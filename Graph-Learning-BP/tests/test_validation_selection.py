"""Independent mask sums, validation isolation, and checkpoint-rule enforcement."""
from copy import deepcopy
from dataclasses import replace
from itertools import product
import math

import pytest
import torch
from torch_geometric.loader import DataLoader

from gib4cl.config import TrainingConfig, load_config
from gib4cl.data.generate import create_dataset
from gib4cl.learning.checkpoints import read_checkpoint, make_model, warm_start
from gib4cl.learning.objectives import gib_objective
from gib4cl.learning.train import train_gnn
from gib4cl.learning import validation
from gib4cl.runtime import training_sources
from tests.test_objectives import make_graph


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def graph():
    return make_graph([[3.,4.]], [[0.,0.],[8.,0.]], [[0,1],[0,2]],
                      [6.,7.], [1.,2.], targets=[[3.4,4.3]])


def test_validation_average_matches_independent_enumeration_of_all_masks(monkeypatch):
    data = graph()
    objective = load_config('configs/smoke-pipeline.json').objective
    class EqualProbability(torch.nn.Module):
        def score_edges(self, data):
            return torch.zeros(data.num_edges)
    model = EqualProbability().train()
    uniforms = {validation.stream_seed('validation',1700,data.sample_id,r):torch.tensor(u)
                for r,u in enumerate(product((.25,.75),repeat=2))}
    monkeypatch.setattr(validation,'uniform_noise',lambda shape,dtype,seed:uniforms[seed])
    actual = validation.validation_probe(model,data,objective,repeats=4,seed=1700)
    logits = torch.zeros(2)
    expected = [float(gib_objective(logits,torch.tensor(mask,dtype=torch.float32),data,
                                    objective=objective).loss) for mask in product((0,1),repeat=2)]
    assert actual['random_mean'] == pytest.approx(math.fsum(expected)/4,rel=0,abs=1e-12)
    zero = float(gib_objective(logits,torch.zeros(2),data,objective=objective).loss)
    assert actual['deterministic_total'] == zero  # p==0.5 is removed
    assert actual['random_mean'] != pytest.approx(zero)
    assert model.training


def test_noise_prefix_repeatability_rng_parameters_and_mixed_modes_are_preserved(monkeypatch):
    cfg = load_config('configs/smoke-pipeline.json')
    model = make_model(cfg.model).train()
    model.unused_dropout = torch.nn.Dropout(.5).eval()
    modes = {name:module.training for name,module in model.named_modules()}
    parameters = {name:value.clone() for name,value in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    original = model.score_edges
    def score(data):
        assert all(not module.training for module in model.modules())
        return original(data)
    monkeypatch.setattr(model,'score_edges',score)
    a = validation.validation_probe(model,graph(),cfg.objective,repeats=4,seed=1700)
    b = validation.validation_probe(model,graph(),cfg.objective,repeats=8,seed=1700)
    assert a['random_totals'] == b['random_totals'][:4]
    assert torch.equal(rng,torch.random.get_rng_state())
    assert modes == {name:module.training for name,module in model.named_modules()}
    assert all(torch.equal(value,model.state_dict()[name]) for name,value in parameters.items())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_epoch_is_graph_equal_and_independent_of_order_and_batch_size():
    cfg = load_config('configs/smoke-pipeline.json')
    model = make_model(cfg.model)
    a = graph()
    b = make_graph([[1.,2.],[3.,1.]],[[0.,0.]],[[0,2],[1,2]],
                   [2.6,3.8],[1.,2.],targets=[[1.5,2.2],[3.3,.5]])
    b.sample_id = 'validation-second-graph'
    options = dict(repeats=4,seed=1700,threshold=.5)
    probes = [validation.validation_probe(model,g,cfg.objective,**options) for g in (a,b)]
    rng = torch.random.get_rng_state().clone()
    x = validation.validation_epoch(model,DataLoader([a,b],batch_size=1,
        generator=torch.Generator().manual_seed(4)),cfg.objective,'cpu',**options)
    y = validation.validation_epoch(model,DataLoader([b,a],batch_size=2,
        generator=torch.Generator().manual_seed(4)),cfg.objective,'cpu',**options)
    assert x == y
    assert torch.equal(rng,torch.random.get_rng_state())
    assert x[0]['total'] == math.fsum(p['random_mean'] for p in probes)/2
    assert x[1]['total'] == math.fsum(p['deterministic_total'] for p in probes)/2
    assert x[2]['prefix_repeats'] == 2
    assert x[2]['prefix_minus_selected_total'] == x[2]['prefix_total']-x[0]['total']


@pytest.mark.parametrize('options',[{'validation_repeats':0},{'validation_repeats':True},
    {'validation_seed':-1},{'validation_seed':True},{'validation_seed':2**63}])
def test_invalid_validation_config_is_rejected(options):
    with pytest.raises(ValueError):
        TrainingConfig(**options)


@pytest.fixture
def saved_run(tmp_path):
    cfg = load_config('configs/smoke-pipeline.json')
    cfg = replace(cfg,training=replace(cfg.training,epochs=3,validation_repeats=4))
    for split in ('train','val'):
        create_dataset(split,cfg.split_sizes[split],settings=cfg,output_root=tmp_path/'data')
    models = train_gnn(settings=cfg,data_root=tmp_path/'data',output_root=tmp_path/'run',
                       stop_after_epoch=1,device='cpu',threads=1)
    return cfg,models,tmp_path/'data'


def test_checkpoint_records_rule_and_rejects_protocol_change_or_legacy_resume(saved_run,tmp_path):
    cfg,models,data_root = saved_run
    current = read_checkpoint(models/'last_checkpoint.pt')
    assert current['checkpoint_version'] == 'gib4cl.checkpoint.v2'
    assert current['validation_protocol']['selection'] == 'minimum_mc_mean_total'
    assert current['validation_protocol']['num_graphs'] == cfg.split_sizes['val']
    assert not cfg.dataset_path('test',data_root).exists()
    assert 'learning/validation.py' in training_sources()
    corrupt = deepcopy(current)
    corrupt['validation_protocol']['seed'] += 1
    path = tmp_path/'corrupt.pt'
    torch.save(corrupt,path)
    with pytest.raises(ValueError,match='validation protocol'):
        read_checkpoint(path)
    with pytest.raises(ValueError,match='original full config'):
        train_gnn(resume=models/'last_checkpoint.pt',settings=replace(cfg,
            training=replace(cfg.training,validation_repeats=8)),data_root=data_root,device='cpu')
    legacy = deepcopy(current)
    legacy['checkpoint_version'] = 'gib4cl.checkpoint.v1'
    del legacy['validation_protocol']
    legacy['config']['training'].pop('validation_repeats')
    legacy['config']['training'].pop('validation_seed')
    for row in legacy['history']:
        row.pop('val_deterministic')
        row.pop('val_sampling')
    path = tmp_path/'legacy.pt'
    torch.save(legacy,path)
    assert read_checkpoint(path)['checkpoint_version'] == 'gib4cl.checkpoint.v1'
    model = make_model(cfg.model)
    assert warm_start(model,path)['model_digest'] == current['model_digest']
    with pytest.raises(ValueError,match='Legacy checkpoint'):
        train_gnn(resume=path,data_root=data_root,device='cpu')


def test_best_epoch_uses_random_mean_not_threshold_and_keeps_earliest_tie(saved_run,tmp_path,monkeypatch):
    cfg,_,data_root = saved_run
    import gib4cl.learning.train as trainer
    totals = iter([(3.,0.),(1.,-1.),(1.,-2.)])
    def controlled(model,loader,objective,device,*,repeats,seed,threshold):
        random,deterministic = next(totals)
        return {'total':random},{'total':deterministic},{'repeats':repeats,
            'prefix_repeats':max(1,repeats//2),'prefix_total':random,'prefix_minus_selected_total':0.}
    monkeypatch.setattr(trainer,'validation_epoch',controlled)
    models = train_gnn(settings=cfg,data_root=data_root,output_root=tmp_path/'controlled',
                       device='cpu',threads=1)
    last = read_checkpoint(models/'last_checkpoint.pt')
    best = read_checkpoint(models/'best_checkpoint.pt')
    assert last['epoch'] == 3 and best['epoch'] == last['best_epoch'] == 2
    assert last['best_metric'] == 1.
    assert last['history'][2]['val_deterministic']['total'] == -2.
