"""Training observations must preserve the original update and expose failures."""
from copy import deepcopy
from dataclasses import replace
import json
import math

import pytest
import torch
from torch_geometric.loader import DataLoader

from gib4cl.config import load_config
from gib4cl.data.generate import create_dataset
from gib4cl.learning.checkpoints import make_model, read_checkpoint
from gib4cl.learning.diagnostics import EpochDiagnostics, selection_statistics, diagnostic_protocol
from gib4cl.learning.objectives import compute_gib_loss
from gib4cl.learning.train import objective_epoch, train_gnn
from tests.audits.checkpoints import identical
from tests.test_objectives import make_graph


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def triangle():
    return make_graph([[0.,0.],[3.,0.],[0.,4.]],[[5.,5.]],
                      [[0,1],[0,2],[1,2],[0,3]],[3.2,4.1,4.9,7.],[1.]*4,
                      targets=[[.1,.2],[3.1,.1],[-.2,4.2]])


def loader():
    a,b = triangle(),triangle()
    b.sample_id = 'second-training-graph'
    return DataLoader([a,b,a],batch_size=2,shuffle=False,
                      generator=torch.Generator().manual_seed(77))


def original_epoch(model, batches, objective, optimizer, max_norm):
    """Independent copy of the training operations before diagnostics existed."""
    model.train()
    totals, count = {},0
    for batch in batches:
        optimizer.zero_grad()
        weights,logits = model(batch,tau=1.,hard=True)
        loss,metrics = compute_gib_loss(logits,weights,batch,objective=objective)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=max_norm,error_if_nonfinite=True)
        optimizer.step()
        count += int(batch.num_graphs)
        for key,value in metrics.items():
            totals[key] = totals.get(key,0.)+value*int(batch.num_graphs)
    return {key:value/count for key,value in totals.items()}


def test_logged_training_is_bit_identical_to_original_update_and_rng(tmp_path):
    cfg = load_config('configs/smoke-pipeline.json')
    reference = make_model(cfg.model)
    logged = deepcopy(reference)
    left = torch.optim.AdamW(reference.parameters(),lr=.001,weight_decay=1e-4)
    right = torch.optim.AdamW(logged.parameters(),lr=.001,weight_decay=1e-4)
    torch.manual_seed(123)
    expected = original_epoch(reference,loader(),cfg.objective,left,.1)
    expected_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(123)
    diagnostic = EpochDiagnostics(logged,right,epoch=1,max_grad_norm=.1,directory=tmp_path)
    actual = objective_epoch(logged,loader(),cfg.objective,'cpu',tau=1.,optimizer=right,
                              max_grad_norm=.1,diagnostics=diagnostic)
    assert expected == actual
    assert identical(reference.state_dict(),logged.state_dict())
    assert identical(left.state_dict(),right.state_dict())
    assert torch.equal(expected_rng,torch.random.get_rng_state())
    summary = diagnostic.summary()
    assert summary['steps'] == summary['clipped_steps'] == 2
    assert summary['clip_fraction'] == 1.
    assert summary['post_clip_norm']['max'] <= .1+1e-7
    assert summary['selection']['graphs'] == 3
    files = list((tmp_path/'epoch-0001').glob('*.json'))
    assert len(files) == 2
    assert all(json.loads(path.read_text())['optimizer_step_applied'] for path in files)


def test_selection_counts_include_anchor_degrees_and_only_agent_triangles():
    batch = next(iter(DataLoader([triangle()],batch_size=1)))
    mask = torch.tensor([1.,0.,1.,1.])
    row = selection_statistics(batch,mask,torch.tensor([-5.,0.,5.,2.]))
    assert row['candidate_edges'] == 4 and row['selected_edges'] == 3
    assert row['retention_fraction_graph_mean'] == .75
    assert row['agents_below_degree_three'] == 3
    assert row['degree_deficit_graph_mean'] == 4
    assert row['unavoidable_degree_deficit_graph_mean'] == 2
    assert row['selected_triangles_graph_mean'] == 0
    assert row['selected_cost_graph_mean'] == float(batch.edge_cost[[0,2,3]].double().sum())
    assert row['probability_low_count'] == row['probability_high_count'] == 1


def test_empty_edge_graph_has_null_edge_fractions_and_retains_degree_deficit():
    graph = make_graph([[0.,0.]],[],[],[],[])
    batch = next(iter(DataLoader([graph],batch_size=1)))
    row = selection_statistics(batch,torch.empty(0),torch.empty(0))
    assert row['retention_fraction_graph_mean'] is None
    assert row['probability_mean'] is None
    assert row['graphs_with_candidate_edges'] == row['selected_edges'] == 0
    assert row['degree_deficit_graph_mean'] == row['unavoidable_degree_deficit_graph_mean'] == 3
    json.dumps(row,allow_nan=False)


def test_adamw_nominal_decay_excludes_parameters_without_gradients(tmp_path):
    model = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([3.,4.],dtype=torch.float64)),
                                   torch.nn.Parameter(torch.tensor([12.],dtype=torch.float64))])
    optimizer = torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.1)
    model[0].grad = torch.zeros_like(model[0])
    batch = next(iter(loader()))
    diagnostic = EpochDiagnostics(model,optimizer,epoch=1,max_grad_norm=5.,directory=tmp_path)
    diagnostic.begin(1,batch)
    diagnostic.gradients()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
    diagnostic.clipped(norm)
    diagnostic.before_step()
    diagnostic.stage('optimizer_step')
    optimizer.step()
    diagnostic.after_step()
    row = diagnostic.current
    assert row['gradient']['clip_applied'] is False
    assert row['gradient']['parameters_without_gradient'] == 1
    assert row['update']['parameter_norm_before'] == 13.
    assert row['update']['nominal_adamw_decay_norm'] == pytest.approx(.005,abs=1e-14)
    assert row['update']['actual_step_norm'] == pytest.approx(.005,abs=1e-14)
    assert row['update']['relative_step_norm'] == pytest.approx(.005/13,abs=1e-14)
    assert model[1].item() == 12.


def test_nonfinite_gradient_failure_is_saved_without_nan_or_optimizer_step(tmp_path):
    cfg = load_config('configs/smoke-pipeline.json')
    model = make_model(cfg.model)
    before = deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001)
    handle = next(model.parameters()).register_hook(lambda gradient:gradient*float('nan'))
    diagnostic = EpochDiagnostics(model,optimizer,epoch=1,max_grad_norm=5.,directory=tmp_path)
    for _ in range(2):
        with pytest.raises(RuntimeError,match='non-finite'):
            objective_epoch(model,loader(),cfg.objective,'cpu',tau=1.,optimizer=optimizer,diagnostics=diagnostic)
    handle.remove()
    files = list((tmp_path/'failures').glob('*.json'))
    assert len(files) == 2  # retain failed attempts instead of overwriting one
    row = json.loads(files[0].read_text())
    assert row['stage'] == 'gradient_clip' and row['status'] == 'failed'
    assert row['gradient']['nonfinite_elements'] > 0
    assert row['gradient']['measured_pre_clip_norm'] is None and row['nonfinite_fields']
    assert row['optimizer_step_applied'] is False
    assert identical(before,model.state_dict())
    assert not list(tmp_path.glob('epoch-*/*.json'))


def test_log_summaries_and_batch_records_resume_exactly(tmp_path):
    cfg = load_config('configs/smoke-pipeline.json')
    cfg = replace(cfg,training=replace(cfg.training,validation_repeats=4))
    for split in ('train','val'):
        create_dataset(split,cfg.split_sizes[split],settings=cfg,output_root=tmp_path/'data')
    full_path = train_gnn(settings=cfg,data_root=tmp_path/'data',output_root=tmp_path/'full',device='cpu',threads=1)
    split_path = train_gnn(settings=cfg,data_root=tmp_path/'data',output_root=tmp_path/'split',device='cpu',threads=1,stop_after_epoch=1)
    train_gnn(resume=split_path/'last_checkpoint.pt',data_root=tmp_path/'data',device='cpu',threads=1)
    full,split = (read_checkpoint(path/'last_checkpoint.pt') for path in (full_path,split_path))
    for key in ('model_state','optimizer_state','scheduler_state','history','rng_state','best_metric','best_epoch'):
        assert identical(full[key],split[key]),key
    assert full['training_diagnostics'] == diagnostic_protocol()
    for row in full['history']:
        assert row['train_diagnostics']['steps'] == 2
        assert row['train_diagnostics']['gradient_nonfinite_elements'] == 0
    full_logs = {p.relative_to(full_path.parent/'diagnostics').as_posix():json.loads(p.read_text())
                 for p in (full_path.parent/'diagnostics').glob('epoch-*/*.json')}
    split_logs = {p.relative_to(split_path.parent/'diagnostics').as_posix():json.loads(p.read_text())
                  for p in (split_path.parent/'diagnostics').glob('epoch-*/*.json')}
    assert len(full_logs) == 4 and full_logs == split_logs
    corrupt = deepcopy(full)
    corrupt['history'][0]['train_diagnostics']['clipped_steps'] = 999
    path = tmp_path/'corrupt.pt'
    torch.save(corrupt,path)
    with pytest.raises(ValueError,match='diagnostics history'):
        read_checkpoint(path)
