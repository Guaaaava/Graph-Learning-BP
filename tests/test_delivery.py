"""Directory isolation, independent RNG roles and fixed-data suite state transitions."""
from contextlib import redirect_stdout,redirect_stderr
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import pytest
import torch

from gib4cl.config import REPO_ROOT,SeedConfig,config_from_dict,load_config,with_seeds,fingerprint
from gib4cl.data.generate import build_dataset,create_dataset
from gib4cl.experiments import suite
from gib4cl.experiments.cli import main
from gib4cl.experiments.worker import main as worker_main
from gib4cl.learning.train import resolve_training_config
from gib4cl.learning.train import train_gnn
from gib4cl.learning.checkpoints import read_checkpoint
from gib4cl.evaluation.evaluate import evaluate
from gib4cl.runtime import sha256
from tools.provenance import preserved_core,preserved_references


@pytest.fixture(autouse=True,scope='module')
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(scope='module')
def panel(tmp_path_factory):
    root = tmp_path_factory.mktemp('seed-panel')
    cfg = load_config('configs/smoke-pipeline.json')
    cfg = replace(cfg,evaluation=replace(cfg.evaluation,bootstrap_samples=12))
    base = root/'base.json'
    base.write_text(json.dumps(cfg.to_dict()),encoding='utf-8')
    spec = dict(suite_id='seed-test',base_config=str(base),
                training_seeds=[101,102],bp_seeds=[700,701],seed_bootstrap_samples=12)
    spec_path = root/'suite.json'
    spec_path.write_text(json.dumps(spec),encoding='utf-8')
    for split in ('train','val','test'):
        create_dataset(split,cfg.split_sizes[split],settings=cfg,output_root=root/'data')
    return root,cfg,spec,spec_path


@pytest.fixture
def inline_child(monkeypatch):
    calls = []
    def execute(command,log_path,training_seed):
        calls.append(command)
        with log_path.open('w',encoding='utf-8') as handle,redirect_stdout(handle),redirect_stderr(handle):
            try:
                worker_main(command[4:])
                return 0
            except Exception:
                traceback.print_exc()
                return 1
    monkeypatch.setattr(suite,'_run_child',execute)
    return calls,execute


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def options(panel,tmp_path):
    root,_,_,spec = panel
    return dict(suite_config=spec,data_root=root/'data',output_root=tmp_path/'runs',
                reuse_data=True,device='cpu',threads=1)


def destination(options):
    spec = suite.load_suite(options['suite_config'])
    return Path(options['output_root'])/spec['suite_id']/fingerprint(spec)[:16]


def test_core_provenance_is_portable_and_old_top_level_imports_are_absent():
    assert len(preserved_core())==3 and len(preserved_references())==8
    for name in ('GNN_learning','BP','NBP_ST','NEBP','references'):
        assert not (REPO_ROOT/name).exists()
        assert importlib.util.find_spec(name) is None


def test_configuration_has_one_input_structure_and_roundtrips():
    cfg = load_config('configs/smoke-pipeline.json')
    value = cfg.to_dict()
    assert 'config_version' not in value and 'seed' not in value
    assert 'architecture_version' not in value['model']
    assert 'algorithm_version' not in value['bp']
    assert config_from_dict(json.loads(json.dumps(value))) == cfg


def test_current_presets_use_five_fixed_anchors_and_one_radius():
    for name in ('experiment.json', 'smoke-pipeline.json'):
        cfg = load_config(REPO_ROOT/'configs'/name)
        assert cfg.data.num_anchors == 5
        assert cfg.data.comm_radius > 0
        assert set(cfg.to_dict()['data']) == {'num_agents_min', 'num_agents_max', 'area_size',
            'comm_radius', 'initial_covariance', 'base_noise_std', 'distance_noise_scale',
            'communication_cost_scale', 'measurement_encoding'}


@pytest.mark.parametrize('role',('training','bp','bootstrap'))
def test_nondata_seed_changes_do_not_change_data_path_or_realizations(role):
    cfg = load_config('configs/smoke-pipeline.json')
    changed = with_seeds(cfg,**{role:999})
    assert changed.data_fingerprint==cfg.data_fingerprint
    assert changed.dataset_path('test')==cfg.dataset_path('test')
    before,after = (build_dataset(value,'test',1)['graphs'][0] for value in (cfg,changed))
    assert before['sample_id']==after['sample_id']
    for key,value in before['inputs'].items():
        if isinstance(value,torch.Tensor):
            torch.testing.assert_close(value,after['inputs'][key],rtol=0,atol=0)
    torch.testing.assert_close(before['targets']['agent_positions'],after['targets']['agent_positions'],rtol=0,atol=0)


def test_data_seed_changes_realizations_and_identity():
    cfg = load_config('configs/smoke-pipeline.json')
    changed = with_seeds(cfg,data=101)
    assert cfg.data_fingerprint != changed.data_fingerprint
    before,after = (build_dataset(value,'test',1)['graphs'][0] for value in (cfg,changed))
    assert before['sample_id']!=after['sample_id']


@pytest.mark.parametrize('value',(-1,True,2**63,1.5))
def test_invalid_stream_values_rejected(value):
    with pytest.raises(ValueError):
        SeedConfig(training=value)


def test_current_configuration_rejects_legacy_seed_aliases():
    value = load_config('configs/smoke-pipeline.json').to_dict()
    value['seed'] = 99
    with pytest.raises(ValueError,match='Unknown'):
        config_from_dict(value)
    del value['seed']
    value['evaluation']['seed'] = 99
    with pytest.raises(ValueError,match='Unknown'):
        config_from_dict(value)


def test_training_cli_override_changes_only_requested_roles():
    cfg = load_config('configs/smoke-pipeline.json')
    resolved = resolve_training_config(settings=cfg,training_seed=111)
    assert resolved.seeds==replace(cfg.seeds,training=111)
    assert resolved.data_fingerprint==cfg.data_fingerprint


def test_suite_plan_has_no_filesystem_side_effect_and_reuses_one_data_identity(panel,tmp_path):
    opts = options(panel,tmp_path)
    value = suite.run_suite(**opts,plan_only=True)
    assert len(value['jobs'])==6 and not Path(opts['output_root']).exists()
    assert len({config_from_dict(j['config']).data_fingerprint for j in value['jobs'].values()})==1


@pytest.mark.parametrize('field,value',[
    ('training_seeds',[]),('training_seeds',[1,1]),('bp_seeds',[True]),
    ('split','train'),('max_graphs',0),('seed_bootstrap_samples',-1),
    ('unexpected',2),
])
def test_suite_schema_rejects_incomplete_ambiguous_or_unrecognized_plans(panel,tmp_path,field,value):
    *_,spec,_ = panel
    changed = dict(spec,**{field:value})
    path = tmp_path/'suite.json'
    path.write_text(json.dumps(changed),encoding='utf-8')
    with pytest.raises(ValueError):
        suite.load_suite(path)


@pytest.mark.parametrize('key,replacement',[
    ('training_seeds',[102]),('bp_seeds',[701]),('suite_id','overridden'),
    ('training_seeds',[101]),
])
def test_suite_rejects_duplicate_json_keys_before_planning(tmp_path,key,replacement):
    spec = dict(suite_id='duplicate-check',base_config=str(REPO_ROOT/'configs/smoke-pipeline.json'),
                training_seeds=[101],bp_seeds=[700])
    content = json.dumps(spec)[:-1]+','+json.dumps(key)+':'+json.dumps(replacement)+'}'
    path = tmp_path/'duplicate-suite.json'
    path.write_text(content,encoding='utf-8')
    output = tmp_path/'outputs'
    with pytest.raises(ValueError,match='Duplicate configuration key: '+key):
        suite.run_suite(suite_config=path,output_root=output,plan_only=True)
    assert not output.exists()


def test_suite_job_boundary_resume_keeps_successes_and_data_bytes(panel,tmp_path,inline_child):
    calls,_ = inline_child
    opts = options(panel,tmp_path)
    root,cfg,_,_ = panel
    initial = {s:sha256(cfg.dataset_path(s,root/'data')) for s in ('train','val','test')}
    path = suite.run_suite(**opts,stop_after_jobs=1)
    first = read(path/'manifest.json')
    assert first['status']=='partial' and len(calls)==1
    assert first['jobs']['train-101']['status']=='complete'
    digest = first['jobs']['train-101']['artifacts']['best_checkpoint']['sha256']
    suite.run_suite(**opts,resume=True)
    manifest,summary = read(path/'manifest.json'),read(path/'summary.json')
    assert manifest['status']=='complete' and len(calls)==6
    assert manifest['jobs']['train-101']['artifacts']['best_checkpoint']['sha256']==digest
    assert len(manifest['jobs']['train-101']['attempts'])==1
    assert all(sha256(cfg.dataset_path(s,root/'data'))==initial[s] for s in initial)
    assert len({j['artifacts']['best_checkpoint']['sha256'] for j in manifest['jobs'].values() if j['kind']=='train'})==2
    assert len({read_checkpoint(path/j['artifacts']['best_checkpoint']['path'])['model_digest']
                for j in manifest['jobs'].values() if j['kind']=='train'})==2
    assert summary['unique_evaluation_graphs']==2 and summary['used_training_seeds']==[101,102]
    assert summary['gnn_across_training_seeds']['rmse']['training_seeds']==2
    assert summary['fixed_baselines']['dense']['graphs']==2
    assert summary['fixed_baselines']['dense']['bp_trials']==8  # 2 graphs x 2 BP seeds x 2 repeats
    assert summary['per_training_seed']['101']['topologies']['dense']==summary['per_training_seed']['102']['topologies']['dense']
    saved = sha256(path/'manifest.json')
    suite.run_suite(**opts,resume=True)
    assert len(calls)==6 and sha256(path/'manifest.json')==saved
    with pytest.raises(FileExistsError):
        suite.run_suite(**opts)


def test_failure_is_visible_retry_is_explicit_and_completed_jobs_are_not_repeated(panel,tmp_path,inline_child,monkeypatch):
    calls,execute = inline_child
    opts = options(panel,tmp_path)
    rejected = set()
    def fail_once(command,log,seed):
        if command[4]=='train' and seed==101 and seed not in rejected:
            rejected.add(seed)
            log.write_text('injected child failure for acceptance\n',encoding='utf-8')
            return 17
        return execute(command,log,seed)
    monkeypatch.setattr(suite,'_run_child',fail_once)
    with pytest.raises(suite.SuiteRunError):
        suite.run_suite(**opts)
    path = destination(opts)
    failed,partial = read(path/'manifest.json'),read(path/'partial_summary.json')
    assert failed['status']=='failed' and failed['jobs']['train-101']['attempts'][0]['exit_code']==17
    assert failed['jobs']['evaluate-101-700']['status']=='blocked'
    assert failed['jobs']['train-102']['status']=='complete'
    assert partial['used_training_seeds']==[102] and partial['missing_training_seeds']==[101]
    assert partial['gnn_across_training_seeds']['rmse']['std'] is None
    previous_calls = len(calls)
    with pytest.raises(suite.SuiteRunError):
        suite.run_suite(**opts,resume=True)
    assert len(calls)==previous_calls
    suite.run_suite(**opts,resume=True,retry_failed=True)
    done = read(path/'manifest.json')
    assert done['status']=='complete'
    assert len(done['jobs']['train-101']['attempts'])==2
    assert len(done['jobs']['train-102']['attempts'])==1
    assert (path/done['jobs']['train-101']['attempts'][0]['log']).read_text(encoding='utf-8').startswith('injected')


def test_resume_rejects_damaged_success_artifact(panel,tmp_path,inline_child):
    opts = options(panel,tmp_path)
    path = suite.run_suite(**opts,stop_after_jobs=1)
    manifest = read(path/'manifest.json')
    checkpoint = path/manifest['jobs']['train-101']['artifacts']['best_checkpoint']['path']
    checkpoint.write_bytes(b'corrupt-checkpoint')
    with pytest.raises(ValueError,match='checksum'):
        suite.run_suite(**opts,resume=True)


def test_resume_rejects_changed_production_sources(panel,tmp_path,inline_child,monkeypatch):
    opts = options(panel,tmp_path)
    suite.run_suite(**opts,stop_after_jobs=1)
    actual = suite.source_manifest()
    monkeypatch.setattr(suite,'source_manifest',lambda:dict(actual,changed='checksum'))
    with pytest.raises(ValueError,match='configuration/source/runtime'):
        suite.run_suite(**opts,resume=True)


def test_bp_and_bootstrap_streams_have_separate_effects_on_real_evaluation(panel,tmp_path):
    root,cfg,_,_ = panel
    models = train_gnn(settings=cfg,data_root=root/'data',run_directory=tmp_path/'model',device='cpu',threads=1)
    paths = [evaluate(checkpoint=models/'best_checkpoint.pt',settings=settings,data_root=root/'data',
                      evaluation_directory=tmp_path/name,device='cpu',threads=1)
             for name,settings in [('original',cfg),('bootstrap',with_seeds(cfg,bootstrap=701)),
                                   ('bp',with_seeds(cfg,bp=701))]]
    original,bootstrap,bp = [read(p/'results.json') for p in paths]
    assert original['records']==bootstrap['records'] and original['graphs']==bootstrap['graphs']
    assert original['summary']['bootstrap_seed']==700 and bootstrap['summary']['bootstrap_seed']==701
    assert original['graphs']==bp['graphs']  # selector and selected physical edges unchanged
    assert original['records'][0]['support_digest']!=bp['records'][0]['support_digest']
    assert original['dataset']['sha256']==bootstrap['dataset']['sha256']==bp['dataset']['sha256']


def test_canonical_package_runs_without_any_old_or_historical_directories(tmp_path):
    for directory in ('gib4cl','configs'):
        shutil.copytree(REPO_ROOT/directory,tmp_path/directory,ignore=shutil.ignore_patterns('__pycache__'))
    env = dict(os.environ,PYTHONPATH=str(tmp_path))
    command = [sys.executable,'-B','-c',
        "import sys; from gib4cl.experiments.cli import main; main(['run','--config','configs/smoke-pipeline.json',"
        "'--data-root','data','--output-root','outputs','--device','cpu','--threads','1']); "
        "assert not any(k.split('.')[0] in {'legacy','references','GNN_learning','BP','NBP_ST'} for k in sys.modules)"]
    result = subprocess.run(command,cwd=tmp_path,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert len(list((tmp_path/'outputs').rglob('results.json')))==1
