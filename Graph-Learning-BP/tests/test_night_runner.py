"""Bounded execution, stage gates, OS locks, and exact training-slice continuation."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

from tools import night_runner as night
from tools.selection_state import Workspace,WorkLimit
from tools.selection_candidates import train_candidate
from tools.selection_evaluate import prepare_data
from gib4cl.config import load_config
from gib4cl.runtime import execution
from gib4cl.learning.checkpoints import read_checkpoint
from tests.audits.checkpoints import identical


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def batch(tmp_path):
    cfg = load_config('configs/smoke-pipeline.json')
    cfg = replace(cfg,training=replace(cfg.training,validation_repeats=2),
                  evaluation=replace(cfg.evaluation,bootstrap_samples=10))
    config = tmp_path/'config.json';night.atomic(config,cfg.to_dict())
    spec = night.read('configs/night/smoke.json')
    spec['config'] = str(config)
    spec['protocol'] = str(Path('configs/selection/smoke-protocol.json').resolve())
    spec['budget']['max_seconds'] = 300
    path = tmp_path/'batch.json';night.atomic(path,spec)
    return cfg,spec,path


def arguments(path,root,*extras):
    return night.parser().parse_args(['run','--batch',str(path),'--output-root',str(root),*extras])


def synchronous_launch(root,request_path,log_path):
    # Execute the real selection entry, using the supervisor's structured worker contract.
    code = night.worker(request_path)
    class Done:
        pid = os.getpid()
        returncode = code
        def poll(self): return self.returncode
    return Done()


def fake_launch(status='partial',units=1,callback=None):
    def launch(root,request_path,log_path):
        request = night.read(request_path)
        if callback: callback(root,request)
        outcome = {'status':status,'new_units':units,'selection_result':{'status':status}}
        if status=='failed': outcome['error']='injected worker failure'
        night.atomic(request['result_file'],outcome)
        class Done:
            pid=12345
            returncode=1 if status=='failed' else 0
            def poll(self): return self.returncode
        return Done()
    return launch


def test_plan_is_stdlib_only_no_dataset_or_gpu_and_formal_estimate_unset():
    command = [sys.executable,'-B','-c',
        "import sys; from tools.night_runner import main; main(['plan','--batch','configs/night/iterations-first.json']); assert 'torch' not in sys.modules"]
    result = subprocess.run(command,capture_output=True,text=True,check=True)
    data = json.loads(result.stdout)
    assert data['plan_only'] and not data['automatic_next_stage']
    assert data['batch']['budget']['unit_estimate_seconds'] is None


@pytest.mark.parametrize('change',['unbounded','zero_units','zero_epochs','estimate_nan','unknown_command','stage_skip'])
def test_batch_rejects_invalid_budgets_and_unapproved_stage_order(batch,change):
    _,spec,path = batch
    if change=='unbounded': spec['budget']['max_seconds']=7201
    elif change=='zero_units': spec['budget']['max_units']=0
    elif change=='zero_epochs': spec['budget']['epochs_per_slice']=0
    elif change=='estimate_nan':
        path.write_text(json.dumps({**spec,'budget':{**spec['budget'],'unit_estimate_seconds':float('nan')}}))
        with pytest.raises(ValueError):night.load_batch(path)
        return
    elif change=='unknown_command': spec['stages'][0]['command']='shell'
    else: spec['stages'][0]['options']['stage']='B'
    night.atomic(path,spec)
    with pytest.raises(ValueError):night.load_batch(path)


def test_unit_budget_pause_resume_and_stage_confirmation_gate(batch,tmp_path):
    _,_,path=batch;root=tmp_path/'night'
    state = night.run_batch(arguments(path,root),launch=fake_launch())
    assert state['status']=='budget_paused' and state['submissions'][0]['units']==2
    state = night.run_batch(arguments(path,root,'--resume'),launch=fake_launch('complete'))
    assert state['status']=='stage_complete' and state['stage_index']==0
    def forbidden(*args):raise AssertionError('Implicit stage advance')
    assert night.run_batch(arguments(path,root,'--resume'),launch=forbidden)['stage_index']==0
    state = night.run_batch(arguments(path,root,'--resume','--advance'),launch=fake_launch('complete'))
    assert state['status']=='complete' and state['stage_index']==1
    assert night.run_batch(arguments(path,root,'--resume'),launch=forbidden)['status']=='complete'
    assert night.read(root/'morning-report.json')['status']=='complete'


def test_failed_worker_stops_without_next_work_or_implicit_retry(batch,tmp_path):
    _,_,path=batch;root=tmp_path/'night'
    state = night.run_batch(arguments(path,root),launch=fake_launch('failed'))
    assert state['status']=='failed' and len(state['submissions'][0]['workers'])==1
    assert state['submissions'][0]['units']==1 and not state['stage_results']
    with pytest.raises(ValueError,match='retry-failed'):
        night.run_batch(arguments(path,root,'--resume'),launch=fake_launch())
    state = night.run_batch(arguments(path,root,'--resume','--retry-failed'),launch=fake_launch('complete'))
    assert state['status']=='stage_complete'
    assert state['submissions'][0]['status']=='failed'
    assert 'injected worker failure' in (root/'submissions/0001/unit-0001-result.json').read_text()


def test_abrupt_worker_exit_retains_started_count_and_failure_report(batch,tmp_path):
    _,_,path=batch;root=tmp_path/'abrupt'
    def launch(root,request_path,log_path):
        request=night.read(request_path)
        night.atomic(request['activity_file'],{'started_units':1})
        class Dead:
            pid=12345
            returncode=137
            def poll(self):return self.returncode
        return Dead()
    state=night.run_batch(arguments(path,root),launch=launch)
    assert state['status']=='failed' and state['submissions'][0]['units']==1
    assert 'without a complete result' in night.read(root/'morning-report.json')['last_error']


def test_unknown_throughput_is_rejected_before_launch_or_directory_creation(batch,tmp_path):
    _,spec,path=batch;root=tmp_path/'no-estimate'
    spec['budget']['unit_estimate_seconds']=None;spec['budget']['estimate_note']=''
    night.atomic(path,spec)
    with pytest.raises(ValueError,match='estimate'):
        night.run_batch(arguments(path,root))
    with pytest.raises(ValueError,match='estimate'):
        night.detached(arguments(path,root))
    assert not root.exists()


def test_source_drift_and_completed_result_tampering_are_rejected(batch,tmp_path,monkeypatch):
    _,_,path=batch;root=tmp_path/'night'
    state=night.run_batch(arguments(path,root),launch=fake_launch('complete'))
    entry=state['stage_results']['A']
    original=(root/entry['path']).read_bytes()
    night.atomic(root/entry['path'],{'changed':True})
    with pytest.raises(ValueError,match='missing or changed'):
        night.run_batch(arguments(path,root,'--resume'))
    (root/entry['path']).write_bytes(original)
    monkeypatch.setattr(night,'source_identity',lambda:{'changed':'hash'})
    with pytest.raises(ValueError,match='identical'):
        night.run_batch(arguments(path,root,'--resume'))


def test_os_lock_blocks_duplicate_supervisor_and_orphan_worker(batch,tmp_path):
    _,_,path=batch;root=tmp_path/'night'
    night.run_batch(arguments(path,root),launch=fake_launch())
    for target in (root,root/'worker-lock'):
        with night.lock(target):
            assert night.status(root)['active_lock']
            with pytest.raises(RuntimeError,match='holds this batch lock'):
                night.run_batch(arguments(path,root,'--resume'),launch=fake_launch())
    assert not night.status(root)['active_lock']
    state=night.read(root/'progress.json');state['status']='running';night.write_state(root,state)
    assert night.status(root)['supervisor_or_worker_unexpected_exit']
    recovered=night.run_batch(arguments(path,root,'--resume'),launch=fake_launch('complete'))
    assert recovered['status']=='stage_complete'


def test_cooperative_stop_and_time_admission_do_not_hard_kill_or_launch_next(batch,tmp_path,monkeypatch):
    _,_,path=batch;root=tmp_path/'stop'
    def stopping(root,request):night.atomic(root/'STOP.json',{'stop':True})
    state=night.run_batch(arguments(path,root),launch=fake_launch(callback=stopping))
    assert state['status']=='stopped' and state['submissions'][0]['units']==1
    state=night.run_batch(arguments(path,root,'--resume'),launch=fake_launch('complete'))
    assert state['status']=='stage_complete' and not (root/'STOP.json').exists()
    ticks=[0.]
    monkeypatch.setattr(night.time,'monotonic',lambda:ticks[0])
    def overrun(root,request):ticks[0]=310.
    timed=night.run_batch(arguments(path,tmp_path/'time'),launch=fake_launch(callback=overrun))
    assert timed['status']=='budget_paused' and timed['submissions'][0]['units']==1
    assert timed['submissions'][0]['overrun_seconds']==10.


def test_promoted_budget_requires_explicit_values_and_cannot_change_mid_stage(batch,tmp_path):
    _,spec,path=batch;root=tmp_path/'review-budget'
    spec['stages']=[{'id':s,'command':'candidates','options':{'stage':s}} for s in night.STAGES]
    night.atomic(path,spec)
    state=night.run_batch(arguments(path,root),launch=fake_launch('complete'))
    for _ in range(3):
        state=night.run_batch(arguments(path,root,'--resume','--advance'),launch=fake_launch('complete'))
    assert state['stage_index']==3
    with pytest.raises(ValueError,match='explicit BP'):
        night.run_batch(arguments(path,root,'--resume','--advance'),launch=fake_launch())
    assert night.read(root/'progress.json')['stage_index']==3
    state=night.run_batch(arguments(path,root,'--resume','--advance','--bp-iterations','5','--bp-chunk-size','8'),launch=fake_launch())
    assert state['status']=='budget_paused' and state['stage_index']==4
    with pytest.raises(ValueError,match='Cannot change'):
        night.run_batch(arguments(path,root,'--resume','--bp-iterations','3','--bp-chunk-size','8'),launch=fake_launch())
    state=night.run_batch(arguments(path,root,'--resume'),launch=fake_launch('complete'))
    assert state['stage_settings']['promote']=={'bp_iterations':5,'bp_chunk_size':8}


def test_cooperative_training_pause_and_epoch_slicing_preserve_exact_state(batch,tmp_path):
    cfg,spec,_=batch
    device=execution('cpu',1,True,0)
    protocol=night.read(spec['protocol'])
    coef={'beta':.01,'gamma':0,'kappa':0,'eta':0}
    full=Workspace(tmp_path/'full',kind='candidates',config=cfg,protocol=protocol,device=device)
    prepare_data(full,cfg)
    full_spec=train_candidate(full,cfg,coef,101,2,device,1)
    expected=read_checkpoint(full.root/full_spec['last_snapshot']['path'])
    sliced=Workspace(tmp_path/'slice',kind='candidates',config=cfg,protocol=protocol,device=device,max_training_epochs=1)
    prepare_data(sliced,cfg,source_root=full.root/'data')
    with pytest.raises(WorkLimit):train_candidate(sliced,cfg,coef,101,2,device,1)
    assert not list((sliced.root/'failures').glob('*.json'))
    sliced=Workspace(sliced.root,kind='candidates',config=cfg,protocol=protocol,device=device,resume=True,max_training_epochs=1)
    got=train_candidate(sliced,cfg,coef,101,2,device,1)
    actual=read_checkpoint(sliced.root/got['last_snapshot']['path'])
    for key in ('model_state','optimizer_state','scheduler_state','history','rng_state','best_metric','best_epoch'):
        assert identical(expected[key],actual[key]),key
    # Also exercise the optional time/stop callback with no epoch slicing.
    class StopAfterCommit:
        def stop_requested(self):
            return bool(list((tmp_path/'cooperative/runs').glob('*/models/last_checkpoint.pt')))
        def job_started(self,key):pass
    paused=Workspace(tmp_path/'cooperative',kind='candidates',config=cfg,protocol=protocol,device=device,control=StopAfterCommit())
    prepare_data(paused,cfg,source_root=full.root/'data')
    with pytest.raises(WorkLimit):train_candidate(paused,cfg,coef,101,2,device,1)
    saved=read_checkpoint(next((paused.root/'runs').glob('*/models/last_checkpoint.pt')))
    assert saved['epoch']==1 and len(saved['history'])==1
    run=next((paused.root/'runs').glob('*/run.json'))
    broken=night.read(run);broken['last_epoch']=0;night.atomic(run,broken)
    resumed=Workspace(paused.root,kind='candidates',config=cfg,protocol=protocol,device=device,resume=True)
    with pytest.raises(ValueError,match='committed boundary'):
        train_candidate(resumed,cfg,coef,101,2,device,1)


def test_successful_retry_then_budget_pause_is_not_a_new_failure(batch,tmp_path):
    cfg,spec,_=batch;protocol=night.read(spec['protocol']);device=execution('cpu',1,True,0)
    root=tmp_path/'retry-pause'
    ws=Workspace(root,kind='candidates',config=cfg,protocol=protocol,device=device)
    def fail():raise RuntimeError('first attempt failed')
    with pytest.raises(RuntimeError):ws.job({'unit':'one'},fail)
    ws=Workspace(root,kind='candidates',config=cfg,protocol=protocol,device=device,resume=True,retry_failed=True)
    def pause():raise WorkLimit()
    with pytest.raises(WorkLimit):ws.job({'unit':'one'},pause)
    assert night.read(next((root/'jobs').glob('*.json')))['status']=='paused'
    ws=Workspace(root,kind='candidates',config=cfg,protocol=protocol,device=device,resume=True)
    assert ws.job({'unit':'one'},lambda:({'ok':True},[]))['ok']
    assert len(list((root/'failures').glob('*.json')))==1


def test_real_candidate_worker_progress_reports_and_complete_cache(batch,tmp_path):
    _,_,path=batch;root=tmp_path/'real'
    for attempt in range(15):
        extras=[] if attempt==0 else ['--resume']
        state=night.run_batch(arguments(path,root,*extras),launch=synchronous_launch)
        assert state['status'] in ('budget_paused','stage_complete')
        if state['status']=='stage_complete':break
    else:raise AssertionError('Tiny stage did not finish')
    assert state['stage_index']==0
    report=night.read(root/'morning-report.json')
    assert report['progress']['candidates']['complete_comparisons']['comparison-A']['method']=='gaussian_proxy'
    assert report['progress']['candidates']['training_runs'][0]['last_training_diagnostics'] is not None
    assert not list((root/'workspaces/candidates/data').rglob('*test_dataset.pt'))
    before={str(p):night.file_sha(p) for p in (root/'workspaces/candidates/checkpoints').glob('*.pt')}
    def forbidden(*args):raise AssertionError('Completed stage reran worker')
    night.run_batch(arguments(path,root,'--resume'),launch=forbidden)
    assert before=={str(p):night.file_sha(p) for p in (root/'workspaces/candidates/checkpoints').glob('*.pt')}


def test_real_detached_launch_stop_and_resume_independent_of_parent_session(batch,tmp_path):
    _,spec,path=batch
    spec['stages']=[{'id':'rounds','command':'iterations','options':{'graph_indices':[0],'iterations':[0,1]}}]
    spec['budget']['max_units']=1
    night.atomic(path,spec);root=tmp_path/'detached'
    args=arguments(path,root)
    receipt=night.detached(args)
    assert Path(receipt['launch_log']).is_file()
    deadline=time.monotonic()+60
    while time.monotonic()<deadline:
        if (root/'progress.json').is_file():
            state=night.read(root/'progress.json')
            if state['status']!='running' and not night.status(root)['active_lock']:break
        time.sleep(.1)
    else:raise AssertionError('Detached software batch did not finish; '+str(receipt))
    assert state['status']=='budget_paused'
    assert state['submissions'][0]['units']==1
    assert night.read(root/'morning-report.json')['progress']['iterations']['trajectory_points']
    # New foreground supervisor continues the detached process's durable selection cache.
    resumed=night.run_batch(arguments(path,root,'--resume'),launch=synchronous_launch)
    assert resumed['status']=='complete'
    assert resumed['submissions'][1]['units']==1
