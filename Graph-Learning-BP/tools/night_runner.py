"""Bounded, detached selection batches. Standard-library supervisor; explicit stage gates."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'gib4cl.night-batch.v1'
ITERATION_OPTIONS = {'graph_indices','seeds','particles','iterations','chunk_size','support_reference'}
CANDIDATE_OPTIONS = {'data_root','bp_particles','bp_iterations','bp_chunk_size','bp_seeds'}
STAGES = ('A','B','C','D','promote','final')


def io_path(path):
    value = os.path.abspath(path)
    if os.name=='nt' and not value.startswith('\\\\?\\'):
        value = '\\\\?\\UNC\\'+value[2:] if value.startswith('\\\\') else '\\\\?\\'+value
    return Path(value)


def read(path):
    def pairs(items):
        out = {}
        for key,value in items:
            if key in out:
                raise ValueError('Duplicate JSON key: '+key)
            out[key] = value
        return out
    def invalid(value):
        raise ValueError('Nonfinite JSON: '+value)
    return json.loads(io_path(path).read_text(encoding='utf-8'),object_pairs_hook=pairs,parse_constant=invalid)


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    return digest(io_path(path).read_bytes())


def atomic(path,value):
    path = io_path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name = tempfile.mkstemp(dir=path.parent,prefix='.night-',suffix='.json')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
    finally:
        if Path(name).exists():
            Path(name).unlink()


def utc():
    return datetime.now(timezone.utc).isoformat()


def positive(value,name,integer=False):
    if isinstance(value,bool) or not isinstance(value,int if integer else (int,float)) or not math.isfinite(value) or value<=0:
        raise ValueError(name+' must be positive'+(' integer' if integer else ' finite number'))


def resolved(path):
    path = Path(path)
    return (ROOT/path).resolve() if not path.is_absolute() else path.resolve()


def load_batch(path):
    spec = read(path)
    allowed = {'schema','name','config','protocol','device','threads','budget','stages'}
    if set(spec)!=allowed or spec['schema']!=SCHEMA:
        raise ValueError('Invalid night batch schema/keys')
    if not isinstance(spec['name'],str) or not spec['name'].strip():
        raise ValueError('Batch name is required')
    if spec['device'] not in ('cpu','cuda'):
        raise ValueError('Explicit cpu/cuda device required')
    positive(spec['threads'],'threads',True)
    cfg,protocol = read(resolved(spec['config'])),read(resolved(spec['protocol']))
    budget = spec['budget']
    if set(budget)!={'max_seconds','max_units','epochs_per_slice','unit_estimate_seconds','estimate_note','safety_factor'}:
        raise ValueError('Invalid budget keys')
    for key in ('max_seconds','safety_factor'):
        positive(budget[key],key)
    if budget['safety_factor']<1:
        raise ValueError('safety_factor must be >= 1')
    for key in ('max_units','epochs_per_slice'):
        positive(budget[key],key,True)
    limit = protocol['resource_policy']['max_planned_wallclock_seconds_per_submission']
    if budget['max_seconds']>limit:
        raise ValueError('Batch exceeds protocol submission time budget')
    if budget['unit_estimate_seconds'] is not None:
        positive(budget['unit_estimate_seconds'],'unit_estimate_seconds')
    if not isinstance(budget['estimate_note'],str):
        raise ValueError('estimate_note must be text')
    stages = spec['stages']
    if not isinstance(stages,list) or not stages:
        raise ValueError('Nonempty explicit stages required')
    ids,candidate_stages = set(),[]
    for stage in stages:
        if set(stage)!={'id','command','options'} or not isinstance(stage['id'],str) or not stage['id'].isascii() or not stage['id'].replace('-','').replace('_','').isalnum() or stage['id'] in ids:
            raise ValueError('Invalid/duplicate stage ID')
        ids.add(stage['id'])
        options = stage['options']
        if not isinstance(options,dict):
            raise ValueError('Stage options must be an object')
        if stage['command']=='candidates':
            if set(options)-CANDIDATE_OPTIONS-{'stage'} or options.get('stage') not in STAGES:
                raise ValueError('Invalid candidate stage/options')
            candidate_stages.append(options['stage'])
        elif stage['command']=='iterations':
            if set(options)-ITERATION_OPTIONS:
                raise ValueError('Invalid iteration options')
        else:
            raise ValueError('Only targeted iterations and candidate stages are allowed')
    if candidate_stages and candidate_stages!=list(STAGES[:len(candidate_stages)]):
        raise ValueError('Candidate stages must be an explicit prefix A/B/C/D/promote/final')
    # Normal selection parser validates all numeric inference options in the worker.
    # Parse the experiment now only as JSON: plan/status need no Torch or GPU.
    if not isinstance(cfg,dict):
        raise ValueError('Invalid experiment config')
    return spec


def source_identity():
    return {p.relative_to(ROOT).as_posix():file_sha(p)
            for base in ('gib4cl','tools') for p in sorted((ROOT/base).rglob('*.py'))
            if '__pycache__' not in p.parts}


def identity(spec):
    external = {}
    for stage in spec['stages']:
        for key in ('support_reference',):
            if stage['options'].get(key):
                path = resolved(stage['options'][key]);external[str(path)] = file_sha(path)
    return {'batch':spec,'config_path':str(resolved(spec['config'])),'config_sha256':file_sha(resolved(spec['config'])),
            'protocol_path':str(resolved(spec['protocol'])),'protocol_sha256':file_sha(resolved(spec['protocol'])),
            'source_sha256':source_identity(),'external_files':external,
            'python':platform.python_version(),'executable':str(Path(sys.executable).resolve()),'system':platform.system()}


@contextmanager
def lock(root):
    """OS-owned lock: released on crash/reboot, never removed as a stale PID guess."""
    path = io_path(Path(root)/'RUN.lock')
    path.parent.mkdir(parents=True,exist_ok=True)
    stream = path.open('a+b')
    try:
        stream.seek(0)
        # Reading a byte locked by another Windows handle itself raises EACCES.
        # Inspect file size instead; let the nonblocking lock report contention.
        if os.fstat(stream.fileno()).st_size==0:
            stream.write(b'0');stream.flush()
        stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('Another supervisor/worker holds this batch lock') from error
        try:
            yield
        finally:
            if os.name=='nt':
                import msvcrt
                stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(stream,fcntl.LOCK_UN)
    finally:
        stream.close()


def write_state(root,state):
    state['updated_at'] = utc()
    atomic(Path(root)/'progress.json',state)


def verify_artifact(root,entry):
    path = (Path(root)/entry['path']).resolve()
    if not path.is_relative_to(Path(root).resolve()) or file_sha(path)!=entry['sha256']:
        raise ValueError('Night result missing or changed: '+entry['path'])
    return read(path)


def artifact(root,path):
    return {'path':Path(path).relative_to(root).as_posix(),'sha256':file_sha(path)}


def progress_snapshot(root):
    """Read durable counters without deserializing Torch weights or changing selection files."""
    out = {}
    for command in ('iterations','candidates'):
        base = Path(root)/'workspaces'/command
        path = base/'manifest.json'
        if not io_path(path).exists():
            continue
        manifest = read(path)
        rows = []
        for run in io_path(base/'runs').glob('*/run.json'):
            data = read(run)
            history_path = run.parent/'history.json'
            history = read(history_path) if history_path.is_file() else []
            rows.append({'run':run.parent.name,'last_epoch':data.get('last_epoch',0),
                         'best_epoch':data.get('best_epoch'),'best_metric':data.get('best_metric'),
                         'complete':data.get('complete',False),
                         'last_training_diagnostics':history[-1].get('train_diagnostics') if history else None})
        comparisons,trajectories,report_errors = {},[],{}
        for name,entry in manifest['reports'].items():
            if not (name.startswith('comparison-') or name=='iterations'):
                continue
            try:
                data = verify_artifact(base,entry)
                if name.startswith('comparison-') and data['status']=='complete':
                    comparisons[name] = {'method':data['context']['method'],'rows':data['rows'],
                                         'selection':data['selection']}
                elif name=='iterations':
                    for trajectory in data['trajectories']:
                        for row in trajectory['rows']:
                            trajectories.append({'sample_index':trajectory['sample_index'],'seed':trajectory['seed'],
                                'topology':trajectory['topology'],'iterations':row['requested_iterations'],
                                'rmse':row['metrics']['rmse'],'coverage95':row['coverage95'],
                                'ess_min':min(row['ess']) if row['ess'] else None,
                                'maximum_weight':max(row['maximum_weight']) if row['maximum_weight'] else None,
                                'trajectory_seconds':trajectory['trajectory_seconds'],
                                'cuda_peak_allocated_bytes':trajectory['cuda_peak_allocated_bytes']})
            except Exception as error:
                report_errors[name] = repr(error)
        out[command] = {'registered_job_records':len(manifest['jobs']),
                        'reports':manifest['reports'],'training_runs':rows,
                        'complete_comparisons':comparisons,'trajectory_points':trajectories,
                        'report_integrity_errors':report_errors,
                        'failure_records':len(list(io_path(base/'failures').glob('*.json')))}
    return out


def report(root,state=None):
    root = Path(root).resolve()
    state = read(root/'progress.json') if state is None else state
    # status/report never certifies source correctness, never needs CUDA, never resumes work.
    result = {'schema':'gib4cl.night-report.v1','generated_at':utc(),'status':state['status'],
              'name':state['identity']['batch']['name'],'current_stage':state['stage_index'],
              'stage_results':state['stage_results'],'submissions':state['submissions'],
              'progress':progress_snapshot(root),'last_error':state.get('last_error'),
              'limits':['Time is cooperative; an in-flight epoch/trajectory may overrun.',
                        'No automatic next stage, budget expansion, retry or parameter selection.',
                        'Partial stages have no valid cross-candidate ranking.']}
    atomic(root/'morning-report.json',result)
    labels = {'running':'运行中','budget_paused':'预算暂停','stopped':'人工暂停','failed':'失败停止',
              'stage_complete':'本阶段完成，等待确认下一阶段','complete':'已完成全部登记阶段','ready':'就绪'}
    lines = ['# 夜间运行报告','',f"生成时间（UTC）：{result['generated_at']}",'',
             f"状态：**{labels.get(result['status'],result['status'])}**。",'',
             '时间预算按安全边界暂停；正在进行的一轮训练或一次 BP 可能超时。',
             '每个阶段完成后停止，需审阅报告并显式推进；失败不会自动跳过。','',
             '|阶段|状态|结果|','|---|---|---|']
    for stage in state['identity']['batch']['stages']:
        entry = state['stage_results'].get(stage['id'])
        lines.append('|'+stage['id']+'|'+('complete' if entry else '未完成')+'|'+(entry['path'] if entry else '')+'|')
    lines += ['','|批次|状态|新增工作单元|耗时/预算（秒）|超时（秒）|',
              '|---|---|---:|---:|---:|']
    for sub in state['submissions']:
        lines.append(f"|{sub['number']}|{sub['status']}|{sub['units']}|{sub.get('elapsed_seconds',0):.2f}/{sub['budget']['max_seconds']}|{sub.get('overrun_seconds',0):.2f}|")
    lines += ['','工作单元：一个训练切片（默认一轮）、一个 BP 图预测或一条定向轨迹。',
              '缓存命中不计新增单元；比较汇总和进程/数据准备耗时也计入时间预算。','']
    if result['last_error']:
        lines += ['## 最近错误','', '```text',result['last_error'],'```','']
    for kind,info in result['progress'].items():
        lines += [f"## {kind} 持久进度",'',f"已登记工作记录：{info['registered_job_records']}；失败记录：{info['failure_records']}。",'',
                  '|训练目录|已保存轮次|最好轮次|最好验证损失（仅同系数选轮次）|', '|---|---:|---:|---:|']
        for row in info['training_runs']:
            lines.append(f"|{row['run']}|{row['last_epoch']}|{row['best_epoch']}|{row['best_metric']}|")
        for name,entry in info['reports'].items():
            lines.append(f"- {name}：workspaces/{kind}/{entry['path']}")
        lines.append('')
        for name,comparison in info['complete_comparisons'].items():
            lines += [f"### {name}（{comparison['method']}）",'',
                      '|候选|RMSE（米）|95% 覆盖率|通信成本|无效协方差比例|', '|---|---:|---:|---:|---:|']
            for row in comparison['rows']:
                lines.append('|{}|{:.6g}|{:.6g}|{:.6g}|{:.6g}|'.format(row['candidate_id'],row['rmse_graph_equal'],
                    row['coverage95'],row['communication_cost_graph_mean'],row['invalid_covariance_fraction']))
            lines += ['', '权衡角色：'+str(comparison['selection']['roles']),'']
        if info['trajectory_points']:
            lines += ['### 已完成的 BP 轨迹点（不代表整个阶段已完成）','',
                      '|图|种子|拓扑|轮数|RMSE（米）|95% 覆盖率|最小 ESS|最大权重|',
                      '|---|---|---|---:|---:|---:|---:|---:|']
            for row in info['trajectory_points']:
                lines.append('|{sample_index}|{seed}|{topology}|{iterations}|{rmse:.6g}|{coverage95:.6g}|{ess_min}|{maximum_weight}|'.format(**row))
            lines.append('')
        if info['report_integrity_errors']:
            lines += ['**报告完整性检查失败，不能据此选参：** '+str(info['report_integrity_errors']),'']
    lines += ['## 如何继续','',
              '预算暂停：检查进度后用 run/start --resume 继续当前阶段。',
              '失败：先处理错误，再显式 --resume --retry-failed；不自动重试。',
              '本阶段完成：确认后用 --resume --advance 进入下一个登记阶段。',
              '服务器重启或监督进程意外退出：先用 status 检查锁；可恢复时再 --resume。',
              '共同指标排名只读取完整阶段的 comparison 报告；不要把当前验证损失跨系数排序。','']
    text = '\n'.join(lines)
    # JSON is authoritative; writing this derived view cannot affect training state.
    path = io_path(root/'morning-report.md')
    path.write_text(text,encoding='utf-8')
    return result


def stage_argv(spec,stage,root,overrides):
    argv = [stage['command'],'--config',str(resolved(spec['config'])),'--protocol',str(resolved(spec['protocol'])),
            '--output-root',str(Path(root)/'workspaces'/stage['command']),'--execute',
            '--device',spec['device'],'--threads',str(spec['threads']),'--max-jobs','1']
    options = {**stage['options'],**overrides}
    for key,value in options.items():
        if value is None:
            continue
        if key in ('support_reference','data_root'):
            value = str(resolved(value))
        argv += ['--'+key.replace('_','-')]+[str(x) for x in (value if isinstance(value,list) else [value])]
    return argv


def effective_budget(spec,args):
    budget = dict(spec['budget'])
    if args.unit_estimate_seconds is not None:
        positive(args.unit_estimate_seconds,'unit_estimate_seconds')
        budget['unit_estimate_seconds'] = args.unit_estimate_seconds
        budget['estimate_note'] = args.estimate_note or ''
    elif args.estimate_note is not None:
        raise ValueError('Estimate note override also requires --unit-estimate-seconds')
    if budget['unit_estimate_seconds'] is None or not budget['estimate_note'].strip():
        raise ValueError('Measured unit estimate and its source note required before execution')
    if budget['unit_estimate_seconds']*budget['safety_factor']>=budget['max_seconds']:
        raise ValueError('Estimated atomic unit does not fit this submission budget')
    return budget


class Control:
    def __init__(self,request):
        self.request,self.started = request,0
        remaining = max(0,request['deadline_unix']-time.time())
        self.deadline = time.monotonic()+remaining

    def stop_requested(self):
        reserve = self.request['budget']['unit_estimate_seconds']*self.request['budget']['safety_factor']
        return io_path(self.request['stop_file']).exists() or time.monotonic()+reserve>=self.deadline

    def job_started(self,key):
        self.started += 1
        atomic(self.request['activity_file'],{'at':utc(),'started_units':self.started,'key':key})


def worker(request_path):
    request = read(request_path)
    # An orphan worker holds the same OS lock as the supervisor until it exits.
    with lock(Path(request['root'])/'worker-lock'):
        control = Control(request)
        try:
            if identity(request['spec'])!=request['identity']:
                raise ValueError('Source/config/protocol changed before worker execution')
            from tools.selection_runner import main as selection_main
            argv = list(request['argv'])
            output = Path(argv[argv.index('--output-root')+1])
            if io_path(output/'manifest.json').exists():
                argv.append('--resume')
            if request['retry_failed']:
                argv.append('--retry-failed')
            value = selection_main(argv,control=control,max_training_epochs=request['budget']['epochs_per_slice'])
            if value['status'] not in ('partial','complete'):
                raise ValueError('Unexpected selection status')
            outcome = {'status':value['status'],'new_units':control.started,'selection_result':value}
        except BaseException:
            outcome = {'status':'failed','new_units':control.started,'error':traceback.format_exc()}
        atomic(request['result_file'],outcome)
        return 1 if outcome['status']=='failed' else 0


def launch_worker(root,request_path,log_path):
    env = dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8')
    with io_path(log_path).open('ab') as log:
        return subprocess.Popen([sys.executable,'-B','-m','tools.night_runner','_worker','--request',str(request_path)],
                                cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                env=env,close_fds=True)


def run_batch(args,*,launch=launch_worker):
    root = args.output_root.resolve()
    spec = load_batch(args.batch)
    current_identity = identity(spec)
    budget = effective_budget(spec,args)
    if args.resume:
        if not (root/'progress.json').exists():
            raise ValueError('No night progress to resume')
    else:
        root.mkdir(parents=True,exist_ok=False)
    with lock(root),lock(root/'worker-lock'):
        # Probe the orphan-worker lock before releasing it for our new child.
        pass
    with lock(root):
        if args.resume:
            state = read(root/'progress.json')
            if state['identity']!=current_identity:
                raise ValueError('Night resume requires identical batch, sources, config, protocol and Python')
            for entry in state['stage_results'].values():
                verify_artifact(root,entry)
            if state['status']=='complete':
                report(root,state);return state
            if state['status']=='failed' and not args.retry_failed:
                raise ValueError('Failed batch retained; explicitly --retry-failed after investigation')
            if args.advance:
                if state['status']!='stage_complete':
                    raise ValueError('Advance requires a complete preceding stage and explicit review')
                state['stage_index'] += 1
            elif state['status']=='stage_complete':
                report(root,state);return state
            for old in state['submissions']:
                if old['status']=='running':
                    old['status']='interrupted';old['recovery_note']='Supervisor exited; durable workspace will be verified on resume'
                    recovered_units = 0
                    for unit in old['workers']:
                        request = verify_artifact(root,unit['request'])
                        if io_path(request['result_file']).exists():
                            recovered_units += read(request['result_file'])['new_units']
                        elif io_path(request['activity_file']).exists():
                            recovered_units += read(request['activity_file'])['started_units']
                    old['units'] = max(old['units'],recovered_units)
                    old['elapsed_scope'] = 'last supervisor heartbeat; crash/downtime duration unknown'
        else:
            if args.advance or args.retry_failed:
                raise ValueError('Advance/retry requires --resume')
            state = {'schema':SCHEMA,'identity':current_identity,'status':'ready','stage_index':0,
                     'stage_results':{},'stage_settings':{},'submissions':[],'created_at':utc()}
        index = state['stage_index']
        stage = spec['stages'][index]
        overrides = {k:getattr(args,k) for k in ('bp_iterations','bp_chunk_size') if getattr(args,k) is not None}
        if overrides and (stage['command']!='candidates' or stage['options']['stage'] not in ('promote','final')):
            raise ValueError('BP overrides apply only to promoted/final review stages')
        previous_settings = state['stage_settings'].get(stage['id'])
        if previous_settings is not None:
            if overrides and overrides!=previous_settings:
                raise ValueError('Cannot change approved stage BP settings on resume')
            overrides = previous_settings
        merged = {**stage['options'],**overrides}
        if stage['command']=='candidates' and merged['stage'] in ('promote','final'):
            if merged.get('bp_iterations') is None or merged.get('bp_chunk_size') is None:
                raise ValueError('Review stage needs explicit BP iterations/chunk size; no default T3')
        state['stage_settings'][stage['id']] = overrides
        stop_file = root/'STOP.json'
        if io_path(stop_file).exists():
            io_path(stop_file).unlink()  # --resume explicitly acknowledges the previous stop.
        number = len(state['submissions'])+1
        folder = root/'submissions'/f'{number:04d}'
        io_path(folder).mkdir(parents=True,exist_ok=False)
        begin = time.monotonic()
        sub = {'number':number,'stage':stage['id'],'status':'running','budget':budget,'units':0,
               'started_at':utc(),'pid':os.getpid(),'workers':[],'elapsed_seconds':0,'overrun_seconds':0}
        state['submissions'].append(sub);state['status']='running';state.pop('last_error',None)
        write_state(root,state);report(root,state)
        try:
            while True:
                elapsed = time.monotonic()-begin
                reserve = budget['unit_estimate_seconds']*budget['safety_factor']
                if io_path(stop_file).exists():
                    state['status']='stopped';break
                if sub['units']>=budget['max_units'] or elapsed+reserve>=budget['max_seconds']:
                    state['status']='budget_paused';break
                if identity(spec)!=state['identity']:
                    raise ValueError('Night sources/config/protocol changed during execution')
                sequence = len(sub['workers'])+1
                request_path = folder/f'unit-{sequence:04d}-request.json'
                result_path = folder/f'unit-{sequence:04d}-result.json'
                activity_path = folder/f'unit-{sequence:04d}-activity.json'
                log_path = folder/f'unit-{sequence:04d}.log'
                request = {'spec':spec,'identity':state['identity'],'root':str(root),
                           'argv':stage_argv(spec,stage,root,overrides),'budget':budget,
                           'deadline_unix':time.time()+max(0,budget['max_seconds']-elapsed),
                           'stop_file':str(stop_file),'result_file':str(result_path),'activity_file':str(activity_path),
                           'retry_failed':args.retry_failed}
                atomic(request_path,request)
                child = launch(root,request_path,log_path)
                unit = {'pid':child.pid,'request':artifact(root,request_path),'log':str(log_path.relative_to(root)),
                        'started_at':utc()}
                sub['workers'].append(unit);write_state(root,state)
                while child.poll() is None:
                    sub['elapsed_seconds'] = time.monotonic()-begin
                    sub['overrun_seconds'] = max(0,sub['elapsed_seconds']-budget['max_seconds'])
                    write_state(root,state)
                    time.sleep(.5)
                if not io_path(result_path).exists():
                    if io_path(activity_path).exists():
                        sub['units'] += read(activity_path)['started_units']
                    raise RuntimeError(f'Worker exited {child.returncode} without a complete result; see {log_path}')
                outcome = read(result_path)
                unit.update({'returncode':child.returncode,'result':artifact(root,result_path),'finished_at':utc()})
                sub['units'] += outcome['new_units']
                write_state(root,state)
                if child.returncode!=0 or outcome['status']=='failed':
                    raise RuntimeError(outcome.get('error','Worker failed; see '+str(log_path)))
                if outcome['status']=='complete':
                    state['stage_results'][stage['id']] = artifact(root,result_path)
                    state['status']='complete' if index==len(spec['stages'])-1 else 'stage_complete'
                    break
                if outcome['new_units']==0:
                    # Time/stop admission rejected work, never spin on cached partial output.
                    state['status']='stopped' if io_path(stop_file).exists() else 'budget_paused'
                    break
        except BaseException:
            state['status']='failed';state['last_error']=traceback.format_exc()
        finally:
            sub['status']=state['status'];sub['finished_at']=utc()
            sub['elapsed_seconds']=time.monotonic()-begin
            sub['overrun_seconds']=max(0,sub['elapsed_seconds']-budget['max_seconds'])
            write_state(root,state);report(root,state)
        return state


def status(root):
    root = Path(root).resolve()
    state = read(root/'progress.json')
    active = False
    try:
        with lock(root),lock(root/'worker-lock'):
            pass
    except RuntimeError:
        active = True
    return {'status':state['status'],'active_lock':active,'stage_index':state['stage_index'],
            'last_updated_at':state['updated_at'],'supervisor_or_worker_unexpected_exit':state['status']=='running' and not active,
            'report':str(root/'morning-report.md')}


def detached(args):
    spec = load_batch(args.batch)
    effective_budget(spec,args)
    root = args.output_root.resolve()
    # The foreground implementation owns all execution validation and both OS locks.
    # A launch receipt alone is not proof of successful execution; inspect status/log.
    if not args.resume and root.exists():
        raise ValueError('New background output directory must not exist')
    if args.resume and not (root/'progress.json').is_file():
        raise ValueError('Resume requires progress.json')
    root.parent.mkdir(parents=True,exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    log_path = root.parent/(root.name+'-launch-'+stamp+'.log')
    argv = [sys.executable,'-B','-m','tools.night_runner','run','--batch',str(args.batch.resolve()),'--output-root',str(root)]
    for flag in ('resume','advance','retry_failed'):
        if getattr(args,flag):
            argv.append('--'+flag.replace('_','-'))
    for name in ('unit_estimate_seconds','estimate_note','bp_iterations','bp_chunk_size'):
        if getattr(args,name) is not None:
            argv += ['--'+name.replace('_','-'),str(getattr(args,name))]
    options = {'start_new_session':True} if os.name!='nt' else {
        'creationflags':subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW}
    with io_path(log_path).open('xb') as log:
        child = subprocess.Popen(argv,cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                 env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8'),
                                 close_fds=True,**options)
    receipt = {'pid':child.pid,'launch_log':str(log_path),'output_root':str(root),
               'note':'Detached launch requested; verify status and log. Server must remain powered on.'}
    atomic(root.parent/(root.name+'-launch-'+stamp+'.json'),receipt)
    return receipt


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest='command',required=True)
    plan = commands.add_parser('plan');plan.add_argument('--batch',type=Path,required=True)
    for name in ('run','start'):
        sub = commands.add_parser(name)
        sub.add_argument('--batch',type=Path,required=True);sub.add_argument('--output-root',type=Path,required=True)
        for flag in ('resume','advance','retry-failed'):
            sub.add_argument('--'+flag,action='store_true')
        sub.add_argument('--unit-estimate-seconds',type=float);sub.add_argument('--estimate-note')
        sub.add_argument('--bp-iterations',type=int);sub.add_argument('--bp-chunk-size',type=int)
    for name in ('status','stop','report'):
        sub = commands.add_parser(name);sub.add_argument('--output-root',type=Path,required=True)
    child = commands.add_parser('_worker',help=argparse.SUPPRESS)
    child.add_argument('--request',type=Path,required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command=='_worker':
        return worker(args.request)
    if args.command in ('run','start') and hasattr(signal,'SIGHUP'):
        signal.signal(signal.SIGHUP,signal.SIG_IGN)
    if args.command=='plan':
        spec = load_batch(args.batch)
        value = {'plan_only':True,'batch':spec,'automatic_next_stage':False,'test_access':False,
                 'note':'No processes, datasets, models or CUDA initialized; estimates required to execute.'}
    elif args.command=='run':
        state = run_batch(args);value = {'status':state['status'],'output_root':str(args.output_root.resolve())}
    elif args.command=='start':
        value = detached(args)
    elif args.command=='status':
        value = status(args.output_root)
    elif args.command=='stop':
        if not (args.output_root/'progress.json').is_file():
            raise ValueError('Stop requires an existing night run')
        atomic(args.output_root/'STOP.json',{'requested_at':utc(),'reason':'Explicit operator stop at safe boundary'})
        value = {'status':'stop_requested','note':'Wait for the current epoch/trajectory and status active_lock=false.'}
    else:
        value = report(args.output_root)
    print(json.dumps(value,ensure_ascii=False,indent=2))
    return 1 if args.command=='run' and value['status']=='failed' else 0


if __name__=='__main__':
    sys.exit(main())
