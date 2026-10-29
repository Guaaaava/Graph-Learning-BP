"""One-command CPU acceptance: provenance, all tests, independent audits and panel replay."""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET

from gib4cl.config import REPO_ROOT
from gib4cl.experiments.suite import run_suite
from gib4cl.learning.checkpoints import read_checkpoint
from gib4cl.runtime import atomic_json,sha256,source_manifest
from tools.provenance import preserved_core,preserved_references,paper_sha256,provenance_scope


def structural_checks():
    forbidden = {'legacy','references','tests','tools','GNN_learning','BP','NBP_ST'}
    files = list((REPO_ROOT/'gib4cl').rglob('*.py'))
    for path in files:
        tree = ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                imports = [name.name for name in node.names]
            elif isinstance(node,ast.ImportFrom) and not node.level:
                imports = [node.module or '']
            else:
                continue
            if any(name.split('.')[0] in forbidden for name in imports):
                raise AssertionError('Production import crosses the historical/test boundary: '+str(path))
    for directory in ('GNN_learning','BP','NBP_ST','NEBP','references','results'):
        if (REPO_ROOT/directory).exists():
            raise AssertionError('Historical directory remains outside legacy/: '+directory)
    return {'production_files':len(files),'preserved_core':preserved_core(),
            'reference_source_hashes':preserved_references(),'provenance_scope':provenance_scope(),
            'paper_sha256':paper_sha256()}


def _command(root,name,arguments):
    log = root/(name+'.log')
    print('Acceptance: '+name,flush=True)
    with log.open('w',encoding='utf-8') as handle:
        code = subprocess.run([sys.executable,'-B',*arguments],cwd=REPO_ROOT,stdout=handle,
                              stderr=subprocess.STDOUT).returncode
    if code:
        raise RuntimeError(name+' failed; see '+str(log))
    return {'command':[sys.executable,'-B',*arguments],'log':str(log),'log_sha256':sha256(log)}


def _read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=reject)


def _test_workspace(root=REPO_ROOT):
    parent = Path(root)/'data'
    parent.mkdir(parents=True,exist_ok=True)
    return parent/('acpu-'+uuid.uuid4().hex[:8])


def panel_checks(root):
    spec = REPO_ROOT/'configs/suites/smoke-seeds.json'
    first = run_suite(suite_config=spec,output_root=root/'panel-a',device='cpu',threads=1,stop_after_jobs=1)
    initial = _read(first/'manifest.json')
    assert initial['status']=='partial'
    completed = initial['jobs']['train-101']['artifacts']['best_checkpoint']
    fixed_data = Path(initial['data_root'])
    hashes = {name:value['sha256'] for name,value in initial['datasets'].items()}
    run_suite(suite_config=spec,output_root=root/'panel-a',device='cpu',threads=1,resume=True)
    full = _read(first/'manifest.json')
    assert full['status']=='complete'
    assert full['jobs']['train-101']['artifacts']['best_checkpoint']==completed
    assert all(len(j['attempts'])==1 for j in full['jobs'].values())
    assert {k:v['sha256'] for k,v in full['datasets'].items()}==hashes
    before = sha256(first/'manifest.json')
    run_suite(suite_config=spec,output_root=root/'panel-a',device='cpu',threads=1,resume=True)
    assert sha256(first/'manifest.json')==before
    replay = run_suite(suite_config=spec,output_root=root/'panel-b',data_root=fixed_data,reuse_data=True,device='cpu',threads=1)
    other = _read(replay/'manifest.json')
    assert _read(first/'summary.json')==_read(replay/'summary.json')
    model_digests = {}
    for key,job in full['jobs'].items():
        counterpart = other['jobs'][key]
        if job['kind']=='train':
            left = read_checkpoint(first/job['artifacts']['best_checkpoint']['path'])['model_digest']
            right = read_checkpoint(replay/counterpart['artifacts']['best_checkpoint']['path'])['model_digest']
            assert left==right
            model_digests[str(job['training_seed'])] = left
        else:
            left = _read(first/job['artifacts']['results']['path'])
            right = _read(replay/counterpart['artifacts']['results']['path'])
            assert left['records']==right['records'] and left['summary']==right['summary']
    assert len(set(model_digests.values()))==2
    summary = _read(first/'summary.json')
    assert summary['unique_evaluation_graphs']==2 and summary['used_training_seeds']==[101,102]
    assert summary['fixed_baselines']['dense']['graphs']==2
    return {'first_panel':str(first),'replay_panel':str(replay),'planned_jobs_per_panel':6,
            'job_boundary_resume_and_idempotence_passed':True,'fixed_dataset_checksums':hashes,
            'model_digest_by_training_seed':model_digests,'replay_records_and_summaries_exact':True,
            'training_seeds':2,'bp_seeds':2,'unique_test_graphs':2,
            'summary_sha256':sha256(first/'summary.json')}


def accept(root):
    root = Path(root).resolve()
    root.mkdir(parents=True,exist_ok=False)
    result = {'status':'running',
              'scope':'CPU source layout, RNG streams, suite orchestration and mathematical/pipeline regression',
              'source_sha256':source_manifest()}
    path = root/'acceptance.json'
    atomic_json(path,result)
    try:
        if (REPO_ROOT/'PACKAGE-MANIFEST.json').is_file():
            from tools.package_project import verify_tree
            result['package_integrity'] = verify_tree(REPO_ROOT)
        else:
            result['package_integrity'] = {'mode':'working checkout; no release manifest'}
        result['layout'] = structural_checks()
        temporary = _test_workspace()
        xml = root/'tests.xml'
        result['unit_tests'] = _command(root,'unit-tests',['-m','pytest','tests','-q','-p','no:cacheprovider',
                                                        '--basetemp',str(temporary),'--junitxml',str(xml)])
        suites = ET.parse(xml).getroot().findall('testsuite')
        counts = {key:sum(int(s.attrib.get(key,0)) for s in suites)
                  for key in ('tests','failures','errors','skipped')}
        assert counts['failures']==counts['errors']==0
        skips = list(ET.parse(xml).getroot().iter('skipped'))
        assert all('CUDA runtime unavailable' in s.attrib.get('message','') for s in skips)
        result['unit_tests'].update(counts=counts,junit_sha256=sha256(xml),temporary_test_root=str(temporary))
        result['resume_audit'] = _command(root,'resume-audit',['-m','tests.audits.checkpoints','--output-root',str(root/'resume')])
        result['metrics_audit'] = _command(root,'metrics-audit',['-m','tests.audits.metrics','--output-root',str(root/'metrics')])
        result['panel_audit'] = panel_checks(root)
        if source_manifest()!=result['source_sha256']:
            raise AssertionError('Production code changed during acceptance')
        result['status'] = 'passed'
        result['limits'] = ['Current Python environment only; does not certify an independently installed dependency environment',
                            'CPU only; no GPU precision, throughput or memory acceptance',
                            'Smoke parameters and datasets only; no formal performance or BP budget recommendation',
                            'Graph and training-seed uncertainty are separate conditional summaries, not a joint confidence interval']
    except BaseException as error:
        result.update(status='failed',error=str(error))
        atomic_json(path,result)
        raise
    atomic_json(path,result)
    print(json.dumps({'acceptance':str(path),'status':result['status'],'tests':counts},ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    args = parser.parse_args()
    accept(args.output_root)
