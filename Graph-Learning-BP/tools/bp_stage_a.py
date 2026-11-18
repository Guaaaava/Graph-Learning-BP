"""Explicit local-only stage A artifact builder. No CUDA, server or formal test IO."""
from __future__ import annotations
import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch

from gib4cl.config import REPO_ROOT
from gib4cl.data.dataset import graph_to_data
from gib4cl.inference import ParticleBP, ParticleSupport
from gib4cl.runtime import atomic_json, atomic_torch, sha256, source_manifest
from tools.bp_cost import workload, estimate
from tools.bp_joint_reference import prepare_joint, joint_cpu_reference
from tools.bp_sample_plan import prepare_samples
from tools.bp_timing import panel_ledger, benchmark_pairs

EVIDENCE_DIRS = (
    'kappa-ablation-20261107', 'kappa60-cost-controls-20261107',
    'positive-coefficients-20261108', 'beta-seed-check-20261108', 'three-seed-review-20261108')
REPORT_SUFFIX = Path('evidence/night-candidates-screen-20261106/workspaces/candidates/reports')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, torch.Tensor): return value.detach().cpu().tolist()
    if isinstance(value, dict): return {str(k):jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    return value


def reference_fixtures():
    # Existing independent hand graphs: no scenario data or trained weights.
    from tests.test_particle_bp import graph_case, explicit_support
    from tests.test_objectives import make_graph
    out = []
    for kind in ('chain', 'tree', 'cycle', 'forest', 'underflow'):
        if kind == 'underflow':
            data = make_graph([[0.,0.], [5.,0.]], [], [[0,1]], [-100.], [.01])
            support = ParticleSupport(torch.tensor([[[0.,0.],[1.,0.],[2.,0.]],
                [[5.,0.],[6.,0.],[7.,0.]]],dtype=torch.float64), torch.zeros(2,3,dtype=torch.float64),torch.tensor([0,1]))
            support.log_prior_weights[:,0] = -torch.inf
        else:
            data = graph_case(kind); support = explicit_support(data, 5)
        p = support.points.shape[1]
        masks = torch.stack([torch.ones(data.num_edges), torch.arange(data.num_edges)%2, torch.zeros(data.num_edges)])
        problem = prepare_joint(data, masks, ('dense', 'partial', 'empty'), support)
        joint = joint_cpu_reference(problem, iterations=4, return_history=True)
        comparisons, original = {}, {}
        for row, mid in enumerate(problem.model_ids):
            baseline = ParticleBP(p, 4, chunk_size=3).infer(data, masks[row], support=support, return_history=True)
            fields = ('mean', 'covariance', 'weights', 'log_messages', 'ess')
            original[mid] = {k:getattr(baseline,k).numpy() for k in fields}
            differences = {}
            for name in fields:
                a, b = joint['models'][mid][name], original[mid][name]
                np.testing.assert_allclose(a,b,rtol=2e-12,atol=2e-13)
                differences[name] = float(np.max(np.abs(a-b))) if a.size else 0.
            comparisons[mid] = differences
        # Store raw support/input evidence; serialize -inf prior mass as None
        # plus an explicit zero-mass mask, rather than emitting invalid JSON.
        masses = support.log_prior_weights.numpy()
        encoded_masses = [[None if not np.isfinite(x) else float(x) for x in row] for row in masses]
        out.append(dict(case=kind, identity=problem.identity,
                        inputs=dict(x=data.x, edges=data.edge_index, edge_attr=data.edge_attr,
                                    prior_covariance=data.agent_prior_cov, agent_node_index=data.agent_node_index,
                                    points=support.points, log_prior_weights=encoded_masses,
                                    null_prior_mass_means_negative_infinity=True, masks=masks),
                        original=original, joint=joint, maximum_absolute_difference=comparisons))
    return jsonable(dict(cpu_reference_only=True, gpu_speed_claim=False, cases=out))


def build(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    previous_threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        previous = read(root/'bp-local-20261110/cpu-benchmark/progress.json')['identity']['sources']
        current = source_manifest()
        if current != previous:
            raise ValueError('Production sources differ from the preceding local reference snapshot')
        paths = sorted(p for d in EVIDENCE_DIRS for p in (root/d/REPORT_SUFFIX).glob('records-*.json'))
        if len(paths) != 12: raise ValueError('Expected twelve explicitly scoped historical panels')
        ledger = panel_ledger(paths)
        atomic_json(output/'timing-ledger.json', ledger)
        columns = ('graph_index','model_id','topology','training_seed','bp_seed','agent_edges','physical_edges',
                   'inference_seconds','load_seconds','support_seconds','metrics_seconds','save_seconds')
        with (output/'timing-records.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
            for r in ledger['records']:
                row={k:r[k] for k in columns if k in r}
                row.update({k+'_seconds':v for k,v in r['phase_seconds'].items()})
                writer.writerow(row)
        atomic_json(output/'prior-cpu-benchmark-review.json', benchmark_pairs(read(root/'bp-local-20261110/cpu-benchmark/progress.json')))
        reference_path = root/'three-seed-review-20261108'/REPORT_SUFFIX/'records-three-seed-balanced-three.json'
        reference_panel = read(reference_path)
        samples, bundle = prepare_samples(reference_panel, diagnostic_regeneration=True)
        samples['reference_panel'] = dict(path=str(reference_path),sha256=sha256(reference_path))
        samples['structural_crosschecks'] = []
        for r in reference_panel['records']:
            if r['model_id'] != 'dense': continue
            i = int(r['sample_id'].rsplit(':',1)[-1]); local = samples['all_graphs'][i]
            data = graph_to_data(bundle['graphs'][i])
            truth = np.array(r['mean'])-np.array(r['metrics']['agent_error_vectors'])
            truth_difference = float(np.max(np.abs(truth-data.y.numpy())))
            edges_match = (r['metrics']['edges']==local['dense_physical_edges']
                           and r['metrics']['agent_edges']==local['dense_agent_edges'])
            samples['structural_crosschecks'].append(dict(index=i,edges_match=edges_match,truth_max_abs_difference=truth_difference))
            if not edges_match or truth_difference > 1e-10:
                raise ValueError('Rebuilt geometry is not adequate even for structural sample planning')
        atomic_json(output/'fixed-samples.json', samples)
        atomic_torch(output/'validation-prefix32-diagnostic.pt', bundle)
        atomic_json(output/'joint-cpu-reference.json', reference_fixtures())

        models200_path = root/'full200-training-20261109/evidence/full200-training-20261109/models200.json'
        models200 = read(models200_path)
        atomic_json(output/'pilot-models.json', [s for s in models200 if s['training_seed']==101])
        models = {}
        for path in paths:
            for spec in read(path)['context']['models']:
                if not spec.get('checkpoint'): continue
                key = spec['checkpoint']['sha256']
                entry = models.setdefault(key,dict(model_id=spec['model_id'],selected_epoch=spec['selected_epoch'],
                    checkpoint=spec['checkpoint'],training_seed=spec['training_seed'],target_epoch=spec['target_epoch'],
                    coefficients=spec['coefficients'],seen_in=[],weights_available_locally=False))
                entry['seen_in'].append(str(path))
        for spec in models200:
            models[spec['checkpoint']['sha256']] = dict(**spec,seen_in=[str(models200_path)],weights_available_locally=False)
        atomic_json(output/'reuse-catalog.json',dict(
            models=list(models.values()), unique_prediction_ids=sorted({r['prediction_id'] for r in ledger['records']}),
            no_automatic_cache_import=True,
            requirement='Verify original server artifact SHA, input/mask/support/budget/backend identity before explicit reuse',
            report_scope='Local archived evidence; current live 200-epoch BP panel is not included'))

        cost_source = root/'kappa60-cost-controls-20261107'/REPORT_SUFFIX/'records-kappa60-cost-controls.json'
        old = read(cost_source)['records']
        role = lambda r:'dense' if r['model_id']=='dense' else 'bfs' if r['model_id']=='bfs' else 'learned'
        rates = {name:sum(r['inference_seconds'] for r in old if role(r)==name)/sum(role(r)==name for r in old)
                 for name in ('dense','bfs','learned')}
        methods = [dict(method_id=name,role='learned' if name=='learned' else 'baseline',
            training_seeds=[101,102,103] if name=='learned' else [],seconds_per_prediction=rates[name],
            overhead_seconds=None,inference_speedup=1.,timing_source=str(cost_source)) for name in rates]
        identity = dict(particles=10000,iterations=10,dtype='float64',
                        dataset='not generated/accessed; hypothetical future test',models='not selected/frozen')
        scenarios = []
        for speed, devices, repeats, extra in ((1,1,3,False),(2,1,3,False),(3,1,3,False),(5,1,3,False),
                                               (5,3,3,False),(5,4,3,False),(5,1,1,False),(5,1,1,True)):
            adjusted = deepcopy(methods)
            for row in adjusted: row['inference_speedup']=speed
            work = workload(graphs=range(500),bp_seeds=[700,701,702][:repeats],methods=adjusted,
                            inference_identity=identity,extra_graphs=range(64) if extra else (),
                            extra_bp_seeds=[701,702] if extra else ())
            result = estimate(work,devices=devices,margin=1.3)
            for shard in result['shards']:
                shard['units']=len(shard.pop('unit_ids'))
            scenarios.append(dict(speedup_assumption=speed,devices=devices,full_graph_bp_repeats=repeats,
                                  extra_64graph_repeats=extra,**result))
        atomic_json(output/'cost-scenarios.json',dict(
            measured_source=dict(path=str(cost_source),sha256=sha256(cost_source),mean_seconds=rates),
            caveat='Historical 60-epoch masks on 16 validation graphs; not a forecast validated for 200-epoch test models. '
                   'Overhead unmeasured; margin=1.3 is an engineering allowance, not a statistical bound.',
            scenarios=scenarios,formal_test_access=False))
        example = workload(graphs=range(8),bp_seeds=[700,701,702],methods=methods,inference_identity=identity)
        atomic_json(output/'shard-contract-example.json',dict(workload=example,estimate=estimate(example,devices=3)))
        inventory = [
            ('full200','complete','Six original best/last states; no retraining for equivalent BP optimization','full200-training-result-20261110.md'),
            ('common-bp200','running_at_last_local_snapshot','Do not treat stale local snapshot as live status; completion review is stage B','common-bp200-launch-20261110.md'),
            ('beta-seeds','complete_at_60_epochs','No new beta grid needed without new evidence','three-seed-review-result-20261109.md'),
            ('kappa-mechanism','20_and_60_epoch_controls_complete','Matched final-setting 200-epoch zero-kappa evidence is conditional on intended claim','kappa60-cost-controls-review-20261108.md'),
            ('eta-controls','complete_at_60_epochs','Do not repeat broad eta sweep by default','positive-coefficients-review-20261108.md'),
            ('clip-decay','logging_complete_ablation_missing','Analyze logs first; no automatic six-model 200-epoch retraining','full200-training-result-20261110.md'),
            ('threshold','0.5_fixed_sensitivity_pending','Structure-first check; trained masks/weights needed on server; do not tune on test','Graph-Learning-BP/docs/selection-protocol.md'),
            ('formal-test','not_accessed','Freeze protocol/resources after validated backend; scope not automatically reduced','BP提速与后续实验执行计划.md')]
        atomic_json(output/'remaining-evidence.json',dict(items=[dict(item=i,status=s,next_step=n,report=p,
            report_sha256=sha256(root/p) if (root/p).is_file() else None) for i,s,n,p in inventory],
            extra_training_authorized=False,extra_server_execution=False))
        sources = {name:sha256(REPO_ROOT/'tools'/name) for name in
            ('bp_joint_reference.py','bp_timing.py','bp_cost.py','bp_sample_plan.py','bp_stage_a.py')}
        files = {p.name:sha256(p) for p in output.iterdir() if p.is_file()}
        atomic_json(output/'manifest.json',dict(status='prepared_local',production_sources_unchanged=current,
            tool_sources=sources,outputs=files,server_access=False,gpu_execution=False,formal_test_access=False))
        return dict(output=str(output),panels=len(paths),timings=ledger['unique_timing_observations'],
                    inference_hours=ledger['inference_seconds']/3600,
                    fixed_samples=[r['index'] for r in samples['expanded']],
                    exact_input_matches=samples['historical_inputs_verified'])
    finally:
        torch.set_num_threads(previous_threads)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace-root',type=Path,default=REPO_ROOT.parent)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(plan='local CPU/JSON preparation only',server=False,gpu=False,test=False)))
        return
    print(json.dumps(build(args.workspace_root,args.output),ensure_ascii=True,indent=2))


if __name__=='__main__': main()
