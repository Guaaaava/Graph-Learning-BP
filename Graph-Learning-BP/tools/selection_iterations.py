"""Targeted fixed-support BP trajectory checks, without modifying particle BP."""
from dataclasses import replace, asdict
import time
import torch

from gib4cl.data.generate import build_dataset
from gib4cl.data.dataset import graph_to_data
from gib4cl.inference import ParticleBP
from gib4cl.runtime import state_digest, sha256
from tools.selection_evaluate import masks, support_for, metric_record
from tools.selection_state import WorkLimit, read_json


def historical_reference(value,particles):
    """Accept our attestations or the unchanged original stage-13 report JSON."""
    if 'supports' in value:
        return value['supports'],[]
    if 'baseline_input_digests' not in value or 'trials' not in value or 'graphs' not in value:
        raise ValueError('Unknown support reference format')
    rows = [r for r in value['trials'] if r['condition']=='baseline' and r['particles']==particles]
    supports = {}
    for row in rows:
        i = row['sample_number']
        sample_id = value['graphs'][str(i)]['baseline']['sample_id']
        if not sample_id.endswith(':baseline'):
            raise ValueError('Unexpected stage-13 baseline sample identity')
        entry = {'sample_index':i,'sample_id':sample_id[:-len(':baseline')],
                 'seed':row['seed'],'particles':particles,
                 'input_digest':value['baseline_input_digests'][i],
                 'support_digest':row['support_digest']}
        key = (i,row['seed'])
        if key in supports and supports[key] != entry:
            raise ValueError('Historical topologies disagree on common support')
        supports[key] = entry
    return list(supports.values()),rows


def moment_change(row,reference):
    mean,ref_mean = (torch.tensor(r['mean'],dtype=torch.float64) for r in (row,reference))
    cov,ref_cov = (torch.tensor(r['covariance'],dtype=torch.float64) for r in (row,reference))
    distance = (mean-ref_mean).norm(dim=1)
    difference = torch.linalg.matrix_norm(cov-ref_cov)
    denominator = torch.linalg.matrix_norm(ref_cov)
    valid = denominator > 0
    relative = difference[valid]/denominator[valid]
    return {'position_change_mean_m':float(distance.mean()),'position_change_max_m':float(distance.max()),
            'covariance_absolute_frobenius_mean':float(difference.mean()),
            'covariance_relative_frobenius_mean':float(relative.mean()) if len(relative) else None,
            'zero_reference_covariances':int((~valid).sum()),
            'rmse_delta_m':row['metrics']['rmse']-reference['metrics']['rmse'],
            'coverage95_delta':row['coverage95']-reference['coverage95']}


@torch.no_grad()
def run_iterations(workspace,cfg,*,indices,seeds,particles,iterations,chunk,device,reference=None):
    if cfg.bp.early_stopping:
        raise ValueError('Iteration check requires fixed rounds, early_stopping=false')
    if not indices or len(set(indices))!=len(indices) or min(indices)<0 or max(indices)>=cfg.split_sizes['val']:
        raise ValueError('Graph indices must be unique and within val')
    if not seeds or len(set(seeds))!=len(seeds) or any(isinstance(s,bool) or not 0<=s<2**63 for s in seeds):
        raise ValueError('Invalid BP seeds')
    if not iterations or len(set(iterations))!=len(iterations) or min(iterations)<0 or particles<2 or chunk<1:
        raise ValueError('Invalid particle/iteration/chunk grid')
    iterations = sorted(iterations)
    reference_rows,reference_outputs = historical_reference(read_json(reference),particles) if reference else ([],[])
    request = {'indices':indices,'seeds':seeds,'particles':particles,'iterations':iterations,'chunk':chunk,
               'reference_sha256':sha256(reference) if reference else None}
    workspace.bind('iterations',request)
    bundle = build_dataset(cfg,'val',max(indices)+1)
    graphs = {i:graph_to_data(bundle['graphs'][i]) for i in indices}
    workspace.bind('iteration-inputs',{str(i):state_digest(data.to_dict()) for i,data in graphs.items()})
    trajectories,supports = [],[]
    complete = True
    try:
        for i,data_cpu in graphs.items():
            for seed in seeds:
                support,digest = support_for(data_cpu,cfg,seed,particles,device)
                input_digest = state_digest(data_cpu.to_dict())
                attestation = {'sample_index':i,'sample_id':data_cpu.sample_id,'seed':seed,'particles':particles,
                               'input_digest':input_digest,'support_digest':digest}
                if reference:
                    matches = [r for r in reference_rows if r['sample_id']==data_cpu.sample_id and r['seed']==seed]
                    if len(matches)!=1 or any(matches[0][k]!=v for k,v in attestation.items()):
                        raise ValueError('Requested historical input/support reference does not match')
                supports.append(attestation)
                data = data_cpu.clone().to(device)
                for topology,mask in masks(data).items():
                    key = {'type':'trajectory',**request,'sample_index':i,'seed':seed,
                           'topology':topology,'input_digest':input_digest,'support_digest':digest}
                    def compute():
                        bp = ParticleBP.from_config(replace(cfg.bp,num_particles=particles,
                            num_iterations=max(iterations),chunk_size=chunk))
                        if device.type=='cuda':
                            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                        begin = time.perf_counter()
                        result = bp.infer(data,mask,support=support,return_history=True)
                        if device.type=='cuda': torch.cuda.synchronize()
                        seconds = time.perf_counter()-begin
                        peak = torch.cuda.max_memory_allocated() if device.type=='cuda' else None
                        snapshots = {r['iteration']:r for r in result.history}
                        rows = []
                        for t in iterations:
                            actual = t if result.diagnostics['agent_agent_edges'] else 0
                            snapshot = snapshots[actual]
                            weights = snapshot['weights']
                            residual = snapshot['message_residual'] if actual else result.message_residual if not result.diagnostics['agent_agent_edges'] else None
                            prediction = replace(result,mean=snapshot['mean'],covariance=snapshot['covariance'],
                                weights=weights,log_messages=snapshot['log_messages'],iterations=actual,
                                message_residual=residual,converged=residual is not None and residual<=cfg.bp.message_tolerance,
                                ess=weights.square().sum(dim=1).reciprocal(),history=None,
                                covariance_min_eigenvalue=torch.linalg.eigvalsh(snapshot['covariance'])[:,0])
                            row = metric_record(data,mask,prediction.mean,prediction.covariance,cfg,prediction=prediction)
                            row.update(requested_iterations=t,actual_iterations=actual,ess=prediction.ess.cpu().tolist(),
                                maximum_weight=weights.max(dim=1).values.cpu().tolist(),message_residual=residual,
                                converged=prediction.converged,support_digest=digest)
                            rows.append(row)
                        # Verify a saved intermediate result against an independent run budget.
                        check_t = min(iterations)
                        single = ParticleBP.from_config(replace(cfg.bp,num_particles=particles,
                            num_iterations=check_t,chunk_size=chunk)).infer(data,mask,support=support)
                        snapshot = snapshots[check_t if result.diagnostics['agent_agent_edges'] else 0]
                        for name in ('mean','covariance','weights'):
                            torch.testing.assert_close(getattr(single,name),snapshot[name],rtol=1e-12,atol=1e-12)
                        reference_row = rows[-1]
                        changes = [{'iterations':r['requested_iterations'],'relative_to':max(iterations),
                                    **moment_change(r,reference_row)} for r in rows]
                        increments = [{'from':a['requested_iterations'],'to':b['requested_iterations'],
                                       **moment_change(b,a)} for a,b in zip(rows,rows[1:])]
                        historical_checked = False
                        if reference_outputs and 3 in iterations:
                            old = [r for r in reference_outputs if r['sample_number']==i and r['seed']==seed and r['topology']==topology]
                            if len(old)!=1:
                                raise ValueError('Historical T3 result missing or duplicated')
                            current = next(r for r in rows if r['requested_iterations']==3)
                            for name in ('mean','covariance'):
                                torch.testing.assert_close(torch.tensor(current[name],dtype=torch.float64),
                                    torch.tensor(old[0][name],dtype=torch.float64),rtol=1e-8,atol=1e-9)
                            historical_checked = True
                        return {'sample_index':i,'sample_id':data.sample_id,'seed':seed,'topology':topology,
                                'rows':rows,'relative_to_max':changes,'successive_changes':increments,
                                'trajectory_seconds':seconds,'cuda_peak_allocated_bytes':peak,
                                'timing_scope':'single max-round trajectory including history; independent verification extra',
                                'independent_round_checked':check_t,'verification_tolerance':1e-12,
                                'historical_T3_checked':historical_checked},[]
                    trajectories.append(workspace.job(key,compute))
    except WorkLimit:
        complete = False
    report = {'schema':'gib4cl.iteration-check.v1','status':'complete' if complete else 'partial',
              'request':request,'trajectories':trajectories,
              'algorithm':asdict(replace(cfg.bp,num_particles=particles,num_iterations=max(iterations),chunk_size=chunk)),
              'comparison_points':sum(len(t['rows']) for t in trajectories),
              'planned_comparison_points':len(indices)*len(seeds)*2*len(iterations),
              'historical_reuse':'verified_against_supplied_reference' if reference else 'unverified_new_run_same_bp_budget_stream_rule',
              'reference_note':'Largest requested T only, not exact posterior or automatic budget selection'}
    workspace.publish('support-reference',{'supports':supports})
    return workspace.publish('iterations',report)
