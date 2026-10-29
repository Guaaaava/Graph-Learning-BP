"""2D accuracy, covariance calibration and graph-cluster evaluation statistics."""
from __future__ import annotations
import hashlib
import json
import math
import numpy as np
import torch

METRICS_VERSION = 'gib4cl.metrics.v2'
DEFAULT_OUTAGE_THRESHOLDS = (.5,1.,2.,3.,5.,10.)
DEFAULT_COVERAGE_LEVELS = (.5,.8,.9,.95,.99)

# Scalar metrics are averaged over BP repeats within a graph, then across graphs.
SCALAR_METRICS = (
    'rmse','mae','error_p50','error_p90','error_p95','error_max',
    'outage','ellipse_coverage','ellipse_coverage_valid','nees_interval_pass',
    'nees_below_interval','nees_above_interval','mean_nees_valid','invalid_covariance_rate',
    'gaussian_nll_valid','ellipse_area_valid','predicted_rmse_valid',
    'edges','candidate_edges','edge_retention','edge_saving',
    'communication_cost','candidate_communication_cost','communication_retention','communication_saving',
    'agent_edges','anchor_edges','agent_triangles','candidate_agent_triangles','agent_triangle_retention',
    'agent_cycle_rank','connected_components','agent_components',
    'agent_degree_min','agent_degree_mean','agent_degree_max','isolated_agents',
    'agents_below_degree_three','agents_below_degree_three_fraction',
    'unanchored_agents','anchor_reachable_fraction',
    'ess_min','ess_mean','ess_fraction_min','ess_fraction_mean','low_ess_agents','low_ess_fraction',
    'bp_converged','bp_iterations','bp_message_residual',
)
PAIRED_METRICS = (
    'rmse','mae','error_p95','outage','ellipse_coverage','nees_interval_pass',
    'mean_nees_valid','invalid_covariance_rate','gaussian_nll_valid','ellipse_area_valid',
    'edges','communication_cost','agent_triangles','agent_cycle_rank',
    'unanchored_agents','ess_fraction_min','low_ess_fraction','bp_converged',
)


def metric_protocol():
    """Machine-readable scope; full definitions and units are in docs/evaluation-metrics.md."""
    return {
        'version':METRICS_VERSION,'position_unit':'m','squared_position_unit':'m^2',
        'gaussian_nll_unit':'nats, 2D density in metre coordinates, constants included',
        'communication_cost_unit':'rho * m^2; model cost, not measured joules or packet counts',
        'rate_unit':'fraction in [0,1]','scalar_metrics':list(SCALAR_METRICS),
        'conditional_metrics':['mean_nees_valid','ellipse_coverage_valid','gaussian_nll_valid',
                               'ellipse_area_valid','predicted_rmse_valid'],
        'zero_denominator_policy':'null for retention/saving ratios and conditional means; no invented 0 or 1',
        'covariance_scope':'moment-matched Gaussian ellipses and scores of BP marginals; not particle HPD regions',
        'cycle_scope':'agent-only triangles and agent-only cycle rank E_agent - A + components_agent',
        'calibration_scope':'empirical error/covariance consistency; not a directional comparison with the exact posterior',
        'uncertainty_scope':'graph bootstrap conditional on the selected training checkpoint; no training-seed uncertainty',
    }


def _grid(values,name,*,probability=False):
    if not isinstance(values,(tuple,list)) or not values:
        raise ValueError(name+' must be a nonempty sequence')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)
           or v <= 0 or (probability and v >= 1) for v in values):
        raise ValueError(name+' must contain finite positive values'+(' below 1' if probability else ''))
    result = tuple(float(v) for v in values)
    if tuple(sorted(set(result))) != result:
        raise ValueError(name+' must be strictly increasing and unique')
    return result


def _mean_valid(values):
    present = [value for value in values if value is not None]
    return math.fsum(present)/len(present) if present else None


def _finite_or_none(value):
    return value if math.isfinite(value) else None


def calculate_metrics(est_pos,true_pos,est_cov,e_th=3.,alpha=.05,*,covariance_rtol=1e-12,
                      outage_thresholds=DEFAULT_OUTAGE_THRESHOLDS,coverage_levels=DEFAULT_COVERAGE_LEVELS):
    if not est_cov.is_floating_point():
        raise ValueError('Covariance must have floating dtype')
    symmetry_rtol = 64*torch.finfo(est_cov.dtype).eps
    positions,truth,covariance = (value.detach().cpu().double() for value in (est_pos,true_pos,est_cov))
    if positions.ndim != 2:
        raise ValueError('Expected nonempty positions=(A,2), covariance=(A,2,2)')
    n = positions.shape[0]
    if positions.shape != (n,2) or truth.shape != (n,2) or covariance.shape != (n,2,2) or n < 1:
        raise ValueError('Expected nonempty positions=(A,2), covariance=(A,2,2)')
    if not torch.isfinite(positions).all() or not torch.isfinite(truth).all():
        raise ValueError('Position estimates and targets must be finite')
    if isinstance(e_th,bool) or not math.isfinite(e_th) or e_th <= 0 or not 0 < alpha < 1:
        raise ValueError('Require positive outage threshold and alpha in (0,1)')
    if not math.isfinite(covariance_rtol) or not 0 <= covariance_rtol < 1:
        raise ValueError('covariance_rtol must be in [0,1)')
    thresholds = sorted(set(_grid(outage_thresholds,'outage_thresholds')) | {float(e_th)})
    levels = sorted(set(_grid(coverage_levels,'coverage_levels',probability=True)) | {1-float(alpha)})
    errors = positions-truth
    distances = errors.norm(dim=1)
    rmse = float(torch.sqrt(distances.square().mean()))
    if not torch.isfinite(distances).all() or not math.isfinite(rmse):
        raise ValueError('Position error statistics overflow; check coordinate scale')
    # Chi-square_2 CDF is 1-exp(-q/2). No SciPy approximation or sample-fitted cutoff.
    lower,upper = -2*math.log1p(-alpha/2),-2*math.log(alpha/2)
    ellipse_upper = -2*math.log(alpha)
    nees,status,nll,area,trace_cov = [],[],[],[],[]
    for error,cov in zip(errors,covariance):
        reason = 'valid'
        value = score = ellipse_area = trace = None
        if not bool(torch.isfinite(cov).all()):
            reason = 'nonfinite_covariance'
        elif not bool((cov-cov.T).abs().max() <= symmetry_rtol*cov.abs().max()):
            reason = 'asymmetric_covariance'
        else:
            # Symmetrize only after checking the dtype-relative roundoff tolerance.
            cov = (cov/2)+(cov.T/2)
            eigenvalues = torch.linalg.eigvalsh(cov)
            if not bool(eigenvalues[0] > 0):
                reason = 'nonpositive_covariance'
            elif not bool(eigenvalues[0] > covariance_rtol*eigenvalues[-1]):
                reason = 'ill_conditioned_covariance'
            else:
                factor,info = torch.linalg.cholesky_ex(cov)
                if int(info) != 0:
                    reason = 'cholesky_failure'
                else:
                    value = float(error @ torch.cholesky_solve(error[:,None],factor)[:,0])
                    if not math.isfinite(value) or value < 0:
                        reason,value = 'nonfinite_nees',None
                    else:
                        log_sqrt_det = float(factor.diagonal().log().sum())
                        score = _finite_or_none(math.log(2*math.pi)+log_sqrt_det+value/2)
                        log_area = math.log(math.pi)+math.log(ellipse_upper)+log_sqrt_det
                        ellipse_area = math.exp(log_area) if log_area <= math.log(np.finfo(float).max) else None
                        trace = _finite_or_none(float(cov.diagonal().sum()))
        nees.append(value)
        status.append(reason)
        nll.append(score)
        area.append(ellipse_area)
        trace_cov.append(trace)
    valid = [s == 'valid' for s in status]
    coverage = [v is not None and v <= ellipse_upper for v in nees]
    interval = [v is not None and lower <= v <= upper for v in nees]
    below = [v is not None and v < lower for v in nees]
    above = [v is not None and v > upper for v in nees]
    nv = sum(valid)
    quantiles = np.percentile(distances.numpy(),[50,90,95])
    outage_curve = [{'threshold':t,'outage':float((distances > t).double().mean()),
                     'error_cdf':float((distances <= t).double().mean())} for t in thresholds]
    coverage_curve = []
    for level in levels:
        cutoff = ellipse_upper if level == 1-alpha else -2*math.log1p(-level)
        count = sum(value is not None and value <= cutoff for value in nees)
        coverage_curve.append({'nominal':level,'chi_square_threshold':cutoff,'coverage':count/n,
                               'coverage_valid':count/nv if nv else None})
    mean_trace = _mean_valid(trace_cov)
    return {
        'rmse':rmse,'mae':float(distances.mean()),
        **dict(zip(('error_p50','error_p90','error_p95'),map(float,quantiles))),
        'error_max':float(distances.max()),'outage':float((distances > e_th).double().mean()),
        'ellipse_coverage':sum(coverage)/n,'ellipse_coverage_valid':sum(coverage)/nv if nv else None,
        'nees_interval_pass':sum(interval)/n,'nees_below_interval':sum(below)/n,
        'nees_above_interval':sum(above)/n,'mean_nees_valid':_mean_valid(nees),
        'nees_invalid':n-nv,'invalid_covariance_rate':(n-nv)/n,'num_agents':n,
        'gaussian_nll_valid':_mean_valid(nll),'ellipse_area_valid':_mean_valid(area),
        'predicted_rmse_valid':math.sqrt(mean_trace) if mean_trace is not None else None,
        'covariance_status_counts':{s:status.count(s) for s in sorted(set(status))},
        'gaussian_nll_valid_agents':sum(v is not None for v in nll),
        'ellipse_area_valid_agents':sum(v is not None for v in area),
        'agent_errors':distances.tolist(),'agent_error_vectors':errors.tolist(),'agent_nees':nees,
        'agent_ellipse_covered':coverage,'agent_nees_interval_pass':interval,
        'agent_nees_below_interval':below,'agent_nees_above_interval':above,
        'agent_covariance_valid':valid,'agent_covariance_status':status,
        'agent_gaussian_nll':nll,'agent_ellipse_area':area,'agent_covariance_trace':trace_cov,
        'outage_threshold':e_th,'alpha':alpha,'covariance_rtol':covariance_rtol,
        'nees_interval':[lower,upper],'ellipse_threshold':ellipse_upper,
        'outage_curve':outage_curve,'coverage_curve':coverage_curve,
    }


def bp_diagnostic_metrics(prediction,num_particles,ess_fraction_threshold=.1):
    if isinstance(num_particles,bool) or not isinstance(num_particles,int) or num_particles < 2:
        raise ValueError('num_particles must be an integer >= 2')
    if not 0 < ess_fraction_threshold <= 1:
        raise ValueError('ess_fraction_threshold must be in (0,1]')
    ess = prediction.ess.detach().cpu().double()
    if ess.ndim != 1 or not len(ess) or not torch.isfinite(ess).all() or (ess <= 0).any():
        raise ValueError('Expected finite positive per-agent ESS')
    fraction = ess/num_particles
    # Small rounding beyond 1 is possible for uniform weights. Keep raw ESS and bound the rate.
    if (fraction > 1+1e-5).any():
        raise ValueError('ESS exceeds the particle count')
    fraction = fraction.clamp(max=1)
    low = fraction < ess_fraction_threshold
    reachable = prediction.anchor_reachable.detach().cpu()
    if reachable.shape != ess.shape or reachable.dtype != torch.bool:
        raise ValueError('Anchor reachability and ESS agent dimensions differ')
    return {'unanchored_agents':int((~reachable).sum()),'anchor_reachable_fraction':float(reachable.double().mean()),
            'ess_min':float(ess.min()),'ess_mean':float(ess.mean()),
            'ess_fraction_min':float(fraction.min()),'ess_fraction_mean':float(fraction.mean()),
            'low_ess_agents':int(low.sum()),'low_ess_fraction':float(low.double().mean()),
            'ess_fraction_threshold':ess_fraction_threshold,
            'bp_converged':int(prediction.converged),'bp_iterations':prediction.iterations,
            'bp_message_residual':prediction.message_residual}


def _rng(seed,domain):
    encoded = json.dumps(['graph-bootstrap-v2',seed,domain],separators=(',',':'))
    return np.random.default_rng(int.from_bytes(hashlib.sha256(encoded.encode()).digest()[:8],'big'))


def _summary(values,alpha,bootstrap_samples,rng):
    values = np.asarray([value for value in values if value is not None],dtype=np.float64)
    n = len(values)
    if not np.isfinite(values).all():
        raise ValueError('Scalar evaluation metrics must be finite or null')
    if not n:
        return {'mean':None,'std':None,'graphs':0,'ci_low':None,'ci_high':None}
    low = high = None
    if n >= 2 and bootstrap_samples:
        # Bound bootstrap working memory independently of dataset and resample count.
        draws = []
        for start in range(0,bootstrap_samples,128):
            indices = rng.integers(0,n,size=(min(128,bootstrap_samples-start),n))
            draws.extend(values[indices].mean(axis=1).tolist())
        low,high = map(float,np.quantile(draws,[alpha/2,1-alpha/2]))
    return {'mean':float(values.mean()),'std':float(values.std(ddof=1)) if n >= 2 else None,
            'graphs':n,'ci_low':low,'ci_high':high}


def _quantiles(values):
    if not values:
        return {'p50':None,'p90':None,'p95':None,'max':None}
    values = np.asarray(values,dtype=np.float64)
    return dict(zip(('p50','p90','p95'),map(float,np.percentile(values,[50,90,95]))),max=float(values.max()))


def _curve_stats(graphs,key,xkey,fields,stats,domain):
    first = next(iter(graphs.values()))[0]
    axis = [point[xkey] for point in first.get(key,[])]
    if any([point[xkey] for point in row.get(key,[])] != axis for rows in graphs.values() for row in rows):
        raise ValueError('All evaluation records must use the same '+key+' grid')
    result = []
    for index,value in enumerate(axis):
        entry = {xkey:value}
        for field in fields:
            per_graph = [_mean_valid([row[key][index][field] for row in rows]) for _,rows in sorted(graphs.items())]
            entry[field] = stats(per_graph,[domain,key,value,field])
        result.append(entry)
    return result


def summarize_records(records,*,alpha=.05,bootstrap_samples=2000,seed=700):
    if not records:
        raise ValueError('At least one graph is required for evaluation')
    if not 0 < alpha < 1 or isinstance(bootstrap_samples,bool) or not isinstance(bootstrap_samples,int) or bootstrap_samples < 0:
        raise ValueError('Require alpha in (0,1) and nonnegative integer bootstrap_samples')
    grouped,identities = {},set()
    protocol = None
    for row in records:
        identity = (row['topology'],row['sample_id'],row['bp_repeat'])
        if identity in identities:
            raise ValueError('Duplicate topology/sample/BP-repeat evaluation record')
        identities.add(identity)
        metrics = row['metrics']
        definitions = (metrics['outage_threshold'],metrics['alpha'],metrics['covariance_rtol'],
                       metrics.get('ess_fraction_threshold'),
                       tuple(point['threshold'] for point in metrics.get('outage_curve',[])),
                       tuple(point['nominal'] for point in metrics.get('coverage_curve',[])))
        if protocol is not None and definitions != protocol:
            raise ValueError('Cannot mix different evaluation metric definitions or curve grids')
        protocol = definitions
        if metrics['alpha'] != alpha:
            raise ValueError('Summary alpha must match the evaluated records')
        grouped.setdefault(row['topology'],{}).setdefault(row['sample_id'],[]).append(row)
    for graphs in grouped.values():
        for sample,rows in graphs.items():
            rows.sort(key=lambda row:row['bp_repeat'])
            graphs[sample] = [dict(row['metrics'],_bp_repeat=row['bp_repeat']) for row in rows]
    def stats(values,domain):
        return _summary(values,alpha,bootstrap_samples,_rng(seed,domain))
    summaries,graph_values = {},{}
    for topology,graphs in sorted(grouped.items()):
        names = [name for name in SCALAR_METRICS if any(name in row for rows in graphs.values() for row in rows)]
        graph_values[topology] = {
            sample:{name:_mean_valid([row.get(name) for row in rows]) for name in names}
            for sample,rows in sorted(graphs.items())}
        scalar_stats = {name:stats([row[name] for row in graph_values[topology].values()],[topology,name]) for name in names}
        errors = [value for sample in sorted(graphs) for row in graphs[sample] for value in row['agent_errors']]
        nees = [value for sample in sorted(graphs) for row in graphs[sample] for value in row['agent_nees']]
        invalid = sum(row['nees_invalid'] for rows in graphs.values() for row in rows)
        counts = {}
        for rows in graphs.values():
            for row in rows:
                for name,count in row.get('covariance_status_counts',{}).items():
                    counts[name] = counts.get(name,0)+count
        summary = {
            'graph_metrics':scalar_stats,'graphs':len(graphs),'bp_trials':sum(len(rows) for rows in graphs.values()),
            'agent_runs':len(errors),'invalid_covariance_agent_runs':invalid,
            'all_invalid_nees_graphs':sum(row['mean_nees_valid'] is None for row in graph_values[topology].values()),
            'pooled_agent_run_rmse':float(np.sqrt(np.mean(np.square(errors)))),
            'pooled_agent_run_mae':_mean_valid(errors),'pooled_mean_nees_valid':_mean_valid(nees),
            'agent_error_quantiles':_quantiles(errors),
            'graph_rmse_quantiles':_quantiles([row['rmse'] for row in graph_values[topology].values()]),
            'covariance_status_counts':counts,
            'bp_unconverged_trials':sum(not row['bp_converged'] for rows in graphs.values() for row in rows),
            'low_ess_agent_runs':sum(row.get('low_ess_agents',0) for rows in graphs.values() for row in rows),
            'per_graph_metrics':[{'sample_id':sample,'bp_repeats':len(graphs[sample]),'metrics':values}
                                 for sample,values in graph_values[topology].items()],
        }
        for metric,agent_key in [('gaussian_nll','agent_gaussian_nll'),('ellipse_area','agent_ellipse_area')]:
            values = [v for rows in graphs.values() for row in rows for v in row.get(agent_key,[])]
            summary['pooled_'+metric+'_valid'] = _mean_valid(values)
            summary[metric+'_valid_agent_runs'] = sum(v is not None for v in values)
        summary['outage_curve'] = _curve_stats(graphs,'outage_curve','threshold',('outage','error_cdf'),stats,topology)
        summary['coverage_curve'] = _curve_stats(graphs,'coverage_curve','nominal',('coverage','coverage_valid'),stats,topology)
        for entry in summary['coverage_curve']:
            entry['coverage_gap'] = {key:(entry['coverage'][key]-entry['nominal']
                                         if entry['coverage'][key] is not None else None)
                                     for key in ('mean','ci_low','ci_high')}
        summaries[topology] = summary
    differences = {}
    if 'dense' in graph_values:
        for other in ('bfs','gnn'):
            if other not in graph_values:
                continue
            shared = sorted(graph_values['dense'])
            if set(shared) != set(graph_values[other]) or any(
                [r['_bp_repeat'] for r in grouped[other][sample]] != [r['_bp_repeat'] for r in grouped['dense'][sample]]
                for sample in shared):
                raise ValueError('Paired topologies must have identical graph and BP-repeat identities')
            paired = {}
            for name in PAIRED_METRICS:
                if name not in next(iter(graph_values[other].values())) or name not in next(iter(graph_values['dense'].values())):
                    continue
                values = []
                for sample in shared:
                    left,right = graph_values[other][sample][name],graph_values['dense'][sample][name]
                    values.append(left-right if left is not None and right is not None else None)
                paired[name] = stats(values,['paired',other,name])
            differences[other+'_minus_dense'] = paired
    return {
        'metrics_version':METRICS_VERSION,'topologies':summaries,'paired_graph_differences':differences,
        'confidence_level':1-alpha,'bootstrap_samples':bootstrap_samples,'bootstrap_seed':seed,
        'bootstrap_unit':'independent sample_id graph; all nodes and all BP repeats remain in one cluster',
        'bootstrap_rng_rule':'sha256(graph-bootstrap-v2, seed, topology/metric); independent of row/metric traversal order',
        'mean_nees_scope':'conditional on positive-definite finite covariance; invalid counts retained',
        'invalid_covariance_policy':'invalid agents fail ellipse coverage and NEES interval; conditional means are null when all invalid',
        'conditional_aggregation':'valid agents within trial, then available BP trials within graph, then available graphs; denominators retained',
        'paired_difference_scope':'other minus dense on identical graph/repeat identities; conditional metrics use common available graphs',
        'single_graph_ci_policy':'null; graph-to-graph uncertainty cannot be estimated from one graph',
    }


