"""Directional uncertainty, weighted intervals, and paired trial bootstrap."""
from __future__ import annotations

import math
import numpy as np
import torch

from gib4cl.evaluation.metrics import calculate_metrics
from .model import seed_for


def finite_mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(v)]
    return math.fsum(values)/len(values) if values else None


def weighted_quantile(values, weights, probabilities):
    """Left-continuous inverse of the normalized weighted empirical CDF."""
    values, weights, probabilities = (np.asarray(a, dtype=np.float64) for a in (values, weights, probabilities))
    if values.ndim != 1 or weights.shape != values.shape or not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError('Invalid weighted marginal')
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Quantile probabilities must be in [0,1]')
    order = np.argsort(values, kind='mergesort')
    positive = weights[order] > 0
    ordered_values, ordered_weights = values[order][positive], weights[order][positive]
    cumulative = np.cumsum(ordered_weights/ordered_weights.sum())
    cumulative[-1] = 1.
    return ordered_values[np.minimum(np.searchsorted(cumulative, probabilities, side='left'), len(cumulative)-1)]


def covariance_comparison(covariance, reference):
    covariance, reference = np.asarray(covariance), np.asarray(reference)
    result = {'ratio_x': [], 'lambda_min': [], 'lambda_max': [], 'trace_ratio': [], 'area_ratio': []}
    for cov, ref in zip(covariance, reference):
        if not np.isfinite(ref).all() or not np.allclose(ref, ref.T, rtol=1e-12, atol=1e-14):
            raise ValueError('Reference covariance must be finite and symmetric')
        chol = np.linalg.cholesky(ref)
        if not np.isfinite(cov).all():
            for v in result.values():
                v.append(None)
            continue
        relative = np.linalg.solve(chol, cov)
        relative = np.linalg.solve(chol, relative.T).T
        eigenvalues = np.linalg.eigvalsh((relative+relative.T)/2)
        result['ratio_x'].append(float(cov[0, 0]/ref[0, 0]))
        result['lambda_min'].append(float(eigenvalues[0]))
        result['lambda_max'].append(float(eigenvalues[-1]))
        result['trace_ratio'].append(float(np.trace(cov)/np.trace(ref)))
        determinant = float(np.linalg.det(relative))
        result['area_ratio'].append(math.sqrt(determinant) if determinant >= 0 else None)
    return result


def evaluate_posterior(points, weights, mean, covariance, truth, reference, levels):
    core = calculate_metrics(torch.as_tensor(mean), torch.as_tensor(truth), torch.as_tensor(covariance),
                             coverage_levels=levels)
    directions = covariance_comparison(covariance, reference.covariance)
    covered, intervals = [], []
    for level in levels:
        bounds = np.stack([weighted_quantile(p[:, 0], w, [(1-level)/2, (1+level)/2])
                           for p, w in zip(points, weights)])
        intervals.append(bounds.tolist())
        covered.append(float(np.mean((truth[:, 0] >= bounds[:, 0]) & (truth[:, 0] <= bounds[:, 1]))))
    kl, covariance_nees, bias_nees = [], [], []
    for node, (w, target) in enumerate(zip(weights, reference.weights)):
        positive = target > 0
        with np.errstate(divide='ignore', invalid='ignore'):
            value = float(np.sum(target[positive]*(np.log(target[positive])-np.log(w[positive]))))
        kl.append(max(0., value) if math.isfinite(value) else None)
        if core['agent_covariance_valid'][node]:
            cov, ref = covariance[node], reference.covariance[node]
            delta = mean[node]-reference.mean[node]
            covariance_nees.append(float(np.trace(np.linalg.solve(cov, ref))/2))
            bias_nees.append(float(delta @ np.linalg.solve(cov, delta)/2))
        else:
            covariance_nees.append(None)
            bias_nees.append(None)
    scalar = {
        'mse': float(np.mean(np.sum((mean-truth)**2, axis=1))),
        'mean_error_to_reference': float(np.sqrt(np.mean(np.sum((mean-reference.mean)**2, axis=1)))),
        'nees_normalized': None if core['mean_nees_valid'] is None else core['mean_nees_valid']/2,
        'invalid_covariance_rate': core['invalid_covariance_rate'],
        'ellipse_coverage_95': core['ellipse_coverage'],
        'variance_ratio_x': finite_mean(directions['ratio_x']),
        **{k: finite_mean(directions[k]) for k in ('lambda_min', 'lambda_max', 'trace_ratio', 'area_ratio')},
        'kl_reference_to_bp': finite_mean(kl),
        'infinite_kl_fraction': sum(v is None for v in kl)/len(weights),
        'max_weight_error': float(np.max(np.abs(weights-reference.weights))),
        'min_ess_fraction': float(np.min(1/np.sum(weights**2, axis=1)/points.shape[1])),
        'covariance_nees_normalized': finite_mean(covariance_nees),
        'mean_bias_nees_normalized': finite_mean(bias_nees),
        'directions_low_fraction': sum(v is not None and v < 1-1e-8 for v in directions['lambda_min'])/len(weights),
        'all_directions_low_fraction': sum(v is not None and v < 1-1e-8 for v in directions['lambda_max'])/len(weights),
    }
    return {'scalar': scalar, 'coverage_x': covered,
            'agents': {**directions, 'nees': core['agent_nees'],
                       'covariance_status': core['agent_covariance_status'],
                       'intervals_x': intervals, 'kl_reference_to_bp': kl,
                       'covariance_nees_normalized': covariance_nees,
                       'mean_bias_nees_normalized': bias_nees}}


def history_metrics(history, reference):
    rows, previous = [], None
    for item in history:
        weights = item['weights'].cpu().numpy()
        comparison = covariance_comparison(item['covariance'].cpu().numpy(), reference.covariance)
        rows.append({'iteration': item['iteration'],
                     **{k: comparison[k] for k in ('ratio_x', 'lambda_min', 'lambda_max')},
                     'message_residual': item['message_residual'],
                     'belief_tv_max': None if previous is None else float(np.abs(weights-previous).sum(axis=1).max()/2)})
        previous = weights
    return rows


def bootstrap_stat(values, indices):
    values = np.array([np.nan if v is None else v for v in values], dtype=np.float64)
    valid = np.isfinite(values)
    count = int(valid.sum())
    if not count:
        return {'mean': None, 'median': None, 'std': None, 'ci_low': None, 'ci_high': None, 'valid_cases': 0}
    low = high = None
    if count >= 2 and len(indices):
        sampled = values[indices]
        present = np.isfinite(sampled)
        counts = present.sum(axis=1)
        totals = np.where(present, sampled, 0).sum(axis=1)
        means = totals[counts > 0]/counts[counts > 0]
        if len(means):
            low, high = map(float, np.percentile(means, [2.5, 97.5]))
    return {'mean': float(values[valid].mean()), 'median': float(np.median(values[valid])),
            'std': float(values[valid].std(ddof=1)) if count > 1 else None,
            'ci_low': low, 'ci_high': high, 'valid_cases': count}


def summarize_cases(cases, settings, particle_counts):
    """Average particle repeats WITHIN trials; bootstrap complete paired trials."""
    groups = {}
    for condition in settings.conditions:
        selected = sorted((c for c in cases if c['condition'] == condition.name), key=lambda c: c['case_index'])
        if not selected:
            continue
        by_particles = {}
        for p in particle_counts:
            trials = [[r for r in c['runs'] if r['particles'] == p] for c in selected]
            if any(not rows for rows in trials):
                continue
            rng = np.random.default_rng(seed_for(settings.bootstrap_seed, condition.name, p, 'paired-bootstrap'))
            indices = rng.integers(0, len(trials), (settings.bootstrap_samples, len(trials)))
            topologies, per_trial = {}, {}
            for name in settings.topologies:
                scalar_names = sorted(trials[0][0]['topologies'][name]['scalar'])
                scalars = {key: [finite_mean([r['topologies'][name]['scalar'][key] for r in rows]) for rows in trials]
                           for key in scalar_names}
                per_trial[name] = scalars
                coverage = [[finite_mean([r['topologies'][name]['coverage_x'][k] for r in rows]) for rows in trials]
                            for k in range(len(settings.coverage_levels))]
                exact_coverage = [[finite_mean([r['topologies'][name]['exact_coverage_x'][k] for r in rows]) for rows in trials]
                                  for k in range(len(settings.coverage_levels))]
                stats = {k: bootstrap_stat(v, indices) for k, v in scalars.items()}
                rmse = dict(stats['mse'])
                for key in ('mean', 'ci_low', 'ci_high'):
                    rmse[key] = None if rmse[key] is None else math.sqrt(max(0., rmse[key]))
                # Median/std of MSE are not the median/std of pooled RMSE.
                rmse.pop('std')
                rmse.pop('median')
                stats['pooled_rmse'] = rmse
                topologies[name] = {'statistics': stats,
                                    'coverage_x': [bootstrap_stat(v, indices) for v in coverage],
                                    'finite_reference_coverage_x': [bootstrap_stat(v, indices) for v in exact_coverage]}
            primary = settings.primary_loop
            paired_all = {name: {key: bootstrap_stat([None if a is None or b is None else a-b
                                                     for a, b in zip(per_trial[name][key], per_trial['tree'][key])], indices)
                                 for key in per_trial[name]} for name in settings.topologies if name.startswith('loop')}
            paired = paired_all[primary]
            converged = [[r['topologies'][primary]['scalar'] for r in rows
                          if r['topologies'][primary]['scalar']['converged']] for rows in trials]
            subset = {'runs': sum(len(rows) for rows in converged),
                      'independent_cases': sum(bool(rows) for rows in converged),
                      'statistics': {key: bootstrap_stat([finite_mean([r[key] for r in rows]) for rows in converged], indices)
                                     for key in ('variance_ratio_x', 'lambda_min', 'lambda_max', 'nees_normalized')}}
            by_particles[str(p)] = {'independent_cases': len(trials), 'particle_repeats': settings.repeats,
                                    'topologies': topologies, 'paired_loop_minus_tree': paired,
                                    'primary_loop': primary, 'paired_topology_minus_tree': paired_all,
                                    'converged_loop_subset': subset}
        groups[condition.name] = by_particles
    return groups


def nees_chi_square_summary(cases, settings, particles):
    """Supplementary NEBP-style two-sided consistency check with separate tails.

    Chi-square_2 calibration is conditional on a Gaussian error/posterior model;
    arbitrary non-Gaussian particle beliefs do not automatically satisfy it.
    Resample independent trials, never treat agents/support repeats as IID.
    """
    result = {}
    for condition in settings.conditions:
        selected = [c for c in cases if c['condition'] == condition.name]
        if not selected:
            continue
        rng = np.random.default_rng(seed_for(settings.bootstrap_seed, condition.name, 'nees-test'))
        indices = rng.integers(0, len(selected), (settings.bootstrap_samples, len(selected)))
        graphs = {}
        for name in settings.topologies:
            levels = []
            for nominal in settings.coverage_levels:
                alpha = 1-nominal
                lower, upper = -2*math.log1p(-alpha/2), -2*math.log(alpha/2)
                counts = {key: [] for key in ('below', 'inside', 'above', 'invalid')}
                for case in selected:
                    per_repeat = {key: [] for key in counts}
                    for run in case['runs']:
                        if run['particles'] != particles:
                            continue
                        values = run['topologies'][name]['agents']['nees']
                        n = len(values)
                        per_repeat['below'].append(sum(v is not None and v < lower for v in values)/n)
                        per_repeat['inside'].append(sum(v is not None and lower <= v <= upper for v in values)/n)
                        per_repeat['above'].append(sum(v is not None and v > upper for v in values)/n)
                        per_repeat['invalid'].append(sum(v is None for v in values)/n)
                    for key in counts:
                        counts[key].append(finite_mean(per_repeat[key]))
                levels.append({'nominal': nominal, 'lower': lower, 'upper': upper,
                               'expected_tail': alpha/2,
                               **{key: bootstrap_stat(value, indices) for key, value in counts.items()}})
            graphs[name] = levels
        result[condition.name] = graphs
    return {'assumption': 'Gaussian posterior/error and correct mean/covariance imply NEES ~ chi-square(2).',
            'scope': 'Supplementary consistency test. Excess upper tail is compatible with overconfidence or mean bias; two-sided rejection alone does not determine the direction.',
            'groups': result}
