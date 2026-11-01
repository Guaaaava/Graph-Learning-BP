"""Auditable execution, numerical pilot, trial-level persistence and reports."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from gib4cl.config import REPO_ROOT
from gib4cl.inference import ParticleBP
from gib4cl.runtime import atomic_json, runtime_info, sha256
from .config import from_dict
from .metrics import covariance_comparison, evaluate_posterior, history_metrics, summarize_cases, nees_chi_square_summary
from .model import graph_spec, as_support, make_trial, prior_points
from .reference import continuous_reference, exact_marginals


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix='.tmp-', suffix='.npz')
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_text(path, text):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix='.tmp-', suffix='.txt')
    os.close(descriptor)
    try:
        Path(temporary).write_text(text, encoding='utf-8')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def experiment_sources():
    paths = sorted(Path(__file__).parent.glob('*.py'))
    paths += [REPO_ROOT/'gib4cl/inference/particle_bp.py', REPO_ROOT/'gib4cl/evaluation/metrics.py',
              REPO_ROOT/'gib4cl/runtime.py']
    return {p.relative_to(REPO_ROOT).as_posix(): sha256(p) for p in paths}


def _safe_artifact(root, relative):
    path = (root/relative).resolve()
    if root not in path.parents:
        raise ValueError('Artifact path escapes experiment directory')
    return path


def load_cases(root, manifest, verify=True):
    root = Path(root).resolve()
    cases = []
    for item in manifest['completed_cases']:
        path = _safe_artifact(root, item['path'])
        if verify and sha256(path) != item['sha256']:
            raise ValueError('Case checksum mismatch: '+str(path))
        case = json.loads(path.read_text(encoding='utf-8'))
        if verify:
            artifacts = [a for run in case['runs'] for a in [run['artifact']]]
            if case['continuous'] is not None:
                artifacts.append(case['continuous']['artifact'])
            for artifact in artifacts:
                if sha256(_safe_artifact(root, artifact['path'])) != artifact['sha256']:
                    raise ValueError('Raw artifact checksum mismatch: '+artifact['path'])
        cases.append(case)
    return cases


def _continuous_errors(mean, covariance, reference, prior_std):
    comparison = covariance_comparison(covariance, reference.covariance)
    error = max(abs(v-1) for key in ('lambda_min', 'lambda_max') for v in comparison[key])
    return error, float(np.abs(mean-reference.mean).max()/prior_std)


def run_case(settings, condition, case_index, root, phase, particle_counts):
    trial = make_trial(condition, settings.data_seed, case_index, phase)
    spec = graph_spec(condition)
    masks = {name: spec.masks[name] for name in settings.topologies}
    reference_edges = None if condition.graph_family == 'triangle' else spec.edges
    case = {'condition': condition.name, 'case_index': case_index, 'data_seed': trial.data_seed,
            'truth': trial.truth.tolist(), 'measurements': trial.measurements.tolist(),
            'variances': trial.variances.tolist(), 'runs': [], 'continuous': None,
            'graph_family': condition.graph_family, 'edges': spec.edges.tolist(),
            'node_labels': list(spec.labels), 'focus_node': spec.focus, 'cycle_lengths': spec.cycle_lengths}
    continuous = None
    if case_index < settings.continuous_cases:
        start = time.perf_counter()
        check, continuous_points, continuous = continuous_reference(
            condition, trial, masks, settings.quadrature_orders, settings.quadrature_tolerance,
            settings.max_reference_particles, edges=reference_edges)
        arrays = {'points': continuous_points, 'truth': trial.truth, 'measurements': trial.measurements}
        for name, value in continuous.items():
            arrays.update({name+'_mean': value.mean, name+'_covariance': value.covariance,
                           name+'_weights': value.weights})
        path = root/'raw'/condition.name/f'{case_index:06d}-quadrature.npz'
        atomic_npz(path, **arrays)
        case['continuous'] = {'validation': check, 'elapsed_seconds': time.perf_counter()-start,
                              'artifact': {'path': path.relative_to(root).as_posix(), 'sha256': sha256(path)},
                              'topologies': {}}
        for name, value in continuous.items():
            stats = evaluate_posterior(continuous_points, value.weights, value.mean, value.covariance,
                                       trial.truth, value, settings.coverage_levels)
            case['continuous']['topologies'][name] = {'mean': value.mean.tolist(),
                                                     'covariance': value.covariance.tolist(),
                                                     'coverage_x': stats['coverage_x']}
    budget = max(settings.num_particles, max(settings.particle_grid))
    for repeat in range(settings.repeats):
        points_max, seed = prior_points(condition, settings.particle_seed, case_index, repeat, budget, phase)
        for p in particle_counts:
            points = points_max[:, :p].copy()
            support = as_support(points)
            log_mass = support.log_prior_weights.numpy()
            run = {'particles': p, 'repeat': repeat, 'particle_seed': seed,
                   'support_prefix_budget': budget, 'topologies': {}}
            arrays = {'points': points, 'log_prior_mass': log_mass,
                      'truth': trial.truth, 'measurements': trial.measurements, 'variances': trial.variances,
                      'edges': spec.edges}
            for name, mask in masks.items():
                start = time.perf_counter()
                reference = exact_marginals(points, log_mass, trial.measurements, trial.variances, mask,
                                            max_particles=settings.max_reference_particles, edges=reference_edges)
                reference_seconds = time.perf_counter()-start
                bp = ParticleBP(p, settings.iterations, chunk_size=settings.chunk_size,
                                compute_dtype='float64', damping=settings.damping,
                                message_tolerance=settings.message_tolerance, early_stopping=False)
                start = time.perf_counter()
                result = bp.infer(trial.graph, torch.tensor(mask), support=support, return_history=True)
                bp_seconds = time.perf_counter()-start
                mean, covariance, weights = (v.cpu().numpy() for v in (result.mean, result.covariance, result.weights))
                metrics = evaluate_posterior(points, weights, mean, covariance, trial.truth, reference, settings.coverage_levels)
                metrics['scalar'].update({'converged': int(result.converged),
                                          'iterations': result.iterations, 'message_residual': result.message_residual,
                                          'bp_seconds': bp_seconds, 'reference_seconds': reference_seconds,
                                          'continuous_covariance_error': None, 'continuous_mean_error': None,
                                          'finite_reference_continuous_covariance_error': None})
                metrics['scalar'].update({'focus_variance_ratio_x': metrics['agents']['ratio_x'][spec.focus],
                                          'focus_lambda_min': metrics['agents']['lambda_min'][spec.focus],
                                          'focus_lambda_max': metrics['agents']['lambda_max'][spec.focus]})
                if continuous is not None and case['continuous']['validation']['converged']:
                    error, mean_error = _continuous_errors(mean, covariance, continuous[name], condition.prior_std)
                    finite_error, _ = _continuous_errors(reference.mean, reference.covariance, continuous[name], condition.prior_std)
                    metrics['scalar'].update({'continuous_covariance_error': error,
                                              'continuous_mean_error': mean_error,
                                              'finite_reference_continuous_covariance_error': finite_error})
                if name not in spec.cycle_lengths and metrics['scalar']['max_weight_error'] > settings.tree_tolerance:
                    raise AssertionError(f'Tree BP/reference mismatch ({name}, P={p}): '
                                         f"{metrics['scalar']['max_weight_error']:.3g}; check iteration budget")
                metrics['exact_coverage_x'] = evaluate_posterior(
                    points, reference.weights, reference.mean, reference.covariance, trial.truth,
                    reference, settings.coverage_levels)['coverage_x']
                metrics['history'] = history_metrics(result.history, reference)
                metrics['mean'] = mean.tolist()
                metrics['covariance'] = covariance.tolist()
                metrics['reference_mean'] = reference.mean.tolist()
                metrics['reference_covariance'] = reference.covariance.tolist()
                run['topologies'][name] = metrics
                arrays.update({name+'_weights': weights, name+'_mean': mean, name+'_covariance': covariance,
                               name+'_edge_mask': np.asarray(mask),
                               name+'_reference_weights': reference.weights, name+'_reference_mean': reference.mean,
                               name+'_reference_covariance': reference.covariance,
                               name+'_history_mean': np.stack([h['mean'].cpu().numpy() for h in result.history]),
                               name+'_history_covariance': np.stack([h['covariance'].cpu().numpy() for h in result.history]),
                               name+'_log_partition_relative': np.array(reference.log_partition_relative)})
            path = root/'raw'/condition.name/f'{case_index:06d}-p{p}-r{repeat:02d}.npz'
            atomic_npz(path, **arrays)
            run['artifact'] = {'path': path.relative_to(root).as_posix(), 'sha256': sha256(path)}
            case['runs'].append(run)
    return case


def numerical_checks(cases, groups, settings, particle_counts):
    tree_errors, convergence, continuous = [], [], []
    for case in cases:
        if case['continuous'] is not None:
            continuous.append(case['continuous']['validation']['converged'])
        for run in case['runs']:
            for name, value in run['topologies'].items():
                if not name.startswith('loop'):
                    tree_errors.append(value['scalar']['max_weight_error'])
                else:
                    convergence.append(value['scalar']['converged'])
    pairs = []
    for small, large in zip(particle_counts, particle_counts[1:]):
        changes = {}
        for condition, values in groups.items():
            if str(small) not in values or str(large) not in values:
                continue
            changes[condition] = {name+':'+key: abs(values[str(small)]['topologies'][name]['statistics'][key]['mean']-
                                                   values[str(large)]['topologies'][name]['statistics'][key]['mean'])
                                  for name in settings.topologies if name.startswith('loop')
                                  for key in ('variance_ratio_x', 'lambda_min')}
        pairs.append({'particles': [small, large], 'absolute_mean_ratio_changes': changes,
                      'within_tolerance': bool(changes) and all(v <= settings.particle_ratio_tolerance
                                                               for row in changes.values() for v in row.values())})
    return {'tree_max_marginal_error': max(tree_errors) if tree_errors else None,
            'tree_tolerance': settings.tree_tolerance,
            'continuous_cases_checked': len(continuous), 'continuous_cases_converged': sum(continuous),
            'loop_runs': len(convergence), 'loop_runs_converged': sum(convergence),
            'particle_ratio_tolerance': settings.particle_ratio_tolerance, 'particle_refinements': pairs,
            'note': 'Finite-support exactness, quadrature refinement and BP convergence are separate checks.'}


def build_summary(cases, manifest, settings):
    groups = summarize_cases(cases, settings, manifest['particle_counts'])
    expected = manifest['expected_cases']
    p = settings.num_particles if settings.num_particles in manifest['particle_counts'] else max(manifest['particle_counts'])
    return {'status': 'complete' if len(cases) == expected else 'partial', 'phase': manifest['phase'],
            'purpose': settings.purpose, 'completed_cases': len(cases), 'expected_cases': expected,
            'analysis_sources': experiment_sources(),
            'reference_scope': 'Exact marginals of the SAME finite prior support, separately for each topology.',
            'continuous_scope': 'Independent Gauss-Hermite refinement; numerical stability, not an analytic solution.',
            'calibration_scope': 'x-direction equal-tail weighted empirical intervals; finite-reference curves retain discretization error.',
            'statistics_scope': 'Particle repeats averaged within independent data trials; paired trial bootstrap.',
            'groups': groups, 'numerical_checks': numerical_checks(cases, groups, settings, manifest['particle_counts']),
            'nees_consistency_supplement': nees_chi_square_summary(cases, settings, p)}


def write_tables(root, summary, cases, settings):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['condition', 'particles', 'topology', 'metric', 'mean', 'ci_low', 'ci_high', 'valid_cases'])
    lines = ['# Tree / loop particle BP', '',
             f"Purpose: **{summary['purpose']}**; phase: **{summary['phase']}**; status: **{summary['status']}**.",
             f"Data trials completed: {summary['completed_cases']}/{summary['expected_cases']}.", '',
             summary['reference_scope'], summary['statistics_scope'], '',
             '| Condition | P | Graph | Trials | Mean x variance ratio | Pooled RMSE (m) | All-agent 95% x coverage |',
             '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
    focus = summary.get('focus_calibration')
    for condition, by_p in summary['groups'].items():
        for p, group in by_p.items():
            for name, values in group['topologies'].items():
                stats = values['statistics']
                for metric, stat in stats.items():
                    writer.writerow([condition, p, name, metric, stat['mean'], stat['ci_low'], stat['ci_high'], stat['valid_cases']])
                coverage = values['coverage_x'][settings.coverage_levels.index(.95)]['mean'] if .95 in settings.coverage_levels else None
                def number(v):
                    return 'NA' if v is None else f'{v:.4f}'
                def interval(stat):
                    center = number(stat['mean'])
                    return center if stat['ci_low'] is None else f"{center} [{number(stat['ci_low'])}, {number(stat['ci_high'])}]"
                lines.append(f"| {condition} | {p} | {name} | {group['independent_cases']} | "
                             f"{interval(stats['variance_ratio_x'])} | "
                             f"{number(stats['pooled_rmse']['mean'])} | {number(coverage)} |")
    if focus:
        lines += ['', '## Focus node (used in the main mechanism and calibration figures)', '',
                  '| Condition | P | Node | Graph | x variance ratio | 95% x coverage |',
                  '| --- | ---: | --- | --- | ---: | ---: |']
        index = focus['coverage_levels'].index(.95)
        for condition,by_p in focus['groups'].items():
            for p,group in by_p.items():
                for name,value in group['topologies'].items():
                    ratio = summary['groups'][condition][p]['topologies'][name]['statistics']['focus_variance_ratio_x']
                    coverage = value['coverage_x'][index]
                    lines.append(f"| {condition} | {p} | {group['node_label']} | {name} | "
                                 f"{interval(ratio)} | {interval(coverage)} |")
        lines += ['', 'The preceding pooled table averages all agents. The focus-node table and main seven-node calibration figure use node i only.']
    lines += ['', '## Numerical checks', '', '```json', json.dumps(summary['numerical_checks'], indent=2), '```', '',
              'A finite-reference variance ratio below 1 directly identifies underestimation in that direction.',
              'Unconverged BP runs and invalid covariance rates remain in the results. '
              'Moment ellipses are Gaussian-equivalent shapes, not particle HPD regions.', '',
              'The mechanism illustration uses trial 0, particle repeat 0 and the configured focus node; aggregate curves use all completed trials.']
    atomic_text(root/'statistics.csv', stream.getvalue())
    atomic_text(root/'report.md', '\n'.join(lines)+'\n')
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['condition', 'case', 'particles', 'repeat', 'topology', 'metric', 'value'])
    for case in cases:
        for run in case['runs']:
            for name, value in run['topologies'].items():
                for key, scalar in value['scalar'].items():
                    writer.writerow([case['condition'], case['case_index'], run['particles'], run['repeat'], name, key, scalar])
    atomic_text(root/'runs.csv', stream.getvalue())


def render_outputs(root, manifest=None, verify=True, make_plots=True):
    root = Path(root).resolve()
    if manifest is None:
        manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    settings = from_dict(manifest['settings'])
    cases = load_cases(root, manifest, verify=verify)
    summary = build_summary(cases, manifest, settings)
    if cases and settings.conditions[0].graph_family=='branched7':
        from .calibration import summarize_focus_coverage
        focus = summarize_focus_coverage(root,cases,settings,manifest['particle_counts'])
        summary['focus_calibration'] = focus
        summary['calibration_scope'] = 'Main seven-node calibration uses focus node i only; pooled all-agent coverage is supplementary.'
        atomic_json(root/'focus-calibration.json',focus)
    atomic_json(root/'summary.json', summary)
    write_tables(root, summary, cases, settings)
    if make_plots and cases:
        from .plotting import plot_experiment
        plot_experiment(root, cases, summary, settings, manifest)
    atomic_json(root/'status.json', {'status': summary['status'], 'completed_cases': len(cases)})
    return summary


def run_experiment(settings, output_dir, *, phase='study', resume=False, stop_after_cases=None,
                   make_plots=True, progress=print):
    if phase not in ('study', 'pilot'):
        raise ValueError('phase must be study or pilot')
    if stop_after_cases is not None and (isinstance(stop_after_cases, bool) or not isinstance(stop_after_cases, int) or stop_after_cases < 1):
        raise ValueError('stop_after_cases must be a positive integer')
    root = Path(output_dir).resolve()
    torch.set_num_threads(settings.threads)
    sources = experiment_sources()
    particles = list(settings.particle_grid) if phase == 'pilot' else [settings.num_particles]
    counts = {c.name: min(c.cases, settings.pilot_cases) if phase == 'pilot' else c.cases for c in settings.conditions}
    if resume:
        manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        if manifest['settings'] != settings.to_dict() or manifest['phase'] != phase or manifest['sources'] != sources:
            raise ValueError('Resume requires identical settings, phase and inference/experiment sources; choose a new output directory')
        if any(manifest['runtime'][key] != runtime_info('cpu')[key] for key in ('python', 'numpy', 'torch')):
            raise ValueError('Resume runtime versions differ')
        load_cases(root, manifest, verify=True)
    else:
        root.mkdir(parents=True, exist_ok=False)
        manifest = {'settings': settings.to_dict(), 'phase': phase, 'particle_counts': particles,
                    'case_counts': counts, 'expected_cases': sum(counts.values()),
                    'sources': sources, 'runtime': runtime_info('cpu'),
                    'created_at': datetime.now(timezone.utc).isoformat(), 'completed_cases': []}
        atomic_json(root/'manifest.json', manifest)
    completed = {(item['condition'], item['case_index']) for item in manifest['completed_cases']}
    new_cases = 0
    try:
        for condition in settings.conditions:
            for index in range(counts[condition.name]):
                if (condition.name, index) in completed:
                    continue
                start = time.perf_counter()
                case = run_case(settings, condition, index, root, phase, particles)
                path = root/'cases'/condition.name/f'{index:06d}.json'
                atomic_json(path, case)
                manifest['completed_cases'].append({'condition': condition.name, 'case_index': index,
                                                    'path': path.relative_to(root).as_posix(), 'sha256': sha256(path)})
                atomic_json(root/'manifest.json', manifest)
                new_cases += 1
                if progress:
                    progress(f"[{len(manifest['completed_cases'])}/{manifest['expected_cases']}] "
                             f'{condition.name} case {index}: {time.perf_counter()-start:.2f}s', flush=True)
                if stop_after_cases is not None and new_cases >= stop_after_cases:
                    return render_outputs(root, manifest, make_plots=make_plots)
        return render_outputs(root, manifest, make_plots=make_plots)
    except Exception as error:
        atomic_json(root/'status.json', {'status': 'failed', 'completed_cases': len(manifest['completed_cases']),
                                        'error': str(error), 'error_type': type(error).__name__})
        raise
