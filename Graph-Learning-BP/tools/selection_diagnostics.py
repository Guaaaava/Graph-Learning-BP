"""Training-only scale probes and validation-only BP/MC diagnostics. Never optimizes."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time
import torch

from gib4cl.config import REPO_ROOT, ObjectiveConfig, canonical_json, fingerprint, load_config
from gib4cl.data.generate import build_dataset
from gib4cl.data.dataset import graph_to_data
from gib4cl.learning.checkpoints import load_selector, make_model, read_checkpoint
from gib4cl.learning.model import bernoulli_sigmoid
from gib4cl.learning.validation import stream_seed, uniform_noise, validation_probe
from gib4cl.learning.objectives import gib_objective
from gib4cl.inference import ParticleBP, ParticleSupport
from gib4cl.inference.spanning_tree import bfs_forest
from gib4cl.evaluation.metrics import calculate_metrics, bp_diagnostic_metrics
from gib4cl.evaluation.topology import select_topologies, selection_metrics
from gib4cl.runtime import atomic_json, execution, runtime_info, sha256, source_manifest, state_digest


def nested_support(support, particles):
    """Each node gets the exact prefix of one maximum-P prior draw."""
    if not 2 <= particles <= support.points.shape[1]:
        raise ValueError('particles must fit the maximum support')
    # This helper is only valid for the uniform prior-sampled support used here.
    expected = support.log_prior_weights.new_full(support.log_prior_weights.shape,
                                                 -math.log(support.points.shape[1]))
    if not torch.allclose(support.log_prior_weights, expected, rtol=0, atol=1e-12):
        raise ValueError('Nested prefixes require uniform prior proposal masses')
    points = support.points[:, :particles].clone()
    return ParticleSupport(points, points.new_full(points.shape[:2], -math.log(particles)),
                           support.agent_node_index.clone())


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def grad_vector(value, parameters):
    if not value.requires_grad:
        return torch.cat([torch.zeros_like(p).flatten() for p in parameters])
    grads = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                      for p, g in zip(parameters, grads)])


def scale_probe(model, data, *, tau, offset, seed, objective=None):
    """No optimizer, clipping, decay, or parameter mutation. Values are unweighted."""
    model.eval()
    logits = model.score_edges(data) + offset
    uniform = uniform_noise(logits.shape, logits.dtype, seed).to(logits.device)
    weights = bernoulli_sigmoid(logits, tau=tau, training=True, uniform=uniform)
    objective = objective or ObjectiveConfig(beta=1., gamma=1., kappa=1., eta=1.)
    result = gib_objective(logits, weights, data, objective=objective)
    terms = {k: result.per_graph[k].mean() for k in
             ('task', 'negative_entropy', 'retained_data_cost', 'expected_energy',
              'expected_triangles', 'degree')}
    terms['base_compression'] = terms['negative_entropy'] + terms['retained_data_cost']
    params = tuple(model.parameters())
    gradients = {k: grad_vector(v, params) for k, v in terms.items()}
    norms = {k: float(v.double().norm()) for k, v in gradients.items()}
    if any(not math.isfinite(v) for v in norms.values()):
        raise FloatingPointError('Nonfinite component gradient')
    p = logits.detach().sigmoid()
    energy = float(data.edge_cost.double().median()) if data.num_edges else 0.
    log_m = float(data.edge_log_cardinality.double().median()) if data.num_edges else 0.
    triangles = data.triangle_edge_index.shape[1]
    # p=.5 reference is for cost scale only; actual model probabilities are logged.
    retained_ref = .5 * float(data.edge_log_cardinality.double().sum())
    return {'values': {k: float(v.detach()) for k, v in terms.items()}, 'gradient_norms': norms,
            'zero_gradient_terms': [k for k, v in norms.items() if v == 0],
            'task_gradient_cosines': {k: safe_ratio(float(gradients['task'].double() @ v.double()), norms['task']*norms[k])
                                      for k, v in gradients.items() if k != 'task'},
            'probability_mean': float(p.mean()) if p.numel() else None,
            'objective_compute_dtype': objective.compute_dtype,
            'prediction_variance_floor_applied_edges': int((data.edge_attr[:, 1] < objective.prediction_variance_floor).sum()),
            'active_edges': int(weights.detach().sum()), 'candidate_edges': data.num_edges,
            'unavoidable_degree_deficit': float(torch.relu(3-torch.bincount(data.edge_index.flatten(),
                     minlength=data.num_nodes)[data.agent_node_index]).sum()),
            'scale_centers_not_selected': {
                'beta': safe_ratio(norms['task'], norms['base_compression']),
                'eta': safe_ratio(norms['task'], norms['degree']),
                'gamma': safe_ratio(log_m, energy),
                'kappa': safe_ratio(retained_ref, triangles * .5**3)}}


def fixed_masks(data):
    masks = {'dense': torch.ones(data.num_edges, device=data.x.device),
             'bfs': bfs_forest(data.num_nodes, data.edge_index, data.anchor_node_index).edge_weights.float()}
    u = torch.rand(data.num_edges, generator=torch.Generator().manual_seed(stream_seed('fixed-mask', data.sample_id)))
    masks.update({'sparse'+str(int(p*100)): (u < p).float().to(data.x.device) for p in (.25, .5, .75)})
    return masks


def evidence_sources():
    return {**source_manifest(), 'tools/selection_diagnostics.py': sha256(__file__)}


def process_peak_rss():
    try:
        import resource
        import sys
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    except ImportError:
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('scale', 'bp', 'validate'))
    parser.add_argument('--config', default=str(REPO_ROOT/'configs/experiment.json'))
    parser.add_argument('--protocol', default=str(REPO_ROOT/'configs/selection/protocol.json'))
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--graphs', type=int)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--temperatures', type=float, nargs='+')
    parser.add_argument('--offsets', type=float, nargs='+')
    parser.add_argument('--particles', type=int, nargs='+')
    parser.add_argument('--iterations', type=int, nargs='+')
    parser.add_argument('--chunk-sizes', type=int, nargs='+')
    parser.add_argument('--bp-stage', choices=('pilot', 'budget'), default='pilot')
    parser.add_argument('--checkpoint')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    protocol = json.loads(Path(args.protocol).read_text(encoding='utf-8'))
    section = protocol[{'scale': 'scale', 'bp': 'bp_'+args.bp_stage, 'validate': 'validation'}[args.mode]]
    graphs = args.graphs if args.graphs is not None else section['graphs']
    split = 'train' if args.mode == 'scale' else 'val'
    if isinstance(graphs, bool) or not 1 <= graphs <= cfg.split_sizes.get(split, 0):
        parser.error('graphs must fit the configured '+split+' split; test is not supported')
    seeds = args.seeds if args.seeds is not None else section.get('training_seeds', section.get('seeds', [section.get('seed', 1700)]))
    particles = args.particles or section.get('particles', [64])
    rounds = args.iterations or section.get('iterations', [3])
    chunks = args.chunk_sizes or section.get('chunk_sizes', [256])
    temperatures = args.temperatures or section.get('temperatures', [1.])
    offsets = args.offsets if args.offsets is not None else section.get('logit_offsets', [0.])
    if any(v < 0 for v in seeds) or any(v < 2 for v in particles) or any(v < 0 for v in rounds) or any(v < 1 for v in chunks) or any(v <= 0 or not math.isfinite(v) for v in temperatures) or any(not math.isfinite(v) for v in offsets):
        parser.error('Invalid seed/particle/iteration/chunk/temperature/offset values')
    settings = {'mode': args.mode, 'split': split, 'graphs': graphs, 'seeds': seeds,
                'particles': particles, 'iterations': rounds, 'chunk_sizes': chunks,
                'temperatures': temperatures, 'offsets': offsets, 'bp_stage': args.bp_stage}
    topologies = 5 + bool(args.checkpoint)
    trials = graphs*len(seeds)*(len(temperatures)*len(offsets) if args.mode == 'scale' else
              len(particles)*len(rounds)*len(chunks)*topologies if args.mode == 'bp' else 1)
    if args.mode == 'validate' and not args.checkpoint:
        parser.error('validate requires an explicit trained checkpoint')
    if args.mode == 'scale' and args.checkpoint:
        parser.error('scale measures initialization; checkpoint is only for bp/validate')
    if any(len(values) != len(set(values)) for values in
           (seeds, particles, rounds, chunks, temperatures, offsets)):
        parser.error('Duplicate grid values are not allowed')
    identity = {'settings': settings, 'config': cfg.to_dict(), 'protocol_sha256': sha256(args.protocol),
                'checkpoint_sha256': sha256(args.checkpoint) if args.checkpoint else None,
                'sources': evidence_sources(), 'device': args.device}
    if args.plan:
        value = {'plan_only': True, 'trials': trials, **settings,
                 'test_access': False, 'identity': fingerprint(identity)}
        print(json.dumps(value, indent=2)); return value
    if args.output_root is None:
        parser.error('Execution requires --output-root; --plan performs no experiment')
    target = execution(args.device, 1, True, 0)
    identity['runtime'] = runtime_info(target)
    # Dataclass tuples become lists in JSON; compare the same representation.
    identity = json.loads(canonical_json(identity))
    output = args.output_root.resolve()
    manifest_path, rows_path = output/'manifest.json', output/'rows.jsonl'
    old_rows = []
    if args.resume:
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if old['identity'] != identity:
            raise ValueError('Resume requires identical source, inputs and runtime')
        old_rows = [json.loads(path.read_text(encoding='utf-8')) for path in sorted((output/'trials').glob('*.json'))]
    else:
        output.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'identity': identity, 'planned_trials': trials,
                'completed_trials': len(old_rows), 'failed_trials': sum(r['status'] != 'ok' for r in old_rows)}
    atomic_json(manifest_path, manifest)
    existing = {row['trial_id']: row for row in old_rows}
    bundle = build_dataset(cfg, split, graphs)
    data_identity = {'metadata': bundle['metadata'],
        'graph_input_digests': [state_digest(graph_to_data(g).to_dict()) for g in bundle['graphs']]}
    if args.resume:
        saved_data = json.loads((output/'data-identity.json').read_text(encoding='utf-8'))
        if (saved_data['graph_input_digests'] != data_identity['graph_input_digests'] or
                saved_data['metadata']['dataset_id'] != bundle['metadata']['dataset_id']):
            raise ValueError('Regenerated diagnostic inputs differ on resume')
    else:
        atomic_json(output/'data-identity.json', data_identity)
    model = objective = None
    if args.checkpoint:
        saved = read_checkpoint(args.checkpoint)
        from gib4cl.config import config_from_dict
        checkpoint_cfg = config_from_dict(saved['config'])
        if checkpoint_cfg.data_fingerprint != cfg.data_fingerprint:
            raise ValueError('Checkpoint and diagnostic data scenario/seed differ')
        model, _, _ = load_selector(args.checkpoint, cfg.model, target)
        objective = checkpoint_cfg.objective.require_ready()
    def emit(key, function):
        trial_id = fingerprint(key)
        old = existing.get(trial_id)
        if old is not None and (old['status'] == 'ok' or not args.retry_failed):
            return
        begin = time.perf_counter()
        try:
            value = function()
            row = {'trial_id': trial_id, 'key': key, 'status': 'ok', **value}
        except Exception as error:
            row = {'trial_id': trial_id, 'key': key, 'status': 'failed', 'error': repr(error)}
            if target.type == 'cuda':
                torch.cuda.empty_cache()
        row['wall_seconds'] = time.perf_counter()-begin
        atomic_json(output/'trials'/(trial_id+'.json'), row)
        separator = ''
        if rows_path.exists() and rows_path.stat().st_size:
            with rows_path.open('rb') as handle:
                handle.seek(-1, 2)
                separator = '' if handle.read(1) == b'\n' else '\n'
        with rows_path.open('a', encoding='utf-8') as handle:
            handle.write(separator+json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            handle.flush()
        existing[trial_id] = row
        manifest.update(completed_trials=len(existing), failed_trials=sum(r['status'] != 'ok' for r in existing.values()))
        atomic_json(manifest_path, manifest)
    for raw in bundle['graphs']:
        data = graph_to_data(raw).to(target)
        if args.mode == 'scale':
            for seed in seeds:
                torch.manual_seed(seed)
                probe_model = make_model(cfg.model).to(target)
                for tau in temperatures:
                    for offset in offsets:
                        key = {'sample_id': data.sample_id, 'seed': seed, 'tau': tau, 'offset': offset}
                        emit(key, lambda: scale_probe(probe_model, data, tau=tau, offset=offset,
                                       seed=stream_seed('scale', seed, data.sample_id),
                                       objective=replace(cfg.objective, beta=1., gamma=1., kappa=1., eta=1.)))
        elif args.mode == 'validate':
            for seed in seeds:
                key = {'sample_id': data.sample_id, 'seed': seed}
                def validate():
                    result = validation_probe(model, data, objective, repeats=max(section['repeats']), seed=seed,
                                              threshold=cfg.evaluation.edge_threshold)
                    result['prefix_means'] = {str(k): sum(result['random_totals'][:k])/k for k in section['repeats']}
                    return result
                emit(key, validate)
        else:
            masks = fixed_masks(data)
            if model is not None:
                masks['gnn'] = select_topologies(data, model, cfg.evaluation.edge_threshold)[0]['gnn']
            for seed in seeds:
                # Draw on CPU, once at max P; all P/T/chunk/topology runs reuse it.
                cpu_support = ParticleBP.from_config(replace(cfg.bp, num_particles=max(particles), num_iterations=0)).draw_support(data.clone().cpu(),
                    generator=torch.Generator().manual_seed(stream_seed('bp-budget', seed, data.sample_id)))
                data = data.to(target)
                for p in particles:
                    prefix = nested_support(cpu_support, p)
                    support = ParticleSupport(prefix.points.to(target), prefix.log_prior_weights.to(target),
                                              prefix.agent_node_index.to(target))
                    support_id = state_digest({'points': prefix.points, 'masses': prefix.log_prior_weights})
                    for t in rounds:
                        for chunk in chunks:
                            for topology, mask in masks.items():
                                key = {'sample_id': data.sample_id, 'seed': seed, 'particles': p,
                                       'iterations': t, 'chunk': chunk, 'topology': topology}
                                def bp_trial():
                                    bp = ParticleBP.from_config(replace(cfg.bp, num_particles=p, num_iterations=t, chunk_size=chunk))
                                    if target.type == 'cuda':
                                        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                                    baseline = torch.cuda.memory_allocated() if target.type == 'cuda' else None
                                    begin = time.perf_counter()
                                    result = bp.infer(data, mask, support=support)
                                    if target.type == 'cuda': torch.cuda.synchronize()
                                    seconds = time.perf_counter()-begin
                                    peak = torch.cuda.max_memory_allocated() if target.type == 'cuda' else None
                                    metrics = calculate_metrics(result.mean, data.y, result.covariance,
                                        cfg.evaluation.outage_threshold, cfg.evaluation.alpha,
                                        covariance_rtol=cfg.evaluation.covariance_rtol,
                                        outage_thresholds=cfg.evaluation.outage_thresholds,
                                        coverage_levels=cfg.evaluation.coverage_levels)
                                    return {'inference_seconds': seconds, 'cuda_peak_allocated_bytes': peak,
                                            'cpu_process_peak_rss_bytes': process_peak_rss(),
                                            'cuda_baseline_allocated_bytes': baseline,
                                            'cuda_incremental_peak_bytes': peak-baseline if peak is not None else None,
                                            'support_digest': support_id, 'mean': result.mean.tolist(),
                                            'covariance': result.covariance.tolist(), 'ess': result.ess.tolist(),
                                            'metrics': metrics, 'topology': selection_metrics(data, mask),
                                            'bp': bp_diagnostic_metrics(result, p, cfg.evaluation.ess_fraction_threshold),
                                            'diagnostics': result.diagnostics}
                                emit(key, bp_trial)
    if evidence_sources() != identity['sources']:
        raise RuntimeError('Sources changed during diagnostic run')
    ok = sorted((r for r in existing.values() if r['status'] == 'ok'), key=lambda r: r['trial_id'])
    summary = {'trials': len(existing), 'ok': len(ok), 'failed': len(existing)-len(ok),
               'scope': 'Scale centers/resource measurements only; no formal parameters selected',
               'all_planned_trials_present': len(existing) == trials}
    if args.mode == 'scale':
        summary['center_medians_not_selected'] = {}
        for name in ('beta', 'gamma', 'kappa', 'eta'):
            values = [r['scale_centers_not_selected'][name] for r in ok if r['scale_centers_not_selected'][name] is not None]
            summary['center_medians_not_selected'][name] = float(torch.tensor(values, dtype=torch.float64).median()) if values else None
    if args.mode == 'validate':
        summary['graph_equal_weight_prefix_means'] = {str(k): sum(r['prefix_means'][str(k)] for r in ok)/len(ok) if ok else None for k in section['repeats']}
    if args.mode == 'bp':
        comparisons = []
        lookup = {fingerprint(r['key']): r for r in ok}
        for row in ok:
            for kind, values in [('particles', {'particles': max(particles)}),
                                 ('iterations', {'iterations': max(rounds)}),
                                 ('joint', {'particles': max(particles), 'iterations': max(rounds)})]:
                reference = lookup.get(fingerprint({**row['key'], **values}))
                if reference is None:
                    continue
                mean, ref_mean = (torch.tensor(r['mean'], dtype=torch.float64) for r in (row, reference))
                cov, ref_cov = (torch.tensor(r['covariance'], dtype=torch.float64) for r in (row, reference))
                rel = torch.linalg.matrix_norm(cov-ref_cov)/torch.linalg.matrix_norm(ref_cov).clamp_min(1e-30)
                comparisons.append({'trial_id': row['trial_id'], 'reference_id': reference['trial_id'],
                    'comparison': kind, 'position_change_max_m': float((mean-ref_mean).norm(dim=1).max()),
                    'covariance_relative_frobenius_mean': float(rel.mean()),
                    'covariance_relative_frobenius_max': float(rel.max()),
                    'rmse_delta_m': row['metrics']['rmse']-reference['metrics']['rmse'],
                    'coverage_delta': row['metrics']['ellipse_coverage']-reference['metrics']['ellipse_coverage']})
        atomic_json(output/'budget-comparisons.json', comparisons)
        summary['budget_comparisons'] = len(comparisons)
        summary['reference_note'] = 'Largest budget in this run only; not exact posterior or automatic selection.'
    atomic_json(output/'summary.json', summary)
    manifest['status'] = 'complete' if summary['failed'] == 0 and summary['all_planned_trials_present'] else 'failed'
    atomic_json(manifest_path, manifest)
    print(json.dumps({'output': str(output), **summary}, ensure_ascii=False))
    if manifest['status'] != 'complete':
        raise RuntimeError('Diagnostic contains failed/incomplete trials; all records retained')
    return summary


if __name__ == '__main__':
    main()
