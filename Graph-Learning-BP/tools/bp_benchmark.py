"""Bounded validation-only BP comparisons, separate from accepted job caches.

Without --execute this prints a plan. CUDA execution refuses other GPU compute
processes. Profiling is a separate mode: its timings never count as speed tests.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
import traceback
import torch

from gib4cl.config import REPO_ROOT, config_from_dict, load_config
from gib4cl.inference import ParticleBP
from gib4cl.learning.checkpoints import load_selector
from gib4cl.runtime import (atomic_json as runtime_atomic_json, atomic_torch, dataset_identity, execution,
                            load_split, runtime_info, sha256, source_manifest,
                            state_digest, _io_path)
from tools.bp_cached import CachedParticleBP
from tools.night_runner import lock
from tools.selection_evaluate import masks, metric_record, support_for
from tools.selection_state import read_json

FIELDS = ('mean', 'covariance', 'weights', 'log_messages', 'ess',
          'covariance_min_eigenvalue', 'anchor_reachable')


def atomic_json(path, value):
    # Windows file scanners can briefly deny replacement after a read. Restrict
    # this bounded retry to this experimental tool, without modifying runtime.
    for attempt in range(6):
        try:
            return runtime_atomic_json(path, value)
        except PermissionError:
            if os.name != 'nt' or attempt == 5:
                raise
            time.sleep(.05 * 2**attempt)


def export_trace(prof, destination):
    # Kineto's native Windows writer cannot reliably open Unicode/extended paths.
    # Export through a system-temp ASCII filename, then copy with Python Unicode IO.
    fd, name = tempfile.mkstemp(prefix='bp-profiler-', suffix='.json')
    os.close(fd)
    temporary = Path(name)
    try:
        prof.export_chrome_trace(str(temporary))
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError('Profiler did not write its trace')
        _io_path(destination).write_bytes(temporary.read_bytes())
    finally:
        temporary.unlink(missing_ok=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(REPO_ROOT/'configs/experiment.json'))
    p.add_argument('--data-root', type=Path)
    p.add_argument('--graphs', default='0,6')
    p.add_argument('--models-json', type=Path, help='optional accepted model-spec list')
    p.add_argument('--checkpoint-root', type=Path, help='root of relative checkpoint paths')
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    p.add_argument('--particles', type=int, default=10000)
    p.add_argument('--iterations', type=int, default=10)
    p.add_argument('--reference-chunk', type=int, default=2048)
    p.add_argument('--variant', action='append', help='chunk:cache_MiB; repeatable')
    p.add_argument('--bp-seed', type=int, default=700)
    p.add_argument('--repeats', type=int, default=2,
                   help='each repeat rebuilds cache; repeat0 is cold and excluded from speed summary')
    p.add_argument('--rtol', type=float, default=1e-8)
    p.add_argument('--atol', type=float, default=1e-9)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--max-inferences', type=int, default=12)
    p.add_argument('--max-seconds', type=float, default=3600,
                   help='cooperative limit checked before each inference, not a forced timeout')
    p.add_argument('--profile', action='store_true', help='separate profiling run, no speed claims')
    p.add_argument('--output', type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--execute', action='store_true')
    return p


def make_plan(args):
    graphs = [int(x) for x in args.graphs.split(',')]
    if not graphs or min(graphs) < 0 or len(set(graphs)) != len(graphs):
        raise ValueError('Graph indices must be distinct and nonnegative')
    variants = []
    for text in args.variant or ['2048:4096']:
        chunk, mib = map(int, text.split(':'))
        if chunk < 1 or mib < 0:
            raise ValueError('Require positive chunk and nonnegative cache MiB')
        v = dict(name=f'chunk{chunk}-cache{mib}MiB', chunk=chunk, cache_bytes=mib*2**20)
        if v in variants:
            raise ValueError('Duplicate variant')
        variants.append(v)
    if (args.particles < 2 or args.iterations < 0 or args.reference_chunk < 1
            or args.repeats < 1 or args.threads < 1 or args.max_inferences < 1
            or not math.isfinite(args.max_seconds) or args.max_seconds <= 0
            or not math.isfinite(args.rtol) or not 0 <= args.rtol <= 1e-8
            or not math.isfinite(args.atol) or not 0 <= args.atol <= 1e-9
            or not 0 <= args.bp_seed < 2**63):
        raise ValueError('Invalid benchmark budget or overly permissive comparison tolerance')
    specs = read_json(args.models_json) if args.models_json else []
    if not isinstance(specs, list) or any(not isinstance(s, dict) for s in specs):
        raise ValueError('models-json must contain a list')
    ids = [s['model_id'] for s in specs]
    if len(ids) != len(set(ids)) or set(ids) & {'dense', 'bfs'}:
        raise ValueError('Model IDs must be distinct and cannot shadow baselines')
    return dict(schema='gib4cl.bp-performance.v1', split='val', graphs=graphs,
                particles=args.particles, iterations=args.iterations, dtype='float64',
                bp_seed=args.bp_seed, reference_chunk=args.reference_chunk,
                variants=variants, models=specs, repeats=args.repeats,
                inference_units=len(graphs)*(2+len(specs))*(1+len(variants))*args.repeats,
                rtol=args.rtol, atol=args.atol, profile=args.profile,
                cache_scope='one infer; cache construction included in timing',
                accepted_training_or_evaluation_modified=False, test_access=False)


def require_idle_cuda():
    result = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True)
    others = [line.strip() for line in result.stdout.splitlines()
              if line.strip().isdigit() and int(line.strip()) != os.getpid()]
    if others:
        raise RuntimeError('GPU compute processes are active; defer the benchmark until idle')


def snapshot(result, data, mask, cfg):
    out = {name: getattr(result, name).detach().cpu().clone() for name in FIELDS}
    out.update(iterations=result.iterations, converged=result.converged,
               message_residual=result.message_residual,
               metrics=metric_record(data, mask, result.mean, result.covariance, cfg)['metrics'])
    return out


def compare_snapshots(actual, reference, rtol, atol):
    differences = {}
    for name in FIELDS:
        x, y = actual[name], reference[name]
        torch.testing.assert_close(x, y, rtol=rtol, atol=atol, msg=lambda msg: name+': '+msg)
        both = torch.isfinite(x) & torch.isfinite(y)
        delta = (x[both].double()-y[both].double()).abs()
        differences[name] = float(delta.max()) if delta.numel() else 0.
    if actual['iterations'] != reference['iterations'] or actual['converged'] != reference['converged']:
        raise AssertionError('Iteration/convergence decisions differ')
    x, y = actual['message_residual'], reference['message_residual']
    if (x is None) != (y is None) or (x is not None and not math.isclose(x, y, rel_tol=rtol, abs_tol=atol)):
        raise AssertionError('Message residual differs')
    for key in ('covariance_status_counts', 'agent_ellipse_covered', 'nees_invalid'):
        if actual['metrics'][key] != reference['metrics'][key]:
            raise AssertionError('Diagnostic/coverage decision differs: '+key)
    return differences


def solver(cfg, plan, variant, profile=False):
    settings = replace(cfg.bp, num_particles=plan['particles'], num_iterations=plan['iterations'],
                       chunk_size=variant['chunk'], compute_dtype='float64')
    if variant['name'] == 'reference':
        return ParticleBP.from_config(settings)
    return CachedParticleBP(
        settings.num_particles, settings.num_iterations, chunk_size=settings.chunk_size,
        compute_dtype=settings.compute_dtype, measurement_variance_floor=settings.measurement_variance_floor,
        damping=settings.damping, message_tolerance=settings.message_tolerance,
        early_stopping=settings.early_stopping, cache_bytes=variant['cache_bytes'], profile_ranges=profile)


def run_unit(bp, data, mask, support, cfg, device, trace=None):
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    sync()
    before = torch.cuda.memory_allocated(device) if device.type == 'cuda' else None
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    if trace is None:
        result = bp.infer(data, mask, support=support)
        sync()
        seconds = time.perf_counter()-start
    else:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, profile_memory=True) as prof:
            result = bp.infer(data, mask, support=support)
            sync()
        seconds = time.perf_counter()-start
        export_trace(prof, trace.with_suffix('.trace.json'))
        key = 'self_cuda_time_total' if device.type == 'cuda' else 'self_cpu_time_total'
        _io_path(trace.with_suffix('.operators.txt')).write_text(
            prof.key_averages().table(sort_by=key, row_limit=40), encoding='utf-8')
    peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
    timing = dict(seconds=seconds, profiling=trace is not None,
                  cuda_allocated_before_bytes=before, cuda_peak_allocated_bytes=peak,
                  cuda_peak_increment_bytes=peak-before if peak is not None else None,
                  cache_stats=getattr(bp, 'last_cache_stats', None))
    return snapshot(result, data, mask, cfg), timing


def speed_summary(state):
    # Complete warm repetitions per CASE; aggregate by total elapsed work, never
    # by averaging per-case speedup ratios. Incomplete/profile runs stay visible.
    if state['identity']['plan']['profile']:
        return dict(speed_claims=False, reason='Profiler overhead; separate unprofiled run required')
    repeats = state['identity']['plan']['repeats']
    groups = {}
    for row in state['records'].values():
        if row['repeat'] > 0:
            groups.setdefault((row['case'], row['variant']), []).append(row['seconds'])
    rows = []
    for v in state['identity']['plan']['variants']:
        paired = []
        for case in sorted({key[0] for key in groups}):
            ref, candidate = groups.get((case, 'reference'), []), groups.get((case, v['name']), [])
            if repeats > 1 and len(ref) == len(candidate) == repeats-1:
                paired.append((statistics.median(ref), statistics.median(candidate)))
        rows.append(dict(variant=v['name'], paired_cases=len(paired),
                         speedup=sum(x for x, _ in paired)/sum(y for _, y in paired) if paired else None))
    return dict(speed_claims=any(r['paired_cases'] for r in rows), rows=rows,
                caveat='Exploratory within-device timing; repeat0 excluded; no CPU-to-CUDA extrapolation')


@torch.no_grad()
def execute(args, plan):
    if args.output is None or args.data_root is None:
        raise ValueError('--execute requires explicit --output and --data-root')
    cfg = load_config(args.config)
    if args.device == 'cuda':
        require_idle_cuda()
    device = execution(args.device, args.threads, True, None)
    ds, ds_identity = load_split(cfg, 'val', args.data_root)
    if max(plan['graphs']) >= len(ds):
        raise ValueError('Graph index is outside validation split')
    nets, checkpoints = {}, {}
    for spec in plan['models']:
        if args.checkpoint_root is None:
            raise ValueError('Model specs require --checkpoint-root')
        ref = spec['checkpoint']
        path = (args.checkpoint_root/ref['path']).resolve()
        if not path.is_relative_to(args.checkpoint_root.resolve()):
            raise ValueError('Checkpoint path escapes root')
        if sha256(path) != ref['sha256']:
            raise ValueError('Checkpoint hash differs')
        net, saved, info = load_selector(path, cfg.model, device)
        if (config_from_dict(saved['config']).data_fingerprint != cfg.data_fingerprint
                or dataset_identity(saved['datasets']['val']) != dataset_identity(ds_identity)):
            raise ValueError('Checkpoint validation data identity differs')
        nets[spec['model_id']], checkpoints[spec['model_id']] = net, info['sha256']
    identity = dict(plan=plan, config=cfg.to_dict(), dataset=dataset_identity(ds_identity),
                    checkpoints=checkpoints, sources=source_manifest(),
                    tool_sources={name: sha256(REPO_ROOT/'tools'/name) for name in
                                  ('bp_cached.py', 'bp_benchmark.py', 'selection_evaluate.py',
                                   'selection_state.py', 'night_runner.py')},
                    runtime=runtime_info(device), gpu_name=torch.cuda.get_device_name(device)
                    if device.type == 'cuda' else None)
    # Config fields may contain tuples; compare the persisted JSON representation
    # rather than rejecting an identical plan after its first serialization.
    identity = json.loads(json.dumps(identity, allow_nan=False))
    root = args.output.resolve()
    if not args.resume:
        root.mkdir(parents=True, exist_ok=False)
    with lock(root):
        path = root/'progress.json'
        if args.resume:
            state = read_json(path)
            if state['identity'] != identity:
                raise ValueError('Resume requires identical sources, data, models, plan and runtime')
            if state['status'] == 'failed':
                raise ValueError('Failed run requires investigation and a fresh output directory')
        else:
            state = dict(identity=identity, records={}, references={}, status='running')
        started, count = time.monotonic(), 0

        def publish(status):
            state.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(),
                         speed_summary=speed_summary(state))
            atomic_json(path, state)

        try:
            publish('running')
            variants = [dict(name='reference', chunk=args.reference_chunk, cache_bytes=0)]+plan['variants']
            for index in plan['graphs']:
                data = ds[index].to(device)
                mask_map = masks(data)
                threshold = math.log(cfg.evaluation.edge_threshold/(1-cfg.evaluation.edge_threshold))
                for mid, net in nets.items():
                    mask_map[mid] = (net.score_edges(data) > threshold).float()
                support_cfg = replace(cfg, bp=replace(cfg.bp, compute_dtype='float64'))
                support, digest = support_for(data, support_cfg, args.bp_seed, args.particles, device)
                for mid, mask in mask_map.items():
                    case = f'graph{index}-{mid}'
                    input_digest = state_digest(data.clone().cpu().to_dict())
                    mask_digest = state_digest({'mask': mask})
                    for variant in variants:
                        for repeat in range(args.repeats):
                            key = f'{case}/{variant["name"]}/repeat{repeat}'
                            if key in state['records']:
                                continue
                            if count >= args.max_inferences or time.monotonic()-started >= args.max_seconds:
                                publish('budget-reached')
                                return state
                            state['active_unit'] = key
                            publish('running')
                            bp = solver(cfg, plan, variant, profile=args.profile)
                            actual, timing = run_unit(bp, data, mask, support, cfg, device,
                                                      trace=root/f'profile-{len(state["records"])}'
                                                      if args.profile else None)
                            count += 1
                            if variant['name'] == 'reference' and repeat == 0:
                                label = state_digest({'model_id': mid})[:16]
                                reference_path = root/f'reference-{index}-{label}.pt'
                                atomic_torch(reference_path, actual)
                                state['references'][case] = dict(path=reference_path.name,
                                    sha256=sha256(reference_path), input_digest=input_digest,
                                    mask_digest=mask_digest, support_digest=digest)
                                reference = actual
                            else:
                                ref = state['references'][case]
                                reference_path = root/ref['path']
                                if (sha256(reference_path) != ref['sha256'] or ref['support_digest'] != digest
                                        or ref['mask_digest'] != mask_digest or ref['input_digest'] != input_digest):
                                    raise ValueError('Reference snapshot or paired inputs changed')
                                reference = torch.load(_io_path(reference_path), map_location='cpu', weights_only=True)
                            differences = compare_snapshots(actual, reference, args.rtol, args.atol)
                            state['records'][key] = dict(case=case, variant=variant['name'], repeat=repeat,
                                sample_id=data.sample_id, input_digest=input_digest, mask_digest=mask_digest,
                                support_digest=digest, differences=differences, **timing)
                            state.pop('active_unit', None)
                            publish('running')
                            del actual, reference, bp
            publish('complete')
            return state
        except Exception:
            state['error'] = traceback.format_exc()
            publish('failed')
            raise


def main(argv=None):
    args = parser().parse_args(argv)
    plan = make_plan(args)
    if not args.execute:
        print(json.dumps(dict(plan=plan, execution=False, data_opened=False,
                              per_invocation_budget=dict(inferences=args.max_inferences, seconds=args.max_seconds)),
                         ensure_ascii=True, indent=2))
        return plan
    state = execute(args, plan)
    print(json.dumps(dict(status=state['status'], records=len(state['records']),
                          speed_summary=state['speed_summary']), ensure_ascii=True, indent=2))
    return state


if __name__ == '__main__':
    main()
