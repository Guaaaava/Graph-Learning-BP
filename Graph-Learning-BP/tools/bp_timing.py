"""Offline timing ledger and non-overlapping phase timer; no server/GPU actions."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

PHASES = ('load', 'support', 'inference', 'metrics', 'save')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


class PhaseTimer:
    """Future runner hook. Optional GPU synchronization is supplied by caller.

    Historical missing phases remain None; never infer them as zero. Snapshot
    wall time includes gaps, synchronization and any uninstrumented work.
    """
    def __init__(self, *, clock=time.perf_counter, synchronize=lambda: None):
        self.clock, self.sync = clock, synchronize
        self.started = clock()
        self.seconds = {name: None for name in PHASES}
        self.active = None
        self.failed_phases = []

    @contextmanager
    def phase(self, name):
        if name not in PHASES or self.active is not None:
            raise ValueError('Unknown or overlapping timing phase')
        self.active = name
        try:
            self.sync()
            start = self.clock()
            try:
                yield
            except BaseException:
                self.failed_phases.append(name)
                raise
            finally:
                self.sync()
                delta = self.clock()-start
                if delta < 0: raise ValueError('Clock moved backwards')
                self.seconds[name] = (self.seconds[name] or 0.)+delta
        finally:
            self.active = None

    def snapshot(self):
        if self.active is not None:
            raise ValueError('Cannot snapshot an active phase')
        wall = self.clock()-self.started
        accounted = sum(x for x in self.seconds.values() if x is not None)
        if wall+1e-9 < accounted:
            raise ValueError('Phase time exceeds wall time')
        return dict(phase_seconds=dict(self.seconds), wall_seconds=wall,
                    uninstrumented_seconds=max(0., wall-accounted),
                    failed_phases=list(self.failed_phases))


def _stats(values):
    return dict(n=len(values), total_seconds=sum(values), mean_seconds=statistics.mean(values),
                median_seconds=statistics.median(values), minimum_seconds=min(values), maximum_seconds=max(values))


def panel_ledger(paths):
    """Deduplicate cached panel records by inference identity AND timing evidence.

    Distinct timings for identical predictions are separate observations, never
    silently averaged or counted as distinct scientific samples.
    """
    observations, predictions, files = {}, {}, []
    appearances = 0
    for path in paths:
        path = Path(path)
        data = json.loads(path.read_text(encoding='utf-8'), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        context = data.get('context', {})
        if (data.get('schema') != 'gib4cl.selection-records.v1' or data.get('status') != 'complete'
                or context.get('method') != 'particle_bp' or context.get('dataset', {}).get('split') != 'val'):
            raise ValueError('Require complete validation particle-BP record panels')
        files.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        specs = {row['model_id']: row for row in context['models']}
        budget = {k: v for k, v in context['bp_settings'].items() if k != 'seeds'}
        for r in data['records']:
            if r.get('status') != 'complete': raise ValueError('Incomplete prediction')
            seconds = r['inference_seconds']
            if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
                raise ValueError('Invalid inference timing')
            spec = specs[r['model_id']]
            key = digest(dict(sample_id=r['sample_id'], input_digest=r['input_digest'],
                support_digest=r['support_digest'], bp_seed=r['bp_seed'],
                selected_edges=r['selected_edge_ids'], model_id=r['model_id'],
                checkpoint=spec.get('checkpoint', {}).get('sha256'), budget=budget,
                dtype=context['base_config']['bp']['compute_dtype'], evaluation=context['evaluation'],
                bp_source=context['sources'].get('gib4cl/inference/particle_bp.py')))
            numerical = digest({k: r[k] for k in ('mean', 'covariance', 'mse', 'coverage95', 'communication_cost')})
            if key in predictions and predictions[key] != numerical:
                raise ValueError('Conflicting predictions under one inference identity')
            predictions[key] = numerical
            timing_key = digest([key, seconds, r.get('cuda_peak_allocated_bytes')])
            appearances += 1
            if timing_key in observations:
                observations[timing_key]['seen_in'].append(str(path.resolve()))
                continue
            metrics = r['metrics']
            observations[timing_key] = dict(
                prediction_id=key, timing_id=timing_key, sample_id=r['sample_id'],
                graph_index=int(r['sample_id'].rsplit(':', 1)[-1]), model_id=r['model_id'],
                candidate_id=spec['candidate_id'], training_seed=spec.get('training_seed'), bp_seed=r['bp_seed'],
                topology='dense' if r['model_id']=='dense' else 'bfs' if r['model_id']=='bfs' else 'learned',
                agent_edges=metrics['agent_edges'], physical_edges=metrics['edges'],
                budget=budget, phase_seconds={k: seconds if k=='inference' else None for k in PHASES},
                wall_seconds=None, seconds_per_agent_edge=seconds/metrics['agent_edges'] if metrics['agent_edges'] else None,
                hardware='not recorded in historical panel; do not mix machines without external provenance',
                seen_in=[str(path.resolve())])
    records = list(observations.values())
    summaries = {}
    for name in ('topology', 'model_id', 'graph_index', 'training_seed', 'bp_seed', 'agent_edges'):
        grouped = {}
        for r in records: grouped.setdefault(str(r[name]), []).append(r['phase_seconds']['inference'])
        summaries[name] = {k: _stats(v) for k, v in sorted(grouped.items())}
    return dict(schema='gib4cl.bp-timing-ledger.v1', scope='only supplied local validation panels',
                sources=files, record_appearances=appearances, unique_predictions=len(predictions),
                unique_timing_observations=len(records), repeated_appearances=appearances-len(records),
                inference_seconds=sum(r['phase_seconds']['inference'] for r in records),
                missing_phase_policy='None means unmeasured, never zero', summaries=summaries,
                slowest=sorted(records, key=lambda r:r['phase_seconds']['inference'], reverse=True)[:10], records=records)


def benchmark_pairs(state):
    """Warm, paired sum-of-times ratio; profiling never yields a speedup."""
    plan = state['identity']['plan']
    if plan['profile']:
        return dict(comparable=False, reason='profiling overhead', rows=[])
    rows = []
    for variant in plan['variants']:
        cases = {}
        for r in state['records'].values():
            if r['repeat'] > 0:
                cases.setdefault(r['case'], {}).setdefault(r['variant'], []).append(r['seconds'])
        paired = []
        for case, values in sorted(cases.items()):
            ref, actual = values.get('reference', []), values.get(variant['name'], [])
            if plan['repeats'] > 1 and len(ref) == len(actual) == plan['repeats']-1:
                paired.append(dict(case=case, reference=statistics.median(ref), candidate=statistics.median(actual)))
        rows.append(dict(variant=variant['name'], cases=paired,
                         speedup=sum(r['reference'] for r in paired)/sum(r['candidate'] for r in paired)
                         if paired else None))
    return dict(comparable=any(r['cases'] for r in rows), rows=rows,
                caveat='warm exploratory timings only; no CPU-to-GPU extrapolation')
