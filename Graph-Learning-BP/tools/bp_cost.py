"""Planning-only workload accounting, graph sharding and strict merge contract.

No datasets, checkpoints, devices or remote services are opened. Timing speedup
applies to inference only, not to measured overhead. A margin is a planning
assumption, not a confidence interval or automatic run deadline.
"""
from __future__ import annotations
import math
from tools.bp_timing import digest


def _unique_ints(values, name):
    values = list(values)
    if (not values or len(set(values)) != len(values)
            or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in values)):
        raise ValueError(name+' must be distinct nonnegative integers')
    return values


def _number(value, name, *, zero=True):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < 0 or (not zero and value == 0)):
        raise ValueError('Invalid '+name)
    return float(value)


def workload(*, graphs, bp_seeds, methods, inference_identity, extra_graphs=(), extra_bp_seeds=(), completed=()):
    graphs = _unique_ints(graphs, 'graphs'); seeds = _unique_ints(bp_seeds, 'bp_seeds')
    extra_graphs = _unique_ints(extra_graphs, 'extra_graphs') if extra_graphs else []
    extra_bp_seeds = _unique_ints(extra_bp_seeds, 'extra_bp_seeds') if extra_bp_seeds else []
    if bool(extra_graphs) != bool(extra_bp_seeds) or not set(extra_graphs) <= set(graphs):
        raise ValueError('Extra repeats require an explicit subset and seed list')
    if not isinstance(inference_identity, dict) or not inference_identity:
        raise ValueError('Explicit inference identity/assumptions required')
    if not methods or len({r['method_id'] for r in methods}) != len(methods):
        raise ValueError('Require distinct method IDs')
    science_methods = []
    units = []
    for method in methods:
        role = method['role']
        if role not in ('baseline', 'learned'): raise ValueError('Unknown role')
        training = list(method['training_seeds'])
        if role == 'baseline' and training:
            raise ValueError('Baselines must be shared, not repeated by training seed')
        training = _unique_ints(training, 'training seeds') if role == 'learned' else [None]
        seconds = _number(method['seconds_per_prediction'], 'inference seconds')
        overhead = method.get('overhead_seconds')
        if overhead is not None: overhead = _number(overhead, 'overhead seconds')
        speedup = _number(method.get('inference_speedup', 1.), 'speedup', zero=False)
        if not method.get('timing_source'): raise ValueError('Timing source/assumption must be recorded')
        science_methods.append(dict(method_id=method['method_id'], role=role,
                                    training_seeds=training, checkpoints=method.get('checkpoints')))
        for graph in graphs:
            graph_seeds = sorted(set(seeds) | (set(extra_bp_seeds) if graph in extra_graphs else set()))
            for bp_seed in graph_seeds:
                for training_seed in training:
                    key = dict(graph=graph, method_id=method['method_id'], training_seed=training_seed, bp_seed=bp_seed)
                    units.append(dict(**key, unit_id=digest(key), role=role,
                        inference_seconds=seconds/speedup, overhead_seconds=overhead,
                        estimated_seconds=seconds/speedup+(overhead or 0.), timing_source=method['timing_source']))
    identity = digest(dict(inference=inference_identity, methods=science_methods, graphs=graphs,
                           bp_seeds=seeds, extra_graphs=extra_graphs, extra_bp_seeds=extra_bp_seeds))
    completed = list(completed)
    if len(set(completed)) != len(completed): raise ValueError('Duplicate completed unit IDs')
    if not set(completed) <= {r['unit_id'] for r in units}: raise ValueError('Completed unit outside workload')
    return dict(schema='gib4cl.bp-cost-workload.v1', planning_only=True, identity=identity,
                inference_identity=inference_identity, units=units, completed=completed,
                note='Unit IDs require this workload identity; no cache is trusted or accessed by this tool')


def estimate(work, *, devices=1, margin=1.3, setup_seconds_per_device=0., hourly_price=None):
    if isinstance(devices, bool) or not isinstance(devices, int) or devices < 1:
        raise ValueError('devices must be positive')
    margin = _number(margin, 'margin', zero=False)
    if margin < 1: raise ValueError('Planning margin must be at least 1')
    setup = _number(setup_seconds_per_device, 'setup seconds')
    completed = set(work['completed'])
    new = [r for r in work['units'] if r['unit_id'] not in completed]
    groups = {}
    for row in new: groups.setdefault(row['graph'], []).append(row)
    # Each complete graph, all BP seeds and methods, stays on one device.
    bins = [dict(device=i, graphs=[], unit_ids=[], seconds=0.) for i in range(devices)]
    for graph, rows in sorted(groups.items(), key=lambda kv:(-sum(r['estimated_seconds'] for r in kv[1]), kv[0])):
        shard = min(bins, key=lambda x:(x['seconds'], x['device']))
        shard['graphs'].append(graph); shard['unit_ids'] += [r['unit_id'] for r in rows]
        shard['seconds'] += sum(r['estimated_seconds'] for r in rows)
    active = [b for b in bins if b['unit_ids']]
    wall = max((r['seconds']+setup for r in active), default=0.)
    planned = wall*margin
    price = _number(hourly_price, 'hourly price') if hourly_price is not None else None
    return dict(planning_only=True, workload_identity=work['identity'],
                units=len(work['units']), completed_units=len(work['completed']), new_units=len(new),
                baseline_units=sum(r['role']=='baseline' for r in new),
                inference_hours=sum(r['inference_seconds'] for r in new)/3600,
                overhead_unmeasured_units=sum(r['overhead_seconds'] is None for r in new),
                estimated_wall_hours=wall/3600, planned_wall_hours=planned/3600,
                planned_billed_device_hours=planned/3600*len(active),
                planned_cost=planned/3600*len(active)*price if price is not None else None,
                margin=margin, margin_is_confidence_interval=False,
                hardware_assumption='identical independent devices; no measured multi-device overhead',
                phases_not_measured_are_not_zero=True, shards=bins)


def validate_merge(work, shards, outputs):
    """Dry-run merge contract; does not execute or publish scientific evaluation.

    Each output must carry workload identity, device, complete unit IDs and the
    input/support digests that pair methods on the same graph/BP seed.
    """
    completed = set(work['completed'])
    expected = {u['unit_id']:u for u in work['units'] if u['unit_id'] not in completed}
    if (len({s['device'] for s in shards}) != len(shards)
            or any(len(s['unit_ids']) != len(set(s['unit_ids'])) for s in shards)):
        raise ValueError('Duplicate device or unit in shard assignment')
    assignments = {s['device']:set(s['unit_ids']) for s in shards}
    if sum(map(len, assignments.values())) != len(expected) or set().union(*assignments.values()) != set(expected):
        raise ValueError('Shard assignment missing or duplicate units')
    if len({o['device'] for o in outputs}) != len(outputs): raise ValueError('Duplicate device output')
    seen, inputs, supports = {}, {}, {}
    for output in outputs:
        if output['workload_identity'] != work['identity']: raise ValueError('Workload identity differs')
        device = output['device']
        if device not in assignments: raise ValueError('Unknown device')
        for row in output['records']:
            uid = row['unit_id']
            if uid not in assignments[device] or uid in seen or row.get('status') != 'complete':
                raise ValueError('Wrong shard, duplicate or incomplete unit')
            unit = expected[uid]
            for name in ('input_digest', 'support_digest'):
                if not isinstance(row.get(name), str) or not row[name]: raise ValueError('Missing paired digest')
            g, key = unit['graph'], (unit['graph'], unit['bp_seed'])
            if g in inputs and inputs[g] != row['input_digest']: raise ValueError('Graph inputs differ')
            if key in supports and supports[key] != row['support_digest']: raise ValueError('Particle supports differ')
            inputs[g], supports[key], seen[uid] = row['input_digest'], row['support_digest'], row
    if set(seen) != set(expected): raise ValueError('Incomplete merged panel')
    return dict(planning_only=True, validated_units=len(seen), records=[seen[k] for k in sorted(seen)])
