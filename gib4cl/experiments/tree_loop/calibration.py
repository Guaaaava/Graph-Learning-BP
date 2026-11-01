"""Focus-node interval coverage from saved particles; never reruns inference."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .metrics import bootstrap_stat, weighted_quantile
from .model import graph_spec, seed_for


def summarize_focus_coverage(root, cases, settings, particle_counts):
    """Use the SAME focus node as the mechanism figure and paired trial draws.

    The caller verifies saved case/NPZ checksums before invoking this analysis.
    A finer probability grid evaluates additional quantiles of existing weights;
    it adds no data trials and no independent observations.
    """
    root = Path(root)
    levels = sorted(set(np.round(np.arange(.50,1.,.01),2).tolist()) | set(settings.coverage_levels))
    probabilities = np.array([[(1-c)/2,(1+c)/2] for c in levels]).ravel()
    groups = {}
    for condition in settings.conditions:
        selected = sorted((c for c in cases if c['condition']==condition.name), key=lambda c:c['case_index'])
        if not selected:
            continue
        spec = graph_spec(condition)
        by_particles = {}
        for p in particle_counts:
            samples = {name:{key:[] for key in ('coverage_x','finite_reference_coverage_x')}
                       for name in settings.topologies}
            for case in selected:
                repeats = {name:{key:[] for key in samples[name]} for name in settings.topologies}
                for run in case['runs']:
                    if run['particles'] != p:
                        continue
                    with np.load(root/run['artifact']['path'],allow_pickle=False) as data:
                        x = data['points'][spec.focus,:,0]
                        truth = float(case['truth'][spec.focus][0])
                        for name in settings.topologies:
                            for key,suffix in (('coverage_x','_weights'),
                                               ('finite_reference_coverage_x','_reference_weights')):
                                bounds = weighted_quantile(x,data[name+suffix][spec.focus],probabilities).reshape(-1,2)
                                repeats[name][key].append((bounds[:,0]<=truth)&(truth<=bounds[:,1]))
                for name in settings.topologies:
                    for key,values in repeats[name].items():
                        if not values:
                            raise ValueError(f'Missing particle budget {p} in focus calibration')
                        samples[name][key].append(np.mean(values,axis=0))
            rng = np.random.default_rng(seed_for(settings.bootstrap_seed,condition.name,p,'focus-calibration'))
            indices = rng.integers(0,len(selected),(settings.bootstrap_samples,len(selected)))
            graphs = {name:{key:[bootstrap_stat(np.asarray(values)[:,j].tolist(),indices)
                                  for j in range(len(levels))] for key,values in samples[name].items()}
                      for name in settings.topologies}
            by_particles[str(p)] = {'node_index':spec.focus,'node_label':spec.labels[spec.focus],
                                    'independent_cases':len(selected),'particle_repeats':settings.repeats,
                                    'topologies':graphs}
        groups[condition.name] = by_particles
    return {'scope':'Configured focus node only; x-coordinate equal-tail credible intervals.',
            'statistics_scope':'Average support repeats within each data trial; bootstrap independent paired trials.',
            'coverage_levels':levels,'groups':groups}
