"""Explicit, independent settings for the Chapter 2 diagnostic experiment."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import re


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


@dataclass(frozen=True)
class Condition:
    name: str
    centers: tuple
    cases: int
    range_std: float = .5
    prior_std: float = 1.
    graph_family: str = 'triangle'

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(r'[a-z][a-z0-9_-]{0,39}', self.name):
            raise ValueError('Condition name must be a safe lowercase identifier')
        sizes = {'triangle': 3, 'branched7': 7}
        if self.graph_family not in sizes:
            raise ValueError('graph_family must be triangle or branched7')
        if len(self.centers) != sizes[self.graph_family] or any(len(row) != 2 for row in self.centers):
            raise ValueError(f'centers must have shape ({sizes[self.graph_family]},2)')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for row in self.centers for v in row):
            raise ValueError('centers must be finite coordinates')
        object.__setattr__(self, 'centers', tuple(tuple(float(v) for v in row) for row in self.centers))
        integer(self.cases, 'cases', 1)
        positive(self.range_std, 'range_std')
        positive(self.prior_std, 'prior_std')


@dataclass(frozen=True)
class Settings:
    conditions: tuple
    purpose: str = 'study'
    num_particles: int = 256
    particle_grid: tuple = (128, 256, 512)
    repeats: int = 3
    iterations: int = 50
    chunk_size: int = 256
    message_tolerance: float = 1e-8
    damping: float = 0.
    data_seed: int = 1029
    particle_seed: int = 700
    bootstrap_seed: int = 1700
    bootstrap_samples: int = 2000
    coverage_levels: tuple = (.5, .8, .9, .95, .99)
    pilot_cases: int = 20
    continuous_cases: int = 20
    quadrature_orders: tuple = (9, 13, 17, 25, 33, 41)
    quadrature_tolerance: float = .01
    particle_ratio_tolerance: float = .02
    max_reference_particles: int = 2048
    tree_tolerance: float = 1e-8
    topologies: tuple = ('tree', 'loop')
    threads: int = 1
    primary_loop: str = 'loop'
    plot_rounds: int = 15

    def __post_init__(self):
        if not self.conditions or not all(isinstance(c, Condition) for c in self.conditions):
            raise ValueError('At least one Condition is required')
        if len({c.name for c in self.conditions}) != len(self.conditions):
            raise ValueError('Condition names must be unique')
        families = {c.graph_family for c in self.conditions}
        if len(families) != 1:
            raise ValueError('Use one graph_family per experiment configuration')
        if self.purpose not in ('study', 'smoke'):
            raise ValueError('purpose must be study or smoke')
        for name in ('num_particles', 'max_reference_particles'):
            integer(getattr(self, name), name, 4)
        for name in ('repeats', 'iterations', 'chunk_size', 'bootstrap_samples', 'pilot_cases', 'threads'):
            integer(getattr(self, name), name, 1)
        integer(self.continuous_cases, 'continuous_cases', 0)
        integer(self.plot_rounds, 'plot_rounds', 1)
        for name in ('data_seed', 'particle_seed', 'bootstrap_seed'):
            integer(getattr(self, name), name)
            if getattr(self, name) >= 2**63:
                raise ValueError(f'{name} must be below 2**63')
        for name, minimum in (('particle_grid', 4), ('quadrature_orders', 2)):
            values = getattr(self, name)
            if not values or tuple(sorted(set(values))) != tuple(values):
                raise ValueError(f'{name} must be strictly increasing and nonempty')
            for v in values:
                integer(v, name, minimum)
        if max(self.num_particles, max(self.particle_grid), max(self.quadrature_orders)**2) > self.max_reference_particles:
            raise ValueError('Requested particles/quadrature exceed max_reference_particles')
        if len(self.quadrature_orders) < 3:
            raise ValueError('At least three quadrature orders are needed for two refinement checks')
        levels = self.coverage_levels
        if not levels or tuple(sorted(set(levels))) != tuple(levels):
            raise ValueError('coverage_levels must be strictly increasing')
        for v in levels:
            if positive(v, 'coverage level') >= 1:
                raise ValueError('coverage levels must be below 1')
        for name in ('message_tolerance', 'quadrature_tolerance', 'particle_ratio_tolerance', 'tree_tolerance'):
            positive(getattr(self, name), name)
        if isinstance(self.damping, bool) or not math.isfinite(self.damping) or not 0 <= self.damping < 1:
            raise ValueError('damping must be in [0,1)')
        allowed = ({'tree', 'loop', 'tree_01_02', 'tree_12_02'} if next(iter(families)) == 'triangle'
                   else {'tree', 'loop3', 'loop4', 'loop5'})
        if ('tree' not in self.topologies or self.primary_loop not in self.topologies
                or self.primary_loop not in {'loop', 'loop3', 'loop4', 'loop5'}
                or len(set(self.topologies)) != len(self.topologies) or not set(self.topologies) <= allowed):
            raise ValueError('topologies must include tree and a valid primary_loop for this graph_family')

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))


def from_dict(value):
    if not isinstance(value, dict):
        raise ValueError('Configuration must be an object')
    unknown = set(value) - {f.name for f in fields(Settings)}
    if unknown:
        raise ValueError('Unknown tree-loop settings: '+str(sorted(unknown)))
    data = dict(value)
    data['conditions'] = tuple(Condition(**c) for c in data.get('conditions', ()))
    for name in ('particle_grid', 'coverage_levels', 'quadrature_orders', 'topologies'):
        if name in data:
            data[name] = tuple(data[name])
    return Settings(**data)


def load_settings(path):
    return from_dict(json.loads(Path(path).read_text(encoding='utf-8')))
