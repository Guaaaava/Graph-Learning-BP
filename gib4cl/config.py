"""GIB4CL experiment settings."""

from __future__ import annotations # postpone the evaluation of type annotations, support self-reference

from dataclasses import asdict, dataclass, field, fields, replace
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1] # `Graph-Learning-BP/`
# Fixed representation used by measurement validation and the compression cost.
MEASUREMENT_ENCODING = 'ieee754-binary32-canonical-finite'
# `smoke` for quick check the overall flow


# 1. Load the single-scene experiment configuration.
def load_config(path=None):
    if path is None:
        path = REPO_ROOT / 'configs' / 'experiment.json'
    with Path(path).open(encoding='utf-8') as handle:
        return config_from_dict(json.load(handle, object_pairs_hook=_unique_object))


# Only the current input structure is accepted; historical inputs are not migrated.
def config_from_dict(value):
    if not isinstance(value, dict):
        raise ValueError('Experiment configuration must be an object')
    value = dict(value)
    for name, cls in _CONFIG_SECTIONS:
        if name in value:
            value[name] = _construct(cls, value[name])
    return _construct(ExperimentConfig, value)


# Check if there's duplicate configuration key
def _unique_object(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError('Duplicate configuration key: ' + key)
        value[key] = item
    return value


# 2. Check if there's unknown attributes, create dataclass instances
def _construct(cls, value):
    if not isinstance(value, dict): # `isinstance(v, t)`: `v` is an instance of type `t`? ; `dict`: dictionary
        raise ValueError(f'{cls.__name__} must be an object')
    unknown = set(value) - {f.name for f in fields(cls)} # `set(dictionary)`: a set of all attribute names, i.e. keys; `dataclasses.fields()`: get name and type of all attributes; `f.name`: attribute name 
    if unknown:
        raise ValueError(f'Unknown {cls.__name__} settings: {sorted(unknown)}')
    return cls(**value) # unpack dictionary to create a dataclass instance


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False) # get standard JSON string


# 4. Calculate data fingerprint, output path, and sample seed, and write the final configuration into the operation record and checkpoint
# Generate a stable identity using config
def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


# Create a new config, modify only fields in `value`, keep other settings unchanged, and re-check
def with_seeds(config,**values):
    """Change only explicitly supplied streams; never couple training to data."""
    return replace(config,seeds=replace(config.seeds,**{k:v for k,v in values.items() if v is not None}))


# Using only one global seed can also reproduce a fixed execution. However, adding random operations or skipping samples in the middle may change all subsequent samples. The per-sample seed supports the independent reconstruction of a graph number under the same configuration and generation code; adding other graphs will not change it
def sample_seed(master_seed, split, seed_offset, sample_number):
    """Stable per-sample streams: splits do not share a sequential RNG state."""
    validate_name(split, 'split')
    for value, name in ((master_seed, 'seed'), (seed_offset, 'seed_offset'), (sample_number, 'sample_number')):
        _integer(value, name, 0)
    encoded = f'gib4cl.sample.v1:{master_seed}:{split}:{seed_offset}:{sample_number}'
    return int.from_bytes(hashlib.sha256(encoded.encode('ascii')).digest()[:8], 'big') & ((1 << 63) - 1)


# 3↓ Check functions
def validate_name(value, field_name):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', value):
        raise ValueError(f'{field_name} must contain 1-80 ASCII letters/digits/_/- and start with a letter/digit')
    return value


def _positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if value < 0 or (not allow_zero and value == 0):
        raise ValueError(f'{name} must be {"nonnegative" if allow_zero else "positive"}')


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


# 3. Check attributes and values while creating instances
# The scenario configuration to be used
@dataclass(frozen=True)
class DataConfig:
    # Five fixed anchors (corners and centre), uniform agents, Gaussian LOS ranges.
    num_agents_min: int = 25
    num_agents_max: int = 35
    area_size: float = 100.0
    comm_radius: float = 30.0
    initial_covariance: object = 25.0
    base_noise_std: float = 0.5 # σ_0
    distance_noise_scale: float = 0.05 # α
    communication_cost_scale: float = 1.0 # ρ in energy constraint
    measurement_encoding: str = MEASUREMENT_ENCODING

    # Check all the configuration value (name, type, ...)
    def __post_init__(self):
        _integer(self.num_agents_min, 'num_agents_min')
        _integer(self.num_agents_max, 'num_agents_max')
        if self.num_agents_max < self.num_agents_min:
            raise ValueError('num_agents_max must be >= num_agents_min')
        for name in ('area_size', 'base_noise_std', 'communication_cost_scale'):
            _positive(getattr(self, name), name)
        for name in ('comm_radius', 'distance_noise_scale'):
            _positive(getattr(self, name), name, allow_zero=True)
        if self.measurement_encoding != MEASUREMENT_ENCODING:
            raise ValueError(f'Only {MEASUREMENT_ENCODING} is currently supported')
        cov = self.initial_covariance
        # JSON uses a scalar, two diagonal values, or a symmetric 2x2 matrix.
        if isinstance(cov, (int, float)):
            _positive(cov, 'initial_covariance')
        elif isinstance(cov, (tuple, list)) and len(cov) == 2:
            if all(isinstance(v, (int, float)) for v in cov):
                for v in cov:
                    _positive(v, 'initial_covariance diagonal')
                cov = tuple(cov)
            elif all(isinstance(v, (tuple, list)) and len(v) == 2 for v in cov):
                a, b = cov[0]
                c, d = cov[1]
                for value in (a, b, c, d):
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError('initial_covariance matrix must be finite')
                if a <= 0 or d <= 0 or b != c or a * d - b * c <= 0:
                    raise ValueError('initial_covariance must be symmetric positive definite')
                cov = tuple(tuple(row) for row in cov)
            else:
                raise ValueError('Invalid initial_covariance shape')
            object.__setattr__(self, 'initial_covariance', cov)
        else:
            raise ValueError('initial_covariance must be scalar, length 2, or a 2x2 matrix')

    @property
    def num_anchors(self):
        return 5

    @property
    def anchor_coordinates(self):
        size = self.area_size
        return ((0., 0.), (size, 0.), (0., size), (size, size), (size / 2, size / 2))


# The GNN configuration to be used
@dataclass(frozen=True)
class ModelConfig:
    node_in_dim: int = 5
    edge_in_dim: int = 4
    hidden_dim: int = 64
    num_layers: int = 3

    def __post_init__(self):
        _integer(self.node_in_dim, 'node_in_dim')
        _integer(self.edge_in_dim, 'edge_in_dim')
        if self.node_in_dim != 5 or self.edge_in_dim != 4:
            raise ValueError('The paper requires 5 node and 4 edge features')
        _integer(self.hidden_dim, 'hidden_dim', 2)
        _integer(self.num_layers, 'num_layers')


# The training configuration to be used
@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 0.001
    tau_init: float = 5.0
    tau_min: float = 0.1
    tau_decay: float = 0.98
    weight_decay: float = 1e-4
    max_grad_norm: float = 5.0 # `max_grad_norm`: maximum L2 norm of gradients for gradient clipping, avoid exploding gradient
    lr_min: float = 1e-5
    deterministic: bool = True # `deterministic`: use deterministic cuda kernels for reproducible results, slower training

    def __post_init__(self):
        _integer(self.epochs, 'epochs')
        _integer(self.batch_size, 'batch_size')
        for name in ('learning_rate', 'tau_init', 'tau_min', 'tau_decay'):
            _positive(getattr(self, name), name)
        if self.tau_min > self.tau_init or self.tau_decay > 1:
            raise ValueError('Require tau_min <= tau_init and tau_decay <= 1')
        _positive(self.weight_decay, 'weight_decay', allow_zero=True)
        _positive(self.max_grad_norm, 'max_grad_norm')
        _positive(self.lr_min, 'lr_min', allow_zero=True)
        if self.lr_min > self.learning_rate or not isinstance(self.deterministic, bool):
            raise ValueError('Require lr_min <= learning_rate and boolean deterministic')


# The hyperparameter configuration to be used
@dataclass(frozen=True)
class ObjectiveConfig:
    # Formal experiment coefficients must be explicit; smoke coefficients live in their own preset.
    beta: Optional[float] = None
    gamma: Optional[float] = None
    kappa: Optional[float] = None
    eta: Optional[float] = None
    prediction_variance_floor: float = 1e-6 # the lower bound of measurement variance for the Gaussian predictor, prevents numerical instability caused by too small variance
    compute_dtype: str = 'float64' # precision used in calculation, has nothing to do with the information

    def __post_init__(self):
        for name in ('beta', 'gamma', 'kappa', 'eta'):
            value = getattr(self, name)
            if value is not None:
                _positive(value, name, allow_zero=name != 'beta')
        _positive(self.prediction_variance_floor, 'prediction_variance_floor')
        if self.compute_dtype not in ('float32', 'float64'):
            raise ValueError('objective.compute_dtype must be float32 or float64')

    # Check all parameters are not `None`
    def require_ready(self):
        missing = [name for name in ('beta', 'gamma', 'kappa', 'eta') if getattr(self, name) is None]
        if missing:
            raise ValueError(f'Set explicit objective coefficients {missing}; old lambda/FIM settings are not substituted')
        return self


# The BP parameter configuration to be used
@dataclass(frozen=True)
class BPConfig:
    num_particles: int = 10000
    num_iterations: int = 3
    chunk_size: int = 256 # particle pair (i,j): combine i-th candidate position particle of u and j-th particle of v, P^2=10000*10000 too large, separate to many 256*256 chunk each calculation, prevent exceeding the memory
    compute_dtype: str = 'float64'
    measurement_variance_floor: float = 0.0 # whether the measurement variance can be too small, but a non-zero lower limit will change the inference model, 0 by default
    damping: float = 0.0 # how old and new messages mix, m=(1-λ)*m_new+λ*m_old, now m=m_new
    message_tolerance: float = 1e-8 # max change in two iteration of normalized log messages ≤ 1e-8, convergence
    early_stopping: bool = False

    def __post_init__(self):
        _integer(self.num_particles, 'num_particles', 2)
        _integer(self.num_iterations, 'num_iterations', 0)
        _integer(self.chunk_size, 'bp.chunk_size')
        if self.compute_dtype not in ('float32', 'float64'):
            raise ValueError('bp.compute_dtype must be float32 or float64')
        for name in ('measurement_variance_floor', 'damping', 'message_tolerance'):
            _positive(getattr(self, name), 'bp.'+name, allow_zero=True)
        if self.damping >= 1 or not isinstance(self.early_stopping, bool):
            raise ValueError('Require bp.damping < 1 and boolean early_stopping')


# The evaluation parameter configuration to be used
@dataclass(frozen=True)
class EvaluationConfig:
    outage_threshold: float = 3.0
    alpha: float = 0.05
    edge_threshold: float = 0.5
    bp_repeats: int = 1 # each graph currently only uses one set of random particles, we can increase it to see the difference caused by particle randomness
    bootstrap_samples: int = 2000 # repeatedly resampling (with replacement, drawing one and putting it back, drawing one and putting it back) on the already produced results, and repeatedly calculating the indicator, to estimate the sampling uncertainty of the indicator
    covariance_rtol: float = 1e-12 # when the ratio of the min/max eigenvalues ≤ 1e-12, cov invalid
    outage_thresholds: tuple = (.5, 1., 2., 3., 5., 10.) # calculate the failure rate at thresholds of 0.5, 1, 2, 3, 5, 10m, generate a curve
    coverage_levels: tuple = (.5, .8, .9, .95, .99) # assuming Gaussian distribution, check how much ellipse achieves this coverage level
    ess_fraction_threshold: float = .1 # when the effective particle count ESS < 10%, is low ESS, indicating weight degradation

    def __post_init__(self):
        _positive(self.outage_threshold, 'outage_threshold')
        _positive(self.alpha, 'alpha')
        if not self.alpha < 1:
            raise ValueError('alpha must be between 0 and 1')
        _positive(self.edge_threshold, 'edge_threshold')
        if self.edge_threshold >= 1:
            raise ValueError('edge_threshold must be in (0,1)')
        _integer(self.bp_repeats, 'bp_repeats')
        _integer(self.bootstrap_samples, 'bootstrap_samples', 0)
        _positive(self.covariance_rtol, 'covariance_rtol', allow_zero=True)
        if self.covariance_rtol >= 1:
            raise ValueError('covariance_rtol must be in [0,1)')
        for name, probability in (('outage_thresholds', False), ('coverage_levels', True)):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError(name+' must be a nonempty sequence')
            for value in values:
                _positive(value, name)
                if probability and value >= 1:
                    raise ValueError(name+' values must be below 1')
            values = tuple(float(value) for value in values)
            if tuple(sorted(set(values))) != values:
                raise ValueError(name+' must be strictly increasing and unique')
            object.__setattr__(self, name, values)
        _positive(self.ess_fraction_threshold, 'ess_fraction_threshold')
        if self.ess_fraction_threshold > 1:
            raise ValueError('ess_fraction_threshold must be in (0,1]')


# The seed configuration to be used
@dataclass(frozen=True)
class SeedConfig:
    data: int = 100
    training: int = 100
    bp: int = 700
    bootstrap: int = 700

    def __post_init__(self):
        for name in ('data','training','bp','bootstrap'):
            value = getattr(self,name)
            _integer(value,'seeds.'+name,0)
            if value >= 2**63:
                raise ValueError('seeds.'+name+' must be below 2**63')


# The overall experiment configuration to be used
@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str = 'five-anchors-los'
    seeds: SeedConfig = field(default_factory=SeedConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    bp: BPConfig = field(default_factory=BPConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    split_sizes: Mapping = field(default_factory=lambda: {'train': 2000, 'val': 500, 'test': 500})

    def __post_init__(self):
        validate_name(self.experiment_id, 'experiment_id')
        for name, cls in _CONFIG_SECTIONS:
            if not isinstance(getattr(self, name), cls):
                raise ValueError(name + ' must be a ' + cls.__name__)
        if not isinstance(self.split_sizes, Mapping) or not self.split_sizes:
            raise ValueError('split_sizes must be a nonempty dictionary')
        for split, count in self.split_sizes.items():
            validate_name(split, 'split')
            _integer(count, f'split_sizes.{split}', 0)
        object.__setattr__(self, 'split_sizes', MappingProxyType(dict(self.split_sizes)))

    def to_dict(self):
        return {'experiment_id': self.experiment_id,
                **{name: asdict(getattr(self, name)) for name, _ in _CONFIG_SECTIONS},
                'split_sizes': dict(self.split_sizes)}

    @property
    def data_fingerprint(self):
        # Training/metric changes must not alter the identity of the same observations.
        return fingerprint({'data': asdict(self.data), 'seed': self.seeds.data})

    def dataset_path(self, split, output_root=None):
        validate_name(split, 'split')
        root = Path(output_root) if output_root is not None else REPO_ROOT / 'data'
        return root / self.experiment_id / self.data_fingerprint[:16] / f'{split}_dataset.pt'


_CONFIG_SECTIONS = (('seeds', SeedConfig), ('data', DataConfig), ('model', ModelConfig),
                    ('training', TrainingConfig), ('objective', ObjectiveConfig),
                    ('bp', BPConfig), ('evaluation', EvaluationConfig))
