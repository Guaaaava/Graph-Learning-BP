"""Compatibility constants backed by the explicit default experiment configuration.

Data generation uses ExperimentConfig directly. The old objective settings below
remain labelled legacy until stages 3-6 replace the training/inference algorithms.
"""
if __package__:
    from gib4cl.config import load_config
else:
    from gib4cl.config import load_config

DEFAULT_EXPERIMENT = load_config()
_DATA = DEFAULT_EXPERIMENT.data
_MODEL = DEFAULT_EXPERIMENT.model
_TRAIN = DEFAULT_EXPERIMENT.training

TORCH_SEED = DEFAULT_EXPERIMENT.seed
NUM_AGENTS_MIN = _DATA.num_agents_min
NUM_AGENTS_MAX = _DATA.num_agents_max
NUM_AGENTS = (NUM_AGENTS_MIN + NUM_AGENTS_MAX) // 2
AREA_SIZE = _DATA.area_size
COMM_RADIUS = _DATA.agent_comm_radius
ANCHOR_COMM_RADIUS = _DATA.anchor_comm_radius
BASE_NOISE = _DATA.base_noise_std
NOISE_SCALE = _DATA.distance_noise_scale
SCENARIO_TYPE = _DATA.scenario
AGENT_DISTRIBUTION = _DATA.agent_distribution
ENABLE_NLOS = _DATA.enable_nlos
NLOS_RATIO = _DATA.nlos_ratio
NLOS_BIAS_MIN = _DATA.nlos_bias_min
NLOS_BIAS_MAX = _DATA.nlos_bias_max
INIT_POS_COV = _DATA.initial_covariance

NODE_IN_DIM = _MODEL.node_in_dim
EDGE_IN_DIM = _MODEL.edge_in_dim
HIDDEN_DIM = _MODEL.hidden_dim
NUM_LAYERS = _MODEL.num_layers
EPOCHS = _TRAIN.epochs
BATCH_SIZE = _TRAIN.batch_size
LR = _TRAIN.learning_rate
TAU_INIT = _TRAIN.tau_init
TAU_MIN = _TRAIN.tau_min
TAU_DECAY = _TRAIN.tau_decay

# Historical objective only. These are NOT the new paper objective configuration.
GAMMA = 0.002
LAMBDA_REG = 3.0
ETA = 10.0
SPARSITY_WEIGHT = 0.0
LAMBDA_SCHEDULE = [(0, 0.5), (50, 2.0), (150, 10.0)]
FIM_PRIOR = 0.5

BP_NUM_PARTICLES_TEST = DEFAULT_EXPERIMENT.bp.num_particles
BP_NUM_ITER = DEFAULT_EXPERIMENT.bp.num_iterations
# Historical aliases only; new BP reads full covariances and known variances from data.
BP_SIGMA_MEAS = _DATA.base_noise_std
BP_INIT_COV = INIT_POS_COV
OUTAGE_THRESHOLD = DEFAULT_EXPERIMENT.evaluation.outage_threshold
ALPHA = DEFAULT_EXPERIMENT.evaluation.alpha


def get_dataset_path(split):
    return str(DEFAULT_EXPERIMENT.dataset_path(split))
