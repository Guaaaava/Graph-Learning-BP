# ================= 全局种子 =================
TORCH_SEED = 100

# ================= 定位网络拓扑 =================
NUM_AGENTS = 30
NUM_ANCHORS = 4
AREA_SIZE = 100.0
COMM_RADIUS = 30.0
BASE_NOISE = 0.5
NOISE_SCALE = 0.05
SCENARIO_TYPE = 'challenge'

# Agent/Anchor 数量范围 (数据集生成时随机采样)
NUM_AGENTS_MIN = 25
NUM_AGENTS_MAX = 35
NUM_ANCHORS_MIN = 3
NUM_ANCHORS_MAX = 6

# 初始位置不确定度 (m²)
INIT_POS_COV = 25.0   # std ≈ 5m

# ================= GNN 模型 =================
NODE_IN_DIM = 5   # [pos_x, pos_y, σ²_x, σ²_y, is_anchor]
EDGE_IN_DIM = 4   # [z, σ²_z, r_residual, is_anchor_edge]
HIDDEN_DIM = 64
NUM_LAYERS = 3

# ================= 训练 =================
EPOCHS = 200
BATCH_SIZE = 32
LR = 0.001

# Gumbel-Sigmoid 温度
TAU_INIT = 5.0
TAU_MIN = 0.1
TAU_DECAY = 0.98     # 每 epoch 衰减率

# ================= GIB 损失 =================
GAMMA = 0.002        # 先验 q(d) = 1/(1+exp(γ·d²)), d=30m→q≈0.14
LAMBDA_REG = 5.0     # KL 项权重 (FIM归一化后, 1-10 即有压缩效果)
ETA = 20.0           # 度数约束权重
FIM_PRIOR = 0.5      # FIM 对角先验 (提高稳定性)

# ================= 粒子 BP =================
BP_NUM_PARTICLES_TEST = 10000  # 粒子 BP: 仅评估使用, 训练不涉及
BP_NUM_ITER = 3
BP_SIGMA_MEAS = 0.5
BP_INIT_COV = 25.0

# ================= 评估 =================
OUTAGE_THRESHOLD = 3.0   # 中断概率阈值 (m)
ALPHA = 0.05              # 卡方检验显著性水平
