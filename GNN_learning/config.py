# ================= 全局种子 =================
TORCH_SEED = 100

# ================= 定位网络拓扑 =================
NUM_AGENTS = 30
AREA_SIZE = 100.0
COMM_RADIUS = 30.0
BASE_NOISE = 0.5
NOISE_SCALE = 0.05

# --- 旋钮 1: 锚点配置 (锚点数与分布由此唯一决定, 见 generate_network.SCENARIO_ANCHORS) ---
# normal(16锚点4x4网格) / hard(5锚点四角+中心) / challenge(4锚点四角)
SCENARIO_TYPE = 'challenge'

# --- 旋钮 2: 盲节点分布 ---
# 'auto' 跟随场景默认 (normal/hard → uniform, challenge → clustered)
# 'uniform' 全区域均匀 / 'clustered' 向中心聚集
AGENT_DISTRIBUTION = 'auto'

# Agent 数量范围 (数据集生成时随机采样); 锚点数不随机
NUM_AGENTS_MIN = 25
NUM_AGENTS_MAX = 35

# --- 旋钮 3: NLOS 陷阱 (可作用于任意场景的 agent-agent 边) ---
# 正偏差加在测量值上, 不放大方差 → 从 edge_attr 看不出异常
ENABLE_NLOS = False
NLOS_RATIO = 0.15
NLOS_BIAS_MIN = 2.0
NLOS_BIAS_MAX = 10.0

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
LAMBDA_REG = 3.0     # KL 项权重 (FIM归一化后)
ETA = 10.0           # 度数约束权重
SPARSITY_WEIGHT = 0.0  # 直接稀疏惩罚权重 (乘 p.mean()), 0 = 关闭。被 LAMBDA_SCHEDULE 取代

# Lambda 退火调度 (epoch -> lambda)
# 格式: [(epoch_start, lambda_start), (epoch_end, lambda_end)]
# lambda 在区间内线性插值
LAMBDA_SCHEDULE = [
    (0,   0.5),    # epoch 1-50: warmup, focus on accuracy
    (50,  2.0),    # epoch 51-150: standard training
    (150, 10.0),   # epoch 151-200: increase compression
]
FIM_PRIOR = 0.5      # FIM 对角先验 (提高稳定性)

# ================= 粒子 BP =================
BP_NUM_PARTICLES_TEST = 10000  # 粒子 BP: 仅评估使用, 训练不涉及
BP_NUM_ITER = 3
BP_SIGMA_MEAS = 0.5
BP_INIT_COV = 25.0

# ================= 评估 =================
OUTAGE_THRESHOLD = 3.0   # 中断概率阈值 (m)
ALPHA = 0.05              # 卡方检验显著性水平
