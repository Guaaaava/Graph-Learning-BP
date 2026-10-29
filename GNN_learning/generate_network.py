"""
协同定位网络图生成器

生成包含 Agent 与 Anchor 的测距图。拓扑由三个正交旋钮决定:

  1. scenario_type — 锚点配置 (锚点数与分布由此唯一决定, 见 SCENARIO_ANCHORS)
       - normal:    16 个锚点, 4x4 网格均匀铺设 (基准)
       - hard:      5 个锚点, 四角 + 中心
       - challenge: 4 个锚点, 仅四角
  2. agent_distribution — 盲节点分布
       - 'uniform':   全区域均匀分布
       - 'clustered': 向中心聚集
       - 'auto':      跟随场景默认 (normal/hard → uniform, challenge → clustered)
  3. enable_nlos — 是否注入 NLOS 正偏差"陷阱"边 (默认关闭, 可作用于任意场景)

三者组合共 3 x 2 x 2 = 12 种拓扑。
"""

import torch
import math
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# 每个场景的锚点数 (唯一真源, 调用方不可覆盖)
SCENARIO_ANCHORS = {'normal': 16, 'hard': 5, 'challenge': 4}

# 盲节点分布可选值 (agent_distribution='auto' 时由场景推断, 见 _resolve_agent_distribution)
AGENT_DISTRIBUTIONS = ('uniform', 'clustered')

# 聚集分布的标准差 (相对 area_size)。锚点通信半径的推导依赖此值, 故设为单一真源。
_AGENT_CLUSTER_SIGMA_RATIO = 1.0 / 6.0


def _grid_anchors(k, area_size):
    """k x k 网格顶点上的锚点。

    坐标取 linspace(0, area_size, k), 因此 k=4 时四角与 corner_anchors 重合。
    """
    g = torch.linspace(0.0, area_size, k)
    grid = torch.meshgrid(g, g, indexing='ij')
    return torch.stack(grid, dim=-1).reshape(-1, 2)


# 场景默认的盲节点分布: 锚点越少 → 越需要向中心聚集以维持连通
_SCENARIO_AGENT_DIST = {'normal': 'uniform', 'hard': 'uniform', 'challenge': 'clustered'}


def _resolve_agent_distribution(agent_distribution, scenario_type):
    """把 'auto' 解析成具体分布, 并校验取值。"""
    d = _SCENARIO_AGENT_DIST[scenario_type] if agent_distribution == 'auto' else agent_distribution
    if d not in AGENT_DISTRIBUTIONS:
        raise ValueError(
            f"未知盲节点分布: {agent_distribution}, "
            f"可选: 'auto' / {' / '.join(repr(x) for x in AGENT_DISTRIBUTIONS)}"
        )
    return d


def _sample_agents(distribution, num_agents, area_size):
    """按指定分布采样盲节点位置。"""
    if distribution == 'uniform':
        return torch.rand((num_agents, 2)) * area_size
    # clustered: 以区域中心为均值的高斯聚集, 夹到区域内部
    sigma = area_size * _AGENT_CLUSTER_SIGMA_RATIO
    pos = torch.randn((num_agents, 2)) * sigma + (area_size / 2.0)
    return torch.clamp(pos, 10.0, area_size - 10.0)


def _anchor_comm_radius(scenario_type, comm_radius, area_size):
    """锚点通信半径。

    锚点是基础设施, 部署时可调发射功率/天线增益, 故其通信半径允许随锚点配置
    变化; 盲节点是用户设备, 通信半径恒为 comm_radius, 不随盲节点分布改变。

    normal / hard: 取 comm_radius 即可
        normal  4x4 网格间距 area/3, 网格 Voronoi 半径 area/3/sqrt(2)=23.6m
                < comm_radius(30m), 天然全覆盖;
        hard    中心锚点承担主覆盖, 实测可定位率 99.7%。

    challenge: 锚点仅在四角, 而盲节点聚集于区域中心, comm_radius 够不到 —
        实测 (area=100, comm=30, 30 seeds) 可定位率仅 33%, 2/3 的 agent
        没有通向任何锚点的路径。所需半径由几何决定:
            角点 -> 区域中心的距离  = area/sqrt(2)                  = 70.71 m
            聚集分布的标准差        = area * _AGENT_CLUSTER_SIGMA_RATIO = 16.67 m
        取二者之差 54.04 m (= 1.80 x comm_radius), 含义是"四角锚点恰好够到
        簇内距中心 1 sigma 的环带", 此时直接覆盖约半数盲节点。
        实测: 该值下可定位率饱和到 100%(最差种子亦然);
              最低可行半径约 1.40 x (42 m), 但 r=40 时最差种子仍为 0%,
              紧贴悬崖不留余量, 故不取。
        注: 用几何量而非固定倍率, 是为了在 area_size / comm_radius 改动后仍成立。
    """
    if scenario_type == 'challenge':
        return area_size / math.sqrt(2.0) - area_size * _AGENT_CLUSTER_SIGMA_RATIO
    return comm_radius


def generate_localization_network(num_agents=30, num_anchors=None,
                                   area_size=100.0, comm_radius=30.0,
                                   base_noise=0.5, noise_scale=0.05,
                                   init_pos_cov=25.0,
                                   scenario_type='challenge',
                                   agent_distribution='auto',
                                   enable_nlos=False, nlos_ratio=0.15,
                                   nlos_bias_range=(2.0, 10.0)):
    """
    生成单张协同定位图的完整数据

    Parameters
    ----------
    num_agents : int
        Agent 数量
    num_anchors : int or None
        Anchor 数量。由 scenario_type 唯一决定 (见 SCENARIO_ANCHORS), 一般不用传;
        若显式传入且与场景要求不符会报错。
    area_size : float
        正方形区域边长 (m)
    comm_radius : float
        通信半径 (m)
    base_noise : float
        测距噪声基础值 (m)
    noise_scale : float
        测距噪声随距离的衰减系数
    init_pos_cov : float
        初始位置估计的方差 (m²), 用于生成含噪初值
    scenario_type : str
        锚点配置, 决定锚点数量与分布:
        normal:    16 个锚点 4x4 网格均布 + 锚点通信半径 = comm_radius (基准)
        hard:      5 个锚点(四角+中心) + 锚点通信半径 = comm_radius
        challenge: 4 个锚点(仅四角) + 锚点通信半径由几何覆盖要求决定
                   (见 _anchor_comm_radius, ≈ 1.80 x comm_radius)
    agent_distribution : str
        盲节点分布: 'uniform' / 'clustered' / 'auto'(跟随场景默认, 见
        _SCENARIO_AGENT_DIST)。与 scenario_type 正交。
    enable_nlos : bool
        是否注入 NLOS 陷阱边, 默认 False。只作用于 agent-agent 边。
    nlos_ratio : float
        NLOS 边占 agent-agent 边的比例
    nlos_bias_range : (float, float)
        NLOS 正偏差的均匀分布区间 (m)。偏差只加在测量值上,
        方差不放大 —— 因此从 edge_attr 的方差维度看不出异常。

    Returns
    -------
    data : dict
        true_agents_pos, init_agents_pos, anchors_pos,
        init_agents_cov (标量, 每 agent 相同),
        edge_index, is_anchor_edge, is_nlos_edge,
        measurements, edge_variances, pseudo_range_residual
    """
    # ============================================
    # 1. 位置采样
    # ============================================
    corner_anchors = torch.tensor([
        [0.0, 0.0],
        [area_size, 0.0],
        [0.0, area_size],
        [area_size, area_size]
    ], dtype=torch.float32)

    if scenario_type not in SCENARIO_ANCHORS:
        raise ValueError(f"未知场景类型: {scenario_type}, 可选: normal / hard / challenge")

    # 锚点数由场景唯一决定, 不接受调用方覆盖
    required_anchors = SCENARIO_ANCHORS[scenario_type]
    if num_anchors is not None and num_anchors != required_anchors:
        raise ValueError(
            f"场景 {scenario_type} 的锚点数固定为 {required_anchors}, "
            f"但收到 num_anchors={num_anchors}"
        )
    num_anchors = required_anchors

    # --- 锚点配置 (由 scenario_type 唯一决定) ---
    if scenario_type == 'normal':
        # 基准: 锚点 4x4 网格均布
        anchors_pos = _grid_anchors(4, area_size)

    elif scenario_type == 'hard':
        # 困难: 锚点固定为四角+中心(共5个)
        center = torch.tensor([[area_size / 2.0, area_size / 2.0]], dtype=torch.float32)
        anchors_pos = torch.cat([corner_anchors, center], dim=0)

    else:  # challenge
        # 挑战: 锚点仅四角(共4个)
        anchors_pos = corner_anchors.clone()

    anchor_comm_radius = _anchor_comm_radius(scenario_type, comm_radius, area_size)

    # --- 盲节点分布 (与锚点配置正交) ---
    agent_distribution = _resolve_agent_distribution(agent_distribution, scenario_type)
    true_agents_pos = _sample_agents(agent_distribution, num_agents, area_size)

    # 含噪初始估计
    init_std = init_pos_cov ** 0.5
    init_agents_pos = true_agents_pos + torch.randn_like(true_agents_pos) * init_std

    # ============================================
    # 2. 建图
    # ============================================
    edge_list, is_anchor_list, true_dists = [], [], []

    # Agent-Agent 边
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            d = torch.norm(true_agents_pos[i] - true_agents_pos[j])
            if d < comm_radius:
                edge_list.append([i, j])
                is_anchor_list.append(False)
                true_dists.append(d)

    # Agent-Anchor 边
    for i in range(num_agents):
        for k in range(num_anchors):
            d = torch.norm(true_agents_pos[i] - anchors_pos[k])
            if d < anchor_comm_radius:
                edge_list.append([i, k])
                is_anchor_list.append(True)
                true_dists.append(d)

    edge_index = torch.tensor(edge_list).T               # (2, E)
    is_anchor_edge = torch.tensor(is_anchor_list, dtype=torch.bool)
    true_distances = torch.tensor(true_dists)

    # ============================================
    # 3. 测距 & NLOS 陷阱注入
    # ============================================
    stds = base_noise + noise_scale * true_distances
    is_nlos_edge = torch.zeros(len(stds), dtype=torch.bool)

    # LOS 测距: 噪声如实, 方差字段与之一致
    measurements = true_distances + torch.randn_like(true_distances) * stds

    if enable_nlos:
        # NLOS 只给测量值加正向偏差 (信号被遮挡/多径 → 走更长的路 → 测距偏大),
        # stds 保持不变 → edge_variances 对 NLOS 边也是 LOS 水平,
        # 因此从 edge_attr 的方差维度看不出异常, 只能靠 pseudo_range_residual
        # 与图的互不闭合性识别。仅作用于 agent-agent 边。
        agt_edge_idx = torch.where(~is_anchor_edge)[0]
        num_nlos = int(len(agt_edge_idx) * nlos_ratio)
        if num_nlos > 0:
            perm = torch.randperm(len(agt_edge_idx))
            nlos_idx = agt_edge_idx[perm[:num_nlos]]
            is_nlos_edge[nlos_idx] = True
            bias_lo, bias_hi = nlos_bias_range
            measurements[nlos_idx] += bias_lo + torch.rand(num_nlos) * (bias_hi - bias_lo)

    measurements = torch.clamp(measurements, min=0.1)
    edge_variances = stds ** 2

    # ============================================
    # 4. 伪距残差: r = |z - ||x_i_init - x_j_init|||
    # ============================================
    u_idx = edge_index[0]        # agent 索引
    v_idx = edge_index[1]        # 对端索引 (relative)
    pos_u = init_agents_pos[u_idx]

    # 计算对端坐标
    pos_v = torch.zeros_like(pos_u)
    mask_anc = is_anchor_edge
    mask_agt = ~is_anchor_edge
    if mask_anc.any():
        pos_v[mask_anc] = anchors_pos[v_idx[mask_anc]]
    if mask_agt.any():
        pos_v[mask_agt] = init_agents_pos[v_idx[mask_agt]]

    init_dist = torch.norm(pos_u - pos_v, dim=1) + 1e-8
    pseudo_range_residual = torch.abs(measurements - init_dist)

    data = {
        "true_agents_pos": true_agents_pos,
        "init_agents_pos": init_agents_pos,
        "init_agents_cov": init_pos_cov,       # 标量, 每个 agent 相同
        "anchors_pos": anchors_pos,
        "agent_distribution": agent_distribution,   # 已解析 ('uniform' / 'clustered')
        "edge_index": edge_index,
        "is_anchor_edge": is_anchor_edge,
        "is_nlos_edge": is_nlos_edge,
        "measurements": measurements,
        "edge_variances": edge_variances,
        "pseudo_range_residual": pseudo_range_residual,
    }
    return data


# ================= 可视化 =================
if __name__ == "__main__":
    # 与本目录其它脚本一致 (generate_datasets.py:15), 在 GNN_learning/ 下运行
    from config import TORCH_SEED, NUM_AGENTS, AREA_SIZE, \
        COMM_RADIUS, BASE_NOISE, NOISE_SCALE, SCENARIO_TYPE, AGENT_DISTRIBUTION, \
        ENABLE_NLOS, NLOS_RATIO, NLOS_BIAS_MIN, NLOS_BIAS_MAX

    torch.manual_seed(TORCH_SEED)
    demo = generate_localization_network(
        num_agents=NUM_AGENTS,
        area_size=AREA_SIZE, comm_radius=COMM_RADIUS,
        base_noise=BASE_NOISE, noise_scale=NOISE_SCALE,
        scenario_type=SCENARIO_TYPE,
        agent_distribution=AGENT_DISTRIBUTION,
        enable_nlos=ENABLE_NLOS, nlos_ratio=NLOS_RATIO,
        nlos_bias_range=(NLOS_BIAS_MIN, NLOS_BIAS_MAX)
    )

    print(f"场景: {SCENARIO_TYPE}  (盲节点分布: {demo['agent_distribution']})")
    print(f"Agent 数: {demo['true_agents_pos'].shape[0]}")
    print(f"Anchor 数: {demo['anchors_pos'].shape[0]}")
    print(f"总边数: {demo['edge_index'].shape[1]}")
    print(f"NLOS 边数: {demo['is_nlos_edge'].sum().item()}")

    agents_pos = demo['true_agents_pos'].numpy()
    anchors_pos = demo['anchors_pos'].numpy()
    ei = demo['edge_index'].numpy()
    iae = demo['is_anchor_edge'].numpy()
    ine = demo['is_nlos_edge'].numpy()

    plt.figure(figsize=(10, 8))
    for idx in range(ei.shape[1]):
        u, v = ei[:, idx]
        if iae[idx]:
            xc, yc = [agents_pos[u, 0], anchors_pos[v, 0]], [agents_pos[u, 1], anchors_pos[v, 1]]
            plt.plot(xc, yc, color='red', alpha=0.5, lw=1.5, zorder=1)
        elif ine[idx]:
            xc, yc = [agents_pos[u, 0], agents_pos[v, 0]], [agents_pos[u, 1], agents_pos[v, 1]]
            plt.plot(xc, yc, color='darkorange', linestyle=':', alpha=0.9, lw=2.5, zorder=2)
        else:
            xc, yc = [agents_pos[u, 0], agents_pos[v, 0]], [agents_pos[u, 1], agents_pos[v, 1]]
            plt.plot(xc, yc, color='gray', linestyle='--', alpha=0.3, lw=1.0, zorder=1)

    plt.scatter(agents_pos[:, 0], agents_pos[:, 1], c='blue', marker='o', s=80, zorder=3)
    plt.scatter(anchors_pos[:, 0], anchors_pos[:, 1], c='red', marker='^', s=150, zorder=3)
    plt.title(f'{SCENARIO_TYPE.capitalize()} Scenario — {demo["agent_distribution"]} agents',
              fontsize=14)
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()
