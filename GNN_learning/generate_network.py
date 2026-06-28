"""
协同定位网络图生成器

生成包含 Agent 与 Anchor 的测距图，支持三种递进场景:
  - normal:    锚点均匀分布 + Agent均匀分布 + 无NLOS (基准)
  - hard:      锚点固定(四角+中心) + Agent均匀分布 + 无NLOS
  - challenge: 锚点仅四角 + Agent中心聚集 + NLOS陷阱边
  - 初始位置不确定度
  - 伪距残差 (pseudo-range residual)
"""

import torch
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def generate_localization_network(num_agents=30, num_anchors=4,
                                   area_size=100.0, comm_radius=30.0,
                                   base_noise=0.5, noise_scale=0.05,
                                   init_pos_cov=25.0,
                                   scenario_type='challenge'):
    """
    生成单张协同定位图的完整数据

    Parameters
    ----------
    num_agents : int
        Agent 数量
    num_anchors : int
        Anchor 数量
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
        'normal' — 锚点均匀分布, Agent均匀分布, 无NLOS (基准)
        'hard' — 锚点固定(四角+中心), Agent均匀分布, 无NLOS
        'challenge' — 锚点仅四角, Agent中心聚集, 含NLOS陷阱边

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

    if scenario_type == 'normal':
        # 正常场景: 锚点均匀分布, Agent均匀分布, 统一通信半径, 无NLOS
        true_agents_pos = torch.rand((num_agents, 2)) * area_size
        anchors_pos = torch.rand((num_anchors, 2)) * area_size
        anchor_comm_radius = comm_radius

    elif scenario_type == 'hard':
        # 恶劣场景: 锚点固定(四角+中心), Agent均匀分布
        true_agents_pos = torch.rand((num_agents, 2)) * area_size
        center = torch.tensor([[area_size / 2.0, area_size / 2.0]], dtype=torch.float32)
        fixed_anchors = torch.cat([corner_anchors, center], dim=0)  # 5个固定位置
        if num_anchors <= 5:
            anchors_pos = fixed_anchors[:num_anchors]
        else:
            extra = torch.rand((num_anchors - 5, 2)) * area_size
            anchors_pos = torch.cat([fixed_anchors, extra], dim=0)
        anchor_comm_radius = comm_radius

    elif scenario_type == 'challenge':
        # 挑战场景: 锚点仅四角, Agent中心聚集, 扩展锚点通信半径, 含NLOS边
        if num_anchors <= 4:
            anchors_pos = corner_anchors[:num_anchors]
        else:
            extra = torch.rand((num_anchors - 4, 2)) * area_size
            anchors_pos = torch.cat([corner_anchors, extra], dim=0)

        true_agents_pos = (
            torch.randn((num_agents, 2)) * (area_size / 6.0) + (area_size / 2.0)
        )
        true_agents_pos = torch.clamp(true_agents_pos, 10.0, area_size - 10.0)
        anchor_comm_radius = comm_radius * 1.8

    else:
        raise ValueError(f"未知场景类型: {scenario_type}, 可选: normal / hard / challenge")

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
    # 3. 测距 & NLOS 注入
    # ============================================
    stds = base_noise + noise_scale * true_distances
    is_nlos_edge = torch.zeros(len(stds), dtype=torch.bool)

    if scenario_type == 'challenge':
        agt_edge_idx = torch.where(~is_anchor_edge)[0]
        num_nlos = int(len(agt_edge_idx) * 0.15)
        if num_nlos > 0:
            perm = torch.randperm(len(agt_edge_idx))
            nlos_idx = agt_edge_idx[perm[:num_nlos]]
            is_nlos_edge[nlos_idx] = True
            stds[nlos_idx] += 10.0 + torch.rand(num_nlos) * 10.0

    measurements = true_distances + torch.randn_like(true_distances) * stds
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
    from GNN_learning.config import TORCH_SEED, NUM_AGENTS, NUM_ANCHORS, AREA_SIZE, \
        COMM_RADIUS, BASE_NOISE, NOISE_SCALE, SCENARIO_TYPE

    torch.manual_seed(TORCH_SEED)
    demo = generate_localization_network(
        num_agents=NUM_AGENTS, num_anchors=NUM_ANCHORS,
        area_size=AREA_SIZE, comm_radius=COMM_RADIUS,
        base_noise=BASE_NOISE, noise_scale=NOISE_SCALE,
        scenario_type=SCENARIO_TYPE
    )

    print(f"Agent 数: {demo['true_agents_pos'].shape[0]}")
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
    plt.title('Challenge Scenario', fontsize=14)
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()
