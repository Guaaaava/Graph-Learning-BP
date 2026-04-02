# %%

import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# import GNN_learning.config as config
import config

def generate_localization_network(num_agents=30, num_anchors=4, area_size=100.0, comm_radius=30.0, base_noise=0.5, noise_scale=0.05, scenario_type='challenge'):
    """
    生成协同定位虚拟网络数据
    新增: 支持 'challenge' 模式，生成带有 NLOS 陷阱、集群化、边缘锚点的挑战场景
    """
    # ==========================================
    # 1. 坐标撒点 (分布逻辑升级)
    # ==========================================
    if scenario_type == 'challenge':
        # (A) 锚点边缘化：死死卡在四个角落
        anchors_pos = torch.tensor([
            [0.0, 0.0],
            [area_size, 0.0],
            [0.0, area_size],
            [area_size, area_size]
        ], dtype=torch.float32)
        # 如果锚点数不是4，做个简单兼容
        if num_anchors < 4:
            anchors_pos = anchors_pos[:num_anchors]
        elif num_anchors > 4:
            extra = torch.rand((num_anchors - 4, 2)) * area_size
            anchors_pos = torch.cat([anchors_pos, extra], dim=0)
            
        # (B) Agent 集群化陷阱：用高斯分布把大家挤在中心地带
        true_agents_pos = torch.randn((num_agents, 2)) * (area_size / 6.0) + (area_size / 2.0)
        true_agents_pos = torch.clamp(true_agents_pos, 10.0, area_size - 10.0)
        
        # (C) 挑战模式下，为了保证中心群体能勉强连上角落锚点，稍微放宽对锚点的通信距离
        anchor_comm_radius = comm_radius * 1.8 
    else:
        # 传统均匀随机分布
        true_agents_pos = torch.rand((num_agents, 2)) * area_size
        anchors_pos = torch.rand((num_anchors, 2)) * area_size
        anchor_comm_radius = comm_radius

    # 模拟初始粗略位置估计
    init_agents_pos = true_agents_pos + torch.randn_like(true_agents_pos) * 5.0

    # ==========================================
    # 2. 建图与连边
    # ==========================================
    edge_index_list = []
    is_anchor_edge_list = []
    true_distances = []
    
    # (A) 建立 Agent 与 Agent 之间的边
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            dist = torch.norm(true_agents_pos[i] - true_agents_pos[j])
            if dist < comm_radius:
                edge_index_list.append([i, j])
                is_anchor_edge_list.append(False)
                true_distances.append(dist)

    # (B) 建立 Agent 与 Anchor 之间的边
    for i in range(num_agents):
        for k in range(num_anchors):
            dist = torch.norm(true_agents_pos[i] - anchors_pos[k])
            if dist < anchor_comm_radius:
                edge_index_list.append([i, k])
                is_anchor_edge_list.append(True)
                true_distances.append(dist)

    edge_index = torch.tensor(edge_index_list).T # (2, E)
    is_anchor_edge = torch.tensor(is_anchor_edge_list, dtype=torch.bool)
    true_distances = torch.tensor(true_distances)

    # ==========================================
    # 3. 测距生噪与 NLOS 陷阱注入
    # ==========================================
    stds = base_noise + noise_scale * true_distances
    is_nlos_edge = torch.zeros(len(stds), dtype=torch.bool)
    
    if scenario_type == 'challenge':
        # 挑选出 Agent-Agent 的纯物理边
        agent_edge_indices = torch.where(~is_anchor_edge)[0]
        # 随机挑选 15% 作为 NLOS 恶劣边
        num_nlos = int(len(agent_edge_indices) * 0.15)
        if num_nlos > 0:
            # 随机打乱并截取前 num_nlos 个索引
            shuffled_idx = torch.randperm(len(agent_edge_indices))
            nlos_idx = agent_edge_indices[shuffled_idx[:num_nlos]]
            is_nlos_edge[nlos_idx] = True
            
            # 注入巨大的非视距误差方差 (例如直接增加 10~20 米的标准差)
            stds[nlos_idx] += 10.0 + torch.rand(num_nlos) * 10.0

    measurements = true_distances + torch.randn_like(true_distances) * stds
    measurements = torch.clamp(measurements, min=0.1)
    edge_variances = stds ** 2

    data = {
        "true_agents_pos": true_agents_pos,
        "init_agents_pos": init_agents_pos,
        "anchors_pos": anchors_pos,
        "edge_index": edge_index,
        "is_anchor_edge": is_anchor_edge,
        "is_nlos_edge": is_nlos_edge, # 供画图和评估使用
        "measurements": measurements,
        "edge_variances": edge_variances
    }
    
    return data

# ================= 测试运行与可视化 =================
if __name__ == "__main__":
    torch.manual_seed(config.TORCH_SEED)
    # 使用新参数生成挑战图
    demo_data = generate_localization_network(
        num_agents=config.NUM_AGENTS, num_anchors=config.NUM_ANCHORS, area_size=config.AREA_SIZE, 
        comm_radius=config.COMM_RADIUS, base_noise=config.BASE_NOISE, noise_scale=config.NOISE_SCALE, 
        scenario_type=config.SCENARIO_TYPE
    )
    
    print(f"生成的 Agent 数量: {demo_data['true_agents_pos'].shape[0]}")
    print(f"生成的有效测距边总数: {demo_data['edge_index'].shape[1]}")
    print(f"其中包含的 NLOS 陷阱边数: {demo_data['is_nlos_edge'].sum().item()}")

    # 提取 Tensor 备用
    agents_pos = demo_data['true_agents_pos'].numpy()
    anchors_pos = demo_data['anchors_pos'].numpy()
    edge_index = demo_data['edge_index'].numpy()
    is_anchor_edge = demo_data['is_anchor_edge'].numpy()
    is_nlos_edge = demo_data['is_nlos_edge'].numpy()

    plt.figure(figsize=(10, 8))
    
    # 1. 绘制连线 (先画底层的灰色线，再画重要的红线和橙色陷阱线)
    for idx in range(edge_index.shape[1]):
        u, v = edge_index[:, idx]
        
        if is_anchor_edge[idx]:
            x_coords = [agents_pos[u, 0], anchors_pos[v, 0]]
            y_coords = [agents_pos[u, 1], anchors_pos[v, 1]]
            plt.plot(x_coords, y_coords, color='red', linestyle='-', alpha=0.5, linewidth=1.5, zorder=1)
        elif is_nlos_edge[idx]:
            # [新增] 绘制极高噪声的陷阱边 (橙色加粗虚线)
            x_coords = [agents_pos[u, 0], agents_pos[v, 0]]
            y_coords = [agents_pos[u, 1], agents_pos[v, 1]]
            plt.plot(x_coords, y_coords, color='darkorange', linestyle=':', alpha=0.9, linewidth=2.5, zorder=2)
        else:
            x_coords = [agents_pos[u, 0], agents_pos[v, 0]]
            y_coords = [agents_pos[u, 1], agents_pos[v, 1]]
            plt.plot(x_coords, y_coords, color='gray', linestyle='--', alpha=0.3, linewidth=1.0, zorder=1)

    # 2. 绘制节点
    plt.scatter(agents_pos[:, 0], agents_pos[:, 1], c='blue', marker='o', s=80, label='Agent (True Pos)', zorder=3)
    for i in range(agents_pos.shape[0]):
        plt.text(agents_pos[i, 0] + 1, agents_pos[i, 1] + 1, str(i), fontsize=9, color='darkblue', zorder=4)

    plt.scatter(anchors_pos[:, 0], anchors_pos[:, 1], c='red', marker='^', s=150, label='Anchor', zorder=3)
    for i in range(anchors_pos.shape[0]):
        plt.text(anchors_pos[i, 0] + 1.5, anchors_pos[i, 1] + 1.5, f"A{i}", fontsize=11, fontweight='bold', color='darkred', zorder=4)

    # 3. 设置图表属性
    plt.title('Challenge Scenario: Clustered Agents & NLOS Traps', fontsize=16, fontweight='bold')
    plt.xlabel('X Coordinate (m)', fontsize=12)
    plt.ylabel('Y Coordinate (m)', fontsize=12)
    
    custom_lines = [
        Line2D([0], [0], color='blue', marker='o', linestyle='None', markersize=8),
        Line2D([0], [0], color='red', marker='^', linestyle='None', markersize=10),
        Line2D([0], [0], color='gray', linestyle='--', lw=1.5),
        Line2D([0], [0], color='red', linestyle='-', lw=1.5),
        Line2D([0], [0], color='darkorange', linestyle=':', lw=2.5) # NLOS 图例
    ]
    plt.legend(custom_lines, ['Agent', 'Anchor', 'Agent-Agent Edge', 'Agent-Anchor Edge', 'NLOS Trap Edge'], loc='upper right')

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axis('equal') 
    plt.tight_layout()
    plt.show()
# %%
