# %%

import torch
import matplotlib.pyplot as plt

def generate_localization_network(num_agents=20, num_anchors=4, area_size=100.0, comm_radius=40.0, base_noise=0.5, noise_scale=0.05):
    """
    生成协同定位虚拟网络数据
    新增：衰减系数 noise_scale，按照实际情况，距离越远标准差越大
    """
    # 1. 坐标撒点 (在 area_size x area_size 的区域内均匀分布)
    # Agent 的真实坐标 (不可见，仅用于生成观测数据和最终评估)
    true_agents_pos = torch.rand((num_agents, 2)) * area_size
    # Anchor 的真实坐标 (作为绝对准确的先验知识)
    anchors_pos = torch.rand((num_anchors, 2)) * area_size

    # 模拟 Agent 的初始粗略位置估计 (真实位置 + 较大的随机误差，模拟尚未收敛的状态)
    init_agents_pos = true_agents_pos + torch.randn_like(true_agents_pos) * 5.0

    # 2. 建图与连边 (找出在通信半径内的节点对)
    edge_index_list = []
    is_anchor_edge_list = []
    true_distances = []
    
    # (A) 建立 Agent 与 Agent 之间的边
    # 严格保持物理边的唯一性 (i < j)，避免 FIM 信息重复计算
    for i in range(num_agents):
        for j in range(i + 1, num_agents): # 避免重复连边和自环
            dist = torch.norm(true_agents_pos[i] - true_agents_pos[j])
            if dist < comm_radius:
                edge_index_list.append([i, j])
                is_anchor_edge_list.append(False)
                true_distances.append(dist)

    # (B) 建立 Agent 与 Anchor 之间的边
    for i in range(num_agents):
        for k in range(num_anchors):
            dist = torch.norm(true_agents_pos[i] - anchors_pos[k])
            if dist < comm_radius:
                edge_index_list.append([i, k]) # 第一项是 Agent 索引，第二项是 Anchor 索引
                is_anchor_edge_list.append(True)
                true_distances.append(dist)

    # 转化为 Tensor
    edge_index = torch.tensor(edge_index_list).T # 形状: (2, E)
    is_anchor_edge = torch.tensor(is_anchor_edge_list, dtype=torch.bool) # 形状: (E,)
    true_distances = torch.tensor(true_distances)

    # 3. 测距生噪 (注入高斯噪声)
    # 实际观测距离 = 真实距离 + 高斯噪声
    # 距离越远，标准差越大 -> 方差越大 -> Fisher信息越小
    stds = base_noise + noise_scale * true_distances
    measurements = true_distances + torch.randn_like(true_distances) * stds
    # 保证距离不能为负
    measurements = torch.clamp(measurements, min=0.1)

    # 每条边的测距方差
    edge_variances = stds ** 2

    data = {
        "true_agents_pos": true_agents_pos,
        "init_agents_pos": init_agents_pos,
        "anchors_pos": anchors_pos,
        "edge_index": edge_index,
        "is_anchor_edge": is_anchor_edge,
        "measurements": measurements,
        "edge_variances": edge_variances
    }
    
    return data

# ================= 测试运行 =================
if __name__ == "__main__":
    torch.manual_seed(1) # 固定随机种子以复现结果
    # 缩小一点范围或者增加节点以确保能看到丰富的连线
    demo_data = generate_localization_network(num_agents=20, num_anchors=4, area_size=100.0,comm_radius=40.0, base_noise=0.5, noise_scale=0.05)
    
    print(f"生成的 Agent 数量: {demo_data['true_agents_pos'].shape[0]}")
    print(f"生成的 Anchor 数量: {demo_data['anchors_pos'].shape[0]}")
    print(f"生成的有效测距边总数: {demo_data['edge_index'].shape[1]}")
    print(f"其中包含 Anchor 的边数: {demo_data['is_anchor_edge'].sum().item()}")

    # ================= 网络可视化 =================
    # 将 Tensor 转换为 numpy 以便 matplotlib 绘图
    agents_pos = demo_data['true_agents_pos'].numpy()
    anchors_pos = demo_data['anchors_pos'].numpy()
    edge_index = demo_data['edge_index'].numpy()
    is_anchor_edge = demo_data['is_anchor_edge'].numpy()

    plt.figure(figsize=(10, 8))
    
    # 1. 绘制连线 (Edges)
    for idx in range(edge_index.shape[1]):
        u, v = edge_index[:, idx]
        
        if is_anchor_edge[idx]:
            # Agent 到 Anchor 的连线 (红色实线)
            x_coords = [agents_pos[u, 0], anchors_pos[v, 0]]
            y_coords = [agents_pos[u, 1], anchors_pos[v, 1]]
            plt.plot(x_coords, y_coords, color='red', linestyle='-', alpha=0.4, linewidth=1.5, zorder=1)
        else:
            # Agent 到 Agent 的连线 (灰色虚线)
            x_coords = [agents_pos[u, 0], agents_pos[v, 0]]
            y_coords = [agents_pos[u, 1], agents_pos[v, 1]]
            plt.plot(x_coords, y_coords, color='gray', linestyle='--', alpha=0.4, linewidth=1.0, zorder=1)

    # 2. 绘制节点 (Nodes)
    # 绘制 Agent 节点 (蓝色圆点)
    plt.scatter(agents_pos[:, 0], agents_pos[:, 1], c='blue', marker='o', s=80, label='Agent (True Pos)', zorder=2)
    for i in range(agents_pos.shape[0]):
        plt.text(agents_pos[i, 0] + 1, agents_pos[i, 1] + 1, str(i), fontsize=9, color='darkblue')

    # 绘制 Anchor 节点 (红色三角)
    plt.scatter(anchors_pos[:, 0], anchors_pos[:, 1], c='red', marker='^', s=150, label='Anchor', zorder=2)
    for i in range(anchors_pos.shape[0]):
        plt.text(anchors_pos[i, 0] + 1.5, anchors_pos[i, 1] + 1.5, f"A{i}", fontsize=11, fontweight='bold', color='darkred')

    # 3. 设置图表属性
    plt.title('Cooperative Localization Virtual Network', fontsize=16)
    plt.xlabel('X Coordinate (m)', fontsize=12)
    plt.ylabel('Y Coordinate (m)', fontsize=12)
    
    # 自定义图例，避免重复绘制线条
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='blue', marker='o', linestyle='None', markersize=8),
        Line2D([0], [0], color='red', marker='^', linestyle='None', markersize=10),
        Line2D([0], [0], color='gray', linestyle='--', lw=1.5),
        Line2D([0], [0], color='red', linestyle='-', lw=1.5)
    ]
    plt.legend(custom_lines, ['Agent', 'Anchor', 'Agent-Agent Edge', 'Agent-Anchor Edge'], loc='upper right')

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axis('equal')  # 保证X轴和Y轴的比例一致，这样通信半径才会看起来是个正圆
    plt.tight_layout()
    plt.show()
# %%
