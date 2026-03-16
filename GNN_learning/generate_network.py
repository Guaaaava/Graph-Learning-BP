import torch

def generate_localization_network(num_agents=30, num_anchors=5, area_size=100.0, comm_radius=40.0, base_noise=0.5, noise_scale=0.05):
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
    torch.manual_seed(42) # 固定随机种子以复现结果
    demo_data = generate_localization_network(num_agents=20, num_anchors=4, comm_radius=40.0)
    
    print(f"生成的 Agent 数量: {demo_data['true_agents_pos'].shape[0]}")
    print(f"生成的 Anchor 数量: {demo_data['anchors_pos'].shape[0]}")
    print(f"生成的有效测距边总数: {demo_data['edge_index'].shape[1]}")
    print(f"其中包含 Anchor 的边数: {demo_data['is_anchor_edge'].sum().item()}")