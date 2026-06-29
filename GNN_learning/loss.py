"""
GIB (Graph Information Bottleneck) 损失函数

L = log det(J^{-1}_{global}) + λ * KL(p_θ || q) + η * Σ max(0, 3 - D_i)

组件:
  1. FIM 任务充分性项: log det(J^{-1}) — 保留包含丰富 Fisher 信息的边
  2. KL 结构压缩项: KL(p||q) — 驱使模型逼近基于物理距离的先验
  3. 度数约束项: ReLU(3-D_i) — 保证每个盲节点至少有 3 条边(2D定位消歧)

先验: q(d) = 1 / (1 + exp(γ · d²))

CRLB: compute_crlb() — 计算每 agent 的定位误差 Cramér-Rao 下界 (米)
  CRLB_i = sqrt(trace([FIM^{-1}]_{2×2, agent i}))
"""

import torch
import torch.nn.functional as F


def compute_gib_loss(edge_logits, edge_weights, data,
                     gamma=0.001, lambda_reg=1.0, eta=10.0,
                     prior_weight=1e-3, sparsity_weight=0.0):
    """
    GIB 损失函数 — 支持 PyG Batch

    Parameters
    ----------
    edge_logits : Tensor (E_total,)
        GNN 边评分器的原始输出 (不含 Sigmoid)
    edge_weights : Tensor (E_total,)
        STE 硬掩码 z^{out} ∈ {0, 1}
    data : PyG Data / Batch
        包含 x, edge_index, edge_attr, y, batch, ptr, num_agents, num_anchors
    gamma : float
        先验衰减系数: q(d) = 1/(1+exp(γ·d²))
        建议取 1/(R/2)² 量级 (R为通信半径)
    lambda_reg : float
        KL 散度正则化权重
    eta : float
        度数约束惩罚权重
    prior_weight : float
        FIM 对角线先验, 防止矩阵奇异

    Returns
    -------
    total_loss : Tensor (scalar)
    loss_dict : dict
        {'fim': ..., 'kl': ..., 'degree': ..., 'active_edges': ...}
    """
    device = data.x.device
    is_batch = hasattr(data, 'num_graphs')
    batch_size = data.num_graphs if is_batch else 1

    # ============================================================
    # 0. 构建全局真值位置张量
    # ============================================================
    is_anchor = data.x[:, 4].bool()  # col4 = is_anchor flag
    is_agent = ~is_anchor

    true_pos_all = torch.zeros(data.x.shape[0], 2, device=device)
    true_pos_all[is_anchor] = data.x[is_anchor, :2].clone()
    true_pos_all[is_agent] = data.y

    # ============================================================
    # 1. 任务充分性: FIM 组装 & CRLB
    # ============================================================
    row, col = data.edge_index
    diff = true_pos_all[row] - true_pos_all[col]
    dist = diff.norm(dim=1) + 1e-8
    dir_vec = diff / dist.unsqueeze(1)

    var = data.edge_attr[:, 1] + 1e-6
    is_anchor_edge = data.edge_attr[:, 3].bool()  # col3 = is_anchor_edge flag
    w = edge_weights
    info_coeff = w / var

    J_e = dir_vec.unsqueeze(2) * info_coeff.view(-1, 1, 1)
    J_e = J_e * dir_vec.unsqueeze(1)

    # 准备 batch 分段信息
    if is_batch:
        ptr = data.ptr
        batch = data.batch
        num_agents = data.num_agents
    else:
        N_a = int(data.num_agents) if hasattr(data, 'num_agents') else \
              int((~is_anchor).sum().item())
        ptr = torch.tensor([0, data.x.shape[0]], device=device)
        node_to_graph = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)
        batch = node_to_graph
        num_agents = torch.tensor([N_a], device=device)

    total_fim = 0.0
    total_dof = 0  # 总自由度 = Σ 2*N_a，用于归一化

    for g in range(batch_size):
        n_start = ptr[g].item()
        n_end = ptr[g + 1].item()
        N_a = int(num_agents[g].item())
        total_dof += 2 * N_a  # 每 agent 贡献 2 个自由度 (x,y)

        edge_mask = (batch[row] == g)
        local_row = row[edge_mask] - n_start
        local_col = col[edge_mask] - n_start

        local_J_e = J_e[edge_mask]
        local_is_anc = is_anchor_edge[edge_mask]

        FIM_blocks = torch.zeros(N_a, N_a, 2, 2, device=device)

        # Anchor 边
        anc_mask = local_is_anc
        if anc_mask.any():
            u_a = local_row[anc_mask]
            J_a = local_J_e[anc_mask]
            FIM_blocks.index_put_((u_a, u_a), J_a, accumulate=True)

        # Agent-Agent 边
        agt_mask = ~local_is_anc
        if agt_mask.any():
            u_a = local_row[agt_mask]
            v_a = local_col[agt_mask]
            J_a = local_J_e[agt_mask]
            FIM_blocks.index_put_((u_a, u_a), J_a, accumulate=True)
            FIM_blocks.index_put_((v_a, v_a), J_a, accumulate=True)
            FIM_blocks.index_put_((u_a, v_a), -J_a, accumulate=True)
            FIM_blocks.index_put_((v_a, u_a), -J_a, accumulate=True)

        FIM = FIM_blocks.permute(0, 2, 1, 3).reshape(2 * N_a, 2 * N_a)
        FIM = FIM + torch.eye(2 * N_a, device=device) * prior_weight

        try:
            eigvals = torch.linalg.eigvalsh(FIM)
            eigvals = torch.clamp(eigvals, min=1e-8)
            total_fim += torch.sum(torch.log(1.0 / eigvals))
        except RuntimeError:
            total_fim += torch.trace(torch.linalg.inv(FIM))

    # 按总自由度归一化: FIM 随 agent 数线性增长, 除以 2*N_a 使量级与 KL 对齐
    loss_fim = total_fim / total_dof if total_dof > 0 else total_fim / batch_size

    # ============================================================
    # 2. 结构压缩: KL(p_θ || q)
    # ============================================================
    p = torch.sigmoid(edge_logits)                           # (E,) 软概率
    # 先验 q(d) = 1 / (1 + exp(γ * d²))
    true_dist = dist                                         # (E,) 边真实距离
    q = 1.0 / (1.0 + torch.exp(gamma * true_dist ** 2))     # (E,)

    eps = 1e-8
    # KL( Bernoulli(p) || Bernoulli(q) )
    kl_per_edge = (
        p * torch.log((p + eps) / (q + eps))
        + (1 - p) * torch.log((1 - p + eps) / (1 - q + eps))
    )
    loss_kl = kl_per_edge.mean()

    # ============================================================
    # 2b. 直接稀疏惩罚 (可选)
    # ============================================================
    if sparsity_weight > 0:
        loss_sparsity = p.mean()
    else:
        loss_sparsity = torch.tensor(0.0, device=device)

    # ============================================================
    # 3. 几何约束: Σ max(0, 3 - D_i)
    # ============================================================
    # 计算每个 Agent 在剪枝后的度数
    N_total = data.x.shape[0]
    degree = torch.zeros(N_total, device=device)

    # 每条激活的边为其 agent 端贡献度数
    active = edge_weights > 0.5
    # edge_index[0] 总是 agent
    degree.scatter_add_(0, row[active],
                        torch.ones(active.sum(), device=device))
    # Agent-Agent 边另一端也是 agent
    agt_active = active & ~is_anchor_edge
    degree.scatter_add_(0, col[agt_active],
                        torch.ones(agt_active.sum(), device=device))

    # 只惩罚 agent 节点
    agent_deg = degree[is_agent]
    degree_penalty = F.relu(3.0 - agent_deg)
    loss_degree = degree_penalty.mean()

    # ============================================================
    # 4. 总损失
    # ============================================================
    total_loss = loss_fim + lambda_reg * loss_kl + sparsity_weight * loss_sparsity + eta * loss_degree

    loss_dict = {
        'fim': loss_fim.item(),
        'kl': loss_kl.item(),
        'sparsity': loss_sparsity.item() if sparsity_weight > 0 else 0.0,
        'degree': loss_degree.item(),
        'active_edges': (edge_weights > 0.5).sum().item() / batch_size,
        'total': total_loss.item(),
    }

    return total_loss, loss_dict


def compute_crlb(data, edge_weights, prior_weight=0.5):
    """
    计算每 agent 的定位误差 Cramér-Rao 下界 (CRLB)

    CRLB_i = sqrt(trace([FIM^{-1}]_{2×2 block for agent i}))

    Parameters
    ----------
    data : PyG Data / Batch
    edge_weights : Tensor (E,)
        二值边掩码 (0=剪枝, 1=保留)
    prior_weight : float
        FIM 对角线先验, 防止矩阵奇异

    Returns
    -------
    crlb_mean : float
        所有 agent 的平均 CRLB (米)
    crlb_per_agent : Tensor (N_a,)
        每个 agent 的 CRLB (米)
    """
    device = data.x.device
    is_batch = hasattr(data, 'num_graphs')
    batch_size = data.num_graphs if is_batch else 1

    # ---- 解析节点 ----
    is_anchor = data.x[:, 4].bool()
    is_agent = ~is_anchor
    true_pos_all = torch.zeros(data.x.shape[0], 2, device=device)
    true_pos_all[is_anchor] = data.x[is_anchor, :2].clone()
    true_pos_all[is_agent] = data.y

    # ---- 方向向量 ----
    row, col = data.edge_index
    diff = true_pos_all[row] - true_pos_all[col]
    dist = diff.norm(dim=1) + 1e-8
    dir_vec = diff / dist.unsqueeze(1)

    var = data.edge_attr[:, 1] + 1e-6
    is_anchor_edge = data.edge_attr[:, 3].bool()
    w = edge_weights
    info_coeff = w / var

    # 每条边对 FIM 的贡献 (2×2 块)
    J_e = dir_vec.unsqueeze(2) * info_coeff.view(-1, 1, 1)
    J_e = J_e * dir_vec.unsqueeze(1)

    # ---- batch 分段 ----
    if is_batch:
        ptr = data.ptr
        batch = data.batch
        num_agents = data.num_agents
    else:
        N_a = int(data.num_agents) if hasattr(data, 'num_agents') else \
              int((~is_anchor).sum().item())
        ptr = torch.tensor([0, data.x.shape[0]], device=device)
        batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)
        num_agents = torch.tensor([N_a], device=device)

    all_crlb = []

    for g in range(batch_size):
        n_start = ptr[g].item()
        n_end = ptr[g + 1].item()
        N_a = int(num_agents[g].item())

        edge_mask = (batch[row] == g)
        local_row = row[edge_mask] - n_start
        local_col = col[edge_mask] - n_start

        local_J_e = J_e[edge_mask]
        local_is_anc = is_anchor_edge[edge_mask]

        FIM_blocks = torch.zeros(N_a, N_a, 2, 2, device=device)

        # Anchor 边
        anc_mask = local_is_anc
        if anc_mask.any():
            u_a = local_row[anc_mask]
            J_a = local_J_e[anc_mask]
            FIM_blocks.index_put_((u_a, u_a), J_a, accumulate=True)

        # Agent-Agent 边
        agt_mask = ~local_is_anc
        if agt_mask.any():
            u_a = local_row[agt_mask]
            v_a = local_col[agt_mask]
            J_a = local_J_e[agt_mask]
            FIM_blocks.index_put_((u_a, u_a), J_a, accumulate=True)
            FIM_blocks.index_put_((v_a, v_a), J_a, accumulate=True)
            FIM_blocks.index_put_((u_a, v_a), -J_a, accumulate=True)
            FIM_blocks.index_put_((v_a, u_a), -J_a, accumulate=True)

        FIM = FIM_blocks.permute(0, 2, 1, 3).reshape(2 * N_a, 2 * N_a)
        FIM = FIM + torch.eye(2 * N_a, device=device) * prior_weight

        try:
            FIM_inv = torch.linalg.inv(FIM)
        except RuntimeError:
            # FIM 奇异时的回退: 伪逆
            FIM_inv = torch.linalg.pinv(FIM)

        # 提取每个 agent 的 2×2 对角块
        for i in range(N_a):
            block = FIM_inv[2*i:2*i+2, 2*i:2*i+2]
            crlb_i = torch.sqrt(torch.trace(block)).item()
            all_crlb.append(crlb_i)

    crlb_tensor = torch.tensor(all_crlb, device=device)
    crlb_mean = crlb_tensor.mean().item()

    return crlb_mean, crlb_tensor
