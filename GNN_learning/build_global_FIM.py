import torch

def build_global_fim_vectorized(agents_pos, anchors_pos, edge_index, edge_weights, edge_variances, is_anchor_edge):
    """
    使用散点累加 (Scatter Add) 彻底向量化的 FIM 组装函数，极大地加速前向与反向传播。
    """
    N = agents_pos.shape[0]
    E = edge_index.shape[1]
    
    # 1. 向量化计算所有边的方向向量 u 和基础信息权重
    u_idx = edge_index[0] # Agent 的索引
    v_idx = edge_index[1] # 另一端的索引
    
    pos_u = agents_pos[u_idx]
    
    # 创建空张量，通过掩码分别赋值，避免数组越界
    pos_v = torch.zeros_like(pos_u)
    
    mask_ak = is_anchor_edge
    mask_aa = ~is_anchor_edge
    
    # 如果有连接 Anchor 的边，取出对应的 Anchor 坐标
    if mask_ak.any():
        pos_v[mask_ak] = anchors_pos[v_idx[mask_ak]]
    
    # 如果有连接 Agent 的边，取出对应的 Agent 坐标
    if mask_aa.any():
        pos_v[mask_aa] = agents_pos[v_idx[mask_aa]]
    
    diff = pos_u - pos_v
    dist = torch.norm(diff, dim=1, keepdim=True) + 1e-8
    u_vec = diff / dist # 形状: (E, 2)
    
    # 计算 I_base_weight = w / sigma^2
    I_weight = edge_weights / edge_variances # 形状: (E,)
    
    # 提取方向向量的 x 和 y 分量
    ux = u_vec[:, 0]
    uy = u_vec[:, 1]
    
    # 向量化计算 2x2 信息矩阵的三个独立元素 (由于对称矩阵, I_xy = I_yx)
    I_xx = I_weight * ux * ux
    I_yy = I_weight * uy * uy
    I_xy = I_weight * ux * uy
    
    # 2. 准备一维展平的全局 FIM 容器，大小为 (2N * 2N)
    N2 = 2 * N
    # 初始化微小先验
    # TODO 1e-6 还是 1e-4？
    J_global_flat = (torch.eye(N2) * 1e-4).view(-1) 
    
    # 辅助函数：将元素加到一维 FIM 的指定行列上
    def add_to_J(row, col, vals):
        # 计算在一维展平张量中的绝对索引
        flat_indices = row * N2 + col
        J_global_flat.scatter_add_(0, flat_indices, vals)

    # 3. 分类处理边并向量化填入
    # --- 处理 Agent-Anchor 边 (只加对角线) ---
    mask_ak = is_anchor_edge
    if mask_ak.any():
        u_ak = u_idx[mask_ak]
        r2u = 2 * u_ak
        # (2u, 2u) += I_xx,  (2u+1, 2u+1) += I_yy
        add_to_J(r2u, r2u, I_xx[mask_ak])
        add_to_J(r2u+1, r2u+1, I_yy[mask_ak])
        # (2u, 2u+1) += I_xy, (2u+1, 2u) += I_xy
        add_to_J(r2u, r2u+1, I_xy[mask_ak])
        add_to_J(r2u+1, r2u, I_xy[mask_ak])
        
    # --- 处理 Agent-Agent 边 (加对角线，减交叉项) ---
    mask_aa = ~is_anchor_edge
    if mask_aa.any():
        u_aa = u_idx[mask_aa]
        v_aa = v_idx[mask_aa]
        
        r2u = 2 * u_aa
        r2v = 2 * v_aa
        
        Ixx_aa, Iyy_aa, Ixy_aa = I_xx[mask_aa], I_yy[mask_aa], I_xy[mask_aa]
        
        # 自身信息相加 (对于 u)
        add_to_J(r2u, r2u, Ixx_aa);     add_to_J(r2u+1, r2u+1, Iyy_aa)
        add_to_J(r2u, r2u+1, Ixy_aa);   add_to_J(r2u+1, r2u, Ixy_aa)
        
        # 自身信息相加 (对于 v)
        add_to_J(r2v, r2v, Ixx_aa);     add_to_J(r2v+1, r2v+1, Iyy_aa)
        add_to_J(r2v, r2v+1, Ixy_aa);   add_to_J(r2v+1, r2v, Ixy_aa)
        
        # 交叉信息相减 (u, v) 和 (v, u)
        add_to_J(r2u, r2v, -Ixx_aa);    add_to_J(r2u+1, r2v+1, -Iyy_aa)
        add_to_J(r2u, r2v+1, -Ixy_aa);  add_to_J(r2u+1, r2v, -Ixy_aa)
        
        add_to_J(r2v, r2u, -Ixx_aa);    add_to_J(r2v+1, r2u+1, -Iyy_aa)
        add_to_J(r2v, r2u+1, -Ixy_aa);  add_to_J(r2v+1, r2u, -Ixy_aa)

    # 4. 重新 Reshape 回 2N x 2N 矩阵
    J_global = J_global_flat.view(N2, N2)
    return J_global