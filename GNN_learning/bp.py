import torch

def gaussian_bp_localization(agents_pos_init, anchors_pos, edge_index, measurements, edge_variances, is_anchor_edge, edge_weights=None, num_iters=15):
    """
    标准线性化高斯置信传播 (Linearized Gaussian BP) 算法
    """
    N = agents_pos_init.shape[0]
    E = edge_index.shape[1]

    # 如果没有传入权重（比如在全图上跑），默认所有存在的边权重为 1
    if edge_weights is None:
        edge_weights = torch.ones(E, device=agents_pos_init.device)

    # --- 1. 状态初始化 ---
    mu = agents_pos_init.clone()
    # 初始协方差：假设一开始大家对自己位置非常不确定 (100.0 是一个极大的方差)
    Sigma = torch.eye(2).unsqueeze(0).repeat(N, 1, 1) * 100.0
    
    # 加入极小的先验信息，防止有些节点被完全剪枝孤立时，矩阵求逆报错
    J_prior = torch.eye(2) * 1e-4
    h_prior = torch.zeros(2)

    # --- 2. 迭代消息传递 ---
    for it in range(num_iters):
        J_new = J_prior.unsqueeze(0).repeat(N, 1, 1)
        h_new = h_prior.unsqueeze(0).repeat(N, 1)
        
        # 遍历每一条物理测距边
        for e in range(E):
            w = edge_weights[e]
            if w < 1e-3: # 如果 GNN 把这条边剪了，则不传递信息
                continue
                
            idx_u = edge_index[0, e] # 边的一端 (必定是 Agent)
            idx_v = edge_index[1, e] # 边的另一端
            
            # 确定 v 的状态 (如果 v 是 Anchor，则没有任何不确定性)
            if is_anchor_edge[e]:
                pos_v = anchors_pos[idx_v]
                Sigma_v = torch.zeros(2, 2) 
            else:
                pos_v = mu[idx_v]
                Sigma_v = Sigma[idx_v]
                
            pos_u = mu[idx_u]

            # --- [前向消息]：从 v 发送给 u ---
            diff = pos_u - pos_v
            dist = torch.norm(diff) + 1e-8
            u_vec = (diff / dist).view(2, 1) # 从 v 指向 u
            
            # 测距方差 + 发送方 v 的投影不确定度
            proj_var_v = torch.mm(torch.mm(u_vec.T, Sigma_v), u_vec).squeeze()
            msg_var_vu = edge_variances[e] + proj_var_v
            
            # 几何推断位置
            z_pos_u = pos_v + measurements[e] * u_vec.squeeze()
            
            # 组装消息并存入接收方 u
            J_msg_vu = (w / msg_var_vu) * torch.mm(u_vec, u_vec.T)
            h_msg_vu = torch.mv(J_msg_vu, z_pos_u)
            J_new[idx_u] += J_msg_vu
            h_new[idx_u] += h_msg_vu

            # --- [反向消息]：从 u 发送给 v (仅限 Agent-Agent 边) ---
            # 因为我们的 edge_index 只存了单向物理边 (i<j)，但 BP 是互相沟通的
            if not is_anchor_edge[e]:
                u_vec_rev = -u_vec # 方向翻转
                Sigma_u = Sigma[idx_u]
                
                # u 的投影不确定度
                proj_var_u = torch.mm(torch.mm(u_vec_rev.T, Sigma_u), u_vec_rev).squeeze()
                msg_var_uv = edge_variances[e] + proj_var_u
                
                # v 的几何推断位置
                z_pos_v = pos_u + measurements[e] * u_vec_rev.squeeze()
                
                # 组装消息并存入接收方 v
                J_msg_uv = (w / msg_var_uv) * torch.mm(u_vec_rev, u_vec_rev.T)
                h_msg_uv = torch.mv(J_msg_uv, z_pos_v)
                J_new[idx_v] += J_msg_uv
                h_new[idx_v] += h_msg_uv
                
        # --- 3. 节点信念更新 (Belief Update) ---
        for i in range(N):
            try:
                # 协方差等于信息矩阵的逆
                Sigma[i] = torch.linalg.inv(J_new[i])
                # 均值更新
                mu[i] = torch.mv(Sigma[i], h_new[i])
            except RuntimeError:
                pass # 忽略孤立节点的奇异矩阵，保持原位
                
    return mu, Sigma