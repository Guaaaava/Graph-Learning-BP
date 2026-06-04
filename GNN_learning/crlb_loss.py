import torch

def compute_batched_crlb_loss(edge_weights, data, lambda_reg=0.2, prior_weight=1e-3):
    """
    计算基于 Batch 的大图 CRLB 损失函数
    :param edge_weights: GNN 预测的边保留权重 (E,)
    :param data: PyG 的 DataBatch 对象
    :param lambda_reg: 稀疏惩罚系数
    :param prior_weight: 先验信息权重 (防止矩阵奇异)
    """
    batch_size = data.num_graphs
    device = data.x.device
    
    # =================================================================
    # 1. 全局向量化预计算 Fisher 信息矩阵的基础组件 (J_e)
    # =================================================================
    true_pos_all = data.x[:, 0:2].clone()
    is_anchor_node = data.x[:, 2].bool()
    is_agent_node = ~is_anchor_node
    true_pos_all[is_agent_node] = data.y
    
    row, col = data.edge_index
    diff = true_pos_all[row] - true_pos_all[col]
    dist = torch.norm(diff, dim=1) + 1e-8
    dir_vec = diff / dist.unsqueeze(1) 
    
    w = edge_weights 
    var = data.edge_attr[:, 1] + 1e-6 
    info_coeff = w / var 
    
    J_e_global = torch.bmm(dir_vec.unsqueeze(2), dir_vec.unsqueeze(1)) * info_coeff.view(-1, 1, 1)
    
    # =================================================================
    # 2. 性能优化：无循环的张量化 FIM 组装
    # =================================================================
    total_crlb = 0.0
    n_offset = 0
    
    for i in range(batch_size):
        N_a = data.num_agents[i].item()
        N_anc = data.num_anchors[i].item()
        N_tot = N_a + N_anc
        
        edge_mask = (data.batch[data.edge_index[0]] == i)
        local_edge_index = data.edge_index[:, edge_mask] - n_offset
        local_J_e = J_e_global[edge_mask]
        local_is_anchor = data.edge_attr[edge_mask, 2].bool()
        
        # 创建一个 4 维的分块矩阵: (Agent数, Agent数, 2, 2)
        FIM_blocks = torch.zeros((N_a, N_a, 2, 2), device=device)
        
        u = local_edge_index[0]
        v = local_edge_index[1]
        
        # --- A. 处理所有 Anchor 边 ---
        is_anc = local_is_anchor
        if is_anc.any():
            u_anc = u[is_anc]
            J_anc = local_J_e[is_anc]
            # 一条指令：将所有的 J_anc 累加到 FIM_blocks 对角线对应位置
            FIM_blocks.index_put_((u_anc, u_anc), J_anc, accumulate=True)
            
        # --- B. 处理所有 Agent 边 ---
        is_agt = ~local_is_anchor
        if is_agt.any():
            u_agt = u[is_agt]
            v_agt = v[is_agt]
            J_agt = local_J_e[is_agt]
            
            # 四条指令：瞬间完成原本需要循环几百次的交叉赋值
            FIM_blocks.index_put_((u_agt, u_agt), J_agt, accumulate=True)
            FIM_blocks.index_put_((v_agt, v_agt), J_agt, accumulate=True)
            FIM_blocks.index_put_((u_agt, v_agt), -J_agt, accumulate=True)
            FIM_blocks.index_put_((v_agt, u_agt), -J_agt, accumulate=True)
            
        # --- C. 降维重构：将 4 维块矩阵完美转换为 2N_a x 2N_a 平面矩阵 ---
        # permute(0, 2, 1, 3) 是极其巧妙的数学变换，它自动把 [x, y] 交织在一起
        FIM = FIM_blocks.permute(0, 2, 1, 3).reshape(2 * N_a, 2 * N_a)
        
        # 加上先验，防止矩阵奇异
        FIM = FIM + torch.eye(2 * N_a, device=device) * prior_weight
        
        CRB = torch.linalg.inv(FIM)
        total_crlb += torch.trace(CRB)
        
        n_offset += N_tot
        
    avg_crlb_loss = total_crlb / batch_size
    avg_sparsity_loss = torch.mean(edge_weights) # 保留了"百分之几"的边 (0.0 ~ 1.0)
    
    # 因为平均 CRLB 大约在 800 左右，且 avg_sparsity_loss 最大为 1.0
    # 我们设定 SCALING_FACTOR = 800.0 来平衡二者水平
    SCALING_FACTOR = 800.0 
    total_loss = avg_crlb_loss + lambda_reg * SCALING_FACTOR * avg_sparsity_loss
    
    return total_loss, avg_crlb_loss, avg_sparsity_loss