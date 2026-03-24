import torch
import torch.optim as optim
import math

from GNN_learning.generate_network import generate_localization_network
from GNN_learning.build_global_FIM import build_global_fim_vectorized
from GNN_learning.edge_predictor_GNN import EdgePredictorGNN

def train_gnn_sparsifier(epochs=1000, lr=0.01, lambda_reg=1.0):
    # 1. 初始化物理场景与数据
    print(">>> [1/3] 初始化协同定位物理场景...")
    data = generate_localization_network(
        num_agents=20, num_anchors=4, area_size=100.0, 
        comm_radius=40.0, base_noise=0.5, noise_scale=0.05
    )
    
    # 提取网络信息
    agents_pos = data['true_agents_pos']
    anchors_pos = data['anchors_pos']
    edge_index = data['edge_index']
    measurements = data['measurements']
    edge_variances = data['edge_variances']
    is_anchor_edge = data['is_anchor_edge']
    E = edge_index.shape[1]
    
    # 计算全图基准 CRLB (所有边权重设为 1.0)
    print(">>> [2/3] 计算稠密全图基准 CRLB...")
    with torch.no_grad():
        J_full = build_global_fim_vectorized(
            agents_pos, anchors_pos, edge_index, torch.ones(E), edge_variances, is_anchor_edge
        )
        eigenvalues_full = torch.linalg.eigvalsh(J_full)
        crlb_full = torch.sum(1.0 / torch.clamp(eigenvalues_full, min=1e-6))
        log_crlb_full = math.log(crlb_full.item())
    
    print(f"    原始稠密图边数: {E}")
    print(f"    原始稠密图 CRLB: {crlb_full.item():.4f}")
    
    # 2. 初始化 GNN 模型与优化器
    print("\n>>> [3/3] 开始 GNN 剪枝训练...")
    model = EdgePredictorGNN(node_in_dim=3, edge_in_dim=2, hidden_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 学习率衰减：每 50 轮将学习率乘以 0.5
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # Gumbel-Softmax 温度退火设置
    tau_init = 1.0
    tau_min = 0.1
    tau_decay = 0.997126 # 每个 epoch 衰减率
    tau = tau_init
    
    # 3. 主训练循环
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # --- A. 前向传播：预测边权重 ---
        edge_weights, logits = model(
            agents_pos, anchors_pos, edge_index, 
            measurements, edge_variances, is_anchor_edge, tau=tau
        )
        
        # --- B. 组装 FIM 并计算理论误差 ---
        J_global = build_global_fim_vectorized(
            agents_pos, anchors_pos, edge_index, edge_weights, edge_variances, is_anchor_edge
        )
        
        # 计算特征值与 CRLB
        eigenvalues = torch.linalg.eigvalsh(J_global)
        valid_eigenvalues = torch.clamp(eigenvalues, min=1e-6)
        crlb_raw = torch.sum(1.0 / valid_eigenvalues)

        # 相对精度恶化率 (当前 CRLB 取对数减去基准全图的 CRLB 对数)
        crlb_penalty = torch.log(crlb_raw) - log_crlb_full
        
        # 稀疏惩罚
        sparsity_loss = lambda_reg * torch.sum(edge_weights)
        
        # --- C. 计算总损失 (CRLB + 稀疏惩罚) ---
        # L1 稀疏正则化：鼓励 edge_weights 尽可能多地变成 0
        total_loss = crlb_penalty + sparsity_loss
        
        # --- D. 反向传播与优化 ---
        total_loss.backward()
        
        # 梯度裁剪 (防止 CRLB 倒数爆炸导致梯度失控)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        
        optimizer.step()
        scheduler.step()
        
        # --- E. 温度退火 (Annealing) ---
        tau = max(tau_min, tau * tau_decay)

        # --- F. 日志打印 ---
        if epoch % 20 == 0 or epoch == epochs - 1:
            active_edges = int(torch.sum(edge_weights).item())
            retention_rate = crlb_raw.item() / crlb_full.item()
            print(f"Epoch {epoch:03d} | Total Loss: {total_loss.item():.4f} "
                  f"| CRLB: {crlb_raw.item():.4f} (是基准的 {retention_rate:.2f} 倍) "
                  f"| 保留边数: {active_edges}/{E} | Tau: {tau:.3f}")

    # 4. 训练结束，输出最终稀疏图拓扑
    print("\n>>> 训练完成！")
    model.eval()
    with torch.no_grad():
        _, logits = model(
            agents_pos, anchors_pos, edge_index, 
            measurements, edge_variances, is_anchor_edge, tau=tau_min
        )
        # 剥离噪声，根据网络学到的 Logits 输出 0 或 1
        final_weights = (logits > 0).float()
        final_edges = int(torch.sum(final_weights).item())
        
        # 最终干净图的真实 CRLB
        J_final = build_global_fim_vectorized(agents_pos, anchors_pos, edge_index, final_weights, edge_variances, is_anchor_edge)
        crlb_final = torch.sum(1.0 / torch.clamp(torch.linalg.eigvalsh(J_final), min=1e-6))
        print(f"最终成果：网络从 {E} 条边精简至 {final_edges} 条边！")
        print(f"最终干净图 CRLB: {crlb_final.item():.4f} (是基准的 {crlb_final.item()/crlb_full.item():.2f} 倍)")
        
    # return model, final_weights, data

    return model, final_weights, data, final_edges, crlb_final.item(), crlb_full.item()

# 保留边数为什么会 12-77-60-65-73-74-... 这样浮动呢？--> Gumbel 噪声带来的探索