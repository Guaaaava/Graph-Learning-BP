import torch
import numpy as np
import scipy.stats as stats
from torch_geometric.loader import DataLoader

# 导入你项目的模块
from GNN_learning.dataset import LocalizationDataset
from GNN_learning.model import EdgePredictorGNN
from GNN_learning.crlb_loss import compute_batched_crlb_loss
import GNN_learning.config as config

def calculate_metrics(estimated_pos, true_pos, estimated_sigma, e_th=3.0, alpha=0.05):
    """
    严谨的评价指标计算
    """
    N = estimated_pos.shape[0]
    errors = estimated_pos - true_pos

    # 1. 基础物理指标 (RMSE)
    distances = torch.norm(errors, dim=1)
    rmse = torch.sqrt(torch.mean(distances**2)).item()

    # 2. 中断概率 (Outage Probability)
    outage_prob = (distances > e_th).float().mean().item()
    
    # 3. 一致性分析 (NEES & 卡方检验)
    nees_list = []
    for i in range(N):
        e_i = errors[i].unsqueeze(1) 
        sigma_i = estimated_sigma[i]
        try:
            sigma_inv = torch.linalg.inv(sigma_i)
            nees = torch.mm(torch.mm(e_i.T, sigma_inv), e_i).item()
        except RuntimeError:
            nees = float('inf') 
        nees_list.append(nees)
        
    nees_array = np.array(nees_list)
    r1 = stats.chi2.ppf(alpha / 2, df=2) 
    r2 = stats.chi2.ppf(1 - alpha / 2, df=2) 
    
    consistent_nodes = ((nees_array >= r1) & (nees_array <= r2)).sum()
    consistency_rate = consistent_nodes / N if N > 0 else 0
    
    valid_nees = nees_array[nees_array != float('inf')]
    mean_nees = np.mean(valid_nees) if len(valid_nees) > 0 else 0.0
    
    return rmse, outage_prob, consistency_rate, mean_nees

def extract_bfs_tree_weights_pyg(data):
    """
    [新增] 原生适配 PyG 的广度优先搜索 (BFS) 生成树提取器。
    从所有 Anchor 节点同时出发进行多源 BFS，保证生成的树最浅，通信延迟最低。
    """
    E = data.edge_index.shape[1]
    N = data.x.shape[0]
    weights = torch.zeros(E, device=data.x.device)

    # 1. 构建邻接表
    adj = {i: [] for i in range(N)}
    for e in range(E):
        u = data.edge_index[0, e].item()
        v = data.edge_index[1, e].item()
        adj[u].append((v, e))

    # 2. 找到所有 Anchor 作为 BFS 的起点集合 (多源 BFS)
    is_anchor = data.x[:, 2].bool()
    anchor_indices = torch.where(is_anchor)[0].tolist()
    
    visited = set(anchor_indices)
    queue = list(anchor_indices)

    # 3. 执行 BFS 遍历
    while queue:
        curr = queue.pop(0)
        for neighbor, edge_idx in adj[curr]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                
                # 激活正向边
                weights[edge_idx] = 1.0
                
                # 在无向图中，我们必须同时激活反向边 (neighbor -> curr) 以保证 BP 消息双向传递
                for n2, e_rev in adj[neighbor]:
                    if n2 == curr:
                        weights[e_rev] = 1.0
                        break

    # 4. 防止有极其孤立的 Agent 群连不上 Anchor
    for i in range(N):
        if i not in visited:
            visited.add(i)
            queue.append(i)
            while queue:
                curr = queue.pop(0)
                for neighbor, edge_idx in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        weights[edge_idx] = 1.0
                        for n2, e_rev in adj[neighbor]:
                            if n2 == curr:
                                weights[e_rev] = 1.0
                                break
    return weights

def gaussian_bp_evaluator_pyg(data, edge_weights=None, num_iters=25, tol=1e-2):
    """
    原生适配 PyG Data 对象的 Gaussian BP 裁判员
    """
    N_total = data.x.shape[0]
    E = data.edge_index.shape[1]
    
    if edge_weights is None:
        edge_weights = torch.ones(E, device=data.x.device)
        
    # 初始化信念 (mu 和 Sigma)
    # data.x 的前两列是初始坐标 (Agent是估计值，Anchor是真实值)，第三列是 Anchor 标志
    mu = data.x[:, :2].clone()
    Sigma = torch.eye(2, device=data.x.device).unsqueeze(0).repeat(N_total, 1, 1) * 100.0 
    
    is_anchor = data.x[:, 2].bool()
    is_agent = ~is_anchor
    Sigma[is_anchor] = torch.zeros(2, 2, device=data.x.device) # Anchor 方差为 0
    
    J_prior = torch.eye(2, device=data.x.device) * 1e-4
    h_prior = torch.zeros(2, device=data.x.device) 
    
    actual_iters = num_iters 
    
    for it in range(num_iters):
        mu_old = mu.clone() 
        J_new = J_prior.unsqueeze(0).repeat(N_total, 1, 1)
        h_new = h_prior.unsqueeze(0).repeat(N_total, 1)
        
        # --- 消息传递 ---
        for e in range(E):
            w = edge_weights[e]
            if w < 1e-3: continue
                
            u = data.edge_index[0, e] 
            v = data.edge_index[1, e] 
            is_anc_edge = data.edge_attr[e, 2].bool()
            
            pos_u, Sigma_u = mu[u], Sigma[u]
            pos_v, Sigma_v = mu[v], Sigma[v]
            z, var = data.edge_attr[e, 0], data.edge_attr[e, 1]

            # v 流向 u (u 是 Agent)
            diff = pos_u - pos_v
            dist = torch.norm(diff) + 1e-8
            u_vec = (diff / dist).view(2, 1) 
            
            proj_var_v = torch.mm(torch.mm(u_vec.T, Sigma_v), u_vec).squeeze()
            msg_var_vu = var + proj_var_v
            z_pos_u = pos_v + z * u_vec.squeeze()
            J_msg_vu = (w / msg_var_vu) * torch.mm(u_vec, u_vec.T)
            h_msg_vu = torch.mv(J_msg_vu, z_pos_u)
            
            J_new[u] += J_msg_vu
            h_new[u] += h_msg_vu

            # u 流向 v (仅当 v 也是 Agent 时)
            if not is_anc_edge:
                u_vec_rev = -u_vec 
                proj_var_u = torch.mm(torch.mm(u_vec_rev.T, Sigma_u), u_vec_rev).squeeze()
                msg_var_uv = var + proj_var_u
                z_pos_v = pos_u + z * u_vec_rev.squeeze()
                J_msg_uv = (w / msg_var_uv) * torch.mm(u_vec_rev, u_vec_rev.T)
                h_msg_uv = torch.mv(J_msg_uv, z_pos_v)
                
                J_new[v] += J_msg_uv
                h_new[v] += h_msg_uv
                
        # --- 信念更新 (仅更新 Agent) ---
        for i in range(N_total):
            if is_agent[i]:
                try:
                    Sigma[i] = torch.linalg.inv(J_new[i])
                    mu[i] = torch.mv(Sigma[i], h_new[i])
                except RuntimeError:
                    pass

        # --- 收敛检查 ---
        max_pos_change = torch.max(torch.norm(mu[is_agent] - mu_old[is_agent], dim=1))
        if max_pos_change < tol:
            actual_iters = it + 1 
            break 
            
    # 只返回 Agent 的预测结果
    return mu[is_agent], Sigma[is_agent], actual_iters

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 正在使用计算设备: {device}")
    
    # 1. 加载测试集和模型
    test_dataset = LocalizationDataset("GNN_learning/datasets/test_dataset.pt")
    # batch_size=1 让我们能够对每张图单独进行严格的 BP 迭代评估
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = EdgePredictorGNN(node_in_dim=3, edge_in_dim=3, hidden_dim=64, num_layers=3).to(device)
    model.load_state_dict(torch.load("GNN_learning/gnn_generalized_model.pth", map_location=device))
    model.eval()
    
    # 统计累加器
    metrics = {
        'dense': {'rmse': 0, 'outage': 0, 'consist': 0, 'nees': 0, 'iters': 0, 'crlb': 0, 'edges': 0},
        'bfs':   {'rmse': 0, 'outage': 0, 'consist': 0, 'nees': 0, 'iters': 0, 'crlb': 0, 'edges': 0},
        'gnn':   {'rmse': 0, 'outage': 0, 'consist': 0, 'nees': 0, 'iters': 0, 'crlb': 0, 'edges': 0}
    }
    
    num_graphs = len(test_dataset)
    print(f"\n>>> 开始评估 {num_graphs} 张泛化测试图...")
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            agents_pos_true = data.y
            E_total = data.edge_index.shape[1]
            
            # --- 1. 获取三种拓扑的权重 ---
            dense_weights = torch.ones(E_total, device=device)
            bfs_weights = extract_bfs_tree_weights_pyg(data)
            gnn_weights, _ = model(data, tau=1.0, hard=True)
            
            # --- 2. 运行 Gaussian BP ---
            mu_d, Sigma_d, iters_d = gaussian_bp_evaluator_pyg(data, dense_weights)
            mu_b, Sigma_b, iters_b = gaussian_bp_evaluator_pyg(data, bfs_weights)
            mu_g, Sigma_g, iters_g = gaussian_bp_evaluator_pyg(data, gnn_weights)
            
            # --- 3. 计算 CRLB ---
            _, crlb_d, _ = compute_batched_crlb_loss(dense_weights, data)
            _, crlb_b, _ = compute_batched_crlb_loss(bfs_weights, data)
            _, crlb_g, _ = compute_batched_crlb_loss(gnn_weights, data)
            
            # --- 4. 计算详细指标 ---
            r_d, o_d, c_d, n_d = calculate_metrics(mu_d, agents_pos_true, Sigma_d)
            r_b, o_b, c_b, n_b = calculate_metrics(mu_b, agents_pos_true, Sigma_b)
            r_g, o_g, c_g, n_g = calculate_metrics(mu_g, agents_pos_true, Sigma_g)
            
            # --- 5. 累加指标 ---
            # Dense
            metrics['dense']['rmse'] += r_d; metrics['dense']['outage'] += o_d
            metrics['dense']['consist'] += c_d; metrics['dense']['nees'] += n_d
            metrics['dense']['iters'] += iters_d; metrics['dense']['crlb'] += crlb_d.item()
            metrics['dense']['edges'] += E_total
            
            # BFS
            metrics['bfs']['rmse'] += r_b; metrics['bfs']['outage'] += o_b
            metrics['bfs']['consist'] += c_b; metrics['bfs']['nees'] += n_b
            metrics['bfs']['iters'] += iters_b; metrics['bfs']['crlb'] += crlb_b.item()
            metrics['bfs']['edges'] += bfs_weights.sum().item()
            
            # GNN
            metrics['gnn']['rmse'] += r_g; metrics['gnn']['outage'] += o_g
            metrics['gnn']['consist'] += c_g; metrics['gnn']['nees'] += n_g
            metrics['gnn']['iters'] += iters_g; metrics['gnn']['crlb'] += crlb_g.item()
            metrics['gnn']['edges'] += gnn_weights.sum().item()
            
            if (i + 1) % 10 == 0:
                print(f"进度: {i + 1} / {num_graphs} 图已评估...")
                
    # ==========================================
    # 打印并保存最终对比成绩单
    # ==========================================
    for k in metrics:
        for m in metrics[k]:
            metrics[k][m] /= num_graphs
            
    d = metrics['dense']
    b = metrics['bfs']
    g = metrics['gnn']
    edge_reduction = (1 - g['edges'] / d['edges']) * 100

    # 1. 使用 f-string 拼接多行字符串，保持完美的排版对齐
    report_text = f"""
{'=' * 120}
测试集实验结果 (500 张图)
{'=' * 120}
{'拓扑模型':<14} | {'平均边数':<8} | {'均次迭代':<8} | {'RMSE (m)':<10} | {'CRLB Trace':<10} | {'中断概率 (3m)':<13} | {'一致性达标率':<15}
{'-' * 120}
{'1. 原图 (Dense)':<16} | {d['edges']:<12.1f} | {d['iters']:<12.1f} | {d['rmse']:<12.4f} | {d['crlb']:<14.2f} | {d['outage']*100:<15.1f}% | {d['consist']*100:<15.1f}%
{'2. BFS 生成树':<15} | {b['edges']:<12.1f} | {b['iters']:<12.1f} | {b['rmse']:<12.4f} | {b['crlb']:<14.2f} | {b['outage']*100:<15.1f}% | {b['consist']*100:<15.1f}%
{'3. GNN 剪枝':<16} | {g['edges']:<12.1f} | {g['iters']:<12.1f} | {g['rmse']:<12.4f} | {g['crlb']:<14.2f} | {g['outage']*100:<15.1f}% | {g['consist']*100:<15.1f}%
{'=' * 120}
"""

    # 2. 在终端屏幕上打印出来
    print(report_text)
    
    # 3. 将字符串保存到本地文件中 (使用 utf-8 编码防止中文乱码)
    filename = "evaluate_batch_report.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    print(f"\n>>> ✅ 报告已成功保存至本地文件: {filename}")