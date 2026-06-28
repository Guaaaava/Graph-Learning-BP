"""
评估脚本: 在测试集上对比 Dense / BFS / GNN 三种拓扑

使用粒子化非参数 BP:
  1. 定位精度 (RMSE)         — mean ± std over graphs
  2. 理论下界 (CRLB)         — per-agent CRLB 均值
  3. 通信开销 (边数)          — 平均保留边数
  4. 中断概率 (Outage, >3m)  — 比例 + 95% CI
  5. 一致性 (NEES χ² 检验)   — 达标率
  6. RMSE 分位数            — P50 / P90 / P95

统计:
  - 所有指标报告 mean ± std (图间标准差)
  - 中断率和一致性报告 Wilson 95% 置信区间
  - 输出 LaTeX 表格行便于论文使用
"""

import torch
import numpy as np
import scipy.stats as stats
from torch_geometric.loader import DataLoader
from collections import defaultdict
import sys
import os
import json
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from BP.particle_bp import ParticleBP
from dataset import LocalizationDataset
from model import EdgePredictorGNN
from loss import compute_crlb
import config as config


# ================= Baseline 拓扑生成 =================
import networkx as nx


def baseline_mst(data):
    """MST by measurement variance (优先保留低噪声边)"""
    N, E = int(data.x.shape[0]), data.edge_index.shape[1]
    G = nx.Graph()
    for e in range(E):
        u, v = data.edge_index[0, e].item(), data.edge_index[1, e].item()
        G.add_edge(u, v, weight=data.edge_attr[e, 1].item(), idx=e)
    weights = torch.zeros(E, device=data.x.device)
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        if sub.number_of_edges() == 0:
            continue
        for u, v in nx.minimum_spanning_tree(sub, weight='weight').edges():
            weights[G[u][v]['idx']] = 1.0
    return weights


def baseline_knn(data, k=3):
    """k-NN: per-agent 保留最近 k 条 agent 边, anchor 边全留"""
    E = data.edge_index.shape[1]
    na = int(data.num_agents)
    is_anc = data.edge_attr[:, 3].bool()
    agt_mask = ~is_anc
    weights = torch.zeros(E, device=data.x.device)
    weights[is_anc] = 1.0
    row = data.edge_index[0, agt_mask]
    meas = data.edge_attr[agt_mask, 0]
    agt_idx = torch.where(agt_mask)[0]
    for i in range(na):
        m = (row == i)
        if m.sum() == 0:
            continue
        _, topk = torch.topk(meas[m], k=min(k, int(m.sum())), largest=False)
        weights[agt_idx[m][topk]] = 1.0
    return weights


def baseline_random(data, keep_ratio=0.5):
    """随机剪枝: anchor 全留, agent 边随机保留 keep_ratio"""
    is_anc = data.edge_attr[:, 3].bool()
    agt_mask = ~is_anc
    weights = torch.zeros(data.edge_index.shape[1], device=data.x.device)
    weights[is_anc] = 1.0
    agt_idx = torch.where(agt_mask)[0]
    n_keep = max(1, int(len(agt_idx) * keep_ratio))
    weights[agt_idx[torch.randperm(len(agt_idx))[:n_keep]]] = 1.0
    return weights


# ================= BFS 生成树提取 =================
def extract_bfs_weights(data):
    """
    从 PyG Data 对象的 Anchor 出发做多源 BFS，返回 0-1 边权重向量。

    策略:
      1. 从所有 Anchor 同时出发做多源 BFS，覆盖所有能连接到锚点的节点
      2. 对剩余孤立组件: 在稠密图中找最短路径连接到已访问集合，
         将该路径上的边加入 BFS 树 — 确保每个 Agent 都有到 Anchor 的路径
    """
    E = data.edge_index.shape[1]
    N = data.x.shape[0]
    weights = torch.zeros(E, device=data.x.device)

    # 构建无向邻接表 (稠密图的所有边)
    adj = {i: [] for i in range(N)}
    for e in range(E):
        u = data.edge_index[0, e].item()
        v = data.edge_index[1, e].item()
        adj[u].append((v, e))
        adj[v].append((u, e))

    is_anchor = data.x[:, 4].bool()
    anchors = torch.where(is_anchor)[0].tolist()

    # --- 第1步: 从所有 Anchor 出发做 BFS ---
    visited = set(anchors)
    queue = list(anchors)

    while queue:
        cur = queue.pop(0)
        for nxt, e_idx in adj[cur]:
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
                weights[e_idx] = 1.0

    # --- 第2步: 将孤立组件通过最短路径连接到已访问集合 ---
    remaining = set(range(N)) - visited

    while remaining:
        # BFS 从 visited 集合出发，找最近的 remaining 节点
        parent_edge = {}       # node -> (parent_node, edge_idx)
        local_visited = set(visited)
        q = list(visited)
        found = None

        while q and found is None:
            cur = q.pop(0)
            for nxt, e_idx in adj[cur]:
                if nxt not in local_visited:
                    local_visited.add(nxt)
                    parent_edge[nxt] = (cur, e_idx)
                    q.append(nxt)
                    if nxt in remaining:
                        found = nxt
                        break

        if found is None:
            break  # 理论上不会发生 (图是连通的)

        # 回溯: 将 found → ... → visited 路径上的所有边加入 BFS 树
        node = found
        while node in parent_edge:
            parent, e_idx = parent_edge[node]
            weights[e_idx] = 1.0
            visited.add(node)
            node = parent

        remaining = set(range(N)) - visited

    return weights


# ================= 指标计算 =================
def calculate_metrics(est_pos, true_pos, est_cov, e_th=3.0, alpha=0.05):
    """
    Parameters
    ----------
    est_pos : Tensor (N, 2)
    true_pos : Tensor (N, 2)
    est_cov : Tensor (N, 2, 2)
    e_th : 中断概率阈值 (m)
    alpha : 卡方检验显著性水平

    Returns
    -------
    rmse, outage_prob, consistency_rate, mean_nees, per_agent_errors
    """
    N = est_pos.shape[0]
    errors = est_pos - true_pos
    dists = errors.norm(dim=1)

    rmse = torch.sqrt((dists ** 2).mean()).item()
    outage = (dists > e_th).float().mean().item()

    nees_vals = []
    for i in range(N):
        e = errors[i].unsqueeze(1)
        try:
            sigma_inv = torch.linalg.inv(est_cov[i])
            nees = (e.T @ sigma_inv @ e).item()
        except RuntimeError:
            nees = float('inf')
        nees_vals.append(nees)

    nees_arr = np.array(nees_vals)
    r1 = stats.chi2.ppf(alpha / 2, df=2)
    r2 = stats.chi2.ppf(1 - alpha / 2, df=2)
    consistent = ((nees_arr >= r1) & (nees_arr <= r2)).sum()
    consistency = consistent / N if N > 0 else 0
    valid = nees_arr[nees_arr != float('inf')]
    mean_nees = np.mean(valid) if len(valid) > 0 else 0.0

    return rmse, outage, consistency, mean_nees, dists.cpu().numpy()


def wilson_ci(p, n, z=1.96):
    """Wilson 二项比例置信区间"""
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0, centre - margin), min(1, centre + margin)


# ================= 评估主函数 =================
def evaluate(quick=False):
    """
    Parameters
    ----------
    quick : bool
        True 则只用少量图和粒子快速验证
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 1. 加载模型
    model = EdgePredictorGNN(
        node_in_dim=config.NODE_IN_DIM,
        edge_in_dim=config.EDGE_IN_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
    ).to(device)

    models_dir = os.path.join(SCRIPT_DIR, "models")
    model_path = os.path.join(models_dir, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"警告: 未找到 {model_path}, 使用随机权重")
    else:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 2. 加载测试集
    test_path = os.path.join(SCRIPT_DIR, "datasets/test_dataset.pt")
    test_dataset = LocalizationDataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 3. 初始化粒子 BP
    if quick:
        num_particles = 500
        max_graphs = 10
        bp_iter = 5
    else:
        num_particles = config.BP_NUM_PARTICLES_TEST
        max_graphs = len(test_dataset)
        bp_iter = config.BP_NUM_ITER

    bp = ParticleBP(
        num_particles=num_particles,
        sigma_meas=config.BP_SIGMA_MEAS,
        num_iter=bp_iter,
        init_cov=config.BP_INIT_COV,
    )

    # 4. 收集 per-graph 指标
    topo_names = ['dense', 'bfs', 'mst', 'knn', 'random', 'gnn']
    # 每张图每条拓扑一个 dict
    per_graph = {name: [] for name in topo_names}
    # 所有 agent 的误差 (用于分位数)
    all_agent_errors = {name: [] for name in topo_names}

    count = 0

    print(f"\n开始评估 (粒子数={num_particles}, iter={bp_iter}, 最大图数={max_graphs})")
    print("=" * 70)

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            if idx >= max_graphs:
                break

            data = data.to(device)
            agents_true = data.y
            E_total = data.edge_index.shape[1]

            # --- GNN 掩码 (GPU) ---
            gnn_w, _ = model(data, tau=0.1, hard=True)

            # --- 数据移到 CPU 跑 BP (避免 P×P 矩阵炸显存) ---
            data_cpu = data.to('cpu')
            agents_true_cpu = agents_true.to('cpu')
            gnn_w_cpu = gnn_w.to('cpu')

            dense_w = torch.ones(E_total)
            bfs_w = extract_bfs_weights(data_cpu)
            mst_w = baseline_mst(data_cpu)
            knn_w = baseline_knn(data_cpu, k=3)
            rnd_w = baseline_random(data_cpu, keep_ratio=0.5)

            masks = {
                'dense': dense_w, 'bfs': bfs_w, 'mst': mst_w,
                'knn': knn_w, 'random': rnd_w, 'gnn': gnn_w_cpu,
            }

            # --- 运行粒子 BP (CPU) ---
            results = {}
            for name, w in masks.items():
                pos, cov, _ = bp.run(data_cpu, edge_weights=w)
                crlb, _ = compute_crlb(data_cpu, w, prior_weight=config.FIM_PRIOR)
                results[name] = (pos, cov, w, crlb)

            # --- 详细指标 ---
            topo_results = []
            for name in ['dense', 'bfs', 'mst', 'knn', 'random', 'gnn']:
                pos, cov, w, crlb = results[name]
                topo_results.append((name, pos, cov, w, crlb))

            for name, pos, cov, w, crlb in topo_results:
                rmse, outage, consist, nees, agent_dists = calculate_metrics(
                    pos, agents_true_cpu, cov,
                    e_th=config.OUTAGE_THRESHOLD, alpha=config.ALPHA,
                )
                per_graph[name].append({
                    'rmse': rmse,
                    'crlb': crlb,
                    'outage': outage,
                    'consist': consist,
                    'nees': nees,
                    'edges': (w > 0.5).sum().item(),
                })
                all_agent_errors[name].extend(agent_dists.tolist())

            count += 1
            if count % 10 == 0:
                print(f"  已评估 {count}/{max_graphs} 张图...", flush=True)

    # 5. 统计计算
    print(f"\n评估完成，共 {count} 张图。计算统计...\n")

    def summarize(records):
        """records: list of dicts"""
        keys = ['rmse', 'crlb', 'outage', 'consist', 'nees', 'edges']
        result = {}
        for k in keys:
            vals = np.array([r[k] for r in records])
            result[k] = {'mean': np.mean(vals), 'std': np.std(vals, ddof=1)}
        # 中断率和一致性用 Wilson CI (以 agent 为单位)
        n_total_outage = sum(1 for r in records for _ in range(1))  # approximation
        # 使用图级 outage 来计算 CI
        outage_mean = result['outage']['mean']
        consist_mean = result['consist']['mean']
        result['outage']['ci_low'], result['outage']['ci_high'] = wilson_ci(outage_mean, len(records))
        result['consist']['ci_low'], result['consist']['ci_high'] = wilson_ci(consist_mean, len(records))
        return result

    stats_summary = {}
    for name in topo_names:
        stats_summary[name] = summarize(per_graph[name])

    # 6. RMSE 分位数
    rmse_quantiles = {}
    for name in topo_names:
        errs = np.array(all_agent_errors[name])
        rmse_quantiles[name] = {
            'p50': np.percentile(errs, 50),
            'p90': np.percentile(errs, 90),
            'p95': np.percentile(errs, 95),
            'max': np.max(errs),
        }

    # 7. 格式化输出
    s = {name: stats_summary[name] for name in topo_names}  # dense, bfs, mst, knn, random, gnn

    def fm(val, width=8, decimals=3):
        return f"{val['mean']:{width}.{decimals}f} ± {val['std']:.{decimals}f}"

    def fm_pct(val, width=7):
        return (f"{val['mean']*100:{width}.1f}% "
                f"[{val['ci_low']*100:.0f}–{val['ci_high']*100:.0f}]")

    edge_reduction = (1 - s['gnn']['edges']['mean'] / s['dense']['edges']['mean']) * 100

    # LaTeX 行
    latex_rows = "\n".join(
        f"    {name.capitalize():>7} & {s[name]['edges']['mean']:.0f} & "
        f"{s[name]['rmse']['mean']:.2f} +/- {s[name]['rmse']['std']:.2f} & "
        f"{s[name]['crlb']['mean']:.3f} & {s[name]['outage']['mean']*100:.1f} & "
        f"{s[name]['nees']['mean']:.1f} \\\\"
        for name in ['dense', 'bfs', 'mst', 'knn', 'random', 'gnn']
    )

    # 生成汇总表列名
    col_names = {'dense': 'Dense', 'bfs': 'BFS', 'mst': 'MST',
                 'knn': 'k-NN', 'random': 'Random', 'gnn': 'GNN'}
    name_list = ['dense', 'bfs', 'mst', 'knn', 'random', 'gnn']

    def row_fmt(label, key, width=10, decimals=3, pct=False):
        parts = [f"{label:<18}"]
        for name in name_list:
            val = s[name][key]
            if pct:
                parts.append(f" | {fm_pct(val, 7):<30}")
            else:
                parts.append(f" | {fm(val, width, decimals):<30}")
        return "".join(parts)

    header = f"{'指标':<18}" + "".join(f" | {col_names[n]:<30}" for n in name_list)
    sep = "-" * (18 + 32 * len(name_list))

    report = f"""
{'=' * (18 + 32 * len(name_list))}
测试集评估结果 ({count} 张图, 粒子 BP, P={num_particles}, iter={bp_iter})
{'=' * (18 + 32 * len(name_list))}

【指标汇总 — mean ± std (图间)】
{header}
{sep}
{row_fmt('边数', 'edges', 8, 0)}
{row_fmt('RMSE (m)', 'rmse', 8, 3)}
{row_fmt('CRLB (m)', 'crlb', 8, 4)}
{row_fmt('中断率 (>3m)', 'outage', pct=True)}
{row_fmt('一致性 (95% χ²)', 'consist', pct=True)}
{row_fmt('NEES', 'nees', 8, 2)}

【RMSE 分位数 — per-agent (m)】
{'分位数':<12} """ + "".join(f" | {col_names[n]:<12}" for n in name_list) + f"""
{'-' * (12 + 14 * len(name_list))}
""" + "\n".join(
    f"{q:<12} " + " ".join(f" | {rmse_quantiles[n][q.lower()]:<12.3f}" for n in name_list)
    for q in ['P50', 'P90', 'P95', 'Max']
) + f"""

【关键对比】
  GNN vs Dense:  RMSE 比 {s['gnn']['rmse']['mean']/s['dense']['rmse']['mean']:.3f}  边减少 {(1-s['gnn']['edges']['mean']/s['dense']['edges']['mean'])*100:.1f}%
  GNN vs BFS:    RMSE 比 {s['gnn']['rmse']['mean']/s['bfs']['rmse']['mean']:.3f}
  GNN vs MST:    RMSE 比 {s['gnn']['rmse']['mean']/s['mst']['rmse']['mean']:.3f}
  GNN vs k-NN:   RMSE 比 {s['gnn']['rmse']['mean']/s['knn']['rmse']['mean']:.3f}

【LaTeX 表格行 (论文用)】
{latex_rows}
{'=' * (18 + 32 * len(name_list))}
"""

    print(report)

    # 8. 保存报告
    report_path = os.path.join(SCRIPT_DIR, "evaluate_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"报告已保存至 {report_path}")

    # 9. 保存 JSON (便于后续分析)
    results_json = {
        'timestamp': datetime.now().isoformat(),
        'num_graphs': count,
        'num_particles': num_particles,
        'bp_iter': bp_iter,
        'summary': {
            name: {
                k: {'mean': float(v['mean']), 'std': float(v['std'])}
                for k, v in stats_summary[name].items()
                if 'mean' in v
            }
            for name in topo_names
        },
        'rmse_quantiles': rmse_quantiles,
        'per_graph': {
            name: per_graph[name]
            for name in topo_names
        },
    }
    json_path = os.path.join(SCRIPT_DIR, "evaluate_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, default=float)
    print(f"JSON 已保存至 {json_path}")

    return stats_summary, per_graph


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='快速测试模式 (10图, P=500)')
    args = parser.parse_args()
    evaluate(quick=args.quick)
