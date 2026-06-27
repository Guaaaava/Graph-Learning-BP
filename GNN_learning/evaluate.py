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


# ================= BFS 生成树提取 =================
def extract_bfs_weights(data):
    """
    从 PyG Data 对象的 Anchor 出发做多源 BFS，返回 0-1 边权重向量。
    保证所有节点可达（未被 BFS 覆盖的孤立分量做额外 BFS）。
    """
    E = data.edge_index.shape[1]
    N = data.x.shape[0]
    weights = torch.zeros(E, device=data.x.device)

    adj = {i: [] for i in range(N)}
    for e in range(E):
        u = data.edge_index[0, e].item()
        v = data.edge_index[1, e].item()
        adj[u].append((v, e))
        adj[v].append((u, e))

    is_anchor = data.x[:, 4].bool()
    anchors = torch.where(is_anchor)[0].tolist()

    visited = set(anchors)
    queue = list(anchors)

    while queue:
        cur = queue.pop(0)
        for nxt, e_idx in adj[cur]:
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
                weights[e_idx] = 1.0

    for i in range(N):
        if i not in visited:
            visited.add(i)
            queue.append(i)
            while queue:
                cur = queue.pop(0)
                for nxt, e_idx in adj[cur]:
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
                        weights[e_idx] = 1.0

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
    topo_names = ['dense', 'bfs', 'gnn']
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

            # --- 生成三种拓扑的边掩码 ---
            dense_w = torch.ones(E_total, device=device)
            bfs_w = extract_bfs_weights(data)
            gnn_w, _ = model(data, tau=0.1, hard=True)

            # --- 运行粒子 BP ---
            pos_d, cov_d, _ = bp.run(data, edge_weights=dense_w)
            pos_b, cov_b, _ = bp.run(data, edge_weights=bfs_w)
            pos_g, cov_g, _ = bp.run(data, edge_weights=gnn_w)

            # --- CRLB ---
            crlb_d, _ = compute_crlb(data, dense_w, prior_weight=config.FIM_PRIOR)
            crlb_b, _ = compute_crlb(data, bfs_w, prior_weight=config.FIM_PRIOR)
            crlb_g, _ = compute_crlb(data, gnn_w, prior_weight=config.FIM_PRIOR)

            # --- 详细指标 ---
            topo_results = [
                ('dense', pos_d, cov_d, dense_w, crlb_d),
                ('bfs',   pos_b, cov_b, bfs_w,   crlb_b),
                ('gnn',   pos_g, cov_g, gnn_w,   crlb_g),
            ]

            for name, pos, cov, w, crlb in topo_results:
                rmse, outage, consist, nees, agent_dists = calculate_metrics(
                    pos, agents_true, cov,
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
            if count % 50 == 0:
                print(f"  已评估 {count} 张图...")

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
    d, b, g = stats_summary['dense'], stats_summary['bfs'], stats_summary['gnn']

    def fm(val, width=8, decimals=3):
        """format mean ± std"""
        return f"{val['mean']:{width}.{decimals}f} ± {val['std']:.{decimals}f}"

    def fm_pct(val, width=7):
        """format percentage with CI"""
        return (f"{val['mean']*100:{width}.1f}% "
                f"[{val['ci_low']*100:.0f}–{val['ci_high']*100:.0f}]")

    edge_reduction = (1 - g['edges']['mean'] / d['edges']['mean']) * 100

    # 7. LaTeX 行
    latex_rows = (
        f"    Dense & {d['edges']['mean']:.0f} & {d['rmse']['mean']:.2f} +/- {d['rmse']['std']:.2f} "
        f"& {d['crlb']['mean']:.3f} & {d['outage']['mean']*100:.1f} & {d['nees']['mean']:.1f} \\\\\n"
        f"    BFS   & {b['edges']['mean']:.0f} & {b['rmse']['mean']:.2f} +/- {b['rmse']['std']:.2f} "
        f"& {b['crlb']['mean']:.3f} & {b['outage']['mean']*100:.1f} & {b['nees']['mean']:.1f} \\\\\n"
        f"    GNN   & {g['edges']['mean']:.0f} & {g['rmse']['mean']:.2f} +/- {g['rmse']['std']:.2f} "
        f"& {g['crlb']['mean']:.3f} & {g['outage']['mean']*100:.1f} & {g['nees']['mean']:.1f} \\\\"
    )

    report = f"""
{'=' * 110}
测试集评估结果 ({count} 张图, 粒子 BP, P={num_particles}, iter={bp_iter})
{'=' * 110}

【指标汇总 — mean ± std (图间)】
{'指标':<18} | {'Dense':<32} | {'BFS':<32} | {'GNN':<32}
{'-' * 110}
{'边数':<20} | {d['edges']['mean']:<8.0f} ± {d['edges']['std']:<8.0f}     | {b['edges']['mean']:<8.0f} ± {b['edges']['std']:<8.0f}     | {g['edges']['mean']:<8.0f} ± {g['edges']['std']:<8.0f}
{'RMSE (m)':<20} | {fm(d['rmse'], 8, 3):<32} | {fm(b['rmse'], 8, 3):<32} | {fm(g['rmse'], 8, 3):<32}
{'CRLB (m)':<20} | {fm(d['crlb'], 8, 4):<32} | {fm(b['crlb'], 8, 4):<32} | {fm(g['crlb'], 8, 4):<32}
{'中断率 (>3m)':<20} | {fm_pct(d['outage'], 7):<32} | {fm_pct(b['outage'], 7):<32} | {fm_pct(g['outage'], 7):<32}
{'一致性 (95% χ²)':<20} | {fm_pct(d['consist'], 7):<32} | {fm_pct(b['consist'], 7):<32} | {fm_pct(g['consist'], 7):<32}
{'NEES':<20} | {fm(d['nees'], 7, 2):<32} | {fm(b['nees'], 7, 2):<32} | {fm(g['nees'], 7, 2):<32}

【RMSE 分位数 — per-agent (m)】
{'分位数':<12} | {'Dense':<12} | {'BFS':<12} | {'GNN':<12}
{'-' * 50}
{'P50':<12} | {rmse_quantiles['dense']['p50']:<12.3f} | {rmse_quantiles['bfs']['p50']:<12.3f} | {rmse_quantiles['gnn']['p50']:<12.3f}
{'P90':<12} | {rmse_quantiles['dense']['p90']:<12.3f} | {rmse_quantiles['bfs']['p90']:<12.3f} | {rmse_quantiles['gnn']['p90']:<12.3f}
{'P95':<12} | {rmse_quantiles['dense']['p95']:<12.3f} | {rmse_quantiles['bfs']['p95']:<12.3f} | {rmse_quantiles['gnn']['p95']:<12.3f}
{'Max':<12} | {rmse_quantiles['dense']['max']:<12.3f} | {rmse_quantiles['bfs']['max']:<12.3f} | {rmse_quantiles['gnn']['max']:<12.3f}

【关键对比】
  边减少: {edge_reduction:.1f}%
  GNN vs Dense RMSE 比: {g['rmse']['mean']/d['rmse']['mean']:.3f}
  GNN vs BFS  RMSE 比: {g['rmse']['mean']/b['rmse']['mean']:.3f}

【LaTeX 表格行 (论文用)】
{latex_rows}
{'=' * 110}
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
