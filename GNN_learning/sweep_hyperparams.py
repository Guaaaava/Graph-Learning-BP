"""
超参数网格搜索 — 平衡边压缩率 vs 定位精度

对 SPARSITY_WEIGHT 和 LAMBDA_REG 做网格搜索，
每轮训练后在验证集上运行粒子 BP 评估 RMSE 和边数，
找 Pareto 最优配置。

用法:
    # 快速筛选 (少 epoch, 少粒子)
    python sweep_hyperparams.py --quick

    # 完整搜索
    python sweep_hyperparams.py
"""

import torch
import sys
import os
import itertools
import json
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from BP.particle_bp import ParticleBP
from GNN_learning.dataset import LocalizationDataset
from GNN_learning.model import EdgePredictorGNN
from GNN_learning.loss import compute_gib_loss, compute_crlb
import GNN_learning.config as config
from GNN_learning.train import train_gnn
from torch_geometric.loader import DataLoader
import numpy as np


def evaluate_model_on_val(model, val_loader, bp, device):
    """
    在验证集上评估模型: RMSE + 边数 + CRLB
    返回 dict{metric: mean_over_graphs}
    """
    model.eval()
    metrics = {'rmse': [], 'edges': [], 'crlb': [], 'outage': [], 'consist': [], 'nees': []}

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            # GNN 剪枝
            gnn_w, _ = model(data, tau=0.1, hard=True)
            n_edges = (gnn_w > 0.5).sum().item()

            # 粒子 BP
            pos, cov, _ = bp.run(data, edge_weights=gnn_w)

            # 指标
            errors = pos - data.y
            dists = errors.norm(dim=1)
            rmse = torch.sqrt((dists ** 2).mean()).item()
            outage = (dists > config.OUTAGE_THRESHOLD).float().mean().item()

            # NEES + consistency
            nees_vals = []
            for i in range(pos.shape[0]):
                e = errors[i].unsqueeze(1)
                try:
                    si = torch.linalg.inv(cov[i])
                    nees_vals.append((e.T @ si @ e).item())
                except RuntimeError:
                    nees_vals.append(float('inf'))
            na = np.array(nees_vals)
            import scipy.stats as stats
            r1 = stats.chi2.ppf(config.ALPHA / 2, df=2)
            r2 = stats.chi2.ppf(1 - config.ALPHA / 2, df=2)
            consist = ((na >= r1) & (na <= r2)).sum() / len(na) if len(na) > 0 else 0
            mean_nees = np.mean(na[na != float('inf')]) if len(na) > 0 else 0

            # CRLB
            crlb_mean, _ = compute_crlb(data, gnn_w, prior_weight=config.FIM_PRIOR)

            metrics['rmse'].append(rmse)
            metrics['edges'].append(n_edges)
            metrics['crlb'].append(crlb_mean)
            metrics['outage'].append(outage)
            metrics['consist'].append(consist)
            metrics['nees'].append(mean_nees)

    return {k: float(np.mean(v)) for k, v in metrics.items()}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='快速模式 (少epoch, 少粒子)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ---- 搜索网格 ----
    sparsity_grid = [50, 80, 100, 120, 150]
    lambda_grid = [10, 20, 40]

    if args.quick:
        sparsity_grid = [80, 120, 150]
        lambda_grid = [20]
        screen_epochs = 50
        bp_particles = 500
        bp_iter = 5
        val_subset = 10
    else:
        screen_epochs = config.EPOCHS
        bp_particles = config.BP_NUM_PARTICLES_TRAIN
        bp_iter = config.BP_NUM_ITER
        val_subset = 50

    # ---- 加载验证集 ----
    val_path = os.path.join(SCRIPT_DIR, "datasets/val_dataset.pt")
    val_dataset = LocalizationDataset(val_path)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    bp = ParticleBP(
        num_particles=bp_particles,
        sigma_meas=config.BP_SIGMA_MEAS,
        num_iter=bp_iter,
        init_cov=config.BP_INIT_COV,
    )

    # ---- 计算 Dense baseline ----
    print("\n>>> 计算 Dense baseline ...")
    dense_metrics = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if i >= val_subset:
                break
            data = data.to(device)
            dense_w = torch.ones(data.edge_index.shape[1], device=device)
            pos, cov, _ = bp.run(data, edge_weights=dense_w)
            errors = pos - data.y
            rmse = torch.sqrt((errors.norm(dim=1) ** 2).mean()).item()
            dense_metrics.append({'rmse': rmse, 'edges': dense_w.sum().item()})

    dense_rmse = np.mean([m['rmse'] for m in dense_metrics])
    dense_edges = np.mean([m['edges'] for m in dense_metrics])
    print(f"Dense: RMSE={dense_rmse:.3f}m, Edges={dense_edges:.0f}")

    # ---- 网格搜索 ----
    results = []
    total = len(sparsity_grid) * len(lambda_grid)
    idx = 0

    for sp_w, lam in itertools.product(sparsity_grid, lambda_grid):
        idx += 1
        label = f"sp_w={sp_w}, λ={lam}"
        print(f"\n{'='*50}")
        print(f"[{idx}/{total}] {label}")
        print(f"{'='*50}")

        # 训练
        train_gnn(
            sparsity_weight=sp_w,
            lambda_reg=lam,
            eta=config.ETA,
            epochs=screen_epochs,
            lr=config.LR,
            tau_init=config.TAU_INIT,
            tau_decay=config.TAU_DECAY,
        )

        # 加载训练好的模型
        model = EdgePredictorGNN(
            node_in_dim=config.NODE_IN_DIM,
            edge_in_dim=config.EDGE_IN_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
        ).to(device)
        model_path = os.path.join(SCRIPT_DIR, "models/best_model.pth")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        # 评估
        metrics = evaluate_model_on_val(model, val_loader, bp, device)
        metrics['sparsity_weight'] = sp_w
        metrics['lambda_reg'] = lam
        metrics['edge_reduction'] = (1 - metrics['edges'] / dense_edges) * 100
        metrics['rmse_ratio'] = metrics['rmse'] / dense_rmse

        print(f"  边数: {metrics['edges']:.0f} ({metrics['edge_reduction']:.0f}%↓)")
        print(f"  RMSE: {metrics['rmse']:.3f}m (×{metrics['rmse_ratio']:.2f} vs Dense)")
        print(f"  CRLB: {metrics['crlb']:.3f}m")
        print(f"  中断率: {metrics['outage']*100:.1f}%")
        print(f"  NEES: {metrics['nees']:.2f}")

        results.append(metrics)

    # ---- Pareto 排序 ----
    # 按 RMSE 升序（精度优先）和 edge_reduction 降序（压缩率优先）
    print(f"\n{'='*80}")
    print("搜索结果汇总 (按 RMSE 升序)")
    print(f"{'='*80}")
    print(f"{'sp_w':<8} {'λ':<8} {'边数':<8} {'压缩%':<8} {'RMSE':<10} {'RMSE比':<8} {'CRLB':<10} {'中断%':<8} {'NEES':<8}")
    print("-" * 80)

    results.sort(key=lambda r: r['rmse'])
    for r in results:
        print(f"{r['sparsity_weight']:<8.0f} {r['lambda_reg']:<8.0f} "
              f"{r['edges']:<8.0f} {r['edge_reduction']:<7.1f}% "
              f"{r['rmse']:<10.3f} {r['rmse_ratio']:<8.3f} "
              f"{r['crlb']:<10.3f} {r['outage']*100:<7.1f}% "
              f"{r['nees']:<8.2f}")

    # ---- 推荐 ----
    # Pareto front: 不可被其他配置同时在 RMSE 和 edge_reduction 上超越
    print(f"\n{'='*80}")
    print("Pareto 前沿 (不可被同时在 RMSE 和压缩率上超越)")
    print(f"{'='*80}")

    pareto = []
    for r in results:
        dominated = any(
            (other['rmse'] <= r['rmse'] and other['edge_reduction'] >= r['edge_reduction'])
            and (other['rmse'] < r['rmse'] or other['edge_reduction'] > r['edge_reduction'])
            for other in results
        )
        if not dominated:
            pareto.append(r)

    pareto.sort(key=lambda r: r['rmse'])
    for r in pareto:
        print(f"  sp_w={r['sparsity_weight']:.0f}, λ={r['lambda_reg']:.0f}: "
              f"RMSE={r['rmse']:.3f}m, Edges={r['edges']:.0f} ({r['edge_reduction']:.0f}%↓)")

    # ---- 保存 ----
    save_path = os.path.join(SCRIPT_DIR, "sweep_results.json")
    with open(save_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'dense_rmse': dense_rmse,
            'dense_edges': dense_edges,
            'pareto': [{'sparsity_weight': r['sparsity_weight'],
                        'lambda_reg': r['lambda_reg'],
                        'rmse': r['rmse'],
                        'edges': r['edges'],
                        'edge_reduction': r['edge_reduction']} for r in pareto],
            'all_results': results,
        }, f, indent=2)
    print(f"\n结果已保存至 {save_path}")


if __name__ == "__main__":
    main()
