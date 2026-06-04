# %%

import torch
import matplotlib.pyplot as plt
import numpy as np

import GNN_learning.config as config
from GNN_learning.train import train_gnn_sparsifier

def search_best_lambda():
    # 1. 定义要探索的 lambda 候选列表
    # lambda_candidates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    lambda_candidates = [0.07, 0.09, 0.11, 0.12, 0.13]
    
    results_edges = []
    results_crlb_ratio = []
    
    print("==================================================")
    print("开始执行 Lambda 搜索")
    print("==================================================")
    
    # 2. 遍历每一个 lambda 进行完整训练
    for lam in lambda_candidates:
        print(f"\n>>> 正在测试 lambda_reg = {lam} ...")
        torch.manual_seed(config.TORCH_SEED)
        _, _, data, final_edges, crlb_final, crlb_full = train_gnn_sparsifier(
            epochs=config.EPOCHS, lr=0.01, lambda_reg=lam
        )
        
        crlb_ratio = crlb_final / crlb_full
        results_edges.append(final_edges)
        results_crlb_ratio.append(crlb_ratio)
        
        print(f"[测试完成] Lambda: {lam} | 保留边数: {final_edges} | CRLB 倍数: {crlb_ratio:.2f}")

    # 3. 绘制曲线
    print("\n>>> 所有测试完成，正在绘制曲线...")
    plt.figure(figsize=(10, 6))
    
    # 画出散点和连线
    plt.plot(results_edges, results_crlb_ratio, marker='o', linestyle='', color='#4A90E2', linewidth=2, markersize=8)
    
    # 标注每个点对应的 lambda 值
    for i, lam in enumerate(lambda_candidates):
        plt.annotate(
            f'λ={lam}', 
            (results_edges[i], results_crlb_ratio[i]),
            textcoords="offset points", 
            xytext=(0,10), 
            ha='center',
            fontsize=10,
            fontweight='bold',
            color='#E94A4A'
        )

    # 设置图表格式
    E_total = data['edge_index'].shape[1]
    plt.title('Pareto Frontier: Accuracy vs. Sparsity', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel(f'Retained Edges (Total: {E_total})', fontsize=12)
    plt.ylabel('CRLB Degradation Ratio (Current / Baseline)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

    # 4. 输出结果汇总表
    print("\n===========================================================")
    print("搜索结果汇总表")
    print("===========================================================")
    # 打印表头，确保对齐
    print(f"{'Lambda (λ)':<12} | {'保留边数 / 总边数':<18} | {'CRLB 倍率':<12}")
    print("-" * 59)
    
    # 遍历输出每一行数据
    for i, lam in enumerate(lambda_candidates):
        edge_str = f"{results_edges[i]} / {E_total}"
        crlb_str = f"{results_crlb_ratio[i]:.4f}"
        print(f"{lam:<12} | {edge_str:<21} | {crlb_str:<12}")
    print("===========================================================")

if __name__ == "__main__":
    search_best_lambda()

# seed | lambda_reg | challege
#  1   |    0.25    |   0.19
#  2   |    0.04    |   0.11
#  3   |    0.25    |   0.05
#  4   |    0.1     |   0.6
#  5   |    1.1     |   0.4
#  6   |    0.3     |   0.7
# %%
