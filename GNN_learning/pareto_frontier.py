# %%

import torch
import matplotlib.pyplot as plt
import numpy as np

# 假设你的训练函数保存在 train.py 中，请根据实际文件名导入
from train import train_gnn_sparsifier

def search_best_lambda():
    # 1. 定义我们要探索的 lambda 候选列表
    # 从“几乎不剪 (0.01)” 到 “疯狂乱剪 (0.4)”
    lambda_candidates = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
    results_edges = []
    results_crlb_ratio = []
    
    print("==================================================")
    print("开始执行 Lambda 网格搜索与帕累托前沿分析")
    print("==================================================")
    
    # 2. 遍历每一个 lambda 进行完整训练
    for lam in lambda_candidates:
        print(f"\n>>> 正在测试 lambda_reg = {lam} ...")
        _, _, data, final_edges, crlb_final, crlb_full = train_gnn_sparsifier(
            epochs=300, lr=0.01, lambda_reg=lam
        )
        
        crlb_ratio = crlb_final / crlb_full
        results_edges.append(final_edges)
        results_crlb_ratio.append(crlb_ratio)
        
        print(f"[测试完成] Lambda: {lam} | 保留边数: {final_edges} | CRLB 倍数: {crlb_ratio:.2f}")

    # 3. 绘制帕累托前沿曲线 (Pareto Frontier)
    print("\n>>> 所有测试完成，正在绘制帕累托前沿曲线...")
    plt.figure(figsize=(10, 6))
    
    # 画出散点和连线
    plt.plot(results_edges, results_crlb_ratio, marker='o', linestyle='-', color='#4A90E2', linewidth=2, markersize=8)
    
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
    
    # 反转 X 轴（通常稀疏度越小，边数越少，我们希望图向右看是精度变高，向左看是边数变少）
    plt.gca().invert_xaxis()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    search_best_lambda()