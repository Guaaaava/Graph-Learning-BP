import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from dataset import LocalizationDataset
from model import EdgePredictorGNN
from crlb_loss import compute_batched_crlb_loss
import config

def train_and_eval_for_lambda(train_loader, val_loader, lambda_reg, device, epochs=40):
    """
    针对特定的 lambda_reg 训练模型，并返回在验证集上的最终表现
    """
    print(f"\n[{'-'*15} 正在测试 LAMBDA_REG = {lambda_reg} {'-'*15}]")
    
    # 初始化全新的模型和优化器，防止状态残留
    model = EdgePredictorGNN(node_in_dim=3, edge_in_dim=3, hidden_dim=64, num_layers=3).to(device)
    optimizer = Adam(model.parameters(), lr=config.LR)
    
    for epoch in range(1, epochs + 1):
        # 较短的温度退火，适配较少的 epoch 数
        tau = max(0.5, 5.0 * np.exp(-0.1 * epoch))
        
        # --- 训练 ---
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            edge_weights, _ = model(batch, tau=tau, hard=True)
            loss, _, _ = compute_batched_crlb_loss(edge_weights, batch, lambda_reg=lambda_reg, prior_weight=1e-3)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
    # --- 验证 (跑完 epochs 后评估最终性能) ---
    model.eval()
    total_val_rmse = 0.0
    total_val_sparsity = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            edge_weights, _ = model(batch, tau=1.0, hard=True) # 硬剪枝输出
            
            # 1. 计算稀疏度 (保留了百分之几的边)
            # edge_weights 是 0 和 1 的向量，均值就是保留的比例
            sparsity = edge_weights.mean().item() 
            total_val_sparsity += sparsity
            
            # 2. 用 CRLB 的值来近似评估物理精度，CRLB 越小，精度下限越好
            _, loss_crlb, _ = compute_batched_crlb_loss(edge_weights, batch, lambda_reg=lambda_reg, prior_weight=1e-3)
            total_val_rmse += loss_crlb.item()

    avg_sparsity = total_val_sparsity / len(val_loader)
    avg_crlb = total_val_rmse / len(val_loader)
    
    print(f"测试完毕 -> 边保留率: {avg_sparsity*100:.1f}% | CRLB (误差下界): {avg_crlb:.4f}")
    return avg_sparsity, avg_crlb

def plot_pareto_frontier(results):
    """
    绘制散点图
    """
    lambdas = [res['lambda'] for res in results]
    sparsities = [res['sparsity'] * 100 for res in results] # 转换为百分比
    crlbs = [res['crlb'] for res in results]

    plt.figure(figsize=(10, 6))
    plt.scatter(sparsities, crlbs, color='#4A90E2', s=80, zorder=3)
    
    # 在每个点旁边标上对应的 lambda 值
    for i, txt in enumerate(lambdas):
        plt.annotate(f"λ={txt}", (sparsities[i], crlbs[i]), 
                     textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')

    plt.title('Pareto Frontier: Communication Cost vs. Localization Accuracy', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('Edges Retained (%) -> Communication Cost', fontsize=12)
    plt.ylabel('CRLB Trace (m²) -> Localization Error Bound', fontsize=12)
    
    # 反转 X 轴：让图表符合常理 (X轴向左代表边数越来越少/惩罚越来越大)
    plt.gca().invert_xaxis()
    
    plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
    plt.tight_layout()
    # 替换为保存图片，dpi=300 保证插入论文时非常清晰
    plt.savefig("tune_lambda_batch.png", dpi=300, bbox_inches='tight') 
    print("\n>>> 📊 网格搜索图已保存为 'tune_lambda_batch.png'")
    plt.close() # 释放内存

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据集 (使用较小的 batch size 保证内存安全)
    train_dataset = LocalizationDataset("datasets/train_dataset.pt")
    val_dataset = LocalizationDataset("datasets/val_dataset.pt")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=4,        # 开启 4 个子线程拼图
        pin_memory=True       # 锁页内存，加速数据传输到 GPU
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # 候选的 lambda 惩罚系数 (从极弱惩罚到极强惩罚)
    lambda_candidates = [14.0, 16.0, 18.0, 20.0]
    search_results = []
    
    print("=" * 60)
    print("🚀 开始启动 LAMBDA_REG 帕累托网格搜索...")
    print("=" * 60)
    
    for l_reg in lambda_candidates:
        # 为了加快搜索速度，每个 lambda 试探训练 40 轮即可看出趋势
        sparsity, crlb = train_and_eval_for_lambda(train_loader, val_loader, lambda_reg=l_reg, device=device, epochs=40)
        
        search_results.append({
            'lambda': l_reg,
            'sparsity': sparsity,
            'crlb': crlb
        })
        
    print("\n>>> 搜索完成！正在生成帕累托前沿散点图...")
    plot_pareto_frontier(search_results)
    
    # 在最后打印出格式化的综合指标表格
    print("\n" + "=" * 60)
    print(f"{'LAMBDA_REG':<12} | {'边保留率 (%)':<15} | {'CRLB (误差下界)':<15}")
    print("-" * 60)
    for res in search_results:
        print(f"{res['lambda']:<12.2f} | {res['sparsity']*100:<17.1f} | {res['crlb']:<15.4f}")
    print("=" * 60)