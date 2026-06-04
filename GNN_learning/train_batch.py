import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
import numpy as np

from dataset import LocalizationDataset
from model import EdgePredictorGNN
from crlb_loss import compute_batched_crlb_loss
import config

def train_generalized_gnn():
    # ==========================================
    # 1. 基础配置与设备选择
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 正在使用计算设备: {device}")
    
    # ==========================================
    # 2. 加载数据集
    # ==========================================
    print(">>> 正在加载泛化数据集...")
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
    
    # ==========================================
    # 3. 初始化模型与优化器
    # ==========================================
    model = EdgePredictorGNN(node_in_dim=3, edge_in_dim=3, hidden_dim=64, num_layers=3).to(device)
    optimizer = Adam(model.parameters(), lr=config.LR)
    # 当验证集 Loss 不再下降时，自动减小学习率
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # ==========================================
    # 4. 训练大循环
    # ==========================================
    print("\n==================================================")
    print("🚀 开始批量泛化训练 (Mini-batch Generalization)")
    print("==================================================")
    
    for epoch in range(1, config.EPOCHS_BATCH + 1):
        # --- 动态温度退火策略 (Temperature Annealing) ---
        # 让 tau 从 5.0 慢慢降到 0.5，前期剧烈探索，后期稳定剪枝
        tau = max(0.5, 5.0 * np.exp(-0.05 * epoch))
        
        # ----------------- [ 训练阶段 ] -----------------
        model.train()
        train_loss_total = 0.0
        train_crlb_total = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # 前向传播 (此时开启 hard=True 以应用 STE)
            edge_weights, _ = model(batch, tau=tau, hard=True)
            
            # 计算 Batch 级别的 CRLB 损失
            loss, loss_crlb, loss_reg = compute_batched_crlb_loss(
                edge_weights, batch, lambda_reg=config.LAMBDA_REG_BATCH, prior_weight=1e-3
            )
            
            # 反向传播与梯度更新
            loss.backward()
            
            # 加入梯度裁剪，防止矩阵求逆引发的梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            train_loss_total += loss.item()
            train_crlb_total += loss_crlb.item()
            
        avg_train_loss = train_loss_total / len(train_loader)
        avg_train_crlb = train_crlb_total / len(train_loader)
        
        # ----------------- [ 验证阶段 ] -----------------
        model.eval()
        val_loss_total = 0.0
        val_crlb_total = 0.0
        
        with torch.no_grad(): # 验证阶段不计算梯度
            for batch in val_loader:
                batch = batch.to(device)
                
                # 推断时 tau 失去意义，直接输出确定性的 Mask
                edge_weights, _ = model(batch, tau=1.0, hard=True)
                
                loss, loss_crlb, _ = compute_batched_crlb_loss(
                    edge_weights, batch, lambda_reg=config.LAMBDA_REG_BATCH, prior_weight=1e-3
                )
                val_loss_total += loss.item()
                val_crlb_total += loss_crlb.item()
                
        avg_val_loss = val_loss_total / len(val_loader)
        avg_val_crlb = val_crlb_total / len(val_loader)
        
        # 调整学习率
        scheduler.step(avg_val_loss)
        
        # 打印华丽的训练日志
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3}/{config.EPOCHS_BATCH} | Tau: {tau:.2f} | "
                  f"Train Loss: {avg_train_loss:.2f} (CRLB:{avg_train_crlb:.2f}) | "
                  f"Val Loss: {avg_val_loss:.2f} (CRLB:{avg_val_crlb:.2f})")
            
    # 保存泛化模型权重
    torch.save(model.state_dict(), "gnn_generalized_model.pth")
    print("\n>>> 🎉 训练完成！泛化模型权重已保存为 'gnn_generalized_model.pth'")

if __name__ == "__main__":
    train_generalized_gnn()