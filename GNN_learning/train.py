"""
GNN 图拓扑优化训练

使用 GIB 损失训练 EdgePredictorGNN:
  - 前向: GNN → logits → Gumbel-Sigmoid STE → hard mask z^{out}
  - 损失: logdet(FIM^{-1}) + λ·KL(p||q) + η·Σ ReLU(3-D_i)
  - 退火: τ 从高到低衰减, 逐步从探索转向确定
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
import numpy as np
import os

from dataset import LocalizationDataset
from model import EdgePredictorGNN
from loss import compute_gib_loss
import config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def train_gnn(sparsity_weight=None, lambda_reg=None, eta=None, epochs=None,
              lr=None, tau_init=None, tau_decay=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 允许 CLI 覆盖 config 默认值
    _sp_w = sparsity_weight if sparsity_weight is not None else config.SPARSITY_WEIGHT
    _lam   = lambda_reg if lambda_reg is not None else config.LAMBDA_REG
    _eta   = eta if eta is not None else config.ETA
    _epochs = epochs if epochs is not None else config.EPOCHS
    _lr    = lr if lr is not None else config.LR
    _tau_init = tau_init if tau_init is not None else config.TAU_INIT
    _tau_decay = tau_decay if tau_decay is not None else config.TAU_DECAY

    # ==========================================
    # 1. 加载数据
    # ==========================================
    train_path = os.path.join(SCRIPT_DIR, "datasets/train_dataset.pt")
    val_path = os.path.join(SCRIPT_DIR, "datasets/val_dataset.pt")
    train_dataset = LocalizationDataset(train_path)
    val_dataset = LocalizationDataset(val_path)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                               shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                             shuffle=False, num_workers=0)

    # ==========================================
    # 2. 模型 & 优化器
    # ==========================================
    model = EdgePredictorGNN(
        node_in_dim=config.NODE_IN_DIM,
        edge_in_dim=config.EDGE_IN_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=_lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=_epochs, eta_min=1e-5)

    models_dir = os.path.join(SCRIPT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    best_val_loss = float('inf')

    print("=" * 60)
    print(f"开始训练 · Epochs={_epochs} · Batch={config.BATCH_SIZE}")
    print(f"GIB: λ={_lam}, γ={config.GAMMA}, η={_eta}, sp_w={_sp_w}")
    print("=" * 60)

    for epoch in range(1, _epochs + 1):
        # 温度退火
        tau = max(config.TAU_MIN, _tau_init * (_tau_decay ** epoch))

        # ================ 训练 ================
        model.train()
        train_loss, train_fim, train_kl, train_deg = 0, 0, 0, 0
        train_edges = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            edge_weights, logits = model(batch, tau=tau, hard=True)
            loss, d = compute_gib_loss(
                logits, edge_weights, batch,
                gamma=config.GAMMA,
                lambda_reg=_lam,
                eta=_eta,
                prior_weight=config.FIM_PRIOR,
                sparsity_weight=_sp_w,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += d['total']
            train_fim += d['fim']
            train_kl += d['kl']
            train_deg += d['degree']
            train_edges += d['active_edges']

        n_batch = len(train_loader)
        train_loss /= n_batch
        train_fim /= n_batch
        train_kl /= n_batch
        train_deg /= n_batch
        train_edges /= n_batch

        scheduler.step()

        # ================ 验证 ================
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                edge_weights, logits = model(batch, tau=0.1, hard=True)
                loss, _ = compute_gib_loss(
                    logits, edge_weights, batch,
                    gamma=config.GAMMA,
                    lambda_reg=_lam,
                    eta=_eta,
                    prior_weight=config.FIM_PRIOR,
                    sparsity_weight=_sp_w,
                )
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(models_dir, "best_model.pth"))

        # 日志
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | τ={tau:.2f} | "
                f"Train: {train_loss:.2f} (FIM:{train_fim:.1f} KL:{train_kl:.3f} Deg:{train_deg:.3f}) "
                f"Edges:{train_edges:.0f} | Val: {val_loss:.2f}"
            )

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(models_dir, "final_model.pth"))
    print(f"\n训练完成！最佳验证损失: {best_val_loss:.2f}")
    print(f"模型已保存至 {models_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GNN 图拓扑优化训练")
    parser.add_argument('--sparsity-weight', type=float, default=None,
                        help=f'稀疏惩罚权重 (默认: {config.SPARSITY_WEIGHT})')
    parser.add_argument('--lambda-reg', type=float, default=None,
                        help=f'KL 正则化权重 (默认: {config.LAMBDA_REG})')
    parser.add_argument('--eta', type=float, default=None,
                        help=f'度数约束权重 (默认: {config.ETA})')
    parser.add_argument('--epochs', type=int, default=None,
                        help=f'训练轮数 (默认: {config.EPOCHS})')
    parser.add_argument('--lr', type=float, default=None,
                        help=f'学习率 (默认: {config.LR})')
    parser.add_argument('--tau-init', type=float, default=None,
                        help=f'初始温度 (默认: {config.TAU_INIT})')
    parser.add_argument('--tau-decay', type=float, default=None,
                        help=f'温度衰减率 (默认: {config.TAU_DECAY})')
    args = parser.parse_args()
    train_gnn(
        sparsity_weight=args.sparsity_weight,
        lambda_reg=args.lambda_reg,
        eta=args.eta,
        epochs=args.epochs,
        lr=args.lr,
        tau_init=args.tau_init,
        tau_decay=args.tau_decay,
    )
