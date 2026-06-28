#!/bin/bash
# 超参数网格搜索 — 30 epoch 快扫
# 用法: bash tune_hyperparams.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py310
cd /home/huangcy/Graph-Learning-BP/GNN_learning

echo "LAMBDA_REG  ETA  val_loss"
echo "------------------------"

for LAMBDA in 20 50 100 200; do
    for ETA in 20 50 100; do
        VAL=$(python -u -c "
import torch, sys, warnings; warnings.filterwarnings('ignore')
from dataset import LocalizationDataset
from model import EdgePredictorGNN
from loss import compute_gib_loss
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
import config
device = 'cuda'
train_ds = LocalizationDataset('datasets/train_dataset.pt')
val_ds = LocalizationDataset('datasets/val_dataset.pt')
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
model = EdgePredictorGNN(5,4,64,3).to(device)
opt = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
best_val = float('inf')
for epoch in range(1, 31):
    tau = max(0.1, 5.0 * (0.98**epoch))
    model.train()
    for batch in train_loader:
        batch = batch.to(device); opt.zero_grad()
        ew, logits = model(batch, tau=tau, hard=True)
        loss, _ = compute_gib_loss(logits, ew, batch,
            gamma=0.002, lambda_reg=$LAMBDA, eta=$ETA,
            prior_weight=0.5, sparsity_weight=0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    model.eval(); v_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            ew, logits = model(batch, tau=0.1, hard=True)
            loss, _ = compute_gib_loss(logits, ew, batch,
                gamma=0.002, lambda_reg=$LAMBDA, eta=$ETA,
                prior_weight=0.5, sparsity_weight=0)
            v_loss += loss.item()
    v_loss /= len(val_loader)
    if v_loss < best_val: best_val = v_loss
print(best_val)
" 2>/dev/null)
        echo "  $LAMBDA         $ETA   $VAL"
    done
done
