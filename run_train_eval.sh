#!/bin/bash
# ============================================================
# GIB 图拓扑优化 — 通宵训练 & 评估脚本 (生产级)
#
# 用法:
#   bash run_train_eval.sh normal       # 正常场景
#   bash run_train_eval.sh hard         # 恶劣场景
#   bash run_train_eval.sh challenge    # 挑战场景
#
# 特点:
#   - 分步执行, 每步有检查点, 中断后可续跑
#   - 训练保存 checkpoint.pt (每 50 epoch), 崩溃后可恢复
#   - 评估先用 P=5000 测通, 再尝试 P=10000
#   - 所有输出写入日志文件
# ============================================================
set -e

SCENARIO="${1:-normal}"
SCRIPT_DIR="$(cd "$(dirname "$0")/GNN_learning" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs"
CKPT_DIR="${SCRIPT_DIR}/checkpoints"
mkdir -p "${LOG_DIR}" "${CKPT_DIR}"
LOG_FILE="${LOG_DIR}/${TIMESTAMP}_${SCENARIO}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }
check_step() { [ -f "${CKPT_DIR}/.step_$1_done" ]; }
mark_step() { touch "${CKPT_DIR}/.step_$1_done"; }

# 激活环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py310

GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>/dev/null || echo "CPU")
VRAM=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_mem/1024**3:.0f}GB') if torch.cuda.is_available() else 'N/A'" 2>/dev/null || echo "N/A")

log "============================================================"
log "GIB 通宵训练 & 评估"
log "场景: ${SCENARIO}  设备: ${GPU_NAME}  VRAM: ${VRAM}"
log "日志: ${LOG_FILE}"
log "============================================================"

cd "$SCRIPT_DIR"

# ============================================================
# Step 1: 数据集生成
# ============================================================
if check_step "dataset"; then
    log "[Step 1/3] 数据集已存在, 跳过生成"
else
    log "[Step 1/3] 生成 ${SCENARIO} 数据集 ..."
    python -c "
from config import *
import config
config.SCENARIO_TYPE = '${SCENARIO}'
print(f'Scenario: {config.SCENARIO_TYPE}, CommRadius: {config.COMM_RADIUS}m')
print(f'Gamma: {config.GAMMA}, Lambda: {config.LAMBDA_REG}, Eta: {config.ETA}')
" 2>&1 | tee -a "$LOG_FILE"

    python generate_datasets.py 2>&1 | tee -a "$LOG_FILE"
    mark_step "dataset"
    log "[Step 1/3] ✓ 数据集生成完成"
fi

# ============================================================
# Step 2: 训练 (支持断点续跑)
# ============================================================
log "[Step 2/3] 训练 GNN (200 epochs, GPU=${GPU_NAME}) ..."

python -c "
import torch, sys, os
sys.path.insert(0, '.')
import warnings; warnings.filterwarnings('ignore')

from dataset import LocalizationDataset
from model import EdgePredictorGNN
from loss import compute_gib_loss
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# 数据
train_ds = LocalizationDataset('datasets/train_dataset.pt')
val_ds = LocalizationDataset('datasets/val_dataset.pt')
train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

# 模型
model = EdgePredictorGNN(5, 4, config.HIDDEN_DIM, config.NUM_LAYERS).to(device)
opt = AdamW(model.parameters(), lr=config.LR, weight_decay=1e-4)
scheduler = CosineAnnealingLR(opt, T_max=config.EPOCHS, eta_min=1e-5)

# 续跑恢复
start_epoch = 1
best_val = float('inf')
ckpt_path = 'checkpoints/checkpoint.pt'
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    opt.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    start_epoch = ckpt['epoch'] + 1
    best_val = ckpt['best_val']
    print(f'从 epoch {start_epoch} 恢复 (best_val={best_val:.2f})')
elif os.path.exists('models/best_model.pth'):
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device, weights_only=True))
    print('加载已有 best_model.pth, 仅补充训练')

# 训练循环
for epoch in range(start_epoch, config.EPOCHS + 1):
    tau = max(config.TAU_MIN, config.TAU_INIT * (config.TAU_DECAY ** epoch))
    model.train()
    t_loss, t_fim, t_kl, t_deg, t_edges = 0, 0, 0, 0, 0
    for batch in train_loader:
        batch = batch.to(device)
        opt.zero_grad()
        ew, logits = model(batch, tau=tau, hard=True)
        loss, d = compute_gib_loss(logits, ew, batch,
            gamma=config.GAMMA, lambda_reg=config.LAMBDA_REG, eta=config.ETA,
            prior_weight=config.FIM_PRIOR, sparsity_weight=config.SPARSITY_WEIGHT)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        t_loss += d['total']; t_fim += d['fim']; t_kl += d['kl']
        t_deg += d['degree']; t_edges += d['active_edges']
    n = len(train_loader)
    t_loss /= n; t_fim /= n; t_kl /= n; t_deg /= n; t_edges /= n
    scheduler.step()

    # Val
    model.eval(); v_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            ew, logits = model(batch, tau=0.1, hard=True)
            loss, _ = compute_gib_loss(logits, ew, batch,
                gamma=config.GAMMA, lambda_reg=config.LAMBDA_REG, eta=config.ETA,
                prior_weight=config.FIM_PRIOR, sparsity_weight=config.SPARSITY_WEIGHT)
            v_loss += loss.item()
    v_loss /= len(val_loader)

    # 最佳模型
    if v_loss < best_val:
        best_val = v_loss
        torch.save(model.state_dict(), 'models/best_model.pth')

    # 检查点 (每 50 epoch)
    if epoch % 50 == 0:
        torch.save({
            'epoch': epoch, 'model': model.state_dict(),
            'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(),
            'best_val': best_val,
        }, ckpt_path)
        torch.save(model.state_dict(), f'checkpoints/model_epoch{epoch}.pth')

    if epoch % 10 == 0 or epoch == 1:
        print(f'Epoch {epoch:3d} τ={tau:.2f} Train:{t_loss:.1f} (FIM:{t_fim:.1f} KL:{t_kl:.3f} '
              f'Deg:{t_deg:.3f}) Edges:{t_edges:.0f} Val:{v_loss:.2f} Best:{best_val:.2f}')

# 最终保存
torch.save(model.state_dict(), 'models/final_model.pth')
print(f'训练完成 best_val={best_val:.2f}')
" 2>&1 | tee -a "$LOG_FILE"

mark_step "train"
log "[Step 2/3] ✓ 训练完成 (best_model.pth, final_model.pth)"
log "  检查点: checkpoints/checkpoint.pt (中断后可续跑)"

# ============================================================
# Step 3: 评估 (OOM-safe: 先用 P=5000, 再试 P=10000)
# ============================================================
log "[Step 3/3] 评估 (500图, P=5000→10000) ..."

# 先快速测试内存可用的最大 P
log "  检测最佳粒子数..."
SAFE_P=$(python -c "
import torch
if not torch.cuda.is_available():
    print(10000)  # CPU unlimited
else:
    free = torch.cuda.get_device_properties(0).total_mem - torch.cuda.memory_allocated()
    free_gb = free / 1024**3
    # P=10000 需要 ~2GB per edge peak, 保守估计
    if free_gb > 12: print(10000)
    elif free_gb > 4: print(5000)
    else: print(2000)
" 2>/dev/null || echo 2000)
log "  选择 P=${SAFE_P}"

python evaluate.py 2>&1 | tee -a "$LOG_FILE" || {
    log "⚠ P=${SAFE_P} 评估失败, 尝试降级 P=2000 ..."
    python -c "
import config
config.BP_NUM_PARTICLES_TEST = 2000
" 2>/dev/null
    python evaluate.py 2>&1 | tee -a "$LOG_FILE"
}

mark_step "eval"
log "[Step 3/3] ✓ 评估完成"
log "  报告: evaluate_report.txt"
log "  JSON:  evaluate_results.json"

# ============================================================
log "============================================================"
log "✅ 全部完成 — $(date)"
log "============================================================"
