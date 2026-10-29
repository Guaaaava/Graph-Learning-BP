"""
Round 1: Lambda annealing + 50-agent challenge
  - Train (25-35 agents): 200 epoch with lambda schedule 0.5->2->10
  - Fine-tune (45-55 agents): 30 epoch, lambda = final schedule value
  - Evaluate (45-55 agents): 500 graphs P=10000
"""
import torch, sys, os, warnings, json
warnings.filterwarnings('ignore')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from dataset import LocalizationDataset
from model import EdgePredictorGNN
from loss import compute_gib_loss
from generate_datasets import create_dataset
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import config


def ds_path(split):
    return os.path.join(SCRIPT_DIR, f'datasets/{split}_dataset.pt')


def mdl_path(name):
    return os.path.join(SCRIPT_DIR, f'models/{name}.pth')


device = 'cuda'
ETA = 10  # fixed from previous sweep

def lambda_at_epoch(epoch):
    """Linear interpolation of config.LAMBDA_SCHEDULE"""
    sched = config.LAMBDA_SCHEDULE
    for i in range(len(sched) - 1):
        e0, l0 = sched[i]; e1, l1 = sched[i+1]
        if e0 <= epoch <= e1:
            return l0 + (l1 - l0) * (epoch - e0) / (e1 - e0)
    return sched[-1][1]

print(f"Device: {device}, Scenario: {config.SCENARIO_TYPE}")
print(f"Lambda schedule: {config.LAMBDA_SCHEDULE}")
print(f"ETA: {ETA}, GAMMA: {config.GAMMA}")

# ============================================================
# Phase 1: Generate small-scale data (25-35 agents)
# ============================================================
print("\n" + "="*60)
print("Phase 1: Small-scale data (25-35 agents)")
print("="*60)
generate_data = create_dataset          # (split, num_graphs, seed_offset, agents_min, agents_max)
generate_data('train', 2000, seed_offset=0, agents_min=25, agents_max=35)
generate_data('val', 500, seed_offset=1, agents_min=25, agents_max=35)

train_ds = LocalizationDataset(ds_path('train'))
val_ds = LocalizationDataset(ds_path('val'))
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

# ============================================================
# Phase 2: Train with lambda annealing (200 epochs)
# ============================================================
print("\n" + "="*60)
print("Phase 2: Training 200 epochs with lambda annealing")
print("="*60)

model = EdgePredictorGNN(5, 4, 64, 3).to(device)
opt = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = CosineAnnealingLR(opt, T_max=200, eta_min=1e-5)
best_val = float('inf')

for epoch in range(1, 201):
    tau = max(0.1, 5.0 * (0.98**epoch))
    lam = lambda_at_epoch(epoch)

    model.train()
    for batch in train_loader:
        batch = batch.to(device); opt.zero_grad()
        ew, logits = model(batch, tau=tau, hard=True)
        loss, d = compute_gib_loss(logits, ew, batch,
            gamma=config.GAMMA, lambda_reg=lam, eta=ETA,
            prior_weight=config.FIM_PRIOR, sparsity_weight=0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

    model.eval(); v_loss = 0; v_edges = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            ew, logits = model(batch, tau=0.1, hard=True)
            loss, d = compute_gib_loss(logits, ew, batch,
                gamma=config.GAMMA, lambda_reg=lam, eta=ETA,
                prior_weight=config.FIM_PRIOR, sparsity_weight=0)
            v_loss += loss.item(); v_edges += d['active_edges']
    v_loss /= len(val_loader); v_edges /= len(val_loader)
    scheduler.step()

    if v_loss < best_val:
        best_val = v_loss
        torch.save(model.state_dict(), mdl_path('model_anneal_small'))

    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} tau={tau:.2f} lam={lam:.1f} "
              f"val={v_loss:.2f} edges={v_edges:.0f} best={best_val:.2f}", flush=True)

print(f"Small-scale training done. Best val: {best_val:.2f}")

# ============================================================
# Phase 3: Generate large-scale data (45-55 agents)
# ============================================================
print("\n" + "="*60)
print("Phase 3: Large-scale data (45-55 agents)")
print("="*60)
# ft_train 微调 / ft_val 选模型 / test 最终评估。三者的 seed_offset 必须互不相同,
# 否则同一 seed + 同一 agent 数会生成完全相同的 agent 位置 (torch.randn 确定性)。
generate_data('ft_train', 500, seed_offset=3, agents_min=45, agents_max=55)
generate_data('ft_val', 500, seed_offset=4, agents_min=45, agents_max=55)
generate_data('test', 500, seed_offset=2, agents_min=45, agents_max=55)

ft_loader = DataLoader(LocalizationDataset(ds_path('ft_train')),
                       batch_size=16, shuffle=True, num_workers=2)
ft_val_loader = DataLoader(LocalizationDataset(ds_path('ft_val')),
                           batch_size=16, shuffle=False, num_workers=2)

test_ds = LocalizationDataset(ds_path('test'))

# ============================================================
# Phase 4: Fine-tune on large-scale data (30 epochs)
# ============================================================
print("\n" + "="*60)
print("Phase 4: Fine-tuning 30 epochs on large-scale data")
print("="*60)

model.load_state_dict(torch.load(mdl_path('model_anneal_small'), map_location=device))
# Fine-tune with fixed lambda = final schedule value
ft_lam = config.LAMBDA_SCHEDULE[-1][1]
print(f"Fine-tune lambda: {ft_lam} (final schedule value)")

best_val = float('inf')
for epoch in range(1, 31):
    tau = max(0.1, 5.0 * (0.98**epoch))
    model.train()
    for batch in ft_loader:
        batch = batch.to(device); opt.zero_grad()
        ew, logits = model(batch, tau=tau, hard=True)
        loss, _ = compute_gib_loss(logits, ew, batch,
            gamma=config.GAMMA, lambda_reg=ft_lam, eta=ETA,
            prior_weight=config.FIM_PRIOR, sparsity_weight=0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    model.eval(); v_loss = 0; v_edges = 0
    with torch.no_grad():
        for batch in ft_val_loader:
            batch = batch.to(device)
            ew, logits = model(batch, tau=0.1, hard=True)
            loss, d = compute_gib_loss(logits, ew, batch,
                gamma=config.GAMMA, lambda_reg=ft_lam, eta=ETA,
                prior_weight=config.FIM_PRIOR, sparsity_weight=0)
            v_loss += loss.item(); v_edges += d['active_edges']
    v_loss /= len(ft_val_loader); v_edges /= len(ft_val_loader)
    if v_loss < best_val:
        best_val = v_loss
        torch.save(model.state_dict(), mdl_path('model_anneal_ft'))
    if epoch % 10 == 0:
        print(f"FT Epoch {epoch:2d} val={v_loss:.2f} edges={v_edges:.0f} best={best_val:.2f}", flush=True)

print(f"Fine-tuning done. Best val: {best_val:.2f}")

# ============================================================
# Phase 5: Evaluate
# ============================================================
print("\n" + "="*60)
print("Phase 5: Evaluation (500 graphs, P=10000)")
print("="*60)

# Load best fine-tuned model for evaluation
model.load_state_dict(torch.load(mdl_path('model_anneal_ft'), map_location=device))

import subprocess
subprocess.run([sys.executable, "-u", "evaluate.py"], check=True, cwd=SCRIPT_DIR)
os.rename(os.path.join(SCRIPT_DIR, 'evaluate_report.txt'),
          os.path.join(SCRIPT_DIR, 'evaluate_report_anneal.txt'))
os.rename(os.path.join(SCRIPT_DIR, 'evaluate_results.json'),
          os.path.join(SCRIPT_DIR, 'evaluate_results_anneal.json'))

print("\n=== Lambda annealing experiment COMPLETE ===")
print(f"Schedule: {config.LAMBDA_SCHEDULE}")
print(f"ETA: {ETA}")
