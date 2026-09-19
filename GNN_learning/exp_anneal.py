"""
Round 1: Lambda annealing + 50-agent challenge
  - Train (25-35 agents): 200 epoch with lambda schedule 0.5->2->10
  - Fine-tune (45-55 agents): 30 epoch, lambda = final schedule value
  - Evaluate (45-55 agents): 500 graphs P=10000
"""
import torch, sys, os, warnings, json
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

from dataset import LocalizationDataset
from model import EdgePredictorGNN
from loss import compute_gib_loss
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import config

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

def generate_data(agents_min, agents_max, suffix):
    import random, numpy as np, generate_network
    from tqdm import tqdm
    num_graphs = {'train_': 2000, 'val_': 500, 'test_': 500}.get(suffix, 500)
    seed_offsets = {'train_': 0, 'val_': 1, 'test_': 2}
    offset = seed_offsets[suffix]
    random.seed(config.TORCH_SEED + offset)
    np.random.seed(config.TORCH_SEED + offset)
    torch.manual_seed(config.TORCH_SEED + offset)
    ds = []
    for _ in tqdm(range(num_graphs), desc=f"Generating {suffix}"):
        na = random.randint(agents_min, agents_max)
        nc = random.randint(config.NUM_ANCHORS_MIN, config.NUM_ANCHORS_MAX)
        d = generate_network.generate_localization_network(
            num_agents=na, num_anchors=nc,
            area_size=config.AREA_SIZE, comm_radius=config.COMM_RADIUS,
            base_noise=config.BASE_NOISE, noise_scale=config.NOISE_SCALE,
            init_pos_cov=config.INIT_POS_COV, scenario_type=config.SCENARIO_TYPE)
        d['num_agents'] = na; d['num_anchors'] = nc
        ds.append(d)
    path = f'datasets/{suffix}dataset.pt'
    torch.save(ds, path)
    return ds

print(f"Device: {device}, Scenario: {config.SCENARIO_TYPE}")
print(f"Lambda schedule: {config.LAMBDA_SCHEDULE}")
print(f"ETA: {ETA}, GAMMA: {config.GAMMA}")

# ============================================================
# Phase 1: Generate small-scale data (25-35 agents)
# ============================================================
print("\n" + "="*60)
print("Phase 1: Small-scale data (25-35 agents)")
print("="*60)
generate_data(25, 35, 'train_')
generate_data(25, 35, 'val_')

train_ds = LocalizationDataset('datasets/train_dataset.pt')
val_ds = LocalizationDataset('datasets/val_dataset.pt')
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
        torch.save(model.state_dict(), 'models/model_anneal_small.pth')

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
generate_data(45, 55, 'val_')
generate_data(45, 55, 'test_')

ft_ds = LocalizationDataset('datasets/val_dataset.pt')
ft_loader = DataLoader(ft_ds, batch_size=16, shuffle=True, num_workers=2)

test_ds = LocalizationDataset('datasets/test_dataset.pt')

# ============================================================
# Phase 4: Fine-tune on large-scale data (30 epochs)
# ============================================================
print("\n" + "="*60)
print("Phase 4: Fine-tuning 30 epochs on large-scale data")
print("="*60)

model.load_state_dict(torch.load('models/model_anneal_small.pth', map_location=device))
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
        for batch in ft_loader:
            batch = batch.to(device)
            ew, logits = model(batch, tau=0.1, hard=True)
            loss, d = compute_gib_loss(logits, ew, batch,
                gamma=config.GAMMA, lambda_reg=ft_lam, eta=ETA,
                prior_weight=config.FIM_PRIOR, sparsity_weight=0)
            v_loss += loss.item(); v_edges += d['active_edges']
    v_loss /= len(ft_loader); v_edges /= len(ft_loader)
    if v_loss < best_val:
        best_val = v_loss
        torch.save(model.state_dict(), 'models/model_anneal_ft.pth')
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
model.load_state_dict(torch.load('models/model_anneal_ft.pth', map_location=device))

import subprocess
subprocess.run(["/root/miniconda3/bin/python", "-u", "evaluate.py"], check=True)
os.rename('evaluate_report.txt', 'evaluate_report_anneal.txt')
os.rename('evaluate_results.json', 'evaluate_results_anneal.json')

print("\n=== Lambda annealing experiment COMPLETE ===")
print(f"Schedule: {config.LAMBDA_SCHEDULE}")
print(f"ETA: {ETA}")
