"""
实验: FIM归一化 + 50 agent + 小规模训练/大规模测试
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


def generate_data(agents_min, agents_max, suffix):
    """生成指定 agent 范围的数据集"""
    import random, numpy as np, generate_network
    from tqdm import tqdm

    old_min, old_max = config.NUM_AGENTS_MIN, config.NUM_AGENTS_MAX
    config.NUM_AGENTS_MIN, config.NUM_AGENTS_MAX = agents_min, agents_max

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
    config.NUM_AGENTS_MIN, config.NUM_AGENTS_MAX = old_min, old_max
    return ds


def sweep_params(train_loader, val_loader, name="sweep"):
    """网格搜索最优 hyperparams"""
    device = 'cuda'
    grid = [(l, e) for l in [2, 5, 10, 20] for e in [10, 20, 50]]
    results = []
    for idx, (LAMBDA, ETA) in enumerate(grid):
        model = EdgePredictorGNN(5, 4, 64, 3).to(device)
        opt = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        best_val = float('inf'); best_epoch = 0
        for epoch in range(1, 31):
            tau = max(0.1, 5.0 * (0.98**epoch))
            model.train()
            for batch in train_loader:
                batch = batch.to(device); opt.zero_grad()
                ew, logits = model(batch, tau=tau, hard=True)
                loss, d = compute_gib_loss(logits, ew, batch,
                    gamma=config.GAMMA, lambda_reg=LAMBDA, eta=ETA,
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
                        gamma=config.GAMMA, lambda_reg=LAMBDA, eta=ETA,
                        prior_weight=config.FIM_PRIOR, sparsity_weight=0)
                    v_loss += loss.item(); v_edges += d['active_edges']
            v_loss /= len(val_loader); v_edges /= len(val_loader)
            if v_loss < best_val: best_val = v_loss; best_epoch = epoch
        results.append({'lambda': LAMBDA, 'eta': ETA, 'val_loss': round(best_val,2), 'best_epoch': best_epoch, 'edges': round(v_edges,0)})
        print(f"[{idx+1}/12] L={LAMBDA} E={ETA} val={best_val:.2f} @ep{best_epoch} edges={v_edges:.0f}", flush=True)
    best = min(results, key=lambda x: x['val_loss'])
    print(f"Best: L={best['lambda']} E={best['eta']} val={best['val_loss']}")
    with open(f'{name}.json', 'w') as f: json.dump({'results': results, 'best': best}, f)
    return best['lambda'], best['eta']


def train_model(train_loader, val_loader, LAMBDA, ETA, epochs=200, save="best_model"):
    """完整训练"""
    device = 'cuda'
    model = EdgePredictorGNN(5, 4, 64, 3).to(device)
    opt = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    best_val = float('inf')
    for epoch in range(1, epochs+1):
        tau = max(0.1, 5.0 * (0.98**epoch))
        model.train()
        for batch in train_loader:
            batch = batch.to(device); opt.zero_grad()
            ew, logits = model(batch, tau=tau, hard=True)
            loss, _ = compute_gib_loss(logits, ew, batch,
                gamma=config.GAMMA, lambda_reg=LAMBDA, eta=ETA,
                prior_weight=config.FIM_PRIOR, sparsity_weight=0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval(); v_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                ew, logits = model(batch, tau=0.1, hard=True)
                loss, _ = compute_gib_loss(logits, ew, batch,
                    gamma=config.GAMMA, lambda_reg=LAMBDA, eta=ETA,
                    prior_weight=config.FIM_PRIOR, sparsity_weight=0)
                v_loss += loss.item()
        v_loss /= len(val_loader); scheduler.step()
        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), f'models/{save}.pth')
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} val={v_loss:.2f} best={best_val:.2f}", flush=True)
    print(f"Training done, best val={best_val:.2f}")
    return model


def main():
    device = 'cuda'
    print(f"Device: {device}, Scenario: {config.SCENARIO_TYPE}")
    print(f"FIM normalization: ON (per degree-of-freedom)")

    # ============================================================
    # Phase 1: 生成小规模数据 (25-35 agents)
    # ============================================================
    print("\n" + "="*60)
    print("Phase 1: 生成小规模数据 (25-35 agents)")
    print("="*60)
    generate_data(25, 35, 'train_')   # 2000 graphs for training
    generate_data(25, 35, 'val_')     # 500 graphs for validation

    train_ds = LocalizationDataset('datasets/train_dataset.pt')
    val_ds = LocalizationDataset('datasets/val_dataset.pt')
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    # ============================================================
    # Phase 2: 网格搜索 (小规模)
    # ============================================================
    print("\n" + "="*60)
    print("Phase 2: 网格搜索 (小规模数据)")
    print("="*60)
    best_L, best_E = sweep_params(train_loader, val_loader, "sweep_small")

    # ============================================================
    # Phase 3: 小规模训练
    # ============================================================
    print("\n" + "="*60)
    print(f"Phase 3: 小规模训练 200 epochs (L={best_L}, E={best_E})")
    print("="*60)
    model = train_model(train_loader, val_loader, best_L, best_E, 200, "model_small")

    # ============================================================
    # Phase 4: 生成大规模数据 (45-55 agents)
    # ============================================================
    print("\n" + "="*60)
    print("Phase 4: 生成大规模数据 (45-55 agents)")
    print("="*60)
    generate_data(45, 55, 'val_')     # 500 graphs for fine-tuning validation
    generate_data(45, 55, 'test_')    # 500 graphs for final test

    ft_train_ds = LocalizationDataset('datasets/val_dataset.pt')  # reuse val as ft-train
    ft_val_ds = LocalizationDataset('datasets/test_dataset.pt')   # test as ft-val (we'll eval on same)

    # Actually: generate separate fine-tune data
    # For simplicity, use val_ as ft_train and test_ as test
    generate_data(45, 55, 'val_')
    generate_data(45, 55, 'test_')

    ft_train_ds = LocalizationDataset('datasets/val_dataset.pt')
    ft_loader = DataLoader(ft_train_ds, batch_size=16, shuffle=True, num_workers=2)

    test_ds = LocalizationDataset('datasets/test_dataset.pt')

    # ============================================================
    # Phase 5: 规模微调
    # ============================================================
    print("\n" + "="*60)
    print(f"Phase 5: 大规模微调 30 epochs (L={best_L}, E={best_E})")
    print("="*60)
    # 加载小规模训练权重
    model.load_state_dict(torch.load('models/model_small.pth', map_location=device))
    model = train_model(ft_loader, ft_loader, best_L, best_E, 30, "model_ft")
    model.load_state_dict(torch.load('models/model_ft.pth', map_location=device))

    # ============================================================
    # Phase 6: 大规模测试评估
    # ============================================================
    print("\n" + "="*60)
    print("Phase 6: 大规模测试评估 (500 graphs, P=10000)")
    print("="*60)

    import subprocess
    subprocess.run(["/root/miniconda3/bin/python", "-u", "evaluate.py"], check=True)
    os.rename('evaluate_report.txt', 'evaluate_report_50agent.txt')
    os.rename('evaluate_results.json', 'evaluate_results_50agent.json')

    print("\n=== 50-agent experiment COMPLETE ===")


if __name__ == "__main__":
    main()
