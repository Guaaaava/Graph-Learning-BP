#!/usr/bin/env python
"""Minimal 100-graph evaluation for sp_w=80 model"""
import sys, os, torch, numpy as np, scipy.stats as stats
os.chdir('/home/huangcy/Graph-Learning-BP/GNN_learning')
sys.path.insert(0, '/home/huangcy/Graph-Learning-BP')

from dataset import LocalizationDataset
from torch_geometric.loader import DataLoader
from model import EdgePredictorGNN
from BP.particle_bp import ParticleBP
from loss import compute_crlb
from evaluate import extract_bfs_weights, calculate_metrics
import config

device = torch.device('cuda')
print(f'Device: {device}')

model = EdgePredictorGNN(
    node_in_dim=5, edge_in_dim=4, hidden_dim=64, num_layers=3,
).to(device)
model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
model.eval()

ds = LocalizationDataset('datasets/test_dataset.pt')
loader = DataLoader(ds, batch_size=1, shuffle=False)

bp = ParticleBP(num_particles=500, sigma_meas=0.5, num_iter=3, init_cov=25.0)

results = {'dense': [], 'bfs': [], 'gnn': []}
count = 0

for data in loader:
    if count >= 100:
        break
    data = data.to(device)
    E = data.edge_index.shape[1]

    dense_w = torch.ones(E, device=device)
    bfs_w = extract_bfs_weights(data)
    gnn_w, _ = model(data, tau=0.1, hard=True)

    for name, w in [('dense', dense_w), ('bfs', bfs_w), ('gnn', gnn_w)]:
        pos, cov, _ = bp.run(data, edge_weights=w)
        rmse, outage, consist, nees, _ = calculate_metrics(pos, data.y, cov)
        crlb, _ = compute_crlb(data, w, prior_weight=config.FIM_PRIOR)
        results[name].append({
            'rmse': rmse, 'crlb': crlb, 'edges': (w > 0.5).sum().item(),
            'outage': outage, 'consist': consist, 'nees': nees,
        })

    count += 1
    if count % 20 == 0:
        print(f'{count} graphs...', flush=True)

print(f'\nDone {count} graphs\n', flush=True)

for name in ['dense', 'bfs', 'gnn']:
    recs = results[name]
    rmse = np.mean([r['rmse'] for r in recs])
    rmse_std = np.std([r['rmse'] for r in recs], ddof=1)
    edges = np.mean([r['edges'] for r in recs])
    crlb = np.mean([r['crlb'] for r in recs])
    outage = np.mean([r['outage'] for r in recs]) * 100
    nees = np.mean([r['nees'] for r in recs])
    consist = np.mean([r['consist'] for r in recs]) * 100
    print(f'{name:8s}: Edges={edges:5.0f}, RMSE={rmse:.3f}±{rmse_std:.3f}m, CRLB={crlb:.4f}m, Outage={outage:.1f}%, NEES={nees:.2f}, Consist={consist:.0f}%')

# Save
import json
with open('eval_results_100.json', 'w') as f:
    json.dump({name: {'rmse_mean': float(np.mean([r['rmse'] for r in results[name]])),
                      'rmse_std': float(np.std([r['rmse'] for r in results[name]], ddof=1)),
                      'edges': float(np.mean([r['edges'] for r in results[name]])),
                      'crlb': float(np.mean([r['crlb'] for r in results[name]])),
                      'outage': float(np.mean([r['outage'] for r in results[name]])),
                      'nees': float(np.mean([r['nees'] for r in results[name]])),
                      'consist': float(np.mean([r['consist'] for r in results[name]]))}
               for name in ['dense', 'bfs', 'gnn']}, f, indent=2)
print('Saved eval_results_100.json')
