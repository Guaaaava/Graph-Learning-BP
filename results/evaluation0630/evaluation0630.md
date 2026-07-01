# GIB Graph Topology Optimization - Final Report

**Date**: 2026-06-30 | **Hardware**: RTX 3090 24GB

---
## 1. Experiment Overview

| # | Scenario | Agents | FIM Norm | Small-Train/Large-Test |
|---|----------|--------|----------|----------------------|
| 1 | normal | 25-45 | No | No |
| 2 | hard | 25-45 | No | No |
| 3 | challenge | 25-45 | No | No |
| 4 | challenge | 25-35 train, 45-55 test | Yes | Yes |

---
## 2. Final: 50-Agent + FIM Norm

### 2.1 Sweep (L=2,5,10,20 x E=10,20,50, 30 epochs)

| LAMBDA | ETA | val_loss | edges |
|--------|-----|---------|-------|
| 2 | 10 | 4.31 | 56 | <-- best
| 2 | 20 | 8.57 | 59 |
| 2 | 50 | 21.96 | 56 |
| 5 | 10 | 9.11 | 32 |
| 5 | 20 | 19.57 | 37 |
| 5 | 50 | 50.64 | 36 |
| 10 | 10 | 15.28 | 22 |
| 10 | 20 | 28.40 | 22 |
| 10 | 50 | 76.78 | 23 |
| 20 | 10 | 20.04 | 15 |
| 20 | 20 | 31.26 | 11 |
| 20 | 50 | 92.70 | 11 |

L=5 shows 43% compression (32 vs 56 edges). Before FIM norm, needed L>100 for this effect.

### 2.2 50-Agent Results (P=10000, 500 graphs)

| Topo | Edges | RMSE (m) | CRLB (m) | Outage | Chi2 | NEES |
|------|-------|----------|----------|--------|------|------|
| Dense  | 740 | 1.27 +/- 0.36 | 0.679 | 2.1% | 82.6% | 4.1 |
| BFS    | 50 | 5.59 +/- 0.51 | 1.838 | 70.4% | 93.1% | 2.3 |
| MST    | 54 | 4.22 +/- 0.46 | 1.533 | 56.2% | 92.9% | 2.4 |
| k-NN   | 188 | 2.79 +/- 0.57 | 1.192 | 26.8% | 87.0% | 3.4 |
| Random | 395 | 1.72 +/- 0.41 | 0.897 | 6.7% | 87.5% | 3.3 |
| GNN    | 596 | 1.51 +/- 0.44 | 0.692 | 4.5% | 75.5% | 5.2 |

**GNN vs Dense**: RMSE ratio=1.184, Edge reduction=19.5%

---
## 3. Cross-Experiment Comparison

### 3.1 GNN Performance Summary

| Experiment | Dense Edges | GNN Edges | Reduction | Dense RMSE | GNN RMSE | RMSE Ratio |
|------------|------------|-----------|-----------|-----------|---------|-----------|
| 30ag normal               | 166 | 161 | 2.9% | 2.27 | 2.30 | 1.017 |
| 30ag hard                 | 148 | 143 | 3.5% | 2.49 | 2.54 | 1.022 |
| 30ag challenge            | 379 | 302 | 20.5% | 1.57 | 1.84 | 1.172 |
| 50ag challenge+FIM        | 740 | 596 | 19.5% | 1.27 | 1.51 | 1.184 |

### 3.2 Key Findings

1. **FIM normalization works**: Lambda=2-5 effective across scales (was 20-100)
2. **Small-train/large-test validated**: 30-agent training generalizes to 50-agent with 30-epoch fine-tune
3. **Compression stabilizes at ~20%**: Both 30ag (20.3%) and 50ag (19.5%) in challenge
4. **Redundancy required for compression**: Normal/hard (<5%) vs challenge (~20%)
5. **GNN consistency drops after pruning**: Chi2 75.5% vs Dense 82.6%. Covariance estimation needs work.

---
## 4. Complete Data

### Normal (30-agent, no FIM norm)

| Topo | Edges | RMSE (m) | CRLB (m) | Outage | Chi2 | NEES |
|------|-------|----------|----------|--------|------|------|
| Dense  | 166 | 2.27 +/- 0.60 | 1.081 | 15.1% | 86.3% | 3.5 |
| BFS    | 35 | 5.27 +/- 0.58 | 1.728 | 64.7% | 93.5% | 2.3 |
| MST    | 38 | 4.50 +/- 0.57 | 1.565 | 57.7% | 93.0% | 2.4 |
| k-NN   | 109 | 2.83 +/- 0.58 | 1.215 | 25.3% | 87.0% | 3.4 |
| Random | 100 | 3.29 +/- 0.69 | 1.327 | 31.2% | 88.8% | 3.2 |
| GNN    | 161 | 2.30 +/- 0.59 | 1.086 | 15.6% | 86.1% | 3.6 |

### Hard (30-agent, no FIM norm)

| Topo | Edges | RMSE (m) | CRLB (m) | Outage | Chi2 | NEES |
|------|-------|----------|----------|--------|------|------|
| Dense  | 148 | 2.49 +/- 0.65 | 1.126 | 20.7% | 82.6% | 4.1 |
| BFS    | 35 | 5.26 +/- 0.62 | 1.725 | 65.5% | 93.2% | 2.3 |
| MST    | 38 | 4.49 +/- 0.58 | 1.572 | 58.9% | 93.1% | 2.3 |
| k-NN   | 92 | 3.12 +/- 0.62 | 1.276 | 33.7% | 86.6% | 3.5 |
| Random | 82 | 3.62 +/- 0.76 | 1.397 | 40.2% | 88.8% | 3.1 |
| GNN    | 143 | 2.54 +/- 0.65 | 1.133 | 21.8% | 82.2% | 4.1 |

### Challenge (30-agent, no FIM norm)

| Topo | Edges | RMSE (m) | CRLB (m) | Outage | Chi2 | NEES |
|------|-------|----------|----------|--------|------|------|
| Dense  | 379 | 1.57 +/- 0.44 | 0.811 | 4.7% | 81.6% | 4.2 |
| BFS    | 35 | 5.58 +/- 0.57 | 1.837 | 70.0% | 93.6% | 2.3 |
| MST    | 38 | 4.31 +/- 0.50 | 1.547 | 57.7% | 92.8% | 2.4 |
| k-NN   | 128 | 2.83 +/- 0.61 | 1.216 | 28.2% | 86.1% | 3.5 |
| Random | 207 | 2.16 +/- 0.57 | 1.056 | 13.5% | 86.6% | 3.5 |
| GNN    | 302 | 1.84 +/- 0.54 | 0.827 | 8.6% | 75.1% | 5.2 |

### Challenge (50-agent, FIM norm + small-train/large-test)

| Topo | Edges | RMSE (m) | CRLB (m) | Outage | Chi2 | NEES |
|------|-------|----------|----------|--------|------|------|
| Dense  | 740 | 1.27 +/- 0.36 | 0.679 | 2.1% | 82.6% | 4.1 |
| BFS    | 50 | 5.59 +/- 0.51 | 1.838 | 70.4% | 93.1% | 2.3 |
| MST    | 54 | 4.22 +/- 0.46 | 1.533 | 56.2% | 92.9% | 2.4 |
| k-NN   | 188 | 2.79 +/- 0.57 | 1.192 | 26.8% | 87.0% | 3.4 |
| Random | 395 | 1.72 +/- 0.41 | 0.897 | 6.7% | 87.5% | 3.3 |
| GNN    | 596 | 1.51 +/- 0.44 | 0.692 | 4.5% | 75.5% | 5.2 |

---
## 5. Conclusions & Next Steps

### 5.1 Conclusions
1. **GIB approach validated**: GNN distinguishes high-quality edges from noisy/redundant ones
2. **FIM normalization is essential**: Makes lambda scale-independent, enables small-train/large-test
3. **Compression requires redundancy**: Normal/hard <5%, challenge ~20%. NLOS edges are the key target
4. **Small-train/large-test saves significant compute**: Train on 30ag (20min) vs 50ag (1h+)

### 5.2 Next Steps
- Larger lambda sweep (5-20) with full 200-epoch training
- Lambda annealing: small lambda early, large lambda late
- 100-agent scale test
- Multi-scenario joint training for universal pruning policy
- NLOS-supervised auxiliary loss

---
## Appendix: LaTeX Tables

```latex
% Normal (30-agent)
  Dense & 166 & 2.27 $\pm$ 0.60 & 1.081 & 15.1 & 3.5 \\
    BFS & 35 & 5.27 $\pm$ 0.58 & 1.728 & 64.7 & 2.3 \\
    MST & 38 & 4.50 $\pm$ 0.57 & 1.565 & 57.7 & 2.4 \\
   k-NN & 109 & 2.83 $\pm$ 0.58 & 1.215 & 25.3 & 3.4 \\
 Random & 100 & 3.29 $\pm$ 0.69 & 1.327 & 31.2 & 3.2 \\
    GNN & 161 & 2.30 $\pm$ 0.59 & 1.086 & 15.6 & 3.6 \\

% Hard (30-agent)
  Dense & 148 & 2.49 $\pm$ 0.65 & 1.126 & 20.7 & 4.1 \\
    BFS & 35 & 5.26 $\pm$ 0.62 & 1.725 & 65.5 & 2.3 \\
    MST & 38 & 4.49 $\pm$ 0.58 & 1.572 & 58.9 & 2.3 \\
   k-NN & 92 & 3.12 $\pm$ 0.62 & 1.276 & 33.7 & 3.5 \\
 Random & 82 & 3.62 $\pm$ 0.76 & 1.397 & 40.2 & 3.1 \\
    GNN & 143 & 2.54 $\pm$ 0.65 & 1.133 & 21.8 & 4.1 \\

% Challenge (30-agent)
  Dense & 379 & 1.57 $\pm$ 0.44 & 0.811 & 4.7 & 4.2 \\
    BFS & 35 & 5.58 $\pm$ 0.57 & 1.837 & 70.0 & 2.3 \\
    MST & 38 & 4.31 $\pm$ 0.50 & 1.547 & 57.7 & 2.4 \\
   k-NN & 128 & 2.83 $\pm$ 0.61 & 1.216 & 28.2 & 3.5 \\
 Random & 207 & 2.16 $\pm$ 0.57 & 1.056 & 13.5 & 3.5 \\
    GNN & 302 & 1.84 $\pm$ 0.54 & 0.827 & 8.6 & 5.2 \\

% Challenge + FIM norm (50-agent)
  Dense & 740 & 1.27 $\pm$ 0.36 & 0.679 & 2.1 & 4.1 \\
    BFS & 50 & 5.59 $\pm$ 0.51 & 1.838 & 70.4 & 2.3 \\
    MST & 54 & 4.22 $\pm$ 0.46 & 1.533 & 56.2 & 2.4 \\
   k-NN & 188 & 2.79 $\pm$ 0.57 & 1.192 & 26.8 & 3.4 \\
 Random & 395 & 1.72 $\pm$ 0.41 & 0.897 & 6.7 & 3.3 \\
    GNN & 596 & 1.51 $\pm$ 0.44 & 0.692 & 4.5 & 5.2 \\

```