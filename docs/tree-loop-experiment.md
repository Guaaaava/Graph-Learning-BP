# 第二章树／单环粒子 BP 实验：运行说明

实验设计见 [已确认的计划](tree-loop-overconfidence-plan.md)。实现入口为 `gib4cl/experiments/tree_loop/`，复用 `gib4cl/inference/particle_bp.py`；不训练 GNN，不调用生产场景生成器，不修改训练数据或正式 GNN 配置。

概率、可信区间、置信区间与 NEES 历史检验的解释见[读图说明](tree-loop-reader-guide.md)。七节点和 3／4／5 环长的配置、运行与限制见[拓展说明](tree-loop-extensions.md)。当前图表与报告主表暂不使用 NEES；机制图展开前 6 轮并放大第 3-6 轮，完整协方差和消息残差另存。七节点主校准图与机制图均针对中央节点 i，另有全节点平均补充图。阴影保持逐点 95% bootstrap 置信区间，横纵轴均明确 x 坐标。图内省略标题和长说明，含义移至独立图注。重画时的绘图实现为 `figures.py`。

## 环境

在 `Graph-Learning-BP` 项目目录执行。已有环境可使用 `.\.venv-data\Scripts\python.exe`，或在激活环境后使用 `python`。依赖沿用项目 requirements，并补充 Matplotlib；不需要 SciPy、LaTeX 或图形桌面。

## 推荐顺序

先跑小规模完整流程（4＋4 个独立试验，每次两个粒子重复）：

```powershell
python -B -m gib4cl.experiments.tree_loop run --config configs/diagnostics/tree-loop-smoke.json --output-dir outputs/tree-loop-smoke
```

做正式预算的数值预实验（每种几何 20 组数据，P=128/256/512，3 个支持重复）：

```powershell
python -B -m gib4cl.experiments.tree_loop pilot --config configs/diagnostics/tree-loop.json --output-dir outputs/tree-loop-pilot
```

检查 `summary.json` 的 `numerical_checks`、`statistics.csv` 及 `figures/particle-stability.png`，据此在正式试验前冻结粒子预算。配置中的 P=256 是计划起点，并非自动认定足够；连续参照未稳定时会保留未通过状态。

固定预算试验（走廊 200 次＋等边 100 次，每次 3 个粒子重复）：

```powershell
python -B -m gib4cl.experiments.tree_loop run --config configs/diagnostics/tree-loop.json --output-dir outputs/tree-loop-study
```

试跑与正式试验使用不同的数据／粒子种子命名空间。每个独立试验编号与重复编号共同派生支持种子。同一配对中的树与环共享粒子与公共测量；跨几何条件也共享底层标准正态扰动。所有试验均为二维非线性测距、高斯先验预测生成、独立的常方差测距噪声，不裁剪观测。

## 断点续跑与单独重画

结果目录默认必须不存在，避免覆盖已有结果。可以分段运行：

```powershell
python -B -m gib4cl.experiments.tree_loop run --config configs/diagnostics/tree-loop.json --output-dir outputs/tree-loop-study --stop-after-cases 10 --no-plots
python -B -m gib4cl.experiments.tree_loop run --config configs/diagnostics/tree-loop.json --output-dir outputs/tree-loop-study --resume
```

`--stop-after-cases` 统计本次新增的独立数据试验数，包含两个几何条件；不会把粒子重复算作独立试验。续跑前验证配置、阶段、相关源码、运行版本和已有原始文件校验和。改变推理／实验源码或配置后应选择新目录；已完成试验不会重新推理。

从已保存结果重新生成图片与表格：

```powershell
python -B -m gib4cl.experiments.tree_loop plot --output-dir outputs/tree-loop-study
```

重画会验证原始记录，不运行 BP。可以调整绘图代码后重画，新的绘图源码摘要记录在 `figures/index.json`。缺失／被改动的数据文件会明确报错。`--no-plots` 仅跳过图片，仍写出统计和报告。

原始推理源码摘要保存在 `manifest.json`；重新汇总／绘图时所用源码另记于 `summary.json` 的 `analysis_sources`。这样可以区分原始数据与后续统计表达的来源。

## 输出

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 完整配置、阶段、种子设置、源码摘要、运行环境、已完成试验及校验和 |
| `status.json` | complete / partial / failed 和已完成试验数 |
| `cases/<condition>/<id>.json` | 真值、测量、每次粒子重复的逐节点指标、逐轮方差比／残差、连续参照检查 |
| `raw/<condition>/<id>-p<P>-r<R>.npz` | 共享支持、先验质量、两图 BP 与精确边缘权重／矩、逐轮均值和协方差 |
| `raw/<condition>/<id>-quadrature.npz` | 最后一级独立求积支持和各图后验边缘／矩 |
| `summary.json` | 按条件和 P 分组的统计、成对差值区间、数值检查 |
| `statistics.csv` | 汇总指标、95% 区间、实际有效独立试验数 |
| `runs.csv` | 每个独立试验、P、粒子重复、图的标量指标 |
| `report.md` | 可阅读的结果和口径说明 |
| `figures/<condition>-mechanism.*` | 左树右环；节点 2 的椭圆与两个方向极值方差比的迭代曲线 |
| `figures/<condition>-calibration.*` | 七节点主图看节点 i；左为完整范围，右放大高覆盖率；阴影为逐点 95% bootstrap 置信区间 |
| `figures/<condition>-calibration-all-agents.*` | 七节点平均校准的补充口径 |
| `focus-calibration.json` | 节点 i 在加密概率网格上的 BP／同图精确后验覆盖统计；仅分析已有粒子权重 |
| `figures/<condition>-covariance-iterations-full.*` | 与机制图一致的焦点节点、完整迭代范围的方向方差比 |
| `figures/<condition>-message-residual.*` | 完整轮数消息残差，与协方差平台分开解释 |
| `figures/<condition>-cycle-length-comparison.*` | 中央节点／全节点平均 x 方差比叠加到同一坐标轴，并附闭环拓扑 |
| `figures/particle-stability.*` | pilot 阶段的粒子预算复核 |

每张图同时输出 PNG、PDF 和 SVG。图中上排示例固定为数据试验 0、支持重复 0；下排曲线使用全部已完成独立试验。默认图示主粒子数为配置的 P；若它不在 pilot 网格中，采用网格中的最大 P。smoke / pilot 图会明确标记阶段，不可直接当作正式统计结论。

## 精确参照与数值检查

`reference.py` 独立用 NumPy 构建三个测距核。相同有限支持联合后验通过两次矩阵乘法缩并，时间 O(P³)、存储 O(P²)，不调用生产 BP 的 cavity、核计算或矩函数。已用现有独立 P³ 状态穷举交叉核对，也覆盖非均匀先验质量和零质量支持。

每张图使用各自的观测集合求精确边缘。树 BP 与精确边缘的最大质量误差超过 `tree_tolerance` 时立即停止并保存失败状态。环 BP 不收敛的运行继续保留，其收敛比例与残差一并报告。没有为协方差添加 jitter，也没有给测量似然加底噪。

连续数值参照采用二维节点 Gauss–Hermite 支持进行独立联合积分。连续两次求积加密的均值变化和全部方向的相对协方差变化都不超过阈值，才标记稳定。检查结果是一种数值证据，不是严格积分误差界。默认只在每条件最前 20 个独立试验上做连续复核；`continuous_cases=0` 可用于纯流程验证。

参考粒子数由 `max_reference_particles` 显式限制；求积阶数 q 对应每节点 P=q²。超过预算、总质量下溢、参照协方差无效都会明确报错；不自动改成近似后验。

## 指标与统计口径

- `variance_ratio_x`：每节点 BP x 方差／该图同一支持精确 x 方差，再对节点平均。
- `lambda_min/max`：相对精确协方差的广义方差比两个特征值，对节点平均；逐节点值完整保留。
- `nees_normalized`：有效协方差节点的平均 NEES/2；无效协方差率单列，不用修正矩阵掩盖失败。
- `covariance_nees_normalized` 与 `mean_bias_nees_normalized`：相对同图参照的条件预期 NEES/2 的协方差项、均值偏差项。
- `pooled_rmse`：先平均各独立试验及粒子重复的 MSE，再开平方；不是各图 RMSE 的简单平均。
- 覆盖曲线：x 方向加权经验等尾区间，逆 CDF 使用左分位数；树与环分别与其有限支持精确参照比较。
- 图中的椭圆为矩匹配高斯等效 95% 椭圆，不声称非高斯后验具有 95% 椭圆覆盖率。
- KL 为同一支持上的 KL(p* || bBP)。零 BP 质量导致的无穷 KL 不写成有限数值，另报其比例。

每组数据内先对粒子重复平均，然后按独立数据试验做成对 bootstrap，默认 2000 次。`paired_loop_minus_tree` 是相同试验上的差值区间。没有把节点或粒子种子当成新增独立观测。每个指标保留有效试验数。

`converged_loop_subset` 补充报告满足消息容限的环图运行，先在独立试验内平均再计算区间；主表仍保留全部运行。只有一个有效独立试验时，不给出零宽的置信区间。

有限支持参照曲线仍包含对连续分布的离散误差，所以不要求它严格位于理想校准对角线上。连续参照达到稳定时，额外记录 BP／有限精确协方差相对它的误差。粒子数改变后效应是否稳定、连续参照是否稳定、BP 是否收敛，是三个不同的问题。

## 可选扩展

三节点模式中，在配置 `topologies` 中增加 `tree_01_02`、`tree_12_02` 即可比较另外两棵生成树。增加新 condition 并设置不同 `range_std` 可做噪声强度复核。七节点模式设置 `graph_family=branched7`，采用 `tree/loop3/loop4/loop5` 及 `primary_loop=loop5`；详见拓展说明。所有 condition 名称必须是安全的小写路径标识符。一般参照支持树／每连通分量至多一个环，拒绝将多环图冒充精确单环模型。

## 验证

```powershell
python -B -m pytest tests/test_tree_loop.py tests/test_particle_bp.py tests/test_metrics.py -q
```

新增测试覆盖：精确缩并对穷举、三棵树的 BP 一致性、可解析一维高斯极限、共享支持与标签隔离、方向比、求积分布矩、加权区间、试验级成对统计、续跑及原始记录完整性、无推理的重新出图。测试使用的可解析辅助模型不替代正式二维实验。
