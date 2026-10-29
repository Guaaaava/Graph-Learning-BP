# 第三步③：按最新版论文重建训练目标

已实现第 3 章的高斯预测分布和四项压缩目标，以及第 4 章的度数项和总目标。入口为 [objectives.py](../GNN_learning/objectives.py)，主训练通过 [loss.py](../GNN_learning/loss.py) 导出的 `compute_gib_loss` 调用新版实现。

本阶段历史验证记录见 [stage3-verification.json](stage3-verification.json)。随后第四步已完成 [GNN 与采样](gnn-sampling.md)，第五步已完成[粒子 BP 和独立参照](particle-bp.md)。这里的高斯预测协方差不是粒子 BP 协方差，也不宣称等于真实后验协方差。

## 1. 公式与函数

| 论文位置 | 实现 | 当前行为 |
| --- | --- | --- |
| §3.2 独立节点先验 | `gaussian_predictions` | 从完整 `agent_prior_cov` 的 Cholesky 分解构造先验精度块；不读取旧 `FIM_PRIOR` 或精度缓存 |
| `eq:local_prediction_product`、`eq:prediction_precision` | `gaussian_predictions` | 初值处的全局距离 Jacobian、精度 `J_s`、信息向量 `b_s` |
| `eq:prediction_update` | `gaussian_predictions` | Cholesky 线性求解 `J_s delta=b_s`，保留通过求解的梯度 |
| `eq:prediction_distribution` | `GaussianPrediction` | `mean=Ybar+delta`，协方差按需由 Cholesky 因子计算 |
| `eq:linear_task_loss` | `gaussian_task_terms` | 行列式项与完整联合误差二次项，保留 `1/2` |
| `eq:entropy_of_p_theta`、`eq:expected_energy`、`eq:expected_triangle`、`eq:compression_loss` | `compression_terms` | 直接由概率计算四项解析期望 |
| `eq:degree_loss` | `degree_terms` | 直接累加 STE 权重并对智能体求和，锚点边计入度数 |
| `eq:gib_train_objective`、`eq:total_loss` | `gib_objective`、`compute_gib_loss` | 每图内求和，对图等权平均，返回分项日志 |

互信息恒等式、变分界及最大熵推导作为目标的理论依据，不通过本代码直接估计互信息或求解大图配分函数。

## 2. 输入限定与高斯预测分布

输入采用第二步的唯一物理测量边格式。每条边对应一个 logit 和一个浮点 STE 权重；权重的前向值必须严格为 0 或 1，双向消息边不能重复传入本目标。

预测器只读初始坐标、锚点标志、边端点、测距、已知方差及节点先验，不读取 `data.y`。在初值处定义：

```text
dbar_e = ||rbar_u-rbar_v||
u_e = (rbar_u-rbar_v)/dbar_e
r_e = z_e-dbar_e
vtilde_e = max(v_e, prediction_variance_floor)
```

构造全局 Jacobian 行 `g_e^T` 时，智能体 u 的坐标块为 `u_e`，智能体 v 的块为 `-u_e`；已知锚点没有未知坐标块。因而智能体—智能体边产生负的交叉信息块，锚点边只贡献智能体的块。

```text
J0 = block_diag(Sigma0_i^-1)
J_s = J0 + sum_e s_e*g_e*g_e^T/vtilde_e
b_s = sum_e s_e*g_e*r_e/vtilde_e
J_s*delta = b_s
mu = Ybar+delta
S = J_s^-1
```

`GaussianPrediction` 暴露 `gradient`、`prior_precision`、`precision`、`information_vector`、`delta`、`mean`、`logdet_precision` 及按需计算的 `covariance`，用于独立验证和诊断。

默认内部计算为 float64；存储的测量仍是第二步定义的 float32，GNN 参数也可保持 float32。类型转换保留反向梯度，不改变测量的编码支持集。可显式设置 `compute_dtype='float32'`，不能依靠全局默认 dtype 暗中改变目标精度。

计算均值使用线性求解，任务损失无需构造完整联合协方差；仅小型节点先验块需要求逆，诊断请求协方差时才计算联合逆。没有任意对角先验、特征值裁剪、伪逆或失败后替换目标的回退。非正定先验、重合初值、非有限值或 Cholesky 失败会报错。

无保留边时预测均值、协方差严格回到指定的高斯先验。构造矩阵时仍包含所有候选行并乘 `s_e`，不会用 `s_e>0` 筛掉零权重边：零权重决定前向删边，但它对应的 STE 导数必须保留。

## 3. 完整任务项

仅在预测分布构造完之后，任务函数才读取位置标签：

```text
e = Y-mu
task = 0.5*(-logdet(J_s)+e^T*J_s*e)
```

这里的二次项使用完整联合精度，不能改成逐智能体独立误差求和后忽略交叉块，也不能把 `mu` 从计算图中分离。位置标签变化只改变误差项，不改变预测器几何或预测均值。

训练用 `task` 与论文的 `doteq` 表述一致，省略 `A*log(2*pi)`。日志同时提供包含该常数的 `task_nll`，便于与规范化高斯负对数似然对照。不同规模数据集之间比较数值时要区分这两个量。

## 4. 四项压缩损失

令 `p=sigmoid(logits)`：

```text
compression = negative_entropy
              + gamma*expected_energy
              + kappa*expected_triangles
              + retained_data_cost

negative_entropy = sum_e [p_e*log(p_e)+(1-p_e)*log(1-p_e)]
expected_energy = sum_e p_e*E_e(C)
expected_triangles = sum_{agent triangles} p_ij*p_jk*p_ik
retained_data_cost = sum_e p_e*log(M_e(C))
```

成本和编码基数来自第二步的输入派生字段，不使用真值距离。锚点边参与成本和测量保留项，但不进入智能体三角形项。三角形必须由三条不同物理边组成、只含三个智能体且每个只计算一次；无三角形时此项为零。

负熵以 `logsigmoid` 与两侧稳定 sigmoid 计算，在极端 logits 下不对概率任意截断、不出现 `log(0)`；同时避免 `1-p` 在接近 1 时相消。

不计算训练无关的 `log Z(C)`；移去此常数后的 `compression` 不要求非负，也不把它的数值叫作完整 KL。三边小图的 8 种掩码穷举验证了：解析式加上精确 `log Z` 与完整参考分布 KL 相等，其关于 logits 的梯度也相等。

## 5. 度数、总损失及归一化

```text
D_i = sum_{physical edges incident to agent i} s_e
degree = sum_{agents i} relu(3-D_i)
total = task + beta*compression + eta*degree
loss = mean_graphs(total)
```

度数直接累加 STE 权重，不能布尔化后累加常数。智能体边给两个智能体各贡献一次，锚点边给其智能体贡献一次。惩罚少于三条边不等于保证连通、刚性或定位精度。

不分别按边数、智能体数或自由度归一化各项。变规模批处理与逐图计算后平均一致；主训练入口在汇总各批次日志时还按实际图数加权，避免最后一个小批次被过度加权。

`gib_objective` 返回可微的 `ObjectiveResult`：标量 `loss`、每图分项张量 `per_graph` 和预测器对象列表。`compute_gib_loss` 返回标量及已脱离计算图的每图平均日志。

日志包括 `task/task_logdet/task_mahalanobis/task_nll`、`compression`、负熵、期望成本、期望三角形、测量保留成本、加权惩罚、度数、保留边数及总值，不再把新版任务称为 FIM 项或把去常数的压缩项称为完整 KL。

## 6. 配置、主入口与历史实验

新版调用示例：

```python
from GNN_learning.experiment_config import load_config
from GNN_learning.loss import compute_gib_loss

cfg = load_config('configs/smoke-objective.json')
loss, metrics = compute_gib_loss(logits, ste_weights, batch, objective=cfg.objective)
loss.backward()
```

`ObjectiveConfig` 要求显式指定 `beta>0`、`gamma>=0`、`kappa>=0`、`eta>=0`。配置未填时调用 `require_ready()` 会报错，不从旧 lambda、旧 KL 尺度或 `FIM_PRIOR` 推断系数。

正式场景配置继续保留未确定的权重；[smoke-objective.json](../configs/smoke-objective.json) 专用于实现验证，采用 `beta=0.02,gamma=0.002,kappa=0.2,eta=1.0`，并使用 1 轮/批大小 2 的小数据设定。这些值不是调参结论，不作为论文正式实验参数。其数据配置与 `smoke-data.json` 相同，因此可读取第二步生成的小数据。

主 [train.py](../GNN_learning/train.py) 已接入新版接口：

- `--config` 读取完整配置；`--beta/--gamma/--kappa/--eta` 可显式覆盖。
- 去掉旧 lambda 退火及额外稀疏项；同一次训练/验证使用固定目标权重。
- `--data-root`、`--output-root` 分别指定数据和模型根目录。
- 模型位于 `outputs/<experiment_id>/<完整配置指纹前16位>/models/`，保存配置和分项历史，默认拒绝覆盖已有运行输出。
- 不再接受 `lambda_reg/prior_weight/sparsity_weight` 作为新版目标参数，也不提供静默兼容换算。

旧 `compute_legacy_fim_loss` 与 `compute_crlb` 仅作为历史工具保留，不参与主训练/评估。第六步已将两个历史实验文件改为统一启动器，原实验内容保留在第一步快照；主评估另提供明确标注含义的信息尺度代理。

主训练入口在第四步接入修正后的 GNN，第五步完成 BP 与独立参照；第六步已完成[统一训练/恢复/评估](unified-experiment.md)及 CPU 小规模全流程。正式场景、多训练种子和 GPU/资源验收仍待后续，不将这些验证配置当作正式实验结果。

## 7. 验证与限制

本阶段新增 31 项目标/训练接口测试，并与第二步 30 项数据测试联合运行，61 项通过、无失败、无跳过。见 [test_objectives.py](../tests/test_objectives.py)。

验证包括：单锚点手算 `J,b,delta,mu,S`；智能体交叉块及相关先验；完整高斯 NLL 对照；带符号残差；标签隔离；删边前向及零权重反向；方差下限；距离 Jacobian 有限差分；8 掩码精确 KL 及梯度；极端 logits；度数 STE 梯度；均值/协方差/任务的 gradcheck；变规模和空图；节点置换；固定权重配置；训练批次汇总与参数更新；可选 float32。

对第二步的 7 张小图还完成了三 split 的新版目标及协方差有限性检查，并用现有真实 GNN 做一次集成梯度更新：梯度有限且非零，参数有变化，不保存模型。它只验证接口和梯度通路，不验证旧 GNN 采样分布、评分对称性、训练收敛或定位效果；未进行正式训练。

测试命令（仓库根目录，使用已配置的项目验证环境）：

```powershell
.\.venv-data\Scripts\python.exe -B -m pytest tests/test_objectives.py tests/test_data_contract.py -q -p no:cacheprovider --basetemp=data/test-tmp-objective-local
```

`gaussian_predictions(...,strict_binary=False)` 仅用于平滑诊断及有限差分；正式 `gib_objective` 始终要求二值前向。有限差分验证的是平滑矩阵运算，STE 另用链式法则和度数导数验证，不把硬阈值跳变的有限差分当作 STE 的数学导数。

本次只在 CPU 验证，未验证 GPU/服务器环境。论文和第一步版本快照保持不变。
