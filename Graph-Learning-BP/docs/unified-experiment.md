# 当前实验运行入口

统一入口为 python -m gib4cl，可在不含 legacy 的发布包中使用。

```text
python -B -m gib4cl --help
python -B -m gib4cl generate --config configs/smoke-pipeline.json --data-root data/demo
python -B -m gib4cl train --config configs/smoke-pipeline.json --data-root data/demo --output-root outputs/train --training-seed 101 --device cpu --threads 1
python -B -m gib4cl evaluate --checkpoint 实际完整检查点.pt --data-root data/demo --output-root outputs/eval --bp-seed 700 --bootstrap-seed 1700 --device cpu --threads 1
```

generate 拒绝覆盖已有数据；run 可一次完成生成、训练、加载最佳完整检查点及 Dense/BFS/GNN 评价。复用已有 run 数据需显式 --reuse-data 并通过校验。训练/评价新目录也拒绝覆盖。

train --resume 从原检查点继续相同计划、优化器、调度和随机状态；只接受相同源码、数据、验证规则和执行环境。train --init-from 先载入指定参数，再建立新训练状态。当前每轮对每图随机选边 16 次，以图等权平均验证总损失挑选最好一轮；概率门槛子图单独作对照，不参与挑选。规则、日志与限制见 [验证和选模](validation-selection.md)，test 不参与选模。

新完整检查点为 gib4cl.checkpoint.v2，保存验证规则和随机流设置。旧 v1 可读取/评估/用作 init-from，不能在新规则下直接 resume；恢复旧任务需其原始源码与环境。正式配置的 validation_repeats=16、validation_seed=1700 均在 training 段，改值须建立新运行。

当前训练另在运行目录 diagnostics/ 按批次保存梯度、裁剪、更新和选边统计，每轮汇总存于 history.json 的 train_diagnostics；失败记录单独保留。格式及衰减/裁剪的解释见 [训练日志](training-diagnostics.md)。日志版本也写入检查点；更新源码后恢复旧 v2 运行同样需要原始源码，不直接混用。

evaluate 必须指定完整版本化检查点，不缺省退回随机权重。三拓扑在同图/同重复下共享先验支持；BFS 为确定性多源森林，GNN 默认严格 p>0.5，不隐式补边。逐条结果保留配置、数据/检查点/源码身份、支持种子和摘要、均值/协方差及数值诊断。

多训练种子、BP 种子面板、失败恢复和分层汇总见 [种子与编排](delivery-and-seeds.md)。指标定义见 [评价口径](evaluation-metrics.md)，正式操作前的完整验收见 [acceptance.md](acceptance.md)。

正式选参的定向轮数轨迹、候选阶段和共同指标入口为 python -B -m tools.selection_runner，默认只列计划。见 [选参执行说明](selection-execution.md)。旧 run/suite 是完整生成/测试流程，选参阶段不使用它们生成正式 test；新入口只使用 train/val。
