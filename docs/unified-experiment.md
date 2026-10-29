# 当前实验运行入口

统一入口为 python -m gib4cl，可在不含 legacy 的发布包中使用。

```text
python -B -m gib4cl --help
python -B -m gib4cl generate --config configs/smoke-pipeline.json --data-root data/demo
python -B -m gib4cl train --config configs/smoke-pipeline.json --data-root data/demo --output-root outputs/train --training-seed 101 --device cpu --threads 1
python -B -m gib4cl evaluate --checkpoint 实际完整检查点.pt --data-root data/demo --output-root outputs/eval --bp-seed 700 --bootstrap-seed 1700 --device cpu --threads 1
```

generate 拒绝覆盖已有数据；run 可一次完成生成、训练、加载最佳完整检查点及 Dense/BFS/GNN 评价。复用已有 run 数据需显式 --reuse-data 并通过校验。训练/评价新目录也拒绝覆盖。

train --resume 从原检查点继续相同计划、优化器、调度和随机状态；只接受相同源码、数据和执行环境。train --init-from 先载入指定参数，再建立新训练状态。最佳模型以固定系数下验证 total 最小选择，测试集不参与选模。

evaluate 必须指定完整版本化检查点，不缺省退回随机权重。三拓扑在同图/同重复下共享先验支持；BFS 为确定性多源森林，GNN 默认严格 p>0.5，不隐式补边。逐条结果保留配置、数据/检查点/源码身份、支持种子和摘要、均值/协方差及数值诊断。

多训练种子、BP 种子面板、失败恢复和分层汇总见 [种子与编排](delivery-and-seeds.md)。指标定义见 [评价口径](evaluation-metrics.md)，正式操作前的完整验收见 [acceptance.md](acceptance.md)。
