# 历史归档区

本目录只用于保留来源和追溯旧行为，不参与当前生产运行或源代码发布包。

- compat/：原顶层 GNN_learning、BP、NBP_ST 的旧兼容目录；
- config_constants.py、fim_objective.py：旧配置常量与旧 FIM 目标；
- references/NEBP/：用户提供的原始外部参考，源码字节保持不变；
- audits/：旧阶段审计脚本，保留原名和原文；
- snapshots/：第一步与整理前的不可变源码快照；
- records/、docs/、manifests/：历史证据、搬移前文档及文件清单；
- results/：原有历史结果目录。

旧兼容脚本不再作为默认导入路径维护，部分导入已随归档失效。要运行当前实现请使用顶层 gib4cl；要复现特定历史版本请使用相应完整快照和原环境，不要将这些脚本混回发布包。

当前独立验收使用 tests/references 中的数学参照和 tests/fixtures/provenance.json 中的固定来源数据，无需导入本目录。
