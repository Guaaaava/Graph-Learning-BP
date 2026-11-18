# 独立随机种子与多次实验编排

当前配置使用一套固定输入结构，种子字段为：

```json
{"seeds": {"data": 100, "training": 101, "bp": 700, "bootstrap": 1700}}
```

data 控制 split/样本的数据生成；training 控制初始化、训练采样和 DataLoader；bp 控制每图/重复的先验支持；bootstrap 控制统计重抽。数值均为 [0,2^63) 整数，拒绝 bool。只改后三者不改变数据指纹。固定数据应复用已生成并校验的文件，不能将重新生成文件的创建时间/来源元数据也假定为相同。

仅接受当前字段；旧顶层 seed、evaluation.seed 和配置版本字段不再迁移。输入中的重复 JSON 键和未知字段均报错。配置、数据身份、源码与执行环境一致时才允许原样恢复训练；改变条件应建立新运行。完整配置通过 to_dict 保存为普通字典，split_sizes 在内存中只读。

## 固定数据面板

一个 suite 对应一个明确基础配置、数据种子和固定划分；训练种子与 BP 种子组成面板。示例见 [smoke-seeds.json](../configs/suites/smoke-seeds.json)：

```text
python -B -m gib4cl suite --suite-config configs/suites/smoke-seeds.json --plan
python -B -m gib4cl suite --suite-config configs/suites/smoke-seeds.json --output-root outputs/panel --device cpu --threads 1
```

--plan 只显示解析后的配置/任务，不生成数据。默认在面板目录生成一次 train/val/test；已有数据用 --data-root 指明，并显式 --reuse-data。会核对 split、图数、指纹、文件/sidecar 校验。

面板由单个编排进程顺序运行，每项为独立 Python 子进程。当前只有五锚点均匀 LOS 场景；不同超参数条件分别使用基础配置与 suite_id。suite 输入、运行清单及汇总不再附加未参与读取校验的版本字段；恢复时直接核对配置、源码、环境、任务计划与文件校验值。smoke 面板含 2 个训练任务与 4 个评价任务；每次评价分别运行三拓扑，并沿用基础配置的 BP 重复数。

## 状态、失败与恢复

```text
python -B -m gib4cl suite --suite-config configs/suites/smoke-seeds.json --output-root outputs/panel --resume --device cpu --threads 1
```

完成任务先核对来源、数据、配置、检查点和产物/日志校验，再跳过；未执行任务继续。失败任务保留退出码、异常、日志、开始/结束时间和尝试历史，依赖其训练的评价标为 blocked，其他训练种子仍可完成。

失败/中断任务需显式 --resume --retry-failed，创建新 attempt，保留原记录并从头执行该任务。单次训练的 epoch 续训使用独立的 train --resume。--stop-after-jobs 1 可在完整任务边界停止，不改变训练 epochs 计划。

新输出拒绝覆盖；配置/源码/环境变化、成功产物缺失或被修改时拒绝继续。整个面板已完成时再次恢复只校验并返回，不重复执行。

只有全部计划任务完成才写 summary.json。其余写 partial_summary.json 和缺失种子列表，并在 manifest 中指明当前摘要；失败返回非零退出码。

## 统计单位

每个模型合并固定 BP 种子面板，重新映射重复编号但仍以 sample_id 为图簇，全部 BP 种子/重复及节点留在该簇内。跨训练种子再汇总每模型的图级均值、训练种子标准差和 bootstrap 区间，条件于同一固定数据与 BP 面板。

两层不确定性分别报告，不宣称联合数据/训练区间。一个有效训练种子时 std/CI 为 null；条件评分保留有效数量。Dense/BFS 跨模型必须一致，核对后只保存一份 fixed_baselines，重复副本不增加独立试验数。

面板保存 manifest、每训练种子的完整图统计、固定基线、GNN 跨训练种子标量/曲线/配对差值，以及逐任务配置/日志/模型/原始评价。完整源包与恢复/重放验收见 [验收说明](acceptance.md)。
