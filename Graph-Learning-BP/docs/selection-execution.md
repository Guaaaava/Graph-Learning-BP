# 定向 BP、候选训练与共同指标：执行入口

接入日期：2026-11-05。统一入口为 python -B -m tools.selection_runner。默认只列计划，必须显式 --execute 和新 --output-root 才执行；没有自动下一阶段，不填写正式四系数或自动选定 BP 轮数。夜间分批后台运行、时间预算和早间报告由 [night_runner](night-running.md) 调用本入口提供；两者都不自动推进到未确认阶段。

## 四个命令

| 命令 | 作用 |
| --- | --- |
| iterations | 指定验证图编号，比较同支持上的多个 BP 轮数，保留 ESS/最大权重/残差及位置、协方差变化 |
| candidates --stage A/B/C/D | 按已登记协议生成候选、训练到筛选暂停点，使用共同的高斯代理指标比较 |
| candidates --stage promote/final | 显式晋级同一训练日程，并以共同粒子 BP 预算复核；最终按全部指定训练种子汇总 |
| review | 导入已预选检查点，单独进行代理或 BP 复核，共享 Dense/BFS 基线 |
| compare | 读取一个完整 records 报告，重新生成共同指标、区间和权衡代表；不训练、不运行 BP |

后面三个阶段命令分别调用，不因 A 完成就自动运行 B。--max-jobs 限制本次新增的工作单元数：一次候选训练到暂停点、一次模型/图/粒子种子预测，或一次 BP 最大轮数轨迹，各算一个单元；已完成缓存不重复计数。这些单元耗时不同，数量限制不是墙钟时间限制。

## 先看正式计划

在源码根目录执行（无需开服务器，不生成数据或读取模型；仍需要项目 Python 依赖）：

```text
python -B -m tools.selection_runner iterations --plan
python -B -m tools.selection_runner candidates --stage A --plan
python -B -m tools.selection_runner candidates --stage B --plan
python -B -m tools.selection_runner candidates --stage promote --plan
```

默认轮数首批是 val 0/1/5/6、BP 种子 700、P10000、T3/5/10/20、Dense/BFS，共 8 条轨迹、32 个比较点；不自动扩到三种子。A 列出 12 个具体 beta/eta 组合，训练总日程 200 轮，在第 20 轮暂停。后续依赖实际前一阶段的结果，计划中显示依赖和数量上限，不凭空构造前阶段赢家。

promote/final 执行必须显式 --bp-iterations 和 --bp-chunk-size，未确认时不回退到配置里的 T3。P 和重复种子从协议取值，也可用显式参数声明复核预算；候选晋级阶段的重复种子必须符合该阶段的协议。预算覆盖写入报告，不自动改正式配置。

## 本地微型流程用法

以下均为软件夹具：3 张训练图、2 张验证图、16 粒子、总日程 2 轮；不是正式预实验或参数推荐。smoke-protocol.json 只缩小软件检查规模，不替代 protocol.json。

```text
python -B -m tools.selection_runner iterations --config configs/smoke-pipeline.json --protocol configs/selection/smoke-protocol.json --execute --output-root outputs/software-rounds
python -B -m tools.selection_runner candidates --config configs/smoke-pipeline.json --protocol configs/selection/smoke-protocol.json --stage A --execute --output-root outputs/software-search --max-jobs 1
python -B -m tools.selection_runner candidates --config configs/smoke-pipeline.json --protocol configs/selection/smoke-protocol.json --stage A --execute --output-root outputs/software-search --resume
```

在同一 software-search 上，继续显式使用 --resume 和 --stage B、C、D。然后 promote/final 还需 --bp-iterations 1 --bp-chunk-size 8（这些仅是微型夹具数值）。完整软件流程已由 test_selection_runner 自动执行，不需要用户为准备正式运行再重复手工跑一遍。

正式服务器命令在执行前按 GPU 短试结果确认；本页没有授权启动正式网格或承诺整晚完成量。

## BP 轨迹和历史参考

iterations 支持 --graph-indices、--seeds、--particles、--iterations、--chunk-size，仅使用 val。每图/种子在 CPU 按 bp-budget 随机流一次抽取支持；两拓扑和各轮数复用同一支持。一次运行到最大 T 保存中间状态，再独立运行最小请求 T 核对均值/协方差/权重，容差为 1e-12。无智能体边时实际传播为 0 轮，仍输出各请求轮数对应的相同局部结果。

轨迹报告中的耗时/显存属于最大 T 的整次轨迹（含历史保存），不伪分摊成各轮数单独耗时；独立核对有额外计算。相邻轮数和相对最大 T 的位置/协方差/误差/覆盖变化均输出，最大 T 不是真实后验参照或自动合格线。零参照协方差的相对变化不除零，另报绝对变化和分母为零的数量。

--support-reference 可直接读取阶段 13 的原始 stage13-oat-…json：用 baseline_input_digests、原 sample_id、P/种子和 support_digest 核对输入及支持；若请求 T3，再核对历史均值/协方差（跨设备核对 rtol1e-8/atol1e-9，只是数值一致性容差）。不匹配则停止，不自动换种子或忽略参考。

也可使用另一轮新入口输出的 reports/support-reference.json。没有参考时明确标注“同随机流规则的新运行，历史复用未验证”，不会只凭种子相同声称配对。阶段 13 报告保留原样；本工具只读适配，不改写历史记录。

## 候选日程与模型副本

- A：gamma=kappa=0，beta×eta；B：取 A 精度代表分别加入 gamma/kappa；C：前两名非零分支值加零值联合复核；D：取 C 精度代表复核 beta/eta 邻域。重复组合按系数与种子共用模型，最多 36 个不同组合，不自动扩展保留边界。
- A–D 在同一筛选暂停点、相同验证前缀上用高斯代理比较；D 后汇集 A–D 所有不同候选，挑最多三个不同权衡代表。
- promote 继续同配置/同种子的模型到第 60 轮，主选模仍用完整验证集，再以 val 前 16 图和显式 BP 预算比较。
- final 按晋级后的权衡代表、训练种子 101/102/103 运行完整 200 轮，再用 val 前 32 图与三个 BP 种子复核。已有 101 同配置状态继续，其他种子独立初始化。
- 20/60 是暂停点，学习率和温度一直按 200 轮日程运行；不把 20 轮当完整余弦日程，不跨系数初始化。
- 每个暂停点的 best 和 last 都复制为不可变文件，带 SHA 校验。后来继续训练不会改变早期筛选依据；没有早期副本却已有更晚权重时拒绝倒填早期模型。

Dense/BFS 结果按图、粒子种子及预算缓存，所有候选复用；同一训练种子初始权重摘要必须一致。训练目标、裁剪、衰减、验证选模规则和粒子算法不变。

## 数据、恢复与失败

候选首次只生成完整 train/val，后续共用同一文件；也可 --data-root 精确复制已有的数据及校验 sidecar。绝不生成/读取正式 test。

review 默认使用检查点登记的原 train/val 文件；若原路径不可达，用 --data-root 指定含原文件的目录。重新生成相同样本不保证文件相同，因为文件包含时间和生成配置等元数据；不能因此放宽数据 SHA 检查。缺原文件或身份不匹配就停止。

首次输出目录必须不存在，续跑必须 --resume，并保持配置、协议、执行代码、设备/环境和线程相同。manifest 记录身份，jobs 下每单元原子保存记录和自身内容摘要；模型副本、数据和报告有独立校验。失败保留在 failures，须显式 --retry-failed 才重试。源代码、协议、原数据或副本被改不会自动重跑掩盖差异。

训练已有完整 epoch 时从 last_checkpoint 恢复；首轮尚未提交就失败时，显式重试会保留原目录，在 -retryN 新目录从同一种子重做。不能从批次日志恢复未提交的优化状态。进程在已完成记录写入与 manifest 更新之间中断时，可验证并接纳该完整记录，不重复其计算。

夜间调度器已接入后台启动、分批预算和报告；本入口提供可复用单元与恢复接口。时间预算在完整轮次/图预测/轨迹边界暂停，不保证到点强制退出。

## 共同指标及文件

每个模型/图/粒子种子保留位置、协方差、误差、多个覆盖水平、椭圆面积、NEES/NLL、成本和结构；BP 另保留 ESS、最大权重、残差和收敛状态。无效协方差计为不覆盖，仍保留失败比例，不删除困难图。

高斯代理使用联合精度矩阵逆的节点边缘协方差，不取精度对角块的逆。它只用于低成本粗筛，不当成 BP 结果。两种方法分别写报告，比较器只接受一种明确方法、完整 val 面板、相同图/重复身份；不接受缺失模型、混乱支持、重复预测、非有限主指标或 test 报告。

先图内平均训练与粒子重复的 MSE、覆盖和成本，再图等权平均，RMSE 最后开方；不平均各图 RMSE，也不按节点总数加权。最终所有候选必须具有同一完整训练种子集合，另列每个种子的结果，不选最好种子。

比较 RMSE、95% 椭圆覆盖差、通信成本和无效协方差比例，按协议保留不被全面胜过的权衡集合，并列精度/校准/成本代表。不同系数加权后的总损失不参与排名。图级 bootstrap 保留同图全部重复，给出条件于这些模型的区间及相对 Dense/BFS 的配对误差变化；单图不报图间区间，Dense 成本为零时不编造相对比值。区间不是独立测试性能保证。

主要输出：

| 路径 | 内容 |
| --- | --- |
| manifest.json / jobs / failures | 运行身份、逐单元结果及失败历史 |
| checkpoints / runs | 冻结的里程碑模型与可续训的当前状态 |
| reports/iterations.json | 轨迹、变化量、ESS 和参考核验 |
| reports/stage-A…final.json | 当前阶段是否完整、模型清单和权衡角色；partial 不提供排名 |
| reports/records-A…final.json | 全部原始预测与输入/模型/支持身份 |
| reports/comparison-A…final.json | 共同指标、种子分项和条件统计区间 |
| tables/comparison-….json/.csv/.md | 可审阅表，列出四系数及 beta*gamma/beta*kappa |

compare 可对 records-….json 生成新的表；必须给新输出目录。报告 JSON 为权威数据，CSV/Markdown 是可重新生成的阅读副本。任何软件夹具选择出的“代表”都不是正式参数。
