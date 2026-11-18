# 夜间分批执行与早间报告

本地实现日期：2026-11-05。入口是 `python -B -m tools.night_runner`，在源码根目录运行。脚本只调度现有选参代码，不改变损失、粒子算法、论文场景或选模规则。未启动服务器或正式训练。

## 可以退出本地电脑，但服务器要保持开机

在**远程服务器终端**执行 `start` 后，监督进程脱离 SSH 会话，标准输入关闭、日志写文件；Linux 使用独立会话并忽略 SSH 断开产生的 SIGHUP。此后可以关闭 SSH、退出 Codex 或关闭自己的电脑，服务器上的进程继续工作。服务器本身关机、被平台回收或终止进程时不能继续计算。

`start` 返回 PID 和启动日志路径，只代表提交了后台进程。需紧接着用 `status` 核实，启动失败详情在启动日志。`run` 是前台执行；它用于本地检查或主动观察运行。

## 一批做多少，什么时候停

每次提交同时受下列限制：

- `max_units`：新增工作单元数量。一个单元是最多 `epochs_per_slice` 轮训练（默认 1 轮）、一个图的 BP 预测，或一条从初始状态运行到最大请求轮数的定向轨迹。缓存命中不计数，失败开始计算的单元也计数。
- `max_seconds`：本次提交的时间预算，不超过协议的 7200 秒。进程准备、数据准备、评价汇总和写报告之前的执行耗时都计入；最后报告本身还有少量写盘开销。
- 耗时预留：必须提供实际短试得到的单元耗时估计及来源说明。每次启动下一个单元前，要求剩余时间能容纳“估计 × safety_factor”；正式模板的安全倍数是 1.5。训练/代理/BP 的估计不同，应在相应阶段重新测量和填写，按较慢单元及初始化开销保守估计。

时间预算是**在安全边界暂停**：训练完成一轮验证和检查点写入后停，BP 完成一次图预测/定向轨迹后停。不会在矩阵计算中途杀进程。因此单次计算比估计慢时，实际时间可能超出预算；早间报告记录实际耗时和超时量，不承诺到点立即退出。大轨迹首次短试仍需人为选择可承受规模。

训练切片不改变完整学习率/温度日程；达到原有 20/60/200 轮暂停点之前，不发布里程碑模型或候选排名。恢复时检查 last、best、history 和 run 登记是否形成一致的完整轮次；异常退出发生在多文件提交之间时停止并保留现场，不自动猜测修复。

**一阶段完成就停，阶段失败也停。** `--resume` 只继续当前阶段；看到“本阶段完成，等待确认”后，审阅并确认，再显式 `--resume --advance`。不自动 A→B，不自动扩种子/预算，不自动决定 BP 轮数，也不自动重试失败。

## 配置文件

| 文件 | 用途 |
| --- | --- |
| `configs/night/iterations-first.json` | 正式 P10000 定向检查首批：2 个新增轨迹单元上限、7200 秒；耗时估计待 GPU 短试 |
| `configs/night/candidates.json` | 登记 A/B/C/D/晋级/最终阶段，每批最多 20 单元、7200 秒，每个阶段之间确认；不会在一批内跨阶段 |
| `configs/night/smoke.json` | 软件检查夹具：CPU、小图、微型协议；不能作为正式参数或 GPU 速度依据 |

单批上限是资源安排，不是方法超参数或理论推荐。阶段确认之后，同一阶段可分多批运行。模板一经开始便固定；恢复严格检查批次、配置、协议、源码、Python 和引用的历史报告身份。实际训练还检查 PyTorch/CUDA 运行环境及原数据身份。代码更新或环境改变后，不会静默沿用旧状态。

## 先查看计划

下列命令不初始化 GPU、不加载模型、不生成数据；监督入口只依赖 Python 标准库：

```text
python -B -m tools.night_runner plan --batch configs/night/iterations-first.json
python -B -m tools.night_runner plan --batch configs/night/candidates.json
```

阶段 13 原报告的历史支持核对，可在开始前将其路径填写在定向批次的 `stages[0].options.support_reference`。否则清楚登记为新运行，不能只凭相同种子声称复用了历史支持。相对路径统一按源码根目录解释；输出目录按执行命令时的当前目录解释。

## 服务器运行命令模板

以下路径按既有远程工作目录编写；**先同步已验收的源码、核验环境、完成更新代码的 GPU 短试**。模板中的 `<实测秒数>` 和 `<短试记录>` 必须替换，不能原样执行。每批估计包含最慢原子单元、数据和模型加载开销；未知估计或一个单元已放不进预算时拒绝启动。

```text
cd /root/autodl-tmp/gib4cl-work/Graph-Learning-BP
python -B -m tools.night_runner start --batch configs/night/iterations-first.json --output-root /root/autodl-tmp/gib4cl-work/night-bp-rounds --unit-estimate-seconds <实测秒数> --estimate-note "<短试记录>"
python -B -m tools.night_runner status --output-root /root/autodl-tmp/gib4cl-work/night-bp-rounds
```

定向阶段未完成且只是预算暂停时，用原命令增加 `--resume`，再分批继续。定向检查结束后由人审阅位置、协方差、覆盖率、ESS、时间/显存，再决定复核 BP 预算。不会因为 T20 是最大请求值就自动认为它是真值或正式选择。

确认可以开始候选阶段后：

```text
python -B -m tools.night_runner start --batch configs/night/candidates.json --output-root /root/autodl-tmp/gib4cl-work/night-candidates --unit-estimate-seconds <训练切片实测秒数> --estimate-note "<训练及代理短试记录>"
```

当前阶段预算暂停：原命令增加 `--resume`。当前阶段完成且已确认下一阶段：增加 `--resume --advance`。推进到 promote/final 时还须提供已确认的 `--bp-iterations <轮数> --bp-chunk-size <分块大小>`，并换用训练切片/BP 两者保守耗时估计。这些 BP 设置一经该阶段开始便固定，续跑可以省略以沿用登记值。最终阶段仍按协议使用所有指定训练种子，不选择最好种子。

本地软件用例可前台运行，不需要远程服务器：

```text
python -B -m tools.night_runner run --batch configs/night/smoke.json --output-root outputs/night-software
```

自动化测试已运行实际微型选参和后台子进程；用户不必为了准备正式运行重复手动执行该夹具。

## 中止、失败与服务器重启

```text
python -B -m tools.night_runner stop --output-root <本次夜间目录>
python -B -m tools.night_runner status --output-root <本次夜间目录>
python -B -m tools.night_runner report --output-root <本次夜间目录>
```

`stop` 写入暂停请求，不杀进程；等当前完整单元结束，再看到 `active_lock=false` 才代表相关进程已释放锁。`status` 同时检查监督进程和工作进程的操作系统锁，避免监督进程退出后残留的工作进程被重复启动。锁随进程退出或服务器重启释放，不根据 PID 数字猜测并删除锁文件。

失败时保留工作目录、子进程日志、错误堆栈及原有失败记录。先调查错误，再用原命令加 `--resume --retry-failed`；不会跳过失败模型去排名。若监督进程意外退出，状态可能仍为 running，但 `status` 会标注无活动锁；恢复后保留中断记录，并让现有执行入口核验已完成单元和检查点。

服务器重启不会自动启动续跑，没有安装定时服务或开机服务。重新连接后检查状态及环境，再显式续跑。可在 00:00 有空余资源时提交；它没有替用户创建每日定时任务。

## 早上看什么

夜间目录主要文件：

| 文件 | 内容 |
| --- | --- |
| `progress.json` | 当前阶段、每批预算/耗时/单元计数、子进程 PID、阶段结果校验值 |
| `morning-report.md` / `.json` | 可读报告与权威数据；状态、失败原因、已保存训练轮次、完整阶段共同指标、已完成 BP 点的 ESS/最大权重 |
| `submissions/0001/unit-…` | 每个单元的结构化请求、开始记录、原始结果和完整输出日志 |
| `workspaces/candidates` | 原训练日志、验证记录、冻结模型和共同指标表 |
| `workspaces/iterations` | 定向轨迹、支持摘要及数值一致性核对 |

完整训练诊断保存在各 run 的 diagnostics/history，报告 JSON 附最后一轮训练诊断。可读报告列出的最好验证损失只用于同系数选轮次，不能跨系数排名；共同指标表才用于候选比较。部分完成的阶段不提供排名。

报告在每批结束时自动生成，早上如果仍在运行，可以显式 `report` 刷新；该命令不需要 GPU，也不修改模型。没有接入邮件、消息推送或服务器自动关机。GPU 吞吐、显存、日志开销和一晚能完成多少正式候选，仍须服务器短试确认。
