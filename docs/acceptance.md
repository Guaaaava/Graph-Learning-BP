# 发布包与完整 CPU 验收

验收对象应为固定版本的源代码发布包。历史代码保留在源仓库 legacy，不参与包内运行；环境、数据、旧结果、模型权重与 Git 元数据不打包。

## 1. 源码打包

在源工程根目录、现有 Python 环境中运行：

```powershell
.\.venv-data\Scripts\python.exe -B -m tools.package_project build --output-root dist/my-release-01
.\.venv-data\Scripts\python.exe -B -m tools.package_project verify --archive dist/my-release-01/Graph-Learning-BP.zip
```

每次使用新目录，拒绝覆盖。白名单为 gib4cl、configs、tests、tools、docs 及明确列出的根文件。打包程序拒绝历史路径、符号链接、源码目录内意外产物；所有 ZIP 项固定顺序和时间戳，同样源码可生成字节一致的 ZIP。

保存 ZIP、PACKAGE-MANIFEST.json 和 ZIP.sha256。MANIFEST 不把自身加入自身哈希清单；它由 ZIP 外部 SHA-256 以及包外同名副本提供核对依据。源码指纹由所有源文件的相对路径、SHA-256、字节数确定。

## 2. 解压与环境

在新位置解压 ZIP，进入其中 Graph-Learning-BP。发布包不包含任何本机虚拟环境。选择明确的项目解释器，确认 Python/PyTorch/PyG/NumPy/pytest 环境；新环境可按 requirements.txt 安装，但独立安装结果仍须验证。

在服务器上选择 CUDA 构建、验证驱动/设备及资源是后续环境验收，不由当前 CPU 验收替代。不要将 ZIP 里的源码与其他版本的源码合并后继续沿用原清单。

## 3. 验收命令

已激活目标解释器后：

```text
python -B -m tools.package_project verify --directory .
python -B -m tools.accept_cpu --output-root outputs/acceptance-01
```

也可使用已知解释器的绝对路径调用。发布包中的完整验收会自动先核对 PACKAGE-MANIFEST；源码工作区没有该文件时，明确记录为工作区模式。不要编辑候选源码后继续声称它通过原包校验。

## 4. 自动执行的项目

1. 生产代码不导入历史/测试/旧根路径，顶层不存在 BP、GNN_learning、NBP_ST、NEBP、references、results。三份核心算法与固定 AST 校验数据相符。
2. 全部 tests 回归，包括数据身份/先验/索引、目标和梯度、对称评分/采样、粒子 BP 的独立参照、校准/统计、种子独立性、失败重试及打包篡改检测。
3. checkpoint 审计：生成→短训练→最佳检查点→三拓扑；连续训练与 epoch 恢复的模型、优化器、调度、历史、随机状态一致；显式微调载入来源正确。
4. metrics 审计：NumPy 线性求解/行列式、独立 DFS/组合计数复核；共享粒子、JSON/CSV/报告和单图/零边/全部无效协方差的处理一致。
5. 两训练种子 × 两 BP 种子真实子进程面板：任务边界停止后继续，校验成功任务未重做、固定数据 SHA 不变，完成后再次调用不改 manifest。
6. 新目录重放整个面板，模型参数摘要和逐条评价/摘要完全一致；两种训练种子产生不同参数，独立测试图数量仍为原数据图数。

数学参照位于 tests/references，属于当前可执行验收代码；它与已归档的外部 NEBP 不同。来源固件 tests/fixtures/provenance.json 仅含冻结哈希和身份数据，不含需要执行的旧代码。详见[来源说明](provenance.md)。

## 5. 通过条件与证据

- 主命令退出码为 0，outputs/acceptance-01/acceptance.json 的 status 为 passed。
- 包内运行时 package_integrity.status 为 verified，且运行期间生产源码校验值不变。
- tests.xml 的 failures/errors 为 0；无 CUDA 环境仅允许 CUDA runtime unavailable 对应检查跳过，无其他未说明跳过。当前测试数以该候选版本 JUnit 为准。
- resume-audit.log、metrics-audit.log 及各自 verification.json 全部完成。
- 两个面板的 manifest 均 complete，全部计划任务完成；无 failed/blocked/running 或缺失训练种子。
- 固定数据校验值、任务边界恢复、完成状态幂等、跨种子基线一致、整套重放检查均通过。
- 无效协方差、低 ESS、未收敛状态按既定口径保留，不靠删除样本通过验收。

保留 acceptance.json、tests.xml、阶段日志和面板 manifest/summary，以及验收时使用的 ZIP 与外部 SHA-256。首次验收失败时先读 acceptance.json 的 error 和对应日志；修正后生成新候选包，在新目录重新验收。

本轮打包与验收证据另存于 dist 发布目录；该证据不加入它所校验 ZIP 的源文件清单，避免自引用。

## 6. 正式训练前的后续门槛

本命令覆盖 CPU 源码与小规模流程，不自动认证独立依赖安装、服务器 GPU 精度/吞吐/峰值内存或正式模型性能。正式配置中的目标权重、数据规模、训练种子数、BP 粒子/轮数等需预实验后冻结；最终测试集不用于反复选择参数。
