# 本地场景与网络预览

`tests/test_data_contract.py` 和 `tests/test_experiment_pipeline.py` 已分别覆盖生成器和 BFS 的自动检查。
`tools/preview_network.py` 将现有生成、BFS、结构指标和绘图接口组合成一个可直接运行的预览脚本，代码不放在 `gib4cl/` 中。

在项目根目录运行：

```powershell
.\.venv-data\Scripts\python.exe -B -m tools.preview_network --num-agents 30
```

默认读取 `configs/experiment.json`：四角及中心五个锚点、全区域均匀智能体、统一通信半径、LOS 高斯测距。
`--num-agents 30` 把本次有效配置的节点数量上下限都改为 30；不传时按配置区间确定节点数。
脚本只使用生成相关配置；不要求训练损失系数已填写，也不训练 GNN、运行 BP 或读取模型检查点。

每次默认创建新的 `outputs/network-preview/<ID>/`，输出：

| 文件 | 内容 |
| --- | --- |
| `scene.png` | 真值场景，以及真值与带噪初始位置的对照；初值不裁剪到区域内 |
| `topology.png`、`topology.svg` | 同一场景上的 Dense、真实多源 BFS 森林和示例二值掩码 |
| `config.json` | 应用了命令行覆盖值的完整配置 |
| `summary.json` | 实际节点数、种子、锚点坐标、三组 `s`、边编号、结构指标及来源校验值 |
| `graph.pt`、`graph.json` | 一张原始图的数据集及校验侧文件，可通过 `LocalizationDataset` 重新读取 |

第三张图明确标注 `Example s (not trained GNN)`。默认随机保留约一半物理边；保留比例只服务于展示。
当候选边数 E≥2 时至少保留一条、删除一条；E=1 时删除唯一边，E=0 时保持空图。
不隐式补边或修复连通性，因此示例掩码可能产生未连接锚点的节点。

可用参数：

```powershell
# 换一张可复现的图，使用一个新的显式输出目录
.\.venv-data\Scripts\python.exe -B -m tools.preview_network --num-agents 30 --data-seed 101 --seed-offset 1 --output-dir outputs/network-preview/my-check-01

# 保持场景不变，只改变示例选边
.\.venv-data\Scripts\python.exe -B -m tools.preview_network --num-agents 30 --mask-seed 123 --keep-ratio 0.4

# 小规模配置，保存图片后同时显示窗口
.\.venv-data\Scripts\python.exe -B -m tools.preview_network --config configs/smoke-pipeline.json --show
```

`--output-dir` 必须是新目录。场景使用独立的 `preview` 种子命名空间，不写入正式 train/val/test 数据目录。
只改 `--mask-seed` 或 `--keep-ratio` 不改变场景、测量和 BFS。
