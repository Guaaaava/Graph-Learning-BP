"""
批量数据集生成

生成 train / val / test 三个集合, 每个集合包含多张独立拓扑图。
每张图有随机变化的 Agent 数量 (NUM_AGENTS_MIN~MAX);
Anchor 数量与分布由 config.SCENARIO_TYPE 唯一决定, 不随机。
"""

import os
import torch
import random
import numpy as np
from tqdm import tqdm

from generate_network import generate_localization_network
import config


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataset(split_name, num_graphs, seed_offset,
                   agents_min=None, agents_max=None):
    """生成并保存一个 split。

    Parameters
    ----------
    split_name : str
        'train' / 'val' / 'test' → 存为 datasets/{split_name}_dataset.pt
    num_graphs : int
        图的张数
    seed_offset : int
        全局种子 = config.TORCH_SEED + seed_offset
    agents_min, agents_max : int or None
        agent 数区间覆盖。None 则用 config.NUM_AGENTS_MIN / MAX。
        跨规模泛化实验用这个参数在小规模/大规模之间切换, 无需改动 config。
    """
    lo = config.NUM_AGENTS_MIN if agents_min is None else agents_min
    hi = config.NUM_AGENTS_MAX if agents_max is None else agents_max

    print(f"\n>>> 生成 [{split_name.upper()}] 数据集 "
          f"({num_graphs} 张图, agents {lo}~{hi}) ...")

    set_global_seed(config.TORCH_SEED + seed_offset)

    dataset_list = []
    for i in tqdm(range(num_graphs), desc=f"Generating {split_name}"):
        current_agents = random.randint(lo, hi)

        data_dict = generate_localization_network(
            num_agents=current_agents,
            area_size=config.AREA_SIZE,
            comm_radius=config.COMM_RADIUS,
            base_noise=config.BASE_NOISE,
            noise_scale=config.NOISE_SCALE,
            init_pos_cov=config.INIT_POS_COV,
            scenario_type=config.SCENARIO_TYPE,
            agent_distribution=config.AGENT_DISTRIBUTION,
            enable_nlos=config.ENABLE_NLOS,
            nlos_ratio=config.NLOS_RATIO,
            nlos_bias_range=(config.NLOS_BIAS_MIN, config.NLOS_BIAS_MAX),
        )

        data_dict['num_agents'] = current_agents
        data_dict['num_anchors'] = data_dict['anchors_pos'].shape[0]
        dataset_list.append(data_dict)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, 'datasets')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{split_name}_dataset.pt")
    torch.save(dataset_list, save_path)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"[{split_name.upper()}] 已保存: {save_path} ({size_mb:.2f} MB)")
    return dataset_list


if __name__ == "__main__":
    NUM_TRAIN = 2000
    NUM_VAL = 500
    NUM_TEST = 500

    print("=" * 50)
    print("批量数据集生成引擎")
    print(f"场景: {config.SCENARIO_TYPE}")
    print(f"区域: {config.AREA_SIZE}x{config.AREA_SIZE} m")
    print(f"通信半径: {config.COMM_RADIUS} m")
    print("=" * 50)

    create_dataset('train', NUM_TRAIN, seed_offset=0)
    create_dataset('val', NUM_VAL, seed_offset=1)
    create_dataset('test', NUM_TEST, seed_offset=2)

    print("\n>>> 全部数据集生成完毕！")
