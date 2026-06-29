"""
批量数据集生成

生成 train / val / test 三个集合, 每个集合包含多张独立拓扑图。
每张图有随机变化的 Agent 数量 (25~45) 和 Anchor 数量 (3~6)。
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


def create_dataset(split_name, num_graphs, seed_offset):
    print(f"\n>>> 生成 [{split_name.upper()}] 数据集 ({num_graphs} 张图) ...")

    set_global_seed(config.TORCH_SEED + seed_offset)

    dataset_list = []
    for i in tqdm(range(num_graphs), desc=f"Generating {split_name}"):
        current_agents = random.randint(config.NUM_AGENTS_MIN, config.NUM_AGENTS_MAX)
        current_anchors = random.randint(config.NUM_ANCHORS_MIN, config.NUM_ANCHORS_MAX)

        data_dict = generate_localization_network(
            num_agents=current_agents,
            num_anchors=current_anchors,
            area_size=config.AREA_SIZE,
            comm_radius=config.COMM_RADIUS,
            base_noise=config.BASE_NOISE,
            noise_scale=config.NOISE_SCALE,
            init_pos_cov=config.INIT_POS_COV,
            scenario_type=config.SCENARIO_TYPE,
        )

        data_dict['num_agents'] = current_agents
        data_dict['num_anchors'] = current_anchors
        dataset_list.append(data_dict)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, 'datasets')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{split_name}_dataset.pt")
    torch.save(dataset_list, save_path)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"[{split_name.upper()}] 已保存: {save_path} ({size_mb:.2f} MB)")


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
