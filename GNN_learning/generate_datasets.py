import os
import torch
import random
import numpy as np
from tqdm import tqdm

from generate_network import generate_localization_network
import config

def set_global_seed(seed):
    """设置全局种子，确保数据集生成的绝对可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_dataset(split_name, num_graphs, seed_offset):
    """
    生成指定规模的图数据集并保存。
    :param split_name: 数据集名称 (train, val, test)
    :param num_graphs: 要生成的图数量
    :param seed_offset: 种子偏移量，保证不同数据集的随机状态不同
    """
    print(f"\n>>> 开始生成 [{split_name.upper()}] 数据集 ({num_graphs} 张图) ...")
    
    # 加上偏移量，确保 train, val, test 集合的数据完全不同
    set_global_seed(config.TORCH_SEED + seed_offset)
    
    dataset_list = []

    # 使用 tqdm 包装 range，提供进度条
    for i in tqdm(range(num_graphs), desc=f"Generating {split_name}"):
        # 1. 动态抖动网络规模，强制 GNN 学习尺度不变性 (Scale Invariance)
        # 让 Agent 数量在 25 到 45 之间随机浮动
        current_agents = random.randint(25, 45)
        # 让 Anchor 数量在 3 到 6 之间随机浮动
        current_anchors = random.randint(3, 6)

        # 2. 调用挑战级场景生成器
        data_dict = generate_localization_network(
            num_agents=current_agents, 
            num_anchors=current_anchors, 
            area_size=100.0, 
            comm_radius=25.0, 
            base_noise=0.5, 
            noise_scale=0.05, 
            scenario_type='challenge'
        )
        
        # 记录这张图的基础信息，方便后续查阅
        data_dict['num_agents'] = current_agents
        data_dict['num_anchors'] = current_anchors
        
        dataset_list.append(data_dict)

    # 3. 确保存储目录存在
    os.makedirs('datasets', exist_ok=True)
    save_path = f"datasets/{split_name}_dataset.pt"

    # 4. 使用 PyTorch 的高效序列化方法保存数据
    torch.save(dataset_list, save_path)
    print(f"[{split_name.upper()}] 数据集已保存至: {save_path} (文件大小: {os.path.getsize(save_path) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    # ==========================================
    # 数据集规模配置
    # ==========================================
    NUM_TRAIN = 2000  # 训练集：让 GNN 见识各种极端情况
    NUM_VAL   = 500   # 验证集：用于在训练中途监控泛化能力，防止过拟合
    NUM_TEST  = 500   # 测试集：绝对隔离，用于最终跑分
    
    print("==================================================")
    print("初始化泛化图数据集构建引擎...")
    print("==================================================")
    
    # 依次生成三大数据集 (传入不同的 seed_offset 防止数据重复)
    create_dataset('train', NUM_TRAIN, seed_offset=0)
    create_dataset('val', NUM_VAL, seed_offset=1)
    create_dataset('test', NUM_TEST, seed_offset=2)
    
    print("\n>>> 所有数据集生成完毕！准备进行 PyG 批处理")