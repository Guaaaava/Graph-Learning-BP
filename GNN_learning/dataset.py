"""
PyG 数据集适配器

将 generate_network 输出的字典格式转换为 PyG Data 对象。
支持批量加载与 Mini-batch 训练。

节点特征 (5 维): [pos_x, pos_y, σ²_x, σ²_y, is_anchor]
边特征 (4 维):   [measurement, variance, pseudo_range_residual, is_anchor_edge]
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os


class LocalizationDataset(Dataset):
    def __init__(self, dataset_path):
        super().__init__()
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"找不到数据集文件: {dataset_path}")

        print(f"正在加载数据集: {dataset_path} ...")
        self.raw_data_list = torch.load(dataset_path, weights_only=False)
        print(f"加载完成，共 {len(self.raw_data_list)} 张图。")

    def __len__(self):
        return len(self.raw_data_list)

    def __getitem__(self, idx):
        raw = self.raw_data_list[idx]

        num_agents = raw['true_agents_pos'].shape[0]
        num_anchors = raw['anchors_pos'].shape[0]

        # ==========================================
        # 1. 节点特征 (5 维)
        # ==========================================
        # Agent 节点: [init_pos, σ², σ², 0]
        # Anchor 节点: [true_pos, 0, 0, 1]
        init_cov = raw.get('init_agents_cov', 25.0)
        agent_cov = torch.full((num_agents, 1), init_cov, dtype=torch.float32)

        agent_feat = torch.cat([
            raw['init_agents_pos'],                 # pos_x, pos_y
            agent_cov,                               # σ²_x
            agent_cov,                               # σ²_y
            torch.zeros(num_agents, 1),              # is_anchor = 0
        ], dim=1)  # (N, 5)

        anchor_feat = torch.cat([
            raw['anchors_pos'],                      # true pos (无噪声)
            torch.zeros(num_anchors, 1),             # σ²_x = 0
            torch.zeros(num_anchors, 1),             # σ²_y = 0
            torch.ones(num_anchors, 1),              # is_anchor = 1
        ], dim=1)  # (M, 5)

        x = torch.cat([agent_feat, anchor_feat], dim=0)  # (N+M, 5)

        # ==========================================
        # 2. 边特征 (4 维)
        # ==========================================
        meas = raw['measurements'].view(-1, 1)
        var = raw['edge_variances'].view(-1, 1)
        res = raw['pseudo_range_residual'].view(-1, 1)
        is_anc = raw['is_anchor_edge'].view(-1, 1).float()

        edge_attr = torch.cat([meas, var, res, is_anc], dim=1)  # (E, 4)

        # ==========================================
        # 3. 边索引 (转绝对索引)
        # ==========================================
        edge_index = raw['edge_index'].clone()
        anc_mask = raw['is_anchor_edge']
        edge_index[1, anc_mask] += num_agents  # anchor 局部索引 → 全局索引

        # ==========================================
        # 4. 组装 PyG Data
        # ==========================================
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=raw['true_agents_pos'],
            num_agents=num_agents,
            num_anchors=num_anchors,
            is_nlos_edge=raw.get('is_nlos_edge', torch.zeros(edge_index.shape[1], dtype=torch.bool)),
        )
        return data


if __name__ == "__main__":
    train_dataset = LocalizationDataset("datasets/train_dataset.pt")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for batch in train_loader:
        print(f"\nBatch 信息:")
        print(f"  子图数: {batch.num_graphs}")
        print(f"  总节点数: {batch.x.shape[0]}")
        print(f"  节点特征维: {batch.x.shape[1]}")
        print(f"  总边数: {batch.edge_index.shape[1]}")
        print(f"  边特征维: {batch.edge_attr.shape[1]}")
        print(f"  前5个节点特征:\n{batch.x[:5]}")
        print(f"  前3条边特征:\n{batch.edge_attr[:3]}")
        break
