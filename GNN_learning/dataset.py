import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os

class LocalizationDataset(Dataset):
    def __init__(self, dataset_path):
        """
        加载之前生成的离线 .pt 图数据集
        """
        super().__init__()
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"找不到数据集文件: {dataset_path}")
            
        print(f"正在加载内存数据集: {dataset_path} ...")
        self.raw_data_list = torch.load(dataset_path)
        print(f"加载完成！共包含 {len(self.raw_data_list)} 张图。")

    def __len__(self):
        return len(self.raw_data_list)
    
    def __getitem__(self, idx):
        """
        [核心逻辑]：将原始的字典格式，转换为 PyG 的标准 Data 对象
        """
        raw = self.raw_data_list[idx]
        
        num_agents = raw['true_agents_pos'].shape[0]
        num_anchors = raw['anchors_pos'].shape[0]
        num_nodes = num_agents + num_anchors

        # ==========================================
        # 1. 统一节点特征 (Node Features: x)
        # ==========================================
        # Agent 使用初始的粗略估计位置，Anchor 使用绝对准确位置
        pos_agents = raw['init_agents_pos']
        pos_anchors = raw['anchors_pos']
        pos_all = torch.cat([pos_agents, pos_anchors], dim=0) # 形状: (num_nodes, 2)

        # 引入一个 One-hot 标识位，告诉 GNN 谁是 Agent (0), 谁是 Anchor (1)
        node_type = torch.zeros(num_nodes, 1)
        node_type[num_agents:] = 1.0

        # 最终节点特征: [X坐标, Y坐标, 节点类型] 
        x = torch.cat([pos_all, node_type], dim=1) # 形状: (num_nodes, 3)

        # ==========================================
        # 2. 统一边特征 (Edge Attributes: edge_attr)
        # ==========================================
        meas = raw['measurements'].view(-1, 1)
        var = raw['edge_variances'].view(-1, 1)
        is_anchor_edge = raw['is_anchor_edge'].view(-1, 1).float()
        
        # 最终边特征: [物理测距, 测距方差, 是否连向Anchor]
        edge_attr = torch.cat([meas, var, is_anchor_edge], dim=1) # 形状: (E, 3)

        # ==========================================
        # 3. 修复边索引 (Edge Index) -> 绝对索引
        # ==========================================
        edge_index = raw['edge_index'].clone()
        anchor_mask = raw['is_anchor_edge']

        # [关键修复]：如果这条边是连向 Anchor 的，把相对索引(0~3)加上 num_agents，变成绝对索引
        # 例如 Agent 数量为 30，那么 Anchor 的绝对索引就是 30, 31, 32, 33
        edge_index[1, anchor_mask] += num_agents

        # ==========================================
        # 4. 组装 PyG Data 对象
        # ==========================================
        # 把真实的 Agent 位置存入 y 作为监督标签 (计算 CRLB 误差时需要)
        y_true = raw['true_agents_pos']

        data = Data(
            x=x, 
            edge_index=edge_index, 
            edge_attr=edge_attr, 
            y=y_true,
            # 将一些额外的元数据打包进去，方便后续 Loss 计算
            num_agents=num_agents, 
            num_anchors=num_anchors,
            is_nlos_edge=raw.get('is_nlos_edge', torch.zeros(edge_index.shape[1], dtype=torch.bool))
        )

        return data

if __name__ == "__main__":
    # 1. 实例化数据集
    train_dataset = LocalizationDataset("datasets/train_dataset.pt")
    
    # 2. PyG 的 DataLoader
    # 这里设置 batch_size=32，意味着把 32 张图拼成一张大图
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 3. 抓取一个 Batch 看看
    for batch in train_loader:
        print("\n>>> 成功抓取一个 Batch (32 张图组合而成的大图)！")
        print("=" * 60)
        print(batch)
        print("-" * 60)
        print(f"大图的总节点数 (x.shape): {batch.x.shape}")
        print(f"大图的总边数 (edge_index.shape): {batch.edge_index.shape}")
        print(f"大图的边特征维 (edge_attr.shape): {batch.edge_attr.shape}")
        
        # PyG 的精髓：batch 向量 (告诉我们每个节点到底属于哪一张原子图)
        print(f"\nbatch.batch 向量展示 (前 100 个元素): \n{batch.batch[:100]}")
        print("=" * 60)
        break # 只看第一个 batch
