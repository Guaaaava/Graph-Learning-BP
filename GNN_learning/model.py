import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

# ================= Gumbel-Sigmoid 直通估计器 =================
def gumbel_sigmoid(logits, tau=1.0, hard=True, training=True):
    if training:
        # 训练阶段：采样 Gumbel 噪声以进行探索
        gumbels = -torch.empty_like(logits).exponential_().log()
        y_soft = torch.sigmoid((logits + gumbels) / tau)
    else:
        # 推断阶段：不需要噪声，直接过 Sigmoid
        y_soft = torch.sigmoid(logits)
        
    if hard:
        # 前向硬截断，反向软梯度 (STE 魔法)
        y_hard = (y_soft > 0.5).float()
        y_out = y_hard - y_soft.detach() + y_soft
        return y_out
    else:
        return y_soft

class GNNLayer(MessagePassing):
    def __init__(self, hidden_dim):
        super(GNNLayer, self).__init__(aggr='add')
        
        # 消息计算 MLP (Message Function)
        # 输入: 源节点特征 + 目标节点特征 + 边特征
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 节点状态更新 MLP (Update Function)
        # 输入: 节点原本特征 + 聚合后的邻居消息
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        # 开始消息传递流程，会自动调用 message() 和 update()
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """
        构建消息: x_i 是目标节点, x_j 是源节点 (信息从 j 流向 i)
        将它们与物理边特征拼在一起，通过 MLP 提取高维特征
        """
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.msg_mlp(msg_input)

    def update(self, aggr_out, x):
        """
        更新节点: 将自己的特征 x 与汇总来的邻居信息 aggr_out 融合
        """
        update_input = torch.cat([x, aggr_out], dim=1)
        return self.update_mlp(update_input)


class EdgePredictorGNN(nn.Module):
    def __init__(self, node_in_dim=3, edge_in_dim=3, hidden_dim=32, num_layers=1):
        super(EdgePredictorGNN, self).__init__()

        # 1. 初始特征编码器 (将低维物理特征映射到高维潜空间)
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim)
        
        # 2. 堆叠多层消息传递网络 (扩大感受野)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GNNLayer(hidden_dim))
            
        # 3. 边评分器
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # 输出标量 Logit
        )

    def forward(self, data, tau=1.0, hard=True):
        # --- A & B. 准备节点与边特征 ---
        x = self.node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr)
        
        # --- C. 消息传递 ---
        for layer in self.layers:
            x_new = layer(x, data.edge_index, e)
            x = x + x_new # 加入残差，防止多层网络梯度消失

        # --- D. 边评分与 Gumbel 采样 ---
        row, col = data.edge_index
        h_u_new = x[row]
        h_v_new = x[col]
        
        score_input = torch.cat([h_u_new, h_v_new, e], dim=1)
        logits = self.edge_scorer(score_input).squeeze() # 形状: (E,)
        
        edge_weights = gumbel_sigmoid(logits, tau=tau, hard=hard, training=self.training)
        
        return edge_weights, logits

if __name__ == "__main__":
    from dataset import LocalizationDataset
    from torch_geometric.loader import DataLoader
    
    # 初始化模型
    model = EdgePredictorGNN(node_in_dim=3, edge_in_dim=3, hidden_dim=32, num_layers=3)
    
    # 加载大图 Batch
    dataset = LocalizationDataset("datasets/train_dataset.pt")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    
    print("\n>>> 开始进行大图前向传播测试...")
    
    # 只需要这一行，GNN 就能在几毫秒内并行处理 32 张图上的数千条边！
    edge_weights, logits = model(batch, tau=1.0, hard=True)
    
    print("=" * 60)
    print(f"大图包含的原子图数量: {batch.num_graphs}")
    print(f"大图的总节点数: {batch.x.shape[0]}")
    print(f"大图的总边数: {batch.edge_index.shape[1]}")
    print("-" * 60)
    print(f"模型输出 掩码 (edge_weights) 形状: {edge_weights.shape}")
    print(f"模型输出 原始打分 (logits) 形状: {logits.shape}")
    print(f"掩码的取值范围: 最小 {edge_weights.min().item():.4f}, 最大 {edge_weights.max().item():.4f}")
    
    # 验证 STE 直通估计器是否起效
    unique_vals = torch.unique(edge_weights)
    if len(unique_vals) <= 2 and 0.0 in unique_vals and 1.0 in unique_vals:
        print(">>> 验证成功：当前处于绝对的 [硬剪枝 (Hard)] 模式，输出仅包含 0.0 和 1.0")
    else:
        print(">>> 提示：当前处于 [软掩码 (Soft)] 模式，输出为连续概率")
    print("=" * 60)