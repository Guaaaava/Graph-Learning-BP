"""
Edge-Predictor GNN — 图拓扑优化网络

输入: 协同定位图 (节点=agents+anchors, 边=测距链路的全连接子图)
输出: 每条边的保留概率 p_ij ∈ {0, 1} (通过 Gumbel-Sigmoid STE 离散化)

节点特征 (5 维):
  [pos_x, pos_y, σ²_x, σ²_y, is_anchor]
边特征 (4 维):
  [测距值 z, 测距方差 σ², 伪距残差 r, 是否 anchor 边]

架构: 多层 MessagePassing + 残差连接 + 边评分器 + Gumbel-Sigmoid STE
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


# ================= Gumbel-Sigmoid 直通估计器 =================
def gumbel_sigmoid(logits, tau=1.0, hard=True, training=True):
    """
    Gumbel-Sigmoid 重参数化 + 直通估计器 (STE)

    前向: 输出严格 0/1 (hard=True) 或连续值 (hard=False)
    反向: 梯度流过平滑 sigmoid, 实现可微的离散采样

    Parameters
    ----------
    logits : Tensor (E,)
        边评分器输出的原始 logits
    tau : float
        温度参数: τ大→探索性强(soft), τ小→趋近离散(hard)
    hard : bool
        True 则前向输出二值, False 则输出连续概率
    training : bool
        训练模式加入 Gumbel 噪声, 推断模式关闭
    """
    if training:
        # Gumbel(0,1) 噪声: -log(-log(u)), u ~ Uniform(0,1)
        gumbels = -torch.empty_like(logits).exponential_().log()
        y_soft = torch.sigmoid((logits + gumbels) / tau)
    else:
        y_soft = torch.sigmoid(logits / tau)

    if hard:
        y_hard = (y_soft > 0.5).float()
        # STE: 前向 = hard, 反向梯度 = soft 的梯度
        y_out = y_hard - y_soft.detach() + y_soft
        return y_out
    else:
        return y_soft


# ================= 消息传递层 =================
class GNNLayer(MessagePassing):
    """单层消息传递: m_ij = MLP_msg(h_i || h_j || e_ij), h_i' = MLP_upd(h_i || Σ m_ji)"""

    def __init__(self, hidden_dim):
        super().__init__(aggr='add')

        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.upd_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=1))

    def update(self, aggr_out, x):
        return self.upd_mlp(torch.cat([x, aggr_out], dim=1))


# ================= 主网络 =================
class EdgePredictorGNN(nn.Module):
    """
    图拓扑边预测网络

    Parameters
    ----------
    node_in_dim : int
        节点输入特征维度 (默认 5)
    edge_in_dim : int
        边输入特征维度 (默认 4)
    hidden_dim : int
        隐层维度
    num_layers : int
        消息传递层数 (扩大感受野)
    """

    def __init__(self, node_in_dim=5, edge_in_dim=4, hidden_dim=64, num_layers=3):
        super().__init__()

        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GNNLayer(hidden_dim))

        # 边评分器: h_u || h_v || e → logit
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data, tau=1.0, hard=True):
        """
        Parameters
        ----------
        data : PyG Data / Batch
        tau : float
            Gumbel-Sigmoid 温度
        hard : bool
            是否输出二值掩码

        Returns
        -------
        edge_weights : Tensor (E,)
            0/1 硬掩码 (hard=True) 或连续概率 (hard=False)
        logits : Tensor (E,)
            评分器原始输出 (用于调试/分析)
        """
        x = self.node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr)

        # 多层消息传递 + 残差连接
        for layer in self.layers:
            x_new = layer(x, data.edge_index, e)
            x = x + x_new

        # 边评分
        row, col = data.edge_index
        h_u = x[row]
        h_v = x[col]
        score_input = torch.cat([h_u, h_v, e], dim=1)
        logits = self.edge_scorer(score_input).squeeze(-1)

        edge_weights = gumbel_sigmoid(logits, tau=tau, hard=hard, training=self.training)

        return edge_weights, logits
