import torch
import torch.nn as nn
import torch.nn.functional as F

# ================= 辅助函数：Gumbel-Sigmoid 直通估计器 =================
def gumbel_sigmoid(logits, tau=1.0, hard=True):
    """
    带直通估计器 (Straight-Through Estimator) 的 Gumbel-Sigmoid
    保证前向传播输出严格的 0 或 1，反向传播拥有平滑梯度。
    """
    # 1. 采样 Gumbel 噪声
    gumbels = -torch.empty_like(logits).exponential_().log()
    
    # 2. 加上噪声并除以温度系数进行平滑松弛
    y_soft = torch.sigmoid((logits + gumbels) / tau)
    
    if hard:
        # 3. 前向传播：硬截断为 0 或 1
        y_hard = (y_soft > 0.5).float()
        # 4. 反向传播：剥离前向的梯度，嫁接 y_soft 的梯度
        y_out = y_hard - y_soft.detach() + y_soft
        return y_out
    else:
        return y_soft

# ================= 核心网络：Edge-Predictor GNN =================
class EdgePredictorGNN(nn.Module):
    def __init__(self, node_in_dim=3, edge_in_dim=2, hidden_dim=32):
        super(EdgePredictorGNN, self).__init__()

        # 1. 特征编码器：将原始物理量映射到高维特征空间
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim)

        # 2. 消息传递层 (Message Passing)
        # 聚合邻居节点特征和边特征来更新自身
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 3. 边评分器 (Edge Scorer)
        # 将两端节点特征与边特征拼接，输出保留这条边的 Logit 分数
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # 输出一个标量 Logit
        )
    
    def forward(self, agents_pos, anchors_pos, edge_index, measurements, edge_variances, is_anchor_edge, tau=1.0):
        N_agents = agents_pos.shape[0]
        N_anchors = anchors_pos.shape[0]
        N_total = N_agents + N_anchors
        E = edge_index.shape[1]

        # --- A. 准备节点特征 ---
        # 节点特征：[x, y, is_anchor]。Agent 为 0，Anchor 为 1
        agent_features = torch.cat([agents_pos, torch.zeros(N_agents, 1)], dim=1)
        anchor_features = torch.cat([anchors_pos, torch.ones(N_anchors, 1)], dim=1)
        # 将 Agent 和 Anchor 的特征拼在一起，索引 0~N_agents-1 是 Agent
        x = torch.cat([agent_features, anchor_features], dim=0)
        h = self.node_encoder(x) # 形状: (N_total, hidden_dim)

        # --- B. 准备边特征 ---
        # 边特征：[测量距离, 测量方差]
        edge_attr = torch.stack([measurements, edge_variances], dim=1)
        e = self.edge_encoder(edge_attr) # 形状: (E, hidden_dim)

        # --- C. 消息传递 (进行 1 次迭代示例，可按需循环多次) ---
        idx_u = edge_index[0] # 发送方 (必定是 Agent)

        # 接收方的真实索引：如果是 Anchor 边，其索引需要加上 N_agents 的偏移量
        idx_v = edge_index[1].clone()
        idx_v[is_anchor_edge] += N_agents

        # 获取两端节点的隐藏状态
        h_u = h[idx_u]
        h_v = h[idx_v]

        # 计算消息 m_{u->v}
        msg_input = torch.cat([h_u, h_v, e], dim=1)
        msg = self.msg_mlp(msg_input) # 形状: (E, hidden_dim)

        # 聚合消息 (Scatter Add)：把所有的 msg 累加到接收节点 idx_u 上
        aggr_msg = torch.zeros(N_total, msg.shape[1])
        aggr_msg.scatter_add_(0, idx_u.unsqueeze(1).expand(-1, msg.shape[1]), msg)

        # 节点状态更新
        update_input = torch.cat([h, aggr_msg], dim=1)
        h_new = self.update_mlp(update_input)

        # --- D. 边评分与 Gumbel 采样 ---
        # 用更新后的节点特征再次获取两端状态
        h_u_new = h_new[idx_u]
        h_v_new = h_new[idx_v]
        
        # 评分
        score_input = torch.cat([h_u_new, h_v_new, e], dim=1)
        logits = self.edge_scorer(score_input).squeeze() # 形状: (E,)
        
        # Gumbel-Sigmoid 重参数化采样
        # 前向输出 0 或 1，反向保留平滑梯度
        edge_weights = gumbel_sigmoid(logits, tau=tau, hard=True)
        
        return edge_weights, logits