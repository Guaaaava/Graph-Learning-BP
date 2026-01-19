import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.linalg import inv, norm


# ==========================================
# 1. 基础数学工具函数
# ==========================================

def get_inverse(matrix):
    """求逆矩阵，增加微小扰动以防止奇异矩阵 (Singular Matrix)"""
    try:
        return inv(matrix)
    except np.linalg.LinAlgError:
        # 如果矩阵不可逆，加一个微小的对角阵（正则化）
        return inv(matrix + np.eye(matrix.shape[0]) * 1e-6)

def product_of_gaussians(means, covs):
    """
    高斯乘积公式 (Product of Gaussians)
    输入：一堆均值列表 [mu1, mu2...] 和 协方差列表 [sigma1, sigma2...]
    输出：融合后的新高斯分布 (mu_new, sigma_new)
    原理：信息矩阵相加
    """
    if len(means) == 0:
        return None, None
    
    # 转换为信息形式 (Information Form)
    # 信息矩阵 Lambda = Sigma^-1
    # 信息向量 eta = Sigma^-1 * mu
    lambda_sum = np.zeros_like(covs[0], dtype=float)
    eta_sum = np.zeros_like(means[0], dtype=float)

    for mu, sigma in zip(means, covs):
        lam = get_inverse(sigma)
        eta = lam @ mu
        lambda_sum += lam
        eta_sum += eta

    # 转换回标准形式
    sigma_new = get_inverse(lambda_sum)
    mu_new = sigma_new @ eta_sum
    return mu_new, sigma_new

# ==========================================
# 2. 节点类 (Variable Node)
# ==========================================
class Node:
    def __init__(self, node_id, true_pos, is_anchor=False):
        self.id = node_id
        self.true_pos = np.array(true_pos, dtype=float)
        self.is_anchor = is_anchor

        # --- 初始化 Belief (置信度) ---
        if is_anchor:
            self.mu = self.true_pos
            self.sigma = np.zeros((2, 2)) # 锚点无误差
            # 为了计算方便，给锚点一个极小的协方差，避免除以0
            self.sigma_prior = np.eye(2) * 1e-8
        else:
            # 未知节点：随机初始化位置，并给予极大的初始协方差（表示完全不知道在哪）
            self.mu = np.random.rand(2) * 50 # 随机分布在 50x50 区域内
            self.sigma_prior = np.eye(2) * 10000 # 初始不确定性极大
            self.sigma = self.sigma_prior.copy()

        # 收件箱：存储邻居发来的消息 {neighbor_id: (mu_msg, sigma_msg)}
        self.incoming_messages = {}
        self.neighbors = []

    def get_belief(self):
        return self.mu, self.sigma
    
    def compute_outgoing_message(self, target_neighbor_id):
        """
        计算发给特定邻居的消息 (Sum-Product Rule)
        规则：融合除了 target_neighbor 以外的所有消息 + 自身的先验信息
        """
        means = []
        covs = []

        # 1. 加入自身的先验 (Prior)
        # 对于锚点，这是强约束；对于未知节点，这是弱约束
        means.append(self.mu if self.is_anchor else self.mu) # 注意：这里简化处理，实际应该存一个初始prior
        covs.append(self.sigma_prior)

        # 2. 加入其他邻居的消息 (排除 target_neighbor_id)
        for nid, msg in self.incoming_messages.items():
            if nid != target_neighbor_id:
                means.append(msg[0])
                covs.append(msg[1])
        
        # 3. 高斯乘积融合
        mu_out, sigma_out = product_of_gaussians(means, covs)
        return mu_out, sigma_out
    
    def update_belief(self):
        """
        一轮迭代结束后，融合所有消息更新自己的位置
        """
        if self.is_anchor:
            return # 锚点不更新
        
        means = []
        covs = []
        # 加入先验
        means.append(self.mu) # 这里简化，实际应使用上一时刻的Belief作为这一时刻的Prior
        covs.append(self.sigma_prior) # 保持先验极弱

        # 加入所有收到的消息
        for msg in self.incoming_messages.values():
            means.append(msg[0])
            covs.append(msg[1])
            
        mu_new, sigma_new = product_of_gaussians(means, covs)
        
        # 更新状态
        self.mu = mu_new
        self.sigma = sigma_new

# ==========================================
# 3. 边类 (Factor Node 逻辑)
# ==========================================
class Edge:
    def __init__(self, node_a, node_b, measurement, noise_std):
        self.node_a = node_a
        self.node_b = node_b
        self.measurement = measurement # 测距值 distance
        self.R = np.eye(2) * (noise_std**2) # 测距噪声协方差 (简化为对角阵)
    
    def linearize_and_pass_message(self, source_node, target_node, msg_in):
        """
        核心难点：通过非线性测距约束传递消息
        从 Source -> Target
        """
        mu_in, sigma_in = msg_in
        
        # 获取当前线性化点 (Linearization Point)
        # 我们使用当前估计的位置来计算梯度方向
        pos_source = source_node.mu
        pos_target = target_node.mu
        
        dist_est = norm(pos_source - pos_target)
        
        # 防止重合导致除以0
        if dist_est < 1e-3:
            dist_est = 1e-3

        # 1. 计算单位方向向量 (Unit Vector)
        # 指向：从 Source 指向 Target
        u_vec = (pos_target - pos_source) / dist_est
        
        # 2. 预测 Target 的均值 (Mean Prediction)
        # 含义：Target 应该在 Source 沿 u_vec 方向距离 measurement 处
        mu_out = mu_in + u_vec * self.measurement
        
        # 3. 预测 Target 的协方差 (Covariance Prediction)
        # 这里使用简化的线性化模型：
        # 新协方差 = 原协方差 + 测距噪声 (沿着连线方向)
        # 这是一个近似，严谨做法需要雅可比矩阵 J * Sigma * J.T
        
        # 构建旋转矩阵，将测距噪声对齐到连线方向
        angle = np.arctan2(u_vec[1], u_vec[0])
        c, s = np.cos(angle), np.sin(angle)
        rot_mat = np.array([[c, -s], [s, c]])
        
        # 噪声在径向(radial)上是 R，在切向(tangential)上如果不确定，通常设得大一点
        # 这里简化：直接叠加各向同性的噪声 R
        sigma_out = sigma_in + self.R
        
        return mu_out, sigma_out

# ==========================================
# 4. 主程序 & 仿真环境（含误差分析图）
# ==========================================

def run_simulation():
    # --- 配置参数 ---
    AREA_SIZE = 50
    NUM_ANCHORS = 4
    NUM_AGENTS = 5
    COMM_RANGE = 35 # 通信半径
    NOISE_STD = 0.5 # 测距噪声标准差
    ITERATIONS = 50 # 迭代次数

    nodes = []
    edges = []

    # 1. 生成锚点 (固定在四个角)
    anchors_pos = [
        [0, 0], [AREA_SIZE, 0], [AREA_SIZE, AREA_SIZE], [0, AREA_SIZE]
    ]
    for i in range(NUM_ANCHORS):
        nodes.append(Node(i, anchors_pos[i], is_anchor=True))

    # 2. 生成未知节点 (随机位置)
    true_agent_positions = []
    for i in range(NUM_AGENTS):
        # 真实位置随机
        true_pos = np.random.rand(2) * AREA_SIZE
        true_agent_positions.append(true_pos)
        nodes.append(Node(NUM_ANCHORS + i, true_pos, is_anchor=False))

    # 3. 生成边 (基于真实距离 + 噪声)
    print("Building Factor Graph...")
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            dist = norm(nodes[i].true_pos - nodes[j].true_pos)
            if dist < COMM_RANGE:
                # 添加噪声的测量值
                measured_dist = dist + np.random.normal(0, NOISE_STD)
                edge = Edge(nodes[i], nodes[j], measured_dist, NOISE_STD)
                edges.append(edge)
                
                # 注册邻居关系
                nodes[i].neighbors.append(nodes[j].id)
                nodes[j].neighbors.append(nodes[i].id)
                
                # 初始化空的收件箱消息，防止第一轮报错
                # 初始消息设为无信息 (均值0，协方差无穷大)
                nodes[i].incoming_messages[nodes[j].id] = (np.zeros(2), np.eye(2) * 1e5)
                nodes[j].incoming_messages[nodes[i].id] = (np.zeros(2), np.eye(2) * 1e5)

    print(f"Graph Created: {len(nodes)} nodes, {len(edges)} edges.")

    # --- 准备绘图 ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-10, AREA_SIZE + 10)
    ax.set_ylim(-10, AREA_SIZE + 10)
    
    # 绘图元素
    scat_true = ax.scatter([n.true_pos[0] for n in nodes], [n.true_pos[1] for n in nodes], c='g', marker='x', label='True Pos')
    scat_est = ax.scatter([], [], c='r', marker='o', label='Est BP')
    lines = [] # 绘制边

    # 绘制连接线
    for edge in edges:
        ln, = ax.plot([], [], 'k-', alpha=0.1)
        lines.append(ln)

    # --- BP 迭代主循环 ---
    def update(frame):
        print(f"Iteration {frame}...")
        
        # 步骤 A: 计算所有消息 (Variable -> Factor -> Variable)
        # 为了模拟同步BP，我们需要先计算所有边上的新消息，暂存起来，然后再统一放入收件箱
        new_messages = [] # list of (target_node_id, source_node_id, message)

        for edge in edges:
            node_a = edge.node_a
            node_b = edge.node_b

            # 1. 计算 A -> B 的消息
            # a. Node A 聚合除了 B 以外的信息
            msg_a_internal, cov_a_internal = node_a.compute_outgoing_message(node_b.id)
            # b. 通过 Edge (非线性测距) 转换
            mu_a_to_b, sigma_a_to_b = edge.linearize_and_pass_message(node_a, node_b, (msg_a_internal, cov_a_internal))
            new_messages.append((node_b.id, node_a.id, mu_a_to_b, sigma_a_to_b))

            # 2. 计算 B -> A 的消息
            msg_b_internal, cov_b_internal = node_b.compute_outgoing_message(node_a.id)
            mu_b_to_a, sigma_b_to_a = edge.linearize_and_pass_message(node_b, node_a, (msg_b_internal, cov_b_internal))
            new_messages.append((node_a.id, node_b.id, mu_b_to_a, sigma_b_to_a))

        # 步骤 B: 投递消息 (更新收件箱)
        for target_id, source_id, mu, sigma in new_messages:
            # 找到目标节点对象
            target_node = next(n for n in nodes if n.id == target_id)
            target_node.incoming_messages[source_id] = (mu, sigma)

        # 步骤 C: 节点更新自己的 Belief
        est_positions = []
        for node in nodes:
            node.update_belief()
            est_positions.append(node.mu)

        # --- 更新绘图数据 ---
        est_positions = np.array(est_positions)
        scat_est.set_offsets(est_positions)
        
        # 更新连接线位置 (基于估计位置)
        for i, edge in enumerate(edges):
            p1 = edge.node_a.mu
            p2 = edge.node_b.mu
            lines[i].set_data([p1[0], p2[0]], [p1[1], p2[1]])
            
        ax.set_title(f"BP Cooperative Localization - Iteration {frame}")
        return scat_est, *lines
    
    ani = animation.FuncAnimation(fig, update, frames=ITERATIONS, interval=500, blit=False, repeat=False)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_simulation()