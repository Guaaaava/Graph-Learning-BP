import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.linalg import inv, norm

# ==========================================
# 1. 基础数学工具
# ==========================================
def get_inverse(matrix):
    try:
        return inv(matrix)
    except np.linalg.LinAlgError:
        return inv(matrix + np.eye(matrix.shape[0]) * 1e-6)

def product_of_gaussians(means, covs):
    if len(means) == 0:
        return None, None
    lambda_sum = np.zeros_like(covs[0], dtype=float)
    eta_sum = np.zeros_like(means[0], dtype=float)
    for mu, sigma in zip(means, covs):
        lam = get_inverse(sigma)
        eta = lam @ mu
        lambda_sum += lam
        eta_sum += eta
    sigma_new = get_inverse(lambda_sum)
    mu_new = sigma_new @ eta_sum
    return mu_new, sigma_new

# ==========================================
# 2. 节点类 (支持指定初始猜测)
# ==========================================
class Node:
    def __init__(self, node_id, true_pos, is_anchor=False, initial_guess=None):
        self.id = node_id
        self.true_pos = np.array(true_pos, dtype=float)
        self.is_anchor = is_anchor
        self.incoming_messages = {}
        self.neighbors = []

        if is_anchor:
            self.mu = self.true_pos
            self.sigma_prior = np.eye(2) * 1e-8
            self.sigma = np.zeros((2, 2))
        else:
            # 关键：使用传入的 initial_guess，确保两张图的起点完全一样
            self.mu = np.array(initial_guess, dtype=float) if initial_guess is not None else np.random.rand(2)*100
            self.sigma_prior = np.eye(2) * 10000
            self.sigma = self.sigma_prior.copy()

    def update_belief(self):
        if self.is_anchor: return
        means = [self.mu]; covs = [self.sigma_prior]
        for msg in self.incoming_messages.values():
            means.append(msg[0]); covs.append(msg[1])
        self.mu, self.sigma = product_of_gaussians(means, covs)

    def compute_outgoing_message(self, target_neighbor_id):
        means = [self.mu]; covs = [self.sigma_prior]
        for nid, msg in self.incoming_messages.items():
            if nid != target_neighbor_id:
                means.append(msg[0]); covs.append(msg[1])
        return product_of_gaussians(means, covs)

# ==========================================
# 3. 边类 (支持模式切换)
# ==========================================
class Edge:
    def __init__(self, node_a, node_b, measurement, noise_std, mode='simple'):
        self.node_a = node_a
        self.node_b = node_b
        self.measurement = measurement
        self.noise_var = noise_std**2
        self.mode = mode 
        self.R_simple = np.eye(2) * self.noise_var

    def linearize_and_pass_message(self, source_node, target_node, msg_in):
        mu_in, sigma_in = msg_in
        diff = target_node.mu - source_node.mu
        dist_est = norm(diff)
        if dist_est < 1e-3: dist_est = 1e-3
        u_vec = diff / dist_est
        
        mu_out = mu_in + u_vec * self.measurement
        
        if self.mode == 'simple':
            # 简化版：圆形噪声
            sigma_out = sigma_in + self.R_simple
        elif self.mode == 'rigorous':
            # 严谨版：椭圆噪声 (投影)
            angle = np.arctan2(u_vec[1], u_vec[0])
            c, s = np.cos(angle), np.sin(angle)
            Rot = np.array([[c, -s], [s, c]])
            # 径向精准，切向模糊
            R_local = np.array([[self.noise_var, 0], [0, 500]]) 
            R_global = Rot @ R_local @ Rot.T
            sigma_out = sigma_in + R_global
            
        return mu_out, sigma_out

# ==========================================
# 4. 核心逻辑：执行一步仿真
# ==========================================
def run_one_step_logic(nodes, edges):
    """辅助函数：让某个图跑一步，并返回位置列表和当前RMSE"""
    # 1. 计算消息
    new_msgs = []
    for edge in edges:
        na, nb = edge.node_a, edge.node_b
        # A -> B
        ma, ca = na.compute_outgoing_message(nb.id)
        mu_ab, sig_ab = edge.linearize_and_pass_message(na, nb, (ma, ca))
        new_msgs.append((nb.id, na.id, mu_ab, sig_ab))
        # B -> A
        mb, cb = nb.compute_outgoing_message(na.id)
        mu_ba, sig_ba = edge.linearize_and_pass_message(nb, na, (mb, cb))
        new_msgs.append((na.id, nb.id, mu_ba, sig_ba))
    
    # 2. 投递消息
    for tid, sid, m, s in new_msgs:
        target = next(n for n in nodes if n.id == tid)
        target.incoming_messages[sid] = (m, s)
        
    # 3. 更新 Belief
    positions = []
    sq_err_sum = 0
    count = 0
    for n in nodes:
        n.update_belief()
        positions.append(n.mu)
        if not n.is_anchor:
            sq_err_sum += norm(n.mu - n.true_pos)**2
            count += 1
    
    current_rmse = np.sqrt(sq_err_sum/count) if count > 0 else 0
    return positions, current_rmse

# ==========================================
# 5. 主程序：三联屏动画
# ==========================================
def run_visual_comparison():
    # --- 参数 ---
    AREA_SIZE = 100
    NUM_ANCHORS = 4
    NUM_AGENTS = 20
    COMM_RANGE = 65
    NOISE_STD = 0.5
    ITERATIONS = 40
    
    # --- 1. 生成上帝视角数据 (保证公平) ---
    np.random.seed(10) # 固定种子方便复现
    anchors_pos = [[0,0], [AREA_SIZE,0], [AREA_SIZE,AREA_SIZE], [0,AREA_SIZE]]
    agents_true_pos = [np.random.rand(2) * AREA_SIZE for _ in range(NUM_AGENTS)]
    agents_init_guess = [np.random.rand(2) * AREA_SIZE for _ in range(NUM_AGENTS)]
    
    # 生成连接
    connectivity = []
    all_true_pos = anchors_pos + agents_true_pos
    for i in range(len(all_true_pos)):
        for j in range(i+1, len(all_true_pos)):
            dist = norm(np.array(all_true_pos[i]) - np.array(all_true_pos[j]))
            if dist < COMM_RANGE:
                noise = np.random.normal(0, NOISE_STD)
                connectivity.append((i, j, dist + noise))

    # --- 2. 构建两张图 ---
    def build_graph(mode_name):
        nodes = []
        for i in range(NUM_ANCHORS):
            nodes.append(Node(i, anchors_pos[i], is_anchor=True))
        for i in range(NUM_AGENTS):
            nodes.append(Node(NUM_ANCHORS+i, agents_true_pos[i], is_anchor=False, initial_guess=agents_init_guess[i]))
        edges = []
        for (i, j, meas) in connectivity:
            edge = Edge(nodes[i], nodes[j], meas, NOISE_STD, mode=mode_name)
            edges.append(edge)
            nodes[i].neighbors.append(nodes[j].id)
            nodes[j].neighbors.append(nodes[i].id)
            nodes[i].incoming_messages[nodes[j].id] = (np.zeros(2), np.eye(2)*1e5)
            nodes[j].incoming_messages[nodes[i].id] = (np.zeros(2), np.eye(2)*1e5)
        return nodes, edges

    nodes_simp, edges_simp = build_graph('simple')
    nodes_rig, edges_rig = build_graph('rigorous')

    # --- 3. 设置三联屏画布 ---
    # figsize=(18, 6) 宽屏显示
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 左图：Simple
    ax1.set_title("Method A: Simplified (Isotropic)")
    ax1.set_xlim(-10, AREA_SIZE+10); ax1.set_ylim(-10, AREA_SIZE+10)
    ax1.grid(True)
    scat_true_1 = ax1.scatter([n.true_pos[0] for n in nodes_simp], [n.true_pos[1] for n in nodes_simp], c='g', marker='x', s=60, label='True')
    scat_est_1 = ax1.scatter([], [], c='r', marker='o', label='Est Simple') # 红色代表简化
    lines_1 = [ax1.plot([], [], 'k-', alpha=0.1)[0] for _ in edges_simp]
    ax1.legend(loc='lower left')

    # 右图：Rigorous
    ax3.set_title("Method B: Rigorous (Geometric)")
    ax3.set_xlim(-10, AREA_SIZE+10); ax3.set_ylim(-10, AREA_SIZE+10)
    ax3.grid(True)
    scat_true_3 = ax3.scatter([n.true_pos[0] for n in nodes_rig], [n.true_pos[1] for n in nodes_rig], c='g', marker='x', s=60, label='True')
    scat_est_3 = ax3.scatter([], [], c='b', marker='o', label='Est Rigorous') # 蓝色代表严谨
    lines_3 = [ax3.plot([], [], 'k-', alpha=0.1)[0] for _ in edges_rig]
    ax3.legend(loc='lower left')

    # 中图：RMSE 曲线
    ax2.set_title("RMSE Comparison")
    ax2.set_xlim(0, ITERATIONS)
    ax2.set_ylim(0, AREA_SIZE/2)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("RMSE (m)")
    ax2.grid(True)
    
    line_rmse_s, = ax2.plot([], [], 'r--o', label='Simplified') # 红色虚线
    line_rmse_r, = ax2.plot([], [], 'b-s', label='Rigorous')   # 蓝色实线
    ax2.legend()

    # 数据记录
    hist_rmse_s = []
    hist_rmse_r = []
    hist_iter = []

    # --- 4. 动画更新函数 ---
    def update(frame):
        # 跑一步 Simple
        pos_s, rmse_s = run_one_step_logic(nodes_simp, edges_simp)
        # 跑一步 Rigorous
        pos_r, rmse_r = run_one_step_logic(nodes_rig, edges_rig)
        
        # 更新数据记录
        hist_rmse_s.append(rmse_s)
        hist_rmse_r.append(rmse_r)
        hist_iter.append(frame)
        
        # --- 更新左图 (Simple) ---
        pos_s_arr = np.array(pos_s)
        scat_est_1.set_offsets(pos_s_arr)
        for i, edge in enumerate(edges_simp):
            p1 = edge.node_a.mu; p2 = edge.node_b.mu
            lines_1[i].set_data([p1[0], p2[0]], [p1[1], p2[1]])
            
        # --- 更新右图 (Rigorous) ---
        pos_r_arr = np.array(pos_r)
        scat_est_3.set_offsets(pos_r_arr)
        for i, edge in enumerate(edges_rig):
            p1 = edge.node_a.mu; p2 = edge.node_b.mu
            lines_3[i].set_data([p1[0], p2[0]], [p1[1], p2[1]])
            
        # --- 更新中图 (RMSE) ---
        line_rmse_s.set_data(hist_iter, hist_rmse_s)
        line_rmse_r.set_data(hist_iter, hist_rmse_r)
        
        # 自动调整Y轴
        if frame > 1:
            max_val = max(max(hist_rmse_s), max(hist_rmse_r))
            ax2.set_ylim(0, max_val * 1.1)
            
        print(f"Iter {frame}: Simple={rmse_s:.2f}m, Rigorous={rmse_r:.2f}m")
        
        return scat_est_1, scat_est_3, line_rmse_s, line_rmse_r, *lines_1, *lines_3

    ani = animation.FuncAnimation(fig, update, frames=ITERATIONS, interval=300, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_visual_comparison()