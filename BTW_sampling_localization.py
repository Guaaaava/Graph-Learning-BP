import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, norm
import copy

# ==========================================
# 1. 基础数学工具函数 (Math Utils)
# ==========================================

def get_inverse(matrix):
    """求逆矩阵，增加微小扰动以防止奇异矩阵"""
    try:
        return inv(matrix)
    except np.linalg.LinAlgError:
        return inv(matrix + np.eye(matrix.shape[0]) * 1e-6)

def product_of_gaussians(means, covs):
    """高斯乘积公式 (Product of Gaussians)"""
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
# 2. 核心类：Node 和 Edge (Rigorous Version)
# ==========================================

class Node:
    def __init__(self, node_id, true_pos, is_anchor=False, initial_guess=None):
        self.id = node_id
        self.true_pos = np.array(true_pos, dtype=float)
        self.is_anchor = is_anchor

        # Belief 初始化
        if is_anchor:
            self.mu = self.true_pos
            self.sigma = np.zeros((2, 2))
            self.sigma_prior = np.eye(2) * 1e-8
        else:
            # 如果有初始猜测则使用，否则随机
            if initial_guess is not None:
                self.mu = np.array(initial_guess, dtype=float)
            else:
                self.mu = np.random.rand(2) * 100 
            
            self.sigma_prior = np.eye(2) * 10000 
            self.sigma = self.sigma_prior.copy()

        self.incoming_messages = {}
        self.neighbors = [] # 存储邻居ID

    def compute_outgoing_message(self, target_neighbor_id):
        means = []
        covs = []
        # 1. 自身先验
        means.append(self.mu if self.is_anchor else self.mu) 
        covs.append(self.sigma_prior)
        # 2. 其他邻居的消息
        for nid, msg in self.incoming_messages.items():
            if nid != target_neighbor_id:
                means.append(msg[0])
                covs.append(msg[1])
        return product_of_gaussians(means, covs)

    def update_belief(self):
        if self.is_anchor: return
        means = [self.mu]
        covs = [self.sigma_prior]
        for msg in self.incoming_messages.values():
            means.append(msg[0])
            covs.append(msg[1])
        self.mu, self.sigma = product_of_gaussians(means, covs)

class Edge:
    def __init__(self, node_a, node_b, measurement, noise_std):
        self.node_a = node_a
        self.node_b = node_b
        self.measurement = measurement 
        self.noise_var = noise_std**2 
    
    def linearize_and_pass_message(self, source_node, target_node, msg_in):
        """严谨版：基于几何投影的消息传递"""
        mu_in, sigma_in = msg_in
        
        pos_source = source_node.mu
        pos_target = target_node.mu
        
        diff = pos_target - pos_source
        dist_est = norm(diff)
        if dist_est < 1e-3: dist_est = 1e-3

        u_vec = diff / dist_est
        mu_out = mu_in + u_vec * self.measurement
        
        # 几何投影噪声矩阵
        angle = np.arctan2(u_vec[1], u_vec[0])
        c, s = np.cos(angle), np.sin(angle)
        Rot = np.array([[c, -s], [s, c]])
        
        # 径向精准，切向模糊
        R_local = np.array([
            [self.noise_var, 0],
            [0, 1e4] 
        ])
        
        R_global = Rot @ R_local @ Rot.T
        sigma_out = sigma_in + R_global
        
        return mu_out, sigma_out

# ==========================================
# 3. BTW 图采样器 (BTW Graph Sampler)
# ==========================================

class BTWGraphSampler:
    def __init__(self, original_nodes, edges_with_weights, k=2):
        """
        original_nodes: 节点对象列表
        edges_with_weights: 列表 [(u_id, v_id, weight), ...]
        k: 目标树宽
        """
        self.nodes = original_nodes
        self.node_ids = [n.id for n in original_nodes]
        self.edges_map = self._build_adj(edges_with_weights) 
        self.k = k
        
    def _build_adj(self, edges):
        adj = {nid: {} for nid in self.node_ids}
        for u, v, w in edges:
            if u in adj and v in adj: # 确保节点存在
                adj[u][v] = w
                adj[v][u] = w
        return adj

    def compute_score(self, node_id, clique):
        score = 0
        for existing_node_id in clique:
            if existing_node_id in self.edges_map[node_id]:
                score += self.edges_map[node_id][existing_node_id]
        # Tie-breaker
        score += np.random.uniform(0, 1e-5)
        return score

    def sample_subgraph(self):
        sampled_edges = [] # 存储 (u_id, v_id)
        visited_nodes = set()
        
        # 1. 初始化：优先选锚点作为种子，保证连通性
        anchors = [n.id for n in self.nodes if n.is_anchor]
        if len(anchors) >= self.k + 1:
            seed_nodes = anchors[:self.k+1]
        else:
            seed_nodes = anchors + [n.id for n in self.nodes if not n.is_anchor][:self.k+1-len(anchors)]
            
        visited_nodes.update(seed_nodes)
        
        # 将种子 clique 内存在的边加入
        initial_clique = list(seed_nodes)
        self._add_existing_edges(initial_clique, sampled_edges)
        
        active_cliques = [initial_clique]
        remaining_nodes = set(self.node_ids) - visited_nodes
        
        # 2. 贪婪扩展
        while remaining_nodes:
            best_score = -1
            best_node = None
            best_clique = None
            
            # Naive search (可优化为 Heap)
            for node_id in remaining_nodes:
                for clique in active_cliques:
                    score = self.compute_score(node_id, clique)
                    if score > best_score:
                        best_score = score
                        best_node = node_id
                        best_clique = clique
            
            if best_node is None:
                # 理论上全连通图不会进这里，但如果是非连通图可能发生
                if remaining_nodes:
                    # 强制取一个剩余节点
                    best_node = list(remaining_nodes)[0]
                    best_clique = active_cliques[0]
                else:
                    break

            # 将 best_node 连入子图
            for existing_node in best_clique:
                if existing_node in self.edges_map[best_node]:
                    # 保证边顺序一致，方便去重 (小ID在前)
                    u, v = sorted((best_node, existing_node))
                    sampled_edges.append((u, v))
            
            # 更新 K-tree 状态
            new_clique = list(best_clique)
            if len(new_clique) >= self.k:
                 new_clique.pop(0) # 简单的FIFO策略维持团大小
            new_clique.append(best_node)
            active_cliques.append(new_clique)
            
            visited_nodes.add(best_node)
            remaining_nodes.remove(best_node)
            
        return list(set(sampled_edges)) # 去重返回

    def _add_existing_edges(self, nodes_group, edge_list):
        for i in range(len(nodes_group)):
            for j in range(i+1, len(nodes_group)):
                u, v = nodes_group[i], nodes_group[j]
                if v in self.edges_map[u]:
                    u_s, v_s = sorted((u, v))
                    edge_list.append((u_s, v_s))

# ==========================================
# 4. 仿真与融合逻辑
# ==========================================

def run_bp_on_subgraph(nodes_meta, selected_edges_ids, measurements_map, noise_std, iterations=10):
    """
    在特定的子图上运行 BP
    nodes_meta: 节点元数据列表 (id, true_pos, is_anchor, init_guess)
    selected_edges_ids: 子图包含的边 [(u, v), ...]
    """
    # 1. 实例化节点 (深拷贝，隔离环境)
    sim_nodes = []
    node_map = {}
    for meta in nodes_meta:
        # 使用相同的初始猜测，保证起点一致
        n = Node(meta['id'], meta['true_pos'], meta['is_anchor'], meta['init_guess'])
        sim_nodes.append(n)
        node_map[n.id] = n
        
    # 2. 实例化边
    sim_edges = []
    for u_id, v_id in selected_edges_ids:
        if (u_id, v_id) in measurements_map:
            meas = measurements_map[(u_id, v_id)]
            edge = Edge(node_map[u_id], node_map[v_id], meas, noise_std)
            sim_edges.append(edge)
            
            # 注册邻居
            node_map[u_id].neighbors.append(v_id)
            node_map[v_id].neighbors.append(u_id)
            
            # 初始化消息
            node_map[u_id].incoming_messages[v_id] = (np.zeros(2), np.eye(2)*1e5)
            node_map[v_id].incoming_messages[u_id] = (np.zeros(2), np.eye(2)*1e5)

    # 3. 运行 BP
    for _ in range(iterations):
        # 计算消息
        new_msgs = []
        for edge in sim_edges:
            na, nb = edge.node_a, edge.node_b
            ma, ca = na.compute_outgoing_message(nb.id)
            mu_ab, sig_ab = edge.linearize_and_pass_message(na, nb, (ma, ca))
            new_msgs.append((nb.id, na.id, mu_ab, sig_ab))
            
            mb, cb = nb.compute_outgoing_message(na.id)
            mu_ba, sig_ba = edge.linearize_and_pass_message(nb, na, (mb, cb))
            new_msgs.append((na.id, nb.id, mu_ba, sig_ba))
            
        # 投递消息
        for tid, sid, m, s in new_msgs:
            node_map[tid].incoming_messages[sid] = (m, s)
            
        # 更新 Belief
        for n in sim_nodes:
            n.update_belief()
            
    # 4. 返回结果字典 {node_id: (mu, sigma)}
    results = {}
    for n in sim_nodes:
        results[n.id] = (n.mu, n.sigma)
    return results, sim_edges

def main_pipeline():
    # --- 参数设置 ---
    AREA_SIZE = 100
    NUM_ANCHORS = 4
    NUM_AGENTS = 15
    COMM_RANGE = 70
    NOISE_STD = 0.5
    
    K_SUBGRAPHS = 3  # 分解成几个子图
    TREEWIDTH_K = 3  # 每个子图的树宽限制
    BP_ITERATIONS = 10 # 每个子图跑几轮BP (树宽小，收敛快，10轮足够)
    
    # --- Step 1: 生成上帝视角数据 ---
    np.random.seed(42)
    anchors_pos = [[0, 0], [AREA_SIZE, 0], [AREA_SIZE, AREA_SIZE], [0, AREA_SIZE]]
    agents_true_pos = [np.random.rand(2) * AREA_SIZE for _ in range(NUM_AGENTS)]
    
    # 统一初始猜测 (让所有子图起点一致)
    agents_init_guess = [np.random.rand(2) * AREA_SIZE for _ in range(NUM_AGENTS)]
    
    # 构建节点元数据 (方便后续克隆)
    nodes_meta = []
    for i in range(NUM_ANCHORS):
        nodes_meta.append({'id': i, 'true_pos': anchors_pos[i], 'is_anchor': True, 'init_guess': None})
    for i in range(NUM_AGENTS):
        nodes_meta.append({'id': NUM_ANCHORS+i, 'true_pos': agents_true_pos[i], 'is_anchor': False, 'init_guess': agents_init_guess[i]})
        
    # 生成物理边 (Ground Truth Constraints)
    physical_edges = [] # 存储 (u, v)
    measurements_map = {} # 存储 (u, v) -> dist
    nodes_temp = [Node(m['id'], m['true_pos'], m['is_anchor']) for m in nodes_meta]
    
    for i in range(len(nodes_temp)):
        for j in range(i+1, len(nodes_temp)):
            dist = norm(nodes_temp[i].true_pos - nodes_temp[j].true_pos)
            if dist < COMM_RANGE:
                meas = dist + np.random.normal(0, NOISE_STD)
                u, v = sorted((nodes_temp[i].id, nodes_temp[j].id))
                physical_edges.append((u, v))
                measurements_map[(u, v)] = meas

    print(f"Total Physical Edges: {len(physical_edges)}")

    # --- Step 2: 定义初始权重 (Fisher Information) ---
    # Weight = 1 / Variance. 这里假设所有边方差相同，初始权重都设为 1.0 (或 1/0.25)
    # 用字典存储权重: {(u, v): weight}
    edge_weights = {edge: 1.0/(NOISE_STD**2) for edge in physical_edges}
    
    # --- Step 3: 分解循环 (Decomposition Loop) ---
    subgraph_results_list = [] # 存储 K 个子图的推断结果
    visual_edges_list = [] # 存储 K 个子图的边用于画图
    
    for k in range(K_SUBGRAPHS):
        print(f"\n>>> Processing Subgraph {k+1}/{K_SUBGRAPHS}...")
        
        # 3.1 准备 BTW 输入
        # BTW 需要 edges_with_weights 格式 [(u, v, w), ...]
        current_edges_weighted = []
        for (u, v), w in edge_weights.items():
            current_edges_weighted.append((u, v, w))
            
        # 3.2 运行 BTW 采样
        sampler = BTWGraphSampler(nodes_temp, current_edges_weighted, k=TREEWIDTH_K)
        selected_edges = sampler.sample_subgraph() # 返回 [(u, v), ...]
        print(f"    Selected {len(selected_edges)} edges (Treewidth <= {TREEWIDTH_K})")
        
        # 3.3 权重惩罚 (Diversity Mechanism)
        # 降低已选边的权重，迫使下一个子图选不同的边
        for edge in selected_edges:
            u, v = sorted(edge)
            if (u, v) in edge_weights:
                edge_weights[(u, v)] *= 0.1 # 惩罚因子 0.1
        
        # 3.4 在该子图上运行 BP
        # 注意：这里我们使用严谨的几何线性化 Edge 类
        res, sim_edges_objs = run_bp_on_subgraph(nodes_meta, selected_edges, measurements_map, NOISE_STD, iterations=BP_ITERATIONS)
        subgraph_results_list.append(res)
        visual_edges_list.append(selected_edges)

    # --- Step 4: 融合 (Fusion) ---
    print("\n>>> Fusing Results...")
    fused_results = {} # {node_id: mu_fused}
    final_rmse = 0
    count_agents = 0
    
    for meta in nodes_meta:
        nid = meta['id']
        if meta['is_anchor']:
            fused_results[nid] = meta['true_pos']
            continue
            
        # 收集 K 个子图对该节点的 Belief
        means = []
        covs = []
        for res in subgraph_results_list:
            if nid in res:
                mu, sigma = res[nid]
                means.append(mu)
                covs.append(sigma)
        
        # 高斯融合
        if means:
            mu_f, _ = product_of_gaussians(means, covs)
            fused_results[nid] = mu_f
            
            # 计算 RMSE
            err = norm(mu_f - meta['true_pos'])
            final_rmse += err**2
            count_agents += 1
            
    final_rmse = np.sqrt(final_rmse / count_agents)
    print(f"Final Fused RMSE: {final_rmse:.4f} m")

    # --- Step 5: 可视化 (Visualization) ---
    fig, axes = plt.subplots(1, K_SUBGRAPHS + 1, figsize=(5 * (K_SUBGRAPHS+1), 5))
    
    # 绘制每个子图的结果
    for k in range(K_SUBGRAPHS):
        ax = axes[k]
        ax.set_title(f"Subgraph {k+1} (Tree-BP)")
        ax.set_xlim(-10, AREA_SIZE+10); ax.set_ylim(-10, AREA_SIZE+10)
        ax.grid(True)
        
        # 画真实位置
        true_xy = np.array([m['true_pos'] for m in nodes_meta])
        ax.scatter(true_xy[:,0], true_xy[:,1], c='g', marker='x', alpha=0.5, label='True')
        
        # 画子图的边
        node_pos_map = {m['id']: m['true_pos'] for m in nodes_meta}
        for u, v in visual_edges_list[k]:
            p1 = node_pos_map[u]
            p2 = node_pos_map[v]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.2)
            
        # 画该子图的估计位置
        res = subgraph_results_list[k]
        est_xy = []
        for meta in nodes_meta:
            if not meta['is_anchor']:
                est_xy.append(res[meta['id']][0])
            else:
                est_xy.append(meta['true_pos'])
        est_xy = np.array(est_xy)
        ax.scatter(est_xy[:,0], est_xy[:,1], c='b', marker='.', label='Est')

    # 绘制最终融合结果
    ax_final = axes[-1]
    ax_final.set_title(f"Fused Result (RMSE={final_rmse:.2f}m)")
    ax_final.set_xlim(-10, AREA_SIZE+10); ax_final.set_ylim(-10, AREA_SIZE+10)
    ax_final.grid(True)
    
    # 画全图背景边 (淡)
    for u, v in physical_edges:
        p1 = node_pos_map[u]
        p2 = node_pos_map[v]
        ax_final.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.05)

    ax_final.scatter(true_xy[:,0], true_xy[:,1], c='g', marker='x', s=80, label='True')
    
    fused_xy = []
    for meta in nodes_meta:
        fused_xy.append(fused_results[meta['id']])
    fused_xy = np.array(fused_xy)
    
    ax_final.scatter(fused_xy[:,0], fused_xy[:,1], c='r', marker='o', label='Fused')
    ax_final.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main_pipeline()