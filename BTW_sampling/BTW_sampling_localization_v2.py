# 用BTW分解为若干子图，分别模拟消息传递，但是全局上融合各子图意见进行belief更新，取代各图的结果融合，但分解损失几何刚性和环的约束，效果不佳（特别12这种）

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, norm
import copy

# ==========================================
# 1. 基础数学工具函数 (保持不变)
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
# 2. 核心类 (Node & Edge)
# ==========================================
class Node:
    def __init__(self, node_id, true_pos, is_anchor=False, current_est=None):
        self.id = node_id
        self.true_pos = np.array(true_pos, dtype=float)
        self.is_anchor = is_anchor

        # Belief 初始化
        if is_anchor:
            self.mu = self.true_pos
            self.sigma_prior = np.eye(2) * 1e-8
        else:
            self.mu = np.array(current_est, dtype=float) if current_est is not None else np.random.rand(2) * 100
            self.sigma_prior = np.eye(2) * 10000 

        self.incoming_messages = {} 
        self.neighbors = [] # 已修复：添加 neighbors 属性
        self.sigma = self.sigma_prior.copy() 

    def compute_outgoing_message(self, target_neighbor_id):
        means = [self.mu] 
        covs = [self.sigma_prior]
        for nid, msg in self.incoming_messages.items():
            if nid != target_neighbor_id:
                means.append(msg[0])
                covs.append(msg[1])
        return product_of_gaussians(means, covs)

    def compute_marginal(self):
        if self.is_anchor:
            return self.mu, self.sigma_prior
        means = [self.mu]
        covs = [self.sigma_prior]
        for msg in self.incoming_messages.values():
            means.append(msg[0])
            covs.append(msg[1])
        return product_of_gaussians(means, covs)

class Edge:
    def __init__(self, node_a, node_b, measurement, noise_std):
        self.node_a = node_a
        self.node_b = node_b
        self.measurement = measurement 
        self.noise_var = noise_std**2 
    
    def linearize_and_pass_message(self, source_node, target_node, msg_in):
        mu_in, sigma_in = msg_in
        pos_source = source_node.mu
        pos_target = target_node.mu
        
        diff = pos_target - pos_source
        dist_est = norm(diff)
        if dist_est < 1e-3: dist_est = 1e-3

        u_vec = diff / dist_est
        mu_out = mu_in + u_vec * self.measurement
        
        angle = np.arctan2(u_vec[1], u_vec[0])
        c, s = np.cos(angle), np.sin(angle)
        Rot = np.array([[c, -s], [s, c]])
        
        # 严谨几何线性化：径向准，切向松
        R_local = np.array([[self.noise_var, 0], [0, 1e4]]) 
        R_global = Rot @ R_local @ Rot.T
        
        sigma_out = sigma_in + R_global
        return mu_out, sigma_out

# ==========================================
# 3. BTW 采样器 (已修复 tuple 问题)
# ==========================================
class BTWGraphSampler:
    def __init__(self, original_nodes, edges_with_weights, k=2):
        self.nodes = original_nodes
        self.node_ids = [n.id for n in original_nodes]
        self.edges_map = self._build_adj(edges_with_weights) 
        self.k = k
        
    def _build_adj(self, edges):
        adj = {nid: {} for nid in self.node_ids}
        for u, v, w in edges:
            if u in adj and v in adj:
                adj[u][v] = w
                adj[v][u] = w
        return adj

    def compute_score(self, node_id, clique):
        score = 0
        for existing_node_id in clique:
            if existing_node_id in self.edges_map[node_id]:
                score += self.edges_map[node_id][existing_node_id]
        score += np.random.uniform(0, 1e-5)
        return score

    def sample_subgraph(self):
        sampled_edges = []
        visited_nodes = set()
        
        anchors = [n.id for n in self.nodes if n.is_anchor]
        if len(anchors) >= self.k + 1:
            seed_nodes = anchors[:self.k+1]
        else:
            seed_nodes = anchors + [n.id for n in self.nodes if not n.is_anchor][:self.k+1-len(anchors)]
        visited_nodes.update(seed_nodes)
        
        initial_clique = list(seed_nodes)
        self._add_existing_edges(initial_clique, sampled_edges)
        
        active_cliques = [initial_clique]
        remaining_nodes = set(self.node_ids) - visited_nodes
        
        while remaining_nodes:
            best_score = -1
            best_node = None
            best_clique = None
            
            for node_id in remaining_nodes:
                for clique in active_cliques:
                    score = self.compute_score(node_id, clique)
                    if score > best_score:
                        best_score = score; best_node = node_id; best_clique = clique
            
            if best_node is None: 
                if remaining_nodes: 
                     best_node = list(remaining_nodes)[0]
                     best_clique = active_cliques[0]
                else: break

            for existing_node in best_clique:
                if existing_node in self.edges_map[best_node]:
                    edge_tuple = tuple(sorted((best_node, existing_node)))
                    sampled_edges.append(edge_tuple)
            
            new_clique = list(best_clique)
            if len(new_clique) >= self.k: new_clique.pop(0)
            new_clique.append(best_node)
            active_cliques.append(new_clique)
            visited_nodes.add(best_node)
            remaining_nodes.remove(best_node)
            
        return list(set(sampled_edges))

    def _add_existing_edges(self, nodes_group, edge_list):
        for i in range(len(nodes_group)):
            for j in range(i+1, len(nodes_group)):
                u, v = nodes_group[i], nodes_group[j]
                if v in self.edges_map[u]:
                    edge_list.append(tuple(sorted((u, v))))

# ==========================================
# 4. 新的运行逻辑 (Global Loop + Inflated Product)
# ==========================================

def run_one_step_bp(nodes_meta, global_estimates, selected_edges_ids, measurements_map, noise_std):
    node_map = {}
    sim_nodes = []
    for meta in nodes_meta:
        current_mu = global_estimates[meta['id']]
        n = Node(meta['id'], meta['true_pos'], meta['is_anchor'], current_mu)
        node_map[n.id] = n
        sim_nodes.append(n)
        
    sim_edges = []
    for u_id, v_id in selected_edges_ids:
        if (u_id, v_id) in measurements_map:
            meas = measurements_map[(u_id, v_id)]
            edge = Edge(node_map[u_id], node_map[v_id], meas, noise_std)
            sim_edges.append(edge)
            node_map[u_id].neighbors.append(v_id)
            node_map[v_id].neighbors.append(u_id)
            node_map[u_id].incoming_messages[v_id] = (np.zeros(2), np.eye(2)*1e5)
            node_map[v_id].incoming_messages[u_id] = (np.zeros(2), np.eye(2)*1e5)

    for _ in range(2): 
        new_msgs = []
        for edge in sim_edges:
            na, nb = edge.node_a, edge.node_b
            ma, ca = na.compute_outgoing_message(nb.id)
            mu_ab, sig_ab = edge.linearize_and_pass_message(na, nb, (ma, ca))
            new_msgs.append((nb.id, na.id, mu_ab, sig_ab))
            
            mb, cb = nb.compute_outgoing_message(na.id)
            mu_ba, sig_ba = edge.linearize_and_pass_message(nb, na, (mb, cb))
            new_msgs.append((na.id, nb.id, mu_ba, sig_ba))
            
        for tid, sid, m, s in new_msgs:
            node_map[tid].incoming_messages[sid] = (m, s)

    results = {}
    for n in sim_nodes:
        mu_marg, sigma_marg = n.compute_marginal()
        results[n.id] = (mu_marg, sigma_marg)
    
    return results, selected_edges_ids

def main_pipeline():
    # --- 参数 ---
    AREA_SIZE = 200
    NUM_ANCHORS = 4
    NUM_AGENTS = 40
    COMM_RANGE = 120
    NOISE_STD = 0.5
    
    K_SUBGRAPHS = 5      # 增加到 5，保证覆盖率
    TREEWIDTH_K = 3      # 保持 3
    GLOBAL_ITERATIONS = 10
    
    # --- Step 1: 数据生成 ---
    np.random.seed(32) # 测试 Seed=12 (之前报错的 Seed)
    anchors_pos = [[0, 0], [AREA_SIZE, 0], [AREA_SIZE, AREA_SIZE], [0, AREA_SIZE]]
    agents_true_pos = [np.random.rand(2) * AREA_SIZE for _ in range(NUM_AGENTS)]
    
    # 模拟“真值 + 5m 噪声”的初始值
    agents_init_guess = [pos + np.random.normal(0, 5, size=2) for pos in agents_true_pos]
    
    nodes_meta = []
    global_estimates = {} 
    
    for i in range(NUM_ANCHORS):
        meta = {'id': i, 'true_pos': anchors_pos[i], 'is_anchor': True}
        nodes_meta.append(meta)
        global_estimates[i] = anchors_pos[i]
        
    for i in range(NUM_AGENTS):
        meta = {'id': NUM_ANCHORS+i, 'true_pos': agents_true_pos[i], 'is_anchor': False}
        nodes_meta.append(meta)
        global_estimates[meta['id']] = agents_init_guess[i]
        
    physical_edges = []
    measurements_map = {}
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

    # --- Step 2: 定义初始权重 (关键修复：顺序调整) ---
    
    # 1. 先定义基础权重
    edge_weights = {edge: 1.0/(NOISE_STD**2) for edge in physical_edges}
    
    # 2. 再添加虚拟骨架 (现在 edge_weights 已经存在了，不会报错)
    virtual_edges = []
    for i in range(NUM_ANCHORS):
        for j in range(i + 1, NUM_ANCHORS):
            u, v = sorted((i, j))
            virtual_edges.append((u, v))
            edge_weights[(u, v)] = 1000.0 # 强制连接锚点

    # --- Step 2.5: 生成 K 个子图 ---
    print("Generating Subgraphs...")
    subgraphs_edges_list = []
    
    # 临时权重字典用于 BTW 采样惩罚
    temp_weights = copy.deepcopy(edge_weights)
    
    for k in range(K_SUBGRAPHS):
        # 准备 BTW 数据
        current_edges_weighted = [(u, v, temp_weights.get((u,v), 0)) for u,v in temp_weights]
        sampler = BTWGraphSampler(nodes_temp, current_edges_weighted, k=TREEWIDTH_K)
        selected = sampler.sample_subgraph()
        subgraphs_edges_list.append(selected)
        print(f"  Subgraph {k+1}: {len(selected)} edges")
        
        # 权重惩罚
        for edge in selected:
            if edge in temp_weights: temp_weights[edge] *= 0.1

    # --- Step 3: 全局同步迭代 ---
    print("\nStarting Global Iterations...")
    rmse_history = []
    
    # 关键参数：膨胀因子 (决定了融合的容错率)
    COV_INFLATION_FACTOR = 10.0 
    DAMPING_FACTOR = 0.6 
    
    for g_iter in range(GLOBAL_ITERATIONS):
        
        # 3.1 所有子图并行计算
        all_subgraph_results = []
        for k in range(K_SUBGRAPHS):
            res, _ = run_one_step_bp(nodes_meta, global_estimates, subgraphs_edges_list[k], measurements_map, NOISE_STD)
            all_subgraph_results.append(res)
            
        # 3.2 融合 (回归高斯乘积 + 膨胀)
        current_rmse = 0
        count = 0
        
        for meta in nodes_meta:
            nid = meta['id']
            if meta['is_anchor']: continue
            
            means = []
            covs = []
            
            # Weak Prior：拉住它别飞太远
            means.append(global_estimates[nid])
            covs.append(np.eye(2) * 10.0)
            
            for res in all_subgraph_results:
                if nid in res:
                    m, s = res[nid]
                    means.append(m)
                    # 【关键】协方差膨胀
                    covs.append(s * COV_INFLATION_FACTOR)
            
            if means:
                # 几何求交
                mu_fused, _ = product_of_gaussians(means, covs)
                
                # 阻尼更新
                if g_iter > 0:
                    old_pos = global_estimates[nid]
                    new_pos = (1 - DAMPING_FACTOR) * old_pos + DAMPING_FACTOR * mu_fused
                else:
                    new_pos = mu_fused
                
                global_estimates[nid] = new_pos
                current_rmse += norm(new_pos - meta['true_pos'])**2
                count += 1
        
        current_rmse = np.sqrt(current_rmse / count) if count > 0 else 0
        rmse_history.append(current_rmse)
        print(f"Global Iter {g_iter+1}: RMSE = {current_rmse:.4f} m")

    # --- Step 4: 可视化 ---
    fig, ax = plt.subplots(figsize=(8, 8))
    true_xy = np.array([m['true_pos'] for m in nodes_meta])
    est_xy = np.array([global_estimates[m['id']] for m in nodes_meta])
    
    ax.scatter(true_xy[:,0], true_xy[:,1], c='g', marker='x', s=100, label='True')
    ax.scatter(est_xy[:,0], est_xy[:,1], c='b', marker='o', label='Fused Est')
    
    node_pos_map = {m['id']: m['true_pos'] for m in nodes_meta}
    for u, v in physical_edges:
        p1 = node_pos_map[u]; p2 = node_pos_map[v]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.05)
        
    ax.set_title(f"Final Result (RMSE={rmse_history[-1]:.2f}m)")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main_pipeline()