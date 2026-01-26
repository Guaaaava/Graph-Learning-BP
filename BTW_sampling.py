import numpy as np
import heapq

class BTWGraphSampler:
    def __init__(self, original_nodes, original_edges, k=2):
        """
        original_nodes: 节点列表
        original_edges: 边列表 [(u, v, weight), ...]
        k: 目标树宽 (Treewidth Bound)
        """
        self.nodes = original_nodes
        self.edges_map = self._build_adj(original_edges) # 邻接表，存储边权重
        self.k = k
        
    def _build_adj(self, edges):
        # 构建邻接表，方便查询边权重
        adj = {node.id: {} for node in self.nodes}
        for u, v, w in edges:
            adj[u][v] = w
            adj[v][u] = w
        return adj

    def compute_score(self, node, clique):
        """
        核心评分函数 (论文公式 2)
        计算将 node 连接到 clique 时，能保留多少权重
        """
        score = 0
        for existing_node in clique:
            if existing_node in self.edges_map[node]:
                # 累加边的权重（如果是定位，这里可以是 1/sigma^2）
                score += self.edges_map[node][existing_node] 
        
        # 加入微小的随机噪声打破平局 (Tie-breaker)
        score += np.random.uniform(0, 1e-4)
        return score

    def sample_subgraph(self):
        """
        执行 BTW 算法，返回一个低树宽的边列表
        """
        sampled_edges = []
        visited_nodes = set()
        
        # 1. 初始化：随机选 k+1 个节点组成初始 clique (K-tree的种子)
        # 在定位中，建议选一个锚点及其周围节点，保证图是连通的
        seed_nodes = self._pick_seed_nodes(self.k + 1)
        visited_nodes.update(seed_nodes)
        
        # 将初始 clique 内部存在的边全部加入子图
        initial_clique = list(seed_nodes)
        self._add_existing_edges(initial_clique, sampled_edges)
        
        # 2. 维护一个 K-tree 的 clique 列表
        # 这里简化处理，只维护当前的 active cliques
        active_cliques = [initial_clique]
        
        # 3. 贪婪扩展：直到所有节点都被加入
        remaining_nodes = set(n.id for n in self.nodes) - visited_nodes
        
        while remaining_nodes:
            best_score = -1
            best_node = None
            best_clique = None
            
            # --- 优化点：这里可以用最大堆 (Max-Heap) 加速 ---
            # 简单遍历所有候选节点和所有 active cliques (论文中的 naive 方法)
            # 寻找获益最大的 (node, clique) 对
            for node_id in remaining_nodes:
                for clique in active_cliques:
                    score = self.compute_score(node_id, clique)
                    if score > best_score:
                        best_score = score
                        best_node = node_id
                        best_clique = clique
            
            if best_node is None:
                break # 理论上连通图不会发生
            
            # 4. 更新状态
            # 将 best_node 加入，并连接到 best_clique 中的节点
            # 注意：只添加原图中实际存在的边！
            for existing_node in best_clique:
                if existing_node in self.edges_map[best_node]:
                    sampled_edges.append((best_node, existing_node))
            
            # 更新 K-tree: best_node 和 best_clique 形成了一个新的 k+1 clique
            # 新的 clique 由 best_node 和 best_clique 中的 k 个节点组成
            # 这是一个简化，严格实现需要维护 K-tree 的三角剖分结构
            # 但作为 heuristics，我们可以保留 best_node 和 best_clique 组成的集合作为新的 active clique
            new_clique = list(best_clique)
            if len(new_clique) >= self.k:
                 # 保持 clique 大小不超过 k (这里是近似实现，论文中是 k-tree 严格定义)
                 # 实际实现时，通常移除 clique 中最旧的一个节点
                 new_clique.pop(0) 
            new_clique.append(best_node)
            active_cliques.append(new_clique)
            
            visited_nodes.add(best_node)
            remaining_nodes.remove(best_node)
            
        return sampled_edges

    def _pick_seed_nodes(self, n):
        # 简单实现：随机选 n 个
        return [n.id for n in self.nodes[:n]]

    def _add_existing_edges(self, nodes, edge_list):
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                u, v = nodes[i], nodes[j]
                if v in self.edges_map[u]:
                    edge_list.append((u, v))