"""
粒子化非参数 Belief Propagation (Particle-based BP) — 静态协同定位

从 NEBP 的 cooperative_localization.py 改写而来:
- 去掉运动模型 (F/Q/W) 和 particle_predict()
- 用 PyTorch Geometric (PyG) 替代 DGL
- 适配静态单快照定位场景
- 支持 edge_weights 做拓扑剪枝
"""

import torch
import torch.nn.functional as F


class ParticleBP:
    """
    粒子化非参数置信传播算法 (静态定位)

    每个 agent 维护 num_particles 个 2D 位置粒子,
    通过 anchor 测距和 agent 间测距的迭代消息传递更新粒子权重,
    最终输出加权均值作为位置估计。

    Parameters
    ----------
    num_particles : int
        粒子数。训练评估建议 2000，最终测试建议 10000
    sigma_meas : float
        测距噪声基准标准差 (当边特征中无方差时使用)
    num_iter : int
        BP 迭代次数，建议 5~15
    init_cov : float
        粒子初始化的协方差 (对角, 单位 m²)，默认 25.0 (std≈5m)
    """

    def __init__(self, num_particles=2000, sigma_meas=0.5,
                 num_iter=10, init_cov=25.0, reg_scale=0.3):
        self.num_particles = num_particles
        self.sigma_meas = sigma_meas
        self.num_iter = num_iter
        self.init_cov = init_cov
        self.reg_scale = reg_scale  # roughening noise scale (relative to init_std)

    def run(self, data, edge_weights=None):
        """
        在单张 PyG Data 图上运行粒子 BP。

        Parameters
        ----------
        data : torch_geometric.data.Data
            - x: (N+M, node_dim)  节点特征 [pos_x, pos_y, is_anchor, ...]
            - edge_index: (2, E)  边索引
            - edge_attr: (E, 3)   [measurement, variance, is_anchor_edge]
        edge_weights : Tensor (E,) or None
            二值边掩码 (0=剪枝, 1=保留)。None 表示使用所有边。

        Returns
        -------
        est_pos : Tensor (N, 2)
            Agent 估计位置
        est_cov : Tensor (N, 2, 2)
            Agent 位置估计协方差矩阵
        particles : Tensor (N, 2, P)
            重采样后的最终粒子集
        """
        device = data.x.device

        # ---- 解析节点信息 ----
        is_anchor = data.x[:, 4].bool()  # col4 = is_anchor flag
        agent_mask = ~is_anchor
        N = int(agent_mask.sum().item())

        agent_init_pos = data.x[agent_mask, :2].clone()
        anchor_pos = data.x[is_anchor, :2].clone()

        # ---- 解析边信息 ----
        edge_index = data.edge_index
        E = edge_index.shape[1]
        is_anchor_edge = data.edge_attr[:, 3].bool()  # col3 = is_anchor_edge flag
        meas_all = data.edge_attr[:, 0]
        var_all = data.edge_attr[:, 1]

        # 分离 anchor 边和 agent 边
        anc_mask = is_anchor_edge
        agt_mask = ~is_anchor_edge

        # Anchor 边: (agent_idx, anchor_local_idx)
        anc_u = edge_index[0, anc_mask]
        anc_v = edge_index[1, anc_mask] - N       # 转回局部 anchor 索引
        anc_meas = meas_all[anc_mask]
        anc_var = var_all[anc_mask]

        # Agent 边: (u, v)
        agt_u = edge_index[0, agt_mask]
        agt_v = edge_index[1, agt_mask]
        agt_meas = meas_all[agt_mask]
        agt_var = var_all[agt_mask]

        # ---- 应用边权重掩码 ----
        if edge_weights is not None:
            anc_active = edge_weights[anc_mask] > 0.5
            agt_active = edge_weights[agt_mask] > 0.5
        else:
            anc_active = torch.ones(anc_mask.sum(), dtype=torch.bool, device=device)
            agt_active = torch.ones(agt_mask.sum(), dtype=torch.bool, device=device)

        # ---- 初始化粒子 ----
        # 从高斯先验 N(init_pos, init_cov * I) 采样
        P = self.num_particles
        std_init = self.init_cov ** 0.5
        particles = (
            agent_init_pos.unsqueeze(2)
            + torch.randn(N, 2, P, device=device) * std_init
        )

        sigma2_base = self.sigma_meas ** 2
        eps = torch.finfo(torch.float32).eps

        # ---- 主 BP 循环 ----
        for _ in range(self.num_iter):
            log_weights = torch.zeros(N, P, device=device)

            # ============= 1. Anchor 测距权重 =============
            n_anc = int(anc_mask.sum().item())
            for e in range(n_anc):
                if not anc_active[e]:
                    continue
                i = anc_u[e].item()
                k = anc_v[e].item()
                z = anc_meas[e]
                var = max(anc_var[e].item(), sigma2_base)

                diff = particles[i] - anchor_pos[k].unsqueeze(1)  # (2, P)
                pred_dist = diff.norm(dim=0)                       # (P,)
                log_weights[i] += -(pred_dist - z) ** 2 / (2.0 * var)

            # 稳定化
            log_weights = log_weights - log_weights.max(dim=1, keepdim=True)[0]

            # ============= 2. Agent 间消息传递 =============
            # 用 anchor 加权后的 belief 作为消息传播的基础
            log_w_base = log_weights.clone()
            n_agt = int(agt_mask.sum().item())
            for e in range(n_agt):
                if not agt_active[e]:
                    continue
                u = agt_u[e].item()
                v = agt_v[e].item()
                z = agt_meas[e]
                var = max(agt_var[e].item(), sigma2_base)

                belief_v = F.softmax(log_w_base[v], dim=0)  # (P,)
                belief_u = F.softmax(log_w_base[u], dim=0)  # (P,)

                # --- v → u ---
                # diff: (2, P_u, P_v) — 所有粒子对的距离
                diff_vu = particles[u].unsqueeze(2) - particles[v].unsqueeze(1)
                pred_vu = diff_vu.norm(dim=0)                        # (P_u, P_v)
                w_vu = torch.exp(-(pred_vu - z) ** 2 / (2.0 * var))  # (P_u, P_v)
                msg_vu = (w_vu * belief_v.unsqueeze(0)).sum(dim=1)   # (P_u,)
                log_weights[u] += torch.log(msg_vu + eps)

                # --- u → v ---
                diff_uv = particles[v].unsqueeze(2) - particles[u].unsqueeze(1)
                pred_uv = diff_uv.norm(dim=0)
                w_uv = torch.exp(-(pred_uv - z) ** 2 / (2.0 * var))
                msg_uv = (w_uv * belief_u.unsqueeze(0)).sum(dim=1)   # (P_v,)
                log_weights[v] += torch.log(msg_uv + eps)

            # 稳定化
            log_weights = log_weights - log_weights.max(dim=1, keepdim=True)[0]

            # ============= 3. 信念更新 & 位置估计 =============
            belief = F.softmax(log_weights, dim=1)  # (N, P)

            est_pos = (particles * belief.unsqueeze(1)).sum(dim=2)         # (N, 2)
            diff = particles - est_pos.unsqueeze(2)                         # (N, 2, P)
            est_cov = torch.bmm(
                diff * belief.unsqueeze(1),
                diff.transpose(1, 2)
            )                                                              # (N, 2, 2)

            # ============= 4. ESS 条件重采样 + Roughening =============
            # 有效样本量: N_eff = 1 / Σ w_i²
            ess = 1.0 / (belief.square().sum(dim=1) + 1e-12)  # (N,)
            ess_thresh = P / 2

            for agent_i in range(N):
                if ess[agent_i] < ess_thresh:
                    # 粒子退化严重 → 重采样 + roughening
                    indices = torch.multinomial(belief[agent_i], P, replacement=True)
                    particles[agent_i] = particles[agent_i][:, indices]
                    if self.reg_scale > 0:
                        noise_std = self.reg_scale * (self.init_cov ** 0.5)
                        particles[agent_i] += torch.randn_like(particles[agent_i]) * noise_std

        return est_pos, est_cov, particles


def particle_bp_batch(data, edge_weights=None, **kwargs):
    """
    对 PyG batch 中的每张子图分别运行粒子 BP。

    Parameters
    ----------
    data : torch_geometric.data.Batch
    edge_weights : Tensor (E,) or None
    **kwargs : 传递给 ParticleBP 的参数

    Returns
    -------
    all_est_pos : list[Tensor]
    all_est_cov : list[Tensor]
    all_particles : list[Tensor]
    """
    bp = ParticleBP(**kwargs)

    all_est_pos, all_est_cov, all_particles = [], [], []
    for i in range(data.num_graphs):
        sub = data.get_example(i)
        ew = None
        if edge_weights is not None:
            mask = data.batch[data.edge_index[0]] == i
            ew = edge_weights[mask]
        ep, ec, pa = bp.run(sub, edge_weights=ew)
        all_est_pos.append(ep)
        all_est_cov.append(ec)
        all_particles.append(pa)

    return all_est_pos, all_est_cov, all_particles
