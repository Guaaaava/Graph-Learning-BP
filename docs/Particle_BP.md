# Particle-based Belief Propagation

> **本文档对应代码：** `particle_bp.py`
> **算法类型：** Particle-based Belief Propagation (Particle BP) / 非参数 BP（NBP）
> **应用场景：** 静态单快照 Cooperative Localization
>
> 本文档同时说明**数学公式、算法步骤、代码实现和数据流之间的一一对应关系**，并特别标出当前实现中与 NEBP 思路对应的部分。

---

## 1. 问题定义

考虑一个包含 anchor 和 agent 的无线协作定位网络。设：

- Anchor 数量为 $K_a$；
- Agent 数量为 $K_b$；
- 每个节点的位置为二维向量 
  $$ \mathbf{x}_i=[x_i,y_i]^\mathsf{T}\in\mathbb{R}^2. $$

Anchor 的位置已知：$ \mathbf{x}_k,\qquad k\in\mathcal{V}_a. $

Agent 的位置未知，需要根据 anchor-agent 和 agent-agent 测距信息估计：$ \mathbf{x}_i,\qquad i\in\mathcal{V}_b. $

整个网络表示为图：$ \mathcal{G}=(\mathcal{V},\mathcal{E}). $ 其中：

- $\mathcal{V}=\mathcal{V}_a\cup\mathcal{V}_b$；
- $\mathcal{E}$ 表示存在测距/通信关系的节点对。

当前代码只处理一个静态快照，因此没有 NEBP 原始代码中的运动模型 $(F,Q,W)$，也没有 `particle_predict()`。

---

## 2. 测距观测模型

对于两个存在测距关系的节点 $(i,j)$，观测距离记为：$ z_{ij}. $

理想距离：$ d_{ij}(\mathbf{x}_i,\mathbf{x}_j) = \|\mathbf{x}_i-\mathbf{x}_j\|_2. $

采用高斯测距噪声模型：$ z_{ij} = \|\mathbf{x}_i-\mathbf{x}_j\|_2 + \epsilon_{ij} $，其中：$ \epsilon_{ij}\sim\mathcal{N}(0,\sigma_{ij}^2). $

因此对应的似然为：

$$
p(z_{ij}\mid\mathbf{x}_i,\mathbf{x}_j) = \frac{1}{\sqrt{2\pi\sigma_{ij}^2}} \exp\left[ -\frac{ \left( z_{ij}-\|\mathbf{x}_i-\mathbf{x}_j\|_2 \right)^2 }{ 2\sigma_{ij}^2 } \right]
$$
代码中使用的实际上是与该似然**成正比**的部分，即：

$$
\psi_{ij}(\mathbf{x}_i,\mathbf{x}_j) \propto \exp\left[ -\frac{ \left( \|\mathbf{x}_i-\mathbf{x}_j\|_2-z_{ij} \right)^2 }{ 2\sigma_{ij}^2 } \right]
$$
为什么代码没有写 $1/\sqrt{2\pi\sigma^2}$？

因为在当前实现中，对于固定的一条边，$\sigma_{ij}$ 是已知常数，因此归一化系数与粒子位置无关。

BP 中最终要对不同粒子位置比较相对权重：

$$ w^{(p)} \propto p(z\mid\mathbf{x}^{(p)}). $$

所以可以省略与 $\mathbf{x}$ 无关的常数。

当前代码：

```python
likelihood_uv = torch.exp(
    -(pred_uv - z) ** 2 / (2.0 * var)
)
```

---

## 3. Factor Graph / BP 的基本形式

对于 agent $i$，其 belief 可以写成：

$$
b_i(\mathbf{x}_i) \propto \phi_i(\mathbf{x}_i) \prod_{j\in\mathcal{N}_b(i)} m_{j\rightarrow i}(\mathbf{x}_i)
$$
其中：

- $\phi_i(\mathbf{x}_i)$：来自 anchor 测距的 local factor；
- $m_{j\rightarrow i}$：agent $j$ 向 agent $i$ 传递的消息；
- $\mathcal{N}_b(i)$：agent-agent 邻居集合。

Anchor factor：$ \phi_i(\mathbf{x}_i) = \prod_{k\in\mathcal{N}_a(i)} p(z_{ik}\mid\mathbf{x}_i,\mathbf{x}_k). $

Agent-agent pairwise factor：$ \psi_{ij}(\mathbf{x}_i,\mathbf{x}_j) = p(z_{ij}\mid\mathbf{x}_i,\mathbf{x}_j). $

因此：

$$
b_i(\mathbf{x}_i) \propto  \prod_{k\in\mathcal{N}_a(i)} p(z_{ik}\mid\mathbf{x}_i,\mathbf{x}_k)  \prod_{j\in\mathcal{N}_b(i)} m_{j\rightarrow i}(\mathbf{x}_i)
$$
这就是当前 Particle BP 的核心理论基础。

---

## 4. Particle Approximation

由于 $\mathbf{x}_i\in\mathbb{R}^2$ 是连续变量，直接计算 $ b_i(\mathbf{x}_i) $ 通常很困难。

因此用 $P$ 个粒子近似：

$$
b_i(\mathbf{x}_i) \approx \sum_{p=1}^{P} w_i^{(p)} \delta( \mathbf{x}_i-\mathbf{x}_i^{(p)} )
$$
其中：

- $P$：粒子数量；
- $\mathbf{x}_i^{(p)}$：第 $p$ 个粒子的位置；
- $w_i^{(p)}$：对应归一化权重；
- $\delta(\cdot)$：Dirac delta 函数，用于表示一个确定位置上的概率质量。

权重满足：

$$ w_i^{(p)}\ge 0, \qquad \sum_{p=1}^{P}w_i^{(p)}=1. $$

---

## 5. Step 1：读取图结构和节点信息

代码：

```python
is_anchor = data.x[:, 4].bool()
agent_mask = ~is_anchor
N = int(agent_mask.sum().item())

agent_init_pos = data.x[agent_mask, :2].clone()
anchor_pos = data.x[is_anchor, :2].clone()
```

数据流：

```text
data.x
  │
  ├── is_anchor
  │
  ├── agent_mask
  │
  ├── agent_init_pos
  │
  └── anchor_pos
```

其中：$ \mathbf{x}_{i,\mathrm{init}} = \texttt{data.x[i,:2]}. $

需要特别注意：

> `agent_init_pos` 是粒子初始化中心。如果它实际上是真实 Ground Truth position，那么会产生严重的数据泄漏。因此 baseline 中这里必须是合法的 prior / initialization，而不能是测试目标真值。

---

## 6. Step 2：读取测距边

代码：

```python
edge_index = data.edge_index
is_anchor_edge = data.edge_attr[:, 3].bool()

meas_all = data.edge_attr[:, 0]
var_all = data.edge_attr[:, 1]
```

每条边至少包含：

$$ [z_{ij},\sigma_{ij}^2,\ldots,\mathrm{isAnchorEdge}]. $$

随后把边分成两类：

### Anchor-agent edge

$$ (i,k),\qquad i\in\mathcal{V}_b,\quad k\in\mathcal{V}_a. $$

### Agent-agent edge

$$ (i,j),\qquad i,j\in\mathcal{V}_b. $$

代码：

```python
anc_mask = is_anchor_edge
agt_mask = ~is_anchor_edge
```

---

## 7. Step 3：兼容应用 GIB 的边剪枝结果

如果 `edge_weights is None`：

$$ w_e=1,\qquad\forall e. $$

即使用完整 Dense Graph。

如果提供 `edge_weights`：

$$ w_e\in\{0,1\}. $$

其中：$ w_e=1 $ 表示保留该边，$ w_e=0 $ 表示 GIB 将该边剪掉。

代码：

```python
anc_active = edge_weights[anc_mask] > 0.5
agt_active = edge_weights[agt_mask] > 0.5
```

因此整个方法可以形成：

$$
\boxed{ \text{Dense Graph} \xrightarrow{\mathrm{GIB}} \text{Sparse Graph} \xrightarrow{\mathrm{Particle\ BP}} \text{Localization} }
$$
Particle BP 本身不负责学习拓扑；它只接受 `edge_weights` 指定的图。

---

## 8. Step 4：初始化粒子

每个 agent $i$ 初始化 $P$ 个二维位置粒子：

$$ \mathbf{x}_i^{(p)} \sim \mathcal{N} \left( \mathbf{x}_{i,\mathrm{init}}, \sigma_0^2\mathbf{I}_2 \right). $$

其中：

$$ \sigma_0^2 = \texttt{init\_cov}. $$

代码：

```python
P = self.num_particles
std_init = self.init_cov ** 0.5

particles = (
    agent_init_pos.unsqueeze(2)
    + torch.randn(
        N, 2, P,
        device=device,
        dtype=dtype
    ) * std_init
)
```

最终：

$$ \texttt{particles.shape} = (N,2,P). $$

也就是：

```text
particles[i, :, p]
       ↓
agent i 的第 p 个二维位置粒子
```

---

## 9. Step 5：Anchor Likelihood

### 9.1 Anchor factor

对于 agent $i$，如果它连接多个 anchor：

$$ \phi_i(\mathbf{x}_i) = \prod_{k\in\mathcal{N}_a(i)} p(z_{ik}\mid\mathbf{x}_i,\mathbf{x}_k). $$

对于第 $p$ 个粒子：

$$ \phi_i^{(p)} = \prod_{k\in\mathcal{N}_a(i)} \psi_{ik} ( \mathbf{x}_i^{(p)}, \mathbf{x}_k ). $$

---

### 9.2 转换到 log domain

乘法变成加法：

$$ \log\phi_i^{(p)} = \sum_{k\in\mathcal{N}_a(i)} \log \psi_{ik} ( \mathbf{x}_i^{(p)}, \mathbf{x}_k ). $$

由于：

$$ \log\psi_{ik} = -\frac{ (d_{ik}^{(p)}-z_{ik})^2 }{ 2\sigma_{ik}^2 }, $$

因此：

$$
L_{i,\mathrm{anchor}}^{(p)} = -\sum_{k\in\mathcal{N}_a(i)} \frac{ ( \|\mathbf{x}_i^{(p)}-\mathbf{x}_k\|_2-z_{ik} )^2 }{ 2\sigma_{ik}^2 }
$$
代码：

```python
anchor_log_weights[i] += (
    -(pred_dist - z) ** 2
    / (2.0 * var)
)
```

---

## 10. 为什么这里使用 log probability？

原始概率需要连续相乘：

$$ \phi_i^{(p)} = \prod_k \psi_{ik}^{(p)}. $$

如果有很多边，每个 $\psi$ 都小于 1，那么连续相乘很容易产生数值下溢（如1e-1000）。

使用：$ \log\prod_k\psi_k = \sum_k\log\psi_k. $

因此代码内部维护：$ \log b_i $ 而不是直接维护 $b_i$。

最终需要概率权重时，再通过：

```python
belief = F.softmax(belief_log, dim=1)
```

得到归一化权重。

> 注：$softmax(\mathbf{z})_i=\sigma(\mathbf{z})_i=\frac{e^{z_i}}{\sum^K_{j=1}e^{z_i}}$，并且具有平移不变性，即$\sigma(\mathbf{z}+c)_i=\sigma(\mathbf{z})_i$

---

## 11. Log-domain Numerical Normalization

代码：

```python
anchor_log_weights = (
    anchor_log_weights
    - anchor_log_weights.max(
        dim=1,
        keepdim=True
    )[0]
)
```

数学上：

$$ \tilde{L}_i^{(p)} = L_i^{(p)} - \max_q L_i^{(q)}. $$

因此：

$$ \exp(\tilde{L}_i^{(p)}) = \frac{ \exp(L_i^{(p)}) }{ \exp(\max_q L_i^{(q)}) }. $$

所有粒子的相对比例不发生改变。这是标准的 log-weight stabilization。

---

## 12. Step 6：初始化 Belief

代码：

```python
belief_log = anchor_log_weights.clone()
```

数学上：

$$ b_i^{(0)}(\mathbf{x}_i) \propto \phi_i(\mathbf{x}_i) $$

粒子形式：

$$ w_i^{(p,0)} \propto \phi_i^{(p)}. $$

因此初始 belief 只包含 agent 的 anchor measurement information。

如果一个 agent 没有 anchor 边，则：$ \phi_i(\mathbf{x}_i)=1, $

因此初始 belief 是均匀粒子权重。

---

## 13. Step 7：Iterative Agent-Agent BP

这是整个 Particle BP 的核心。

设 BP 总迭代次数为：

$$ T=\texttt{num\_iter}. $$

第 $t$ 轮：

$$ b^{(t)} \rightarrow m^{(t)} \rightarrow b^{(t+1)}. $$

当前代码采用**同步（synchronous）belief update**。

---

## 14. Step 7.1：保存上一轮 Belief

代码：

```python
belief_prev = F.softmax(
    belief_log,
    dim=1
)
```

数学上：

$$
w_i^{(p,t)} = \frac{ \exp(L_i^{(p,t)}) }{ \sum_q\exp(L_i^{(q,t)}) }
$$
因此：

$$ \sum_p w_i^{(p,t)}=1. $$

这就是第 $t$ 轮用于发送消息的粒子 belief。

---

## 15. Step 7.2：初始化新的 Belief

代码：

```python
new_belief_log = anchor_log_weights.clone()
```

这对应：

$$ \log b_i^{(t+1)} = \log\phi_i + \sum_j\log m_{j\rightarrow i}^{(t)}. $$

这里先放入固定的 anchor factor：

$$ \log b_i^{(t+1)} \leftarrow \log\phi_i. $$

之后把所有 agent-agent message 加进去。

---

## 16. Step 7.3：Agent-Agent Message

对于 agent $u$ 和 $v$，定义：

$$ \psi_{uv} (\mathbf{x}_u,\mathbf{x}_v) = p(z_{uv}\mid\mathbf{x}_u,\mathbf{x}_v). $$

BP message：

$$
m_{u\rightarrow v}^{(t)} (\mathbf{x}_v) = \int \psi_{uv} (\mathbf{x}_u,\mathbf{x}_v) b_u^{(t)}(\mathbf{x}_u) d\mathbf{x}_u
$$
这是标准 BP 的积分形式。

---

## 17. Step 7.4：Particle Approximation of Message

用粒子近似：

$$ b_u^{(t)}(\mathbf{x}_u)\ \approx\ \sum_{p=1}^{P} w_u^{(p,t)} \delta ( \mathbf{x}_u-\mathbf{x}_u^{(p)} ). $$

代入 BP message：

$$ m_{u\rightarrow v}^{(t)} (\mathbf{x}_v) \approx \sum_{p=1}^{P} w_u^{(p,t)} \psi_{uv} ( \mathbf{x}_u^{(p)},\mathbf{x}_v ). $$

在 $v$ 的第 $q$ 个粒子处：

$$
m_{u\rightarrow v}^{(t)} (\mathbf{x}_v^{(q)}) \approx \sum_{p=1}^{P} w_u^{(p,t)} \psi_{uv} ( \mathbf{x}_u^{(p)}, \mathbf{x}_v^{(q)} )
$$

---

## 18. Step 7.5：代码如何构造所有粒子对

代码：

```python
diff_uv = (
    particles[u].unsqueeze(2)
    - particles[v].unsqueeze(1)
)
```

维度：

```text
particles[u]          		[2, P]
particles[u].unsqueeze(2)
                      		[2, P, 1]

particles[v]          		[2, P]
particles[v].unsqueeze(1)
                      		[2, 1, P]

相减得                 	  [2, P, P]
```

即：

$$ \Delta\mathbf{x}_{uv}^{p,q} = \mathbf{x}_u^{(p)} - \mathbf{x}_v^{(q)}. $$

> 注：unsqueeze(x) 函数代表在第 x 维前面加一维，如 a = [6, 7, 8]，则 a.unsqueeze(1) = [6, 1, 7, 8]，维数从 0 开始，unsqueeze(-1) 代表在最后加一维，相当于 a.unsqueeze(3) = [6, 7, 8, 1]

然后：

```python
pred_uv = diff_uv.norm(dim=0)
```

得到：

$$ d_{uv}^{p,q} = \| \mathbf{x}_u^{(p)} - \mathbf{x}_v^{(q)} \|_2. $$

维度：

$$ [P_u,P_v]. $$

---

## 19. Step 7.6：计算 Pairwise Likelihood

代码：

```python
likelihood_uv = torch.exp(
    -(pred_uv - z) ** 2
    / (2.0 * var)
)
```

数学上：

$$
\psi_{uv}^{p,q} = \exp \left[ -\frac{ (d_{uv}^{p,q}-z_{uv})^2 }{ 2\sigma_{uv}^2 } \right]
$$
其中：

$$ d_{uv}^{p,q} = \| \mathbf{x}_u^{(p)} - \mathbf{x}_v^{(q)} \|_2. $$

---

## 20. Step 7.7：Message ($u\rightarrow v$)

代码：

```python
msg_v = (
    likelihood_uv
    * belief_u.unsqueeze(1)
).sum(dim=0)
```

维度：

```text
likelihood_uv          [P_u, P_v]

belief_u               [P_u]

belief_u.unsqueeze(1)  [P_u, 1]

逐元素相乘得             [P_u, P_v]

sum(dim=0)             [P_v]
```

因此：

$$
m_{u\rightarrow v}^{(t)} (\mathbf{x}_v^{(q)}) = \sum_{p=1}^{P} \psi_{uv}^{p,q} w_u^{(p,t)}
$$
这里：

> **Message 的结果定义在接收节点 $v$ 的粒子上。**

因此最终：

```python
msg_v.shape == [P]
```

---

## 21. Step 7.8：Message ($v\rightarrow u$)

同理：

```python
diff_vu = (
    particles[v].unsqueeze(2)
    - particles[u].unsqueeze(1)
)

pred_vu = diff_vu.norm(dim=0)

likelihood_vu = torch.exp(
    -(pred_vu - z) ** 2
    / (2.0 * var)
)

msg_u = (
    likelihood_vu
    * belief_v.unsqueeze(1)
).sum(dim=0)
```

数学上：

$$
m_{v\rightarrow u}^{(t)} (\mathbf{x}_u^{(p)}) = \sum_{q=1}^{P} w_v^{(q,t)} \psi_{uv} ( \mathbf{x}_v^{(q)}, \mathbf{x}_u^{(p)} )
$$
因此：

```text
msg_u.shape == [P]
```

---

## 22. Step 7.9：把 Message 乘入 Belief

BP belief equation：

$$
b_v^{(t+1)}(\mathbf{x}_v) \propto \phi_v(\mathbf{x}_v) \prod_{u\in\mathcal{N}(v)} m_{u\rightarrow v}^{(t)} (\mathbf{x}_v)
$$
对于粒子 $q$：

$$
w_v^{(q,t+1)} \propto \phi_v^{(q)} \prod_{u\in\mathcal{N}(v)} m_{u\rightarrow v}^{(q,t)}
$$
由于代码工作在 log domain：

$$
L_v^{(q,t+1)} = L_{v,\mathrm{anchor}}^{(q)} + \sum_{u\in\mathcal{N}(v)} \log m_{u\rightarrow v}^{(q,t)}
$$
代码：

```python
new_belief_log[v] += log_msg_u_to_v
new_belief_log[u] += log_msg_v_to_u
```

这就是：

$$ \text{product of factors} \quad\longrightarrow\quad \text{sum of log factors}. $$

---

## 23. Step 7.10：为什么一定要使用上一轮 Belief？

当前代码：

```python
belief_prev = F.softmax(belief_log, dim=1)
```

之后整个 iteration 都使用：

```python
belief_u = belief_prev[u]
belief_v = belief_prev[v]
```

而不是使用正在修改的：

```python
new_belief_log
```

因此实现：

$$ \boxed{ b^{(t)} \rightarrow \text{all messages} \rightarrow b^{(t+1)} } $$

而不是：

$$ b_u^{(t)} \rightarrow b_v^{(t+1)} \rightarrow b_u^{(t+1)} \rightarrow\cdots $$

这与标准的同步 BP 更新更一致，也避免一轮 iteration 内部出现更新顺序依赖。

---

## 24. Step 7.11：完成一轮 BP 后归一化

代码：

```python
new_belief_log = (
    new_belief_log
    - new_belief_log.max(
        dim=1,
        keepdim=True
    )[0]
)

belief_log = new_belief_log
```

数学上：

$$ L_i^{(p,t+1)} \leftarrow L_i^{(p,t+1)} - \max_q L_i^{(q,t+1)}. $$

这不会改变归一化后的 belief：

$$ \operatorname{softmax}(L) = \operatorname{softmax}(L+c) $$

对任意常数 $c$ 都成立。

---

## 25. 完整 BP 迭代方程

因此当前代码的完整迭代可以总结为：

#### 第 $t$ 轮 belief

$$
b_i^{(t)} \approx \sum_p w_i^{(p,t)} \delta( \mathbf{x}_i-\mathbf{x}_i^{(p)} )
$$

#### Message

$$
m_{j\rightarrow i}^{(t)} (\mathbf{x}_i^{(p)}) \approx \sum_q w_j^{(q,t)} \psi_{ij} ( \mathbf{x}_j^{(q)}, \mathbf{x}_i^{(p)} )
$$

#### Belief

$$
b_i^{(t+1)} (\mathbf{x}_i^{(p)}) \propto \phi_i(\mathbf{x}_i^{(p)}) \prod_{j\in\mathcal{N}(i)} m_{j\rightarrow i}^{(t)} (\mathbf{x}_i^{(p)})
$$

#### Weight normalization

$$
w_i^{(p,t+1)} = \frac{ \exp L_i^{(p,t+1)} }{ \sum_q \exp L_i^{(q,t+1)} }
$$

---

## 26. Step 8：得到最终 Belief

完成：

$$ T=\texttt{num\_iter} $$

轮以后：

$$ L_i^{(p,T)} $$

就是最终 log-belief。

代码：

```python
belief = F.softmax(
    belief_log,
    dim=1
)
```

得到：

$$ w_i^{(p)} = \frac{ \exp L_i^{(p,T)} }{ \sum_q \exp L_i^{(q,T)} } $$

满足：

$$ \sum_p w_i^{(p)}=1. $$

---

## 27. Step 9：最终位置估计

使用粒子 belief 的加权均值：

$$
\hat{\mathbf{x}}_i = \sum_{p=1}^{P} w_i^{(p)} \mathbf{x}_i^{(p)}
$$
代码：

```python
est_pos = (
    particles
    * belief.unsqueeze(1)
).sum(dim=2)
```

输出：

```text
est_pos.shape = [N, 2]
```

因此：

$$ \texttt{est\_pos}[i] = [\hat{x}_i,\hat{y}_i]. $$

---

## 28. Step 10：估计位置协方差

定义粒子相对于估计均值的偏差：

$$ \Delta\mathbf{x}_i^{(p)} = \mathbf{x}_i^{(p)} - \hat{\mathbf{x}}_i. $$

代码：

```python
diff = (
    particles
    - est_pos.unsqueeze(2)
)
```

加权协方差：

$$
\hat{\mathbf\Sigma}_i = \sum_{p=1}^{P} w_i^{(p)} \left( \mathbf{x}_i^{(p)} - \hat{\mathbf{x}}_i \right) \left( \mathbf{x}_i^{(p)} - \hat{\mathbf{x}}_i \right)^\mathsf{T}
$$
代码：

```python
est_cov = torch.bmm(
    diff * belief.unsqueeze(1),
    diff.transpose(1, 2)
)
```

输出：

$$ \texttt{est\_cov.shape} = (N,2,2). $$

---

## 29. Step 11：最终一次 Resampling

当前 baseline 不在每一轮进行 resampling，BP 完成以后才进行一次：

```python
indices = torch.multinomial(
    belief,
    P,
    replacement=True
)
```

数学上：

$$ \mathbf{x}_i^{(p')} \sim \operatorname{Categorical} \left( w_i^{(1)},\ldots,w_i^{(P)} \right) $$

即按照最终 belief 的粒子权重重新采样。

---

## 30. Step 12：(Optional) Roughening

最终 resampling 后，如果：

$$ \texttt{reg\_scale}=0, $$

则不加入任何扰动：

$$ \mathbf{x}_{i,\mathrm{final}}^{(p)} = \mathbf{x}_{i,\mathrm{resampled}}^{(p)} $$

这就是当前建议的严格 baseline。

代码：

```python
if self.reg_scale > 0:
    ...
```

如果：

$$ \texttt{reg\_scale}>0, $$

加入：

$$ \boldsymbol\epsilon_i^{(p)} \sim \mathcal{N} ( 0, \sigma_r^2\mathbf{I} ) $$

其中：

$$ \sigma_r = \texttt{reg\_scale}\sqrt{\texttt{init\_cov}}. $$

因此：

$$ \mathbf{x}_{i,\mathrm{final}}^{(p)} = \mathbf{x}_{i,\mathrm{resampled}}^{(p)} + \boldsymbol\epsilon_i^{(p)} $$

代码：

```python
noise_std = (
    self.reg_scale
    * (self.init_cov ** 0.5)
)

particles_resampled += (
    torch.randn_like(particles_resampled)
    * noise_std
)
```

因此：

| `reg_scale` | 含义                                 |
| :---------: | :----------------------------------- |
|     `0`     | 严格 NBP baseline，不进行 roughening |
|    `>0`     | roughening ablation                  |

---

## 31. 完整算法流程

整个 Particle BP 可以压缩成以下数学流程。

### Step 1：粒子初始化

$$
\mathbf{x}_i^{(p)} \sim \mathcal{N} ( \mathbf{x}_{i,\mathrm{init}}, \sigma_0^2\mathbf{I} )
$$

---

### Step 2：计算 Anchor factor

$$
\log\phi_i^{(p)} = -\sum_{k\in\mathcal{N}_a(i)} \frac{ ( \|\mathbf{x}_i^{(p)}-\mathbf{x}_k\|_2 -z_{ik} )^2 }{ 2\sigma_{ik}^2 }
$$

---

### Step 3：初始化 belief

$$
L_i^{(p,0)} = \log\phi_i^{(p)}
$$

---

### Step 4：BP iteration

对于 $(t=0,\ldots,T-1)$：

#### 4.1 Normalize previous belief

$$
w_i^{(p,t)} = \operatorname{softmax} ( L_i^{(p,t)} )
$$

#### 4.2 Calculate message

$$
m_{j\rightarrow i}^{(t)}(p) = \sum_q w_j^{(q,t)} \exp \left[ -\frac{ ( \| \mathbf{x}_i^{(p)} - \mathbf{x}_j^{(q)} \|_2 -z_{ij} )^2 }{ 2\sigma_{ij}^2 } \right]
$$

#### 4.3 Update belief

$$
L_i^{(p,t+1)} = \log\phi_i^{(p)} + \sum_{j\in\mathcal{N}(i)} \log m_{j\rightarrow i}^{(t)}(p)
$$

#### 4.4 Normalize log weights

$$
L_i^{(p,t+1)} \leftarrow L_i^{(p,t+1)} - \max_q L_i^{(q,t+1)}
$$

---

### Step 5：Final belief

$$
w_i^{(p)} = \operatorname{softmax} ( L_i^{(p,T)} )
$$

---

### Step 6：Position estimation

$$
\hat{\mathbf{x}}_i = \sum_p w_i^{(p)} \mathbf{x}_i^{(p)}
$$

---

### Step 7：Covariance estimation

$$
\hat{\mathbf\Sigma}_i = \sum_p w_i^{(p)} ( \mathbf{x}_i^{(p)} -\hat{\mathbf{x}}_i ) ( \mathbf{x}_i^{(p)} -\hat{\mathbf{x}}_i )^\mathsf{T}
$$

---

### Step 8：Final resampling

$$
\mathbf{x}_i^{(p')} \sim \operatorname{Categorical} (w_i^{(1)},\ldots,w_i^{(P)})
$$

---

### Step 9：Optional roughening

$$ \mathbf{x}_i^{(p')} \leftarrow \mathbf{x}_i^{(p')} + \mathcal{N} ( 0, \sigma_r^2\mathbf{I} ) $$

仅当：

$$ \texttt{reg\_scale}>0. $$

---

## 32. 数学 ↔ 代码一一对应表

| 数学对象                   | 数学表达式                            | 当前代码                                  |
| :------------------------- | :------------------------------------ | :---------------------------------------- |
| Agent prior                | $\mathbf{x}_{i,\mathrm{init}}$        | `agent_init_pos`                          |
| Anchor position            | $\mathbf{x}_k$                        | `anchor_pos`                              |
| Particle                   | $\mathbf{x}_i^{(p)}$                  | `particles[i,:,p]`                        |
| Anchor factor              | $\phi_i$                              | `anchor_log_weights`                      |
| Initial belief             | $b_i^{(0)}$                           | `belief_log = anchor_log_weights.clone()` |
| Normalized weight          | $w_i^{(p,t)}$                         | `F.softmax(belief_log, dim=1)`            |
| Pairwise distance          | $\|\mathbf{x}_u^p-\mathbf{x}_v^q\|_2$ | `pred_uv`                                 |
| Pairwise likelihood        | $\psi_{uv}^{p,q}$                     | `likelihood_uv`                           |
| ($u\rightarrow v$) message | $m_{u\rightarrow v}$                  | `msg_v`                                   |
| ($v\rightarrow u$) message | $m_{v\rightarrow u}$                  | `msg_u`                                   |
| Log message                | $\log m$                              | `log_msg_u_to_v`, `log_msg_v_to_u`        |
| New belief                 | $b_i^{(t+1)}$                         | `new_belief_log`                          |
| Final belief               | $b_i^{(T)}$                           | `belief`                                  |
| Position mean              | $E[\mathbf{x}_i]$                     | `est_pos`                                 |
| Position covariance        | $\operatorname{Cov}[\mathbf{x}_i]$    | `est_cov`                                 |
| Resampling                 | $x^{p'}\sim\operatorname{Cat}(w)$     | `torch.multinomial`                       |
| Roughening                 | $x'=x+\epsilon$                       | `particles_resampled += ...`              |

---

## 33. 当前代码与 NEBP 原始代码的对应关系

当前 Particle BP 是从 NEBP 的核心粒子消息传递结构改写而来，但针对静态单快照进行了简化。

#### NEBP 原始代码中的核心结构

```text
init_particles()
        ↓
particle_predict()
        ↓
calculate_weight_anchor()
        ↓
calculate_weight_agent()
        ↓
calculate_belief()
        ↓
perform_estimation()
```

其中：

- `particle_predict()`：时间状态转移；
- `calculate_weight_anchor()`：anchor likelihood；
- `calculate_weight_agent()`：agent-agent message；
- `calculate_belief()`：belief aggregation；
- `perform_estimation()`：最终位置估计和 resampling。

---

## 当前代码

```text
Particle initialization
        ↓
Anchor likelihood
        ↓
belief^(0)
        ↓
┌─────────────────────┐
│  synchronous BP     │
│                     │
│  belief^(t)         │
│       ↓             │
│  agent messages     │
│       ↓             │
│  belief^(t+1)       │
└─────────────────────┘
        ↓
final belief
        ↓
mean + covariance
        ↓
final resampling
        ↓
optional roughening
```

主要区别：

### 1. 去掉运动模型

原 NEBP：

$$ \mathbf{x}_{t+1} = F\mathbf{x}_t + \mathbf{q}_t. $$

当前为静态快照，因此没有：

- $F$
- $Q$
- $W$
- `particle_predict()`

---

### 2. DGL → PyG

原始 NEBP：

```python
g.update_all(...)
```

当前：

```python
for e in range(n_agt):
    ...
```

因此消息聚合从 DGL graph primitive 改为显式 PyTorch tensor computation。

---

### 3. 当前版本使用同步 belief 更新

当前：

```python
belief_prev = F.softmax(belief_log, dim=1)
```

所有 message 都基于：

$$ b^{(t)} $$

计算，然后统一得到：

$$ b^{(t+1)}. $$

这是当前版本为了明确对应标准 BP 方程而采用的实现方式。

---

## 34. 与 GIB-guided Cooperative Localization 的整体数据流

你的整体方法可以表示为：

$$ \boxed{ \mathcal{G}_{\mathrm{dense}} \xrightarrow{\mathrm{GIB}} \mathcal{G}_{\mathrm{sparse}} \xrightarrow{\mathrm{Particle\ BP}} \{\hat{\mathbf{x}}_i\} } $$

更具体地：

```text
                 Dense Graph
                     │
                     │
                     ▼
             ┌───────────────┐
             │      GNN      │
             │   GIB-guided  │
             └───────┬───────┘
                     │
                     │ edge weights
                     ▼
               Sparse Graph
                     │
                     │ edge_weights
                     ▼
        ┌────────────────────────┐
        │      Particle BP       │
        │                        │
        │ Particle initialization│
        │          ↓             │
        │ Anchor likelihood      │
        │          ↓             │
        │ BP message passing     │
        │          ↓             │
        │ Final belief            │
        │          ↓             │
        │ Position estimation    │
        └────────────┬───────────┘
                     │
                     ▼
              Localization
              \hat{x}_i
```

因此 GIB 与 Particle BP 的职责是分离的：

$$ \boxed{ \mathrm{GIB}: \text{学习保留哪些边} } $$

而：

$$ \boxed{ \mathrm{Particle\ BP}: \text{在给定图上进行概率推断} } $$

这对于你的 baseline 设计尤其重要：NBP 不应该自己学习拓扑，它应该在指定的 Dense Graph 或 Sparse Graph 上执行相同的推断过程。

---

## 35. 总结

当前 Particle BP 的完整理论链条是：

$$
\boxed{
\begin{aligned}
&\text{Measurement Model}\\
&\downarrow\\
&p(z_{ij}\mid x_i,x_j)\\
&\downarrow\\
&\text{Particle Approximation}\\
&\downarrow\\
&b_i^{(t)}(x_i) \approx \sum_p w_i^{(p,t)} \delta(x_i-x_i^{(p)})\\
&\downarrow\\
&\text{Anchor Update}\\
&\downarrow\\
&\phi_i(x_i)\\
&\downarrow\\
&\text{Agent-Agent Message}\\
&\downarrow\\
&m_{j\rightarrow i}^{(t)}(x_i) = \int \psi_{ij}(x_i,x_j) b_j^{(t)}(x_j)dx_j\\
&\downarrow\\
&\text{Particle Message Approximation}\\
&\downarrow\\
&m_{j\rightarrow i}^{(t)}(x_i^{(p)}) \approx \sum_q w_j^{(q,t)} \psi_{ij}(x_i^{(p)},x_j^{(q)})\\
&\downarrow\\
&\text{Belief Update}\\
&\downarrow\\
&b_i^{(t+1)} \propto \phi_i \prod_jm_{j\rightarrow i}^{(t)}\\
&\downarrow\\
&\text{Repeat }T\text{ iterations}\\
&\downarrow\\
&\text{Final Belief}\\
&\downarrow\\
&\hat{x}_i = \sum_pw_i^{(p)}x_i^{(p)}\\
&\downarrow\\
&\hat{\Sigma}_i\\
&\downarrow\\
&\text{Final Resampling}\\
&\downarrow\\
&\text{Optional Roughening}
\end{aligned}
}
$$

这就是当前 `particle_bp.py` 所实现的完整 Particle-based BP 理论与数据流。