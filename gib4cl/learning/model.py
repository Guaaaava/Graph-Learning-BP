"""Paper chapter 4: bidirectional messages, symmetric physical-edge scores, STE.

Only x, edge_attr and the physical/message-edge indices are read. Labels and
diagnostics are not model inputs. A physical edge produces exactly one logit.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

# tau: temperature, the smaller it is, the sharper `soft` becomes, the closer it approaches 0/1
# logit: maps the probability p∈(0,1) to number∈R, logit(p)=log(p/(1-p))
# sigmoid: maps real number∈R back to probabilities p∈(0,1), σ(z)=1/(1+e^(-z))
#           p = σ(logit(p))
def bernoulli_sigmoid(logits, tau=1.0, hard=True, training=True, *,
                      generator=None, uniform=None):
    """Independent Logistic relaxation; hard training samples have law Bernoulli(p).

    p=sigmoid(logits), L=log(u)-log(1-u), soft=sigmoid((logits+L)/tau).
    The hard forward decision is logits+L>0; its gradient follows soft (biased
    STE). Temperature changes the relaxation, not the Bernoulli hard-sample law.

    Evaluation consumes no randomness: hard=True returns logit>0, hard=False
    returns p, independent of tau. Optional uniforms must lie strictly in (0,1)
    and match logits; they support reproducible paired comparisons/tests.
    """
    if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
        raise TypeError('logits must be a floating tensor')
    if not torch.isfinite(logits).all():
        raise ValueError('logits must be finite')
    if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not math.isfinite(tau):
        raise ValueError('tau must be a finite positive scalar')
    limits = torch.finfo(logits.dtype)
    if not limits.tiny <= tau <= limits.max:
        raise ValueError('tau must be positive and numerically representable in logits dtype')
    if not isinstance(hard, bool) or not isinstance(training, bool):
        raise TypeError('hard and training must be boolean')
    if uniform is not None and generator is not None:
        raise ValueError('Pass uniform or generator, not both')
    if not training:
        if uniform is not None or generator is not None:
            raise ValueError('Evaluation is deterministic; noise inputs require training=True')
        soft = torch.sigmoid(logits)
        perturbed = logits
    else:
        if uniform is None:
            uniform = torch.rand(logits.shape, dtype=logits.dtype,
                                 device=logits.device, generator=generator)
            # torch.rand is [0,1). Reject the possible zero rather than clip the
            # distribution or evaluate log(0); all other representable draws stay.
            zero = uniform == 0
            while zero.any():
                uniform[zero] = torch.rand((int(zero.sum()),), dtype=logits.dtype,
                                          device=logits.device, generator=generator)
                zero = uniform == 0
        elif (not isinstance(uniform, torch.Tensor) or uniform.shape != logits.shape
              or uniform.dtype != logits.dtype or uniform.device != logits.device):
            raise ValueError('uniform must match logits shape, dtype and device')
        if not (torch.isfinite(uniform).all() and ((uniform > 0) & (uniform < 1)).all()):
            raise ValueError('uniform must be finite and strictly inside (0,1)')
        noise = uniform.detach().log() - torch.log1p(-uniform.detach()) # Logistic noise L
        perturbed = logits + noise
        soft = torch.sigmoid(perturbed / tau)
    if not hard:
        return soft
    # Comparing before sigmoid avoids rounding a tiny positive logit to 0.5.
    binary = (perturbed > 0).to(logits.dtype)
    return binary + (soft - soft.detach()) # STE


def gumbel_sigmoid(logits, tau=1.0, hard=True, training=True, *,
                   generator=None, uniform=None):
    """Compatibility name: now uses Logistic noise (difference of two Gumbels)."""
    return bernoulli_sigmoid(logits, tau, hard, training,
                             generator=generator, uniform=uniform)

# Inherit from `MessagePassing`
class GNNLayer(MessagePassing):
    """m_ij=MLP([receiver_i,sender_j,e]); h_i'=MLP([h_i,sum_j m_ij]).

    Both MLPs have Linear -> LayerNorm -> ReLU -> Linear. There is no
    additional residual update. Edge features remain the raw four features.
    """

    def __init__(self, node_in_dim, edge_in_dim, hidden_dim):
        super().__init__(aggr='add', flow='source_to_target')
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * node_in_dim + edge_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.upd_mlp = nn.Sequential(
            nn.Linear(node_in_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr,
                              size=(x.shape[0], x.shape[0]))

    def message(self, x_i, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=1))

    def update(self, aggr_out, x):
        return self.upd_mlp(torch.cat([x, aggr_out], dim=1))


# Inherit from `nn.Module`
class EdgePredictorGNN(nn.Module):
    """One symmetric score and one decision per undirected measurement edge."""

    def __init__(self, node_in_dim=5, edge_in_dim=4, hidden_dim=64, num_layers=3):
        super().__init__()
        if node_in_dim != 5 or edge_in_dim != 4:
            raise ValueError('The paper requires 5 node and 4 edge features')
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim < 2:
            raise ValueError('hidden_dim must be an integer >= 2')
        if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError('num_layers must be an integer >= 1')
        self.layers = nn.ModuleList([
            GNNLayer(node_in_dim if layer == 0 else hidden_dim, edge_in_dim, hidden_dim)
            for layer in range(num_layers)])
        self.edge_scorer = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1))

    @staticmethod
    def _validate_inputs(data):
        required = ('x', 'edge_attr', 'edge_index', 'message_edge_index',
                    'message_edge_to_measurement', 'message_reverse_index')
        if any(getattr(data, name, None) is None for name in required):
            raise ValueError('Expected v1 graph with explicit bidirectional message-edge mappings')
        x, e, physical = data.x, data.edge_attr, data.edge_index
        arcs, mapping, reverse = (data.message_edge_index,
                                 data.message_edge_to_measurement, data.message_reverse_index)
        if x.ndim != 2 or x.shape[1] != 5 or e.ndim != 2 or e.shape[1] != 4:
            raise ValueError('Expected x=(N,5), edge_attr=(E,4)')
        if (not x.is_floating_point() or e.dtype != x.dtype
                or not torch.isfinite(x).all() or not torch.isfinite(e).all()):
            raise ValueError('Node and edge features must share a finite floating dtype')
        count, nodes = e.shape[0], x.shape[0]
        if physical.shape != (2, count) or arcs.shape != (2, 2 * count):
            raise ValueError('Expected one physical edge and two message arcs per measurement')
        if mapping.shape != (2 * count,) or reverse.shape != (2 * count,):
            raise ValueError('Invalid message mapping/reverse shape')
        for value in (e, physical, arcs, mapping, reverse):
            if value.device != x.device:
                raise ValueError('All model inputs must share a device')
        for value in (physical, arcs, mapping, reverse):
            if value.dtype != torch.long:
                raise ValueError('Edge indices and mappings must be torch.long')
        if count == 0:
            return
        if (physical.min() < 0 or physical.max() >= nodes or arcs.min() < 0 or arcs.max() >= nodes
                or mapping.min() < 0 or mapping.max() >= count
                or reverse.min() < 0 or reverse.max() >= 2 * count):
            raise ValueError('Edge index or message mapping out of bounds')
        lo, hi = physical.min(dim=0).values, physical.max(dim=0).values
        if (lo == hi).any() or torch.unique(lo * nodes + hi).numel() != count:
            raise ValueError('Physical edges must be unique, without self-loops')
        endpoints = physical[:, mapping]
        valid_arc = (arcs == endpoints).all(dim=0) | (arcs == endpoints.flip(0)).all(dim=0)
        if not valid_arc.all() or not (torch.bincount(mapping, minlength=count) == 2).all():
            raise ValueError('Message arcs must map to their physical measurement')
        if (not torch.equal(arcs[:, reverse], arcs.flip(0))
                or not torch.equal(mapping[reverse], mapping)
                or not torch.equal(reverse[reverse], torch.arange(2 * count, device=x.device))):
            raise ValueError('Each message arc must have its paired reverse arc')
        batch = getattr(data, 'batch', None)
        if batch is not None and (batch[physical[0]] != batch[physical[1]]).any():
            raise ValueError('Physical edges cannot connect separate graphs in a batch')

    def node_embeddings(self, data):
        self._validate_inputs(data)
        hidden = data.x
        message_features = data.edge_attr[data.message_edge_to_measurement]
        for layer in self.layers:
            hidden = layer(hidden, data.message_edge_index, message_features)
        return hidden

    def score_edges(self, data):
        """Symmetrize logits, not probabilities; no sampling or label access."""
        hidden = self.node_embeddings(data)
        row, col = data.edge_index
        forward = self.edge_scorer(torch.cat([hidden[row], hidden[col], data.edge_attr], dim=1))
        backward = self.edge_scorer(torch.cat([hidden[col], hidden[row], data.edge_attr], dim=1))
        return (0.5 * (forward + backward)).squeeze(-1)

    def edge_probabilities(self, data):
        """Temperature-independent Bernoulli parameters, in physical-edge order."""
        return torch.sigmoid(self.score_edges(data))

    def forward(self, data, tau=1.0, hard=True, *, generator=None, uniform=None):
        logits = self.score_edges(data)
        weights = bernoulli_sigmoid(logits, tau=tau, hard=hard, training=self.training,
                                    generator=generator, uniform=uniform)
        return weights, logits
