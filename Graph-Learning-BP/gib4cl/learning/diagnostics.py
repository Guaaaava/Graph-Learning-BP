"""Observations of existing training steps; no extra backward or optimizer step."""
from __future__ import annotations

import math
from pathlib import Path
import torch

from gib4cl.runtime import atomic_json


DIAGNOSTICS_VERSION = 'gib4cl.training-diagnostics.v1'


def diagnostic_protocol():
    return {'version': DIAGNOSTICS_VERSION, 'step_unit': 'optimizer_batch',
            'probability_low': .01, 'probability_high': .99,
            'clip_trigger': 'actual_clip_scale_less_than_one',
            'clip_scale': 'min(1,max_grad_norm/(returned_preclip_norm+1e-6))',
            'norms': 'L2; measured norms use detached float64 tensors',
            'parameter_scope': 'all_optimizer_parameters',
            'decay_scope': 'AdamW nominal lr*weight_decay*prestep_parameter, only grad-present parameters',
            'epoch_gradient_weighting': 'equal_optimizer_steps',
            'epoch_selection_weighting': 'equal_graphs; probabilities pooled over physical edges',
            'timings_in_reproducible_history': False}


def _norm(tensors):
    values = [torch.linalg.vector_norm(value.detach().to(dtype=torch.float64)) for value in tensors]
    return float(torch.linalg.vector_norm(torch.stack(values))) if values else 0.


def _distribution(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return {'count': 0, 'mean': None, 'p50': None, 'p95': None, 'max': None}
    def quantile(p):
        index = (len(values)-1)*p
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo]+(values[hi]-values[lo])*(index-lo)
    return {'count': len(values), 'mean': math.fsum(values)/len(values),
            'p50': quantile(.5), 'p95': quantile(.95), 'max': values[-1]}


@torch.no_grad()
def selection_statistics(batch, weights, logits):
    """Actual hard selections, counting each physical edge once."""
    graph_count = int(batch.num_graphs)
    membership = batch.batch
    u, v = batch.edge_index
    edge_graph = membership[u]
    active = weights.detach().to(dtype=torch.float64)
    probabilities = logits.detach().sigmoid().to(dtype=torch.float64)
    def per_graph(values):
        return values.new_zeros(graph_count).index_add_(0, edge_graph, values)
    edges = per_graph(torch.ones_like(active))
    retained = per_graph(active)
    nonempty = edges > 0
    degree = active.new_zeros(batch.num_nodes).index_add_(0,u,active).index_add_(0,v,active)
    ones = torch.ones_like(active)
    candidate_degree = active.new_zeros(batch.num_nodes).index_add_(0,u,ones).index_add_(0,v,ones)
    agents = batch.agent_node_index
    triangles = batch.triangle_edge_index
    triangle_retained = active[triangles].prod(dim=0)
    return {'graphs': graph_count, 'agents': int(agents.numel()),
            'candidate_edges': int(active.numel()), 'selected_edges': float(active.sum()),
            'retention_fraction_graph_mean': float((retained[nonempty]/edges[nonempty]).mean()) if nonempty.any() else None,
            'graphs_with_candidate_edges': int(nonempty.sum()),
            'selected_cost_graph_mean': float(per_graph(active*batch.edge_cost.detach().double()).mean()),
            'selected_triangles_graph_mean': float(triangle_retained.sum())/graph_count,
            'agents_below_degree_three': int((degree[agents] < 3).sum()),
            'degree_deficit_graph_mean': float(torch.relu(3-degree[agents]).sum())/graph_count,
            'unavoidable_degree_deficit_graph_mean': float(torch.relu(3-candidate_degree[agents]).sum())/graph_count,
            'probability_mean': float(probabilities.mean()) if probabilities.numel() else None,
            'probability_min': float(probabilities.min()) if probabilities.numel() else None,
            'probability_max': float(probabilities.max()) if probabilities.numel() else None,
            'probability_low_count': int((probabilities < .01).sum()),
            'probability_high_count': int((probabilities > .99).sum())}


def _json_safe(value, path='', nonfinite=None):
    if nonfinite is None:
        nonfinite = {}
    if isinstance(value, float) and not math.isfinite(value):
        nonfinite[path] = str(value)
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item, path+'.'+key, nonfinite) for key,item in value.items()}
    if isinstance(value, (list,tuple)):
        return [_json_safe(item, path+f'[{i}]', nonfinite) for i,item in enumerate(value)]
    return value


class EpochDiagnostics:
    def __init__(self, model, optimizer, *, epoch, max_grad_norm, directory):
        self.model, self.optimizer = model, optimizer
        self.epoch, self.max_grad_norm = epoch, max_grad_norm
        self.directory = Path(directory)
        self.rows = []
        self.current = None
        self.before = None
        self.parameters = [p for group in optimizer.param_groups for p in group['params']]

    def begin(self, batch_index, batch):
        self.current = {'version': DIAGNOSTICS_VERSION, 'status': 'running',
                        'epoch': self.epoch, 'batch': batch_index,
                        'sample_ids': list(batch.sample_id), 'stage': 'forward',
                        'optimizer_step_started': False, 'optimizer_step_applied': False}

    def stage(self, name):
        self.current['stage'] = name
        if name == 'optimizer_step':
            self.current['optimizer_step_started'] = True
            self.current['optimizer_step_applied'] = None

    def forward(self, batch, weights, logits, metrics):
        self.current['loss_components'] = dict(metrics)
        self.current['selection'] = selection_statistics(batch,weights,logits)

    @torch.no_grad()
    def gradients(self):
        parameters = list(self.model.parameters())
        gradients = [p.grad for p in parameters if p.grad is not None]
        self.current['gradient'] = {
            'parameters_without_gradient': sum(p.grad is None for p in parameters),
            'elements': sum(g.numel() for g in gradients),
            'nonfinite_elements': sum(int((~torch.isfinite(g)).sum()) for g in gradients),
            'measured_pre_clip_norm': _norm(gradients), 'threshold': self.max_grad_norm}

    @torch.no_grad()
    def clipped(self, returned_norm):
        gradient = self.current['gradient']
        # This is the existing clip_grad_norm_ scale, including its epsilon.
        scale = float(torch.clamp(self.max_grad_norm/(returned_norm+1e-6),max=1.))
        gradient.update(returned_pre_clip_norm=float(returned_norm), clip_scale=scale,
                        clip_applied=scale < 1., norm_above_threshold=float(returned_norm) > self.max_grad_norm,
                        post_clip_norm=_norm(p.grad for p in self.model.parameters() if p.grad is not None))

    @torch.no_grad()
    def before_step(self):
        self.before = [p.detach().clone() for p in self.parameters]
        groups, decay = [], []
        offset = 0
        is_adamw = isinstance(self.optimizer,torch.optim.AdamW)
        for group in self.optimizer.param_groups:
            parameters = group['params']
            saved = self.before[offset:offset+len(parameters)]
            lr, weight_decay = float(group['lr']), float(group.get('weight_decay',0.))
            active = [old for p,old in zip(parameters,saved) if p.grad is not None]
            if is_adamw:
                decay.extend(old.double()*(lr*weight_decay) for old in active)
            groups.append({'learning_rate': lr, 'weight_decay': weight_decay,
                           'parameters': len(parameters), 'parameters_with_gradient': len(active)})
            offset += len(parameters)
        self.current['update'] = {'parameter_norm_before': _norm(self.before),
                                  'optimizer': type(self.optimizer).__name__, 'groups': groups,
                                  'nominal_adamw_decay_norm': _norm(decay) if is_adamw else None}

    @torch.no_grad()
    def after_step(self):
        self.current['optimizer_step_applied'] = True
        update = self.current['update']
        update['parameter_nonfinite_elements'] = sum(int((~torch.isfinite(p)).sum()) for p in self.parameters)
        update['parameter_norm_after'] = _norm(self.parameters)
        update['actual_step_norm'] = _norm(p.detach().double()-old.double() for p,old in zip(self.parameters,self.before))
        before_norm, step_norm = update['parameter_norm_before'], update['actual_step_norm']
        update['relative_step_norm'] = step_norm/before_norm if before_norm > 0 else None
        nominal = update['nominal_adamw_decay_norm']
        update['nominal_decay_to_actual_step_ratio'] = nominal/step_norm if nominal is not None and step_norm > 0 else None
        self.before = None
        if update['parameter_nonfinite_elements']:
            raise FloatingPointError('Optimizer produced nonfinite parameters')

    def finish(self):
        self.current.update(status='complete',stage='complete')
        atomic_json(self.directory/f'epoch-{self.epoch:04d}'/f'batch-{self.current["batch"]:05d}.json',self.current)
        self.rows.append(self.current)
        self.current = None

    def failure(self, error):
        row = dict(self.current or {'epoch':self.epoch})
        row.update(status='failed',error_type=type(error).__name__,error=str(error))
        nonfinite = {}
        row = _json_safe(row, nonfinite=nonfinite)
        row['nonfinite_fields'] = nonfinite
        directory = self.directory/'failures'
        attempt = 1
        path = directory/f'epoch-{self.epoch:04d}-batch-{row.get("batch",0):05d}-attempt-{attempt:03d}.json'
        while path.exists():
            attempt += 1
            path = directory/f'epoch-{self.epoch:04d}-batch-{row.get("batch",0):05d}-attempt-{attempt:03d}.json'
        atomic_json(path,row)
        self.before = None

    def summary(self):
        rows = self.rows
        if not rows:
            raise ValueError('Cannot summarize an empty diagnostic epoch')
        selections = [row['selection'] for row in rows]
        graphs, edges = sum(s['graphs'] for s in selections), sum(s['candidate_edges'] for s in selections)
        nonempty = sum(s['graphs_with_candidate_edges'] for s in selections)
        def graph_mean(key):
            return math.fsum(s[key]*s['graphs'] for s in selections)/graphs
        return {'version': DIAGNOSTICS_VERSION, 'steps': len(rows),
                'clipped_steps': sum(row['gradient']['clip_applied'] for row in rows),
                'clip_fraction': sum(row['gradient']['clip_applied'] for row in rows)/len(rows),
                'gradient_nonfinite_elements': sum(row['gradient']['nonfinite_elements'] for row in rows),
                'pre_clip_norm': _distribution(row['gradient']['returned_pre_clip_norm'] for row in rows),
                'post_clip_norm': _distribution(row['gradient']['post_clip_norm'] for row in rows),
                'clip_scale': _distribution(row['gradient']['clip_scale'] for row in rows),
                'actual_step_norm': _distribution(row['update']['actual_step_norm'] for row in rows),
                'relative_step_norm': _distribution(row['update']['relative_step_norm'] for row in rows),
                'nominal_adamw_decay_norm': _distribution(row['update']['nominal_adamw_decay_norm'] for row in rows),
                'nominal_decay_to_actual_step_ratio': _distribution(row['update']['nominal_decay_to_actual_step_ratio'] for row in rows),
                'parameter_norm_before': _distribution(row['update']['parameter_norm_before'] for row in rows),
                'parameter_norm_after': _distribution(row['update']['parameter_norm_after'] for row in rows),
                'selection': {'graphs': graphs, 'candidate_edges': edges,
                    'selected_edges': math.fsum(s['selected_edges'] for s in selections),
                    'retention_fraction_graph_mean': math.fsum(s['retention_fraction_graph_mean']*s['graphs_with_candidate_edges']
                        for s in selections if s['graphs_with_candidate_edges'])/nonempty if nonempty else None,
                    'graphs_with_candidate_edges': nonempty,
                    'selected_cost_graph_mean': graph_mean('selected_cost_graph_mean'),
                    'selected_triangles_graph_mean': graph_mean('selected_triangles_graph_mean'),
                    'degree_deficit_graph_mean': graph_mean('degree_deficit_graph_mean'),
                    'unavoidable_degree_deficit_graph_mean': graph_mean('unavoidable_degree_deficit_graph_mean'),
                    'agents': sum(s['agents'] for s in selections),
                    'agents_below_degree_three': sum(s['agents_below_degree_three'] for s in selections),
                    'probability_mean': math.fsum(s['probability_mean']*s['candidate_edges'] for s in selections if s['candidate_edges'])/edges if edges else None,
                    'probability_low_fraction': sum(s['probability_low_count'] for s in selections)/edges if edges else None,
                    'probability_high_fraction': sum(s['probability_high_count'] for s in selections)/edges if edges else None}}
