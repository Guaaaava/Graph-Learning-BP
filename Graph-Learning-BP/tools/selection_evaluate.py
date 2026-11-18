"""Shared proxy/BP validation records, with CPU-drawn paired particle supports."""
from __future__ import annotations
from dataclasses import replace
import math
import time
from pathlib import Path
import torch

from gib4cl.config import config_from_dict
from gib4cl.data.generate import create_dataset
from gib4cl.data.dataset import graph_to_data
from gib4cl.runtime import load_split, dataset_identity, state_digest, sha256, _io_path
from gib4cl.inference import ParticleBP, ParticleSupport
from gib4cl.inference.spanning_tree import bfs_forest
from gib4cl.learning.checkpoints import load_selector
from gib4cl.learning.objectives import gaussian_predictions
from gib4cl.learning.validation import stream_seed
from gib4cl.evaluation.metrics import calculate_metrics, bp_diagnostic_metrics
from gib4cl.evaluation.topology import selection_metrics
from tools.selection_state import freeze_copy


def prepare_data(workspace,cfg,*,source_root=None,source_paths=None):
    data_root = workspace.root/'data'
    result = {}
    for split in ('train','val'):
        path = cfg.dataset_path(split,data_root)
        if source_root is not None or source_paths is not None:
            source = cfg.dataset_path(split,source_root) if source_root is not None else source_paths[split]
            source = Path(source)
            if not _io_path(source).is_file() or not _io_path(source.with_suffix('.json')).is_file():
                raise ValueError('Original dataset unavailable; pass --data-root with the exact dataset and sidecar files')
            freeze_copy(source,path)
            freeze_copy(source.with_suffix('.json'),path.with_suffix('.json'))
        if not path.exists():
            if path.with_suffix('.json').exists():
                raise ValueError('Incomplete dataset pair: '+str(path))
            create_dataset(split,cfg.split_sizes[split],settings=cfg,output_root=data_root)
        dataset,identity = load_split(cfg,split,data_root,training=True)
        if len(dataset) != cfg.split_sizes[split]:
            raise ValueError('Candidate training requires the full configured split')
        result[split] = dataset_identity(identity)
    workspace.bind('datasets',result)
    return data_root


def masks(data):
    return {'dense':torch.ones(data.num_edges,device=data.x.device),
            'bfs':bfs_forest(data.num_nodes,data.edge_index,data.anchor_node_index).edge_weights.to(data.x.device)}


def support_for(data,cfg,seed,particles,device):
    cpu = data.clone().cpu()
    bp = ParticleBP.from_config(replace(cfg.bp,num_particles=particles,num_iterations=0))
    support = bp.draw_support(cpu,generator=torch.Generator().manual_seed(stream_seed('bp-budget',seed,data.sample_id)))
    digest = state_digest({'points':support.points,'masses':support.log_prior_weights})
    return ParticleSupport(support.points.to(device),support.log_prior_weights.to(device),
                           support.agent_node_index.to(device)),digest


def metric_record(data,mask,mean,covariance,cfg,*,prediction=None):
    metrics = calculate_metrics(mean,data.y,covariance,cfg.evaluation.outage_threshold,cfg.evaluation.alpha,
        covariance_rtol=cfg.evaluation.covariance_rtol,outage_thresholds=cfg.evaluation.outage_thresholds,
        coverage_levels=tuple(sorted(set(cfg.evaluation.coverage_levels)|{.95})))
    metrics.update(selection_metrics(data,mask))
    if prediction is not None:
        metrics.update(bp_diagnostic_metrics(prediction,prediction.weights.shape[1],cfg.evaluation.ess_fraction_threshold))
    return {'status':'complete','sample_id':data.sample_id,
            'input_digest':state_digest(data.clone().cpu().to_dict()),
            'mse':float((mean.double()-data.y.double()).square().sum(dim=1).mean()),
            'coverage95':next(r['coverage'] for r in metrics['coverage_curve'] if r['nominal']==.95),
            'communication_cost':metrics['communication_cost'],
            'invalid_covariance_fraction':metrics['invalid_covariance_rate'],
            'mean':mean.detach().cpu().tolist(),'covariance':covariance.detach().cpu().tolist(),
            'selected_edge_ids':torch.where(mask==1)[0].cpu().tolist(),'metrics':metrics}


@torch.no_grad()
def gaussian_moments(data,mask,cfg):
    prediction, = gaussian_predictions(data,mask,variance_floor=cfg.objective.prediction_variance_floor,
                                       compute_dtype=cfg.objective.compute_dtype)
    joint = prediction.covariance
    # Marginal blocks of the inverse JOINT precision, never inverse diagonal blocks.
    covariance = torch.stack([joint[2*i:2*i+2,2*i:2*i+2] for i in range(len(prediction.mean))])
    return prediction.mean,covariance


@torch.no_grad()
def evaluate_panel(workspace,cfg,model_specs,*,graphs,method,device,bp_settings=None):
    if method not in ('gaussian_proxy','particle_bp'):
        raise ValueError('Unknown evaluator')
    data_root = workspace.root/'data'
    dataset,identity = load_split(cfg,'val',data_root)
    if not 1 <= graphs <= len(dataset):
        raise ValueError('Validation prefix must fit the fixed val split')
    base = [{'model_id':k,'candidate_id':k,'baseline':True,'training_seed':None,'coefficients':None}
            for k in ('dense','bfs')]
    specs = base+model_specs
    loaded = {}
    for spec in model_specs:
        path = workspace.verify(spec['checkpoint'])
        model,saved,_ = load_selector(path,cfg.model,device)
        checkpoint_cfg = config_from_dict(saved['config'])
        if checkpoint_cfg.data_fingerprint != cfg.data_fingerprint or checkpoint_cfg.objective.require_ready().beta != spec['coefficients']['beta']:
            raise ValueError('Checkpoint scenario/objective differs from candidate')
        if any(getattr(checkpoint_cfg.objective,k) != v for k,v in spec['coefficients'].items()) or checkpoint_cfg.seeds.training != spec['training_seed']:
            raise ValueError('Checkpoint coefficients or training seed mismatch')
        for split in ('train','val'):
            current = workspace.manifest['requests']['datasets'][split]
            if dataset_identity(saved['datasets'][split]) != current:
                raise ValueError('Checkpoint dataset differs from fixed candidate data')
        loaded[spec['model_id']] = model
    seeds = bp_settings['seeds'] if method=='particle_bp' else [None]
    context = {'method':method,'dataset':dataset_identity(identity),'data_fingerprint':cfg.data_fingerprint,
               'base_config':cfg.to_dict(),
               'evaluation':{'threshold':cfg.evaluation.edge_threshold,'alpha':cfg.evaluation.alpha,
                    'covariance_rtol':cfg.evaluation.covariance_rtol,'coverage_levels':list(cfg.evaluation.coverage_levels),
                    'outage_thresholds':list(cfg.evaluation.outage_thresholds),'outage_threshold':cfg.evaluation.outage_threshold},
               'bp_settings':bp_settings,'bp_seeds':seeds,
               'training_seeds':sorted(set(s['training_seed'] for s in model_specs)),
               'models':specs,'sample_ids':[dataset[i].sample_id for i in range(graphs)],
               'input_digests':{dataset[i].sample_id:state_digest(dataset[i].to_dict()) for i in range(graphs)},
               'sources':workspace.identity['sources']}
    records = []
    for index in range(graphs):
        data = dataset[index].to(device)
        mask_map = masks(data)
        for spec in model_specs:
            logits = loaded[spec['model_id']].score_edges(data)
            threshold = cfg.evaluation.edge_threshold
            mask_map[spec['model_id']] = (logits > math.log(threshold/(1-threshold))).to(logits.dtype)
        for seed in seeds:
            support,digest = (support_for(data,cfg,seed,bp_settings['particles'],device)
                              if method=='particle_bp' else (None,None))
            for spec in specs:
                key = {'type':'prediction','method':method,'sample_id':data.sample_id,
                       'input_digest':context['input_digests'][data.sample_id],
                       'mask_digest':state_digest({'mask':mask_map[spec['model_id']]}),
                       'checkpoint_sha256':spec['checkpoint']['sha256'] if spec.get('checkpoint') else None,'model_id':spec['model_id'],
                       'bp_settings':{k:v for k,v in bp_settings.items() if k!='seeds'} if bp_settings else None,
                       'bp_seed':seed,'support_digest':digest,
                       'evaluation':context['evaluation']}
                def predict():
                    prediction = None
                    if method=='particle_bp':
                        bp = ParticleBP.from_config(replace(cfg.bp,num_particles=bp_settings['particles'],
                            num_iterations=bp_settings['iterations'],chunk_size=bp_settings['chunk_size']))
                        if device.type=='cuda':
                            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                        begin = time.perf_counter()
                        prediction = bp.infer(data,mask_map[spec['model_id']],support=support)
                        if device.type=='cuda': torch.cuda.synchronize()
                        seconds = time.perf_counter()-begin
                        mean,covariance = prediction.mean,prediction.covariance
                    else:
                        begin = time.perf_counter()
                        mean,covariance = gaussian_moments(data,mask_map[spec['model_id']],cfg)
                        seconds = time.perf_counter()-begin
                    row = metric_record(data,mask_map[spec['model_id']],mean,covariance,cfg,prediction=prediction)
                    row.update(model_id=spec['model_id'],bp_seed=seed,support_digest=digest,
                               inference_seconds=seconds,
                               cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if method=='particle_bp' and device.type=='cuda' else None)
                    if prediction is not None:
                        row.update(ess=prediction.ess.cpu().tolist(),maximum_weight=prediction.weights.max(dim=1).values.cpu().tolist(),
                                   message_residual=prediction.message_residual,actual_iterations=prediction.iterations,
                                   converged=prediction.converged)
                    return row,[]
                records.append(workspace.job(key,predict))
    return {'schema':'gib4cl.selection-records.v1','status':'complete','context':context,'records':records}
