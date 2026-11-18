"""Versioned model/checkpoint loading shared by training and evaluation."""
from pathlib import Path
import math
import torch
from .model import EdgePredictorGNN
from gib4cl.config import config_from_dict
from gib4cl.runtime import CHECKPOINT_VERSION, sha256, state_digest
from .validation import validation_protocol
from .diagnostics import diagnostic_protocol


def make_model(config):
    return EdgePredictorGNN(config.node_in_dim,config.edge_in_dim,config.hidden_dim,config.num_layers)


def read_checkpoint(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint does not exist: {path}')
    value = torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(value,dict) or value.get('checkpoint_version') not in (CHECKPOINT_VERSION, 'gib4cl.checkpoint.v1'):
        raise ValueError('A versioned full checkpoint is required; evaluate best_checkpoint.pt, not a raw state_dict')
    required = {'model_state','optimizer_state','scheduler_state','config','epoch','history','rng_state',
                'best_metric','best_epoch','datasets','runtime','training_sources','run_dir'}
    if not required <= value.keys():
        raise ValueError('Incomplete training checkpoint')
    if value['checkpoint_version'] == CHECKPOINT_VERSION:
        cfg = config_from_dict(value['config'])
        expected = validation_protocol(cfg, value['datasets']['val']['num_graphs'])
        if value.get('validation_protocol') != expected:
            raise ValueError('Checkpoint validation protocol is inconsistent')
    if state_digest(value['model_state']) != value.get('model_digest'):
        raise ValueError('Checkpoint model digest mismatch')
    epoch,history = value['epoch'],value['history']
    if (isinstance(epoch,bool) or not isinstance(epoch,int) or epoch < 1
            or not isinstance(history,list) or len(history) != epoch
            or [row.get('epoch') for row in history] != list(range(1,epoch+1))):
        raise ValueError('Checkpoint epoch/history is inconsistent')
    if any(not isinstance(row.get('val'), dict)
           or not isinstance(row['val'].get('total'), (int, float))
           or not math.isfinite(row['val']['total']) for row in history):
        raise ValueError('Checkpoint validation history must contain finite totals')
    if value['checkpoint_version'] == CHECKPOINT_VERSION:
        for row in history:
            sampling = row.get('val_sampling', {})
            monitor = row.get('val_deterministic', {}).get('total')
            prefix = sampling.get('prefix_total')
            gap = sampling.get('prefix_minus_selected_total')
            if (sampling.get('repeats') != cfg.training.validation_repeats
                    or sampling.get('prefix_repeats') != max(1,cfg.training.validation_repeats//2)
                    or not isinstance(monitor, (int,float)) or not math.isfinite(monitor)
                    or not isinstance(prefix, (int,float)) or not math.isfinite(prefix)
                    or gap != prefix-row['val']['total']):
                raise ValueError('Checkpoint validation diagnostics are inconsistent')
    best = min(history,key=lambda row:row['val']['total'])
    if (not math.isfinite(value['best_metric']) or value['best_metric'] != best['val']['total']
            or value['best_epoch'] != best['epoch']):
        raise ValueError('Checkpoint best validation selection is inconsistent')
    if 'training_diagnostics' in value:
        if value['training_diagnostics'] != diagnostic_protocol():
            raise ValueError('Checkpoint training diagnostics protocol is inconsistent')
        for row in history:
            diagnostic = row.get('train_diagnostics', {})
            steps, clipped = diagnostic.get('steps'), diagnostic.get('clipped_steps')
            if (diagnostic.get('version') != diagnostic_protocol()['version']
                    or isinstance(steps,bool) or not isinstance(steps,int) or steps < 1
                    or isinstance(clipped,bool) or not isinstance(clipped,int) or not 0 <= clipped <= steps
                    or diagnostic.get('clip_fraction') != clipped/steps
                    or diagnostic.get('gradient_nonfinite_elements') != 0):
                raise ValueError('Checkpoint training diagnostics history is inconsistent')
    return value


def load_selector(path,expected_model,device='cpu'):
    checkpoint = read_checkpoint(path)
    cfg = config_from_dict(checkpoint['config'])
    if cfg.model != expected_model:
        raise ValueError('Evaluation model configuration differs from checkpoint')
    model = make_model(expected_model)
    model.load_state_dict(checkpoint['model_state'])
    return model.to(device).eval(),checkpoint,{'path':str(Path(path).resolve()),
            'sha256':sha256(path),'epoch':checkpoint['epoch'],'best_epoch':checkpoint['best_epoch'],
            'model_digest':checkpoint['model_digest'],'training_config':checkpoint['config'],
            'datasets':checkpoint['datasets'], 'checkpoint_version':checkpoint['checkpoint_version'],
            'validation_protocol':checkpoint.get('validation_protocol',
                {'selection':'legacy_deterministic_validation_total'})}


def warm_start(model,path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Initial weights do not exist: {path}')
    value = torch.load(path,map_location='cpu',weights_only=True)
    if isinstance(value,dict) and 'checkpoint_version' in value:
        value = read_checkpoint(path)['model_state']
    model.load_state_dict(value)  # Strict PyTorch loading checks parameter keys and tensor shapes.
    return {'path':str(path),'sha256':sha256(path),'model_digest':state_digest(model.state_dict())}
