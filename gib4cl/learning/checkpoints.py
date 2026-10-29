"""Versioned model/checkpoint loading shared by training and evaluation."""
from pathlib import Path
import math
import torch
from .model import EdgePredictorGNN
from gib4cl.config import config_from_dict
from gib4cl.runtime import CHECKPOINT_VERSION, sha256, state_digest


def make_model(config):
    return EdgePredictorGNN(config.node_in_dim,config.edge_in_dim,config.hidden_dim,config.num_layers)


def read_checkpoint(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint does not exist: {path}')
    value = torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(value,dict) or value.get('checkpoint_version') != CHECKPOINT_VERSION:
        raise ValueError('A versioned full checkpoint is required; evaluate best_checkpoint.pt, not a raw state_dict')
    required = {'model_state','optimizer_state','scheduler_state','config','epoch','history','rng_state',
                'best_metric','best_epoch','datasets','runtime','training_sources','run_dir'}
    if not required <= value.keys():
        raise ValueError('Incomplete training checkpoint')
    if state_digest(value['model_state']) != value.get('model_digest'):
        raise ValueError('Checkpoint model digest mismatch')
    epoch,history = value['epoch'],value['history']
    if (isinstance(epoch,bool) or not isinstance(epoch,int) or epoch < 1
            or not isinstance(history,list) or len(history) != epoch
            or [row.get('epoch') for row in history] != list(range(1,epoch+1))):
        raise ValueError('Checkpoint epoch/history is inconsistent')
    best = min(history,key=lambda row:row['val']['total'])
    if (not math.isfinite(value['best_metric']) or value['best_metric'] != best['val']['total']
            or value['best_epoch'] != best['epoch']):
        raise ValueError('Checkpoint best validation selection is inconsistent')
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
            'datasets':checkpoint['datasets']}


def warm_start(model,path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Initial weights do not exist: {path}')
    value = torch.load(path,map_location='cpu',weights_only=True)
    if isinstance(value,dict) and 'checkpoint_version' in value:
        value = read_checkpoint(path)['model_state']
    model.load_state_dict(value)  # Strict PyTorch loading checks parameter keys and tensor shapes.
    return {'path':str(path),'sha256':sha256(path),'model_digest':state_digest(model.state_dict())}
