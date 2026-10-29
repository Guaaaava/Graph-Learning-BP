"""Shared execution, provenance and persistence helpers; imports have no effects."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import tempfile
import numpy as np
import torch
import torch_geometric

from gib4cl.data.dataset import LocalizationDataset
from gib4cl.config import REPO_ROOT

CHECKPOINT_VERSION = 'gib4cl.checkpoint.v1'

# The runtime tool specifically addresses the most troublesome issues in academic papers:
# non-reproducible results, dataset confusion, corrupted saved files, forgetting to record code changes...

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# Atomic write is to first write to a temporary file, and then replace the target file at one time
def atomic_json(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,name = tempfile.mkstemp(dir=path.parent,prefix='.tmp-',suffix=path.suffix)
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        os.replace(temporary,path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_torch(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,name = tempfile.mkstemp(dir=path.parent,prefix='.tmp-',suffix=path.suffix)
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(value,temporary)
        os.replace(temporary,path)
    finally:
        if temporary.exists():
            temporary.unlink()


def execution(device='auto',threads=None,deterministic=True,seed=None):
    if threads is not None:
        if isinstance(threads,bool) or not isinstance(threads,int) or threads < 1:
            raise ValueError('threads must be a positive integer')
        torch.set_num_threads(threads)
    if device not in ('auto','cpu','cuda'):
        raise ValueError('device must be auto, cpu or cuda')
    device = 'cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu' if device == 'auto' else device
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(deterministic)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed % (2**63))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed % (2**63))
    return torch.device(device)


def runtime_info(device):
    return {'python':platform.python_version(),'torch':str(torch.__version__),
            'torch_geometric':str(torch_geometric.__version__),'system':platform.system(),
            'machine':platform.machine(),
            'numpy':str(np.__version__),'device':str(device),'num_threads':torch.get_num_threads(),
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}


def training_sources():
    names = ('learning/model.py','learning/objectives.py','learning/train.py','runtime.py',
             'learning/checkpoints.py','config.py','data/schema.py','data/dataset.py')
    return {name:sha256(REPO_ROOT/'gib4cl'/name) for name in names}


def source_manifest():
    return {path.relative_to(REPO_ROOT).as_posix():sha256(path)
            for path in sorted((REPO_ROOT/'gib4cl').rglob('*.py'))}


def load_split(cfg,split,data_root=None,*,training=False):
    path = cfg.dataset_path(split,data_root).resolve()
    dataset = LocalizationDataset(path)
    if not len(dataset):
        raise ValueError(f'{split} dataset must contain at least one graph')
    if dataset.metadata.get('split') != split or dataset.metadata.get('data_fingerprint') != cfg.data_fingerprint:
        raise ValueError(f'Dataset split/config identity mismatch: {path}')
    return dataset,{'path':str(path),'sha256':sha256(path),'dataset_id':dataset.metadata['dataset_id'],
                    'series_id':dataset.metadata['series_id'],'split':split,'num_graphs':len(dataset),
                    'data_fingerprint':cfg.data_fingerprint,
                    'resolved_split_config':dataset.metadata['resolved_split_config'],
                    'generation_sources':dataset.metadata['code_version'],
                    'representation':dataset.metadata['representation']}


def dataset_identity(value):
    return {key:value[key] for key in ('sha256','dataset_id','series_id','split','num_graphs','data_fingerprint')}


# Save the complete state of all current random number generators (Resume training at breakpoints, precise reproduction...)
def capture_rng(loaders):
    np_state = np.random.get_state()
    return {'python':random.getstate(),'numpy':{'kind':np_state[0],
            'keys':torch.from_numpy(np_state[1].astype(np.int64)),'position':np_state[2],
            'has_gauss':np_state[3],'cached_gaussian':np_state[4]},
            'torch_cpu':torch.random.get_rng_state(),
            'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'loaders':{key:generator.get_state() for key,generator in loaders.items()}}


def restore_rng(state,loaders):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n['kind'],n['keys'].numpy().astype(np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))
    torch.random.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:
        if not torch.cuda.is_available() or len(state['torch_cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA RNG/device count differs from checkpoint; use init-from for a new run')
        torch.cuda.set_rng_state_all(state['torch_cuda'])
    for key,generator in loaders.items():
        generator.set_state(state['loaders'][key])


def state_digest(state):
    h = hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode())
        if isinstance(value,torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            h.update(str((str(tensor.dtype),list(tensor.shape))).encode())
            h.update(tensor.numpy().tobytes())
        else:
            h.update(json.dumps(value,sort_keys=True).encode())
    return h.hexdigest()
