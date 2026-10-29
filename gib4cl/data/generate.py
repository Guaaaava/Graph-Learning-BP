"""Reproducible per-sample generation and versioned dataset persistence."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import torch

from gib4cl.config import (REPO_ROOT, ExperimentConfig, fingerprint, load_config,
                          sample_seed, validate_name, _integer)
from gib4cl.data.schema import DATASET_VERSION, representation_metadata, validate_bundle
from gib4cl.data.network import generate_graph


# `main` (command-line)
#  → `create_dataset`
#     → `build_dataset`
#       for i=0~num_graphs-1
#         → `sample_seed`: get seed of the sample
#         → `random.Random`: randomly choose the amount of agents（lo~hi）
#         → `generate_graph`: get a record
#       collect all records → pack bundle + metadata、fingerprint...
#     → `save_dataset`
#       → `validate_bundle`
#       → `torch.save`: save bundle to `.pt` file
#       → output `.json` file


# Reproducible design: Fingerprint the source code of the generated dataset itself for traceability
def _source_version():
    base = REPO_ROOT
    names = ('gib4cl/config.py','gib4cl/data/schema.py','gib4cl/data/network.py',
             'gib4cl/data/generate.py','gib4cl/data/dataset.py')
    hashes = {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name in names}
    try:
        # An unpacked package may live inside another Git repository. Only
        # record a revision when this source root owns its own Git checkout.
        head = (subprocess.check_output(['git', '-C', str(base), 'rev-parse', 'HEAD'],
                                       stderr=subprocess.DEVNULL, text=True).strip()
                if (base/'.git').exists() else None)
    except (OSError, subprocess.CalledProcessError):
        head = None
    return {'git_head': head, 'source_sha256': hashes,
            'source_fingerprint': fingerprint(hashes)}


# Construct bundles
# Metadata can serve as a reference for experimental conditions
def build_dataset(settings, split_name, num_graphs, seed_offset=0,
                  agents_min=None, agents_max=None):
    if not isinstance(settings, ExperimentConfig):
        raise TypeError('settings must be an ExperimentConfig')
    validate_name(split_name, 'split')
    _integer(num_graphs, 'num_graphs', 0)
    _integer(seed_offset, 'seed_offset', 0)
    lo = settings.data.num_agents_min if agents_min is None else agents_min
    hi = settings.data.num_agents_max if agents_max is None else agents_max
    _integer(lo, 'agents_min')
    _integer(hi, 'agents_max')
    if hi < lo:
        raise ValueError('agents_max must be >= agents_min')
    identity = {'experiment_id': settings.experiment_id,
                'data_fingerprint': settings.data_fingerprint, 'split': split_name,
                'seed_offset': seed_offset, 'agents_min': lo, 'agents_max': hi}
    series_id = fingerprint(identity)
    graphs = []
    for i in range(num_graphs):
        seed = sample_seed(settings.seeds.data, split_name, seed_offset, i)
        # Python's private RNG chooses size; torch's private RNG generates this graph.
        count = random.Random(seed).randint(lo, hi)
        generator = torch.Generator(device='cpu').manual_seed(seed)
        graphs.append(generate_graph(
            settings.data, count, generator=generator, sample_id=f'{series_id}:{i}',
            metadata={'sample_number': i, 'sample_seed': seed, 'split': split_name,
                      'series_id': series_id, 'data_fingerprint': settings.data_fingerprint}))
    metadata = {**identity, 'series_id': series_id,
                'dataset_id': fingerprint({'series_id': series_id, 'num_graphs': num_graphs}),
                'num_graphs': num_graphs, 'config': settings.to_dict(),
                'resolved_split_config': {'split': split_name, 'num_graphs': num_graphs,
                                          'agents_min': lo, 'agents_max': hi, 'seed_offset': seed_offset},
                'representation': representation_metadata(),
                'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
                'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__)},
                'code_version': _source_version(),
                'summary': {
                    'physical_edges': sum(g['inputs']['edge_index'].shape[1] for g in graphs),
                    'unanchored_agents': sum(g['diagnostics']['num_unanchored_agents'] for g in graphs),
                    'agents_below_degree_three': sum(g['diagnostics']['num_agents_below_degree_three'] for g in graphs),
                    'sample_seeds_unique': len({g['metadata']['sample_seed'] for g in graphs}) == num_graphs}}
    return {'schema_version': DATASET_VERSION, 'metadata': metadata, 'graphs': graphs}


def save_dataset(bundle, path, *, overwrite=False):
    path = Path(path)
    sidecar = path.with_suffix('.json')
    if not overwrite and (path.exists() or sidecar.exists()):
        raise FileExistsError(f'Dataset already exists: {path}; choose another experiment/output root or explicitly overwrite')
    validate_bundle(bundle, require_targets=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb' if overwrite else 'xb') as handle:
        torch.save(bundle, handle)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {'schema_version': DATASET_VERSION, 'file': path.name, 'sha256': checksum,
              'metadata': bundle['metadata']}
    with sidecar.open('w' if overwrite else 'x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    return path


def create_dataset(split_name, num_graphs, seed_offset=0, agents_min=None,
                   agents_max=None, *, settings=None, output_root=None, overwrite=False):
    """Legacy entry name, new versioned bundle/path; no global RNG resets."""
    settings = load_config() if settings is None else settings
    path = settings.dataset_path(split_name, output_root)
    # Refuse existing output before any expensive generation.
    if not overwrite and (path.exists() or path.with_suffix('.json').exists()):
        raise FileExistsError(f'Dataset already exists: {path}')
    bundle = build_dataset(settings, split_name, num_graphs, seed_offset, agents_min, agents_max)
    save_dataset(bundle, path, overwrite=overwrite)
    print(f"Saved {split_name}: graphs={num_graphs}, path={path}")
    return path


# Command-line entry
def main():
    parser = argparse.ArgumentParser(description='Generate versioned localization datasets')
    parser.add_argument('--config')
    parser.add_argument('--split', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--num-graphs', type=int)
    parser.add_argument('--seed-offset', type=int, default=0)
    parser.add_argument('--agents-min', type=int)
    parser.add_argument('--agents-max', type=int)
    parser.add_argument('--output-root')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    settings = load_config(args.config)
    for split in args.split:
        count = args.num_graphs if args.num_graphs is not None else settings.split_sizes.get(split)
        if count is None:
            parser.error(f'Provide --num-graphs for split {split}')
        create_dataset(split, count, args.seed_offset, args.agents_min, args.agents_max,
                       settings=settings, output_root=args.output_root, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
