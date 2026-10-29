"""Reproducible stage-4 CPU smoke: train twice, reload weights, audit 7 graphs.

Run from the repository root:
  python -B -m tests.verify_stage4 --output-root outputs/stage4-smoke
Requires the stage-2 smoke data; this is verification, not a BP experiment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import zipfile

import torch
from tools.provenance import source_path
import torch_geometric
from torch_geometric.data import Batch

from gib4cl.data.dataset import LocalizationDataset
from gib4cl.config import REPO_ROOT, MODEL_VERSION, load_config
from gib4cl.learning.model import EdgePredictorGNN
from gib4cl.learning.objectives import gib_objective
from gib4cl.learning.train import train_gnn


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_model(cfg):
    return EdgePredictorGNN(cfg.model.node_in_dim, cfg.model.edge_in_dim,
                            cfg.model.hidden_dim, cfg.model.num_layers)


def verify(output_root, data_root):
    if torch.cuda.is_available():
        raise RuntimeError('This smoke audit is explicitly CPU-only; run it in the CPU environment')
    torch.set_num_threads(1)
    cfg = load_config(REPO_ROOT / 'configs/smoke-objective.json')
    torch.manual_seed(cfg.seed)
    initial = new_model(cfg).state_dict()
    paths = [train_gnn(settings=cfg, data_root=data_root, output_root=output_root / run)
             for run in ('run-a', 'run-b')]
    histories = [json.loads((path.parent / 'history.json').read_text()) for path in paths]
    assert histories[0] == histories[1], 'Identical seeds must reproduce training/validation metrics'
    states = [torch.load(path / 'final_model.pth', map_location='cpu', weights_only=True)
              for path in paths]
    for key, value in states[0].items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, states[1][key], rtol=0, atol=0)
        else:
            assert value == states[1][key]
    changed = sum(not torch.equal(value, initial[key]) for key, value in states[0].items()
                  if isinstance(value, torch.Tensor))
    assert changed > 0
    loaded = new_model(cfg).eval()
    loaded.load_state_dict(states[0])
    loaded_again = new_model(cfg).eval()
    loaded_again.load_state_dict(torch.load(paths[0] / 'best_model.pth', weights_only=True))
    graph_results, split_results = [], {}
    with torch.no_grad():
        for split in ('train', 'val', 'test'):
            dataset_path = cfg.dataset_path(split, data_root)
            dataset = LocalizationDataset(dataset_path)
            split_results[split] = {'graphs': len(dataset), 'dataset_sha256': sha256(dataset_path)}
            for graph in dataset:
                batch = Batch.from_data_list([graph])
                weights, logits = loaded(batch)
                other_weights, other_logits = loaded_again(batch)
                torch.testing.assert_close(logits, other_logits, rtol=0, atol=0)
                torch.testing.assert_close(weights, other_weights, rtol=0, atol=0)
                result = gib_objective(logits, weights, batch, objective=cfg.objective)
                assert torch.isfinite(result.loss)
                for prediction in result.predictions:
                    assert torch.isfinite(prediction.mean).all()
                    assert torch.isfinite(prediction.covariance).all()
                graph_results.append({'split': split, 'sample_id': graph.sample_id,
                                      'physical_edges': graph.num_edges,
                                      'message_arcs': graph.message_edge_index.shape[1],
                                      'selected_edges': int(weights.sum()),
                                      'metrics': result.metrics()})
    baseline = REPO_ROOT / 'docs/baselines/2026-10-22/source-snapshot.zip'
    preservation = {}
    with zipfile.ZipFile(baseline) as archive:
        for name in archive.namelist():
            # BP was intentionally rewritten in stage 5. Keep this stage-4
            # training audit rerunnable without requiring obsolete BP sources.
            if name.startswith('repository/NEBP/') and name.endswith('.py'):
                relative = name[len('repository/'):]
                assert archive.read(name) == source_path(relative).read_bytes()
                preservation[relative] = sha256(source_path(relative))
        assert archive.read('paper/GIB-Chap234-1016.tex') == (REPO_ROOT.parent / 'GIB-Chap234-1016.tex').read_bytes()
    result = {
        'stage': '04-gnn-and-discrete-sampling', 'date': '2026-10-22',
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                    'torch_geometric': torch_geometric.__version__, 'device': 'cpu',
                    'tensor_threads': torch.get_num_threads()},
        'architecture_version': MODEL_VERSION,
        'configuration': cfg.to_dict(),
        'configuration_sha256': sha256(REPO_ROOT / 'configs/smoke-objective.json'),
        'training_runs': [path.relative_to(REPO_ROOT).as_posix() for path in paths],
        'epochs_per_run': cfg.training.epochs, 'parameter_tensors_changed': changed,
        'training_metrics_exactly_reproduced': True, 'parameter_tensors_exactly_reproduced': True,
        'best_final_reload_outputs_equal': True, 'history': histories[0],
        'splits': split_results, 'graphs': graph_results,
        'baseline_archive_sha256': sha256(baseline),
        'paper_sha256': sha256(REPO_ROOT.parent / 'GIB-Chap234-1016.tex'),
        'unchanged_nebp_sources': preservation,
        'limitations': ['CPU only', 'one-epoch smoke, no convergence or BP accuracy claim',
                        'verification coefficients only', 'optimizer/resume and final BP evaluation pending']
    }
    destination = output_root / 'verification.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(destination), 'graphs': len(graph_results),
                      'changed_parameter_tensors': changed,
                      'histories_and_weights_exactly_reproduced': True}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=REPO_ROOT / 'outputs/stage4-smoke')
    parser.add_argument('--data-root', type=Path, default=REPO_ROOT / 'data/stage2-smoke')
    arguments = parser.parse_args()
    verify(arguments.output_root.resolve(), arguments.data_root.resolve())
