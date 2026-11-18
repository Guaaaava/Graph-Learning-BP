"""本地检查场景生成与拓扑；输出图片、配置和一张原始图，不训练 GNN 或运行 BP。

在项目根目录运行：
    python -B -m tools.preview_network --num-agents 30
    python -B -m tools.preview_network --config configs/smoke-pipeline.json --show
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import math
from pathlib import Path
import sys
import uuid

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from gib4cl.config import REPO_ROOT, load_config, sample_seed, with_seeds
from gib4cl.data.dataset import graph_to_data
from gib4cl.data.generate import build_dataset, save_dataset
from gib4cl.evaluation.topology import selection_metrics
from gib4cl.inference.spanning_tree import bfs_forest
from gib4cl.runtime import atomic_json, sha256, source_manifest


def _seed(value):
    value = int(value)
    if not 0 <= value < 2**63:
        raise argparse.ArgumentTypeError('seed must be an integer in [0, 2**63)')
    return value


def _example_mask(num_edges, keep_ratio, seed):
    """仅用于展示的随机二值 s；有至少两条边时保证既选边又删边。"""
    count = min(num_edges - 1, max(1, math.floor(num_edges * keep_ratio))) if num_edges >= 2 else 0
    generator = torch.Generator(device='cpu').manual_seed(seed)
    selected = torch.randperm(num_edges, generator=generator)[:count]
    mask = torch.zeros(num_edges, dtype=torch.bool)
    mask[selected] = True
    return mask


def _plot_scene(record, size, plt):
    """真值用于展示场景；初值不裁剪，保留可能位于正方形之外的先验均值。"""
    from matplotlib.patches import Rectangle

    inputs = record['inputs']
    truth = record['targets']['agent_positions'].numpy()
    initial = inputs['initial_positions'][~inputs['is_anchor']].numpy()
    anchors = inputs['initial_positions'][inputs['is_anchor']].numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    for ax in axes:
        ax.add_patch(Rectangle((0, 0), size, size, fill=False, linestyle='--',
                               edgecolor='0.5', linewidth=1))
        ax.scatter(truth[:, 0], truth[:, 1], c='#4A90E2', s=40, label='Agent truth', zorder=3)
        ax.scatter(anchors[:, 0], anchors[:, 1], c='#E94A4A', marker='s', s=100,
                   label='Known anchors', zorder=4)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(alpha=.2)
        ax.margins(.08)
    for true_position, prior_mean in zip(truth, initial):
        axes[1].plot([true_position[0], prior_mean[0]], [true_position[1], prior_mean[1]],
                     color='0.6', linewidth=.7, alpha=.6, zorder=1)
    axes[1].scatter(initial[:, 0], initial[:, 1], c='#E69228', marker='x', s=45,
                    label='Prior means', zorder=3)
    axes[0].set_title('Uniform agents and five fixed anchors')
    axes[1].set_title('Truth and noisy prior means')
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .07, 1, 1))
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=REPO_ROOT/'configs/experiment.json',
                        help='Existing experiment JSON; only data settings and the data seed generate the scene')
    parser.add_argument('--num-agents', type=int,
                        help='Fix the agent count; otherwise use the configured range')
    parser.add_argument('--data-seed', type=_seed, help='Override seeds.data')
    parser.add_argument('--seed-offset', type=int, default=0,
                        help='Choose another reproducible preview sample without changing the config')
    parser.add_argument('--mask-seed', type=_seed,
                        help='Override the example-mask seed; otherwise derive it independently from the scene seed')
    parser.add_argument('--keep-ratio', type=float, default=.5,
                        help='Example-mask retention target, strictly between 0 and 1; not a GNN parameter')
    parser.add_argument('--output-dir', type=Path,
                        help='New output directory; default: outputs/network-preview/<unique ID>')
    parser.add_argument('--show', action='store_true', help='Also display the figures after saving')
    args = parser.parse_args(argv)
    if args.num_agents is not None and args.num_agents < 1:
        parser.error('--num-agents must be positive')
    if args.seed_offset < 0:
        parser.error('--seed-offset must be nonnegative')
    if not math.isfinite(args.keep_ratio) or not 0 < args.keep_ratio < 1:
        parser.error('--keep-ratio must be finite and strictly between 0 and 1')

    cfg = with_seeds(load_config(args.config), data=args.data_seed)
    if args.num_agents is not None:
        cfg = replace(cfg, data=replace(cfg.data, num_agents_min=args.num_agents,
                                       num_agents_max=args.num_agents))
    destination = (args.output_dir if args.output_dir is not None else
                   REPO_ROOT/'outputs/network-preview'/uuid.uuid4().hex[:8]).resolve()
    destination.mkdir(parents=True, exist_ok=False)

    # 使用正式生成链；preview 是独立的样本种子命名空间，不写入训练／测试数据目录。
    torch.set_num_threads(1)
    bundle = build_dataset(cfg, 'preview', 1, seed_offset=args.seed_offset)
    record = bundle['graphs'][0]
    data = graph_to_data(record)
    forest = bfs_forest(len(data.x), data.edge_index, data.anchor_node_index)
    num_edges = data.edge_index.shape[1]
    mask_seed = (args.mask_seed if args.mask_seed is not None else
                 sample_seed(record['metadata']['sample_seed'], 'example-mask', 0, 0))
    masks = {'dense': torch.ones(num_edges, dtype=torch.bool),
             'bfs': forest.edge_weights.bool(),
             'example_gnn': _example_mask(num_edges, args.keep_ratio, mask_seed)}
    topologies = {name: {'s': mask.to(torch.int64).tolist(),
                         'selected_edge_ids': torch.where(mask)[0].tolist(),
                         'metrics': selection_metrics(data, mask)}
                  for name, mask in masks.items()}

    save_dataset(bundle, destination/'graph.pt')
    atomic_json(destination/'config.json', cfg.to_dict())

    # 无 --show 时无需图形桌面；延迟导入绘图模块，先确定 Matplotlib 后端。
    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from gib4cl.evaluation.plotting import plot_topology_comparison

    scene = _plot_scene(record, cfg.data.area_size, plt)
    comparison, axes = plot_topology_comparison(record, masks['example_gnn'], masks['bfs'], show=False)
    axes[2].set_title(f"3. Example s (not trained GNN)\nEdges: {topologies['example_gnn']['metrics']['edges']}",
                      fontsize=14, fontweight='bold', pad=10)
    legend = axes[0].get_legend()
    if legend is not None:
        legend.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    comparison.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
    margin = .05 * cfg.data.area_size
    for ax in axes:
        ax.set_axis_on()
        ax.tick_params(bottom=True, left=True, labelbottom=True, labelleft=True)
        ax.set_xlim(-margin, cfg.data.area_size + margin)
        ax.set_ylim(-margin, cfg.data.area_size + margin)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(alpha=.2)
    comparison.tight_layout(rect=(0, .07, 1, 1))
    scene.savefig(destination/'scene.png', dpi=150, bbox_inches='tight')
    comparison.savefig(destination/'topology.png', dpi=150, bbox_inches='tight')
    comparison.savefig(destination/'topology.svg', bbox_inches='tight')

    summary = {
        'status': 'complete', 'purpose': 'Scene/network generation preview; no training or BP inference',
        'source_config': str(args.config.resolve()), 'sample_id': record['sample_id'],
        'data_seed': cfg.seeds.data, 'sample_seed': record['metadata']['sample_seed'],
        'seed_offset': args.seed_offset, 'split_namespace': 'preview',
        'num_agents': data.num_agents, 'num_anchors': data.num_anchors,
        'anchor_coordinates': cfg.data.anchor_coordinates,
        'example_mask': {'source': 'Random fixed-count binary mask; not a trained GNN',
                         'seed': mask_seed, 'requested_keep_ratio': args.keep_ratio,
                         'rule': 'For E>=2, keep min(E-1,max(1,floor(E*ratio))) random edges; otherwise keep none',
                         'note': 'E=0 has no edges to remove; E=1 requires an empty mask to avoid all ones'},
        'topologies': topologies, 'bfs_forest_root': forest.forest_root.tolist(),
        'source_sha256': source_manifest(), 'preview_script_sha256': sha256(__file__),
        'files_sha256': {name: sha256(destination/name) for name in
                         ('config.json', 'graph.pt', 'graph.json', 'scene.png', 'topology.png', 'topology.svg')},
    }
    atomic_json(destination/'summary.json', summary)
    print(f'Output: {destination}')
    print(f'Agents={data.num_agents}; anchors={data.num_anchors}; area={cfg.data.area_size:g} m square; radius={cfg.data.comm_radius:g} m')
    print(f'Data seed={cfg.seeds.data}; sample seed={summary["sample_seed"]}; example-mask seed={mask_seed}')
    print('Topology       Edges   Mean agent degree   Unanchored agents')
    for name, values in topologies.items():
        metrics = values['metrics']
        print(f'{name:14} {metrics["edges"]:5d} {metrics["agent_degree_mean"]:19.4f} {metrics["unanchored_agents"]:19d}')
    print('example_gnn is a display-only mask, not a trained-model result.')
    if args.show:
        plt.show()
    plt.close(scene)
    plt.close(comparison)
    return destination


if __name__ == '__main__':
    main()
