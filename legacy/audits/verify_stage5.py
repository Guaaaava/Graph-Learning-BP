"""CPU audit: independent posterior references, particle convergence, GNN->BP.

Run from repository root: python -B -m tests.verify_stage5
Uses existing stage-2 smoke data and stage-4 smoke checkpoint; no training.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import zipfile
import numpy as np
import torch
from tools.provenance import source_path
import torch_geometric

from gib4cl.inference import ParticleBP, ParticleSupport
from tests.references.particle_posterior import exact_discrete, gaussian_hermite_support, grid_support, radial_single_anchor
from gib4cl.data.schema import GRAPH_VERSION
from gib4cl.data.dataset import LocalizationDataset, graph_to_data
from gib4cl.config import REPO_ROOT, BP_VERSION, load_config, sample_seed
from gib4cl.learning.model import EdgePredictorGNN


def make_case(agents, anchors, edges, measurements, variances, covariance):
    a = len(agents)
    return graph_to_data({'schema_version':GRAPH_VERSION,'sample_id':'stage5-reference',
        'inputs':{'initial_positions':torch.tensor(agents+anchors,dtype=torch.float32),
                  'initial_covariances':torch.tensor(covariance,dtype=torch.float32),
                  'is_anchor':torch.tensor([False]*a+[True]*len(anchors)),
                  'edge_index':torch.tensor(edges,dtype=torch.long).t().contiguous(),
                  'measurements':torch.tensor(measurements,dtype=torch.float32),
                  'measurement_variances':torch.tensor(variances,dtype=torch.float32),
                  'communication_cost_scale':1.,
                  'measurement_encoding':'ieee754-binary32-canonical-finite'}} ,include_targets=False)


def problem_description(data):
    return {name:getattr(data,name).tolist() for name in
            ('x','edge_index','edge_attr','agent_prior_cov','agent_node_index')}


def convergence(data, reference_mean, reference_covariance, counts, seeds):
    errors = {p:[] for p in counts}
    for seed in seeds:
        whole = ParticleBP(max(counts),2).draw_support(data,generator=torch.Generator().manual_seed(seed))
        for p in counts:
            support = ParticleSupport(whole.points[:,:p],whole.points.new_full((len(whole.points),p),-math.log(p)),whole.agent_node_index)
            started = time.perf_counter()
            result = ParticleBP(p,2,chunk_size=128).infer(data,support=support)
            errors[p].append({'mean_error':float(np.linalg.norm(result.mean.numpy()-reference_mean)),
                              'covariance_error':float(np.linalg.norm(result.covariance.numpy()-reference_covariance)),
                              'min_ess':float(result.ess.min()),'seconds':time.perf_counter()-started})
    rows = []
    for p in counts:
        rows.append({'particles_per_agent':p,'seeds':len(seeds),
                     'mean_rmse':math.sqrt(np.mean([r['mean_error']**2 for r in errors[p]])),
                     'covariance_frobenius_rmse':math.sqrt(np.mean([r['covariance_error']**2 for r in errors[p]])),
                     'min_ess_over_trials':min(r['min_ess'] for r in errors[p]),
                     'mean_min_ess':float(np.mean([r['min_ess'] for r in errors[p]])),
                     'mean_seconds':float(np.mean([r['seconds'] for r in errors[p]]))})
    assert rows[-1]['mean_rmse'] < .55*rows[0]['mean_rmse']
    assert rows[-1]['covariance_frobenius_rmse'] < .55*rows[0]['covariance_frobenius_rmse']
    return {'counts':rows,'seed_values':list(seeds),'paired_prefix_supports_across_counts':True,
            'per_seed_errors':{str(p):values for p,values in errors.items()}}


def discrete_checks():
    checks = []
    for kind,a,pairs in (('single',2,[[0,1]]),('chain',3,[[0,1],[1,2]]),
                         ('tree',4,[[0,1],[1,2],[1,3]]),
                         ('cycle',3,[[0,1],[0,2],[1,2]])):
        agents = [[0.,0.],[2.,0.],[4.,1.],[2.,3.]][:a]
        anchors = [[-2.,1.],[6.,2.]]
        edges = sorted(pairs+[[0,a],[a-1,a+1]])
        xy = np.asarray(agents+anchors)
        measurements = [float(np.linalg.norm(xy[u]-xy[v]))+.4 for u,v in edges]
        data = make_case(agents,anchors,edges,measurements,[1.1]*len(edges),
                         [[[1.2,.2],[.2,2.]]]*a)
        points,masses = gaussian_hermite_support(data,2)
        support = ParticleSupport(torch.tensor(points),torch.tensor(masses),data.agent_node_index)
        for mode in ('all','none','partial'):
            mask = torch.ones(data.num_edges)
            if mode == 'none':
                mask.zero_()
            elif mode == 'partial':
                mask[::2] = 0
            result = ParticleBP(4,30,chunk_size=2).infer(data,mask,support=support)
            exact = exact_discrete(data,points,masses,mask)
            differences = {'marginal_max_abs':float(np.max(abs(result.weights.numpy()-exact.marginals))),
                           'mean_max_abs':float(np.max(abs(result.mean.numpy()-exact.mean))),
                           'covariance_max_abs':float(np.max(abs(result.covariance.numpy()-exact.covariance)))}
            if kind != 'cycle':
                assert max(differences.values()) < 1e-12
            checks.append({'graph':kind,'mask':mode,'joint_states':exact.joint_states,
                           'problem':problem_description(data),'selection_mask':mask.tolist(),
                           'support_points':points.tolist(),'log_prior_weights':masses.tolist(),
                           'exact_mean':exact.mean.tolist(),'exact_covariance':exact.covariance.tolist(),
                           'exact_marginals':exact.marginals.tolist(),
                           'bp_mean':result.mean.tolist(),'bp_covariance':result.covariance.tolist(),
                           'converged':result.converged,'message_residual':result.message_residual,
                           'differences_from_exact':differences})
    return checks


def verify(output_root, data_root):
    if torch.cuda.is_available():
        raise RuntimeError('This audit is explicitly CPU-only')
    output_root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    finite_state_checks = discrete_checks()
    radial_data = make_case([[.1,0.]],[[0.,0.]],[[0,1]],[3.2],[.7],[[[4.,0.],[0.,4.]]])
    radial_data.x[0,:2] = 0  # BP is nonlinear; no linearization at the mean.
    z,v = map(float,radial_data.edge_attr[0,:2])
    radial_levels = [radial_single_anchor(4.,z,v,order=order,radius=radius)
                     for order,radius in ((128,24.),(256,24.),(512,30.))]
    radial = radial_levels[-1]
    assert np.max(abs(radial_levels[-2]['covariance']-radial['covariance'])) < 1e-11
    grids = []
    for n in (201,401):
        points,masses = grid_support(radial_data,[-12.,-12.],[12.,12.],n)
        reference = exact_discrete(radial_data,points,masses)
        grids.append({'points_per_axis':n,'mean':reference.mean.tolist(),
                      'covariance':reference.covariance.tolist(),
                      'max_covariance_difference_from_radial':float(np.max(abs(reference.covariance[0]-radial['covariance'])))})
    assert grids[-1]['max_covariance_difference_from_radial'] < 2e-6
    radial_convergence = convergence(radial_data,radial['mean'][None,:],radial['covariance'][None,:,:],
                                    [64,256,1024,4096],range(10000,10032))
    print('Radial reference and 32-seed particle convergence passed.',flush=True)

    cooperative = make_case([[-3.,0.],[3.,0.]], [[-7.,0.],[7.,1.]], [[0,1],[0,2],[1,3]],
                            [6.2,4.3,3.9],[1.,1.,1.],
                            [[[.5,.1],[.1,.7]],[[.6,-.1],[-.1,.5]]])
    gh_levels,previous = [],None
    for order in (13,21,29,37):
        points,masses = gaussian_hermite_support(cooperative,order)
        reference = exact_discrete(cooperative,points,masses)
        change = None if previous is None else {
            'mean_max_abs':float(np.max(abs(reference.mean-previous.mean))),
            'covariance_max_abs':float(np.max(abs(reference.covariance-previous.covariance)))}
        gh_levels.append({'order_per_axis':order,'joint_states':reference.joint_states,
                          'mean':reference.mean.tolist(),'covariance':reference.covariance.tolist(),
                          'change_from_previous':change})
        previous = reference
    assert gh_levels[-1]['change_from_previous']['mean_max_abs'] < 1e-11
    assert gh_levels[-1]['change_from_previous']['covariance_max_abs'] < 1e-11
    means = cooperative.x[cooperative.agent_node_index,:2].double().numpy()
    sd = np.sqrt(cooperative.agent_prior_cov.diagonal(dim1=-2,dim2=-1).double().numpy())
    cooperative_grids = []
    for n in (25,37):
        points,masses = grid_support(cooperative,means-6*sd,means+6*sd,n)
        grid = exact_discrete(cooperative,points,masses)
        cooperative_grids.append({'points_per_axis':n,'joint_states':grid.joint_states,
                                  'mean_max_abs_difference':float(np.max(abs(grid.mean-reference.mean))),
                                  'covariance_max_abs_difference':float(np.max(abs(grid.covariance-reference.covariance)))})
    assert cooperative_grids[-1]['covariance_max_abs_difference'] < 1e-8
    cooperative_convergence = convergence(cooperative,reference.mean,reference.covariance,
                                         [64,256,1024],range(20000,20016))
    print('Two-agent nonlinear reference and 16-seed particle convergence passed.',flush=True)

    cfg = load_config(REPO_ROOT/'configs/smoke-objective.json')
    old_report = json.loads((REPO_ROOT/'docs/stage4-verification.json').read_text(encoding='utf-8'))
    checkpoint = REPO_ROOT/old_report['smoke_audit']['training_runs'][0]/'best_model.pth'
    model = EdgePredictorGNN(cfg.model.node_in_dim,cfg.model.edge_in_dim,
                             cfg.model.hidden_dim,cfg.model.num_layers).eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    bp = ParticleBP(128,6,chunk_size=64)
    graph_results,split_hashes = [],{}
    with torch.no_grad():
        for split in ('train','val','test'):
            path = cfg.dataset_path(split,data_root)
            split_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
            dataset = LocalizationDataset(path,include_targets=False)
            for index,data in enumerate(dataset):
                support = bp.draw_support(data,generator=torch.Generator().manual_seed(sample_seed(cfg.seed,split,5000,index)))
                learned,_ = model(data)
                topologies = {}
                for name,mask in (('full',None),('empty',torch.zeros(data.num_edges)),('gnn',learned)):
                    result = bp.infer(data,mask,support=support)
                    assert torch.isfinite(result.mean).all() and torch.isfinite(result.covariance).all()
                    topologies[name] = {'mean':result.mean.tolist(),'covariance':result.covariance.tolist(),
                                        'selected_edges':result.diagnostics['selected_physical_edges'],
                                        'min_ess':float(result.ess.min()),'ess':result.ess.tolist(),
                                        'converged':result.converged,'message_residual':result.message_residual,
                                        'iterations':result.iterations,'anchor_reachable':result.anchor_reachable.tolist(),
                                        'covariance_min_eigenvalue':result.covariance_min_eigenvalue.tolist(),
                                        'max_kernel_elements':result.diagnostics['max_kernel_elements']}
                graph_results.append({'sample_id':data.sample_id,'split':split,'shared_support':True,'topologies':topologies})
    preservation = {}
    baseline = REPO_ROOT/'docs/baselines/2026-10-22/source-snapshot.zip'
    with zipfile.ZipFile(baseline) as archive:
        assert archive.read('paper/GIB-Chap234-1016.tex') == (REPO_ROOT.parent/'GIB-Chap234-1016.tex').read_bytes()
        for name in archive.namelist():
            if name.startswith('repository/NEBP/') and name.endswith('.py'):
                relative = name[len('repository/'):]
                assert archive.read(name) == source_path(relative).read_bytes()
                preservation[relative] = hashlib.sha256(source_path(relative).read_bytes()).hexdigest()
    result = {
        'stage':'05-particle-bp-and-independent-references','date':'2026-10-22',
        'runtime':{'python':platform.python_version(),'torch':torch.__version__,
                   'torch_geometric':torch_geometric.__version__,'numpy':np.__version__,
                   'device':'cpu','tensor_threads':1},
        'algorithm_version':BP_VERSION,
        'finite_state_checks':finite_state_checks,
        'radial_reference':{'problem':problem_description(radial_data),
                            'integration_levels':[dict(level,mean=level['mean'].tolist(),covariance=level['covariance'].tolist()) for level in radial_levels],
                            'cartesian_grid_crosschecks':grids,'particle_convergence':radial_convergence},
        'cooperative_reference':{'problem':problem_description(cooperative),
                                 'gauss_hermite_refinement':gh_levels,'cartesian_grid_crosschecks':cooperative_grids,
                                 'particle_convergence':cooperative_convergence},
        'gnn_bp_smoke':{'graphs':len(graph_results),'particles_per_agent':128,'rounds':6,
                        'checkpoint':checkpoint.relative_to(REPO_ROOT).as_posix(),
                        'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        'dataset_sha256':split_hashes,'results':graph_results},
        'baseline_archive_sha256':hashlib.sha256(baseline.read_bytes()).hexdigest(),
        'paper_sha256':hashlib.sha256((REPO_ROOT.parent/'GIB-Chap234-1016.tex').read_bytes()).hexdigest(),
        'unchanged_nebp_sources':preservation,'duration_seconds':time.perf_counter()-started,
        'limitations':['CPU only; CUDA test skipped when unavailable',
                       'Numerical-reference refinement is empirical accuracy evidence, not a rigorous integration error bound',
                       'Finite prior supports can have low ESS and miss posterior modes; no adaptive resampling or accuracy guarantee',
                       'Loopy BP need not converge or equal the exact posterior; no universal covariance-order claim',
                       'Formal BP particle count/iteration/damping choices and unified evaluation/statistics remain pending']}
    destination = output_root/'verification.json'
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(destination),'gnn_bp_graphs':len(graph_results),
                      'radial_convergence':radial_convergence['counts'],
                      'cooperative_convergence':cooperative_convergence['counts']},ensure_ascii=False),flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=REPO_ROOT/'outputs/stage5-audit')
    parser.add_argument('--data-root',type=Path,default=REPO_ROOT/'data/stage2-smoke')
    args = parser.parse_args()
    verify(args.output_root.resolve(),args.data_root.resolve())
