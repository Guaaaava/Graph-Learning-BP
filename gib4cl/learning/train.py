"""Paper-aligned training with explicit warm starts and exact epoch-boundary resume."""
from __future__ import annotations
from dataclasses import replace
import argparse
import json
from pathlib import Path
import sys
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from gib4cl.learning.objectives import compute_gib_loss
from gib4cl.config import (ExperimentConfig,REPO_ROOT,fingerprint,load_config,
                                            config_from_dict,validate_name,with_seeds)
from gib4cl.learning.checkpoints import make_model,read_checkpoint,warm_start
from gib4cl.runtime import (CHECKPOINT_VERSION,atomic_json,atomic_torch,capture_rng,
    restore_rng,execution,runtime_info,training_sources,load_split,dataset_identity,state_digest)


def resolve_training_config(config_path=None, *, settings=None, beta=None, gamma=None,
                            kappa=None, eta=None, epochs=None, lr=None,data_seed=None,training_seed=None,
                            tau_init=None, tau_decay=None):
    if settings is not None and config_path is not None:
        raise ValueError('Pass settings or config_path, not both')
    cfg = load_config(config_path) if settings is None else settings
    if not isinstance(cfg,ExperimentConfig):
        raise TypeError('settings must be an ExperimentConfig')
    obj = {key:value for key,value in (('beta',beta),('gamma',gamma),('kappa',kappa),('eta',eta)) if value is not None}
    objective = replace(cfg.objective,**obj).require_ready()
    options = {key:value for key,value in (('epochs',epochs),('learning_rate',lr),
                 ('tau_init',tau_init),('tau_decay',tau_decay)) if value is not None}
    return with_seeds(replace(cfg,objective=objective,training=replace(cfg.training,**options)),
                      data=data_seed,training=training_seed)


# Each training batch perform the following:
# Random edge selection → Gaussian predictor and loss → STE backpropagation → Gradient clipping → AdamW update
def objective_epoch(model,loader,objective,device,*,tau,optimizer=None,max_grad_norm=5.):
    training = optimizer is not None
    model.train(training)
    totals,count = {},0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = batch.to(device)
            if training:
                optimizer.zero_grad()
            weights,logits = model(batch,tau=tau,hard=True)
            loss,metrics = compute_gib_loss(logits,weights,batch,objective=objective)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=max_grad_norm,error_if_nonfinite=True)
                optimizer.step()
            graph_count = int(batch.num_graphs)
            count += graph_count
            for key,value in metrics.items():
                totals[key] = totals.get(key,0.)+value*graph_count
    if not count:
        raise ValueError('A training/validation split must contain at least one graph')
    return {key:value/count for key,value in totals.items()}


def train_gnn(*,settings=None,config_path=None,beta=None,gamma=None,kappa=None,eta=None,
              epochs=None,lr=None,tau_init=None,tau_decay=None,data_root=None,output_root=None,
              data_seed=None,training_seed=None,
              resume=None,init_from=None,run_name=None,device='auto',threads=None,stop_after_epoch=None,
              run_directory=None):
    if resume is not None and init_from is not None:
        raise ValueError('resume and init_from are mutually exclusive')
    saved = read_checkpoint(resume) if resume is not None else None
    if saved is not None and settings is None and config_path is None:
        settings = config_from_dict(saved['config'])
    cfg = resolve_training_config(config_path,settings=settings,beta=beta,gamma=gamma,kappa=kappa,
                                  eta=eta,epochs=epochs,lr=lr,tau_init=tau_init,tau_decay=tau_decay,
                                  data_seed=data_seed,training_seed=training_seed)
    if stop_after_epoch is not None and (isinstance(stop_after_epoch,bool) or
            not isinstance(stop_after_epoch,int) or not 1 <= stop_after_epoch <= cfg.training.epochs):
        raise ValueError('stop_after_epoch must be an absolute epoch in the configured schedule')
    if run_directory is not None and (output_root is not None or run_name is not None):
        raise ValueError('Explicit run_directory and output_root/run_name are mutually exclusive')
    if saved is not None and (output_root is not None or run_name is not None or run_directory is not None):
        raise ValueError('Resume continues the checkpoint run directory; use init_from for a new output run')
    if run_name is not None:
        validate_name(run_name,'run_name')
    if saved is not None:
        # A fresh CLI process should inherit execution choices from its run.
        # Explicit overrides are still checked against the checkpoint below.
        if threads is None:
            threads = saved['runtime']['num_threads']
        if device == 'auto':
            device = saved['runtime']['device']
    target_device = execution(device,threads,cfg.training.deterministic,cfg.seeds.training)
    runtime,sources = runtime_info(target_device),training_sources()
    train_data,train_id = load_split(cfg,'train',data_root,training=True)
    val_data,val_id = load_split(cfg,'val',data_root,training=True)
    identities = {'train':train_id,'val':val_id}
    if set(g['sample_id'] for g in train_data.raw_data_list) & set(g['sample_id'] for g in val_data.raw_data_list):
        raise ValueError('Training and validation samples overlap')
    generators = {'train':torch.Generator().manual_seed(cfg.seeds.training),
                  'val':torch.Generator().manual_seed((cfg.seeds.training+1) % (2**63))}
    loaders = {key:DataLoader(data,batch_size=cfg.training.batch_size,shuffle=key=='train',
                  num_workers=0,generator=generators[key])
               for key,data in (('train',train_data),('val',val_data))}
    model = make_model(cfg.model).to(target_device)
    initial = warm_start(model,init_from) if init_from is not None else None
    initial_digest = state_digest(model.state_dict())
    optimizer = AdamW(model.parameters(),lr=cfg.training.learning_rate,weight_decay=cfg.training.weight_decay)
    scheduler = CosineAnnealingLR(optimizer,T_max=cfg.training.epochs,eta_min=cfg.training.lr_min)
    if saved is not None:
        if cfg.to_dict() != saved['config']:
            raise ValueError('Resume requires the original full config/schedule; use init_from for changes')
        if runtime != saved['runtime'] or sources != saved['training_sources']:
            raise ValueError('Resume runtime/source differs; use init_from for a new run')
        if any(dataset_identity(identities[key]) != dataset_identity(saved['datasets'][key]) for key in identities):
            raise ValueError('Resume dataset identity/checksum differs')
        models_dir = Path(resume).resolve().parent
        run_dir = models_dir.parent
        if models_dir.name != 'models' or not (run_dir/'config.json').is_file():
            raise ValueError('Resume expects the saved run layout (run/models/checkpoint.pt)')
        latest = models_dir/'last_checkpoint.pt'
        if latest.exists() and read_checkpoint(latest)['epoch'] > saved['epoch']:
            raise ValueError('A newer checkpoint exists; resume last_checkpoint.pt or use init_from')
        model.load_state_dict(saved['model_state'])
        optimizer.load_state_dict(saved['optimizer_state'])
        scheduler.load_state_dict(saved['scheduler_state'])
        history,best,best_epoch,start = saved['history'],saved['best_metric'],saved['best_epoch'],saved['epoch']+1
        manifest = json.loads((run_dir/'run.json').read_text(encoding='utf-8'))
        restore_rng(saved['rng_state'],generators)
    else:
        root = Path(output_root) if output_root is not None else REPO_ROOT/'outputs'
        run_dir = Path(run_directory) if run_directory is not None else root/cfg.experiment_id/fingerprint(cfg.to_dict())[:16]
        if run_name is not None:
            run_dir = run_dir/run_name
        run_dir.mkdir(parents=True,exist_ok=False)
        models_dir = run_dir/'models'
        models_dir.mkdir()
        history,best,best_epoch,start = [],None,None,1
        manifest = {'config':cfg.to_dict(),
                    'datasets':identities,'runtime':runtime,'training_sources':sources,
                    'init_from':initial,'initial_model_digest':initial_digest,
                    'validation_selection':'minimum deterministic-graph validation total; fixed objective coefficients'}
        atomic_json(run_dir/'config.json',cfg.to_dict())
        atomic_json(run_dir/'run.json',manifest)
    last_epoch = cfg.training.epochs if stop_after_epoch is None else stop_after_epoch
    if last_epoch < start-1:
        raise ValueError('Cannot resume backwards to an earlier epoch')
    print(f'Train device={target_device}; epochs {start}..{last_epoch}; run={run_dir}')
    for epoch in range(start,last_epoch+1):
        tau = max(cfg.training.tau_min,cfg.training.tau_init*cfg.training.tau_decay**epoch)
        learning_rate = optimizer.param_groups[0]['lr']
        train_metrics = objective_epoch(model,loaders['train'],cfg.objective,target_device,tau=tau,
                                        optimizer=optimizer,max_grad_norm=cfg.training.max_grad_norm)
        val_metrics = objective_epoch(model,loaders['val'],cfg.objective,target_device,tau=cfg.training.tau_min)
        scheduler.step()
        improved = best is None or val_metrics['total'] < best
        if improved:
            best,best_epoch = val_metrics['total'],epoch
        history.append({'epoch':epoch,'tau':tau,'learning_rate':learning_rate,'train':train_metrics,'val':val_metrics})
        checkpoint = {'checkpoint_version':CHECKPOINT_VERSION,'model_state':model.state_dict(),
                      'model_digest':state_digest(model.state_dict()),'optimizer_state':optimizer.state_dict(),
                      'scheduler_state':scheduler.state_dict(),'epoch':epoch,'best_metric':best,'best_epoch':best_epoch,
                      'config':cfg.to_dict(),'history':history,'rng_state':capture_rng(generators),
                      'datasets':identities,'runtime':runtime,'training_sources':sources,'run_dir':str(run_dir.resolve())}
        if improved:
            atomic_torch(models_dir/'best_model.pth',model.state_dict())
            atomic_torch(models_dir/'best_checkpoint.pt',checkpoint)
        atomic_torch(models_dir/'last_checkpoint.pt',checkpoint)
        atomic_torch(models_dir/'final_model.pth',model.state_dict())
        atomic_json(run_dir/'history.json',history)
        manifest.update({'last_epoch':epoch,'best_epoch':best_epoch,'best_metric':best,
                         'complete':epoch==cfg.training.epochs,'last_model_digest':checkpoint['model_digest']})
        atomic_json(run_dir/'run.json',manifest)
        print(f"Epoch {epoch}: train={train_metrics['total']:.6f}; val={val_metrics['total']:.6f}")
    return models_dir


def main():
    # Both the old module path and the unified CLI use exactly this implementation.
    from gib4cl.experiments.cli import main as unified_main
    return unified_main(['train',*sys.argv[1:]])


if __name__ == '__main__':
    main()
