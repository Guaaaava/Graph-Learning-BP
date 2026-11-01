"""Reader-facing scientific figures with explicit probability/CI semantics."""
from __future__ import annotations

import csv
import io
import math
from pathlib import Path
import numpy as np

from gib4cl.runtime import atomic_json, sha256
from .metrics import bootstrap_stat
from .model import graph_spec, seed_for
from .gaussian_control import controlled_rings

COLORS = {'tree':'#0072B2', 'loop':'#D55E00', 'loop3':'#009E73', 'loop4':'#CC79A7', 'loop5':'#D55E00'}


def label(name):
    if name.startswith('tree_'):
        return 'Tree BP ('+name.removeprefix('tree_')+')'
    return {'tree':'Tree BP', 'loop':'Loopy BP'}.get(name, r'BP ($L='+name.removeprefix('loop')+'$)')


def _panel(ax, letter):
    ax.text(0,1.025,'('+letter+')',transform=ax.transAxes,fontsize=11)


def _ellipse(ax, mean, covariance, color, fill=False, name=None):
    from matplotlib.patches import Ellipse
    eigenvalues, vectors = np.linalg.eigh(covariance)
    if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 0:
        ax.text(.03,.03,'Invalid covariance',transform=ax.transAxes,color=color)
        return
    angle = math.degrees(math.atan2(vectors[1,1],vectors[0,1]))
    width,height = 2*np.sqrt(-2*math.log(.05)*eigenvalues[::-1])
    if fill:
        ax.add_patch(Ellipse(mean,width,height,angle=angle,facecolor=color,edgecolor='none',alpha=.09))
    ax.add_patch(Ellipse(mean,width,height,angle=angle,fill=False,edgecolor=color,lw=2,label=name))
    ax.plot(*mean,'+',color=color,ms=8,mew=1.5)


def _paper_layout(spec, family):
    if family=='triangle':
        return np.array([[0,0],[1,.8],[2,0]]),spec.labels
    # Raw indices are unchanged. Swap only a/b display names to match Fig.1:
    # the saved edge raw-a--c is the symmetric inner-leaf edge paper-b--c.
    return (np.array([[0,2.25],[-1,1.1],[1,1.1],[-.6,0],[-1.8,0],[.6,0],[1.8,0]]),
            tuple({'a':'b','b':'a'}.get(name,name) for name in spec.labels))


def _draw_topology(inset, spec, name, family):
    from matplotlib.patches import FancyArrowPatch
    xy,labels=_paper_layout(spec,family)
    for edge,(active,(u,v)) in enumerate(zip(spec.masks[name],spec.edges)):
        if active:
            closure = not spec.masks['tree'][edge]
            inset.add_patch(FancyArrowPatch(xy[u],xy[v],arrowstyle='-',lw=1.8 if closure else 1.1,
                                            color='#D55E00' if closure else '.58',zorder=1))
    faces=['#e6f2f8' if node==spec.focus else 'white' for node in range(len(labels))]
    inset.scatter(*xy.T,s=115,c=faces,edgecolors='.35',lw=1.1,zorder=3)
    for text,(x,y) in zip(labels,xy):
        inset.text(x,y,'$'+text+'$',ha='center',va='center',fontsize=8,zorder=4)
    inset.set_xlim(xy[:,0].min()-.35,xy[:,0].max()+.35)
    inset.set_ylim(xy[:,1].min()-.35,xy[:,1].max()+.35)
    inset.set_aspect('equal')
    inset.axis('off')


def _topology(ax, spec, name, family):
    inset=ax.inset_axes([.67,.65,.32,.33] if family=='branched7' else [.70,.76,.27,.20])
    _draw_topology(inset,spec,name,family)


def _indices(settings, condition, p, n, key):
    rng=np.random.default_rng(seed_for(settings.bootstrap_seed,condition,p,key))
    return rng.integers(0,n,(settings.bootstrap_samples,n))


def _curve(cases,p,name,key,focus):
    return np.array([np.mean([[h[key][focus] for h in run['topologies'][name]['history']]
                              for run in case['runs'] if run['particles']==p],axis=0) for case in cases])


def _series(array,indices):
    values=[bootstrap_stat(array[:,j].tolist(),indices) for j in range(array.shape[1])]
    return [np.array([np.nan if v[key] is None else v[key] for v in values]) for key in ('mean','ci_low','ci_high')]


def _errors(stat):
    if stat['mean'] is None or stat['ci_low'] is None:
        return None
    return [[max(0.,stat['mean']-stat['ci_low'])],[max(0.,stat['ci_high']-stat['mean'])]]


def _save(fig,root,stem,plt):
    paths=[]
    for suffix in ('png','pdf','svg'):
        path=root/'figures'/f'{stem}.{suffix}'
        fig.savefig(path,dpi=190,bbox_inches='tight',facecolor='white')
        paths.append(path)
    plt.close(fig)
    return paths


def _stamp(fig,settings,manifest):
    if settings.purpose=='smoke' or manifest['phase']=='pilot':
        fig.text(.985,.99,'SMOKE RUN' if settings.purpose=='smoke' else 'NUMERICAL PILOT',
                 ha='right',va='top',fontsize=8,color='.45',weight='bold')


def _mechanism(root,c,selected,settings,manifest,p,plt):
    spec=graph_spec(c)
    node=spec.focus
    run=next(r for r in selected[0]['runs'] if r['particles']==p and r['repeat']==0)
    center=np.array(c.centers[node])
    truth=np.array(selected[0]['truth'][node])-center
    graphs=('tree',settings.primary_loop)
    indices=_indices(settings,c.name,p,len(selected),'plot-curves')
    fig,axes=plt.subplots(2,2,figsize=(8.6,7.5),gridspec_kw={'height_ratios':[1.15,1]})
    bound=max(2.,float(np.abs(truth).max())+.3)
    for name in graphs:
        value=run['topologies'][name]
        for m,s in (('mean','covariance'),('reference_mean','reference_covariance')):
            radius=np.sqrt(-2*math.log(.05)*np.maximum(0.,np.diag(value[s][node])))
            bound=max(bound,float(np.max(np.abs(np.array(value[m][node])-center)+radius))*1.12)
    handles,labels=[],[]
    curves={name:{key:_series(_curve(selected,p,name,key,node),indices)
                  for key in ('lambda_min','lambda_max')} for name in graphs}
    early=min(6,settings.plot_rounds,settings.iterations)
    late=max(0,early-3)
    late_values=np.concatenate([np.array([1.])]+[v[late:early+1] for name in graphs
                                                for values in curves[name].values() for v in values])
    late_values=late_values[np.isfinite(late_values)]
    pad=max(.025,float(np.ptp(late_values))*.15)
    late_limits=(float(late_values.min())-pad,float(late_values.max())+pad)
    maximum=max(float(np.nanmax(np.r_[values[0][:early+1],values[2][:early+1]]))
                for name in graphs for values in curves[name].values())
    for column,name in enumerate(graphs):
        value,color=run['topologies'][name],COLORS[name]
        ax=axes[0,column]
        _ellipse(ax,np.array(value['reference_mean'][node])-center,value['reference_covariance'][node],'.25',name='Exact posterior')
        _ellipse(ax,np.array(value['mean'][node])-center,value['covariance'][node],color,fill=True,name=label(name))
        ax.scatter(*truth,c='black',marker='*',s=100,zorder=5,label='True position')
        ax.set(xlim=(-bound,bound),ylim=(-bound,bound),aspect='equal',xlabel='x relative to prior mean (m)',ylabel='y relative to prior mean (m)')
        _panel(ax,'a' if column==0 else 'b')
        _topology(ax,spec,name,c.graph_family)
        for h,l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
        ax=axes[1,column]
        detail=ax.inset_axes([.42,.15,.54,.42])
        for key,linestyle,marker,text in (('lambda_min','-','o',r'$\lambda_{\min}$'),
                                         ('lambda_max','--','^',r'$\lambda_{\max}$')):
            mean,low,high=curves[name][key]
            t=np.arange(len(mean))
            for target,size in ((ax,4.2),(detail,3.2)):
                target.plot(t,mean,ls=linestyle,marker=marker,ms=size,mfc='white',c=color,lw=1.6,label=text)
                target.fill_between(t,low,high,color=color,alpha=.12,lw=0)
        detail.axhline(1,c='.4',ls=':',lw=.9)
        detail.set(xlim=(late-.08,early+.08),ylim=late_limits,xticks=np.arange(late,early+1))
        detail.tick_params(labelsize=7,pad=2)
        detail.grid(alpha=.13)
        ax.axhline(1,c='.3',ls=':',lw=1)
        ax.set(xlim=(-.08,early+.12),xticks=np.arange(early+1),xlabel='Iteration',ylabel='Directional variance ratio\n(BP / exact)')
        _panel(ax,'c' if column==0 else 'd')
        ax.legend(loc='upper right',fontsize=9,frameon=False)
    for ax in axes[1]:
        ax.set_ylim(0,maximum*1.08)
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.995),ncol=4,fontsize=9,frameon=False)
    _stamp(fig,settings,manifest)
    fig.tight_layout(rect=(0,0,1,.94),h_pad=1.5,w_pad=1.6)
    paths=_save(fig,root,c.name+'-mechanism',plt)
    fig,axes=plt.subplots(1,2,figsize=(8.8,3.3))
    for ax,name,letter in zip(axes,graphs,('a','b')):
        for key,ls,text in (('lambda_min','-',r'$\lambda_{\min}$'),('lambda_max','--',r'$\lambda_{\max}$')):
            mean,low,high=curves[name][key]
            ax.plot(np.arange(len(mean)),mean,c=COLORS[name],ls=ls,label=text)
            ax.fill_between(np.arange(len(mean)),low,high,color=COLORS[name],alpha=.12,lw=0)
        ax.axhline(1,c='.4',ls=':')
        ax.set(xlim=(0,settings.iterations),xlabel='Iteration',ylabel='Directional variance ratio\n(BP / exact)')
        ax.legend(frameon=False,fontsize=9)
        _panel(ax,letter)
    _stamp(fig,settings,manifest)
    fig.tight_layout(rect=(0,0,1,.97))
    return paths+_save(fig,root,c.name+'-covariance-iterations-full',plt)


def _calibration(root,c,selected,group,settings,manifest,p,plt,levels=None,suffix=''):
    from matplotlib.ticker import PercentFormatter,MultipleLocator
    primary=settings.primary_loop
    levels=np.array(settings.coverage_levels if levels is None else levels)
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.4),gridspec_kw={'width_ratios':[1.25,1]})
    zoom=max(float(levels.min()),.9 if levels.max()>=.9 else float(levels.max())-.1)
    series={name:[np.array([np.nan if row[key] is None else row[key] for row in group['topologies'][name]['coverage_x']])
                  for key in ('mean','ci_low','ci_high')] for name in ('tree',primary)}
    exact=np.array([v['mean'] for v in group['topologies'][primary]['finite_reference_coverage_x']])
    handles=[]
    for column,ax in enumerate(axes):
        mask=levels>=zoom-1e-10 if column else np.ones(len(levels),dtype=bool)
        x=levels[mask]
        limits=[]
        for name,marker in (('tree','o'),(primary,'s')):
            mean,low,high=(v[mask] for v in series[name])
            ax.fill_between(x,low,high,color=COLORS[name],alpha=.10,lw=0,zorder=1)
            marks=[j for j,v in enumerate(x) if any(abs(v-c)<1e-10 for c in settings.coverage_levels)]
            line,=ax.plot(x,mean,marker=marker,markevery=marks,mfc='white',mew=1.1,
                          c=COLORS[name],lw=1.8,ms=4.2,zorder=4,label='Tree BP' if name=='tree' else 'Loopy BP')
            limits.extend(np.r_[mean,low].tolist())
            if column==0:handles.append(line)
        line,=ax.plot(x,exact[mask],ls=(0,(4,2)),c='.25',lw=1.3,zorder=3,label='Exact (loop)')
        ideal,=ax.plot([0,1],[0,1],ls=':',c='.45',lw=1.2,zorder=2,label='Ideal')
        if column==0:handles.extend([line,ideal])
        limits=np.array(limits)
        bottom=max(0.,math.floor(float(np.nanmin(limits))/.05)*.05-.015)
        ax.set(xlim=(zoom-.008,1.002) if column else (max(0.,float(levels.min())-.015),1.002),
               ylim=(bottom,1.015),xlabel=r'Nominal coverage of $x$-credible intervals',
               ylabel=r'Empirical coverage of $x$-credible intervals')
        if column:ax.set_xticks(sorted(set([zoom,.95,.99])&set(levels.tolist())))
        ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.yaxis.set_major_locator(MultipleLocator(.05 if column else .1))
        _panel(ax,'b' if column else 'a')
    fig.legend(handles,[h.get_label() for h in handles],loc='upper center',bbox_to_anchor=(.5,.995),
               ncol=4,frameon=False,fontsize=9)
    _stamp(fig,settings,manifest)
    fig.tight_layout(rect=(0,0,1,.92),w_pad=2)
    return _save(fig,root,c.name+'-calibration'+suffix,plt)


def _residual(root,c,selected,settings,manifest,p,plt):
    fig,ax=plt.subplots(figsize=(7.5,4.6))
    indices=_indices(settings,c.name,p,len(selected),'residuals')
    for name in settings.topologies:
        array=np.array([np.mean([[h['message_residual'] for h in r['topologies'][name]['history'][1:]]
                                  for r in case['runs'] if r['particles']==p],axis=0) for case in selected])
        mean,low,high=_series(array,indices)
        t=np.arange(1,len(mean)+1)
        ax.plot(t,np.maximum(mean,1e-16),c=COLORS.get(name,'.5'),label=label(name))
        ax.fill_between(t,np.maximum(low,1e-16),np.maximum(high,1e-16),color=COLORS.get(name,'.5'),alpha=.1)
    ax.axhline(settings.message_tolerance,c='.25',ls='--',label='Tolerance')
    ax.set(yscale='log',xlabel='Message-passing iteration',ylabel='Maximum normalized log-message change',
           xlim=(1,settings.iterations))
    ax.legend(fontsize=8,frameon=False)
    _stamp(fig,settings,manifest)
    fig.tight_layout(rect=(0,0,1,.97))
    return _save(fig,root,c.name+'-message-residual',plt)


def _cycle_comparison(root,c,group,settings,manifest,plt):
    spec=graph_spec(c)
    names=sorted((name for name in settings.topologies if name in spec.cycle_lengths),key=lambda name:spec.cycle_lengths[name])
    lengths=[spec.cycle_lengths[name] for name in names]
    fig=plt.figure(figsize=(7.4,5.3))
    grid=fig.add_gridspec(2,len(names),height_ratios=[1.15,2.8],hspace=.25)
    for k,name in enumerate(names):
        top=fig.add_subplot(grid[0,k])
        _draw_topology(top,spec,name,c.graph_family)
        top.text(.5,-.10,f'$L={spec.cycle_lengths[name]}$',transform=top.transAxes,ha='center',fontsize=10)
    ax=fig.add_subplot(grid[1,:])
    lows=[]
    for key,text,color,marker in (('focus_variance_ratio_x',r'Node $i$','#0072B2','o'),
                                 ('variance_ratio_x','All agents','#009E73','s')):
        values=[group['topologies'][name]['statistics'][key] for name in names]
        ax.plot(lengths,[v['mean'] for v in values],c=color,lw=1.8,ls='-' if marker=='o' else '--',label=text)
        for length,stat in zip(lengths,values):
            ax.errorbar(length,stat['mean'],yerr=_errors(stat),fmt=marker,c=color,capsize=4,ms=6,
                        mfc='white',mew=1.3,lw=1.2)
            lows.append(stat['mean'] if stat['ci_low'] is None else stat['ci_low'])
    ax.axhline(1,c='.4',ls=':',lw=1.2,label='Exact scale')
    ax.set(xlabel='Cycle length',ylabel=r'$x$ variance ratio (BP / exact)',xticks=lengths,
           xlim=(min(lengths)-.22,max(lengths)+.22),ylim=(min(lows)-.035,1.025))
    ax.legend(loc='lower right',fontsize=9,frameon=False)
    _stamp(fig,settings,manifest)
    fig.subplots_adjust(left=.13,right=.98,bottom=.12,top=.97)
    return _save(fig,root,c.name+'-cycle-length-comparison',plt)


def plot_experiment(root,cases,summary,settings,manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root)
    (root/'figures').mkdir(exist_ok=True)
    p=settings.num_particles if settings.num_particles in manifest['particle_counts'] else max(manifest['particle_counts'])
    paths=[]
    style={'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
           'axes.grid':True,'grid.alpha':.17,'grid.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'}
    with plt.rc_context(style):
        for c in settings.conditions:
            selected=sorted((case for case in cases if case['condition']==c.name),key=lambda case:case['case_index'])
            if not selected:
                continue
            group=summary['groups'][c.name][str(p)]
            paths += _mechanism(root,c,selected,settings,manifest,p,plt)
            if c.graph_family=='branched7':
                focused=summary['focus_calibration']['groups'][c.name][str(p)]
                paths += _calibration(root,c,selected,focused,settings,manifest,p,plt,
                                      levels=summary['focus_calibration']['coverage_levels'])
                paths += _calibration(root,c,selected,group,settings,manifest,p,plt,suffix='-all-agents')
            else:
                paths += _calibration(root,c,selected,group,settings,manifest,p,plt)
            paths += _residual(root,c,selected,settings,manifest,p,plt)
            if c.graph_family=='branched7':
                paths += _cycle_comparison(root,c,group,settings,manifest,plt)
        if len(manifest['particle_counts'])>1:
            fig,axes=plt.subplots(1,2,figsize=(10.5,4.5))
            for c,groups in summary['groups'].items():
                particles=sorted(map(int,groups))
                for name in settings.topologies:
                    if not name.startswith('loop'):
                        continue
                    for ax,key in zip(axes,('variance_ratio_x','lambda_min')):
                        values=[groups[str(p0)]['topologies'][name]['statistics'][key] for p0 in particles]
                        ax.plot(particles,[v['mean'] for v in values],'o-',label=c+' / '+name)
            for ax,key in zip(axes,('Mean x variance ratio','Mean minimum directional ratio')):
                ax.set_xscale('log',base=2)
                ax.set(xlabel='Particles per agent',ylabel=key,xticks=particles)
                ax.set_xticklabels([str(p0) for p0 in particles])
                ax.axhline(1,c='.4',ls=':')
                ax.legend(fontsize=7,frameon=False)
            _stamp(fig,settings,manifest)
            fig.tight_layout(rect=(0,0,1,.97))
            paths += _save(fig,root,'particle-stability',plt)
        if settings.conditions[0].graph_family=='branched7':
            data=controlled_rings()
            atomic_json(root/'gaussian-ring-control.json',data)
            stream=io.StringIO(newline='')
            writer=csv.DictWriter(stream,fieldnames=list(data['rows'][0]))
            writer.writeheader(); writer.writerows(data['rows'])
            (root/'gaussian-ring-control.csv').write_bytes(stream.getvalue().encode('utf-8'))
            fig,ax=plt.subplots(figsize=(7,4.8))
            for b in (1.,4.):
                rows=[r for r in data['rows'] if r['edge_precision']==b]
                ax.plot([r['ring_length'] for r in rows],[r['variance_ratio'] for r in rows],'o-',label=f'$b={b:g}$')
            ax.axhline(1,c='.4',ls=':',label='Exact')
            ax.set(xlabel='Cycle length',ylabel='Gaussian BP variance / exact variance')
            ax.legend(frameon=False)
            fig.tight_layout()
            paths += _save(fig,root,'gaussian-ring-control',plt)
    (root/'figures/captions.md').write_text(
        '# Figure interpretation\n\n'
        '- Exact posterior is the benchmark posterior obtained by exact marginalization on the SAME finite particle support, priors and observations as BP, separately for each graph. Reference was a description of its benchmark role, not an additional distribution or algorithm. Exactness is limited to the finite model.\n'
        '- Mechanism: left column is the tree, right column is the configured primary loop. Upper panels use fixed trial 0 / support repeat 0 and the configured focus node. Ellipses summarize posterior means/covariances; Gaussian-equivalent 95% scaling does not guarantee a true 95% posterior region. Crosses mark posterior means; the star is simulation truth. Coordinates are relative to the prior mean. Lower panels expand iterations 0-6 and inset iterations 3-6 near unit scale; every integer iteration has a marker. Full covariance trajectories are saved separately as covariance-iterations-full.\n'
        '- Seven-node topology display: node i is at the top. Raw indices/data are unchanged; the original a/b display labels are exchanged to match Fig.1. Thus saved raw-a--c is displayed as the symmetric inner-leaf b--c edge, and raw-a--k is displayed as b--k. Diagrams are schematic, not physical coordinate plots.\n'
        '- Directional curves take minimum/maximum over spatial directions within each run, then average support repeats within each data trial and average independent trials. Shading is a pointwise 95% bootstrap confidence interval for that mean, not the range of individual results.\n'
        '- Calibration: each method constructs its own equal-tail posterior credible interval for the unknown x coordinate. A nominal 95% interval uses the weighted 2.5% and 97.5% posterior quantiles. Probability mass is the sum of included particle weights, not the fraction of particle counts. Credible refers to probability assigned by that posterior, not a guarantee of empirical correctness. Both axes refer to x-coordinate intervals.\n'
        '- Main seven-node calibration uses node i ONLY, matching the mechanism figure. Support repeats are averaged within each trial before the trial bootstrap. A finer 50%-99% probability grid evaluates saved particle weights and adds no new trials. Panel (a) shows the full reported probability range; panel (b) enlarges 90%-99%. The calibration-all-agents file retains the pooled seven-node result as supplementary. Three-node calibration retains its original all-agent scope.\n'
        '- Calibration shading is a pointwise 95% bootstrap confidence interval for mean empirical coverage, obtained by resampling complete independent data trials. Band widths are not artificially reduced; genuine overlap remains. The dashed Exact (loop) curve is the exact finite-support posterior for the SAME loop observations. Tree BP coincides with its own exact posterior. The dotted Ideal line is observed = nominal.\n'
        '- A 95% posterior credible interval targets 95% probability mass under the given posterior/belief; discrete atoms can make the included mass slightly larger. A 95% bootstrap confidence interval is a repeated-sampling interval for a reported mean. These are different objects.\n'
        '- Cycle-length comparison overlays node-i and all-agent mean variance ratios on the same axes with different colors/markers. Whiskers are 95% bootstrap confidence intervals. The topology strip identifies the actual three closures; orange highlights the added edge. Closing-edge endpoints, direction and degrees also change; ring length is not the sole physical change. No intermediate ring lengths or fitted smooth curves are invented.\n'
        '- Message-residual shading is a pointwise 95% bootstrap confidence interval for the trial mean. Exact zeros are displayed at 1e-16 for logarithmic plotting only. The tolerance is recorded in the experiment configuration.\n'
        '- The scalar Gaussian control fixes unary precision a=1, local edge coupling b within each curve and degree two. It is an auxiliary model, not the nonlinear 2D ranging experiment.\n'
        '- Figure titles and explanatory footers are omitted for paper use; interpretation belongs in these captions. NEES is not used in the current figure/report set; previously computed raw statistics remain archived.\n',encoding='utf-8')
    atomic_json(root/'figures/index.json',{'plot_source_sha256':sha256(__file__),
                                          'paper_node_labels':{c.name:dict(zip(graph_spec(c).labels,
                                              _paper_layout(graph_spec(c),c.graph_family)[1])) for c in settings.conditions},
                                          'main_calibration_scope':{c.name:'node i only' if c.graph_family=='branched7'
                                              else 'all agents' for c in settings.conditions},
                                          'main_covariance_iterations':[0,min(6,settings.plot_rounds,settings.iterations)],
                                          'files':{path.relative_to(root).as_posix():sha256(path) for path in paths}})
    return paths
