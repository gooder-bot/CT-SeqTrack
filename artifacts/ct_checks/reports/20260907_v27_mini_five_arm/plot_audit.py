"""Static engineering figures; source tables retained alongside exports."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

D = Path(__file__).resolve().parent
ROOT = D.parents[3]
data = json.loads((D/'metrics_summary.json').read_text(encoding='utf8'))
curves = list(csv.DictReader((D/'dev_validation_curves.csv').open(encoding='utf-8-sig')))
losses = list(csv.DictReader((D/'b0_view0_training_loss.csv').open(encoding='utf-8-sig')))
arms = ['B0', 'B1-CfC', 'B1-GRU', 'B1+B2', 'Full']
colors = ['#333333', '#0072B2', '#009E73', '#D55E00', '#AA4499']
markers = ['o', 's', '^', 'D', 'v']
with plt.rc_context({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42}):
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    for arm, color, marker in zip(arms, colors, markers):
        rr = [r for r in curves if r['arm'] == arm]
        axs[0,0].plot([int(r['epoch']) for r in rr], [float(r['utility']) for r in rr],
                      color=color, marker=marker, markersize=4, label=arm, linewidth=1.3)
        rr = [r for r in losses if r['arm'] == arm]
        axs[0,1].plot([int(r['epoch']) for r in rr], [float(r['view0_loss_mean']) for r in rr],
                      color=color, label=arm, linewidth=1.2, marker=marker, markevery=12, markersize=3)
    axs[0,0].set(title='A. Observed dev scores (one scene, 321 frames)', xlabel='Completed epoch',
                 ylabel='U = (Success + Precision) / 2 (%)', ylim=(0,40))
    axs[0,0].legend(ncol=3, fontsize=8)
    axs[0,1].set(title='B. B0 canonical-view training loss', xlabel='Completed epoch',
                 ylabel='Mean loss over 1,057 batches', yscale='log')
    gate_rows = []
    stages = ['Novel\nvisible', 'Novel\nsupport', '768\npool', '256\nselected', 'Top\nmode']
    for arm in ['B1+B2', 'Full']:
        rec = next(r for r in data['summary'] if r['arm']==arm)
        f = ROOT/'output'/rec['run']/'lightning_logs/version_0/dev_diagnostics/epoch_60_endpoints.csv'
        rows = [r for r in csv.DictReader(f.open()) if r['is_initial']=='False']
        values = [sum(float(r.get('acquisition_global_novel_target_count') or 0) for r in rows)]
        values += [rec[k+'_target_points'] for k in ['support_novel','prepool','selected','consensus_top_mode']]
        idx = arms.index(arm)
        axs[1,0].plot(stages, values, color=colors[idx], marker=markers[idx], label=arm)
        for x, y in enumerate(values):
            axs[1,0].annotate(f'{y:,.0f}', (x,y), textcoords='offset points',
                              xytext=(0,6 if arm=='Full' or y<600 else -13), ha='center', fontsize=8, color=colors[idx])
        true_count = sum(float(r['ct_search_extension_selected_count'])>0 for r in rows)
        old_count = sum(r['structural_available']=='True' for r in rows)
        actions = sum(r['action_applied']=='True' for r in rows)
        gate_rows.append([true_count, old_count, actions])
    axs[1,0].set(title='C. Target points across the evidence chain (epoch 60)',
                 ylabel='Sum of unique target points over endpoints', ylim=(0,14500))
    axs[1,0].legend(fontsize=8)
    positions = np.arange(3)
    for i, (arm, values) in enumerate(zip(['B1+B2','Full'],gate_rows)):
        bars=axs[1,1].bar(positions+(i-.5)*.32,values,.30,color=colors[arms.index(arm)],
                         label=arm,hatch='' if i==0 else '//')
        axs[1,1].bar_label(bars,padding=3,fontsize=9)
    axs[1,1].set(title='D. Legacy presence gate blocks structural candidates',
                 xticks=positions,xticklabels=['Nonempty\nselected evidence','Legacy gate\nreported valid','Actually\napplied'],
                 ylabel='Frames (309 prediction endpoints)',ylim=(0,210))
    axs[1,1].legend(fontsize=8)
    for ax in axs.flat:
        ax.grid(axis='y',alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle('CT-SeqTrack v27 mini audit | final epoch 60 | seed 42',fontsize=15)
    fig.savefig(D/'audit_overview.png',dpi=180)
    fig.savefig(D/'audit_overview.pdf')
    plt.close(fig)

(D/'figure_provenance.json').write_text(json.dumps({
    'purpose':'engineering audit; no journal-specific formatting claim',
    'source_tables':['dev_validation_curves.csv','b0_view0_training_loss.csv','metrics_summary.json'],
    'source_endpoints':'output/20260905-173*/lightning_logs/version_0/dev_diagnostics/epoch_60_endpoints.csv',
    'transformations':{'A':'unsmoothed observed every-five-epoch U; connecting segments are visual guides',
                      'B':'per-epoch arithmetic batch mean; logarithmic y axis',
                      'C':'sum of unique target point IDs at each endpoint, then sum over all 309 prediction endpoints',
                      'D':'nonempty selected count versus recorded structural flag versus recorded applied flag'},
    'uncertainty':'none shown; one seed and one dev scene; curves are not independent replicates',
    'known_limits':['B0 trajectories differ across arms','Full is uncalibrated observation output',
                    'historical SeqTrack uses a different evaluation split and is excluded from this chart'],
    'alt_text':'All dev curves remain low while training loss falls. Most visible novel target points never reach the support. Selection retains most prepool target points. The old presence gate leaves only 2 and 12 candidates marked valid, compared with 177 and 140 frames with selected evidence.'
},indent=2),encoding='utf8')
print('Wrote audit_overview.png, audit_overview.pdf and figure_provenance.json')
