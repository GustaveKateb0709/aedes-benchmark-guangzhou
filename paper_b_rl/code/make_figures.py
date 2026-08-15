#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Publication figures (English labels, 300 dpi PNG + PDF).

Reads results/*.csv produced by evaluate.py / sensitivity.py and the
training logs under results/monitor/, and writes Figure1..Figure8 into
../figures/.
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(BASE, '..', 'results'))
FIGS = os.path.abspath(os.path.join(BASE, '..', 'figures'))
os.makedirs(FIGS, exist_ok=True)

C_PPO = '#b2182b'
C_RULE = '#2166ac'
C_CAL = '#1b7837'
C_NONE = '#7f7f7f'
ARM_STYLE = {'none': ('No intervention', C_NONE),
             'threshold': ('Threshold rule (BI≥15 → spray)', C_RULE),
             'calendar': ('Preventive calendar rule', C_CAL)}
PPO_LABELS = {'ppo_a0.1_s0_acts012': 'PPO (seed 0)',
              'ppo_a0.1_s1_acts012': 'PPO (seed 1)',
              'ppo_a0.1_s2_acts012': 'PPO (seed 2)'}


def save(fig, name):
    fig.savefig(os.path.join(FIGS, f'{name}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIGS, f'{name}.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  saved', name, flush=True)


def fig1_framework():
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.axis('off')

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02',
                                    fc=fc, ec='#333333', lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=9.5)

    box(0.02, 0.30, 0.24, 0.40, 'ERA5 reanalysis\nweather 2015–2022\n(10 grids)', '#dbe9f6')
    box(0.33, 0.30, 0.26, 0.40, 'Mechanistic PFI model\nweekly baseline BI\nper grid', '#e8f4e5')
    box(0.66, 0.52, 0.30, 0.34, 'MosquitoEnv (MDP)\nstate: weather, BI, residual\nprotection, last action', '#fdeeda')
    box(0.66, 0.06, 0.30, 0.30, 'PPO agent\naction per grid: none /\nsource reduction / spraying', '#f6dcdc')
    box(0.33, 0.02, 0.26, 0.20, 'Reward\n− excess BI² − α·cost', '#f0f0f0')

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle='-|>', mutation_scale=14,
                                     color='#333333', lw=1.2))
    arrow(0.26, 0.50, 0.33, 0.50)
    arrow(0.59, 0.50, 0.66, 0.62)
    arrow(0.81, 0.52, 0.81, 0.36)
    arrow(0.66, 0.18, 0.59, 0.14)
    arrow(0.46, 0.22, 0.46, 0.30)
    ax.text(0.62, 0.90, 'Weekly decision loop (one step = one week)',
            fontsize=10, ha='center', style='italic', color='#555555')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, 'Figure1')


def fig2_learning_curves():
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for f, lab, c in [
            ('ppo_a0.1_s0_acts012.monitor.csv', 'seed 0', '#b2182b'),
            ('ppo_a0.1_s1_acts012.monitor.csv', 'seed 1', '#d6604d'),
            ('ppo_a0.1_s2_acts012.monitor.csv', 'seed 2', '#f4a582')]:
        p = os.path.join(RES, 'monitor', f)
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p, skiprows=1)
        r = d['r'].rolling(200, min_periods=20).mean()
        ax.plot(np.arange(len(r)), r, label=f'PPO α=0.1 ({lab})', color=c, lw=1.2)
    ax.set_xlabel('Training episode (1 episode = 1 simulated year)')
    ax.set_ylabel('Episode reward (200-episode moving average)')
    ax.set_title('PPO training convergence (400k steps, three seeds)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    save(fig, 'Figure2')


def _arm_label(arm):
    if arm in PPO_LABELS:
        return PPO_LABELS[arm], C_PPO
    return ARM_STYLE.get(arm, (arm, C_PPO))


def fig3_trajectories():
    arms = ['none', 'threshold', 'calendar']
    ppo = sorted(glob.glob(os.path.join(RES, 'trajectory_ppo_a0.1_s0*.csv')))
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    for arm in arms:
        p = os.path.join(RES, f'trajectory_{arm}.csv')
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        bi_cols = [c for c in d.columns if c.startswith('BI_')]
        mb = d[bi_cols].mean(axis=1)
        lab, c = _arm_label(arm)
        ax.plot(d['week_in_year'], mb, label=lab, color=c, lw=1.4)
    if ppo:
        d = pd.read_csv(ppo[0])
        bi_cols = [c for c in d.columns if c.startswith('BI_')]
        ax.plot(d['week_in_year'], d[bi_cols].mean(axis=1),
                label='PPO dynamic policy', color=C_PPO, lw=2.0)
        ax2 = ax.twinx()
        ax2.bar(d['week_in_year'], d['n_sr'], color='#1b7837', alpha=0.25,
                label='grids: source reduction')
        if d['n_sp'].sum() > 0:
            ax2.bar(d['week_in_year'], d['n_sp'], bottom=d['n_sr'],
                    color='#2166ac', alpha=0.25, label='grids: spraying')
        ax2.set_ylabel('Interventions per week (grids)')
        ax2.set_ylim(0, 40)
        ax2.legend(fontsize=8, loc='upper left')
    ax.axhline(5, color='black', ls=':', lw=1)
    ax.text(1, 5.3, 'BI = 5 (transmission-risk threshold)', fontsize=8)
    ax.set_xlabel('Week of year, 2022 (held-out)')
    ax.set_ylabel('Mean BI across 10 grids')
    ax.set_title('Counterfactual 2022 trajectories under the four control arms')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.25)
    save(fig, 'Figure3')


def fig4_cost_bi():
    p = os.path.join(RES, 'counterfactual_summary.csv')
    s = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    # stagger annotation offsets so the three PPO seeds stay readable
    offsets = {'none': (8, 6), 'threshold': (8, 6), 'calendar': (8, 6),
               'ppo_a0.1_s0_acts012': (-24, 12),
               'ppo_a0.1_s1_acts012': (34, -6),
               'ppo_a0.1_s2_acts012': (-10, -18)}
    for _, row in s.iterrows():
        lab, c = _arm_label(row['arm'])
        ax.errorbar(row['cost'], row['mean_bi'],
                    xerr=row['cost_sd'], yerr=row['mean_bi_sd'],
                    fmt='o', color=c, capsize=4, markersize=9)
        ax.annotate(lab, (row['cost'], row['mean_bi']),
                    textcoords='offset points',
                    xytext=offsets.get(row['arm'], (8, 6)), fontsize=8,
                    ha='center' if row['arm'].startswith('ppo') else 'left')
    ax.set_xlabel('Total control cost in 2022 (relative units, mean ± sd)')
    ax.set_ylabel('Mean BI in 2022 (mean ± sd)')
    ax.set_title('Cost–health trade-off by control arm (paired noise replicates)')
    ax.grid(alpha=0.25)
    save(fig, 'Figure4')


def fig5_alpha():
    p = os.path.join(RES, 'alpha_sweep.csv')
    if not os.path.exists(p):
        return
    d = pd.read_csv(p).sort_values('alpha')
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(d['alpha'], d['mean_bi'], 'o-', color=C_PPO, label='Mean BI')
    ax.set_xlabel('Cost weight α')
    ax.set_ylabel('Mean BI (2022 held-out)', color=C_PPO)
    ax2 = ax.twinx()
    ax2.plot(d['alpha'], d['total_cost'], 's--', color=C_RULE, label='Total cost')
    ax2.set_ylabel('Total cost (relative units)', color=C_RULE)
    ax.set_title('Policy response to the cost weight α (retrained per value)')
    ax.grid(alpha=0.25)
    save(fig, 'Figure5')


def fig6_robustness():
    p = os.path.join(RES, 'robustness_perturbation.csv')
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    nom = d[d['scenario'] == 'nominal']['mean_bi'].iloc[0]
    d = d[d['scenario'] != 'nominal']
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    colors = [C_PPO if v > nom else C_CAL for v in d['mean_bi']]
    ax.bar(d['scenario'], d['mean_bi'], color=colors)
    ax.axhline(nom, color='black', ls='--', lw=1)
    ax.text(len(d) - 0.5, nom + 0.05, f'nominal ({nom:.2f})', fontsize=8,
            ha='right')
    ax.set_ylabel('Mean BI (2022, fixed trained policy)')
    ax.set_title('Robustness to perturbed intervention dynamics (no retraining)')
    ax.tick_params(axis='x', rotation=25)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    save(fig, 'Figure6')


def fig7_ablation():
    p = os.path.join(RES, 'action_ablation.csv')
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    labels = {'(0, 1, 2)': 'both tools', '(0, 1)': 'source reduction only',
              '(0, 2)': 'spraying only'}
    d['label'] = d['action_set'].map(labels)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(d['label'], d['mean_bi'], color=[C_PPO, C_CAL, C_RULE])
    for i, (v, c) in enumerate(zip(d['mean_bi'], d['total_cost'])):
        ax.text(i, v + 0.05, f'BI {v:.2f}\ncost {c:.0f}', ha='center',
                fontsize=9)
    ax.set_ylim(0, max(d['mean_bi']) * 1.35)
    ax.set_ylabel('Mean BI (2022 held-out)')
    ax.set_title('Action-space ablation')
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    save(fig, 'Figure7')


def fig8_policy():
    """Intervention intensity versus the week's *baseline* (uncontrolled)
    risk level: mean grids treated per week, binned by the baseline BI the
    week would show without intervention (PFI × urbanisation scale)."""
    tp = os.path.join(RES, 'trajectory_ppo_a0.1_s0_acts012.csv')
    pp = os.path.join(RES, 'panel_data.csv')
    if not (os.path.exists(tp) and os.path.exists(pp)):
        return
    d = pd.read_csv(tp)
    panel = pd.read_csv(pp)
    p22 = panel[panel['year'] == 2022].copy()
    p22['mu'] = p22['PFI'] * p22['pop_density']
    baseline = p22.groupby('week_in_year')['mu'].mean()
    j = d.set_index('week_in_year').join(baseline.rename('baseline'))
    bins = [0, 2, 4, 6, 8, 10, 15, 100]
    labels = ['0–2', '2–4', '4–6', '6–8', '8–10', '10–15', '>15']
    j['bin'] = pd.cut(j['baseline'], bins=bins, labels=labels)
    g = j.groupby('bin', observed=True)[['n_sr', 'n_sp']].mean()
    counts = j.groupby('bin', observed=True).size()
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(g))
    ax.bar(x - 0.2, g['n_sr'], width=0.4, color=C_CAL, label='source reduction')
    ax.bar(x + 0.2, g['n_sp'], width=0.4, color=C_RULE, label='space spraying')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{lab}\n(n={int(counts.get(lab, 0))})'
                        for lab in g.index], fontsize=8)
    ax.set_xlabel('Baseline mean BI of the week (no-intervention level, all grids)')
    ax.set_ylabel('Mean number of grids treated')
    ax.set_title('Learned policy behaviour: intervention intensity vs. baseline risk')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    save(fig, 'Figure8')


def main():
    print('generating figures ...', flush=True)
    fig1_framework()
    fig2_learning_curves()
    fig3_trajectories()
    fig4_cost_bi()
    fig5_alpha()
    fig6_robustness()
    fig7_ablation()
    fig8_policy()
    print('figures ->', FIGS, flush=True)


if __name__ == '__main__':
    main()
