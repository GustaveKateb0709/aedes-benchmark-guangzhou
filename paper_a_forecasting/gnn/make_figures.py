#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""publication figures (English labels, 300 dpi PNG + PDF).

Reads the CSVs produced by run_gat.py / run_dml.py / loo_baselines.py.
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
def _find_baseline_dir():
    """Resolve the baseline pipeline directory (panel data + features):
    the BASELINE_DIR environment variable, or the package root of this
    self-contained code-and-data package."""
    cands = [os.environ.get('BASELINE_DIR', ''),
             os.path.abspath(os.path.join(BASE, '..'))]      # this package
    for c in cands:
        if c and os.path.exists(os.path.join(c, 'results', 'panel_data.csv')):
            return c
    raise FileNotFoundError('panel_data.csv not found; set BASELINE_DIR')


BASELINE_DIR = _find_baseline_dir()
RES = os.path.abspath(os.path.join(BASE, '..', 'results'))
FIGS = os.path.abspath(os.path.join(BASE, '..', 'figures'))
os.makedirs(FIGS, exist_ok=True)
sys.path.insert(0, BASELINE_DIR)
import pipeline as base
from run_gat import (load_coords, distance_matrix, build_adjacency,
                        GRIDS, N)

GRID_EN = {'conghua': 'Conghua', 'huadu': 'Huadu', 'zengcheng': 'Zengcheng',
           'baiyun': 'Baiyun', 'tianhe': 'Tianhe', 'yuexiu': 'Yuexiu',
           'haizhu': 'Haizhu', 'panyu': 'Panyu', 'nansha': 'Nansha',
           'huangpu': 'Huangpu'}
C_MAIN = '#2166ac'
C_GAT = '#b2182b'
C_XGB = '#1b7837'
C_GREY = '#7f7f7f'


def save(fig, name):
    fig.savefig(os.path.join(FIGS, f'{name}.png'), dpi=300,
                bbox_inches='tight')
    fig.savefig(os.path.join(FIGS, f'{name}.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('  saved', name, flush=True)


def fig1_graph():
    coords = load_coords()
    dist = distance_matrix(coords)
    A = build_adjacency(dist, 'knn', 3)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    lon = [coords[g][1] for g in GRIDS]
    lat = [coords[g][0] for g in GRIDS]
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] > 0:
                ax.plot([lon[i], lon[j]], [lat[i], lat[j]], '-',
                        color=C_GREY, lw=0.8, alpha=0.6, zorder=1)
    sizes = None
    ax.scatter(lon, lat, s=160, color=C_MAIN, zorder=2, edgecolor='white',
               linewidth=1.2)
    # per-grid label offsets (points) chosen to avoid overlaps between
    # the closely spaced central grids (Tianhe / Haizhu / Yuexiu)
    offsets = {'conghua': (8, 6), 'huadu': (8, 6), 'zengcheng': (8, 6),
               'baiyun': (8, 6), 'tianhe': (10, 8), 'yuexiu': (-8, 10),
               'haizhu': (10, -16), 'panyu': (8, 6), 'nansha': (8, 6),
               'huangpu': (8, 6)}
    for i, g in enumerate(GRIDS):
        ha = 'right' if offsets[g][0] < 0 else 'left'
        ax.annotate(GRID_EN[g], (lon[i], lat[i]), ha=ha,
                    textcoords='offset points', xytext=offsets[g], fontsize=9)
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title('Study grids and k-nearest-neighbor graph (k = 3), Guangzhou')
    ax.grid(alpha=0.25)
    save(fig, 'Figure1')


def fig2_data_overview():
    panel = pd.read_csv(os.path.join(BASELINE_DIR, 'results', 'panel_data.csv'))
    th = panel[panel['grid'] == 'tianhe'].reset_index(drop=True)
    x = np.arange(len(th))
    fig, ax = plt.subplots(3, 1, figsize=(9.5, 7.5), sharex=True)
    ax[0].plot(x, th['T'], color='#b2182b', lw=0.8)
    ax[0].set_ylabel('Weekly mean\ntemperature (°C)')
    ax[1].bar(x, th['Rain'], color='#2166ac', width=1.0)
    ax[1].set_ylabel('Weekly rainfall (mm)')
    ax[2].plot(x, th['PFI'], color='#1b7837', lw=0.9, label='PFI (mechanistic)')
    ax[2].plot(x, th['BI'], color=C_GREY, lw=0.5, alpha=0.7,
               label='Simulated BI (label)')
    ax[2].set_ylabel('PFI / simulated BI')
    ax[2].set_xlabel('Week (2015–2022)')
    ax[2].legend(loc='upper right', fontsize=8)
    ticks = [0, 52, 104, 156, 208, 260, 312, 364]
    ax[2].set_xticks(ticks)
    ax[2].set_xticklabels([str(2015 + t // 52) for t in ticks])
    fig.suptitle('ERA5-driven PFI and simulated Breteau index — Tianhe grid')
    fig.tight_layout()
    save(fig, 'Figure2')


def fig3_model_comparison():
    tbl = pd.read_csv(os.path.join(RES, 'gat_baseline_compare.csv'),
                      index_col=0)
    order = ['XGBoost', 'RandomForest', 'Lasso', 'SARIMA', 'LSTM',
             'GAT (mean of 3 seeds)']
    order = [o for o in order if o in tbl.index]
    labels = ['GAT' if o.startswith('GAT') else o for o in order]
    colors = [C_GAT if o.startswith('GAT') else C_MAIN for o in order]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    rmse = tbl.loc[order, 'RMSE'].values
    r2 = tbl.loc[order, 'R2'].values
    rmse_err = tbl.loc[order, 'RMSE_sd'].fillna(0).values \
        if 'RMSE_sd' in tbl.columns else np.zeros(len(order))
    r2_err = tbl.loc[order, 'R2_sd'].fillna(0).values \
        if 'R2_sd' in tbl.columns else np.zeros(len(order))
    axes[0].bar(labels, rmse, color=colors, yerr=rmse_err, capsize=3)
    axes[0].set_ylabel('RMSE (lower is better)')
    for i, v in enumerate(rmse):
        axes[0].text(i, v + rmse_err[i] + 0.08, f'{v:.2f}', ha='center',
                     fontsize=8)
    axes[1].bar(labels, r2, color=colors, yerr=r2_err, capsize=3)
    axes[1].set_ylabel('R² (higher is better)')
    axes[1].set_ylim(0, 1.0)
    for i, v in enumerate(r2):
        axes[1].text(i, v + r2_err[i] + 0.02, f'{v:.2f}', ha='center',
                     fontsize=8)
    for a in axes:
        a.tick_params(axis='x', rotation=25)
        a.grid(axis='y', alpha=0.25)
    fig.suptitle('One-week-ahead BI prediction, rolling-origin evaluation '
                 '(test years 2019–2022)')
    fig.tight_layout()
    save(fig, 'Figure3')


def fig4_loo():
    gat = pd.read_csv(os.path.join(RES, 'loo_gat.csv'))
    xgb = pd.read_csv(os.path.join(RES, 'loo_xgb_baseline.csv'))
    g = gat.groupby('held_out_grid')['GAT_RMSE'].mean().reindex(xgb['held_out_grid'])
    x_ = np.arange(len(xgb))
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.bar(x_ - 0.2, g.values, width=0.4, color=C_GAT, label='GAT (spatial)')
    ax.bar(x_ + 0.2, xgb['XGB_RMSE'].values, width=0.4, color=C_XGB,
           label='XGBoost (non-spatial)')
    ax.set_xticks(x_)
    ax.set_xticklabels([GRID_EN[g] for g in xgb['held_out_grid']], rotation=25)
    ax.set_ylabel('LOO RMSE (2019–2022)')
    ax.set_title('Leave-one-grid-out extrapolation: held-out grid predicted '
                 'from the remaining nine')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    save(fig, 'Figure4')


def fig5_graph_sensitivity():
    sens = pd.read_csv(os.path.join(RES, 'graph_sensitivity.csv'))
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    colors = [C_GAT if 'main' in g else (C_GREY if 'MLP' in g else C_MAIN)
              for g in sens['graph']]
    ax.bar(sens['graph'], sens['RMSE'], color=colors)
    for i, v in enumerate(sens['RMSE']):
        ax.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=8)
    ax.set_ylabel('RMSE (rolling-origin 2019–2022)')
    ax.set_title('Graph-structure sensitivity of the spatial model')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    save(fig, 'Figure5')


def fig6_dml_lag():
    d = pd.read_csv(os.path.join(RES, 'dml_lag_effects.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, tvar, unit, title in [
            (axes[0], 'Rain', 'BI units per mm', 'Rainfall'),
            (axes[1], 'T', 'BI units per °C', 'Temperature')]:
        sub = d[d['treatment'] == tvar].sort_values('lag_weeks')
        ax.axhline(0, color='black', lw=0.8)
        ax.plot(sub['lag_weeks'], sub['naive_ols'], 'o--', color=C_GREY,
                label='Naive (unadjusted)', lw=1)
        ax.errorbar(sub['lag_weeks'], sub['dml_ate'],
                    yerr=1.96 * sub['se_cluster'], fmt='o-', color=C_MAIN,
                    capsize=3, label='DML estimate (95% CI)')
        ax.plot(sub['lag_weeks'], sub['truth'], 'D', color='#b2182b',
                markersize=7, label='Ground truth (PFI mechanism)')
        ax.set_xlabel('Lag (weeks)')
        ax.set_ylabel(f'Estimated effect ({unit})')
        ax.set_title(f'{title}: DML vs. ground truth')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, 'Figure6')


def fig7_dml_dose():
    d = pd.read_csv(os.path.join(RES, 'dml_dose_response.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, tvar, xlabel, title in [
            (axes[0], 'Rain', 'Weekly rainfall (mm)',
             'Rainfall dose–response (aligned at median dose)'),
            (axes[1], 'T', 'Weekly mean temperature (°C)',
             'Temperature dose–response (aligned at median dose)')]:
        sub = d[d['treatment'] == tvar].sort_values('value')
        ax.plot(sub['value'], sub['naive_response'], '--', color=C_GREY,
                lw=1.2, label='Naive (unadjusted spline)')
        ax.plot(sub['value'], sub['dml_response'], '-', color=C_MAIN, lw=1.6,
                label='DML spline estimate')
        ax.fill_between(sub['value'].values.astype(float),
                        (sub['dml_response'] - 1.96 * sub['dml_se']).values.astype(float),
                        (sub['dml_response'] + 1.96 * sub['dml_se']).values.astype(float),
                        color=C_MAIN, alpha=0.18, lw=0)
        ax.plot(sub['value'], sub['truth_response'], color='#b2182b', lw=1.6,
                label='Ground truth (PFI partial dependence)')
        ax.axhline(0, color='black', lw=0.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Δ simulated BI vs. median dose')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, 'Figure7')


def main():
    print('generating publication figures ...', flush=True)
    fig1_graph()
    fig2_data_overview()
    fig3_model_comparison()
    fig4_loo()
    fig5_graph_sensitivity()
    fig6_dml_lag()
    fig7_dml_dose()
    print('figures ->', FIGS, flush=True)


if __name__ == '__main__':
    main()
