#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Panel-robust Diebold-Mariano tests (revision 2026-08-15).

The original dm_test() in pipeline.py pools all grid-week loss differences and
applies a Newey-West HAC correction for serial correlation only. Because the
rolling-origin predictions form a 10-grid panel whose errors are cross-
sectionally correlated (shared weather shocks), the pooled variance can be too
small and p-values overstate significance. This script reports:

  (i)   HAC DM with lag orders 1-8 (robustness of the original lag-4 choice)
  (ii)  cluster-robust DM: variance estimated by summing loss-difference
        means per grid cluster (10 clusters), with a small-sample correction
  (iii) a grid-level paired t-test on per-grid mean loss differences
        (the most conservative "effective n = 10" view)

Input: results/preds_*.csv from the de-leaked pipeline run.
Output: results/dm_tests_panelrobust.csv
"""
import os
import numpy as np
import pandas as pd
from scipy import stats as sst

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, 'results')

MODELS = ['SARIMA', 'Lasso', 'RandomForest', 'LSTM']


def dm_hac(d, lags):
    """Newey-West HAC DM statistic at a given lag order."""
    n = len(d)
    m = d.mean()
    g0 = np.mean((d - m) ** 2)
    var = g0
    for L in range(1, lags + 1):
        g = np.mean((d[L:] - m) * (d[:-L] - m))
        var += 2 * (1 - L / (lags + 1)) * g
    var = max(var, 1e-12)
    dm = m / np.sqrt(var / n)
    return dm, 2 * (1 - sst.norm.cdf(abs(dm)))


def dm_cluster(d, groups):
    """Cluster-robust (by grid) DM: variance of the mean from cluster sums.

    d: loss difference series; groups: grid labels aligned with d.
    Let m = mean(d), and S_g = sum of d within cluster g, G clusters, n obs.
    Var(m) = G/(G-1) * (n-1)/(n-G) * sum_g n_g^2 (S_g/n_g - m)^2 / n^2 ... we use
    the standard cluster-robust sandwich in the simple-mean form:
    Var(m) = [G/(G-1)] * [(n-1)/(n-G)] * sum_g n_g^2 (dbar_g - m)^2 / n^2.
    """
    n = len(d)
    m = d.mean()
    G = len(np.unique(groups))
    var = 0.0
    for g in np.unique(groups):
        dg = d[groups == g]
        ng = len(dg)
        var += ng ** 2 * (dg.mean() - m) ** 2
    var *= (G / (G - 1)) * ((n - 1) / (n - G)) / n ** 2
    var = max(var, 1e-12)
    dm = m / np.sqrt(var)
    return dm, 2 * (1 - sst.norm.cdf(abs(dm)))


def main():
    preds = {name: pd.read_csv(os.path.join(RES, f'preds_{name}.csv'))
             for name in ['XGBoost'] + MODELS if os.path.exists(
                 os.path.join(RES, f'preds_{name}.csv'))}
    ref = preds['XGBoost']
    rows = []
    for other in [m for m in MODELS if m in preds]:
        m = ref.merge(preds[other], on=['grid', 'year', 'week_in_year'],
                      suffixes=('_xgb', '_o'))
        e1 = (m['y_true_xgb'] - m['y_pred_xgb']).values
        e2 = (m['y_true_o'] - m['y_pred_o']).values
        d = e1 ** 2 - e2 ** 2                      # >0: XGB worse
        g = m['grid'].values
        row = {'comparison': f'XGBoost vs {other}', 'n': len(d)}
        for L in (1, 2, 4, 8):
            dm, p = dm_hac(d, L)
            row[f'dm_hac{L}'] = round(dm, 3)
            row[f'p_hac{L}'] = round(p, 4)
        dm, p = dm_cluster(d, g)
        row['dm_cluster'] = round(dm, 3)
        row['p_cluster'] = round(p, 4)
        # grid-level paired t-test on per-grid mean loss differences
        per_grid = pd.DataFrame({'d': d, 'g': g}).groupby('g')['d'].mean()
        t, pt = sst.ttest_1samp(per_grid.values, 0.0)
        row['grid_t'] = round(float(t), 3)
        row['grid_t_p'] = round(float(pt), 4)
        rows.append(row)
        print(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, 'dm_tests_panelrobust.csv'), index=False)
    print(f'\nsaved -> {os.path.join(RES, "dm_tests_panelrobust.csv")}')


if __name__ == '__main__':
    main()
