#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""leave-one-grid-out non-spatial baselines (torch-free).

Trains XGBoost on 9 grids (years 2015-2018) and evaluates on the held-out
grid (years 2019-2022): the 'dense weather / sparse BI monitoring' scenario.
Kept separate from run_gat.py so that XGBoost never shares a process
with torch (memory safety).
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

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
os.makedirs(RES, exist_ok=True)
sys.path.insert(0, BASELINE_DIR)
import pipeline as base  # renamed from pipeline_redo during repo unification

SEED = 2024


def main():
    from xgboost import XGBRegressor
    panel = pd.read_csv(os.path.join(BASELINE_DIR, 'results', 'panel_data.csv'))
    df = base.prepare_Xy(panel)          # baseline features + BI_future, dropna
    rows = []
    for hold in base.GRIDS:
        tr = df[(df['grid'] != hold) & (df['year'] < 2019)]
        te = df[(df['grid'] == hold) & (df['year'] >= 2019)]
        m = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                         random_state=SEED)
        m.fit(tr[base.FEATS], tr['BI_future'])
        yp = m.predict(te[base.FEATS])
        rows.append({'held_out_grid': hold,
                     'XGB_RMSE': round(float(np.sqrt(mean_squared_error(
                         te['BI_future'], yp))), 3),
                     'XGB_R2': round(float(r2_score(te['BI_future'], yp)), 3),
                     'n': len(te)})
        print(f'  held-out {hold:9s}: XGB RMSE={rows[-1]["XGB_RMSE"]}', flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, 'loo_xgb_baseline.csv'), index=False)
    print('\n', out.to_string(index=False), flush=True)
    print('saved ->', os.path.join(RES, 'loo_xgb_baseline.csv'), flush=True)


if __name__ == '__main__':
    main()
