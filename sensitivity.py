# -*- coding: utf-8 -*-
"""Structural sensitivity: does the ranking survive alternative mechanism assumptions?"""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import sim
from exp import make_features, fit_ml, metrics, F0
import stats as st

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
SEEDS = [1, 2, 3, 4, 5]
ROWS = []

def run_variant(rain_form, use_hum):
    """Rebuild the panel under an alternative mechanism, then refit the F0 arm."""
    orig = sim.hum_mod
    if not use_hum:
        sim.hum_mod = lambda h: np.ones_like(np.asarray(h, float))
    try:
        res = {}
        for sd in SEEDS:
            p = sim.build_panel(seed=sd, scenario='L', rain_form=rain_form)
            df = make_features(p); df['y'] = df.groupby('grid')['BI'].shift(-1)
            sub = df.dropna(subset=F0 + ['y'])
            yp, yt = [], []
            for ty in (2019, 2020, 2021, 2022):
                tr = sub[sub.year < ty]; te = sub[sub.year == ty]
                if len(tr) < 100 or len(te) == 0: continue
                pr = fit_ml('XGBoost', tr[F0].values, tr['y'].values, te[F0].values, sd)
                yp.extend(pr); yt.extend(te['y'].values)
            m = metrics(yt, yp)
            # SARIMA on the same variant panel, one seed only (expensive)
            res.setdefault('XGBoost', []).append(m['R2'])
            res.setdefault('BImean', []).append(float(p['BI'].mean()))
        return res
    finally:
        sim.hum_mod = orig

for rain_form, use_hum in itertools.product(['saturating', 'linear', 'gate'], [True, False]):
    r = run_variant(rain_form, use_hum)
    ROWS.append(dict(rain_form=rain_form, humidity=use_hum,
                     XGB_R2=float(np.mean(r['XGBoost'])),
                     XGB_SD=float(np.std(r['XGBoost'], ddof=1)),
                     BI_mean=float(np.mean(r['BImean'])), n_panels=len(r['XGBoost'])))
    print(ROWS[-1], flush=True)

pd.DataFrame(ROWS).to_csv(os.path.join(OUT, 'sensitivity_structure.csv'), index=False)
print('saved')
