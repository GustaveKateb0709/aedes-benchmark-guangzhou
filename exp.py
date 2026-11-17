#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment driver for simulator v2.

Design grid
-----------
scenario : M (memoryless) | L (lagged) | S (lagged + spatial coupling)
arm      : F0  14 weather+season features (primary)
           F1  F0 + BI(t), BI(t-1)              information-set parity
           F2  F0 + grid dummies                legitimate spatial identity
           F3  F0 + urbanisation proxy          the v1 "leakage" feature
           F4  F0 + noisified proxy             proxy with 10% / 30% error
           F5  F1 + F2                          everything observable
model    : Lasso | RandomForest | XGBoost | SARIMA (feature-free)

Outputs one tidy CSV per (scenario, seed) with per-observation predictions,
plus a metrics CSV. Resumable: existing files are skipped.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim

OUT = os.environ.get("AEGIS_OUT", "/tmp/at_work/v2/results")
os.makedirs(OUT, exist_ok=True)

F0 = ["GDD", "GDD_cum2", "GDD_cum4", "Rain_cum2", "Rain_cum4",
      "sin_week", "cos_week", "T_lag1", "Rain_lag1", "Hum_lag1",
      "TempRain_inter", "T", "Rain", "Hum"]
GD = ["gd_" + g for g in sim.GRIDS[1:]]


def make_features(df, proxies=("pop_density",)):
    df = df.sort_values(["grid", "week_idx"]).copy()
    df["GDD"] = (df["T"] - sim.T_DEV0).clip(lower=0)
    for lb in (2, 4):
        df[f"GDD_cum{lb}"] = df.groupby("grid")["GDD"].transform(lambda s: s.rolling(lb).sum())
        df[f"Rain_cum{lb}"] = df.groupby("grid")["Rain"].transform(lambda s: s.rolling(lb).sum())
    df["sin_week"] = np.sin(2 * np.pi * df["week_in_year"] / 52.0)
    df["cos_week"] = np.cos(2 * np.pi * df["week_in_year"] / 52.0)
    for c in ("T", "Rain", "Hum"):
        df[c + "_lag1"] = df.groupby("grid")[c].shift(1)
    df["TempRain_inter"] = (df["T"] - sim.T_OPT_ADULT).clip(-5, 5) * df["Rain"]
    df["BI_lag1"] = df.groupby("grid")["BI"].shift(1)
    df["BI_lag2"] = df.groupby("grid")["BI"].shift(2)
    for g in sim.GRIDS[1:]:
        df["gd_" + g] = (df["grid"] == g).astype(int)
    return df


def arms_for(df, want):
    """Return a dict arm -> feature list; noisy-proxy arms computed from the panel."""
    out = {"F0": F0, "F1": F0 + ["BI_lag1", "BI_lag2"], "F2": F0 + GD,
           "F3": F0 + ["pop_density"], "F5": F0 + ["BI_lag1", "BI_lag2"] + GD}
    if any(a.startswith("F4") for a in want):
        rng = np.random.default_rng(7)
        base = df["pop_density"].values
        df["proxy_n10"] = np.round(base * np.exp(rng.normal(0, 0.10, len(base))), 3)
        df["proxy_n30"] = np.round(base * np.exp(rng.normal(0, 0.30, len(base))), 3)
        out["F4a"] = F0 + ["proxy_n10"]
        out["F4b"] = F0 + ["proxy_n30"]
    return {k: v for k, v in out.items() if k in want}


def metrics(y, p, thr=10.0):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    e = y - p
    ss_res = float(np.sum(e ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    yb = (y >= thr).astype(int)
    pb = (p >= thr).astype(int)
    tp = int(((yb == 1) & (pb == 1)).sum()); tn = int(((yb == 0) & (pb == 0)).sum())
    fp = int(((yb == 0) & (pb == 1)).sum()); fn = int(((yb == 1) & (pb == 0)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    try:
        auc = roc_auc_score(yb, p)
    except ValueError:
        auc = np.nan
    return dict(n=len(y), RMSE=float(np.sqrt(np.mean(e ** 2))),
                MAE=float(np.mean(np.abs(e))), R2=1 - ss_res / ss_tot,
                AUC=auc, Sens=sens, Spec=spec,
                F1=2 * prec * sens / (prec + sens) if prec + sens > 0 else 0.0,
                BalAcc=(sens + spec) / 2)


def fit_ml(name, Xtr, ytr, Xte, seed):
    if name == "Lasso":
        from sklearn.linear_model import Lasso
        m = Lasso(alpha=0.01)
    elif name == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=400, max_depth=10, min_samples_split=5,
                                  n_jobs=2, random_state=seed)
    elif name == "XGBoost":
        from xgboost import XGBRegressor
        m = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=seed, n_jobs=2, verbosity=0)
    else:
        raise ValueError(name)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def run_sarima(panel):
    from statsmodels.tsa.arima.model import ARIMA
    rows = []
    for g, grp in panel.groupby("grid"):
        grp = grp.sort_values("week_idx").reset_index(drop=True)
        bi = grp["BI"].values; years = grp["year"].values
        for ty in (2019, 2020, 2021, 2022):
            idx = np.where(years == ty)[0]
            if len(idx) == 0:
                continue
            train = bi[:idx[0]]
            if len(train) < 150:
                continue
            try:
                fit = ARIMA(train, order=(1, 1, 1), seasonal_order=(0, 1, 1, 52)).fit(
                    method_kwargs={"maxiter": 100})
                cur = fit
                for ix in idx:
                    rows.append({"grid": g, "year": ty,
                                 "week_in_year": int(grp.loc[ix, "week_in_year"]),
                                 "y_true": bi[ix], "y_pred": float(max(cur.forecast(1)[0], 0))})
                    cur = cur.append([bi[ix]], refit=False)
            except Exception as exc:                      # noqa: BLE001
                print(f"    SARIMA fail {g}/{ty}: {type(exc).__name__}", flush=True)
    return pd.DataFrame(rows)


def run_panel(scenario, seed, arm_names, models, kappa):
    tag = f"{scenario}{int(kappa*100):02d}_{seed}"
    pred_f = os.path.join(OUT, f"preds_{tag}.csv")
    met_f = os.path.join(OUT, f"metrics_{tag}.csv")
    if os.path.exists(met_f) and os.path.exists(pred_f):
        return

    t0 = time.time()
    panel = sim.build_panel(seed=seed, scenario=scenario, kappa=kappa)
    df = make_features(panel)
    df["y"] = df.groupby("grid")["BI"].shift(-1)
    arms = arms_for(df, arm_names)

    preds, mets = [], []

    for arm, cols in arms.items():
        sub = df.dropna(subset=cols + ["y"])
        for name in [m for m in models if m != "SARIMA"]:
            yp, yt, meta = [], [], []
            for ty in (2019, 2020, 2021, 2022):
                tr = sub[sub["year"] < ty]
                te = sub[sub["year"] == ty]
                if len(tr) < 100 or len(te) == 0:
                    continue
                p = fit_ml(name, tr[cols].values, tr["y"].values, te[cols].values, seed)
                yp.extend(p); yt.extend(te["y"].values)
                meta.append(te[["grid", "year", "week_in_year", "y"]].assign(y_pred=p))
            if not yt:
                continue
            m = metrics(yt, yp)
            m.update(scenario=scenario, seed=seed, kappa=kappa, arm=arm, model=name)
            mets.append(m)
            preds.append(pd.concat(meta).assign(scenario=scenario, seed=seed,
                                                kappa=kappa, arm=arm, model=name))

    if "SARIMA" in models:
        sp = run_sarima(panel)
        if len(sp):
            m = metrics(sp["y_true"], sp["y_pred"])
            m.update(scenario=scenario, seed=seed, kappa=kappa, arm="F0", model="SARIMA")
            mets.append(m)
            preds.append(sp.rename(columns={"y_true": "y"}).assign(
                scenario=scenario, seed=seed, kappa=kappa, arm="F0", model="SARIMA"))

    # oracle ceilings: perfect information and week-t-only (persistence)
    for mode, label in (("full", "oracle_full"), ("persist", "oracle_persist")):
        yy, pp = [], []
        for g, grp in panel.groupby("grid"):
            grp = grp.sort_values("week_idx").reset_index(drop=True)
            T = grp["T"].values; R = grp["Rain"].values; H = grp["Hum"].values
            if mode == "persist":
                T = np.concatenate([[T[0]], T[:-1]])
                R = np.concatenate([[R[0]], R[:-1]])
                H = np.concatenate([[H[0]], H[:-1]])
            pred = sim.pfi(T, R, H, scenario) * grp["pop_density"].values
            keep = grp["year"].values >= 2019
            yy.extend(grp["BI"].values[keep]); pp.extend(pred[keep])
        m = metrics(yy, pp)
        m.update(scenario=scenario, seed=seed, kappa=kappa, arm="oracle", model=label)
        mets.append(m)

    pd.concat(preds).to_csv(pred_f, index=False)
    pd.DataFrame(mets).to_csv(met_f, index=False)
    print(f"  {tag}: {len(mets)} rows, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="L")
    ap.add_argument("--seeds", default="2024")
    ap.add_argument("--arms", default="F0,F1,F2,F3")
    ap.add_argument("--models", default="Lasso,RandomForest,XGBoost,SARIMA")
    ap.add_argument("--kappa", type=float, default=0.0)
    a = ap.parse_args()
    for sc in a.scenarios.split(","):
        for sd in [int(x) for x in a.seeds.split(",")]:
            run_panel(sc, sd, a.arms.split(","), a.models.split(","),
                      a.kappa if sc == "S" else 0.0)
