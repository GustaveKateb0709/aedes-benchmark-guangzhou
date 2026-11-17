#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DML audit, v2 -- simulator with a genuine multi-week lag structure
=================================================================

What changed relative to gn/run_dml.py
--------------------------------------
The v1 script hard-coded

    true_val = float(np.mean(truth0)) if lag == 0 else 0.0

because its simulator had no memory: BI(t) depended on week-t weather only,
so every lagged effect was zero by construction.  Scenario L of sim.py puts
immature development on a three-week kernel (0.50, 0.30, 0.20) and carries
breeding-site water over weeks with rho = 0.60, so

    BI(t) <- T(t), T(t-1), T(t-2)          (development kernel)
    BI(t) <- Rain(t), Rain(t-1), ...       (habitat recursion, geometric)

and the lagged ground truth is no longer zero.  This script therefore
replaces the hard-coded zero with a finite-difference truth computed through
the same simulator that generated the data.

Ground truth (finite difference, whole-series recomputation)
------------------------------------------------------------
For grid g, perturb weather variable v at source week s by delta and
recompute the *entire* PFI series with sim.pfi().  Because the kernel and the
recursion carry information forward, a perturbation at s moves PFI at
s, s+1, s+2, ... -- and that is exactly the true lagged effect we want:

    truth[lag k](g,t) = [ (PFI_new(t) - PFI_old(t)) * pop_density(g) ] / delta
                        with the perturbation placed at s = t - k.

delta_T = 0.1 degC, delta_Rain = 1 mm.  Averaged over all observations in the
analysis window for the scalar "truth" reported per (variable, lag).

Two truths are reported: the unweighted mean over observations (`truth`), and
-- for lag 0 only -- the variance-weighted mean (`truth_weighted`) with weights
equal to the squared cross-fitted treatment residual, i.e. the estimand that a
linear partially-linear DML actually targets.

Estimator (unchanged specification)
-----------------------------------
Partially linear DML, XGBoost nuisance learners, 5-fold cross-fitting, five
independent random splits, median of the five ATEs, cluster-robust (CR1 by
grid) standard errors.  Naive unadjusted OLS as the confounded contrast.
Dose-response by FWL-spline DML, curves anchored at the median dose (as in v1).

One deliberate deviation from v1's confounder_for()
---------------------------------------------------
v1 built the confounder matrix as

    [sin_week, cos_week, pop_density, Hum, T, Rain, <other>_lag<k>]

and then removed only '<tvar>_lag<k>'.  At lag 0 the treatment column *is*
'T' (or 'Rain'), so the raw current-week value of the treatment stayed inside
X, making the treatment collinear with its own residualiser.  That is not a
defensible specification -- t_res collapses to noise and theta is arbitrary.
Here the treatment column itself is dropped from X for every lag, which is the
only change to the estimator.  Everything else is copied verbatim.

Outputs (results/)
------------------
dml_lag_effects_v2.csv   treatment, lag_weeks, dml_ate, se_cluster, ci_lo,
                         ci_hi, naive_ols, truth, truth_weighted,
                         ci_covers_truth, seed
dml_dose_response_v2.csv treatment, value, dml_response, dml_se,
                         naive_response, truth_response, seed
dml_summary_v2.csv       per (treatment, lag) across panels: mean/SD of the
                         DML ATE, mean truth, coverage, detection flag

Usage
-----
python run_dml_v2.py --seeds 1,2,...,10
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim  # noqa: E402

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

SCENARIO = "L"
LAGS = list(range(0, 7))
K_FOLDS = 5
N_SPLIT_REPS = 5
DELTA = {"T": 0.1, "Rain": 1.0}
SEED0 = 2024


# ---------------------------------------------------------------- truth -----
def truth_tables(panel):
    """Per-observation finite-difference truth, keyed by (variable, lag).

    Returns dict[(var, lag)] -> DataFrame[grid, week_idx, truth].
    """
    out = {}
    for var, delta in DELTA.items():
        frames = []
        for g, grp in panel.groupby("grid", sort=True):
            grp = grp.sort_values("week_idx")
            T = grp["T"].to_numpy(float)
            R = grp["Rain"].to_numpy(float)
            H = grp["Hum"].to_numpy(float)
            widx = grp["week_idx"].to_numpy(int)
            pop = float(grp["pop_density"].iloc[0])
            n = len(grp)
            base = sim.pfi(T, R, H, SCENARIO) * pop

            eff = np.full((len(LAGS), n), np.nan)   # eff[k, t] = d BI(t)/d var(t-k)
            for s in range(n):
                T2, R2 = T, R
                if var == "T":
                    T2 = T.copy()
                    T2[s] += delta
                else:
                    R2 = R.copy()
                    R2[s] += delta
                d = (sim.pfi(T2, R2, H, SCENARIO) * pop - base) / delta
                for k in LAGS:
                    t = s + k
                    if t < n:
                        eff[k, t] = d[t]

            for k in LAGS:
                frames.append(pd.DataFrame({"grid": g, "week_idx": widx,
                                            "lag_weeks": k, "truth": eff[k]}))
        out[var] = pd.concat(frames, ignore_index=True)
    return out


# ------------------------------------------------------------ DML core ------
def _xgb(seed=SEED0):
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                        subsample=0.8, colsample_bytree=0.8,
                        tree_method="hist", random_state=seed, n_jobs=2,
                        verbosity=0)


def crossfit_residuals(Y, T, X, seed=SEED0, k=K_FOLDS):
    from sklearn.model_selection import KFold
    n = len(Y)
    g_hat = np.zeros(n)
    m_hat = np.zeros(n)
    for tr, te in KFold(n_splits=k, shuffle=True, random_state=seed).split(X):
        g_hat[te] = _xgb(seed).fit(X[tr], Y[tr]).predict(X[te])
        m_hat[te] = _xgb(seed).fit(X[tr], T[tr]).predict(X[te])
    return Y - g_hat, T - m_hat


def _theta_se(y_res, t_res, groups):
    n = len(y_res)
    denom = float(np.sum(t_res ** 2))
    if denom <= 1e-10:
        return np.nan, np.nan
    theta = float(np.sum(t_res * y_res) / denom)
    psi = (t_res * y_res - theta * t_res ** 2) / max(denom / n, 1e-12)
    uniq = np.unique(groups)
    G = len(uniq)
    meat = sum(psi[groups == g].sum() ** 2 for g in uniq)
    var_cl = meat / n ** 2 * (G / max(G - 1, 1)) * ((n - 1) / max(n - 8, 1))
    return theta, float(np.sqrt(max(var_cl, 0.0)))


def dml_ate(Y, T, X, groups, reps=N_SPLIT_REPS):
    thetas, ses = [], []
    for r in range(reps):
        y_res, t_res = crossfit_residuals(Y, T, X, seed=SEED0 + 101 * r)
        th, sc = _theta_se(y_res, t_res, groups)
        thetas.append(th)
        ses.append(sc)
    thetas = np.asarray(thetas, float)
    ses = np.asarray(ses, float)
    if not np.isfinite(thetas).any():
        return np.nan, np.nan
    i = int(np.argsort(np.nan_to_num(thetas, nan=np.inf))[len(thetas) // 2])
    return float(thetas[i]), float(ses[i])


def naive_ols(Y, T):
    from sklearn.linear_model import LinearRegression
    return float(LinearRegression().fit(T.reshape(-1, 1), Y).coef_[0])


def spline_basis(t, knots):
    cols = [np.asarray(t, float)]
    for k_ in knots:
        cols.append(np.maximum(np.asarray(t, float) - k_, 0.0))
    return np.column_stack(cols)


def dml_dose_curve(Y, T, X, groups, knots, grid_pts):
    from sklearn.model_selection import KFold
    n = len(Y)
    B = spline_basis(T, knots)
    p = B.shape[1]
    y_res = np.zeros(n)
    B_res = np.zeros_like(B)
    for tr, te in KFold(n_splits=K_FOLDS, shuffle=True,
                        random_state=SEED0).split(X):
        y_res[te] = Y[te] - _xgb().fit(X[tr], Y[tr]).predict(X[te])
        for j in range(p):
            B_res[te, j] = B[te, j] - _xgb().fit(X[tr], B[tr, j]).predict(X[te])
    XtX_inv = np.linalg.pinv(B_res.T @ B_res)
    theta = XtX_inv @ B_res.T @ y_res
    eps = y_res - B_res @ theta
    meat = np.zeros((p, p))
    for g in np.unique(groups):
        m = groups == g
        s = B_res[m].T @ eps[m]
        meat += np.outer(s, s)
    G = len(np.unique(groups))
    cov = XtX_inv @ meat @ XtX_inv * (G / max(G - 1, 1)) * ((n - 1) / max(n - p, 1))
    Bg = spline_basis(grid_pts, knots)
    curve = Bg @ theta
    se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Bg, cov, Bg), 0))
    return theta, curve, se


# ------------------------------------------------------------- analysis -----
def build_analysis_frame(panel):
    df = panel.sort_values(["grid", "week_idx"]).reset_index(drop=True).copy()
    df["sin_week"] = np.sin(2 * np.pi * df["week_in_year"] / 52.0)
    df["cos_week"] = np.cos(2 * np.pi * df["week_in_year"] / 52.0)
    for var in ("Rain", "T"):
        for lag in LAGS:
            df[f"{var}_lag{lag}"] = df.groupby("grid")[var].shift(lag)
    need = [f"{v}_lag{l}" for v in ("Rain", "T") for l in LAGS]
    return df.dropna(subset=need + ["BI"]).reset_index(drop=True)


def confounders_for(df, tvar, lag):
    cols = ["sin_week", "cos_week", "pop_density", "Hum", "T", "Rain"]
    other = "T" if tvar == "Rain" else "Rain"
    cols.append(f"{other}_lag{lag}")
    cols = [c for c in dict.fromkeys(cols) if c in df.columns]
    drop = {f"{tvar}_lag{lag}"}
    if lag == 0:
        drop.add(tvar)          # see module docstring: treatment must not be in X
    return [c for c in cols if c not in drop]


def truth_response_curve(panel, df, tvar, grid_pts):
    """E[BI_mean | do(var = gp)] under the known mechanism, on the rows used."""
    keep = {g: set(s) for g, s in df.groupby("grid")["week_idx"]}
    cache = {}
    for g, grp in panel.groupby("grid", sort=True):
        grp = grp.sort_values("week_idx")
        m = grp["week_idx"].isin(keep.get(g, set())).to_numpy()
        cache[g] = (grp["T"].to_numpy(float), grp["Rain"].to_numpy(float),
                    grp["Hum"].to_numpy(float),
                    float(grp["pop_density"].iloc[0]), m)
    out = []
    for gp in grid_pts:
        acc = []
        for g, (T, R, H, pop, m) in cache.items():
            if not m.any():
                continue
            if tvar == "T":
                T = np.full_like(T, gp)
            else:
                R = np.full_like(R, gp)
            acc.append(sim.pfi(T, R, H, SCENARIO)[m] * pop)
        out.append(float(np.concatenate(acc).mean()))
    return np.array(out)


def run_panel(seed, reps=N_SPLIT_REPS, lags=LAGS, verbose=True):
    panel = sim.build_panel(seed=seed, scenario=SCENARIO)
    df = build_analysis_frame(panel)
    groups = df["grid"].to_numpy()
    Y = df["BI"].to_numpy(float)

    tr = truth_tables(panel)

    lag_rows = []
    for tvar in ("Rain", "T"):
        for lag in lags:
            tcol = f"{tvar}_lag{lag}"
            T = df[tcol].to_numpy(float)
            X = df[confounders_for(df, tvar, lag)].to_numpy(float)
            theta, se_cl = dml_ate(Y, T, X, groups, reps=reps)
            naive = naive_ols(Y, T)

            tt = tr[tvar]
            tt = tt[tt["lag_weeks"] == lag][["grid", "week_idx", "truth"]]
            merged = df[["grid", "week_idx"]].merge(
                tt, on=["grid", "week_idx"], how="left", validate="1:1")
            tvec = merged["truth"].to_numpy(float)
            true_val = float(np.nanmean(tvec))

            wtruth = np.nan
            if lag == 0:
                y_res, t_res = crossfit_residuals(Y, T, X, seed=SEED0)
                w = t_res ** 2
                if np.sum(w) > 0:
                    wtruth = float(np.sum(w * tvec) / np.sum(w))

            lo = theta - 1.96 * se_cl
            hi = theta + 1.96 * se_cl
            lag_rows.append({
                "treatment": tvar, "lag_weeks": lag,
                "dml_ate": round(theta, 4),
                "se_cluster": round(se_cl, 4),
                "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                "naive_ols": round(naive, 4),
                "truth": round(true_val, 4),
                "truth_weighted": (round(wtruth, 4) if np.isfinite(wtruth) else np.nan),
                "ci_covers_truth": int(lo <= true_val <= hi),
                "seed": seed,
            })
            if verbose:
                print(f"  {tvar:4s} lag{lag}: DML={theta:8.4f} "
                      f"(CI {lo:8.4f},{hi:8.4f}) truth={true_val:8.4f} "
                      f"naive={naive:8.4f}", flush=True)

    dose_rows = []
    for tvar in ("Rain", "T"):
        T = df[f"{tvar}_lag0"].to_numpy(float)
        qs = np.quantile(T, [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99])
        knots = list(np.round(qs[1:-1], 1))
        grid_pts = np.linspace(qs[0], qs[-1], 41)
        X = df[confounders_for(df, tvar, 0)].to_numpy(float)
        _, curve, se = dml_dose_curve(Y, T, X, groups, knots, grid_pts)

        B = spline_basis(T, knots)
        th_n = np.linalg.lstsq(np.column_stack([np.ones(len(T)), B]), Y,
                              rcond=None)[0]
        naive_curve = np.column_stack([np.ones(len(grid_pts)),
                                       spline_basis(grid_pts, knots)]) @ th_n

        resp_truth = truth_response_curve(panel, df, tvar, grid_pts)
        ref = int(np.argmin(np.abs(grid_pts - np.median(T))))
        curve_al = curve - curve[ref]
        naive_al = naive_curve - naive_curve[ref]
        truth_al = resp_truth - resp_truth[ref]
        for i, gp in enumerate(grid_pts):
            dose_rows.append({
                "treatment": tvar, "value": round(float(gp), 2),
                "dml_response": round(float(curve_al[i]), 4),
                "dml_se": round(float(se[i]), 4),
                "naive_response": round(float(naive_al[i]), 4),
                "truth_response": round(float(truth_al[i]), 4),
                "seed": seed,
            })
    return pd.DataFrame(lag_rows), pd.DataFrame(dose_rows)


def summarize(lag):
    rows = []
    for (tvar, lag_), grp in lag.groupby(["treatment", "lag_weeks"]):
        est = grp["dml_ate"].to_numpy(float)
        est = est[np.isfinite(est)]
        sd = float(np.std(est, ddof=1)) if len(est) > 1 else np.nan
        mean_est = float(np.mean(est)) if len(est) else np.nan
        mean_truth = float(np.nanmean(grp["truth"].to_numpy(float)))
        rows.append({
            "treatment": tvar, "lag_weeks": int(lag_),
            "n_panels": int(len(est)),
            "dml_ate_mean": round(mean_est, 4),
            "dml_ate_sd": round(sd, 4),
            "naive_ols_mean": round(float(np.nanmean(grp["naive_ols"])), 4),
            "truth_mean": round(mean_truth, 4),
            "coverage": round(float(np.nanmean(grp["ci_covers_truth"])), 3),
            "abs_bias": round(abs(mean_est - mean_truth), 4),
            "detected_truth": int(abs(mean_est - mean_truth) < 1.96 * sd)
                              if np.isfinite(sd) else 0,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--reps", type=int, default=N_SPLIT_REPS)
    ap.add_argument("--lags", default="0,1,2,3,4,5,6")
    ap.add_argument("--tag", default="v2")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    lags = [int(s) for s in a.lags.split(",")]
    t0 = time.time()

    lag_all, dose_all = [], []
    for sd in seeds:
        ts = time.time()
        lg, dz = run_panel(sd, reps=a.reps, lags=lags)
        lag_all.append(lg)
        dose_all.append(dz)
        pd.concat(lag_all, ignore_index=True).to_csv(
            os.path.join(OUT, f"dml_lag_effects_{a.tag}.csv"), index=False)
        pd.concat(dose_all, ignore_index=True).to_csv(
            os.path.join(OUT, f"dml_dose_response_{a.tag}.csv"), index=False)
        print(f"  seed {sd} done in {time.time() - ts:.0f}s "
              f"(total {time.time() - t0:.0f}s)", flush=True)

    lag = pd.concat(lag_all, ignore_index=True)
    dose = pd.concat(dose_all, ignore_index=True)
    lag.to_csv(os.path.join(OUT, f"dml_lag_effects_{a.tag}.csv"), index=False)
    dose.to_csv(os.path.join(OUT, f"dml_dose_response_{a.tag}.csv"), index=False)
    summ = summarize(lag)
    summ.to_csv(os.path.join(OUT, f"dml_summary_{a.tag}.csv"), index=False)

    print(f"\n95% CI coverage: {lag['ci_covers_truth'].mean():.3f}")
    print(summ.to_string(index=False))
    print(f"\ndone in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
