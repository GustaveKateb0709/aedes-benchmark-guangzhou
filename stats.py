#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Statistics for the benchmark.

Implements the inference the reviewers asked for:
  * Diebold-Mariano with Newey-West HAC
  * cluster-robust (CR1) with both normal and t(G-1) reference distributions
  * CR3 (leverage-corrected) for the mean loss differential
  * exact sign-flip randomisation test over all 2^G sign patterns (G = 10 grids)
  * DeLong test for paired AUC differences
  * Benjamini-Hochberg control across the family of pairwise comparisons
  * Monte Carlo summaries across simulated panels (mean, SD, 2.5-97.5 percentiles)
"""
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats as sst

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("AEGIS_OUT", os.path.join(HERE, "results"))


# ------------------------------------------------------------------ DM family
def dm_hac(d, lags):
    d = np.asarray(d, float)
    n = len(d); m = d.mean()
    var = np.mean((d - m) ** 2)
    for L in range(1, lags + 1):
        g = np.mean((d[L:] - m) * (d[:-L] - m))
        var += 2 * (1 - L / (lags + 1)) * g
    var = max(var, 1e-12)
    stat = m / np.sqrt(var / n)
    return stat, 2 * (1 - sst.norm.cdf(abs(stat)))


def dm_cluster(d, groups, small_sample="CR1"):
    """Cluster-robust variance of the mean loss differential.

    CR1: G/(G-1) * (n-1)/(n-G) scaling.
    CR3: CR1 with a leverage correction; when each cluster contributes several
         observations the correction inflates the variance further, which
         matters at G = 10.
    """
    d = np.asarray(d, float); groups = np.asarray(groups)
    n = len(d); m = d.mean(); gs = np.unique(groups); G = len(gs)
    meat = 0.0
    for g in gs:
        dg = d[groups == g]; ng = len(dg)
        # leverage of an observation in its own cluster (equal within cluster)
        h = ng / n
        corr = 1.0 / (1.0 - h) ** 2 if small_sample == "CR3" else 1.0
        meat += corr * ng ** 2 * (dg.mean() - m) ** 2
    var = meat * (G / (G - 1)) * ((n - 1) / (n - G)) / n ** 2
    var = max(var, 1e-12)
    stat = m / np.sqrt(var)
    p_norm = 2 * (1 - sst.norm.cdf(abs(stat)))
    p_t = 2 * (1 - sst.t.cdf(abs(stat), df=G - 1))
    return stat, p_norm, p_t


def signflip_exact(d):
    """Exact two-sided sign-flip test; enumerates all 2^G patterns (G <= 16)."""
    d = np.asarray(d, float)
    G = len(d)
    if G > 16:
        rng = np.random.default_rng(0)
        S = rng.choice([-1.0, 1.0], size=(200000, G))
        stats_ = np.abs((S * d).mean(axis=1))
    else:
        S = np.array(list(itertools.product([-1.0, 1.0], repeat=G)))
        stats_ = np.abs((S * d).mean(axis=1))
    obs = abs(d.mean())
    return float(np.mean(stats_ >= obs - 1e-12))


# --------------------------------------------------------------------- DeLong
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x)
    T = np.zeros(N, float); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, float); T2[J] = T
    return T2


def _auc_fast(x, y):
    """AUC of scores x for positives (y==1) vs negatives."""
    m = int((y == 1).sum()); n = int((y == 0).sum())
    if m == 0 or n == 0:
        return np.nan, np.nan
    r = _midrank(np.concatenate([x[y == 1], x[y == 0]]))
    auc = (r[:m].sum() - m * (m + 1) / 2) / (m * n)
    return auc, (r, m, n)


def delong_test(y, p1, p2):
    """Paired DeLong test for two correlated ROC AUCs."""
    y = np.asarray(y).astype(int)
    p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
    auc1, (r1, m, n) = _auc_fast(p1, y)
    auc2, (r2, _, _) = _auc_fast(p2, y)
    if np.isnan(auc1) or np.isnan(auc2):
        return auc1, auc2, np.nan, np.nan, np.nan
    v10_1 = (r1[:m] - np.arange(1, m + 1)) / n
    v01_1 = 1 - (r1[m:] - np.arange(1, n + 1)) / m
    v10_2 = (r2[:m] - np.arange(1, m + 1)) / n
    v01_2 = 1 - (r2[m:] - np.arange(1, n + 1)) / m
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    S = s10 / m + s01 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return auc1, auc2, np.nan, np.nan, np.nan
    z = (auc1 - auc2) / np.sqrt(var)
    return auc1, auc2, float(z), float(2 * (1 - sst.norm.cdf(abs(z)))), float(np.sqrt(var))


# ------------------------------------------------------------------ assembly
def load_metrics():
    fs = [f for f in os.listdir(OUT) if f.startswith("metrics_") and f.endswith(".csv")]
    if not fs:
        raise SystemExit(f"no metrics_*.csv in {OUT}")
    d = pd.concat([pd.read_csv(os.path.join(OUT, f)) for f in fs], ignore_index=True)
    return d


def mc_summary(d):
    keep = d[~d.model.str.startswith("oracle")]
    g = keep.groupby(["scenario", "arm", "model"])
    out = g.agg(RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
                R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
                AUC_mean=("AUC", "mean"), AUC_sd=("AUC", "std"),
                RMSE_lo=("RMSE", lambda s: np.percentile(s, 2.5)),
                RMSE_hi=("RMSE", lambda s: np.percentile(s, 97.5)),
                R2_lo=("R2", lambda s: np.percentile(s, 2.5)),
                R2_hi=("R2", lambda s: np.percentile(s, 97.5)),
                n_panels=("RMSE", "size")).reset_index()
    return out


def paired_tests(preds, scenario, arm, ref, cmp_, by_seed=True):
    """Paired tests on matched grid-week errors, averaged over panels."""
    rows = []
    sub = preds[(preds.scenario == scenario) & (preds.arm == arm)]
    for seed, blk in sub.groupby("seed"):
        a = blk[blk.model == ref][["grid", "year", "week_in_year", "y", "y_pred"]]
        b = blk[blk.model == cmp_][["grid", "year", "week_in_year", "y", "y_pred"]]
        m = a.merge(b, on=["grid", "year", "week_in_year"], suffixes=("_a", "_b"))
        if len(m) == 0:
            continue
        d = (m.y_a - m.y_pred_a) ** 2 - (m.y_b - m.y_pred_b) ** 2
        dm4, p4 = dm_hac(d.values, 4)
        s1, pn, pt = dm_cluster(d.values, m.grid.values, "CR1")
        s3, pn3, pt3 = dm_cluster(d.values, m.grid.values, "CR3")
        per_grid = pd.DataFrame({"d": d.values, "g": m.grid.values}).groupby("g")["d"].mean()
        t_g, p_g = sst.ttest_1samp(per_grid.values, 0.0)
        rows.append(dict(scenario=scenario, arm=arm, ref=ref, cmp=cmp_, seed=seed,
                         n=len(d), DM_hac4=dm4, p_hac4=p4,
                         DM_CR1=s1, p_CR1_norm=pn, p_CR1_t=pt,
                         DM_CR3=s3, p_CR3_norm=pn3, p_CR3_t=pt3,
                         t_grid=t_g, p_grid=p_g,
                         p_signflip=signflip_exact(per_grid.values)))
    return pd.DataFrame(rows)


def bh(pvals, alpha=0.05):
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    tmp = np.empty(n)
    tmp[order] = np.clip(adj, 0, 1)
    out[ok] = tmp
    return out


if __name__ == "__main__":
    d = load_metrics()
    print(f"loaded {len(d)} metric rows; scenarios={sorted(d.scenario.unique())}")
    s = mc_summary(d)
    s.to_csv(os.path.join(OUT, "summary_mc.csv"), index=False)
    print(s.round(4).to_string(index=False))
