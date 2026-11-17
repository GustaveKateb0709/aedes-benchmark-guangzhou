#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Publication figures for the Acta Tropica revision.

House rules applied throughout:
  * panel letters (a)(b)... sit BELOW each panel, above the x-axis title,
    bold, centred;
  * no legend ever overlaps a line, marker or text;
  * two-channel encoding (colour + linestyle/marker) for every multi-series
    panel so the figure survives greyscale printing;
  * vector PDF (primary), 1000-dpi TIFF (Acta Tropica's bitmapped minimum),
    and a 200-dpi PNG preview;
  * single column 90 mm, full width 190 mm.

Usage:  python figs.py --which 3,4
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim

RES = os.environ.get("AEGIS_OUT", "/tmp/at_work/v2/results")
FIG = os.environ.get("AEGIS_FIG", "/tmp/at_work/v2/figures")
os.makedirs(FIG, exist_ok=True)

MM = 1 / 25.4
W1, W2 = 90 * MM, 190 * MM          # single column, full width (inches)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C = {"SARIMA": "#1f4e79", "XGBoost": "#b2182b", "Lasso": "#1b7837",
     "RandomForest": "#7f7f7f", "GAT": "#6a3d9a", "MLP": "#e08214",
     "GAT_identity": "#b2abd2", "obs": "#252525"}

ARM_LABEL = {"F0": "Weather\n+ season", "F1": "+ lagged\nBI",
             "F2": "+ grid\nidentity", "F3": "+ urbanisation\nproxy",
             "F4a": "+ proxy\n10% noise", "F4b": "+ proxy\n30% noise",
             "F5": "+ BI + grid"}
ARM_ORDER = ["F0", "F1", "F2", "F3", "F4a", "F4b", "F5"]
ARM_MAIN = ["F0", "F1", "F2", "F3"]


# ------------------------------------------------------------------ helpers
def load_metrics():
    fs = glob.glob(os.path.join(RES, "metrics_*.csv"))
    if not fs:
        raise SystemExit("no metrics_*.csv yet")
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def panel_letter(ax, letter, y=-0.30):
    """Bold centred letter BELOW the panel, above the x-axis title."""
    ax.annotate(letter, xy=(0.5, y), xycoords="axes fraction",
                ha="center", va="top", fontsize=8, fontweight="bold")


def save(fig, name):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", **kw)
    # TIFF at 1000 dpi (line art requirement); Pillow keeps LZW
    fig.savefig(os.path.join(FIG, f"{name}.tif"), dpi=1000, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("  saved", name, flush=True)


# --------------------------------------------------------------------- Fig 3
def fig3_arm_comparison(scen="L"):
    d = load_metrics()
    d = d[(d.scenario == scen) & (~d.model.str.startswith("oracle"))]
    if d.empty:
        print("  fig3: no data"); return
    order_m = [m for m in ["SARIMA", "Lasso", "RandomForest", "XGBoost"]
               if m in d.model.unique()]
    arms = [a for a in ARM_MAIN if a in d.arm.unique()]

    g = d.groupby(["arm", "model"])["R2"].agg(["mean", "std"]).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(W2, 62 * MM),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    n_arm, n_mod = len(arms), len(order_m)
    w = 0.8 / n_mod
    hatch = {"SARIMA": "", "Lasso": "//", "RandomForest": "..", "XGBoost": ""}
    for k, m in enumerate(order_m):
        sub = g[g.model == m].set_index("arm").reindex(arms)
        x = np.arange(n_arm) + (k - (n_mod - 1) / 2) * w
        ax.bar(x, sub["mean"], w * 0.92, yerr=sub["std"], capsize=1.2,
               color=C.get(m, "#333333"), edgecolor="white", linewidth=0.4,
               hatch=hatch.get(m, ""), error_kw={"lw": 0.6}, label=m, zorder=3)
    ax.set_xticks(np.arange(n_arm))
    ax.set_xticklabels([ARM_LABEL.get(a, a) for a in arms], fontsize=6.2, linespacing=1.1)
    ax.set_ylabel("R² (rolling origin, 2019–2022)")
    ax.set_ylim(0.60, 0.86)
    ax.grid(axis="y", lw=0.4, alpha=0.35, zorder=0)
    ax.legend(ncol=2, frameon=False, loc="upper left", handlelength=1.4,
              columnspacing=0.9, handletextpad=0.5)
    panel_letter(ax, "(a)", y=-0.30)

    # paired difference of every model against SARIMA, computed panel by panel
    ax = axes[1]
    fs = [f for f in glob.glob(os.path.join(RES, f"preds_{scen}*.csv"))
          if "_gat" not in f]
    P = pd.concat([pd.read_csv(f, usecols=["seed", "arm", "model", "grid", "year",
                                          "week_in_year", "y", "y_pred"])
                   for f in fs], ignore_index=True)
    sat = P[P.model == "SARIMA"][["seed", "grid", "year", "week_in_year", "y", "y_pred"]]
    rows = []
    for arm in ["F0", "F2"]:
        for m in order_m:
            if m == "SARIMA" or arm not in P.arm.unique():
                continue
            a = P[(P.model == m) & (P.arm == arm)]
            if a.empty:
                continue
            j = a.merge(sat, on=["seed", "grid", "year", "week_in_year"],
                        suffixes=("_m", "_s"))
            per_seed = []
            for sd, blk in j.groupby("seed"):
                if len(blk) < 100:
                    continue
                r2a = 1 - ((blk.y_m - blk.y_pred_m) ** 2).sum() / ((blk.y_m - blk.y_m.mean()) ** 2).sum()
                r2b = 1 - ((blk.y_s - blk.y_pred_s) ** 2).sum() / ((blk.y_s - blk.y_s.mean()) ** 2).sum()
                per_seed.append(r2a - r2b)
            if per_seed:
                rows.append({"model": m, "arm": arm, "d": float(np.mean(per_seed)),
                             "sd": float(np.std(per_seed, ddof=1))})
    dd = pd.DataFrame(rows)
    ypos, labels = [], []
    for i, (m, arm) in enumerate([(m, a) for a in ["F0", "F2"] for m in order_m if m != "SARIMA"]):
        sub = dd[(dd.model == m) & (dd.arm == arm)]
        if sub.empty:
            continue
        ax.barh(i, sub.d.iloc[0], 0.7, color=C.get(m, "#333"), alpha=0.55 if arm == "F0" else 1.0,
                edgecolor="white", linewidth=0.4, zorder=3)
        ax.errorbar(sub.d.iloc[0], i, xerr=sub.sd.iloc[0], fmt="none",
                    ecolor="#333", lw=0.6, capsize=1.2, zorder=4)
        ypos.append(i); labels.append(f"{m} · {'weather' if arm=='F0' else '+grid'}")
    ax.axvline(0, color="black", lw=0.7, zorder=2)
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=6.4)
    ax.invert_yaxis()
    ax.set_xlabel("ΔR² vs SARIMA (same panel)")
    ax.grid(axis="x", lw=0.4, alpha=0.35, zorder=0)
    panel_letter(ax, "(b)", y=-0.30)
    fig.tight_layout(w_pad=1.5)
    save(fig, "Fig3_information_arms")


# --------------------------------------------------------------------- Fig 4
def fig4_pred_vs_obs(seed=2024, grid="tianhe", year=2022, scen="L"):
    fs = glob.glob(os.path.join(RES, f"preds_{scen}00_{seed}.csv"))
    if not fs:
        print("  fig4: no preds"); return
    p = pd.read_csv(fs[0])
    p = p[(p.grid == grid) & (p.year == year)]
    if p.empty:
        print("  fig4: empty"); return
    fig, axes = plt.subplots(2, 1, figsize=(W2, 82 * MM), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1]})
    ax = axes[0]
    base = p[(p.model == "SARIMA") & (p.arm == "F0")].sort_values("week_in_year")
    ax.plot(base.week_in_year, base.y, "-", color=C["obs"], lw=1.5,
            label="Simulated BI (label)", zorder=6)
    ax.plot(base.week_in_year, base.y_pred, "--", color=C["SARIMA"], lw=1.1,
            label="SARIMA (outcome history)", zorder=5)
    for arm, ls, lab in (("F0", ":", "XGBoost, weather only"),
                         ("F2", "-.", "XGBoost, + grid identity")):
        s_ = p[(p.model == "XGBoost") & (p.arm == arm)].sort_values("week_in_year")
        if len(s_):
            ax.plot(s_.week_in_year, s_.y_pred, ls, color=C["XGBoost"], lw=1.1,
                    label=lab, zorder=4)
    ax.axhline(10, color="#888", lw=0.6, ls=(0, (1, 2)), zorder=1)
    ax.set_ylabel("Breteau index", labelpad=2)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncol=2, loc="upper left", handlelength=2.2,
              columnspacing=1.0, handletextpad=0.6)
    ax.grid(lw=0.4, alpha=0.3)
    panel_letter(ax, "(a)", y=-0.10)

    ax = axes[1]
    for arm, mk, col, lab in (("F0", "o", "#999999", "weather only"),
                              ("F2", "^", C["XGBoost"], "+ grid identity")):
        s_ = p[(p.model == "XGBoost") & (p.arm == arm)].sort_values("week_in_year")
        if len(s_):
            ax.axhline(0, color="black", lw=0.6, zorder=1)
            ax.plot(s_.week_in_year, s_.y - s_.y_pred, mk, ms=3.0, mfc="none",
                    mec=col, mew=0.8, ls="none", label=lab, zorder=3)
    ax.set_xlabel("Week of 2022")
    ax.set_ylabel("Observed − predicted", labelpad=2)
    ax.legend(frameon=False, ncol=2, loc="lower left", handletextpad=0.4)
    ax.grid(lw=0.4, alpha=0.3)
    panel_letter(ax, "(b)", y=-0.46)
    fig.tight_layout(h_pad=0.6)
    save(fig, "Fig4_pred_vs_obs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="3,4")
    a = ap.parse_args()
    for w in a.which.split(","):
        {"3": fig3_arm_comparison, "4": fig4_pred_vs_obs}[w]()


# --------------------------------------------------------------------- Fig 1
def fig1_system():
    import json
    from matplotlib.patches import FancyArrowPatch
    coords = {}
    for g in sim.GRIDS:
        with open(os.path.join(sim.DATA, f"era5_{g}.json")) as f:
            j = json.load(f)
        coords[g] = (j["longitude"], j["latitude"])
    A = sim.adjacency(3)
    fig = plt.figure(figsize=(W2, 78 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25], height_ratios=[1, 1],
                          wspace=0.30, hspace=0.55)
    ax = fig.add_subplot(gs[:, 0])
    lon = [coords[g][0] for g in sim.GRIDS]
    lat = [coords[g][1] for g in sim.GRIDS]
    for i in range(len(sim.GRIDS)):
        for j2 in range(i + 1, len(sim.GRIDS)):
            if A[i, j2] > 0:
                ax.plot([lon[i], lon[j2]], [lat[i], lat[j2]], "-", color="#9a9a9a",
                        lw=0.7, zorder=1)
    ax.scatter(lon, lat, s=42, c="#1f4e79", edgecolor="white", linewidth=0.8, zorder=3)
    off = {"tianhe": (-2, 11), "yuexiu": (-6, -2), "haizhu": (6, -10), "huangpu": (9, -3),
           "panyu": (7, -9), "nansha": (5, -9), "baiyun": (-7, 4), "huadu": (-6, 6),
           "conghua": (5, 3), "zengcheng": (6, 2)}
    for g, x, y in zip(sim.GRIDS, lon, lat):
        dx, dy = off[g]
        ax.annotate(g.capitalize(), (x, y), textcoords="offset points",
                    xytext=(dx, dy), fontsize=5.6,
                    ha="right" if dx < 0 else "left", va="bottom" if dy > 0 else "top")
    ax.set_xlabel("Longitude (°E)", labelpad=1)
    ax.set_ylabel("Latitude (°N)", labelpad=1)
    ax.set_aspect("equal", adjustable="datalim")
    panel_letter(ax, "(a)", y=-0.16)

    ax = fig.add_subplot(gs[0, 1])
    pM = sim.build_panel(seed=1, scenario="M")
    pL = sim.build_panel(seed=1, scenario="L")
    pS = sim.build_panel(seed=1, scenario="S", kappa=0.6)
    yr = 2019
    for p, col, ls, lab in ((pM, "#999999", ":", "M: memoryless"),
                            (pL, "#1f4e79", "-", "L: 2–3 week lag"),
                            (pS, "#b2182b", "--", "S: lag + spatial")):
        s_ = p[(p.grid == "tianhe") & (p.year == yr)].sort_values("week_in_year")
        ax.plot(s_.week_in_year, s_.BI, ls, color=col, lw=1.0, label=lab)
    ax.set_xlabel("Week", labelpad=1); ax.set_ylabel("Simulated BI", labelpad=1)
    ax.legend(frameon=False, fontsize=5.6, loc="upper center", ncol=3,
              handlelength=1.6, columnspacing=0.9, handletextpad=0.4)
    ax.set_ylim(0, 36)
    ax.grid(lw=0.35, alpha=0.3)
    panel_letter(ax, "(b)", y=-0.34)

    ax = fig.add_subplot(gs[1, 1])
    d = load_metrics()
    o = d[d.model.str.startswith("oracle")].groupby(["scenario", "model"])["R2"].mean().unstack()
    labs, vals = [], []
    for scen in ["M", "L"]:
        if scen in o.index:
            for m, col in (("oracle_full", "#bbbbbb"), ("oracle_persist", "#1f4e79")):
                if m in o.columns:
                    labs.append(f"{scen}\n{'perfect' if 'full' in m else 'week-t only'}")
                    vals.append(o.loc[scen, m])
    x = np.arange(len(labs))
    cols = ["#bbbbbb", "#1f4e79", "#bbbbbb", "#1f4e79"][:len(labs)]
    ax.bar(x, vals, 0.65, color=cols, edgecolor="white", linewidth=0.4, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=5.8)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=5.8, linespacing=1.0)
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("Achievable R²", labelpad=1)
    ax.grid(axis="y", lw=0.35, alpha=0.3)
    panel_letter(ax, "(c)", y=-0.52)
    save(fig, "Fig1_system")


# --------------------------------------------------------------------- Fig 2
def fig2_simulator(seed=1, scen="L"):
    p = sim.build_panel(seed=seed, scenario=scen)
    th = p[p.grid == "tianhe"].sort_values("week_idx")
    fig, axes = plt.subplots(1, 3, figsize=(W2, 52 * MM),
                             gridspec_kw={"width_ratios": [1.5, 1, 1]})
    ax = axes[0]
    x = np.arange(len(th)) / 52 + 2015
    ax.plot(x, th["T"].values, color="#b2182b", lw=0.7, label="Temperature")
    ax.set_ylabel("Weekly mean T (°C)", color="#b2182b", labelpad=1)
    ax.tick_params(axis="y", colors="#b2182b")
    ax2 = ax.twinx()
    ax2.fill_between(x, 0, th["Rain"].values, color="#2166ac", alpha=0.35, lw=0)
    ax2.set_ylabel("Weekly rainfall (mm)", color="#2166ac", labelpad=1)
    ax2.tick_params(axis="y", colors="#2166ac")
    ax2.spines["top"].set_visible(False)
    ax.set_xlabel("Year", labelpad=1)
    ax.set_xlim(2015, 2023)
    panel_letter(ax, "(a)", y=-0.28)

    ax = axes[1]
    wk = p.groupby("week_in_year")["BI"].mean()
    ax.plot(wk.index, wk.values, "-", color="#1f4e79", lw=1.1)
    ax.axhline(10, color="#888", lw=0.6, ls=(0, (1, 2)))
    ax.axhline(5, color="#888", lw=0.6, ls=(0, (1, 2)))
    ax.set_xlabel("Week of year", labelpad=1); ax.set_ylabel("Mean simulated BI", labelpad=1)
    ax.set_xlim(1, 52); ax.grid(lw=0.35, alpha=0.3)
    panel_letter(ax, "(b)", y=-0.28)

    ax = axes[2]
    gm = p.groupby("grid")["BI"].mean().sort_values()
    ax.barh(range(len(gm)), gm.values, 0.68, color="#1f4e79", edgecolor="white",
            linewidth=0.4, zorder=3)
    ax.set_yticks(range(len(gm)))
    ax.set_yticklabels([g.capitalize() for g in gm.index], fontsize=6)
    ax.set_xlabel("Annual mean BI", labelpad=1)
    ax.grid(axis="x", lw=0.35, alpha=0.3)
    panel_letter(ax, "(c)", y=-0.28)
    fig.tight_layout(w_pad=1.2)
    save(fig, "Fig2_simulator")


# --------------------------------------------------------------------- Fig 5
def fig5_proxy_dose():
    d = load_metrics()
    d = d[(d.scenario == "L") & (d.model.isin(["Lasso", "RandomForest", "XGBoost"]))]
    arms = [a for a in ["F3", "F4a", "F4b", "F2"] if a in d.arm.unique()]
    if len(arms) < 3:
        print("  fig5: need F3/F4a/F4b, run exp with those arms"); return
    lab = {"F3": "proxy\n(no noise)", "F4a": "proxy\n10% noise",
           "F4b": "proxy\n30% noise", "F2": "grid identity\n(no proxy)"}
    fig, axes = plt.subplots(1, 2, figsize=(W2, 55 * MM),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    mk = {"Lasso": "o", "RandomForest": "s", "XGBoost": "^"}
    for m in ["Lasso", "RandomForest", "XGBoost"]:
        sub = d[d.model == m].groupby("arm")["R2"].agg(["mean", "std"]).reindex(arms)
        ax.errorbar(range(len(arms)), sub["mean"], yerr=sub["std"], marker=mk[m],
                    ms=3.2, lw=0.9, capsize=1.5, color=C.get(m, "#333"),
                    mfc="none" if m == "RandomForest" else C.get(m, "#333"),
                    label=m, zorder=3)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([lab[a] for a in arms], fontsize=6, linespacing=1.0)
    ax.set_ylabel("R²", labelpad=1)
    ax.legend(frameon=False, fontsize=6.2, loc="lower right", handlelength=1.6)
    ax.grid(lw=0.35, alpha=0.3)
    panel_letter(ax, "(a)", y=-0.36)

    ax = axes[1]
    fs = [f for f in glob.glob(os.path.join(RES, "preds_L00_*.csv")) if "_gat" not in f]
    P = pd.concat([pd.read_csv(f, usecols=["seed", "arm", "model", "grid", "year",
                                          "week_in_year", "y", "y_pred"]) for f in fs],
                  ignore_index=True)
    rows = []
    for sd, blk in P[P.model == "XGBoost"].groupby("seed"):
        a = blk[blk.arm == "F2"][["grid", "year", "week_in_year", "y", "y_pred"]]
        b = blk[blk.arm == "F3"][["grid", "year", "week_in_year", "y", "y_pred"]]
        m = a.merge(b, on=["grid", "year", "week_in_year"], suffixes=("_g", "_p"))
        if len(m) < 100:
            continue
        r2g = 1 - ((m.y_g - m.y_pred_g) ** 2).sum() / ((m.y_g - m.y_g.mean()) ** 2).sum()
        r2p = 1 - ((m.y_p - m.y_pred_p) ** 2).sum() / ((m.y_p - m.y_p.mean()) ** 2).sum()
        rows.append(r2p - r2g)
    if rows:
        ax.hist(rows, bins=12, color="#b2182b", alpha=0.75, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="black", lw=0.8)
        ax.axvline(np.mean(rows), color="#1f4e79", lw=1.0, ls="--")
        ax.set_xlabel("R²(proxy) − R²(grid identity), per panel", labelpad=1)
        ax.set_ylabel("Panels", labelpad=1)
        ax.text(0.97, 0.93, f"mean {np.mean(rows):+.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.2)
    panel_letter(ax, "(b)", y=-0.34)
    fig.tight_layout(w_pad=1.2)
    save(fig, "Fig5_proxy_dose")


# --------------------------------------------------------------------- Fig 6
def fig6_graph_boundary():
    fs = glob.glob(os.path.join(RES, "metrics_S*_gat.csv"))
    if not fs:
        print("  fig6: no GAT results"); return
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d["k"] = d.kappa.round(2)
    fig, axes = plt.subplots(1, 2, figsize=(W2, 55 * MM))
    ax = axes[0]
    style = {"GAT_knn3": ("o", "#6a3d9a", "-"), "GAT_identity": ("s", "#b2abd2", "--"),
             "MLP": ("^", "#e08214", "-.")}
    for m, (mk, col, ls) in style.items():
        sub = d[d.model == m].groupby("k")["R2"].agg(["mean", "std"]).reset_index()
        if sub.empty:
            continue
        ax.errorbar(sub.k, sub["mean"], yerr=sub["std"].fillna(0), marker=mk, ms=3.2,
                    lw=1.0, ls=ls, capsize=1.5, color=col, label=m, zorder=3)
    ax.set_xlabel("Spatial coupling κ", labelpad=1)
    ax.set_ylabel("R²", labelpad=1)
    ax.legend(frameon=False, fontsize=6.2, loc="lower left", handlelength=1.8)
    ax.grid(lw=0.35, alpha=0.3)
    panel_letter(ax, "(a)", y=-0.30)

    ax = axes[1]
    d0 = d[d.k == d.k.min()]
    gm = d0.groupby("model")["R2"].agg(["mean", "std"]).reindex(
        ["GAT_knn3", "GAT_identity", "MLP"])
    ax.barh(range(len(gm)), gm["mean"], xerr=gm["std"].fillna(0), height=0.55,
            color=["#6a3d9a", "#b2abd2", "#e08214"], edgecolor="white",
            linewidth=0.4, error_kw={"lw": 0.6}, zorder=3)
    ax.set_yticks(range(len(gm)))
    ax.set_yticklabels(["GAT kNN-3", "GAT no edges", "MLP (no graph)"], fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xlabel("R² (κ ≈ 0)", labelpad=1)
    ax.grid(axis="x", lw=0.35, alpha=0.3)
    panel_letter(ax, "(b)", y=-0.30)
    fig.tight_layout(w_pad=1.2)
    save(fig, "Fig6_graph_boundary")
