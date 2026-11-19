#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Supplementary Figure S1: lag structure of the simulated mechanism.

Scenario L of sim.py puts immature development on a three-week kernel and
carries breeding-site water over weeks with rho = 0.60, so BI(t) responds to
T(t), T(t-1), T(t-2) and to Rain(t), Rain(t-1), ... .  run_dml_v2.py scores its
double-machine-learning estimates against that ground truth; this figure shows
the ground truth itself, which is what makes the audit's specificity and
sensitivity checks interpretable.

Data source: results/dml_lag_effects_v2.csv, column `truth`
(written by run_dml_v2.py; one row per treatment x lag x panel).
Points are means over the ten panels.

Outputs (into $AEGIS_FIG, default ./figures):
    FigS1 Lag Structure.tif   1000 dpi, LZW, RGB  (submission file)
    FigS1 Lag Structure.pdf   vector              (submission file)
    FigS1 Lag Structure.png   200 dpi              (preview)
    FigS1 Lag Structure_embed.png  2400 px        (inline copy for the .docx)

Usage:  python figS1_lag_structure.py
Env:    AEGIS_OUT / AEGIS_FIG override the results and figure directories.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figs  # noqa: E402  (rcParams, save(), panel_letter(), raise_frame())

RES, FIG = figs.RES, figs.FIG
MM = figs.MM

T_COL = "#b2182b"      # temperature, same red as Fig. 2a
R_COL = "#2166ac"      # rainfall, same blue as Fig. 2a
EMBED_PX = 2400


def load_truth():
    p = os.path.join(RES, "dml_lag_effects_v2.csv")
    if not os.path.isfile(p):
        raise SystemExit(f"FAIL: {p} not found -- run run_dml_v2.py first.")
    d = pd.read_csv(p)
    need = {"treatment", "lag_weeks", "truth"}
    if not need <= set(d.columns):
        raise SystemExit(f"FAIL: {p} lacks columns {sorted(need - set(d.columns))}")
    m = d.groupby(["lag_weeks", "treatment"])["truth"].mean().unstack("treatment")
    return m.sort_index()


def figS1_lag_structure():
    m = load_truth()
    lags = m.index.to_numpy(int)
    T = m["T"].to_numpy(float)
    R = m["Rain"].to_numpy(float)

    fig, axes = plt.subplots(1, 2, figsize=(150 * MM, 58 * MM))

    # ------------------------------------------------ (a) temperature, linear
    ax = axes[0]
    ax.axhline(0, color="#8a8a8a", lw=0.6, ls=(0, (1, 2)), zorder=2)
    ax.plot(lags, T, "-", color=T_COL, lw=1.1, marker="o", ms=3.8,
            mfc=T_COL, mec="white", mew=0.6, zorder=4)
    ax.set_ylim(-0.016, 0.218)
    ax.set_xlim(lags.min() - 0.35, lags.max() + 0.35)
    ax.set_xticks(lags)
    ax.set_xlabel("Lag (weeks)", labelpad=5)
    ax.set_ylabel("True effect on BI (per \u00b0C)", labelpad=3)
    ax.grid(lw=0.35, alpha=0.3, zorder=0)
    zl = lags[T == 0]
    if len(zl) >= 2:
        ax.annotate(f"exactly zero\nat lags {zl.min()}\u2013{zl.max()}",
                    xy=(zl.mean(), 0.0), xytext=(0, 11), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.2, color="#444444")
    figs.panel_letter(ax, "(a)", y=-0.34)

    # -------------------------------------------------- (b) rainfall, log scale
    ax = axes[1]
    ax.set_yscale("log")
    ax.plot(lags, R, "-", color=R_COL, lw=1.1, marker="s", ms=3.2,
            mfc=R_COL, mec="white", mew=0.6, zorder=4)
    ax.set_ylim(5e-5, 1.6e-1)          # keeps the 10^-4 tick inside the frame
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_xlim(lags.min() - 0.35, lags.max() + 0.35)
    ax.set_xticks(lags)
    ax.set_xlabel("Lag (weeks)", labelpad=5)
    ax.set_ylabel("True effect on BI (per mm)", labelpad=3)
    ax.grid(which="major", lw=0.35, alpha=0.3, zorder=0)
    figs.panel_letter(ax, "(b)", y=-0.34)

    fig.tight_layout(w_pad=1.8)
    figs.raise_frame(fig)

    # native 2400-px copy for the inline figure in the supplementary .docx
    # (rendered at savefig time, never upsampled from the 200-dpi preview)
    emb = os.path.join(FIG, "FigS1 Lag Structure_embed.png")
    w_in = fig.get_size_inches()[0]
    fig.savefig(emb, dpi=EMBED_PX / w_in, bbox_inches="tight")
    from PIL import Image
    im = Image.open(emb)
    if im.width != EMBED_PX:
        h = round(im.height * EMBED_PX / im.width)
        im.resize((EMBED_PX, h), Image.LANCZOS).save(emb, dpi=(600, 600))
    print(f"  embed copy {Image.open(emb).size} px")

    figs.save(fig, "FigS1 Lag Structure")

    print("  values (mean over panels):")
    for k, row in m.iterrows():
        print(f"    lag {int(k)}:  T = {row['T']:.4f}   Rain = {row['Rain']:.6f}")


if __name__ == "__main__":
    figS1_lag_structure()
