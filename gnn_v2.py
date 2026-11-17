#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Graph attention network under varying strengths of spatial coupling.

The question is whether message passing starts to pay once the system actually
contains cross-grid dependence. kappa = 0 is the no-coupling case; kappa > 0
lets each grid inherit part of its neighbours' productivity.

Everything is batched over weeks, so one epoch is a handful of tensor ops.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim
from exp import make_features, metrics, F0

# graph models see the same weather features plus the local lagged outcome,
# so that a neighbour's observation is available to message passing.
FEATS_G = F0 + ['BI_lag1']

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("AEGIS_OUT", os.path.join(HERE, "results"))
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(2)


def knn_adj(k=3, self_loops=True):
    coords = np.array([[sim.GRID_LAT[g], sim.GRID_LON[g]] for g in sim.GRIDS])
    n = len(sim.GRIDS)
    A = np.zeros((n, n))
    for i in range(n):
        d = np.hypot(coords[:, 0] - coords[i, 0], coords[:, 1] - coords[i, 1])
        for j in np.argsort(d)[1:k + 1]:
            A[i, j] = A[j, i] = 1.0
    if self_loops:
        A = A + np.eye(n)
    return A


class GAT(nn.Module):
    """Batched two-head graph attention over a fixed G-node graph."""

    def __init__(self, d_in, hidden=32, heads=2, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.lin = nn.Linear(d_in, hidden * heads, bias=False)
        self.att = nn.Linear(2 * hidden, 1, bias=False)
        self.act = nn.ELU()
        self.out = nn.Linear(hidden * heads, 1)

    def forward(self, x, A, keep=None):
        # x: (B, G, d)   A: (G, G) 0/1   keep: (B, G) bool mask for observed labels
        B, G, _ = x.shape
        H = self.heads
        h = self.lin(x).view(B, G, H, -1)                      # (B,G,H,D)
        outs = []
        for hd in range(H):
            hh = h[:, :, hd, :]                                # (B,G,D)
            src = hh.unsqueeze(2).expand(B, G, G, hh.size(-1))
            dst = hh.unsqueeze(1).expand(B, G, G, hh.size(-1))
            e = self.att(torch.cat([src, dst], -1)).squeeze(-1)          # (B,G,G)
            e = e.masked_fill(A.view(1, G, G) == 0, -1e9)
            a = torch.softmax(e, dim=-1)
            outs.append(self.act(a @ hh))
        return self.out(torch.cat(outs, dim=-1)).squeeze(-1)   # (B,G)


class MLP(nn.Module):
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x, A=None, keep=None):
        return self.net(x).squeeze(-1)


def build_tensors(df, cols):
    grids, weeks = sim.GRIDS, sorted(df["week_idx"].unique())
    def piv(col):
        return df.pivot(index="week_idx", columns="grid", values=col)\
                 .reindex(index=weeks, columns=grids).values
    X = np.stack([piv(c) for c in cols], axis=-1).astype(np.float32)
    Y = piv("y").astype(float)
    YR = piv("year").astype(float)
    return X, Y, YR


def train_eval(X, Y, YR, A, MakeModel, tr_years, te_year, seed, epochs=150):
    torch.manual_seed(seed)
    tr = np.isin(YR, tr_years) & ~np.isnan(Y)
    te = (YR == te_year) & ~np.isnan(Y)
    if tr.sum() < 100 or te.sum() == 0:
        return None
    mu = np.nanmean(X[tr], axis=0, keepdims=True)
    sd = np.nanstd(X[tr], axis=0, keepdims=True) + 1e-6
    Xn = np.nan_to_num((X - mu) / sd)
    Xt = torch.tensor(Xn); Yt = torch.tensor(np.nan_to_num(Y), dtype=torch.float32)
    Mt = torch.tensor(tr)
    model = MakeModel(len(FEATS_G))
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()
    best, wait = np.inf, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        pred = model(Xt, A)
        loss = lossf(pred[Mt], Yt[Mt])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if float(loss) < best - 1e-5:
            best, wait = float(loss), 0
        else:
            wait += 1
            if wait >= 20:
                break
    model.eval()
    with torch.no_grad():
        pr = model(Xt, A).numpy()
    mt = torch.tensor(te).numpy()
    return pr[mt], Y[mt]


def main(scenarios, seeds, kappa):
    for sc in scenarios:
        for sd in seeds:
            tag = f"{sc}{int(kappa*100):02d}_{sd}_gat"
            met_f = os.path.join(OUT, f"metrics_{tag}.csv")
            if os.path.exists(met_f):
                continue
            t0 = time.time()
            panel = sim.build_panel(seed=sd, scenario=sc, kappa=kappa)
            df = make_features(panel)
            df["y"] = df.groupby("grid")["BI"].shift(-1)
            sub = df.dropna(subset=FEATS_G + ["y"])
            X, Y, YR = build_tensors(sub, FEATS_G)
            A3 = torch.tensor(knn_adj(3), dtype=torch.float32)
            AI = torch.eye(len(sim.GRIDS))
            A0 = torch.zeros(len(sim.GRIDS), len(sim.GRIDS))
            refs = {"GAT_knn3": (A3, GAT), "GAT_identity": (AI, GAT),
                    "MLP": (A0, MLP)}
            rows = []
            for name, (A, cls) in refs.items():
                yp, yt = [], []
                for ty in (2019, 2020, 2021, 2022):
                    out = train_eval(X, Y, YR, A, cls, list(range(2015, ty)), ty, sd)
                    if out is None:
                        continue
                    p, y = out
                    yp.extend(p); yt.extend(y)
                if yt:
                    m = metrics(yt, yp)
                    m.update(scenario=sc, seed=sd,
                             kappa=kappa if sc == "S" else 0.0, arm="F0", model=name)
                    rows.append(m)
                print(f"  {tag} {name} R2={rows[-1]['R2']:.3f} ({time.time()-t0:.0f}s)",
                      flush=True)
            pd.DataFrame(rows).to_csv(met_f, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="S")
    ap.add_argument("--seeds", default="2024")
    ap.add_argument("--kappa", type=float, default=0.35)
    a = ap.parse_args()
    main(a.scenarios.split(","), [int(x) for x in a.seeds.split(",")], a.kappa)
