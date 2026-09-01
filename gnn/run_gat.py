#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment 1: spatial graph model vs. non-spatial baselines
======================================================================
Unified-protocol comparison of a graph attention network (GAT) against the
baseline baselines (XGBoost / RandomForest / Lasso / LSTM / SARIMA) for one-week-
ahead prediction of the simulated Breteau index (BI), plus:

  * leave-one-grid-out (LOO) extrapolation  (feature-dense / label-sparse)
  * graph-structure sensitivity (kNN k in {2,3,5}, complete, identity, MLP)

Implementation notes (key design decisions):
  [1] Node features are standardised (StandardScaler fitted on the
      training years only, per fold). Feeding raw features lets large-scale
      predictors such as Rain_cum4 (~10^2 mm) dominate sin/cos_week (~1) and
      slows / biases GAT optimisation; the LSTM baseline scales, the GAT does
      not, so standardisation is applied explicitly.
  [2] The validation year is EXCLUDED from training (train = years <= ty-2,
      val = year ty-1), matching the XGBoost early-stopping protocol so that
      all models are tuned on the same held-out window.
  [3] True two-head attention is implemented (heads=2, genuinely multi-head).
  [4] Three random seeds per configuration; mean +/- sd reported.
  [5] LOO: the held-out grid is excluded from the training LOSS (obs=0)
      while its weather features still participate in message passing
      — the honest "dense weather / sparse BI monitoring" setting (features
      are retained, labels are masked in the loss).

The non-spatial baselines, panel_data.csv, feature engineering and the
rolling-origin protocol are reused from the shared baseline pipeline
(pipeline_redo.py) so that all numbers remain directly comparable.
"""
import os, json, sys, gc, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- paths ----
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
RES_A = os.path.abspath(os.path.join(BASE, '..', 'results'))
os.makedirs(RES_A, exist_ok=True)
sys.path.insert(0, BASELINE_DIR)
import pipeline as base  # renamed from pipeline_redo during repo unification            # baseline feature engineering / FEATS / dm_test

DEVICE = torch.device('cpu')
torch.set_num_threads(1)              # keep memory footprint low (sandbox-safe)

TEST_YEARS = (2019, 2020, 2021, 2022)
GRIDS = base.GRIDS
N = len(GRIDS)
gidx = {g: i for i, g in enumerate(GRIDS)}
WINDOW = 8                            # kept for parity with baseline LSTM (not used by GAT)
SEEDS = (2024, 7, 42)


# ------------------------------------------------------------- adjacency ----
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_coords():
    coords = {}
    for g in GRIDS:
        with open(os.path.join(BASELINE_DIR, 'data', f'era5_{g}.json')) as f:
            j = json.load(f)
        coords[g] = (j['latitude'], j['longitude'])
    return coords


def distance_matrix(coords):
    dist = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dist[i, j] = haversine(*coords[GRIDS[i]], *coords[GRIDS[j]])
    return dist


def build_adjacency(dist, mode='knn', k=3):
    """Graph variants for the sensitivity analysis."""
    A = np.eye(N)                                # self-loops always included
    if mode == 'identity':                       # ablation: no neighbour edges
        return A
    if mode == 'complete':
        return np.ones((N, N))
    if mode == 'random':                         # ablation: random topology
        rng = np.random.default_rng(0)
        R = (rng.random((N, N)) < 0.3).astype(float)
        R = np.maximum(R, R.T)
        np.fill_diagonal(R, 1.0)
        return R
    # default: symmetric kNN on great-circle distance
    for i in range(N):
        order = np.argsort(dist[i])
        for j in order[1:k + 1]:
            A[i, j] = 1.0
            A[j, i] = 1.0
    return A


# -------------------------------------------------------------- snapshots ---
def build_snapshots(panel):
    """One graph snapshot per calendar week; node feature = the same 15
    biology-informed features used by the baseline XGBoost (fair comparison)."""
    df = base.feature_engineering(panel)
    df['BI_future'] = df.groupby('grid')['BI'].shift(-1)      # as in the baseline prepare_Xy
    df = df.dropna(subset=base.FEATS + ['BI_future']).reset_index(drop=True)
    snaps = {}
    for g in GRIDS:
        gg = df[df['grid'] == g].sort_values(['year', 'week_in_year'])
        F = gg[base.FEATS].values.astype(float)
        Y = gg['BI_future'].values.astype(float)
        yr = gg['year'].values
        wi = gg['week_in_year'].values
        for i in range(len(gg)):
            key = (int(yr[i]), int(wi[i]))
            snaps.setdefault(key, {})[gidx[g]] = (F[i], float(Y[i]))
    return snaps


def snapshot_tensor(snaps, keys, scaler, mask_grid_loss=None):
    """Stack weekly snapshots -> X(B,N,F) y(B,N) obs(B,N).
    mask_grid_loss: index of a grid whose LABEL is excluded from the loss
    (obs=0) while its features remain visible to neighbours (LOO setting)."""
    Xs, Ys, Os = [], [], []
    F = len(base.FEATS)
    for key in keys:
        if key not in snaps:
            continue
        node = snaps[key]
        x = np.zeros((N, F))
        y = np.zeros(N)
        obs = np.zeros(N)
        for gi in range(N):
            if gi in node:
                xv, yv = node[gi]
                x[gi] = xv
                y[gi] = yv
                obs[gi] = 0.0 if gi == mask_grid_loss else 1.0
        Xs.append(x)
        Ys.append(y)
        Os.append(obs)
    if not Xs:
        return None, None, None
    X = np.array(Xs)
    B, n, F = X.shape
    X = scaler.transform(X.reshape(-1, F)).reshape(B, n, F)   # fix-1
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(np.array(Ys), dtype=torch.float32),
            torch.tensor(np.array(Os), dtype=torch.float32))


# ------------------------------------------------------------------- GAT ----
class GAT(nn.Module):
    """Two-layer-free single GAT block with honest multi-head attention (fix-3)."""

    def __init__(self, in_f, hid=32, heads=2):
        super().__init__()
        self.heads = heads
        self.hid = hid
        self.W = nn.Linear(in_f, hid * heads, bias=False)
        self.attn = nn.Linear(2 * hid, 1, bias=False)
        self.act = nn.ELU()
        self.out = nn.Linear(hid * heads, 1)

    def forward(self, x, adj):
        B, Nn, _ = x.shape
        h = self.W(x).view(B, Nn, self.heads, self.hid)        # (B,N,H,hid)
        outs = []
        for hd in range(self.heads):
            hh = h[:, :, hd, :]                                # (B,N,hid)
            src = hh.unsqueeze(2).expand(B, Nn, Nn, self.hid)  # i
            dst = hh.unsqueeze(1).expand(B, Nn, Nn, self.hid)  # j
            e = self.attn(torch.cat([src, dst], -1)).squeeze(-1)   # (B,N,N)
            e = e.masked_fill(adj == 0, -1e9)
            a = torch.softmax(e, dim=-1)                       # over neighbours j
            outs.append(self.act(a @ hh))                      # (B,N,hid)
        out = torch.cat(outs, dim=-1)                          # (B,N,H*hid)
        return self.out(out).squeeze(-1)                       # (B,N)


class MLP(nn.Module):
    """Non-spatial ablation with matched capacity (no message passing)."""

    def __init__(self, in_f, hid=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_f, hid), nn.ELU(), nn.Linear(hid, 1))

    def forward(self, x, adj=None):
        return self.net(x).squeeze(-1)


# -------------------------------------------------------------- training ----
def fit_model(model, snaps, train_keys, val_keys, scaler, A_t,
              mask_grid_loss=None, epochs=150, lr=0.005, patience=20, seed=2024):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # re-initialise weights deterministically per seed
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.reset_parameters()
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.MSELoss(reduction='none')

    Xtr, Ytr, Otr = snapshot_tensor(snaps, train_keys, scaler, mask_grid_loss)
    Xtr, Ytr, Otr = Xtr.to(DEVICE), Ytr.to(DEVICE), Otr.to(DEVICE)
    Xva = Yva = Ova = None
    if val_keys:
        Xva, Yva, Ova = snapshot_tensor(snaps, val_keys, scaler)
        Xva, Yva, Ova = Xva.to(DEVICE), Yva.to(DEVICE), Ova.to(DEVICE)

    best, best_state, wait = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(Xtr, A_t)
        loss = (crit(pred, Ytr) * Otr).sum() / Otr.sum().clamp(min=1)
        if not torch.isfinite(loss):
            print(f'  [warn] non-finite loss at epoch {ep}; stop', flush=True)
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if Xva is not None:
            model.eval()
            with torch.no_grad():
                pv = model(Xva, A_t)
                vl = ((crit(pv, Yva) * Ova).sum() / Ova.sum().clamp(min=1)).item()
            if vl < best - 1e-6:
                best, wait = vl, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model, A_t, snaps, keys, scaler):
    model.eval()
    X, Y, O = snapshot_tensor(snaps, keys, scaler)
    with torch.no_grad():
        P = model(X.to(DEVICE), A_t).cpu()
    rows = []
    kept = [k for k in keys if k in snaps]
    for bi, key in enumerate(kept):
        for gi in range(N):
            if gi in snaps[key]:                   # every observed node
                rows.append({'grid': GRIDS[gi], 'year': key[0],
                             'week_in_year': key[1],
                             'y_true': Y[bi, gi].item(),
                             'y_pred': P[bi, gi].item()})
    return pd.DataFrame(rows)


def metric_dict(y_true, y_pred):
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = float(r2_score(yt, yp))
    try:
        auc = float(roc_auc_score((yt >= 10).astype(int), yp))
    except ValueError:
        auc = float('nan')
    return {'RMSE': round(rmse, 3), 'R2': round(r2, 3),
            'AUC': round(auc, 3), 'n': len(yt)}


# ------------------------------------------------------------------- CV -----
def run_cv(snaps, dist, seeds=SEEDS):
    """Rolling-origin evaluation, test years 2019-2022 (unified rolling-origin protocol)."""
    A = build_adjacency(dist, 'knn', 3)
    A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)
    per_seed_preds = []
    for sd in seeds:
        frames = []
        for ty in TEST_YEARS:
            train_keys = [k for k in snaps if k[0] <= ty - 2]      # fix-2
            val_keys = [k for k in snaps if k[0] == ty - 1]
            test_keys = [k for k in snaps if k[0] == ty]
            tr_rows = np.vstack(
                [snaps[k][gi][0] for k in train_keys for gi in snaps[k]])
            scaler = StandardScaler().fit(tr_rows)                 # fix-1
            model = GAT(len(base.FEATS))
            model = fit_model(model, snaps, train_keys, val_keys, scaler, A_t, seed=sd)
            dfp = predict(model, A_t, snaps, test_keys, scaler)
            frames.append(dfp)
            m = metric_dict(dfp['y_true'], dfp['y_pred'])
            print(f'  seed={sd} test-year={ty}: GAT RMSE={m["RMSE"]} R2={m["R2"]}',
                  flush=True)
            del model
            gc.collect()
        d = pd.concat(frames, ignore_index=True)
        d['seed'] = sd
        per_seed_preds.append(d)
    return pd.concat(per_seed_preds, ignore_index=True)


def run_graph_sensitivity(snaps, dist, seed=2024):
    """Graph-structure sensitivity: does the learned topology matter?"""
    variants = [('kNN-3 (main)', 'knn', 3), ('kNN-2', 'knn', 2),
                ('kNN-5', 'knn', 5), ('complete', 'complete', None),
                ('identity (no edges)', 'identity', None),
                ('random', 'random', None), ('MLP (no graph)', 'mlp', None)]
    rows = []
    for label, mode, k in variants:
        A = build_adjacency(dist, mode, k or 3)
        A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)
        frames = []
        for ty in TEST_YEARS:
            train_keys = [kk for kk in snaps if kk[0] <= ty - 2]
            val_keys = [kk for kk in snaps if kk[0] == ty - 1]
            test_keys = [kk for kk in snaps if kk[0] == ty]
            tr_rows = np.vstack(
                [snaps[kk][gi][0] for kk in train_keys for gi in snaps[kk]])
            scaler = StandardScaler().fit(tr_rows)
            model = MLP(len(base.FEATS)) if mode == 'mlp' else GAT(len(base.FEATS))
            model = fit_model(model, snaps, train_keys, val_keys, scaler, A_t, seed=seed)
            frames.append(predict(model, A_t, snaps, test_keys, scaler))
            del model
            gc.collect()
        d = pd.concat(frames, ignore_index=True)
        m = metric_dict(d['y_true'], d['y_pred'])
        rows.append({'graph': label, **m})
        print(f'  graph={label:22s} RMSE={m["RMSE"]} R2={m["R2"]}', flush=True)
    return pd.DataFrame(rows)


def run_loo(snaps, dist, seeds=(2024,)):
    """Leave-one-grid-out extrapolation (dense weather / sparse BI labels)."""
    A = build_adjacency(dist, 'knn', 3)
    A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)
    rows = []
    for hold in range(N):
        for sd in seeds:
            train_keys = [k for k in snaps if k[0] < 2019]
            test_keys = [k for k in snaps if k[0] >= 2019]
            tr_rows = np.vstack(
                [snaps[k][gi][0] for k in train_keys for gi in snaps[k]
                 if gi != hold])
            scaler = StandardScaler().fit(tr_rows)
            model = GAT(len(base.FEATS))
            model = fit_model(model, snaps, train_keys, None, scaler, A_t,
                              mask_grid_loss=hold, epochs=100, seed=sd)   # fix-5
            dfp = predict(model, A_t, snaps, test_keys, scaler)
            dfp = dfp[dfp['grid'] == GRIDS[hold]]
            rmse = float(np.sqrt(mean_squared_error(dfp['y_true'], dfp['y_pred'])))
            rows.append({'held_out_grid': GRIDS[hold], 'seed': sd,
                         'GAT_RMSE': round(rmse, 3), 'n': len(dfp)})
            print(f'  hold={GRIDS[hold]:9s} seed={sd}: GAT LOO RMSE={rmse:.3f}',
                  flush=True)
            del model
            gc.collect()
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-loo', action='store_true')
    ap.add_argument('--skip-sens', action='store_true')
    args = ap.parse_args()

    print('loading baseline panel and building snapshots ...', flush=True)
    panel = pd.read_csv(os.path.join(BASELINE_DIR, 'results', 'panel_data.csv'))
    snaps = build_snapshots(panel)
    coords = load_coords()
    dist = distance_matrix(coords)
    A3 = build_adjacency(dist, 'knn', 3)
    print(f'  graph: {N} nodes, edges={int(A3.sum() - N) // 2}, '
          f'mean off-diag distance={dist[~np.eye(N, dtype=bool)].mean():.1f} km',
          flush=True)

    # ---- 1. main rolling-origin CV (3 seeds) -------------------------------
    print('\n[1/4] rolling-origin CV, GAT, 3 seeds', flush=True)
    gat = run_cv(snaps, dist)
    gat.to_csv(os.path.join(RES_A, 'preds_GAT_allseeds.csv'), index=False)
    seed_metrics = (gat.groupby('seed')
                    .apply(lambda d: pd.Series(metric_dict(d['y_true'], d['y_pred'])),
                         include_groups=False)
                    .reset_index())
    print(seed_metrics.to_string(index=False), flush=True)

    # predictions of the reference seed for DM tests / figures
    ref = gat[gat['seed'] == SEEDS[0]].drop(columns=['seed'])
    ref.to_csv(os.path.join(RES_A, 'preds_GAT.csv'), index=False)

    # ---- 2. comparison with baseline baselines on identical samples --------------
    print('\n[2/4] comparison with baseline baselines (identical samples)', flush=True)
    comp = ref.rename(columns={'y_pred': 'y_pred_GAT'})
    table = {'GAT (mean of 3 seeds)': {
        'RMSE': round(seed_metrics['RMSE'].mean(), 3),
        'R2': round(seed_metrics['R2'].mean(), 3),
        'RMSE_sd': round(seed_metrics['RMSE'].std(), 3),
        'R2_sd': round(seed_metrics['R2'].std(), 3),
        'n': int(seed_metrics['n'].iloc[0])}}
    for name in ['XGBoost', 'RandomForest', 'Lasso', 'LSTM', 'SARIMA']:
        p = pd.read_csv(os.path.join(BASELINE_DIR, 'results', f'preds_{name}.csv'))
        pm = p.rename(columns={'y_pred': f'y_pred_{name}'})[
            ['grid', 'year', 'week_in_year', f'y_pred_{name}']]
        m = comp.merge(pm, on=['grid', 'year', 'week_in_year'], how='inner')
        m = m.dropna(subset=[f'y_pred_{name}'])
        table[name] = metric_dict(m['y_true'], m[f'y_pred_{name}'])
    tbl = pd.DataFrame(table).T
    tbl.to_csv(os.path.join(RES_A, 'gat_baseline_compare.csv'))
    print(tbl.to_string(), flush=True)

    # ---- 3. Diebold-Mariano tests ------------------------------------------
    print('\n[3/4] Diebold-Mariano tests (GAT vs each baseline)', flush=True)
    dm_rows = []
    for name in ['XGBoost', 'RandomForest', 'Lasso', 'LSTM', 'SARIMA']:
        p = pd.read_csv(os.path.join(BASELINE_DIR, 'results', f'preds_{name}.csv'))
        pm = p.rename(columns={'y_pred': f'y_pred_{name}'})[
            ['grid', 'year', 'week_in_year', f'y_pred_{name}']]
        m = comp.merge(pm, on=['grid', 'year', 'week_in_year'], how='inner').dropna()
        e_g = m['y_true'] - m['y_pred_GAT']
        e_o = m['y_true'] - m[f'y_pred_{name}']
        dm, pval = base.dm_test(e_g.values, e_o.values)
        verdict = ('GAT significantly better' if (pval < 0.05 and dm < 0)
                   else ('baseline significantly better' if (pval < 0.05 and dm > 0)
                         else 'no significant difference'))
        dm_rows.append({'comparison': f'GAT vs {name}', 'DM': round(dm, 3),
                        'p_value': f'{pval:.3e}', 'n': len(m), 'verdict': verdict})
        print(f'  GAT vs {name:12s}: DM={dm:.3f}, p={pval:.2e} -> {verdict}',
              flush=True)
    pd.DataFrame(dm_rows).to_csv(os.path.join(RES_A, 'dm_gat_vs_all.csv'), index=False)

    # ---- 4. graph sensitivity + LOO ----------------------------------------
    if not args.skip_sens:
        print('\n[4/4] graph-structure sensitivity (seed=2024)', flush=True)
        sens = run_graph_sensitivity(snaps, dist)
        sens.to_csv(os.path.join(RES_A, 'graph_sensitivity.csv'), index=False)
    if not args.skip_loo:
        print('\n[LOO] leave-one-grid-out extrapolation', flush=True)
        loo = run_loo(snaps, dist)
        loo.to_csv(os.path.join(RES_A, 'loo_gat.csv'), index=False)

    print(f'\ndone. results -> {RES_A}', flush=True)


if __name__ == '__main__':
    main()
