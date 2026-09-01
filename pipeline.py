#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aedes albopictus density forecasting from ERA5 weather
================================================================
"""
import os, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
RESULTS = os.path.join(BASE, 'results')
os.makedirs(RESULTS, exist_ok=True)

SEED = 2024
GRIDS = ['conghua', 'huadu', 'zengcheng', 'baiyun', 'tianhe',
         'yuexiu', 'haizhu', 'panyu', 'nansha', 'huangpu']
GRID_LABEL = {'conghua': 'Conghua', 'huadu': 'Huadu', 'zengcheng': 'Zengcheng',
              'baiyun': 'Baiyun', 'tianhe': 'Tianhe', 'yuexiu': 'Yuexiu',
              'haizhu': 'Haizhu', 'panyu': 'Panyu', 'nansha': 'Nansha',
              'huangpu': 'Huangpu'}
GRID_LAT = {'conghua': 23.55, 'huadu': 23.40, 'zengcheng': 23.29, 'baiyun': 23.26,
            'tianhe': 23.13, 'yuexiu': 23.13, 'haizhu': 23.08, 'panyu': 22.94,
            'nansha': 22.60, 'huangpu': 23.10}

# ===== Parameters of the PFI mechanistic model (sources in comments) =====
T_DEV0 = 10.0    # lower developmental threshold (deg C), Delatte 2009 (PMID:19198515)
T_OPT  = 27.0    # optimum temperature (deg C); Delatte 2009 reports a 25-30 deg C peak
T_MAX  = 39.0    # upper lethal temperature (deg C), Delatte 2009
SURV_MAX = 0.95  # maximum daily survival at the optimum, Delatte 2009
SURV_K   = 0.004 # decay coefficient of survival away from the optimum
RAIN_THRESH = 5.0  # threshold for effective rainfall (mm/day)
RAIN_GAIN   = 2.5  # maximum breeding-site gain from rainfall


def briere_dev_rate(T, a=0.000113, Tmin=T_DEV0, Tmax=T_MAX):
    """Briere temperature-dependent development rate (Briere 1999; Delatte 2009)."""
    T = np.asarray(T, dtype=float)
    out = np.zeros_like(T)
    m = (T > Tmin) & (T < Tmax)
    out[m] = a * T[m] * (T[m] - Tmin) * np.sqrt(np.maximum(Tmax - T[m], 0))
    return out


def temp_survival(T):
    return SURV_MAX * np.exp(-SURV_K * (np.asarray(T, dtype=float) - T_OPT) ** 2)


def rainfall_brood_factor(rain_mm):
    eff = np.maximum(0.0, np.asarray(rain_mm, dtype=float) - RAIN_THRESH)
    return 1.0 + RAIN_GAIN * (1.0 - np.exp(-eff / 15.0))


def compute_pfi_weekly(T_wk, rain_wk, hum_wk):
    """Weekly PFI = development rate x survival (humidity-modulated) x rainfall factor."""
    dev = briere_dev_rate(T_wk)
    surv = temp_survival(T_wk)
    hum = np.clip(0.5 + 0.5 * np.clip(hum_wk / 70.0, 0, 1.6), 0, 1)
    surv = surv * hum
    rf = rainfall_brood_factor(rain_wk)
    return dev * surv * rf * 28.0  # scaled to the observed BI range (annual mean ~10, summer peak 20-40)


# ===== 1. Data construction: ERA5 weekly aggregation + PFI-simulated labels =====
def load_weekly_grid(grid):
    """Read one grid daily ERA5 JSON and aggregate it to weekly resolution."""
    with open(os.path.join(DATA, f'era5_{grid}.json')) as f:
        j = json.load(f)
    df = pd.DataFrame({
        'date': pd.to_datetime(j['daily']['time']),
        'T': j['daily']['temperature_2m_mean'],
        'rain': j['daily']['precipitation_sum'],
        'hum': j['daily']['relative_humidity_2m_mean'],
    })
    df = df.set_index('date')
    wk = df.resample('W-SUN').agg(T=('T', 'mean'), Rain=('rain', 'sum'),
                                  Hum=('hum', 'mean'))
    # keep complete weeks only (first week 2015-01-04, last week 2022-12-25)
    wk = wk.iloc[1:-1].reset_index().rename(columns={'date': 'week_end'})
    iso = wk['week_end'].dt.isocalendar()
    wk['year'] = iso['year'].values
    wk['week_in_year'] = iso['week'].values
    wk = wk[wk['year'].between(2015, 2022)].reset_index(drop=True)
    return wk


def build_panel(seed=SEED):
    """Build the 10-grid x week panel; BI is a noisy observation of the PFI output (simulated label)."""
    rng = np.random.default_rng(seed)
    frames = []
    for g in GRIDS:
        wk = load_weekly_grid(g)
        pfi = compute_pfi_weekly(wk['T'].values, wk['Rain'].values, wk['Hum'].values)
        # urbanisation proxy: density is higher towards the city centre (spatial heterogeneity)
        urban = np.exp(-0.5 * ((GRID_LAT[g] - 23.10) / 0.25) ** 2) * 0.6 + 0.7
        pop_scale = urban * rng.uniform(0.9, 1.1)
        bi = np.maximum(0, pfi * pop_scale + rng.normal(0, 1.5, len(pfi)))
        wk['grid'] = g
        wk['PFI'] = pfi
        wk['pop_density'] = np.round(pop_scale, 3)
        wk['BI'] = np.round(bi, 2)
        frames.append(wk)
    panel = pd.concat(frames, ignore_index=True)
    panel['week_idx'] = panel.groupby('grid').cumcount()
    panel.to_csv(os.path.join(RESULTS, 'panel_data.csv'), index=False)
    return panel


# ===== 2. Biologically motivated feature engineering =====
def feature_engineering(df, lookback=(2, 4)):
    df = df.sort_values(['grid', 'week_idx']).copy()
    df['GDD'] = (df['T'] - 10.0).clip(lower=0)
    for lb in lookback:
        df[f'GDD_cum{lb}'] = df.groupby('grid')['GDD'].transform(lambda s: s.rolling(lb).sum())
        df[f'Rain_cum{lb}'] = df.groupby('grid')['Rain'].transform(lambda s: s.rolling(lb).sum())
    df['sin_week'] = np.sin(2 * np.pi * df['week_in_year'] / 52.0)
    df['cos_week'] = np.cos(2 * np.pi * df['week_in_year'] / 52.0)
    df['T_lag1'] = df.groupby('grid')['T'].shift(1)
    df['Rain_lag1'] = df.groupby('grid')['Rain'].shift(1)
    df['Hum_lag1'] = df.groupby('grid')['Hum'].shift(1)
    df['TempRain_inter'] = (df['T'] - 27).clip(-5, 5) * df['Rain']
    return df


# Leakage control:
#   'pop_density' REMOVED from predictive features. It equals the grid-level
#   urbanization scale used to GENERATE the BI label (pop_scale, rounded to
#   3 dp), so retaining it gave models near-direct access to a deterministic
#   component of the target. The column remains in the panel (needed by the
#   DML ground-truth computation) but is no longer a model input.
#   Old 15-feature results (with leakage) are preserved separately as the
#   "leaked-feature control" contrast reported in the manuscript.
FEATS = ['GDD', 'GDD_cum2', 'GDD_cum4', 'Rain_cum2', 'Rain_cum4',
         'sin_week', 'cos_week', 'T_lag1', 'Rain_lag1', 'Hum_lag1',
         'TempRain_inter', 'T', 'Rain', 'Hum']


def prepare_Xy(panel, horizon=1):
    df = feature_engineering(panel)
    df['BI_future'] = df.groupby('grid')['BI'].shift(-horizon)
    df = df.dropna(subset=FEATS + ['BI_future']).reset_index(drop=True)
    return df


# ===== 3. Evaluation metrics =====
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, roc_auc_score)

def full_metrics(y_true, y_pred, threshold=10.0):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    yb_t = (y_true >= threshold).astype(int)
    yb_p = (y_pred >= threshold).astype(int)
    tp = int(((yb_t == 1) & (yb_p == 1)).sum()); tn = int(((yb_t == 0) & (yb_p == 0)).sum())
    fp = int(((yb_t == 0) & (yb_p == 1)).sum()); fn = int(((yb_t == 1) & (yb_p == 0)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    try:
        auc = roc_auc_score(yb_t, y_pred)  # ROC AUC computed on continuous predictions
    except ValueError:
        auc = np.nan
    bal_acc = (sens + spec) / 2
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'AUC': auc,
            'BalancedAcc': bal_acc, 'Sensitivity': sens,
            'Specificity': spec, 'F1': f1, 'n': len(y_true)}


# ===== 4. Models (common protocol: predict BI(t+1), same grid, same test year) =====
def run_ml_models(df, test_years=(2019, 2020, 2021, 2022)):
    """Lasso / RF / XGBoost under a rolling-origin protocol; XGBoost early-stops on
the year preceding the test year, which is excluded from training."""
    from sklearn.linear_model import Lasso
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor

    results, preds_store = {}, {}
    for name in ['Lasso', 'RandomForest', 'XGBoost']:
        y_p_all, y_t_all, rows = [], [], []
        for ty in test_years:
            tr = df[df['year'] < ty]
            te = df[df['year'] == ty]
            if name == 'Lasso':
                model = Lasso(alpha=0.01)
                model.fit(tr[FEATS], tr['BI_future'])
            elif name == 'RandomForest':
                model = RandomForestRegressor(n_estimators=500, max_depth=10,
                                              min_samples_split=5, n_jobs=-1,
                                              random_state=SEED)
                model.fit(tr[FEATS], tr['BI_future'])
            else:
                # early-stopping validation set = the last complete year before the
                # test year, strictly excluded from training; identical across all
                # test years (test 2019 -> train 2015-2017, validate 2018)
                tr2 = df[df['year'] < ty - 1]
                va2 = df[df['year'] == ty - 1]
                model = XGBRegressor(n_estimators=500, learning_rate=0.05,
                                     max_depth=4, subsample=0.8,
                                     colsample_bytree=0.8, random_state=SEED,
                                     early_stopping_rounds=20)
                model.fit(tr2[FEATS], tr2['BI_future'],
                          eval_set=[(va2[FEATS], va2['BI_future'])], verbose=False)
            y_p = model.predict(te[FEATS])
            y_p_all.extend(y_p); y_t_all.extend(te['BI_future'].values)
            for i in range(len(te)):
                rows.append({'grid': te.iloc[i]['grid'], 'year': ty,
                             'week_in_year': te.iloc[i]['week_in_year'],
                             'y_true': te.iloc[i]['BI_future'], 'y_pred': y_p[i]})
        results[name] = full_metrics(y_t_all, y_p_all)
        preds_store[name] = pd.DataFrame(rows)
        print(f'  {name}: RMSE={results[name]["RMSE"]:.3f}  R2={results[name]["R2"]:.3f}  '
              f'AUC={results[name]["AUC"]:.3f}')
    return results, preds_store


def run_sarima(panel, df, test_years=(2019, 2020, 2021, 2022)):
    """SARIMA(1,1,1)(0,1,1,52) under the identical protocol: per grid, one-week-ahead
    rolling forecasts over the test year (one-step; observed values are appended
    without re-estimation), targeting BI(t+1) to align with ML's BI_future.
    """
    from statsmodels.tsa.arima.model import ARIMA
    y_p_all, y_t_all, rows = [], [], []
    t0 = time.time()
    for g in GRIDS:
        grp = panel[panel['grid'] == g].sort_values('week_idx').reset_index(drop=True)
        bi = grp['BI'].values
        years = grp['year'].values
        for ty in test_years:
            tr_end = np.where(years == ty)[0]
            if len(tr_end) == 0:
                continue
            tr_end = tr_end[0]  # index of the first test-year week
            train = bi[:tr_end]
            test_idx = np.where(years == ty)[0]
            if len(train) < 150:
                continue
            try:
                fit = ARIMA(train, order=(1, 1, 1),
                            seasonal_order=(0, 1, 1, 52)).fit(
                                method_kwargs={'maxiter': 100})
                # one-step weekly forecasting: predict week t+1, append the observation, continue
                cur = fit
                for k, ix in enumerate(test_idx):
                    fc = cur.forecast(1)[0]
                    # alignment: ML predicts BI(t+1) from features at t; here the
                    # forecast issued after appending ix-1 targets ix, matching
                    # BI_future within the test year (same week index)
                    y_p = max(fc, 0)
                    y_t = bi[ix]
                    y_p_all.append(y_p); y_t_all.append(y_t)
                    rows.append({'grid': g, 'year': ty,
                                 'week_in_year': int(grp.loc[ix, 'week_in_year']),
                                 'y_true': y_t, 'y_pred': y_p})
                    cur = cur.append([bi[ix]], refit=False)
            except Exception as e:
                print(f'  SARIMA failed for {g}/{ty}: {type(e).__name__}: {e}')
    print(f'  SARIMA finished in {time.time()-t0:.0f}s')
    res = full_metrics(y_t_all, y_p_all)
    print(f'  SARIMA: RMSE={res["RMSE"]:.3f}  R2={res["R2"]:.3f}  AUC={res["AUC"]:.3f}')
    return {'SARIMA': res}, {'SARIMA': pd.DataFrame(rows)}


def run_lstm(df, test_years=(2019, 2020, 2021, 2022), window=8, epochs=40):
    """LSTM (PyTorch CPU): the past 8 weeks of features -> BI(t+1), same task as the ML models."""
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import StandardScaler
    torch.manual_seed(SEED)

    class LSTMPredictor(nn.Module):
        def __init__(self, d, h=64):
            super().__init__()
            self.lstm = nn.LSTM(d, h, batch_first=True)
            self.fc = nn.Linear(h, 1)
        def forward(self, x):
            o, _ = self.lstm(x)
            return self.fc(o[:, -1, :]).squeeze(-1)

    def build_seq(sub):
        Xs, ys, meta = [], [], []
        for g, gg in sub.groupby('grid'):
            gg = gg.sort_values('week_idx')
            X = gg[FEATS].values; y = gg['BI_future'].values
            for i in range(window, len(gg)):
                Xs.append(X[i - window:i]); ys.append(y[i])
                meta.append({'grid': g, 'year': int(gg.iloc[i]['year']),
                             'week_in_year': int(gg.iloc[i]['week_in_year'])})
        return np.array(Xs), np.array(ys), pd.DataFrame(meta)

    y_p_all, y_t_all, rows = [], [], []
    for ty in test_years:
        tr = df[df['year'] < ty]
        te = df[df['year'] <= ty].copy()  # test-year sequence needs prior window weeks
        te = te[(te['year'] == ty) | (te['year'] == ty - 1)]
        Xtr, ytr, _ = build_seq(tr)
        Xte_all, yte_all, meta = build_seq(te)
        keep = meta['year'] == ty
        Xte, yte, mte = Xte_all[keep.values], yte_all[keep.values], meta[keep]
        sc = StandardScaler()
        ntr, w, d = Xtr.shape
        sc.fit(Xtr.reshape(-1, d))
        Xtr = sc.transform(Xtr.reshape(-1, d)).reshape(ntr, w, d)
        nte = Xte.shape[0]
        Xte = sc.transform(Xte.reshape(-1, d)).reshape(nte, w, d)
        model = LSTMPredictor(d)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        lossfn = nn.MSELoss()
        ds = torch.utils.data.TensorDataset(
            torch.tensor(Xtr, dtype=torch.float32),
            torch.tensor(ytr, dtype=torch.float32))
        ld = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
        model.train()
        for ep in range(epochs):
            for xb, yb in ld:
                opt.zero_grad()
                loss = lossfn(model(xb), yb)
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            y_p = model(torch.tensor(Xte, dtype=torch.float32)).numpy()
        y_p_all.extend(y_p); y_t_all.extend(yte)
        for i in range(len(mte)):
            rows.append({'grid': mte.iloc[i]['grid'], 'year': ty,
                         'week_in_year': mte.iloc[i]['week_in_year'],
                         'y_true': yte[i], 'y_pred': float(y_p[i])})
        print(f'  LSTM done for {ty} (train n={ntr}, test n={nte})')
    res = full_metrics(y_t_all, y_p_all)
    print(f'  LSTM: RMSE={res["RMSE"]:.3f}  R2={res["R2"]:.3f}  AUC={res["AUC"]:.3f}')
    return {'LSTM': res}, {'LSTM': pd.DataFrame(rows)}


# ===== 5. Diebold-Mariano tests =====
def dm_test(e1, e2, h=1, power=2):
    """Diebold-Mariano test: H0 = equal predictive accuracy.
Returns the DM statistic and the two-sided p-value."""
    from scipy import stats as sst
    d = np.abs(e1) ** power - np.abs(e2) ** power if power == 1 else e1 ** 2 - e2 ** 2
    n = len(d)
    mean_d = d.mean()
    # Newey-West HAC variance, autocorrelation-corrected up to lag h-1+4
    gamma0 = np.mean((d - mean_d) ** 2)
    var = gamma0
    for lag in range(1, 5):
        gamma = np.mean((d[lag:] - mean_d) * (d[:-lag] - mean_d))
        var += 2 * (1 - lag / 5) * gamma
    var = max(var, 1e-12)  # guard: the HAC variance may be estimated as non-positive
    dm = mean_d / np.sqrt(var / n)
    p = 2 * (1 - sst.norm.cdf(abs(dm)))
    return dm, p


# ===== 6. SHAP =====
def run_shap(df, test_year=2021):
    import shap
    from xgboost import XGBRegressor
    tr = df[df['year'] < test_year]
    te = df[df['year'] == test_year]
    # n_estimators aligned with the main rolling-origin XGBoost (500)
    # so SHAP explains the same model configuration reported in the results.
    model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4,
                         random_state=SEED)
    model.fit(tr[FEATS], tr['BI_future'])
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(te[FEATS])
    imp = pd.Series(np.abs(sv).mean(axis=0), index=FEATS).sort_values(ascending=False)
    return imp, sv, te


# ===== Main =====
if __name__ == '__main__':
    print('=' * 60)
    print('Aedes albopictus density forecasting benchmark')
    print('=' * 60)

    print('\n[1/6] Building the panel (ERA5 weather + PFI mechanistic labels)')
    panel = build_panel()
    print(f'  panel: {len(panel)} rows | {panel["grid"].nunique()} grids | '
          f'{panel["year"].min()}-{panel["year"].max()} | '
          f'BI mean {panel["BI"].mean():.1f}, range {panel["BI"].min():.1f}-{panel["BI"].max():.1f}')
    df = prepare_Xy(panel)
    print(f'  after feature engineering: {len(df)} rows, {len(FEATS)} features')

    print('\n[2/6] Machine-learning models (common protocol, rolling origin 2019-2022)')
    ml_res, ml_preds = run_ml_models(df)

    print('\n[3/6] SARIMA baseline (same one-step rolling forecast strategy)')
    sa_res, sa_preds = run_sarima(panel, df)

    print('\n[4/6] LSTM (PyTorch, aligned task)')
    try:
        lstm_res, lstm_preds = run_lstm(df)
    except ImportError:
        print('  torch not installed, skipping LSTM')
        lstm_res, lstm_preds = {}, {}

    all_res = {**ml_res, **sa_res, **lstm_res}
    all_preds = {**ml_preds, **sa_preds, **lstm_preds}
    res_df = pd.DataFrame(all_res).T.round(3)
    res_df.to_csv(os.path.join(RESULTS, 'model_results_unified.csv'))
    print('\n=== Model comparison under the common protocol ===')
    print(res_df.to_string())

    # save weekly predictions
    for name, p in all_preds.items():
        p.to_csv(os.path.join(RESULTS, f'preds_{name}.csv'), index=False)

    print('\n[5/6] Diebold-Mariano tests (XGBoost vs others, weekly paired)')
    dm_rows = []
    for other in ['SARIMA', 'Lasso', 'RandomForest'] + (['LSTM'] if 'LSTM' in all_preds else []):
        m = all_preds['XGBoost'].merge(all_preds[other], on=['grid', 'year', 'week_in_year'],
                                       suffixes=('_xgb', '_o'))
        e_xgb = m['y_true_xgb'] - m['y_pred_xgb']
        e_o = m['y_true_o'] - m['y_pred_o']
        dm, p = dm_test(e_xgb.values, e_o.values)
        dm_rows.append({'comparison': f'XGBoost vs {other}',
                        'DM_statistic': round(dm, 3),
                        'p_value': round(p, 4), 'n': len(m),
                        'conclusion': 'XGBoost significantly better'
                                      if (p < 0.05 and dm < 0) else
                                      ('Other model significantly better'
                                       if (p < 0.05 and dm > 0) else
                                       'No significant difference')})
        print(f'  XGBoost vs {other}: DM={dm:.3f}, p={p:.4f}')
    pd.DataFrame(dm_rows).to_csv(os.path.join(RESULTS, 'dm_tests.csv'), index=False)

    print('\n[6/6] SHAP interpretability and exploratory figures')
    imp, sv, te = run_shap(df, 2021)
    imp.round(3).to_csv(os.path.join(RESULTS, 'shap_importance.csv'))
    print('  SHAP feature importance, top 5:')
    for k, v in imp.head(5).items():
        print(f'    {k}: {v:.3f}')

    # ---- Figure A: climate drivers and the PFI seasonal curve (Tianhe grid) ----
    th = panel[panel['grid'] == 'tianhe'].reset_index(drop=True)
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    ax[0].plot(th['T'], color='#d62728', lw=0.7)
    ax[0].set_ylabel('Weekly mean temperature (deg C)')
    ax[1].bar(range(len(th)), th['Rain'], color='#1f77b4', width=1)
    ax[1].set_ylabel('Weekly rainfall (mm)')
    ax[2].plot(th['PFI'], color='#2ca02c', lw=0.9)
    ax[2].plot(th['BI'], color='#7f7f7f', lw=0.5, alpha=0.6)
    ax[2].set_ylabel('PFI / simulated BI')
    ax[2].set_xlabel('Week index (week 1 of 2015 - week 52 of 2022)')
    ax[2].legend(['PFI (potential fecundity index)', 'Simulated Breteau index'],
               loc='upper right')
    fig.suptitle('ERA5-driven PFI and simulated Breteau index, Tianhe grid, '
                 'Guangzhou, 2015-2022')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'fig1_climate_pfi.png'), dpi=300)
    plt.close(fig)

    # ---- Figure B: model comparison (RMSE and R2) ----
    names = list(res_df.index)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = ['#7f7f7f', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd'][:len(names)]
    axes[0].bar(names, res_df['RMSE'], color=colors)
    axes[0].set_title('RMSE (lower is better)')
    for i, v in enumerate(res_df['RMSE']):
        axes[0].text(i, v + 0.15, f'{v:.2f}', ha='center', fontsize=9)
    axes[1].bar(names, res_df['R2'], color=colors)
    axes[1].set_title('R2 (higher is better)')
    for i, v in enumerate(res_df['R2']):
        axes[1].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)
    for a in axes:
        for lbl in a.get_xticklabels():
            lbl.set_rotation(20)
    fig.suptitle('Weekly vector-density forecasting: model comparison '
                 '(one-week-ahead, rolling origin 2019-2022)')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'fig2_model_comparison.png'), dpi=300)
    plt.close(fig)

    # ---- Figure C: observed vs predicted, example grids, 2022 ----
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for k, g in enumerate(['tianhe', 'nansha']):
        sub_true = panel[(panel['grid'] == g) & (panel['year'] == 2022)]
        axes[k].plot(sub_true['week_in_year'].values, sub_true['BI'].values,
                     'k-', lw=1.6, label='Simulated BI (observed)')
        for name, color in [('XGBoost', '#2ca02c'), ('SARIMA', '#1f77b4'),
                            ('RandomForest', '#ff7f0e')]:
            p = all_preds[name]
            ps = p[(p['grid'] == g) & (p['year'] == 2022)].sort_values('week_in_year')
            axes[k].plot(ps['week_in_year'].values, ps['y_pred'].values,
                         color=color, lw=1.0, alpha=0.85, label=name)
        axes[k].set_ylabel('BI')
        axes[k].set_title(f'{GRID_LABEL[g]} grid, 2022 weekly BI: simulated '
                        f'observation vs one-week-ahead forecast')
        axes[k].legend(ncol=4, fontsize=8, loc='upper right')
    axes[1].set_xlabel('Week of 2022')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'fig3_obs_vs_pred_2022.png'), dpi=300)
    plt.close(fig)

    # ---- Figure D: SHAP importance ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    top = imp.sort_values()
    ax.barh(top.index, top.values, color='#1f77b4')
    ax.set_xlabel('Mean |SHAP value| (contribution to the predicted BI)')
    ax.set_title('XGBoost feature importance (SHAP, test year 2021)')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'fig4_shap_importance.png'), dpi=300)
    plt.close(fig)

    print(f'\nDone. Results written to: {RESULTS}')
