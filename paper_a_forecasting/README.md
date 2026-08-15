# Paper A — Leakage-controlled forecasting benchmark

Reproduces every number, table and figure in *Climate-driven forecasting of
Aedes albopictus density on a known-mechanism simulator: seasonal,
machine-learning and graph models under a leakage-controlled benchmark in
Guangzhou, China*.

## What this benchmark controls

The simulator generates the Breteau index from grid-level urbanisation
scales. An earlier feature set that included the urbanisation proxy leaked
the label-generation coefficient into the predictors and inflated pooled
ML performance. All results here use a **leakage-controlled 14-feature set**
(weather, seasonal harmonics and lags only); the proxy-inclusive numbers are
reproduced only as a quantitative leakage contrast (Table 3).

## Layout

```
pipeline.py        panel construction, five baselines (SARIMA, Lasso, RF,
                   XGBoost, LSTM), DM tests, PFI mechanism check, SHAP,
                   leakage-controlled vs proxy-inclusive contrast
dm_panelrobust.py  HAC / grid-cluster / grid-level DM robustness
gnn/
  run_gat.py       GAT with kNN graphs + graph sensitivity (identity/random)
  loo_baselines.py leave-one-grid-out extrapolation (GAT vs XGBoost)
  run_dml.py       double-machine-learning audit of weather effects
  make_figures.py  Figures 1-7
data/              ERA5 weekly weather per grid (10 JSON files)
results/           every CSV behind the tables and figures
```

## Run

```bash
python pipeline.py            # ~10 min, CPU; writes results/*.csv
python dm_panelrobust.py
cd gnn
python run_gat.py             # ~2 min (torch CPU)
python loo_baselines.py
python run_dml.py
python make_figures.py        # writes ../figures/Figure1..7 (PNG+PDF)
```

Outputs map directly onto manuscript tables: `model_results_unified.csv`
(Tables 1-2), `dm_tests_panelrobust.csv` (Table 2), the proxy contrast within
`pipeline.py` output (Table 3), `loo_xgb_baseline.csv` / `loo_gat.csv`
(Table 4), `graph_sensitivity.csv` (Table 5), `dml_lag_effects.csv` /
`dml_dose_response.csv` (Appendix A). Dependencies: numpy, pandas, scipy,
scikit-learn, xgboost, statsmodels, matplotlib, shap, torch.
