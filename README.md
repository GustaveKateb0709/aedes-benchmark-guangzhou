# Leakage-controlled forecasting benchmark for Aedes albopictus density, Guangzhou

Code and data for *Climate-driven forecasting of Aedes albopictus density on a
known-mechanism simulator: seasonal, machine-learning and graph models under a
leakage-controlled benchmark in Guangzhou, China*.

## What the benchmark controls for

The simulator generates the Breteau index from grid-level urbanisation scales.
An earlier feature set that included the urbanisation proxy leaked the
label-generation coefficient into the predictors and inflated pooled
machine-learning skill. All results reported here use a **leakage-controlled
14-feature set** (weather, seasonal harmonics and lags only); the
proxy-inclusive numbers are reproduced only as a quantitative leakage contrast.

## Repository layout

```
pipeline.py        panel construction, five baselines (SARIMA, Lasso, random
                   forest, XGBoost, LSTM), Diebold-Mariano tests, permutation
                   feature importance, SHAP, leakage contrast
dm_panelrobust.py  HAC / grid-cluster / grid-level DM robustness
gnn/
  run_gat.py       graph attention network with kNN graphs + graph sensitivity
  loo_baselines.py leave-one-grid-out extrapolation
  run_dml.py       double-machine-learning audit of weather effects
  make_figures.py  publication figures
data/              ERA5 weekly weather per grid (10 JSON files)
results/           every CSV behind the tables and figures
```

## Run

```bash
python pipeline.py            # ~10 min, CPU; writes results/*.csv
python dm_panelrobust.py
cd gnn
python run_gat.py             # ~2 min (PyTorch CPU)
python loo_baselines.py
python run_dml.py
python make_figures.py        # writes figures/
```

The CSV outputs correspond to the manuscript as follows:

| Output | Corresponds to |
|---|---|
| `model_results_unified.csv` | Tables 1-2 |
| `dm_tests_panelrobust.csv` | Table 2 (robustness) |
| `pipeline.py` leakage contrast | Table 3 |
| `loo_xgb_baseline.csv` / `loo_gat.csv` | Table 4 |
| `graph_sensitivity.csv` | Table 5 |
| `dml_lag_effects.csv` / `dml_dose_response.csv` | Appendix A |

Dependencies: numpy, pandas, scipy, scikit-learn, xgboost, statsmodels,
matplotlib, shap, torch. Random seed fixed at 2024.

## Data sources

- **ERA5 weather**: Copernicus Climate Change Service (C3S), weekly
  aggregation per grid (2015-2022), included in `data/`.
- **Breteau index**: simulated by the literature-parameterised mechanistic
  model described in the manuscript. No identifiable surveillance microdata
  are redistributed.

## License

MIT (see `LICENSE`). Python 3.11+.
