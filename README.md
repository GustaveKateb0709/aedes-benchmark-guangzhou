# Information sets outweigh algorithm choice in a simulation benchmark for *Aedes albopictus* density forecasting

Code and data for the manuscript submitted to *Acta Tropica* (ACTROP-D-26-01728).

## What the benchmark does

The repository simulates weekly Breteau index (BI) labels for ten Guangzhou grid cells from
ERA5 weather, using a mechanistic model parameterised from laboratory studies of
*Aedes albopictus*. Because the data-generating process is known, the value of any feature can
be measured rather than assumed, and the accuracy ceiling attainable from a given information
set can be computed exactly.

Six information sets are compared under one rolling-origin protocol, holding the target, the
grids, the test weeks and the tuning rules fixed:

| Arm | Features | What it tests |
|---|---|---|
| F0 | 14 weather and season features | the usual machine-learning setup |
| F1 | F0 + BI at weeks *t* and *t*−1 | information a seasonal model gets for free |
| F2 | F0 + nine grid dummies | legitimate spatial identity |
| F3 | F0 + the urbanisation proxy | the feature that motivated the study |
| F4a, F4b | F3 with 10% and 30% measurement error | how much of the proxy's value depends on being exact |
| F5 | F1 + F2 | everything a deployed system could observe |

## Three simulator scenarios

- **M** — memoryless: BI in week *t* depends on week-*t* weather only.
- **L** — lagged (primary): a three-week development kernel and week-to-week carry-over of
  breeding-site water, so BI depends on weather at *t*, *t*−1 and *t*−2.
- **S** — lagged plus spatial coupling: a grid-specific AR(1) shock and one-week dispersal from
  the three nearest neighbours, with the coupling strength set by κ.

## Repository layout

```
sim.py                 simulator: three scenarios, literature-anchored parameters
exp.py                 information-arm experiment driver (six arms x four models x 30 panels)
stats.py               DM / CR1 / CR3 / exact sign-flip / DeLong / Benjamini-Hochberg
gnn_v2.py              graph attention network, identity-graph ablation, MLP control, kappa sweep
run_dml_v2.py          double machine learning audit with finite-difference ground truth
sensitivity.py         structural sensitivity: rainfall functional form and humidity term
figs.py                every figure in the manuscript
requirements-lock.txt  exact package versions and platform
data/                  weekly ERA5 aggregates for the ten grids, 2015-2022
results/               per-panel metrics, DML output, sensitivity and calibration tables
```

## Running it

```bash
pip install -r requirements-lock.txt

python exp.py --scenarios L --seeds 1,2,3 --arms F0,F1,F2,F3 --models Lasso,XGBoost,SARIMA
python gnn_v2.py --scenarios S --seeds 1 --kappa 0.6
python run_dml_v2.py
python sensitivity.py
python figs.py --which 1,2,3,4,5,6
```

A single panel takes about four minutes on one CPU core with the full arm set, most of it in
the per-grid SARIMA fits. Thirty panels were run for the manuscript.

Figures 1 and 2 are produced from `data/` and `results/` alone. Figures 3 to 5 also read the
per-observation predictions that `exp.py` writes; figure 6 additionally needs `shap`.

## What is in `results/`

| File | Contents |
|---|---|
| `metrics_L00_*.csv` | per-panel metrics for every arm and model, scenario L, 30 panels |
| `metrics_M00_*.csv` | the same for the memoryless scenario, 10 panels |
| `metrics_S*_gat.csv` | graph models by coupling strength |
| `dml_truth_v2.csv` | finite-difference ground truth for lags 0 to 6 |
| `dml_lag_effects_v2.csv` | DML estimates, standard errors and coverage flags, 10 panels |
| `dml_dose_response_v2.csv` | spline dose-response curves |
| `dml_summary_v2.csv` | per-lag summary across panels |
| `sensitivity_structure.csv` | accuracy under three rainfall forms with humidity on and off |
| `calibration_v2.csv` | logistic recalibration slope and intercept of the BI ≥ 10 alert |

Per-observation predictions are not stored; running `exp.py` regenerates them.

## Data sources

- **ERA5 weather** — Copernicus Climate Change Service, retrieved through the Open-Meteo
  historical archive and aggregated to weeks ending Sunday. Included in `data/`.
- **Breteau index** — simulated by the mechanistic model described in the manuscript. No
  surveillance microdata are redistributed.

## Licence

MIT (see `LICENSE`).
