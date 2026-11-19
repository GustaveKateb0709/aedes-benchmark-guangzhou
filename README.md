# Reproducibility package — mechanistic simulation and multi-model forecasting of *Aedes* density in Guangzhou

Code and data accompanying the manuscript's simulation study: a weekly mechanistic
larval-habitat model (`sim.py`) drives synthetic Breteau-index panels for the ten
Guangzhou districts, and four model families are scored against them under three
design scenarios. This package reproduces the per-panel metrics that underlie the
manuscript tables, and regenerates Figures 1–6 and Supplementary Figure S1.

```
05_Code_and_Data/
├── sim.py                    panel generator (mechanistic core, 3 scenarios)
├── exp.py                    experiment driver → preds_*.csv + metrics_*.csv
├── stats.py                  Monte-Carlo summary → summary_mc.csv
├── figs.py                   Figures 1–6
├── figS1_lag_structure.py    Supplementary Figure S1
├── gnn_v2.py                 GAT / identity-GAT / MLP comparison (scenario S)
├── run_dml_v2.py             double machine learning for T and Rain effects
├── sensitivity.py            structural sensitivity of the ranking
├── requirements-lock.txt     pinned dependencies (all ten, incl. Pillow)
├── data/                     ten ERA5-derived district files, era5_<district>.json
├── figures/                  delivered figures, exactly as submitted
└── results/                  delivered metrics, DML tables and products
```

## 1. Environment

Python 3.11 or later, tested with the versions below.

```bash
pip install -r requirements-lock.txt
```

`requirements-lock.txt` pins all ten dependencies: numpy 2.4.6, pandas 3.0.5,
Pillow 12.3.0, scipy 1.18.0, matplotlib 3.11.1, scikit-learn 1.9.0,
xgboost 3.4.0, statsmodels 0.14.6, shap 0.52.0 and torch 2.13.0. Pillow is
imported by `figs.py` (TIFF LZW flattening) and `figS1_lag_structure.py`
(`from PIL import Image`).

Three environment variables relocate in- and outputs. All commands below write
to `repro/` so that the delivered `results/` stays untouched:

| Variable     | Default      | Used by                                        |
|--------------|--------------|------------------------------------------------|
| `AEGIS_DATA` | `./data`     | `sim.py` (ERA5 JSON lookup)                    |
| `AEGIS_OUT`  | `./results`  | `exp.py`, `stats.py`, `figs.py`, `gnn_v2.py`    |
| `AEGIS_FIG`  | `./figures`  | `figs.py`, `figS1_lag_structure.py`             |

## 2. Reproducing the delivered metrics

The delivered `results/metrics_*.csv` were **not** produced with the script
defaults. `exp.py` defaults to `--arms F0,F1,F2,F3 --seeds 2024 --scenarios L`,
which writes a 15-row `metrics_L00_2024.csv` (4 arms × 3 learners, plus SARIMA
on F0 and two oracle ceilings) under the same file name as the delivered 24-row
version. Use the explicit invocations below.

Scenario **L** (lagged): 7 arms × 3 learners plus SARIMA on F0, 30 panels
(seeds 1–29 and 2024) → 30 × `metrics_L00_<seed>.csv`, 24 data rows each.

```bash
mkdir -p repro/results
AEGIS_OUT=repro/results python exp.py --scenarios L \
  --arms F0,F1,F2,F3,F4a,F4b,F5 \
  --models Lasso,RandomForest,XGBoost,SARIMA \
  --seeds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,2024
```

Scenario **M** (memoryless): 3 arms only, Lasso/XGBoost/SARIMA, 10 panels
(seeds 1–10) → 10 × `metrics_M00_<seed>.csv`, 9 data rows each.

```bash
AEGIS_OUT=repro/results python exp.py --scenarios M \
  --arms F0,F2,F3 --models Lasso,XGBoost,SARIMA \
  --seeds 1,2,3,4,5,6,7,8,9,10
```

Each run writes a companion `preds_L00_<seed>.csv` / `preds_M00_<seed>.csv`
with per-observation predictions. Runs are resumable: a panel is skipped only
when both its `preds_*` and `metrics_*` files already exist. The delivered
`results/` holds the metrics and the scenario-L predictions, so re-running the
scenario-L command as-is skips every panel and writes nothing. Scenario M
predictions are not delivered: a run without `AEGIS_OUT` would rewrite the
delivered M metrics in place — keep the variable.

Monte-Carlo summary (mean ± SD across panels, BH-adjusted paired tests):

```bash
AEGIS_OUT=repro/results python stats.py     # → summary_mc.csv
```

## 3. Reproducing the figures

Figures 3 and 4 read the per-observation predictions, not the metrics files:

| Figure                  | Inputs in `$AEGIS_OUT`                                   |
|-------------------------|----------------------------------------------------------|
| Fig 1, Fig 2            | none (simulator only)                                     |
| Fig 3 arm comparison    | `preds_L00_*.csv`                                         |
| Fig 4 proxy dose        | `preds_L00_*.csv`                                         |
| Fig 5 graph boundary    | `metrics_S*_gat.csv` (delivered)                          |
| Fig 6 SHAP              | none (refits XGBoost on panel seed 2024)                  |
| FigS1 Lag Structure     | `dml_lag_effects_v2.csv` (delivered)                      |

**The delivered `results/` contains the scenario-L predictions
(`preds_L00_*.csv`)**, so `python figs.py` as-is draws all six figures
directly. The commands below are for regenerating predictions into a scratch
directory and drawing figures from there:

```bash
AEGIS_OUT=repro/results AEGIS_FIG=repro/figures python figs.py --which 3
AEGIS_OUT=repro/results AEGIS_FIG=repro/figures python figs.py --which 4
AEGIS_OUT=repro/results AEGIS_FIG=repro/figures python figs.py --which 1,2,5,6
```

Each figure is written as `.pdf`, `.png` (200 dpi preview) and `.tif`
(1000 dpi, LZW, flattened to RGB — the submission file).

The `figures/` directory holds the figure files delivered with the manuscript
(Figs 1–6 from `figs.py`, FigS1 from `figS1_lag_structure.py`). Since
`AEGIS_FIG` defaults to `./figures`, regenerating a figure without the
variable would overwrite the delivered file — keep `AEGIS_FIG=repro/figures`
when drawing into a scratch directory.

## 4. Double machine learning, graph model, structural sensitivity

```bash
python run_dml_v2.py --seeds 1,2,3,4,5,6,7,8,9,10 --lags 0,1,2,3,4,5,6 --tag v2
AEGIS_OUT=repro/results python gnn_v2.py --scenarios S --seeds 2024 --kappa 0.0
AEGIS_OUT=repro/results python gnn_v2.py --scenarios S --seeds 2024 --kappa 0.6
AEGIS_OUT=repro/results python gnn_v2.py --scenarios S --seeds 2024 --kappa 1.5
python sensitivity.py
AEGIS_OUT=repro/results AEGIS_FIG=repro/figures python figS1_lag_structure.py
```

`--tag v2` reproduces the delivered `dml_*_v2.csv`; the default tag is `main`.
The three `--kappa` values reproduce `metrics_S00_2024_gat.csv`,
`metrics_S60_2024_gat.csv` and `metrics_S150_2024_gat.csv`. Note that
`run_dml_v2.py` (line 81) and `sensitivity.py` (line 10) hard-code their output
directory to `./results` and ignore `AEGIS_OUT`, so these two commands write
into the delivered `results/` folder.

## 5. Known limitations

- `stats.py` exposes only the Monte-Carlo summary on the command line. The
  functions `paired_tests()` and `delong_test()` are importable but have no CLI
  entry point, so their outputs must be produced interactively.
- `results/calibration_v2.csv` and `results/dml_truth_v2.csv` ship as finished
  products; no script in this package generates them.
- `results/summary_mc.csv` is not part of the delivery. Run `stats.py` (§2) to
  produce it.
- `figs.py` stops at the first failing figure; pass `--which` one figure at a
  time when debugging.
