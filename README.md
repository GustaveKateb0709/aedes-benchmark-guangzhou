# Aedes albopictus benchmark, Guangzhou: forecasting and control under a leakage-controlled evaluation

Code and data for two companion studies on weekly *Aedes albopictus* immature
density (Breteau index, BI) across ten grids of Guangzhou, China (2015-2022),
both driven by the same ERA5 weather panel and a literature-parameterised
mechanistic BI simulator.

## Papers

| Folder | Paper | Target journal |
|---|---|---|
| `paper_a_forecasting/` | *Climate-driven forecasting of Aedes albopictus density on a known-mechanism simulator: seasonal, machine-learning and graph models under a leakage-controlled benchmark in Guangzhou, China* | Acta Tropica |
| `paper_b_rl/` | *Proactive mosquito control for chikungunya prevention: a deep reinforcement learning policy trained on a weather-driven simulator and evaluated on a held-out year* | PLOS Neglected Tropical Diseases |

## Repository layout

```
paper_a_forecasting/
  pipeline.py        panel construction + five baselines + DM tests + PFI/SHAP
  dm_panelrobust.py  panel- and cluster-robust Diebold-Mariano tests
  gnn/               GAT, graph sensitivity, leave-one-grid-out, DML audit, figures
  data/              ERA5 weekly weather per grid (JSON)
  results/           all CSV outputs behind every table and figure
paper_b_rl/
  code/              MosquitoEnv (Gymnasium), PPO training, rule baselines,
                     paired evaluation, sensitivity, tuned-rule fairness
                     analysis (tune_rules.py), PPO episode-return measurement
                     (ppo_return.py), figures
  results/           all CSV outputs (incl. rule_tuning_train, rule_tuned_2022,
                     ppo_return_2022)
  requirements.txt
```

## Quick start

```bash
# Paper A (forecasting benchmark)
cd paper_a_forecasting
python pipeline.py            # panel + baselines + leakage-controlled benchmark
python dm_panelrobust.py      # robust DM tests
cd gnn
python run_gat.py             # GAT + graph sensitivity + LOO
python run_dml.py             # double-machine-learning audit
python loo_baselines.py       # leave-one-grid-out baselines
python make_figures.py        # Figure 1-7

# Paper B (reinforcement learning control)
cd ../paper_b_rl
pip install -r requirements.txt
cd code
python evaluate.py --reps 10  # main table + trajectories (uses shipped models)
python tune_rules.py          # tuned-rule fairness analysis (Table 5)
python ppo_return.py          # PPO 2022 episode return under the same protocol
```

All scripts resolve paths relative to their own location; no configuration
is required. Random seeds are fixed (SEED=2024 for forecasting; PPO seeds
0/1/2 with noise seeds 0-9 for the control study).

## Trained models and figures

The trained PPO policy checkpoints (`models/`, ~7 MB of `.zip`/`.pkl`) and
the rendered publication figures are distributed through a Zenodo archive
(DOI will appear here upon publication) rather than this git repository.
All figures can be regenerated from the CSVs in `results/` with
`make_figures.py`.

## Data sources

- **ERA5 weather**: Copernicus Climate Change Service (C3S), weekly
  aggregation per grid (2015-2022), included in
  `paper_a_forecasting/data/` and `paper_b_rl/results/panel_data.csv`.
- **Breteau index**: simulated by the literature-parameterised mechanism
  model described in each manuscript (no identifiable surveillance microdata
  are redistributed).

## License

MIT (see `LICENSE`). Python 3.11+.
