#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulator v2 for the Aedes albopictus BI benchmark.

Three scenarios share one parameter set:

  M  memoryless   - the original structure (BI at week t depends on week-t
                    weather only), with the temperature optimum corrected and
                    log-normal observation noise.
  L  lagged       - immature development is spread over the preceding three
                    weeks and breeding-site water carries over between weeks,
                    so BI at week t depends on weather at t, t-1, t-2.
  S  spatial      - scenario L plus adult dispersal from the k=3 nearest
                    neighbours, which gives message passing something to learn.

Parameter provenance
--------------------
T_DEV0 = 10.4 C     Delatte et al. 2009 (J Med Entomol 46:33-41), minimal
                    developmental threshold for Ae. albopictus.
DEV_A              Briere coefficient, unchanged from v1 so that the
                    development curve keeps its published shape.
T_OPT_ADULT = 24.5  Doeurk et al. 2025 (Parasit Vectors 18:362), fitted optimum
                    of female adult cumulative survival for Ae. albopictus.
                    (v1 used 27.0, which is the Ae. aegypti value in the same
                    paper - see response to Reviewer 3, comment 12.)
SURV_MAX, SURV_K    unchanged from v1; the Gaussian width is a modelling
                    assumption and is tested in the structural sensitivity.
Humidity term       retained but flagged as an assumption: no source in the
                    reference set supplies a humidity response for adult
                    survival (AedesTraits explicitly excludes humidity).
Rainfall gain       retained as a phenomenological saturating gain, flagged as
                    an assumption, with three alternative forms tested in the
                    structural sensitivity.

Observation noise
-----------------
v1 drew additive N(0, 1.5^2) and clipped at zero, which biases low-density
weeks upward (Reviewer 3). v2 uses multiplicative log-normal noise with
sigma_log = 0.15, which is positive by construction and has no boundary bias.
"""
import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("AEGIS_DATA", "/tmp/at_work/repo/data")

GRIDS = ["conghua", "huadu", "zengcheng", "baiyun", "tianhe",
         "yuexiu", "haizhu", "panyu", "nansha", "huangpu"]
GRID_LAT = {"conghua": 23.55, "huadu": 23.40, "zengcheng": 23.29, "baiyun": 23.26,
            "tianhe": 23.13, "yuexiu": 23.13, "haizhu": 23.08, "panyu": 22.94,
            "nansha": 22.60, "huangpu": 23.10}
GRID_LON = {"conghua": 113.59, "huadu": 113.22, "zengcheng": 113.81, "baiyun": 113.27,
            "tianhe": 113.36, "yuexiu": 113.27, "haizhu": 113.32, "panyu": 113.38,
            "nansha": 113.53, "huangpu": 113.46}

# ---------------------------------------------------------------- parameters
T_DEV0 = 10.4
T_MAX = 39.0
DEV_A = 1.13e-4
T_OPT_ADULT = 24.5
SURV_MAX = 0.95
SURV_K = 0.004
RAIN_THRESH = 5.0
RAIN_GAIN = 2.5
K_SCALE = 28.0
SIGMA_LOG = 0.15

LAG_W = (0.50, 0.30, 0.20)   # development kernel over weeks t, t-1, t-2
RHO_HAB = 0.60               # weekly carry-over of breeding-site water


# ---------------------------------------------------------------- mechanism
def briere(T):
    T = np.asarray(T, float)
    out = np.zeros_like(T)
    m = (T > T_DEV0) & (T < T_MAX)
    out[m] = DEV_A * T[m] * (T[m] - T_DEV0) * np.sqrt(np.maximum(T_MAX - T[m], 0.0))
    return out


def surv_adult(T):
    return SURV_MAX * np.exp(-SURV_K * (np.asarray(T, float) - T_OPT_ADULT) ** 2)


def hum_mod(h):
    return np.clip(0.5 + 0.5 * np.clip(np.asarray(h, float) / 70.0, 0, 1.6), 0, 1)


def rain_gain(x, form="saturating"):
    x = np.asarray(x, float)
    if form == "saturating":
        return 1.0 + RAIN_GAIN * (1.0 - np.exp(-np.maximum(0.0, x - RAIN_THRESH) / 15.0))
    if form == "linear":
        return 1.0 + RAIN_GAIN * np.maximum(0.0, x - RAIN_THRESH) / 60.0
    if form == "gate":                      # step: sites open above threshold
        return np.where(x > RAIN_THRESH * 2, 1.0 + RAIN_GAIN, 1.0)
    raise ValueError(form)


def _lagged_mean(x, weights):
    """Weighted mean of x over the current and preceding weeks (k = 0 is now)."""
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    for k, w in enumerate(weights):
        if k == 0:
            shifted = x
        else:
            shifted = np.concatenate([np.full(k, x[0]), x[:-k]])
        out += w * shifted
    return out


def _habitat_recursion(rain, rho=RHO_HAB):
    """Breeding-site water: fills with rain, drains at rate rho per week."""
    rain = np.asarray(rain, float)
    eff = np.maximum(0.0, rain - RAIN_THRESH)
    hab = np.zeros_like(eff)
    acc = 0.0
    for i, e in enumerate(eff):
        acc = (1.0 - rho) * acc + e
        hab[i] = acc
    return hab * (1.0 - rho)          # normalise so the steady state equals rain


def pfi(T, Rain, Hum, scenario="L", rain_form="saturating"):
    T = np.asarray(T, float)
    Rain = np.asarray(Rain, float)
    Hum = np.asarray(Hum, float)
    if scenario == "M":
        return briere(T) * surv_adult(T) * hum_mod(Hum) * rain_gain(Rain, rain_form) * K_SCALE
    dev = _lagged_mean(briere(T), LAG_W)
    hab = _habitat_recursion(Rain)
    return dev * surv_adult(T) * hum_mod(Hum) * rain_gain(hab, rain_form) * K_SCALE


# ---------------------------------------------------------------- data
def load_weekly(grid):
    with open(os.path.join(DATA, f"era5_{grid}.json")) as f:
        j = json.load(f)
    df = pd.DataFrame({
        "date": pd.to_datetime(j["daily"]["time"]),
        "T": j["daily"]["temperature_2m_mean"],
        "rain": j["daily"]["precipitation_sum"],
        "hum": j["daily"]["relative_humidity_2m_mean"],
    }).set_index("date")
    wk = df.resample("W-SUN").agg(T=("T", "mean"), Rain=("rain", "sum"), Hum=("hum", "mean"))
    wk = wk.iloc[1:-1].reset_index().rename(columns={"date": "week_end"})
    iso = wk["week_end"].dt.isocalendar()
    wk["year"] = iso["year"].values
    wk["week_in_year"] = iso["week"].values
    return wk[wk["year"].between(2015, 2022)].reset_index(drop=True)


def adjacency(k=3):
    coords = np.array([[GRID_LAT[g], GRID_LON[g]] for g in GRIDS])
    n = len(GRIDS)
    A = np.zeros((n, n))
    for i in range(n):
        d = np.hypot(coords[:, 0] - coords[i, 0], coords[:, 1] - coords[i, 1])
        order = np.argsort(d)
        for j in order[1:k + 1]:
            A[i, j] = A[j, i] = 1.0
    return A


def urban_scale(lat, jitter):
    return (np.exp(-0.5 * ((lat - 23.10) / 0.25) ** 2) * 0.6 + 0.7) * jitter


def build_panel(seed=2024, scenario="L", kappa=0.0, rain_form="saturating",
                sigma_log=SIGMA_LOG, urban_jitter=0.10,
                eta_sd=0.15, eta_phi=0.80):
    """One simulated panel: 10 grids x weeks, log-normal observation noise.

    Scenario S adds two things the other scenarios lack:
      eta_g(t)  a grid-specific AR(1) shock (local habitat, control effort,
                container composition) that no weather feature can recover;
      spill     the previous week's mean neighbour shock, scaled by kappa.
    Together they make the neighbours' *observations* informative, which is the
    only situation in which message passing can earn its keep. Without eta the
    ten grids share 0.998-correlated weather, so neighbour information is a
    duplicate of local information and no graph model can help by construction.
    """
    rng = np.random.default_rng(seed)
    raw = {g: load_weekly(g) for g in GRIDS}
    base = {g: pfi(raw[g]["T"].values, raw[g]["Rain"].values,
                   raw[g]["Hum"].values, scenario, rain_form) for g in GRIDS}

    if scenario == "S":
        # A grid-specific AR(1) shock always exists in scenario S; kappa only
        # controls how much of the neighbours' previous-week shock is imported.
        # Holding the shock fixed across kappa makes the coupling a clean
        # dose-response rather than a confound with added noise.
        A = adjacency(3)
        n_week = len(raw[GRIDS[0]])
        rng_e = np.random.default_rng(seed + 999)
        eta = np.zeros((len(GRIDS), n_week))
        for t in range(n_week):
            eps = rng_e.normal(0.0, eta_sd * np.sqrt(1 - eta_phi ** 2), len(GRIDS))
            eta[:, t] = eta_phi * eta[:, t - 1] + eps if t > 0 else eps
        nb_eta = (A @ eta) / np.maximum(A.sum(axis=1, keepdims=True), 1)
        for i, g in enumerate(GRIDS):
            spill = np.concatenate([[0.0], nb_eta[i, :-1]])   # one-week dispersal lag
            mult = np.clip(1.0 + eta[i] + kappa * spill, 0.2, 3.0)
            base[g] = base[g] * mult


    frames = []
    for g in GRIDS:
        wk = raw[g].copy()
        u = urban_scale(GRID_LAT[g], rng.uniform(1 - urban_jitter, 1 + urban_jitter))
        mean_bi = base[g] * u
        bi = mean_bi * np.exp(rng.normal(0.0, sigma_log, len(mean_bi)))
        wk["grid"] = g
        wk["PFI"] = base[g]
        wk["BI_mean"] = mean_bi
        wk["pop_density"] = round(float(u), 3)
        wk["BI"] = np.round(bi, 3)
        frames.append(wk)

    panel = pd.concat(frames, ignore_index=True)
    panel["week_idx"] = panel.groupby("grid").cumcount()
    return panel.sort_values(["grid", "week_idx"]).reset_index(drop=True)


# ---------------------------------------------------------------- diagnostics
def truth_curves(panel, tvar, grid_pts, delta):
    """Finite-difference marginal effect of a weather variable on BI_mean."""
    out = []
    for g, grp in panel.groupby("grid"):
        grp = grp.sort_values("week_idx")
        out.append(grp["BI_mean"].values.copy())
    return out


if __name__ == "__main__":
    for sc in ("M", "L", "S"):
        p = build_panel(seed=2024, scenario=sc, kappa=0.35 if sc == "S" else 0.0)
        print(f"{sc}: n={len(p)}  BI mean={p.BI.mean():.2f} sd={p.BI.std():.2f} "
              f"max={p.BI.max():.2f}  >=10: {(p.BI >= 10).mean():.3f}")
