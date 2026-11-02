"""Physical load disturbances and causal forecasts, independent of the solver.

Nominal load and PV are known exogenous profiles. Only physical LOAD is noisy.
AR(1) state continues across all windows/days; it is never a forecast-time draw.
At a clipped load observation, the measured (censored) residual is used as a
certainty-equivalent state estimate, not the hidden negative latent state.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

MODEL_VERSION = "seasonal-actual-load-causal-v2"
SEED = 20250903
SIGMA_W = 0.3  # kW innovation std, NOT equal marginal variance across scenarios
RHO = 0.9
AMPLITUDE_COMPARISON = "set by the selected experiment configuration"


def build_realization(nominal, scenario, *, seed=SEED, sigma_w=SIGMA_W, rho=RHO):
    if scenario not in ("perfect", "iid", "ar1"):
        raise ValueError(f"Unknown physical load scenario: {scenario}")
    if sigma_w < 0 or not 0 <= rho < 1:
        raise ValueError("Require sigma_w >= 0 and 0 <= rho < 1")
    if "process" in nominal:
        raise ValueError("Pass nominal profiles, not an already disturbed realization")
    actual = {v: {b: np.asarray(a, dtype=float).copy() for b, a in nominal[v].items()}
              for v in ("load", "pv")}
    actual["nominal_load"] = {b: a.copy() for b, a in actual["load"].items()}
    actual["disturbance"] = {}
    actual["innovation"] = {}
    actual["clipped"] = {}
    for bus, base in actual["nominal_load"].items():
        if len(base) == 0 or not np.isfinite(base).all() or (base < 0).any():
            raise ValueError(f"Invalid nominal load for {bus}")
        token = f"{seed}|physical-load|{bus}".encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8], "little"))
        w = rng.normal(0.0, sigma_w, len(base))
        # Common, known initial condition. After t=0 there are no resets.
        w[0] = 0.0
        d = np.zeros(len(base))
        if scenario == "iid":
            d = w.copy()
        elif scenario == "ar1":
            for k in range(1, len(base)):
                d[k] = rho * d[k - 1] + w[k]
        else:
            w[:] = 0.0
        actual["disturbance"][bus] = d
        actual["innovation"][bus] = w
        actual["clipped"][bus] = base + d < 0.0
        actual["load"][bus] = np.maximum(base + d, 0.0)
    actual["process"] = {
        "version": MODEL_VERSION, "scenario": scenario, "seed": seed,
        "sigma_innovation_kw": sigma_w, "rho": rho, "initial_disturbance_kw": 0.0,
        "pv_noise": False, "load_floor_kw": 0.0,
        "clipping_forecast": "certainty-equivalent measured residual; not exact censored conditional mean",
        "amplitude_comparison": AMPLITUDE_COMPARISON,
    }
    return actual


def causal_forecast(actual, scenario, origin, horizon, *, target_start=None):
    """Uses only the observation at origin plus known nominal future profiles.

    target_start permits precommitted OL blocks all using information at t=0.
    Changing future realizations/innovations cannot change this forecast.
    """
    meta = actual.get("process")
    if not meta or meta.get("version") != MODEL_VERSION or meta["scenario"] != scenario:
        raise ValueError("Forecast requires a matching physical-load realization; old forecast-noise data rejected")
    start = origin if target_start is None else target_start
    if origin < 0 or start < origin or horizon < 1:
        raise ValueError("Invalid causal forecast window")
    leads = np.arange(start, start + horizon) - origin
    result = {"load": {}, "pv": {}}
    for bus, base in actual["nominal_load"].items():
        if start + horizon > len(base):
            raise ValueError("Insufficient look-ahead buffer; do not shorten the horizon")
        measured_residual = float(actual["load"][bus][origin] - base[origin])
        correction = (meta["rho"] ** leads * measured_residual
                      if scenario == "ar1" else np.zeros(horizon))
        predicted = np.maximum(base[start:start + horizon] + correction, 0.0)
        if start == origin:
            predicted[0] = actual["load"][bus][origin]
        result["load"][bus] = predicted
        # PV is deterministic and declared known; no future physical noise here.
        result["pv"][bus] = actual["pv"][bus][start:start + horizon].copy()
    return result


def save_realization(actual, timestamps, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    data = {"timestamp": timestamps}
    for bus in actual["load"]:
        tag = bus.replace(" ", "_").lower()
        for key in ("nominal_load", "load", "pv", "disturbance", "innovation", "clipped"):
            data[f"{tag}_{key}"] = actual[key][bus]
    path = folder / "realization.csv"
    frame = pd.DataFrame(data)
    if path.exists():
        old = pd.read_csv(path, parse_dates=["timestamp"])
        pd.testing.assert_frame_equal(old, frame, check_dtype=False, check_exact=False,
                                      rtol=1e-12, atol=1e-12)
    else:
        frame.to_csv(path, index=False)
    meta = dict(actual["process"])
    meta["clipped_samples_by_bus"] = {b: int(a.sum()) for b, a in actual["clipped"].items()}
    (folder / "process.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def ensure_run_manifest(folder, configuration):
    """Fail closed on old/mismatched checkpoints instead of silently reusing them."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "run_manifest.json"
    expected = {"model_version": MODEL_VERSION, **configuration}
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Configuration differs from checkpoint: {path}")
    else:
        if any(folder.glob("*.csv")):
            raise ValueError(f"Unversioned output exists: {folder}. Use a new directory.")
        path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
