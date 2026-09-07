from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_core.stackelberg_mpc import (
    BUS_CONFIG, BUS_ORDER, DT, HORIZON, PHASE4_SETTINGS,
    _solve_stackelberg_window,
)

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
PHASE4 = PROJECT_ROOT / "stackelberg_scenario_analysis" / "phase4"
if str(PHASE4) not in sys.path:
    sys.path.insert(0, str(PHASE4))

from p4_common import day_inputs, load_simbench_profiles, read_pa_year, read_pv_source

START_DATE = pd.Timestamp("2025-01-15")
SIM_STEPS = 2 * 96
BASE_SEED = 20250903
SIGMA_EPSILON_KW = 0.3
AR1_RHO = 0.9
PV_ACTIVE_THRESHOLD_KW = 0.01

EXPERIMENTS = {
    "t5_perfect": ("perfect", ROOT / "t5_perfect_mpc_mpc" / "results_online_2days"),
    "t5": ("iid", ROOT / "t5_iid_mpc_mpc" / "results_online_2days"),
    "t6": ("ar1", ROOT / "t6_ar1_mpc_mpc" / "results_online_2days"),
}


def _seed(*parts) -> int:
    token = "|".join(map(str, (BASE_SEED,) + parts)).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")


def _load_period():
    settings = deepcopy(PHASE4_SETTINGS)
    settings["LOAD_MODE"] = "annual"
    settings["SOLVER_OUTPUT"] = 1
    pa_year, _ = read_pa_year(settings)
    pv_source = read_pv_source(settings)
    profiles, full_days = load_simbench_profiles(settings)
    parts = []
    for offset in range(3):
        date = START_DATE + pd.Timedelta(days=offset)
        loads, pv, pa = day_inputs(date, profiles, full_days, pa_year, pv_source, settings)
        parts.append((date, loads, pv, pa))
    realization = {v: {b: np.concatenate([p[1 if v == "load" else 2][b] for p in parts])
                       for b in BUS_ORDER} for v in ("load", "pv")}
    # Apply the same explicit PV activity convention to the simulated meter data.
    for bus in BUS_ORDER:
        realization["pv"][bus][realization["pv"][bus] < PV_ACTIVE_THRESHOLD_KW] = 0.0
    pa = np.concatenate([p[3] for p in parts])
    timestamps = pd.date_range(START_DATE, periods=len(pa), freq="15min")
    return settings, realization, pa, timestamps


def _forecast(realization, scenario, origin, horizon=HORIZON):
    forecast = {v: {} for v in ("load", "pv")}
    for variable in forecast:
        for bus in BUS_ORDER:
            actual = realization[variable][bus][origin:origin + horizon].copy()
            if scenario == "perfect":
                error = np.zeros(horizon)
            else:
                rng = np.random.default_rng(_seed(scenario, origin, bus, variable))
                if scenario == "iid":
                    error = rng.normal(0.0, SIGMA_EPSILON_KW, horizon)
                else:
                    error = np.zeros(horizon)
                    if horizon > 1:
                        error[1] = rng.normal(0.0, SIGMA_EPSILON_KW / np.sqrt(1.0 - AR1_RHO**2))
                    for lead in range(2, horizon):
                        error[lead] = AR1_RHO * error[lead - 1] + rng.normal(0.0, SIGMA_EPSILON_KW)
                error[0] = 0.0  # current measurement is known exactly
            values = actual + error
            if variable == "pv":
                active = actual >= PV_ACTIVE_THRESHOLD_KW
                capacity = float(PHASE4_SETTINGS["PV_CAPACITY_KW"][bus])
                values = np.where(active, np.clip(values, 0.0, capacity), 0.0)
            else:
                values = np.maximum(values, 0.0)
            forecast[variable][bus] = values
    return forecast


def _metrics(name, frame):
    rows = []
    groups = [("full_48h", frame)] + [(str(d.date()), g) for d, g in frame.groupby(frame.timestamp.dt.normalize())]
    for period, data in groups:
        net = data.realized_net_kw.to_numpy()
        base = data.uncontrolled_net_kw.to_numpy()
        rows.append({"experiment": name, "period": period,
                     "peak_kw": net.max(), "valley_kw": net.min(), "peak_valley_kw": np.ptp(net),
                     "uncontrolled_peak_valley_kw": np.ptp(base),
                     "max_abs_ramp_kw": np.max(np.abs(np.diff(net))),
                     "import_energy_kwh": np.maximum(net, 0).sum() * DT,
                     "export_energy_kwh": np.maximum(-net, 0).sum() * DT,
                     "bess_throughput_kwh": np.abs(data.bess_kw).sum() * DT})
    return pd.DataFrame(rows)


def _plot(name, frame, plans, output):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    x = frame.step.to_numpy() * DT
    ax = axes[0, 0]
    ax.plot(x, frame.uncontrolled_net_kw, color="#7A7A7A", lw=1.0, label="Uncontrolled")
    ax.plot(x, frame.realized_net_kw, color="#0072B2", lw=1.1, label="Controlled realization")
    ax.set_title("Grid exchange")
    ax.legend(frameon=False)
    ax = axes[0, 1]
    one = plans[plans.lead == 1].groupby("target_step", as_index=False).agg(
        predicted=("forecast_baseline_kw", "sum"), realized=("actual_baseline_kw", "sum"))
    ax.plot(one.target_step * DT, one.realized, color="#0072B2", lw=1.0, label="Realization")
    ax.plot(one.target_step * DT, one.predicted, color="#E69F00", lw=1.0, ls="--", label="Updated prediction")
    ax.set_title("One-step-ahead prediction")
    ax.legend(frameon=False)
    axes[1, 0].plot(x, frame.bess_kw, color="#009E73", lw=1.0)
    axes[1, 0].axhline(0, color="0.4", lw=0.6)
    axes[1, 0].set_title("Aggregate BESS action")
    axes[1, 1].step(x, frame.pb_eur_per_kwh, where="post", color="#CC79A7", lw=1.1)
    axes[1, 1].set_title("Leader tariff PB")
    for ax in axes.flat:
        ax.set_xlabel("Time (h)")
        ax.grid(True, ls="--", alpha=0.3, lw=0.5)
        ax.set_xlim(0, 48)
    axes[0, 0].set_ylabel("Power (kW)")
    axes[0, 1].set_ylabel("Net load (kW)")
    axes[1, 0].set_ylabel("Power (kW)")
    axes[1, 1].set_ylabel("PB (EUR/kWh)")
    fig.suptitle(f"{name}: 48 h strict rolling Stackelberg MPC", fontsize=13)
    fig.tight_layout()
    fig.savefig(output / "stackelberg_mpc_overview_2days.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(name):
    scenario, output = EXPERIMENTS[name]
    output.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] Loading continuous SimBench/price/PV data...", flush=True)
    _, actual, pa, timestamps = _load_period()
    energy = {b: float(BUS_CONFIG[b]["bess_e_init"]) for b in BUS_ORDER}
    previous_pb = None
    warm_pb = None
    series_rows, plan_rows, solver_rows = [], [], []
    partial_series = output / "mpc_timeseries.partial.csv"
    partial_solver = output / "solver_log.partial.csv"
    partial_plans = output / "rolling_plans.partial.csv"
    start_origin = 0
    if partial_series.exists():
        old_series = pd.read_csv(partial_series, parse_dates=["timestamp"])
        if not old_series.empty:
            expected = np.arange(len(old_series))
            if len(old_series) > SIM_STEPS or not np.array_equal(old_series["step"].to_numpy(), expected):
                raise RuntimeError("Partial time series is not a contiguous prefix of this experiment.")
            series_rows = old_series.to_dict("records")
            start_origin = len(series_rows)
            last = old_series.iloc[-1]
            energy = {b: float(last[f"soc_{b.replace(' ', '_').lower()}_kwh"]) for b in BUS_ORDER}
            previous_pb = float(last["pb_eur_per_kwh"])
            if partial_solver.exists():
                solver_rows = pd.read_csv(partial_solver, parse_dates=["timestamp"]).to_dict("records")
            if partial_plans.exists():
                plan_rows = pd.read_csv(
                    partial_plans, parse_dates=["origin_timestamp", "target_timestamp"]
                ).to_dict("records")
            else:
                # The older checkpoint format did not persist rolling plans.
                # Recreate its deterministic forecasts; unavailable optimized
                # plan entries remain explicit NaN values.
                for old_origin in range(start_origin):
                    old_forecast = _forecast(actual, scenario, old_origin)
                    for lead in range(HORIZON):
                        for bus in BUS_ORDER:
                            plan_rows.append({"origin_step": old_origin, "target_step": old_origin + lead,
                                              "lead": lead, "origin_timestamp": timestamps[old_origin],
                                              "target_timestamp": timestamps[old_origin + lead], "bus": bus,
                                              "forecast_load_kw": old_forecast["load"][bus][lead],
                                              "forecast_pv_kw": old_forecast["pv"][bus][lead],
                                              "forecast_baseline_kw": old_forecast["load"][bus][lead] - old_forecast["pv"][bus][lead],
                                              "actual_baseline_kw": actual["load"][bus][old_origin + lead] - actual["pv"][bus][old_origin + lead],
                                              "planned_bess_kw": np.nan, "planned_net_kw": np.nan,
                                              "planned_pb_eur_per_kwh": np.nan})
            print(f"[{name}] Resuming from MPC step {start_origin + 1}/{SIM_STEPS}; "
                  f"restored {start_origin} completed steps.", flush=True)
    started = time.time()
    for origin in range(start_origin, SIM_STEPS):
        forecast = _forecast(actual, scenario, origin)
        # The first two updates receive extra time, but a 180 s ceiling avoids
        # spending many minutes proving a weak KKT-MILP bound after the
        # incumbent has stopped improving. Later updates retain the 600 s cap.
        limit = 180.0 if origin < 2 else 600.0
        print(f"\n[{name}] MPC {origin + 1:03d}/{SIM_STEPS} | {timestamps[origin]} | limit={limit:.0f}s", flush=True)
        solution = _solve_stackelberg_window(
            timestamps[origin], origin, forecast, energy, pa[origin:origin + HORIZON],
            previous_pb, None, warm_pb=warm_pb, mip_gap=0.05, time_limit_s=limit)
        for lead in range(HORIZON):
            for bus in BUS_ORDER:
                plan_rows.append({"origin_step": origin, "target_step": origin + lead, "lead": lead,
                                  "origin_timestamp": timestamps[origin], "target_timestamp": timestamps[origin + lead],
                                  "bus": bus,
                                  "forecast_load_kw": forecast["load"][bus][lead],
                                  "forecast_pv_kw": forecast["pv"][bus][lead],
                                  "forecast_baseline_kw": forecast["load"][bus][lead] - forecast["pv"][bus][lead],
                                  "actual_baseline_kw": actual["load"][bus][origin + lead] - actual["pv"][bus][origin + lead],
                                  "planned_bess_kw": solution["bess_plans"][bus][lead],
                                  "planned_net_kw": solution["net_plans"][bus][lead],
                                  "planned_pb_eur_per_kwh": solution["pb"][lead]})
        bess = sum(solution["actions"].values())
        uncontrolled = sum(actual["load"][b][origin] - actual["pv"][b][origin] for b in BUS_ORDER)
        realized = uncontrolled + bess
        for bus in BUS_ORDER:
            energy[bus] += DT * solution["actions"][bus]
        series_rows.append({"step": origin, "timestamp": timestamps[origin], "uncontrolled_net_kw": uncontrolled,
                            "realized_net_kw": realized, "bess_kw": bess, "pb_eur_per_kwh": solution["pb"][0],
                            **{f"soc_{b.replace(' ', '_').lower()}_kwh": energy[b] for b in BUS_ORDER}})
        solver_rows.append({"step": origin, "timestamp": timestamps[origin],
                            **{k: solution[k] for k in ("status", "objective", "best_bound", "gap", "runtime_s", "nodes")}})
        previous_pb = float(solution["pb"][0])
        warm_pb = np.r_[solution["pb"][1:], solution["pb"][-1]]
        print(f"[{name}] executed: net={realized:.3f} kW, PB={previous_pb:.4f}, "
              f"gap={100*solution['gap']:.2f}%, runtime={solution['runtime_s']:.1f}s", flush=True)
        pd.DataFrame(series_rows).to_csv(partial_series, index=False)
        pd.DataFrame(solver_rows).to_csv(partial_solver, index=False)
        pd.DataFrame(plan_rows).to_csv(partial_plans, index=False)
    frame = pd.DataFrame(series_rows)
    plans = pd.DataFrame(plan_rows)
    frame.to_csv(output / "mpc_timeseries.csv", index=False)
    plans.to_csv(output / "rolling_plans.csv", index=False)
    pd.DataFrame(solver_rows).to_csv(output / "solver_log.csv", index=False)
    _metrics(name, frame).to_csv(output / "period_metrics.csv", index=False)
    _plot(name, frame, plans, output)
    print(f"[{name}] Complete in {(time.time()-started)/60:.1f} min -> {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["all", *EXPERIMENTS], default="all")
    args = parser.parse_args()
    for name in EXPERIMENTS if args.experiment == "all" else (args.experiment,):
        run(name)


if __name__ == "__main__":
    main()
