from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
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
from experiment_core.load_process import (
    build_realization, causal_forecast, save_realization, ensure_run_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
SOFT_SOC_ROOT = ROOT / "soft_soc_simulations"
PROJECT_ROOT = ROOT.parent
PHASE4 = PROJECT_ROOT / "stackelberg_scenario_analysis" / "phase4"
if str(PHASE4) not in sys.path:
    sys.path.insert(0, str(PHASE4))

from p4_common import day_inputs, load_simbench_profiles, read_pa_year, read_pv_source

START_DATE = pd.Timestamp("2025-01-15")
SIM_STEPS = 3 * 96
BASE_SEED = 20250903
AR1_RHO = 0.9
PHYSICAL_SIGMA_KW = 0.3
PV_ACTIVE_THRESHOLD_KW = 0.01
MPC_MIP_GAP = 0.05
MPC_TIME_LIMIT_S = 600.0
MPC_INITIAL_TIME_LIMIT_S = 180.0
MPC_SECOND_TIME_LIMIT_S = 120.0
MPC_RETRY_TIME_LIMIT_S = None
PROGRESS_HOOK = None

EXPERIMENTS = {
    "t5_perfect": ("perfect", SOFT_SOC_ROOT / "t56_3days" / "t5_perfect"),
    "t5": ("iid", SOFT_SOC_ROOT / "t56_3days" / "t5_iid"),
    "t6": ("ar1", SOFT_SOC_ROOT / "t56_3days" / "t6_ar1"),
}


def _seed(*parts) -> int:
    token = "|".join(map(str, (BASE_SEED,) + parts)).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")


def _load_period(sim_steps=None, forecast_horizon=None, scenario="perfect"):
    sim_steps = SIM_STEPS if sim_steps is None else sim_steps
    forecast_horizon = HORIZON if forecast_horizon is None else forecast_horizon
    settings = deepcopy(PHASE4_SETTINGS)
    settings["LOAD_MODE"] = "annual"
    settings["SOLVER_OUTPUT"] = 1
    pa_year, _ = read_pa_year(settings)
    pv_source = read_pv_source(settings)
    profiles, full_days = load_simbench_profiles(settings)
    parts = []
    # Three days are executed. One additional calendar day is loaded so that
    # the final updates retain the full 16-step (4 h) prediction horizon.
    planning_days = (sim_steps + forecast_horizon - 1 + 95) // 96
    for offset in range(planning_days):
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
    return settings, build_realization(
        realization,
        scenario,
        seed=BASE_SEED,
        sigma_w=PHYSICAL_SIGMA_KW,
        rho=AR1_RHO,
    ), pa, timestamps


def _forecast(realization, scenario, origin, horizon=None):
    if horizon is None:
        horizon = HORIZON
    return causal_forecast(realization, scenario, origin, horizon)


def _metrics(name, frame):
    rows = []
    groups = [(f"full_{SIM_STEPS * DT:g}h", frame)] + [
        (str(d.date()), g) for d, g in frame.groupby(frame.timestamp.dt.normalize())
    ]
    for period, data in groups:
        net = data.realized_net_kw.to_numpy()
        base = data.uncontrolled_net_kw.to_numpy()
        rows.append({"experiment": name, "period": period,
                     "peak_kw": net.max(), "valley_kw": net.min(), "peak_valley_kw": np.ptp(net),
                     "uncontrolled_peak_valley_kw": np.ptp(base),
                     "max_abs_ramp_kw": float(np.max(np.abs(np.diff(net)))) if len(net) > 1 else 0.0,
                     "import_energy_kwh": np.maximum(net, 0).sum() * DT,
                     "export_energy_kwh": np.maximum(-net, 0).sum() * DT,
                     "bess_throughput_kwh": np.abs(data.bess_kw).sum() * DT,
                     "pv_waste_energy_kwh": data.get("pv_waste_kw", pd.Series(0.0, index=data.index)).sum() * DT,
                     "pv_waste_steps": int((data.get("pv_waste_kw", pd.Series(0.0, index=data.index)) > 1e-9).sum())})
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
        ax.set_xlim(0, SIM_STEPS * DT)
    axes[0, 0].set_ylabel("Power (kW)")
    axes[0, 1].set_ylabel("Net load (kW)")
    axes[1, 0].set_ylabel("Power (kW)")
    axes[1, 1].set_ylabel("PB (EUR/kWh)")
    fig.suptitle(f"{name}: {SIM_STEPS * DT:g} h rolling Stackelberg MPC", fontsize=13)
    fig.tight_layout()
    fig.savefig(output / "stackelberg_mpc_overview_3days.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(name):
    scenario, output = EXPERIMENTS[name]
    output.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] Loading continuous SimBench/price/PV data...", flush=True)
    _, actual, pa, timestamps = _load_period(scenario=scenario)
    ensure_run_manifest(output, {
        "method": name, "scenario": scenario, "steps": SIM_STEPS, "horizon": HORIZON,
        "c_deg": float(PHASE4_SETTINGS["BESS_DEGRADATION_EUR_PER_KWH"]),
        "c_ramp": float(PHASE4_SETTINGS.get("LEADER_RAMP_WEIGHT", 0.0)),
        "mip_gap": MPC_MIP_GAP, "process": actual["process"],
        "solver_settings": json.loads(json.dumps(PHASE4_SETTINGS, default=str)),
    })
    save_realization(actual, timestamps, output)
    final_path = output / "mpc_timeseries.csv"
    if final_path.exists():
        completed = pd.read_csv(final_path)
        if len(completed) == SIM_STEPS and completed.step.tolist() == list(range(SIM_STEPS)):
            print(f"[{name}] Already complete: {SIM_STEPS}/{SIM_STEPS}", flush=True)
            return
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
                saved_plans = pd.read_csv(
                    partial_plans, parse_dates=["origin_timestamp", "target_timestamp"]
                )
                plan_rows = saved_plans.to_dict("records")
                # Continue with the shifted tariff trajectory of the last solved
                # MPC window instead of restarting the MILP from a flat guess.
                last_origin = int(saved_plans["origin_step"].max())
                last_pb = (
                    saved_plans.loc[saved_plans["origin_step"] == last_origin]
                    .groupby("lead")["planned_pb_eur_per_kwh"]
                    .first().sort_index().to_numpy(dtype=float)
                )
                if len(last_pb) and np.isfinite(last_pb).all():
                    warm_pb = np.r_[last_pb[1:], last_pb[-1]][:HORIZON]
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
        # Per-run time limits are supplied by the versioned experiment runner.
        limit = MPC_INITIAL_TIME_LIMIT_S if origin < 2 else MPC_TIME_LIMIT_S
        print(f"\n[{name}] MPC {origin + 1:03d}/{SIM_STEPS} | {timestamps[origin]} | limit={limit:.0f}s", flush=True)
        previous_realized_net = (
            float(series_rows[-1]["realized_net_kw"])
            if series_rows and float(PHASE4_SETTINGS.get("LEADER_RAMP_WEIGHT", 0.0)) > 0.0
            else None
        )
        attempt_limits = []
        attempt_runtimes = []

        def solve_with_limit(time_limit):
            attempt_limits.append(float(time_limit))
            result = _solve_stackelberg_window(
                timestamps[origin], origin, forecast, energy, pa[origin:origin + HORIZON],
                previous_pb, None, warm_pb=warm_pb, mip_gap=MPC_MIP_GAP,
                time_limit_s=float(time_limit), previous_realized_net=previous_realized_net)
            attempt_runtimes.append(float(result["runtime_s"]))
            return result

        solution = None
        first_error = None
        try:
            solution = solve_with_limit(limit)
        except RuntimeError as exc:
            if "no incumbent" not in str(exc):
                raise
            first_error = exc
            attempt_runtimes.append(float(limit))

        if solution is None or solution["gap"] > float(MPC_MIP_GAP) + 1e-9:
            second_limit = float(MPC_SECOND_TIME_LIMIT_S)
            reason = "no incumbent" if solution is None else f"gap={100*solution['gap']:.2f}%"
            print(f"[{name}] {reason} after {limit:.0f}s; second solve with "
                  f"{second_limit:.0f}s", flush=True)
            try:
                second_solution = solve_with_limit(second_limit)
                solution = second_solution
            except RuntimeError as exc:
                if "no incumbent" not in str(exc):
                    raise
                attempt_runtimes.append(second_limit)
                if solution is None:
                    first_error = exc

        if solution is None:
            if MPC_RETRY_TIME_LIMIT_S is None:
                raise first_error
            retry_limit = float(MPC_RETRY_TIME_LIMIT_S)
            print(f"[{name}] no incumbent after 60s and 120s; final solve with "
                  f"{retry_limit:.0f}s", flush=True)
            solution = solve_with_limit(retry_limit)
        quality_status = (
            "gap_target_reached"
            if solution["gap"] <= float(MPC_MIP_GAP) + 1e-9
            else "feasible_above_target"
        )
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
        exchange_mode = str(PHASE4_SETTINGS.get("GRID_EXCHANGE_MODE", "nonnegative_clip"))
        uncontrolled_by_bus = {
            b: actual["load"][b][origin] - actual["pv"][b][origin] for b in BUS_ORDER
        }
        realized_by_bus = {
            b: uncontrolled_by_bus[b] + solution["actions"][b] for b in BUS_ORDER
        }
        uncontrolled_raw = sum(uncontrolled_by_bus.values())
        realized_raw = sum(realized_by_bus.values())
        if exchange_mode == "grid_export":
            uncontrolled = uncontrolled_raw
            realized = realized_raw
            public_import = sum(max(value, 0.0) for value in realized_by_bus.values())
            public_export = sum(max(-value, 0.0) for value in realized_by_bus.values())
            local_sale = 0.0
        elif exchange_mode == "local_sale":
            uncontrolled = sum(max(value, 0.0) for value in uncontrolled_by_bus.values())
            realized = sum(max(value, 0.0) for value in realized_by_bus.values())
            public_import = realized
            public_export = 0.0
            local_sale = sum(max(-value, 0.0) for value in realized_by_bus.values())
        else:
            uncontrolled = max(0.0, uncontrolled_raw)
            realized = max(0.0, realized_raw)
            public_import = realized
            public_export = 0.0
            local_sale = max(-realized_raw, 0.0)
        # Neither new formulation curtails energy: every negative local balance
        # is either public-grid export or an explicitly settled local sale.
        uncontrolled_waste = 0.0
        pv_waste = 0.0
        for bus in BUS_ORDER:
            energy[bus] += DT * solution["actions"][bus]
            cfg = BUS_CONFIG[bus]
            if not cfg["bess_e_min"] - 1e-6 <= energy[bus] <= cfg["bess_e_max"] + 1e-6:
                raise RuntimeError(f"Physical SOC violation at step {origin} for {bus}: {energy[bus]}")
        series_rows.append({"step": origin, "timestamp": timestamps[origin],
                            "uncontrolled_raw_net_kw": uncontrolled_raw,
                            "uncontrolled_net_kw": uncontrolled,
                            "uncontrolled_pv_waste_kw": uncontrolled_waste,
                            "realized_raw_net_kw": realized_raw,
                            "realized_net_kw": realized,
                            "public_grid_import_kw": public_import,
                            "public_grid_export_kw": public_export,
                            "local_sale_kw": local_sale,
                            "pv_waste_kw": pv_waste, "bess_kw": bess,
                            "pa_eur_per_kwh": pa[origin],
                            "pb_eur_per_kwh": solution["pb"][0],
                            "total_price_eur_per_kwh": pa[origin] + solution["pb"][0],
                            **{f"soc_{b.replace(' ', '_').lower()}_kwh": energy[b] for b in BUS_ORDER}})
        solver_rows.append({"step": origin, "timestamp": timestamps[origin],
                            **{k: solution[k] for k in (
                                "status", "objective", "best_bound", "gap", "runtime_s", "nodes",
                                "constraint_violation", "bound_violation", "integrality_violation",
                            )}, "solution_quality": quality_status,
                            "attempt_count": len(attempt_limits),
                            "attempt_limits_s": "+".join(f"{v:g}" for v in attempt_limits),
                            "total_solver_runtime_s": float(sum(attempt_runtimes))})
        previous_pb = float(solution["pb"][0])
        warm_pb = np.r_[solution["pb"][1:], solution["pb"][-1]]
        print(f"[{name}] executed: net={realized:.3f} kW, PB={previous_pb:.4f}, "
              f"gap={100*solution['gap']:.2f}%, runtime={solution['runtime_s']:.1f}s", flush=True)
        pd.DataFrame(series_rows).to_csv(partial_series, index=False)
        pd.DataFrame(solver_rows).to_csv(partial_solver, index=False)
        pd.DataFrame(plan_rows).to_csv(partial_plans, index=False)
        (output / "progress.json").write_text(json.dumps({
            "method": name, "completed_steps": origin + 1, "total_steps": SIM_STEPS,
            "timestamp": str(timestamps[origin]), "updated_at": pd.Timestamp.now().isoformat(),
            "last_runtime_s": solution["runtime_s"], "last_mip_gap": solution["gap"],
        }, indent=2), encoding="utf-8")
        if PROGRESS_HOOK is not None:
            PROGRESS_HOOK(name, origin + 1, SIM_STEPS, timestamps[origin])
    frame = pd.DataFrame(series_rows)
    plans = pd.DataFrame(plan_rows)
    frame.to_csv(output / "mpc_timeseries.csv", index=False)
    pd.DataFrame([{
        "pv_waste_energy_kwh": float(frame["pv_waste_kw"].sum() * DT),
        "pv_waste_steps": int((frame["pv_waste_kw"] > 1e-9).sum()),
        "maximum_pv_waste_kw": float(frame["pv_waste_kw"].max()),
        "minimum_grid_kw": float(frame["realized_net_kw"].min()),
    }]).to_csv(output / "curtailment_summary.csv", index=False)
    plans.to_csv(output / "rolling_plans.csv", index=False)
    pd.DataFrame(solver_rows).to_csv(output / "solver_log.csv", index=False)
    _metrics(name, frame).to_csv(output / "period_metrics.csv", index=False)
    _plot(name, frame, plans, output)
    print(f"[{name}] Complete in {(time.time()-started)/60:.1f} min -> {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["all", *EXPERIMENTS], default="all")
    args = parser.parse_args()
    # The old CLI must not silently resume forecast-noise output directories.
    import runpy
    entry = ROOT / "soft_soc_simulations" / "actual_load_noise" / "run_experiments.py"
    methods = {"t5_perfect": "T56", "t5": "T5", "t6": "T6"}
    chosen = list(methods.values()) if args.experiment == "all" else [methods[args.experiment]]
    runpy.run_path(str(entry))["main"](["--methods", *chosen])


if __name__ == "__main__":
    main()
