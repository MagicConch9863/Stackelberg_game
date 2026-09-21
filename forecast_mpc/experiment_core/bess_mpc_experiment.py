from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import time

import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
PHASE4 = PROJECT_ROOT / "stackelberg_scenario_analysis" / "phase4"
if str(PHASE4) not in sys.path:
    sys.path.insert(0, str(PHASE4))

from p4_common import BUS_CONFIG, BUS_ORDER, DT, daily_metrics, make_result_item, read_pa_year
from p4_run import SETTINGS as PHASE4_SETTINGS
from soc_soft_band import add_soft_soc_band


SCENARIO_FILE = ROOT / "results" / "control_inputs" / "bus_scenarios" / "forecast_scenarios.csv"
STEPS_PER_DAY = 96
HORIZON = 16
SIGMA_EPSILON_KW = 0.3
AR1_RHO = 0.9
PV_ACTIVE_THRESHOLD_KW = 0.01
BASE_SEED = 20250903
FLOW_TIE_BREAK = 1e-5

EXPERIMENTS = {
    "t3": {
        "label": "T3 IID Gaussian: OL tariff + BESS MPC",
        "scenario": "iid",
        "ol_folder": "t1_iid_ol_ol",
        "output": ROOT / "t3_iid_ol_mpc" / "results_soft_soc",
    },
    "t4": {
        "label": "T4 AR(1) Gaussian: OL tariff + BESS MPC",
        "scenario": "ar1",
        "ol_folder": "t2_ar1_ol_ol",
        "output": ROOT / "t4_ar1_ol_mpc" / "results_soft_soc",
    },
}


def configure_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _stable_rng(date, origin_step, bus, variable, scenario):
    key = f"{BASE_SEED}|{pd.Timestamp(date).date()}|{origin_step}|{bus}|{variable}|{scenario}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))


def _rolling_forecast(realization, date, origin_step, bus, variable, scenario, capacity_kw):
    realization = np.asarray(realization, dtype=float)
    horizon = len(realization)
    forecast = realization.copy()
    raw_error = np.zeros(horizon, dtype=float)
    if horizon > 1 and not (variable == "pv" and capacity_kw <= 0.0):
        rng = _stable_rng(date, origin_step, bus, variable, scenario)
        if scenario == "iid":
            raw_error[1:] = rng.normal(0.0, SIGMA_EPSILON_KW, horizon - 1)
        elif scenario == "ar1":
            errors = np.empty(horizon - 1, dtype=float)
            errors[0] = rng.normal(
                0.0, SIGMA_EPSILON_KW / np.sqrt(1.0 - AR1_RHO ** 2)
            )
            innovations = rng.normal(0.0, SIGMA_EPSILON_KW, horizon - 1)
            for index in range(1, horizon - 1):
                errors[index] = AR1_RHO * errors[index - 1] + innovations[index]
            raw_error[1:] = errors
        else:
            raise ValueError(f"Unknown rolling forecast scenario: {scenario}")
        forecast[1:] += raw_error[1:]

    if variable == "load":
        forecast = np.maximum(forecast, 0.0)
    elif variable == "pv":
        active = realization >= PV_ACTIVE_THRESHOLD_KW
        forecast[~active] = 0.0
        forecast[active] = np.clip(forecast[active], 0.0, capacity_kw)
    else:
        raise ValueError(f"Unknown variable: {variable}")

    # The currently executed interval is measured, not forecast.
    forecast[0] = realization[0]
    return forecast, raw_error


def _load_inputs():
    if not SCENARIO_FILE.exists():
        raise FileNotFoundError(f"Missing scenario input: {SCENARIO_FILE}")
    frame = pd.read_csv(SCENARIO_FILE, parse_dates=["timestamp"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame


def _day_realizations(frame, date):
    day = frame[frame["date"] == pd.Timestamp(date).normalize()]
    result = {"load": {}, "pv": {}}
    for variable in result:
        for bus in BUS_ORDER:
            values = day[
                (day["variable"] == variable) & (day["bus"] == bus)
            ].sort_values("step")["realization_kW"].to_numpy(float)
            if len(values) != STEPS_PER_DAY:
                raise RuntimeError(f"{date} {bus} {variable}: expected 96 values, got {len(values)}")
            result[variable][bus] = values
    return result


def _load_ol_day(config, date):
    path = ROOT / config["ol_folder"] / "results_soft_soc" / "planned_and_realized_timeseries.csv"
    frame = pd.read_csv(path)
    day = frame[frame["date"].astype(str) == pd.Timestamp(date).date().isoformat()].copy()
    if day.empty:
        raise KeyError(f"No OL result for {date} in {path}")
    price_check = day.groupby("step")["pb_eur_per_kwh"].nunique()
    if len(price_check) != STEPS_PER_DAY or (price_check != 1).any():
        raise RuntimeError(f"{date}: OL tariff is incomplete or inconsistent across buses")
    pb = day.groupby("step")["pb_eur_per_kwh"].first().sort_index().to_numpy(float)
    ol_net = day.groupby("step")["realized_net_kW"].sum().sort_index().to_numpy(float)
    metrics = pd.read_csv(ROOT / config["ol_folder"] / "results_soft_soc" / "daily_metrics.csv")
    metric = metrics[metrics["date"].astype(str) == pd.Timestamp(date).date().isoformat()].iloc[0]
    return pb, ol_net, float(metric["solver_mip_gap"]), str(metric["solver_result_status"])


def _solve_bus_mpc(bus, current_energy, baseline, pa, pb, end_step):
    baseline = np.asarray(baseline, dtype=float)
    pa = np.asarray(pa, dtype=float)
    pb = np.asarray(pb, dtype=float)
    horizon = len(baseline)
    cfg = BUS_CONFIG[bus]
    model = gp.Model(f"MPC_{bus.replace(' ', '_')}_{end_step - horizon:02d}")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1

    bess = model.addVars(horizon, lb=-cfg["bess_kw"], ub=cfg["bess_kw"], name="bess")
    imp = model.addVars(horizon, lb=0.0, ub=PHASE4_SETTINGS["GRID_BOUND_KW"], name="import")
    exp = model.addVars(horizon, lb=0.0, ub=PHASE4_SETTINGS["GRID_BOUND_KW"], name="export")
    energy = model.addVars(
        horizon + 1,
        lb=cfg["bess_e_min"],
        ub=cfg["bess_e_max"],
        name="energy",
    )
    model.addConstr(energy[0] == float(current_energy))
    for step in range(horizon):
        model.addConstr(imp[step] - exp[step] == baseline[step] + bess[step])
        model.addConstr(energy[step + 1] == energy[step] + DT * bess[step])

    soc_penalty, _, _ = add_soft_soc_band(
        model, energy, horizon, cfg, PHASE4_SETTINGS, bus.replace(" ", "_")
    )
    objective = gp.quicksum(
        DT * (
            (pa[step] + pb[step] + FLOW_TIE_BREAK) * imp[step]
            + (-pa[step] + FLOW_TIE_BREAK) * exp[step]
        )
        for step in range(horizon)
    )
    model.setObjective(objective + soc_penalty, GRB.MINIMIZE)
    started = time.perf_counter()
    model.optimize()
    runtime = time.perf_counter() - started
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"{bus} step {end_step - horizon}: MPC status={model.Status}")
    return {
        "bess": np.array([bess[j].X for j in range(horizon)]),
        "import": np.array([imp[j].X for j in range(horizon)]),
        "export": np.array([exp[j].X for j in range(horizon)]),
        "energy": np.array([energy[j].X for j in range(horizon + 1)]),
        "objective": float(model.ObjVal),
        "runtime_s": runtime,
        "status": int(model.Status),
        "soft_soc_low_kWh": PHASE4_SETTINGS["SOC_SOFT_MIN_FRAC"] * cfg["bess_kwh"],
        "soft_soc_high_kWh": PHASE4_SETTINGS["SOC_SOFT_MAX_FRAC"] * cfg["bess_kwh"],
    }


def _simulate_day(config, date, realized, pa, pb):
    scenario = config["scenario"]
    energy = {bus: float(BUS_CONFIG[bus]["bess_e_init"]) for bus in BUS_ORDER}
    executed_bess = {bus: np.zeros(STEPS_PER_DAY) for bus in BUS_ORDER}
    energy_path = {bus: np.zeros(STEPS_PER_DAY + 1) for bus in BUS_ORDER}
    for bus in BUS_ORDER:
        energy_path[bus][0] = energy[bus]
    solver_rows, forecast_rows = [], []

    day_started = time.perf_counter()
    for origin in range(STEPS_PER_DAY):
        horizon = min(HORIZON, STEPS_PER_DAY - origin)
        stop = origin + horizon
        step_started = time.perf_counter()
        for bus in BUS_ORDER:
            forecasts = {}
            for variable in ("load", "pv"):
                capacity = float(PHASE4_SETTINGS["PV_CAPACITY_KW"].get(bus, 0.0))
                actual_window = realized[variable][bus][origin:stop]
                forecast, raw_error = _rolling_forecast(
                    actual_window, date, origin, bus, variable, scenario, capacity
                )
                forecasts[variable] = forecast
                for lead in range(horizon):
                    forecast_rows.append({
                        "experiment": config["label"],
                        "date": pd.Timestamp(date).date().isoformat(),
                        "forecast_origin_step": origin,
                        "forecast_origin_hour": origin * DT,
                        "target_step": origin + lead,
                        "lead_step": lead,
                        "bus": bus,
                        "variable": variable,
                        "realization_kW": actual_window[lead],
                        "forecast_kW": forecast[lead],
                        "raw_error_kW": raw_error[lead],
                    })

            solution = _solve_bus_mpc(
                bus,
                energy[bus],
                forecasts["load"] - forecasts["pv"],
                pa[origin:stop],
                pb[origin:stop],
                stop,
            )
            action = float(solution["bess"][0])
            executed_bess[bus][origin] = action
            energy[bus] = float(solution["energy"][1])
            energy_path[bus][origin + 1] = energy[bus]
            solver_rows.append({
                "experiment": config["label"],
                "date": pd.Timestamp(date).date().isoformat(),
                "step": origin,
                "hour": origin * DT,
                "bus": bus,
                "horizon_steps": horizon,
                "status": solution["status"],
                "runtime_s": solution["runtime_s"],
                "objective_eur": solution["objective"],
                "executed_bess_kW": action,
                "soc_before_kWh": solution["energy"][0],
                "soc_after_kWh": solution["energy"][1],
                "horizon_end_soc_kWh": solution["energy"][-1],
                "reachable_low_kWh": solution["reachable_low_kWh"],
                "reachable_high_kWh": solution["reachable_high_kWh"],
                "daily_target_low_kWh": solution["daily_target_low_kWh"],
                "daily_target_high_kWh": solution["daily_target_high_kWh"],
            })

        step_runtime = time.perf_counter() - step_started
        print(
            f"[{config['label']}] step {origin + 1:02d}/96 "
            f"({origin * DT:05.2f}h), H={horizon:02d}, "
            f"aggregate BESS={sum(executed_bess[b][origin] for b in BUS_ORDER):+.3f} kW, "
            f"SOC={sum(energy.values()):.3f} kWh, solve={step_runtime:.3f}s",
            flush=True,
        )

    result = {}
    timeseries_rows = []
    for bus in BUS_ORDER:
        net = realized["load"][bus] - realized["pv"][bus] + executed_bess[bus]
        result[bus] = make_result_item(
            executed_bess[bus], np.maximum(net, 0.0), np.maximum(-net, 0.0), energy_path[bus]
        )
        for step in range(STEPS_PER_DAY):
            timeseries_rows.append({
                "experiment": config["label"],
                "date": pd.Timestamp(date).date().isoformat(),
                "step": step,
                "hour": step * DT,
                "bus": bus,
                "pa_eur_per_kwh": pa[step],
                "pb_eur_per_kwh": pb[step],
                "load_real_kW": realized["load"][bus][step],
                "pv_real_kW": realized["pv"][bus][step],
                "bess_mpc_kW": executed_bess[bus][step],
                "soc_before_kWh": energy_path[bus][step],
                "soc_after_kWh": energy_path[bus][step + 1],
                "realized_import_kW": result[bus]["import"][step],
                "realized_export_kW": result[bus]["export"][step],
                "realized_net_kW": result[bus]["net"][step],
            })
    return result, timeseries_rows, solver_rows, forecast_rows, time.perf_counter() - day_started


def _daily_row(key, config, date, result, pa, pb, solver_rows, forecast_rows, runtime, ol_gap, ol_status):
    values = daily_metrics("s3", date, result, pa=pa, pb=pb)
    row = {"experiment": key, "label": config["label"], "date": pd.Timestamp(date).date().isoformat()}
    row.update({key_: value for key_, value in values.items() if key_ not in ("scenario", "date", "month")})
    row["peak_to_valley_kw"] = row["peak_kw"] - row["valley_kw"]
    row["bess_throughput_kwh"] = float(sum(np.sum(np.abs(result[b]["bess"])) * DT for b in BUS_ORDER))
    row["mpc_total_wall_time_s"] = runtime
    row["mpc_lp_solve_time_s"] = float(sum(item["runtime_s"] for item in solver_rows))
    row["mpc_max_single_lp_time_s"] = float(max(item["runtime_s"] for item in solver_rows))
    row["mpc_all_optimal"] = all(item["status"] == int(GRB.OPTIMAL) for item in solver_rows)
    forecasts = pd.DataFrame(forecast_rows)
    forecasts = forecasts[forecasts["lead_step"] > 0]
    for variable in ("load", "pv"):
        subset = forecasts[forecasts["variable"] == variable]
        error = subset["forecast_kW"] - subset["realization_kW"]
        row[f"rolling_{variable}_forecast_MAE_kW"] = float(error.abs().mean())
        row[f"rolling_{variable}_forecast_RMSE_kW"] = float(np.sqrt(np.mean(error ** 2)))
    row["source_ol_solver_mip_gap"] = ol_gap
    row["source_ol_solver_status"] = ol_status
    return row


def plot_mpc(timeseries, ol_curves, label, destination):
    configure_plot_style()
    dates = sorted(timeseries["date"].unique())
    fig, axes = plt.subplots(len(dates), 4, figsize=(12.8, 5.0), squeeze=False)
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
    for row_index, date in enumerate(dates):
        day = timeseries[timeseries["date"] == date]
        aggregate = day.groupby("step", as_index=False).agg({
            "hour": "first", "load_real_kW": "sum", "pv_real_kW": "sum",
            "bess_mpc_kW": "sum", "realized_net_kW": "sum", "pb_eur_per_kwh": "first",
        })
        hour = aggregate["hour"]
        uncontrolled = aggregate["load_real_kW"] - aggregate["pv_real_kW"]
        ax = axes[row_index, 0]
        ax.plot(hour, uncontrolled, color="#777777", ls=":", lw=1.0, label="Uncontrolled net")
        ax.plot(hour, ol_curves[date], color="#E69F00", ls="--", lw=1.0, label="OL realized net")
        ax.plot(hour, aggregate["realized_net_kW"], color="#0072B2", lw=1.1, label="MPC realized net")
        ax.axhline(0.0, color="0.35", lw=0.5)
        ax.set_ylabel("Net exchange (kW)")
        ax.set_title(f"{date}: grid exchange")

        ax = axes[row_index, 1]
        ax.plot(hour, aggregate["bess_mpc_kW"], color="#009E73", lw=1.0, label="Aggregate BESS")
        ax.axhline(0.0, color="0.35", lw=0.5)
        ax.set_ylabel("BESS power (kW)")
        ax.set_title(f"{date}: MPC dispatch")

        ax = axes[row_index, 2]
        for bus, color in zip(BUS_ORDER, colors):
            bus_day = day[day["bus"] == bus].sort_values("step")
            ax.plot(bus_day["hour"], bus_day["soc_after_kWh"], color=color, lw=0.9, label=bus)
        ax.set_ylabel("BESS energy (kWh)")
        ax.set_title(f"{date}: SOC")

        ax = axes[row_index, 3]
        ax.step(hour, aggregate["pb_eur_per_kwh"], where="post", color="#CC79A7", lw=1.0, label="PB")
        ax.set_ylabel("PB (EUR/kWh)")
        ax.set_title(f"{date}: fixed OL tariff")

        for ax in axes[row_index, :]:
            ax.set_xlabel("Time (h)")
            ax.set_xlim(0.0, 24.0)
            ax.set_xticks(np.arange(0.0, 25.0, 4.0))
            ax.grid(True, ls="--", lw=0.4, alpha=0.3)

    handles, names = [], []
    for ax in axes[0, :]:
        h, n = ax.get_legend_handles_labels()
        handles.extend(h)
        names.extend(n)
    unique = dict(zip(names, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(label, fontsize=13, y=0.995)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.10, top=0.85, wspace=0.38, hspace=0.55)
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_mpc_experiment(key):
    key = key.lower()
    if key not in EXPERIMENTS:
        raise ValueError(f"Unknown MPC experiment: {key}")
    config = EXPERIMENTS[key]
    output = config["output"]
    output.mkdir(parents=True, exist_ok=True)
    settings = dict(PHASE4_SETTINGS)
    frame = _load_inputs()
    dates = sorted(frame["date"].unique())
    pa_year, _ = read_pa_year(settings)
    all_metrics, all_timeseries, all_solver, all_forecasts = [], [], [], []
    ol_curves = {}

    print(f"[{config['label']}] Running {len(dates)} dates, 96 MPC updates/date...", flush=True)
    for date_index, date in enumerate(dates, start=1):
        date = pd.Timestamp(date)
        print(f"\n[{config['label']}] date {date_index}/{len(dates)}: {date.date()}", flush=True)
        realized = _day_realizations(frame, date)
        pa = pa_year.loc[(pa_year.index >= date) & (pa_year.index < date + pd.Timedelta(days=1))].to_numpy(float)
        if len(pa) != STEPS_PER_DAY:
            raise RuntimeError(f"{date.date()}: expected 96 PA values, got {len(pa)}")
        pb, ol_net, ol_gap, ol_status = _load_ol_day(config, date)
        ol_curves[date.date().isoformat()] = ol_net
        result, series, solver_rows, forecasts, runtime = _simulate_day(
            config, date, realized, pa, pb
        )
        row = _daily_row(
            key, config, date, result, pa, pb, solver_rows, forecasts, runtime, ol_gap, ol_status
        )
        all_metrics.append(row)
        all_timeseries.extend(series)
        all_solver.extend(solver_rows)
        all_forecasts.extend(forecasts)
        pd.DataFrame(all_metrics).to_csv(output / "daily_metrics.csv", index=False)
        pd.DataFrame(all_timeseries).to_csv(output / "mpc_timeseries.csv", index=False)
        pd.DataFrame(all_solver).to_csv(output / "mpc_solver_log.csv", index=False)
        pd.DataFrame(all_forecasts).to_csv(output / "rolling_forecasts.csv", index=False)
        print(
            f"[{config['label']}] completed {date.date()} in {runtime:.2f}s; "
            f"MPC peak-to-valley={row['peak_to_valley_kw']:.3f} kW",
            flush=True,
        )

    timeseries = pd.DataFrame(all_timeseries)
    # Use unambiguous ASCII names for downstream scripts.
    timeseries.to_csv(output / "mpc_timeseries.csv", index=False)
    pd.DataFrame(all_solver).to_csv(output / "mpc_solver_log.csv", index=False)
    plot_mpc(timeseries, ol_curves, config["label"], output / "pb_and_mpc_overview.png")
    print(f"[{config['label']}] Results: {output}", flush=True)
    return pd.DataFrame(all_metrics)
