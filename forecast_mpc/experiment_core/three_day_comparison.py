from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
import sys
import time


import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

from experiment_core.phase1_t3_t4 import _solve_follower
import experiment_core.stackelberg_mpc as sm

ROOT = Path(__file__).resolve().parents[1]
PHASE4 = ROOT.parent / "stackelberg_scenario_analysis" / "phase4"
if str(PHASE4) not in sys.path:
    sys.path.insert(0, str(PHASE4))
import p4_s3_stackelberg as p4_stackelberg

OUTPUT_ROOT = ROOT / "comparison_3days"
EXECUTION_DAYS = 3
EXECUTION_STEPS = EXECUTION_DAYS * sm.STEPS_PER_DAY
INPUT_STEPS = EXECUTION_STEPS + sm.HORIZON - 1
OL_TIME_LIMIT_S = 1200.0
MIP_GAP = 0.05

METHODS = {
    "t12_perfect": ("perfect", "ol_ol"),
    "t1": ("iid", "ol_ol"),
    "t2": ("ar1", "ol_ol"),
    "t34_perfect": ("perfect", "ol_mpc"),
    "t3": ("iid", "ol_mpc"),
    "t4": ("ar1", "ol_mpc"),
}

OUTPUT_FOLDERS = {
    "t12_perfect": "t12_perfect_ol_ol",
    "t1": "t1_iid_ol_ol",
    "t2": "t2_ar1_ol_ol",
    "t34_perfect": "t34_perfect_ol_mpc",
    "t3": "t3_iid_ol_mpc",
    "t4": "t4_ar1_ol_mpc",
    "t56_perfect": "t56_perfect_mpc_mpc",
    "t5_perfect": "t5_perfect_mpc_mpc",
    "t5": "t5_iid_mpc_mpc",
    "t6": "t6_ar1_mpc_mpc",
}


def _solve_daily_stackelberg(date, loads, pv, pa, current_energy, previous_pb=None):
    """Use the proven Phase-4 OL KKT structure with an inherited SOC state."""
    original_follower_lp = p4_stackelberg.follower_lp

    def follower_lp_with_state(bus, baseline, settings):
        cfg = sm.BUS_CONFIG[bus]
        horizon = sm.STEPS_PER_DAY
        q = 3 * horizon
        aeq, beq, aub, bub = [], [], [], []
        for step in range(horizon):
            row = np.zeros(q)
            row[step] = 1.0
            row[horizon + step] = -1.0
            row[2 * horizon + step] = 1.0
            aeq.append(row)
            beq.append(-baseline[step])
        initial = float(current_energy[bus])
        for count in range(1, horizon + 1):
            row = np.zeros(q)
            row[:count] = sm.DT
            aub.extend((row, -row))
            bub.extend((cfg["bess_e_max"] - initial, initial - cfg["bess_e_min"]))
        tolerance = settings["TERMINAL_ENERGY_TOL_FRAC"] * (
            cfg["bess_e_max"] - cfg["bess_e_min"]
        )
        target_low = max(cfg["bess_e_min"], cfg["bess_e_init"] - tolerance)
        target_high = min(cfg["bess_e_max"], cfg["bess_e_init"] + tolerance)
        row = np.zeros(q)
        row[:horizon] = sm.DT
        aub.extend((row, -row))
        bub.extend((target_high - initial, initial - target_low))
        grid_bound = float(settings["GRID_BOUND_KW"])
        lb = np.r_[np.full(horizon, -cfg["bess_kw"]), np.zeros(2 * horizon)]
        ub = np.r_[np.full(horizon, cfg["bess_kw"]), np.full(2 * horizon, grid_bound)]
        return tuple(map(np.asarray, (aeq, beq, aub, bub))) + (lb, ub)

    settings = deepcopy(sm.PHASE4_SETTINGS)
    settings.update({"SOLVER_OUTPUT": 1, "TIME_LIMIT_S_PER_DAY": OL_TIME_LIMIT_S, "MIP_GAP": MIP_GAP})
    started = time.perf_counter()
    p4_stackelberg.follower_lp = follower_lp_with_state
    try:
        model = gp.Model(f"comparison_OL_{date:%Y%m%d}")
        model.Params.OutputFlag = int(settings.get("SOLVER_OUTPUT", 1))
        model.Params.LogToConsole = int(settings.get("SOLVER_OUTPUT", 1))
        model.Params.DisplayInterval = int(settings.get("GUROBI_DISPLAY_INTERVAL_S", 2))
        model.Params.MIPGap = MIP_GAP
        model.Params.TimeLimit = OL_TIME_LIMIT_S
        model.Params.NumericFocus = 1
        model.Params.MIPFocus = 1
        model.Params.Heuristics = 0.50
        model.Params.NoRelHeurTime = 60.0
        model.Params.PumpPasses = 10
        model.Params.RINS = 10
        model.Params.Seed = 1
        horizon = sm.STEPS_PER_DAY
        pb_vars = model.addVars(
            horizon, lb=settings["PB_MIN"], ub=settings["PB_MAX"], name="PB"
        )
        if settings.get("ENFORCE_PB_DAILY_MEAN", False):
            model.addConstr(
                gp.quicksum(pb_vars[step] for step in range(horizon))
                == horizon * float(settings["PB_DAILY_MEAN"])
            )
        ramp = float(settings["PB_RAMP"])
        if previous_pb is not None:
            model.addConstr(pb_vars[0] - float(previous_pb) <= ramp)
            model.addConstr(float(previous_pb) - pb_vars[0] <= ramp)
        for step in range(1, horizon):
            model.addConstr(pb_vars[step] - pb_vars[step - 1] <= ramp)
            model.addConstr(pb_vars[step - 1] - pb_vars[step] <= ramp)
        x = {}
        for bus in sm.BUS_ORDER:
            x[bus] = p4_stackelberg.add_kkt(
                model, bus, loads[bus] - pv[bus], pa, pb_vars, settings
            )
        peak = model.addVar(lb=-GRB.INFINITY, name="peak")
        valley = model.addVar(lb=-GRB.INFINITY, name="valley")
        for step in range(horizon):
            aggregate = gp.quicksum(
                x[bus][horizon + step] - x[bus][2 * horizon + step]
                for bus in sm.BUS_ORDER
            )
            model.addConstr(peak >= aggregate)
            model.addConstr(valley <= aggregate)
        model.setObjective(peak - valley, GRB.MINIMIZE)
        model.optimize()
        if model.SolCount == 0:
            raise RuntimeError(f"Daily OL {date.date()}: no incumbent, status={model.Status}")
        pb = np.array([pb_vars[step].X for step in range(horizon)])
        result = {}
        for bus in sm.BUS_ORDER:
            values = np.array([x[bus][j].X for j in range(3 * horizon)])
            bess, imports, exports = np.split(values, 3)
            result[bus] = {
                "bess": bess,
                "net": imports - exports,
                "bess_energy": float(current_energy[bus])
                + np.r_[0.0, np.cumsum(sm.DT * bess)],
            }
        gap = float(model.MIPGap)
        status = int(model.Status)
        objective = float(model.ObjVal)
        best_bound = float(model.ObjBound)
        nodes = float(model.NodeCount)
    finally:
        p4_stackelberg.follower_lp = original_follower_lp
    runtime = time.perf_counter() - started
    return {
        "actions": {bus: float(result[bus]["bess"][0]) for bus in sm.BUS_ORDER},
        "bess_plans": {bus: np.asarray(result[bus]["bess"], float) for bus in sm.BUS_ORDER},
        "net_plans": {bus: np.asarray(result[bus]["net"], float) for bus in sm.BUS_ORDER},
        "pb": np.asarray(pb, float), "status": status,
        "objective": objective, "best_bound": best_bound, "gap": float(gap),
        "runtime_s": runtime, "nodes": nodes,
    }


def configure_duration(days: int, output_root: Path) -> None:
    global EXECUTION_DAYS, EXECUTION_STEPS, INPUT_STEPS, OUTPUT_ROOT
    EXECUTION_DAYS = int(days)
    EXECUTION_STEPS = EXECUTION_DAYS * sm.STEPS_PER_DAY
    INPUT_STEPS = EXECUTION_STEPS + sm.HORIZON - 1
    OUTPUT_ROOT = Path(output_root)
    sm.CONTINUOUS_EXECUTION_DAYS = EXECUTION_DAYS
    sm.CONTINUOUS_EXECUTION_STEPS = EXECUTION_STEPS
    sm.CONTINUOUS_INPUT_STEPS = INPUT_STEPS


def _forecast_slice(forecasts, start, length):
    return {
        variable: {
            bus: forecasts[variable][bus][start:start + length]
            for bus in sm.BUS_ORDER
        }
        for variable in ("load", "pv")
    }


def _solver_row(step, timestamp, solution, limit):
    return {
        "step": step,
        "timestamp": timestamp,
        "status": solution["status"],
        "objective": solution["objective"],
        "best_bound": solution["best_bound"],
        "mip_gap": solution["gap"],
        "runtime_s": solution["runtime_s"],
        "nodes": solution["nodes"],
        "time_limit_s": limit,
        "time_limit_reached": solution["status"] == int(GRB.TIME_LIMIT),
        "gap_target_met": solution["gap"] <= MIP_GAP + 1e-9,
    }


def _write_metrics(frame, solver, output, wall_time_s, day_ahead_solver):
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    solver = solver.copy()
    solver["timestamp"] = pd.to_datetime(solver["timestamp"])
    metrics = sm._continuous_metrics(
        frame, solver, wall_time_s, day_ahead_solver=day_ahead_solver
    )
    metrics.iloc[1:].to_csv(output / "daily_metrics.csv", index=False)
    metrics.iloc[[0]].to_csv(output / "overall_metrics.csv", index=False)


def _save_day_ahead_plan(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _run_ol_ol(name, scenario, actual, pa, timestamps):
    output = OUTPUT_ROOT / OUTPUT_FOLDERS[name]
    output.mkdir(parents=True, exist_ok=True)
    forecasts = sm._day_ahead_forecasts(actual, scenario)
    sm._write_day_ahead_forecasts(output, forecasts, actual, timestamps)
    series_path = output / "timeseries.partial.csv"
    solver_path = output / "solver_log.partial.csv"
    plan_path = output / "day_ahead_plan.partial.csv"
    series = pd.read_csv(series_path, parse_dates=["timestamp"]).to_dict("records") if series_path.exists() else []
    solver_rows = pd.read_csv(solver_path, parse_dates=["timestamp"]).to_dict("records") if solver_path.exists() else []
    plan_rows = pd.read_csv(plan_path, parse_dates=["timestamp"]).to_dict("records") if plan_path.exists() else []
    if len(series) % sm.STEPS_PER_DAY:
        raise RuntimeError(f"{name}: OL checkpoint is not aligned to a complete day")
    start_day = len(series) // sm.STEPS_PER_DAY
    energy = {
        bus: (float(series[-1][f"soc_{bus.lower().replace(' ', '_')}_kwh"])
              if series else float(sm.BUS_CONFIG[bus]["bess_e_init"]))
        for bus in sm.BUS_ORDER
    }
    previous_pb = float(series[-1]["pb_eur_per_kwh"]) if series else None
    started = time.perf_counter()
    for day in range(start_day, EXECUTION_DAYS):
        origin = day * sm.STEPS_PER_DAY
        forecast = _forecast_slice(forecasts, origin, sm.STEPS_PER_DAY)
        print(f"[{name}] daily Stackelberg OL {day + 1}/{EXECUTION_DAYS}", flush=True)
        solution = _solve_daily_stackelberg(
            timestamps[origin], forecast["load"], forecast["pv"],
            pa[origin:origin + sm.STEPS_PER_DAY], energy, previous_pb,
        )
        solver_rows.append(_solver_row(origin, timestamps[origin], solution, OL_TIME_LIMIT_S))
        day_start_energy = dict(energy)
        for local in range(sm.STEPS_PER_DAY):
            step = origin + local
            actions = {bus: float(solution["bess_plans"][bus][local]) for bus in sm.BUS_ORDER}
            uncontrolled = sum(
                actual["load"][bus][step] - actual["pv"][bus][step]
                for bus in sm.BUS_ORDER
            )
            bess = sum(actions.values())
            for bus in sm.BUS_ORDER:
                energy[bus] += sm.DT * actions[bus]
            series.append({
                "step": step, "timestamp": timestamps[step],
                "pa_eur_per_kwh": pa[step],
                "pb_eur_per_kwh": solution["pb"][local],
                "uncontrolled_net_kw": uncontrolled,
                "realized_net_kw": uncontrolled + bess,
                "bess_kw": bess,
                **{f"soc_{bus.lower().replace(' ', '_')}_kwh": energy[bus] for bus in sm.BUS_ORDER},
            })
            for bus in sm.BUS_ORDER:
                plan_rows.append({
                    "day": day, "step": step, "timestamp": timestamps[step], "bus": bus,
                    "load_forecast_kw": forecast["load"][bus][local],
                    "pv_forecast_kw": forecast["pv"][bus][local],
                    "pb_eur_per_kwh": solution["pb"][local],
                    "bess_plan_kw": actions[bus],
                    "soc_plan_kwh": day_start_energy[bus]
                    + sm.DT * np.sum(solution["bess_plans"][bus][:local + 1]),
                })
        previous_pb = float(solution["pb"][-1])
        pd.DataFrame(series).to_csv(series_path, index=False)
        pd.DataFrame(solver_rows).to_csv(solver_path, index=False)
        _save_day_ahead_plan(plan_path, plan_rows)
    frame = pd.DataFrame(series)
    solver = pd.DataFrame(solver_rows)
    frame.to_csv(output / "timeseries.csv", index=False)
    solver.to_csv(output / "solver_log.csv", index=False)
    _save_day_ahead_plan(output / "day_ahead_plan.csv", plan_rows)
    _write_metrics(frame, solver, output, time.perf_counter() - started, solver)
    print(f"[{name}] complete -> {output}", flush=True)


def _day_ahead_pb_plans(name, scenario, output, actual, pa, timestamps):
    plan_path = output / "day_ahead_pb_plans.csv"
    solver_path = output / "day_ahead_solver_log.csv"
    if plan_path.exists() and solver_path.exists():
        plan = pd.read_csv(plan_path, parse_dates=["timestamp"])
        solver = pd.read_csv(solver_path, parse_dates=["timestamp"])
        if len(plan) == (EXECUTION_DAYS + 1) * sm.STEPS_PER_DAY:
            return plan["pb_eur_per_kwh"].to_numpy(float), solver
    forecasts = sm._day_ahead_forecasts(actual, scenario)
    sm._write_day_ahead_forecasts(output, forecasts, actual, timestamps)
    rows, solver_rows = [], []
    previous_pb = None
    initial_energy = {bus: float(sm.BUS_CONFIG[bus]["bess_e_init"]) for bus in sm.BUS_ORDER}
    for day in range(EXECUTION_DAYS + 1):
        origin = day * sm.STEPS_PER_DAY
        day_forecast = _forecast_slice(forecasts, origin, sm.STEPS_PER_DAY)
        solution = _solve_daily_stackelberg(
            timestamps[origin], day_forecast["load"], day_forecast["pv"],
            pa[origin:origin + sm.STEPS_PER_DAY], initial_energy, previous_pb,
        )
        solver_rows.append(_solver_row(origin, timestamps[origin], solution, OL_TIME_LIMIT_S))
        rows.extend({
            "day": day, "step": origin + local, "timestamp": timestamps[origin + local],
            "pb_eur_per_kwh": solution["pb"][local],
        } for local in range(sm.STEPS_PER_DAY))
        previous_pb = float(solution["pb"][-1])
        pd.DataFrame(rows).to_csv(plan_path, index=False)
        pd.DataFrame(solver_rows).to_csv(solver_path, index=False)
        print(f"[{name}] day-ahead PB {day + 1}/{EXECUTION_DAYS + 1} saved", flush=True)
    return pd.DataFrame(rows)["pb_eur_per_kwh"].to_numpy(float), pd.DataFrame(solver_rows)


def _run_ol_mpc(name, scenario, actual, pa, timestamps):
    output = OUTPUT_ROOT / OUTPUT_FOLDERS[name]
    output.mkdir(parents=True, exist_ok=True)
    pb, day_ahead_solver = _day_ahead_pb_plans(
        name, scenario, output, actual, pa, timestamps
    )
    series_path = output / "timeseries.partial.csv"
    solver_path = output / "solver_log.partial.csv"
    forecast_path = output / "rolling_forecasts.partial.csv"
    series = pd.read_csv(series_path, parse_dates=["timestamp"]).to_dict("records") if series_path.exists() else []
    solver_rows = pd.read_csv(solver_path, parse_dates=["timestamp"]).to_dict("records") if solver_path.exists() else []
    forecast_rows = pd.read_csv(forecast_path, parse_dates=["origin_timestamp", "target_timestamp"]).to_dict("records") if forecast_path.exists() else []
    start = len(series)
    if start and [int(row["step"]) for row in series] != list(range(start)):
        raise RuntimeError(f"{name}: non-contiguous checkpoint")
    energy = {
        bus: (float(series[-1][f"soc_{bus.lower().replace(' ', '_')}_kwh"])
              if series else float(sm.BUS_CONFIG[bus]["bess_e_init"]))
        for bus in sm.BUS_ORDER
    }
    started = time.perf_counter()
    for origin in range(start, EXECUTION_STEPS):
        forecast = sm._rolling_forecast(actual, scenario, origin)
        actions, runtime = {}, 0.0
        for bus in sm.BUS_ORDER:
            baseline = forecast["load"][bus] - forecast["pv"][bus]
            plan, energy_path, elapsed = _solve_follower(
                bus, energy[bus], baseline,
                pa[origin:origin + sm.HORIZON], pb[origin:origin + sm.HORIZON], origin,
            )
            actions[bus] = float(plan[0])
            energy[bus] = float(energy_path[1])
            runtime += elapsed
            for lead in range(sm.HORIZON):
                forecast_rows.append({
                    "origin_step": origin, "target_step": origin + lead, "lead_step": lead,
                    "origin_timestamp": timestamps[origin], "target_timestamp": timestamps[origin + lead],
                    "bus": bus,
                    "load_real_kw": actual["load"][bus][origin + lead],
                    "load_forecast_kw": forecast["load"][bus][lead],
                    "pv_real_kw": actual["pv"][bus][origin + lead],
                    "pv_forecast_kw": forecast["pv"][bus][lead],
                })
        uncontrolled = sum(
            actual["load"][bus][origin] - actual["pv"][bus][origin]
            for bus in sm.BUS_ORDER
        )
        bess = sum(actions.values())
        series.append({
            "step": origin, "timestamp": timestamps[origin],
            "pa_eur_per_kwh": pa[origin], "pb_eur_per_kwh": pb[origin],
            "uncontrolled_net_kw": uncontrolled, "realized_net_kw": uncontrolled + bess,
            "bess_kw": bess,
            **{f"soc_{bus.lower().replace(' ', '_')}_kwh": energy[bus] for bus in sm.BUS_ORDER},
        })
        solver_rows.append({
            "step": origin, "timestamp": timestamps[origin], "status": int(GRB.OPTIMAL),
            "objective": np.nan, "best_bound": np.nan, "mip_gap": 0.0,
            "runtime_s": runtime, "nodes": 0.0, "time_limit_s": np.nan,
            "time_limit_reached": False, "gap_target_met": True,
        })
        pd.DataFrame(series).to_csv(series_path, index=False)
        pd.DataFrame(solver_rows).to_csv(solver_path, index=False)
        pd.DataFrame(forecast_rows).to_csv(forecast_path, index=False)
        print(f"[{name}] follower MPC {origin + 1:03d}/{EXECUTION_STEPS}", flush=True)
    frame, solver = pd.DataFrame(series), pd.DataFrame(solver_rows)
    frame.to_csv(output / "timeseries.csv", index=False)
    solver.to_csv(output / "solver_log.csv", index=False)
    pd.DataFrame(forecast_rows).to_csv(output / "rolling_forecasts.csv", index=False)
    _write_metrics(frame, solver, output, time.perf_counter() - started, day_ahead_solver)
    print(f"[{name}] complete -> {output}", flush=True)


def run_method(name):
    sm.CONTINUOUS_EXECUTION_DAYS = EXECUTION_DAYS
    sm.CONTINUOUS_EXECUTION_STEPS = EXECUTION_STEPS
    sm.CONTINUOUS_INPUT_STEPS = INPUT_STEPS
    if name in ("t56_perfect", "t5_perfect", "t5", "t6"):
        scenario = {"t56_perfect": "perfect", "t5_perfect": "perfect", "t5": "iid", "t6": "ar1"}[name]
        label = {
            "t56_perfect": "T56-Perfect: perfect forecast + MPC-MPC",
            "t5_perfect": "T5-perfect: perfect forecast + MPC-MPC",
            "t5": "T5-IID: IID forecast + MPC-MPC",
            "t6": "T6-AR(1): AR(1) forecast + MPC-MPC",
        }[name]
        output = OUTPUT_ROOT / OUTPUT_FOLDERS[name]
        config_key = name
        sm.EXPERIMENTS[config_key] = {
            "label": label,
            "scenario": scenario, "output": output,
            "mip_gap": MIP_GAP, "time_limit_s": 900.0,
            "initial_time_limit_s": 1200.0, "initial_long_steps": 2,
        }
        return sm._run_continuous_experiment(config_key)
    scenario, architecture = METHODS[name]
    actual, pa, timestamps = sm._load_continuous_inputs()
    if architecture == "ol_ol":
        return _run_ol_ol(name, scenario, actual, pa, timestamps)
    return _run_ol_mpc(name, scenario, actual, pa, timestamps)
