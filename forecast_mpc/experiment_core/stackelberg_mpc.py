from __future__ import annotations

import json
from copy import deepcopy
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

from p4_common import (
    BUS_CONFIG, BUS_ORDER, DT, daily_metrics, day_inputs, load_simbench_profiles,
    make_result_item, read_pa_year, read_pv_source,
)
from p4_run import SETTINGS as PHASE4_SETTINGS
from p4_s3_stackelberg import solve_day as solve_stackelberg_day

from experiment_core.ol_experiment import solve_day_with_feasibility_focus


SCENARIO_FILE = ROOT / "results" / "control_inputs" / "bus_scenarios" / "forecast_scenarios.csv"
STEPS_PER_DAY = 96
HORIZON = 16
TIME_LIMIT_S_PER_UPDATE = 600.0
CONTINUOUS_START_DATE = pd.Timestamp("2025-01-15")
CONTINUOUS_EXECUTION_DAYS = 10
CONTINUOUS_EXECUTION_STEPS = CONTINUOUS_EXECUTION_DAYS * STEPS_PER_DAY
CONTINUOUS_INPUT_STEPS = CONTINUOUS_EXECUTION_STEPS + HORIZON - 1
BASE_SEED = 20250903
FORECAST_MARGINAL_SIGMA_KW = 0.3
AR1_RHO = 0.9
AR1_INNOVATION_SIGMA_KW = FORECAST_MARGINAL_SIGMA_KW * np.sqrt(1.0 - AR1_RHO ** 2)
PV_ACTIVE_THRESHOLD_KW = 0.01
SOC_REFERENCE_BAND_FRAC = 0.15
DAY_AHEAD_TIME_LIMIT_S = 1200.0

EXPERIMENTS = {
    "t5_perfect": {
        "label": "T5 diagnostic: perfect forecast + tariff MPC + BESS MPC",
        "scenario": "perfect",
        "output": ROOT / "t5_perfect_mpc_mpc" / "results",
        "dates": ("2025-01-15",),
        "mip_gap": 0.05,
        "time_limit_s": 600.0,
    },
    "t5": {
        "label": "T5 IID Gaussian: tariff MPC + BESS MPC",
        "scenario": "iid",
        "output": ROOT / "t5_iid_mpc_mpc" / "results_10days_soc_reference",
        "mip_gap": 0.05,
        "time_limit_s": 900.0,
        "initial_time_limit_s": 1200.0,
        "initial_long_steps": 2,
    },
    "t6": {
        "label": "T6 AR(1) Gaussian: tariff MPC + BESS MPC",
        "scenario": "ar1",
        "output": ROOT / "t6_ar1_mpc_mpc" / "results_10days_soc_reference",
        "mip_gap": 0.05,
        "time_limit_s": 900.0,
        "initial_time_limit_s": 1200.0,
        "initial_long_steps": 2,
    },
}


def _load_inputs():
    frame = pd.read_csv(SCENARIO_FILE, parse_dates=["timestamp"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame


def _day_arrays(frame, date):
    day = frame[frame["date"] == pd.Timestamp(date).normalize()]
    result = {"realization": {"load": {}, "pv": {}}}
    for scenario in ("perfect", "iid", "ar1"):
        result[scenario] = {"load": {}, "pv": {}}
    for variable in ("load", "pv"):
        for bus in BUS_ORDER:
            rows = day[(day["variable"] == variable) & (day["bus"] == bus)].sort_values("step")
            if len(rows) != STEPS_PER_DAY:
                raise RuntimeError(f"{date} {bus} {variable}: expected 96 values, got {len(rows)}")
            actual = rows["realization_kW"].to_numpy(float)
            result["realization"][variable][bus] = actual
            result["perfect"][variable][bus] = actual.copy()
            result["iid"][variable][bus] = rows["forecast_iid_kW"].to_numpy(float)
            result["ar1"][variable][bus] = rows["forecast_ar1_kW"].to_numpy(float)
    return result


def _forecast_window(data, scenario, origin, stop):
    forecast = {"load": {}, "pv": {}}
    for variable in forecast:
        for bus in BUS_ORDER:
            values = data[scenario][variable][bus][origin:stop].copy()
            # Current interval is measured; only future intervals are forecasts.
            values[0] = data["realization"][variable][bus][origin]
            forecast[variable][bus] = values
    return forecast


def _terminal_reachable_bounds(bus, end_step, boundary_step=None):
    cfg = BUS_CONFIG[bus]
    tolerance = PHASE4_SETTINGS["TERMINAL_ENERGY_TOL_FRAC"] * (
        cfg["bess_e_max"] - cfg["bess_e_min"]
    )
    target_low = max(cfg["bess_e_min"], cfg["bess_e_init"] - tolerance)
    target_high = min(cfg["bess_e_max"], cfg["bess_e_init"] + tolerance)
    if boundary_step is None:
        boundary_step = STEPS_PER_DAY
    remaining_energy = max(0, int(boundary_step) - int(end_step)) * DT * cfg["bess_kw"]
    return (
        max(cfg["bess_e_min"], target_low - remaining_energy),
        min(cfg["bess_e_max"], target_high + remaining_energy),
        target_low,
        target_high,
    )


def _follower_lp(
    bus, baseline, current_energy, origin_step, day_steps=STEPS_PER_DAY,
    enforce_daily_terminal=True, terminal_soc_band=None,
):
    baseline = np.asarray(baseline, dtype=float)
    horizon = len(baseline)
    cfg = BUS_CONFIG[bus]
    q = 3 * horizon
    aeq, beq, aub, bub = [], [], [], []

    # -BESS + import - export = baseline, i.e. import-export=load-PV+BESS.
    for step in range(horizon):
        row = np.zeros(q)
        row[step] = -1.0
        row[horizon + step] = 1.0
        row[2 * horizon + step] = -1.0
        aeq.append(row)
        beq.append(baseline[step])

    for count in range(1, horizon + 1):
        row = np.zeros(q)
        row[:count] = DT
        aub.extend((row, -row))
        bub.extend((cfg["bess_e_max"] - current_energy, current_energy - cfg["bess_e_min"]))

    end_step = int(origin_step) + horizon
    _, _, target_low, target_high = _terminal_reachable_bounds(bus, end_step)
    # Enforce the daily SOC band whenever a midnight boundary lies inside the
    # rolling horizon. SOC remains continuous; it is never reset at midnight.
    first_boundary = ((int(origin_step) // day_steps) + 1) * day_steps
    if enforce_daily_terminal:
        for boundary in range(first_boundary, end_step + 1, day_steps):
            count = boundary - int(origin_step)
            boundary_row = np.zeros(q)
            boundary_row[:count] = DT
            aub.extend((boundary_row, -boundary_row))
            bub.extend((target_high - current_energy, current_energy - target_low))

    # Before the next midnight enters the horizon, retain only the necessary
    # reachability band so the daily target cannot become physically impossible.
    next_boundary = ((end_step + day_steps - 1) // day_steps) * day_steps
    if not enforce_daily_terminal:
        reachable_low, reachable_high = cfg["bess_e_min"], cfg["bess_e_max"]
    elif next_boundary == end_step:
        reachable_low, reachable_high = target_low, target_high
    else:
        reachable_low, reachable_high, _, _ = _terminal_reachable_bounds(
            bus, end_step, next_boundary
        )
        # Add only bounds stricter than the ordinary horizon-end SOC bounds.
        # At midnight the exact band above already exists; do not duplicate it.
        row = np.zeros(q)
        row[:horizon] = DT
        if reachable_high < cfg["bess_e_max"] - 1e-10:
            aub.append(row.copy())
            bub.append(reachable_high - current_energy)
        if reachable_low > cfg["bess_e_min"] + 1e-10:
            aub.append(-row.copy())
            bub.append(current_energy - reachable_low)

    # Optional long-timescale energy guidance. This constrains only the end of
    # the rolling window; all within-window BESS actions remain free.
    if terminal_soc_band is not None:
        requested_low, requested_high = map(float, terminal_soc_band)
        reference_low = max(requested_low, cfg["bess_e_min"])
        reference_high = min(requested_high, cfg["bess_e_max"])
        if end_step % day_steps == 0 and enforce_daily_terminal:
            overlap_low = max(reference_low, target_low)
            overlap_high = min(reference_high, target_high)
            if overlap_low > overlap_high + 1e-10:
                raise RuntimeError(
                    f"SOC reference conflicts with daily 5% band: bus={bus}, "
                    f"energy_step={end_step}, reference=[{reference_low:.6f}, "
                    f"{reference_high:.6f}], daily=[{target_low:.6f}, {target_high:.6f}]"
                )
        physical_low = max(cfg["bess_e_min"], current_energy - horizon * DT * cfg["bess_kw"])
        physical_high = min(cfg["bess_e_max"], current_energy + horizon * DT * cfg["bess_kw"])
        reference_low = max(reference_low, physical_low)
        reference_high = min(reference_high, physical_high)
        if reference_low > reference_high + 1e-10:
            raise RuntimeError(
                f"SOC reference is unreachable: bus={bus}, energy_step={end_step}, "
                f"requested=[{requested_low:.6f}, {requested_high:.6f}], "
                f"reachable=[{physical_low:.6f}, {physical_high:.6f}]"
            )
        reference_row = np.zeros(q)
        reference_row[:horizon] = DT
        aub.extend((reference_row, -reference_row))
        bub.extend((reference_high - current_energy, current_energy - reference_low))

    # At an optimum import and export cannot both be positive because PB >= 0
    # and epsilon > 0. These per-period bounds are therefore exact, not heuristic.
    bess_limit = float(cfg["bess_kw"])
    import_ub = np.maximum(baseline + bess_limit, 0.0)
    export_ub = np.maximum(-baseline + bess_limit, 0.0)
    lb = np.r_[np.full(horizon, -cfg["bess_kw"]), np.zeros(2 * horizon)]
    ub = np.r_[
        np.full(horizon, cfg["bess_kw"]),
        import_ub,
        export_ub,
    ]
    return (
        np.asarray(aeq), np.asarray(beq), np.asarray(aub), np.asarray(bub), lb, ub,
        reachable_low, reachable_high, target_low, target_high,
    )


def _complementarity(model, slack_expression, slack_max, dual, name, dual_max):
    maximum = max(float(slack_max), 0.0)
    binary = model.addVar(vtype=GRB.BINARY, name=f"z_{name}")
    slack = model.addVar(lb=0.0, ub=maximum, name=f"slack_{name}")
    model.addConstr(slack == slack_expression)
    model.addConstr(slack <= maximum * binary)
    model.addConstr(dual <= dual_max * (1.0 - binary))


def _add_follower_kkt(
    model, bus, baseline, current_energy, pa, pb, origin_step,
    enforce_daily_terminal=True, terminal_soc_band=None,
):
    packed = _follower_lp(
        bus, baseline, current_energy, origin_step,
        enforce_daily_terminal=enforce_daily_terminal,
        terminal_soc_band=terminal_soc_band,
    )
    aeq, beq, aub, bub, lb, ub = packed[:6]
    horizon = len(baseline)
    q = 3 * horizon
    tag = bus.replace(" ", "_")
    epsilon = float(PHASE4_SETTINGS["FLOW_TIE_BREAK_EUR_PER_KWH"])
    max_marginal_cost = DT * max(
        np.max(np.abs(-np.asarray(pa) + epsilon)),
        np.max(np.abs(np.asarray(pa) + PHASE4_SETTINGS["PB_MIN"] + epsilon)),
        np.max(np.abs(np.asarray(pa) + PHASE4_SETTINGS["PB_MAX"] + epsilon)),
    )
    min_dynamic_coefficient = min(DT, 1.0)
    derived_dual_bound = 1.0 + horizon * max_marginal_cost / min_dynamic_coefficient
    dual_big_m = min(float(PHASE4_SETTINGS["DUAL_BIG_M"]), derived_dual_bound)

    x = model.addVars(q, lb=-GRB.INFINITY, name=f"x_{tag}")
    lam = model.addVars(len(beq), lb=-GRB.INFINITY, name=f"lam_{tag}")
    mu = model.addVars(len(bub), lb=0.0, ub=dual_big_m, name=f"mu_{tag}")

    for index, row in enumerate(aeq):
        nz = np.flatnonzero(row)
        model.addConstr(gp.quicksum(row[j] * x[j] for j in nz) == beq[index])

    for index, row in enumerate(aub):
        nz = np.flatnonzero(row)
        lhs = gp.quicksum(row[j] * x[j] for j in nz)
        model.addConstr(lhs <= bub[index])
        minimum_lhs = sum(row[j] * (lb[j] if row[j] >= 0 else ub[j]) for j in nz)
        _complementarity(
            model, bub[index] - lhs, bub[index] - minimum_lhs, mu[index],
            f"ineq_{tag}_{index}", dual_big_m,
        )

    lower_dual, upper_dual = {}, {}
    for j in range(q):
        span = ub[j] - lb[j]
        if span <= 1e-12:
            # Eliminate a fixed primal component. Its stationarity condition and
            # bound multipliers belong to the eliminated variable, so no binary
            # complementarity pair is required.
            model.addConstr(x[j] == lb[j])
            lower_dual[j] = None
            upper_dual[j] = None
            continue

        model.addConstr(x[j] >= lb[j])
        model.addConstr(x[j] <= ub[j])
        lower_dual[j] = model.addVar(lb=0.0, ub=dual_big_m, name=f"dl_{tag}_{j}")
        upper_dual[j] = model.addVar(lb=0.0, ub=dual_big_m, name=f"du_{tag}_{j}")
        _complementarity(model, x[j] - lb[j], span, lower_dual[j], f"lb_{tag}_{j}", dual_big_m)
        _complementarity(model, ub[j] - x[j], span, upper_dual[j], f"ub_{tag}_{j}", dual_big_m)

    for j in range(q):
        if j < horizon:
            cost = 0.0
        elif j < 2 * horizon:
            step = j - horizon
            cost = DT * (pa[step] + pb[step] + epsilon)
        else:
            step = j - 2 * horizon
            cost = DT * (-pa[step] + epsilon)
        equality_terms = gp.quicksum(aeq[i, j] * lam[i] for i in np.flatnonzero(aeq[:, j]))
        inequality_terms = gp.quicksum(aub[i, j] * mu[i] for i in np.flatnonzero(aub[:, j]))
        if lower_dual[j] is not None:
            model.addConstr(
                cost + equality_terms + inequality_terms
                - lower_dual[j] + upper_dual[j] == 0.0
            )
    bounded_duals = list(mu.values())
    bounded_duals.extend(value for value in lower_dual.values() if value is not None)
    bounded_duals.extend(value for value in upper_dual.values() if value is not None)
    return x, packed[6:], bounded_duals, dual_big_m


def _solve_stackelberg_window(
    date, origin, forecasts, energies, pa, previous_pb, current_hour_pb,
    warm_pb=None, mip_gap=None, time_limit_s=None, enforce_daily_terminal=True,
    terminal_soc_bands=None, previous_realized_net=None, feasibility_focus=False,
):
    horizon = len(pa)
    model = gp.Model(f"T5_T6_{date:%Y%m%d}_{origin:02d}")
    solver_output = int(PHASE4_SETTINGS.get("SOLVER_OUTPUT", 1))
    model.Params.OutputFlag = solver_output
    model.Params.LogToConsole = solver_output
    model.Params.DisplayInterval = int(PHASE4_SETTINGS.get("GUROBI_DISPLAY_INTERVAL_S", 2))
    model.Params.MIPGap = float(
        PHASE4_SETTINGS["MIP_GAP"] if mip_gap is None else mip_gap
    )
    model.Params.TimeLimit = float(
        TIME_LIMIT_S_PER_UPDATE if time_limit_s is None else time_limit_s
    )
    model.Params.NumericFocus = 1
    # OL helper routines set several Gurobi parameters globally. Explicitly
    # restore MPC-local values so a preceding day-ahead solve cannot add a
    # 60-second NoRel heuristic phase to every rolling update.
    model.Params.Heuristics = 0.50 if feasibility_focus else 0.05
    model.Params.NoRelHeurTime = 60.0 if feasibility_focus else 0.0
    model.Params.PumpPasses = 10 if feasibility_focus else -1
    model.Params.RINS = 10 if feasibility_focus else -1
    # Full-day OL models prioritize finding an incumbent. Rolling MPC models
    # prioritize bound improvement once their shifted warm start is available.
    model.Params.MIPFocus = 1 if feasibility_focus else 3
    model.Params.Cuts = -1 if feasibility_focus else 3
    model.Params.Presolve = 2
    model.Params.Symmetry = 2
    model.Params.Seed = 1

    pb = model.addVars(
        horizon,
        lb=PHASE4_SETTINGS["PB_MIN"],
        ub=PHASE4_SETTINGS["PB_MAX"],
        name="PB",
    )
    if warm_pb is not None:
        warm_pb = np.asarray(warm_pb, dtype=float)
        for step in range(min(horizon, len(warm_pb))):
            pb[step].Start = float(warm_pb[step])
    ramp = float(PHASE4_SETTINGS["PB_RAMP"])
    if previous_pb is not None:
        model.addConstr(pb[0] - previous_pb <= ramp)
        model.addConstr(previous_pb - pb[0] <= ramp)
    for step in range(1, horizon):
        model.addConstr(pb[step] - pb[step - 1] <= ramp)
        model.addConstr(pb[step - 1] - pb[step] <= ramp)

    x, bounds, bounded_duals, dual_bounds = {}, {}, {}, {}
    for bus in BUS_ORDER:
        baseline = forecasts["load"][bus] - forecasts["pv"][bus]
        x[bus], bounds[bus], bounded_duals[bus], dual_bounds[bus] = _add_follower_kkt(
            model, bus, baseline, energies[bus], pa, pb, origin,
            enforce_daily_terminal=enforce_daily_terminal,
            terminal_soc_band=(None if terminal_soc_bands is None else terminal_soc_bands[bus]),
        )

    peak = model.addVar(lb=-GRB.INFINITY, name="peak")
    valley = model.addVar(lb=-GRB.INFINITY, name="valley")
    if previous_realized_net is not None:
        # Include the last measured aggregate exchange in the range without
        # imposing a hard ramp. A discontinuity remains feasible, but it is no
        # longer invisible to the receding-horizon leader objective.
        model.addConstr(peak >= float(previous_realized_net), name="anchor_peak")
        model.addConstr(valley <= float(previous_realized_net), name="anchor_valley")
    for step in range(horizon):
        aggregate = gp.quicksum(
            x[bus][horizon + step] - x[bus][2 * horizon + step] for bus in BUS_ORDER
        )
        model.addConstr(peak >= aggregate)
        model.addConstr(valley <= aggregate)
    model.setObjective(peak - valley, GRB.MINIMIZE)
    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError(f"{date.date()} step {origin}: no incumbent, status={model.Status}")

    actions = {bus: float(x[bus][0].X) for bus in BUS_ORDER}
    bess_plans = {
        bus: np.array([x[bus][step].X for step in range(horizon)], dtype=float)
        for bus in BUS_ORDER
    }
    net_plans = {
        bus: np.array([
            x[bus][horizon + step].X - x[bus][2 * horizon + step].X
            for step in range(horizon)
        ], dtype=float)
        for bus in BUS_ORDER
    }
    pb_values = np.array([pb[j].X for j in range(horizon)])
    max_dual_fraction = max(
        dual.X / dual_bounds[bus]
        for bus in BUS_ORDER
        for dual in bounded_duals[bus]
    )
    return {
        "actions": actions,
        "bess_plans": bess_plans,
        "net_plans": net_plans,
        "pb": pb_values,
        "status": int(model.Status),
        "objective": float(model.ObjVal),
        "best_bound": float(model.ObjBound),
        "gap": float(model.MIPGap),
        "runtime_s": float(model.Runtime),
        "nodes": float(model.NodeCount),
        "bounds": bounds,
        "dual_big_m": max(dual_bounds.values()),
        "max_dual_bound_fraction": float(max_dual_fraction),
    }


def _checkpoint_paths(output, date):
    folder = output / "checkpoint" / date.date().isoformat()
    return folder, folder / "state.npz", folder / "meta.json", folder / "solver_rows.csv", folder / "forecast_rows.csv"


def _simulate_day(config, date, data, pa_day, output):
    scenario = config["scenario"]
    energies = {bus: float(BUS_CONFIG[bus]["bess_e_init"]) for bus in BUS_ORDER}
    energy_paths = {bus: np.zeros(STEPS_PER_DAY + 1) for bus in BUS_ORDER}
    bess_paths = {bus: np.zeros(STEPS_PER_DAY) for bus in BUS_ORDER}
    for bus in BUS_ORDER:
        energy_paths[bus][0] = energies[bus]
    pb_executed = np.zeros(STEPS_PER_DAY)
    rows, solver_rows, forecast_rows = [], [], []
    previous_pb = None
    previous_pb_plan = None
    start_origin = 0
    checkpoint_dir, state_path, meta_path, solver_path, forecast_path = _checkpoint_paths(output, date)
    if state_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = {"label": config["label"], "scenario": scenario, "horizon": HORIZON}
        if all(meta.get(key) == value for key, value in expected.items()):
            state = np.load(state_path)
            start_origin = int(state["origin_next"])
            pb_executed[:] = state["pb_executed"]
            previous_pb = None if start_origin == 0 else float(state["previous_pb"])
            previous_pb_plan = state["previous_pb_plan"]
            for index, bus in enumerate(BUS_ORDER):
                bess_paths[bus][:] = state[f"bess_{index}"]
                energy_paths[bus][:] = state[f"energy_{index}"]
                energies[bus] = float(energy_paths[bus][start_origin])
            solver_rows = pd.read_csv(solver_path).to_dict("records") if solver_path.exists() else []
            forecast_rows = pd.read_csv(forecast_path).to_dict("records") if forecast_path.exists() else []
            print(f"[{config['label']}] resumed checkpoint at step {start_origin + 1}/96", flush=True)
    day_started = time.perf_counter()

    for origin in range(start_origin, STEPS_PER_DAY):
        horizon = min(HORIZON, STEPS_PER_DAY - origin)
        stop = origin + horizon
        forecasts = _forecast_window(data, scenario, origin, stop)
        step_time_limit_s = config.get("time_limit_s")
        if origin < int(config.get("initial_long_steps", 0)):
            step_time_limit_s = config.get("initial_time_limit_s", step_time_limit_s)
        warm_pb = None
        if previous_pb_plan is not None:
            warm_pb = np.r_[previous_pb_plan[1:], previous_pb_plan[-1]][:horizon]
        solution = _solve_stackelberg_window(
            date,
            origin,
            forecasts,
            energies,
            pa_day[origin:stop],
            previous_pb,
            None,
            warm_pb,
            config.get("mip_gap"),
            step_time_limit_s,
        )
        current_pb_plan = solution["pb"]
        actions = solution["actions"]
        update_type = "leader_kkt"
        pb_executed[origin] = current_pb_plan[0]
        previous_pb = pb_executed[origin]
        previous_pb_plan = solution["pb"]
        for bus in BUS_ORDER:
            action = solution["actions"][bus]
            bess_paths[bus][origin] = action
            energies[bus] += DT * action
            energy_paths[bus][origin + 1] = energies[bus]
            for variable in ("load", "pv"):
                for lead in range(horizon):
                    target = origin + lead
                    actual = data["realization"][variable][bus][target]
                    forecast_rows.append({
                        "date": date.date().isoformat(), "origin_step": origin,
                        "target_step": target, "lead_step": lead, "bus": bus,
                        "variable": variable, "realization_kW": actual,
                        "forecast_kW": forecasts[variable][bus][lead],
                    })
        solver_rows.append({
            "date": date.date().isoformat(), "step": origin, "hour": origin * DT,
            "update_type": update_type,
            "horizon_steps": horizon, "status": solution["status"],
            "objective_peak_to_valley_kW": solution["objective"],
            "best_bound_kW": solution["best_bound"], "mip_gap": solution["gap"],
            "runtime_s": solution["runtime_s"], "nodes": solution["nodes"],
            "executed_pb_eur_per_kwh": pb_executed[origin],
            "dual_big_m": solution["dual_big_m"],
            "max_dual_bound_fraction": solution["max_dual_bound_fraction"],
            "target_gap": float(config.get("mip_gap", PHASE4_SETTINGS["MIP_GAP"])),
            "time_limit_s": float(
                TIME_LIMIT_S_PER_UPDATE if step_time_limit_s is None else step_time_limit_s
            ),
            "gap_target_met": bool(
                solution["gap"] <= float(config.get("mip_gap", PHASE4_SETTINGS["MIP_GAP"])) + 1e-9
            ),
            "time_limit_reached": bool(solution["status"] == int(GRB.TIME_LIMIT)),
            "solution_accepted": True,
        })
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            state_path,
            origin_next=origin + 1,
            pb_executed=pb_executed,
            previous_pb=previous_pb,
            previous_pb_plan=previous_pb_plan,
            **{f"bess_{index}": bess_paths[bus] for index, bus in enumerate(BUS_ORDER)},
            **{f"energy_{index}": energy_paths[bus] for index, bus in enumerate(BUS_ORDER)},
        )
        meta_path.write_text(
            json.dumps({"label": config["label"], "scenario": scenario, "horizon": HORIZON}, indent=2),
            encoding="utf-8",
        )
        pd.DataFrame(solver_rows).to_csv(solver_path, index=False)
        pd.DataFrame(forecast_rows).to_csv(forecast_path, index=False)
        print(
            f"[{config['label']}] step {origin + 1:02d}/96, {update_type}, H={horizon:02d}, "
            f"status={solution['status']}, obj={solution['objective']:.4f} kW, "
            f"bound={solution['best_bound']:.4f}, gap={solution['gap']:.3%}, "
            f"PB={pb_executed[origin]:.4f}, "
            f"BESS={sum(solution['actions'].values()):+.3f} kW, "
            f"dual_cap={solution['max_dual_bound_fraction']:.1%}, "
            f"time={solution['runtime_s']:.2f}s",
            flush=True,
        )

    result = {}
    for bus in BUS_ORDER:
        actual_baseline = data["realization"]["load"][bus] - data["realization"]["pv"][bus]
        net = actual_baseline + bess_paths[bus]
        result[bus] = make_result_item(
            bess_paths[bus], np.maximum(net, 0.0), np.maximum(-net, 0.0), energy_paths[bus]
        )
        for step in range(STEPS_PER_DAY):
            rows.append({
                "experiment": config["label"], "date": date.date().isoformat(),
                "step": step, "hour": step * DT, "bus": bus,
                "pa_eur_per_kwh": pa_day[step], "pb_eur_per_kwh": pb_executed[step],
                "load_real_kW": data["realization"]["load"][bus][step],
                "pv_real_kW": data["realization"]["pv"][bus][step],
                "bess_mpc_kW": bess_paths[bus][step],
                "soc_before_kWh": energy_paths[bus][step],
                "soc_after_kWh": energy_paths[bus][step + 1],
                "realized_import_kW": result[bus]["import"][step],
                "realized_export_kW": result[bus]["export"][step],
                "realized_net_kW": result[bus]["net"][step],
            })
    return result, pb_executed, rows, solver_rows, forecast_rows, time.perf_counter() - day_started


def _metric_row(key, config, date, result, pa, pb, solver_rows, forecast_rows, runtime):
    values = daily_metrics("s3", date, result, pa=pa, pb=pb)
    row = {"experiment": key, "label": config["label"], "date": date.date().isoformat()}
    row.update({name: value for name, value in values.items() if name not in ("scenario", "date", "month")})
    row["peak_to_valley_kw"] = row["peak_kw"] - row["valley_kw"]
    row["bess_throughput_kwh"] = sum(np.sum(np.abs(result[b]["bess"])) * DT for b in BUS_ORDER)
    logs = pd.DataFrame(solver_rows)
    row["total_wall_time_s"] = runtime
    row["total_solver_time_s"] = logs["runtime_s"].sum()
    row["max_solver_gap"] = logs["mip_gap"].max()
    row["all_updates_have_incumbent"] = True
    forecasts = pd.DataFrame(forecast_rows)
    for variable in ("load", "pv"):
        subset = forecasts[(forecasts["variable"] == variable) & (forecasts["lead_step"] > 0)]
        error = subset["forecast_kW"] - subset["realization_kW"]
        row[f"rolling_{variable}_forecast_MAE_kW"] = error.abs().mean()
        row[f"rolling_{variable}_forecast_RMSE_kW"] = np.sqrt(np.mean(error ** 2))
    row["power_balance_definition"] = "import-export=load-PV+BESS"
    return row


def _plot(timeseries, label, destination):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 10,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    })
    dates = sorted(timeseries["date"].unique())
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
    for date in dates:
        fig, axes_grid = plt.subplots(2, 2, figsize=(7.2, 6.1))
        axes = axes_grid.ravel()
        day = timeseries[timeseries["date"] == date]
        agg = day.groupby("step", as_index=False).agg({
            "hour": "first", "load_real_kW": "sum", "pv_real_kW": "sum",
            "bess_mpc_kW": "sum", "realized_net_kW": "sum", "pb_eur_per_kwh": "first",
        })
        hour = agg["hour"]
        uncontrolled = agg["load_real_kW"] - agg["pv_real_kW"]
        axes[0].plot(hour, uncontrolled, color="#777777", ls=":", label="Uncontrolled")
        axes[0].plot(hour, agg["realized_net_kW"], color="#0072B2", label="Realized")
        axes[0].set_ylabel("Net exchange (kW)")
        axes[1].plot(hour, agg["bess_mpc_kW"], color="#009E73", label="Aggregate BESS")
        axes[1].set_ylabel("BESS power (kW)")
        for bus, color in zip(BUS_ORDER, colors):
            bus_day = day[day["bus"] == bus].sort_values("step")
            axes[2].plot(bus_day["hour"], bus_day["soc_after_kWh"], color=color, label=bus)
        axes[2].set_ylabel("BESS energy (kWh)")
        axes[3].step(hour, agg["pb_eur_per_kwh"], where="post", color="#CC79A7", label="PB")
        axes[3].set_ylabel("PB (EUR/kWh)")
        for column, ax in enumerate(axes):
            ax.set_title(("Grid exchange", "MPC dispatch", "SOC", "Tariff MPC")[column])
            ax.set_xlabel("Time (h)")
            ax.set_xlim(0, 24)
            ax.set_xticks(np.arange(0, 25, 4))
            ax.grid(True, ls="--", lw=0.4, alpha=0.3)
        handles, names = [], []
        for ax in axes:
            h, n = ax.get_legend_handles_labels(); handles.extend(h); names.extend(n)
        unique = dict(zip(names, handles))
        fig.legend(
            unique.values(), unique.keys(), loc="upper center", ncol=4,
            frameon=False, bbox_to_anchor=(0.5, 0.925),
        )
        fig.suptitle(f"{label}\n{date}", fontsize=12, y=0.995)
        fig.subplots_adjust(left=0.11, right=0.98, bottom=0.09, top=0.82, wspace=0.32, hspace=0.38)
        target = destination
        if len(dates) > 1:
            target = destination.with_name(f"{destination.stem}_{date}{destination.suffix}")
        fig.savefig(target, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _rolling_seed(*parts):
    token = "|".join(map(str, (BASE_SEED,) + parts)).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")


def _forecast_error(scenario, length, seed_parts, measured_first=False):
    if scenario == "perfect":
        error = np.zeros(length)
    else:
        rng = np.random.default_rng(_rolling_seed(*seed_parts))
        if scenario == "iid":
            error = rng.normal(0.0, FORECAST_MARGINAL_SIGMA_KW, length)
        elif scenario == "ar1":
            error = np.zeros(length)
            error[0] = rng.normal(0.0, FORECAST_MARGINAL_SIGMA_KW)
            for step in range(1, length):
                error[step] = (
                    AR1_RHO * error[step - 1]
                    + rng.normal(0.0, AR1_INNOVATION_SIGMA_KW)
                )
        else:
            raise ValueError(f"Unknown forecast scenario: {scenario}")
    if measured_first:
        error[0] = 0.0
    return error


def _apply_forecast_physics(actual, error, variable, bus):
    values = np.asarray(actual, float) + np.asarray(error, float)
    if variable == "load":
        return np.maximum(values, 0.0)
    active = np.asarray(actual, float) >= PV_ACTIVE_THRESHOLD_KW
    capacity = float(PHASE4_SETTINGS["PV_CAPACITY_KW"][bus])
    return np.where(active, np.clip(values, 0.0, capacity), 0.0)


def _load_continuous_inputs():
    settings = deepcopy(PHASE4_SETTINGS)
    settings["LOAD_MODE"] = "annual"
    pa_year, _ = read_pa_year(settings)
    pv_source = read_pv_source(settings)
    profiles, full_days = load_simbench_profiles(settings)
    parts = []
    for offset in range(CONTINUOUS_EXECUTION_DAYS + 1):
        date = CONTINUOUS_START_DATE + pd.Timedelta(days=offset)
        loads, pv, pa = day_inputs(date, profiles, full_days, pa_year, pv_source, settings)
        parts.append((loads, pv, pa))
    realization = {
        variable: {
            bus: np.concatenate([part[0 if variable == "load" else 1][bus] for part in parts])
            for bus in BUS_ORDER
        }
        for variable in ("load", "pv")
    }
    for bus in BUS_ORDER:
        realization["pv"][bus][realization["pv"][bus] < PV_ACTIVE_THRESHOLD_KW] = 0.0
    pa = np.concatenate([part[2] for part in parts])
    timestamps = pd.date_range(CONTINUOUS_START_DATE, periods=len(pa), freq="15min")
    return realization, pa, timestamps


def _day_ahead_forecasts(realization, scenario):
    forecasts = {variable: {} for variable in ("load", "pv")}
    for variable in forecasts:
        for bus in BUS_ORDER:
            days = []
            for day in range(CONTINUOUS_EXECUTION_DAYS + 1):
                start = day * STEPS_PER_DAY
                actual = realization[variable][bus][start:start + STEPS_PER_DAY]
                error = _forecast_error(
                    scenario, STEPS_PER_DAY,
                    ("day_ahead", scenario, day, bus, variable),
                )
                days.append(_apply_forecast_physics(actual, error, variable, bus))
            forecasts[variable][bus] = np.concatenate(days)
    return forecasts


def _rolling_forecast(realization, scenario, origin):
    forecast = {variable: {} for variable in ("load", "pv")}
    for variable in forecast:
        for bus in BUS_ORDER:
            actual = realization[variable][bus][origin:origin + HORIZON].copy()
            if len(actual) != HORIZON:
                raise RuntimeError(f"origin {origin}: incomplete {variable} horizon for {bus}")
            error = _forecast_error(
                scenario, HORIZON, ("rolling", scenario, origin, bus, variable),
                measured_first=True,
            )
            forecast[variable][bus] = _apply_forecast_physics(
                actual, error, variable, bus
            )
    return forecast


def _continuous_checkpoint_paths(output):
    checkpoint = output / "checkpoint"
    return (
        checkpoint,
        checkpoint / "state.npz",
        checkpoint / "meta.json",
        output / "mpc_timeseries.partial.csv",
        output / "solver_log.partial.csv",
        output / "rolling_forecasts.partial.csv",
    )


def _continuous_metadata(config):
    return {
        "version": 2,
        "label": config["label"],
        "scenario": config["scenario"],
        "start_date": str(CONTINUOUS_START_DATE.date()),
        "execution_steps": CONTINUOUS_EXECUTION_STEPS,
        "rolling_input_steps": CONTINUOUS_INPUT_STEPS,
        "day_ahead_input_steps": (CONTINUOUS_EXECUTION_DAYS + 1) * STEPS_PER_DAY,
        "horizon": HORIZON,
        "base_seed": BASE_SEED,
        "forecast_marginal_sigma_kw": FORECAST_MARGINAL_SIGMA_KW,
        "ar1_rho": AR1_RHO,
        "ar1_innovation_sigma_kw": AR1_INNOVATION_SIGMA_KW,
        "soc_reference_band_frac": SOC_REFERENCE_BAND_FRAC,
        "day_ahead_time_limit_s": DAY_AHEAD_TIME_LIMIT_S,
    }


def _day_ahead_metadata(config):
    return _continuous_metadata(config) | {
        "reference_days": CONTINUOUS_EXECUTION_DAYS + 1,
        "reference_method": "daily_96_step_stackelberg_ol_from_noisy_day_ahead_forecast",
    }


def _write_day_ahead_forecasts(output, forecasts, realization, timestamps):
    rows = []
    total = (CONTINUOUS_EXECUTION_DAYS + 1) * STEPS_PER_DAY
    for step in range(total):
        for bus in BUS_ORDER:
            rows.append({
                "day": step // STEPS_PER_DAY,
                "step": step,
                "step_in_day": step % STEPS_PER_DAY,
                "timestamp": timestamps[step],
                "bus": bus,
                "load_real_kw": realization["load"][bus][step],
                "load_forecast_kw": forecasts["load"][bus][step],
                "pv_real_kw": realization["pv"][bus][step],
                "pv_forecast_kw": forecasts["pv"][bus][step],
            })
    pd.DataFrame(rows).to_csv(output / "day_ahead_forecasts.csv", index=False)


def _load_or_solve_day_ahead_reference(output, config, realization, pa, timestamps):
    meta_path = output / "day_ahead_cache_meta.json"
    reference_path = output / "day_ahead_soc_reference.csv"
    solver_path = output / "day_ahead_solver_log.csv"
    expected_meta = _day_ahead_metadata(config)
    if meta_path.exists():
        saved_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if saved_meta != expected_meta:
            raise RuntimeError(
                f"Day-ahead cache metadata mismatch in {output}; refusing to overwrite it"
            )

    forecasts = _day_ahead_forecasts(realization, config["scenario"])
    _write_day_ahead_forecasts(output, forecasts, realization, timestamps)
    meta_path.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")

    reference_rows = []
    solver_rows = []
    completed_days = set()
    if reference_path.exists():
        existing = pd.read_csv(reference_path, parse_dates=["timestamp"])
        reference_rows = existing.to_dict("records")
        counts = existing.groupby("day").size()
        completed_days = {
            int(day) for day, count in counts.items()
            if int(count) == (STEPS_PER_DAY + 1) * len(BUS_ORDER)
        }
    if solver_path.exists():
        solver_rows = pd.read_csv(solver_path, parse_dates=["date"]).to_dict("records")

    settings = deepcopy(PHASE4_SETTINGS)
    settings.update({
        "SOLVER_OUTPUT": 1,
        "TIME_LIMIT_S_PER_DAY": DAY_AHEAD_TIME_LIMIT_S,
        "MIP_GAP": config.get("mip_gap", 0.05),
    })
    for day in range(CONTINUOUS_EXECUTION_DAYS + 1):
        if day in completed_days:
            print(
                f"[{config['scenario']}] day-ahead OL {day + 1}/"
                f"{CONTINUOUS_EXECUTION_DAYS + 1} loaded from cache", flush=True
            )
            continue
        start = day * STEPS_PER_DAY
        date = timestamps[start]
        day_forecast = {
            variable: {
                bus: forecasts[variable][bus][start:start + STEPS_PER_DAY]
                for bus in BUS_ORDER
            }
            for variable in ("load", "pv")
        }
        print(
            f"[{config['scenario']}] day-ahead OL {day + 1}/"
            f"{CONTINUOUS_EXECUTION_DAYS + 1}: {date.date()}", flush=True
        )
        started = time.perf_counter()
        result, pb, gap = solve_day_with_feasibility_focus(
            solve_stackelberg_day, date, day_forecast["load"], day_forecast["pv"],
            pa[start:start + STEPS_PER_DAY], settings,
        )
        runtime = time.perf_counter() - started
        reference_rows = [row for row in reference_rows if int(row["day"]) != day]
        for bus in BUS_ORDER:
            energy = np.asarray(result[bus]["bess_energy"], float)
            bess = np.asarray(result[bus]["bess"], float)
            for local_energy_step in range(STEPS_PER_DAY + 1):
                stage = min(local_energy_step, STEPS_PER_DAY - 1)
                reference_rows.append({
                    "day": day,
                    "energy_step": start + local_energy_step,
                    "energy_step_in_day": local_energy_step,
                    "timestamp": date + pd.Timedelta(minutes=15 * local_energy_step),
                    "bus": bus,
                    "soc_reference_kwh": energy[local_energy_step],
                    "pb_day_ahead_eur_per_kwh": float(pb[stage]),
                    "bess_day_ahead_kw": float(bess[stage]),
                })
        solver_rows = [row for row in solver_rows if int(row["day"]) != day]
        solver_rows.append({
            "day": day,
            "date": date.normalize(),
            "status": "incumbent_available",
            "mip_gap": float(gap),
            "runtime_s": float(runtime),
            "time_limit_s": DAY_AHEAD_TIME_LIMIT_S,
            "gap_target_met": bool(gap <= config.get("mip_gap", 0.05) + 1e-9),
        })
        pd.DataFrame(reference_rows).sort_values(
            ["day", "bus", "energy_step_in_day"]
        ).to_csv(reference_path, index=False)
        pd.DataFrame(solver_rows).sort_values("day").to_csv(solver_path, index=False)
        print(
            f"[{config['scenario']}] day-ahead OL saved: gap={gap:.3%}, "
            f"runtime={runtime:.1f}s", flush=True,
        )

    reference = pd.DataFrame(reference_rows)
    if len(reference) != (CONTINUOUS_EXECUTION_DAYS + 1) * (STEPS_PER_DAY + 1) * len(BUS_ORDER):
        raise RuntimeError(f"Incomplete day-ahead SOC reference: {len(reference)} rows")
    lookup = {
        (bus, day * STEPS_PER_DAY + local): float(group.iloc[0]["soc_reference_kwh"])
        for day in range(CONTINUOUS_EXECUTION_DAYS + 1)
        for local in range(STEPS_PER_DAY)
        for bus in BUS_ORDER
        for group in [reference[
            (reference["day"] == day)
            & (reference["energy_step_in_day"] == local)
            & (reference["bus"] == bus)
        ]]
    }
    return lookup, pd.DataFrame(solver_rows)


def _soc_reference_bands(reference_lookup, energy_step):
    bands = {}
    for bus in BUS_ORDER:
        cfg = BUS_CONFIG[bus]
        center = reference_lookup[(bus, int(energy_step))]
        delta = SOC_REFERENCE_BAND_FRAC * (cfg["bess_e_max"] - cfg["bess_e_min"])
        bands[bus] = (center - delta, center + delta)
    return bands


def _validate_reference_boundary_compatibility(reference_lookup):
    for energy_step in range(
        STEPS_PER_DAY,
        CONTINUOUS_EXECUTION_STEPS + 1,
        STEPS_PER_DAY,
    ):
        for bus, (reference_low, reference_high) in _soc_reference_bands(
            reference_lookup, energy_step
        ).items():
            _, _, daily_low, daily_high = _terminal_reachable_bounds(bus, energy_step)
            if max(reference_low, daily_low) > min(reference_high, daily_high) + 1e-10:
                raise RuntimeError(
                    f"SOC reference conflicts with daily 5% band before MPC: bus={bus}, "
                    f"energy_step={energy_step}, reference=[{reference_low:.6f}, "
                    f"{reference_high:.6f}], daily=[{daily_low:.6f}, {daily_high:.6f}]"
                )


def _continuous_metrics(frame, solver, wall_time_s, day_ahead_solver=None):
    rows = []
    overall_period = f"overall_{CONTINUOUS_EXECUTION_DAYS}days"
    periods = [(overall_period, frame)] + [
        (str(date.date()), day) for date, day in frame.groupby(frame["timestamp"].dt.normalize())
    ]
    for period, data in periods:
        net = data["realized_net_kw"].to_numpy(float)
        uncontrolled = data["uncontrolled_net_kw"].to_numpy(float)
        imports = np.maximum(net, 0.0)
        exports = np.maximum(-net, 0.0)
        cost = DT * np.sum(
            (data["pa_eur_per_kwh"] + data["pb_eur_per_kwh"]) * imports
            - data["pa_eur_per_kwh"] * exports
        )
        period_solver = solver if period == overall_period else solver[
            solver["timestamp"].dt.normalize() == pd.Timestamp(period)
        ]
        terminal_deviations = []
        terminal_soc = {}
        if period != overall_period:
            last = data.iloc[-1]
            for bus in BUS_ORDER:
                cfg = BUS_CONFIG[bus]
                tolerance = PHASE4_SETTINGS["TERMINAL_ENERGY_TOL_FRAC"] * (
                    cfg["bess_e_max"] - cfg["bess_e_min"]
                )
                low = max(cfg["bess_e_min"], cfg["bess_e_init"] - tolerance)
                high = min(cfg["bess_e_max"], cfg["bess_e_init"] + tolerance)
                value = float(last[f"soc_{bus.lower().replace(' ', '_')}_kwh"])
                terminal_soc[f"terminal_soc_{bus.lower().replace(' ', '_')}_kwh"] = value
                terminal_soc[f"terminal_soc_deviation_{bus.lower().replace(' ', '_')}_kwh"] = abs(
                    value - cfg["bess_e_init"]
                )
                terminal_deviations.append(max(low - value, value - high, 0.0))
        rows.append({
            "period": period,
            "peak_kw": float(net.max()),
            "valley_kw": float(net.min()),
            "peak_to_valley_kw": float(np.ptp(net)),
            "uncontrolled_peak_to_valley_kw": float(np.ptp(uncontrolled)),
            "maximum_ramp_kw_per_15min": float(np.max(np.abs(np.diff(net)))) if len(net) > 1 else 0.0,
            "net_electricity_cost_eur": float(cost),
            "bess_throughput_kwh": float(DT * np.abs(data["bess_kw"]).sum()),
            "max_terminal_soc_band_violation_kwh": (
                float(max(terminal_deviations)) if terminal_deviations else np.nan
            ),
            "mean_mip_gap": float(period_solver["mip_gap"].mean()),
            "max_mip_gap": float(period_solver["mip_gap"].max()),
            "timeout_count": int(period_solver["time_limit_reached"].sum()),
            "gap_target_met_fraction": float(period_solver["gap_target_met"].mean()),
            "total_solver_time_s": float(period_solver["runtime_s"].sum()),
            **terminal_soc,
        })
    metrics = pd.DataFrame(rows)
    boundary_ramps = [
        abs(float(frame.iloc[boundary]["realized_net_kw"] - frame.iloc[boundary - 1]["realized_net_kw"]))
        for boundary in range(STEPS_PER_DAY, len(frame), STEPS_PER_DAY)
    ]
    overall = metrics["period"] == overall_period
    metrics.loc[overall, "max_midnight_ramp_kw_per_15min"] = (
        max(boundary_ramps) if boundary_ramps else np.nan
    )
    metrics.loc[overall, "total_wall_time_s"] = float(wall_time_s)
    if day_ahead_solver is not None and not day_ahead_solver.empty:
        metrics.loc[overall, "day_ahead_ol_solver_time_s"] = float(
            day_ahead_solver["runtime_s"].sum()
        )
        metrics.loc[overall, "day_ahead_ol_mean_mip_gap"] = float(
            day_ahead_solver["mip_gap"].mean()
        )
        metrics.loc[overall, "day_ahead_ol_max_mip_gap"] = float(
            day_ahead_solver["mip_gap"].max()
        )
    daily_violations = metrics.loc[~overall, "max_terminal_soc_band_violation_kwh"]
    metrics.loc[overall, "max_terminal_soc_band_violation_kwh"] = float(daily_violations.max())
    return metrics


def _plot_continuous(frame, output, label):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    hours = frame["step"].to_numpy() * DT
    controlled_mean = frame["realized_net_kw"].rolling(4, center=True, min_periods=1).mean()
    uncontrolled_mean = frame["uncontrolled_net_kw"].rolling(4, center=True, min_periods=1).mean()
    bess_mean = frame["bess_kw"].rolling(4, center=True, min_periods=1).mean()
    pb_mean = frame["pb_eur_per_kwh"].rolling(4, center=True, min_periods=1).mean()
    total_hours = CONTINUOUS_EXECUTION_DAYS * 24
    fig, axes = plt.subplots(4, 1, figsize=(20.0, 12.0), sharex=True)
    axes[0].plot(hours, frame["uncontrolled_net_kw"], color="#777777", ls="--", lw=0.75,
                 alpha=0.65, label="Uncontrolled (15 min)")
    axes[0].plot(hours, frame["realized_net_kw"], color="#0072B2", lw=0.75,
                 alpha=0.65, label="Controlled (15 min)")
    axes[0].plot(hours, uncontrolled_mean, color="#444444", ls="--", lw=1.35,
                 label="Uncontrolled (1 h mean)")
    axes[0].plot(hours, controlled_mean, color="#0072B2", lw=1.55,
                 label="Controlled (1 h mean)")
    axes[0].set_ylabel("Grid exchange (kW)"); axes[0].legend(frameon=False, ncol=4, loc="upper left")
    axes[1].plot(hours, frame["bess_kw"], color="#66C2A5", lw=0.7, alpha=0.55,
                 label="15 min")
    axes[1].plot(hours, bess_mean, color="#009E73", lw=1.5, label="1 h mean")
    axes[1].axhline(0.0, color="0.35", lw=0.6)
    axes[1].set_ylabel("BESS power (kW)")
    axes[1].legend(frameon=False, ncol=2, loc="upper left")
    axes[2].step(hours, frame["pb_eur_per_kwh"], where="post", color="#CC79A7", lw=0.75,
                 alpha=0.55, label="15 min")
    axes[2].plot(hours, pb_mean, color="#A64D79", lw=1.5, label="1 h mean")
    axes[2].set_ylabel("PB (EUR/kWh)"); axes[2].legend(frameon=False, ncol=2, loc="upper left")
    for bus, color in zip(BUS_ORDER, ("#0072B2", "#E69F00", "#009E73", "#CC79A7")):
        column = f"soc_{bus.lower().replace(' ', '_')}_kwh"
        axes[3].plot(hours, frame[column], color=color, lw=1.15, label=bus)
    axes[3].set_ylabel("BESS SOC (kWh)")
    axes[3].set_xlabel("Continuous time (h)")
    axes[3].legend(frameon=False, ncol=4, loc="upper left")
    for axis in axes:
        for day in range(CONTINUOUS_EXECUTION_DAYS):
            if day % 2:
                axis.axvspan(24 * day, 24 * (day + 1), color="#EAF2F8", alpha=0.32, zorder=0)
        for boundary in range(24, total_hours, 24):
            axis.axvline(boundary, color="0.30", ls=":", lw=1.0)
        axis.grid(True, ls="--", alpha=0.3, lw=0.5)
        axis.set_xlim(0, total_hours)
    axes[-1].set_xticks(np.arange(0, total_hours + 1, 6))
    for day in range(CONTINUOUS_EXECUTION_DAYS):
        center = 24 * day + 12
        date_text = str((CONTINUOUS_START_DATE + pd.Timedelta(days=day)).date())
        axes[0].text(center, 1.025, date_text, transform=axes[0].get_xaxis_transform(),
                     ha="center", va="bottom", fontsize=8, fontweight="bold")
    fig.suptitle(
        f"{label}: continuous {total_hours} h rolling Stackelberg MPC",
        fontsize=14, y=0.995,
    )
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.06, top=0.945, hspace=0.16)
    fig.savefig(
        output / f"stackelberg_mpc_overview_{CONTINUOUS_EXECUTION_DAYS}days.png",
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)


def _plot_daily(frame, output, label):
    for date, day in frame.groupby(frame["timestamp"].dt.normalize()):
        hours = np.arange(len(day)) * DT
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)
        axes = axes.ravel()
        axes[0].plot(hours, day["uncontrolled_net_kw"], color="#777777", ls=":", label="Uncontrolled")
        axes[0].plot(hours, day["realized_net_kw"], color="#0072B2", label="Controlled")
        axes[0].legend(frameon=False); axes[0].set_ylabel("Grid exchange (kW)")
        axes[1].plot(hours, day["bess_kw"], color="#009E73"); axes[1].set_ylabel("BESS power (kW)")
        axes[2].step(hours, day["pb_eur_per_kwh"], where="post", color="#CC79A7")
        axes[2].set_ylabel("PB (EUR/kWh)")
        for bus, color in zip(BUS_ORDER, ("#0072B2", "#E69F00", "#009E73", "#CC79A7")):
            axes[3].plot(hours, day[f"soc_{bus.lower().replace(' ', '_')}_kwh"], color=color, label=bus)
        axes[3].set_ylabel("SOC (kWh)"); axes[3].legend(frameon=False, ncol=2, fontsize=7)
        for axis in axes:
            axis.set_xlabel("Time (h)"); axis.grid(True, ls="--", alpha=0.3, lw=0.5); axis.set_xlim(0, 24)
        fig.suptitle(f"{label}: {date.date()}", fontsize=13); fig.tight_layout()
        fig.savefig(output / f"daily_overview_{date.date()}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def _run_continuous_experiment(key, execution_steps=None, output_override=None):
    config = dict(EXPERIMENTS[key])
    output = Path(output_override) if output_override is not None else config["output"]
    output.mkdir(parents=True, exist_ok=True)
    execution_steps = CONTINUOUS_EXECUTION_STEPS if execution_steps is None else int(execution_steps)
    if not 1 <= execution_steps <= CONTINUOUS_EXECUTION_STEPS:
        raise ValueError(f"execution_steps must be in [1, {CONTINUOUS_EXECUTION_STEPS}]")
    actual, pa, timestamps = _load_continuous_inputs()
    scenario = config["scenario"]
    reference_lookup, day_ahead_solver = _load_or_solve_day_ahead_reference(
        output, config, actual, pa, timestamps
    )
    _validate_reference_boundary_compatibility(reference_lookup)
    expected_meta = _continuous_metadata(config) | {"execution_steps_this_run": execution_steps}
    checkpoint_dir, state_path, meta_path, series_path, solver_path, forecast_path = _continuous_checkpoint_paths(output)
    energy = {bus: float(BUS_CONFIG[bus]["bess_e_init"]) for bus in BUS_ORDER}
    previous_pb = None; previous_pb_plan = None; previous_realized_net = None
    series_rows, solver_rows, forecast_rows = [], [], []
    start_origin = 0
    if state_path.exists() and meta_path.exists():
        saved_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if saved_meta != expected_meta:
            raise RuntimeError(f"Checkpoint metadata mismatch in {checkpoint_dir}")
        state = np.load(state_path)
        start_origin = int(state["origin_next"])
        previous_pb = float(state["previous_pb"])
        previous_pb_plan = state["previous_pb_plan"]
        previous_realized_net = float(state["previous_realized_net"])
        energy = {bus: float(state[f"energy_{index}"]) for index, bus in enumerate(BUS_ORDER)}
        series_rows = pd.read_csv(series_path, parse_dates=["timestamp"]).to_dict("records")
        solver_rows = pd.read_csv(solver_path, parse_dates=["timestamp"]).to_dict("records")
        forecast_rows = pd.read_csv(forecast_path, parse_dates=["origin_timestamp", "target_timestamp"]).to_dict("records")
        print(f"[{key}] resumed at global step {start_origin + 1}/{execution_steps}", flush=True)
    started = time.perf_counter()
    for origin in range(start_origin, execution_steps):
        forecast = _rolling_forecast(actual, scenario, origin)
        reference_energy_step = origin + HORIZON
        terminal_soc_bands = _soc_reference_bands(
            reference_lookup, reference_energy_step
        )
        limit = config.get("initial_time_limit_s") if origin < int(config.get("initial_long_steps", 0)) else config.get("time_limit_s")
        solution = _solve_stackelberg_window(
            timestamps[origin], origin, forecast, energy, pa[origin:origin + HORIZON],
            previous_pb, None, warm_pb=(None if previous_pb_plan is None else np.r_[previous_pb_plan[1:], previous_pb_plan[-1]]),
            mip_gap=config.get("mip_gap"), time_limit_s=limit, enforce_daily_terminal=True,
            terminal_soc_bands=terminal_soc_bands,
            previous_realized_net=previous_realized_net,
        )
        uncontrolled = sum(actual["load"][bus][origin] - actual["pv"][bus][origin] for bus in BUS_ORDER)
        bess = sum(solution["actions"].values()); realized = uncontrolled + bess
        for bus in BUS_ORDER:
            energy[bus] += DT * solution["actions"][bus]
        previous_pb = float(solution["pb"][0]); previous_pb_plan = solution["pb"].copy()
        previous_realized_net = float(realized)
        series_rows.append({
            "step": origin, "timestamp": timestamps[origin], "pa_eur_per_kwh": pa[origin],
            "pb_eur_per_kwh": previous_pb, "uncontrolled_net_kw": uncontrolled,
            "realized_net_kw": realized, "bess_kw": bess,
            "soc_reference_energy_step": reference_energy_step,
            **{
                f"soc_reference_{bus.lower().replace(' ', '_')}_kwh":
                0.5 * sum(terminal_soc_bands[bus])
                for bus in BUS_ORDER
            },
            **{f"soc_{bus.lower().replace(' ', '_')}_kwh": energy[bus] for bus in BUS_ORDER},
        })
        solver_rows.append({
            "step": origin, "timestamp": timestamps[origin], "status": solution["status"],
            "objective": solution["objective"], "best_bound": solution["best_bound"],
            "mip_gap": solution["gap"], "runtime_s": solution["runtime_s"], "nodes": solution["nodes"],
            "time_limit_s": limit, "time_limit_reached": solution["status"] == int(GRB.TIME_LIMIT),
            "gap_target_met": solution["gap"] <= config.get("mip_gap", 0.05) + 1e-9,
        })
        for lead in range(HORIZON):
            for bus in BUS_ORDER:
                forecast_rows.append({
                    "origin_step": origin, "target_step": origin + lead, "lead_step": lead,
                    "origin_timestamp": timestamps[origin], "target_timestamp": timestamps[origin + lead],
                    "bus": bus, "load_real_kw": actual["load"][bus][origin + lead],
                    "load_forecast_kw": forecast["load"][bus][lead], "pv_real_kw": actual["pv"][bus][origin + lead],
                    "pv_forecast_kw": forecast["pv"][bus][lead],
                })
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
        np.savez(state_path, origin_next=origin + 1, previous_pb=previous_pb,
                 previous_pb_plan=previous_pb_plan, previous_realized_net=previous_realized_net,
                 **{f"energy_{index}": energy[bus] for index, bus in enumerate(BUS_ORDER)})
        pd.DataFrame(series_rows).to_csv(series_path, index=False)
        pd.DataFrame(solver_rows).to_csv(solver_path, index=False)
        pd.DataFrame(forecast_rows).to_csv(forecast_path, index=False)
        print(f"[{key}] step {origin + 1:03d}/{execution_steps}, gap={solution['gap']:.3%}, "
              f"PB={previous_pb:.4f}, net={realized:.3f} kW, time={solution['runtime_s']:.1f}s", flush=True)
    frame = pd.DataFrame(series_rows); frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    solver = pd.DataFrame(solver_rows); forecasts = pd.DataFrame(forecast_rows)
    frame.to_csv(output / "mpc_timeseries.csv", index=False)
    solver.to_csv(output / "solver_log.csv", index=False)
    forecasts.to_csv(output / "rolling_forecasts.csv", index=False)
    if execution_steps == CONTINUOUS_EXECUTION_STEPS:
        metrics = _continuous_metrics(
            frame, solver, time.perf_counter() - started, day_ahead_solver
        )
        metrics.iloc[1:].to_csv(output / "daily_metrics.csv", index=False)
        metrics.iloc[[0]].to_csv(output / "overall_metrics.csv", index=False)
        _plot_continuous(frame, output, config["label"])
    print(f"[{key}] complete in {(time.perf_counter() - started) / 60:.1f} min -> {output}", flush=True)
    return frame


def run_experiment(key, execution_steps=None, output_override=None):
    if key in ("t5", "t6"):
        return _run_continuous_experiment(key, execution_steps=execution_steps, output_override=output_override)
    config = EXPERIMENTS[key]
    output = config["output"]
    output.mkdir(parents=True, exist_ok=True)
    frame = _load_inputs()
    dates = sorted(frame["date"].unique())
    if config.get("dates"):
        selected_dates = {pd.Timestamp(value).normalize() for value in config["dates"]}
        dates = [value for value in dates if pd.Timestamp(value).normalize() in selected_dates]
        missing = selected_dates.difference(pd.Timestamp(value).normalize() for value in dates)
        if missing:
            raise KeyError(f"Requested experiment dates are unavailable: {sorted(missing)}")
    pa_year, _ = read_pa_year(dict(PHASE4_SETTINGS))
    all_metrics, all_rows, all_solver, all_forecasts = [], [], [], []
    print(f"[{config['label']}] {len(dates)} dates; 96 rolling Stackelberg updates/date", flush=True)
    for date_index, value in enumerate(dates, 1):
        date = pd.Timestamp(value)
        print(f"\n[{config['label']}] date {date_index}/{len(dates)}: {date.date()}", flush=True)
        data = _day_arrays(frame, date)
        pa = pa_year.loc[(pa_year.index >= date) & (pa_year.index < date + pd.Timedelta(days=1))].to_numpy(float)
        result, pb, rows, solver_rows, forecasts, runtime = _simulate_day(config, date, data, pa, output)
        metric = _metric_row(key, config, date, result, pa, pb, solver_rows, forecasts, runtime)
        all_metrics.append(metric); all_rows.extend(rows); all_solver.extend(solver_rows); all_forecasts.extend(forecasts)
        pd.DataFrame(all_metrics).to_csv(output / "daily_metrics.csv", index=False)
        pd.DataFrame(all_rows).to_csv(output / "mpc_timeseries.csv", index=False)
        pd.DataFrame(all_solver).to_csv(output / "solver_log.csv", index=False)
        pd.DataFrame(all_forecasts).to_csv(output / "rolling_forecasts.csv", index=False)
        _plot(
            pd.DataFrame(all_rows),
            config["label"] + " (completed dates)",
            output / "stackelberg_mpc_overview.png",
        )
        print(f"[{config['label']}] completed {date.date()} in {runtime:.1f}s; realized range={metric['peak_to_valley_kw']:.3f} kW", flush=True)
    timeseries = pd.DataFrame(all_rows)
    _plot(timeseries, config["label"], output / "stackelberg_mpc_overview.png")
    print(f"[{config['label']}] results: {output}", flush=True)
    return pd.DataFrame(all_metrics)
