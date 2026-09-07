from __future__ import annotations

import json
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


SCENARIO_FILE = ROOT / "results" / "control_inputs" / "bus_scenarios" / "forecast_scenarios.csv"
STEPS_PER_DAY = 96
HORIZON = 16
TIME_LIMIT_S_PER_UPDATE = 600.0

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
        "output": ROOT / "t5_iid_mpc_mpc" / "results",
        "dates": ("2025-01-15",),
        "mip_gap": 0.05,
        "time_limit_s": 600.0,
        "initial_time_limit_s": 1200.0,
        "initial_long_steps": 2,
    },
    "t6": {
        "label": "T6 AR(1) Gaussian: tariff MPC + BESS MPC",
        "scenario": "ar1",
        "output": ROOT / "t6_ar1_mpc_mpc" / "results",
        "dates": ("2025-01-15",),
        "mip_gap": 0.05,
        "time_limit_s": 600.0,
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


def _follower_lp(bus, baseline, current_energy, origin_step, day_steps=STEPS_PER_DAY):
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
    for boundary in range(first_boundary, end_step + 1, day_steps):
        count = boundary - int(origin_step)
        boundary_row = np.zeros(q)
        boundary_row[:count] = DT
        aub.extend((boundary_row, -boundary_row))
        bub.extend((target_high - current_energy, current_energy - target_low))

    # Before the next midnight enters the horizon, retain only the necessary
    # reachability band so the daily target cannot become physically impossible.
    next_boundary = ((end_step + day_steps - 1) // day_steps) * day_steps
    if next_boundary == end_step:
        # Already enforced above; retain these values for diagnostics only.
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


def _add_follower_kkt(model, bus, baseline, current_energy, pa, pb, origin_step):
    packed = _follower_lp(bus, baseline, current_energy, origin_step)
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
    warm_pb=None, mip_gap=None, time_limit_s=None,
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
    # The trivial nonnegative peak-to-valley bound is often weak. Prioritize
    # bound improvement and exact MILP cuts rather than additional heuristics.
    model.Params.MIPFocus = 3
    model.Params.Cuts = 3
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
            model, bus, baseline, energies[bus], pa, pb, origin
        )

    peak = model.addVar(lb=-GRB.INFINITY, name="peak")
    valley = model.addVar(lb=-GRB.INFINITY, name="valley")
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


def run_experiment(key):
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
