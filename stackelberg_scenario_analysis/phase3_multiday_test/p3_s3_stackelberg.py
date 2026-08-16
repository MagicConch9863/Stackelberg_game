"""Phase 3 S3: 31 independent daily Stackelberg problems using Big-M KKT."""
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

from p3_common import (
    BUS_CONFIG, BUS_ORDER, DEFAULT_SETTINGS, DT, T, day_frame, day_inputs,
    day_metrics, experiment_dates, load_fixed_simbench_day, plot_month,
    read_pa_month, read_pv_month, save_month, trip_vector,
)


def follower_lp(bus, baseline, settings):
    """Build Aeq*x=beq, Aub*x<=bub; x=[BESS, EV, import, export]."""
    cfg = BUS_CONFIG[bus]
    q = 4 * T
    aeq, beq, aub, bub = [], [], [], []

    for t in range(T):
        row = np.zeros(q)
        row[t] = 1.0
        row[T + t] = 1.0
        row[2 * T + t] = -1.0
        row[3 * T + t] = 1.0
        aeq.append(row)
        beq.append(-baseline[t])

    for k in range(1, T + 1):
        row = np.zeros(q)
        row[:k] = DT
        aub.extend((row, -row))
        bub.extend((cfg["bess_e_max"] - cfg["bess_e_init"],
                    cfg["bess_e_init"] - cfg["bess_e_min"]))

    bess_tol = settings["TERMINAL_ENERGY_TOL_FRAC"] * (
        cfg["bess_e_max"] - cfg["bess_e_min"])
    row = np.zeros(q)
    row[:T] = DT
    aub.extend((row, -row))
    bub.extend((bess_tol, bess_tol))

    if "ev_kwh" in cfg:
        cumulative_trip = np.r_[0.0, np.cumsum(trip_vector(bus))]
        for k in range(1, T + 1):
            row = np.zeros(q)
            row[T:T + k] = DT
            aub.extend((row, -row))
            bub.extend((0.3 * cfg["ev_kwh"] + cumulative_trip[k],
                        0.4 * cfg["ev_kwh"] - cumulative_trip[k]))
        ev_tol = settings["TERMINAL_ENERGY_TOL_FRAC"] * 0.7 * cfg["ev_kwh"]
        row = np.zeros(q)
        row[T:2 * T] = DT
        aub.extend((row, -row))
        bub.extend((cfg["ev_trip_kwh"] + ev_tol,
                    -cfg["ev_trip_kwh"] + ev_tol))

    grid_bound = float(settings["GRID_BOUND_KW"])
    lb = np.r_[np.full(T, -cfg["bess_kw"]), np.zeros(T),
               np.zeros(T), np.zeros(T)]
    ub = np.r_[np.full(T, cfg["bess_kw"]), np.zeros(T),
               np.full(T, grid_bound), np.full(T, grid_bound)]
    if "ev_kwh" in cfg:
        for t in range(T):
            hour = t * DT
            if hour < 7 or hour >= 18:
                lb[T + t], ub[T + t] = 0.0, cfg["ev_kw_ch"]
            elif 8 <= hour < 17:
                lb[T + t], ub[T + t] = -cfg["ev_kw_dis"], 0.0
    return tuple(map(np.asarray, (aeq, beq, aub, bub))) + (lb, ub)


def complementarity(model, slack_expression, slack_max, dual, name, dual_max):
    maximum = max(float(slack_max), 0.0)
    binary = model.addVar(vtype=GRB.BINARY, name=f"z_{name}")
    slack = model.addVar(lb=0.0, ub=maximum, name=f"slack_{name}")
    model.addConstr(slack == slack_expression)
    model.addConstr(slack <= maximum * binary)
    model.addConstr(dual <= dual_max * (1.0 - binary))


def add_follower_kkt(model, bus, baseline, pa, pb, settings):
    aeq, beq, aub, bub, lb, ub = follower_lp(bus, baseline, settings)
    q = 4 * T
    tag = bus.replace(" ", "_")
    x = model.addVars(q, lb=-GRB.INFINITY, name=f"x_{tag}")
    lam = model.addVars(len(beq), lb=-GRB.INFINITY, name=f"lambda_{tag}")
    mu = model.addVars(len(bub), lb=0.0, ub=settings["DUAL_BIG_M"],
                        name=f"mu_{tag}")

    for i, row in enumerate(aeq):
        nz = np.flatnonzero(row)
        model.addConstr(gp.quicksum(row[j] * x[j] for j in nz) == beq[i])
    for i, row in enumerate(aub):
        nz = np.flatnonzero(row)
        lhs = gp.quicksum(row[j] * x[j] for j in nz)
        model.addConstr(lhs <= bub[i])
        minimum_lhs = sum(row[j] * (lb[j] if row[j] >= 0 else ub[j]) for j in nz)
        complementarity(model, bub[i] - lhs, bub[i] - minimum_lhs, mu[i],
                        f"ineq_{tag}_{i}", settings["DUAL_BIG_M"])

    lower_dual, upper_dual = {}, {}
    for j in range(q):
        model.addConstr(x[j] >= lb[j])
        model.addConstr(x[j] <= ub[j])
        lower_dual[j] = model.addVar(lb=0.0, ub=settings["DUAL_BIG_M"],
                                     name=f"dl_{tag}_{j}")
        upper_dual[j] = model.addVar(lb=0.0, ub=settings["DUAL_BIG_M"],
                                     name=f"du_{tag}_{j}")
        span = ub[j] - lb[j]
        complementarity(model, x[j] - lb[j], span, lower_dual[j],
                        f"lb_{tag}_{j}", settings["DUAL_BIG_M"])
        complementarity(model, ub[j] - x[j], span, upper_dual[j],
                        f"ub_{tag}_{j}", settings["DUAL_BIG_M"])

    eps = float(settings["FLOW_TIE_BREAK_EUR_PER_KWH"])
    export_mode = settings["EXPORT_PRICE_MODE"].upper()
    for j in range(q):
        if j < 2 * T:
            cost = 0.0
        elif j < 3 * T:
            t = j - 2 * T
            cost = DT * (pa[t] + pb[t] + eps)
        else:
            t = j - 3 * T
            if export_mode == "PA":
                cost = DT * (-pa[t] + eps)
            elif export_mode == "PB":
                cost = DT * (-pb[t] + eps)
            else:
                raise ValueError("EXPORT_PRICE_MODE must be 'PA' or 'PB'")
        equality_terms = gp.quicksum(
            aeq[i, j] * lam[i] for i in np.flatnonzero(aeq[:, j]))
        inequality_terms = gp.quicksum(
            aub[i, j] * mu[i] for i in np.flatnonzero(aub[:, j]))
        model.addConstr(cost + equality_terms + inequality_terms
                        - lower_dual[j] + upper_dual[j] == 0.0)
    return x


def solve_day(date, loads, pv, pa, settings):
    model = gp.Model(f"P3_S3_{pd.Timestamp(date):%Y%m%d}")
    model.Params.OutputFlag = settings["SOLVER_OUTPUT"]
    model.Params.MIPGap = settings["MIP_GAP"]
    model.Params.TimeLimit = settings["TIME_LIMIT_S_PER_DAY"]
    model.Params.NumericFocus = 1
    model.Params.MIPFocus = 1
    model.Params.Heuristics = 0.20
    model.Params.Cuts = 2
    model.Params.Seed = 1

    pb = model.addVars(T, lb=settings["PB_MIN"], ub=settings["PB_MAX"], name="PB")
    if settings["ENFORCE_PB_DAILY_MEAN"]:
        target = float(settings["PB_DAILY_MEAN"])
        if not settings["PB_MIN"] <= target <= settings["PB_MAX"]:
            raise ValueError("PB_DAILY_MEAN must lie inside [PB_MIN, PB_MAX]")
        model.addConstr(gp.quicksum(pb[t] for t in range(T)) == T * target,
                        name="PB_daily_mean")
    for t in range(1, T):
        model.addConstr(pb[t] - pb[t - 1] <= settings["PB_RAMP"])
        model.addConstr(pb[t - 1] - pb[t] <= settings["PB_RAMP"])

    x = {
        bus: add_follower_kkt(model, bus, loads[bus] - pv[bus], pa, pb, settings)
        for bus in BUS_ORDER
    }
    peak = model.addVar(lb=-GRB.INFINITY, name="aggregate_peak")
    valley = model.addVar(lb=-GRB.INFINITY, name="aggregate_valley")
    for t in range(T):
        aggregate = gp.quicksum(
            x[bus][2 * T + t] - x[bus][3 * T + t] for bus in BUS_ORDER)
        model.addConstr(peak >= aggregate)
        model.addConstr(valley <= aggregate)
    model.setObjective(peak - valley, GRB.MINIMIZE)
    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError(
            f"S3 {pd.Timestamp(date).date()}: no incumbent, status={model.Status}")

    pb_values = np.array([pb[t].X for t in range(T)])
    result = {}
    for bus in BUS_ORDER:
        values = np.array([x[bus][j].X for j in range(4 * T)])
        bess, ev, imp, exp = np.split(values, 4)
        cfg = BUS_CONFIG[bus]
        item = {
            "bess": bess, "ev": ev, "import": imp, "export": exp,
            "net": imp - exp,
            "bess_energy": cfg["bess_e_init"]
                           + np.r_[0.0, np.cumsum(DT * bess)],
            "sell_price_mode": settings["EXPORT_PRICE_MODE"].upper(),
        }
        item["sell_price"] = (pa if settings["EXPORT_PRICE_MODE"].upper() == "PA"
                              else pb_values)
        if "ev_kwh" in cfg:
            item["ev_energy"] = (0.6 * cfg["ev_kwh"]
                + np.r_[0.0, np.cumsum(DT * ev - trip_vector(bus))])
        result[bus] = item
    gap = float(model.MIPGap) if model.IsMIP else 0.0
    return result, pb_values, gap


def main(settings):
    loads = load_fixed_simbench_day(settings)
    pa_month = read_pa_month(settings)
    pv_month = read_pv_month(settings)
    frames, metrics = [], []
    for number, date in enumerate(experiment_dates(settings), 1):
        _, pv, pa = day_inputs(date, loads, pa_month, pv_month, settings)
        result, pb, gap = solve_day(date, loads, pv, pa, settings)
        frame = day_frame(date, loads, pv, result, pa, pb)
        frames.append(frame)
        row = day_metrics("s3", date, frame, result, pa, pb)
        row["mip_gap"] = gap
        row["pb_mean_eur_per_kwh"] = float(np.mean(pb))
        row["pb_range_eur_per_kwh"] = float(np.ptp(pb))
        metrics.append(row)
        print(f"S3 {number:02d}: {date.date()}, range={row['peak_valley_kw']:.4f} "
              f"kW, gap={gap:.3%}")

    ts_path, metric_path = save_month("s3", frames, metrics)
    month_frame = pd.concat(frames, ignore_index=True)
    metric_frame = pd.DataFrame(metrics)
    figure = plot_month("s3", "Phase 3 S3: Stackelberg coordination",
                        month_frame, metric_frame, show_price=True)
    print(f"Timeseries: {ts_path}\nMetrics: {metric_path}\nFigure: {figure}")


if __name__ == "__main__":
    SETTINGS = {
        **DEFAULT_SETTINGS,
        "PB_MIN": 0.0,
        "PB_MAX": 0.30,
        "PB_RAMP": 0.15,
        "ENFORCE_PB_DAILY_MEAN": False,
        "PB_DAILY_MEAN": 0.10,
        "EXPORT_PRICE_MODE": "PA",
        "GRID_BOUND_KW": 20.0,
        "DUAL_BIG_M": 1000.0,
        "MIP_GAP": 0.05,
        "TIME_LIMIT_S_PER_DAY": 600,
        "SOLVER_OUTPUT": 1,
    }
    main(SETTINGS)
