"""C-B S3: one leader/four load-integrated followers, Big-M KKT reformulation."""
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from cb_common import (BUS_CONFIG, BUS_ORDER, DEFAULT_SETTINGS, DT, T,
                       load_inputs, output_folders, plot_overview,
                       save_timeseries, trip_vector)


def follower_lp(bus, load, settings):
    """Return Aeq*x=beq, Aub*x<=bub and bounds; x=[BESS, EV, import, export]."""
    cfg, q = BUS_CONFIG[bus], 4*T
    aeq, beq, aub, bub = [], [], [], []
    for t in range(T):
        row = np.zeros(q)
        row[t], row[T+t], row[2*T+t], row[3*T+t] = 1, 1, -1, 1
        aeq.append(row); beq.append(-load[t])
    grid_bound = settings["GRID_BOUND_KW"]
    for k in range(1, T+1):
        row = np.zeros(q); row[:k] = DT
        aub.extend((row, -row))
        bub.extend((cfg["bess_e_max"]-cfg["bess_e_init"],
                    cfg["bess_e_init"]-cfg["bess_e_min"]))
    # End-of-day BESS energy may deviate by 5% of its usable energy range.
    bess_terminal_tol = settings["TERMINAL_ENERGY_TOL_FRAC"] * (
        cfg["bess_e_max"]-cfg["bess_e_min"])
    row = np.zeros(q); row[:T] = DT
    aub.extend((row, -row)); bub.extend((bess_terminal_tol, bess_terminal_tol))
    if "ev_kwh" in cfg:
        cumulative_trip = np.r_[0.0, np.cumsum(trip_vector(bus))]
        for k in range(1, T+1):
            row = np.zeros(q); row[T:T+k] = DT
            aub.extend((row, -row))
            bub.extend((.3*cfg["ev_kwh"] + cumulative_trip[k],
                        .4*cfg["ev_kwh"] - cumulative_trip[k]))
        # Ee[T]-Ee[0] = sum(EV*DT)-trip may deviate by 5% of usable EV energy.
        ev_terminal_tol = settings["TERMINAL_ENERGY_TOL_FRAC"] * .7*cfg["ev_kwh"]
        row = np.zeros(q); row[T:2*T] = DT
        aub.extend((row, -row))
        bub.extend((cfg["ev_trip_kwh"] + ev_terminal_tol,
                    -cfg["ev_trip_kwh"] + ev_terminal_tol))
    lb = np.r_[np.full(T, -cfg["bess_kw"]), np.zeros(T), np.zeros(T), np.zeros(T)]
    ub = np.r_[np.full(T, cfg["bess_kw"]), np.zeros(T),
               np.full(T, grid_bound), np.full(T, grid_bound)]
    if "ev_kwh" in cfg:
        for t in range(T):
            hour = t*DT
            if hour < 7 or hour >= 18:
                lb[T+t], ub[T+t] = 0, cfg["ev_kw_ch"]
            elif 8 <= hour < 17:
                lb[T+t], ub[T+t] = -cfg["ev_kw_dis"], 0
    return tuple(map(np.asarray, (aeq, beq, aub, bub))) + (lb, ub)


def complementarity(model, slack_expression, slack_max, dual, name, dual_max):
    maximum = max(float(slack_max), 0.0)
    z = model.addVar(vtype=GRB.BINARY, name=f"z_{name}")
    slack = model.addVar(lb=0, ub=maximum, name=f"slack_{name}")
    model.addConstr(slack == slack_expression)
    model.addConstr(slack <= maximum*z)
    model.addConstr(dual <= dual_max*(1-z))


def add_follower_kkt(model, bus, load, pa, pb, settings):
    Aeq, beq, Aub, bub, lb, ub = follower_lp(bus, load, settings)
    q, tag = 4*T, bus.replace(" ", "_")
    x = model.addVars(q, lb=-GRB.INFINITY, name=f"x_{tag}")
    lam = model.addVars(len(beq), lb=-GRB.INFINITY, name=f"lambda_{tag}")
    mu = model.addVars(len(bub), lb=0, ub=settings["DUAL_BIG_M"], name=f"mu_{tag}")
    for i, row in enumerate(Aeq):
        nz = np.flatnonzero(row)
        model.addConstr(gp.quicksum(row[j]*x[j] for j in nz) == beq[i])
    for i, row in enumerate(Aub):
        nz = np.flatnonzero(row)
        lhs = gp.quicksum(row[j]*x[j] for j in nz)
        model.addConstr(lhs <= bub[i])
        minimum_lhs = sum(row[j]*(lb[j] if row[j] >= 0 else ub[j]) for j in nz)
        complementarity(model, bub[i]-lhs, bub[i]-minimum_lhs, mu[i],
                        f"ineq_{tag}_{i}", settings["DUAL_BIG_M"])
    lower_dual, upper_dual = {}, {}
    for j in range(q):
        model.addConstr(x[j] >= lb[j]); model.addConstr(x[j] <= ub[j])
        lower_dual[j] = model.addVar(lb=0, ub=settings["DUAL_BIG_M"], name=f"dl_{tag}_{j}")
        upper_dual[j] = model.addVar(lb=0, ub=settings["DUAL_BIG_M"], name=f"du_{tag}_{j}")
        span = ub[j]-lb[j]
        complementarity(model, x[j]-lb[j], span, lower_dual[j],
                        f"lb_{tag}_{j}", settings["DUAL_BIG_M"])
        complementarity(model, ub[j]-x[j], span, upper_dual[j],
                        f"ub_{tag}_{j}", settings["DUAL_BIG_M"])
    for j in range(q):
        eps = float(settings["FLOW_TIE_BREAK_EUR_PER_KWH"])
        if j < 2*T:
            # BESS and EV have no cycling, smoothing or tie-break cost.
            cost = 0
        elif 2*T <= j < 3*T:
            t = j-2*T
            cost = DT*(pa[t]+pb[t]+eps)
        elif 3*T <= j:
            t = j-3*T
            if settings["EXPORT_PRICE_MODE"].upper() == "PA":
                cost = DT*(-pa[t]+eps)
            elif settings["EXPORT_PRICE_MODE"].upper() == "PB":
                cost = DT*(-pb[t]+eps)
            else:
                raise ValueError("EXPORT_PRICE_MODE must be 'PA' or 'PB'")
        equality_terms = gp.quicksum(Aeq[i, j]*lam[i] for i in np.flatnonzero(Aeq[:, j]))
        inequality_terms = gp.quicksum(Aub[i, j]*mu[i] for i in np.flatnonzero(Aub[:, j]))
        model.addConstr(cost + equality_terms + inequality_terms
                        - lower_dual[j] + upper_dual[j] == 0)
    return x


def main(settings):
    csv_folder, figure_folder = output_folders()
    loads, pv, pa = load_inputs(settings, need_price=True)
    m = gp.Model("CB_S3_one_leader_four_followers")
    m.Params.OutputFlag = settings["SOLVER_OUTPUT"]
    m.Params.MIPGap = settings["MIP_GAP"]
    m.Params.TimeLimit = settings["TIME_LIMIT_S"]
    m.Params.NumericFocus = 1
    m.Params.MIPFocus = 1
    m.Params.Heuristics = .20
    m.Params.Cuts = 2
    m.Params.Seed = 1
    # 96 independent PB values: one for every 15-minute interval.
    pb = m.addVars(T, lb=settings["PB_MIN"], ub=settings["PB_MAX"], name="PB")
    # Optional daily mean; disabled by default, so PB otherwise remains free.
    if settings["ENFORCE_PB_DAILY_MEAN"]:
        target = float(settings["PB_DAILY_MEAN"])
        if not settings["PB_MIN"] <= target <= settings["PB_MAX"]:
            raise ValueError("PB_DAILY_MEAN must lie inside [PB_MIN, PB_MAX]")
        m.addConstr(gp.quicksum(pb[t] for t in range(T)) == T*target,
                    name="PB_daily_mean")
    for t in range(1, T):
        m.addConstr(pb[t]-pb[t-1] <= settings["PB_RAMP"])
        m.addConstr(pb[t-1]-pb[t] <= settings["PB_RAMP"])
    x = {bus: add_follower_kkt(m, bus, loads[bus]-pv[bus], pa, pb, settings)
         for bus in BUS_ORDER}
    peak = m.addVar(lb=-GRB.INFINITY, name="aggregate_peak")
    valley = m.addVar(lb=-GRB.INFINITY, name="aggregate_valley")
    for t in range(T):
        aggregate = gp.quicksum(x[bus][2*T+t]-x[bus][3*T+t] for bus in BUS_ORDER)
        m.addConstr(peak >= aggregate)
        m.addConstr(valley <= aggregate)
    m.setObjective(peak-valley, GRB.MINIMIZE)
    m.optimize()
    if m.SolCount == 0:
        raise RuntimeError(f"S3: no feasible incumbent, status={m.Status}")
    result = {}
    for bus in BUS_ORDER:
        values = np.array([x[bus][j].X for j in range(4*T)])
        bess, ev, imp, exp = np.split(values, 4)
        cfg = BUS_CONFIG[bus]
        item = {"bess": bess, "ev": ev, "import": imp, "export": exp,
                "net": imp-exp,
                "bess_energy": cfg["bess_e_init"] + np.r_[0.0, np.cumsum(DT*bess)]}
        item["sell_price"] = pa if settings["EXPORT_PRICE_MODE"].upper() == "PA" else None
        item["sell_price_mode"] = settings["EXPORT_PRICE_MODE"].upper()
        if "ev_kwh" in cfg:
            item["ev_energy"] = .6*cfg["ev_kwh"] + np.r_[0.0, np.cumsum(DT*ev-trip_vector(bus))]
        result[bus] = item
    pb_values = np.array([pb[t].X for t in range(T)])
    if settings["EXPORT_PRICE_MODE"].upper() == "PB":
        for bus in BUS_ORDER:
            result[bus]["sell_price"] = pb_values
    max_simultaneous = max(float(np.max(np.minimum(result[bus]["import"],
                                                    result[bus]["export"])))
                           for bus in BUS_ORDER)
    aggregate = save_timeseries(csv_folder / "s3_timeseries.csv", loads, pv,
                                result, pa, pb_values)
    figure = plot_overview(figure_folder, "s3", "C-B S3: Stackelberg coordination",
                           loads, pv, result, pa, pb_values)
    gap = float(m.MIPGap)
    print(f"S3 complete: peak-valley={np.ptp(aggregate):.4f} kW, gap={gap:.4%}")
    print(f"PB mean={pb_values.mean():.5f}, PB range={np.ptp(pb_values):.5f}, "
          f"PA+PB range={np.ptp(pa+pb_values):.5f} EUR/kWh")
    print(f"Maximum simultaneous import/export={max_simultaneous:.8f} kW")
    if max_simultaneous > 1e-5:
        print("WARNING: increase FLOW_TIE_BREAK_EUR_PER_KWH slightly.")
    print(f"CSV: {csv_folder}\nFigure: {figure}")


if __name__ == "__main__":
    SETTINGS = {
        **DEFAULT_SETTINGS,
        "PB_MIN": 0.0,
        "PB_MAX": 0.30,
        "PB_RAMP": 0.15,
        "ENFORCE_PB_DAILY_MEAN": False,
        "PB_DAILY_MEAN": 0.10,
        "EXPORT_PRICE_MODE": "PA",
        "FLOW_TIE_BREAK_EUR_PER_KWH": 1e-5,
        "TERMINAL_ENERGY_TOL_FRAC": 0.05,
        "GRID_BOUND_KW": 20.0,
        "DUAL_BIG_M": 1000.0,
        "MIP_GAP": 0.02,
        "TIME_LIMIT_S": 1200,
        "SOLVER_OUTPUT": 1,
    }
    main(SETTINGS)
