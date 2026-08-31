import gurobipy as gp
from gurobipy import GRB
import numpy as np

from p4_common import BUS_CONFIG, BUS_ORDER, DT, T, make_result_item


def follower_lp(bus, baseline, settings):
    cfg = BUS_CONFIG[bus]
    q = 3 * T

    aeq, beq, aub, bub = [], [], [], []

    # import - export = baseline + BESS
    for t in range(T):
        row = np.zeros(q)
        row[t] = 1.0
        row[T + t] = -1.0
        row[2 * T + t] = 1.0
        aeq.append(row)
        beq.append(-baseline[t])

    # Cumulative BESS energy limits.
    for k in range(1, T + 1):
        row = np.zeros(q)
        row[:k] = DT
        aub.extend((row, -row))
        bub.extend((
            cfg["bess_e_max"] - cfg["bess_e_init"],
            cfg["bess_e_init"] - cfg["bess_e_min"],
        ))

    # Terminal-energy tolerance.
    terminal_tol = settings["TERMINAL_ENERGY_TOL_FRAC"] * (
        cfg["bess_e_max"] - cfg["bess_e_min"]
    )
    row = np.zeros(q)
    row[:T] = DT
    aub.extend((row, -row))
    bub.extend((terminal_tol, terminal_tol))

    grid_bound = float(settings["GRID_BOUND_KW"])

    lb = np.r_[
        np.full(T, -cfg["bess_kw"]),
        np.zeros(T),
        np.zeros(T),
    ]
    ub = np.r_[
        np.full(T, cfg["bess_kw"]),
        np.full(T, grid_bound),
        np.full(T, grid_bound),
    ]

    return tuple(map(np.asarray, (aeq, beq, aub, bub))) + (lb, ub)


def complementarity(model, slack_expr, slack_max, dual, name, dual_max):
    maximum = max(float(slack_max), 0.0)
    binary = model.addVar(vtype=GRB.BINARY, name=f"z_{name}")
    slack = model.addVar(lb=0.0, ub=maximum, name=f"slack_{name}")

    model.addConstr(slack == slack_expr)
    model.addConstr(slack <= maximum * binary)
    model.addConstr(dual <= dual_max * (1.0 - binary))


def add_kkt(model, bus, baseline, pa, pb, settings):
    aeq, beq, aub, bub, lb, ub = follower_lp(bus, baseline, settings)
    q = 3 * T
    tag = bus.replace(" ", "_")

    x = model.addVars(q, lb=-GRB.INFINITY, name=f"x_{tag}")
    lam = model.addVars(len(beq), lb=-GRB.INFINITY, name=f"lam_{tag}")
    mu = model.addVars(
        len(bub),
        lb=0.0,
        ub=settings["DUAL_BIG_M"],
        name=f"mu_{tag}",
    )

    for i, row in enumerate(aeq):
        nz = np.flatnonzero(row)
        model.addConstr(
            gp.quicksum(row[j] * x[j] for j in nz) == beq[i]
        )

    for i, row in enumerate(aub):
        nz = np.flatnonzero(row)
        lhs = gp.quicksum(row[j] * x[j] for j in nz)
        model.addConstr(lhs <= bub[i])

        minimum_lhs = sum(
            row[j] * (lb[j] if row[j] >= 0 else ub[j])
            for j in nz
        )

        complementarity(
            model,
            bub[i] - lhs,
            bub[i] - minimum_lhs,
            mu[i],
            f"ineq_{tag}_{i}",
            settings["DUAL_BIG_M"],
        )

    lower_dual, upper_dual = {}, {}

    for j in range(q):
        model.addConstr(x[j] >= lb[j])
        model.addConstr(x[j] <= ub[j])

        lower_dual[j] = model.addVar(
            lb=0.0,
            ub=settings["DUAL_BIG_M"],
            name=f"dl_{tag}_{j}",
        )
        upper_dual[j] = model.addVar(
            lb=0.0,
            ub=settings["DUAL_BIG_M"],
            name=f"du_{tag}_{j}",
        )

        span = ub[j] - lb[j]

        complementarity(
            model,
            x[j] - lb[j],
            span,
            lower_dual[j],
            f"lb_{tag}_{j}",
            settings["DUAL_BIG_M"],
        )
        complementarity(
            model,
            ub[j] - x[j],
            span,
            upper_dual[j],
            f"ub_{tag}_{j}",
            settings["DUAL_BIG_M"],
        )

    eps = float(settings["FLOW_TIE_BREAK_EUR_PER_KWH"])

    for j in range(q):
        if j < T:
            cost = 0.0
        elif j < 2 * T:
            t = j - T
            cost = DT * (pa[t] + pb[t] + eps)
        else:
            t = j - 2 * T
            cost = DT * (-pa[t] + eps)

        equality_terms = gp.quicksum(
            aeq[i, j] * lam[i]
            for i in np.flatnonzero(aeq[:, j])
        )
        inequality_terms = gp.quicksum(
            aub[i, j] * mu[i]
            for i in np.flatnonzero(aub[:, j])
        )

        model.addConstr(
            cost
            + equality_terms
            + inequality_terms
            - lower_dual[j]
            + upper_dual[j]
            == 0.0
        )

    return x


def solve_day(date, loads, pv, pa, settings):
    model = gp.Model(f"P4_S3_{date:%Y%m%d}")
    model.Params.OutputFlag = int(settings.get("SOLVER_OUTPUT", 1))
    model.Params.LogToConsole = int(settings.get("SOLVER_OUTPUT", 1))
    model.Params.DisplayInterval = int(settings.get("GUROBI_DISPLAY_INTERVAL_S", 2))
    model.Params.MIPGap = settings["MIP_GAP"]
    model.Params.TimeLimit = settings["TIME_LIMIT_S_PER_DAY"]
    model.Params.NumericFocus = 1
    model.Params.MIPFocus = 1
    model.Params.Seed = 1

    pb = model.addVars(
        T,
        lb=settings["PB_MIN"],
        ub=settings["PB_MAX"],
        name="PB",
    )

    if settings["ENFORCE_PB_DAILY_MEAN"]:
        model.addConstr(
            gp.quicksum(pb[t] for t in range(T))
            == T * float(settings["PB_DAILY_MEAN"])
        )

    for t in range(1, T):
        model.addConstr(pb[t] - pb[t - 1] <= settings["PB_RAMP"])
        model.addConstr(pb[t - 1] - pb[t] <= settings["PB_RAMP"])

    x = {}
    for idx, bus in enumerate(BUS_ORDER, start=1):
        print(
            f"[S3][BUILD] follower {idx}/{len(BUS_ORDER)}: "
            f"building KKT system for {bus}...",
            flush=True,
        )
        x[bus] = add_kkt(
            model,
            bus,
            loads[bus] - pv[bus],
            pa,
            pb,
            settings,
        )
        model.update()
        print(
            f"[S3][BUILD] {bus} finished -> vars={model.NumVars}, "
            f"constraints={model.NumConstrs}, binaries={model.NumBinVars}",
            flush=True,
        )

    peak = model.addVar(lb=-GRB.INFINITY, name="peak")
    valley = model.addVar(lb=-GRB.INFINITY, name="valley")

    for t in range(T):
        aggregate = gp.quicksum(
            x[bus][T + t] - x[bus][2 * T + t]
            for bus in BUS_ORDER
        )
        model.addConstr(peak >= aggregate)
        model.addConstr(valley <= aggregate)

    model.setObjective(peak - valley, GRB.MINIMIZE)
    model.update()

    print(
        f"[S3][BUILD] complete model for {date.date()}: "
        f"vars={model.NumVars}, constraints={model.NumConstrs}, "
        f"binaries={model.NumBinVars}",
        flush=True,
    )
    print(
        "[S3][GUROBI] optimize() starts now. "
        "Watch Incumbent, BestBd, Gap, Nodes and Time below.",
        flush=True,
    )

    model.optimize()

    if model.SolCount:
        print(
            f"[S3][GUROBI] status={model.Status}, runtime={model.Runtime:.2f}s, "
            f"objective={model.ObjVal:.6f}, best_bound={model.ObjBound:.6f}, "
            f"gap={model.MIPGap:.6g}, nodes={model.NodeCount:.0f}",
            flush=True,
        )
    else:
        print(
            f"[S3][GUROBI] no incumbent. status={model.Status}, "
            f"runtime={model.Runtime:.2f}s, best_bound={model.ObjBound:.6f}, "
            f"nodes={model.NodeCount:.0f}",
            flush=True,
        )

    if model.SolCount == 0:
        raise RuntimeError(
            f"S3 {date.date()}: no incumbent, status={model.Status}"
        )

    pb_values = np.array([pb[t].X for t in range(T)])
    result = {}

    for bus in BUS_ORDER:
        values = np.array([x[bus][j].X for j in range(3 * T)])
        bess, imp, exp = np.split(values, 3)
        cfg = BUS_CONFIG[bus]
        energy = cfg["bess_e_init"] + np.r_[0.0, np.cumsum(DT * bess)]

        result[bus] = make_result_item(
            bess,
            imp,
            exp,
            energy,
        )

    mip_gap = float(model.MIPGap) if model.IsMIP else 0.0
    return result, pb_values, mip_gap
