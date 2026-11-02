import gurobipy as gp
from gurobipy import GRB
import numpy as np

from p4_common import BUS_CONFIG, BUS_ORDER, DT, T, make_result_item
from soc_soft_band import soc_limits


def follower_lp(bus, baseline, settings):
    cfg = BUS_CONFIG[bus]
    initial_by_bus = settings.get("BESS_INITIAL_ENERGY_KWH", {})
    initial_energy = float(initial_by_bus.get(bus, cfg["bess_e_init"]))
    if not cfg["bess_e_min"] <= initial_energy <= cfg["bess_e_max"]:
        raise ValueError(f"Invalid initial BESS energy for {bus}: {initial_energy}")
    has_degradation = float(settings.get("BESS_DEGRADATION_EUR_PER_KWH", 0.0)) > 0.0
    # Base: [BESS, import, export, SOC-under, SOC-over].
    # Degradation: [charge, discharge, import, export, SOC-under, SOC-over].
    q = (6 if has_degradation else 5) * T

    aeq, beq, aub, bub = [], [], [], []

    # import - export = baseline + BESS
    for t in range(T):
        row = np.zeros(q)
        if has_degradation:
            row[t] = 1.0
            row[T + t] = -1.0
            row[2 * T + t] = -1.0
            row[3 * T + t] = 1.0
        else:
            row[t] = 1.0
            row[T + t] = -1.0
            row[2 * T + t] = 1.0
        aeq.append(row)
        beq.append(-baseline[t])

    # Cumulative BESS energy limits.
    for k in range(1, T + 1):
        row = np.zeros(q)
        row[:k] = DT
        if has_degradation:
            row[T:T + k] = -DT
        aub.extend((row, -row))
        bub.extend((
            cfg["bess_e_max"] - initial_energy,
            initial_energy - cfg["bess_e_min"],
        ))

    physical_low, physical_high, soft_low, soft_high = soc_limits(cfg, settings)
    # Time-invariant soft comfort band at every future state:
    # E_k + v^-_k >= 20% C and E_k - v^+_k <= 80% C.
    for k in range(1, T + 1):
        low_row = np.zeros(q)
        low_row[:k] = -DT
        if has_degradation:
            low_row[T:T + k] = DT
            low_row[4 * T + k - 1] = -1.0
        else:
            low_row[3 * T + k - 1] = -1.0
        aub.append(low_row)
        bub.append(initial_energy - soft_low)

        high_row = np.zeros(q)
        high_row[:k] = DT
        if has_degradation:
            high_row[T:T + k] = -DT
            high_row[5 * T + k - 1] = -1.0
        else:
            high_row[4 * T + k - 1] = -1.0
        aub.append(high_row)
        bub.append(soft_high - initial_energy)

    grid_bound = float(settings["GRID_BOUND_KW"])
    if has_degradation:
        lb = np.zeros(6 * T)
        ub = np.r_[
            np.full(T, cfg["bess_kw"]), np.full(T, cfg["bess_kw"]),
            np.full(T, grid_bound), np.full(T, grid_bound),
            np.full(T, soft_low - physical_low),
            np.full(T, physical_high - soft_high),
        ]
    else:
        lb = np.r_[np.full(T, -cfg["bess_kw"]), np.zeros(4 * T)]
        ub = np.r_[
            np.full(T, cfg["bess_kw"]),
            np.full(T, grid_bound), np.full(T, grid_bound),
            np.full(T, soft_low - physical_low),
            np.full(T, physical_high - soft_high),
        ]

    return tuple(map(np.asarray, (aeq, beq, aub, bub))) + (lb, ub)


def complementarity(model, slack_expr, slack_max, dual, name, dual_max, start_slack=None):
    maximum = max(float(slack_max), 0.0)
    binary = model.addVar(vtype=GRB.BINARY, name=f"z_{name}")
    slack = model.addVar(lb=0.0, ub=maximum, name=f"slack_{name}")

    model.addConstr(slack == slack_expr)
    model.addConstr(slack <= maximum * binary)
    model.addConstr(dual <= dual_max * (1.0 - binary))
    if start_slack is not None:
        start_slack = max(float(start_slack), 0.0)
        slack.Start = start_slack
        binary.Start = 1.0 if start_slack > 1e-7 else 0.0


def follower_primal_warm_start(bus, baseline, pa, pb_values, settings):
    """Solve the fixed-tariff follower LP and return a primal KKT start."""
    aeq, beq, aub, bub, lb, ub = follower_lp(bus, baseline, settings)
    q = len(lb)
    model = gp.Model(f"warm_{bus.replace(' ', '_')}")
    model.Params.OutputFlag = 0
    x = model.addVars(q, lb=-GRB.INFINITY, name="x")
    for j in range(q):
        model.addConstr(x[j] >= float(lb[j]))
        model.addConstr(x[j] <= float(ub[j]))
    for i, row in enumerate(aeq):
        nz = np.flatnonzero(row)
        model.addConstr(gp.quicksum(row[j] * x[j] for j in nz) == float(beq[i]))
    for i, row in enumerate(aub):
        nz = np.flatnonzero(row)
        model.addConstr(gp.quicksum(row[j] * x[j] for j in nz) <= float(bub[i]))

    eps = float(settings["FLOW_TIE_BREAK_EUR_PER_KWH"])
    degradation = DT * float(settings.get("BESS_DEGRADATION_EUR_PER_KWH", 0.0))
    objective = gp.LinExpr()
    for t in range(T):
        objective += degradation * (x[t] + x[T + t])
        objective += DT * (pa[t] + pb_values[t] + eps) * x[2 * T + t]
        objective += DT * (-pa[t] + eps) * x[3 * T + t]
        objective += float(settings["SOC_SOFT_PENALTY_EUR_PER_KWH"]) * (
            x[4 * T + t] + x[5 * T + t]
        )
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Warm-start follower {bus}: status={model.Status}")
    values = np.array([x[j].X for j in range(q)], dtype=float)
    return values, {
        "objective": float(model.ObjVal),
        "max_eq_residual": float(np.max(np.abs(aeq @ values - beq))),
        "max_ineq_violation": float(np.max(np.maximum(aub @ values - bub, 0.0))),
    }


def add_kkt(model, bus, baseline, pa, pb, settings, primal_start=None):
    aeq, beq, aub, bub, lb, ub = follower_lp(bus, baseline, settings)
    q = len(lb)
    tag = bus.replace(" ", "_")

    x = model.addVars(q, lb=-GRB.INFINITY, name=f"x_{tag}")
    lam = model.addVars(len(beq), lb=-GRB.INFINITY, name=f"lam_{tag}")
    mu = model.addVars(
        len(bub),
        lb=0.0,
        ub=settings["DUAL_BIG_M"],
        name=f"mu_{tag}",
    )
    if primal_start is not None:
        for j in range(q):
            x[j].Start = float(primal_start[j])

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
            f"ineq_{tag}_{i}", settings["DUAL_BIG_M"],
            None if primal_start is None else bub[i] - float(row @ primal_start),
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
            None if primal_start is None else float(primal_start[j] - lb[j]),
        )
        complementarity(
            model,
            ub[j] - x[j],
            span,
            upper_dual[j],
            f"ub_{tag}_{j}",
            settings["DUAL_BIG_M"],
            None if primal_start is None else float(ub[j] - primal_start[j]),
        )

    eps = float(settings["FLOW_TIE_BREAK_EUR_PER_KWH"])
    has_degradation = float(settings.get("BESS_DEGRADATION_EUR_PER_KWH", 0.0)) > 0.0
    degradation = DT * float(settings.get("BESS_DEGRADATION_EUR_PER_KWH", 0.0))

    for j in range(q):
        if has_degradation:
            if j < 2 * T:
                cost = degradation
            elif j < 3 * T:
                t = j - 2 * T
                cost = DT * (pa[t] + pb[t] + eps)
            elif j < 4 * T:
                t = j - 3 * T
                cost = DT * (-pa[t] + eps)
            else:
                cost = float(settings["SOC_SOFT_PENALTY_EUR_PER_KWH"])
        else:
            if j < T:
                cost = 0.0
            elif j < 2 * T:
                t = j - T
                cost = DT * (pa[t] + pb[t] + eps)
            elif j < 3 * T:
                t = j - 2 * T
                cost = DT * (-pa[t] + eps)
            else:
                cost = float(settings["SOC_SOFT_PENALTY_EUR_PER_KWH"])

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
    model.Params.MIPFocus = int(settings.get("MIP_FOCUS", 1))
    model.Params.Seed = 1
    model.Params.Threads = int(settings.get("GUROBI_THREADS", 1))

    pb = model.addVars(
        T,
        lb=settings["PB_MIN"],
        ub=settings["PB_MAX"],
        name="PB",
    )
    pb_warm = settings.get("PB_WARM_START")
    if pb_warm is not None:
        pb_warm = np.asarray(pb_warm, dtype=float)
        if len(pb_warm) != T:
            raise ValueError(f"PB_WARM_START must contain {T} values")
        for t in range(T):
            pb[t].Start = float(pb_warm[t])

    if settings["ENFORCE_PB_DAILY_MEAN"]:
        model.addConstr(
            gp.quicksum(pb[t] for t in range(T))
            == T * float(settings["PB_DAILY_MEAN"])
        )

    for t in range(1, T):
        model.addConstr(pb[t] - pb[t - 1] <= settings["PB_RAMP"])
        model.addConstr(pb[t - 1] - pb[t] <= settings["PB_RAMP"])
    previous_pb = settings.get("PB_PREVIOUS_EUR_PER_KWH")
    if previous_pb is not None:
        model.addConstr(pb[0] - float(previous_pb) <= settings["PB_RAMP"])
        model.addConstr(float(previous_pb) - pb[0] <= settings["PB_RAMP"])

    x = {}
    has_degradation = float(settings.get("BESS_DEGRADATION_EUR_PER_KWH", 0.0)) > 0.0
    primal_starts = {}
    if pb_warm is not None and has_degradation:
        for bus in BUS_ORDER:
            primal_starts[bus], info = follower_primal_warm_start(
                bus, loads[bus] - pv[bus], pa, pb_warm, settings
            )
            print(
                f"[S3][WARM] {bus}: objective={info['objective']:.6f}, "
                f"eq_res={info['max_eq_residual']:.2e}, "
                f"ineq_violation={info['max_ineq_violation']:.2e}", flush=True,
            )
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
            primal_start=primal_starts.get(bus),
        )
        model.update()
        print(
            f"[S3][BUILD] {bus} finished -> vars={model.NumVars}, "
            f"constraints={model.NumConstrs}, binaries={model.NumBinVars}",
            flush=True,
        )

    peak = model.addVar(lb=-GRB.INFINITY, name="peak")
    valley = model.addVar(lb=-GRB.INFINITY, name="valley")
    aggregates = []
    exchange_mode = str(settings.get("GRID_EXCHANGE_MODE", "nonnegative_clip"))
    grid_clip_m = float(len(BUS_ORDER) * settings["GRID_BOUND_KW"])
    for t in range(T):
        imports = gp.quicksum(
            x[bus][2 * T + t] if has_degradation else x[bus][T + t]
            for bus in BUS_ORDER
        )
        exports = gp.quicksum(
            x[bus][3 * T + t] if has_degradation else x[bus][2 * T + t]
            for bus in BUS_ORDER
        )
        if exchange_mode == "grid_export":
            # Signed public-grid exchange. Negative values are physical export.
            aggregate = imports - exports
        elif exchange_mode == "local_sale":
            # Export is settled with an external local buyer and is deliberately
            # outside the public-grid KPI boundary.
            aggregate = imports
        elif exchange_mode == "nonnegative_clip":
            # Legacy formulation retained only for reproducibility of old runs.
            aggregate_raw = imports - exports
            aggregate = model.addVar(lb=0.0, ub=grid_clip_m, name=f"grid_after_curtailment_{t}")
            import_mode = model.addVar(vtype=GRB.BINARY, name=f"grid_import_mode_{t}")
            model.addConstr(aggregate >= aggregate_raw, name=f"grid_clip_lb_raw_{t}")
            model.addConstr(aggregate <= aggregate_raw + grid_clip_m * (1 - import_mode), name=f"grid_clip_ub_raw_{t}")
            model.addConstr(aggregate <= grid_clip_m * import_mode, name=f"grid_clip_zero_{t}")
        else:
            raise ValueError(f"Unsupported GRID_EXCHANGE_MODE: {exchange_mode}")
        aggregates.append(aggregate)
        model.addConstr(peak >= aggregate)
        model.addConstr(valley <= aggregate)

    ramp_weight = float(settings.get("LEADER_RAMP_WEIGHT", 0.0))
    ramp_terms = []
    previous_grid = settings.get("LEADER_PREVIOUS_GRID_KW")
    if ramp_weight > 0.0 and previous_grid is not None:
        ramp0 = model.addVar(lb=0.0, name="leader_grid_ramp_0")
        model.addConstr(ramp0 >= aggregates[0] - float(previous_grid))
        model.addConstr(ramp0 >= float(previous_grid) - aggregates[0])
        ramp_terms.append(ramp0)
    if ramp_weight > 0.0:
        for t in range(1, T):
            ramp_abs = model.addVar(lb=0.0, name=f"leader_grid_ramp_{t}")
            model.addConstr(ramp_abs >= aggregates[t] - aggregates[t - 1])
            model.addConstr(ramp_abs >= aggregates[t - 1] - aggregates[t])
            ramp_terms.append(ramp_abs)
    ramp_l1 = gp.quicksum(ramp_terms) if ramp_terms else 0.0
    model.setObjective(peak - valley + ramp_weight * ramp_l1, GRB.MINIMIZE)
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
        if has_degradation:
            values = np.array([x[bus][j].X for j in range(4 * T)])
            charge, discharge, imp, exp = np.split(values, 4)
            bess = charge - discharge
        else:
            values = np.array([x[bus][j].X for j in range(3 * T)])
            bess, imp, exp = np.split(values, 3)
        cfg = BUS_CONFIG[bus]
        initial_by_bus = settings.get("BESS_INITIAL_ENERGY_KWH", {})
        initial_energy = float(initial_by_bus.get(bus, cfg["bess_e_init"]))
        energy = initial_energy + np.r_[0.0, np.cumsum(DT * bess)]

        result[bus] = make_result_item(
            bess,
            imp,
            exp,
            energy,
        )

    mip_gap = float(model.MIPGap) if model.IsMIP else 0.0
    return result, pb_values, mip_gap
