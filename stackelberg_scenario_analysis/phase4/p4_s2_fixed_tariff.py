import gurobipy as gp
from gurobipy import GRB
import numpy as np

from p4_common import BUS_CONFIG, BUS_ORDER, DT, T, make_result_item
from soc_soft_band import add_soft_soc_band


def solve_follower(bus, baseline, pa, pb, settings):
    cfg = BUS_CONFIG[bus]
    model = gp.Model(f"P4_S2_{bus.replace(' ', '_')}")
    model.Params.OutputFlag = int(settings.get("SOLVER_OUTPUT", 1))
    model.Params.LogToConsole = int(settings.get("SOLVER_OUTPUT", 1))
    model.Params.DisplayInterval = int(settings.get("GUROBI_DISPLAY_INTERVAL_S", 2))

    bess = model.addVars(
        T,
        lb=-cfg["bess_kw"],
        ub=cfg["bess_kw"],
        name="bess",
    )

    eb = model.addVars(
        T + 1,
        lb=cfg["bess_e_min"],
        ub=cfg["bess_e_max"],
        name="Eb",
    )

    flow_bound = float(
        np.max(np.abs(baseline)) + cfg["bess_kw"] + 1.0
    )

    imp = model.addVars(T, lb=0.0, ub=flow_bound, name="import")
    exp = model.addVars(T, lb=0.0, ub=flow_bound, name="export")

    model.addConstr(eb[0] == cfg["bess_e_init"])

    for t in range(T):
        model.addConstr(eb[t + 1] == eb[t] + DT * bess[t])
        model.addConstr(imp[t] - exp[t] == baseline[t] + bess[t])

    eps = float(settings["FLOW_TIE_BREAK_EUR_PER_KWH"])
    soc_penalty, _, _ = add_soft_soc_band(model, eb, T, cfg, settings, bus.replace(" ", "_"))

    model.setObjective(
        gp.quicksum(
            DT * (
                (pa[t] + pb[t]) * imp[t]
                - pa[t] * exp[t]
                + eps * (imp[t] + exp[t])
            )
            for t in range(T)
        ) + soc_penalty,
        GRB.MINIMIZE,
    )

    model.update()
    print(
        f"[S2][{bus}] Gurobi model: vars={model.NumVars}, "
        f"constraints={model.NumConstrs}. Starting optimize()...",
        flush=True,
    )
    model.optimize()
    print(
        f"[S2][{bus}] status={model.Status}, runtime={model.Runtime:.3f}s, "
        f"objective={model.ObjVal if model.SolCount else float('nan'):.6f}",
        flush=True,
    )

    if model.SolCount == 0:
        raise RuntimeError(f"{bus}: S2 no solution, status={model.Status}")

    values = lambda var, n=T: np.array([var[t].X for t in range(n)])

    return make_result_item(
        values(bess),
        values(imp),
        values(exp),
        values(eb, T + 1),
    )


def solve_day(loads, pv, pa, settings):
    pb = np.full(T, float(settings["FIXED_PB_EUR_PER_KWH"]))

    result = {}
    for idx, bus in enumerate(BUS_ORDER, start=1):
        print(f"[S2] follower {idx}/{len(BUS_ORDER)}: {bus}", flush=True)
        result[bus] = solve_follower(
            bus,
            loads[bus] - pv[bus],
            pa,
            pb,
            settings,
        )

    return result, pb
