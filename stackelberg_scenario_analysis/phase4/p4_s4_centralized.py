import gurobipy as gp
from gurobipy import GRB
import numpy as np

from p4_common import BUS_CONFIG, BUS_ORDER, DT, T, make_result_item
from soc_soft_band import add_soft_soc_band


def solve_day(date, loads, pv, settings):
    model = gp.Model(f"P4_S4_{date:%Y%m%d}")
    model.Params.OutputFlag = int(settings.get("SOLVER_OUTPUT", 1))
    model.Params.LogToConsole = int(settings.get("SOLVER_OUTPUT", 1))
    model.Params.DisplayInterval = int(settings.get("GUROBI_DISPLAY_INTERVAL_S", 2))

    bess, eb, net = {}, {}, {}
    soc_penalties = []

    for bus in BUS_ORDER:
        cfg = BUS_CONFIG[bus]
        tag = bus.replace(" ", "_")

        bess[bus] = model.addVars(
            T,
            lb=-cfg["bess_kw"],
            ub=cfg["bess_kw"],
            name=f"bess_{tag}",
        )
        eb[bus] = model.addVars(
            T + 1,
            lb=cfg["bess_e_min"],
            ub=cfg["bess_e_max"],
            name=f"Eb_{tag}",
        )
        net[bus] = model.addVars(T, lb=-GRB.INFINITY, name=f"net_{tag}")

        model.addConstr(eb[bus][0] == cfg["bess_e_init"])

        for t in range(T):
            model.addConstr(
                eb[bus][t + 1] == eb[bus][t] + DT * bess[bus][t]
            )
            model.addConstr(
                net[bus][t]
                == loads[bus][t] - pv[bus][t] + bess[bus][t]
            )
        penalty, _, _ = add_soft_soc_band(model, eb[bus], T, cfg, settings, tag)
        soc_penalties.append(penalty)

    peak = model.addVar(lb=-GRB.INFINITY, name="peak")
    valley = model.addVar(lb=-GRB.INFINITY, name="valley")

    for t in range(T):
        aggregate = gp.quicksum(net[bus][t] for bus in BUS_ORDER)
        model.addConstr(peak >= aggregate)
        model.addConstr(valley <= aggregate)

    model.setObjective(peak - valley + gp.quicksum(soc_penalties), GRB.MINIMIZE)
    model.update()
    print(
        f"[S4][GUROBI] model for {date.date()}: vars={model.NumVars}, "
        f"constraints={model.NumConstrs}. Starting optimize()...",
        flush=True,
    )
    model.optimize()
    if model.SolCount:
        print(
            f"[S4][GUROBI] status={model.Status}, runtime={model.Runtime:.3f}s, "
            f"objective={model.ObjVal:.6f}",
            flush=True,
        )

    if model.SolCount == 0:
        raise RuntimeError(
            f"S4 {date.date()}: no solution, status={model.Status}"
        )

    values = lambda var, n=T: np.array([var[t].X for t in range(n)])
    result = {}

    for bus in BUS_ORDER:
        net_values = values(net[bus])
        result[bus] = make_result_item(
            values(bess[bus]),
            np.maximum(net_values, 0.0),
            np.maximum(-net_values, 0.0),
            values(eb[bus], T + 1),
        )

    return result
