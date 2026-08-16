"""C-B S4: direct centralized peak-valley benchmark with SimBench loads."""
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from cb_common import (BUS_CONFIG, BUS_ORDER, DEFAULT_SETTINGS, DT, T,
                       load_inputs, output_folders, plot_overview,
                       save_timeseries, trip_vector)


def main(settings):
    csv_folder, figure_folder = output_folders()
    loads, pv, _ = load_inputs(settings)
    m = gp.Model("CB_S4_centralized")
    m.Params.OutputFlag = settings["SOLVER_OUTPUT"]
    bess, eb, ev, ee, net = {}, {}, {}, {}, {}
    for bus in BUS_ORDER:
        cfg, tag = BUS_CONFIG[bus], bus.replace(" ", "_")
        bess[bus] = m.addVars(T, lb=-cfg["bess_kw"], ub=cfg["bess_kw"], name=f"bess_{tag}")
        eb[bus] = m.addVars(T+1, lb=cfg["bess_e_min"], ub=cfg["bess_e_max"], name=f"Eb_{tag}")
        net[bus] = m.addVars(T, lb=-GRB.INFINITY, name=f"net_{tag}")
        m.addConstr(eb[bus][0] == cfg["bess_e_init"])
        terminal_tol = settings["TERMINAL_ENERGY_TOL_FRAC"]*(cfg["bess_e_max"]-cfg["bess_e_min"])
        m.addConstr(eb[bus][T]-eb[bus][0] <= terminal_tol)
        m.addConstr(eb[bus][0]-eb[bus][T] <= terminal_tol)
        if "ev_kwh" in cfg:
            ev[bus] = m.addVars(T, lb=-GRB.INFINITY, name=f"ev_{tag}")
            ee[bus] = m.addVars(T+1, lb=.2*cfg["ev_kwh"], ub=.9*cfg["ev_kwh"], name=f"Ee_{tag}")
            m.addConstr(ee[bus][0] == .6*cfg["ev_kwh"])
            ev_terminal_tol = settings["TERMINAL_ENERGY_TOL_FRAC"]*.7*cfg["ev_kwh"]
            m.addConstr(ee[bus][T]-ee[bus][0] <= ev_terminal_tol)
            m.addConstr(ee[bus][0]-ee[bus][T] <= ev_terminal_tol)
        trip = trip_vector(bus)
        for t in range(T):
            m.addConstr(eb[bus][t+1] == eb[bus][t] + DT*bess[bus][t])
            evp = 0
            if bus in ev:
                hour = t*DT
                if hour < 7 or hour >= 18:
                    ev[bus][t].LB, ev[bus][t].UB = 0, cfg["ev_kw_ch"]
                elif 8 <= hour < 17:
                    ev[bus][t].LB, ev[bus][t].UB = -cfg["ev_kw_dis"], 0
                else:
                    ev[bus][t].LB = ev[bus][t].UB = 0
                m.addConstr(ee[bus][t+1] == ee[bus][t] + DT*ev[bus][t] - trip[t])
                evp = ev[bus][t]
            m.addConstr(net[bus][t] == loads[bus][t] - pv[bus][t]
                        + bess[bus][t] + evp)
    peak = m.addVar(lb=-GRB.INFINITY, name="peak")
    valley = m.addVar(lb=-GRB.INFINITY, name="valley")
    for t in range(T):
        aggregate = gp.quicksum(net[bus][t] for bus in BUS_ORDER)
        m.addConstr(peak >= aggregate)
        m.addConstr(valley <= aggregate)
    m.setObjective(peak-valley, GRB.MINIMIZE)
    m.optimize()
    if m.SolCount == 0:
        raise RuntimeError(f"S4: no feasible solution, status={m.Status}")
    val = lambda var, n=T: np.array([var[t].X for t in range(n)])
    result = {}
    for bus in BUS_ORDER:
        item = {"bess": val(bess[bus]), "ev": np.zeros(T), "net": val(net[bus]),
                "bess_energy": val(eb[bus], T+1)}
        item["import"], item["export"] = np.maximum(item["net"], 0), np.maximum(-item["net"], 0)
        if bus in ev:
            item["ev"], item["ev_energy"] = val(ev[bus]), val(ee[bus], T+1)
        result[bus] = item
    aggregate = save_timeseries(csv_folder / "s4_timeseries.csv", loads, pv, result)
    figure = plot_overview(figure_folder, "s4", "C-B S4: centralized benchmark",
                           loads, pv, result)
    original = sum(loads[bus]-pv[bus] for bus in BUS_ORDER)
    print(f"S4 complete: original range={np.ptp(original):.4f} kW, "
          f"controlled range={np.ptp(aggregate):.4f} kW")
    print(f"Total BESS power limit={sum(BUS_CONFIG[b]['bess_kw'] for b in BUS_ORDER):.2f} kW")
    print(f"CSV: {csv_folder}\nFigure: {figure}")


if __name__ == "__main__":
    SETTINGS = {**DEFAULT_SETTINGS, "TERMINAL_ENERGY_TOL_FRAC": .05,
                "SOLVER_OUTPUT": 1}
    main(SETTINGS)
