"""S4: centralized theoretical benchmark for the same four followers/devices."""
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from ca_s1_reference import (BUS_CONFIG, BUS_ORDER, DT, T, load_inputs,
                             output_folders, plot_scenario_files, save_timeseries,
                             trip_vector)


def main(settings: dict):
    csv_folder, figure_folder = output_folders(settings)
    loads, pv, _, configuration = load_inputs(settings)
    m = gp.Model("S4_four_bus_centralized")
    m.Params.OutputFlag = settings["SOLVER_OUTPUT"]
    bess, abs_bess, eb, ev, abs_ev, ee, net = {}, {}, {}, {}, {}, {}, {}
    for bus in BUS_ORDER:
        cfg = BUS_CONFIG[bus]; tag = bus.replace(" ", "_")
        bess[bus] = m.addVars(T, lb=-cfg["bess_kw"], ub=cfg["bess_kw"], name=f"bess_{tag}")
        abs_bess[bus] = m.addVars(T, lb=0, name=f"abs_bess_{tag}")
        eb[bus] = m.addVars(T + 1, lb=.1*cfg["bess_kwh"], ub=.9*cfg["bess_kwh"], name=f"Eb_{tag}")
        net[bus] = m.addVars(T, lb=-GRB.INFINITY, name=f"net_{tag}")
        m.addConstr(eb[bus][0] == .5*cfg["bess_kwh"])
        m.addConstr(eb[bus][T] == eb[bus][0])
        if "ev_kwh" in cfg:
            ev[bus] = m.addVars(T, lb=-GRB.INFINITY, name=f"ev_{tag}")
            abs_ev[bus] = m.addVars(T, lb=0, name=f"abs_ev_{tag}")
            ee[bus] = m.addVars(T + 1, lb=.2*cfg["ev_kwh"], ub=.9*cfg["ev_kwh"], name=f"Ee_{tag}")
            m.addConstr(ee[bus][0] == .6*cfg["ev_kwh"])
            m.addConstr(ee[bus][T] == ee[bus][0])
        trip = trip_vector(bus)
        for t in range(T):
            m.addConstr(eb[bus][t+1] == eb[bus][t] + bess[bus][t]*DT)
            m.addGenConstrAbs(abs_bess[bus][t], bess[bus][t])
            evp = 0
            if bus in ev:
                hour = t*DT
                if hour < 7 or hour >= 18:
                    ev[bus][t].LB, ev[bus][t].UB = 0, cfg["ev_kw_ch"]
                elif 8 <= hour < 17:
                    ev[bus][t].LB, ev[bus][t].UB = -cfg["ev_kw_dis"], 0
                else:
                    ev[bus][t].LB = ev[bus][t].UB = 0
                m.addConstr(ee[bus][t+1] == ee[bus][t] + ev[bus][t]*DT - trip[t])
                m.addGenConstrAbs(abs_ev[bus][t], ev[bus][t])
                evp = ev[bus][t]
            m.addConstr(net[bus][t] == loads[bus][t]-pv[bus][t] + bess[bus][t] + evp)

    peak = m.addVar(lb=-GRB.INFINITY, name="aggregate_peak")
    valley = m.addVar(lb=-GRB.INFINITY, name="aggregate_valley")
    for t in range(T):
        aggregate = gp.quicksum(net[bus][t] for bus in BUS_ORDER)
        m.addConstr(peak >= aggregate); m.addConstr(valley <= aggregate)
    # Lexicographic objectives: first obtain the theoretical minimum range;
    # only among equally good range solutions select the least device movement.
    movement = gp.quicksum(abs_bess[bus][t] for bus in BUS_ORDER for t in range(T))
    movement += gp.quicksum(abs_ev[bus][t] for bus in ev for t in range(T))
    m.ModelSense = GRB.MINIMIZE
    m.setObjectiveN(peak-valley, index=0, priority=2, weight=1.0,
                    abstol=1e-8, name="minimum_peak_valley")
    m.setObjectiveN(movement*DT, index=1, priority=1, weight=1.0,
                    name="minimum_movement_tie_break")
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"S4 Gurobi status {m.Status}")

    val = lambda v, n=T: np.array([v[t].X for t in range(n)])
    result = {}
    for bus in BUS_ORDER:
        item = {"bess": val(bess[bus]), "ev": np.zeros(T),
                "net": val(net[bus]), "bess_energy": val(eb[bus], T+1)}
        item["import"] = np.maximum(item["net"], 0); item["export"] = np.maximum(-item["net"], 0)
        if bus in ev:
            item["ev"] = val(ev[bus]); item["ev_energy"] = val(ee[bus], T+1)
        result[bus] = item
    agg = save_timeseries(csv_folder / "s4_timeseries.csv", loads, pv, result)
    plot_scenario_files(figure_folder, "s4", result,
                        "S4: centralized theoretical benchmark")
    print(f"S4 complete: peak={peak.X:.4f}, valley={valley.X:.4f}, range={np.ptp(agg):.4f} kW")
    print(f"CSV: {csv_folder / 's4_timeseries.csv'}")
    print(f"Figures: {figure_folder / 's4_aggregate.png'} and s4_bus1.png ... s4_bus4.png")


if __name__ == "__main__":
    SETTINGS = {
        "ENABLE_PV": False, "PV_FILE": "pv_2019-07-01_local.csv", "PV_COLUMN": "electricity",
        "PRICE_FILE": "price_data.csv", "PRICE_COLUMN": "price_eur_per_mwh_1", "PRICE_UNIT": "EUR/MWH",
        "SOLVER_OUTPUT": 1,
    }
    main(SETTINGS)
