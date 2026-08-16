"""S2: four independent followers respond to PA plus a fixed network tariff PB."""
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from ca_s1_reference import (BUS_CONFIG, BUS_ORDER, DT, T, load_inputs,
                             output_folders, plot_scenario_files, save_timeseries,
                             trip_vector)


def solve_follower(bus: str, load: np.ndarray, pv: np.ndarray,
                   pa: np.ndarray, pb: np.ndarray, solver_output: int):
    cfg = BUS_CONFIG[bus]
    m = gp.Model(f"S2_{bus.replace(' ', '_')}")
    m.Params.OutputFlag = solver_output
    bess = m.addVars(T, lb=-cfg["bess_kw"], ub=cfg["bess_kw"], name="bess_power")
    eb = m.addVars(T + 1, lb=0.1 * cfg["bess_kwh"], ub=0.9 * cfg["bess_kwh"], name="Eb")
    imp = m.addVars(T, lb=0, name="import")
    exp = m.addVars(T, lb=0, name="export")
    has_ev = "ev_kwh" in cfg
    ev = m.addVars(T, lb=-GRB.INFINITY, name="ev_power") if has_ev else None
    ee = m.addVars(T + 1, lb=0.2 * cfg["ev_kwh"], ub=0.9 * cfg["ev_kwh"], name="Ee") if has_ev else None

    m.addConstr(eb[0] == 0.5 * cfg["bess_kwh"])
    m.addConstr(eb[T] == eb[0])
    if has_ev:
        m.addConstr(ee[0] == 0.6 * cfg["ev_kwh"])
        m.addConstr(ee[T] == ee[0])
    trip = trip_vector(bus)
    for t in range(T):
        m.addConstr(eb[t + 1] == eb[t] + bess[t] * DT)
        ev_power = 0
        if has_ev:
            hour = t * DT
            if hour < 7 or hour >= 18:
                ev[t].LB, ev[t].UB = 0, cfg["ev_kw_ch"]
            elif 8 <= hour < 17:
                ev[t].LB, ev[t].UB = -cfg["ev_kw_dis"], 0
            else:
                ev[t].LB = ev[t].UB = 0
            m.addConstr(ee[t + 1] == ee[t] + ev[t] * DT - trip[t])
            ev_power = ev[t]
        m.addConstr(imp[t] - exp[t] == load[t] - pv[t] + bess[t] + ev_power)

    m.setObjective(gp.quicksum(((pa[t] + pb[t]) * imp[t] - pa[t] * exp[t]) * DT
                               for t in range(T)), GRB.MINIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"{bus}: Gurobi status {m.Status}")
    val = lambda v, n=T: np.array([v[t].X for t in range(n)])
    result = {"bess": val(bess), "ev": np.zeros(T),
              "import": val(imp), "export": val(exp), "bess_energy": val(eb, T + 1)}
    if has_ev:
        result["ev"] = val(ev)
        result["ev_energy"] = val(ee, T + 1)
    result["net"] = result["import"] - result["export"]
    return result, m.ObjVal


def main(settings: dict):
    csv_folder, figure_folder = output_folders(settings)
    loads, pv, pa, configuration = load_inputs(settings, need_price=True)
    pb = np.full(T, settings["FIXED_PB_EUR_PER_KWH"])
    result, costs = {}, {}
    for bus in BUS_ORDER:
        result[bus], costs[bus] = solve_follower(bus, loads[bus], pv[bus], pa, pb, settings["SOLVER_OUTPUT"])
    agg = save_timeseries(csv_folder / "s2_timeseries.csv", loads, pv, result, pa, pb)
    plot_scenario_files(figure_folder, "s2", result,
                        "S2: fixed-tariff follower response", pa, pb)
    print(f"S2 complete: range={np.ptp(agg):.4f} kW; total follower cost={sum(costs.values()):.4f} EUR")
    print(f"CSV: {csv_folder / 's2_timeseries.csv'}")
    print(f"Figures: {figure_folder / 's2_aggregate.png'} and s2_bus1.png ... s2_bus4.png")


if __name__ == "__main__":
    SETTINGS = {
        "ENABLE_PV": False, "PV_FILE": "pv_2019-07-01_local.csv", "PV_COLUMN": "electricity",
        "PRICE_FILE": "price_data.csv", "PRICE_COLUMN": "price_eur_per_mwh_1", "PRICE_UNIT": "EUR/MWH",
        "FIXED_PB_EUR_PER_KWH": 0.10, "SOLVER_OUTPUT": 1,
    }
    main(SETTINGS)
