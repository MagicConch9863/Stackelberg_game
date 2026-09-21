"""C-B S2: four independent load-integrated followers under PA + fixed PB."""
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from cb_common import (BUS_CONFIG, BUS_ORDER, DEFAULT_SETTINGS, DT, T,
                       load_inputs, output_folders, plot_overview,
                       save_timeseries, select_export_price, trip_vector)


def solve_follower(bus, load, pa, pb, settings):
    cfg = BUS_CONFIG[bus]
    m = gp.Model(f"CB_S2_{bus.replace(' ', '_')}")
    m.Params.OutputFlag = settings["SOLVER_OUTPUT"]
    bess = m.addVars(T, lb=-cfg["bess_kw"], ub=cfg["bess_kw"], name="bess")
    eb = m.addVars(T + 1, lb=cfg["bess_e_min"], ub=cfg["bess_e_max"], name="Eb")
    has_ev = "ev_kwh" in cfg
    ev_power = max(cfg.get("ev_kw_ch", 0.0), cfg.get("ev_kw_dis", 0.0))
    flow_bound = float(np.max(np.abs(load)) + cfg["bess_kw"] + ev_power + 1.0)
    imp = m.addVars(T, lb=0, ub=flow_bound, name="import")
    exp = m.addVars(T, lb=0, ub=flow_bound, name="export")
    ev = m.addVars(T, lb=-GRB.INFINITY, name="ev") if has_ev else None
    ee = m.addVars(T + 1, lb=.2*cfg["ev_kwh"], ub=.9*cfg["ev_kwh"], name="Ee") if has_ev else None
    m.addConstr(eb[0] == cfg["bess_e_init"])
    terminal_tol = settings["TERMINAL_ENERGY_TOL_FRAC"]*(cfg["bess_e_max"]-cfg["bess_e_min"])
    m.addConstr(eb[T]-eb[0] <= terminal_tol)
    m.addConstr(eb[0]-eb[T] <= terminal_tol)
    if has_ev:
        m.addConstr(ee[0] == .6*cfg["ev_kwh"])
        ev_terminal_tol = settings["TERMINAL_ENERGY_TOL_FRAC"]*.7*cfg["ev_kwh"]
        m.addConstr(ee[T]-ee[0] <= ev_terminal_tol)
        m.addConstr(ee[0]-ee[T] <= ev_terminal_tol)
    trip = trip_vector(bus)
    for t in range(T):
        m.addConstr(eb[t+1] == eb[t] + DT*bess[t])
        evp = 0
        if has_ev:
            hour = t*DT
            if hour < 7 or hour >= 18:
                ev[t].LB, ev[t].UB = 0, cfg["ev_kw_ch"]
            elif 8 <= hour < 17:
                ev[t].LB, ev[t].UB = -cfg["ev_kw_dis"], 0
            else:
                ev[t].LB = ev[t].UB = 0
            m.addConstr(ee[t+1] == ee[t] + DT*ev[t] - trip[t])
            evp = ev[t]
        m.addConstr(imp[t] - exp[t] == load[t] + bess[t] + evp)
    sell_price = select_export_price(pa, pb, settings["EXPORT_PRICE_MODE"])
    eps = float(settings["FLOW_TIE_BREAK_EUR_PER_KWH"])
    m.setObjective(gp.quicksum(
        DT*((pa[t]+pb[t])*imp[t] - sell_price[t]*exp[t]
            + eps*(imp[t]+exp[t])) for t in range(T)),
                   GRB.MINIMIZE)
    m.optimize()
    if m.SolCount == 0:
        raise RuntimeError(f"{bus}: no feasible solution, status={m.Status}")
    val = lambda var, n=T: np.array([var[t].X for t in range(n)])
    item = {"bess": val(bess), "ev": np.zeros(T), "import": val(imp),
            "export": val(exp), "bess_energy": val(eb, T+1)}
    if has_ev:
        item["ev"], item["ev_energy"] = val(ev), val(ee, T+1)
    item["net"] = item["import"] - item["export"]
    item["sell_price"] = sell_price
    item["sell_price_mode"] = settings["EXPORT_PRICE_MODE"].upper()
    simultaneous = float(np.max(np.minimum(item["import"], item["export"])))
    if simultaneous > 1e-5:
        print(f"WARNING {bus}: simultaneous import/export={simultaneous:.6g} kW")
    return item


def main(settings):
    raise RuntimeError("Legacy terminal-SOC model. Use phase4 soft_soc_20_80_v1.")
    csv_folder, figure_folder = output_folders()
    loads, pv, pa = load_inputs(settings, need_price=True)
    pb = np.full(T, settings["FIXED_PB_EUR_PER_KWH"])
    result = {bus: solve_follower(bus, loads[bus]-pv[bus], pa, pb, settings)
              for bus in BUS_ORDER}
    aggregate = save_timeseries(csv_folder / "s2_timeseries.csv", loads, pv, result, pa, pb)
    figure = plot_overview(figure_folder, "s2", "C-B S2: fixed-tariff response",
                           loads, pv, result, pa, pb)
    print(f"S2 complete: range={np.ptp(aggregate):.4f} kW\nCSV: {csv_folder}\nFigure: {figure}")


if __name__ == "__main__":
    SETTINGS = {**DEFAULT_SETTINGS, "FIXED_PB_EUR_PER_KWH": .10,
                "EXPORT_PRICE_MODE": "PA",
                "FLOW_TIE_BREAK_EUR_PER_KWH": 1e-5,
                "TERMINAL_ENERGY_TOL_FRAC": .05, "SOLVER_OUTPUT": 1}
    main(SETTINGS)
