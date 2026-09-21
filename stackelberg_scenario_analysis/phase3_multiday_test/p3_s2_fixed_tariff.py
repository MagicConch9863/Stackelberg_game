"""Phase 3 S2: daily follower response to monthly PA plus fixed PB."""
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

from p3_common import (
    BUS_CONFIG, BUS_ORDER, DEFAULT_SETTINGS, DT, T, day_frame, day_inputs,
    day_metrics, experiment_dates, load_fixed_simbench_day, plot_month,
    read_pa_month, read_pv_month, save_month, select_export_price, trip_vector,
)


def solve_follower(bus, baseline, pa, pb, settings):
    cfg = BUS_CONFIG[bus]
    model = gp.Model(f"P3_S2_{bus.replace(' ', '_')}")
    model.Params.OutputFlag = settings["SOLVER_OUTPUT"]
    bess = model.addVars(T, lb=-cfg["bess_kw"], ub=cfg["bess_kw"], name="bess")
    eb = model.addVars(T + 1, lb=cfg["bess_e_min"],
                       ub=cfg["bess_e_max"], name="Eb")
    has_ev = "ev_kwh" in cfg
    ev_limit = max(cfg.get("ev_kw_ch", 0.0), cfg.get("ev_kw_dis", 0.0))
    flow_bound = float(np.max(np.abs(baseline)) + cfg["bess_kw"] + ev_limit + 1.0)
    imp = model.addVars(T, lb=0.0, ub=flow_bound, name="import")
    exp = model.addVars(T, lb=0.0, ub=flow_bound, name="export")
    ev = model.addVars(T, lb=-GRB.INFINITY, name="ev") if has_ev else None
    ee = (model.addVars(T + 1, lb=0.2 * cfg["ev_kwh"],
                        ub=0.9 * cfg["ev_kwh"], name="Ee") if has_ev else None)

    model.addConstr(eb[0] == cfg["bess_e_init"])
    tol = settings["TERMINAL_ENERGY_TOL_FRAC"] * (
        cfg["bess_e_max"] - cfg["bess_e_min"])
    model.addConstr(eb[T] - eb[0] <= tol)
    model.addConstr(eb[0] - eb[T] <= tol)
    if has_ev:
        model.addConstr(ee[0] == 0.6 * cfg["ev_kwh"])
        ev_tol = settings["TERMINAL_ENERGY_TOL_FRAC"] * 0.7 * cfg["ev_kwh"]
        model.addConstr(ee[T] - ee[0] <= ev_tol)
        model.addConstr(ee[0] - ee[T] <= ev_tol)

    trip = trip_vector(bus)
    for t in range(T):
        model.addConstr(eb[t + 1] == eb[t] + DT * bess[t])
        evp = 0.0
        if has_ev:
            hour = t * DT
            if hour < 7 or hour >= 18:
                ev[t].LB, ev[t].UB = 0.0, cfg["ev_kw_ch"]
            elif 8 <= hour < 17:
                ev[t].LB, ev[t].UB = -cfg["ev_kw_dis"], 0.0
            else:
                ev[t].LB = ev[t].UB = 0.0
            model.addConstr(ee[t + 1] == ee[t] + DT * ev[t] - trip[t])
            evp = ev[t]
        model.addConstr(imp[t] - exp[t] == baseline[t] + bess[t] + evp)

    sell_price = select_export_price(pa, pb, settings["EXPORT_PRICE_MODE"])
    eps = float(settings["FLOW_TIE_BREAK_EUR_PER_KWH"])
    model.setObjective(gp.quicksum(
        DT * ((pa[t] + pb[t]) * imp[t] - sell_price[t] * exp[t]
              + eps * (imp[t] + exp[t])) for t in range(T)), GRB.MINIMIZE)
    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError(f"{bus}: no feasible S2 solution, status={model.Status}")

    val = lambda var, n=T: np.array([var[t].X for t in range(n)])
    item = {
        "bess": val(bess), "ev": np.zeros(T),
        "import": val(imp), "export": val(exp),
        "bess_energy": val(eb, T + 1),
        "sell_price": sell_price,
        "sell_price_mode": settings["EXPORT_PRICE_MODE"].upper(),
    }
    if has_ev:
        item["ev"] = val(ev)
        item["ev_energy"] = val(ee, T + 1)
    item["net"] = item["import"] - item["export"]
    return item


def main(settings):
    raise RuntimeError("Legacy terminal-SOC model. Rebuild with phase4 soft_soc_20_80_v1.")
    loads = load_fixed_simbench_day(settings)
    pa_month = read_pa_month(settings)
    pv_month = read_pv_month(settings)
    frames, metrics = [], []
    for number, date in enumerate(experiment_dates(settings), 1):
        _, pv, pa = day_inputs(date, loads, pa_month, pv_month, settings)
        pb = np.full(T, settings["FIXED_PB_EUR_PER_KWH"])
        result = {
            bus: solve_follower(bus, loads[bus] - pv[bus], pa, pb, settings)
            for bus in BUS_ORDER
        }
        frame = day_frame(date, loads, pv, result, pa, pb)
        frames.append(frame)
        metrics.append(day_metrics("s2", date, frame, result, pa, pb))
        print(f"S2 {number:02d}: {date.date()} complete")

    ts_path, metric_path = save_month("s2", frames, metrics)
    month_frame = pd.concat(frames, ignore_index=True)
    metric_frame = pd.DataFrame(metrics)
    figure = plot_month("s2", "Phase 3 S2: fixed-tariff response",
                        month_frame, metric_frame, show_price=True)
    print(f"Timeseries: {ts_path}\nMetrics: {metric_path}\nFigure: {figure}")


if __name__ == "__main__":
    SETTINGS = {
        **DEFAULT_SETTINGS,
        "FIXED_PB_EUR_PER_KWH": 0.10,
        "EXPORT_PRICE_MODE": "PA",
        "SOLVER_OUTPUT": 0,
    }
    main(SETTINGS)
