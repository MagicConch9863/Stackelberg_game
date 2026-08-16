"""Phase 3 S4: daily direct centralized peak-valley benchmark."""
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

from p3_common import (
    BUS_CONFIG, BUS_ORDER, DEFAULT_SETTINGS, DT, T, day_frame, day_inputs,
    day_metrics, experiment_dates, load_fixed_simbench_day, plot_month,
    read_pv_month, save_month, trip_vector,
)


def solve_day(date, loads, pv, settings):
    model = gp.Model(f"P3_S4_{pd.Timestamp(date):%Y%m%d}")
    model.Params.OutputFlag = settings["SOLVER_OUTPUT"]
    bess, eb, ev, ee, net = {}, {}, {}, {}, {}

    for bus in BUS_ORDER:
        cfg = BUS_CONFIG[bus]
        tag = bus.replace(" ", "_")
        bess[bus] = model.addVars(T, lb=-cfg["bess_kw"], ub=cfg["bess_kw"],
                                  name=f"bess_{tag}")
        eb[bus] = model.addVars(T + 1, lb=cfg["bess_e_min"],
                                ub=cfg["bess_e_max"], name=f"Eb_{tag}")
        net[bus] = model.addVars(T, lb=-GRB.INFINITY, name=f"net_{tag}")
        model.addConstr(eb[bus][0] == cfg["bess_e_init"])
        tol = settings["TERMINAL_ENERGY_TOL_FRAC"] * (
            cfg["bess_e_max"] - cfg["bess_e_min"])
        model.addConstr(eb[bus][T] - eb[bus][0] <= tol)
        model.addConstr(eb[bus][0] - eb[bus][T] <= tol)

        if "ev_kwh" in cfg:
            ev[bus] = model.addVars(T, lb=-GRB.INFINITY, name=f"ev_{tag}")
            ee[bus] = model.addVars(T + 1, lb=0.2 * cfg["ev_kwh"],
                                    ub=0.9 * cfg["ev_kwh"], name=f"Ee_{tag}")
            model.addConstr(ee[bus][0] == 0.6 * cfg["ev_kwh"])
            ev_tol = settings["TERMINAL_ENERGY_TOL_FRAC"] * 0.7 * cfg["ev_kwh"]
            model.addConstr(ee[bus][T] - ee[bus][0] <= ev_tol)
            model.addConstr(ee[bus][0] - ee[bus][T] <= ev_tol)

        trip = trip_vector(bus)
        for t in range(T):
            model.addConstr(eb[bus][t + 1] == eb[bus][t] + DT * bess[bus][t])
            evp = 0.0
            if bus in ev:
                hour = t * DT
                if hour < 7 or hour >= 18:
                    ev[bus][t].LB, ev[bus][t].UB = 0.0, cfg["ev_kw_ch"]
                elif 8 <= hour < 17:
                    ev[bus][t].LB, ev[bus][t].UB = -cfg["ev_kw_dis"], 0.0
                else:
                    ev[bus][t].LB = ev[bus][t].UB = 0.0
                model.addConstr(
                    ee[bus][t + 1] == ee[bus][t] + DT * ev[bus][t] - trip[t])
                evp = ev[bus][t]
            model.addConstr(net[bus][t] == loads[bus][t] - pv[bus][t]
                            + bess[bus][t] + evp)

    peak = model.addVar(lb=-GRB.INFINITY, name="peak")
    valley = model.addVar(lb=-GRB.INFINITY, name="valley")
    for t in range(T):
        aggregate = gp.quicksum(net[bus][t] for bus in BUS_ORDER)
        model.addConstr(peak >= aggregate)
        model.addConstr(valley <= aggregate)
    model.setObjective(peak - valley, GRB.MINIMIZE)
    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError(
            f"S4 {pd.Timestamp(date).date()}: no solution, status={model.Status}")

    val = lambda var, n=T: np.array([var[t].X for t in range(n)])
    result = {}
    for bus in BUS_ORDER:
        item = {
            "bess": val(bess[bus]), "ev": np.zeros(T),
            "net": val(net[bus]), "bess_energy": val(eb[bus], T + 1),
        }
        item["import"] = np.maximum(item["net"], 0.0)
        item["export"] = np.maximum(-item["net"], 0.0)
        if bus in ev:
            item["ev"] = val(ev[bus])
            item["ev_energy"] = val(ee[bus], T + 1)
        result[bus] = item
    return result


def main(settings):
    loads = load_fixed_simbench_day(settings)
    pv_month = read_pv_month(settings)
    frames, metrics = [], []
    for number, date in enumerate(experiment_dates(settings), 1):
        _, pv, _ = day_inputs(date, loads, None, pv_month, settings)
        result = solve_day(date, loads, pv, settings)
        frame = day_frame(date, loads, pv, result)
        frames.append(frame)
        metrics.append(day_metrics("s4", date, frame, result))
        print(f"S4 {number:02d}: {date.date()} complete")

    ts_path, metric_path = save_month("s4", frames, metrics)
    month_frame = pd.concat(frames, ignore_index=True)
    metric_frame = pd.DataFrame(metrics)
    figure = plot_month("s4", "Phase 3 S4: centralized benchmark",
                        month_frame, metric_frame, show_price=False)
    print(f"Timeseries: {ts_path}\nMetrics: {metric_path}\nFigure: {figure}")


if __name__ == "__main__":
    SETTINGS = {**DEFAULT_SETTINGS, "SOLVER_OUTPUT": 0}
    main(SETTINGS)
