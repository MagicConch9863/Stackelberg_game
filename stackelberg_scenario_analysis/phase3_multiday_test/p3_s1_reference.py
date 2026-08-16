"""Phase 3 S1: repeated predefined schedules under monthly PV variation."""
import numpy as np
import pandas as pd

from p3_common import (
    BUS_CONFIG, BUS_ORDER, DEFAULT_SETTINGS, DT, T, day_frame, day_inputs,
    day_metrics, experiment_dates, load_fixed_simbench_day, plot_month,
    read_pv_month, save_month, trip_vector,
)


def energy(power, initial, loss=None):
    loss = np.zeros(T) if loss is None else np.asarray(loss)
    return initial + np.r_[0.0, np.cumsum(np.asarray(power) * DT - loss)]


def predefined_dispatch(bus):
    cfg = BUS_CONFIG[bus]
    bess = np.zeros(T)
    interval_h = 16 * DT
    headroom = min(cfg["bess_e_max"] - cfg["bess_e_init"],
                   cfg["bess_e_init"] - cfg["bess_e_min"])
    fixed_power = min(0.5 * cfg["bess_kw"], headroom / interval_h)
    bess[4:20] = fixed_power
    bess[72:88] = -fixed_power

    ev = np.zeros(T)
    if "ev_kwh" in cfg:
        ev[32:68] = -0.25 * cfg["ev_kw_dis"]
        home = np.r_[np.arange(0, 28), np.arange(72, 96)]
        required = -ev.sum() * DT + cfg["ev_trip_kwh"]
        ev[home] = required / (len(home) * DT)
    return bess, ev


def main(settings):
    loads = load_fixed_simbench_day(settings)
    pv_month = read_pv_month(settings)
    frames, metrics = [], []
    for number, date in enumerate(experiment_dates(settings), 1):
        _, pv, _ = day_inputs(date, loads, None, pv_month, settings)
        result = {}
        for bus in BUS_ORDER:
            cfg = BUS_CONFIG[bus]
            bess, ev = predefined_dispatch(bus)
            net = loads[bus] - pv[bus] + bess + ev
            item = {
                "bess": bess, "ev": ev, "net": net,
                "import": np.maximum(net, 0.0),
                "export": np.maximum(-net, 0.0),
                "bess_energy": energy(bess, cfg["bess_e_init"]),
            }
            if "ev_kwh" in cfg:
                item["ev_energy"] = energy(
                    ev, 0.6 * cfg["ev_kwh"], trip_vector(bus))
            result[bus] = item
        frame = day_frame(date, loads, pv, result)
        frames.append(frame)
        metrics.append(day_metrics("s1", date, frame, result))
        print(f"S1 {number:02d}: {date.date()} complete")

    ts_path, metric_path = save_month("s1", frames, metrics)
    month_frame = pd.concat(frames, ignore_index=True)
    metric_frame = pd.DataFrame(metrics)
    figure = plot_month("s1", "Phase 3 S1: predefined operation",
                        month_frame, metric_frame, show_price=False)
    print(f"Timeseries: {ts_path}\nMetrics: {metric_path}\nFigure: {figure}")


if __name__ == "__main__":
    main(DEFAULT_SETTINGS.copy())
