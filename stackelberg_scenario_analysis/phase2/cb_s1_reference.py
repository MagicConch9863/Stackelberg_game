"""C-B S1: SimBench loads plus predefined BESS/EV schedules."""
import numpy as np

from cb_common import (BUS_CONFIG, BUS_ORDER, DEFAULT_SETTINGS, DT, T,
                       load_inputs, output_folders, plot_overview,
                       save_timeseries, trip_vector)


def energy(power, initial, loss=None):
    loss = np.zeros(T) if loss is None else loss
    return initial + np.r_[0.0, np.cumsum(power * DT - loss)]


def predefined_dispatch(bus):
    cfg = BUS_CONFIG[bus]
    bess = np.zeros(T)
    # Four hours charging and four hours discharging.  Limit the predefined
    # power by both the converter rating and the available energy headroom,
    # so S1 remains a physically feasible reference after resizing the BESS.
    interval_h = 16 * DT
    energy_headroom = min(cfg["bess_e_max"] - cfg["bess_e_init"],
                          cfg["bess_e_init"] - cfg["bess_e_min"])
    fixed_power = min(0.5 * cfg["bess_kw"], energy_headroom / interval_h)
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
    csv_folder, figure_folder = output_folders()
    loads, pv, _ = load_inputs(settings)
    result = {}
    for bus in BUS_ORDER:
        cfg = BUS_CONFIG[bus]
        bess, ev = predefined_dispatch(bus)
        net = loads[bus] - pv[bus] + bess + ev
        item = {"bess": bess, "ev": ev, "net": net,
                "import": np.maximum(net, 0), "export": np.maximum(-net, 0),
                "bess_energy": energy(bess, cfg["bess_e_init"])}
        if "ev_kwh" in cfg:
            item["ev_energy"] = energy(ev, .6 * cfg["ev_kwh"], trip_vector(bus))
        result[bus] = item
    aggregate = save_timeseries(csv_folder / "s1_timeseries.csv", loads, pv, result)
    figure = plot_overview(figure_folder, "s1", "C-B S1: predefined operation",
                           loads, pv, result)
    print(f"S1 complete: range={np.ptp(aggregate):.4f} kW\nCSV: {csv_folder}\nFigure: {figure}")


if __name__ == "__main__":
    main(DEFAULT_SETTINGS.copy())
