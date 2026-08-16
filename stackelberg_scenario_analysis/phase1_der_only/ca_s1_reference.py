"""S1: predefined DER-only operation for four independent followers.

This file also contains the common data/configuration helpers imported by S2--S4.
Sign convention: import, BESS charging and EV charging are positive.

The BESS sizes were selected in advance with reference to the later SimBench
application, but this C-A test does not read or use any SimBench load data.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

T, DT = 96, 0.25
BUS_ORDER = ("Bus 1", "Bus 2", "Bus 3", "Bus 4")

# Final four-follower structure from the diagram.
BUS_CONFIG = {
    "Bus 1": {"bess_kwh": 5.0, "bess_kw": 1.0,
              "ev_kwh": 40.0, "ev_kw_ch": 3.6, "ev_kw_dis": 1.0,
              "ev_trip_kwh": 4.0, "pv_scale": 1.8},
    "Bus 2": {"bess_kwh": 4.0, "bess_kw": 0.8,
              "pv_scale": 1.8},
    "Bus 3": {"bess_kwh": 6.0, "bess_kw": 1.2,
              "pv_scale": 2.2},
    "Bus 4": {"bess_kwh": 6.0, "bess_kw": 1.2,
              "ev_kwh": 40.0, "ev_kw_ch": 3.6, "ev_kw_dis": 1.2,
              "ev_trip_kwh": 5.0, "pv_scale": 2.2},
}


def _read_series(path: Path, column: str, scale: float = 1.0) -> np.ndarray:
    frame = pd.read_csv(path)
    if column not in frame:
        raise KeyError(f"{path.name} has no '{column}'. Columns: {list(frame.columns)}")
    values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(float) * scale
    if len(values) == T:
        return values
    if not len(values):
        raise ValueError(f"No numeric values in {path}:{column}")
    return np.interp(np.linspace(0, 1, T), np.linspace(0, 1, len(values)), values)


def load_inputs(settings: dict, need_price: bool = False):
    # C-A is deliberately DER-only.  SimBench is used later, not in S1--S4.
    loads = {bus: np.zeros(T) for bus in BUS_ORDER}
    pv = {bus: np.zeros(T) for bus in BUS_ORDER}
    if settings["ENABLE_PV"]:
        raw = _read_series(Path(settings["PV_FILE"]), settings["PV_COLUMN"])
        pv = {bus: raw * BUS_CONFIG[bus]["pv_scale"] for bus in BUS_ORDER}
    pa = None
    if need_price:
        pa = _read_series(Path(settings["PRICE_FILE"]), settings["PRICE_COLUMN"],
                          0.001 if settings["PRICE_UNIT"].upper() == "EUR/MWH" else 1.0)
    configuration = pd.DataFrame([
        {"follower": bus, "base_load_used": False,
         "bess_capacity_kwh": BUS_CONFIG[bus]["bess_kwh"],
         "bess_power_kw": BUS_CONFIG[bus]["bess_kw"],
         "has_ev": "ev_kwh" in BUS_CONFIG[bus],
         "pv_enabled": settings["ENABLE_PV"]}
        for bus in BUS_ORDER
    ])
    return loads, pv, pa, configuration


def trip_vector(bus: str) -> np.ndarray:
    trip = np.zeros(T)
    if "ev_kwh" not in BUS_CONFIG[bus]:
        return trip
    slots = np.r_[np.arange(28, 32), np.arange(68, 72)]  # 07-08 and 17-18
    trip[slots] = BUS_CONFIG[bus]["ev_trip_kwh"] / len(slots)
    return trip


def energy_series(power: np.ndarray, initial: float, external_loss=None) -> np.ndarray:
    loss = np.zeros(T) if external_loss is None else external_loss
    return initial + np.r_[0.0, np.cumsum(power * DT - loss)]


def predefined_dispatch(bus: str):
    """Fixed, feasible schedules; no optimization and no price response."""
    cfg = BUS_CONFIG[bus]
    bess = np.zeros(T)
    # Charge 01:00-05:00 and discharge 18:00-22:00 with equal energy.
    bess[4:20] = cfg["bess_kw"] * 0.5
    bess[72:88] = -cfg["bess_kw"] * 0.5
    ev = np.zeros(T)
    if "ev_kwh" in cfg:
        ev[32:68] = -0.25 * cfg["ev_kw_dis"]
        home = np.r_[np.arange(0, 28), np.arange(72, 96)]
        required = -ev.sum() * DT + cfg["ev_trip_kwh"]
        ev[home] = required / (len(home) * DT)
        if ev[home[0]] > cfg["ev_kw_ch"] + 1e-9:
            raise ValueError(f"The predefined EV schedule is infeasible for {bus}")
    return bess, ev


def save_timeseries(path: Path, loads, pv, result: dict, pa=None, pb=None):
    data = {"time_h": np.arange(T) * DT}
    aggregate = np.zeros(T)
    for bus in BUS_ORDER:
        key = bus.lower().replace(" ", "_")
        for field in ("bess", "ev", "import", "export", "net", "bess_energy", "ev_energy"):
            if field in result[bus]:
                values = result[bus][field]
                data[f"{key}_{field}_kw" if "energy" not in field else f"{key}_{field}_kwh"] = values[:T]
        data[f"{key}_load_kw"] = loads[bus]
        data[f"{key}_pv_kw"] = pv[bus]
        aggregate += result[bus]["net"]
    data["aggregate_net_kw"] = aggregate
    if pa is not None:
        data["pa_eur_per_kwh"] = pa
    if pb is not None:
        for bus in BUS_ORDER:
            data[f"{bus.lower().replace(' ', '_')}_pb_eur_per_kwh"] = pb[bus]
    pd.DataFrame(data).to_csv(path, index=False)
    return aggregate


def main(settings: dict):
    out = Path(settings["OUTPUT_DIR"]); out.mkdir(parents=True, exist_ok=True)
    loads, pv, _, configuration = load_inputs(settings)
    result = {}
    for bus in BUS_ORDER:
        cfg = BUS_CONFIG[bus]
        bess, ev = predefined_dispatch(bus)
        net = loads[bus] - pv[bus] + bess + ev
        eb = energy_series(bess, 0.5 * cfg["bess_kwh"])
        item = {"bess": bess, "ev": ev, "net": net,
                "import": np.maximum(net, 0), "export": np.maximum(-net, 0),
                "bess_energy": eb}
        if "ev_kwh" in cfg:
            item["ev_energy"] = energy_series(ev, 0.6 * cfg["ev_kwh"], trip_vector(bus))
        result[bus] = item
    agg = save_timeseries(out / "s1_timeseries.csv", loads, pv, result)
    configuration.to_csv(out / "c_a_device_configuration.csv", index=False)
    print(f"S1 complete: aggregate range = {np.ptp(agg):.4f} kW")


if __name__ == "__main__":
    SETTINGS = {
        "ENABLE_PV": False, "PV_FILE": "pv_2019-07-01_local.csv", "PV_COLUMN": "electricity",
        "PRICE_FILE": "price_data.csv", "PRICE_COLUMN": "price_eur_per_mwh_1", "PRICE_UNIT": "EUR/MWH",
        "OUTPUT_DIR": "results_4bus",  # all paths are relative to the working directory
    }
    main(SETTINGS)
