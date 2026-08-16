# -*- coding: utf-8 -*-

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import simbench as sb


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")


def get_paths():
    here = Path(__file__).resolve().parent
    feeder = here if here.name == FEEDER_FOLDER else here / FEEDER_FOLDER
    project = feeder.parent
    out = feeder / OUTPUT_FOLDER
    out.mkdir(parents=True, exist_ok=True)
    return project, feeder, out


def find_file(project, feeder, filename):
    for path in (project / filename, feeder / filename):
        if path.exists():
            return path
    raise FileNotFoundError(filename)


def to_96(values, mode="LINEAR"):
    values = np.asarray(values, dtype=float)

    if len(values) == T:
        return values

    if len(values) > T and len(values) % T == 0:
        start = DAY_INDEX * T
        if start + T > len(values):
            raise ValueError("DAY_INDEX exceeds CSV time-series length.")
        return values[start:start + T]

    if len(values) != 24:
        raise ValueError(
            f"Expected 24, {T}, or full-year values; received {len(values)}."
        )

    target_hour = np.arange(T) * DT

    if mode.upper() == "STEP":
        return values[np.floor(target_hour).astype(int)]

    source_hour = np.arange(24, dtype=float)
    return np.interp(
        target_hour,
        np.append(source_hour, 24.0),
        np.append(values, values[0]),
    )


def read_numeric_csv(path, column):
    df = pd.read_csv(path)

    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' not found in {path.name}. "
            f"Available: {list(df.columns)}"
        )

    values = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy()

    if len(values) == 0:
        raise ValueError(f"No numeric data in {path.name}:{column}")

    return values


def load_simbench_profiles():
    net = sb.get_simbench_net(SIMBENCH_CODE)

    profiles = sb.get_absolute_profiles_from_relative_profiles(
        net=net,
        element="load",
        multiplying_column="p_mw",
        time_as_index=False,
    )

    start = DAY_INDEX * T
    day = profiles.iloc[start:start + T]

    if len(day) != T:
        raise ValueError("DAY_INDEX exceeds SimBench profile length.")

    column_map = {str(c): c for c in day.columns}
    loads, rows = {}, []

    for model_bus, (label, sb_bus) in LOAD_ASSIGNMENT.items():
        load_indices = net.load.index[net.load["bus"] == sb_bus].tolist()
        columns = [
            column_map[str(idx)]
            for idx in load_indices
            if str(idx) in column_map
        ]

        if not columns:
            raise RuntimeError(
                f"No SimBench load profile found for {model_bus}, "
                f"SimBench bus index {sb_bus}."
            )

        loads[model_bus] = (
            day[columns].sum(axis=1).to_numpy(dtype=float) * 1000.0
        )

        rows.append({
            "model_bus": model_bus,
            "plot_label": label,
            "simbench_bus_index": sb_bus,
            "simbench_bus_name": net.bus.at[sb_bus, "name"],
            "load_indices": ",".join(map(str, load_indices)),
        })

    return loads, pd.DataFrame(rows)


def load_external_data(project, feeder):
    pv_file = find_file(project, feeder, PV_FILE)
    price_file = find_file(project, feeder, PRICE_FILE)

    pv_raw = to_96(
        read_numeric_csv(pv_file, PV_COLUMN),
        mode="LINEAR",
    )

    pa = to_96(
        read_numeric_csv(price_file, PRICE_COLUMN),
        mode=PRICE_RESAMPLE,
    )

    if PRICE_UNIT.upper() == "EUR/MWH":
        pa = pa / 1000.0
    elif PRICE_UNIT.upper() != "EUR/KWH":
        raise ValueError("PRICE_UNIT must be EUR/MWh or EUR/kWh.")

    pv = {
        bus: pv_raw * scale
        for bus, scale in PV_SCALE_BY_BUS.items()
    }

    return pv, pa


def build_inputs(loads, pv, pa):
    data = {
        "step": np.arange(T),
        "hour": np.arange(T) * DT,
    }

    aggregate_load = np.zeros(T)
    aggregate_pv = np.zeros(T)

    for bus in LOAD_ASSIGNMENT:
        tag = safe_name(bus)
        load = loads[bus]
        pv_bus = pv.get(bus, np.zeros(T))

        data[f"load_{tag}_kw"] = load
        data[f"pv_{tag}_kw"] = pv_bus
        data[f"net_baseline_{tag}_kw"] = load - pv_bus

        aggregate_load += load
        aggregate_pv += pv_bus

    data["aggregate_load_baseline_kw"] = aggregate_load
    data["aggregate_pv_kw"] = aggregate_pv
    data["aggregate_net_baseline_kw"] = aggregate_load - aggregate_pv
    data["pa_eur_per_kwh"] = pa

    return pd.DataFrame(data)


def build_device_table():
    rows = []

    for bus, (label, sb_bus) in LOAD_ASSIGNMENT.items():
        ev = EV_CONFIG.get(bus, {})
        bess = BESS_CONFIG.get(bus, {})

        roles = ["LOAD"]
        if bus in EV_CONFIG:
            roles.append("EV")
        if bus in PV_SCALE_BY_BUS:
            roles.append("PV")
        if bus in BESS_CONFIG:
            roles.append("BESS")

        rows.append({
            "bus": bus,
            "profile": label,
            "simbench_bus_index": sb_bus,
            "devices": "+".join(roles),
            "pv_scale": PV_SCALE_BY_BUS.get(bus, np.nan),
            "ev_p_max_kw": ev.get("p_max_kw", np.nan),
            "ev_energy_req_kwh": ev.get("energy_req_kwh", np.nan),
            "bess_p_max_kw": bess.get("p_max_kw", np.nan),
            "bess_e_min_kwh": bess.get("e_min_kwh", np.nan),
            "bess_e_max_kwh": bess.get("e_max_kwh", np.nan),
            "bess_e_init_kwh": bess.get("e_init_kwh", np.nan),
            "bess_eta_ch": bess.get("eta_ch", np.nan),
            "bess_eta_dis": bess.get("eta_dis", np.nan),
        })

    return pd.DataFrame(rows)


def sanity_checks(inputs):
    rows = []

    def add(name, value, passed):
        rows.append({
            "check": name,
            "value": value,
            "status": "PASS" if passed else "FAIL",
        })

    core = [
        "aggregate_load_baseline_kw",
        "aggregate_pv_kw",
        "aggregate_net_baseline_kw",
        "pa_eur_per_kwh",
    ]

    identity_error = np.max(np.abs(
        inputs["aggregate_load_baseline_kw"]
        - inputs["aggregate_pv_kw"]
        - inputs["aggregate_net_baseline_kw"]
    ))

    device_buses = (
        set(EV_CONFIG)
        | set(PV_SCALE_BY_BUS)
        | set(BESS_CONFIG)
    )

    add("96 time steps", len(inputs), len(inputs) == T)
    add(
        "no missing core values",
        int(inputs[core].isna().sum().sum()),
        inputs[core].isna().sum().sum() == 0,
    )
    add(
        "nonnegative load",
        float(inputs["aggregate_load_baseline_kw"].min()),
        inputs["aggregate_load_baseline_kw"].min() >= 0,
    )
    add(
        "nonnegative PV",
        float(inputs["aggregate_pv_kw"].min()),
        inputs["aggregate_pv_kw"].min() >= 0,
    )
    add(
        "aggregate identity",
        float(identity_error),
        identity_error <= 1e-8,
    )
    add(
        "all devices assigned to active buses",
        sorted(device_buses),
        device_buses.issubset(LOAD_ASSIGNMENT),
    )

    for bus, cfg in EV_CONFIG.items():
        maximum_energy = cfg["p_max_kw"] * T * DT
        add(
            f"{bus} EV energy feasible",
            f"{cfg['energy_req_kwh']} <= {maximum_energy}",
            0 <= cfg["energy_req_kwh"] <= maximum_energy,
        )

    for bus, cfg in BESS_CONFIG.items():
        add(
            f"{bus} initial SOC feasible",
            cfg["e_init_kwh"],
            cfg["e_min_kwh"]
            <= cfg["e_init_kwh"]
            <= cfg["e_max_kwh"],
        )
        add(
            f"{bus} efficiency feasible",
            f"{cfg['eta_ch']}, {cfg['eta_dis']}",
            0 < cfg["eta_ch"] <= 1
            and 0 < cfg["eta_dis"] <= 1,
        )

    checks = pd.DataFrame(rows)

    if (checks["status"] == "FAIL").any():
        failed = checks.loc[
            checks["status"] == "FAIL", "check"
        ].tolist()
        raise RuntimeError(f"Sanity checks failed: {failed}")

    return checks


def plot_inputs(inputs, out):
    hour = inputs["hour"]

    fig, ax = plt.subplots(
        3,
        1,
        figsize=(12, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1.4, 1.0]},
    )

    for bus in LOAD_ASSIGNMENT:
        tag = safe_name(bus)
        ax[0].plot(
            hour,
            inputs[f"load_{tag}_kw"],
            label=bus,
        )

    ax[0].set_title("Selected uncontrollable load baselines")
    ax[0].set_ylabel("load [kW]")
    ax[0].grid(True, alpha=0.25)
    ax[0].legend(ncol=3)

    ax[1].plot(
        hour,
        inputs["aggregate_load_baseline_kw"],
        label="aggregate load baseline",
    )
    ax[1].plot(
        hour,
        inputs["aggregate_pv_kw"],
        label="PV total",
    )
    ax[1].plot(
        hour,
        inputs["aggregate_net_baseline_kw"],
        label="aggregate net baseline",
    )
    ax[1].set_ylabel("power [kW]")
    ax[1].grid(True, alpha=0.25)
    ax[1].legend()

    ax[2].step(
        hour,
        inputs["pa_eur_per_kwh"],
        where="post",
        label="PA",
    )
    ax[2].set_ylabel("EUR/kWh")
    ax[2].set_xlabel("hour")
    ax[2].grid(True, alpha=0.25)
    ax[2].legend()

    fig.tight_layout()
    fig.savefig(out / "phase1_data_sanity.png", dpi=180)
    plt.close(fig)


def main():
    project, feeder, out = get_paths()

    loads, assignment = load_simbench_profiles()
    pv, pa = load_external_data(project, feeder)

    inputs = build_inputs(loads, pv, pa)
    devices = build_device_table()
    checks = sanity_checks(inputs)

    inputs.to_csv(out / "phase1_inputs.csv", index=False)
    assignment.to_csv(out / "phase1_profile_assignment.csv", index=False)
    devices.to_csv(out / "phase1_device_config.csv", index=False)
    checks.to_csv(out / "phase1_sanity_checks.csv", index=False)

    plot_inputs(inputs, out)

    print("\n[Phase 1 completed]")
    print(f"Output: {out}")
    print(
        f"Load peak: "
        f"{inputs['aggregate_load_baseline_kw'].max():.3f} kW"
    )
    print(
        f"PV peak: "
        f"{inputs['aggregate_pv_kw'].max():.3f} kW"
    )
    print(
        f"Net baseline peak: "
        f"{inputs['aggregate_net_baseline_kw'].max():.3f} kW"
    )
    print(f"EV buses: {list(EV_CONFIG)}")
    print(f"PV buses: {list(PV_SCALE_BY_BUS)}")
    print(f"BESS buses: {list(BESS_CONFIG)}")


# ============================================================
# MODIFY PARAMETERS HERE
# ============================================================

FEEDER_FOLDER = "1_LV_urban_0_no_sw"
OUTPUT_FOLDER = "phase1_outputs"

SIMBENCH_CODE = "1-LV-urban6--0-no_sw"
DAY_INDEX = 0

T = 96
DT = 0.25

PV_FILE = "pv_2019-07-01_local.csv"
PV_COLUMN = "electricity"

PRICE_FILE = "price_data.csv"
PRICE_COLUMN = "price_eur_per_mwh_1"
PRICE_UNIT = "EUR/MWh"
PRICE_RESAMPLE = "STEP"

LOAD_ASSIGNMENT = {
    "Bus R1": ("BUS6", 6),
    "Bus R2": ("BUS8", 8),
    "Bus R3": ("BUS14", 14),
    "Bus R4": ("BUS16", 17),
    "Bus R5": ("BUS5", 5),
}

EV_CONFIG = {
    "Bus R2": {
        "p_max_kw": 3.7,
        "energy_req_kwh": 8.0,
    },
}

PV_SCALE_BY_BUS = {
    "Bus R4": 2.5,
    "Bus R5": 2.5,
}

BESS_CONFIG = {
    "Bus R1": {
        "p_max_kw": 2.5,
        "e_min_kwh": 0.8,
        "e_max_kwh": 8.0,
        "e_init_kwh": 4.0,
        "eta_ch": 0.95,
        "eta_dis": 0.95,
    },
    "Bus R2": {
        "p_max_kw": 1.5,
        "e_min_kwh": 0.5,
        "e_max_kwh": 5.0,
        "e_init_kwh": 2.5,
        "eta_ch": 0.95,
        "eta_dis": 0.95,
    },
    "Bus R3": {
        "p_max_kw": 2.5,
        "e_min_kwh": 0.8,
        "e_max_kwh": 8.0,
        "e_init_kwh": 4.0,
        "eta_ch": 0.95,
        "eta_dis": 0.95,
    },
}


if __name__ == "__main__":
    main()