"""Common configuration, direct SimBench loading, CSV and overview plotting for C-B."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simbench as sb

T, DT = 96, 0.25
BUS_ORDER = ("Bus 1", "Bus 2", "Bus 3", "Bus 4")

# Reuse the four validated load buses from the earlier SimBench input checker.
SIMBENCH_MAPPING = {
    "Bus 1": {"name": "BUS6", "index": 6},
    "Bus 2": {"name": "BUS8", "index": 8},
    "Bus 3": {"name": "BUS14", "index": 14},
    "Bus 4": {"name": "BUS16", "index": 17},
}

# Same device structure and sizes as C-A.
BUS_CONFIG = {
    "Bus 1": {"bess_kwh": 8.0, "bess_kw": 2.5,
              "bess_e_min": .8, "bess_e_max": 8.0, "bess_e_init": 4.0,
              "ev_kwh": 40.0,
              "ev_kw_ch": 3.6, "ev_kw_dis": 1.0, "ev_trip_kwh": 4.0},
    "Bus 2": {"bess_kwh": 5.0, "bess_kw": 1.5,
              "bess_e_min": .5, "bess_e_max": 5.0, "bess_e_init": 2.5},
    "Bus 3": {"bess_kwh": 8.0, "bess_kw": 2.5,
              "bess_e_min": .8, "bess_e_max": 8.0, "bess_e_init": 4.0},
    "Bus 4": {"bess_kwh": 8.0, "bess_kw": 2.5,
              "bess_e_min": .8, "bess_e_max": 8.0, "bess_e_init": 4.0,
              "ev_kwh": 40.0,
              "ev_kw_ch": 3.6, "ev_kw_dis": 1.2, "ev_trip_kwh": 5.0},
}
COLORS = {"Bus 1": "tab:blue", "Bus 2": "tab:orange",
          "Bus 3": "tab:green", "Bus 4": "tab:red"}


def resolve_file(filename: str) -> Path:
    path = Path(filename)
    if path.is_file():
        return path
    here = Path(__file__).resolve()
    for folder in (here.parent, *here.parents):
        candidate = folder / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find {filename}")


def output_folders():
    root = Path(__file__).resolve().parent
    csv_folder, figure_folder = root / "csv_results", root / "figures"
    csv_folder.mkdir(parents=True, exist_ok=True)
    figure_folder.mkdir(parents=True, exist_ok=True)
    return csv_folder, figure_folder


def read_price(settings):
    frame = pd.read_csv(resolve_file(settings["PRICE_FILE"]))
    if settings["PRICE_COLUMN"] not in frame:
        raise KeyError(f"Missing price column {settings['PRICE_COLUMN']}")
    values = pd.to_numeric(frame[settings["PRICE_COLUMN"]], errors="coerce").dropna().to_numpy(float)
    if len(values) == 24:
        values = values[np.floor(np.arange(T) * DT).astype(int)]
    elif len(values) >= (settings["DAY_INDEX"] + 1) * T:
        start = settings["DAY_INDEX"] * T
        values = values[start:start + T]
    elif len(values) != T:
        values = np.interp(np.linspace(0, 1, T), np.linspace(0, 1, len(values)), values)
    if settings["PRICE_UNIT"].upper() == "EUR/MWH":
        values = values / 1000.0
    return values


def read_pv_profile(settings):
    """Read one daily PV shape and normalize it to a 0--1 per-unit curve."""
    frame = pd.read_csv(resolve_file(settings["PV_FILE"]))
    column = settings["PV_COLUMN"]
    if column not in frame:
        numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric:
            raise KeyError(f"Missing PV column {column} and no numeric fallback column")
        column = numeric[-1]
    values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(float)
    if len(values) >= (settings["DAY_INDEX"]+1)*T and len(values) != T:
        start = settings["DAY_INDEX"]*T
        values = values[start:start+T]
    elif len(values) == 24:
        values = np.interp(np.arange(T)*DT, np.arange(24), values)
    elif len(values) != T:
        values = np.interp(np.linspace(0, 1, T), np.linspace(0, 1, len(values)), values)
    values = np.maximum(values, 0.0)
    maximum = float(np.max(values))
    if maximum <= 0:
        raise ValueError("PV profile contains no positive generation")
    return values / maximum


def load_inputs(settings, need_price=False):
    """Read the chosen day directly from SimBench; no intermediate input CSV."""
    net = sb.get_simbench_net(settings["SIMBENCH_CODE"])
    profiles = sb.get_absolute_profiles_from_relative_profiles(
        net=net, element="load", multiplying_column="p_mw", time_as_index=False)
    start = settings["DAY_INDEX"] * T
    day = profiles.iloc[start:start + T]
    if len(day) != T:
        raise ValueError("DAY_INDEX exceeds the available SimBench time series")
    column_map = {str(column): column for column in day.columns}
    loads = {}
    for bus in BUS_ORDER:
        mapping = SIMBENCH_MAPPING[bus]
        sb_index = mapping["index"]
        actual_name = str(net.bus.at[sb_index, "name"])
        load_indices = net.load.index[net.load["bus"] == sb_index].tolist()
        columns = [column_map[str(i)] for i in load_indices if str(i) in column_map]
        if not columns:
            raise RuntimeError(
                f"{bus}: no load profile at SimBench {actual_name} (index {sb_index})")
        loads[bus] = day[columns].sum(axis=1).to_numpy(float) * 1000.0  # MW -> kW
        print(f"{bus} <- {actual_name} (index {sb_index}), load elements {load_indices}")
    pv_shape = read_pv_profile(settings) if settings["ENABLE_PV"] else np.zeros(T)
    capacities = settings["PV_CAPACITY_KW"]
    pv = {bus: pv_shape * float(capacities.get(bus, 0.0)) for bus in BUS_ORDER}
    pa = read_price(settings) if need_price else None
    return loads, pv, pa


def trip_vector(bus):
    trip = np.zeros(T)
    if "ev_kwh" in BUS_CONFIG[bus]:
        slots = np.r_[np.arange(28, 32), np.arange(68, 72)]
        trip[slots] = BUS_CONFIG[bus]["ev_trip_kwh"] / len(slots)
    return trip


def clean(values):
    values = np.asarray(values, dtype=float).copy()
    values[np.abs(values) < 1e-8] = 0.0
    return np.round(values, 8)


def select_export_price(pa, pb, mode):
    """Return the selling-price vector selected by the experiment setting."""
    choice = str(mode).strip().upper()
    if choice == "PA":
        return np.asarray(pa, dtype=float)
    if choice == "PB":
        return np.asarray(pb, dtype=float)
    raise ValueError("EXPORT_PRICE_MODE must be 'PA' or 'PB'")


def save_timeseries(path, loads, pv, result, pa=None, pb=None):
    data = {"time_h": np.arange(T) * DT}
    aggregate_load, aggregate_net = np.zeros(T), np.zeros(T)
    for bus in BUS_ORDER:
        key, item = bus.lower().replace(" ", "_"), result[bus]
        data[f"{key}_load_kw"] = clean(loads[bus])
        data[f"{key}_pv_kw"] = clean(pv[bus])
        for field in ("bess", "ev", "import", "export", "net"):
            data[f"{key}_{field}_kw"] = clean(item[field])
        data[f"{key}_bess_energy_kwh"] = clean(item["bess_energy"][:T])
        if "ev_energy" in item:
            data[f"{key}_ev_energy_kwh"] = clean(item["ev_energy"][:T])
        aggregate_load += loads[bus]
        aggregate_net += item["net"]
    data["aggregate_load_kw"] = clean(aggregate_load)
    aggregate_pv = sum(pv[bus] for bus in BUS_ORDER)
    data["aggregate_pv_kw"] = clean(aggregate_pv)
    data["aggregate_baseline_net_kw"] = clean(aggregate_load-aggregate_pv)
    data["aggregate_net_kw"] = clean(aggregate_net)
    if pa is not None:
        data["pa_eur_per_kwh"] = clean(pa)
    if pb is not None:
        data["pb_eur_per_kwh"] = clean(pb)
        data["total_buy_price_eur_per_kwh"] = clean(pa + pb)
    pd.DataFrame(data).to_csv(path, index=False)
    return aggregate_net


def plot_overview(folder, scenario, title, loads, pv, result, pa=None, pb=None):
    """One readable 4x2 figure instead of one separate image per bus."""
    time = np.arange(T) * DT
    aggregate_load = sum(loads[bus] for bus in BUS_ORDER)
    aggregate_pv = sum(pv[bus] for bus in BUS_ORDER)
    aggregate_net = sum(result[bus]["net"] for bus in BUS_ORDER)
    fig, ax = plt.subplots(4, 2, figsize=(16, 16), sharex=True)

    ax[0, 0].plot(time, aggregate_load, "--", color="0.45", lw=1.5, label="Load")
    ax[0, 0].plot(time, aggregate_load-aggregate_pv, ":", color="tab:green",
                  lw=1.7, label="Baseline (load - PV)")
    ax[0, 0].plot(time, aggregate_net, color="black", lw=2, label="Net exchange")
    ax[0, 0].set_title("Aggregate load and net exchange")
    ax[0, 0].legend()

    if pa is not None:
        total_price = np.asarray(pa) + np.asarray(pb)
        ax[0, 1].step(time, pa, where="post", lw=1.5, label="PA")
        ax[0, 1].step(time, pb, where="post", lw=1.5, label="PB")
        ax[0, 1].step(time, total_price, where="post", color="tab:green", lw=1.5,
                      label="PA + PB")
        ax[0, 1].set_ylabel("EUR/kWh")
        ax[0, 1].set_title(
            f"Common prices (PA+PB range={np.ptp(total_price):.4f} EUR/kWh)")
        ax[0, 1].legend(ncol=3)
    else:
        total_bess = sum(result[bus]["bess"] for bus in BUS_ORDER)
        total_ev = sum(result[bus]["ev"] for bus in BUS_ORDER)
        ax[0, 1].step(time, total_bess, where="post", label="Total BESS")
        ax[0, 1].step(time, total_ev, where="post", label="Total EV")
        ax[0, 1].set_title("Aggregate device power (+ charge, - discharge)")
        ax[0, 1].legend()

    for index, bus in enumerate(BUS_ORDER):
        row, col = 1 + index // 2, index % 2
        item, color = result[bus], COLORS[bus]
        ax[row, col].plot(time, loads[bus], "--", color="0.45", label="Load")
        if np.max(pv[bus]) > 0:
            ax[row, col].plot(time, -pv[bus], ":", color="tab:green", label="-PV")
        ax[row, col].plot(time, item["net"], color=color, lw=1.8, label="Net")
        ax[row, col].step(time, item["import"], where="post", color="tab:olive",
                          lw=1.0, alpha=.75, label="Import")
        ax[row, col].step(time, -np.asarray(item["export"]), where="post",
                          color="tab:pink", lw=1.0, alpha=.8, label="-Export")
        ax[row, col].step(time, item["bess"], where="post", color="tab:purple", alpha=.8,
                          label="BESS")
        if "ev_kwh" in BUS_CONFIG[bus]:
            ax[row, col].step(time, item["ev"], where="post", color="tab:cyan", alpha=.8,
                              label="EV")
        mapped = SIMBENCH_MAPPING[bus]["name"]
        import_kwh = float(np.sum(item["import"]) * DT)
        export_kwh = float(np.sum(item["export"]) * DT)
        lines = [f"Import: {import_kwh:.3f} kWh", f"Export: {export_kwh:.3f} kWh"]
        if pa is not None:
            buy_price = np.asarray(pa) + np.asarray(pb)
            sell_mode = str(DEFAULT_SETTINGS.get("EXPORT_PRICE_MODE", "PA")).upper()
            # The scenario may store its selected selling-price vector explicitly.
            sell_price = np.asarray(item.get("sell_price", pa), dtype=float)
            avg_buy = (float(np.dot(buy_price, item["import"]) /
                       np.sum(item["import"])) if np.sum(item["import"]) > 1e-10 else 0.0)
            avg_sell = (float(np.dot(sell_price, item["export"]) /
                        np.sum(item["export"])) if np.sum(item["export"]) > 1e-10 else 0.0)
            purchase_cost = float(np.dot(buy_price, item["import"]) * DT)
            sales_revenue = float(np.dot(sell_price, item["export"]) * DT)
            lines += [f"Avg buy: {avg_buy:.4f} EUR/kWh",
                      f"Avg sell ({item.get('sell_price_mode', sell_mode)}): "
                      f"{avg_sell:.4f} EUR/kWh",
                      f"Purchase: {purchase_cost:.3f} EUR",
                      f"Sales revenue: {sales_revenue:.3f} EUR"]
        ax[row, col].set_title(f"{bus} ({mapped})")
        ax[row, col].text(.01, .98, "\n".join(lines), transform=ax[row, col].transAxes,
                          va="top", ha="left", fontsize=7.5,
                          bbox=dict(facecolor="white", alpha=.75, edgecolor="0.8"))
        ax[row, col].legend(ncol=3, fontsize=7)

    for bus in BUS_ORDER:
        ax[3, 0].plot(time, result[bus]["bess_energy"][:T], color=COLORS[bus], label=bus)
    ax[3, 0].set_title("BESS stored energy")
    ax[3, 0].set_ylabel("kWh")
    ax[3, 0].legend(ncol=2)
    for bus in ("Bus 1", "Bus 4"):
        ax[3, 1].plot(time, result[bus]["ev_energy"][:T], color=COLORS[bus], label=bus)
    ax[3, 1].set_title("EV stored energy")
    ax[3, 1].set_ylabel("kWh")
    ax[3, 1].legend()

    for axis in ax.flat:
        axis.axhline(0, color="0.5", lw=.6)
        axis.grid(alpha=.25)
        axis.set_xlabel("Time [h]")
        if not axis.get_ylabel():
            axis.set_ylabel("kW")
    fig.suptitle(title, fontsize=16)
    fig.text(.5, .004,
             "Sign convention: Net < 0 = export; BESS/EV power < 0 = discharge.",
             ha="center", fontsize=9, color="0.35")
    fig.tight_layout(rect=(0, .015, 1, .975))
    path = folder / f"{scenario}_overview.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


DEFAULT_SETTINGS = {
    "SIMBENCH_CODE": "1-LV-urban6--0-no_sw",
    "DAY_INDEX": 0,
    "PRICE_FILE": "price_data.csv",
    "PRICE_COLUMN": "price_eur_per_mwh_1",
    "PRICE_UNIT": "EUR/MWH",
    "ENABLE_PV": True,
    "PV_FILE": "pv_2019-07-01_local.csv",
    "PV_COLUMN": "electricity",
    # Installed peak power. Set 0.0 to disconnect PV from that Bus.
    "PV_CAPACITY_KW": {"Bus 1": 3.0, "Bus 2": 2.5,
                       "Bus 3": 0.0, "Bus 4": 0.0},
    # Select "PA" or "PB" as the export remuneration in S2 and S3.
    "EXPORT_PRICE_MODE": "PA",
    # Tiny deterministic transaction cost preventing simultaneous import/export.
    "FLOW_TIE_BREAK_EUR_PER_KWH": 1e-5,
}
