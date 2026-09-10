from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.bus_scenario_generator import (
    BUS_ORDER,
    ForecastScenarioStore,
    generate_bus_scenarios,
)


PROJECT_ROOT = ROOT.parent
PHASE4 = PROJECT_ROOT / "stackelberg_scenario_analysis" / "phase4"
if str(PHASE4) not in sys.path:
    sys.path.insert(0, str(PHASE4))

from p4_common import (  # noqa: E402
    day_inputs,
    load_simbench_profiles,
    read_pa_year,
    read_pv_source,
)
from p4_run import SETTINGS as PHASE4_SETTINGS  # noqa: E402


DATES = (pd.Timestamp("2025-01-15"), pd.Timestamp("2025-01-16"))
BASE_SEED = 20250903
SIGMA_EPSILON_KW = 0.3
AR1_RHO = 0.9
PV_ACTIVE_THRESHOLD_KW = 0.01
MPC_HORIZON = 16
RESULTS = ROOT / "results" / "control_inputs" / "bus_scenarios"

BUS_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")


def configure_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def scenario_settings():
    settings = deepcopy(PHASE4_SETTINGS)
    settings.update({
        "LOAD_MODE": "annual",
        "SOLVER_OUTPUT": 0,
    })
    return settings


def plot_overview(frame, variable):
    label = "Load" if variable == "load" else "PV"
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.2), sharex=True)

    for row, date in enumerate(DATES):
        date_text = date.date().isoformat()
        for col, (bus, color) in enumerate(zip(BUS_ORDER, BUS_COLORS)):
            ax = axes[row, col]
            data = frame[
                (frame["date"] == date_text)
                & (frame["bus"] == bus)
                & (frame["variable"] == variable)
            ].sort_values("step")
            hour = data["step"].to_numpy() * 0.25
            ax.plot(hour, data["realization_kW"], color="#0072B2", lw=1.1,
                    label="Realization")
            ax.plot(hour, data["forecast_iid_kW"], color="#E69F00", lw=0.85,
                    alpha=0.9, label="IID forecast")
            ax.plot(hour, data["forecast_ar1_kW"], color="#009E73", lw=0.9,
                    label="AR(1) forecast")
            ax.set_title(f"{date_text}\n{bus}")
            ax.set_xlim(0, 24)
            ax.set_xticks([0, 6, 12, 18, 24])
            ax.grid(True, ls="--", lw=0.4, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(f"Four-bus {label} forecast scenarios", fontsize=13, y=0.995)
    fig.supxlabel("Time (h)")
    fig.supylabel(f"{label} (kW)")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.10, top=0.84,
                        wspace=0.25, hspace=0.42)
    fig.savefig(RESULTS / f"{variable}_scenario_overview.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def metadata_frame(settings):
    mapping = settings["PV_CAPACITY_KW"]
    records = [
        ("simbench_code", settings["SIMBENCH_CODE"]),
        ("load_mode", settings["LOAD_MODE"]),
        ("dates", ",".join(str(date.date()) for date in DATES)),
        ("base_seed", BASE_SEED),
        ("sigma_epsilon_kW", SIGMA_EPSILON_KW),
        ("ar1_rho", AR1_RHO),
        ("pv_active_threshold_kW", PV_ACTIVE_THRESHOLD_KW),
        ("mpc_horizon_steps", MPC_HORIZON),
        ("steps_per_day", 96),
    ]
    records.extend(
        (f"{bus.lower().replace(' ', '_')}_pv_capacity_kW", mapping[bus])
        for bus in BUS_ORDER
    )
    return pd.DataFrame(records, columns=["parameter", "value"])


def validate_store(store):
    checks = []
    for date in DATES:
        day = store.day_ahead(date, "iid")
        window = store.mpc_window(date, step=20, horizon=MPC_HORIZON, scenario="ar1")
        final_window = store.mpc_window(date, step=90, horizon=MPC_HORIZON, scenario="ar1")
        checks.append({
            "date": date.date().isoformat(),
            "day_ahead_length": len(day["load"]["Bus 1"]),
            "mpc_window_length_step20": len(window["load"]["Bus 1"]),
            "mpc_window_length_step90": len(final_window["load"]["Bus 1"]),
            "bus_count": len(day["load"]),
        })
        current = store.realization(date, 20)
        if set(current["load"]) != set(BUS_ORDER):
            raise RuntimeError("Realization interface returned an invalid bus set.")
    return pd.DataFrame(checks)


def main():
    configure_plot_style()
    RESULTS.mkdir(parents=True, exist_ok=True)
    settings = scenario_settings()

    print("[1/7] Reading PA/PV sources and annual four-bus SimBench profiles...", flush=True)
    pa_year, _ = read_pa_year(settings)
    pv_source = read_pv_source(settings)
    simbench_profiles, full_days = load_simbench_profiles(settings)

    scenario_parts = []
    validation_parts = []
    print("[2/7] Generating deterministic IID and AR(1) scenarios...", flush=True)
    for index, date in enumerate(DATES, start=1):
        print(f"  [{index}/{len(DATES)}] {date.date()}", flush=True)
        loads, pv, _ = day_inputs(
            date, simbench_profiles, full_days, pa_year, pv_source, settings
        )
        scenarios, validation = generate_bus_scenarios(
            date,
            loads,
            pv,
            settings["PV_CAPACITY_KW"],
            base_seed=BASE_SEED,
            sigma_epsilon_kw=SIGMA_EPSILON_KW,
            ar1_rho=AR1_RHO,
            pv_active_threshold_kw=PV_ACTIVE_THRESHOLD_KW,
        )
        scenario_parts.append(scenarios)
        validation_parts.append(validation)

    frame = pd.concat(scenario_parts, ignore_index=True)
    validation = pd.concat(validation_parts, ignore_index=True)

    print("[3/7] Checking deterministic reproducibility...", flush=True)
    first_date = DATES[0]
    loads, pv, _ = day_inputs(
        first_date, simbench_profiles, full_days, pa_year, pv_source, settings
    )
    repeated, _ = generate_bus_scenarios(
        first_date, loads, pv, settings["PV_CAPACITY_KW"],
        base_seed=BASE_SEED,
        sigma_epsilon_kw=SIGMA_EPSILON_KW,
        ar1_rho=AR1_RHO,
        pv_active_threshold_kw=PV_ACTIVE_THRESHOLD_KW,
    )
    original = frame[frame["date"] == first_date.date().isoformat()].reset_index(drop=True)
    if not repeated.equals(original):
        raise RuntimeError("Scenario reproducibility check failed.")

    print("[4/7] Checking physical constraints and aggregate consistency...", flush=True)
    if (frame.filter(regex=r"^forecast_(iid|ar1)_kW$") < -1e-12).any().any():
        raise RuntimeError("A negative physical forecast was detected.")
    pv_rows = frame[frame["variable"] == "pv"]
    inactive = ~pv_rows["pv_active"].astype(bool)
    if (pv_rows.loc[inactive, ["forecast_iid_kW", "forecast_ar1_kW"]] != 0.0).any().any():
        raise RuntimeError("Inactive-period PV forecast violation detected.")
    for bus in BUS_ORDER:
        capacity = float(settings["PV_CAPACITY_KW"][bus])
        bus_pv = pv_rows[pv_rows["bus"] == bus]
        if (bus_pv[["forecast_iid_kW", "forecast_ar1_kW"]] > capacity + 1e-12).any().any():
            raise RuntimeError(f"PV capacity violation detected for {bus}.")

    print("[5/7] Exercising day-ahead and 16-step MPC forecast interfaces...", flush=True)
    store = ForecastScenarioStore(frame)
    interface_checks = validate_store(store)

    print("[6/7] Saving scenario, metadata and validation CSV files...", flush=True)
    frame.to_csv(RESULTS / "forecast_scenarios.csv", index=False)
    validation.to_csv(RESULTS / "validation_summary.csv", index=False)
    metadata_frame(settings).to_csv(RESULTS / "scenario_metadata.csv", index=False)
    interface_checks.to_csv(RESULTS / "interface_checks.csv", index=False)

    print("[7/7] Creating four-bus Load/PV overview figures...", flush=True)
    plot_overview(frame, "load")
    plot_overview(frame, "pv")
    print(f"Completed. Results: {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
