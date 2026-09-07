from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PHASE4 = HERE.parent / "phase4"
if str(PHASE4) not in sys.path:
    sys.path.insert(0, str(PHASE4))

from p4_common import (  # noqa: E402
    BUS_ORDER,
    DT,
    T,
    aggregate_net,
    day_inputs,
    load_simbench_profiles,
    read_pa_year,
    read_pv_source,
)
from p4_run import SETTINGS as PHASE4_SETTINGS  # noqa: E402
from p4_s2_fixed_tariff import solve_day as solve_s2  # noqa: E402


DATES = (pd.Timestamp("2025-01-15"), pd.Timestamp("2025-07-15"))
PB_VALUES = np.round(np.arange(0.05, 0.51, 0.05), 2)
DISPATCH_PB_VALUES = (0.05, 0.20, 0.50)
RESULTS = HERE / "results"

COLORS = ("#0072B2", "#D55E00")


def settings_for_test():
    settings = deepcopy(PHASE4_SETTINGS)
    settings.update({
        "LOAD_MODE": "annual",
        "SOLVER_OUTPUT": 0,
    })
    return settings


def configure_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def compute_metrics(date, pb_value, pa, result):
    aggregate = aggregate_net(result)
    import_energy = sum(
        np.sum(result[bus]["import"]) * DT for bus in BUS_ORDER
    )
    export_energy = sum(
        np.sum(result[bus]["export"]) * DT for bus in BUS_ORDER
    )
    bess_throughput = sum(
        np.sum(np.abs(result[bus]["bess"])) * DT for bus in BUS_ORDER
    )
    pa_charge = sum(
        np.dot(pa, result[bus]["import"]) * DT for bus in BUS_ORDER
    )
    pb_charge = float(pb_value) * import_energy
    total_import_bill = pa_charge + pb_charge
    export_revenue = sum(
        np.dot(pa, result[bus]["export"]) * DT for bus in BUS_ORDER
    )

    row = {
        "date": date.date().isoformat(),
        "pb_eur_per_kwh": float(pb_value),
        "peak_kw": float(np.max(aggregate)),
        "valley_kw": float(np.min(aggregate)),
        "maximum_ramp_kw_per_15min": float(np.max(np.abs(np.diff(aggregate)))),
        "import_energy_kwh": float(import_energy),
        "export_energy_kwh": float(export_energy),
        "bess_throughput_kwh": float(bess_throughput),
        "pa_energy_charge_eur": float(pa_charge),
        "pb_grid_tariff_charge_eur": float(pb_charge),
        "total_import_bill_eur": float(total_import_bill),
        "export_revenue_eur": float(export_revenue),
        "net_electricity_cost_eur": float(total_import_bill - export_revenue),
        "pb_bill_share_percent": (
            100.0 * float(pb_charge) / float(total_import_bill)
            if abs(total_import_bill) > 1e-12 else np.nan
        ),
    }

    for bus in BUS_ORDER:
        key = bus.lower().replace(" ", "_")
        energy = np.asarray(result[bus]["bess_energy"], dtype=float)
        row[f"{key}_bess_throughput_kwh"] = float(
            np.sum(np.abs(result[bus]["bess"])) * DT
        )
        row[f"{key}_soc_min_kwh"] = float(np.min(energy))
        row[f"{key}_soc_max_kwh"] = float(np.max(energy))
        row[f"{key}_soc_final_kwh"] = float(energy[-1])

    return row


def timeseries_rows(date, pb_value, loads, pv, pa, result):
    rows = []
    aggregate_load = sum(np.asarray(loads[bus]) for bus in BUS_ORDER)
    aggregate_pv = sum(np.asarray(pv[bus]) for bus in BUS_ORDER)
    aggregate_bess = sum(np.asarray(result[bus]["bess"]) for bus in BUS_ORDER)
    aggregate = aggregate_net(result)

    for t in range(T):
        row = {
            "date": date.date().isoformat(),
            "pb_eur_per_kwh": float(pb_value),
            "step": t,
            "hour": t * DT,
            "pa_eur_per_kwh": float(pa[t]),
            "buy_price_eur_per_kwh": float(pa[t] + pb_value),
            "aggregate_load_kw": float(aggregate_load[t]),
            "aggregate_pv_kw": float(aggregate_pv[t]),
            "aggregate_bess_kw": float(aggregate_bess[t]),
            "aggregate_net_kw": float(aggregate[t]),
        }
        for bus in BUS_ORDER:
            key = bus.lower().replace(" ", "_")
            row[f"{key}_load_kw"] = float(loads[bus][t])
            row[f"{key}_pv_kw"] = float(pv[bus][t])
            row[f"{key}_bess_kw"] = float(result[bus]["bess"][t])
            row[f"{key}_net_kw"] = float(result[bus]["net"][t])
            row[f"{key}_soc_kwh"] = float(result[bus]["bess_energy"][t])
        rows.append(row)
    return rows


def plot_sensitivity(metrics):
    panels = (
        ("peak_kw", "Peak (kW)"),
        ("valley_kw", "Valley (kW)"),
        ("maximum_ramp_kw_per_15min", "Maximum ramp (kW/15 min)"),
        ("import_energy_kwh", "Import energy (kWh)"),
        ("export_energy_kwh", "Export energy (kWh)"),
        ("bess_throughput_kwh", "BESS throughput (kWh)"),
        ("total_import_bill_eur", "Total import bill (EUR)"),
        ("pb_grid_tariff_charge_eur", "PB tariff charge (EUR)"),
        ("pb_bill_share_percent", "PB bill share (%)"),
    )
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.0), sharex=True)

    for ax, (metric, label) in zip(axes.ravel(), panels):
        for color, date in zip(COLORS, DATES):
            subset = metrics[metrics["date"] == date.date().isoformat()].sort_values(
                "pb_eur_per_kwh"
            )
            ax.plot(
                subset["pb_eur_per_kwh"], subset[metric], marker="o", ms=3.5,
                lw=1.1, color=color, label=str(date.date()),
            )
        mean = metrics.groupby("pb_eur_per_kwh", as_index=False)[metric].mean()
        ax.plot(
            mean["pb_eur_per_kwh"], mean[metric], color="black", lw=1.7,
            ls="--", label="Two-day mean",
        )
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.set_xticks(PB_VALUES[::2])
        ax.grid(True, ls="--", lw=0.45, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("Sensitivity to the fixed grid tariff PB", fontsize=13, y=0.995)
    fig.supxlabel("Fixed PB (EUR/kWh)")
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.07, top=0.91,
                        wspace=0.34, hspace=0.32)
    fig.savefig(RESULTS / "pb_sensitivity_overview.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def plot_dispatch(timeseries):
    selected = timeseries[
        timeseries["pb_eur_per_kwh"].isin(DISPATCH_PB_VALUES)
    ].copy()
    fig, axes = plt.subplots(4, 3, figsize=(7.2, 8.2), sharex=True)

    for date_index, date in enumerate(DATES):
        date_text = date.date().isoformat()
        for col, pb_value in enumerate(DISPATCH_PB_VALUES):
            frame = selected[
                (selected["date"] == date_text)
                & np.isclose(selected["pb_eur_per_kwh"], pb_value)
            ].sort_values("step")
            hour = frame["hour"].to_numpy()

            dispatch_ax = axes[2 * date_index, col]
            dispatch_ax.plot(hour, frame["aggregate_net_kw"], color="#0072B2",
                             lw=1.1, label="Net exchange")
            dispatch_ax.plot(hour, frame["aggregate_bess_kw"], color="#D55E00",
                             lw=1.0, label="BESS power")
            dispatch_ax.axhline(0.0, color="0.3", lw=0.5)
            dispatch_ax.set_title(f"{date_text}, PB={pb_value:.2f}")
            dispatch_ax.set_ylabel("Power (kW)")
            dispatch_ax.grid(True, ls="--", lw=0.4, alpha=0.3)

            soc_ax = axes[2 * date_index + 1, col]
            for bus, color in zip(BUS_ORDER, ("#0072B2", "#E69F00", "#009E73", "#CC79A7")):
                key = bus.lower().replace(" ", "_")
                soc_ax.plot(hour, frame[f"{key}_soc_kwh"], lw=0.9,
                            color=color, label=bus)
            soc_ax.set_ylabel("BESS energy (kWh)")
            soc_ax.set_xlim(0, 24)
            soc_ax.set_xticks([0, 6, 12, 18, 24])
            soc_ax.grid(True, ls="--", lw=0.4, alpha=0.3)

    dispatch_handles, dispatch_labels = axes[0, 0].get_legend_handles_labels()
    soc_handles, soc_labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(dispatch_handles + soc_handles, dispatch_labels + soc_labels,
               loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.975))
    fig.suptitle("S2 dispatch at selected fixed PB values", fontsize=13, y=0.998)
    fig.supxlabel("Time (h)")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.06, top=0.925,
                        wspace=0.27, hspace=0.38)
    fig.savefig(RESULTS / "pb_dispatch_comparison.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


def main():
    configure_plot_style()
    RESULTS.mkdir(parents=True, exist_ok=True)
    settings = settings_for_test()

    print("[1/5] Reading PA, PV and annual four-bus SimBench profiles...", flush=True)
    pa_year, _ = read_pa_year(settings)
    pv_source = read_pv_source(settings)
    simbench_profiles, full_days = load_simbench_profiles(settings)

    metrics_rows = []
    series_rows = []
    total_runs = len(DATES) * len(PB_VALUES)
    run_index = 0

    print(f"[2/5] Solving {total_runs} date-PB combinations...", flush=True)
    for date in DATES:
        loads, pv, pa = day_inputs(
            date, simbench_profiles, full_days, pa_year, pv_source, settings
        )
        for pb_value in PB_VALUES:
            run_index += 1
            print(
                f"  [{run_index:02d}/{total_runs}] date={date.date()}, "
                f"PB={pb_value:.2f} EUR/kWh",
                flush=True,
            )
            run_settings = deepcopy(settings)
            run_settings["FIXED_PB_EUR_PER_KWH"] = float(pb_value)
            result, _ = solve_s2(loads, pv, pa, run_settings)
            metrics_rows.append(compute_metrics(date, pb_value, pa, result))
            series_rows.extend(
                timeseries_rows(date, pb_value, loads, pv, pa, result)
            )

    print("[3/5] Saving metrics and time-series CSV files...", flush=True)
    metrics = pd.DataFrame(metrics_rows)
    timeseries = pd.DataFrame(series_rows)
    metrics.to_csv(RESULTS / "pb_sensitivity_metrics.csv", index=False)
    timeseries.to_csv(RESULTS / "pb_sensitivity_timeseries.csv", index=False)

    print("[4/5] Creating the 3x3 sensitivity overview...", flush=True)
    plot_sensitivity(metrics)
    print("[5/5] Creating dispatch comparison for PB=0.05, 0.20, 0.50...", flush=True)
    plot_dispatch(timeseries)
    print(f"Completed. Results: {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
