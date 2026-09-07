from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import gurobipy as gp
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
PHASE4 = PROJECT_ROOT / "stackelberg_scenario_analysis" / "phase4"
if str(PHASE4) not in sys.path:
    sys.path.insert(0, str(PHASE4))

from p4_common import (  # noqa: E402
    BUS_CONFIG,
    BUS_ORDER,
    DT,
    SIMBENCH_MAPPING,
    daily_metrics,
    make_result_item,
    read_pa_year,
)
from p4_run import SETTINGS as PHASE4_SETTINGS  # noqa: E402
from p4_s3_stackelberg import solve_day as solve_stackelberg_day  # noqa: E402


def solve_day_with_feasibility_focus(base_solver, date, loads, pv, pa, settings):
    """Run the Phase 4 OL solver with forecast-experiment search defaults."""
    run_settings = dict(settings)
    run_settings["TIME_LIMIT_S_PER_DAY"] = 1200
    run_settings["MIP_GAP"] = 0.05
    run_settings["MIP_FOCUS"] = 1
    gp.setParam("Heuristics", 0.50)
    gp.setParam("NoRelHeurTime", 60)
    gp.setParam("PumpPasses", 10)
    gp.setParam("RINS", 10)
    print(
        "[OL solver] feasibility focus: Heuristics=0.50, "
        "NoRelHeurTime=60s, PumpPasses=10, RINS=10, TimeLimit=1200s",
        flush=True,
    )
    return base_solver(date, loads, pv, pa, run_settings)


SCENARIO_FILE = (
    ROOT / "results" / "control_inputs" / "bus_scenarios" / "forecast_scenarios.csv"
)
EXPERIMENTS = {
    "reference": {
        "label": "Reference",
        "forecast_column": "forecast_perfect_kW",
        "output": ROOT / "reference_ol_ol" / "results",
    },
    "t1": {
        "label": "T1 IID Gaussian",
        "forecast_column": "forecast_iid_kW",
        "output": ROOT / "t1_iid_ol_ol" / "results",
    },
    "t2": {
        "label": "T2 AR(1) Gaussian",
        "forecast_column": "forecast_ar1_kW",
        "output": ROOT / "t2_ar1_ol_ol" / "results",
    },
}


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


def experiment_settings():
    settings = deepcopy(PHASE4_SETTINGS)
    settings.update({
        "LOAD_MODE": "annual",
        "SOLVER_OUTPUT": 1,
    })
    return settings


def load_scenario_frame():
    if not SCENARIO_FILE.exists():
        raise FileNotFoundError(
            f"Missing four-bus scenarios: {SCENARIO_FILE}\n"
            "Run forecast_mpc/run_bus_scenarios.py first."
        )
    frame = pd.read_csv(SCENARIO_FILE, parse_dates=["timestamp"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame


def _arrays_for_day(frame, date, value_column):
    day = frame[frame["date"] == pd.Timestamp(date).normalize()]
    if day.empty:
        raise KeyError(f"No scenario data for {pd.Timestamp(date).date()}.")
    result = {"load": {}, "pv": {}}
    for variable in ("load", "pv"):
        for bus in BUS_ORDER:
            values = day[
                (day["variable"] == variable) & (day["bus"] == bus)
            ].sort_values("step")[value_column].to_numpy(float)
            if len(values) != 96:
                raise RuntimeError(f"{date} {bus} {variable}: expected 96 values.")
            result[variable][bus] = values
    return result


def _realized_result(realized_loads, realized_pv, planned_result):
    result = {}
    for bus in BUS_ORDER:
        bess = np.asarray(planned_result[bus]["bess"], dtype=float)
        net = np.asarray(realized_loads[bus]) - np.asarray(realized_pv[bus]) + bess
        result[bus] = make_result_item(
            bess=bess,
            import_kw=np.maximum(net, 0.0),
            export_kw=np.maximum(-net, 0.0),
            bess_energy=np.asarray(planned_result[bus]["bess_energy"], dtype=float),
        )
    return result


def _forecast_scores(forecast, realized):
    rows = {}
    for variable in ("load", "pv"):
        actual = np.concatenate([realized[variable][bus] for bus in BUS_ORDER])
        predicted = np.concatenate([forecast[variable][bus] for bus in BUS_ORDER])
        error = predicted - actual
        rows[f"{variable}_forecast_MAE_kW"] = float(np.mean(np.abs(error)))
        rows[f"{variable}_forecast_RMSE_kW"] = float(np.sqrt(np.mean(error ** 2)))
    return rows


def _result_metrics(prefix, date, result, pa, pb):
    values = daily_metrics("s3", date, result, pa=pa, pb=pb)
    keep = {
        key: value for key, value in values.items()
        if key not in ("scenario", "date", "month")
    }
    keep["bess_throughput_kwh"] = float(sum(
        np.sum(np.abs(result[bus]["bess"])) * DT for bus in BUS_ORDER
    ))
    return {f"{prefix}_{key}": value for key, value in keep.items()}


def _timeseries_rows(experiment, date, pa, pb, forecast, realized, planned, actual):
    rows = []
    for step in range(96):
        for bus in BUS_ORDER:
            rows.append({
                "experiment": experiment,
                "date": pd.Timestamp(date).date().isoformat(),
                "step": step,
                "hour": step * DT,
                "bus": bus,
                "pa_eur_per_kwh": pa[step],
                "pb_eur_per_kwh": pb[step],
                "load_forecast_kW": forecast["load"][bus][step],
                "load_real_kW": realized["load"][bus][step],
                "pv_forecast_kW": forecast["pv"][bus][step],
                "pv_real_kW": realized["pv"][bus][step],
                "bess_planned_kW": planned[bus]["bess"][step],
                "soc_planned_kWh": planned[bus]["bess_energy"][step],
                "planned_import_kW": planned[bus]["import"][step],
                "planned_export_kW": planned[bus]["export"][step],
                "realized_import_kW": actual[bus]["import"][step],
                "realized_export_kW": actual[bus]["export"][step],
                "planned_net_kW": planned[bus]["net"][step],
                "realized_net_kW": actual[bus]["net"][step],
            })
    return rows


def plot_experiment(timeseries, label, destination):
    dates = sorted(timeseries["date"].unique())
    fig, axes = plt.subplots(len(dates), 3, figsize=(10.2, 4.8), squeeze=False)
    axes = np.atleast_2d(axes)
    for row, date in enumerate(dates):
        day = timeseries[timeseries["date"] == date]
        grouped = day.groupby("step", as_index=False).agg({
            "hour": "first",
            "load_real_kW": "sum",
            "pv_real_kW": "sum",
            "planned_net_kW": "sum",
            "realized_net_kW": "sum",
            "bess_planned_kW": "sum",
            "pb_eur_per_kwh": "first",
        })
        hour = grouped["hour"]
        uncontrolled_net = grouped["load_real_kW"] - grouped["pv_real_kW"]
        ax = axes[row, 0]
        ax.plot(hour, uncontrolled_net, color="#6E6E6E", ls=":",
                lw=1.1, label="Uncontrolled net")
        ax.plot(hour, grouped["planned_net_kW"], color="#E69F00", ls="--",
                lw=1.0, label="Planned net")
        ax.plot(hour, grouped["realized_net_kW"], color="#0072B2", lw=1.1,
                label="Realized net")
        ax.axhline(0.0, color="0.35", lw=0.5)
        ax.set_title(f"{date}: planned vs realized")
        ax.set_ylabel("Net exchange (kW)")
        ax.grid(True, ls="--", lw=0.4, alpha=0.3)
        ax.set_xlabel("Time (h)")

        ax = axes[row, 1]
        ax.plot(hour, grouped["bess_planned_kW"], color="#009E73", lw=1.0,
                label="Aggregate BESS")
        ax.set_ylabel("BESS power (kW)")
        ax.set_title(f"{date}: BESS dispatch")
        ax.grid(True, ls="--", lw=0.4, alpha=0.3)
        ax.set_xlabel("Time (h)")

        ax = axes[row, 2]
        ax.step(hour, grouped["pb_eur_per_kwh"], where="post",
                color="#CC79A7", lw=1.0, label="PB")
        ax.set_ylabel("PB (EUR/kWh)")
        ax.set_title(f"{date}: grid tariff")
        ax.grid(True, ls="--", lw=0.4, alpha=0.3)
        ax.set_xlabel("Time (h)")

        for panel in axes[row, :]:
            panel.set_xlim(0.0, 24.0)
            panel.set_xticks(np.arange(0.0, 25.0, 4.0))

    handles, labels = [], []
    for ax in axes[0, :]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(f"{label}: 24h one-shot Stackelberg OL", fontsize=13, y=0.995)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.10, top=0.86,
                        wspace=0.36, hspace=0.55)
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_experiment(key):
    key = key.lower()
    if key not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {key}")
    configure_plot_style()
    config = EXPERIMENTS[key]
    output = config["output"]
    output.mkdir(parents=True, exist_ok=True)
    settings = experiment_settings()
    frame = load_scenario_frame()
    dates = sorted(frame["date"].unique())
    pa_year, _ = read_pa_year(settings)
    metrics_path = output / "daily_metrics.csv"
    series_path = output / "planned_and_realized_timeseries.csv"
    if metrics_path.exists() and series_path.exists():
        existing_metrics = pd.read_csv(metrics_path)
        existing_series = pd.read_csv(series_path)
    else:
        existing_metrics = pd.DataFrame()
        existing_series = pd.DataFrame()
    if not existing_metrics.empty:
        converged = existing_metrics["solver_mip_gap"].astype(float) <= 0.05 + 1e-12
        retry_dates = set(existing_metrics.loc[~converged, "date"].astype(str))
        if retry_dates:
            print(
                f"[{config['label']}] retrying non-converged checkpoints: "
                f"{', '.join(sorted(retry_dates))}",
                flush=True,
            )
        existing_metrics = existing_metrics.loc[converged].copy()
        existing_series = existing_series.loc[
            ~existing_series["date"].astype(str).isin(retry_dates)
        ].copy()
    completed_dates = set(existing_metrics.get("date", pd.Series(dtype=str)).astype(str))
    metrics_rows = existing_metrics.to_dict("records")
    series_rows = existing_series.to_dict("records")

    print(f"[{config['label']}] Starting {len(dates)} daily OL solves...", flush=True)
    for index, date in enumerate(dates, start=1):
        date = pd.Timestamp(date)
        date_text = date.date().isoformat()
        if date_text in completed_dates:
            print(f"[{config['label']}] day {index}/{len(dates)}: {date_text} checkpoint found; skipping.", flush=True)
            continue
        print(f"[{config['label']}] day {index}/{len(dates)}: {date.date()}", flush=True)
        forecast = _arrays_for_day(frame, date, config["forecast_column"])
        realized = _arrays_for_day(frame, date, "realization_kW")
        pa = pa_year.loc[(pa_year.index >= date) & (pa_year.index < date + pd.Timedelta(days=1))]
        if len(pa) != 96:
            raise RuntimeError(f"{date.date()}: expected 96 PA values, got {len(pa)}.")
        pa = pa.to_numpy(float)

        started = time.perf_counter()
        planned, pb, mip_gap = solve_day_with_feasibility_focus(
            solve_stackelberg_day,
            date, forecast["load"], forecast["pv"], pa, settings
        )
        runtime = time.perf_counter() - started
        actual = _realized_result(realized["load"], realized["pv"], planned)

        row = {
            "experiment": key,
            "label": config["label"],
            "date": date.date().isoformat(),
            "solver_wall_time_s": runtime,
            "solver_mip_gap": mip_gap,
            "solver_converged_5pct": bool(mip_gap <= 0.05 + 1e-12),
            "solver_result_status": (
                "converged_5pct" if mip_gap <= 0.05 + 1e-12
                else "time_limit_feasible_not_converged"
            ),
        }
        row.update(_forecast_scores(forecast, realized))
        row.update(_result_metrics("planned", date, planned, pa, pb))
        row.update(_result_metrics("realized", date, actual, pa, pb))
        metrics_rows.append(row)
        series_rows.extend(
            _timeseries_rows(key, date, pa, pb, forecast, realized, planned, actual)
        )
        pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
        pd.DataFrame(series_rows).to_csv(series_path, index=False)
        print(
            f"[{config['label']}] completed {date.date()} in {runtime:.1f}s, "
            f"MIP gap={mip_gap:.4g}; checkpoint saved", flush=True,
        )

    metrics = pd.DataFrame(metrics_rows)
    timeseries = pd.DataFrame(series_rows)
    metrics["solver_converged_5pct"] = metrics["solver_mip_gap"] <= 0.05 + 1e-12
    metrics["solver_result_status"] = np.where(
        metrics["solver_converged_5pct"],
        "converged_5pct",
        "time_limit_feasible_not_converged",
    )
    metrics.to_csv(metrics_path, index=False)
    timeseries.to_csv(series_path, index=False)
    plot_experiment(timeseries, config["label"], output / "pb_and_dispatch_overview.png")
    print(f"[{config['label']}] Results: {output}", flush=True)
    return metrics


def plot_system_configuration(destination, settings=None):
    configure_plot_style()
    settings = experiment_settings() if settings is None else settings
    rows = []
    for bus in BUS_ORDER:
        cfg = BUS_CONFIG[bus]
        rows.append([
            bus,
            SIMBENCH_MAPPING[bus]["index"],
            f"{settings['PV_CAPACITY_KW'][bus]:.1f}",
            f"{cfg['bess_kwh']:.1f}",
            f"±{cfg['bess_kw']:.1f}",
            f"{cfg['bess_e_min']:.1f}–{cfg['bess_e_max']:.1f}",
        ])
    fig, ax = plt.subplots(figsize=(8.4, 1.9))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Follower", "SimBench bus", "PV (kW)", "BESS (kWh)",
                   "BESS power (kW)", "SOC range (kWh)"],
        colWidths=[0.15, 0.17, 0.13, 0.15, 0.20, 0.20],
        cellLoc="center", colLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.45)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#B8C6D9")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#DCE6F1")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("white")
    ax.set_title("Phase 4 four-bus system configuration", fontsize=13, pad=8)
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_comparison(all_metrics):
    configure_plot_style()
    output = ROOT / "reference_t1_t2_comparison" / "results"
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(all_metrics, ignore_index=True)
    frame["solver_converged_5pct"] = frame["solver_mip_gap"] <= 0.05 + 1e-12
    frame["solver_result_status"] = np.where(
        frame["solver_converged_5pct"],
        "converged_5pct",
        "time_limit_feasible_not_converged",
    )
    frame.to_csv(output / "comparison_metrics.csv", index=False)
    plot_system_configuration(output / "system_configuration.png")

    panels = (
        ("realized_peak_kw", "Realized peak (kW)"),
        ("realized_maximum_ramp_kw_per_15min", "Maximum ramp (kW/15 min)"),
        ("realized_import_energy_kwh", "Import energy (kWh)"),
        ("realized_export_energy_kwh", "Export energy (kWh)"),
        ("realized_net_electricity_cost_eur", "Net cost (EUR)"),
        ("realized_bess_throughput_kwh", "BESS throughput (kWh)"),
    )
    valid = frame[frame["solver_converged_5pct"]].copy()
    if valid.empty:
        raise RuntimeError("No solutions reached the 5% MIP-gap threshold.")
    common_dates = (
        valid.groupby("date")["experiment"].nunique()
        .loc[lambda counts: counts == len(EXPERIMENTS)]
        .index
    )
    valid = valid[valid["date"].isin(common_dates)].copy()
    if valid.empty:
        raise RuntimeError(
            "No date has converged solutions for Reference, T1 and T2 together."
        )
    labels = [EXPERIMENTS[key]["label"] for key in EXPERIMENTS]
    colors = ["#0072B2", "#E69F00", "#009E73"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    for ax, (metric, title) in zip(axes.ravel(), panels):
        for date_index, date in enumerate(sorted(valid["date"].unique())):
            day = valid[valid["date"] == date]
            values = [float(day[day["experiment"] == key][metric].iloc[0]) for key in EXPERIMENTS]
            offset = (-0.14, 0.14)[date_index]
            ax.bar(np.arange(3) + offset, values, width=0.27,
                   color=colors, alpha=0.65 + 0.25 * date_index,
                   edgecolor="black", linewidth=0.35,
                   label=date if ax is axes[0, 0] else None)
        ax.set_title(title)
        ax.set_xticks(range(3), labels, rotation=18)
        ax.grid(axis="y", ls="--", lw=0.4, alpha=0.3)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 0.97))
    fig.suptitle("Reference, T1 and T2: converged realized OL/OL results", fontsize=13, y=0.998)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.87,
                        wspace=0.30, hspace=0.42)
    fig.savefig(output / "realized_performance_comparison.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    return frame
