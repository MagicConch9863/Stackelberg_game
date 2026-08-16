"""Compare Phase 3 monthly S1-S4 outputs after all four scripts have run."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIOS = ("s1", "s2", "s3", "s4")
SOFT_COLORS = ("#B0B0B0", "#8ECAE6", "#FFB4A2", "#A8D5BA")


def load_outputs(root):
    frames, daily = {}, {}
    for scenario in SCENARIOS:
        ts_path = root / "csv_results" / f"{scenario}_timeseries_month.csv"
        metric_path = root / "csv_results" / f"{scenario}_daily_metrics.csv"
        if not ts_path.is_file() or not metric_path.is_file():
            raise FileNotFoundError(
                f"Missing {scenario.upper()} output. Run p3_{scenario}_*.py first."
            )
        frames[scenario] = pd.read_csv(ts_path, parse_dates=["timestamp"])
        daily[scenario] = pd.read_csv(metric_path, parse_dates=["date"])
    return frames, daily


def summary_table(daily):
    rows = []
    for scenario in SCENARIOS:
        frame = daily[scenario]
        rows.append({
            "scenario": scenario.upper(),
            "days": len(frame),
            "mean_daily_peak_kw": frame["peak_kw"].mean(),
            "maximum_monthly_peak_kw": frame["peak_kw"].max(),
            "mean_daily_peak_valley_kw": frame["peak_valley_kw"].mean(),
            "p95_daily_peak_valley_kw": frame["peak_valley_kw"].quantile(0.95),
            "mean_daily_maximum_ramp_kw": frame["maximum_ramp_kw_per_15min"].mean(),
            "total_import_energy_kwh": frame["import_energy_kwh"].sum(),
            "total_export_energy_kwh": frame["export_energy_kwh"].sum(),
            "total_purchase_cost_eur": frame["purchase_cost_eur"].sum(min_count=1),
            "total_sales_revenue_eur": frame["sales_revenue_eur"].sum(min_count=1),
            "total_net_electricity_cost_eur": frame["net_electricity_cost_eur"].sum(min_count=1),
        })
    return pd.DataFrame(rows)


def main():
    root = Path(__file__).resolve().parent
    csv_folder = root / "csv_results"
    figure_folder = root / "figures"
    figure_folder.mkdir(parents=True, exist_ok=True)
    frames, daily = load_outputs(root)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True, sharey=True)
    for axis, scenario in zip(axes.flat, SCENARIOS):
        frame = frames[scenario]
        x = np.arange(len(frame)) * 0.25 / 24.0
        axis.plot(x, frame["aggregate_baseline_net_kw"], "--",
                  color="#8C8C8C", lw=0.75, label="Baseline (load - PV)")
        axis.plot(x, frame["aggregate_net_kw"], color=SOFT_COLORS[SCENARIOS.index(scenario)],
                  lw=0.9, label="Net exchange")
        axis.set_title(scenario.upper())
        axis.set_xlabel("Day of July 2026")
        axis.set_ylabel("kW")
        axis.grid(alpha=0.22)
        axis.legend()
    fig.suptitle("Phase 3: S1-S4 over the same fixed load and monthly PA/PV", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    comparison_path = figure_folder / "scenario_monthly_comparison_2x2.png"
    fig.savefig(comparison_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = summary_table(daily)
    summary_path = csv_folder / "scenario_monthly_summary.csv"
    summary.to_csv(summary_path, index=False)
    panels = (
        ("mean_daily_peak_kw", "Mean daily peak [kW]"),
        ("maximum_monthly_peak_kw", "Maximum monthly peak [kW]"),
        ("mean_daily_peak_valley_kw", "Mean daily peak-valley [kW]"),
        ("mean_daily_maximum_ramp_kw", "Mean daily maximum ramp [kW/15 min]"),
        ("total_import_energy_kwh", "Monthly import energy [kWh]"),
        ("total_export_energy_kwh", "Monthly export energy [kWh]"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for axis, (column, title) in zip(axes.flat, panels):
        values = summary[column].to_numpy(float)
        bars = axis.bar(summary["scenario"], values, color=SOFT_COLORS,
                        edgecolor="white", linewidth=0.8)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
        axis.axhline(0.0, color="#888888", lw=0.6)
        span = max(float(np.ptp(values)), float(np.max(np.abs(values))), 1.0)
        for bar, value in zip(bars, values):
            offset = 0.018 * span
            y = value + offset if value >= 0 else value - offset
            axis.text(bar.get_x() + bar.get_width() / 2, y, f"{value:.3f}",
                      ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    fig.suptitle("Phase 3 S1-S4 monthly performance comparison", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    bars_path = figure_folder / "scenario_monthly_performance_bars.png"
    fig.savefig(bars_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(summary.round(5).to_string(index=False))
    print(f"Summary: {summary_path}\nComparison: {comparison_path}\nBars: {bars_path}")


if __name__ == "__main__":
    main()
