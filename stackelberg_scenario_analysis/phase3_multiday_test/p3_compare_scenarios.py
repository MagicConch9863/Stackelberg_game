"""Create line/bar S1-S4 comparisons from existing Phase-3 CSV outputs."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCENARIOS = ("s1", "s2", "s3", "s4")
COLORS = {"s1": "#5B6573", "s2": "#2F6FA3",
          "s3": "#245A8D", "s4": "#3F7D58"}
BASELINE_COLOR, DT = "#555555", 0.25

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8, "figure.titlesize": 14,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": "white",
})


def load_outputs(root):
    frames = {}
    for scenario in SCENARIOS:
        path = root / "csv_results" / f"{scenario}_timeseries_month.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path.name}. Run the corresponding Phase-3 scenario first.")
        frames[scenario] = pd.read_csv(path, parse_dates=["timestamp"])
    return frames


def recompute_daily(frame):
    required = {"date", "aggregate_net_kw", "aggregate_baseline_net_kw"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Timeseries is missing columns: {sorted(missing)}")
    rows = []
    for date, day in frame.groupby("date", sort=True):
        net = day["aggregate_net_kw"].to_numpy(float)
        baseline = day["aggregate_baseline_net_kw"].to_numpy(float)
        rows.append({
            "date": date, "peak_kw": float(net.max()),
            "valley_kw": float(net.min()),
            "peak_valley_kw": float(net.max() - net.min()),
            "maximum_ramp_kw_per_15min": float(np.abs(np.diff(net)).max()),
            "baseline_peak_kw": float(baseline.max()),
            "baseline_valley_kw": float(baseline.min()),
        })
    return pd.DataFrame(rows)


def energy_and_money(frame):
    keys = ("bus_1", "bus_2", "bus_3", "bus_4")
    imported = sum(frame[f"{key}_import_kw"].sum() for key in keys) * DT
    exported = sum(frame[f"{key}_export_kw"].sum() for key in keys) * DT
    purchase_cols = [f"{key}_purchase_cost_eur" for key in keys]
    revenue_cols = [f"{key}_sales_revenue_eur" for key in keys]
    purchase = (sum(frame[col].sum() for col in purchase_cols)
                if all(col in frame for col in purchase_cols) else np.nan)
    revenue = (sum(frame[col].sum() for col in revenue_cols)
               if all(col in frame for col in revenue_cols) else np.nan)
    return float(imported), float(exported), float(purchase), float(revenue)


def summary_table(frames):
    rows = []
    for scenario in SCENARIOS:
        daily = recompute_daily(frames[scenario])
        imported, exported, purchase, revenue = energy_and_money(frames[scenario])
        rows.append({
            "scenario": scenario.upper(), "days": len(daily),
            "mean_daily_peak_kw": daily["peak_kw"].mean(),
            "maximum_monthly_peak_kw": daily["peak_kw"].max(),
            "mean_daily_valley_kw": daily["valley_kw"].mean(),
            "minimum_monthly_valley_kw": daily["valley_kw"].min(),
            "mean_daily_peak_valley_kw": daily["peak_valley_kw"].mean(),
            "p95_daily_peak_valley_kw": daily["peak_valley_kw"].quantile(0.95),
            "mean_daily_maximum_ramp_kw": daily["maximum_ramp_kw_per_15min"].mean(),
            "total_import_energy_kwh": imported,
            "total_export_energy_kwh": exported,
            "total_purchase_cost_eur": purchase,
            "total_sales_revenue_eur": revenue,
            "total_net_electricity_cost_eur": purchase - revenue,
        })
    return pd.DataFrame(rows)


def _draw(axis, x, values, style, color, label):
    if style == "bar":
        axis.bar(x, values, width=DT / 24.0 * 0.90, color=color,
                 edgecolor="none", alpha=0.82, label=label)
    else:
        axis.plot(x, values, color=color, lw=1.0, label=label)


def comparison_figure(frames, folder, style):
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 2, figsize=(15, 15),
                             gridspec_kw={"width_ratios": [2.25, 1]},
                             sharex="col")
    for row, scenario in enumerate(SCENARIOS):
        frame = frames[scenario]
        x = np.arange(len(frame)) * DT / 24.0
        power, price = axes[row]
        power.plot(x, frame["aggregate_baseline_net_kw"], "--",
                   color=BASELINE_COLOR, lw=0.85, label="Baseline (load - PV)")
        _draw(power, x, frame["aggregate_net_kw"], style, COLORS[scenario],
              f"{scenario.upper()} net exchange")
        power.axhline(0.0, color="#777777", lw=0.55)
        power.set_title(f"{scenario.upper()} — aggregate net exchange", loc="left")
        power.set_ylabel("kW")
        power.grid(axis="y", alpha=0.28, linewidth=0.55)
        power.legend(frameon=False, ncol=2)
        power.margins(y=0.12)

        if "pa_eur_per_kwh" in frame:
            price.plot(x, frame["pa_eur_per_kwh"], color="#245A8D", lw=0.85,
                       label="PA")
            if "pb_eur_per_kwh" in frame:
                price.plot(x, frame["pb_eur_per_kwh"], color="#A64B2A", lw=0.85,
                           label="PB")
                price.plot(x, frame["total_buy_price_eur_per_kwh"],
                           color="#3F7D58", lw=0.9, label="PA + PB")
            price.legend(frameon=False, ncol=3)
        else:
            price.text(0.5, 0.5, "No price data", ha="center", va="center",
                       transform=price.transAxes, color="#666666")
        price.set_title("Price signal", loc="left")
        price.set_ylabel("EUR/kWh")
        price.grid(axis="y", alpha=0.28, linewidth=0.55)
        price.margins(y=0.12)
    axes[-1, 0].set_xlabel("Day of July 2026")
    axes[-1, 1].set_xlabel("Day of July 2026")
    fig.suptitle(f"Phase 3: vertical S1-S4 comparison — {style} view", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.982))
    path = folder / "scenario_monthly_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _set_limits(axis, values):
    low, high = min(float(np.min(values)), 0.0), max(float(np.max(values)), 0.0)
    span = max(high - low, 1.0)
    axis.set_ylim(low - 0.12 * span, high + 0.20 * span)


def performance_bars(summary, folder):
    folder.mkdir(parents=True, exist_ok=True)
    panels = (
        ("mean_daily_peak_kw", "Mean daily peak [kW]"),
        ("mean_daily_valley_kw", "Mean daily valley [kW]"),
        ("mean_daily_peak_valley_kw", "Mean daily peak-valley [kW]"),
        ("mean_daily_maximum_ramp_kw", "Mean daily maximum ramp [kW/15 min]"),
        ("total_import_energy_kwh", "Monthly import energy [kWh]"),
        ("total_export_energy_kwh", "Monthly export energy [kWh]"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    colors = [COLORS[s] for s in SCENARIOS]
    for axis, (column, title) in zip(axes.flat, panels):
        values = summary[column].to_numpy(float)
        bars = axis.bar(summary["scenario"], values, color=colors,
                        edgecolor="white", linewidth=0.7)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.28, linewidth=0.55)
        axis.axhline(0.0, color="#777777", lw=0.6)
        _set_limits(axis, values)
        low, high = axis.get_ylim()
        offset = 0.025 * (high - low)
        for bar, value in zip(bars, values):
            y = value + offset if value >= 0 else value - offset
            axis.text(bar.get_x() + bar.get_width() / 2, y, f"{value:.1f}",
                      ha="center", va="bottom" if value >= 0 else "top", fontsize=8.5)
    fig.suptitle("Phase 3 S1-S4 monthly performance comparison", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = folder / "scenario_monthly_performance_bars.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    root = Path(__file__).resolve().parent
    csv_folder = root / "csv_results"
    figure_root = root / "figures" / "comparison"
    frames = load_outputs(root)
    summary = summary_table(frames)
    summary_path = csv_folder / "scenario_monthly_summary.csv"
    summary.to_csv(summary_path, index=False)
    line_path = comparison_figure(frames, figure_root / "line", "line")
    bar_path = comparison_figure(frames, figure_root / "bar", "bar")
    performance_path = performance_bars(summary, figure_root / "performance")
    print(summary.round(1).to_string(index=False))
    print(f"Summary: {summary_path}")
    print(f"Line comparison: {line_path}")
    print(f"Bar comparison: {bar_path}")
    print(f"Performance bars: {performance_path}")


if __name__ == "__main__":
    main()
