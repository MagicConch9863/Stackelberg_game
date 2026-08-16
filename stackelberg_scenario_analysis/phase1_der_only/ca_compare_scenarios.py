"""Compare C-A scenarios S1--S4 using their aggregate_net_kw columns."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCENARIOS = ("S1", "S2", "S3", "S4")
COLORS = {"S1": "0.35", "S2": "tab:blue", "S3": "tab:orange", "S4": "tab:green"}


def folders():
    root = Path(__file__).resolve().parent
    csv_folder, figure_folder = root / "csv_results", root / "figures"
    csv_folder.mkdir(exist_ok=True); figure_folder.mkdir(exist_ok=True)
    return csv_folder, figure_folder


def load_results(csv_folder: Path):
    results = {}
    required = {"time_h", "aggregate_net_kw"}
    for scenario in SCENARIOS:
        path = csv_folder / f"{scenario.lower()}_timeseries.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path.name}; run {scenario} first")
        frame = pd.read_csv(path)
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"{path.name} is missing columns: {sorted(missing)}")
        if len(frame) != 96:
            raise ValueError(f"{path.name} has {len(frame)} rows; expected 96")
        results[scenario] = frame
    return results


def metrics_table(results, dt):
    rows = []
    for scenario, frame in results.items():
        net = frame["aggregate_net_kw"].to_numpy(float)
        rows.append({
            "scenario": scenario,
            "peak_kw": net.max(),
            "valley_kw": net.min(),
            "range_kw": np.ptp(net),
            "maximum_ramp_kw": np.max(np.abs(np.diff(net))),
            "import_energy_kwh": np.maximum(net, 0).sum()*dt,
            "export_energy_kwh": np.maximum(-net, 0).sum()*dt,
        })
    table = pd.DataFrame(rows).set_index("scenario")
    s1, s4 = table.loc["S1", "range_kw"], table.loc["S4", "range_kw"]
    table["improvement_vs_s1_percent"] = np.nan if abs(s1) < 1e-10 else (s1-table["range_kw"])/s1*100
    potential = s1-s4
    table["captured_potential_percent"] = np.nan if abs(potential) < 1e-10 else (s1-table["range_kw"])/potential*100
    return table


def plot_separate(results, path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    for ax, scenario in zip(axes.flat, SCENARIOS):
        frame = results[scenario]
        ax.plot(frame["time_h"], frame["aggregate_net_kw"],
                color=COLORS[scenario], linewidth=1.8)
        ax.axhline(0, color="black", linewidth=.7)
        ax.set_title(f"{scenario}: aggregate net exchange")
        ax.set_ylabel("kW"); ax.set_xlabel("Time [h]"); ax.grid(alpha=.3)
    fig.suptitle("C-A scenario comparison – separate panels", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_overlay(results, path):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for scenario in SCENARIOS:
        frame = results[scenario]
        ax.plot(frame["time_h"], frame["aggregate_net_kw"], label=scenario,
                color=COLORS[scenario], linewidth=1.8)
    ax.axhline(0, color="black", linewidth=.7)
    ax.set(title="C-A aggregate net-exchange comparison", xlabel="Time [h]", ylabel="kW")
    ax.grid(alpha=.3); ax.legend(ncol=4)
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_metrics(table, path):
    columns = (("range_kw", "Peak-valley range [kW]"),
               ("maximum_ramp_kw", "Maximum ramp [kW]"),
               ("import_energy_kwh", "Import energy [kWh]"),
               ("captured_potential_percent", "Captured S4 potential [%]"))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    colors = [COLORS[s] for s in SCENARIOS]
    for ax, (column, title) in zip(axes.flat, columns):
        values = table.loc[list(SCENARIOS), column].to_numpy(float)
        bars = ax.bar(SCENARIOS, values, color=colors)
        ax.set_title(title); ax.grid(axis="y", alpha=.3)
        labels = ["n/a" if not np.isfinite(v) else f"{v:.2f}" for v in values]
        ax.bar_label(bars, labels=labels, padding=3)
    fig.suptitle("C-A scenario performance", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def main():
    csv_folder, figure_folder = folders()
    results = load_results(csv_folder)
    table = metrics_table(results, dt=.25)
    table.to_csv(csv_folder / "scenario_metrics.csv")
    plot_separate(results, figure_folder / "scenario_comparison_2x2.png")
    plot_overlay(results, figure_folder / "scenario_comparison_overlay.png")
    plot_metrics(table, figure_folder / "scenario_metrics.png")
    print(table.round(4))
    print(f"Metrics: {csv_folder / 'scenario_metrics.csv'}")
    print(f"Figures: {figure_folder}")


if __name__ == "__main__":
    main()
