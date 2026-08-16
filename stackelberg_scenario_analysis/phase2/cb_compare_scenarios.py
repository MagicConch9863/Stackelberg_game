"""Create the S1--S4 time-series comparison and soft-color performance bars."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIOS = ("s1", "s2", "s3", "s4")
SOFT_COLORS = ("#A9A9A9", "#8ECAE6", "#FFB4A2", "#A8D5BA")


def load_frames(csv_folder):
    frames = {}
    for scenario in SCENARIOS:
        path = csv_folder/f"{scenario}_timeseries.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Run {scenario.upper()} first: {path}")
        frames[scenario] = pd.read_csv(path)
    return frames


def performance_metrics(frames):
    rows = []
    for scenario, frame in frames.items():
        net = frame["aggregate_net_kw"].to_numpy(float)
        import_cols = [c for c in frame if c.endswith("_import_kw")]
        export_cols = [c for c in frame if c.endswith("_export_kw")]
        rows.append({
            "scenario": scenario.upper(),
            "peak_kw": np.max(net),
            "valley_kw": np.min(net),
            "peak_valley_kw": np.ptp(net),
            "maximum_ramp_kw": np.max(np.abs(np.diff(net))),
            "import_energy_kwh": frame[import_cols].to_numpy(float).sum()*.25,
            "export_energy_kwh": frame[export_cols].to_numpy(float).sum()*.25,
        })
    return pd.DataFrame(rows)


def main():
    root = Path(__file__).resolve().parent
    csv_folder, figure_folder = root/"csv_results", root/"figures"
    figure_folder.mkdir(exist_ok=True)
    frames = load_frames(csv_folder)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, sharey=True)
    for axis, scenario in zip(axes.flat, SCENARIOS):
        frame = frames[scenario]
        baseline_column = ("aggregate_baseline_net_kw"
                           if "aggregate_baseline_net_kw" in frame
                           else "aggregate_load_kw")
        axis.plot(frame["time_h"], frame[baseline_column], "--",
                  color="0.5", lw=1.8, label="Baseline (load - PV)")
        axis.plot(frame["time_h"], frame["aggregate_net_kw"],
                  color="tab:blue", lw=1.8, label="Net exchange")
        axis.set_title(scenario.upper())
        axis.set_xlabel("Time [h]"); axis.set_ylabel("kW")
        axis.grid(alpha=.25); axis.legend()
    fig.suptitle("C-B: effect of S1-S4 on the same SimBench load and PV", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .96))
    output = figure_folder/"scenario_comparison_2x2.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    metrics = performance_metrics(frames)
    metrics.to_csv(csv_folder/"scenario_performance_metrics.csv", index=False)
    panels = (
        ("peak_kw", "Peak [kW]"),
        ("valley_kw", "Valley [kW]"),
        ("peak_valley_kw", "Peak-valley range [kW]"),
        ("maximum_ramp_kw", "Maximum ramp [kW/15 min]"),
        ("import_energy_kwh", "Import energy [kWh]"),
        ("export_energy_kwh", "Export energy [kWh]"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for axis, (column, title) in zip(axes.flat, panels):
        values = metrics[column].to_numpy(float)
        bars = axis.bar(metrics["scenario"], values, color=SOFT_COLORS,
                        edgecolor="white", linewidth=.8)
        axis.set_title(title)
        axis.grid(axis="y", alpha=.22)
        axis.axhline(0, color="0.5", lw=.7)
        for bar, value in zip(bars, values):
            offset = max(np.ptp(values), np.max(np.abs(values)), 1.0)*.018
            y = value + offset if value >= 0 else value-offset
            axis.text(bar.get_x()+bar.get_width()/2, y, f"{value:.3f}",
                      ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    fig.suptitle("C-B S1-S4 performance comparison", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .96))
    bars_output = figure_folder/"scenario_performance_bars.png"
    fig.savefig(bars_output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison figure: {output}\nPerformance bars: {bars_output}")


if __name__ == "__main__":
    main()
