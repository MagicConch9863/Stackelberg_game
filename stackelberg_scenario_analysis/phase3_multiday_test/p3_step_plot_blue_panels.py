"""Blue Phase-3 step overview and 2x2 pairwise comparison figures."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIOS = ("s3", "s4")
DAYS_TO_PLOT = 2
START_DATE = None                 # e.g. "2026-07-02"
LABELS = {"s1": "S1 predefined", "s2": "S2 fixed tariff",
          "s3": "S3 Stackelberg", "s4": "S4 centralized benchmark"}
COLORS = {"s1": "#9E9E9E", "s2": "#86BBD8",
          "s3": "#3B82C4", "s4": "#63A87B"}
BASELINE_COLOR = "#777777"


def find_phase3_folder():
    """Support placement in the project root or in phase3_multiday_test."""
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    candidates = [
        script_dir / "phase3_multiday_test",
        script_dir,
        cwd / "stackelberg_scenario_analysis" / "phase3_multiday_test",
        cwd / "phase3_multiday_test",
    ]
    checked = []
    for folder in dict.fromkeys(path.resolve() for path in candidates):
        csv_folder = folder / "csv_results"
        checked.append(csv_folder)
        if all((csv_folder / f"{s}_timeseries_month.csv").is_file()
               for s in SCENARIOS):
            return folder
    checked_text = "\n".join(f"  {path}" for path in checked)
    raise FileNotFoundError(
        "Could not locate S3/S4 monthly CSV files. Checked:\n" + checked_text)


def read_scenario(folder, scenario):
    path = folder / "csv_results" / f"{scenario}_timeseries_month.csv"
    frame = pd.read_csv(path)
    required = {"timestamp", "aggregate_baseline_net_kw", "aggregate_net_kw"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"{path.name} misses columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].isna().any():
        raise ValueError(f"{path.name} contains invalid timestamps")
    return frame.sort_values("timestamp").drop_duplicates("timestamp")


def select_window(frames):
    first = frames[SCENARIOS[0]]
    start = (first["timestamp"].min() if START_DATE is None
             else pd.Timestamp(START_DATE))
    stop = start + pd.Timedelta(days=DAYS_TO_PLOT)
    expected = pd.date_range(start, stop, freq="15min", inclusive="left")
    selected = {}
    for scenario, frame in frames.items():
        window = (frame.set_index("timestamp")
                  .reindex(expected)[["aggregate_baseline_net_kw",
                                      "aggregate_net_kw"]])
        if window.isna().any().any():
            raise ValueError(
                f"{scenario.upper()} does not contain all {len(expected)} "
                f"quarter-hours from {start} to {stop}.")
        selected[scenario] = window
    baseline = selected[SCENARIOS[0]]["aggregate_baseline_net_kw"].to_numpy()
    for scenario in SCENARIOS[1:]:
        other = selected[scenario]["aggregate_baseline_net_kw"].to_numpy()
        if not np.allclose(baseline, other, atol=1e-6):
            raise ValueError(f"The {scenario.upper()} baseline is different.")
    return selected, expected


def format_time_axis(axis, timestamps):
    start = pd.Timestamp(timestamps[0]).floor("D")
    stop = pd.Timestamp(timestamps[-1]) + pd.Timedelta(minutes=15)
    ticks = pd.date_range(start, stop, freq="6h")
    axis.set_xticks(ticks.to_pydatetime())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b\n%H:%M"))
    axis.set_xlim(timestamps[0], stop)
    axis.axhline(0, color="#999999", lw=0.6)
    axis.grid(alpha=0.20)
    axis.set_xlabel("Time")
    axis.set_ylabel("Aggregate net exchange [kW]")


def step_line(axis, timestamps, values, label, color, width=1.15):
    axis.step(timestamps, values, where="post", color=color, lw=width,
              label=label)


def reduction_text(baseline, net):
    base_peak = float(np.max(baseline))
    base_range = float(np.ptp(baseline))
    peak = float(np.max(net))
    spread = float(np.ptp(net))
    peak_reduction = 100 * (base_peak - peak) / max(abs(base_peak), 1e-9)
    range_reduction = 100 * (base_range - spread) / max(base_range, 1e-9)
    return peak, spread, peak_reduction, range_reduction


def main():
    folder = find_phase3_folder()
    frames = {s: read_scenario(folder, s) for s in SCENARIOS}
    frames, timestamps = select_window(frames)
    baseline = frames[SCENARIOS[0]]["aggregate_baseline_net_kw"].to_numpy()

    nets = {scenario: frames[scenario]["aggregate_net_kw"].to_numpy()
            for scenario in SCENARIOS}
    metrics = []
    for scenario in SCENARIOS:
        peak, spread, peak_change, range_change = reduction_text(
            baseline, nets[scenario])
        metrics.append(f"{scenario.upper()}: peak {peak:.2f} kW "
                       f"(reduction {peak_change:.1f}%), range {spread:.2f} kW "
                       f"(reduction {range_change:.1f}%)")

    # Figure 1: compact overview, matching the reference-paper style.
    fig, ax = plt.subplots(figsize=(13, 5.4))
    step_line(ax, timestamps, baseline, "Baseline (load - PV)", BASELINE_COLOR)
    for scenario in SCENARIOS:
        step_line(ax, timestamps, nets[scenario], LABELS[scenario],
                  COLORS[scenario], 1.25)

    ax.text(0.012, 0.975, "\n".join(metrics), transform=ax.transAxes,
            va="top", fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                  "edgecolor": "#D0D0D0", "alpha": 0.88})
    ax.set_title("Phase 3: 15-minute step comparison")
    format_time_axis(ax, timestamps)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()

    output = folder / "figures" / "scenario_step_comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: pairwise comparisons plus one complete overview.
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True, sharey=True)

    step_line(axes[0, 0], timestamps, baseline,
              "Baseline (load - PV)", BASELINE_COLOR)
    step_line(axes[0, 0], timestamps, nets["s3"], LABELS["s3"], COLORS["s3"], 1.25)
    s3_metric = reduction_text(baseline, nets["s3"])
    axes[0, 0].set_title(
        f"Baseline vs S3 | peak reduction {s3_metric[2]:.1f}%, "
        f"range reduction {s3_metric[3]:.1f}%")

    step_line(axes[0, 1], timestamps, baseline,
              "Baseline (load - PV)", BASELINE_COLOR)
    step_line(axes[0, 1], timestamps, nets["s4"], LABELS["s4"], COLORS["s4"], 1.25)
    s4_metric = reduction_text(baseline, nets["s4"])
    axes[0, 1].set_title(
        f"Baseline vs S4 | peak reduction {s4_metric[2]:.1f}%, "
        f"range reduction {s4_metric[3]:.1f}%")

    step_line(axes[1, 0], timestamps, nets["s3"], LABELS["s3"], COLORS["s3"], 1.25)
    step_line(axes[1, 0], timestamps, nets["s4"], LABELS["s4"], COLORS["s4"], 1.25)
    axes[1, 0].set_title("S3 Stackelberg vs S4 centralized benchmark")

    step_line(axes[1, 1], timestamps, baseline,
              "Baseline (load - PV)", BASELINE_COLOR)
    for scenario in SCENARIOS:
        step_line(axes[1, 1], timestamps, nets[scenario], LABELS[scenario],
                  COLORS[scenario], 1.25)
    axes[1, 1].set_title("Complete comparison")

    for axis in axes.flat:
        format_time_axis(axis, timestamps)
        axis.legend(loc="upper right", fontsize=8)
    fig.suptitle("Phase 3: pairwise 15-minute step comparisons", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    panel_output = folder / "figures" / "scenario_step_comparison_2x2.png"
    fig.savefig(panel_output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"FOUND PHASE-3 FOLDER: {folder}")
    print(f"FIGURE: {output}")
    print(f"PANEL FIGURE: {panel_output}")
    print("\n".join(metrics))


if __name__ == "__main__":
    main()
