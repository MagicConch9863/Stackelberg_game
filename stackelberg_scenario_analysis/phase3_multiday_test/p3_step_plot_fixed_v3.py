"""Path-safe Phase-3 step comparison; reads existing S3/S4 CSV outputs."""
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
          "s3": "#EFA58F", "s4": "#91C6A8"}


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


def main():
    folder = find_phase3_folder()
    frames = {s: read_scenario(folder, s) for s in SCENARIOS}
    frames, timestamps = select_window(frames)
    baseline = frames[SCENARIOS[0]]["aggregate_baseline_net_kw"].to_numpy()

    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.step(timestamps, baseline, where="post", color="#777777", lw=1.15,
            label="Baseline (load - PV)")
    base_peak, base_range = float(baseline.max()), float(np.ptp(baseline))
    metrics = []
    for scenario in SCENARIOS:
        net = frames[scenario]["aggregate_net_kw"].to_numpy()
        ax.step(timestamps, net, where="post", color=COLORS[scenario], lw=1.25,
                label=LABELS[scenario])
        peak, spread = float(net.max()), float(np.ptp(net))
        peak_change = 100 * (base_peak - peak) / max(abs(base_peak), 1e-9)
        range_change = 100 * (base_range - spread) / max(base_range, 1e-9)
        metrics.append(f"{scenario.upper()}: peak {peak:.2f} kW "
                       f"(-{peak_change:.1f}%), range {spread:.2f} kW "
                       f"(-{range_change:.1f}%)")

    ax.text(0.012, 0.975, "\n".join(metrics), transform=ax.transAxes,
            va="top", fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                  "edgecolor": "#D0D0D0", "alpha": 0.88})
    ax.axhline(0, color="#999999", lw=0.6)
    ax.set(title="Phase 3: 15-minute step comparison",
           xlabel="Time", ylabel="Aggregate net exchange [kW]")
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b\n%H:%M"))
    ax.grid(alpha=0.20)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()

    output = folder / "figures" / "scenario_step_comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"FOUND PHASE-3 FOLDER: {folder}")
    print(f"FIGURE: {output}")
    print("\n".join(metrics))


if __name__ == "__main__":
    main()
