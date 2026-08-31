"""Compare C-A (DER-only) and C-B (load-integrated) scenarios S1--S4.

Place this file in ``stackelberg_scenario_analysis`` or change CA_CSV_DIR and
CB_CSV_DIR at the bottom.  It reads existing scenario CSVs only; it does not
rerun Gurobi.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIOS = ("S1", "S2", "S3", "S4")
SYSTEM_COLORS = {"C-A": "#8ECAE6", "C-B": "#F4A6A6"}
SCENARIO_COLORS = {"S1": "#B0B0B0", "S2": "#9ECAE1",
                   "S3": "#F6B48F", "S4": "#A8D5BA"}


def find_column(frame, candidates, source):
    for name in candidates:
        if name in frame.columns:
            return name
    raise KeyError(f"{source.name}: none of these columns exists: {candidates}")


def read_scenario(csv_dir, scenario):
    path = csv_dir / f"{scenario.lower()}_timeseries.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run {scenario} first.")
    frame = pd.read_csv(path)
    time_col = find_column(frame, ("time_h", "hour", "time"), path)
    net_col = find_column(
        frame,
        ("aggregate_net_kw", "net_exchange_kw", "aggregate_net_exchange_kw",
         "net_kw"),
        path,
    )
    time = pd.to_numeric(frame[time_col], errors="coerce").to_numpy(float)
    net = pd.to_numeric(frame[net_col], errors="coerce").to_numpy(float)
    valid = np.isfinite(time) & np.isfinite(net)
    time, net = time[valid], net[valid]
    if len(net) < 2:
        raise ValueError(f"{path}: fewer than two valid time steps")
    order = np.argsort(time)
    time, net = time[order], net[order]
    dt = float(np.median(np.diff(time)))
    if dt <= 0:
        raise ValueError(f"{path}: invalid time spacing")
    return {"path": path, "time": time, "net": net, "dt": dt}


def metrics(record):
    net, dt = record["net"], record["dt"]
    peak, valley = float(np.max(net)), float(np.min(net))
    # These are exchanges with the upstream grid, calculated after aggregating
    # all followers. Internal exchanges between followers cancel here.
    grid_import = float(np.sum(np.maximum(net, 0.0))*dt)
    grid_export = float(np.sum(np.maximum(-net, 0.0))*dt)
    return {
        "peak_kw": peak,
        "valley_kw": valley,
        "peak_valley_kw": peak-valley,
        "maximum_ramp_kw": float(np.max(np.abs(np.diff(net)))),
        "grid_import_kwh": grid_import,
        "grid_export_kwh": grid_export,
        "net_grid_energy_kwh": grid_import-grid_export,
    }


def load_all(ca_dir, cb_dir):
    records, rows = {}, []
    for system, folder in (("C-A", ca_dir), ("C-B", cb_dir)):
        for scenario in SCENARIOS:
            record = read_scenario(folder, scenario)
            records[(system, scenario)] = record
            rows.append({"system": system, "scenario": scenario,
                         **metrics(record), "source_csv": str(record["path"])})
    return records, pd.DataFrame(rows)


def plot_curves(records, output_dir):
    fig, axes = plt.subplots(4, 2, figsize=(15, 14))
    for row, scenario in enumerate(SCENARIOS):
        ca, cb = records[("C-A", scenario)], records[("C-B", scenario)]
        for axis, system, record in ((axes[row, 0], "C-A", ca),
                                     (axes[row, 1], "C-B", cb)):
            axis.plot(record["time"], record["net"],
                      color=SYSTEM_COLORS[system], lw=1.8)
            axis.axhline(0, color="0.45", lw=.7)
            axis.grid(alpha=.22)
            axis.set_title(f"{system} {scenario}")
            axis.set_ylabel("Net exchange [kW]")
            axis.set_xlabel("Time [h]")
    fig.suptitle("C-A DER-only versus C-B load-integrated: S1-S4", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .97))
    path = output_dir/"ca_cb_timeseries_4x2.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_absolute_bars(table, output_dir):
    panels = (
        ("peak_kw", "Peak [kW]"),
        ("valley_kw", "Valley [kW]"),
        ("peak_valley_kw", "Peak-valley range [kW]"),
        ("maximum_ramp_kw", "Maximum ramp [kW/step]"),
        ("grid_import_kwh", "Upstream-grid import [kWh]"),
        ("grid_export_kwh", "Upstream-grid export [kWh]"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    x = np.arange(len(SCENARIOS)); width = .36
    for axis, (column, title) in zip(axes.flat, panels):
        for offset, system in ((-.5, "C-A"), (.5, "C-B")):
            values = (table[table.system == system].set_index("scenario")
                      .loc[list(SCENARIOS), column].to_numpy(float))
            bars = axis.bar(x+offset*width, values, width,
                            label=system, color=SYSTEM_COLORS[system],
                            edgecolor="white")
            for bar, value in zip(bars, values):
                axis.annotate(f"{value:.2f}",
                              (bar.get_x()+bar.get_width()/2, value),
                              xytext=(0, 3 if value >= 0 else -10),
                              textcoords="offset points", ha="center", fontsize=7)
        axis.set_xticks(x, SCENARIOS)
        axis.set_title(title)
        axis.axhline(0, color="0.5", lw=.7)
        axis.grid(axis="y", alpha=.2)
        axis.legend()
    fig.suptitle("Absolute C-A / C-B performance metrics", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .96))
    path = output_dir/"ca_cb_absolute_metrics.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def relative_summary(table):
    rows = []
    for system in ("C-A", "C-B"):
        data = table[table.system == system].set_index("scenario")
        r1, r2, r3, r4 = [float(data.loc[s, "peak_valley_kw"])
                          for s in SCENARIOS]
        rows.append({
            "system": system,
            "s3_improvement_vs_s1_percent":
                100*(r1-r3)/r1 if abs(r1) > 1e-12 else np.nan,
            "s3_improvement_vs_s2_percent":
                100*(r2-r3)/r2 if abs(r2) > 1e-12 else np.nan,
            "s3_minus_s4_range_kw": r3-r4,
            "s3_range_over_s4_percent":
                100*r3/r4 if abs(r4) > 1e-12 else np.nan,
        })
    return pd.DataFrame(rows)


def plot_relative_bars(summary, output_dir):
    columns = (
        ("s3_improvement_vs_s1_percent", "S3 improvement vs S1 [%]"),
        ("s3_improvement_vs_s2_percent", "S3 improvement vs S2 [%]"),
        ("s3_minus_s4_range_kw", "S3 minus S4 range [kW]"),
        ("s3_range_over_s4_percent", "S3 range / S4 range [%]"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, (column, title) in zip(axes.flat, columns):
        values = summary[column].to_numpy(float)
        bars = axis.bar(summary["system"], values,
                        color=[SYSTEM_COLORS[s] for s in summary["system"]],
                        edgecolor="white")
        axis.set_title(title); axis.grid(axis="y", alpha=.2)
        axis.axhline(0, color="0.5", lw=.7)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                axis.annotate(f"{value:.2f}",
                              (bar.get_x()+bar.get_width()/2, value),
                              xytext=(0, 4 if value >= 0 else -11),
                              textcoords="offset points", ha="center", fontsize=8)
    fig.suptitle("Within-system effectiveness of S3", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .95))
    path = output_dir/"ca_cb_relative_effectiveness.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main(settings):
    ca_dir = Path(settings["CA_CSV_DIR"]).resolve()
    cb_dir = Path(settings["CB_CSV_DIR"]).resolve()
    output_dir = Path(settings["OUTPUT_DIR"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records, absolute = load_all(ca_dir, cb_dir)
    relative = relative_summary(absolute)
    absolute.to_csv(output_dir/"ca_cb_absolute_metrics.csv", index=False)
    relative.to_csv(output_dir/"ca_cb_relative_metrics.csv", index=False)
    figures = (plot_curves(records, output_dir),
               plot_absolute_bars(absolute, output_dir),
               plot_relative_bars(relative, output_dir))
    print("\nAbsolute metrics:\n", absolute.to_string(index=False))
    print("\nRelative effectiveness:\n", relative.to_string(index=False))
    print("\nOutputs:")
    for path in figures:
        print(path)


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent
    SETTINGS = {
        # If this script is placed inside stackelberg_scenario_analysis:
        "CA_CSV_DIR": ROOT/"phase1_der_only"/"csv_results",
        "CB_CSV_DIR": ROOT/"phase2"/"csv_results",
        "OUTPUT_DIR": ROOT/"ca_cb_comparison",
    }
    main(SETTINGS)
