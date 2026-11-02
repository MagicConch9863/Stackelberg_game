"""Publication-oriented analysis for the unified three-day experiment suite."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENTS = {
    "T1": "t1_iid_ol_ol",
    "T2": "t2_ar1_ol_ol",
    "T3": "t3_iid_ol_mpc",
    "T4": "t4_ar1_ol_mpc",
    "T5": "t5_iid_mpc_mpc",
    "T6": "t6_ar1_mpc_mpc",
    "T12-perfect": "t12_perfect_ol_ol",
    "T34-perfect": "t34_perfect_ol_mpc",
    "T56-perfect": "t56_perfect_mpc_mpc",
}

COLORS = {
    "T1": "#0072B2", "T2": "#56B4E9", "T3": "#009E73", "T4": "#CC79A7",
    "T5": "#E69F00", "T6": "#D55E00", "T12-perfect": "#4C78A8",
    "T34-perfect": "#59A14F", "T56-perfect": "#B07AA1",
}


def _read_timeseries(folder: Path) -> pd.DataFrame:
    for name in ("timeseries.csv", "mpc_timeseries.csv"):
        path = folder / name
        if path.exists():
            frame = pd.read_csv(path, parse_dates=["timestamp"])
            if len(frame) != 288:
                raise RuntimeError(f"Incomplete result: {path} has {len(frame)} rows, expected 288.")
            return frame
    raise FileNotFoundError(f"No completed time series in {folder}")


def _read_solver(folder: Path) -> pd.DataFrame:
    path = folder / "solver_log.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _metric_and_extrema(label: str, frame: pd.DataFrame, solver: pd.DataFrame):
    net = frame["realized_net_kw"].to_numpy(float)
    ramp = frame["realized_net_kw"].diff()
    peak_index = int(np.argmax(net)); valley_index = int(np.argmin(net))
    ramp_index = int(ramp.abs().iloc[1:].idxmax())
    throughput = (
        float(frame["bess_throughput_step_kwh"].sum())
        if "bess_throughput_step_kwh" in frame else float(frame["bess_kw"].abs().sum() * 0.25)
    )
    gap_col = "gap" if "gap" in solver else "mip_gap" if "mip_gap" in solver else None
    runtime_col = "runtime_s" if "runtime_s" in solver else None
    metric = {
        "experiment": label,
        "peak_kw": net[peak_index], "valley_kw": net[valley_index],
        "peak_to_valley_kw": float(np.ptp(net)),
        "max_abs_ramp_kw_per_15min": float(abs(ramp.iloc[ramp_index])),
        "mean_abs_ramp_kw_per_15min": float(ramp.abs().iloc[1:].mean()),
        "bess_throughput_kwh": throughput,
        "degradation_cost_eur": float(frame.get("bess_degradation_cost_eur", pd.Series(0.0, index=frame.index)).sum()),
        "soc_penalty_eur": float(frame.get("soc_penalty_eur", pd.Series(0.0, index=frame.index)).sum()),
        "solver_runtime_sum_s": float(solver[runtime_col].sum()) if runtime_col else np.nan,
        "solver_gap_mean": float(solver[gap_col].mean()) if gap_col else np.nan,
        "solver_gap_max": float(solver[gap_col].max()) if gap_col else np.nan,
        "timeout_count": int((solver.get("status", pd.Series(dtype=float)) == 9).sum()),
    }
    charge_index = int(frame["bess_kw"].idxmax())
    discharge_index = int(frame["bess_kw"].idxmin())
    soc_cols = [column for column in frame if column.startswith("soc_") and column.endswith("_kwh")]
    soc_values = frame[soc_cols].to_numpy(float) if soc_cols else np.empty((0, 0))
    if soc_values.size:
        soc_min_flat = int(np.argmin(soc_values)); soc_max_flat = int(np.argmax(soc_values))
        soc_min_row, soc_min_bus = np.unravel_index(soc_min_flat, soc_values.shape)
        soc_max_row, soc_max_bus = np.unravel_index(soc_max_flat, soc_values.shape)
    else:
        soc_min_row = soc_max_row = 0; soc_min_bus = soc_max_bus = 0
    extrema = {
        "experiment": label,
        "peak_kw": net[peak_index], "peak_time": frame.iloc[peak_index].timestamp,
        "valley_kw": net[valley_index], "valley_time": frame.iloc[valley_index].timestamp,
        "max_abs_ramp_kw_per_15min": float(abs(ramp.iloc[ramp_index])),
        "ramp_signed_kw_per_15min": float(ramp.iloc[ramp_index]),
        "ramp_start_time": frame.iloc[ramp_index - 1].timestamp,
        "ramp_end_time": frame.iloc[ramp_index].timestamp,
        "max_charge_kw": float(frame.iloc[charge_index].bess_kw),
        "max_charge_time": frame.iloc[charge_index].timestamp,
        "max_discharge_kw": float(frame.iloc[discharge_index].bess_kw),
        "max_discharge_time": frame.iloc[discharge_index].timestamp,
        "soc_min_kwh": float(soc_values[soc_min_row, soc_min_bus]) if soc_values.size else np.nan,
        "soc_min_time": frame.iloc[soc_min_row].timestamp if soc_values.size else pd.NaT,
        "soc_min_bus": soc_cols[soc_min_bus] if soc_values.size else "",
        "soc_max_kwh": float(soc_values[soc_max_row, soc_max_bus]) if soc_values.size else np.nan,
        "soc_max_time": frame.iloc[soc_max_row].timestamp if soc_values.size else pd.NaT,
        "soc_max_bus": soc_cols[soc_max_bus] if soc_values.size else "",
    }
    return metric, extrema


def _style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    })


def _time_boundaries(ax):
    for boundary in (24, 48):
        ax.axvline(boundary, color="0.55", ls="--", lw=0.8, alpha=0.7)
    ax.grid(True, ls="--", lw=0.5, alpha=0.3)
    ax.set_xlim(0, 72)


def _plot_t56(frames: dict[str, pd.DataFrame], output: Path):
    fig, axes = plt.subplots(3, 1, figsize=(12.0, 7.0), sharex=True, sharey=True)
    for ax, label in zip(axes, ("T56-perfect", "T5", "T6")):
        frame = frames[label]; x = np.arange(len(frame)) * 0.25
        ax.plot(x, frame["uncontrolled_net_kw"], color="0.55", ls="--", lw=1.2, label="Uncontrolled")
        ax.plot(x, frame["realized_net_kw"], color=COLORS[label], lw=1.45, label=label)
        ax.set_title(label); ax.set_ylabel("Grid exchange (kW)"); ax.legend(frameon=False, ncol=2)
        _time_boundaries(ax)
    axes[-1].set_xlabel("Time (h)")
    fig.tight_layout(); fig.savefig(output / "T56_comparison.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def _boxplot_t5_t6(frames: dict[str, pd.DataFrame], solvers: dict[str, pd.DataFrame], output: Path):
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.2))
    labels = ["T5", "T6"]
    series = [
        ([frames[x].realized_net_kw for x in labels], "Grid exchange (kW)"),
        ([frames[x].realized_net_kw.diff().dropna().abs() for x in labels], "|15-min ramp| (kW)"),
        ([frames[x].bess_kw for x in labels], "BESS power (kW)"),
        ([frames[x].pb_eur_per_kwh for x in labels], r"$P^B$ (EUR/kWh)"),
        ([solvers[x].runtime_s for x in labels], "Solver runtime (s)"),
        ([100 * solvers[x].gap for x in labels], "MIP gap (%)"),
    ]
    for ax, (values, ylabel) in zip(axes.flat, series):
        box = ax.boxplot(values, labels=labels, patch_artist=True, showfliers=True,
                         medianprops={"color": "black", "linewidth": 1.2})
        for patch, label in zip(box["boxes"], labels):
            patch.set_facecolor(COLORS[label]); patch.set_alpha(0.65)
        ax.set_ylabel(ylabel); ax.grid(True, axis="y", ls="--", alpha=0.3)
    fig.tight_layout(); fig.savefig(output / "T5_T6_statistical_comparison.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def _plot_all(frames: dict[str, pd.DataFrame], output: Path):
    groups = [("IID", ("T1", "T3", "T5")), ("AR(1)", ("T2", "T4", "T6")),
              ("Perfect", ("T12-perfect", "T34-perfect", "T56-perfect"))]
    fig, axes = plt.subplots(3, 1, figsize=(12.0, 8.0), sharex=True)
    for ax, (title, labels) in zip(axes, groups):
        base = frames[labels[0]]; x = np.arange(len(base)) * 0.25
        ax.plot(x, base.uncontrolled_net_kw, color="0.65", ls="--", lw=1.0, label="Uncontrolled")
        for label in labels:
            ax.plot(x, frames[label].realized_net_kw, color=COLORS[label], lw=1.25, label=label)
        ax.set_title(title); ax.set_ylabel("Grid exchange (kW)"); ax.legend(frameon=False, ncol=4)
        _time_boundaries(ax)
    axes[-1].set_xlabel("Time (h)")
    fig.tight_layout(); fig.savefig(output / "all_architectures_timeseries.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def _plot_metrics(metrics: pd.DataFrame, output: Path):
    fields = [("peak_to_valley_kw", "Peak-to-valley (kW)"),
              ("max_abs_ramp_kw_per_15min", "Maximum |ramp| (kW/15 min)"),
              ("bess_throughput_kwh", "BESS throughput (kWh)"),
              ("degradation_cost_eur", "Degradation cost (EUR)"),
              ("soc_penalty_eur", "SOC soft-band penalty (EUR)"),
              ("solver_runtime_sum_s", "Accumulated solver time (s)")]
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.5))
    labels = metrics.experiment.tolist(); x = np.arange(len(labels))
    colors = [COLORS[label] for label in labels]
    for ax, (field, ylabel) in zip(axes.flat, fields):
        ax.bar(x, metrics[field], color=colors, alpha=0.8, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x, labels, rotation=35, ha="right"); ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", ls="--", alpha=0.3)
    fig.tight_layout(); fig.savefig(output / "all_architectures_statistics.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def _plot_extrema_table(extrema: pd.DataFrame, output: Path):
    shown = extrema[["experiment", "peak_kw", "peak_time", "valley_kw", "valley_time",
                     "max_abs_ramp_kw_per_15min", "ramp_end_time"]].copy()
    for field in ("peak_kw", "valley_kw", "max_abs_ramp_kw_per_15min"):
        shown[field] = shown[field].map(lambda value: f"{value:.3f}")
    for field in ("peak_time", "valley_time", "ramp_end_time"):
        shown[field] = pd.to_datetime(shown[field]).dt.strftime("%d.%m %H:%M")
    shown.columns = ["Experiment", "Peak\n(kW)", "Peak time", "Valley\n(kW)",
                     "Valley time", "Max |ramp|\n(kW/15 min)", "Ramp end"]
    fig, ax = plt.subplots(figsize=(11.8, 3.5)); ax.axis("off")
    table = ax.table(cellText=shown.values, colLabels=shown.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(8.5); table.scale(1.0, 1.55)
    for (row, _), cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#E6E6E6"); cell.set_text_props(weight="bold")
    fig.tight_layout(); fig.savefig(output / "extrema_timing_table.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def generate_analysis(root: Path) -> None:
    root = Path(root); output = root / "analysis"; output.mkdir(parents=True, exist_ok=True)
    _style()
    frames = {label: _read_timeseries(root / folder) for label, folder in EXPERIMENTS.items()}
    solvers = {label: _read_solver(root / folder) for label, folder in EXPERIMENTS.items()}
    metrics, extrema = [], []
    for label in EXPERIMENTS:
        metric, extreme = _metric_and_extrema(label, frames[label], solvers[label])
        metrics.append(metric); extrema.append(extreme)
    metric_frame = pd.DataFrame(metrics); extrema_frame = pd.DataFrame(extrema)
    metric_frame.to_csv(output / "overall_performance_metrics.csv", index=False)
    extrema_frame.to_csv(output / "extrema_timing.csv", index=False)
    _plot_t56(frames, output)
    _boxplot_t5_t6(frames, solvers, output)
    _plot_all(frames, output)
    _plot_metrics(metric_frame, output)
    _plot_extrema_table(extrema_frame, output)
    print(f"[analysis] Complete -> {output}", flush=True)
