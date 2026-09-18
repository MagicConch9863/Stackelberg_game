from __future__ import annotations

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis" / "results_soc_reference"

EXPERIMENTS = {
    "3 days": {
        "T5-IID": ROOT / "t5_iid_mpc_mpc" / "results_3days_soc_reference" / "daily_metrics.csv",
        "T6-AR(1)": ROOT / "t6_ar1_mpc_mpc" / "results_3days_soc_reference" / "daily_metrics.csv",
    },
    "10 days": {
        "T5-IID": ROOT / "t5_iid_mpc_mpc" / "results_10days_soc_reference" / "daily_metrics.csv",
        "T6-AR(1)": ROOT / "t6_ar1_mpc_mpc" / "results_10days_soc_reference" / "daily_metrics.csv",
    },
}

METHODS = ("T5-IID", "T6-AR(1)")
COLORS = {"T5-IID": "#0072B2", "T6-AR(1)": "#E69F00"}
METRICS = (
    ("peak_to_valley_kw", "Peak-to-valley range (kW)"),
    ("peak_kw", "Peak (kW)"),
    ("valley_kw", "Valley (kW)"),
    ("maximum_ramp_kw_per_15min", "Maximum ramp (kW/15 min)"),
    ("net_electricity_cost_eur", "Net electricity cost (EUR)"),
    ("bess_throughput_kwh", "BESS throughput (kWh)"),
)


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 13,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def _load_paired(duration: str, paths: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for method, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing daily metrics: {path}")
        frame = pd.read_csv(path)
        required = {"period", "uncontrolled_peak_to_valley_kw"} | {
            metric for metric, _ in METRICS
        }
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"{path} is missing columns: {sorted(missing)}")
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["period"], errors="raise")
        frame["method"] = method
        frame["duration"] = duration
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    counts = combined.groupby(["date", "method"]).size().unstack(fill_value=0)
    if tuple(counts.columns) != METHODS or not (counts == 1).all().all():
        raise RuntimeError(f"Dates are not one-to-one paired for {duration}:\n{counts}")
    expected = int(duration.split()[0])
    if len(counts) != expected:
        raise RuntimeError(f"{duration}: expected {expected} paired dates, found {len(counts)}")

    uncontrolled = combined.pivot(
        index="date", columns="method", values="uncontrolled_peak_to_valley_kw"
    )
    if not np.allclose(uncontrolled[METHODS[0]], uncontrolled[METHODS[1]], atol=1e-12):
        raise RuntimeError(f"T5/T6 uncontrolled values differ for {duration}")
    return combined.sort_values(["date", "method"]).reset_index(drop=True)


def _draw_metric(ax: plt.Axes, frame: pd.DataFrame, metric: str, title: str) -> None:
    positions = np.arange(1, len(METHODS) + 1)
    values = [
        frame.loc[frame["method"] == method].sort_values("date")[metric].to_numpy(float)
        for method in METHODS
    ]
    box = ax.boxplot(
        values,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showmeans=True,
        showfliers=False,
        meanprops={
            "marker": "^", "markerfacecolor": "black", "markeredgecolor": "black",
            "markersize": 5, "zorder": 5,
        },
        medianprops={"color": "black", "linewidth": 1.1},
        whiskerprops={"color": "0.25", "linewidth": 0.8},
        capprops={"color": "0.25", "linewidth": 0.8},
    )
    for patch, method in zip(box["boxes"], METHODS):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.32)
        patch.set_edgecolor(COLORS[method])
        patch.set_linewidth(1.0)

    for position, method, method_values in zip(positions, METHODS, values):
        ax.scatter(
            np.full(len(method_values), position), method_values,
            s=22, facecolor=COLORS[method], edgecolor="white", linewidth=0.45,
            alpha=0.9, zorder=4,
        )

    if metric == "peak_to_valley_kw":
        reference = (
            frame.loc[frame["method"] == METHODS[0], "uncontrolled_peak_to_valley_kw"]
            .to_numpy(float)
        )
        ax.axhline(
            reference.mean(), color="0.30", ls="--", lw=1.0,
            label="Uncontrolled daily mean",
        )
        if np.ptp(reference) > 1e-12:
            ax.axhspan(reference.min(), reference.max(), color="0.6", alpha=0.10)
        ax.legend(frameon=False, fontsize=8, loc="best")

    ax.set_title(title)
    ax.set_xticks(positions, METHODS)
    ax.set_xlim(0.55, 2.45)
    ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3)


def _descriptive_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (duration, method), group in frame.groupby(["duration", "method"], sort=False):
        for metric, _ in METRICS:
            values = pd.to_numeric(group[metric], errors="raise").to_numpy(float)
            rows.append({
                "duration": duration,
                "method": method,
                "metric": metric,
                "n_days": len(values),
                "mean": values.mean(),
                "std": values.std(ddof=1) if len(values) > 1 else np.nan,
                "median": np.median(values),
                "q1": np.quantile(values, 0.25),
                "q3": np.quantile(values, 0.75),
                "minimum": values.min(),
                "maximum": values.max(),
            })
    return pd.DataFrame(rows)


def _paired_differences(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for duration, group in frame.groupby("duration", sort=False):
        for metric, _ in METRICS:
            paired = group.pivot(index="date", columns="method", values=metric)
            for date, values in paired.iterrows():
                rows.append({
                    "duration": duration,
                    "date": date,
                    "metric": metric,
                    "t5_iid": values[METHODS[0]],
                    "t6_ar1": values[METHODS[1]],
                    "t6_minus_t5": values[METHODS[1]] - values[METHODS[0]],
                })
    return pd.DataFrame(rows)


def _plot(frame: pd.DataFrame, duration: str) -> Path:
    n_days = frame["date"].nunique()
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.8))
    for ax, (metric, title) in zip(axes.flat, METRICS):
        _draw_metric(ax, frame, metric, title)
    fig.suptitle(
        f"T5-IID vs T6-AR(1): daily performance distributions\n"
        f"Continuous {duration} experiment | daily observations, n = {n_days}",
        y=0.975,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    destination = OUTPUT / f"distribution_t5_t6_{duration.replace(' ', '')}.png"
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return destination


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_frames = []
    for duration, paths in EXPERIMENTS.items():
        frame = _load_paired(duration, paths)
        all_frames.append(frame)
        destination = _plot(frame, duration)
        print(f"Saved {destination}")
    combined = pd.concat(all_frames, ignore_index=True)
    _descriptive_statistics(combined).to_csv(
        OUTPUT / "distribution_descriptive_statistics.csv", index=False
    )
    _paired_differences(combined).to_csv(
        OUTPUT / "paired_daily_differences.csv", index=False
    )
    print(f"Saved statistics to {OUTPUT}")


if __name__ == "__main__":
    main()
