from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "comparisons"
EXPECTED_DATES = (pd.Timestamp("2025-01-15"), pd.Timestamp("2025-01-16"))
EXPECTED_STEPS = 2 * 96

SERIES = {
    "Reference": (
        ROOT / "reference_ol_ol" / "results" / "planned_and_realized_timeseries.csv",
        "ol",
    ),
    "T1": (ROOT / "t1_iid_ol_ol" / "results" / "planned_and_realized_timeseries.csv", "ol"),
    "T2": (ROOT / "t2_ar1_ol_ol" / "results" / "planned_and_realized_timeseries.csv", "ol"),
    "T3": (ROOT / "t3_iid_ol_mpc" / "results_online_2days" / "mpc_timeseries.csv", "mpc"),
    "T4": (ROOT / "t4_ar1_ol_mpc" / "results_online_2days" / "mpc_timeseries.csv", "mpc"),
    "T5": (ROOT / "t5_iid_mpc_mpc" / "results_online_2days" / "mpc_timeseries.csv", "mpc"),
    "T6": (ROOT / "t6_ar1_mpc_mpc" / "results_online_2days" / "mpc_timeseries.csv", "mpc"),
}

COLORS = {
    "Reference": "#4D4D4D",
    "T1": "#0072B2",
    "T2": "#56B4E9",
    "T3": "#009E73",
    "T4": "#E69F00",
    "T5": "#CC79A7",
    "T6": "#D55E00",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def load_ol(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "step", "bus", "realized_net_kW"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    selected = frame[frame["date"].isin(EXPECTED_DATES)].copy()
    aggregate = (
        selected.groupby(["date", "step"], as_index=False)["realized_net_kW"]
        .sum()
        .sort_values(["date", "step"])
    )
    aggregate["timestamp"] = aggregate["date"] + pd.to_timedelta(aggregate["step"] * 15, unit="min")
    return aggregate.rename(columns={"realized_net_kW": "grid_exchange_kw"})[
        ["timestamp", "grid_exchange_kw"]
    ]


def load_mpc(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "realized_net_kw"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    selected = frame[frame["timestamp"].dt.normalize().isin(EXPECTED_DATES)].copy()
    return selected.rename(columns={"realized_net_kw": "grid_exchange_kw"})[
        ["timestamp", "grid_exchange_kw"]
    ].sort_values("timestamp")


def load_all() -> dict[str, pd.DataFrame]:
    frames = {}
    expected_timestamps = pd.date_range(EXPECTED_DATES[0], periods=EXPECTED_STEPS, freq="15min")
    for label, (path, kind) in SERIES.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} time series: {path}")
        frame = load_ol(path) if kind == "ol" else load_mpc(path)
        frame = frame.reset_index(drop=True)
        if len(frame) != EXPECTED_STEPS:
            raise ValueError(f"{label}: expected {EXPECTED_STEPS} time steps, got {len(frame)}")
        if not frame["timestamp"].reset_index(drop=True).equals(pd.Series(expected_timestamps)):
            raise ValueError(f"{label}: timestamps do not match the common winter 48 h period")
        frames[label] = frame
    return frames


def main() -> None:
    configure_style()
    frames = load_all()
    all_values = np.concatenate([frame["grid_exchange_kw"].to_numpy(float) for frame in frames.values()])
    lower, upper = float(all_values.min()), float(all_values.max())
    padding = max(0.5, 0.06 * (upper - lower))
    y_limits = (lower - padding, upper + padding)
    hours = np.arange(EXPECTED_STEPS) * 0.25

    fig, axes = plt.subplots(7, 1, figsize=(15.5, 18.0), sharex=True, sharey=True)
    for axis, (label, frame) in zip(axes, frames.items()):
        axis.plot(hours, frame["grid_exchange_kw"], color=COLORS[label], lw=1.15)
        axis.axhline(0.0, color="0.35", lw=0.6, alpha=0.7)
        axis.axvline(24.0, color="0.25", lw=0.8, ls=":", alpha=0.8)
        axis.set_ylim(*y_limits)
        axis.set_ylabel("Grid exchange\n(kW)")
        axis.set_title(label, loc="left", fontweight="bold", pad=4)
        axis.grid(True, color="0.75", ls="--", lw=0.5, alpha=0.3)

    axes[-1].set_xlim(0.0, 48.0)
    axes[-1].set_xticks(np.arange(0.0, 48.1, 3.0))
    axes[-1].set_xlabel("Time (h)")
    fig.suptitle("Winter 48 h aggregate grid exchange: Reference and T1–T6", fontsize=14, y=0.997)
    fig.text(0.507, 0.982, "2025-01-15", ha="center", va="top", fontsize=9)
    fig.text(0.758, 0.982, "2025-01-16", ha="center", va="top", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.045, top=0.965, hspace=0.30)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / "reference_t1_t6_vertical_grid_exchange.png"
    pdf = OUTPUT_DIR / "reference_t1_t6_vertical_grid_exchange.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


if __name__ == "__main__":
    main()
