"""Draw a publication-style MIP-gap figure from solver_gap_history.csv.

This script does not run Gurobi. It only reads the convergence data already
written by test_phase.py and creates a new PNG figure.
"""

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter


def load_history(csv_file):
    df = pd.read_csv(csv_file)
    required = {"mode", "window_start", "solve_time_s", "gap"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_file}: {sorted(missing)}")

    for col in ("window_start", "solve_time_s", "gap"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["mode", "solve_time_s", "gap"]
    )
    df["mode"] = df["mode"].astype(str).str.lower()
    df["time_plot"] = df["solve_time_s"].clip(lower=TIME_FLOOR_S)
    df["gap_pct"] = (100.0 * df["gap"]).clip(lower=GAP_FLOOR_PCT)
    return df.sort_values(["mode", "window_start", "solve_time_s"])


def style_axis(ax, title=None, show_xlabel=True, show_ylabel=True,
               show_target=True):
    ax.set_xscale("log")
    ax.set_yscale("log")
    if show_target:
        ax.axhline(TARGET_GAP_PCT, color="#d1495b", ls="--", lw=1.2,
                   label=f"Target: {TARGET_GAP_PCT:g}%", zorder=1)
    ax.grid(True, which="major", color="#c8ced6", lw=.65, alpha=.65)
    ax.grid(True, which="minor", color="#dfe3e8", lw=.4, alpha=.45)
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    if title:
        ax.set_title(title, loc="left", fontsize=10.5, weight="semibold", pad=7)
    if show_xlabel:
        ax.set_xlabel("Solver time [s]")
    if show_ylabel:
        ax.set_ylabel("MIP gap [%]")
    ax.spines[["top", "right"]].set_visible(False)


def draw_curve(ax, data, color, label=None, annotate=True):
    if data.empty:
        ax.text(.5, .5, "No feasible-solution history", ha="center", va="center",
                transform=ax.transAxes, color="#6b7280", fontsize=9)
        return
    ax.step(data["time_plot"], data["gap_pct"], where="post", color=color,
            lw=1.9, label=label, zorder=3)
    ax.scatter(data["time_plot"], data["gap_pct"], s=10, color=color,
               edgecolor="white", linewidth=.25, zorder=4)
    if annotate:
        last = data.iloc[-1]
        text = f'{last["gap_pct"]:.3g}% at {last["solve_time_s"]:.1f} s'
        ax.annotate(text, (last["time_plot"], last["gap_pct"]), xytext=(-5, 7),
                    textcoords="offset points", ha="right", fontsize=8,
                    color=color, weight="semibold")


def make_figure(df, output_png):
    static = df[df["mode"] == "static"]
    rhg = df[df["mode"] == "rhg"]
    windows = sorted(rhg["window_start"].dropna().unique())
    ncols = min(RHG_COLUMNS, max(1, len(windows)))
    nrows = max(1, math.ceil(len(windows) / ncols))

    fig = plt.figure(figsize=(FIGURE_WIDTH, 4.0 + 2.65 * nrows), facecolor="white")
    grid = fig.add_gridspec(nrows + 2, ncols, height_ratios=[1.25, .08] + [1] * nrows,
                            hspace=.58, wspace=.33)

    ax_static = fig.add_subplot(grid[0, :])
    style_axis(ax_static, "A  Static optimization", show_xlabel=True,
               show_target=False)
    draw_curve(ax_static, static, STATIC_COLOR, "Static")
    set_static_limits(ax_static, static)
    ax_static.legend(frameon=False, loc="best")

    fig.text(.075, .545, "B  RHG optimization — independent prediction windows",
             fontsize=11.5, weight="bold", color="#20242a")

    rhg_axes = []
    for i, start in enumerate(windows):
        row, col = divmod(i, ncols)
        ax = fig.add_subplot(grid[row + 2, col])
        hours = start * TIME_STEP_HOURS
        title = f"Window {int(start)}  (starts at {hours:g} h)"
        style_axis(ax, title, show_xlabel=row == nrows - 1, show_ylabel=col == 0)
        draw_curve(ax, rhg[rhg["window_start"] == start], RHG_COLOR, annotate=True)
        rhg_axes.append(ax)

    for i in range(len(windows), nrows * ncols):
        row, col = divmod(i, ncols)
        fig.add_subplot(grid[row + 2, col]).set_visible(False)

    # Equal limits make convergence difficulty directly comparable across windows.
    if rhg_axes:
        xvals = rhg["time_plot"].to_numpy()
        yvals = np.r_[rhg["gap_pct"].to_numpy(), TARGET_GAP_PCT]
        xlim = decade_limits(xvals)
        ylim = decade_limits(yvals)
        for ax in rhg_axes:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

    fig.suptitle("MIP convergence: relative optimality gap over solver time",
                 x=.075, y=.985, ha="left", fontsize=15, weight="bold")
    fig.text(.075, .955,
             "Each RHG panel is a separate optimization problem; curves are not cumulative.",
             ha="left", fontsize=9.5, color="#59616b")
    fig.subplots_adjust(left=.075, right=.975, bottom=.075, top=.91)
    fig.savefig(output_png, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def set_static_limits(ax, data):
    """Crop the Static panel to its relevant gap range."""
    if data.empty:
        return
    ymax = float(data["gap_pct"].max())
    upper = 10 ** math.ceil(math.log10(max(ymax, STATIC_Y_MIN_PCT)))
    if upper <= STATIC_Y_MIN_PCT:
        upper = STATIC_Y_MIN_PCT * 10
    ax.set_ylim(STATIC_Y_MIN_PCT, upper)


def decade_limits(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        return (1e-3, 1.0)
    lo = 10 ** math.floor(math.log10(values.min()))
    hi = 10 ** math.ceil(math.log10(values.max()))
    return (lo, hi if hi > lo else lo * 10)


def main():
    csv_file = Path(INPUT_CSV)
    if not csv_file.exists():
        raise FileNotFoundError(
            f"Cannot find {csv_file.resolve()}\n"
            "Set INPUT_CSV below to your existing solver_gap_history.csv."
        )
    df = load_history(csv_file)
    make_figure(df, Path(OUTPUT_PNG))
    print(f"Saved: {Path(OUTPUT_PNG).resolve()}")


# ========================== ADJUSTABLE PARAMETERS ===========================
INPUT_CSV = "1_LV_urban_0_no_sw/test_phase_outputs/solver_gap_history.csv"
OUTPUT_PNG = "1_LV_urban_0_no_sw/test_phase_outputs/05_gap_vs_time_improved.png"

TARGET_GAP_PCT = 0.1       # MIP_GAP=0.001 corresponds to 0.1%
STATIC_Y_MIN_PCT = 10.0    # crop unused space below the Static curve
TIME_STEP_HOURS = 0.25     # 15-minute data; window 12 starts at hour 3
TIME_FLOOR_S = 1e-3        # log axes cannot display zero
GAP_FLOOR_PCT = 1e-6       # display floor only; source CSV is unchanged
RHG_COLUMNS = 4
FIGURE_WIDTH = 14
OUTPUT_DPI = 300

STATIC_COLOR = "#2457A6"
RHG_COLOR = "#E07A3F"


if __name__ == "__main__":
    main()
