from __future__ import annotations

import os
import tempfile

# Do NOT create phase4/mpl_cache.
# On remote desktops Matplotlib can still use the existing Windows temp folder.
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from p4_common import PAPER_METRIC_LABELS, figure_folder


SCENARIO_ORDER = ("S1", "S2", "S3", "S4")

FIGURE_METRICS = (
    "export_energy_kwh",
    "import_energy_kwh",
    "maximum_ramp_kw_per_15min",
    "net_electricity_cost_eur",
    "peak_kw",
    "valley_kw",
)


plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 13,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


def safe_filename(text: str) -> str:
    return "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in str(text)
    ).strip("_")


def _metric_data(frame: pd.DataFrame, metric: str):
    labels, values = [], []

    # S1 and S4 have no price/cost by model definition.
    allowed = ("S2", "S3") if metric == "net_electricity_cost_eur" else SCENARIO_ORDER

    for scenario in allowed:
        x = pd.to_numeric(
            frame.loc[frame["scenario"] == scenario, metric],
            errors="coerce",
        ).dropna().to_numpy(float)

        if len(x) == 0:
            continue

        labels.append(scenario)
        values.append(x)

    return labels, values


def _boxplot_compatible(ax, values, labels, **kwargs):
    # New Matplotlib uses tick_labels; older versions use labels.
    try:
        return ax.boxplot(values, tick_labels=labels, **kwargs)
    except TypeError:
        return ax.boxplot(values, labels=labels, **kwargs)


def _draw_boxplot(ax, frame: pd.DataFrame, metric: str):
    labels, values = _metric_data(frame, metric)

    if not values:
        ax.set_visible(False)
        return

    box = _boxplot_compatible(
        ax,
        values,
        labels,
        widths=0.58,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "^",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 5,
        },
        medianprops={"color": "black", "linewidth": 1.0},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 4,
        },
    )

    gray_levels = ("0.87", "0.76", "0.64", "0.52")
    for patch, gray in zip(box["boxes"], gray_levels):
        patch.set_facecolor(gray)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)

    ax.set_title(PAPER_METRIC_LABELS[metric])
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3)


def plot_experiment_distribution(frame: pd.DataFrame, experiment_name: str):
    subset = frame.loc[frame["experiment"] == experiment_name].copy()
    if subset.empty:
        return

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.8))

    for ax, metric in zip(axes.flat, FIGURE_METRICS):
        _draw_boxplot(ax, subset, metric)

    occurrences = subset[["replication", "date"]].drop_duplicates().shape[0]
    months = sorted(subset["month"].dropna().astype(int).unique().tolist())

    fig.suptitle(
        f"Performance distributions — {experiment_name}\n"
        f"Months: {months} | sampled day occurrences: {occurrences}",
        y=0.995,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.955])

    folder = figure_folder()
    stem = safe_filename(f"distribution_{experiment_name}")

    fig.savefig(folder / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_all_experiments(frame: pd.DataFrame, settings: dict):
    if not settings.get("CREATE_DISTRIBUTION_FIGURES", True):
        return

    for experiment_name in frame["experiment"].dropna().unique():
        plot_experiment_distribution(frame, str(experiment_name))
