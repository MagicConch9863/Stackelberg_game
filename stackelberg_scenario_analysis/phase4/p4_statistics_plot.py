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
SCENARIO_COLORS = {
    "S1": "#B8BEC5",
    "S2": "#86B6D8",
    "S3": "#D9A66F",
    "S4": "#78A98B",
}

FIGURE_METRICS = (
    "peak_kw",
    "valley_kw",
    "maximum_ramp_kw_per_15min",
    "net_electricity_cost_eur",
)

GERMAN_METRIC_LABELS = {
    "peak_kw": "Spitzenwert [kW]",
    "valley_kw": "Talwert [kW]",
    "maximum_ramp_kw_per_15min": "Maximale Rampe [kW/15 min]",
    "net_electricity_cost_eur": "Netto-Stromkosten [EUR]",
}


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
            "markersize": 6,
        },
        medianprops={"color": "#26323D", "linewidth": 1.8},
        whiskerprops={"color": "#4D5964", "linewidth": 1.35},
        capprops={"color": "#4D5964", "linewidth": 1.35},
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 5,
        },
    )

    for patch, label in zip(box["boxes"], labels):
        patch.set_facecolor(SCENARIO_COLORS[label])
        patch.set_alpha(0.72)
        patch.set_edgecolor("#33404C")
        patch.set_linewidth(1.35)

    ax.set_title(GERMAN_METRIC_LABELS[metric])
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3)


def plot_experiment_distribution(frame: pd.DataFrame, experiment_name: str):
    subset = frame.loc[frame["experiment"] == experiment_name].copy()
    if subset.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.6))

    for ax, metric in zip(axes.flat, FIGURE_METRICS):
        _draw_boxplot(ax, subset, metric)

    occurrences = subset[["replication", "date"]].drop_duplicates().shape[0]
    months = sorted(subset["month"].dropna().astype(int).unique().tolist())

    fig.suptitle(
        "C–B: Ergebnisverteilungen für repräsentative Tage\n"
        f"Monate: {months} | ausgewertete Tagesfälle: {occurrences}",
        y=0.995,
    )

    fig.text(
        0.5, 0.015,
        "S3: indirekte Stackelberg-Koordination   |   S4: direkte zentrale Steuerung (Referenz)",
        ha="center", fontsize=9.5,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])

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
