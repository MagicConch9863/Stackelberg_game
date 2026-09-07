from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "comparisons" / "mpc_forecast_process_explanation.png"


def style_axis(ax, title):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Time")
    ax.set_ylabel("Load (kW)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
    })
    time = np.arange(9)
    labels = ["08:00", "08:15", "08:30", "08:45", "09:00",
              "09:15", "09:30", "09:45", "10:00"]
    actual = np.array([4.0, 4.2, 4.5, 4.7, 5.0, 5.2, 5.0, 4.8, 4.6])
    fixed = np.array([4.3, 3.9, 4.8, 5.1, 4.6, 5.4, 5.2, 4.5, 4.9])
    blue, orange, green, gray = "#0072B2", "#D55E00", "#009E73", "#777777"

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.1), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(time, actual, "o-", color=blue, label="Realization")
    ax.plot(time, fixed, "s--", color=orange, label="Fixed noisy forecast")
    ax.set_xticks(time[::2], labels[::2])
    style_axis(ax, "A. One noisy forecast trajectory is generated first")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(time[:7], fixed[:7], "s--", color=orange, alpha=0.85,
            label="Forecast used in horizon")
    ax.scatter(0, actual[0], s=70, color=green, zorder=4, label="Current measurement")
    ax.axvspan(-0.15, 6.15, color=blue, alpha=0.08, label="Horizon at k")
    ax.annotate("Execute only this step", xy=(0, actual[0]), xytext=(1.0, 5.45),
                arrowprops={"arrowstyle": "->", "color": green}, color=green)
    ax.set_xticks(time[:7], labels[:7])
    style_axis(ax, "B. MPC at k = 08:00")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1, 0]
    ax.plot(time[1:8], fixed[1:8], "s--", color=orange,
            label="Unchanged future forecast")
    ax.scatter(1, actual[1], s=70, color=green, zorder=4, label="New measurement")
    ax.axvspan(0.85, 7.15, color=blue, alpha=0.08, label="Shifted horizon at k+1")
    ax.annotate("Previous plan discarded;\nexecute new first step",
                xy=(1, actual[1]), xytext=(2.4, 4.05),
                arrowprops={"arrowstyle": "->", "color": green}, color=green)
    ax.set_xticks(time[1:8], labels[1:8])
    style_axis(ax, "C. MPC at k+1 = 08:15")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1, 1]
    origins = np.array([0, 1, 2])
    current_method = np.array([fixed[4], fixed[4], fixed[4]])
    online_example = np.array([fixed[4], 4.82, 4.94])
    ax.plot(origins, current_method, "s--", color=orange,
            label="Current fixed-scenario method")
    ax.plot(origins, online_example, "o-", color=blue,
            label="Online forecast update (desired)")
    ax.axhline(actual[4], color=gray, linestyle=":", label="09:00 realization")
    ax.set_xticks(origins, ["08:00", "08:15", "08:30"])
    ax.set_xlabel("Forecast origin for the same 09:00 target")
    ax.set_ylabel("09:00 load forecast (kW)")
    ax.set_title("D. Why assimilating new measurements matters", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Current fixed-scenario MPC forecast process", fontsize=14)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
