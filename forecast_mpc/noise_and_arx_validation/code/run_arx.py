
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from forecast_common import dataset, DirectARX, score, DIRECT_ARX_RESULTS

TRAIN_YEAR = 2024
TEST_YEAR = 2025

N_DAYS = 16
SEED = 20250831

HORIZONS = {
    "24h": 96,
}


def plot_grid(days, data, model, column, label, tag):
    fig, axes = plt.subplots(4, 4, figsize=(16, 10.5))
    axes = axes.ravel()

    rows = []

    for ax, day in zip(axes, days):
        start = pd.Timestamp(day)
        pos = data.index.get_loc(start)

        actual = data[column].iloc[pos:pos + model.H].to_numpy(float)
        pred = model.forecast(data[column], pos)

        x = np.arange(model.H) * 0.25

        ax.plot(x, actual, lw=1.05, label="Actual")
        ax.plot(x, pred, "--", lw=1.15, label="Direct ARX")

        ax.set_title(str(start.date()), fontsize=8.5, pad=3)
        ax.grid(alpha=0.18, linewidth=0.45)
        ax.tick_params(labelsize=7.5)

        if model.H == 16:
            ax.set_xlim(0, 4)
            ax.set_xticks([0, 1, 2, 3, 4])
        else:
            ax.set_xlim(0, 24)
            ax.set_xticks([0, 6, 12, 18, 24])

        mae, rmse = score(actual, pred)
        rows.append([start.date(), mae, rmse])

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.952),
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(
        f"{label}: Direct Multi-step ARX ({tag}) — 2024 training / 2025 test",
        fontsize=12,
        y=0.992,
    )

    fig.supxlabel("Forecast horizon [h]", fontsize=9.5)
    fig.supylabel(f"{label} [kW]", fontsize=9.5)

    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.065,
        top=0.91,
        wspace=0.20,
        hspace=0.32,
    )

    out = DIRECT_ARX_RESULTS / f"direct_arx_{column.replace('_kw', '')}_{tag}_16days_4x4.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return rows, out


def main():
    print("Reading and building dataset...")
    data = dataset()

    train = data[data.index.year == TRAIN_YEAR]
    test = data[data.index.year == TEST_YEAR]

    if train.empty or test.empty:
        raise RuntimeError("2024 training data or 2025 test data is missing.")

    days = [
        day
        for day, group in test.groupby(test.index.normalize())
        if len(group) == 96
    ]

    if len(days) < N_DAYS:
        raise RuntimeError(f"Only {len(days)} complete 2025 days found.")

    chosen = sorted(random.Random(SEED).sample(days, N_DAYS))

    all_rows = []

    for tag, H in HORIZONS.items():
        print(f"\nTraining Direct ARX for {tag} ({H} steps)...")

        load_model = DirectARX(H, "load").fit(train["load_kw"])
        pv_model = DirectARX(H, "pv").fit(train["pv_kw"])

        load_rows, load_fig = plot_grid(
            chosen, data, load_model,
            "load_kw", "Load", tag
        )

        pv_rows, pv_fig = plot_grid(
            chosen, data, pv_model,
            "pv_kw", "PV", tag
        )

        for lr, pr in zip(load_rows, pv_rows):
            all_rows.append([
                tag,
                lr[0],
                lr[1],
                lr[2],
                pr[1],
                pr[2],
            ])

        print("Load figure:", load_fig)
        print("PV figure:  ", pv_fig)

    metrics = pd.DataFrame(
        all_rows,
        columns=[
            "horizon",
            "date",
            "load_MAE_kW",
            "load_RMSE_kW",
            "pv_MAE_kW",
            "pv_RMSE_kW",
        ],
    )

    metrics.to_csv(
        DIRECT_ARX_RESULTS / "direct_arx_metrics.csv",
        index=False,
    )

    summary = metrics.groupby("horizon")[
        [
            "load_MAE_kW",
            "load_RMSE_kW",
            "pv_MAE_kW",
            "pv_RMSE_kW",
        ]
    ].mean()

    summary.to_csv(
        DIRECT_ARX_RESULTS / "direct_arx_metrics_summary.csv"
    )

    print("\nMean metrics:")
    print(summary.round(4))
    print("\nDone.")


if __name__ == "__main__":
    main()
