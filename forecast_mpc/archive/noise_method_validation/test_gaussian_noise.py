
import random
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecast_common import dataset, DirectARX, score, GAUSSIAN_NOISE_RESULTS

TRAIN_YEAR = 2024
TEST_YEAR = 2025
N_DAYS = 16
DAY_SEED = 20250831
NOISE_SEED = 20250901

HORIZONS = {"24h": 96}

# iid Gaussian noise strength = ratio * training-series standard deviation
LOAD_SIGMA_RATIO = 0.10
PV_SIGMA_RATIO = 0.10
PV_MAX_KW = 5.5


def add_gaussian_noise(base, sigma_kw, rng, kind):
    base = np.asarray(base, float)
    noise = rng.normal(0.0, sigma_kw, len(base))
    noisy = base + noise

    if kind == "load":
        noisy = np.clip(noisy, 0.0, None)
    else:
        noisy = np.clip(noisy, 0.0, PV_MAX_KW)
        # Do not create artificial PV at night.
        noisy[base <= 1e-9] = 0.0

    return noisy, noise


def plot_grid(days, data, model, column, label, tag, sigma_kw, rng):
    fig, axes = plt.subplots(4, 4, figsize=(16, 10.5))
    axes = axes.ravel()
    metric_rows, forecast_rows = [], []

    for ax, day in zip(axes, days):
        start = pd.Timestamp(day)
        pos = data.index.get_loc(start)

        actual = data[column].iloc[pos:pos + model.H].to_numpy(float)
        base = model.forecast(data[column], pos)
        noisy, noise = add_gaussian_noise(
            base, sigma_kw, rng, "pv" if column == "pv_kw" else "load"
        )

        x = np.arange(model.H) * 0.25
        ax.plot(x, actual, lw=1.0, label="Actual")
        ax.plot(x, base, "--", lw=1.05, label="Direct ARX")
        ax.plot(x, noisy, lw=0.95, alpha=0.85, label="ARX + Gaussian")
        ax.set_title(str(start.date()), fontsize=8.5, pad=3)
        ax.grid(alpha=0.18, linewidth=0.45)
        ax.tick_params(labelsize=7.5)

        if model.H == 16:
            ax.set_xlim(0, 4)
            ax.set_xticks([0, 1, 2, 3, 4])
        else:
            ax.set_xlim(0, 24)
            ax.set_xticks([0, 6, 12, 18, 24])

        bmae, brmse = score(actual, base)
        nmae, nrmse = score(actual, noisy)
        metric_rows.append([start.date(), sigma_kw, bmae, brmse, nmae, nrmse])

        times = data.index[pos:pos + model.H]
        for t, a, b, n, w in zip(times, actual, base, noisy, noise):
            forecast_rows.append([tag, start.date(), t, column, a, b, n, w])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.952), ncol=3, frameon=False, fontsize=9)
    fig.suptitle(
        f"{label}: Direct ARX + iid Gaussian noise ({tag}), sigma={sigma_kw:.3f} kW",
        fontsize=12, y=0.992
    )
    fig.supxlabel("Forecast horizon [h]", fontsize=9.5)
    fig.supylabel(f"{label} [kW]", fontsize=9.5)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.065,
                        top=0.91, wspace=0.20, hspace=0.32)

    fig_path = GAUSSIAN_NOISE_RESULTS / f"gaussian_{column.replace('_kw','')}_{tag}_16days_4x4.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return metric_rows, forecast_rows, fig_path


def main():
    print("Reading and building dataset...")
    data = dataset()
    train = data[data.index.year == TRAIN_YEAR]
    test = data[data.index.year == TEST_YEAR]

    load_sigma = LOAD_SIGMA_RATIO * float(train["load_kw"].std())
    pv_sigma = PV_SIGMA_RATIO * float(train["pv_kw"].std())

    print(f"Load sigma = {load_sigma:.4f} kW")
    print(f"PV sigma   = {pv_sigma:.4f} kW")

    complete_days = [
        d for d, g in test.groupby(test.index.normalize()) if len(g) == 96
    ]
    chosen = sorted(random.Random(DAY_SEED).sample(complete_days, N_DAYS))

    rng = np.random.default_rng(NOISE_SEED)
    metrics_all, forecasts_all = [], []

    for tag, H in HORIZONS.items():
        print(f"\nTraining Direct ARX: {tag}")
        load_model = DirectARX(H, "load").fit(train["load_kw"])
        pv_model = DirectARX(H, "pv").fit(train["pv_kw"])

        lr, lf, lfig = plot_grid(
            chosen, data, load_model, "load_kw", "Load", tag, load_sigma, rng
        )
        pr, pf, pfig = plot_grid(
            chosen, data, pv_model, "pv_kw", "PV", tag, pv_sigma, rng
        )

        for a, b in zip(lr, pr):
            metrics_all.append([
                tag, a[0], a[1], b[1],
                a[2], a[3], a[4], a[5],
                b[2], b[3], b[4], b[5]
            ])

        forecasts_all.extend(lf)
        forecasts_all.extend(pf)

        print("Load figure:", lfig)
        print("PV figure:  ", pfig)

    metrics = pd.DataFrame(metrics_all, columns=[
        "horizon", "date", "load_sigma_kW", "pv_sigma_kW",
        "load_base_MAE_kW", "load_base_RMSE_kW",
        "load_noisy_MAE_kW", "load_noisy_RMSE_kW",
        "pv_base_MAE_kW", "pv_base_RMSE_kW",
        "pv_noisy_MAE_kW", "pv_noisy_RMSE_kW"
    ])
    metrics.to_csv(GAUSSIAN_NOISE_RESULTS / "gaussian_noise_metrics.csv", index=False)

    pd.DataFrame(forecasts_all, columns=[
        "horizon", "day", "timestamp", "variable",
        "actual_kW", "base_arx_kW", "arx_plus_gaussian_kW", "gaussian_noise_kW"
    ]).to_csv(GAUSSIAN_NOISE_RESULTS / "gaussian_noise_forecasts.csv", index=False)

    summary = metrics.groupby("horizon")[[
        "load_base_MAE_kW", "load_noisy_MAE_kW",
        "load_base_RMSE_kW", "load_noisy_RMSE_kW",
        "pv_base_MAE_kW", "pv_noisy_MAE_kW",
        "pv_base_RMSE_kW", "pv_noisy_RMSE_kW"
    ]].mean()
    summary.to_csv(GAUSSIAN_NOISE_RESULTS / "gaussian_noise_summary.csv")

    print("\nMean error comparison:")
    print(summary.round(4))
    print("\nDone.")


if __name__ == "__main__":
    main()
