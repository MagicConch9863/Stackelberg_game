from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PERFECT_DIR = ROOT / "t5_perfect_mpc_mpc" / "results"
IID_DIR = ROOT / "t5_iid_mpc_mpc" / "results"
OUTPUT_DIR = ROOT / "results" / "comparisons"


def aggregate_timeseries(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return (
        frame.groupby(["date", "step", "hour"], as_index=False)
        .agg(
            load_real_kW=("load_real_kW", "sum"),
            pv_real_kW=("pv_real_kW", "sum"),
            bess_mpc_kW=("bess_mpc_kW", "sum"),
            realized_net_kW=("realized_net_kW", "sum"),
            pb_eur_per_kwh=("pb_eur_per_kwh", "first"),
        )
        .sort_values("step")
    )


def main() -> None:
    perfect = aggregate_timeseries(PERFECT_DIR / "mpc_timeseries.csv")
    iid = aggregate_timeseries(IID_DIR / "mpc_timeseries.csv")
    forecasts = pd.read_csv(IID_DIR / "rolling_forecasts.csv")
    forecasts = forecasts[forecasts["lead_step"] > 0].copy()
    forecasts["error_kW"] = forecasts["forecast_kW"] - forecasts["realization_kW"]

    actual_net = perfect["load_real_kW"] - perfect["pv_real_kW"]
    colors = {
        "actual": "#777777",
        "perfect": "#0072B2",
        "iid": "#D55E00",
        "perfect_bess": "#009E73",
        "iid_bess": "#CC79A7",
    }

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.3), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(perfect["hour"], actual_net, color=colors["actual"], ls=":", lw=1.8,
            label="Uncontrolled")
    ax.plot(perfect["hour"], perfect["realized_net_kW"], color=colors["perfect"], lw=1.6,
            label="T5-perfect")
    ax.plot(iid["hour"], iid["realized_net_kW"], color=colors["iid"], lw=1.4,
            label="T5 IID")
    ax.set(title="Grid exchange", xlabel="Time (h)", ylabel="Net exchange (kW)")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(perfect["hour"], perfect["bess_mpc_kW"], color=colors["perfect_bess"], lw=1.5,
            label="T5-perfect")
    ax.plot(iid["hour"], iid["bess_mpc_kW"], color=colors["iid_bess"], lw=1.4,
            label="T5 IID")
    ax.axhline(0.0, color="#555555", lw=0.7)
    ax.set(title="Aggregate BESS dispatch", xlabel="Time (h)", ylabel="BESS power (kW)")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.step(perfect["hour"], perfect["pb_eur_per_kwh"], where="post",
            color=colors["perfect"], lw=1.5, label="T5-perfect")
    ax.step(iid["hour"], iid["pb_eur_per_kwh"], where="post",
            color=colors["iid"], lw=1.4, label="T5 IID")
    ax.set(title="Tariff decisions", xlabel="Time (h)", ylabel=r"$P^B$ (EUR/kWh)")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    load_error = forecasts.loc[forecasts["variable"] == "load", "error_kW"].to_numpy()
    pv_error = forecasts.loc[forecasts["variable"] == "pv", "error_kW"].to_numpy()
    box = ax.boxplot(
        [load_error, pv_error], tick_labels=["Load", "PV"], widths=0.55,
        showfliers=False, patch_artist=True,
        medianprops={"color": "#222222", "linewidth": 1.2},
    )
    for patch, color in zip(box["boxes"], [colors["iid"], colors["perfect_bess"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.axhline(0.0, color="#555555", lw=0.7, ls="--")
    ax.set(title="T5 rolling forecast errors", xlabel="Forecast variable",
           ylabel="Forecast error (kW)")

    for ax in axes.flat:
        ax.grid(True, ls="--", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("T5-perfect vs T5 IID Gaussian — 2025-01-15", fontsize=14)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "t5_perfect_vs_t5_iid.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved comparison figure: {output}")
    print(
        "Forecast error RMSE (kW): "
        f"Load={np.sqrt(np.mean(load_error**2)):.4f}, "
        f"PV={np.sqrt(np.mean(pv_error**2)):.4f}"
    )


if __name__ == "__main__":
    main()
