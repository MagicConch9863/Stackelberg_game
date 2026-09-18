import random
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecast_common import (
    CORRELATED_GAUSSIAN_RESULTS,
    RANDOM_WALK_RESULTS,
    DirectARX,
    dataset,
    score,
)

TRAIN_YEAR = 2024
TEST_YEAR = 2025
HORIZON = 96
N_DAYS = 16
DAY_SEED = 20250831
NOISE_SEED = 20250902

AR1_RHOS = (0.5, 0.9, 0.99)
AR1_SIGMA_EPSILON_KW = 0.3
RANDOM_WALK_LOAD_SIGMA_EPSILON_KW = 0.45
RANDOM_WALK_PV_SIGMA_EPSILON_KW = 0.3
PV_MAX_KW = 5.5
PV_ACTIVE_THRESHOLD_KW = 0.01

COLORS = {
    "actual": "#0072B2",
    "base": "#E69F00",
    "noisy": "#009E73",
}


def configure_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def temporal_error(length, rho, sigma_epsilon, rng, stationary):
    error = np.empty(length, dtype=float)
    if stationary:
        error[0] = rng.normal(0.0, sigma_epsilon / np.sqrt(1.0 - rho ** 2))
    else:
        error[0] = 0.0

    innovations = rng.normal(0.0, sigma_epsilon, length)
    innovations[0] = error[0]
    for h in range(1, length):
        error[h] = rho * error[h - 1] + innovations[h]
    return error, innovations


def apply_physical_constraints(base, error, kind):
    base = np.asarray(base, dtype=float)
    raw = base + error

    if kind == "load":
        noisy = np.maximum(raw, 0.0)
        active = np.ones(len(base), dtype=bool)
        lower_clipped = raw < 0.0
        upper_clipped = np.zeros(len(base), dtype=bool)
    else:
        active = base >= PV_ACTIVE_THRESHOLD_KW
        noisy = np.zeros(len(base), dtype=float)
        noisy[active] = np.clip(raw[active], 0.0, PV_MAX_KW)
        lower_clipped = active & (raw < 0.0)
        upper_clipped = active & (raw > PV_MAX_KW)

    return noisy, raw, active, lower_clipped, upper_clipped


def lag1_correlation(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.std(values[:-1]) <= 1e-12 or np.std(values[1:]) <= 1e-12:
        return np.nan
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def save_grid(records, variable, label, scenario_label, output_dir, file_stem):
    fig, axes = plt.subplots(4, 4, figsize=(7.2, 7.0), sharex=True)
    axes = axes.ravel()
    x = np.arange(HORIZON) * 0.25

    for ax, record in zip(axes, records):
        ax.plot(x, record["actual"], color=COLORS["actual"], lw=0.9, label="Actual")
        ax.plot(x, record["base"], color=COLORS["base"], ls="--", lw=0.9,
                label="Direct ARX")
        ax.plot(x, record["noisy"], color=COLORS["noisy"], lw=0.8,
                alpha=0.9, label="ARX + error")
        ax.set_title(str(record["day"].date()), pad=2)
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.grid(True, ls="--", lw=0.4, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=3, frameon=False)
    fig.suptitle(f"{label}: {scenario_label}", fontsize=13, y=0.995)
    fig.supxlabel("Forecast horizon (h)", fontsize=11)
    fig.supylabel(f"{label} (kW)", fontsize=11)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.07, top=0.91,
                        wspace=0.20, hspace=0.34)

    png_path = output_dir / f"{file_stem}_{variable}_24h_16days_4x4.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png_path


def build_base_records(data, days, models):
    records = {"load": [], "pv": []}
    for day in days:
        start = pd.Timestamp(day)
        pos = data.index.get_loc(start)
        times = data.index[pos:pos + HORIZON]
        for kind, column in (("load", "load_kw"), ("pv", "pv_kw")):
            records[kind].append({
                "day": start,
                "times": times,
                "actual": data[column].iloc[pos:pos + HORIZON].to_numpy(float),
                "base": models[kind].forecast(data[column], pos),
            })
    return records


def run_scenario(base_records, name, rho, stationary, output_dir, seed):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    metric_rows = []
    forecast_rows = []
    plot_records = {"load": [], "pv": []}

    for kind in ("load", "pv"):
        sigma_epsilon = (
            AR1_SIGMA_EPSILON_KW
            if stationary
            else (
                RANDOM_WALK_LOAD_SIGMA_EPSILON_KW
                if kind == "load"
                else RANDOM_WALK_PV_SIGMA_EPSILON_KW
            )
        )
        print(
            f"    Generating {kind.upper()} paths "
            f"(rho={rho:.2f}, sigma_epsilon={sigma_epsilon:.2f} kW)...",
            flush=True,
        )
        for record in base_records[kind]:
            error, innovations = temporal_error(
                HORIZON, rho, sigma_epsilon, rng, stationary
            )
            noisy, raw, active, lower_clip, upper_clip = apply_physical_constraints(
                record["base"], error, kind
            )
            base_mae, base_rmse = score(record["actual"], record["base"])
            noisy_mae, noisy_rmse = score(record["actual"], noisy)
            inactive_violation_count = int(np.count_nonzero(noisy[~active]))

            metric_rows.append({
                "scenario": name,
                "rho": rho,
                "sigma_epsilon_kW": sigma_epsilon,
                "date": record["day"].date(),
                "variable": kind,
                "base_MAE_kW": base_mae,
                "base_RMSE_kW": base_rmse,
                "noisy_MAE_kW": noisy_mae,
                "noisy_RMSE_kW": noisy_rmse,
                "raw_error_std_kW": float(np.std(error)),
                "empirical_lag1_correlation": lag1_correlation(error),
                "lower_clip_rate": float(np.mean(lower_clip)),
                "upper_clip_rate": float(np.mean(upper_clip)),
                "inactive_PV_rate": float(np.mean(~active)) if kind == "pv" else 0.0,
                "inactive_PV_violation_count": inactive_violation_count,
            })

            plot_records[kind].append({**record, "noisy": noisy})
            for values in zip(
                record["times"], record["actual"], record["base"], raw, noisy,
                error, noisy - record["base"], innovations, active,
                lower_clip, upper_clip,
            ):
                forecast_rows.append([
                    name, rho, sigma_epsilon, record["day"].date(), kind,
                    *values,
                ])

    metrics = pd.DataFrame(metric_rows)
    forecasts = pd.DataFrame(forecast_rows, columns=[
        "scenario", "rho", "sigma_epsilon_kW", "day", "variable",
        "timestamp", "actual_kW", "base_arx_kW", "raw_forecast_kW",
        "physical_forecast_kW", "raw_error_kW", "effective_error_kW",
        "innovation_kW", "pv_active", "lower_clipped", "upper_clipped",
    ])
    summary = metrics.groupby(["scenario", "rho", "variable"], as_index=False).mean(
        numeric_only=True
    )

    metrics.to_csv(output_dir / "metrics.csv", index=False)
    forecasts.to_csv(output_dir / "forecasts.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    stem = "correlated" if stationary else "random_walk"
    for kind, label in (("load", "Load"), ("pv", "PV")):
        sigma_epsilon = (
            AR1_SIGMA_EPSILON_KW
            if stationary
            else (
                RANDOM_WALK_LOAD_SIGMA_EPSILON_KW
                if kind == "load"
                else RANDOM_WALK_PV_SIGMA_EPSILON_KW
            )
        )
        scenario_label = (
            f"AR(1), rho={rho:.2f}, sigma_epsilon={sigma_epsilon:.2f} kW"
            if stationary else
            f"Random walk, rho=1.00, sigma_epsilon={sigma_epsilon:.2f} kW"
        )
        print(f"    Saving {label} figure...", flush=True)
        save_grid(plot_records[kind], kind, label, scenario_label, output_dir, stem)

    return metrics, summary


def save_comparison(all_metrics, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_csv(output_dir / "all_scenarios_metrics.csv", index=False)
    summary = metrics.groupby(["scenario", "rho", "variable"], as_index=False).mean(
        numeric_only=True
    )
    summary.to_csv(output_dir / "all_scenarios_summary.csv", index=False)

    labels = ["rho=0.50", "rho=0.90", "rho=0.99", "RW"]
    order = [("ar1", 0.5), ("ar1", 0.9), ("ar1", 0.99), ("random_walk", 1.0)]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for ax, variable, title in zip(axes, ("load", "pv"), ("Load", "PV")):
        groups = [
            metrics[(metrics["scenario"] == scenario) &
                    np.isclose(metrics["rho"], rho) &
                    (metrics["variable"] == variable)]["noisy_RMSE_kW"].to_numpy()
            for scenario, rho in order
        ]
        box = ax.boxplot(groups, patch_artist=True, widths=0.6, showfliers=True)
        for patch, color in zip(box["boxes"], ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
        ax.set_xticks(range(1, 5), labels, rotation=20)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Forecast RMSE (kW)")
        ax.grid(axis="y", ls="--", lw=0.5, alpha=0.3)
    fig.suptitle("Temporal-error scenario comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "rmse_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    configure_plot_style()
    print("[1/7] Building the SimBench/SMARD dataset...", flush=True)
    data = dataset()
    train = data[data.index.year == TRAIN_YEAR]
    test = data[data.index.year == TEST_YEAR]
    complete_days = [d for d, group in test.groupby(test.index.normalize()) if len(group) == HORIZON]
    if len(complete_days) < N_DAYS:
        raise RuntimeError(f"Only {len(complete_days)} complete test days found.")
    days = sorted(random.Random(DAY_SEED).sample(complete_days, N_DAYS))
    print("[2/7] Training the 96-step Load Direct ARX model...", flush=True)
    load_model = DirectARX(HORIZON, "load").fit(train["load_kw"])
    print("[3/7] Training the 96-step PV Direct ARX model...", flush=True)
    pv_model = DirectARX(HORIZON, "pv").fit(train["pv_kw"])
    models = {"load": load_model, "pv": pv_model}
    print("[4/7] Building the common 16-day base forecasts...", flush=True)
    base_records = build_base_records(data, days, models)

    all_metrics = []
    print("[5/7] Running the stationary AR(1) scenarios...", flush=True)
    for index, rho in enumerate(AR1_RHOS):
        output_dir = CORRELATED_GAUSSIAN_RESULTS / f"rho_{rho:.2f}"
        print(f"  AR(1) rho={rho:.2f}", flush=True)
        metrics, _ = run_scenario(
            base_records, "ar1", rho, True, output_dir, NOISE_SEED + index
        )
        all_metrics.append(metrics)

    print("[6/7] Running the unrestricted Random walk scenario...", flush=True)
    metrics, _ = run_scenario(
        base_records, "random_walk", 1.0, False,
        RANDOM_WALK_RESULTS / "rho_1.00", NOISE_SEED + len(AR1_RHOS),
    )
    all_metrics.append(metrics)
    print("[7/7] Saving comparisons and validating physical constraints...", flush=True)
    save_comparison(all_metrics, CORRELATED_GAUSSIAN_RESULTS / "comparison")

    combined = pd.concat(all_metrics, ignore_index=True)
    violations = int(combined["inactive_PV_violation_count"].sum())
    if violations:
        raise RuntimeError(f"Detected {violations} inactive-period PV violations.")
    print("Completed: inactive-period PV violations = 0", flush=True)


if __name__ == "__main__":
    main()
