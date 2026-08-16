import os
import copy
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import StackelbergConfig, load_price_data, make_profiles
from build_base_network import feeder_data
from stackelberg_kkt_model import solve_stackelberg_mpec
from big_hype_fd_solver import solve_big_hype_fd


HORIZONS = range(4, 21)
N_REPEATS = 3
METHODS = ("sos1", "bigm", "big_hype_fd")


def build_case(cfg, feeder, load, pv):
    buses = list(cfg.active_buses)
    return {
        "buses": buses,
        "net": feeder,
        "load": {b: load[k] for k, b in enumerate(buses)},
        "pv": {b: pv[k] for k, b in enumerate(buses)},
    }


def get_baseline(load, pv):
    return load.sum(axis=0) - pv.sum(axis=0)


def solve_method(cfg, data, market, method):
    if method in ("sos1", "bigm", "duality"):
        return solve_stackelberg_mpec(cfg, data, market, method=method)

    if method == "big_hype_fd":
        return solve_big_hype_fd(cfg, data, market)

    raise ValueError(f"Unknown method: {method}")


def total_dload(result):
    total = None
    for sch in result["schedules"].values():
        arr = sch.get("dload", None)
        if arr is None:
            continue
        total = arr.copy() if total is None else total + arr
    return total


def total_curtailment(result):
    total = None
    for sch in result["schedules"].values():
        arr = sch.get("curt", None)
        if arr is None:
            continue
        total = arr.copy() if total is None else total + arr
    return total


def battery_buses(cfg, results):
    if not results:
        return []

    sample = next(iter(results.values()))
    out = []

    for b in getattr(cfg, "battery_buses", []):
        if b in sample["schedules"]:
            sch = sample["schedules"][b]
            if "ch" in sch and "dis" in sch:
                out.append(b)

    return out


def battery_net(result, bus):
    sch = result["schedules"][bus]
    return sch["ch"] - sch["dis"]


def metric_row(cfg, horizon, repeat_id, method, result, baseline, ref):
    after = result["agg_load"]

    if method == "big_hype_fd":
        mip_gap = np.nan
    else:
        mip_gap = float(result.get("mip_gap", np.nan))

    row = {
        "horizon": horizon,
        "repeat": repeat_id,
        "method": method,
        "status": result.get("status", np.nan),
        "runtime_sec": float(result.get("runtime_sec", np.nan)),
        "mip_gap": mip_gap,
        "objective": float(result.get("obj_value", np.nan)),
        "peak_after_kw": float(np.max(np.abs(after))),
        "std_after_kw": float(np.std(after)),
        "energy_after_kwh": float(np.sum(after) * cfg.dt),
        "energy_diff_kwh": float((np.sum(after) - np.sum(baseline)) * cfg.dt),
    }

    if ref is not None:
        row["max_netload_error_vs_sos1_kw"] = float(
            np.max(np.abs(after - ref["agg_load"]))
        )
        row["max_buy_price_error_vs_sos1"] = float(
            np.max(np.abs(result["price_buy"] - ref["price_buy"]))
        )
        row["max_sell_price_error_vs_sos1"] = float(
            np.max(np.abs(result["price_sell"] - ref["price_sell"]))
        )
    else:
        row["max_netload_error_vs_sos1_kw"] = 0.0
        row["max_buy_price_error_vs_sos1"] = 0.0
        row["max_sell_price_error_vs_sos1"] = 0.0

    return row


def time_series_rows(cfg, horizon, repeat_id, method, result, baseline, batt_buses):
    rows = []
    dload = total_dload(result)
    curt = total_curtailment(result)

    for t in range(cfg.horizon):
        row = {
            "horizon": horizon,
            "repeat": repeat_id,
            "method": method,
            "time_step": t,
            "time_hour": t * cfg.dt,
            "baseline_net_load_kw": float(baseline[t]),
            "optimized_net_load_kw": float(result["agg_load"][t]),
            "price_buy_eur_per_kwh": float(result["price_buy"][t]),
            "price_sell_eur_per_kwh": float(result["price_sell"][t]),
            "total_demand_response_kw": float(dload[t]) if dload is not None else np.nan,
            "total_pv_curtailment_kw": float(curt[t]) if curt is not None else np.nan,
        }

        for b in batt_buses:
            row[f"{b}_battery_net_kw"] = float(battery_net(result, b)[t])

        rows.append(row)

    return rows


def format_value(value, col):
    if pd.isna(value):
        return "N/A"

    if col in ("objective", "runtime_sec"):
        return f"{value:.3f}"

    if col == "mip_gap":
        return f"{value:.4f}"

    return f"{value:.4f}"


def plot_dashboard(cfg, horizon, repeat_id, market, baseline, results, metrics_df, fig_path):
    t = np.arange(cfg.horizon)

    fig, axes = plt.subplots(3, 4, figsize=(18, 10), constrained_layout=True)
    axes = axes.ravel()

    styles = {
        "sos1": {"linestyle": "--", "marker": "o", "linewidth": 1.7, "markersize": 3, "zorder": 3},
        "bigm": {"linestyle": "-", "marker": "x", "linewidth": 1.4, "markersize": 3, "zorder": 2},
        "big_hype_fd": {"linestyle": "-", "marker": "s", "linewidth": 1.4, "markersize": 3, "zorder": 1},
    }

    plot_order = ["big_hype_fd", "bigm", "sos1"]

    ax = axes[0]
    ax.plot(t, baseline, "--", linewidth=2.0, label="baseline")
    for method in plot_order:
        if method in results:
            ax.plot(t, results[method]["agg_load"], label=method, **styles[method])
    ax.set_title("Total Net Load")
    ax.set_ylabel("kW")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.step(t, market, where="post", linestyle="--", linewidth=1.5, label="market")
    for method in plot_order:
        if method in results:
            ax.step(t, results[method]["price_buy"], where="post", label=method)
    ax.set_title("Buy Price")
    ax.set_ylabel("EUR/kWh")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)

    ax = axes[2]
    for method in plot_order:
        if method in results:
            ax.step(t, results[method]["price_sell"], where="post", label=method)
    ax.set_title("Sell Price")
    ax.set_ylabel("EUR/kWh")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)

    ax = axes[3]
    ax.axis("off")
    show_cols = [
        "method",
        "objective",
        "peak_after_kw",
        "std_after_kw",
        "energy_diff_kwh",
        "runtime_sec",
        "mip_gap",
    ]

    table_df = metrics_df[show_cols].copy()
    for c in show_cols[1:]:
        table_df[c] = table_df[c].map(lambda x, col=c: format_value(x, col))

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    ax.set_title("Metrics Table")

    ax = axes[4]
    if "sos1" in results:
        ref = results["sos1"]
        for method in plot_order:
            if method in results and method != "sos1":
                err = results[method]["agg_load"] - ref["agg_load"]
                ax.plot(t, err, label=f"{method} - sos1")
    ax.axhline(0.0, linewidth=1.0, color="black")
    ax.set_title("Net Load Error vs SOS1")
    ax.set_ylabel("kW")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)

    ax = axes[5]
    if "sos1" in results:
        ref = results["sos1"]
        for method in plot_order:
            if method in results and method != "sos1":
                err = results[method]["price_buy"] - ref["price_buy"]
                ax.step(t, err, where="post", label=f"{method} - sos1")
    ax.axhline(0.0, linewidth=1.0, color="black")
    ax.set_title("Buy Price Error vs SOS1")
    ax.set_ylabel("EUR/kWh")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)

    ax = axes[6]
    for method in plot_order:
        if method in results:
            dload = total_dload(results[method])
            if dload is not None:
                ax.plot(t, dload, label=method)
    ax.axhline(0.0, linewidth=1.0, color="black")
    ax.set_title("Total Load Adjustment ΔL")
    ax.set_ylabel("adjustment kW")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)

    ax = axes[7]
    for method in plot_order:
        if method in results:
            curt = total_curtailment(results[method])
            if curt is not None:
                ax.plot(t, curt, label=method)
    ax.set_title("Total PV Curtailment")
    ax.set_ylabel("kW")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)

    batt = battery_buses(cfg, results)[:4]

    for k in range(4):
        ax = axes[8 + k]

        if k >= len(batt):
            ax.axis("off")
            continue

        b = batt[k]

        for method in plot_order:
            if method in results:
                ax.plot(
                    t,
                    battery_net(results[method], b),
                    label=method,
                    **styles[method],
                )

        ax.axhline(0.0, linewidth=1.0, color="black")
        ax.set_title(f"{b}: Battery Net Power (+ch, -dis)")
        ax.set_ylabel("kW")
        ax.set_xlabel("time step")
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"SOS1 vs Big-M vs BIG-Hype-like | horizon={horizon}, repeat={repeat_id}",
        fontsize=15,
    )

    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def run_one_case(base_cfg, horizon, repeat_id, out_dir, fig_dir):
    cfg = copy.copy(base_cfg)

    cfg.horizon = horizon
    cfg.output_dir = out_dir

    cfg.solver_seed = 1000 + 100 * horizon + repeat_id
    cfg.big_hype_seed = 2000 + 100 * horizon + repeat_id

    market = load_price_data("price_data.csv", cfg.horizon)
    load, pv = make_profiles(cfg)
    feeder = feeder_data(cfg.active_buses)
    data = build_case(cfg, feeder, load, pv)
    baseline = get_baseline(load, pv)

    results = {}

    for method in METHODS:
        print(f"\n--- horizon={horizon}, repeat={repeat_id}, method={method} ---")

        try:
            result = solve_method(cfg, data, market, method)
        except Exception as exc:
            print(f"{method} failed with exception:")
            print(exc)
            traceback.print_exc()
            result = None

        if result is None:
            print(f"{method}: no feasible solution")
            continue

        results[method] = result

        gap_text = "N/A" if method == "big_hype_fd" else f"{result.get('mip_gap', np.nan):.6f}"
        print(
            f"{method}: obj={result['obj_value']:.6f}, "
            f"runtime={result['runtime_sec']:.2f}s, gap={gap_text}"
        )

    if not results:
        return [], [], {}

    ref = results.get("sos1", None)
    batt = battery_buses(cfg, results)

    metric_rows = []
    series_rows = []

    for method, result in results.items():
        metric_rows.append(metric_row(cfg, horizon, repeat_id, method, result, baseline, ref))
        series_rows.extend(time_series_rows(cfg, horizon, repeat_id, method, result, baseline, batt))

    metrics_df = pd.DataFrame(metric_rows)

    fig_path = os.path.join(fig_dir, f"h{horizon:02d}_run{repeat_id:02d}_dashboard.png")
    plot_dashboard(cfg, horizon, repeat_id, market, baseline, results, metrics_df, fig_path)

    runtime_row = {
        "horizon": horizon,
        "repeat": repeat_id,
    }

    for method in METHODS:
        if method in results:
            runtime_row[f"{method}_runtime_sec"] = float(results[method]["runtime_sec"])
            runtime_row[f"{method}_objective"] = float(results[method]["obj_value"])
            runtime_row[f"{method}_peak_after_kw"] = float(np.max(np.abs(results[method]["agg_load"])))
            runtime_row[f"{method}_status"] = results[method]["status"]
            runtime_row[f"{method}_mip_gap"] = (
                np.nan if method == "big_hype_fd" else float(results[method].get("mip_gap", np.nan))
            )
        else:
            runtime_row[f"{method}_runtime_sec"] = np.nan
            runtime_row[f"{method}_objective"] = np.nan
            runtime_row[f"{method}_peak_after_kw"] = np.nan
            runtime_row[f"{method}_status"] = np.nan
            runtime_row[f"{method}_mip_gap"] = np.nan

    return metric_rows, series_rows, runtime_row


def main():
    base_cfg = StackelbergConfig()

    out_dir = os.path.join("outputs", "batch_horizon")
    fig_dir = os.path.join(out_dir, "figures")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    all_metrics = []
    all_series = []
    runtime_rows = []

    for horizon in HORIZONS:
        for repeat_id in range(1, N_REPEATS + 1):
            print("\n" + "=" * 80)
            print(f"Running horizon={horizon}, repeat={repeat_id}")
            print("=" * 80)

            metric_rows, series_rows, runtime_row = run_one_case(
                base_cfg,
                horizon,
                repeat_id,
                out_dir,
                fig_dir,
            )

            all_metrics.extend(metric_rows)
            all_series.extend(series_rows)

            if runtime_row:
                runtime_rows.append(runtime_row)

            pd.DataFrame(all_metrics).to_csv(
                os.path.join(out_dir, "batch_metrics_all.csv"),
                index=False,
            )

            pd.DataFrame(all_series).to_csv(
                os.path.join(out_dir, "batch_time_series_all.csv"),
                index=False,
            )

            pd.DataFrame(runtime_rows).to_csv(
                os.path.join(out_dir, "batch_runtime_table.csv"),
                index=False,
            )

    metrics_df = pd.DataFrame(all_metrics)
    runtime_df = pd.DataFrame(runtime_rows)

    if not metrics_df.empty:
        summary = (
            metrics_df
            .groupby(["horizon", "method"], dropna=False)
            .agg(
                runtime_mean_sec=("runtime_sec", "mean"),
                runtime_std_sec=("runtime_sec", "std"),
                objective_mean=("objective", "mean"),
                objective_std=("objective", "std"),
                peak_after_mean_kw=("peak_after_kw", "mean"),
                peak_after_std_kw=("peak_after_kw", "std"),
                std_after_mean_kw=("std_after_kw", "mean"),
                energy_diff_mean_kwh=("energy_diff_kwh", "mean"),
                max_netload_error_mean_kw=("max_netload_error_vs_sos1_kw", "mean"),
                max_buy_price_error_mean=("max_buy_price_error_vs_sos1", "mean"),
            )
            .reset_index()
        )

        summary.to_csv(
            os.path.join(out_dir, "batch_runtime_summary.csv"),
            index=False,
        )

    print("\nBatch simulation finished.")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()