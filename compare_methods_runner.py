import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import StackelbergConfig, load_price_data, make_profiles
from build_base_network import feeder_data
from stackelberg_kkt_model import solve_stackelberg_mpec
from big_hype_fd_solver import solve_big_hype_fd


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


def get_metrics(cfg, baseline, result, ref=None):
    after = result["agg_load"]
    method = result["method"]

    if method == "big_hype_fd":
        mip_gap = np.nan
    else:
        mip_gap = float(result["mip_gap"])

    row = {
        "method": method,
        "status": result["status"],
        "runtime_sec": float(result["runtime_sec"]),
        "mip_gap": mip_gap,
        "objective": float(result["obj_value"]),
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
    else:
        row["max_netload_error_vs_sos1_kw"] = 0.0
        row["max_buy_price_error_vs_sos1"] = 0.0

    return row


def get_total_dload(result):
    total = None

    for sch in result["schedules"].values():
        arr = sch.get("dload", None)
        if arr is None:
            continue

        total = arr.copy() if total is None else total + arr

    return total


def get_total_curtailment(result):
    total = None

    for sch in result["schedules"].values():
        arr = sch.get("curt", None)
        if arr is None:
            continue

        total = arr.copy() if total is None else total + arr

    return total


def get_battery_buses(cfg, results):
    if not results:
        return []

    sample = next(iter(results.values()))
    buses = []

    for b in getattr(cfg, "battery_buses", []):
        if b not in sample["schedules"]:
            continue

        sch = sample["schedules"][b]
        if "ch" in sch and "dis" in sch:
            buses.append(b)

    return buses[:4]


def solve_method(cfg, data, market, method):
    if method in ("sos1", "bigm", "duality"):
        return solve_stackelberg_mpec(cfg, data, market, method=method)

    if method == "big_hype_fd":
        return solve_big_hype_fd(cfg, data, market)

    raise ValueError(f"Unknown method: {method}")


def format_table_value(col, value):
    if pd.isna(value):
        return "N/A"

    if col in ("runtime_sec", "objective"):
        return f"{value:.3f}"

    if col == "mip_gap":
        return f"{value:.4f}"

    return f"{value:.4f}"


def plot_dashboard(cfg, market, baseline, results, metrics_df):
    os.makedirs(cfg.output_dir, exist_ok=True)

    t = np.arange(cfg.horizon)
    fig, axes = plt.subplots(3, 4, figsize=(18, 10), constrained_layout=True)
    axes = axes.ravel()

    # 1. Total Net Load
    ax = axes[0]
    ax.plot(t, baseline, "--", linewidth=2.0, label="baseline")
    for method, result in results.items():
        ax.plot(t, result["agg_load"], linewidth=1.8, label=method)
    ax.set_title("Total Net Load")
    ax.set_ylabel("kW")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)

    # 2. Buy Price
    ax = axes[1]
    ax.step(t, market, where="post", linestyle="--", linewidth=1.5, label="market")
    for method, result in results.items():
        ax.step(t, result["price_buy"], where="post", linewidth=1.7, label=method)
    ax.set_title("Buy Price")
    ax.set_ylabel("EUR/kWh")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)

    # 3. Sell Price
    ax = axes[2]
    for method, result in results.items():
        ax.step(t, result["price_sell"], where="post", linewidth=1.7, label=method)
    ax.set_title("Sell Price")
    ax.set_ylabel("EUR/kWh")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)

    # 4. Metrics Table
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
        table_df[c] = table_df[c].map(lambda x, col=c: format_table_value(col, x))

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)
    ax.set_title("Metrics Table")

    # 5. Net Load Error vs SOS1
    ax = axes[4]
    if "sos1" in results:
        ref = results["sos1"]
        for method, result in results.items():
            if method == "sos1":
                continue

            err = result["agg_load"] - ref["agg_load"]
            ax.plot(t, err, linewidth=1.8, label=f"{method} - sos1")

    ax.axhline(0.0, linewidth=1.0, color="black")
    ax.set_title("Net Load Error vs SOS1")
    ax.set_ylabel("kW")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)

    # 6. Buy Price Error vs SOS1
    ax = axes[5]
    if "sos1" in results:
        ref = results["sos1"]
        for method, result in results.items():
            if method == "sos1":
                continue

            err = result["price_buy"] - ref["price_buy"]
            ax.step(t, err, where="post", linewidth=1.6, label=f"{method} - sos1")

    ax.axhline(0.0, linewidth=1.0, color="black")
    ax.set_title("Buy Price Error vs SOS1")
    ax.set_ylabel("EUR/kWh")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)

    # 7. Total Demand Response
    ax = axes[6]
    for method, result in results.items():
        dload = get_total_dload(result)
        if dload is not None:
            ax.plot(t, dload, linewidth=1.7, label=method)

    ax.axhline(0.0, linewidth=1.0, color="black")
    ax.set_title("Total Demand Response")
    ax.set_ylabel("kW")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)

    # 8. Total PV Curtailment
    ax = axes[7]
    for method, result in results.items():
        curt = get_total_curtailment(result)
        if curt is not None:
            ax.plot(t, curt, linewidth=1.7, label=method)

    ax.set_title("Total PV Curtailment")
    ax.set_ylabel("kW")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)

    # 9-12. Battery Net Power
    battery_buses = get_battery_buses(cfg, results)

    for k in range(4):
        ax = axes[8 + k]

        if k >= len(battery_buses):
            ax.axis("off")
            continue

        b = battery_buses[k]

        for method, result in results.items():
            sch = result["schedules"][b]
            bat_net = sch["ch"] - sch["dis"]
            ax.plot(t, bat_net, linewidth=1.7, label=method)

        ax.axhline(0.0, linewidth=1.0, color="black")
        ax.set_title(f"{b}: Battery Net Power")
        ax.set_ylabel("kW")
        ax.set_xlabel("time step")
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=8)

    fig.suptitle("SOS1 vs Big-M vs BIG-Hype-like Method Dashboard", fontsize=16)

    save_path = os.path.join(cfg.output_dir, "method_dashboard.png")
    fig.savefig(save_path, dpi=150)
    plt.show()

    return save_path


def main():
    cfg = StackelbergConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)

    market = load_price_data("price_data.csv", cfg.horizon)
    load, pv = make_profiles(cfg)
    feeder = feeder_data(cfg.active_buses)
    data = build_case(cfg, feeder, load, pv)
    baseline = get_baseline(load, pv)

    allowed = ("sos1", "bigm", "big_hype_fd")
    methods = [m for m in getattr(cfg, "compare_methods", allowed) if m in allowed]

    print("\n=== Method Setting ===")
    print(f"methods              : {methods}")
    print(f"horizon              : {cfg.horizon}")
    print(f"mip_gap              : {cfg.mip_gap}")
    print(f"time_limit_seconds   : {cfg.time_limit_seconds}")
    print(f"big_m_dual           : {cfg.big_m_dual}")
    print(f"big_hype_outer_iter  : {cfg.big_hype_outer_iter}")
    print(f"big_hype_step        : {cfg.big_hype_step}")
    print(f"big_hype_fd_eps      : {cfg.big_hype_fd_eps}")

    results = {}

    for method in methods:
        print(f"\n=== Running {method} ===")

        result = solve_method(cfg, data, market, method)

        if result is None:
            print(f"{method}: no feasible solution")
            continue

        results[method] = result

        gap_text = "N/A" if method == "big_hype_fd" else f"{result['mip_gap']:.6f}"

        print(
            f"{method}: "
            f"obj={result['obj_value']:.6f}, "
            f"peak={np.max(np.abs(result['agg_load'])):.6f}, "
            f"runtime={result['runtime_sec']:.2f}s, "
            f"gap={gap_text}"
        )

    if not results:
        print("No method solved.")
        return

    ref = results.get("sos1", None)
    rows = []

    for method, result in results.items():
        rows.append(get_metrics(cfg, baseline, result, ref=ref))

    metrics_df = pd.DataFrame(rows)

    metrics_csv = os.path.join(cfg.output_dir, "method_comparison_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    print("\n=== Comparison Metrics ===")
    print(metrics_df.to_string(index=False))

    if "sos1" in results:
        print("\n=== Error Relative to SOS1 ===")
        print(
            metrics_df[
                [
                    "method",
                    "max_netload_error_vs_sos1_kw",
                    "max_buy_price_error_vs_sos1",
                ]
            ].to_string(index=False)
        )

    fig_path = plot_dashboard(cfg, market, baseline, results, metrics_df)

    print("\nSaved:")
    print(metrics_csv)
    print(fig_path)


if __name__ == "__main__":
    main()