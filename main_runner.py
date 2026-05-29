import os
import numpy as np
import matplotlib.pyplot as plt

from data_loader import StackelbergConfig, load_price_data, make_profiles
from build_base_network import feeder_data
from simulation_manager import run_pipeline


def plot_all_in_one(cfg, market, baseline, result, metrics, ac, baseline_by_bus, after_by_bus):
    os.makedirs(cfg.output_dir, exist_ok=True)
    t = np.arange(cfg.horizon)

    fig, ax = plt.subplots(4, 3, figsize=(14, 10), constrained_layout=True)

    ax[0, 0].plot(t, baseline, "--", linewidth=1.8, label="baseline total")
    ax[0, 0].plot(t, result["agg_load"], linewidth=2.0, label="optimized total")
    ax[0, 0].set_title(
        f"Total Net Load | peak={metrics['peak_shaving_percent']:.2f}% "
        f"std={metrics['std_reduction_percent']:.2f}%"
    )
    ax[0, 0].set_ylabel("kW")
    ax[0, 0].grid(True, alpha=0.35)
    ax[0, 0].legend(fontsize=8)

    ax[0, 1].step(t, market, where="post", linewidth=1.5, label="market")
    ax[0, 1].step(t, result["price_buy"], where="post", linewidth=1.7, label="p_buy")
    ax[0, 1].step(t, result["price_sell"], where="post", linewidth=1.7, label="p_sell")
    ax[0, 1].set_title("Leader Price")
    ax[0, 1].set_ylabel("EUR/kWh")
    ax[0, 1].grid(True, alpha=0.35)
    ax[0, 1].legend(fontsize=8)

    energy_vals = [metrics["net_energy_before_kwh"], metrics["net_energy_after_kwh"]]
    ax[0, 2].bar(["before", "after"], energy_vals, width=0.5)
    ax[0, 2].set_title(f"Net Energy | diff={metrics['net_energy_diff_kwh']:.4f} kWh")
    ax[0, 2].set_ylabel("kWh")
    ax[0, 2].grid(True, axis="y", alpha=0.35)

    buses = list(cfg.active_buses)
    bus_axes = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]

    for b, pos in zip(buses, bus_axes):
        r, c = pos
        ax[r, c].plot(t, baseline_by_bus[b], "--", linewidth=1.5, label=f"{b} baseline")
        ax[r, c].plot(t, after_by_bus[b], linewidth=1.8, label=f"{b} optimized")
        ax[r, c].set_title(f"{b}: Before vs After")
        ax[r, c].set_ylabel("kW")
        ax[r, c].grid(True, alpha=0.35)
        ax[r, c].legend(fontsize=8)

    ax[2, 2].set_axis_off()

    for k, b in enumerate(result["bus_order"]):
        ax[3, 0].plot(t, result["v_lin"][k], linewidth=1.5, label=b)
    ax[3, 0].set_title("LinDistFlow Voltage")
    ax[3, 0].set_ylabel("p.u.")
    ax[3, 0].set_xlabel("time step")
    ax[3, 0].grid(True, alpha=0.35)
    ax[3, 0].legend(fontsize=7, ncol=2)

    if ac and ac.get("available", False):
        for k, b in enumerate(result["bus_order"]):
            ax[3, 1].plot(t, ac["v_ac"][:, k], linewidth=1.5, label=b)
        ax[3, 1].set_title(f"Pandapower AC Voltage | err={ac['v_error_max_pu']:.4f} p.u.")
        ax[3, 1].set_ylabel("p.u.")
        ax[3, 1].set_xlabel("time step")
        ax[3, 1].grid(True, alpha=0.35)
        ax[3, 1].legend(fontsize=7, ncol=2)

        ax[3, 2].plot(t, ac["line_loading_max"], linewidth=1.8)
        ax[3, 2].set_title(f"AC Line Loading | max={ac['line_loading_max_percent']:.2f}%")
        ax[3, 2].set_ylabel("%")
        ax[3, 2].set_xlabel("time step")
        ax[3, 2].grid(True, alpha=0.35)
    else:
        ax[3, 1].text(0.5, 0.5, "AC validation unavailable", ha="center", va="center")
        ax[3, 1].set_axis_off()
        ax[3, 2].set_axis_off()

    fig.suptitle(
        f"5-Bus Stackelberg Game | SimBench {cfg.simbench_code}",
        fontsize=13,
    )

    save_path = os.path.join(cfg.output_dir, "stackelberg_summary_5bus.png")
    fig.savefig(save_path, dpi=150)
    plt.show()


def main():
    cfg = StackelbergConfig()

    market = load_price_data("price_data.csv", cfg.horizon)
    load, pv = make_profiles(cfg)
    feeder = feeder_data(cfg.active_buses)

    print(f"SimBench case      : {cfg.simbench_code}")
    print(f"Gurobi time limit  : {cfg.time_limit_seconds} seconds")
    print(f"Active buses       : {cfg.active_buses}")
    print(f"Battery buses      : {cfg.battery_buses}")

    result, baseline, metrics, ac, baseline_by_bus, after_by_bus = run_pipeline(
        cfg, feeder, load, pv, market
    )

    if result is None:
        print("求解失败：没有可用解。")
        print("如果生成了 stackelberg_infeasible.ilp，请检查冲突约束。")
        return

    print("\n=== Solver ===")
    print(f"status                         : {result['status']}")
    print(f"runtime_sec                    : {result['runtime_sec']:.2f}")
    print(f"mip_gap                        : {result['mip_gap']:.6f}")

    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k:30s}: {v:.6f}")

    print("\n=== AC validation ===")
    if ac and ac.get("available", False):
        for k, v in ac.items():
            if k not in ("v_ac", "line_loading_max"):
                print(f"{k:30s}: {v}")
    else:
        print(ac)

    plot_all_in_one(cfg, market, baseline, result, metrics, ac, baseline_by_bus, after_by_bus)

    print(f"\n图片已保存到: {cfg.output_dir}")


if __name__ == "__main__":
    main()