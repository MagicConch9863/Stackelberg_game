import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import StackelbergConfig, load_price_data, make_profiles
from build_base_network import feeder_data
from stackelberg_kkt_model import solve_stackelberg_mpec


def build_case(cfg, feeder, load, pv):
    buses = list(cfg.active_buses)

    return {
        "buses": buses,
        "net": feeder,
        "load": {b: load[k] for k, b in enumerate(buses)},
        "pv": {b: pv[k] for k, b in enumerate(buses)},
    }


def get_metrics(cfg, baseline, result):
    after = result["agg_load"]

    return {
        "method": result["method"],
        "status": result["status"],
        "runtime_sec": result["runtime_sec"],
        "mip_gap": result["mip_gap"],
        "objective": result["obj_value"],
        "peak_after_kw": float(np.max(np.abs(after))),
        "std_after_kw": float(np.std(after)),
        "energy_after_kwh": float(np.sum(after) * cfg.dt),
        "energy_diff_kwh": float((np.sum(after) - np.sum(baseline)) * cfg.dt),
    }


def main():
    cfg = StackelbergConfig()

    market = load_price_data("price_data.csv", cfg.horizon)
    load, pv = make_profiles(cfg)
    feeder = feeder_data(cfg.active_buses)

    data = build_case(cfg, feeder, load, pv)
    baseline = load.sum(axis=0) - pv.sum(axis=0)

    results = {}
    rows = []

    for method in cfg.compare_methods:
        print(f"\n=== Running {method} ===")

        result = solve_stackelberg_mpec(cfg, data, market, method=method)

        if result is None:
            print(f"{method}: no feasible solution")
            continue

        results[method] = result
        rows.append(get_metrics(cfg, baseline, result))

        print(
            f"{method}: "
            f"obj={result['obj_value']:.4f}, "
            f"peak={np.max(np.abs(result['agg_load'])):.4f}, "
            f"runtime={result['runtime_sec']:.2f}s"
        )

    if not results:
        print("No method solved.")
        return

    os.makedirs(cfg.output_dir, exist_ok=True)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(cfg.output_dir, "method_comparison_metrics.csv")
    df.to_csv(csv_path, index=False)

    t = np.arange(cfg.horizon)

    plt.figure(figsize=(10, 5))
    plt.plot(t, baseline, "--", linewidth=2.0, label="baseline")

    for method, result in results.items():
        plt.plot(t, result["agg_load"], linewidth=1.8, label=method)

    plt.xlabel("time step")
    plt.ylabel("kW")
    plt.title("Net Load Comparison: SOS1 vs Big-M vs Duality")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(cfg.output_dir, "method_comparison_netload.png")
    plt.savefig(fig_path, dpi=150)
    plt.show()

    print("\n=== Comparison ===")
    print(df.to_string(index=False))

    print("\nSaved:")
    print(csv_path)
    print(fig_path)


if __name__ == "__main__":
    main()