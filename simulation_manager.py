import numpy as np

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


def compute_baseline_by_bus(cfg, load, pv):
    buses = list(cfg.active_buses)
    return {b: load[k] - pv[k] for k, b in enumerate(buses)}


def compute_after_by_bus(result):
    return {
        b: result["schedules"][b]["buy"] - result["schedules"][b]["sell"]
        for b in result["schedules"]
    }


def compute_metrics(cfg, baseline, result):
    after = result["agg_load"]

    peak_before = float(np.max(np.abs(baseline)))
    peak_after = float(np.max(np.abs(after)))

    std_before = float(np.std(baseline))
    std_after = float(np.std(after))

    before_energy = float(np.sum(baseline) * cfg.dt)
    after_energy = float(np.sum(after) * cfg.dt)
    diff = after_energy - before_energy

    return {
        "method": result["method"],
        "status": result["status"],
        "runtime_sec": result["runtime_sec"],
        "mip_gap": result["mip_gap"],
        "objective": result["obj_value"],

        "peak_before_kw": peak_before,
        "peak_after_kw": peak_after,
        "peak_shaving_percent": 100.0 * (peak_before - peak_after) / max(peak_before, 1e-9),

        "std_before_kw": std_before,
        "std_after_kw": std_after,
        "std_reduction_percent": 100.0 * (std_before - std_after) / max(std_before, 1e-9),

        "net_energy_before_kwh": before_energy,
        "net_energy_after_kwh": after_energy,
        "net_energy_diff_kwh": diff,
        "net_energy_diff_percent": 100.0 * diff / max(abs(before_energy), 1e-9),
    }


def run_method(cfg, feeder, load, pv, market, method):
    data = build_case(cfg, feeder, load, pv)

    baseline_by_bus = compute_baseline_by_bus(cfg, load, pv)
    baseline = np.sum(list(baseline_by_bus.values()), axis=0)

    if method in ("sos1", "bigm", "duality"):
        result = solve_stackelberg_mpec(cfg, data, market, method=method)
    elif method == "big_hype_fd":
        result = solve_big_hype_fd(cfg, data, market)
    else:
        raise ValueError(f"Unknown method: {method}")

    if result is None:
        return None, baseline, None, baseline_by_bus, None

    after_by_bus = compute_after_by_bus(result)
    metrics = compute_metrics(cfg, baseline, result)

    return result, baseline, metrics, baseline_by_bus, after_by_bus