"""Replot existing C-A CSV results as one readable overview per scenario.

Copy this file into ``stackelberg_scenario_analysis/phase1_der_only`` and run:

    python ca_plot_overviews.py

It does not rerun S1--S4.  It reads ``csv_results/s1_timeseries.csv`` through
``s4_timeseries.csv`` and writes exactly four figures to ``figures``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIOS = {
    "s1": "C-A S1: predefined operation",
    "s2": "C-A S2: fixed-tariff response",
    "s3": "C-A S3: Stackelberg coordination",
    "s4": "C-A S4: centralized benchmark",
}
BUS_COLORS = {1: "#5B8FF9", 2: "#F6BD16", 3: "#61DDAA", 4: "#E8684A"}
EPS = 1e-8


def first_column(frame: pd.DataFrame, *names: str) -> str | None:
    return next((name for name in names if name in frame.columns), None)


def values(frame: pd.DataFrame, *names: str) -> np.ndarray | None:
    column = first_column(frame, *names)
    return None if column is None else pd.to_numeric(frame[column], errors="coerce").fillna(0).to_numpy(float)


def nonzero(series: np.ndarray | None) -> bool:
    return series is not None and np.any(np.abs(series) > EPS)


def bus_series(frame: pd.DataFrame, bus: int, field: str) -> np.ndarray | None:
    # Current four-follower schema, plus a few earlier naming variants.
    key = f"bus_{bus}"
    aliases = {
        "net": (f"{key}_net_kw", f"bus{bus}_net_kw"),
        "import": (f"{key}_import_kw", f"bus{bus}_import_kw"),
        "export": (f"{key}_export_kw", f"bus{bus}_export_kw"),
        "bess": (f"{key}_bess_kw", f"bus{bus}_bess_kw"),
        "ev": (f"{key}_ev_kw", f"bus{bus}_ev_kw"),
        "load": (f"{key}_load_kw", f"bus{bus}_load_kw"),
        "pv": (f"{key}_pv_kw", f"bus{bus}_pv_kw"),
        "bess_energy": (f"{key}_bess_energy_kwh", f"bus{bus}_bess_energy_kwh"),
        "ev_energy": (f"{key}_ev_energy_kwh", f"bus{bus}_ev_energy_kwh"),
    }
    return values(frame, *aliases[field])


def aggregate_net(frame: pd.DataFrame) -> np.ndarray:
    direct = values(frame, "aggregate_net_kw", "net_exchange_kw", "aggregate_kw")
    if direct is not None:
        return direct
    nets = [bus_series(frame, bus, "net") for bus in range(1, 5)]
    present = [series for series in nets if series is not None]
    if not present:
        raise KeyError("No aggregate or per-Bus net-exchange column was found")
    return np.sum(present, axis=0)


def step(ax, time, series, *, label, color, linestyle="-"):
    if series is not None:
        ax.step(time, series, where="post", label=label, color=color,
                linestyle=linestyle, linewidth=1.15)


def style_axis(ax, ylabel="kW"):
    ax.axhline(0, color="#9A9A9A", linewidth=.65)
    ax.grid(alpha=.22, linewidth=.6)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel(ylabel)


def draw_bus(ax, frame: pd.DataFrame, time: np.ndarray, bus: int):
    color = BUS_COLORS[bus]
    net = bus_series(frame, bus, "net")
    if net is None:
        ax.text(.5, .5, f"Bus {bus}: no per-Bus columns in this CSV",
                ha="center", va="center", transform=ax.transAxes, color="#666666")
        ax.set_title(f"Bus {bus}")
        style_axis(ax)
        return

    load = bus_series(frame, bus, "load")
    pv = bus_series(frame, bus, "pv")
    bess = bus_series(frame, bus, "bess")
    ev = bus_series(frame, bus, "ev")
    imp = bus_series(frame, bus, "import")
    exp = bus_series(frame, bus, "export")

    if nonzero(load):
        ax.plot(time, load, color="#888888", linestyle="--", linewidth=1.0, label="Load")
    if nonzero(pv):
        ax.plot(time, -pv, color="#7CB342", linestyle=":", linewidth=1.0, label="-PV")
    ax.plot(time, net, color=color, linewidth=1.45, label="Net exchange")
    step(ax, time, bess, label="BESS power", color="#8E6CBB")
    if nonzero(ev):
        step(ax, time, ev, label="EV power", color="#42B7C7")
    if imp is not None and exp is not None:
        # Import/export are intentionally faint: they explain the sign without
        # hiding net/device curves.
        ax.plot(time, imp, color="#E5A93D", linewidth=.75, alpha=.65, label="Import")
        ax.plot(time, -exp, color="#D982A6", linewidth=.75, alpha=.65, label="-Export")
    ax.set_title(f"Bus {bus}")
    style_axis(ax)
    ax.legend(fontsize=7, ncol=3, loc="best")


def plot_one(frame: pd.DataFrame, scenario: str, title: str, destination: Path):
    time = values(frame, "time_h", "time")
    if time is None:
        time = np.arange(len(frame)) * 0.25
    net = aggregate_net(frame)

    fig, axes = plt.subplots(4, 2, figsize=(15, 16), sharex=True)
    ax = axes[0, 0]
    ax.plot(time, net, color="#4C78A8", linewidth=1.55, label="Aggregate net exchange")
    ax.set_title("Aggregate net exchange")
    style_axis(ax)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    pa = values(frame, "pa_eur_per_kwh", "pa")
    pb = values(frame, "pb_eur_per_kwh", "pb")
    total = values(frame, "total_buy_price_eur_per_kwh", "pa_plus_pb_eur_per_kwh")
    if pa is not None or pb is not None or total is not None:
        step(ax, time, pa, label="PA", color="#5B8FF9")
        step(ax, time, pb, label="PB", color="#F6BD16")
        if total is None and pa is not None and pb is not None:
            total = pa + pb
        step(ax, time, total, label="PA + PB", color="#61DDAA")
        ax.set_title("Common prices")
        style_axis(ax, "EUR/kWh")
        ax.legend(fontsize=8, ncol=3)
    else:
        bess_parts = [s for b in range(1, 5)
                      if (s := bus_series(frame, b, "bess")) is not None]
        aggregate_bess = values(frame, "bess_kw", "aggregate_bess_kw")
        total_bess = (np.sum(bess_parts, axis=0) if bess_parts else
                      aggregate_bess if aggregate_bess is not None else np.zeros(len(frame)))
        ev_parts = [s for b in range(1, 5) if (s := bus_series(frame, b, "ev")) is not None]
        aggregate_ev = values(frame, "ev_kw", "aggregate_ev_kw")
        total_ev = (np.sum(ev_parts, axis=0) if ev_parts else
                    aggregate_ev if aggregate_ev is not None else np.zeros(len(frame)))
        step(ax, time, total_bess, label="Total BESS", color="#8E6CBB")
        step(ax, time, total_ev, label="Total EV", color="#42B7C7")
        ax.set_title("Aggregate device power (+ charge, - discharge)")
        style_axis(ax)
        ax.legend(fontsize=8)

    for bus, ax in enumerate(axes[1:3].flat, start=1):
        draw_bus(ax, frame, time, bus)

    ax = axes[3, 0]
    found = False
    for bus in range(1, 5):
        energy = bus_series(frame, bus, "bess_energy")
        if energy is not None:
            ax.plot(time, energy, color=BUS_COLORS[bus], linewidth=1.25, label=f"Bus {bus}")
            found = True
    if not found:
        energy = values(frame, "bess_energy_kwh")
        if energy is not None:
            ax.plot(time, energy, color="#8E6CBB", label="BESS")
            found = True
    ax.set_title("BESS stored energy")
    style_axis(ax, "kWh")
    if found:
        ax.legend(fontsize=8, ncol=4)

    ax = axes[3, 1]
    found = False
    for bus in range(1, 5):
        energy = bus_series(frame, bus, "ev_energy")
        if energy is not None:
            ax.plot(time, energy, color=BUS_COLORS[bus], linewidth=1.25, label=f"Bus {bus}")
            found = True
    if not found:
        energy = values(frame, "ev_energy_kwh")
        if energy is not None:
            ax.plot(time, energy, color="#42B7C7", label="EV")
            found = True
    ax.set_title("EV stored energy")
    style_axis(ax, "kWh")
    if found:
        ax.legend(fontsize=8, ncol=4)

    fig.suptitle(title, fontsize=15)
    fig.text(.5, .008, "Sign convention: net < 0 = export; BESS/EV power < 0 = discharge.",
             ha="center", fontsize=8, color="#666666")
    fig.tight_layout(rect=(0, .018, 1, .975))
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parent
    csv_folder = root / "csv_results"
    figure_folder = root / "figures"
    figure_folder.mkdir(parents=True, exist_ok=True)

    missing = []
    for scenario, title in SCENARIOS.items():
        source = csv_folder / f"{scenario}_timeseries.csv"
        if not source.exists():
            missing.append(source.name)
            continue
        frame = pd.read_csv(source)
        if len(frame) == 0:
            raise ValueError(f"{source} is empty")
        destination = figure_folder / f"{scenario}_overview.png"
        plot_one(frame, scenario, title, destination)
        print(f"Saved: {destination}")

    if missing:
        raise FileNotFoundError(
            "Missing scenario CSV files: " + ", ".join(missing)
            + ". Run only the missing scenarios first."
        )


if __name__ == "__main__":
    main()
