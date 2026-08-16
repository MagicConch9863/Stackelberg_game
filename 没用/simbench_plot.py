# -*- coding: utf-8 -*-

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import simbench as sb


# ============================================================
# 1. User settings
# ============================================================

# 可以改成：
# "1-LV-urban6--0-sw"
# "1-LV-urban6--0-no_sw"
# 或其他 SimBench code
SB_CODE = "1-LV-urban6--0-sw"

# 默认自动选择前 16 个“有 load 的 bus”
N_BUSES = 16

# 如果你想手动指定 SimBench bus index，就在这里填，例如：
# SELECT_BUS_INDICES = [12, 13, 14, 15]
SELECT_BUS_INDICES = None

# 如果你想按 SimBench bus name 指定，就在这里填，例如：
# SELECT_BUS_NAMES = ["Bus 1", "Bus 2"]
# 注意：这里必须和 net.bus["name"] 里的真实名字一致
SELECT_BUS_NAMES = None

# 第几天，0 表示全年 profile 的第 1 天
DAY_INDEX = 0

# SimBench 常规是 15min 分辨率，所以一天 96 个点
POINTS_PER_DAY = 96

# 输出单位："kW" 或 "MW"
POWER_UNIT = "kW"

# 是否把标题显示成 BUS1, BUS2, ..., BUS16
# True:  图里显示 BUS1，但 mapping 表里会保留真实 SimBench bus index/name
# False: 图里直接显示 SimBench bus index/name
RENAME_TO_BUS1_BUS16 = True

# 输出文件夹
OUT_DIR = Path("simbench_bus_24h_output")


# ============================================================
# 2. Helper functions
# ============================================================

def safe_filename(text: str) -> str:
    """Make a safe filename from SimBench code."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def get_profile_column(df: pd.DataFrame, load_idx):
    """
    SimBench absolute profile columns are usually load indices.
    Depending on package versions, they may be int or str.
    This helper makes the matching robust.
    """
    candidates = [load_idx, str(load_idx)]

    for c in candidates:
        if c in df.columns:
            return c

    raise KeyError(
        f"Cannot find load index {load_idx} in profile columns. "
        f"Example columns: {list(df.columns[:10])}"
    )


def choose_buses(net, n_buses=16, bus_indices=None, bus_names=None):
    """
    Choose buses for plotting.

    Priority:
    1. If bus_indices is given, use these SimBench bus indices.
    2. If bus_names is given, find matching names in net.bus['name'].
    3. Otherwise, select first n buses that actually have loads.
    """
    if bus_indices is not None:
        selected = list(bus_indices)
        missing = [b for b in selected if b not in net.bus.index]
        if missing:
            raise ValueError(f"These bus indices are not in net.bus: {missing}")
        return selected

    if bus_names is not None:
        if "name" not in net.bus.columns:
            raise ValueError("net.bus has no column named 'name'.")

        selected = []
        for name in bus_names:
            matches = net.bus.index[net.bus["name"].astype(str) == str(name)].tolist()
            if not matches:
                raise ValueError(
                    f"Cannot find bus name: {name}\n"
                    f"Available examples:\n{net.bus[['name']].head(30)}"
                )
            selected.append(matches[0])
        return selected

    # Default: choose first n buses that have at least one load
    load_buses = sorted(net.load["bus"].dropna().astype(int).unique().tolist())

    if len(load_buses) < n_buses:
        raise ValueError(
            f"Only {len(load_buses)} buses with loads are available, "
            f"but n_buses={n_buses} was requested."
        )

    return load_buses[:n_buses]


def build_bus_load_profiles(net, selected_buses, load_p_mw):
    """
    Aggregate load profiles from load level to bus level.

    Result:
    DataFrame with shape:
        time steps × selected buses
    """
    bus_profiles = {}
    mapping_rows = []

    for i, bus_idx in enumerate(selected_buses, start=1):
        load_indices = net.load.index[net.load["bus"].astype(int) == int(bus_idx)].tolist()

        if len(load_indices) == 0:
            bus_series = pd.Series(0.0, index=load_p_mw.index)
        else:
            cols = [get_profile_column(load_p_mw, load_idx) for load_idx in load_indices]
            bus_series = load_p_mw[cols].sum(axis=1)

        bus_name = str(net.bus.at[bus_idx, "name"]) if "name" in net.bus.columns else ""

        if RENAME_TO_BUS1_BUS16:
            col_name = f"BUS{i}"
        else:
            col_name = f"bus_{bus_idx}_{bus_name}"

        bus_profiles[col_name] = bus_series

        mapping_rows.append({
            "plot_label": col_name,
            "simbench_bus_index": bus_idx,
            "simbench_bus_name": bus_name,
            "load_indices_at_this_bus": load_indices,
            "number_of_loads": len(load_indices),
        })

    bus_profiles = pd.DataFrame(bus_profiles)
    mapping = pd.DataFrame(mapping_rows)

    return bus_profiles, mapping


def slice_24h(df, day_index=0, points_per_day=96):
    """
    Slice one 24h day from full-year SimBench profiles.
    day_index=0 means the first day.
    """
    start = day_index * points_per_day
    end = start + points_per_day

    if end > len(df):
        raise ValueError(
            f"Requested day_index={day_index}, but profile length is only {len(df)}. "
            f"With {points_per_day} points/day, max day_index is {len(df)//points_per_day - 1}."
        )

    return df.iloc[start:end].copy()


def convert_power_unit(df_mw, unit="kW"):
    """Convert MW to selected unit."""
    unit = unit.lower()

    if unit == "mw":
        return df_mw.copy(), "MW"

    if unit == "kw":
        return df_mw * 1000.0, "kW"

    raise ValueError("POWER_UNIT must be either 'MW' or 'kW'.")


def plot_4x4_profiles(df_24h, unit, title, output_path):
    """
    Plot each selected bus in a 4×4 subplot figure.
    """
    n = df_24h.shape[1]
    n_rows = 4
    n_cols = 4

    if n > n_rows * n_cols:
        raise ValueError("This plot function supports up to 16 buses for a 4×4 figure.")

    # x-axis: hour of day
    x_hours = np.arange(len(df_24h)) * 24.0 / len(df_24h)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(18, 12),
        sharex=True,
        constrained_layout=True
    )

    axes_flat = axes.flatten()

    for ax, col in zip(axes_flat, df_24h.columns):
        ax.plot(x_hours, df_24h[col].values, linewidth=1.6)
        ax.set_title(col, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 24)

    # Hide unused axes if fewer than 16 buses
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=16)

    for ax in axes[-1, :]:
        ax.set_xlabel("Hour of day")

    for ax in axes[:, 0]:
        ax.set_ylabel(f"Active load [{unit}]")

    fig.savefig(output_path, dpi=300)
    plt.show()


# ============================================================
# 3. Main workflow
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading SimBench grid: {SB_CODE}")
    net = sb.get_simbench_net(SB_CODE)

    print("\nBasic grid information:")
    print(f"Number of buses: {len(net.bus)}")
    print(f"Number of loads: {len(net.load)}")

    # Optional: save available bus table for checking real SimBench names
    available_bus_table = net.bus.copy()
    available_bus_table.to_csv(
        OUT_DIR / f"{safe_filename(SB_CODE)}_available_buses.csv",
        index=True
    )

    # Generate absolute active load profiles in MW
    # This uses the actual net.load.p_mw and the assigned SimBench profile of each load.
    load_p_mw = sb.get_absolute_profiles_from_relative_profiles(
        net,
        element="load",
        multiplying_column="p_mw",
        profile_column="profile",
        profile_suffix="_pload",
        time_as_index=True
    )

    print("\nAbsolute load profile table:")
    print(f"Shape: {load_p_mw.shape}")
    print(f"Rows = time steps, columns = load indices")

    selected_buses = choose_buses(
        net,
        n_buses=N_BUSES,
        bus_indices=SELECT_BUS_INDICES,
        bus_names=SELECT_BUS_NAMES
    )

    print("\nSelected SimBench bus indices:")
    print(selected_buses)

    # Aggregate load profiles to bus level
    bus_profiles_mw, mapping = build_bus_load_profiles(
        net=net,
        selected_buses=selected_buses,
        load_p_mw=load_p_mw
    )

    # Slice one 24h period
    bus_24h_mw = slice_24h(
        bus_profiles_mw,
        day_index=DAY_INDEX,
        points_per_day=POINTS_PER_DAY
    )

    # Convert MW to kW if needed
    bus_24h, unit = convert_power_unit(bus_24h_mw, POWER_UNIT)

    # Save CSV files
    code_safe = safe_filename(SB_CODE)

    csv_path = OUT_DIR / f"{code_safe}_day{DAY_INDEX + 1}_selected_bus_24h_load_{unit}.csv"
    mapping_path = OUT_DIR / f"{code_safe}_selected_bus_mapping.csv"
    fig_path = OUT_DIR / f"{code_safe}_day{DAY_INDEX + 1}_selected_bus_24h_load_4x4.png"

    bus_24h.to_csv(csv_path, index=False)
    mapping.to_csv(mapping_path, index=False)

    print("\nSelected bus mapping:")
    print(mapping)

    print("\nSaved files:")
    print(f"24h load CSV: {csv_path}")
    print(f"Bus mapping CSV: {mapping_path}")
    print(f"Figure: {fig_path}")

    # Plot 4×4 figure
    plot_title = (
        f"24h active load profiles from SimBench {SB_CODE}\n"
        f"Day index = {DAY_INDEX}, unit = {unit}"
    )

    plot_4x4_profiles(
        df_24h=bus_24h,
        unit=unit,
        title=plot_title,
        output_path=fig_path
    )


if __name__ == "__main__":
    main()