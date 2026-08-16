# -*- coding: utf-8 -*-
"""
Phase 1: selected-bus SimBench load-profile sanity check

This script does NOT run a full SimBench power-flow model.
It only uses SimBench load time series as baseline data.

Baseline:
    Sum of SimBench load profiles connected to ACTIVE_LOAD_BUSES only.

Scenario:
    1 EV
    2 PV
    2 BESS
"""

from pathlib import Path
import sys
import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gurobipy as gp
from gurobipy import GRB


# ============================================================
# 0. Settings
# ============================================================

FEEDER_FOLDER = "1_LV_urban_0_no_sw"
SIMBENCH_CODE = "1-LV-urban6--0-no_sw"

DAY = "2019-07-01"
T = 96
DT = 0.25

PRICE_FILE = "price_data.csv"
PV_FILE = "pv_2019-07-01_local.csv"

# Only these buses are used to build the load baseline.
ACTIVE_LOAD_BUSES = ["Bus R1", "Bus R2", "Bus R3", "Bus R4", "Bus R5"]

# Phase 1 devices
EV_BUS = "Bus R2"
PV_BUSES = ["Bus R4", "Bus R5"]
BESS_BUSES = ["Bus R1", "Bus R3"]

PV_SCALE_TOTAL = 15.0

PB_MIN = 0.00
PB_MAX = 0.12
PB_BLOCKS = 4

EV_P_MAX = 7.2
EV_E_REQ = 18.0

BESS_P_MAX = 5.0
BESS_E_MIN = 1.0
BESS_E_MAX = 10.0
BESS_E_INIT = 5.0
BESS_ETA_CH = 0.95
BESS_ETA_DIS = 0.95

GRID_MIN_UB = 20.0
GRID_MARGIN = 10.0

EPS = 1e-5


# ============================================================
# 1. Paths
# ============================================================

def get_paths():
    here = Path(__file__).resolve().parent

    if here.name == FEEDER_FOLDER:
        feeder_dir = here
        project_root = here.parent
    else:
        project_root = here
        feeder_dir = project_root / FEEDER_FOLDER

    out_dir = feeder_dir / "phase1_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print(f"[path] project_root = {project_root}")
    print(f"[path] feeder_dir   = {feeder_dir}")
    print(f"[path] out_dir      = {out_dir}")

    return project_root, feeder_dir, out_dir


# ============================================================
# 2. General utilities
# ============================================================

def norm_text(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def bus_number(x):
    s = str(x).lower()

    m = re.search(r"bus\s*r?\s*(\d+)", s)
    if m:
        return int(m.group(1))

    m = re.search(r"\br\s*(\d+)", s)
    if m:
        return int(m.group(1))

    return None


def parse_time(x):
    s = pd.Series(x).astype(str).str.strip()

    for fmt in [
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
    ]:
        y = pd.to_datetime(s, format=fmt, errors="coerce")
        if y.notna().mean() > 0.8:
            return y

    return pd.to_datetime(s, errors="coerce")


def find_time_col(df):
    for c in df.columns:
        name = str(c).strip().lower()
        if name in ["time", "datetime", "date", "timestamp", "local_time"]:
            return c

    for c in df.columns:
        name = str(c).strip().lower()
        if "time" in name or "date" in name:
            return c

    return None


def select_day_from_csv(df, day=DAY):
    df = df.copy()
    time_col = find_time_col(df)

    if time_col is None:
        return df

    df[time_col] = parse_time(df[time_col])
    df = df.dropna(subset=[time_col]).sort_values(time_col)

    picked = df[df[time_col].dt.strftime("%Y-%m-%d") == day]
    return picked if len(picked) > 0 else df


def select_day_from_profile(df, day=DAY):
    df = df.copy()
    target = pd.Timestamp(day)

    time_col = find_time_col(df)

    if time_col is not None:
        t = parse_time(df[time_col])
        tmp = df.loc[t.notna()].copy()
        t = t.loc[t.notna()]

        picked = tmp[(t.dt.month == target.month) & (t.dt.day == target.day)]
        if len(picked) > 0:
            return picked.drop(columns=[time_col], errors="ignore")

    if isinstance(df.index, pd.DatetimeIndex):
        picked = df[(df.index.month == target.month) & (df.index.day == target.day)]
        if len(picked) > 0:
            return picked

    if len(df) >= 365 * T:
        start = (target.dayofyear - 1) * T
        end = start + T
        if end <= len(df):
            return df.iloc[start:end]

    return df


def numeric_values(s):
    x = pd.to_numeric(pd.Series(s), errors="coerce").to_numpy(dtype=float)
    return x[np.isfinite(x)]


def to_96(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        raise ValueError("empty time series")

    if len(x) == T:
        return x.copy()

    if T % len(x) == 0:
        return np.repeat(x, T // len(x))

    if len(x) > T:
        return x[:T].copy()

    old = np.linspace(0, 1, len(x))
    new = np.linspace(0, 1, T)
    return np.interp(new, old, x)


def maybe_mw_to_kw(x, name):
    x = np.asarray(x, dtype=float)
    peak = float(np.nanmax(np.abs(x)))

    if 0 < peak < 2.0:
        print(f"[scale] {name}: MW -> kW")
        return x * 1000.0

    return x


def find_csv(project_root, feeder_dir, filename, keywords):
    candidates = [feeder_dir / filename, project_root / filename]

    for folder in [feeder_dir, project_root]:
        if not folder.exists():
            continue

        for f in folder.glob("*.csv"):
            low = f.name.lower()
            if any(k in low for k in keywords):
                candidates.append(f)

    for f in candidates:
        if f.exists():
            return f

    return None


# ============================================================
# 3. SimBench selected-bus baseline
# ============================================================

def match_active_bus_indices(net):
    if not hasattr(net, "bus") or "name" not in net.bus.columns:
        raise RuntimeError("net.bus.name is missing; cannot match selected buses.")

    target_nums = {bus_number(x) for x in ACTIVE_LOAD_BUSES}
    target_nums.discard(None)

    matched = []

    for idx, name in net.bus["name"].astype(str).items():
        n = bus_number(name)
        if n in target_nums:
            matched.append(idx)

    matched = list(dict.fromkeys(matched))

    print("[select] active buses:")
    for idx in matched:
        print(f"         bus index {idx}: {net.bus.at[idx, 'name']}")

    if not matched:
        raise RuntimeError(
            "No active buses matched. Check ACTIVE_LOAD_BUSES and SimBench bus names."
        )

    return matched


def select_active_load_indices(net, active_bus_idx):
    if not hasattr(net, "load") or "bus" not in net.load.columns:
        raise RuntimeError("net.load.bus is missing; cannot select load elements.")

    load_idx = net.load.index[net.load["bus"].isin(active_bus_idx)].tolist()
    load_idx = list(dict.fromkeys(load_idx))

    print("[select] active load elements:")
    for idx in load_idx:
        bus = net.load.at[idx, "bus"]
        bus_name = net.bus.at[bus, "name"] if "name" in net.bus.columns else str(bus)
        load_name = net.load.at[idx, "name"] if "name" in net.load.columns else str(idx)
        p_mw = net.load.at[idx, "p_mw"] if "p_mw" in net.load.columns else np.nan
        print(f"         load index {idx}: {load_name} | bus={bus_name} | p_mw={p_mw}")

    if not load_idx:
        raise RuntimeError(
            "The selected buses have no connected load elements in net.load."
        )

    if len(load_idx) == len(net.load):
        raise RuntimeError(
            "Selected loads equal all feeder loads. This violates selected-bus baseline."
        )

    return load_idx


def find_abs_load_profile_from_dict(abs_values):
    for key, val in abs_values.items():
        if not isinstance(val, pd.DataFrame):
            continue

        if isinstance(key, tuple):
            key_txt = " ".join(map(str, key)).lower()
        else:
            key_txt = str(key).lower()

        if "load" in key_txt and ("p_mw" in key_txt or "pload" in key_txt or ".p" in key_txt):
            return val

    return None


def get_absolute_load_profiles(sb, net):
    fn = getattr(sb, "get_absolute_profiles_from_relative_profiles", None)

    if fn is not None:
        for kwargs in [
            {"time_as_index": True},
            {},
        ]:
            try:
                df = fn(net, "load", "p_mw", **kwargs)
                print("[data] load profile source = get_absolute_profiles_from_relative_profiles")
                return df
            except Exception:
                pass

    fn = getattr(sb, "get_absolute_values", None)

    if fn is not None:
        for call in [
            lambda: fn(net, True),
            lambda: fn(net, profiles_instead_of_study_cases=True),
        ]:
            try:
                abs_values = call()
                df = find_abs_load_profile_from_dict(abs_values)
                if df is not None:
                    print("[data] load profile source = get_absolute_values")
                    return df
            except Exception:
                pass

    raise RuntimeError(
        "Could not get SimBench absolute load profiles. "
        "Your simbench version should provide either "
        "get_absolute_profiles_from_relative_profiles() or get_absolute_values()."
    )


def column_matches_load(col, load_idx, load_names):
    idx_str = {str(i) for i in load_idx}
    name_norm = {norm_text(x) for x in load_names}

    if isinstance(col, tuple):
        parts = [str(x) for x in col]
    else:
        parts = [str(col)]

    for p in parts:
        if p in idx_str:
            return True

        if norm_text(p) in name_norm:
            return True

    return False


def sum_selected_profiles(abs_load_df, net, selected_load_idx):
    df = select_day_from_profile(abs_load_df)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")

    if df.empty:
        raise RuntimeError("Absolute load profile is empty after numeric conversion.")

    selected_names = []
    if "name" in net.load.columns:
        selected_names = net.load.loc[selected_load_idx, "name"].astype(str).tolist()

    selected_cols = [
        c for c in df.columns
        if column_matches_load(c, selected_load_idx, selected_names)
    ]

    if selected_cols:
        selected_df = df[selected_cols]
        print("[select] profile columns matched by load index/name.")
    elif df.shape[1] == len(net.load.index):
        all_load_idx = list(net.load.index)
        pos = [all_load_idx.index(i) for i in selected_load_idx if i in all_load_idx]
        selected_df = df.iloc[:, pos]
        print("[select] profile columns matched by net.load position.")
    else:
        raise RuntimeError(
            "Could not match selected loads to profile columns. "
            "Check SimBench absolute load profile column names."
        )

    selected_kw = maybe_mw_to_kw(
        to_96(selected_df.sum(axis=1).to_numpy(dtype=float)),
        "selected-bus SimBench load profile"
    )

    full_kw = maybe_mw_to_kw(
        to_96(df.sum(axis=1).to_numpy(dtype=float)),
        "full-feeder SimBench reference profile"
    )

    print(f"[check] selected load elements = {len(selected_load_idx)}")
    print(f"[check] selected profile cols  = {selected_df.shape[1]}")
    print(f"[check] full profile cols      = {df.shape[1]}")
    print(f"[check] selected peak         = {selected_kw.max():.3f} kW")
    print(f"[check] full feeder peak      = {full_kw.max():.3f} kW")
    print(f"[check] selected/full ratio   = {selected_kw.max() / max(full_kw.max(), 1e-9):.3f}")

    return np.maximum(selected_kw, 0.0), np.maximum(full_kw, 0.0)


def load_selected_simbench_baseline():
    import simbench as sb

    net = sb.get_simbench_net(SIMBENCH_CODE)

    active_bus_idx = match_active_bus_indices(net)
    selected_load_idx = select_active_load_indices(net, active_bus_idx)

    abs_load_df = get_absolute_load_profiles(sb, net)
    baseline_kw, full_feeder_kw = sum_selected_profiles(abs_load_df, net, selected_load_idx)

    return baseline_kw, full_feeder_kw, selected_load_idx


# ============================================================
# 4. Price and PV
# ============================================================

def load_price(project_root, feeder_dir):
    path = find_csv(project_root, feeder_dir, PRICE_FILE, ["price", "pa"])

    if path is None:
        raise FileNotFoundError("Cannot find price_data.csv")

    df = pd.read_csv(path, comment="#")
    df = select_day_from_csv(df)

    time_col = find_time_col(df)
    numeric_cols = []

    for c in df.columns:
        name = str(c).lower()
        if c == time_col or name in ["hour", "step", "index"]:
            continue
        if len(numeric_values(df[c])) > 0:
            numeric_cols.append(c)

    if not numeric_cols:
        raise ValueError(f"No numeric price column found in {path}")

    col = numeric_cols[0]
    pa = to_96(numeric_values(df[col]))

    name = str(col).lower()
    med = float(np.nanmedian(pa))

    if "ct" in name:
        pa = pa / 100.0
    elif "mwh" in name or med > 5:
        pa = pa / 1000.0

    print(f"[data] PA loaded from {path.name}, column = {col}")
    return pa


def load_pv(project_root, feeder_dir):
    path = find_csv(project_root, feeder_dir, PV_FILE, ["pv", "solar", "ninja"])

    if path is None:
        print("[warn] PV csv not found. Use synthetic PV curve.")
        h = np.arange(T) * DT
        raw = np.maximum(0, np.sin((h - 5.5) / 13.5 * np.pi)) ** 1.8
        raw = raw / max(raw.max(), 1e-9) * 0.704
    else:
        df = pd.read_csv(path, comment="#")
        df = select_day_from_csv(df)

        pv_cols = [c for c in df.columns if "electricity" in str(c).lower()]
        col = pv_cols[0] if pv_cols else None

        if col is None:
            for c in df.columns:
                if len(numeric_values(df[c])) > 0:
                    col = c
                    break

        if col is None:
            raise ValueError(f"No PV column found in {path}")

        raw = to_96(numeric_values(df[col]))
        print(f"[data] PV loaded from {path.name}, column = {col}")

    raw = np.maximum(raw, 0.0)

    pv_total = raw * PV_SCALE_TOTAL
    pv_each = {bus: pv_total / len(PV_BUSES) for bus in PV_BUSES}

    return pv_total, pv_each


# ============================================================
# 5. Fixed PB and grid bounds
# ============================================================

def build_fixed_pb(reference):
    blocks = np.array_split(np.arange(T), PB_BLOCKS)
    avg = np.array([reference[idx].mean() for idx in blocks])
    q1, q2 = np.quantile(avg, [1 / 3, 2 / 3])

    pb_mid = 0.5 * (PB_MIN + PB_MAX)
    pb = np.zeros(T)

    for a, idx in zip(avg, blocks):
        if a <= q1:
            pb[idx] = PB_MIN
        elif a <= q2:
            pb[idx] = pb_mid
        else:
            pb[idx] = PB_MAX

    return pb


def grid_bounds(base_minus_pv):
    base_minus_pv = np.asarray(base_minus_pv, dtype=float)

    max_net = np.max(base_minus_pv) + EV_P_MAX + len(BESS_BUSES) * BESS_P_MAX
    min_net = np.min(base_minus_pv) - len(BESS_BUSES) * BESS_P_MAX

    imp_ub = max(GRID_MIN_UB, max_net + GRID_MARGIN)
    exp_ub = max(GRID_MIN_UB, -min_net + GRID_MARGIN)

    print(f"[bound] import_ub = {imp_ub:.2f} kW")
    print(f"[bound] export_ub = {exp_ub:.2f} kW")

    return imp_ub, exp_ub


# ============================================================
# 6. Follower model
# ============================================================

def solve_follower(pa, pb, base_minus_pv, out_dir):
    imp_ub, exp_ub = grid_bounds(base_minus_pv)

    m = gp.Model("phase1_selected_bus_follower")
    m.Params.OutputFlag = 0
    m.Params.InfUnbdInfo = 1

    TT = range(T)
    BB = range(len(BESS_BUSES))

    ev = m.addVars(TT, lb=0, ub=EV_P_MAX, name="ev")

    imp = m.addVars(TT, lb=0, ub=imp_ub, name="imp")
    exp = m.addVars(TT, lb=0, ub=exp_ub, name="exp")
    grid_import_mode = m.addVars(TT, vtype=GRB.BINARY, name="grid_import_mode")

    ch = m.addVars(BB, TT, lb=0, ub=BESS_P_MAX, name="ch")
    dis = m.addVars(BB, TT, lb=0, ub=BESS_P_MAX, name="dis")
    soc = m.addVars(BB, range(T + 1), lb=BESS_E_MIN, ub=BESS_E_MAX, name="soc")
    bess_ch_mode = m.addVars(BB, TT, vtype=GRB.BINARY, name="bess_ch_mode")

    m.addConstr(gp.quicksum(ev[t] * DT for t in TT) == EV_E_REQ, name="ev_energy")

    for t in TT:
        m.addConstr(imp[t] <= imp_ub * grid_import_mode[t], name=f"imp_mode_{t}")
        m.addConstr(exp[t] <= exp_ub * (1 - grid_import_mode[t]), name=f"exp_mode_{t}")

        m.addConstr(
            imp[t] - exp[t]
            == float(base_minus_pv[t])
            + ev[t]
            + gp.quicksum(ch[b, t] - dis[b, t] for b in BB),
            name=f"power_balance_{t}",
        )

    for b in BB:
        m.addConstr(soc[b, 0] == BESS_E_INIT, name=f"soc_init_{b}")
        m.addConstr(soc[b, T] == BESS_E_INIT, name=f"soc_terminal_{b}")

        for t in TT:
            m.addConstr(ch[b, t] <= BESS_P_MAX * bess_ch_mode[b, t], name=f"ch_mode_{b}_{t}")
            m.addConstr(dis[b, t] <= BESS_P_MAX * (1 - bess_ch_mode[b, t]), name=f"dis_mode_{b}_{t}")

            m.addConstr(
                soc[b, t + 1]
                == soc[b, t]
                + BESS_ETA_CH * ch[b, t] * DT
                - dis[b, t] / BESS_ETA_DIS * DT,
                name=f"soc_dyn_{b}_{t}",
            )

    energy_cost = gp.quicksum(
        ((float(pa[t]) + float(pb[t])) * imp[t] - float(pb[t]) * exp[t]) * DT
        for t in TT
    )

    small_regularization = EPS * gp.quicksum(
        (imp[t] + exp[t] + ev[t]) * DT
        + gp.quicksum((ch[b, t] + dis[b, t]) * DT for b in BB)
        for t in TT
    )

    m.setObjective(energy_cost + small_regularization, GRB.MINIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        try:
            m.computeIIS()
            m.write(str(out_dir / "phase1_infeasible.ilp"))
            print(f"[iis] written to {out_dir / 'phase1_infeasible.ilp'}")
        except Exception:
            pass

        raise RuntimeError(f"Follower model not optimal. Gurobi status = {m.Status}")

    sol = pd.DataFrame({
        "step": np.arange(T),
        "hour": np.arange(T) * DT,
        "pa": pa,
        "pb": pb,
        "base_minus_pv_kw": base_minus_pv,
        "ev_kw": [ev[t].X for t in TT],
        "imp_kw": [imp[t].X for t in TT],
        "exp_kw": [exp[t].X for t in TT],
    })

    for b, bus in enumerate(BESS_BUSES):
        tag = bus.replace(" ", "_")
        sol[f"{tag}_ch_kw"] = [ch[b, t].X for t in TT]
        sol[f"{tag}_dis_kw"] = [dis[b, t].X for t in TT]
        sol[f"{tag}_soc_kwh"] = [soc[b, t].X for t in TT]
        sol[f"{tag}_soc_terminal_kwh"] = soc[b, T].X

    ch_cols = [c for c in sol.columns if c.endswith("_ch_kw")]
    dis_cols = [c for c in sol.columns if c.endswith("_dis_kw")]

    sol["bess_ch_total_kw"] = sol[ch_cols].sum(axis=1)
    sol["bess_dis_total_kw"] = sol[dis_cols].sum(axis=1)
    sol["net_exchange_kw"] = sol["imp_kw"] - sol["exp_kw"]

    sol.attrs["objective"] = m.ObjVal
    sol.attrs["import_ub"] = imp_ub
    sol.attrs["export_ub"] = exp_ub

    return sol


# ============================================================
# 7. Reports and figures
# ============================================================

def save_reports(out_dir, inputs, sol, selected_load_idx):
    inputs.to_csv(out_dir / "phase1_inputs.csv", index=False)
    sol.to_csv(out_dir / "phase1_solution.csv", index=False)

    balance_error = (
        sol["base_minus_pv_kw"]
        + sol["ev_kw"]
        + sol["bess_ch_total_kw"]
        - sol["bess_dis_total_kw"]
        - sol["net_exchange_kw"]
    )

    row = {
        "active_load_buses": ", ".join(ACTIVE_LOAD_BUSES),
        "selected_simbench_load_indices": ", ".join(map(str, selected_load_idx)),
        "EV_bus": EV_BUS,
        "PV_buses": ", ".join(PV_BUSES),
        "BESS_buses": ", ".join(BESS_BUSES),
        "selected_baseline_peak_kw": float(inputs["baseline_kw"].max()),
        "selected_baseline_min_kw": float(inputs["baseline_kw"].min()),
        "full_feeder_peak_kw_reference_only": float(inputs["full_feeder_reference_kw"].max()),
        "selected_to_full_peak_ratio": float(
            inputs["baseline_kw"].max() / max(inputs["full_feeder_reference_kw"].max(), 1e-9)
        ),
        "pv_peak_kw": float(inputs["pv_total_kw"].max()),
        "pv_energy_kwh": float(inputs["pv_total_kw"].sum() * DT),
        "base_minus_pv_peak_kw": float(inputs["base_minus_pv_kw"].max()),
        "base_minus_pv_min_kw": float(inputs["base_minus_pv_kw"].min()),
        "ev_energy_kwh": float(sol["ev_kw"].sum() * DT),
        "import_ub_kw": float(sol.attrs["import_ub"]),
        "export_ub_kw": float(sol.attrs["export_ub"]),
        "max_balance_error_kw": float(np.max(np.abs(balance_error))),
        "objective": float(sol.attrs["objective"]),
    }

    for bus in BESS_BUSES:
        tag = bus.replace(" ", "_")
        row[f"{tag}_soc_init_kwh"] = BESS_E_INIT
        row[f"{tag}_soc_terminal_kwh"] = float(sol[f"{tag}_soc_terminal_kwh"].iloc[0])
        row[f"{tag}_simultaneous_ch_dis_steps"] = int(
            ((sol[f"{tag}_ch_kw"] > 1e-6) & (sol[f"{tag}_dis_kw"] > 1e-6)).sum()
        )

    pd.DataFrame([row]).to_csv(out_dir / "phase1_sanity_checks.csv", index=False)


def plot_data(out_dir, inputs):
    h = inputs["hour"]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(h, inputs["baseline_kw"], label="manual selected baseline")
    ax.plot(h, inputs["pv_total_kw"], label="PV total")
    ax.plot(h, inputs["base_minus_pv_kw"], label="baseline - PV")

    ax.set_xlabel("hour")
    ax.set_ylabel("power [kW]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_dir / "phase1_01_data_sanity.png", dpi=180)
    plt.close(fig)

def plot_model(out_dir, sol):
    h = sol["hour"]

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(h, sol["base_minus_pv_kw"], label="selected baseline - PV")
    axes[0].plot(h, sol["net_exchange_kw"], label="optimized net exchange")
    axes[0].set_ylabel("kW")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(h, sol["imp_kw"], label="import")
    axes[1].plot(h, sol["exp_kw"], label="export")
    axes[1].set_ylabel("kW")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    axes[2].plot(h, sol["ev_kw"], label="EV")
    axes[2].plot(h, sol["bess_ch_total_kw"], label="BESS charge total")
    axes[2].plot(h, sol["bess_dis_total_kw"], label="BESS discharge total")
    axes[2].set_ylabel("kW")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()

    h_soc = np.arange(T + 1) * DT

    for bus in BESS_BUSES:
        tag = bus.replace(" ", "_")
        soc_97 = list(sol[f"{tag}_soc_kwh"].to_numpy()) + [float(sol[f"{tag}_soc_terminal_kwh"].iloc[0])]
        axes[3].plot(h_soc, soc_97, label=f"SOC {bus}")

    axes[3].set_xlabel("hour")
    axes[3].set_ylabel("kWh")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "phase1_02_model_sanity.png", dpi=180)
    plt.close(fig)


# ============================================================
# 8. Main
# ============================================================

def main():
    project_root, feeder_dir, out_dir = get_paths()

    baseline_kw, full_feeder_kw, selected_load_idx = load_selected_simbench_baseline()
    pv_total, pv_each = load_pv(project_root, feeder_dir)
    pa = load_price(project_root, feeder_dir)

    base_minus_pv = baseline_kw - pv_total
    pb = build_fixed_pb(base_minus_pv)

    inputs = pd.DataFrame({
        "step": np.arange(T),
        "hour": np.arange(T) * DT,
        "full_feeder_reference_kw": full_feeder_kw,
        "baseline_kw": baseline_kw,
        "pv_total_kw": pv_total,
        "base_minus_pv_kw": base_minus_pv,
        "pa": pa,
        "pb": pb,
    })

    for bus, arr in pv_each.items():
        inputs[f"pv_{bus.replace(' ', '_')}_kw"] = arr

    sol = solve_follower(pa, pb, base_minus_pv, out_dir)

    save_reports(out_dir, inputs, sol, selected_load_idx)
    plot_data(out_dir, inputs)
    plot_model(out_dir, sol)

    print("\n[done] Phase 1 selected-bus sanity check finished.")
    print(f"[done] results saved to: {out_dir}")
    print("[done] generated:")
    print("       phase1_inputs.csv")
    print("       phase1_solution.csv")
    print("       phase1_sanity_checks.csv")
    print("       phase1_01_data_sanity.png")
    print("       phase1_02_model_sanity.png")


if __name__ == "__main__":
    main()