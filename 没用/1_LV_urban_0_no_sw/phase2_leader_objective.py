# -*- coding: utf-8 -*-

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB


# =========================
# Parameters
# =========================

FEEDER_FOLDER = "1_LV_urban_0_no_sw"
PHASE1_FOLDER = "phase1_outputs"
OUT_FOLDER = "phase2_kkt_outputs"

T = 96
DT = 0.25

OBJECTIVE_CASE = "A"        # A, B, C, D, AB, BCD, ABCD

PV_SCALE_PHASE2 = 0.9

PB_MIN = 0.00
PB_MAX = 0.35
PB_RAMP_LIMIT = 0.1
PB_N_BLOCKS = 8               # 这里设置 PB 有几个窗口

M_DUAL = 200.0

EV_P_MAX = 1.5
EV_E_REQ = 4.0

BESS_BUSES = ["Bus R1", "Bus R3"]
BESS_P_MAX = 0.8
BESS_E_MIN = 0.2
BESS_E_MAX = 2.0
BESS_E_INIT = 1.0
BESS_ETA_CH = 0.95
BESS_ETA_DIS = 0.95

GRID_MIN_UB = 20.0
GRID_MARGIN = 10.0

FOLLOWER_IMPORT_EXPORT_EPS = 1e-5
FOLLOWER_EV_EPS = 1e-5
FOLLOWER_BESS_COST = 8e-4

REF_MODE = "MEAN"
REF_ROLLING_STEPS = 20

W_A_TRACK = 1.0
W_B_RAMP = 0.08
W_C_VAR = 0.08
W_D_PEAK_VALLEY = 0.85

OUTPUT_FLAG = 1
TIME_LIMIT = 1200
MIP_GAP = 0.10


# =========================
# Helpers
# =========================

def safe_name(x):
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(x)).strip("_")


def has_obj(x):
    return x in OBJECTIVE_CASE.upper()


def check_case():
    valid = {"A", "B", "C", "D", "AB", "BCD", "ABCD"}
    case = OBJECTIVE_CASE.upper()
    if case not in valid:
        raise ValueError(f"OBJECTIVE_CASE must be one of {sorted(valid)}")
    return case


def paths():
    here = Path(__file__).resolve().parent
    feeder = here if here.name == FEEDER_FOLDER else here / FEEDER_FOLDER
    p1 = feeder / PHASE1_FOLDER
    out = feeder / OUT_FOLDER / f"case_{OBJECTIVE_CASE.upper()}"
    out.mkdir(parents=True, exist_ok=True)
    if not p1.exists():
        raise FileNotFoundError(p1)
    return p1, out


def read_inputs(p1):
    f = p1 / "phase1_inputs.csv"
    df = pd.read_csv(f)
    need = ["hour", "baseline_kw", "pv_total_kw", "base_minus_pv_kw", "pa"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"Missing columns: {miss}")
    if len(df) != T:
        raise ValueError(f"Expected {T} rows, got {len(df)}")
    return df


def make_data(df):
    hour = df["hour"].to_numpy(float)
    pa = df["pa"].to_numpy(float)
    baseline = df["baseline_kw"].to_numpy(float)
    pv1 = df["pv_total_kw"].to_numpy(float)
    pv2 = PV_SCALE_PHASE2 * pv1
    base = baseline - pv2

    if REF_MODE.upper() == "MEAN":
        ref = np.ones(T) * base.mean()
    else:
        ref = pd.Series(base).rolling(
            REF_ROLLING_STEPS, center=True, min_periods=1
        ).mean().to_numpy(float)

    imp_ub = max(GRID_MIN_UB, base.max() + EV_P_MAX + len(BESS_BUSES) * BESS_P_MAX + GRID_MARGIN)
    exp_ub = max(GRID_MIN_UB, -base.min() + len(BESS_BUSES) * BESS_P_MAX + GRID_MARGIN)

    print(f"[data] base-PV peak = {base.max():.3f} kW")
    print(f"[data] base-PV mean = {base.mean():.3f} kW")
    print(f"[PV] phase1 peak = {pv1.max():.3f} kW")
    print(f"[PV] phase2 peak = {pv2.max():.3f} kW")
    print(f"[PB] windows = {PB_N_BLOCKS}")

    return hour, pa, baseline, pv1, pv2, base, ref, imp_ub, exp_ub


def block_ids():
    ids = np.zeros(T, dtype=int)
    for k, idx in enumerate(np.array_split(np.arange(T), PB_N_BLOCKS)):
        ids[idx] = k
    return ids


def add_abs(m, expr, name):
    z = m.addVar(lb=0.0, name=name)
    m.addConstr(z >= expr)
    m.addConstr(z >= -expr)
    return z


def comp_lb(m, x, dual, ub, name):
    z = m.addVar(vtype=GRB.BINARY, name=f"z_{name}")
    m.addConstr(x <= ub * z)
    m.addConstr(dual <= M_DUAL * (1 - z))


def comp_ub(m, x, dual, ub, name):
    z = m.addVar(vtype=GRB.BINARY, name=f"z_{name}")
    m.addConstr(ub - x <= ub * z)
    m.addConstr(dual <= M_DUAL * (1 - z))


def comp_lb_shift(m, x, lb, dual, span, name):
    z = m.addVar(vtype=GRB.BINARY, name=f"z_{name}")
    m.addConstr(x - lb <= span * z)
    m.addConstr(dual <= M_DUAL * (1 - z))


def comp_ub_shift(m, x, ub, dual, span, name):
    z = m.addVar(vtype=GRB.BINARY, name=f"z_{name}")
    m.addConstr(ub - x <= span * z)
    m.addConstr(dual <= M_DUAL * (1 - z))


# =========================
# Model
# =========================

def solve(df, out):
    case = check_case()
    hour, pa, baseline, pv1, pv2, base, ref, imp_ub, exp_ub = make_data(df)

    m = gp.Model("phase2_kkt")
    m.Params.OutputFlag = OUTPUT_FLAG
    m.Params.TimeLimit = TIME_LIMIT
    m.Params.MIPGap = MIP_GAP
    m.Params.NumericFocus = 1
    m.Params.MIPFocus = 1
    m.Params.Heuristics = 0.3
    m.Params.Cuts = 2
    m.Params.Presolve = 2

    TT = range(T)
    BB = range(len(BESS_BUSES))

    bid = block_ids()
    pb_block = m.addVars(PB_N_BLOCKS, lb=PB_MIN, ub=PB_MAX, name="pb_block")
    pb = {t: pb_block[int(bid[t])] for t in TT}

    for k in range(1, PB_N_BLOCKS):
        m.addConstr(pb_block[k] - pb_block[k - 1] <= PB_RAMP_LIMIT)
        m.addConstr(pb_block[k - 1] - pb_block[k] <= PB_RAMP_LIMIT)

    ev = m.addVars(TT, lb=0, ub=EV_P_MAX, name="ev")
    imp = m.addVars(TT, lb=0, ub=imp_ub, name="imp")
    exp = m.addVars(TT, lb=0, ub=exp_ub, name="exp")
    ch = m.addVars(BB, TT, lb=0, ub=BESS_P_MAX, name="ch")
    dis = m.addVars(BB, TT, lb=0, ub=BESS_P_MAX, name="dis")
    soc = m.addVars(BB, range(T + 1), lb=BESS_E_MIN, ub=BESS_E_MAX, name="soc")

    a_ev = m.addVar(lb=-GRB.INFINITY, name="dual_ev_energy")
    lam = m.addVars(TT, lb=-GRB.INFINITY, name="dual_balance")
    g = m.addVars(BB, TT, lb=-GRB.INFINITY, name="dual_soc_dyn")
    ni = m.addVars(BB, lb=-GRB.INFINITY, name="dual_soc_init")
    nt = m.addVars(BB, lb=-GRB.INFINITY, name="dual_soc_terminal")

    l_ev = m.addVars(TT, lb=0, name="l_ev")
    u_ev = m.addVars(TT, lb=0, name="u_ev")
    l_imp = m.addVars(TT, lb=0, name="l_imp")
    u_imp = m.addVars(TT, lb=0, name="u_imp")
    l_exp = m.addVars(TT, lb=0, name="l_exp")
    u_exp = m.addVars(TT, lb=0, name="u_exp")
    l_ch = m.addVars(BB, TT, lb=0, name="l_ch")
    u_ch = m.addVars(BB, TT, lb=0, name="u_ch")
    l_dis = m.addVars(BB, TT, lb=0, name="l_dis")
    u_dis = m.addVars(BB, TT, lb=0, name="u_dis")
    l_soc = m.addVars(BB, range(T + 1), lb=0, name="l_soc")
    u_soc = m.addVars(BB, range(T + 1), lb=0, name="u_soc")

    m.addConstr(gp.quicksum(ev[t] * DT for t in TT) == EV_E_REQ)

    for t in TT:
        m.addConstr(
            imp[t] - exp[t] - ev[t]
            - gp.quicksum(ch[b, t] for b in BB)
            + gp.quicksum(dis[b, t] for b in BB)
            == float(base[t])
        )

    for b in BB:
        m.addConstr(soc[b, 0] == BESS_E_INIT)
        m.addConstr(soc[b, T] == BESS_E_INIT)
        for t in TT:
            m.addConstr(
                soc[b, t + 1] - soc[b, t]
                - BESS_ETA_CH * ch[b, t] * DT
                + dis[b, t] / BESS_ETA_DIS * DT
                == 0
            )

    for t in TT:
        m.addConstr(DT * (float(pa[t]) + pb[t] + FOLLOWER_IMPORT_EXPORT_EPS) + lam[t] + u_imp[t] - l_imp[t] == 0)
        m.addConstr(DT * (-pb[t] + FOLLOWER_IMPORT_EXPORT_EPS) - lam[t] + u_exp[t] - l_exp[t] == 0)
        m.addConstr(DT * FOLLOWER_EV_EPS + a_ev * DT - lam[t] + u_ev[t] - l_ev[t] == 0)

    for b in BB:
        for t in TT:
            cb = DT * FOLLOWER_BESS_COST
            m.addConstr(cb - lam[t] - BESS_ETA_CH * DT * g[b, t] + u_ch[b, t] - l_ch[b, t] == 0)
            m.addConstr(cb + lam[t] + DT / BESS_ETA_DIS * g[b, t] + u_dis[b, t] - l_dis[b, t] == 0)

        for t in range(T + 1):
            expr = 0
            if t > 0:
                expr += g[b, t - 1]
            if t < T:
                expr -= g[b, t]
            if t == 0:
                expr += ni[b]
            if t == T:
                expr += nt[b]
            m.addConstr(expr + u_soc[b, t] - l_soc[b, t] == 0)

    span = BESS_E_MAX - BESS_E_MIN

    for t in TT:
        comp_lb(m, ev[t], l_ev[t], EV_P_MAX, f"ev_lb_{t}")
        comp_ub(m, ev[t], u_ev[t], EV_P_MAX, f"ev_ub_{t}")
        comp_lb(m, imp[t], l_imp[t], imp_ub, f"imp_lb_{t}")
        comp_ub(m, imp[t], u_imp[t], imp_ub, f"imp_ub_{t}")
        comp_lb(m, exp[t], l_exp[t], exp_ub, f"exp_lb_{t}")
        comp_ub(m, exp[t], u_exp[t], exp_ub, f"exp_ub_{t}")

    for b in BB:
        for t in TT:
            comp_lb(m, ch[b, t], l_ch[b, t], BESS_P_MAX, f"ch_lb_{b}_{t}")
            comp_ub(m, ch[b, t], u_ch[b, t], BESS_P_MAX, f"ch_ub_{b}_{t}")
            comp_lb(m, dis[b, t], l_dis[b, t], BESS_P_MAX, f"dis_lb_{b}_{t}")
            comp_ub(m, dis[b, t], u_dis[b, t], BESS_P_MAX, f"dis_ub_{b}_{t}")

        for t in range(T + 1):
            comp_lb_shift(m, soc[b, t], BESS_E_MIN, l_soc[b, t], span, f"soc_lb_{b}_{t}")
            comp_ub_shift(m, soc[b, t], BESS_E_MAX, u_soc[b, t], span, f"soc_ub_{b}_{t}")

    net = m.addVars(TT, lb=-GRB.INFINITY, name="net")
    for t in TT:
        m.addConstr(net[t] == imp[t] - exp[t])

    terms = []

    if has_obj("A"):
        terms.append(W_A_TRACK * gp.quicksum(add_abs(m, net[t] - float(ref[t]), f"A_{t}") * DT for t in TT))

    if has_obj("B"):
        terms.append(W_B_RAMP * gp.quicksum(add_abs(m, net[t] - net[t - 1], f"B_{t}") for t in range(1, T)))

    if has_obj("C"):
        mean_net = m.addVar(lb=-GRB.INFINITY, name="mean_net")
        m.addConstr(T * mean_net == gp.quicksum(net[t] for t in TT))
        terms.append(W_C_VAR * gp.quicksum(add_abs(m, net[t] - mean_net, f"C_{t}") * DT for t in TT))

    if has_obj("D"):
        peak = m.addVar(lb=-GRB.INFINITY, name="peak")
        valley = m.addVar(lb=-GRB.INFINITY, name="valley")
        for t in TT:
            m.addConstr(peak >= net[t])
            m.addConstr(valley <= net[t])
        terms.append(W_D_PEAK_VALLEY * (peak - valley))

    m.setObjective(gp.quicksum(terms), GRB.MINIMIZE)
    m.optimize()

    if m.SolCount == 0:
        m.computeIIS()
        m.write(str(out / "infeasible.ilp"))
        raise RuntimeError(f"No solution, status={m.Status}")

    sol = pd.DataFrame({
        "step": np.arange(T),
        "hour": hour,
        "baseline_kw": baseline,
        "pv_phase1_kw": pv1,
        "pv_phase2_kw": pv2,
        "base_minus_pv_kw": base,
        "reference_kw": ref,
        "pa": pa,
        "pb_opt": [pb[t].X for t in TT],
        "pb_block_id": bid,
        "ev_kw": [ev[t].X for t in TT],
        "imp_kw": [imp[t].X for t in TT],
        "exp_kw": [exp[t].X for t in TT],
        "net_exchange_kw": [net[t].X for t in TT],
    })

    if "pb" in df.columns:
        sol["pb_phase1"] = df["pb"].to_numpy(float)

    for b, bus in enumerate(BESS_BUSES):
        tag = safe_name(bus)
        sol[f"{tag}_ch_kw"] = [ch[b, t].X for t in TT]
        sol[f"{tag}_dis_kw"] = [dis[b, t].X for t in TT]
        sol[f"{tag}_soc_kwh"] = [soc[b, t].X for t in TT]
        sol[f"{tag}_soc_terminal_kwh"] = soc[b, T].X

    ch_cols = [c for c in sol.columns if c.endswith("_ch_kw")]
    dis_cols = [c for c in sol.columns if c.endswith("_dis_kw")]
    sol["bess_ch_total_kw"] = sol[ch_cols].sum(axis=1)
    sol["bess_dis_total_kw"] = sol[dis_cols].sum(axis=1)
    sol["bess_net_kw"] = sol["bess_ch_total_kw"] - sol["bess_dis_total_kw"]

    net_np = sol["net_exchange_kw"].to_numpy(float)

    comps = pd.DataFrame([{
        "case": case,
        "status": int(m.Status),
        "objective": float(m.ObjVal),
        "mip_gap": float(m.MIPGap) if hasattr(m, "MIPGap") else np.nan,
        "PB_N_BLOCKS": PB_N_BLOCKS,
        "pb_min": float(sol["pb_opt"].min()),
        "pb_max": float(sol["pb_opt"].max()),
        "ev_energy_kwh": float(sol["ev_kw"].sum() * DT),
        "net_peak_kw": float(np.max(net_np)),
        "net_valley_kw": float(np.min(net_np)),
        "net_peak_valley_kw": float(np.max(net_np) - np.min(net_np)),
        "net_ramp_sum_kw": float(np.sum(np.abs(np.diff(net_np)))),
        "net_var_l1_kwh": float(np.sum(np.abs(net_np - np.mean(net_np))) * DT),
    }])

    sol.to_csv(out / "phase2_kkt_solution.csv", index=False)
    comps.to_csv(out / "phase2_kkt_objective_components.csv", index=False)

    return sol, comps


def sanity(sol, out):
    err = sol["base_minus_pv_kw"] + sol["ev_kw"] + sol["bess_ch_total_kw"] - sol["bess_dis_total_kw"] - sol["net_exchange_kw"]

    row = {
        "max_balance_error_kw": float(np.max(np.abs(err))),
        "ev_energy_kwh": float(sol["ev_kw"].sum() * DT),
        "simultaneous_import_export_steps": int(((sol["imp_kw"] > 1e-6) & (sol["exp_kw"] > 1e-6)).sum()),
        "pb_min": float(sol["pb_opt"].min()),
        "pb_max": float(sol["pb_opt"].max()),
    }

    pd.DataFrame([row]).to_csv(out / "phase2_kkt_sanity_checks.csv", index=False)

    print("\n[check]")
    for k, v in row.items():
        print(f"{k}: {v}")


def plot(sol, comps, out):
    h = sol["hour"]
    case = comps["case"].iloc[0]

    fig, ax = plt.subplots(6, 1, figsize=(12, 15), sharex=True)

    ax[0].plot(h, sol["base_minus_pv_kw"], label="baseline - PV")
    ax[0].plot(h, sol["net_exchange_kw"], label="optimized net exchange")
    if has_obj("A"):
        ax[0].plot(h, sol["reference_kw"], linestyle=":", label="reference")
    ax[0].set_title(f"Leader objective: {case}, PB windows = {PB_N_BLOCKS}")
    ax[0].set_ylabel("kW")
    ax[0].legend()
    ax[0].grid(True, alpha=0.25)

    ax[1].plot(h, sol["pv_phase1_kw"], label="PV phase1")
    ax[1].plot(h, sol["pv_phase2_kw"], label="PV phase2")
    ax[1].set_ylabel("PV [kW]")
    ax[1].legend()
    ax[1].grid(True, alpha=0.25)

    ax[2].step(h, sol["pa"], where="post", linestyle="--", label="PA")
    ax[2].step(h, sol["pb_opt"], where="post", linestyle=":", label="optimized PB")
    if "pb_phase1" in sol.columns:
        ax[2].step(h, sol["pb_phase1"], where="post", alpha=0.4, label="phase1 fixed PB")
    ax[2].set_ylabel("EUR/kWh")
    ax[2].legend()
    ax[2].grid(True, alpha=0.25)

    ax[3].plot(h, sol["ev_kw"], label="EV charging")
    ax[3].set_ylabel("EV [kW]")
    ax[3].legend()
    ax[3].grid(True, alpha=0.25)

    ax[4].plot(h, sol["bess_ch_total_kw"], label="BESS charge")
    ax[4].plot(h, -sol["bess_dis_total_kw"], label="BESS discharge (-)")
    ax[4].plot(h, sol["bess_net_kw"], linestyle="--", label="BESS net")
    ax[4].axhline(0, linewidth=0.8)
    ax[4].set_ylabel("BESS [kW]")
    ax[4].legend()
    ax[4].grid(True, alpha=0.25)

    h_soc = np.arange(T + 1) * DT
    for bus in BESS_BUSES:
        tag = safe_name(bus)
        y = list(sol[f"{tag}_soc_kwh"].to_numpy()) + [float(sol[f"{tag}_soc_terminal_kwh"].iloc[0])]
        ax[5].plot(h_soc, y, label=f"SOC {bus}")
    ax[5].set_ylabel("SOC [kWh]")
    ax[5].set_xlabel("hour")
    ax[5].legend()
    ax[5].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out / "phase2_kkt_result.png", dpi=180)
    plt.close(fig)


def main():
    p1, out = paths()
    df = read_inputs(p1)
    sol, comps = solve(df, out)
    sanity(sol, out)
    plot(sol, comps, out)
    print("\n[done]")
    print(out)


if __name__ == "__main__":
    main()