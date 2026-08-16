# -*- coding: utf-8 -*-
"""Static optimistic Stackelberg test: PV + BESS, import/export, no EV."""

from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB


def tag(x):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(x)).strip("_")


def get_paths():
    here = Path(__file__).resolve().parent
    feeder = here if here.name == FEEDER_FOLDER else here / FEEDER_FOLDER
    src = feeder / PHASE1_FOLDER / INPUT_FILE
    out = feeder / OUTPUT_FOLDER
    out.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise FileNotFoundError(f"Input not found: {src}")
    return src, out


def load_data(src):
    df = pd.read_csv(src).iloc[:T].copy()
    required = ["hour", "aggregate_net_baseline_kw", "pa_eur_per_kwh"]
    missing = [c for c in required if c not in df.columns]
    if missing or len(df) != T:
        raise ValueError(f"Invalid input: missing={missing}, rows={len(df)}, expected={T}")

    base = df["aggregate_net_baseline_kw"].to_numpy(float)
    pa = df["pa_eur_per_kwh"].to_numpy(float)
    p_sum = sum(c["p_max_kw"] for c in BESS_CONFIG.values())
    imp_ub = GRID_IMPORT_MAX_KW or max(GRID_MARGIN_KW, base.max() + p_sum + GRID_MARGIN_KW)
    exp_ub = GRID_EXPORT_MAX_KW or max(GRID_MARGIN_KW, -base.min() + p_sum + GRID_MARGIN_KW)
    return {"df": df, "base": base, "pa": pa,
            "imp_ub": float(imp_ub), "exp_ub": float(exp_ub)}


def terminal_bounds(c):
    # Five percent of usable energy range, clipped to physical limits.
    tol = TERMINAL_SOC_TOL_FRAC * (c["e_max_kwh"] - c["e_min_kwh"])
    lo = max(c["e_min_kwh"], c["e_init_kwh"] - tol)
    hi = min(c["e_max_kwh"], c["e_init_kwh"] + tol)
    return lo, hi, tol


def set_solver(m, mip=True, log=True):
    m.Params.OutputFlag = OUTPUT_FLAG if log else 0
    m.Params.TimeLimit = TIME_LIMIT
    m.Params.NumericFocus = NUMERIC_FOCUS
    m.Params.Seed = SOLVER_SEED
    m.Params.Threads = THREADS
    if mip:
        m.Params.MIPGap = MIP_GAP
        m.Params.MIPFocus = MIP_FOCUS
        m.Params.Heuristics = HEURISTICS
        m.Params.Cuts = CUTS


def build_primal(m, d):
    imp = m.addVars(T, lb=0, ub=d["imp_ub"], name="imp")
    exp = m.addVars(T, lb=0, ub=d["exp_ub"], name="exp")
    p, soc = {}, {}

    for b, c in BESS_CONFIG.items():
        n = tag(b)
        p[b] = m.addVars(T, lb=-c["p_max_kw"], ub=c["p_max_kw"], name=f"p_{n}")
        soc[b] = {}
        lo_t, hi_t, _ = terminal_bounds(c)
        for s in range(T + 1):
            lo, hi = (lo_t, hi_t) if s == T else (c["e_min_kwh"], c["e_max_kwh"])
            soc[b][s] = m.addVar(lb=lo, ub=hi, name=f"soc_{n}_{s}")
        m.addConstr(soc[b][0] == c["e_init_kwh"], name=f"soc_init_{n}")
        for t in range(T):
            # p > 0: charge; p < 0: discharge. Frozen test assumes eta=1.
            m.addConstr(soc[b][t + 1] == soc[b][t] + DT * p[b][t],
                        name=f"soc_dyn_{n}_{t}")

    for t in range(T):
        m.addConstr(
            imp[t] - exp[t] - gp.quicksum(p[b][t] for b in BESS_CONFIG)
            == float(d["base"][t]), name=f"balance_{t}"
        )
    return {"imp": imp, "exp": exp, "p": p, "soc": soc}


def follower_objective(d, v, pb):
    # Export receives PB; PA is the upstream purchase component.
    energy = gp.quicksum(DT * ((float(d["pa"][t]) + pb[t]) * v["imp"][t]
                              - pb[t] * v["exp"][t]) for t in range(T))
    # Kept as a parameter for documentation; signed-power model omits it when zero.
    if abs(BESS_CYCLE_COST) > 1e-12:
        raise ValueError("This signed-power test requires BESS_CYCLE_COST = 0.")
    return energy


def complementarity(m, slack_expr, slack_ub, dual, name):
    ub = max(float(slack_ub), 1e-9)
    z = m.addVar(vtype=GRB.BINARY, name=f"z_{name}")
    slack = m.addVar(lb=0, ub=ub, name=f"slack_{name}")
    m.addConstr(slack == slack_expr)
    m.addConstr(slack <= ub * M_SLACK_FACTOR * z)
    m.addConstr(dual <= DUAL_MAX * (1 - z))


def bounded_duals(m, x, lb, ub, name):
    dl = m.addVar(lb=0, ub=DUAL_MAX, name=f"dl_{name}")
    du = m.addVar(lb=0, ub=DUAL_MAX, name=f"du_{name}")
    complementarity(m, x - lb, ub - lb, dl, f"{name}_lb")
    complementarity(m, ub - x, ub - lb, du, f"{name}_ub")
    return dl, du


def verify_follower(d, pb_value):
    m = gp.Model("verify_follower")
    set_solver(m, mip=False, log=False)
    v = build_primal(m, d)
    m.setObjective(follower_objective(d, v, pb_value), GRB.MINIMIZE)
    m.optimize()
    return float(m.ObjVal) if m.SolCount else np.nan


def solve(d):
    m = gp.Model("test_phase_static_stackelberg")
    set_solver(m)
    pb = m.addVars(T, lb=PB_MIN, ub=PB_MAX, name="pb")
    avg = gp.quicksum(pb[t] for t in range(T)) / T
    m.addConstr(avg >= PB_AVG_TARGET - PB_AVG_TOL)
    m.addConstr(avg <= PB_AVG_TARGET + PB_AVG_TOL)
    for t in range(1, T):
        m.addConstr(pb[t] - pb[t - 1] <= PB_RAMP_LIMIT)
        m.addConstr(pb[t - 1] - pb[t] <= PB_RAMP_LIMIT)

    v = build_primal(m, d)
    lam = m.addVars(T, lb=-GRB.INFINITY, name="lambda_balance")
    gamma = {b: m.addVars(T, lb=-GRB.INFINITY, name=f"gamma_{tag(b)}")
             for b in BESS_CONFIG}
    init_dual = {b: m.addVar(lb=-GRB.INFINITY, name=f"init_dual_{tag(b)}")
                 for b in BESS_CONFIG}

    imp_d = [bounded_duals(m, v["imp"][t], 0, d["imp_ub"], f"imp_{t}") for t in range(T)]
    exp_d = [bounded_duals(m, v["exp"][t], 0, d["exp_ub"], f"exp_{t}") for t in range(T)]
    p_d, soc_d = {}, {}
    for b, c in BESS_CONFIG.items():
        n = tag(b)
        p_d[b] = [bounded_duals(m, v["p"][b][t], -c["p_max_kw"], c["p_max_kw"],
                                  f"p_{n}_{t}") for t in range(T)]
        lo_t, hi_t, _ = terminal_bounds(c)
        soc_d[b] = []
        for s in range(T + 1):
            lo, hi = (lo_t, hi_t) if s == T else (c["e_min_kwh"], c["e_max_kwh"])
            soc_d[b].append(bounded_duals(m, v["soc"][b][s], lo, hi, f"soc_{n}_{s}"))

    # KKT stationarity of the continuous convex follower LP.
    for t in range(T):
        li, ui = imp_d[t]
        le, ue = exp_d[t]
        m.addConstr(DT * (float(d["pa"][t]) + pb[t]) + lam[t] - li + ui == 0)
        m.addConstr(-DT * pb[t] - lam[t] - le + ue == 0)

    for b in BESS_CONFIG:
        for t in range(T):
            lp, up = p_d[b][t]
            m.addConstr(-lam[t] - DT * gamma[b][t] - lp + up == 0)
        for s in range(T + 1):
            ls, us = soc_d[b][s]
            expr = -ls + us
            if s == 0:
                expr += -gamma[b][0] + init_dual[b]
            elif s == T:
                expr += gamma[b][T - 1]
            else:
                expr += gamma[b][s - 1] - gamma[b][s]
            m.addConstr(expr == 0)

    peak = m.addVar(lb=-d["exp_ub"], ub=d["imp_ub"], name="peak")
    valley = m.addVar(lb=-d["exp_ub"], ub=d["imp_ub"], name="valley")
    for t in range(T):
        net = v["imp"][t] - v["exp"][t]
        m.addConstr(peak >= net)
        m.addConstr(valley <= net)
    m.setObjective(PEAK_WEIGHT * peak - VALLEY_WEIGHT * valley, GRB.MINIMIZE)

    started = time.time()
    m.optimize()
    runtime = time.time() - started
    if m.SolCount == 0:
        m.computeIIS()
        m.write(str(IIS_FILE))
        raise RuntimeError(f"No feasible solution; status={m.Status}. IIS: {IIS_FILE}")

    pb_value = np.array([pb[t].X for t in range(T)])
    sol = extract_solution(d, v, pb_value)
    follower = follower_cost(sol)
    verified = verify_follower(d, pb_value)
    info = {
        "status": int(m.Status), "runtime_s": runtime,
        "mip_gap": float(m.MIPGap), "leader_objective": float(m.ObjVal),
        "follower_objective": follower, "follower_verify_gap": follower - verified,
    }
    return sol, info


def extract_solution(d, v, pb):
    imp = np.array([v["imp"][t].X for t in range(T)])
    exp = np.array([v["exp"][t].X for t in range(T)])
    sol = pd.DataFrame({
        "step": np.arange(T), "hour": d["df"]["hour"].to_numpy(float),
        "net_baseline_kw": d["base"], "pa_eur_per_kwh": d["pa"],
        "pb_eur_per_kwh": pb, "total_import_price_eur_per_kwh": d["pa"] + pb,
        "import_kw": imp, "export_kw": exp, "net_exchange_kw": imp - exp,
    })
    p_cols = []
    for b in BESS_CONFIG:
        n = tag(b)
        col = f"bess_{n}_kw"
        p_cols.append(col)
        sol[col] = [v["p"][b][t].X for t in range(T)]
        sol[f"soc_{n}_kwh"] = [v["soc"][b][t].X for t in range(T)]
        sol[f"soc_next_{n}_kwh"] = [v["soc"][b][t + 1].X for t in range(T)]
    sol["bess_total_kw"] = sol[p_cols].sum(axis=1)
    return sol


def follower_cost(sol):
    return float(np.sum(DT * ((sol["pa_eur_per_kwh"] + sol["pb_eur_per_kwh"])
                              * sol["import_kw"]
                              - sol["pb_eur_per_kwh"] * sol["export_kw"])))


def metrics(x):
    x = np.asarray(x, float)
    return {"peak_kw": x.max(), "valley_kw": x.min(),
            "peak_valley_gap_kw": x.max() - x.min(),
            "ramp_sum_kw": np.abs(np.diff(x)).sum()}


def make_reports(sol, info):
    base, opt = metrics(sol["net_baseline_kw"]), metrics(sol["net_exchange_kw"])
    curves = pd.DataFrame([{"case": "net_baseline", **base}, {"case": "stackelberg", **opt}])
    denom = max(base["peak_valley_gap_kw"], 1e-12)
    summary = pd.DataFrame([{**info,
        "peak_reduction_pct": (base["peak_kw"] - opt["peak_kw"]) / max(abs(base["peak_kw"]), 1e-12),
        "valley_fill_kw": opt["valley_kw"] - base["valley_kw"],
        "gap_reduction_pct": (base["peak_valley_gap_kw"] - opt["peak_valley_gap_kw"]) / denom,
        "pb_average": sol["pb_eur_per_kwh"].mean(),
        "pb_max_ramp": np.abs(np.diff(sol["pb_eur_per_kwh"])).max(),
        "import_kwh": sol["import_kw"].sum() * DT,
        "export_kwh": sol["export_kw"].sum() * DT}])

    checks = [["power balance", np.max(np.abs(sol["net_baseline_kw"] + sol["bess_total_kw"]
                                                - sol["net_exchange_kw"])), BALANCE_TOL],
              ["simultaneous import/export",
               int(((sol["import_kw"] > POWER_TOL) & (sol["export_kw"] > POWER_TOL)).sum()), 0],
              ["PB average deviation", abs(sol["pb_eur_per_kwh"].mean() - PB_AVG_TARGET),
               PB_AVG_TOL + PRICE_TOL],
              ["PB max ramp", np.abs(np.diff(sol["pb_eur_per_kwh"])).max(),
               PB_RAMP_LIMIT + PRICE_TOL]]
    for b, c in BESS_CONFIG.items():
        _, _, tol = terminal_bounds(c)
        end = sol[f"soc_next_{tag(b)}_kwh"].iloc[-1]
        checks.append([f"{b} terminal SOC", abs(end - c["e_init_kwh"]), tol + SOC_TOL])
    checks = pd.DataFrame(checks, columns=["check", "value", "limit"])
    checks["status"] = np.where(checks["value"] <= checks["limit"], "PASS", "FAIL")
    return curves, summary, checks


def plot_result(sol, out):
    h = sol["hour"]
    fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    ax[0].plot(h, sol["net_baseline_kw"], "--", label="PV-adjusted baseline")
    ax[0].plot(h, sol["net_exchange_kw"], label="optimized net exchange")
    ax[0].axhline(0, color="black", lw=.7); ax[0].set_ylabel("kW"); ax[0].legend()
    ax[1].step(h, sol["pa_eur_per_kwh"], where="post", label="PA")
    ax[1].step(h, sol["pb_eur_per_kwh"], where="post", label="PB")
    ax[1].step(h, sol["total_import_price_eur_per_kwh"], where="post", label="PA+PB")
    ax[1].set_ylabel("EUR/kWh"); ax[1].legend()
    ax[2].plot(h, sol["import_kw"], label="import")
    ax[2].plot(h, -sol["export_kw"], label="export (-)")
    ax[2].plot(h, sol["bess_total_kw"], "--", label="BESS (+ charge)")
    ax[2].set_ylabel("kW"); ax[2].legend()
    for b in BESS_CONFIG:
        ax[3].plot(h, sol[f"soc_{tag(b)}_kwh"], label=b)
    ax[3].set_ylabel("Energy [kWh]"); ax[3].set_xlabel("hour"); ax[3].legend()
    for a in ax: a.grid(True, alpha=.25)
    fig.tight_layout(); fig.savefig(out / "test_phase_result.png", dpi=180); plt.close(fig)


def main():
    src, out = get_paths()
    sol, info = solve(load_data(src))
    curves, summary, checks = make_reports(sol, info)
    sol.to_csv(out / "test_phase_solution.csv", index=False)
    curves.to_csv(out / "test_phase_curve_metrics.csv", index=False)
    summary.to_csv(out / "test_phase_summary.csv", index=False)
    checks.to_csv(out / "test_phase_checks.csv", index=False)
    plot_result(sol, out)
    print(f"\n[test_phase completed] Output: {out}")
    print(summary.to_string(index=False))
    failed = checks[checks["status"] == "FAIL"]
    if len(failed): print("\n[warning]\n" + failed.to_string(index=False))


# ==================== ALL ADJUSTABLE PARAMETERS ====================
FEEDER_FOLDER = "1_LV_urban_0_no_sw"
PHASE1_FOLDER = "phase1_outputs"
INPUT_FILE = "phase1_inputs.csv"
OUTPUT_FOLDER = "test_phase_outputs"
T, DT = 96, 0.25

PB_MIN, PB_MAX = 0.00, 0.20
PB_AVG_TARGET, PB_AVG_TOL = 0.05, 0.005
PB_RAMP_LIMIT = 0.05
PEAK_WEIGHT, VALLEY_WEIGHT = 1.0, 1.0

BESS_CONFIG = {
    "Bus R1": {"p_max_kw": 2.5, "e_min_kwh": 0.8, "e_max_kwh": 8.0, "e_init_kwh": 4.0},
    "Bus R2": {"p_max_kw": 1.5, "e_min_kwh": 0.5, "e_max_kwh": 5.0, "e_init_kwh": 2.5},
    "Bus R3": {"p_max_kw": 2.5, "e_min_kwh": 0.8, "e_max_kwh": 8.0, "e_init_kwh": 4.0},
}
TERMINAL_SOC_TOL_FRAC = 0.05
BESS_CYCLE_COST = 0.0
GRID_IMPORT_MAX_KW = None
GRID_EXPORT_MAX_KW = None
GRID_MARGIN_KW = 5.0

DUAL_MAX, M_SLACK_FACTOR = 100.0, 1.0
TIME_LIMIT, MIP_GAP = 1200, 0.001
OUTPUT_FLAG, NUMERIC_FOCUS, MIP_FOCUS = 1, 1, 1
HEURISTICS, CUTS, THREADS, SOLVER_SEED = 0.20, 2, 0, 1
BALANCE_TOL, SOC_TOL, POWER_TOL, PRICE_TOL = 1e-5, 1e-5, 1e-5, 1e-7
IIS_FILE = Path(__file__).resolve().parent / "test_phase_infeasible.ilp"


if __name__ == "__main__":
    main()
