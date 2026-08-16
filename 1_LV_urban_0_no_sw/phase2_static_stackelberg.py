# -*- coding: utf-8 -*-

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


def paths():
    here = Path(__file__).resolve().parent
    feeder = here if here.name == FEEDER_FOLDER else here / FEEDER_FOLDER
    src = feeder / PHASE1_FOLDER / "phase1_inputs.csv"
    out = feeder / OUTPUT_FOLDER
    out.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        raise FileNotFoundError(src)
    return src, out


def load_data(src):
    df = pd.read_csv(src).iloc[:T].copy()
    need = ["hour", "aggregate_net_baseline_kw", "pa_eur_per_kwh"]
    miss = [c for c in need if c not in df.columns]
    if miss or len(df) != T:
        raise ValueError(f"Invalid phase1_inputs.csv: missing={miss}, rows={len(df)}")

    base = df["aggregate_net_baseline_kw"].to_numpy(float)
    pa = df["pa_eur_per_kwh"].to_numpy(float)

    if base.min() < -1e-8:
        raise ValueError("Net baseline is negative. Export is not modeled here.")

    grid_ub = GRID_IMPORT_MAX_KW or (
        base.max()
        + sum(c["p_max_kw"] for c in EV_CONFIG.values())
        + sum(c["p_max_kw"] for c in BESS_CONFIG.values())
        + GRID_MARGIN_KW
    )

    return {"df": df, "base": base, "pa": pa, "grid_ub": float(grid_ub)}


def terminal_bounds(cfg):
    tol = TERMINAL_SOC_TOL_FRAC * (cfg["e_max_kwh"] - cfg["e_min_kwh"])
    lo = max(cfg["e_min_kwh"], cfg["e_init_kwh"] - tol)
    hi = min(cfg["e_max_kwh"], cfg["e_init_kwh"] + tol)
    return lo, hi, tol


def set_solver(m, mip=True, log=True):
    m.Params.OutputFlag = OUTPUT_FLAG if log else 0
    m.Params.TimeLimit = TIME_LIMIT
    m.Params.NumericFocus = NUMERIC_FOCUS
    m.Params.Seed = SOLVER_SEED
    if mip:
        m.Params.MIPGap = MIP_GAP
        m.Params.MIPFocus = MIP_FOCUS
        m.Params.Heuristics = HEURISTICS
        m.Params.Cuts = CUTS
        m.Params.Threads = THREADS


def build_primal(m, d):
    imp = m.addVars(T, lb=0, ub=d["grid_ub"], name="imp")
    ev, ch, dis, soc = {}, {}, {}, {}

    for b, c in EV_CONFIG.items():
        ev[b] = m.addVars(T, lb=0, ub=c["p_max_kw"], name=f"ev_{tag(b)}")
        m.addConstr(gp.quicksum(ev[b][t] * DT for t in range(T)) == c["energy_req_kwh"])

    for b, c in BESS_CONFIG.items():
        n = tag(b)
        ch[b] = m.addVars(T, lb=0, ub=c["p_max_kw"], name=f"ch_{n}")
        dis[b] = m.addVars(T, lb=0, ub=c["p_max_kw"], name=f"dis_{n}")
        soc[b] = m.addVars(T + 1, lb=c["e_min_kwh"], ub=c["e_max_kwh"], name=f"soc_{n}")

        m.addConstr(soc[b][0] == c["e_init_kwh"])

        for t in range(T):
            m.addConstr(
                soc[b][t + 1]
                == soc[b][t]
                + c["eta_ch"] * ch[b][t] * DT
                - dis[b][t] * DT / c["eta_dis"]
            )

        lo, hi, _ = terminal_bounds(c)
        m.addConstr(soc[b][T] >= lo)
        m.addConstr(soc[b][T] <= hi)

    for t in range(T):
        m.addConstr(
            imp[t]
            == float(d["base"][t])
            + gp.quicksum(ev[b][t] for b in EV_CONFIG)
            + gp.quicksum(ch[b][t] - dis[b][t] for b in BESS_CONFIG)
        )

    return {"imp": imp, "ev": ev, "ch": ch, "dis": dis, "soc": soc}


def follower_obj(d, v, pb):
    energy = gp.quicksum(
        DT * (float(d["pa"][t]) + pb[t]) * v["imp"][t]
        for t in range(T)
    )

    cycle = gp.quicksum(
        DT * BESS_CYCLE_COST * (v["ch"][b][t] + v["dis"][b][t])
        for b in BESS_CONFIG
        for t in range(T)
    )

    return energy + cycle


def bigm_comp(m, slack_expr, slack_ub, dual, name):
    ub = max(float(slack_ub), 1e-9)
    z = m.addVar(vtype=GRB.BINARY, name=f"z_{name}")
    slack = m.addVar(lb=0, ub=ub, name=f"slack_{name}")

    m.addConstr(slack == slack_expr)
    m.addConstr(slack <= ub * M_SLACK_FACTOR * z)
    m.addConstr(dual <= DUAL_MAX * (1 - z))


def bigm_bound(m, x, lb, ub, name):
    dl = m.addVar(lb=0, ub=DUAL_MAX, name=f"dl_{name}")
    du = m.addVar(lb=0, ub=DUAL_MAX, name=f"du_{name}")
    span = max(float(ub - lb), 1e-9)

    bigm_comp(m, x - lb, span, dl, f"{name}_lb")
    bigm_comp(m, ub - x, span, du, f"{name}_ub")

    return dl, du


def bigm_ineq(m, slack_expr, slack_ub, name):
    dual = m.addVar(lb=0, ub=DUAL_MAX, name=f"dual_{name}")
    bigm_comp(m, slack_expr, slack_ub, dual, name)
    return dual


def verify_follower(d, pb_val):
    m = gp.Model("verify_follower")
    set_solver(m, mip=False, log=False)

    v = build_primal(m, d)
    pb = {t: float(pb_val[t]) for t in range(T)}

    m.setObjective(follower_obj(d, v, pb), GRB.MINIMIZE)
    m.optimize()

    if m.SolCount == 0:
        return np.nan

    return float(m.ObjVal)


def solve_stackelberg(d):
    m = gp.Model("phase2_static_stackelberg_bigm")
    set_solver(m, mip=True, log=True)

    pb = m.addVars(T, lb=PB_MIN, ub=PB_MAX, name="pb")

    avg_pb = gp.quicksum(pb[t] for t in range(T)) / T
    m.addConstr(avg_pb >= PB_AVG_TARGET - PB_AVG_TOL)
    m.addConstr(avg_pb <= PB_AVG_TARGET + PB_AVG_TOL)

    for t in range(1, T):
        m.addConstr(pb[t] - pb[t - 1] <= PB_RAMP_LIMIT)
        m.addConstr(pb[t - 1] - pb[t] <= PB_RAMP_LIMIT)

    v = build_primal(m, d)

    lam = m.addVars(T, lb=-GRB.INFINITY, name="lam")
    ev_eq = {b: m.addVar(lb=-GRB.INFINITY, name=f"ev_eq_{tag(b)}") for b in EV_CONFIG}
    gamma = {b: m.addVars(T, lb=-GRB.INFINITY, name=f"gamma_{tag(b)}") for b in BESS_CONFIG}
    init_dual = {b: m.addVar(lb=-GRB.INFINITY, name=f"init_{tag(b)}") for b in BESS_CONFIG}

    imp_dual = [bigm_bound(m, v["imp"][t], 0, d["grid_ub"], f"imp_{t}") for t in range(T)]

    ev_dual, ch_dual, dis_dual, soc_dual, term_dual = {}, {}, {}, {}, {}

    for b, c in EV_CONFIG.items():
        ev_dual[b] = [
            bigm_bound(m, v["ev"][b][t], 0, c["p_max_kw"], f"ev_{tag(b)}_{t}")
            for t in range(T)
        ]

    for b, c in BESS_CONFIG.items():
        n = tag(b)

        ch_dual[b] = [
            bigm_bound(m, v["ch"][b][t], 0, c["p_max_kw"], f"ch_{n}_{t}")
            for t in range(T)
        ]

        dis_dual[b] = [
            bigm_bound(m, v["dis"][b][t], 0, c["p_max_kw"], f"dis_{n}_{t}")
            for t in range(T)
        ]

        soc_dual[b] = [
            bigm_bound(
                m,
                v["soc"][b][s],
                c["e_min_kwh"],
                c["e_max_kwh"],
                f"soc_{n}_{s}",
            )
            for s in range(T + 1)
        ]

        lo, hi, _ = terminal_bounds(c)

        term_dual[b] = (
            bigm_ineq(m, v["soc"][b][T] - lo, c["e_max_kwh"] - lo, f"term_low_{n}"),
            bigm_ineq(m, hi - v["soc"][b][T], hi - c["e_min_kwh"], f"term_high_{n}"),
        )

    for t in range(T):
        li, ui = imp_dual[t]
        m.addConstr(DT * (float(d["pa"][t]) + pb[t]) + lam[t] - li + ui == 0)

    for b in EV_CONFIG:
        for t in range(T):
            li, ui = ev_dual[b][t]
            m.addConstr(-lam[t] + ev_eq[b] * DT - li + ui == 0)

    for b, c in BESS_CONFIG.items():
        for t in range(T):
            lch, uch = ch_dual[b][t]
            ldis, udis = dis_dual[b][t]

            m.addConstr(
                DT * BESS_CYCLE_COST
                - lam[t]
                - c["eta_ch"] * DT * gamma[b][t]
                - lch
                + uch
                == 0
            )

            m.addConstr(
                DT * BESS_CYCLE_COST
                + lam[t]
                + DT * gamma[b][t] / c["eta_dis"]
                - ldis
                + udis
                == 0
            )

        tl, tu = term_dual[b]

        for s in range(T + 1):
            ls, us = soc_dual[b][s]
            expr = -ls + us

            if s == 0:
                expr += -gamma[b][0] + init_dual[b]
            elif s == T:
                expr += gamma[b][T - 1] - tl + tu
            else:
                expr += gamma[b][s - 1] - gamma[b][s]

            m.addConstr(expr == 0)

    peak = m.addVar(lb=0, ub=d["grid_ub"], name="peak")
    valley = m.addVar(lb=0, ub=d["grid_ub"], name="valley")

    for t in range(T):
        m.addConstr(peak >= v["imp"][t])
        m.addConstr(valley <= v["imp"][t])

    m.setObjective(PEAK_WEIGHT * peak - VALLEY_WEIGHT * valley, GRB.MINIMIZE)

    start = time.time()
    m.optimize()
    runtime = time.time() - start

    if m.SolCount == 0:
        m.computeIIS()
        m.write("phase2_static_stackelberg_bigm_infeasible.ilp")
        raise RuntimeError(f"No feasible Stackelberg solution. Status={m.Status}")

    pb_val = np.array([pb[t].X for t in range(T)])
    sol = extract_solution(d, v, pb_val)
    verify_obj = verify_follower(d, pb_val)

    info = {
        "status": int(m.Status),
        "runtime_s": runtime,
        "mip_gap": float(m.MIPGap) if m.IsMIP else np.nan,
        "leader_objective": float(m.ObjVal),
        "follower_objective": follower_cost(sol),
        "follower_verify_gap": follower_cost(sol) - verify_obj,
    }

    return sol, info


def extract_solution(d, v, pb):
    sol = pd.DataFrame({
        "step": np.arange(T),
        "hour": d["df"]["hour"].to_numpy(float),
        "net_baseline_kw": d["base"],
        "pa_eur_per_kwh": d["pa"],
        "pb_eur_per_kwh": pb,
        "total_price_eur_per_kwh": d["pa"] + pb,
        "net_exchange_kw": [v["imp"][t].X for t in range(T)],
    })

    for b in EV_CONFIG:
        sol[f"ev_{tag(b)}_kw"] = [v["ev"][b][t].X for t in range(T)]

    for b in BESS_CONFIG:
        n = tag(b)
        sol[f"bess_ch_{n}_kw"] = [v["ch"][b][t].X for t in range(T)]
        sol[f"bess_dis_{n}_kw"] = [v["dis"][b][t].X for t in range(T)]
        sol[f"soc_{n}_kwh"] = [v["soc"][b][t].X for t in range(T)]
        sol[f"soc_next_{n}_kwh"] = [v["soc"][b][t + 1].X for t in range(T)]

    ev_cols = [c for c in sol.columns if c.startswith("ev_")]
    ch_cols = [c for c in sol.columns if c.startswith("bess_ch_")]
    dis_cols = [c for c in sol.columns if c.startswith("bess_dis_")]

    sol["ev_total_kw"] = sol[ev_cols].sum(axis=1) if ev_cols else 0.0
    sol["bess_ch_total_kw"] = sol[ch_cols].sum(axis=1)
    sol["bess_dis_total_kw"] = sol[dis_cols].sum(axis=1)
    sol["bess_net_kw"] = sol["bess_ch_total_kw"] - sol["bess_dis_total_kw"]

    return sol


def follower_cost(sol):
    c_pa = np.sum(sol["pa_eur_per_kwh"] * sol["net_exchange_kw"] * DT)
    c_pb = np.sum(sol["pb_eur_per_kwh"] * sol["net_exchange_kw"] * DT)
    c_bess = np.sum(
        BESS_CYCLE_COST
        * (sol["bess_ch_total_kw"] + sol["bess_dis_total_kw"])
        * DT
    )
    return float(c_pa + c_pb + c_bess)


def curve_metrics(x):
    x = np.asarray(x, dtype=float)
    return {
        "peak_kw": x.max(),
        "valley_kw": x.min(),
        "peak_valley_gap_kw": x.max() - x.min(),
        "ramp_sum_kw": np.abs(np.diff(x)).sum(),
        "variation_kwh": np.sum(np.abs(x - x.mean()) * DT),
    }


def build_metrics(sol, info):
    base = curve_metrics(sol["net_baseline_kw"])
    opt = curve_metrics(sol["net_exchange_kw"])

    curve_df = pd.DataFrame([
        {"case": "net_baseline", **base},
        {"case": "stackelberg", **opt},
    ])

    c_pa = np.sum(sol["pa_eur_per_kwh"] * sol["net_exchange_kw"] * DT)
    c_pb = np.sum(sol["pb_eur_per_kwh"] * sol["net_exchange_kw"] * DT)
    c_bess = np.sum(
        BESS_CYCLE_COST
        * (sol["bess_ch_total_kw"] + sol["bess_dis_total_kw"])
        * DT
    )

    summary = pd.DataFrame([{
        **info,
        "peak_reduction_pct": (base["peak_kw"] - opt["peak_kw"]) / base["peak_kw"],
        "valley_fill_kw": opt["valley_kw"] - base["valley_kw"],
        "gap_reduction_pct": (
            base["peak_valley_gap_kw"] - opt["peak_valley_gap_kw"]
        ) / base["peak_valley_gap_kw"],
        "ramp_reduction_pct": (
            base["ramp_sum_kw"] - opt["ramp_sum_kw"]
        ) / base["ramp_sum_kw"],
        "variation_reduction_pct": (
            base["variation_kwh"] - opt["variation_kwh"]
        ) / base["variation_kwh"],
        "pb_average": sol["pb_eur_per_kwh"].mean(),
        "pb_max_ramp": np.abs(np.diff(sol["pb_eur_per_kwh"])).max(),
        "cost_PA": c_pa,
        "cost_PB": c_pb,
        "cost_BESS": c_bess,
        "cost_total": c_pa + c_pb + c_bess,
    }])

    return curve_df, summary


def sanity_checks(sol):
    rows = []

    balance = (
        sol["net_baseline_kw"]
        + sol["ev_total_kw"]
        + sol["bess_net_kw"]
        - sol["net_exchange_kw"]
    )

    rows.append(["power balance", np.abs(balance).max(), BALANCE_TOL])

    for b, c in EV_CONFIG.items():
        err = abs(sol[f"ev_{tag(b)}_kw"].sum() * DT - c["energy_req_kwh"])
        rows.append([f"{b} EV energy", err, ENERGY_TOL])

    for b, c in BESS_CONFIG.items():
        n = tag(b)
        _, _, tol = terminal_bounds(c)
        end_soc = sol[f"soc_next_{n}_kwh"].iloc[-1]

        simul = (
            (sol[f"bess_ch_{n}_kw"] > POWER_TOL)
            & (sol[f"bess_dis_{n}_kw"] > POWER_TOL)
        ).sum()

        rows.append([f"{b} terminal SOC", abs(end_soc - c["e_init_kwh"]), tol + SOC_TOL])
        rows.append([f"{b} simultaneous charge/discharge", simul, 0])

    avg_pb = sol["pb_eur_per_kwh"].mean()
    pb_ramp = np.abs(np.diff(sol["pb_eur_per_kwh"])).max()

    rows.append(["PB average deviation", abs(avg_pb - PB_AVG_TARGET), PB_AVG_TOL + PRICE_TOL])
    rows.append(["PB max ramp", pb_ramp, PB_RAMP_LIMIT + PRICE_TOL])

    df = pd.DataFrame(rows, columns=["check", "value", "limit"])
    df["status"] = np.where(df["value"] <= df["limit"], "PASS", "FAIL")
    return df


def plot_result(sol, out):
    h = sol["hour"]

    fig, ax = plt.subplots(5, 1, figsize=(12, 13), sharex=True)

    ax[0].plot(h, sol["net_baseline_kw"], "--", label="net baseline")
    ax[0].plot(h, sol["net_exchange_kw"], label="Stackelberg optimized")
    ax[0].set_title("Phase 2 Static Stackelberg | Big-M KKT")
    ax[0].set_ylabel("kW")
    ax[0].legend()

    ax[1].step(h, sol["pa_eur_per_kwh"], where="post", linestyle="--", label="PA")
    ax[1].step(h, sol["pb_eur_per_kwh"], where="post", linestyle=":", label="PB")
    ax[1].step(h, sol["total_price_eur_per_kwh"], where="post", label="PA + PB")
    ax[1].set_ylabel("EUR/kWh")
    ax[1].legend()

    ax[2].plot(h, sol["ev_total_kw"], label="EV charging")
    ax[2].set_ylabel("EV [kW]")
    ax[2].legend()

    ax[3].plot(h, sol["bess_ch_total_kw"], label="BESS charge")
    ax[3].plot(h, -sol["bess_dis_total_kw"], label="BESS discharge (-)")
    ax[3].plot(h, sol["bess_net_kw"], "--", label="BESS net")
    ax[3].axhline(0, linewidth=0.8)
    ax[3].set_ylabel("BESS [kW]")
    ax[3].legend()

    for b in BESS_CONFIG:
        ax[4].plot(h, sol[f"soc_{tag(b)}_kwh"], label=b)

    ax[4].set_ylabel("SOC [kWh]")
    ax[4].set_xlabel("hour")
    ax[4].legend()

    for a in ax:
        a.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out / "phase2_01_static_stackelberg_bigm.png", dpi=180)
    plt.close(fig)


def plot_metrics(curve_df, summary_df, out):
    metrics = [
        "peak_kw",
        "valley_kw",
        "peak_valley_gap_kw",
        "ramp_sum_kw",
        "variation_kwh",
    ]

    labels = [
        "Peak [kW]",
        "Valley [kW]",
        "Gap [kW]",
        "Ramp sum [kW]",
        "Variation [kWh]",
    ]

    base = curve_df[curve_df["case"] == "net_baseline"].iloc[0]
    opt = curve_df[curve_df["case"] == "stackelberg"].iloc[0]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, [base[m] for m in metrics], width, label="net baseline")
    ax.bar(x + width / 2, [opt[m] for m in metrics], width, label="Stackelberg")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Baseline vs Stackelberg indicators")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    s = summary_df.iloc[0]
    text = (
        f"Peak red.: {100*s['peak_reduction_pct']:.1f}% | "
        f"Gap red.: {100*s['gap_reduction_pct']:.1f}% | "
        f"Ramp red.: {100*s['ramp_reduction_pct']:.1f}% | "
        f"Var red.: {100*s['variation_reduction_pct']:.1f}%"
    )

    ax.text(0.5, -0.28, text, ha="center", va="top", transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(out / "phase2_02_metric_comparison.png", dpi=180)
    plt.close(fig)


def main():
    src, out = paths()
    d = load_data(src)

    sol, info = solve_stackelberg(d)
    curve_df, summary_df = build_metrics(sol, info)
    sanity_df = sanity_checks(sol)

    sol.to_csv(out / "phase2_stackelberg_solution.csv", index=False)
    curve_df.to_csv(out / "phase2_curve_metrics.csv", index=False)
    summary_df.to_csv(out / "phase2_summary_metrics.csv", index=False)
    sanity_df.to_csv(out / "phase2_sanity_checks.csv", index=False)

    plot_result(sol, out)
    plot_metrics(curve_df, summary_df, out)

    print("\n[Phase 2 completed | Big-M KKT | P2 only]")
    print(f"Output: {out}")
    print(summary_df.to_string(index=False))

    failed = sanity_df[sanity_df["status"] == "FAIL"]
    if len(failed):
        print("\n[warning] failed sanity checks:")
        print(failed.to_string(index=False))


# ==================== PARAMETERS ====================

FEEDER_FOLDER = "1_LV_urban_0_no_sw"
PHASE1_FOLDER = "phase1_outputs"
OUTPUT_FOLDER = "phase2_outputs"

T = 96
DT = 0.25

PB_MIN = 0.00
PB_MAX = 0.20
PB_AVG_TARGET = 0.05
PB_AVG_TOL = 0.005
PB_RAMP_LIMIT = 0.05

PEAK_WEIGHT = 1.0
VALLEY_WEIGHT = 1.0

EV_CONFIG = {
    "Bus R2": {
        "p_max_kw": 3.7,
        "energy_req_kwh": 8.0,
    },
}

BESS_CONFIG = {
    "Bus R1": {
        "p_max_kw": 2.5,
        "e_min_kwh": 0.8,
        "e_max_kwh": 8.0,
        "e_init_kwh": 4.0,
        "eta_ch": 0.95,
        "eta_dis": 0.95,
    },
    "Bus R2": {
        "p_max_kw": 1.5,
        "e_min_kwh": 0.5,
        "e_max_kwh": 5.0,
        "e_init_kwh": 2.5,
        "eta_ch": 0.95,
        "eta_dis": 0.95,
    },
    "Bus R3": {
        "p_max_kw": 2.5,
        "e_min_kwh": 0.8,
        "e_max_kwh": 8.0,
        "e_init_kwh": 4.0,
        "eta_ch": 0.95,
        "eta_dis": 0.95,
    },
}

TERMINAL_SOC_TOL_FRAC = 0.05
BESS_CYCLE_COST = 0.01

GRID_IMPORT_MAX_KW = None
GRID_MARGIN_KW = 5.0

DUAL_MAX = 100.0
M_SLACK_FACTOR = 1.0

TIME_LIMIT = 3000
MIP_GAP = 0.02
OUTPUT_FLAG = 1
NUMERIC_FOCUS = 1
MIP_FOCUS = 1
HEURISTICS = 0.20
CUTS = 2
THREADS = 0
SOLVER_SEED = 1

BALANCE_TOL = 1e-5
ENERGY_TOL = 1e-4
SOC_TOL = 1e-5
POWER_TOL = 1e-5
PRICE_TOL = 1e-7


if __name__ == "__main__":
    main()