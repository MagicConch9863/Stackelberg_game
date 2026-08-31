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


def get_paths():
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
        raise ValueError("Negative net baseline found. Export is not modeled.")

    grid_ub = GRID_IMPORT_MAX_KW or (
        base.max()
        + sum(c["p_max_kw"] for c in EV_CONFIG.values())
        + sum(c["p_max_kw"] for c in BESS_CONFIG.values())
        + GRID_MARGIN_KW
    )

    return {"df": df, "base": base, "pa": pa, "grid_ub": float(grid_ub)}


def terminal_band(cfg, target):
    tol = TERMINAL_SOC_TOL_FRAC * (cfg["e_max_kwh"] - cfg["e_min_kwh"])
    lo = max(cfg["e_min_kwh"], target - tol)
    hi = min(cfg["e_max_kwh"], target + tol)
    return lo, hi, tol


def set_solver(model, rhg=False, log=True):
    model.Params.OutputFlag = OUTPUT_FLAG_RHG if rhg else OUTPUT_FLAG_STATIC
    if not log:
        model.Params.OutputFlag = 0

    model.Params.TimeLimit = TIME_LIMIT_RHG if rhg else TIME_LIMIT_STATIC
    model.Params.MIPGap = MIP_GAP_RHG if rhg else MIP_GAP_STATIC
    model.Params.NumericFocus = NUMERIC_FOCUS
    model.Params.MIPFocus = MIP_FOCUS
    model.Params.Heuristics = HEURISTICS
    model.Params.Cuts = CUTS
    model.Params.Threads = THREADS
    model.Params.Seed = SOLVER_SEED


def window_ev_target(k, h, ev_remaining):
    targets = {}

    for bus, cfg in EV_CONFIG.items():
        remaining_steps = T - k
        if k + h >= T:
            target = ev_remaining[bus]
        else:
            target = ev_remaining[bus] * h / remaining_steps

        max_possible = cfg["p_max_kw"] * h * DT
        targets[bus] = min(max(target, 0.0), max_possible)

    return targets


def build_primal(model, data, k, h, init_soc, ev_remaining, terminal_mode):
    imp = model.addVars(h, lb=0, ub=data["grid_ub"], name="imp")
    ev, ch, dis, soc = {}, {}, {}, {}

    ev_target = window_ev_target(k, h, ev_remaining)

    for bus, cfg in EV_CONFIG.items():
        ev[bus] = model.addVars(h, lb=0, ub=cfg["p_max_kw"], name=f"ev_{tag(bus)}")
        model.addConstr(
            gp.quicksum(ev[bus][t] * DT for t in range(h)) == ev_target[bus],
            name=f"ev_energy_{tag(bus)}",
        )

    for bus, cfg in BESS_CONFIG.items():
        name = tag(bus)

        ch[bus] = model.addVars(h, lb=0, ub=cfg["p_max_kw"], name=f"ch_{name}")
        dis[bus] = model.addVars(h, lb=0, ub=cfg["p_max_kw"], name=f"dis_{name}")
        soc[bus] = model.addVars(
            h + 1,
            lb=cfg["e_min_kwh"],
            ub=cfg["e_max_kwh"],
            name=f"soc_{name}",
        )

        model.addConstr(soc[bus][0] == init_soc[bus])

        for t in range(h):
            model.addConstr(
                soc[bus][t + 1]
                == soc[bus][t]
                + cfg["eta_ch"] * ch[bus][t] * DT
                - dis[bus][t] * DT / cfg["eta_dis"]
            )

        if terminal_mode == "CURRENT_BAND" and k + h < T:
            lo, hi, _ = terminal_band(cfg, init_soc[bus])
            model.addConstr(soc[bus][h] >= lo)
            model.addConstr(soc[bus][h] <= hi)

        if terminal_mode == "DAILY" or k + h >= T:
            lo, hi, _ = terminal_band(cfg, cfg["e_init_kwh"])
            model.addConstr(soc[bus][h] >= lo)
            model.addConstr(soc[bus][h] <= hi)

    for t in range(h):
        model.addConstr(
            imp[t]
            == float(data["base"][k + t])
            + gp.quicksum(ev[b][t] for b in EV_CONFIG)
            + gp.quicksum(ch[b][t] - dis[b][t] for b in BESS_CONFIG)
        )

    return {"imp": imp, "ev": ev, "ch": ch, "dis": dis, "soc": soc}


def bigm_comp(model, slack_expr, slack_ub, dual, name):
    ub = max(float(slack_ub), 1e-9)
    z = model.addVar(vtype=GRB.BINARY, name=f"z_{name}")
    slack = model.addVar(lb=0, ub=ub, name=f"slack_{name}")

    model.addConstr(slack == slack_expr)
    model.addConstr(slack <= ub * M_SLACK_FACTOR * z)
    model.addConstr(dual <= DUAL_MAX * (1 - z))


def bigm_bound(model, x, lb, ub, name):
    dl = model.addVar(lb=0, ub=DUAL_MAX, name=f"dl_{name}")
    du = model.addVar(lb=0, ub=DUAL_MAX, name=f"du_{name}")
    span = max(float(ub - lb), 1e-9)

    bigm_comp(model, x - lb, span, dl, f"{name}_lb")
    bigm_comp(model, ub - x, span, du, f"{name}_ub")

    return dl, du


def bigm_ineq(model, slack_expr, slack_ub, name):
    dual = model.addVar(lb=0, ub=DUAL_MAX, name=f"dual_{name}")
    bigm_comp(model, slack_expr, slack_ub, dual, name)
    return dual


def solve_stackelberg_window(
    data,
    k,
    h,
    init_soc,
    ev_remaining,
    case_name,
    rhg=False,
    previous_pb=None,
    terminal_mode="DAILY",
):
    model = gp.Model(case_name)
    set_solver(model, rhg=rhg, log=True)

    pb = model.addVars(h, lb=PB_MIN, ub=PB_MAX, name="pb")

    model.addConstr(
        gp.quicksum(pb[t] for t in range(h)) / h >= PB_AVG_TARGET - PB_AVG_TOL
    )
    model.addConstr(
        gp.quicksum(pb[t] for t in range(h)) / h <= PB_AVG_TARGET + PB_AVG_TOL
    )

    for t in range(1, h):
        model.addConstr(pb[t] - pb[t - 1] <= PB_RAMP_LIMIT)
        model.addConstr(pb[t - 1] - pb[t] <= PB_RAMP_LIMIT)

    if previous_pb is not None:
        model.addConstr(pb[0] - previous_pb <= PB_RAMP_LIMIT)
        model.addConstr(previous_pb - pb[0] <= PB_RAMP_LIMIT)

    v = build_primal(model, data, k, h, init_soc, ev_remaining, terminal_mode)

    lam = model.addVars(h, lb=-GRB.INFINITY, name="lam")
    ev_eq = {
        b: model.addVar(lb=-GRB.INFINITY, name=f"ev_eq_{tag(b)}")
        for b in EV_CONFIG
    }
    gamma = {
        b: model.addVars(h, lb=-GRB.INFINITY, name=f"gamma_{tag(b)}")
        for b in BESS_CONFIG
    }
    init_dual = {
        b: model.addVar(lb=-GRB.INFINITY, name=f"init_{tag(b)}")
        for b in BESS_CONFIG
    }

    imp_dual = [
        bigm_bound(model, v["imp"][t], 0, data["grid_ub"], f"imp_{t}")
        for t in range(h)
    ]

    ev_dual, ch_dual, dis_dual, soc_dual, term_dual = {}, {}, {}, {}, {}

    for b, cfg in EV_CONFIG.items():
        ev_dual[b] = [
            bigm_bound(model, v["ev"][b][t], 0, cfg["p_max_kw"], f"ev_{tag(b)}_{t}")
            for t in range(h)
        ]

    for b, cfg in BESS_CONFIG.items():
        name = tag(b)

        ch_dual[b] = [
            bigm_bound(model, v["ch"][b][t], 0, cfg["p_max_kw"], f"ch_{name}_{t}")
            for t in range(h)
        ]

        dis_dual[b] = [
            bigm_bound(model, v["dis"][b][t], 0, cfg["p_max_kw"], f"dis_{name}_{t}")
            for t in range(h)
        ]

        soc_dual[b] = [
            bigm_bound(
                model,
                v["soc"][b][s],
                cfg["e_min_kwh"],
                cfg["e_max_kwh"],
                f"soc_{name}_{s}",
            )
            for s in range(h + 1)
        ]

        if terminal_mode == "CURRENT_BAND" and k + h < T:
            target = init_soc[b]
        else:
            target = cfg["e_init_kwh"]

        lo, hi, _ = terminal_band(cfg, target)

        term_dual[b] = (
            bigm_ineq(model, v["soc"][b][h] - lo, cfg["e_max_kwh"] - lo, f"term_lo_{name}"),
            bigm_ineq(model, hi - v["soc"][b][h], hi - cfg["e_min_kwh"], f"term_hi_{name}"),
        )

    for t in range(h):
        li, ui = imp_dual[t]
        model.addConstr(
            DT * (float(data["pa"][k + t]) + pb[t]) + lam[t] - li + ui == 0
        )

    for b in EV_CONFIG:
        for t in range(h):
            li, ui = ev_dual[b][t]
            model.addConstr(-lam[t] + ev_eq[b] * DT - li + ui == 0)

    for b, cfg in BESS_CONFIG.items():
        for t in range(h):
            lch, uch = ch_dual[b][t]
            ldis, udis = dis_dual[b][t]

            model.addConstr(
                DT * BESS_CYCLE_COST
                - lam[t]
                - cfg["eta_ch"] * DT * gamma[b][t]
                - lch
                + uch
                == 0
            )

            model.addConstr(
                DT * BESS_CYCLE_COST
                + lam[t]
                + DT * gamma[b][t] / cfg["eta_dis"]
                - ldis
                + udis
                == 0
            )

        tl, tu = term_dual[b]

        for s in range(h + 1):
            ls, us = soc_dual[b][s]
            expr = -ls + us

            if s == 0:
                expr += -gamma[b][0] + init_dual[b]
            elif s == h:
                expr += gamma[b][h - 1] - tl + tu
            else:
                expr += gamma[b][s - 1] - gamma[b][s]

            model.addConstr(expr == 0)

    peak = model.addVar(lb=0, ub=data["grid_ub"], name="peak")
    valley = model.addVar(lb=0, ub=data["grid_ub"], name="valley")

    for t in range(h):
        model.addConstr(peak >= v["imp"][t])
        model.addConstr(valley <= v["imp"][t])

    model.setObjective(PEAK_WEIGHT * peak - VALLEY_WEIGHT * valley, GRB.MINIMIZE)

    start = time.time()
    model.optimize()
    runtime = time.time() - start

    if model.SolCount == 0:
        model.computeIIS()
        model.write(f"{case_name}_infeasible.ilp")
        raise RuntimeError(f"{case_name} infeasible, status={model.Status}")

    pb_val = np.array([pb[t].X for t in range(h)])
    net = np.array([v["imp"][t].X for t in range(h)])

    ev_total = np.zeros(h)
    ev_bus = {}
    for b in EV_CONFIG:
        ev_bus[b] = np.array([v["ev"][b][t].X for t in range(h)])
        ev_total += ev_bus[b]

    ch_total = np.zeros(h)
    dis_total = np.zeros(h)
    ch_bus, dis_bus, soc_bus = {}, {}, {}

    for b in BESS_CONFIG:
        ch_bus[b] = np.array([v["ch"][b][t].X for t in range(h)])
        dis_bus[b] = np.array([v["dis"][b][t].X for t in range(h)])
        soc_bus[b] = np.array([v["soc"][b][s].X for s in range(h + 1)])
        ch_total += ch_bus[b]
        dis_total += dis_bus[b]

    return {
        "pb": pb_val,
        "net": net,
        "ev": ev_total,
        "ev_bus": ev_bus,
        "ch": ch_total,
        "dis": dis_total,
        "ch_bus": ch_bus,
        "dis_bus": dis_bus,
        "soc": soc_bus,
        "leader_objective": float(model.ObjVal),
        "runtime_s": runtime,
        "mip_gap": float(model.MIPGap),
    }


def build_solution(data, case, r, soc_path):
    sol = pd.DataFrame({
        "case": case,
        "step": np.arange(T),
        "hour": data["df"]["hour"].to_numpy(float),
        "net_baseline_kw": data["base"],
        "pa_eur_per_kwh": data["pa"],
        "pb_eur_per_kwh": r["pb"],
        "total_price_eur_per_kwh": data["pa"] + r["pb"],
        "net_exchange_kw": r["net"],
        "ev_total_kw": r["ev"],
        "bess_ch_total_kw": r["ch"],
        "bess_dis_total_kw": r["dis"],
        "bess_net_kw": r["ch"] - r["dis"],
    })

    for b in BESS_CONFIG:
        n = tag(b)
        sol[f"soc_{n}_kwh"] = np.array(soc_path[b][:T])
        sol[f"soc_next_{n}_kwh"] = np.array(soc_path[b][1:T + 1])

    return sol


def run_static(data):
    init_soc = {b: c["e_init_kwh"] for b, c in BESS_CONFIG.items()}
    ev_remaining = {b: c["energy_req_kwh"] for b, c in EV_CONFIG.items()}

    r = solve_stackelberg_window(
        data,
        k=0,
        h=T,
        init_soc=init_soc,
        ev_remaining=ev_remaining,
        case_name="static_full",
        rhg=False,
        terminal_mode="DAILY",
    )

    soc_path = {b: r["soc"][b] for b in BESS_CONFIG}
    sol = build_solution(data, "Static", r, soc_path)
    info = pd.DataFrame([{
        "case": "Static",
        "windows": 1,
        "runtime_s": r["runtime_s"],
        "mip_gap": r["mip_gap"],
        "leader_objective": r["leader_objective"],
    }])

    return sol, info


def run_rhg(data):
    pb, net, ev, ch, dis = [], [], [], [], []
    soc_now = {b: c["e_init_kwh"] for b, c in BESS_CONFIG.items()}
    ev_remaining = {b: c["energy_req_kwh"] for b, c in EV_CONFIG.items()}
    soc_path = {b: [soc_now[b]] for b in BESS_CONFIG}
    previous_pb = None
    log_rows = []

    k = 0
    window_id = 0

    while k < T:
        h = min(RHG_H_STEPS, T - k)
        a = min(RHG_APPLY_STEPS, h)

        r = solve_stackelberg_window(
            data,
            k=k,
            h=h,
            init_soc=soc_now,
            ev_remaining=ev_remaining,
            case_name=f"rhg_window_{window_id}",
            rhg=True,
            previous_pb=previous_pb,
            terminal_mode=RHG_TERMINAL_MODE,
        )

        pb.extend(r["pb"][:a])
        net.extend(r["net"][:a])
        ev.extend(r["ev"][:a])
        ch.extend(r["ch"][:a])
        dis.extend(r["dis"][:a])

        for b in BESS_CONFIG:
            soc_path[b].extend(r["soc"][b][1:a + 1])
            soc_now[b] = float(r["soc"][b][a])

        for b in EV_CONFIG:
            used = np.sum(r["ev_bus"][b][:a]) * DT
            ev_remaining[b] = max(0.0, ev_remaining[b] - used)

        previous_pb = float(r["pb"][a - 1])

        log_rows.append({
            "window": window_id,
            "start_step": k,
            "horizon_steps": h,
            "applied_steps": a,
            "leader_objective": r["leader_objective"],
            "runtime_s": r["runtime_s"],
            "mip_gap": r["mip_gap"],
        })

        k += a
        window_id += 1

    r_all = {
        "pb": np.array(pb),
        "net": np.array(net),
        "ev": np.array(ev),
        "ch": np.array(ch),
        "dis": np.array(dis),
    }

    sol = build_solution(data, f"RHG_H{RHG_H_STEPS}_A{RHG_APPLY_STEPS}", r_all, soc_path)

    info = pd.DataFrame([{
        "case": "RHG",
        "windows": len(log_rows),
        "runtime_s": sum(row["runtime_s"] for row in log_rows),
        "mip_gap": np.mean([row["mip_gap"] for row in log_rows]),
        "leader_objective": curve_metrics(sol["net_exchange_kw"])["gap_kw"],
    }])

    return sol, pd.DataFrame(log_rows), info


def curve_metrics(x):
    x = np.asarray(x, dtype=float)
    return {
        "peak_kw": x.max(),
        "valley_kw": x.min(),
        "gap_kw": x.max() - x.min(),
        "ramp_sum_kw": np.abs(np.diff(x)).sum(),
        "variation_kwh": np.sum(np.abs(x - x.mean()) * DT),
    }


def make_metrics(data, static_sol, rhg_sol):
    rows = [{"case": "Net baseline", **curve_metrics(data["base"])}]
    rows.append({"case": "Static", **curve_metrics(static_sol["net_exchange_kw"])})
    rows.append({"case": "RHG", **curve_metrics(rhg_sol["net_exchange_kw"])})
    return pd.DataFrame(rows)


def sanity(sol):
    rows = []
    case = sol["case"].iloc[0]

    balance = (
        sol["net_baseline_kw"]
        + sol["ev_total_kw"]
        + sol["bess_net_kw"]
        - sol["net_exchange_kw"]
    )
    rows.append([case, "power balance", np.abs(balance).max(), BALANCE_TOL])

    for b, cfg in EV_CONFIG.items():
        error = abs(sol["ev_total_kw"].sum() * DT - cfg["energy_req_kwh"])
        rows.append([case, f"{b} EV energy", error, ENERGY_TOL])

    for b, cfg in BESS_CONFIG.items():
        n = tag(b)
        _, _, tol = terminal_band(cfg, cfg["e_init_kwh"])
        end_soc = sol[f"soc_next_{n}_kwh"].iloc[-1]
        rows.append([case, f"{b} terminal SOC", abs(end_soc - cfg["e_init_kwh"]), tol + SOC_TOL])

    df = pd.DataFrame(rows, columns=["case", "check", "value", "limit"])
    df["status"] = np.where(df["value"] <= df["limit"], "PASS", "FAIL")
    return df


def plot_results(data, static_sol, rhg_sol, out):
    h = data["df"]["hour"]
    fig, ax = plt.subplots(5, 1, figsize=(12, 13), sharex=True)

    ax[0].plot(h, data["base"], "--", label="net baseline")
    ax[0].plot(h, static_sol["net_exchange_kw"], label="Static")
    ax[0].plot(h, rhg_sol["net_exchange_kw"], label="RHG")
    ax[0].set_title("Phase 3: Static Stackelberg vs RHG")
    ax[0].set_ylabel("kW")
    ax[0].legend()

    ax[1].step(h, data["pa"], where="post", linestyle="--", label="PA")
    ax[1].step(h, static_sol["total_price_eur_per_kwh"], where="post", label="PA+PB Static")
    ax[1].step(h, rhg_sol["total_price_eur_per_kwh"], where="post", label="PA+PB RHG")
    ax[1].set_ylabel("EUR/kWh")
    ax[1].legend()

    ax[2].plot(h, static_sol["ev_total_kw"], label="Static")
    ax[2].plot(h, rhg_sol["ev_total_kw"], label="RHG")
    ax[2].set_ylabel("EV [kW]")
    ax[2].legend()

    ax[3].plot(h, static_sol["bess_net_kw"], label="Static")
    ax[3].plot(h, rhg_sol["bess_net_kw"], label="RHG")
    ax[3].axhline(0, linewidth=0.8)
    ax[3].set_ylabel("BESS net [kW]")
    ax[3].legend()

    for b in BESS_CONFIG:
        ax[4].plot(h, static_sol[f"soc_{tag(b)}_kwh"], linestyle="--", label=f"Static {b}")
        ax[4].plot(h, rhg_sol[f"soc_{tag(b)}_kwh"], label=f"RHG {b}")

    ax[4].set_ylabel("SOC [kWh]")
    ax[4].set_xlabel("hour")
    ax[4].legend(ncol=2)

    for a in ax:
        a.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out / "phase3_01_static_vs_rhg.png", dpi=180)
    plt.close(fig)


def plot_metrics(metrics, out):
    cols = ["peak_kw", "valley_kw", "gap_kw", "ramp_sum_kw", "variation_kwh"]
    labels = ["Peak", "Valley", "Gap", "Ramp", "Variation"]

    x = np.arange(len(cols))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (_, row) in enumerate(metrics.iterrows()):
        ax.bar(
            x - 0.4 + width / 2 + i * width,
            [row[c] for c in cols],
            width,
            label=row["case"],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Static vs RHG indicators")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out / "phase3_02_metric_comparison.png", dpi=180)
    plt.close(fig)


def main():
    src, out = get_paths()
    data = load_data(src)

    print("[run] Static Stackelberg")
    static_sol, static_info = run_static(data)

    print("[run] RHG Stackelberg")
    rhg_sol, rhg_log, rhg_info = run_rhg(data)

    metrics = make_metrics(data, static_sol, rhg_sol)
    checks = pd.concat([sanity(static_sol), sanity(rhg_sol)], ignore_index=True)
    info = pd.concat([static_info, rhg_info], ignore_index=True)

    static_sol.to_csv(out / "phase3_static_solution.csv", index=False)
    rhg_sol.to_csv(out / "phase3_rhg_solution.csv", index=False)
    rhg_log.to_csv(out / "phase3_rhg_window_log.csv", index=False)
    metrics.to_csv(out / "phase3_metrics.csv", index=False)
    checks.to_csv(out / "phase3_sanity_checks.csv", index=False)
    info.to_csv(out / "phase3_solver_summary.csv", index=False)

    plot_results(data, static_sol, rhg_sol, out)
    plot_metrics(metrics, out)

    print("\n[Phase 3 completed]")
    print(f"Output: {out}")
    print("\n[metrics]")
    print(metrics.to_string(index=False))
    print("\n[solver summary]")
    print(info.to_string(index=False))

    bad = checks[checks["status"] == "FAIL"]
    if len(bad):
        print("\n[warning] failed sanity checks:")
        print(bad.to_string(index=False))


# ==================== PARAMETERS ====================

FEEDER_FOLDER = "1_LV_urban_0_no_sw"
PHASE1_FOLDER = "phase1_outputs"
OUTPUT_FOLDER = "phase3_outputs"

T = 96
DT = 0.25

RHG_H_STEPS = 24
RHG_APPLY_STEPS = 12
RHG_TERMINAL_MODE = "DAILY"      # "DAILY" or "CURRENT_BAND"

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

TIME_LIMIT_STATIC = 1200
TIME_LIMIT_RHG = 60
MIP_GAP_STATIC = 0.02
MIP_GAP_RHG = 0.05

OUTPUT_FLAG_STATIC = 1
OUTPUT_FLAG_RHG = 0

NUMERIC_FOCUS = 1
MIP_FOCUS = 1
HEURISTICS = 0.20
CUTS = 2
THREADS = 0
SOLVER_SEED = 1

BALANCE_TOL = 1e-5
ENERGY_TOL = 1e-4
SOC_TOL = 1e-5


if __name__ == "__main__":
    main()