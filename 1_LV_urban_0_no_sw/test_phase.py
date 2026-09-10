# -*- coding: utf-8 -*-
"""Static and receding-horizon optimistic Stackelberg test: PV + BESS, no EV."""

from pathlib import Path
import re, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB


def tag(x): return re.sub(r"[^A-Za-z0-9]+", "_", str(x)).strip("_")


def paths():
    root = Path(__file__).resolve().parent
    feeder = root if root.name == FEEDER_FOLDER else root / FEEDER_FOLDER
    src, out = feeder / PHASE1_FOLDER / INPUT_FILE, feeder / OUTPUT_FOLDER
    out.mkdir(parents=True, exist_ok=True)
    if not src.exists(): raise FileNotFoundError(f"Input not found: {src}")
    return src, out


def load_data(src):
    df = pd.read_csv(src).iloc[:T].copy()
    need = ["hour", "aggregate_net_baseline_kw", "pa_eur_per_kwh"]
    miss = [c for c in need if c not in df]
    if miss or len(df) != T: raise ValueError(f"missing={miss}, rows={len(df)}, expected={T}")
    base, pa = df[need[1]].to_numpy(float), df[need[2]].to_numpy(float)
    pmax = sum(c["p_max_kw"] for c in BESS.values())
    imp = GRID_IMPORT_MAX_KW or max(GRID_MARGIN_KW, base.max() + pmax + GRID_MARGIN_KW)
    exp = GRID_EXPORT_MAX_KW or max(GRID_MARGIN_KW, -base.min() + pmax + GRID_MARGIN_KW)
    return {"df": df, "base": base, "pa": pa, "imp_ub": float(imp), "exp_ub": float(exp)}


def terminal_bounds(c, e0):
    tol = TERMINAL_SOC_TOL_FRAC * (c["e_max_kwh"] - c["e_min_kwh"])
    return max(c["e_min_kwh"], e0 - tol), min(c["e_max_kwh"], e0 + tol)


def solver_options(m, mip=True, time_limit=None, log=True):
    m.Params.OutputFlag, m.Params.NumericFocus = (OUTPUT_FLAG if log else 0), NUMERIC_FOCUS
    m.Params.TimeLimit, m.Params.Seed, m.Params.Threads = time_limit or TIME_LIMIT, SOLVER_SEED, THREADS
    if mip:
        m.Params.MIPGap, m.Params.MIPFocus = MIP_GAP, MIP_FOCUS
        m.Params.Heuristics, m.Params.Cuts = HEURISTICS, CUTS


def primal(m, d, e0):
    h = len(d["base"])
    imp, exp = m.addVars(h, lb=0, ub=d["imp_ub"], name="imp"), m.addVars(h, lb=0, ub=d["exp_ub"], name="exp")
    p, e = {}, {}
    for b, c in BESS.items():
        n, (elo, ehi) = tag(b), terminal_bounds(c, e0[b])
        p[b] = m.addVars(h, lb=-c["p_max_kw"], ub=c["p_max_kw"], name=f"p_{n}")
        e[b] = {s: m.addVar(lb=elo if s == h else c["e_min_kwh"],
                            ub=ehi if s == h else c["e_max_kwh"], name=f"e_{n}_{s}") for s in range(h + 1)}
        m.addConstr(e[b][0] == e0[b], name=f"e_init_{n}")
        for t in range(h): m.addConstr(e[b][t + 1] == e[b][t] + DT * p[b][t], name=f"e_dyn_{n}_{t}")
    for t in range(h):
        m.addConstr(imp[t] - exp[t] - gp.quicksum(p[b][t] for b in BESS) == float(d["base"][t]), name=f"balance_{t}")
    return {"imp": imp, "exp": exp, "p": p, "e": e}


def follower_obj(d, v, pb):
    if abs(BESS_CYCLE_COST) > 1e-12: raise ValueError("Signed-power formulation requires BESS_CYCLE_COST=0.")
    return gp.quicksum(DT * ((float(d["pa"][t]) + pb[t]) * v["imp"][t] - pb[t] * v["exp"][t]) for t in range(len(d["base"])))


def comp(m, expr, ub, dual, name):
    z, s = m.addVar(vtype=GRB.BINARY, name=f"z_{name}"), m.addVar(lb=0, ub=ub, name=f"s_{name}")
    m.addConstr(s == expr); m.addConstr(s <= ub * M_SLACK_FACTOR * z); m.addConstr(dual <= DUAL_MAX * (1-z))


def bound_duals(m, x, lo, hi, name):
    dl, du = m.addVar(lb=0, ub=DUAL_MAX, name=f"dl_{name}"), m.addVar(lb=0, ub=DUAL_MAX, name=f"du_{name}")
    comp(m, x-lo, hi-lo, dl, name+"_lo"); comp(m, hi-x, hi-lo, du, name+"_hi")
    return dl, du


def convergence_callback(trace):
    def callback(model, where):
        if where != GRB.Callback.MIP: return
        incumbent = model.cbGet(GRB.Callback.MIP_OBJBST)
        bound = model.cbGet(GRB.Callback.MIP_OBJBND)
        elapsed = model.cbGet(GRB.Callback.RUNTIME)
        if abs(incumbent) >= GRB.INFINITY or abs(bound) >= GRB.INFINITY: return
        gap = abs(incumbent-bound) / max(abs(incumbent), 1e-10)
        if not trace or elapsed-trace[-1]["solve_time_s"] >= GAP_LOG_INTERVAL_S or gap < trace[-1]["gap"]-GAP_LOG_MIN_CHANGE:
            trace.append({"solve_time_s": elapsed, "incumbent": incumbent,
                          "best_bound": bound, "gap": gap})
    return callback


def solve_window(d, e0, mode, previous_pb=None, window_start=0):
    h, m = len(d["base"]), gp.Model(f"{mode}_stackelberg")
    solver_options(m, time_limit=RHG_TIME_LIMIT if mode == "rhg" else TIME_LIMIT)

    pb = m.addVars(h, lb=PB_MIN, ub=PB_MAX, name="pb")

    avg = gp.quicksum(pb[t] for t in range(h)) / h
    m.addConstr(avg >= PB_AVG_TARGET-PB_AVG_TOL); m.addConstr(avg <= PB_AVG_TARGET+PB_AVG_TOL)
    if previous_pb is not None:
        m.addConstr(pb[0]-previous_pb <= PB_RAMP_LIMIT); m.addConstr(previous_pb-pb[0] <= PB_RAMP_LIMIT)
    for t in range(1, h):
        m.addConstr(pb[t]-pb[t-1] <= PB_RAMP_LIMIT); m.addConstr(pb[t-1]-pb[t] <= PB_RAMP_LIMIT)
    v, lam = primal(m, d, e0), m.addVars(h, lb=-GRB.INFINITY, name="lambda")
    gamma = {b: m.addVars(h, lb=-GRB.INFINITY, name=f"gamma_{tag(b)}") for b in BESS}
    init_dual = {b: m.addVar(lb=-GRB.INFINITY, name=f"init_{tag(b)}") for b in BESS}
    imp_d = [bound_duals(m, v["imp"][t], 0, d["imp_ub"], f"imp_{t}") for t in range(h)]
    exp_d = [bound_duals(m, v["exp"][t], 0, d["exp_ub"], f"exp_{t}") for t in range(h)]
    p_d, e_d = {}, {}

    for b, c in BESS.items():
        p_d[b] = [bound_duals(m, v["p"][b][t], -c["p_max_kw"], c["p_max_kw"], f"p_{tag(b)}_{t}") for t in range(h)]
        elo, ehi = terminal_bounds(c, e0[b]); e_d[b] = []
    for t in range(h):
        li, ui = imp_d[t]; le, ue = exp_d[t]
        m.addConstr(DT*(float(d["pa"][t])+pb[t])+lam[t]-li+ui == 0)
        m.addConstr(-DT*pb[t]-lam[t]-le+ue == 0)
    for b in BESS:
        for t in range(h):
            lo, up = p_d[b][t]; m.addConstr(-lam[t]-DT*gamma[b][t]-lo+up == 0)
        for s in range(h+1):
            lo, up = e_d[b][s]; expr = -lo+up
            if s == 0: expr += -gamma[b][0]+init_dual[b]
            elif s == h: expr += gamma[b][h-1]
            else: expr += gamma[b][s-1]-gamma[b][s]
            m.addConstr(expr == 0)


    peak = m.addVar(lb=-d["exp_ub"], ub=d["imp_ub"], name="peak")
    valley = m.addVar(lb=-d["exp_ub"], ub=d["imp_ub"], name="valley")

    for t in range(h): m.addConstr(peak >= v["imp"][t]-v["exp"][t]); m.addConstr(valley <= v["imp"][t]-v["exp"][t])
    m.setObjective(PEAK_WEIGHT*peak-VALLEY_WEIGHT*valley, GRB.MINIMIZE)
    
    trace, tic = [], time.time()
    m.optimize(convergence_callback(trace)); runtime = time.time()-tic
    if not m.SolCount: raise RuntimeError(f"{mode}: no solution, status={m.Status}")
    final = {"solve_time_s": runtime, "incumbent": m.ObjVal,
             "best_bound": m.ObjBound, "gap": m.MIPGap}
    if not trace or abs(trace[-1]["solve_time_s"]-runtime) > 1e-6: trace.append(final)
    history = pd.DataFrame(trace)
    history.insert(0, "window_start", window_start); history.insert(0, "mode", mode)
    info = {"status": m.Status, "runtime_s": runtime, "mip_gap": m.MIPGap,
            "leader_objective": m.ObjVal}
    return extract(d, v, np.array([pb[t].X for t in range(h)])), info, history


def extract(d, v, pb):
    h = len(d["base"]); imp = np.array([v["imp"][t].X for t in range(h)]); exp = np.array([v["exp"][t].X for t in range(h)])
    sol = pd.DataFrame({"net_baseline_kw": d["base"], "pa_eur_per_kwh": d["pa"], "pb_eur_per_kwh": pb,
                        "total_import_price_eur_per_kwh": d["pa"]+pb, "import_kw": imp, "export_kw": exp, "net_exchange_kw": imp-exp})
    pcs = []
    for b in BESS:
        n, col = tag(b), f"bess_{tag(b)}_kw"; pcs.append(col)
        sol[col] = [v["p"][b][t].X for t in range(h)]; sol[f"energy_{n}_kwh"] = [v["e"][b][t].X for t in range(h)]
        sol[f"energy_next_{n}_kwh"] = [v["e"][b][t+1].X for t in range(h)]
    sol["bess_total_kw"] = sol[pcs].sum(axis=1)
    return sol


def initial_energy(): return {b: c["e_init_kwh"] for b, c in BESS.items()}


def solve_static(d):
    sol, info, history = solve_window(d, initial_energy(), "static")
    sol.insert(0, "hour", d["df"]["hour"].to_numpy(float)); sol.insert(0, "step", np.arange(T)); sol["mode"] = "static"
    return sol, info, history


def forecast_slice(d, start):
    # Shorten only the final horizon so its terminal bound applies at day end.
    idx = np.arange(start, min(T, start+RHG_PREDICTION_STEPS))
    return {"base": d["base"][idx], "pa": d["pa"][idx], "imp_ub": d["imp_ub"], "exp_ub": d["exp_ub"]}


def solve_rhg(d):
    e0, previous_pb, blocks, logs, histories = initial_energy(), None, [], [], []
    for start in range(0, T, RHG_EXECUTION_STEPS):
        pred = forecast_slice(d, start)
        win, info, history = solve_window(pred, e0, "rhg", previous_pb, start)
        keep = min(RHG_EXECUTION_STEPS, T-start); block = win.iloc[:keep].copy()
        block.insert(0, "hour", d["df"]["hour"].iloc[start:start+keep].to_numpy(float)); block.insert(0, "step", np.arange(start, start+keep))
        block["mode"], block["rhg_window_start"] = "rhg", start; blocks.append(block)
        previous_pb = float(block["pb_eur_per_kwh"].iloc[-1])
        e0 = {b: float(block[f"energy_next_{tag(b)}_kwh"].iloc[-1]) for b in BESS}
        logs.append({"window_start": start, "executed_steps": keep, **info}); histories.append(history)
    return pd.concat(blocks, ignore_index=True), pd.DataFrame(logs), pd.concat(histories, ignore_index=True)


def curve_metrics(x):
    x = np.asarray(x, float)
    return {"peak_kw": x.max(), "valley_kw": x.min(), "peak_valley_gap_kw": np.ptp(x),
            "ramp_sum_kw": np.abs(np.diff(x)).sum(), "ramp_max_kw": np.abs(np.diff(x)).max()}


def performance(d, static, rhg, static_info, rhg_log):
    rows = [{"case": "baseline", **curve_metrics(d["base"]), "runtime_s": 0.0, "mip_gap": np.nan}]
    for name, sol, runtime, gap in [("static", static, static_info["runtime_s"], static_info["mip_gap"]),
                                     ("rhg", rhg, rhg_log["runtime_s"].sum(), rhg_log["mip_gap"].max())]:
        r = {"case": name, **curve_metrics(sol["net_exchange_kw"]), "runtime_s": runtime, "mip_gap": gap,
             "import_kwh": sol["import_kw"].sum()*DT, "export_kwh": sol["export_kw"].sum()*DT,
             "pb_mean": sol["pb_eur_per_kwh"].mean(), "pb_max_ramp": np.abs(np.diff(sol["pb_eur_per_kwh"])).max()}
        r["follower_cost_eur"] = np.sum(DT*((sol["pa_eur_per_kwh"]+sol["pb_eur_per_kwh"])*sol["import_kw"]-sol["pb_eur_per_kwh"]*sol["export_kw"]))
        rows.append(r)
    out = pd.DataFrame(rows); bgap = out.loc[0, "peak_valley_gap_kw"]
    out["gap_reduction_vs_baseline_pct"] = 100*(bgap-out["peak_valley_gap_kw"])/max(bgap, 1e-12)
    return out


def plot_detail(sol, filename, title):
    h = sol["hour"]; fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    ax[0].plot(h, sol["net_baseline_kw"], "--", label="PV-adjusted baseline"); ax[0].plot(h, sol["net_exchange_kw"], label=title)
    ax[1].step(h, sol["pa_eur_per_kwh"], where="post", label="PA"); ax[1].step(h, sol["pb_eur_per_kwh"], where="post", label="PB"); ax[1].step(h, sol["total_import_price_eur_per_kwh"], where="post", label="PA+PB")
    ax[2].plot(h, sol["import_kw"], label="import"); ax[2].plot(h, -sol["export_kw"], label="export (-)"); ax[2].plot(h, sol["bess_total_kw"], "--", label="BESS (+ charge)")
    for b in BESS: ax[3].plot(h, sol[f"energy_{tag(b)}_kwh"], label=b)
    for a, y in zip(ax, ["kW", "EUR/kWh", "kW", "Energy [kWh]"]): a.set_ylabel(y); a.grid(True, alpha=.25); a.legend()
    ax[0].set_title(title); ax[3].set_xlabel("hour"); fig.tight_layout(); fig.savefig(filename, dpi=180); plt.close(fig)


def plot_comparison(d, static, rhg, filename):
    h = d["df"]["hour"]; fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(h, d["base"], "--", color="0.45", label="baseline"); ax[0].plot(h, static["net_exchange_kw"], label="Static"); ax[0].plot(h, rhg["net_exchange_kw"], label="RHG")
    ax[1].step(h, static["pb_eur_per_kwh"], where="post", label="PB Static"); ax[1].step(h, rhg["pb_eur_per_kwh"], where="post", label="PB RHG")
    for a, y in zip(ax, ["Net exchange [kW]", "PB [EUR/kWh]"]): a.set_ylabel(y); a.grid(True, alpha=.25); a.legend()
    ax[1].set_xlabel("hour"); fig.tight_layout(); fig.savefig(filename, dpi=180); plt.close(fig)


def plot_baseline(d, filename):
    fig, ax = plt.subplots(figsize=(12, 4)); ax.plot(d["df"]["hour"], d["base"], "--", lw=1.8, label="PV-adjusted baseline")
    ax.axhline(0, color="black", lw=.7); ax.set(xlabel="hour", ylabel="Net exchange [kW]", title="Original baseline"); ax.grid(True, alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(filename, dpi=180); plt.close(fig)


def plot_gap(static_hist, rhg_hist, filename):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    s = static_hist.copy(); s["gap_plot"] = np.maximum(100*s["gap"], GAP_PLOT_FLOOR_PCT)
    ax[0].semilogy(s["solve_time_s"], s["gap_plot"], marker="o", ms=3, label="Static")
    for start, g in rhg_hist.groupby("window_start"):
        g = g.copy(); g["gap_plot"] = np.maximum(100*g["gap"], GAP_PLOT_FLOOR_PCT)
        ax[1].semilogy(g["solve_time_s"], g["gap_plot"], marker=".", label=f"window {int(start)}")
    for a, title in zip(ax, ["Static convergence", "RHG convergence by independent window"]):
        a.axhline(100*MIP_GAP, color="red", ls="--", lw=1, label="target gap")
        a.set(xlabel="Solver time [s]", ylabel="MIP gap [%]", title=title)
        a.set_xscale("symlog", linthresh=1.0); a.grid(True, which="both", alpha=.25); a.legend(ncol=3)
    fig.tight_layout(); fig.savefig(filename, dpi=180); plt.close(fig)


def main():
    src, out = paths(); d = load_data(src)
    static, sinfo, shist = solve_static(d); rhg, rlog, rhist = solve_rhg(d)
    perf = performance(d, static, rhg, sinfo, rlog)
    static.to_csv(out/"static_solution.csv", index=False); rhg.to_csv(out/"rhg_solution.csv", index=False)
    rlog.to_csv(out/"rhg_window_performance.csv", index=False); perf.to_csv(out/"static_rhg_performance.csv", index=False)
    convergence = pd.concat([shist, rhist], ignore_index=True)
    convergence.to_csv(out/"solver_gap_history.csv", index=False)
    plot_detail(static, out/"01_static.png", "Static optimized net exchange")
    plot_detail(rhg, out/"02_plus_RHG.png", f"RHG: predict {RHG_PREDICTION_STEPS}, execute {RHG_EXECUTION_STEPS}")
    plot_comparison(d, static, rhg, out/"03_static_vs_RHG.png"); plot_baseline(d, out/"04_original_baseline.png")
    plot_gap(shist, rhist, out/"05_gap_vs_time.png")
    print(f"\n[completed] {out}\n{perf.to_string(index=False)}")


# ========================== ALL ADJUSTABLE PARAMETERS ==========================
FEEDER_FOLDER, PHASE1_FOLDER = "1_LV_urban_0_no_sw", "phase1_outputs"
INPUT_FILE, OUTPUT_FOLDER = "phase1_inputs.csv", "test_phase_outputs"
T, DT = 96, 0.25
RHG_PREDICTION_STEPS, RHG_EXECUTION_STEPS = 24, 12
PB_MIN, PB_MAX = 0.00, 0.20
PB_AVG_TARGET, PB_AVG_TOL, PB_RAMP_LIMIT = 0.05, 0.005, 0.05
PEAK_WEIGHT, VALLEY_WEIGHT = 1.0, 1.0
BESS = {
    "Bus R1": {"p_max_kw": 2.5, "e_min_kwh": 0.8, "e_max_kwh": 8.0, "e_init_kwh": 4.0},
    "Bus R2": {"p_max_kw": 1.5, "e_min_kwh": 0.5, "e_max_kwh": 5.0, "e_init_kwh": 2.5},
    "Bus R3": {"p_max_kw": 2.5, "e_min_kwh": 0.8, "e_max_kwh": 8.0, "e_init_kwh": 4.0},
}
TERMINAL_SOC_TOL_FRAC, BESS_CYCLE_COST = 0.05, 0.0
GRID_IMPORT_MAX_KW = GRID_EXPORT_MAX_KW = None
GRID_MARGIN_KW, DUAL_MAX, M_SLACK_FACTOR = 5.0, 100.0, 1.0
TIME_LIMIT, RHG_TIME_LIMIT, MIP_GAP = 6400, 3600, 0.01
OUTPUT_FLAG, NUMERIC_FOCUS, MIP_FOCUS = 1, 1, 1
HEURISTICS, CUTS, THREADS, SOLVER_SEED = 0.20, 2, 0, 1
GAP_LOG_INTERVAL_S, GAP_LOG_MIN_CHANGE, GAP_PLOT_FLOOR_PCT = 1.0, 1e-4, 1e-6

if __name__ == "__main__": main()
