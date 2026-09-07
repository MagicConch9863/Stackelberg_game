from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time

import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_core.phase1_closed_loop import (
    START_DATE, SIM_STEPS, _forecast, _load_period, _metrics,
)
from experiment_core.stackelberg_mpc import BUS_CONFIG, BUS_ORDER, DT, HORIZON, PHASE4_SETTINGS
from experiment_core.ol_experiment import solve_day_with_feasibility_focus

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE4 = PROJECT_ROOT / "stackelberg_scenario_analysis" / "phase4"
if str(PHASE4) not in sys.path:
    sys.path.insert(0, str(PHASE4))
from p4_s3_stackelberg import solve_day as solve_stackelberg_day

ROOT = Path(__file__).resolve().parents[1]
FLOW_TIE_BREAK = float(PHASE4_SETTINGS["FLOW_TIE_BREAK_EUR_PER_KWH"])
EXPERIMENTS = {
    "t3": ("iid", ROOT / "t3_iid_ol_mpc" / "results_online_2days"),
    "t4": ("ar1", ROOT / "t4_ar1_ol_mpc" / "results_online_2days"),
}


def _soc_targets(bus):
    cfg = BUS_CONFIG[bus]
    tol = PHASE4_SETTINGS["TERMINAL_ENERGY_TOL_FRAC"] * (cfg["bess_e_max"] - cfg["bess_e_min"])
    return max(cfg["bess_e_min"], cfg["bess_e_init"] - tol), min(cfg["bess_e_max"], cfg["bess_e_init"] + tol)


def _solve_follower(bus, current_energy, baseline, pa, pb, origin):
    h = len(baseline)
    cfg = BUS_CONFIG[bus]
    model = gp.Model(f"{bus.replace(' ', '_')}_{origin}")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    bess = model.addVars(h, lb=-cfg["bess_kw"], ub=cfg["bess_kw"], name="bess")
    imp = model.addVars(h, lb=0.0, ub=PHASE4_SETTINGS["GRID_BOUND_KW"], name="imp")
    exp = model.addVars(h, lb=0.0, ub=PHASE4_SETTINGS["GRID_BOUND_KW"], name="exp")
    energy = model.addVars(h + 1, lb=cfg["bess_e_min"], ub=cfg["bess_e_max"], name="energy")
    model.addConstr(energy[0] == current_energy)
    for j in range(h):
        model.addConstr(imp[j] - exp[j] == baseline[j] + bess[j])
        model.addConstr(energy[j + 1] == energy[j] + DT * bess[j])
    end = origin + h
    target_low, target_high = _soc_targets(bus)
    first_boundary = ((origin // 96) + 1) * 96
    for boundary in range(first_boundary, end + 1, 96):
        j = boundary - origin
        model.addConstr(energy[j] >= target_low)
        model.addConstr(energy[j] <= target_high)
    next_boundary = ((end + 95) // 96) * 96
    if next_boundary != end:
        possible = (next_boundary - end) * DT * cfg["bess_kw"]
        model.addConstr(energy[h] >= max(cfg["bess_e_min"], target_low - possible))
        model.addConstr(energy[h] <= min(cfg["bess_e_max"], target_high + possible))
    model.setObjective(gp.quicksum(DT * ((pa[j] + pb[j] + FLOW_TIE_BREAK) * imp[j]
                              + (-pa[j] + FLOW_TIE_BREAK) * exp[j]) for j in range(h)), GRB.MINIMIZE)
    started = time.perf_counter()
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Follower {bus}, origin {origin}: status={model.Status}")
    return np.array([bess[j].X for j in range(h)]), np.array([energy[j].X for j in range(h + 1)]), time.perf_counter() - started


def _day_ahead_pb(name, scenario, output, actual, pa, timestamps):
    path = output / "fixed_pb_plans.csv"
    if path.exists():
        saved = pd.read_csv(path)
        if len(saved) == SIM_STEPS + 96:
            print(f"[{name}] Reusing cached PB plans (including forecast buffer day).", flush=True)
            return saved.pb_eur_per_kwh.to_numpy(float), saved
    settings = deepcopy(PHASE4_SETTINGS)
    settings.update({"SOLVER_OUTPUT": 1, "TIME_LIMIT_S_PER_DAY": 1200, "MIP_GAP": 0.05})
    rows, values = [], []
    for day in range(3):
        origin = day * 96
        forecast = _forecast(actual, scenario, origin, horizon=96)
        print(f"\n[{name}] Day-ahead Stackelberg PB {day+1}/3: {timestamps[origin].date()}", flush=True)
        started = time.perf_counter()
        _, pb, gap = solve_day_with_feasibility_focus(
            solve_stackelberg_day, timestamps[origin], forecast["load"], forecast["pv"],
            pa[origin:origin + 96], settings)
        elapsed = time.perf_counter() - started
        values.extend(pb)
        rows.extend({"step": origin+j, "timestamp": timestamps[origin+j], "date": str(timestamps[origin].date()),
                     "pb_eur_per_kwh": pb[j], "ol_mip_gap": gap, "ol_runtime_s": elapsed}
                    for j in range(96))
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"[{name}] PB day {day+1} saved: gap={100*gap:.2f}%, runtime={elapsed:.1f}s", flush=True)
    return np.asarray(values), pd.DataFrame(rows)


def _plot(name, frame, forecasts, output):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    x = frame.step * DT
    axes[0, 0].plot(x, frame.uncontrolled_net_kw, color="#777777", ls=":", lw=1, label="Uncontrolled")
    axes[0, 0].plot(x, frame.realized_net_kw, color="#0072B2", lw=1.1, label="MPC realization")
    axes[0, 0].set_title("Grid exchange"); axes[0, 0].legend(frameon=False)
    one = forecasts[forecasts.lead == 1].groupby("target_step", as_index=False).agg(
        predicted=("forecast_baseline_kw", "sum"), realized=("actual_baseline_kw", "sum"))
    axes[0, 1].plot(one.target_step*DT, one.realized, color="#0072B2", lw=1, label="Realization")
    axes[0, 1].plot(one.target_step*DT, one.predicted, color="#E69F00", ls="--", lw=1, label="Updated prediction")
    axes[0, 1].set_title("One-step-ahead prediction"); axes[0, 1].legend(frameon=False)
    axes[1, 0].plot(x, frame.bess_kw, color="#009E73", lw=1); axes[1, 0].axhline(0, color=".4", lw=.5)
    axes[1, 0].set_title("Aggregate BESS action")
    axes[1, 1].step(x, frame.pb_eur_per_kwh, where="post", color="#CC79A7", lw=1)
    axes[1, 1].set_title("Fixed day-ahead PB")
    for ax in axes.flat:
        ax.set_xlabel("Time (h)"); ax.grid(True, ls="--", alpha=.3, lw=.5); ax.set_xlim(0,48)
    axes[0,0].set_ylabel("Power (kW)"); axes[0,1].set_ylabel("Net load (kW)")
    axes[1,0].set_ylabel("Power (kW)"); axes[1,1].set_ylabel("PB (EUR/kWh)")
    fig.suptitle(f"{name.upper()}: fixed OL tariff + 4 h rolling BESS MPC", fontsize=13)
    fig.tight_layout(); fig.savefig(output / "mpc_overview_2days.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def run(name):
    scenario, output = EXPERIMENTS[name]
    output.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] Loading continuous SimBench realization...", flush=True)
    _, actual, pa, timestamps = _load_period()
    pb, _ = _day_ahead_pb(name, scenario, output, actual, pa, timestamps)
    energy = {b: float(BUS_CONFIG[b]["bess_e_init"]) for b in BUS_ORDER}
    series, solver, forecast_rows = [], [], []
    for origin in range(SIM_STEPS):
        fc = _forecast(actual, scenario, origin)
        actions = {}; step_runtime = 0.0
        for bus in BUS_ORDER:
            base = fc["load"][bus] - fc["pv"][bus]
            plan, epath, runtime = _solve_follower(bus, energy[bus], base,
                pa[origin:origin+HORIZON], pb[origin:origin+HORIZON], origin)
            actions[bus] = float(plan[0]); energy[bus] = float(epath[1]); step_runtime += runtime
            for lead in range(HORIZON):
                forecast_rows.append({"origin_step":origin,"target_step":origin+lead,"lead":lead,
                    "origin_timestamp":timestamps[origin],"target_timestamp":timestamps[origin+lead],"bus":bus,
                    "forecast_baseline_kw":base[lead],
                    "actual_baseline_kw":actual["load"][bus][origin+lead]-actual["pv"][bus][origin+lead]})
        bess = sum(actions.values())
        uncontrolled = sum(actual["load"][b][origin]-actual["pv"][b][origin] for b in BUS_ORDER)
        series.append({"step":origin,"timestamp":timestamps[origin],"uncontrolled_net_kw":uncontrolled,
                       "realized_net_kw":uncontrolled+bess,"bess_kw":bess,"pb_eur_per_kwh":pb[origin],
                       **{f"soc_{b.replace(' ','_').lower()}_kwh":energy[b] for b in BUS_ORDER}})
        solver.append({"step":origin,"timestamp":timestamps[origin],"runtime_s":step_runtime,"all_optimal":True})
        print(f"[{name}] MPC {origin+1:03d}/{SIM_STEPS} | {timestamps[origin]} | net={uncontrolled+bess:.3f} kW | BESS={bess:+.3f} kW | solve={step_runtime:.3f}s", flush=True)
        pd.DataFrame(series).to_csv(output/"mpc_timeseries.partial.csv",index=False)
        pd.DataFrame(solver).to_csv(output/"solver_log.partial.csv",index=False)
        pd.DataFrame(forecast_rows).to_csv(output/"rolling_forecasts.partial.csv",index=False)
    frame=pd.DataFrame(series); forecasts=pd.DataFrame(forecast_rows)
    frame.to_csv(output/"mpc_timeseries.csv",index=False); forecasts.to_csv(output/"rolling_forecasts.csv",index=False)
    pd.DataFrame(solver).to_csv(output/"solver_log.csv",index=False)
    _metrics(name,frame).to_csv(output/"period_metrics.csv",index=False)
    _plot(name,frame,forecasts,output)
    print(f"[{name}] Complete -> {output}",flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--experiment",choices=["t3","t4","all"],default="all")
    args=parser.parse_args()
    for name in EXPERIMENTS if args.experiment=="all" else (args.experiment,): run(name)


if __name__ == "__main__": main()
