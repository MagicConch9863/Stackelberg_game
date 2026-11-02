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
from soc_soft_band import add_soft_soc_band

ROOT = Path(__file__).resolve().parents[1]
FLOW_TIE_BREAK = float(PHASE4_SETTINGS["FLOW_TIE_BREAK_EUR_PER_KWH"])
EXPERIMENTS = {
    "t3": ("iid", ROOT / "t3_iid_ol_mpc" / "results_soft_soc_3days"),
    "t4": ("ar1", ROOT / "t4_ar1_ol_mpc" / "results_soft_soc_3days"),
}


def _solve_follower(bus, current_energy, baseline, pa, pb, origin):
    h = len(baseline)
    cfg = BUS_CONFIG[bus]
    model = gp.Model(f"{bus.replace(' ', '_')}_{origin}")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    degradation_rate = float(PHASE4_SETTINGS.get("BESS_DEGRADATION_EUR_PER_KWH", 0.0))
    if degradation_rate > 0.0:
        charge = model.addVars(h, lb=0.0, ub=cfg["bess_kw"], name="charge")
        discharge = model.addVars(h, lb=0.0, ub=cfg["bess_kw"], name="discharge")
        bess = {j: charge[j] - discharge[j] for j in range(h)}
    else:
        charge = discharge = None
        bess = model.addVars(h, lb=-cfg["bess_kw"], ub=cfg["bess_kw"], name="bess")
    imp = model.addVars(h, lb=0.0, ub=PHASE4_SETTINGS["GRID_BOUND_KW"], name="imp")
    exp = model.addVars(h, lb=0.0, ub=PHASE4_SETTINGS["GRID_BOUND_KW"], name="exp")
    energy = model.addVars(h + 1, lb=cfg["bess_e_min"], ub=cfg["bess_e_max"], name="energy")
    model.addConstr(energy[0] == current_energy)
    for j in range(h):
        model.addConstr(imp[j] - exp[j] == baseline[j] + bess[j])
        model.addConstr(energy[j + 1] == energy[j] + DT * bess[j])
    soc_penalty, _, _ = add_soft_soc_band(
        model, energy, h, cfg, PHASE4_SETTINGS, bus.replace(" ", "_")
    )
    energy_cost = gp.quicksum(DT * ((pa[j] + pb[j] + FLOW_TIE_BREAK) * imp[j]
                              + (-pa[j] + FLOW_TIE_BREAK) * exp[j]) for j in range(h))
    degradation_cost = (
        gp.quicksum(DT * degradation_rate * (charge[j] + discharge[j]) for j in range(h))
        if charge is not None else 0.0
    )
    model.setObjective(energy_cost + soc_penalty + degradation_cost, GRB.MINIMIZE)
    started = time.perf_counter()
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Follower {bus}, origin {origin}: status={model.Status}")
    if charge is not None:
        bess_values = np.array([charge[j].X - discharge[j].X for j in range(h)])
    else:
        bess_values = np.array([bess[j].X for j in range(h)])
    return bess_values, np.array([energy[j].X for j in range(h + 1)]), time.perf_counter() - started


def _add_follower_problem(model, bus, initial_energy, baseline, pa, pb, prefix):
    """Add one Follower LP and return its variables and complete objective."""
    h = len(baseline)
    cfg = BUS_CONFIG[bus]
    charge = model.addVars(h, lb=0.0, ub=cfg["bess_kw"], name=f"charge_{prefix}")
    discharge = model.addVars(h, lb=0.0, ub=cfg["bess_kw"], name=f"discharge_{prefix}")
    imp = model.addVars(h, lb=0.0, ub=PHASE4_SETTINGS["GRID_BOUND_KW"], name=f"imp_{prefix}")
    exp = model.addVars(h, lb=0.0, ub=PHASE4_SETTINGS["GRID_BOUND_KW"], name=f"exp_{prefix}")
    energy = model.addVars(h + 1, lb=cfg["bess_e_min"], ub=cfg["bess_e_max"], name=f"energy_{prefix}")
    model.addConstr(energy[0] == initial_energy)
    for step in range(h):
        model.addConstr(imp[step] - exp[step] == baseline[step] + charge[step] - discharge[step])
        model.addConstr(energy[step + 1] == energy[step] + DT * (charge[step] - discharge[step]))
    soc_penalty, _, _ = add_soft_soc_band(model, energy, h, cfg, PHASE4_SETTINGS, prefix)
    degradation = float(PHASE4_SETTINGS.get("BESS_DEGRADATION_EUR_PER_KWH", 0.0))
    energy_cost = gp.quicksum(
        DT * ((pa[j] + pb[j] + FLOW_TIE_BREAK) * imp[j]
              + (-pa[j] + FLOW_TIE_BREAK) * exp[j]) for j in range(h)
    )
    degradation_cost = gp.quicksum(
        DT * degradation * (charge[j] + discharge[j]) for j in range(h)
    )
    return {"charge": charge, "discharge": discharge, "imp": imp, "exp": exp,
            "energy": energy, "cost": energy_cost + soc_penalty + degradation_cost}


def _follower_minimum_cost(bus, initial_energy, baseline, pa, pb, origin):
    model = gp.Model(f"primary_{bus.replace(' ', '_')}_{origin}")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    item = _add_follower_problem(model, bus, initial_energy, baseline, pa, pb, "primary")
    model.setObjective(item["cost"], GRB.MINIMIZE)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Primary Follower {bus}, origin {origin}: status={model.Status}")
    return float(model.ObjVal)


def solve_optimistic_followers(energies, forecast, pa, pb, origin,
                               previous_realized_net=None,
                               absolute_tolerance=1e-7, relative_tolerance=1e-6):
    """Select the Leader-best response among cost-optimal fixed-price Followers."""
    started = time.perf_counter()
    baselines = {bus: forecast["load"][bus] - forecast["pv"][bus] for bus in BUS_ORDER}
    primary = {
        bus: _follower_minimum_cost(bus, energies[bus], baselines[bus], pa, pb, origin)
        for bus in BUS_ORDER
    }
    h = len(pa)
    model = gp.Model(f"optimistic_followers_{origin}")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    followers = {}
    for bus in BUS_ORDER:
        tag = bus.replace(" ", "_").lower()
        item = _add_follower_problem(model, bus, energies[bus], baselines[bus], pa, pb, tag)
        tolerance = max(absolute_tolerance, relative_tolerance * max(1.0, abs(primary[bus])))
        model.addConstr(item["cost"] <= primary[bus] + tolerance, name=f"optimal_cost_{tag}")
        followers[bus] = item
    aggregate = [gp.quicksum(followers[b]["imp"][j] - followers[b]["exp"][j]
                             for b in BUS_ORDER) for j in range(h)]
    peak = model.addVar(lb=-GRB.INFINITY, name="peak")
    valley = model.addVar(lb=-GRB.INFINITY, name="valley")
    for value in aggregate:
        model.addConstr(peak >= value); model.addConstr(valley <= value)
    ramp_weight = float(PHASE4_SETTINGS.get("LEADER_RAMP_WEIGHT", 0.0))
    ramp_terms = []
    if ramp_weight > 0.0 and previous_realized_net is not None:
        first = model.addVar(lb=0.0, name="grid_ramp_0")
        model.addConstr(first >= aggregate[0] - previous_realized_net)
        model.addConstr(first >= previous_realized_net - aggregate[0]); ramp_terms.append(first)
    if ramp_weight > 0.0:
        for j in range(1, h):
            term = model.addVar(lb=0.0, name=f"grid_ramp_{j}")
            model.addConstr(term >= aggregate[j] - aggregate[j - 1])
            model.addConstr(term >= aggregate[j - 1] - aggregate[j]); ramp_terms.append(term)
    model.setObjective(peak - valley + ramp_weight * gp.quicksum(ramp_terms), GRB.MINIMIZE)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Optimistic response origin {origin}: status={model.Status}")
    actions = {bus: float(followers[bus]["charge"][0].X - followers[bus]["discharge"][0].X)
               for bus in BUS_ORDER}
    next_energy = {bus: float(followers[bus]["energy"][1].X) for bus in BUS_ORDER}
    selected = {bus: float(followers[bus]["cost"].getValue()) for bus in BUS_ORDER}
    return actions, next_energy, time.perf_counter() - started, {
        "primary_cost": primary, "selected_cost": selected,
        "max_cost_gap_eur": max(selected[b] - primary[b] for b in BUS_ORDER),
        "leader_objective": float(model.ObjVal),
        "leader_peak_to_valley_kw": float(peak.X - valley.X),
        "leader_ramp_l1_kw": float(sum(term.X for term in ramp_terms)),
    }


def _day_ahead_pb(name, scenario, output, actual, pa, timestamps):
    path = output / "fixed_pb_plans.csv"
    if path.exists():
        saved = pd.read_csv(path)
        if len(saved) == SIM_STEPS:
            print(f"[{name}] Reusing cached three-day PB plans.", flush=True)
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
        ax.set_xlabel("Time (h)"); ax.grid(True, ls="--", alpha=.3, lw=.5); ax.set_xlim(0,SIM_STEPS*DT)
    axes[0,0].set_ylabel("Power (kW)"); axes[0,1].set_ylabel("Net load (kW)")
    axes[1,0].set_ylabel("Power (kW)"); axes[1,1].set_ylabel("PB (EUR/kWh)")
    fig.suptitle(f"{name.upper()}: fixed OL tariff + 4 h rolling BESS MPC", fontsize=13)
    fig.tight_layout(); fig.savefig(output / "mpc_overview_3days.png", dpi=300, bbox_inches="tight"); plt.close(fig)


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
