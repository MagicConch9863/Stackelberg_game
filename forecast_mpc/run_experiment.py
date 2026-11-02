from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
PHASE4 = PROJECT_ROOT / "stackelberg_scenario_analysis" / "phase4"
for path in (ROOT, PHASE4):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import experiment_core.phase1_closed_loop as closed_loop
import experiment_core.phase1_t3_t4 as rolling_follower
from experiment_core.comparison_common import soft_soc_values
from experiment_core.ol_experiment import solve_day_with_feasibility_focus
from experiment_core.stackelberg_mpc import BUS_CONFIG, BUS_ORDER, DT, PHASE4_SETTINGS
from p4_s3_stackelberg import solve_day as solve_stackelberg_day

EXECUTION_DAYS = 3
SIM_STEPS = 288
HORIZON = 16
C_DEG = 0.05
OUTPUT = ROOT / "comparison_3days_soft_soc_cdeg005_h16"
FORMULATION = "soft_soc_20_80_v1_plus_linear_degradation_v1"


class _Tee:
    """Mirror console text into the active result directory."""

    def __init__(self, console, log_file):
        self.console = console
        self.log_file = log_file

    def write(self, text):
        self.console.write(text)
        self.log_file.write(text)
        self.log_file.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log_file.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


def _install_terminal_logs() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stdout_log = (OUTPUT / "terminal.log").open("a", encoding="utf-8", buffering=1)
    stderr_log = (OUTPUT / "terminal_error.log").open("a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, stdout_log)
    sys.stderr = _Tee(sys.stderr, stderr_log)


def _meta(name: str, architecture: str, scenario: str) -> dict:
    return {
        "experiment": name,
        "architecture": architecture,
        "forecast_error": scenario,
        "formulation_version": FORMULATION,
        "execution_days": EXECUTION_DAYS,
        "execution_steps": SIM_STEPS,
        "follower_mpc_horizon_steps": HORIZON if architecture == "OL-MPC" else None,
        "bess_degradation_eur_per_kwh_throughput": C_DEG,
        "soc_physical_fraction": [0.0, 1.0],
        "soc_soft_fraction": [0.2, 0.8],
        "rho_soc_eur_per_kwh": float(PHASE4_SETTINGS["SOC_SOFT_PENALTY_EUR_PER_KWH"]),
        "terminal_soc_reference": False,
        "midnight_constraint": False,
    }


def _save_metrics(name: str, frame: pd.DataFrame, folder: Path) -> None:
    closed_loop._metrics(name, frame).to_csv(folder / "period_metrics.csv", index=False)


def _warm_pb(scenario: str, day: int) -> np.ndarray | None:
    candidates = [
        ROOT / "comparison_3days_soft_soc_degradation_h16" / "t12_perfect_ol_ol" / "timeseries.csv",
        ROOT / "comparison_3days_soft_soc_h12" / "t12_perfect_ol_ol" / "timeseries.csv",
    ]
    for path in candidates:
        if path.exists():
            data = pd.read_csv(path)
            values = data.get("pb_eur_per_kwh")
            if values is not None and len(values) >= (day + 1) * 96:
                return values.to_numpy(float)[day * 96:(day + 1) * 96]
    return None


def run_pair(pair: str) -> None:
    definitions = {
        "t13": ("iid", "t1", "t3", "t1_iid_ol_ol", "t3_iid_ol_mpc"),
        "t24": ("ar1", "t2", "t4", "t2_ar1_ol_ol", "t4_ar1_ol_mpc"),
        "t1234perfect": (
            "perfect", "t12", "t34", "t12_perfect_ol_ol", "t34_perfect_ol_mpc"
        ),
    }
    scenario, ol_name, mpc_name, ol_dir, mpc_dir = definitions[pair]
    ol_folder = OUTPUT / ol_dir
    mpc_folder = OUTPUT / mpc_dir
    ol_folder.mkdir(parents=True, exist_ok=True)
    mpc_folder.mkdir(parents=True, exist_ok=True)
    (ol_folder / "formulation_metadata.json").write_text(
        json.dumps(_meta(ol_name, "OL-OL", scenario), indent=2), encoding="utf-8")
    (mpc_folder / "formulation_metadata.json").write_text(
        json.dumps(_meta(mpc_name, "OL-MPC", scenario), indent=2), encoding="utf-8")

    closed_loop.SIM_STEPS = SIM_STEPS
    closed_loop.HORIZON = HORIZON
    _, actual, pa, timestamps = closed_loop._load_period()
    settings = deepcopy(PHASE4_SETTINGS)
    settings.update({"BESS_DEGRADATION_EUR_PER_KWH": C_DEG, "SOLVER_OUTPUT": 1,
                     "TIME_LIMIT_S_PER_DAY": 900.0, "MIP_GAP": 0.05})
    energy = {b: float(BUS_CONFIG[b]["bess_e_init"]) for b in BUS_ORDER}
    original_energy = energy.copy()
    ol_rows, ol_solver, pb_all = [], [], []
    try:
        for day in range(EXECUTION_DAYS):
            origin = day * 96
            for bus in BUS_ORDER:
                BUS_CONFIG[bus]["bess_e_init"] = energy[bus]
            warm = _warm_pb(scenario, day)
            if warm is not None:
                settings["PB_WARM_START"] = warm
            forecast = closed_loop._forecast(actual, scenario, origin, horizon=96)
            print(f"[{ol_name}] OL day {day+1}/3 ({scenario})", flush=True)
            started = time.perf_counter()
            result, pb, gap = solve_day_with_feasibility_focus(
                solve_stackelberg_day, timestamps[origin], forecast["load"], forecast["pv"],
                pa[origin:origin+96], settings)
            elapsed = time.perf_counter() - started
            pb_all.extend(np.asarray(pb, float))
            ol_solver.append({"day": day, "date": str(timestamps[origin].date()),
                              "status": "incumbent", "mip_gap": gap, "runtime_s": elapsed})
            for step in range(96):
                base = sum(actual["load"][b][origin+step] - actual["pv"][b][origin+step]
                           for b in BUS_ORDER)
                bess = sum(float(result[b]["bess"][step]) for b in BUS_ORDER)
                throughput = DT * sum(abs(float(result[b]["bess"][step])) for b in BUS_ORDER)
                under = over = penalty = 0.0
                for bus in BUS_ORDER:
                    u, o, p = soft_soc_values(bus, float(result[bus]["bess_energy"][step+1]))
                    under += u; over += o; penalty += p
                ol_rows.append({"step": origin+step, "timestamp": timestamps[origin+step],
                                "uncontrolled_net_kw": base, "realized_net_kw": base+bess,
                                "bess_kw": bess, "pb_eur_per_kwh": float(pb[step]),
                                "bess_throughput_step_kwh": throughput,
                                "bess_degradation_cost_eur": C_DEG*throughput,
                                "soc_under_slack_kwh": under, "soc_over_slack_kwh": over,
                                "soc_penalty_eur": penalty,
                                **{f"soc_{b.replace(' ', '_').lower()}_kwh":
                                   float(result[b]["bess_energy"][step+1]) for b in BUS_ORDER}})
            energy = {b: float(result[b]["bess_energy"][-1]) for b in BUS_ORDER}
            pd.DataFrame(ol_rows).to_csv(ol_folder / "timeseries.partial.csv", index=False)
            pd.DataFrame(ol_solver).to_csv(ol_folder / "solver_log.partial.csv", index=False)
    finally:
        for bus, value in original_energy.items():
            BUS_CONFIG[bus]["bess_e_init"] = value

    ol = pd.DataFrame(ol_rows)
    ol.to_csv(ol_folder / "timeseries.csv", index=False)
    pd.DataFrame(ol_solver).to_csv(ol_folder / "solver_log.csv", index=False)
    _save_metrics(ol_name, ol, ol_folder)

    pb_buffer = np.r_[np.asarray(pb_all), np.repeat(pb_all[-1], HORIZON-1)]
    energy = original_energy.copy()
    mpc_rows, mpc_solver, forecast_rows = [], [], []
    previous_realized_net = None
    for origin in range(SIM_STEPS):
        forecast = closed_loop._forecast(actual, scenario, origin, horizon=HORIZON)
        actions, next_energy, runtime, optimistic = rolling_follower.solve_optimistic_followers(
            energy, forecast, pa[origin:origin+HORIZON],
            pb_buffer[origin:origin+HORIZON], origin,
            previous_realized_net=previous_realized_net,
        )
        energy.update(next_energy)
        under = over = penalty = 0.0
        for bus in BUS_ORDER:
            baseline = forecast["load"][bus] - forecast["pv"][bus]
            u, o, p = soft_soc_values(bus, energy[bus]); under += u; over += o; penalty += p
            for lead in range(HORIZON):
                forecast_rows.append({"origin_step": origin, "target_step": origin+lead,
                                      "lead": lead, "bus": bus,
                                      "forecast_baseline_kw": baseline[lead],
                                      "actual_baseline_kw": actual["load"][bus][origin+lead]-actual["pv"][bus][origin+lead]})
        base = sum(actual["load"][b][origin]-actual["pv"][b][origin] for b in BUS_ORDER)
        bess = sum(actions.values()); throughput = DT*sum(abs(v) for v in actions.values())
        mpc_rows.append({"step": origin, "timestamp": timestamps[origin],
                         "uncontrolled_net_kw": base, "realized_net_kw": base+bess,
                         "bess_kw": bess, "pb_eur_per_kwh": pb_buffer[origin],
                         "bess_throughput_step_kwh": throughput,
                         "bess_degradation_cost_eur": C_DEG*throughput,
                         "soc_under_slack_kwh": under, "soc_over_slack_kwh": over,
                         "soc_penalty_eur": penalty,
                         **{f"soc_{b.replace(' ', '_').lower()}_kwh": energy[b] for b in BUS_ORDER}})
        mpc_solver.append({"step": origin, "timestamp": timestamps[origin], "status": 2,
                           "mip_gap": 0.0, "runtime_s": runtime,
                           "optimistic_max_follower_cost_gap_eur": optimistic["max_cost_gap_eur"],
                           "leader_objective": optimistic["leader_objective"],
                           "leader_peak_to_valley_kw": optimistic["leader_peak_to_valley_kw"],
                           "leader_ramp_l1_kw": optimistic["leader_ramp_l1_kw"]})
        previous_realized_net = base + bess
        if (origin+1) % 12 == 0:
            print(f"[{mpc_name}] {origin+1:03d}/{SIM_STEPS}", flush=True)
        pd.DataFrame(mpc_rows).to_csv(mpc_folder / "timeseries.partial.csv", index=False)
    mpc = pd.DataFrame(mpc_rows)
    mpc.to_csv(mpc_folder / "timeseries.csv", index=False)
    pd.DataFrame(mpc_solver).to_csv(mpc_folder / "solver_log.csv", index=False)
    pd.DataFrame(forecast_rows).to_csv(mpc_folder / "rolling_forecasts.csv", index=False)
    _save_metrics(mpc_name, mpc, mpc_folder)


def run_t56(name: str) -> None:
    scenario = {"t5perfect": "perfect", "t5": "iid", "t6": "ar1"}[name]
    folder_name = {"t5perfect": "t56_perfect_mpc_mpc", "t5": "t5_iid_mpc_mpc",
                   "t6": "t6_ar1_mpc_mpc"}[name]
    closed_loop.SIM_STEPS = SIM_STEPS
    closed_loop.HORIZON = HORIZON
    closed_loop.EXPERIMENTS = {name: (scenario, OUTPUT / folder_name)}
    closed_loop.PHASE4_SETTINGS["BESS_DEGRADATION_EUR_PER_KWH"] = C_DEG
    closed_loop.EXTERNAL_PB_WARM_STARTS = {}
    source = ROOT / "t56_perfect_2days_cdeg005_h4" / "rolling_plans.csv"
    if name == "t5perfect" and source.exists():
        old = pd.read_csv(source)
        for origin, group in old.groupby("origin_step"):
            one = group[group["bus"] == BUS_ORDER[0]].sort_values("lead")
            if len(one) >= HORIZON:
                closed_loop.EXTERNAL_PB_WARM_STARTS[int(origin)] = one["planned_pb_eur_per_kwh"].to_numpy(float)[:HORIZON]
    closed_loop.run(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=(
            "t13", "t24", "t1234perfect", "t5perfect", "t5", "t6",
            "analysis", "post", "remaining", "resume_after_reboot", "all"
        ),
        default="all",
    )
    args = parser.parse_args()
    _install_terminal_logs()
    PHASE4_SETTINGS["BESS_DEGRADATION_EUR_PER_KWH"] = C_DEG
    if args.task == "all":
        tasks = ("t13", "t24", "t5perfect", "t5", "t6", "t1234perfect", "analysis")
    elif args.task == "remaining":
        tasks = ("t24", "t5perfect", "t5", "t6")
    elif args.task == "post":
        tasks = ("t1234perfect", "analysis")
    elif args.task == "resume_after_reboot":
        tasks = ("t5", "t6", "t1234perfect", "analysis")
    else:
        tasks = (args.task,)
    for task in tasks:
        if task in ("t13", "t24", "t1234perfect"):
            run_pair(task)
        elif task == "analysis":
            from experiment_core.performance_analysis import generate_analysis
            generate_analysis(OUTPUT)
        else:
            run_t56(task)


if __name__ == "__main__":
    main()
