from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent

EXPERIMENTS = {
    "t1": (ROOT / "t1_iid_ol_ol" / "run_t1.py", ROOT / "t1_iid_ol_ol" / "results_soft_soc"),
    "t2": (ROOT / "t2_ar1_ol_ol" / "run_t2.py", ROOT / "t2_ar1_ol_ol" / "results_soft_soc"),
    "t3": (ROOT / "t3_iid_ol_mpc" / "run_t3.py", ROOT / "t3_iid_ol_mpc" / "results_soft_soc_3days"),
    "t4": (ROOT / "t4_ar1_ol_mpc" / "run_t4.py", ROOT / "t4_ar1_ol_mpc" / "results_soft_soc_3days"),
    "t5": (ROOT / "t5_iid_mpc_mpc" / "run_t5.py", ROOT / "t5_iid_mpc_mpc" / "results_soft_soc"),
    "t6": (ROOT / "t6_ar1_mpc_mpc" / "run_t6.py", ROOT / "t6_ar1_mpc_mpc" / "results_soft_soc"),
    "t5_perfect": (
        ROOT / "t5_perfect_mpc_mpc" / "run_t5_perfect.py",
        ROOT / "t5_perfect_mpc_mpc" / "results_soft_soc",
    ),
}

GROUPS = {
    "t5-t6": ("t5", "t6"),
    "followup": ("t1", "t2", "t3", "t4", "t5_perfect"),
    "all": ("t5", "t6", "t1", "t2", "t3", "t4", "t5_perfect"),
}


def preflight() -> None:
    import gurobipy as gp
    import matplotlib
    import numpy
    import pandas
    import scipy
    import simbench

    model = gp.Model("phase1_preflight")
    model.Params.OutputFlag = 0
    variable = model.addVar(lb=0.0, name="license_check")
    model.setObjective(variable, gp.GRB.MINIMIZE)
    model.optimize()
    if model.Status != gp.GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi preflight failed with status {model.Status}.")
    print(
        "Preflight OK | "
        f"Python {sys.version.split()[0]} | Gurobi {'.'.join(map(str, gp.gurobi.version()))} | "
        f"NumPy {numpy.__version__} | Pandas {pandas.__version__} | "
        f"SciPy {scipy.__version__} | Matplotlib {matplotlib.__version__} | "
        f"SimBench {simbench.__version__}",
        flush=True,
    )


def run_experiment(name: str) -> None:
    script, output = EXPERIMENTS[name]
    if not script.is_file():
        raise FileNotFoundError(f"Experiment entry point not found: {script}")
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "terminal.log"
    started = datetime.now().astimezone()
    header = f"\n{'=' * 80}\n{name} started {started.isoformat()}\ncommand: {sys.executable} {script}\n"
    print(header, end="", flush=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(header)
        process = subprocess.Popen(
            [sys.executable, "-u", str(script)],
            cwd=ROOT.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = process.wait()
        footer = f"{name} finished {datetime.now().astimezone().isoformat()} | exit={return_code}\n"
        print(footer, end="", flush=True)
        log.write(footer)
    if return_code != 0:
        raise RuntimeError(f"{name} failed with exit code {return_code}; see {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase-1 T1-T6 experiment suite serially.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--experiment", choices=tuple(EXPERIMENTS))
    selection.add_argument("--group", choices=tuple(GROUPS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preflight()
    names = (args.experiment,) if args.experiment else GROUPS[args.group]
    print(f"Execution order: {', '.join(names)}", flush=True)
    for name in names:
        run_experiment(name)
    print("All requested experiments completed.", flush=True)


if __name__ == "__main__":
    main()
