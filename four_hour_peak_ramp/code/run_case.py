"""Peak+ramp experiment using the same execution, prediction and checkpoint code."""
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE.parent/'four_hour_test/code'))
import run_case as shared
from solve_window import solve_window

if __name__ == '__main__':
    shared.main(experiment_base=BASE, solve_window=solve_window,
                solver_source=BASE/'code/solve_window.py')
