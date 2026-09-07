from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_core.stackelberg_mpc import run_experiment


if __name__ == "__main__":
    run_experiment("t5_perfect")
