from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_core.phase1_t3_t4 import run


if __name__ == "__main__":
    run("t4")
