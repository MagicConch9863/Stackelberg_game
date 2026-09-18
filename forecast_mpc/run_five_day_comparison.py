from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_core.three_day_comparison import OUTPUT_FOLDERS, configure_duration, run_method
from run_four_day_comparison import METHODS, Tee

OUTPUT_ROOT = ROOT / "comparison_5days"


def _complete(name: str) -> bool:
    folder = OUTPUT_ROOT / OUTPUT_FOLDERS[name]
    series = folder / ("mpc_timeseries.csv" if name in {"t56_perfect", "t5", "t6"} else "timeseries.csv")
    metrics = folder / "daily_metrics.csv"
    if not (series.exists() and metrics.exists() and (folder / "overall_metrics.csv").exists()):
        return False
    import pandas as pd
    return len(pd.read_csv(series)) == 480 and len(pd.read_csv(metrics)) == 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("all", *METHODS), default="all")
    parser.add_argument("--skip-complete", action="store_true")
    args = parser.parse_args()
    selected = METHODS if args.experiment == "all" else (args.experiment,)
    configure_duration(5, OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name in selected:
        if args.skip_complete and _complete(name):
            print(f"Skipping complete {name}")
            continue
        folder = OUTPUT_ROOT / OUTPUT_FOLDERS[name]
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "terminal.log").open("a", encoding="utf-8", buffering=1) as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                print(f"Starting {name}")
                run_method(name)
                print(f"Completed {name}")


if __name__ == "__main__":
    main()
