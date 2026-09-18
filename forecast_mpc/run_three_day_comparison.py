from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_core.three_day_comparison import OUTPUT_ROOT, run_method


METHODS = ("t1", "t2", "t3", "t4", "t5_perfect")
OUTPUT_NAMES = {
    "t1": "t1_iid_ol_ol",
    "t2": "t2_ar1_ol_ol",
    "t3": "t3_iid_ol_mpc",
    "t4": "t4_ar1_ol_mpc",
    "t5_perfect": "t5_perfect_mpc_mpc",
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("all", *METHODS), default="all")
    args = parser.parse_args()
    selected = METHODS if args.experiment == "all" else (args.experiment,)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name in selected:
        folder = OUTPUT_ROOT / OUTPUT_NAMES[name]
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "terminal.log").open("a", encoding="utf-8", buffering=1) as log:
            tee_out, tee_err = Tee(sys.stdout, log), Tee(sys.stderr, log)
            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                print(f"Starting {name}")
                run_method(name)
                print(f"Completed {name}")


if __name__ == "__main__":
    main()
