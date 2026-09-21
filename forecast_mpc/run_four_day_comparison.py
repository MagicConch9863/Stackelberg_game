from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METHODS = (
    "t12_perfect", "t1", "t2",
    "t34_perfect", "t3", "t4",
    "t56_perfect", "t5", "t6",
)
OUTPUT_ROOT = ROOT / "comparison_4days"


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
    raise RuntimeError(
        "This legacy comparison runner is invalid after the soft-SOC redesign. "
        "Rebuild T1-T6 with results_soft_soc before creating a new four-day comparison."
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("all", *METHODS), default="all")
    args = parser.parse_args()
    selected = METHODS if args.experiment == "all" else (args.experiment,)
    configure_duration(4, OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name in selected:
        folder = OUTPUT_ROOT / OUTPUT_FOLDERS[name]
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "terminal.log").open("a", encoding="utf-8", buffering=1) as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                print(f"Starting {name}")
                run_method(name)
                print(f"Completed {name}")


if __name__ == "__main__":
    main()
