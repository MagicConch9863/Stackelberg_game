from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

import experiment_core.phase1_soc_reference as experiment

experiment.OUTPUT = ROOT / "t5_perfect_soc_reference" / "results_5days"
experiment.SIM_STEPS = 5 * 96
experiment.FIRST_STEP_LIMIT_S = 900.0

if __name__=="__main__": experiment.run()
