from pathlib import Path
import sys
import unittest

import gurobipy as gp
from gurobipy import GRB


ROOT = Path(__file__).resolve().parents[2]
PHASE4 = ROOT / "stackelberg_scenario_analysis" / "phase4"
sys.path.insert(0, str(PHASE4))

from p4_common import BUS_CONFIG
from p4_run import SETTINGS
from soc_soft_band import add_soft_soc_band, soc_limits


class SoftSocBandTest(unittest.TestCase):
    def test_piecewise_linear_soft_soc_penalty(self):
        cfg = BUS_CONFIG["Bus 1"]
        for fraction, expected_fraction in ((0.50, 0.0), (0.10, 0.10), (0.90, 0.10)):
            with self.subTest(fraction=fraction):
                model = gp.Model()
                model.Params.OutputFlag = 0
                energy = model.addVars(2, lb=cfg["bess_e_min"], ub=cfg["bess_e_max"])
                model.addConstr(energy[0] == cfg["bess_e_init"])
                model.addConstr(energy[1] == fraction * cfg["bess_kwh"])
                penalty, under, over = add_soft_soc_band(
                    model, energy, 1, cfg, SETTINGS, "test"
                )
                model.setObjective(penalty, GRB.MINIMIZE)
                model.optimize()
                self.assertEqual(model.Status, GRB.OPTIMAL)
                self.assertAlmostEqual(
                    under[0].X + over[0].X,
                    expected_fraction * cfg["bess_kwh"],
                )

    def test_physical_soc_limits_are_zero_and_capacity(self):
        cfg = BUS_CONFIG["Bus 1"]
        physical_low, physical_high, soft_low, soft_high = soc_limits(cfg, SETTINGS)
        self.assertEqual(physical_low, 0.0)
        self.assertEqual(physical_high, cfg["bess_kwh"])
        self.assertAlmostEqual(soft_low, 0.20 * cfg["bess_kwh"])
        self.assertAlmostEqual(soft_high, 0.80 * cfg["bess_kwh"])


if __name__ == "__main__":
    unittest.main()
