"""Protect original-plan certificates and deadline scheduling semantics."""
import math
from pathlib import Path
import tempfile
import unittest

from deadline_policy import certificate, task_priority, accept_mpc_stage
from solve_policy import stop_after
from run_suite import complete, finish
import run_suite


class DeadlineTests(unittest.TestCase):
    def test_first_attempt_requires_target_gap(self):
        self.assertTrue(accept_mpc_stage(0, True, .05))
        self.assertFalse(accept_mpc_stage(0, True, .5))

    def test_second_attempt_accepts_high_gap_feasible_plan(self):
        self.assertTrue(accept_mpc_stage(1, True, .9))

    def test_rescue_requires_valid_finite_incumbent(self):
        self.assertFalse(accept_mpc_stage(1, False, .01))
        self.assertFalse(accept_mpc_stage(2, True, math.inf))

    def test_regular_runner_accepts_feasible_second_attempt(self):
        cfg=dict(target_gap=.1,acceptance_gap_after_second=.2)
        self.assertFalse(stop_after(60,dict(gap=.9),cfg))
        self.assertTrue(stop_after(120,dict(gap=.9),cfg))
        self.assertFalse(stop_after(120,None,cfg))

    def test_certificate_uses_original_upper_not_better_candidate(self):
        result=certificate(10., 1., 8.)
        self.assertAlmostEqual(result['gap'], .20)
        self.assertFalse(result['actions_changed'])
        self.assertEqual(result['quality'],'acceptable_20pct')

    def test_inconsistent_bound_is_rejected(self):
        with self.assertRaises(ValueError):
            certificate(1., .1, 1.1)

    def test_incomplete_bound_does_not_pass(self):
        self.assertEqual(certificate(10.,0.,2.)['quality'],'needs_further_review')

    def test_nonfinite_bound_does_not_pass(self):
        with self.assertRaises(ValueError):
            certificate(10.,0.,math.nan)

    def test_completion_precedes_certificates(self):
        self.assertLess(task_priority({'review_mode':'completion'}),
                        task_priority({'review_mode':'certificate','priority_gap_gt20':True}))

    def test_completed_artifact_survives_nonzero_process_code(self):
        import json
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            (folder/'certificate_summary.json').write_text(json.dumps(dict(
                audit_complete=True,windows_checked=2,total_windows=2,windows_unresolved=1)))
            task=dict(output=str(folder),review_mode='certificate')
            self.assertTrue(complete(task))
            finish(task,1)
            self.assertEqual(task['status'],'completed')
            # Audit completion is independent of whether its gaps passed.

    def test_partial_audit_is_not_complete(self):
        import json
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            (folder/'certificate_summary.json').write_text(json.dumps(dict(
                audit_complete=True,windows_checked=1,total_windows=2)))
            self.assertFalse(complete(dict(output=str(folder),review_mode='certificate')))


if __name__=='__main__':
    unittest.main()
