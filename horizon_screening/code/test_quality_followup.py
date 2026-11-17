"""Check the deferred quality stage without running or changing a simulation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_suite
from deadline_policy import accept_mpc_stage
from review_case import targets_for, matching_candidate


class QualityFollowupTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name)
        status=self.base/'status';status.mkdir()
        (status/'deadline_policy.json').write_text(json.dumps(dict(quality_followup=dict(enabled=True))))
        for name,value in [('BASE',self.base),('STATUS',status)]:
            context=patch.object(run_suite,name,value);context.start();self.addCleanup(context.stop)

    def case(self,name,gaps,**extra):
        folder=self.base/name;folder.mkdir()
        (folder/'solver_log.json').write_text(json.dumps([
            dict(origin=16+i,gap=gap) for i,gap in enumerate(gaps)]))
        task=dict(name=name,output=str(folder),status='completed',seed=1,seed_label='S1',
                  season='winter',scenario='iid',objective='p2v_ramp',method='MPC',horizon=16)
        task.update(extra);return task

    def pair(self,gaps,status='completed'):
        source=self.case('case',[.99])
        retry=self.case('case_budget_retry',gaps,review=True,review_mode='budget_retry',
                        source_case='case',status=status)
        return [source,retry]

    def test_waits_for_all_completion_tasks(self):
        tasks=self.pair([.9])
        for status in ('pending','running'):
            with self.subTest(status=status):
                tasks[1]['status']=status
                self.assertEqual(run_suite.queue_quality_reviews(tasks),0)
                self.assertEqual(len(tasks),2)

    def test_latest_retry_uses_two_gap_tiers_and_is_queued_once(self):
        tasks=self.pair([.25,.15,.05])
        self.assertEqual(run_suite.queue_quality_reviews(tasks),1)
        task=tasks[-1]
        self.assertEqual(task['review_source'],tasks[1]['output'])
        self.assertEqual(task['review_target_windows'],2)
        self.assertTrue(task['priority_gap_gt20'])
        self.assertEqual(task['review_mode'],'quality_review')
        self.assertEqual(run_suite.queue_quality_reviews(tasks),0)
        self.assertEqual(len(tasks),3)
        restored=json.loads(json.dumps(tasks))
        self.assertEqual(run_suite.queue_quality_reviews(restored),0)

    def test_passing_latest_result_does_not_reuse_bad_old_result(self):
        tasks=self.pair([.01,.10])
        self.assertEqual(run_suite.queue_quality_reviews(tasks),0)
        self.assertTrue(tasks[1]['quality_followup_scanned'])

    def test_nonretry_latest_cases_are_included(self):
        tasks=self.pair([.01])
        tasks.append(self.case('other',[.15]))
        self.assertEqual(run_suite.queue_quality_reviews(tasks),1)
        self.assertEqual(tasks[-1]['source_case'],'other')
        self.assertFalse(tasks[-1]['priority_gap_gt20'])

    def test_completed_review_supersedes_its_original(self):
        tasks=self.pair([.01])
        tasks.append(self.case('other',[.9]))
        tasks.append(self.case('other_review',[.01],review=True))
        self.assertEqual(run_suite.queue_quality_reviews(tasks),0)
        self.assertTrue(tasks[-1]['quality_followup_scanned'])

    def test_failed_case_uses_saved_failure_origin(self):
        tasks=self.pair([.01],'failed')
        (Path(tasks[1]['output'])/'failure.json').write_text('{"origin":17}')
        self.assertEqual(run_suite.queue_quality_reviews(tasks),1)
        self.assertEqual(tasks[-1]['review_failed_origin'],17)

    def test_strict_quality_never_accepts_high_gap_after_second_stage(self):
        self.assertFalse(accept_mpc_stage(1,True,.9,strict=True))
        self.assertFalse(accept_mpc_stage(2,True,.03,target=.02,strict=True))
        self.assertTrue(accept_mpc_stage(2,True,.02,target=.02,strict=True))
        self.assertTrue(accept_mpc_stage(1,True,.9))

    def test_nonfinite_gap_is_targeted(self):
        self.assertEqual(targets_for([dict(origin=16,gap=None),dict(origin=17,gap=float('nan'))]),
                         {'16':.1,'17':.1})


    def retry_case(self):
        policy=dict(quality_followup=dict(enabled=True,automatic_retry_unqualified=True,
                                         maximum_automatic_rounds=0))
        (self.base/'status/deadline_policy.json').write_text(json.dumps(policy))
        task=self.case('quality',[],status='failed',review=True,review_mode='quality_review',
                       exit_code=1,finished_at='2026-11-16 11:00:00')
        folder=Path(task['output'])
        (folder/'checkpoint.json').write_text('{"next_step":16,"energy":{"Bus 1":4}}')
        (folder/'failure.json').write_text(json.dumps(dict(origin=16,reason='target gap not reached',
            target_gap=.1,attempts=[dict(gap=.4)])))
        (folder/'review_plan.json').write_text('{"quality_review":true,"targets":{"16":0.1}}')
        (folder/'latest_candidate.json').write_text('{"saved":true}')
        return task,folder

    def test_failed_review_requeues_without_changing_checkpoint_or_targets(self):
        task,folder=self.retry_case()
        before=(folder/'checkpoint.json').read_text()
        self.assertEqual(run_suite.queue_quality_retries([task]),1)
        self.assertEqual(task['status'],'pending')
        self.assertEqual(task['automatic_round'],1)
        self.assertTrue(task['priority_gap_gt20'])
        self.assertNotIn('exit_code',task)
        self.assertEqual((folder/'checkpoint.json').read_text(),before)
        self.assertEqual(json.loads((folder/'review_plan.json').read_text())['targets'],{'16':.1})
        self.assertEqual((folder/'latest_candidate.json').read_text(),'{"saved":true}')
        self.assertEqual(len((folder/'review_rounds.jsonl').read_text().splitlines()),1)
        self.assertEqual(run_suite.queue_quality_retries([task]),0)

    def test_automatic_retry_continues_after_three_rounds_and_leaves_passed_cases(self):
        task,folder=self.retry_case()
        for number in range(1,5):
            task['status']='failed'
            self.assertEqual(run_suite.queue_quality_retries([task]),1)
            self.assertEqual(task['automatic_round'],number)
        task['status']='completed'
        self.assertEqual(run_suite.queue_quality_retries([task]),0)
        self.assertEqual(len((folder/'review_rounds.jsonl').read_text().splitlines()),4)

    def test_mismatched_failure_cannot_reuse_the_wrong_state(self):
        task,folder=self.retry_case()
        (folder/'checkpoint.json').write_text('{"next_step":17}')
        self.assertEqual(run_suite.queue_quality_retries([task]),0)
        self.assertEqual(task['status'],'failed')
        self.assertIn('checkpoint',task['automatic_retry_blocked'])

    def test_current_round_and_high_gaps_have_priority(self):
        base=dict(review_mode='quality_review',automatic_round=1,review_last_gap=.9)
        high=dict(base,priority_gap_gt20=True)
        moderate=dict(base,priority_gap_gt20=False,review_last_gap=.15)
        later=dict(high,automatic_round=2)
        self.assertLess(run_suite.dispatch_priority(high),run_suite.dispatch_priority(moderate))
        self.assertLess(run_suite.dispatch_priority(moderate),run_suite.dispatch_priority(later))

    def test_saved_candidate_requires_the_same_window_energy_and_horizon(self):
        candidate=dict(origin=16,energies={'Bus 1':4},pb=[.1,.2],
                       objective=2,best_bound=1,gap=.5)
        path=self.base/'latest_candidate.json'
        path.write_text(json.dumps(candidate))
        self.assertEqual(matching_candidate(self.base,16,{'Bus 1':4},2),candidate)
        self.assertIsNone(matching_candidate(self.base,17,{'Bus 1':4},2))
        self.assertIsNone(matching_candidate(self.base,16,{'Bus 1':3},2))
        self.assertIsNone(matching_candidate(self.base,16,{'Bus 1':4},3))
        candidate['energies']['Bus 1']=float('nan')
        path.write_text(json.dumps(candidate))
        self.assertIsNone(matching_candidate(self.base,16,{'Bus 1':4},2))


if __name__=='__main__':
    unittest.main()
