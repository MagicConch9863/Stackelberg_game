"""Check review selection and state rewind without starting Gurobi."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_suite

from review_case import prepare, targets_for
from run_suite import queue_reviews, dispatch_priority, eight_hour_blocked


class ReviewTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base=Path(temporary.name)
        for name,value in [('BASE',base),('STATUS',base/'status')]:
            context=patch.object(run_suite,name,value)
            context.start()
            self.addCleanup(context.stop)

    def test_failed_reviews_queue_while_six_hour_case_is_running(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            (folder/'solver_log.json').write_text('[{"origin":16,"gap":0.4}]')
            (folder/'failure.json').write_text('{"origin":20}')
            task=dict(name='failed',output=str(folder),status='failed',horizon=24,method='MPC')
            tasks=[task,dict(task,name='running',status='running'),
                   dict(task,name='done',status='completed')]
            queue_reviews(tasks,failed_only=True)
            self.assertEqual(len(tasks),4)
            self.assertEqual(tasks[-1]['name'],'failed_review')
            self.assertNotIn('review_scanned',tasks[1])
            self.assertNotIn('review_scanned',tasks[2])
            queue_reviews(tasks)
            self.assertEqual(len(tasks),5)
            self.assertEqual(tasks[-1]['name'],'done_review')

    def test_reviews_precede_eight_hour_fillers(self):
        tasks=[dict(name='eight',method='MPC',horizon=32),
               dict(name='ol_review',method='OL',horizon=96,review=True),
               dict(name='six_review',method='MPC',horizon=24,review=True)]
        ordered=sorted(tasks,key=dispatch_priority)
        self.assertEqual(ordered[-1]['name'],'eight')
        self.assertTrue(all(t.get('review') for t in ordered[:2]))

    def test_eight_hour_waits_for_six_hour_completion(self):
        cfg={'fill_8h_after_reviews':True}
        self.assertTrue(eight_hour_blocked(cfg,False,True))
        self.assertTrue(eight_hour_blocked(cfg,True,False))
        self.assertFalse(eight_hour_blocked(cfg,False,False))

    def test_gap_tiers_and_failed_origin(self):
        solves=[dict(origin=16,gap=.10),dict(origin=17,gap=.20),
                dict(origin=18,gap=.21)]
        self.assertEqual(targets_for(solves,19),{'17':.02,'18':.10,'19':.10})

    def test_rewind_preserves_source_and_continuous_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            source=Path(temporary)/'source';source.mkdir()
            destination=Path(temporary)/'review'
            rows=[dict(execution_step=k,soc_bus_1_kwh=4+k/100,
                       pb_eur_per_kwh=.1,realized_net_kw=3.) for k in range(20)]
            state=dict(next_step=20,rows=rows,energy={'busA':4.19},
                previous_pb=.1,previous_grid=3.,warm_pb=[.1]*16,
                forecasts=[dict(origin=k) for k in range(16,20)],
                solves=[dict(origin=k,gap=.3 if k==18 else .05) for k in range(16,20)])
            original=json.dumps(state)
            (source/'checkpoint.json').write_text(original)
            (source/'manifest.json').write_text(json.dumps(dict(config=dict(horizon_steps=16))))
            prepare(source,destination)
            after=json.loads((destination/'checkpoint.json').read_text())
            self.assertEqual(after['next_step'],18)
            self.assertEqual(after['energy'],{'busA':4.17})
            self.assertEqual(len(after['rows']),18)
            self.assertEqual([r['origin'] for r in after['solves']],[16,17])
            self.assertEqual((source/'checkpoint.json').read_text(),original)
            # A resumed review must retain its own checkpoint.
            (destination/'checkpoint.json').write_text('{}')
            prepare(source,destination)
            self.assertEqual((destination/'checkpoint.json').read_text(),'{}')

    def test_review_queue_is_idempotent_and_excludes_8h(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            (folder/'solver_log.json').write_text('[{"origin":16,"gap":0.4}]')
            task=dict(name='case',output=str(folder),status='completed',horizon=24,
                      method='MPC')
            tasks=[task,dict(task,name='eight',horizon=32)]
            queue_reviews(tasks);queue_reviews(tasks)
            self.assertEqual(len(tasks),3)
            self.assertTrue(tasks[-1]['review'])
            self.assertEqual(tasks[-1]['review_target_windows'],1)

    def test_explicit_audit_targets_and_saved_price_warm_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            source=Path(temporary)/'source';source.mkdir()
            destination=Path(temporary)/'retry'
            rows=[dict(execution_step=k,soc_bus_1_kwh=4.,
                pb_eur_per_kwh=.1,realized_net_kw=3.) for k in range(20)]
            state=dict(next_step=20,rows=rows,energy={'busA':4.},
                forecasts=[dict(origin=19,bus=1,lead=h,planned_pb=.01*h) for h in range(8)],
                solves=[dict(origin=k,gap=.9) for k in range(16,20)])
            (source/'checkpoint.json').write_text(json.dumps(state))
            (source/'manifest.json').write_text(json.dumps(dict(config=dict(horizon_steps=8))))
            prepare(source,destination,targets={'19':.1})
            after=json.loads((destination/'checkpoint.json').read_text())
            self.assertEqual(after['next_step'],19)
            self.assertEqual(after['warm_pb'],[.01*h for h in range(8)])
            self.assertEqual(len(after['rows']),19)
            self.assertEqual(after['previous_grid'],3.)


if __name__=='__main__':
    unittest.main()
