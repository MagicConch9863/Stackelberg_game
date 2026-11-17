import unittest
from progress_view import render_progress


class ProgressViewTests(unittest.TestCase):
    def test_automatic_round_displays_the_current_strict_budget(self):
        task=dict(name='quality',review=True,review_mode='quality_review',status='running',
                  method='MPC',horizon=24,objective='p2v_ramp',season='winter',scenario='iid',
                  automatic_round=2)
        row=dict(task,seed_label='S1',current_step=20,total_steps=304,display_status='SOLVING')
        text='\n'.join(render_progress([task],[row],
            dict(parallel_cases=14,threads_per_case=16,time_limits_s=[60,120,600]),
            dict(ol_window_s=3600,quality_followup=dict(enabled=True,
                automatic_retry_unqualified=True,mpc_time_limits_s=[120,600,3600],
                ol_time_limit_s=3600))))
        self.assertIn('Aktuelles MPC-Prüfbudget: 120/600/3600 s',text)
        self.assertIn('OL: 3600 s',text)
        self.assertIn('automatische Runde 2',text)
        self.assertNotIn('ab Stufe 2 genügt',text)

    def test_quality_followup_is_visible_and_separate(self):
        task=dict(name='quality',review=True,review_mode='quality_review',status='running',
                  method='MPC',horizon=24,objective='p2v_ramp',season='winter',scenario='iid')
        row=dict(task,seed_label='S1',current_step=20,total_steps=304,display_status='SOLVING')
        text='\n'.join(render_progress([task],[row],
            dict(parallel_cases=14,threads_per_case=16,time_limits_s=[60,120,600]),
            dict(quality_followup=dict(enabled=True))))
        self.assertIn('Qualitätsprüfungen: fertig 0 | aktiv 1',text)
        self.assertIn('Qualitätsprüfung',text)
        self.assertIn('<=2%',text)
        self.assertNotRegex(text,r'[\u3400-\u9fff]')

    def test_names_actual_scenario_horizon_objective_and_executed_steps(self):
        task=dict(name='retry',review=True,review_mode='budget_retry',status='running',
                  method='MPC',horizon=24,objective='peak_ramp',season='autumn',scenario='ar1')
        row=dict(task,seed_label='S1',current_step=56,total_steps=304,display_status='SOLVING',
                 decision_time='2025-10-15 10:00:00')
        text='\n'.join(render_progress([task],[row],
            dict(parallel_cases=14,threads_per_case=16,time_limits_s=[60,120,600]),
            dict(mpc_time_limits_s=[60,120,600],ol_window_s=3600)))
        self.assertIn('Herbst | AR(1), rho=0.9 | Peak+Ramp | MPC 6h | S1',text)
        self.assertIn('Schritt 56/304; Auswertungszeitraum: 40/288 Schritte abgeschlossen',text)
        self.assertIn('2025-10-15 10:00:00',text)
        self.assertNotRegex(text,r'[\u3400-\u9fff]')

    def test_audit_windows_are_not_simulation_steps(self):
        task=dict(name='audit',review=True,review_mode='certificate',status='running',
                  method='MPC',horizon=24,objective='p2v_ramp',season='spring',scenario='iid')
        row=dict(task,seed_label='S1',current_step=12,total_steps=20,display_status='SOLVING')
        text='\n'.join(render_progress([task],[row],
            dict(parallel_cases=14,threads_per_case=16,time_limits_s=[60,120,600]),{}))
        self.assertIn('Geprüfte Fenster: 12/20',text)
        self.assertNotIn('Auswertungszeitraum:',text)
        self.assertNotRegex(text,r'[\u3400-\u9fff]')


if __name__=='__main__':
    unittest.main()
