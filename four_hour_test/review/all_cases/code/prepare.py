"""Inspect all 48 case trajectories and enqueue independent solver-quality audits."""
from pathlib import Path
import csv
import json
import sys
import importlib.util

BASE=Path(__file__).resolve().parents[1]
SHARED=BASE.parents[1]/'code'
sys.path.insert(0,str(SHARED))
from io_utils import atomic_json,atomic_text
spec=importlib.util.spec_from_file_location('window_audit',BASE/'code/run_case.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
inspect=audit.inspect


def main():
    project=BASE.parents[2]; suite=BASE.parents[1]
    (BASE/'logs').mkdir(parents=True,exist_ok=True)
    (BASE/'results').mkdir(exist_ok=True)
    priorities={('peak_ramp','winter','perfect','MPC'),('p2v_ramp','autumn','perfect','OL'),
        ('peak_ramp','autumn','perfect','OL'),('p2v_ramp','spring','perfect','MPC'),
        ('p2v_ramp','winter','iid','MPC'),('p2v_ramp','spring','iid','MPC'),
        ('p2v_ramp','autumn','iid','MPC'),('p2v_ramp','autumn','perfect','MPC'),
        ('p2v_ramp','summer','iid','OL'),('p2v_ramp','autumn','ar1','OL'),('peak_ramp','summer','iid','OL'),
        ('peak_ramp','winter','iid','MPC')}
    tasks=[];summary=[]
    for objective,folder in [('p2v_ramp','four_hour_test'),('peak_ramp','four_hour_peak_ramp')]:
        for season in ('winter','spring','summer','autumn'):
            for scenario in ('perfect','iid','ar1'):
                for method in ('OL','MPC'):
                    root=project/folder
                    if (objective,season,scenario,method)==('p2v_ramp','summer','perfect','MPC'):
                        root=suite/'review/summer_perfect'
                    source=root/'results/matched'/season/scenario/'seed_20250903'/method
                    if not (source/'checkpoint.json').exists():continue
                    state=json.loads((source/'checkpoint.json').read_text());meta=json.loads((source/'manifest.json').read_text())
                    reasons,checks=inspect(state,meta)
                    summary.append(dict(objective=objective,season=season,scenario=scenario,method=method,
                        source_steps=state['next_step'],candidate_windows=len(reasons),physical_errors=len(checks['physical_errors']),
                        forecast_formula_error_kw=checks['forecast_formula_max_error_kw']))
                    name=f'audit_{objective}_{season}_{scenario}_{method}'
                    output=BASE/'results'/objective/season/scenario/'seed_20250903'/method
                    tasks.append(dict(name=name,season=season,scenario=scenario,method=method,
                        objective='audit_'+objective,suite_base=str(BASE),output=str(output),status='pending',
                        isolated_startup=True,total_steps=len(reasons),priority=(objective,season,scenario,method) in priorities,
                        extra_args=['--source-root',str(root),'--source-objective',objective]))
    # Full static screening covers 48 cases. Expensive re-solves cover all 12
    # cases explicitly identified in the preceding review, not 48 new full runs.
    tasks=[t for t in tasks if t['priority']]
    first=['audit_peak_ramp_winter_perfect_MPC','audit_p2v_ramp_autumn_perfect_OL',
           'audit_peak_ramp_autumn_perfect_OL','audit_p2v_ramp_summer_iid_OL',
           'audit_p2v_ramp_autumn_ar1_OL','audit_peak_ramp_summer_iid_OL']
    tasks.sort(key=lambda t:(first.index(t['name']) if t['name'] in first else len(first),t['name']))
    additional=suite/'configs/additional_tasks.json'
    existing=json.loads(additional.read_text()) if additional.exists() else []
    existing=[t for t in existing if not t['name'].startswith('audit_')]
    atomic_json(additional,existing+tasks)
    atomic_json(BASE/'case_screening.json',summary)
    text='objective,season,scenario,method,source_steps,candidate_windows,physical_errors,forecast_formula_error_kw\n'
    text+=''.join(','.join(str(row[k]) for k in row)+'\n' for row in summary)
    atomic_text(BASE/'case_screening.csv',text)
    atomic_text(BASE/'Anleitung.txt','All-case review: fixed-state windows, not new three-day simulations.\n'
        'Every recorded official window with accepted gap >10% is selected, plus amplified-ramp neighbors and case peaks.\n'
        'Amplified-ramp screen: |delta Grid|>=2 kW and |delta Grid|-|delta raw|>=1 kW; this is not a physical limit.\n'
        'Higher-gap windows target 10%; already-low-gap ramp windows target 1%. Attempts:120/600/1200 s.\n'
        'The original state, forecast, PA and model are held fixed; saved original PB provides a warm start.\n'
        'New windows of still-running source cases are picked up after a deferred restart.\n'
        'Source results stay unchanged. Summer Perfect P2V uses the completed corrected replay as its source.\n'
        'Results are independent diagnostics and must not be substituted into a three-day trajectory.\n')
    print('Screened',len(summary),'cases; queued',len(tasks),'audit jobs;',sum(r['total_steps'] for r in tasks),'windows currently selected')
    print('Physical/forecast check errors:',sum(r['physical_errors'] for r in summary))


if __name__=='__main__':main()
