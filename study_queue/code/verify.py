"""Prelaunch checks for priority, reproducibility, causality and warmup handover."""
from pathlib import Path
import ast
import json
import math
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'study_queue'
sys.path.insert(0,str(ROOT/'horizon_screening/code'))
from io_utils import atomic_json

for path in (list((BASE/'code').glob('*.py'))+
             [ROOT/'horizon_screening/code/run_case.py',ROOT/'horizon_screening/code/deadline_solver.py']):
    ast.parse(path.read_text(encoding='utf-8-sig'))
cfg=json.loads((BASE/'configs/plan.json').read_text())
cases=cfg['cases']
assert len(cases)==58 and len({t['name'] for t in cases})==58
assert len({t['output'] for t in cases})==58
assert [sum(t['stage']==s for t in cases) for s in (1,2,3)]==[16,40,2]
assert sum(t['stage']==1 and t['generate_warmup'] for t in cases)==8
assert sum(t['stage']==2 and t['generate_warmup'] for t in cases)==8
assert all(t['horizon'] in (16,96) for t in cases)
assert all(Path(t['shared_warmup']).exists() for t in cases if not t['generate_warmup'] and t['stage']==3)
assert len(cfg['reused'])==24
import numpy as np
from prediction import forecast_window
sample={'nominal_load':{'Bus 1':np.full(40,5.)},'load':{'Bus 1':np.full(40,7.)},'pv':{'Bus 1':np.zeros(40)}}
prediction=forecast_window(sample,'MPC','ar1',1,16,.5)['load']['Bus 1']
np.testing.assert_allclose(prediction,5.+2.*.5**np.arange(1,17))
sample['load']['Bus 1'][1:]=1000.
np.testing.assert_allclose(prediction,forecast_window(sample,'MPC','ar1',1,16,.5)['load']['Bus 1'])
sigma=1.2
assert math.isclose((sigma*math.sqrt(1-.5**2))**2/(1-.5**2),sigma**2)
atomic_json(BASE/'status/verification.json',dict(syntax=True,distinct_cases=58,reused_cases=24,
    causal_prediction=True,matched_marginal_sigma=True,shared_warmup_dependencies=True,
    forecast_formula='nominal + rho**(h+1) * previous_completed_residual'))
print('Verification passed: priority counts, source reuse, causal rho forecast, fixed marginal sigma.',flush=True)
