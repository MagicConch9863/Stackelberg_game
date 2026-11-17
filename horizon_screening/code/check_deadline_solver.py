"""Exercise both objectives at a formerly blocked 6h window; no result edits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from deadline_solver import core, wrap
from peak_window import solve_window

base=Path(__file__).resolve().parents[1]
output=base/'status/deadline_smoke'
output.mkdir(exist_ok=True)
policy=json.loads((base/'status/deadline_policy.json').read_text())
policy.update(mpc_window_s=30,mpc_no_incumbent_total_s=60)
policy.pop('mpc_time_limits_s',None)
core.PHASE4_SETTINGS.update(GRID_EXCHANGE_MODE='grid_export',GUROBI_THREADS=2,
    LEADER_RAMP_WEIGHT=.05,BESS_DEGRADATION_EUR_PER_KWH=.05,SOLVER_OUTPUT=1)
source=base/'review/results/peak_ramp_winter_perfect_MPC_H24'
state=json.loads((source/'checkpoint.json').read_text())
origin=state['next_step']; h=24
frame=pd.read_csv(base/'inputs/winter.csv',parse_dates=['timestamp'])
pa=frame.pa_eur_per_kwh.to_numpy()[origin+1:origin+1+h]
forecast={key:{} for key in ('load','pv')}
for i,bus in enumerate(core.BUS_ORDER,1):
    forecast['load'][bus]=frame[f'bus_{i}_nominal_load'].to_numpy()[origin+1:origin+1+h]
    forecast['pv'][bus]=frame[f'bus_{i}_pv'].to_numpy()[origin+1:origin+1+h]
results=[]
for name,solve in [('p2v_ramp',core._solve_stackelberg_window),('peak_ramp',solve_window)]:
    result=wrap(solve,'MPC',policy,output)(frame.timestamp.iloc[origin+1],origin,forecast,
        state['energy'],pa,state['previous_pb'],None,warm_pb=state['warm_pb'],
        previous_realized_net=state['previous_grid'])
    assert np.isfinite(result['objective']) and np.isfinite(result['gap'])
    assert max(result[k] for k in ('constraint_violation','bound_violation','integrality_violation'))<=3e-5
    results.append(dict(objective=name,origin=origin,gap=result['gap'],
        runtime_s=result['runtime_s'],stages=result['deadline_stages']))
(output/'results.json').write_text(json.dumps(results,indent=2))
print('Real 6h deadline solver checks passed',flush=True)
