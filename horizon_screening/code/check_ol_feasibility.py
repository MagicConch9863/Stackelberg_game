"""Real 96-step smoke test; no experiment outputs are overwritten."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ol_feasibility import core, follower_start, wrap
from ol_warm_start import load_prices, window_prices
from peak_window import solve_window as peak_solve

base=Path(__file__).resolve().parents[1]
cfg=json.loads((base/'configs/experiment.json').read_text())
core.PHASE4_SETTINGS.update(GRID_EXCHANGE_MODE='grid_export',GUROBI_THREADS=2,
    LEADER_RAMP_WEIGHT=.05,BESS_DEGRADATION_EUR_PER_KWH=.05,SOLVER_OUTPUT=1)
frame=pd.read_csv(base/'inputs/winter.csv',parse_dates=['timestamp'])
out=base/'results/p2v_ramp/OL_H96/winter/perfect/seed_20250903'
state=json.loads((out/'checkpoint.json').read_text())
origin=state['next_step']; h=96
timestamps=frame.timestamp.iloc[origin+1:origin+1+h].tolist()
pa=frame.pa_eur_per_kwh.to_numpy()[origin+1:origin+1+h]
forecast={key:{} for key in ('load','pv')}
for i,bus in enumerate(core.BUS_ORDER,1):
    forecast['load'][bus]=frame[f'bus_{i}_nominal_load'].to_numpy()[origin+1:origin+1+h]
    forecast['pv'][bus]=frame[f'bus_{i}_pv'].to_numpy()[origin+1:origin+1+h]
prices,info=load_prices(base.parent,'p2v_ramp','winter','perfect',20250903,frame,cfg,1.)
pb=window_prices(prices,timestamps,state['previous_pb'],core.PHASE4_SETTINGS['PB_MIN'],
                 core.PHASE4_SETTINGS['PB_MAX'],core.PHASE4_SETTINGS['PB_RAMP'])
assert pb is not None
residuals=[]
for bus in core.BUS_ORDER:
    s=follower_start(bus,forecast['load'][bus]-forecast['pv'][bus],state['energy'][bus],pa,np.array(pb),origin)
    stationarity=s['costs']+s['aeq'].T@s['lam']+s['aub'].T@s['mu']-s['lower']+s['upper']
    mask=(s['ub']-s['lb'])>1e-12
    residual=max(np.max(np.abs(s['aeq']@s['primal']-s['beq'])),
                 np.max(np.maximum(s['aub']@s['primal']-s['bub'],0)),
                 np.max(np.abs(stationarity[mask])),
                 np.max(np.abs(s['mu']*(s['bub']-s['aub']@s['primal']))),
                 np.max(np.abs(s['lower']*(s['primal']-s['lb']))),
                 np.max(np.abs(s['upper']*(s['ub']-s['primal']))))
    assert residual<1e-6,(bus,residual)
    residuals.append(float(residual))
results=[]
for name,solve in [('p2v_ramp',core._solve_stackelberg_window),('peak_ramp',peak_solve)]:
    result=wrap(solve)(pd.Timestamp(timestamps[0]),origin,forecast,state['energy'],pa,
        state['previous_pb'],None,warm_pb=pb,mip_gap=.1,time_limit_s=60,
        previous_realized_net=state['previous_grid'])
    assert np.isfinite(result['objective']) and np.isfinite(result['gap'])
    assert max(result[k] for k in ('constraint_violation','bound_violation','integrality_violation'))<=cfg['validation_tolerance']
    results.append(dict(objective=name,value=result['objective'],gap=result['gap'],runtime_s=result['runtime_s']))
path=base/'status/ol_feasibility_smoke.json'
path.write_text(json.dumps(dict(horizon=96,max_follower_kkt_residual=max(residuals),results=results),indent=2))
print('96-step OL smoke passed:',path,flush=True)
