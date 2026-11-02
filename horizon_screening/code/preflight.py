"""Check shared initial states, actual disturbances and every horizon's LP shape."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'source/forecast_mpc'))
from experiment_core import stackelberg_mpc as core
from experiment_core.load_process import build_realization
from prediction import forecast_window

cfg=json.loads((BASE/'configs/experiment.json').read_text())
sources=json.loads((BASE/'inputs/warmup_sources.json').read_text())
cal=json.loads((BASE/'inputs/calibration.json').read_text())
checked=0
for season in cfg['dates']:
    frame=pd.read_csv(BASE/f'inputs/{season}.csv')
    nominal={'load':{},'pv':{}}
    for i,bus in enumerate(core.BUS_ORDER,1):
        nominal['load'][bus]=frame[f'bus_{i}_nominal_load'].to_numpy()
        nominal['pv'][bus]=frame[f'bus_{i}_pv'].to_numpy()*cal[season]['pv_scale']
    mean=float((sum(nominal['load'].values())-sum(nominal['pv'].values()))[17:305].mean())
    sigma=.2*mean/2
    for scenario in ('perfect','iid','ar1'):
        actual=build_realization(nominal,scenario,seed=20250903,
            sigma_w=0 if scenario=='perfect' else sigma*(np.sqrt(1-.9**2) if scenario=='ar1' else 1),
            rho=.9)
        reference=pd.read_csv(BASE.parent/f'four_hour_test/results/matched/{season}/{scenario}/seed_20250903/MPC/realization.csv')
        for i,bus in enumerate(core.BUS_ORDER,1):
            assert np.max(abs(actual['load'][bus][:320]-reference[f'bus_{i}_load']))<1e-9
        for objective in ('p2v_ramp','peak_ramp'):
            warm=pd.read_csv(BASE/'inputs'/sources[f'{objective}/{season}/{scenario}']['file'])
            for j,row in warm.iterrows():
                assert abs(row.uncontrolled_net_kw-sum(actual['load'][b][j+1]-actual['pv'][b][j+1] for b in core.BUS_ORDER))<1e-8
            for h in (8,16,24,32,96):
                method='OL' if h==96 else 'MPC'
                forecast=forecast_window(actual,method,scenario,17,h,.9)
                # Independent primal feasible-point check; no optimization.
                for i,bus in enumerate(core.BUS_ORDER,1):
                    baseline=forecast['load'][bus]-forecast['pv'][bus]
                    energy=float(warm.iloc[-1][f'soc_bus_{i}_kwh'])
                    a,b,g,d,lb,ub,*_=core._follower_lp(bus,baseline,energy,16)
                    capacity=core.BUS_CONFIG[bus]['bess_e_max']
                    y=np.r_[np.zeros(h),np.maximum(baseline,0),np.maximum(-baseline,0),
                        np.full(h,max(0,.2*capacity-energy)),
                        np.full(h,max(0,energy-.8*capacity)),np.zeros(h)]
                    assert a.shape[1]==6*h and g.shape[1]==6*h
                    assert np.max(abs(a@y-b))<3e-5
                    assert np.max(g@y-d)<3e-5
                    assert np.max(lb-y)<3e-5 and np.max(y-ub)<3e-5
                checked+=1
(BASE/'status/preflight.json').write_text(json.dumps({
    'horizon_cases_checked':checked,'shared_warmup_verified':True,
    'actual_noise_matches_S1_reference':True,'primal_lp_shapes_and_feasible_points_checked':True,
    'no_milp_solved':True},indent=2),encoding='utf-8')
print('Preflight passed:',checked,'case/horizon combinations; no MILP solved.')
