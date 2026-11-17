"""Read-only checks of rho-zero equivalence and reusable completed 4h MPC cases."""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'horizon_screening'
sys.path.insert(0,str(ROOT/'venv/Lib/site-packages'))
sys.path.insert(0,str(BASE/'source/forecast_mpc'))
import numpy as np
import pandas as pd
from experiment_core.load_process import build_realization
from prediction import forecast_window


def main():
    frames={}
    amplitude={}
    matched=[]
    forecasts_checked=0
    for season in ('winter','spring','summer','autumn'):
        frame=pd.read_csv(BASE/'inputs'/f'{season}.csv')
        manifest=json.loads((BASE/'results/p2v_ramp/MPC_H16'/season/
                            'ar1/seed_20250903/manifest.json').read_text())
        scale=manifest['pv_scale']
        nominal={key:{} for key in ('load','pv')}
        for i in range(1,5):
            nominal['load'][f'Bus {i}']=frame[f'bus_{i}_nominal_load'].to_numpy(float)
            nominal['pv'][f'Bus {i}']=frame[f'bus_{i}_pv'].to_numpy(float)*scale
        frames[season]=(frame,nominal)
        amplitude[season]=manifest['iid_sigma_per_bus']
        for seed in (20250903,20250917,20251001):
            iid=build_realization(nominal,'iid',seed=seed,sigma_w=amplitude[season],rho=0.)
            ar0=build_realization(nominal,'ar1',seed=seed,sigma_w=amplitude[season],rho=0.)
            for key in ('load','pv','disturbance','innovation'):
                for bus in nominal['load']:
                    np.testing.assert_array_equal(iid[key][bus],ar0[key][bus])
            assert any(np.any(a!=0.) for a in ar0['disturbance'].values())
            for h,method in itertools.product((8,16,24,32),('OL','MPC')):
                for start in range(1,305):
                    a=forecast_window(iid,method,'iid',start,h,.9)
                    b=forecast_window(ar0,method,'ar1',start,h,0.)
                    for key in ('load','pv'):
                        for bus in nominal['load']:
                            np.testing.assert_array_equal(a[key][bus],b[key][bus])
                    forecasts_checked+=1
            matched.append(dict(season=season,seed=seed,
                nonzero_disturbance=True,actual_exactly_equal=True,forecasts_exactly_equal=True))

    selected=list(csv.DictReader((ROOT/'four_hour_test/analysis_data/daily_kpi/data_sources.csv').open()))
    inventory=[]
    reference=json.loads((BASE/'results/p2v_ramp/MPC_H16/winter/ar1/seed_20250903/manifest.json').read_text())
    expected=dict(horizon_steps=16,warmup_steps=16,official_steps=288,dt_hours=.25,rho=.9,
        leader_ramp_weight=.05,degradation_eur_per_kwh=.05,grid_exchange_mode='grid_export',
        measurement='previous_completed_interval',history_steps=1,ol_forecast='nominal')
    for seed,objective,season,scenario in itertools.product(
            (20250917,20251001),('p2v_ramp','peak_ramp'),
            ('winter','spring','summer','autumn'),('iid','ar1')):
        source=None
        if seed==20250917:
            entry=next((v for v in selected if int(v['seed'])==seed and v['objective']==objective
                        and v['season']==season and v['scenario']==scenario and v['method']=='MPC'),None)
            if entry:source=ROOT/entry['source']
        elif objective=='p2v_ramp':
            source=ROOT/'monte_carlo_3seeds/results/matched'/season/scenario/f'seed_{seed}'/'MPC'
        row=dict(seed=seed,objective=objective,season=season,scenario=scenario,
                 method='MPC',horizon=16,source=str(source) if source else '',reusable=False,reason='')
        if source is None or not (source/'metrics.json').exists():
            row['reason']='No completed source'
            inventory.append(row);continue
        manifest=json.loads((source/'manifest.json').read_text())
        try:
            assert all(manifest['config'][key]==value for key,value in expected.items()),'Configuration mismatch'
            shared=manifest.get('shared_solver_sha256',manifest['solver_sha256'])
            assert shared==reference['solver_sha256'],'Core formulation mismatch'
            assert manifest['prediction_sha256']==reference['prediction_sha256'],'Forecast mismatch'
            assert manifest['bus_config']==reference['bus_config'],'BESS configuration mismatch'
            assert abs(manifest['iid_sigma_per_bus']-amplitude[season])<1e-12,'Amplitude mismatch'
            reader=(source/'checkpoint.json').open()
            with reader:header=reader.read(128)
            import re
            assert re.search(r'"next_step"\s*:\s*304\s*[,}]',header),'Incomplete checkpoint'
            output=pd.read_csv(source/'official_timeseries.csv')
            assert len(output)==288,'Incomplete official output'
            actual=pd.read_csv(source/'realization.csv')
            frame,nominal=frames[season]
            prices=frame.set_index('timestamp').loc[output.timestamp,'pa_eur_per_kwh']
            np.testing.assert_allclose(output.pa_eur_per_kwh,prices,rtol=1e-12,atol=1e-12)
            assert actual.timestamp.tolist()==frame.timestamp.iloc[:len(actual)].tolist(),'Timestamp mismatch'
            sigma=amplitude[season]*(np.sqrt(1-.9**2) if scenario=='ar1' else 1.)
            target=build_realization(nominal,scenario,seed=seed,sigma_w=sigma,rho=.9)
            for bus in nominal['load']:
                tag=bus.replace(' ','_').lower()
                for key in ('load','pv','nominal_load'):
                    np.testing.assert_allclose(actual[f'{tag}_{key}'],target[key][bus][:len(actual)],
                                               rtol=1e-12,atol=1e-12)
            row.update(reusable=True,reason='Use this saved MPC prehistory as the seed-specific common warmup',
                       timeseries_sha256=hashlib.sha256((source/'official_timeseries.csv').read_bytes()).hexdigest())
        except (AssertionError,KeyError,FileNotFoundError) as exc:
            row['reason']=str(exc)
        inventory.append(row)
    folder=BASE/'status'
    report=dict(rho_zero_checks=matched,forecast_windows_checked=forecasts_checked,
        disturbance_is_nonzero=True,
        conclusion='Identical disturbances and forecasts with matched sigma and common RNG; identical MILP given identical state and anchors; solver trajectories need not be bitwise identical',
        additional_seeds=[20250917,20251001],reusable_mpc_cases=sum(v['reusable'] for v in inventory),
        baseline_extension_required_cases=64,
        baseline_extension_missing=64-sum(v['reusable'] for v in inventory))
    (folder/'rho_zero_reuse_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (folder/'reusable_4h_cases.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        fields=list(dict.fromkeys(k for row in inventory for k in row))
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(inventory)
    print(json.dumps({k:v for k,v in report.items() if k!='rho_zero_checks'},indent=2),flush=True)
    for row in inventory:
        if not row['reusable']:print('MISSING',row['seed'],row['objective'],row['season'],row['scenario'],row['reason'],flush=True)


if __name__=='__main__':
    main()
