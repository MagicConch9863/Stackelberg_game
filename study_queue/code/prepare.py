"""Build the requested priority plan; reuse only verified, identical realizations."""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
HS = ROOT/'horizon_screening'
sys.path.insert(0, str(HS/'code'))
sys.path.insert(0, str(HS/'source/forecast_mpc'))
from io_utils import atomic_json, atomic_text


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_source(source, season, scenario, seed):
    import numpy as np
    import pandas as pd
    from experiment_core.load_process import build_realization
    manifest = read(source/'manifest.json')
    cfg = read(HS/'configs/experiment.json')
    assert (manifest['season'],manifest['scenario'],manifest['seed'],manifest['method']) == (season,scenario,seed,'MPC')
    assert manifest['prediction_sha256'] == sha(HS/'code/prediction.py')
    assert manifest.get('shared_solver_sha256',manifest['solver_sha256']) == sha(HS/'source/forecast_mpc/experiment_core/stackelberg_mpc.py'), str(source)
    for key in ('rho','degradation_eur_per_kwh','leader_ramp_weight','iid_aggregate_std_fraction','measurement','dt_hours'):
        assert manifest['config'][key] == cfg[key], (source,key)
    assert manifest['config']['horizon_steps'] == 16
    frame = pd.read_csv(HS/'inputs'/f'{season}.csv')
    alpha = read(HS/'inputs/calibration.json')[season]['pv_scale'] if season=='summer' else 1.
    assert abs(manifest['pv_scale']-alpha)<1e-12
    nominal = {'load':{},'pv':{}}
    for i in range(1,5):
        nominal['load'][f'Bus {i}'] = frame[f'bus_{i}_nominal_load'].to_numpy()
        nominal['pv'][f'Bus {i}'] = frame[f'bus_{i}_pv'].to_numpy()*alpha
    mean = float((sum(nominal['load'].values())-sum(nominal['pv'].values()))[17:305].mean())
    sigma = .2*mean/2
    innovation = sigma*math.sqrt(1-.9**2) if scenario=='ar1' else sigma
    actual = build_realization(nominal,scenario,seed=seed,sigma_w=innovation,rho=.9)
    old = pd.read_csv(source/'realization.csv')
    assert old.timestamp.tolist() == frame.timestamp.iloc[:len(old)].tolist()
    for i in range(1,5):
        for key in ('nominal_load','load','pv','disturbance','innovation'):
            np.testing.assert_allclose(old[f'bus_{i}_{key}'],actual[key][f'Bus {i}'][:len(old)],atol=1e-12,rtol=1e-12)
    state = read(source/'checkpoint.json')
    assert len(state['rows'])==state['next_step']
    assert [int(row['execution_step']) for row in state['rows']]==list(range(state['next_step']))
    for row in state['rows']:
        j=int(row['execution_step'])+1
        raw=sum(actual['load'][b][j]-actual['pv'][b][j] for b in actual['load'])
        assert abs(float(row['uncontrolled_net_kw'])-raw)<1e-8
        for i,cap in enumerate((8.,5.,8.,8.),1):
            assert -3e-5<=float(row[f'soc_bus_{i}_kwh'])<=cap+3e-5
    last=state['rows'][-1]
    for i,bus in enumerate(state['energy'],1):
        assert abs(state['energy'][bus]-last[f'soc_bus_{i}_kwh'])<1e-8
    assert abs(state['previous_pb']-last['pb_eur_per_kwh'])<1e-8
    assert abs(state['previous_grid']-last['realized_net_kw'])<1e-8
    return state


def write_warmup(path, source, state):
    import pandas as pd
    path.parent.mkdir(parents=True,exist_ok=True)
    rows=state['rows'][:16]
    assert len(rows)==16
    atomic_text(path,pd.DataFrame(rows).to_csv(index=False))
    atomic_json(path.with_suffix('.json'),dict(source=str(source),checkpoint_sha256=sha(source/'checkpoint.json'),
        warmup_steps=16,validated_identical_realization=True))


def task(stage,season,scenario,seed,objective,method,rho=.9):
    h=16 if method=='MPC' else 96
    name=f'{stage}_{objective}_{season}_{scenario}_S{ {20250903:1,20250917:2,20251001:3,20251015:4,20251029:5}[seed]}_{method}'
    folder=BASE/'results'/('rho_0p5' if stage==1 else 'seed_pairs')/objective/f'{method}_H{h}'/season/scenario/f'seed_{seed}'
    return dict(name=name,stage=stage,season=season,scenario=scenario,seed=seed,objective=objective,
        method=method,horizon=h,rho=rho,output=str(folder),status='pending',attempt=0,retry_after=0)


def main():
    if (BASE/'status/suite_status.json').exists():
        raise RuntimeError('An existing priority queue must be resumed, not overwritten')
    cases=[];reused=[]
    seasons=('winter','spring','summer','autumn')
    for objective in ('p2v_ramp','peak_ramp'):
        for season in seasons:
            warm=BASE/'warmup/rho_0p5'/f'{objective}_{season}_S1.csv'
            for method in ('MPC','OL'):
                item=task(1,season,'ar1',20250903,objective,method,.5)
                item.update(shared_warmup=str(warm),generate_warmup=method=='MPC')
                cases.append(item)
    with (HS/'status/reusable_4h_cases.csv').open(encoding='utf-8-sig',newline='') as stream:
        reusable=list(csv.DictReader(stream))
    assert len(reusable)==32
    for row in reusable:
        season,scenario,objective,seed=row['season'],row['scenario'],row['objective'],int(row['seed'])
        warm=BASE/'warmup/seed_pairs'/f'{objective}_{season}_{scenario}_{seed}.csv'
        if row['reusable']=='True':
            source=Path(row['source'])
            assert sha(source/'official_timeseries.csv')==row['timeseries_sha256'] or sha(source/'timeseries.csv')==row['timeseries_sha256']
            state=validate_source(source,season,scenario,seed)
            assert state['next_step']==304
            write_warmup(warm,source,state)
            reused.append(dict(**row,shared_warmup=str(warm)))
        else:
            item=task(2,season,scenario,seed,objective,'MPC')
            item.update(shared_warmup=str(warm),generate_warmup=True)
            cases.append(item)
        item=task(2,season,scenario,seed,objective,'OL')
        item.update(shared_warmup=str(warm),generate_warmup=False)
        cases.append(item)
    for season,seed,step in [('summer',20251015,99),('autumn',20251029,259)]:
        source=ROOT/f'monte_carlo_3seeds/results/matched/{season}/ar1/seed_{seed}/MPC'
        state=validate_source(source,season,'ar1',seed)
        assert state['next_step']==step
        item=task(3,season,'ar1',seed,'p2v_ramp','MPC')
        output=BASE/'results/recovery'/item['name']
        output.mkdir(parents=True,exist_ok=True)
        atomic_json(output/'checkpoint.json',state)
        warm=BASE/'warmup/recovery'/f'{season}_{seed}.csv'
        write_warmup(warm,source,state)
        atomic_json(output/'recovery_source.json',dict(source=str(source),source_checkpoint_sha256=sha(source/'checkpoint.json'),
            source_manifest_sha256=sha(source/'manifest.json'),resume_step=step,validated_identical_realization=True,
            physical_model_unchanged=True,original_results_preserved=True))
        item.update(output=str(output),shared_warmup=str(warm),generate_warmup=False,recovery_source=str(source))
        cases.append(item)
    assert [sum(t['stage']==s for t in cases) for s in (1,2,3)]==[16,40,2]
    # Copy the latest persisted review state only after the previous coordinator stops.
    atomic_json(BASE/'configs/plan.json',dict(parallel_cases=14,threads_per_case=16,
        refill_seconds=10,two_hour_seconds=7200,hold_8h=True,stages={1:'AR(1), rho=0.5; gleiche marginale Standardabweichung',
        2:'4h MPC und 24h OL, S2/S3',3:'Zwei unterbrochene Monte-Carlo-Faelle',4:'Strenge kontinuierliche Nachpruefung'},
        cases=cases,reused=reused))
    atomic_json(BASE/'status/preflight.json',dict(new_cases=56,recovery_cases=2,reused_MPC_cases=len(reused),
        rho=.5,marginal_amplitude_fixed=True,recovery_steps=[99,259],all_source_checks_passed=True))
    print('Prepared: rho 16, seed pairs 40, recovery 2, reused MPC 24',flush=True)


if __name__=='__main__':
    main()
