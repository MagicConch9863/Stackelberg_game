"""Extend frozen nominal profiles and record one common measured warmup per pair."""
from pathlib import Path
import hashlib
import json
import sys
import pandas as pd
import numpy as np

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parent
sys.path.insert(0,str(ROOT/'hpc_seasonal_actual_load_noise/src/forecast_mpc'))
from experiment_core import phase1_closed_loop as source


def main():
    cfg=json.loads((BASE/'configs/experiment.json').read_text())
    calibration={};warmups={}
    for season,date in cfg['dates'].items():
        old=pd.read_csv(ROOT/f'four_hour_test/inputs/{season}.csv',parse_dates=['timestamp'])
        start=pd.Timestamp(date)-pd.Timedelta(hours=4,minutes=15)
        source.START_DATE=start.normalize()
        offset=int((start-start.normalize()).total_seconds()/900)
        _,nominal,pa,times=source._load_period(offset+336,32,'perfect')
        sl=slice(offset,offset+336)
        data={'timestamp':times[sl],'pa_eur_per_kwh':pa[sl]}
        for i,bus in enumerate(source.BUS_ORDER,1):
            data[f'bus_{i}_nominal_load']=nominal['load'][bus][sl]
            data[f'bus_{i}_pv']=nominal['pv'][bus][sl]
        frame=pd.DataFrame(data)
        assert len(frame)==336
        assert np.array_equal(frame.timestamp.iloc[:320].to_numpy(),old.timestamp.to_numpy())
        assert np.max(abs(frame.iloc[:320,1:].to_numpy()-old.iloc[:,1:].to_numpy()))<1e-9
        frame.to_csv(BASE/f'inputs/{season}.csv',index=False)
        m=json.loads((ROOT/f'four_hour_test/results/matched/{season}/perfect/seed_20250903/MPC/manifest.json').read_text())
        calibration[season]={'pv_scale':m['pv_scale'],'source_320_rows_match':True,
            'nominal_mean_grid_kw':m['nominal_mean_grid_kw']}
        for objective,folder in [('p2v_ramp','four_hour_test'),('peak_ramp','four_hour_peak_ramp')]:
            for scenario in ('perfect','iid','ar1'):
                path=ROOT/f'{folder}/results/matched/{season}/{scenario}/seed_20250903/MPC/timeseries.csv'
                rows=pd.read_csv(path).iloc[:16].copy()
                assert np.array_equal(rows.step,np.arange(-16,0))
                relative=f'warmup/{objective}_{season}_{scenario}.csv'
                dest=BASE/'inputs'/relative
                dest.parent.mkdir(exist_ok=True)
                rows.to_csv(dest,index=False)
                warmups[f'{objective}/{season}/{scenario}']={
                    'file':relative,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
                    'source':str(path.relative_to(ROOT)),
                    'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'interpretation':'Common 4h prehistory from existing S1 MPC; all tested horizons share this state'}
        print(f'{season}: 336 rows, original prefix checked, common warmup saved',flush=True)
    (BASE/'inputs/calibration.json').write_text(json.dumps(calibration,indent=2),encoding='utf-8')
    (BASE/'inputs/warmup_sources.json').write_text(json.dumps(warmups,indent=2),encoding='utf-8')


if __name__=='__main__':main()
