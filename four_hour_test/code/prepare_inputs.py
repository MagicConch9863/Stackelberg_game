"""Prepare nominal inputs only. Does not create or solve an optimization model."""
from pathlib import Path
import json
import sys


def main():
    import pandas as pd
    base = Path(__file__).resolve().parents[1]
    core = base.parent/'hpc_seasonal_actual_load_noise/src/forecast_mpc'
    sys.path.insert(0, str(core))
    from experiment_core import phase1_closed_loop as source
    cfg = json.loads((base/'configs/experiment.json').read_text())
    out = base/'inputs'
    out.mkdir(exist_ok=True)
    for season, date in cfg['dates'].items():
        start = pd.Timestamp(date)-pd.Timedelta(hours=4, minutes=15)
        source.START_DATE = start.normalize()
        offset = int((start-start.normalize()).total_seconds()/900)
        count = cfg['history_steps']+cfg['warmup_steps']+cfg['official_steps']+cfg['horizon_steps']-1
        settings, nominal, pa, times = source._load_period(
            cfg['history_steps']+cfg['warmup_steps']+cfg['official_steps']+offset, cfg['horizon_steps'], 'perfect')
        selected = slice(offset, offset+count)
        data = {'timestamp': times[selected], 'pa_eur_per_kwh': pa[selected]}
        for i, bus in enumerate(source.BUS_ORDER, 1):
            data[f'bus_{i}_nominal_load'] = nominal['load'][bus][selected]
            data[f'bus_{i}_pv'] = nominal['pv'][bus][selected]
        path = out/f'{season}.csv'
        if path.exists():
            raise FileExistsError(f'Refusing to replace existing input: {path}')
        pd.DataFrame(data).to_csv(path, index=False)
        (out/f'{season}_source.json').write_text(json.dumps(dict(
            source_module=str(Path(source.__file__).resolve()), settings=settings,
            pv_scaling='Applied later by run_case; these inputs are unscaled',
            rows=count), indent=2, default=str), encoding='utf-8')
        print(path)


if __name__ == '__main__':
    main()
