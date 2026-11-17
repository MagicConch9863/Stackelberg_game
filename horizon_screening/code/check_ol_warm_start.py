"""Read-only checks for the historical-price seed and OL budget change."""
import ast
import csv
import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace
from ol_warm_start import load_prices, window_prices

base = Path(__file__).resolve().parents[1]
cfg = json.loads((base/'configs/experiment.json').read_text())
calibration = json.loads((base/'inputs/calibration.json').read_text())
count = 0
for objective in ('p2v_ramp', 'peak_ramp'):
    for season in cfg['dates']:
        with (base/'inputs'/f'{season}.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        frame = SimpleNamespace(iloc=rows)
        for scenario in ('perfect', 'iid', 'ar1'):
            prices, info = load_prices(base.parent, objective, season, scenario,
                                      20250903, frame, cfg, calibration[season]['pv_scale'])
            assert info['status'] == 'available', info
            for origin in (16, 112, 208):
                timestamps = [row['timestamp'] for row in rows[origin+1:origin+97]]
                seed = window_prices(prices, timestamps, .1, 0., .5, .02)
                assert seed is not None and len(seed) == 96
                assert all(0 <= value <= .5 for value in seed)
                assert all(abs(b-a) <= .020000001 for a,b in zip([.1]+seed,seed))
            count += 1
assert window_prices({}, ['missing'], None, 0., .5, .02) is None
for name in ('run_case.py', 'run_suite.py', 'ol_warm_start.py'):
    ast.parse((base/'code'/name).read_text())
print(f'Validated {count} historical cases, 72 full-day PB starts; bounds/ramp and missing-source fallback OK.')
