"""Reuse timestamp-aligned prices, never old actions or a feasibility claim."""
import csv
import hashlib
import json
import math
from pathlib import Path


def load_prices(project, objective, season, scenario, seed, frame, cfg, pv_scale):
    batch = 'four_hour_test' if objective == 'p2v_ramp' else 'four_hour_peak_ramp'
    base = Path(project) / batch
    source = base / 'results/matched' / season / scenario / f'seed_{seed}' / 'OL'
    manifest_path = source / 'manifest.json'
    path = source / 'timeseries.csv'
    if not path.exists() or not manifest_path.exists():
        return {}, dict(status='unavailable', source=str(path))
    manifest = json.loads(manifest_path.read_text())
    keys = ('dt_hours', 'rho', 'iid_aggregate_std_fraction', 'leader_ramp_weight',
            'degradation_eur_per_kwh', 'grid_exchange_mode', 'measurement')
    if any(manifest['config'].get(k) != cfg[k] for k in keys):
        return {}, dict(status='incompatible_configuration', source=str(path))
    if abs(manifest['pv_scale'] - pv_scale) > 1e-10:
        return {}, dict(status='incompatible_pv_scale', source=str(path))
    with (Path(project) / 'four_hour_test/inputs' / f'{season}.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    for index, row in enumerate(rows):
        if str(frame.iloc[index]['timestamp']) != row['timestamp']:
            return {}, dict(status='incompatible_timestamps', source=str(path))
        if any(abs(float(frame.iloc[index][key])-float(value)) > 1e-10
               for key, value in row.items() if key != 'timestamp'):
            return {}, dict(status='incompatible_inputs', source=str(path))
    with path.open(newline='', encoding='utf-8') as stream:
        prices = {row['timestamp']:float(row['pb_eur_per_kwh']) for row in csv.DictReader(stream)}
    if not all(math.isfinite(value) for value in prices.values()):
        return {}, dict(status='nonfinite_prices', source=str(path))
    return prices, dict(status='available', source=str(path),
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        kind='partial_PB_MIP_start_only')


def window_prices(prices, timestamps, previous_pb, lower, upper, ramp):
    keys = [str(t) for t in timestamps]
    if not all(key in prices for key in keys):
        return None
    result = []
    previous = previous_pb
    for key in keys:
        lo, hi = lower, upper
        if previous is not None:
            lo, hi = max(lo, previous-ramp), min(hi, previous+ramp)
        value = min(hi, max(lo, prices[key]))
        result.append(value)
        previous = value
    return result
