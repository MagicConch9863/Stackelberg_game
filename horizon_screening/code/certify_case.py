"""Audit saved window plans; keep all original actions, states and KPIs intact."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from io_utils import atomic_json
from deadline_policy import certificate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review-source', type=Path, required=True)
    parser.add_argument('--review-output', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    args, _ = parser.parse_known_args()
    import numpy as np
    import pandas as pd
    from peak_window import core, solve_window as peak_solve
    from deadline_solver import wrap

    base = Path(__file__).resolve().parents[1]
    policy = json.loads((base/'status/deadline_policy.json').read_text())
    source, output = args.review_source, args.review_output
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source/'manifest.json').read_text())
    cfg = manifest['config']
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != manifest['input_sha256']:
        raise ValueError('Input provenance mismatch')
    if hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest() != manifest['solver_sha256']:
        raise ValueError('Core solver provenance mismatch')
    peak_path = base/'code/peak_window.py'
    if hashlib.sha256(peak_path.read_bytes()).hexdigest() != manifest['peak_solver_sha256']:
        raise ValueError('Peak solver provenance mismatch')
    fingerprints = {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                    for name in ('solver_log.json', 'rolling_forecasts.csv', 'timeseries.csv')}
    atomic_json(output/'source.json', dict(source=str(source.resolve()), hashes=fingerprints,
                policy=policy, objective=cfg['objective'], actions_changed=False))
    state_path = output/'certificate_checkpoint.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(hashes=fingerprints, items=[])
    if state['hashes'] != fingerprints:
        raise ValueError('Saved trajectory changed since certificate audit started')
    forecasts = pd.read_csv(source/'rolling_forecasts.csv')
    rows = pd.read_csv(source/'timeseries.csv').set_index('execution_step')
    frame = pd.read_csv(args.input).set_index('timestamp')
    solves = json.loads((source/'solver_log.json').read_text())
    targets = [v for v in solves if v['origin'] >= 16 and v['gap'] > policy['acceptance_gap']]
    completed = {v['origin'] for v in state['items']}
    core.PHASE4_SETTINGS.update(GRID_EXCHANGE_MODE='grid_export',
        GUROBI_THREADS=policy['threads_per_case'], LEADER_RAMP_WEIGHT=cfg['leader_ramp_weight'],
        BESS_DEGRADATION_EUR_PER_KWH=cfg['degradation_eur_per_kwh'], SOLVER_OUTPUT=1)
    method = manifest['method']
    solve = wrap(peak_solve if cfg['objective'] == 'peak_ramp' else core._solve_stackelberg_window,
                 method, policy, output)
    h = cfg['horizon_steps']
    for old in targets:
        origin = old['origin']
        if origin in completed:
            continue
        atomic_json(output/'progress.json', dict(status='certifying', current_step=len(state['items']),
            total=len(targets), current_origin=origin, audit_only=True))
        window = forecasts[forecasts.origin == origin]
        predicted = {'load': {}, 'pv': {}}
        saved_plan = {}
        times, prices = None, None
        for i, bus in enumerate(core.BUS_ORDER, 1):
            data = window[window.bus == i].sort_values('lead')
            if data.lead.tolist() != list(range(h)):
                raise ValueError(f'Incomplete saved forecast at {origin}, bus={bus}')
            predicted['load'][bus] = data.predicted_load.to_numpy(float)
            predicted['pv'][bus] = data.predicted_pv.to_numpy(float)
            saved_plan[bus] = data.planned_bess.to_numpy(float)
            times = data.target_time.tolist()
            prices = data.planned_pb.to_numpy(float)
        current, previous = rows.loc[origin], rows.loc[origin-1]
        energies = {bus: float(current[f'energy_before_bus_{i}'])
                    for i, bus in enumerate(core.BUS_ORDER, 1)}
        pa = frame.loc[times, 'pa_eur_per_kwh'].to_numpy(float)
        tick = time.monotonic()
        item = dict(origin=origin, original_objective=old['objective'], original_gap=old['gap'])
        try:
            result = solve(pd.Timestamp(times[0]), origin, predicted, energies, pa,
                float(previous.pb_eur_per_kwh), None, warm_pb=prices,
                previous_realized_net=float(previous.realized_net_kw), saved_plan=saved_plan)
            if not all(math.isfinite(result[k]) for k in ('objective', 'best_bound', 'gap')):
                raise ValueError('Nonfinite result')
            if max(result[k] for k in ('constraint_violation','bound_violation','integrality_violation')) > cfg['validation_tolerance']:
                raise ValueError('Numerical feasibility check failed')
            old_lower = old['best_bound']
            item.update(certificate(old['objective'], old_lower, result['best_bound'],
                                    policy['target_gap'], policy['acceptance_gap']))
            item.update(candidate_objective=result['objective'], candidate_gap=result['gap'],
                        solver_status=result['status'], stages=result['deadline_stages'])
        except (RuntimeError, ValueError) as exc:
            item.update(quality='unresolved', error=str(exc), actions_changed=False)
        item['elapsed_s'] = time.monotonic()-tick
        state['items'].append(item)
        atomic_json(state_path, state)
        print(f'Certificate {source.name}: {len(state["items"])}/{len(targets)} origin={origin} {item["quality"]}', flush=True)
    unresolved = sum(v['quality'] not in ('target_met', 'acceptable_20pct') for v in state['items'])
    summary = dict(audit_complete=True, source=str(source.resolve()), windows_checked=len(state['items']),
        total_windows=len(targets), windows_unresolved=unresolved,
        quality_status='acceptable_20pct' if not unresolved else 'needs_further_review',
        actions_changed=False, source_hashes=fingerprints)
    atomic_json(output/'certificate_summary.json', summary)
    atomic_json(output/'progress.json', dict(status='completed', current_step=len(targets),
                total=len(targets), audit_only=True, quality_status=summary['quality_status']))


if __name__ == '__main__':
    main()
