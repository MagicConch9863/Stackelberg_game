"""Explicit single-case runner. Importing this module never starts a simulation.

Input: 320 nominal quarter-hour rows starting at 19:45 before the official date.
Columns: timestamp, pa_eur_per_kwh, bus_1_nominal_load, bus_1_pv, ... bus_4_pv.
Both methods use the same KKT solver; only their execution/update interval differs.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time
from io_utils import atomic_json
from solve_policy import merge_candidate, stop_after, quality

BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parent
CORE = BASE / 'source/forecast_mpc'


# Checkpoints produced before the I/O-only repair are mathematically compatible.
IO_COMPATIBLE_RUNNERS = {'6d50fdb6a7f82c030adb9e7f1be82dcdceae9242a0daf39917ac440b543c2d47',
                         'd5644dcb8ce3bd37fb9a9642541eb6e5d5e4bdd88383a6dcfde79155f975a794',
                         '1ecf619f80cef821c5a3f1e63f28c3ab31fdc58fbf43cc678ab432eb8c6c0c9d'}


def main(experiment_base=None, solve_window=None, solver_source=None):
    global BASE
    if experiment_base is not None:
        BASE = Path(experiment_base)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--season', choices=['winter', 'spring', 'summer', 'autumn'], required=True)
    parser.add_argument('--family', choices=['matched', 'same'], required=True)
    parser.add_argument('--scenario', choices=['perfect', 'iid', 'ar1'], required=True)
    parser.add_argument('--method', choices=['OL', 'MPC'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--execute', action='store_true', help='Required to actually solve.')
    args = parser.parse_args()
    if not args.execute:
        parser.error('No simulation started. Explicit --execute is required.')
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(CORE))
    from experiment_core import stackelberg_mpc as solver
    from experiment_core.load_process import build_realization
    from prediction import forecast_window
    solve = solver._solve_stackelberg_window if solve_window is None else solve_window

    cfg = json.loads((BASE/'configs/experiment.json').read_text())
    if cfg['solver_mip_focus'] != 3 or cfg['grid_exchange_mode'] != 'grid_export':
        raise ValueError('Shared solver currently requires MIPFocus=3 and this runner requires signed export.')
    h = cfg['horizon_steps']
    warm = cfg['warmup_steps']
    n = warm + cfg['official_steps']
    if h != 16 or warm != 16 or cfg['measurement'] != 'previous_completed_interval' or cfg['history_steps'] != 1:
        raise ValueError('This version requires H16, four-hour warmup and one completed measurement interval.')
    frame = pd.read_csv(args.input, parse_dates=['timestamp'])
    start = pd.Timestamp(cfg['dates'][args.season]) - pd.Timedelta(hours=4, minutes=15)
    input_times = pd.date_range(start, periods=n+h, freq='15min')
    if len(frame) != len(input_times) or not np.array_equal(frame.timestamp.to_numpy(), input_times.to_numpy()):
        raise ValueError('Input must contain 320 points: history, 304 execution intervals and 15 buffer points.')
    expected = input_times[1:]
    buses = solver.BUS_ORDER
    nominal = {key: {} for key in ('load', 'pv')}
    for i, bus in enumerate(buses, 1):
        for key, column in [('load', f'bus_{i}_nominal_load'), ('pv', f'bus_{i}_pv')]:
            values = frame[column].to_numpy(float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f'Invalid nominal data: {column}')
            nominal[key][bus] = values.copy()
    pa = frame.pa_eur_per_kwh.to_numpy(float)[1:]
    if not np.isfinite(pa).all():
        raise ValueError('Invalid price data')
    loads, pv = sum(nominal['load'].values()), sum(nominal['pv'].values())
    alpha = 1.0
    if args.season == 'summer' and (pv > 0).any():
        alpha = min(1.0, cfg['summer_pv_safety_factor'] * float(np.min(loads[pv > 0]/pv[pv > 0])))
        for bus in buses:
            nominal['pv'][bus] *= alpha
    nominal_grid = sum(nominal['load'].values()) - sum(nominal['pv'].values())
    mean_grid = float(nominal_grid[1+warm:1+n].mean())
    if mean_grid <= 0:
        raise ValueError('Positive nominal mean is required for relative noise scaling')
    sigma_iid = cfg['iid_aggregate_std_fraction'] * mean_grid / math.sqrt(len(buses))
    innovation = sigma_iid * (math.sqrt(1-cfg['rho']**2) if args.family == 'matched' else 1.0)
    sigma = 0.0 if args.scenario == 'perfect' else (sigma_iid if args.scenario == 'iid' else innovation)
    actual = build_realization(nominal, args.scenario, seed=args.seed, sigma_w=sigma, rho=cfg['rho'])
    out = BASE/'results'/args.family/args.season/args.scenario/f'seed_{args.seed}'/args.method
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(args.input.read_bytes()).hexdigest()
    manifest = dict(config=cfg, input_sha256=fingerprint, season=args.season, family=args.family,
                    scenario=args.scenario, seed=args.seed, method=args.method, pv_scale=alpha,
                    nominal_mean_grid_kw=mean_grid, iid_sigma_per_bus=sigma_iid,
                    ar1_innovation_sigma_per_bus=innovation,
                    bus_config=solver.BUS_CONFIG, physical_settings=dict(solver.PHASE4_SETTINGS),
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    prediction_sha256=hashlib.sha256(Path(forecast_window.__code__.co_filename).read_bytes()).hexdigest(),
                    solver_sha256=hashlib.sha256(Path(solver.__file__ if solver_source is None else solver_source).read_bytes()).hexdigest())
    if solver_source is not None:
        manifest['shared_solver_sha256'] = hashlib.sha256(Path(solver.__file__).read_bytes()).hexdigest()
    manifest = json.loads(json.dumps(manifest, allow_nan=False))
    manifest_path = out/'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        previous_hash = previous.pop('runner_sha256', None)
        history = previous.pop('runner_history', [])
        expected_manifest = {k:v for k,v in manifest.items() if k != 'runner_sha256'}
        if previous != expected_manifest or previous_hash not in IO_COMPATIBLE_RUNNERS | {manifest['runner_sha256']}:
            raise ValueError('Existing results use different inputs/configuration/core. Use a new output directory.')
        if previous_hash != manifest['runner_sha256']:
            history = history+[previous_hash]
        if history:
            manifest['runner_history'] = history
    atomic_json(manifest_path, manifest)
    from experiment_core.load_process import save_realization
    save_realization(actual, input_times, out)
    atomic_json(out/'pv_scaling.json', dict(additional_scale=alpha,
        nominal_min_net_kw=float(nominal_grid[1+warm:1+n].min()),
        nominal_mean_grid_kw=mean_grid, source='inputs/'+args.season+'_source.json'))
    solver.PHASE4_SETTINGS.update(GRID_EXCHANGE_MODE='grid_export', GUROBI_THREADS=cfg['threads_per_case'],
        LEADER_RAMP_WEIGHT=cfg['leader_ramp_weight'], BESS_DEGRADATION_EUR_PER_KWH=cfg['degradation_eur_per_kwh'],
        SOLVER_OUTPUT=1)
    state_path = out/'checkpoint.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(
        next_step=0, energy={b: float(solver.BUS_CONFIG[b]['bess_e_init']) for b in buses},
        previous_pb=None, previous_grid=None, warm_pb=None, rows=[], forecasts=[], solves=[])
    stride = h if args.method == 'OL' else 1
    for origin in range(state['next_step'], n, stride):
        forecast = forecast_window(actual, args.method, args.scenario, origin+1, h, cfg['rho'])
        atomic_json(out/'progress.json', dict(status='solving', method=args.method,
            current_step=origin, total=n, decision_time=str(expected[origin]),
            latest_observed_interval=str(input_times[origin])))
        best = None
        attempts = []
        review = cfg.get('solver_review',{})
        strict_review = origin in review.get('origins',[])
        limits = review['time_limits_s'] if strict_review else cfg['time_limits_s']
        for limit in limits:
            with (out/'terminal.log').open('a', encoding='utf-8') as log:
                log.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} origin={origin}/{n} limit={limit}s\n')
            tick = time.monotonic()
            try:
                result = solve(expected[origin], origin, forecast,
                    state['energy'], pa[origin:origin+h], state['previous_pb'], None,
                    warm_pb=(best['pb'] if best is not None else state['warm_pb']),
                    mip_gap=cfg['target_gap'], time_limit_s=limit,
                    previous_realized_net=state['previous_grid'])
                valid = all(math.isfinite(result[k]) for k in ('objective', 'best_bound', 'gap'))
                valid &= max(result[k] for k in ('constraint_violation', 'bound_violation', 'integrality_violation')) <= cfg['validation_tolerance']
                valid &= result['best_bound']<=result['objective']+1e-7*max(1.,abs(result['objective']))
                attempts.append(dict(limit_s=limit, elapsed_s=time.monotonic()-tick, status=result['status'],
                                     gap=result['gap'] if math.isfinite(result['gap']) else None, accepted=valid))
                if valid:
                    best = merge_candidate(best,result)
                atomic_json(out/'progress.json',dict(status='solving',method=args.method,
                    current_step=origin,total=n,decision_time=str(expected[origin]),
                    latest_observed_interval=str(input_times[origin]),last_attempt_limit_s=limit,
                    best_gap=best['gap'] if best is not None and math.isfinite(best['gap']) else None))
                if (strict_review and best is not None and best['gap']<=cfg['target_gap']) or (
                    not strict_review and stop_after(limit,best,cfg)):
                    break
            except RuntimeError as exc:
                if 'no incumbent' not in str(exc).lower():
                    raise
                attempts.append(dict(limit_s=limit, elapsed_s=time.monotonic()-tick, error=str(exc)))
        if best is None or (strict_review and best['gap'] > cfg['target_gap']):
            atomic_json(out/'failure.json', dict(origin=origin, attempts=attempts,
                reason='target gap not reached' if best is not None else 'no valid incumbent'))
            raise RuntimeError(f'No acceptable solution at {origin}; checkpoint retained. No fallback action applied.')
        state['solves'].append(dict(origin=origin, objective=best['objective'], gap=best['gap'],
                                    status=best['status'], attempts=attempts,
                                    best_bound=best['best_bound'], runtime_s=best['runtime_s'],
                                    solver_gap=best['solver_gap'],solver_best_bound=best['solver_best_bound'],
                                    quality=quality(best['gap']),
                                    **{key:best[key] for key in ('objective_kind','peak_component','ramp_component') if key in best}))
        for lead in range(h):
            for i, bus in enumerate(buses, 1):
                state['forecasts'].append(dict(origin=origin, origin_time=str(expected[origin]),
                    target_time=str(expected[origin+lead]), lead=lead, bus=i,
                    latest_observed_interval=str(input_times[origin]),
                    nominal_load=float(actual['nominal_load'][bus][origin+1+lead]),
                    measured_residual=float(actual['load'][bus][origin]-actual['nominal_load'][bus][origin]),
                    predicted_load=float(forecast['load'][bus][lead]),
                    actual_load=float(actual['load'][bus][origin+1+lead]),
                    predicted_pv=float(forecast['pv'][bus][lead]), planned_bess=float(best['bess_plans'][bus][lead]),
                    planned_pb=float(best['pb'][lead])))
        for lead in range(min(stride, n-origin)):
            k = origin+lead
            row = dict(step=k-warm, execution_step=k, timestamp=str(expected[k]), origin=origin,
                       pb_eur_per_kwh=float(best['pb'][lead]), pa_eur_per_kwh=float(pa[k]),
                       uncontrolled_net_kw=float(sum(actual['load'][b][k+1]-actual['pv'][b][k+1] for b in buses)))
            total_action = 0.0
            for i, bus in enumerate(buses, 1):
                action = float(best['bess_plans'][bus][lead])
                if not math.isfinite(action) or abs(action) > solver.BUS_CONFIG[bus]['bess_kw']+cfg['validation_tolerance']:
                    raise RuntimeError(f'Physical power violation at {k}: {bus}')
                before = state['energy'][bus]
                after = before + cfg['dt_hours']*action
                cap = float(solver.BUS_CONFIG[bus]['bess_e_max'])
                if not -cfg['validation_tolerance'] <= after <= cap+cfg['validation_tolerance']:
                    raise RuntimeError(f'Physical energy violation at {k}: {bus}')
                state['energy'][bus] = after
                row.update({f'energy_before_bus_{i}':before, f'soc_bus_{i}_kwh':after,
                            f'bess_bus_{i}_kw':action, f'slack_low_bus_{i}':max(0, .2*cap-after),
                            f'slack_high_bus_{i}':max(0, after-.8*cap)})
                total_action += action
            row['bess_kw'] = total_action
            row['realized_net_kw'] = row['uncontrolled_net_kw'] + total_action
            state['rows'].append(row)
            state['previous_grid'] = row['realized_net_kw']
            state['previous_pb'] = row['pb_eur_per_kwh']
        state['next_step'] = min(origin+stride, n)
        state['warm_pb'] = list(map(float, np.r_[best['pb'][stride:], np.repeat(best['pb'][-1], stride)]))
        atomic_json(state_path, state)
        pd.DataFrame(state['rows']).to_csv(out/'timeseries.csv', index=False)
        pd.DataFrame(state['forecasts']).to_csv(out/'rolling_forecasts.csv', index=False)
        atomic_json(out/'solver_log.json', state['solves'])
        atomic_json(out/'progress.json', dict(status='completed' if state['next_step']==n else 'running',
            method=args.method, current_step=state['next_step'], total=n))
        print(f'{args.method} {args.season} {args.scenario}: {state["next_step"]}/{n}', flush=True)
        with (out/'terminal.log').open('a', encoding='utf-8') as log:
            log.write(f'completed={state["next_step"]}/{n} gap={best["gap"]:.6g}\n')
    official = pd.DataFrame(state['rows']).query('step >= 0')
    official.to_csv(out/'official_timeseries.csv', index=False)
    y = official.realized_net_kw.to_numpy()
    atomic_json(out/'metrics.json', dict(peak_kw=float(y.max()), valley_kw=float(y.min()),
        p2v_kw=float(np.ptp(y)), volatility_kw=float(np.std(np.diff(y), ddof=1)),
        max_ramp_kw=float(np.max(np.abs(np.diff(y)))), total_absolute_ramp_kw=float(np.sum(np.abs(np.diff(y))))))
    audit = pd.DataFrame(state['forecasts']).query('origin >= @warm')
    audit['error_kw'] = audit.predicted_load - audit.actual_load
    audit['squared_error'] = audit.error_kw**2
    audit['absolute_error'] = audit.error_kw.abs()
    scores = audit.groupby(['bus', 'lead']).agg(mse=('squared_error', 'mean'),
        mae_kw=('absolute_error', 'mean'), bias_kw=('error_kw', 'mean'), count=('error_kw', 'size'))
    scores['rmse_kw'] = np.sqrt(scores.pop('mse'))
    scores.to_csv(out/'forecast_accuracy.csv')


if __name__ == '__main__':
    main()
