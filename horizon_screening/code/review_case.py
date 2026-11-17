"""Replay a flagged trajectory in a separate directory, preserving its source."""
import json
import math
from pathlib import Path
import shutil

from io_utils import atomic_json


def matching_candidate(folder, origin, energies, horizon):
    """Only reuse an incumbent for the identical window and initial energy state."""
    try:
        candidate=json.loads((Path(folder)/'latest_candidate.json').read_text())
        if candidate['origin']!=origin or len(candidate['pb'])!=horizon:return None
        if set(candidate['energies'])!=set(energies):return None
        if any(not math.isfinite(float(candidate['energies'][b])) or
               abs(float(candidate['energies'][b])-float(v))>1e-8 for b,v in energies.items()):return None
        if not all(math.isfinite(float(v)) for v in candidate['pb']):return None
        if not all(math.isfinite(float(candidate[k])) for k in ('objective','best_bound','gap')):return None
        return candidate
    except (OSError,ValueError,KeyError,TypeError):
        return None


def targets_for(solves, failed_origin=None):
    targets = {}
    for item in solves:
        gap = item.get('gap')
        if gap is None or not math.isfinite(gap):
            targets[str(item['origin'])] = .10
        elif gap > .10:
            targets[str(item['origin'])] = .10 if gap > .20 else .02
    if failed_origin is not None:
        targets.setdefault(str(failed_origin), .10)
    return targets


def prepare(source, destination, failed_origin=None, targets=None):
    source, destination = Path(source), Path(destination)
    if (destination/'review_plan.json').exists():
        return
    state = json.loads((source/'checkpoint.json').read_text())
    targets = targets_for(state['solves'], failed_origin) if targets is None else dict(targets)
    if not targets:
        raise ValueError('No failed or high-gap window to review')
    origin = min(map(int, targets))
    if origin < 16 or origin > state['next_step']:
        raise ValueError('Review origin outside the saved trajectory')
    rows = [row for row in state['rows'] if row['execution_step'] < origin]
    if not rows or rows[-1]['execution_step'] != origin-1:
        raise ValueError('Missing contiguous state before review origin')
    last = rows[-1]
    # Energy keys preserve the runner's BUS_ORDER.
    manifest = json.loads((source/'manifest.json').read_text())
    horizon = manifest['config']['horizon_steps']
    price_rows = sorted((r for r in state['forecasts']
                         if r['origin']==origin and r.get('bus')==1), key=lambda r:r['lead'])
    warm_prices = ([float(r['planned_pb']) for r in price_rows]
                   if [r['lead'] for r in price_rows]==list(range(horizon))
                   else [float(last['pb_eur_per_kwh'])]*horizon)
    buses = list(state['energy'])
    state.update(next_step=origin, rows=rows,
        energy={bus:float(last[f'soc_bus_{i}_kwh']) for i,bus in enumerate(buses,1)},
        previous_pb=float(last['pb_eur_per_kwh']),
        previous_grid=float(last['realized_net_kw']),
        warm_pb=warm_prices,
        forecasts=[row for row in state['forecasts'] if row['origin'] < origin],
        solves=[row for row in state['solves'] if row['origin'] < origin])
    destination.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source/'manifest.json',destination/'manifest.json')
    shutil.copy2(source/'checkpoint.json',destination/'original_checkpoint.json')
    atomic_json(destination/'checkpoint.json',state)
    atomic_json(destination/'review_plan.json',dict(source=str(source.resolve()),
        replay_from=origin, targets=targets,
        policy='gap_above_20_to_10;gap_10_to_20_to_2;failed_to_10',
        continuous_replay=True, original_results_unchanged=True))


if __name__ == '__main__':
    import sys
    source_index=sys.argv.index('--review-source')
    source=Path(sys.argv[source_index+1])
    del sys.argv[source_index:source_index+2]
    destination=Path(sys.argv[sys.argv.index('--review-output')+1])
    quality_review = '--quality-review' in sys.argv
    if quality_review:
        sys.argv.remove('--quality-review')
    failed_path=source/'failure.json'
    failed_origin=json.loads(failed_path.read_text())['origin'] if failed_path.exists() else None
    if '--failed-origin' in sys.argv:
        index=sys.argv.index('--failed-origin')
        failed_origin=int(sys.argv[index+1])
        del sys.argv[index:index+2]
    # Completed sources can retain stale failure files from recovered attempts.
    if (source/'checkpoint.json').exists():
        if json.loads((source/'checkpoint.json').read_text())['next_step']==304:
            failed_origin=None
    prepare(source,destination,failed_origin)
    if quality_review:
        plan_path=destination/'review_plan.json'
        plan=json.loads(plan_path.read_text())
        plan['quality_review']=True
        atomic_json(plan_path,plan)
    from run_case import main
    main()
