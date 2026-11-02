"""Create a preserved-prefix replay; do not modify the original completed result."""
from pathlib import Path
import hashlib
import json
from io_utils import atomic_json,atomic_text

BASE = Path(__file__).resolve().parents[1]


def main():
    source=BASE/'results/matched/summer/perfect/seed_20250903/MPC'
    review_base=BASE/'review/summer_perfect'
    target=review_base/'results/matched/summer/perfect/seed_20250903/MPC'
    target.mkdir(parents=True,exist_ok=True)
    (review_base/'configs').mkdir(exist_ok=True)
    (review_base/'logs').mkdir(exist_ok=True)
    original=json.loads((source/'checkpoint.json').read_text())
    origin=107
    before=original['rows'][origin]
    previous=original['rows'][origin-1]
    plans=sorted([r for r in original['forecasts'] if r['origin']==origin-1 and r['bus']==1],key=lambda r:r['lead'])
    warm=[r['planned_pb'] for r in plans][1:]+[plans[-1]['planned_pb']]
    state=dict(next_step=origin,energy={f'Bus {i}':before[f'energy_before_bus_{i}'] for i in range(1,5)},
        previous_pb=previous['pb_eur_per_kwh'],previous_grid=previous['realized_net_kw'],warm_pb=warm,
        rows=original['rows'][:origin],forecasts=[r for r in original['forecasts'] if r['origin']<origin],
        solves=[r for r in original['solves'] if r['origin']<origin])
    if (target/'checkpoint.json').exists():
        raise FileExistsError('Review checkpoint already exists; refusing to overwrite progress')
    cfg=json.loads((BASE/'configs/experiment.json').read_text())
    cfg['version']='summer_perfect_origin107_quality_review_v1'
    cfg['solver_review']=dict(origins=[107,108],time_limits_s=[120,600,1200],require_target_gap=True)
    atomic_json(review_base/'configs/experiment.json',cfg)
    atomic_json(target/'checkpoint.json',state)
    atomic_json(target/'progress.json',dict(status='pending',current_step=origin,total=304))
    old=next(r for r in original['solves'] if r['origin']==origin)
    atomic_json(review_base/'baseline_audit.json',dict(source=str(source),source_sha256=hashlib.sha256((source/'checkpoint.json').read_bytes()).hexdigest(),
        replay_origin=origin,time=before['timestamp'],original_solution=old,
        previous_grid=state['previous_grid'],original_grid=before['realized_net_kw'],
        original_bess=before['bess_kw'],initial_energy=state['energy'],
        max_prediction_error=max(abs(r['predicted_load']-r['actual_load']) for r in original['forecasts'] if r['origin']==origin)))
    atomic_text(review_base/'Anleitung.txt','Original summer Perfect P2V MPC replay from origin 107 (15.07.2025 22:45).\n'
        'Original completed results are retained. Prefix 0..106 is unchanged.\n'
        'Origins 107/108 require <=10% gap; attempts 120/600/1200 seconds.\n'
        'If quality is not reached, stop before applying the action. Later origins use the original policy.\n'
        'This is a solver-quality sensitivity run, not a replacement for the original common-budget comparison.\n'
        'Progress is in results/matched/summer/perfect/seed_20250903/MPC/progress.json.\n')
    task=dict(name='review_summer_perfect_MPC',season='summer',scenario='perfect',method='MPC',
        objective='p2v_ramp_review',suite_base=str(review_base),output=str(target),
        status='pending',isolated_startup=True)
    atomic_json(BASE/'configs/additional_tasks.json',[task])
    print('Prepared review from origin 107; original case retained.')


if __name__=='__main__':
    main()
