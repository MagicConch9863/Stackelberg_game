"""Queue continuous retries from unresolved audited windows; preserve source results."""
import hashlib
import json
from pathlib import Path
import time

from io_utils import atomic_json, exclusive_lock
from review_case import prepare


def main():
    base=Path(__file__).resolve().parents[1]
    status=base/'status'
    if not (status/'STOP').exists():
        raise RuntimeError('Pause dispatch before editing the durable queue')
    with exclusive_lock(status/'coordinator.lock'):
        path=status/'suite_status.json'
        suite=json.loads(path.read_text())
        tasks=suite['cases']
        added=[]
        audits=[t for t in tasks if t.get('review_mode')=='certificate']
        # Retain the earlier 6h priority, then 4h/2h, followed by OL.
        order={24:0,16:1,8:2,96:3}
        for audit in sorted(audits,key=lambda t:(order.get(t['horizon'],4),t['name'])):
            if audit['horizon']==32:
                continue
            root=audit.get('source_case',audit['name'].removesuffix('_certificate'))
            name=root+'_budget_retry'
            if any(t['name']==name for t in tasks):
                continue
            audit_path=Path(audit['output'])/'certificate_checkpoint.json'
            if not audit_path.exists():
                continue
            checkpoint=json.loads(audit_path.read_text())
            bad=[item for item in checkpoint['items']
                 if item.get('quality')=='needs_further_review' or item.get('error')]
            if not bad:
                continue
            source=Path(audit['review_source'])
            destination=base/'review/budget_retry'/root
            targets={str(item['origin']):.1 for item in bad}
            prepare(source,destination,targets=targets)
            plan_path=destination/'review_plan.json'
            plan=json.loads(plan_path.read_text())
            plan.update(policy='MPC_60_120_600_accept_feasible_after_second;OL_3600',
                audit_source=str(audit_path), source_checkpoint_sha256=hashlib.sha256(
                    (source/'checkpoint.json').read_bytes()).hexdigest(),
                audit_complete_at_queue_time=audit['status']=='completed',
                quality_target_is_not_an_execution_gate=True)
            atomic_json(plan_path,plan)
            task={key:audit[key] for key in
                  ('seed','seed_label','season','objective','horizon','method')}
            task.update(name=name,source_case=root,review=True,review_mode='budget_retry',
                review_source=str(source),review_failed_origin=None,
                review_target_windows=len(targets),output=str(destination),
                status='pending',retries=0,startup_deferrals=2,retry_after=0,
                ol_time_limit_s=3600,queued_at=time.strftime('%Y-%m-%d %H:%M:%S'))
            task['scenario']=audit['scenario']
            tasks.append(task)
            added.append(dict(name=name,method=task['method'],horizon=task['horizon'],
                replay_from=plan['replay_from'],flagged_windows=len(targets),
                remaining_execution_steps=304-plan['replay_from']))
        suite['counts']={value:sum(t['status']==value for t in tasks)
                         for value in ('completed','running','pending','failed','paused')}
        atomic_json(path,suite)
        atomic_json(status/'budget_retry_queue.json',dict(
            queued_at=time.strftime('%Y-%m-%d %H:%M:%S'),added=len(added),cases=added))
        print(f'Queued {len(added)} continuous budget retries; original trajectories preserved',flush=True)
        for method in ('MPC','OL'):
            selected=[t for t in added if t['method']==method]
            print(f'{method}: {len(selected)} cases, '+
                  f'{sum(t["remaining_execution_steps"] for t in selected)} execution steps',flush=True)


if __name__=='__main__':
    main()
