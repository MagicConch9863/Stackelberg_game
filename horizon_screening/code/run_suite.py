"""Headless 48-case scheduler with PID reattachment and durable checkpoints."""
from pathlib import Path
import argparse
import csv
import ctypes
from ctypes import wintypes
import io
import json
import os
import re
import subprocess
import sys
import time
from io_utils import atomic_json,atomic_text,exclusive_lock
from progress_view import render_progress

BASE=Path(__file__).resolve().parents[1]
STATUS=BASE/'status'
KERNEL=ctypes.WinDLL('kernel32',use_last_error=True)
KERNEL.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
KERNEL.OpenProcess.restype=wintypes.HANDLE
KERNEL.CloseHandle.argtypes=[wintypes.HANDLE]
KERNEL.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
KERNEL.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
KERNEL.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT]


def process_info(pid):
    handle=KERNEL.OpenProcess(0x1000,False,pid)
    if not handle:return None,None
    code=wintypes.DWORD();times=[wintypes.FILETIME() for _ in range(4)]
    try:
        if not KERNEL.GetExitCodeProcess(handle,ctypes.byref(code)):return None,None
        if not KERNEL.GetProcessTimes(handle,*[ctypes.byref(v) for v in times]):return None,None
        return int(code.value),times[0].dwHighDateTime<<32|times[0].dwLowDateTime
    finally:
        KERNEL.CloseHandle(handle)


class Attached:
    def __init__(self,pid,created):
        self.pid,self.created=pid,created
    def poll(self):
        code,created=process_info(self.pid)
        if created!=self.created or code is None:return 1
        return None if code==259 else code


def plan(cfg):
    tasks=[]
    seed=20250903
    for scenario in ('perfect','iid','ar1'):
        for season in cfg['dates']:
            for objective in ('p2v_ramp','peak_ramp'):
                for method,h in [('MPC',v) for v in cfg['horizons_mpc']]+[('OL',96)]:
                    label='S1';variant=f'{method}_H{h:02d}'
                    name=f'{objective}_{season}_{scenario}_{variant}'
                    output=BASE/'results'/objective/variant/season/scenario/f'seed_{seed}'
                    tasks.append(dict(name=name,seed=seed,seed_label=label,season=season,
                        objective=objective,horizon=h,scenario=scenario,method=method,
                        status='pending',output=str(output),
                        retries=0,startup_deferrals=0,retry_after=0))
    assert len(tasks)==120
    return tasks


def external_workers():
    path=BASE.parent/'monte_carlo_3seeds/status/suite_status.json'
    if not path.exists():return 0
    state=json.loads(path.read_text())
    return sum(process_info(t['pid'])[0]==259 and
        process_info(t['pid'])[1]==t.get('process_created')
        for t in state['cases'] if t['status']=='running' and t.get('pid'))


def complete(task):
    folder=Path(task['output'])
    if task.get('review_mode') == 'certificate':
        path=folder/'certificate_summary.json'
        if not path.exists():return False
        summary=json.loads(path.read_text())
        return summary.get('audit_complete') is True and summary['windows_checked']==summary['total_windows']
    if not all((folder/name).exists() for name in ('metrics.json','checkpoint.json','official_timeseries.csv')):
        return False
    # next_step is at the start of these checkpoints; avoid parsing megabytes per status refresh.
    with (folder/'checkpoint.json').open(encoding='utf-8') as stream:
        header=stream.read(1024)
    if not re.search(r'"next_step"\s*:\s*304\s*[,}]',header):return False
    with (folder/'official_timeseries.csv').open(encoding='utf-8-sig',newline='') as stream:
        return sum(1 for _ in csv.DictReader(stream))==288


def finish(task,code):
    if complete(task):
        task.update(status='completed',exit_code=0,finished_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        return
    code=code or 1
    path=BASE/'logs'/f'{task["name"]}.log'
    empty=path.stat().st_size==task.get('launch_log_start_bytes',-1) if path.exists() else True
    task['exit_history']=(task.get('exit_history',[])+[code])[-30:]
    if (STATUS/'deadline_policy.json').exists() and not empty:
        task.update(status='failed',exit_code=code,retry_reason='deadline48_saved_for_diagnosis_no_identical_retry',
                    finished_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        return
    if empty:
        task['startup_deferrals']+=1
        task.update(status='pending',retry_after=time.time()+min(300,15*task['startup_deferrals']),
                    retry_reason='interpreter_startup')
    elif task['method']=='OL' and task.get('ol_time_limit_s',3600)<3600:
        task.update(status='pending',retry_after=time.time()+60,retries=2,
                    ol_time_limit_s=3600,retry_reason='user_escalation_to_3600')
    elif task['retries']<2:
        task['retries']+=1
        task.update(status='pending',retry_after=time.time()+60,retry_reason='case_error')
    else:
        task.update(status='failed',exit_code=code,retry_reason='case_error_requires_attention',
                    finished_at=time.strftime('%Y-%m-%d %H:%M:%S'))


def reap(active):
    for name,(process,stream,task) in list(active.items()):
        code=process.poll()
        if code is not None:
            if stream:stream.close()
            finish(task,code);del active[name]


def queue_reviews(tasks,failed_only=False):
    if (STATUS/'deadline_policy.json').exists():
        for task in list(tasks):
            if task.get('review_mode')!='completion' or task['status']!='completed' or task.get('certificate_scanned'):
                continue
            source=Path(task['output'])
            solves=json.loads((source/'solver_log.json').read_text())
            targets=[v for v in solves if v['origin']>=16 and v['gap']>.20]
            task['certificate_scanned']=True
            if not targets or any(t.get('review_mode')=='certificate' and t.get('review_source')==str(source) for t in tasks):
                continue
            review=dict(task)
            review.update(name=task.get('source_case',task['name'])+'_certificate',review=True,
                review_mode='certificate',review_source=str(source),review_failed_origin=None,
                output=str(BASE/'review/certificates'/task.get('source_case',task['name'])),
                status='pending',priority_gap_gt20=True,retries=0,startup_deferrals=2,retry_after=0)
            for key in ('pid','process_created','started_at','finished_at','first_started_timestamp','exit_code','exit_history'):
                review.pop(key,None)
            tasks.append(review)
        return
    if (STATUS/'COMPLETE_FIRST').exists():
        return
    from review_case import targets_for
    for task in list(tasks):
        if task.get('review') or task['horizon']==32 or task.get('review_scanned'):
            continue
        if task['status'] not in ('completed','failed'):
            continue
        if failed_only and task['status']!='failed':
            continue
        folder=Path(task['output'])
        log=folder/'solver_log.json'
        solves=json.loads(log.read_text()) if log.exists() else []
        failed_origin=None
        if task['status']=='failed' and (folder/'failure.json').exists():
            failed_origin=json.loads((folder/'failure.json').read_text())['origin']
        elif task['status']=='failed' and (folder/'progress.json').exists():
            failed_origin=json.loads((folder/'progress.json').read_text())['current_step']
        targets=targets_for(solves,failed_origin)
        task['review_scanned']=True
        if not targets and task['status']!='failed':continue
        review=dict(task)
        review.update(name=task['name']+'_review', review=True,
            review_source=task['output'], review_target_windows=len(targets),
            review_failed_origin=failed_origin,
            output=str(BASE/'review/results'/task['name']), status='pending',
            retries=0, startup_deferrals=2, retry_after=0)
        for key in ('pid','process_created','started_at','finished_at',
                    'first_started_timestamp','exit_code','exit_history'):
            review.pop(key,None)
        tasks.append(review)


def queue_quality_reviews(tasks):
    """After the completion round, review the latest trajectory of each root once."""
    policy_path=STATUS/'deadline_policy.json'
    if not policy_path.exists():return 0
    policy=json.loads(policy_path.read_text()).get('quality_followup',{})
    retries=[t for t in tasks if t.get('review_mode')=='budget_retry']
    if not policy.get('enabled') or not retries or any(t['status'] in ('pending','running') for t in retries):
        return 0
    from review_case import targets_for
    roots={t['name'] for t in tasks if not t.get('review') and t['horizon']!=32}
    latest={}
    ranks={None:0,'completion':1,'budget_retry':2}
    for task in tasks:
        mode=task.get('review_mode')
        if mode not in ranks or task['horizon']==32:continue
        if task['status']!='completed' and not (mode=='budget_retry' and task['status']=='failed'):continue
        root=task.get('source_case',task['name'])
        if root not in roots and root.endswith('_review'):root=root[:-7]
        if root not in roots:continue
        rank=(ranks[mode]+int(bool(task.get('review'))),task.get('finished_at',''))
        old=latest.get(root)
        if old is None or rank>(ranks[old.get('review_mode')]+int(bool(old.get('review'))),old.get('finished_at','')):
            latest[root]=task
    added=0
    for root,task in sorted(latest.items()):
        name=root+'_quality_review'
        if task.get('quality_followup_scanned') or any(t['name']==name for t in tasks):continue
        source=Path(task['output'])
        try:
            solves=json.loads((source/'solver_log.json').read_text())
            failed_origin=None
            if task['status']=='failed':
                failed=json.loads((source/'failure.json').read_text())
                failed_origin=failed['origin']
            targets=targets_for([v for v in solves if v['origin']>=16],failed_origin)
        except (OSError,ValueError,KeyError) as exc:
            task.update(quality_followup_scanned=True,quality_followup_error=str(exc));continue
        task['quality_followup_scanned']=True
        if not targets:continue
        review={key:task[key] for key in ('seed','seed_label','season','objective','horizon','method','scenario')}
        review.update(name=name,source_case=root,review=True,review_mode='quality_review',
            review_source=str(source),review_failed_origin=failed_origin,
            review_target_windows=len(targets),output=str(BASE/'review/quality'/root),
            status='pending',retries=0,startup_deferrals=2,retry_after=0,ol_time_limit_s=3600,
            priority_gap_gt20=failed_origin is not None or any(value==.10 for value in targets.values()),
            queued_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        tasks.append(review);added+=1
    return added


def queue_quality_retries(tasks):
    """Continue unfinished strict reviews from their saved state, round by round."""
    path=STATUS/'deadline_policy.json'
    if not path.exists():return 0
    policy=json.loads(path.read_text()).get('quality_followup',{})
    if not policy.get('enabled') or not policy.get('automatic_retry_unqualified'):return 0
    maximum=int(policy.get('maximum_automatic_rounds',0))
    queued=0
    for task in tasks:
        if task.get('review_mode')!='quality_review' or task['status']!='failed':continue
        current=int(task.get('automatic_round',0))
        if maximum and current>=maximum:continue
        folder=Path(task['output'])
        try:
            failure=json.loads((folder/'failure.json').read_text())
            if failure.get('reason') not in ('target gap not reached','no valid incumbent'):
                raise ValueError('Failure requires diagnosis before automatic solving')
            checkpoint=json.loads((folder/'checkpoint.json').read_text())
            if checkpoint['next_step']!=failure['origin']:
                raise ValueError('Failure origin does not match the resume checkpoint')
            review_plan=json.loads((folder/'review_plan.json').read_text())
        except (OSError,ValueError,KeyError) as exc:
            task['automatic_retry_blocked']=str(exc)
            continue
        with (folder/'review_rounds.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(automatic_round=current,
                finished_at=task.get('finished_at'),exit_code=task.get('exit_code'),
                failure=failure),allow_nan=False)+'\n')
        gaps=[float(v['gap']) for v in failure.get('attempts',[])
              if v.get('gap') is not None]
        gap=min(gaps) if gaps else None
        review_plan['automatic_round']=current+1
        atomic_json(folder/'review_plan.json',review_plan)
        task.update(status='pending',automatic_round=current+1,retries=0,
            startup_deferrals=max(2,task.get('startup_deferrals',0)),retry_after=0,
            priority_gap_gt20=gap is None or gap>.20,review_last_gap=gap,
            retry_reason='automatic_strict_quality_resume',ol_time_limit_s=3600,
            queued_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        for field in ('pid','process_created','started_at','finished_at','exit_code',
                      'first_started_timestamp','automatic_retry_blocked'):
            task.pop(field,None)
        queued+=1
    return queued


def dispatch_priority(task):
    if (STATUS/'deadline_policy.json').exists():
        from deadline_policy import task_priority
        if task.get('review_mode')=='quality_review':
            return (3,int(task.get('automatic_round',0)),
                    not task.get('priority_gap_gt20',False),
                    -float(task.get('review_last_gap') or 0))
        return (task_priority(task),False)
    if task.get('review'):
        return 0
    if task['method']=='OL':
        return 1
    return 2 if task['horizon']==24 else (4 if task['horizon']==32 else 3)


def eight_hour_blocked(cfg,ol_unfinished,h6_unfinished):
    return ol_unfinished or h6_unfinished or (
        (STATUS/'HOLD_8H').exists() and not cfg.get('fill_8h_after_reviews',False))


def publish(tasks,active,cfg,phase='running'):
    now=time.time();rows=[]
    counts={s:sum(t['status']==s for t in tasks) for s in ('completed','running','pending','failed','paused')}
    external=external_workers()
    original=[t for t in tasks if not t.get('review')]
    reviews=[t for t in tasks if t.get('review')]
    deadline=(STATUS/'deadline_policy.json').exists()
    execution=json.loads((STATUS/'deadline_policy.json').read_text()) if deadline else {}
    mpc_budget=' / '.join(map(str,execution.get('mpc_time_limits_s',cfg['time_limits_s'])))
    original_done=sum(t['status']=='completed' for t in original)
    lines=[f'updated: {time.strftime("%Y-%m-%d %H:%M:%S")}',
        'Horizon screening: MPC 2/4/6/8h + OL24h; two objectives; S1; 120 cases',
        f'original completed: {original_done}/120 | failed (including review): {counts["failed"]} | paused: {counts["paused"]} | own running: {len(active)} | pending: {counts["pending"]}',
        ('Deadline48: missing 6h trajectories first; >20% original-plan certificates next; MPC8h ON HOLD.'
         if deadline else
         'Priority: finish trajectories with valid incumbents; quality review deferred until completion.'
         if (STATUS/'COMPLETE_FIRST').exists() else
         'Priority: failed-case reviews now; finish MPC6h, then high-gap reviews; MPC8h fills remaining slots.'
         if cfg.get('fill_8h_after_reviews',False) else
         'Priority: finish MPC6h queue, then failed/high-gap continuous reviews; MPC8h ON HOLD.'),
        f'reviews: {sum(t["status"]=="completed" for t in reviews)}/{len(reviews)} completed; '+
        ('target 10%, acceptable 20%; audit completion does not imply quality passed' if deadline else
         'accept valid incumbent regardless of gap; gaps retained' if (STATUS/'COMPLETE_FIRST').exists()
         else 'targets >20% -> 10%, 10-20% -> 2%'),
        f'old batch running: {external} | combined running: {external+len(active)}/{cfg["parallel_cases"]}',
        (f'per case: {cfg["threads_per_case"]} Gurobi threads | MPC stage budgets {mpc_budget} s; accept valid incumbent after second stage; OL {execution["ol_window_s"]} s; same model and full KKT start'
         if deadline else f'per case: {cfg["threads_per_case"]} Gurobi threads | MPC stage budgets {mpc_budget} s | OL: 3600 s; fixed-price Follower LP/KKT start, MIPFocus=0'),
        'Progress includes 16 shared warmup points (304 total). process_min = current process age, excluding earlier startup retries.']
    for task in tasks:
        progress={};path=Path(task['output'])/'progress.json'
        if path.exists():
            try:progress=json.loads(path.read_text())
            except (ValueError,OSError):pass
        certificate=task.get('review_mode')=='certificate'
        total=progress.get('total',0) if certificate else 304
        step=progress.get('current_step',0) if certificate else (304 if task['status']=='completed' else progress.get('current_step',0))
        elapsed=(now-task['first_started_timestamp']) if 'first_started_timestamp' in task else 0.
        created=task.get('process_created')
        born=(created/10000000-11644473600) if created else None
        process_minutes=max(0.,(now-born)/60) if task['status']=='running' and born else None
        fresh_progress=path.exists() and born is not None and path.stat().st_mtime >= born
        display_status=('SOLVING' if fresh_progress else 'STARTING') if task['status']=='running' else task['status'].upper()
        rows.append(dict(name=task['name'],seed_label=task['seed_label'],seed=task['seed'],
            season=task['season'],scenario=task['scenario'],method=task['method'],
            status=task['status'],display_status=display_status,current_step=step,total_steps=total,
            first_attempt_age_minutes=round(elapsed/60,2),
            current_process_minutes=round(process_minutes,2) if process_minutes is not None else '',
            startup_retries=task['startup_deferrals'],pid=task.get('pid',''),
            retries=task['retries'],last_attempt_limit_s=progress.get('last_attempt_limit_s',''),
            best_gap=progress.get('best_gap',''),output=task['output']))
        rows[-1]['decision_time']=progress.get('decision_time','')
        if task['status']=='running':
            budget=f' | OL limit {task.get("ol_time_limit_s",3600)} s' if task['method']=='OL' else ''
            unit='certified windows' if certificate else 'step'
            lines.append(f'{display_status} {task["name"]} | {unit} {step}/{total} | process_min {process_minutes:.1f} | startup_retries {task["startup_deferrals"]}'+budget)
        elif task['status']=='failed':
            lines.append(f'FAILED {task["name"]} | checkpoint retained | see logs/{task["name"]}.log')
    if len(active)+external<cfg['parallel_cases'] and counts['pending']:
        lines.append('Free slots will be filled; startup retries may be cooling down.')
    atomic_json(STATUS/'suite_status.json',dict(counts=counts,cases=tasks))
    if deadline:
        brief=render_progress(tasks,rows,cfg,execution,phase)
        atomic_text(STATUS/'progress_brief.log','\n'.join(brief)+'\n')
        atomic_text(STATUS/'diagnostic_status.log','\n'.join(lines)+'\n')
    atomic_text(STATUS/'terminal_status.log','\n'.join(brief if deadline else lines)+'\n')
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_text(STATUS/'cases.csv',stream.getvalue())
    atomic_json(STATUS/'coordinator.json',dict(pid=os.getpid(),process_created=process_info(os.getpid())[1],
        status=phase,checked_timestamp=now,checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        live_processes=len(active),parallel_cases=cfg['parallel_cases'],counts=counts))
    return counts


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--retry-failed',action='store_true')
    args=parser.parse_args()
    cfg=json.loads((BASE/'configs/experiment.json').read_text())
    # Dispatch policy is not part of the numerical experiment manifest.
    cfg['fill_8h_after_reviews']=(STATUS/'FILL_8H_AFTER_REVIEWS').exists()
    assert 1 <= cfg['parallel_cases'] <= 30 and cfg['threads_per_case'] in (8,12,16)
    STATUS.mkdir(exist_ok=True);(BASE/'logs').mkdir(exist_ok=True)
    tasks=plan(cfg);previous={}
    if (STATUS/'suite_status.json').exists():
        previous={t['name']:t for t in json.loads((STATUS/'suite_status.json').read_text())['cases']}
    tasks.extend(dict(t) for t in previous.values() if t.get('review'))
    active={}
    for task in tasks:
        if task['name'] in previous:task.update(previous[task['name']])
        if complete(task):task['status']='completed'
        elif task['horizon']==32 and (STATUS/'HOLD_8H').exists() and not cfg.get('fill_8h_after_reviews',False) and task['status']!='running':
            task.update(status='paused',retry_reason='user_hold_8h_until_further_instruction')
        elif task['horizon']==32 and task['status']=='paused' and cfg.get('fill_8h_after_reviews',False):
            task.update(status='pending',retry_after=0,retry_reason='fill_free_slots_after_6h_and_reviews')
        elif task['status']=='running':
            process=Attached(task['pid'],task.get('process_created'))
            if process.poll() is None:active[task['name']]=(process,None,task)
            else:task.update(status='pending',retry_after=0,retry_reason='resume_after_restart')
        elif task['status']=='failed' and args.retry_failed and not (
                task['method']=='MPC' and task['horizon'] in (24,32)):
            task.update(status='pending',retries=0,retry_after=0)
    env=os.environ.copy()
    env.update({v:'1' for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    env['PYTHONPATH']=str(BASE.parent/'venv/Lib/site-packages')+os.pathsep+env.get('PYTHONPATH','')
    env['PYTHONIOENCODING']='utf-8'
    publish(tasks,active,cfg)
    while True:
        tick=time.monotonic()
        reap(active)
        if (STATUS/'STOP').exists():
            publish(tasks,active,cfg,'dispatch_stopped');break
        ol_unfinished=any(not t.get('review') and t['method']=='OL' and t['status'] in ('pending','running') for t in tasks)
        h6_unfinished=any(not t.get('review') and t['method']=='MPC' and t['horizon']==24 and t['status'] in ('pending','running') for t in tasks)
        queue_reviews(tasks,failed_only=True)
        queue_quality_reviews(tasks)
        queue_quality_retries(tasks)
        if not ol_unfinished and not h6_unfinished:
            queue_reviews(tasks)
        for task in sorted(tasks,key=dispatch_priority):
            if len(active)+external_workers()>=cfg['parallel_cases']:break
            if task['status']!='pending' or time.time()<task.get('retry_after',0):continue
            if task['horizon']==32 and eight_hour_blocked(cfg,ol_unfinished,h6_unfinished):continue
            if ol_unfinished and task['method']!='OL' and not task.get('review'):continue
            if not ol_unfinished and h6_unfinished and task['horizon']==32:continue
            if complete(task):task['status']='completed';continue
            script=BASE/'code/run_case.py'
            cmd=[sys.executable,'-u',str(script),'--input',str(BASE/'inputs'/f'{task["season"]}.csv'),
                '--season',task['season'],'--family','matched','--scenario',task['scenario'],
                '--method',task['method'],'--seed',str(task['seed']),
                '--horizon',str(task['horizon']),'--objective',task['objective'],'--execute']
            if task['method']=='OL':
                cmd+=['--ol-time-limit',str(task.get('ol_time_limit_s',3600))]
            if task.get('review'):
                cmd[2]=str(BASE/'code/certify_case.py' if task.get('review_mode')=='certificate' else BASE/'code/review_case.py')
                cmd+=['--review-source',task['review_source'],'--review-output',task['output']]
                if task.get('review_mode')=='quality_review':cmd+=['--quality-review']
                if task.get('review_failed_origin') is not None:
                    cmd+=['--failed-origin',str(task['review_failed_origin'])]
            if task['startup_deferrals']>=2:
                bootstrap=('import pathlib,runpy,site,sys;site.sethelper();'
                    'entry=sys.argv.pop(1);sys.path.insert(0,str(pathlib.Path(entry).parent));'
                    'sys.argv[0]=entry;runpy.run_path(entry,run_name="__main__")')
                cmd=[sys.executable,'-S','-u','-c',bootstrap]+cmd[2:]
            stream=(BASE/'logs'/f'{task["name"]}.log').open('a',encoding='utf-8')
            task['launch_log_start_bytes']=stream.tell()
            try:
                startup=subprocess.STARTUPINFO()
                startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startup.wShowWindow = 0
                process=subprocess.Popen(cmd,cwd=BASE.parent,stdout=stream,stderr=subprocess.STDOUT,
                    env=env,stdin=subprocess.DEVNULL,startupinfo=startup,
                    creationflags=subprocess.CREATE_NO_WINDOW,close_fds=True)
            except OSError as exc:
                stream.close();task.update(retry_after=time.time()+60,retry_reason=str(exc));continue
            task.setdefault('first_started_timestamp',time.time())
            task.update(status='running',pid=process.pid,process_created=process_info(process.pid)[1],
                        started_at=time.strftime('%Y-%m-%d %H:%M:%S'))
            active[task['name']]=(process,stream,task)
            publish(tasks,active,cfg)
            time.sleep(2);reap(active)
        counts=publish(tasks,active,cfg)
        if not active and counts['pending']==0:
            publish(tasks,active,cfg,'completed' if counts['failed']==0 else 'completed_with_failures')
            break
        time.sleep(max(1.,10-(time.monotonic()-tick)))


if __name__=='__main__':
    STATUS.mkdir(parents=True,exist_ok=True)
    sys.stdout=(STATUS/'coordinator_stdout.log').open('a',encoding='utf-8',buffering=1)
    sys.stderr=(STATUS/'coordinator_stderr.log').open('a',encoding='utf-8',buffering=1)
    with exclusive_lock(STATUS/'coordinator.lock'):
        main()
