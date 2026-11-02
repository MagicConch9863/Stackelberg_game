"""Explicit recovery of interrupted suites; preserves damaged files before retry."""
import csv
from datetime import datetime
import json
from pathlib import Path
from io_utils import atomic_json, atomic_text, exclusive_lock

BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parent


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, UnicodeError):
        return None


def main():
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report = dict(time=stamp, archived=[], restarted=[], resumed=[])
    for root in (BASE, PROJECT/'four_hour_peak_ramp'):
        files = list((root/'results').rglob('*'))+list((root/'status').glob('*'))
        for path in files:
            if not path.is_file() or path.suffix not in ('.json','.csv','.log'):
                continue
            raw = path.read_bytes()
            bad = b'\0' in raw or (path.suffix=='.json' and read_json(path) is None)
            if not bad:
                continue
            relative = path.resolve().relative_to(root.resolve())
            target = root/'recovery'/stamp/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            path.replace(target)
            report['archived'].append(str(target.relative_to(PROJECT)))
    saved_path = BASE/'status/suite_status.json'
    saved = read_json(saved_path)
    if saved is None:
        raise RuntimeError('Task inventory missing; preserved results require manual reconstruction.')
    for task in saved['cases']:
        out = Path(task['output'])
        checkpoint = read_json(out/'checkpoint.json')
        if checkpoint is None:
            backup = read_json(out/'checkpoint.previous.json')
            if backup is not None:
                atomic_json(out/'checkpoint.json',backup)
                checkpoint = backup
        official = out/'official_timeseries.csv'
        complete = bool(read_json(out/'metrics.json')) and official.exists()
        if complete:
            with official.open() as stream:
                complete = len(list(csv.DictReader(stream)))==288
        if complete:
            task.update(status='completed',current_step=304)
            continue
        step = checkpoint['next_step'] if checkpoint else 0
        task.update(status='pending',current_step=step,retries=0,retry_after=0)
        task.pop('pid',None)
        task.pop('exit_code',None)
        task['recovered_at'] = stamp
        label = 'resumed' if checkpoint else 'restarted'
        report[label].append(dict(case=task['name'],step=step))
        if out.exists():
            atomic_json(out/'progress.json',dict(status='pending',current_step=step,total=304))
    saved['counts'] = {s:sum(t['status']==s for t in saved['cases']) for s in ('completed','running','pending','failed')}
    atomic_json(saved_path,saved)
    atomic_json(BASE/'status'/f'recovery_{stamp}.json',report)
    atomic_text(BASE/'status/terminal_status.log',f'Recovered {stamp}\ncounts: {saved["counts"]}\nStarting coordinator next.\n')
    print('Recovered queue:',saved['counts'])
    print('Checkpoint resumes:',len(report['resumed']),'unstarted/restarted:',len(report['restarted']))
    print('Damaged files preserved:',len(report['archived']))


if __name__=='__main__':
    with exclusive_lock(BASE/'status/coordinator.lock'):
        main()
