"""Export pending strict reviews and observed new high-gap cases, without Git data."""
from pathlib import Path
from datetime import datetime
import ast
import csv
import hashlib
import io
import json
import math
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
HS=ROOT/'horizon_screening'


def read(path,default=None):
    path=Path(path)
    if not path.exists():return default
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_text(rows):
    output=io.StringIO(newline='')
    fields=list(dict.fromkeys(k for row in rows for k in row))
    writer=csv.DictWriter(output,fieldnames=fields,lineterminator='\n')
    writer.writeheader();writer.writerows(rows)
    return output.getvalue()


def targets_for(solves):
    return {str(s['origin']):(.1 if s.get('gap') is None or not math.isfinite(s['gap']) or s['gap']>.2 else .02)
        for s in solves if s['origin']>=16 and (s.get('gap') is None or not math.isfinite(s['gap']) or s['gap']>.1)}


def trim(state,origin,horizon):
    rows=[r for r in state['rows'] if r['execution_step']<origin]
    assert rows and rows[-1]['execution_step']==origin-1
    last=rows[-1]
    prices=sorted((r for r in state['forecasts'] if r['origin']==origin and r.get('bus')==1),key=lambda r:r['lead'])
    warm=[float(r['planned_pb']) for r in prices] if len(prices)==horizon else [last['pb_eur_per_kwh']]*horizon
    return dict(state,next_step=origin,rows=rows,
        energy={bus:float(last[f'soc_bus_{i}_kwh']) for i,bus in enumerate(state['energy'],1)},
        previous_pb=float(last['pb_eur_per_kwh']),previous_grid=float(last['realized_net_kw']),warm_pb=warm,
        solves=[s for s in state['solves'] if s['origin']<origin],
        forecasts=[r for r in state['forecasts'] if r['origin']<origin])


def main():
    out=ROOT/'review_transfer'
    assert not out.exists()
    out.mkdir()
    runtime=out/'horizon_screening'
    names=['run_case.py','review_case.py','deadline_solver.py','deadline_policy.py','ol_feasibility.py',
        'ol_warm_start.py','solve_policy.py','io_utils.py','prediction.py','peak_window.py','run_suite.py','progress_view.py']
    for name in names:
        dest=runtime/'code'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(HS/'code'/name,dest)
    for source in (HS/'source').rglob('*.py'):
        if '__pycache__' in source.parts:continue
        dest=runtime/'source'/source.relative_to(HS/'source');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    for folder in ('configs','inputs'):
        shutil.copytree(HS/folder,runtime/folder)
    write(runtime/'status/deadline_policy.json',read(HS/'status/deadline_policy.json'))
    write(runtime/'status/suite_status.json',dict(cases=[]))
    code=out/'study_queue/code';code.mkdir(parents=True)
    shutil.copy2(ROOT/'study_queue/code/run_queue.py',code/'run_queue.py')
    (out/'study_queue/status').mkdir()
    shutil.copy2(ROOT/'tools/run_review_transfer.py',out/'run_review.py')
    versions={}
    for name in ('numpy','pandas','matplotlib','scipy','gurobipy','simbench'):
        records=list((ROOT/'venv/Lib/site-packages').glob(f'{name}-*.dist-info/METADATA'))
        assert len(records)==1, (name,records)
        with records[0].open(encoding='utf-8') as stream:
            for line in stream:
                if line.startswith('Version: '):versions[name]=line.split(': ',1)[1].strip();break
    (out/'requirements.txt').write_text(''.join(f'{name}=={version}\n' for name,version in versions.items()),encoding='utf-8')
    inventory=[];windows=[];tasks=[];provenance=[];omitted=[]
    print('Runtime snapshot copied; collecting review checkpoints.',flush=True)
    suite=read(ROOT/'study_queue/status/suite_status.json')
    for task in suite['cases']:
        if task['stage']==4 and task.get('quality_review') and task['status']!='completed':
            kind='existing_strict_review'
        elif task['stage'] in (1,2,3):
            kind='new_high_gap_snapshot'
        else:continue
        source=Path(task['output'])
        state=read(source/'checkpoint.json')
        if state is None:continue
        manifest=read(source/'manifest.json')
        if manifest is None:
            # Recovery snapshots have not yet acquired the new execution manifest.
            if task.get('recovery_source'):
                archive=out/'incomplete'/task['name'];archive.mkdir(parents=True)
                for filename in ('checkpoint.json','recovery_source.json'):
                    if (source/filename).exists():shutil.copy2(source/filename,archive/filename)
                shutil.copy2(Path(task['recovery_source'])/'manifest.json',archive/'original_manifest.json')
                for filename in ('realization.csv','failure.json','process.json'):
                    path=Path(task['recovery_source'])/filename
                    if path.exists():shutil.copy2(path,archive/filename)
                omitted.append(dict(case=task['name'],step=state['next_step'],reason='Finish initial recovery before strict review'))
            continue
        if kind=='existing_strict_review':
            plan=read(source/'review_plan.json')
            assert plan and plan.get('quality_review')
            targets=dict(plan['targets']);resume=state
            reference=Path(task['review_source'])
        else:
            targets=targets_for(state['solves'])
            if not targets:continue
            plan=dict(targets=targets,replay_from=min(map(int,targets)),quality_review=True,automatic_round=0,
                continuous_replay=True,policy='gap_above_20_to_10;gap_10_to_20_to_2')
            resume=trim(state,plan['replay_from'],task['horizon'])
            reference=source
        if task['horizon']==32:continue
        case=task['name']
        if kind=='new_high_gap_snapshot':case+='_quality_snapshot'
        dest=out/'cases'/case
        ref=dest/'reference';ref.mkdir(parents=True)
        write(dest/'resume/checkpoint.json',resume)
        write(dest/'resume/manifest.json',manifest)
        write(dest/'resume/review_plan.json',plan)
        write(ref/'manifest.json',read(reference/'manifest.json',manifest))
        refsolves=read(reference/'solver_log.json',state['solves'])
        write(ref/'solver_log.json',refsolves)
        for filename in ('official_timeseries.csv','timeseries.csv','realization.csv','process.json'):
            path=reference/filename
            if path.exists():shutil.copy2(path,ref/filename)
        candidate=read(source/'latest_candidate.json')
        matching=bool(candidate and candidate['origin']==resume['next_step'] and
            len(candidate.get('pb',[]))==task['horizon'] and set(candidate['energies'])==set(resume['energy']) and
            all(abs(candidate['energies'][b]-v)<1e-8 for b,v in resume['energy'].items()))
        if matching:
            write(dest/'resume/latest_candidate.json',candidate)
            if (source/'latest_incumbent.mst').exists():shutil.copy2(source/'latest_incumbent.mst',dest/'resume/latest_incumbent.mst')
        warm_path=runtime/'inputs/transfer_warmup'/f'{case}.csv'
        warm_path.parent.mkdir(exist_ok=True)
        if task.get('shared_warmup'):
            original_warm=Path(task['shared_warmup'])
        else:
            original_warm=HS/'inputs'/manifest['shared_warmup']['file']
        assert original_warm.exists()
        shutil.copy2(original_warm,warm_path)
        for solve in refsolves:
            if str(solve['origin']) in targets:
                windows.append(dict(case=case,origin=solve['origin'],original_gap=solve.get('gap'),
                    target_gap=targets[str(solve['origin'])],status=solve.get('status'),
                    objective=solve.get('objective'),best_bound=solve.get('best_bound')))
        gaps=[s['gap'] for s in refsolves if str(s['origin']) in targets and s.get('gap') is not None and math.isfinite(s['gap'])]
        max_gap=max(gaps,default=1.)
        row=dict(case=case,kind=kind,method=task['method'],horizon_hours=task['horizon']/4,
            objective=task['objective'],season=task['season'],scenario=task['scenario'],seed=task['seed'],
            rho=manifest['config']['rho'],source_complete=state['next_step']==304 if kind=='new_high_gap_snapshot' else True,
            source_saved_step=state['next_step'],resume_step=resume['next_step'],total_steps=304,
            remaining_windows=math.ceil((304-resume['next_step'])/(96 if task['method']=='OL' else 1)),
            target_windows=len(targets),max_original_gap_pct=round(100*max_gap,3),
            target_10pct=sum(v==.1 for v in targets.values()),
            target_2pct=sum(v==.02 for v in targets.values()),matching_MIP_start=matching,
            source=str(source.relative_to(ROOT)),bundle_case=f'cases/{case}')
        inventory.append(row)
        tasks.append(dict(name=case,stage=4,season=task['season'],scenario=task['scenario'],seed=task['seed'],
            objective=task['objective'],method=task['method'],horizon=task['horizon'],rho=manifest['config']['rho'],
            output=f'cases/{case}/resume',review_source=f'cases/{case}/reference',
            shared_warmup=f'horizon_screening/inputs/transfer_warmup/{case}.csv',
            quality_review=True,status='pending',attempt=0,retry_after=0,bundle_kind=kind,
            review_last_gap=max_gap,review_target_windows=len(targets)))
        provenance.append(dict(case=case,source=str(source.relative_to(ROOT)),
            checkpoint_sha256=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest(),
            manifest_sha256=digest(source/'manifest.json'),partial_reference=kind=='new_high_gap_snapshot' and state['next_step']<304))
    existing=sum(r['kind']=='existing_strict_review' for r in inventory)
    assert existing==sum(t['stage']==4 and t.get('quality_review') and t['status']!='completed' for t in suite['cases'])
    catalog=dict(parallel_cases=4,threads_per_case=8,refill_seconds=10,
        two_hour_seconds=7200,hold_8h=True,stages={'4':'Kontinuierliche strenge Nachpruefung'},cases=tasks)
    write(out/'study_queue/configs/plan.json',catalog)
    write(out/'catalog.json',catalog)
    for name,data in [('review_cases.csv',inventory),('review_windows.csv',windows)]:
        if name=='review_cases.csv':data=sorted(data,key=lambda r:(r['kind']!='existing_strict_review',-r['max_original_gap_pct'],r['case']))
        (out/name).write_text(csv_text(data),encoding='utf-8')
    write(out/'provenance.json',dict(exported_at=datetime.now().astimezone().isoformat(),source_root=str(ROOT),
        git_branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).strip(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).strip(),
        dirty_code_snapshot=True,existing_reviews=existing,new_high_gap_snapshots=len(inventory)-existing,
        incomplete_recoveries=omitted,cases=provenance,formulation_unchanged=True))
    (out/'Anleitung.txt').write_text(
        'PORTABLE NACHPRUEFUNG / WINDOWS\n\n'
        '1. Das ZIP separat zum Git-Repository uebertragen und in einen neuen Ordner entpacken.\n'
        '   main kann das uebergeordnete Arbeitsverzeichnis sein. Der Branch beschleunigt den Solver nicht.\n'
        '   Das Paket verwendet seinen eingefrorenen Solvercode und liest keine aelteren main-Konfigurationen.\n'
        '2. Python 3.11 und eine gueltige Gurobi-Lizenz benoetigen. Keine Lizenzdatei ist enthalten.\n'
        '   Im gewaehlten Python-Umfeld: python -m pip install -r requirements.txt\n'
        '3. Nur pruefen, keine Optimierung: python run_review.py --check\n'
        f'4. Bestehende {existing} Reviews starten: python run_review.py --run --parallel 4 --threads 8\n'
        '   Bei ausreichender Hardware koennen --parallel 14 --threads 16 verwendet werden.\n'
        '   Neue, teils unvollstaendige Hoch-gap-Schnappschuesse zusaetzlich: --include-new\n'
        '5. Fortschritt: study_queue/status/progress_brief.log; detailliert: study_queue/logs/.\n'
        '   Nach Neustart denselben Startbefehl erneut ausfuehren; Checkpoints werden fortgesetzt.\n'
        '   Dieses Paket installiert keinen systemweiten Autostart.\n\n'
        'Review-Regel: urspruenglicher gap >20% -> Ziel 10%; gap >10% bis20% -> Ziel 2%.\n'
        'MPC-Stufen: 120/600/3600 s; OL: 3600 s. Nicht qualifizierte Fenster werden automatisch erneut versucht.\n'
        'Nach einer geaenderten Aktion wird die gesamte verbleibende Trajektorie weitergerechnet.\n'
        '304 Ausfuehrungsschritte enthalten 16 Anlaufschritte; ausgewertet werden 288 Schritte.\n'
        'Neue Teil-Schnappschuesse sind in review_cases.csv markiert, keine vollstaendigen Vergleichsergebnisse.\n'
        'Zwei unvollstaendige Monte-Carlo-Wiederherstellungen sind nur unter incomplete/ gesichert.\n'
        'Originale auf TSF und Ausgaben auf dem zweiten Rechner bleiben getrennt.\n'
        'Zurueck uebertragen: cases/*/resume/ und study_queue/status/suite_status.json.\n\n'
        'JSON, CSV, Checkpoints und MIP-Starts sind im ZIP erforderlich, werden aber nicht in Git hochgeladen.\n',encoding='utf-8')
    for source in out.rglob('*.py'):ast.parse(source.read_text(encoding='utf-8-sig'))
    file_rows=[dict(file=str(p.relative_to(out)).replace('\\','/'),bytes=p.stat().st_size,sha256=digest(p))
        for p in out.rglob('*') if p.is_file()]
    (out/'files.csv').write_text(csv_text(file_rows),encoding='utf-8')
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in out.rglob('*'):
            if path.is_file():z.write(path,path.relative_to(out))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps(dict(folder=str(out),zip=str(archive),zip_mb=round(archive.stat().st_size/1024**2,2),
        files=len(file_rows)+1,existing_reviews=existing,new_high_gap_snapshots=len(inventory)-existing,
        incomplete_recoveries=len(omitted)),indent=2),flush=True)


if __name__=='__main__':main()
