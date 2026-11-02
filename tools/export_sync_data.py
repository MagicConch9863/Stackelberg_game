"""Export compact, read-only plotting data; never launches or alters simulations."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'sync_data'


def read_json(path, default=None):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else default


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace('\\','/')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    if not fields:fields=['case_id']
    target = OUT/name
    temp = target.with_suffix(target.suffix+'.tmp')
    with temp.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)
    temp.replace(target)


def main():
    OUT.mkdir(exist_ok=True)
    cases=[];trajectories=[];realizations=[];windows=[];daily=[];inputs=[];phase4=[]
    manifests={};configs={};warnings=[];series={}
    selections=read_csv(ROOT/'four_hour_test/results/figure/current_review/data_sources.csv')
    for item in selections:
        if item['dataset']!='original':continue
        cases.append(dict(case_id='four_hour_'+ '_'.join(item[k] for k in
            ('objective','season','scenario','seed','method')),dataset='four_hour_selected',
            objective=item['objective'],season=item['season'],scenario=item['scenario'],
            seed=item['seed'],method=item['method'],horizon_steps=16,status='completed',
            source=item['source'],selection=item['selection'],
            replaced_source=item.get('replaced_source',''),prior_plot_sha256=item['sha256']))
    for batch in ('monte_carlo_3seeds','horizon_screening'):
        state=read_json(ROOT/batch/'status/suite_status.json',{})
        for item in state.get('cases',[]):
            cases.append(dict(case_id=batch+'_'+item['name'],dataset=batch,
                objective=item.get('objective','p2v_ramp'),season=item['season'],
                scenario=item['scenario'],seed=item['seed'],method=item['method'],
                horizon_steps=item.get('horizon',16),status=item['status'],
                source=item['output'],selection='batch_result'))
    for case in cases:
        folder=Path(case.pop('source'))
        case['source_relative']=relative(folder)
        if case.get('replaced_source'):case['replaced_source']=relative(case['replaced_source'])
        manifest=read_json(folder/'manifest.json',{})
        if manifest:manifests[case['case_id']]=manifest
        meta={k:case[k] for k in ('case_id','dataset','objective','season','scenario','seed','method','horizon_steps')}
        log=read_json(folder/'solver_log.json',[])
        for entry in log:
            attempts=entry.get('attempts',[])
            windows.append(dict(meta,origin=entry['origin'],gap=entry.get('gap'),
                solver_gap=entry.get('solver_gap'),status=entry.get('status'),
                objective_value=entry.get('objective'),best_bound=entry.get('best_bound'),
                quality=entry.get('quality',''),attempt_count=len(attempts),
                recorded_attempt_seconds=sum(a.get('elapsed_s',0) for a in attempts)))
        case['saved_windows']=len(log)
        case['high_gap_windows_gt_20pct']=sum(e.get('gap',0)>.2 for e in log)
        checkpoint=folder/'checkpoint.json'
        case['checkpoint_present']=checkpoint.exists()
        case['checkpoint_in_git']=False
        failure=read_json(folder/'failure.json')
        if failure:case['last_failure_reason']=failure.get('reason','')
        source=folder/'official_timeseries.csv'
        case['trajectory_exported']=False
        if case['status']!='completed' or not source.exists():continue
        rows=read_csv(source)
        if len(rows)!=288:
            warnings.append(case['case_id']+': not 288 official rows; omitted');continue
        for row in rows:
            y=float(row['realized_net_kw']);x=float(row['uncontrolled_net_kw']);u=float(row['bess_kw'])
            if not all(math.isfinite(v) for v in (y,x,u)) or abs(y-x-u)>3e-5:
                raise ValueError('Invalid physical balance: '+str(source))
        case['trajectory_exported']=True
        case['timeseries_sha256']=digest(source)
        case['prior_plot_source_changed']=bool(case.get('prior_plot_sha256') and
            case['prior_plot_sha256']!=case['timeseries_sha256'])
        trajectories.extend(dict(meta,**row) for row in rows)
        series[case['case_id']]=(case,rows)
        actual=folder/'realization.csv'
        if actual.exists():
            case['realization_sha256']=digest(actual)
            realizations.extend(dict(meta,**row) for row in read_csv(actual))
        for day in range(3):
            chunk=rows[day*96:(day+1)*96]
            y=[float(r['realized_net_kw']) for r in chunk]
            x=[float(r['uncontrolled_net_kw']) for r in chunk]
            peak=max(y);rawpeak=max(x);p2v=max(y)-min(y);rawp2v=max(x)-min(x)
            changes=[y[j]-y[j-1] for j in range(1,96)]
            if day>0:changes.insert(0,y[0]-float(rows[day*96-1]['realized_net_kw']))
            daily.append(dict(meta,day=day+1,date=chunk[0]['timestamp'][:10],peak_kw=peak,
                valley_kw=min(y),p2v_kw=p2v,uncontrolled_peak_kw=rawpeak,
                uncontrolled_p2v_kw=rawp2v,
                peak_change_pct=(peak/rawpeak-1)*100 if rawpeak>1e-8 else '',
                p2v_change_pct=(p2v/rawp2v-1)*100 if rawp2v>1e-8 else '',
                share_above_70pct_raw_peak_pct=sum(v>=.7*rawpeak for v in y)/96*100 if rawpeak>0 else '',
                raw_share_above_70pct_raw_peak_pct=sum(v>=.7*rawpeak for v in x)/96*100 if rawpeak>0 else '',
                own_95pct_peak_share_pct=sum(v>=.95*peak for v in y)/96*100 if peak>0 else '',
                max_15min_change_kw=max(abs(v) for v in changes),
                volatility_population_std_kw=statistics.pstdev(changes),
                ramp_boundary_note='includes preceding formal day' if day else 'first formal difference excluded; no warmup point in this compact export'))
    for batch in ('four_hour_test','four_hour_peak_ramp','monte_carlo_3seeds','horizon_screening'):
        folder=ROOT/batch/'configs'
        if folder.exists():
            for path in folder.glob('*.json'):configs[relative(path)]=read_json(path)
        folder=ROOT/batch/'inputs'
        if folder.exists():
            for path in folder.rglob('*.json'):configs[relative(path)]=read_json(path)
            for path in folder.rglob('*.csv'):
                inputs.extend(dict(input_file=relative(path),**row) for row in read_csv(path))
    phase=ROOT/'stackelberg_scenario_analysis/phase4/results'
    for path in phase.glob('*.csv'):
        phase4.extend(dict(source_file=relative(path),**row) for row in read_csv(path))
    if (phase/'phase4_settings.json').exists():configs[relative(phase/'phase4_settings.json')]=read_json(phase/'phase4_settings.json')
    for row in daily:
        peers=[d for d in daily if all(d[k]==row[k] for k in
            ('dataset','objective','season','scenario','seed','horizon_steps','day')) and d['method']!=row['method']]
        if len(peers)==1:
            threshold=.95*min(row['peak_kw'],peers[0]['peak_kw'])
            chunk=series[row['case_id']][1][(row['day']-1)*96:row['day']*96]
            row['paired_common_95pct_threshold_kw']=threshold
            row['paired_common_95pct_share_pct']=sum(float(r['realized_net_kw'])>=threshold for r in chunk)/96*100 if threshold>0 else ''
    exports={'cases.csv':cases,'timeseries.csv':trajectories,'realizations.csv':realizations,
        'solver_windows.csv':windows,'daily_kpi.csv':daily,'nominal_inputs.csv':inputs,
        's1_s4_statistics.csv':phase4}
    for name,rows in exports.items():write_csv(name,rows)
    hashes={relative(p):digest(p) for p in ROOT.rglob('requirements.txt') if 'venv' not in p.parts and '.git' not in p.parts}
    metadata=dict(exported_at=datetime.now().astimezone().isoformat(),
        snapshot_only=True,complete_trajectories=len(series),rows={k:len(v) for k,v in exports.items()},
        source_selection='four_hour_selected uses current_review/data_sources.csv dataset=original; may contain explicitly selected complete causal recomputations; no hand smoothing',
        warnings=warnings,configs=configs,case_manifests=manifests,requirements_sha256=hashes,
        exclusions=['images','raw rolling forecast paths','checkpoints','logs','machine PIDs'],
        first_day_ramp_note='Compact daily KPI excludes first formal difference; original 288-point data retained unchanged.',
        s1_s4_note='Only existing aggregate phase4 CSV tables are included, not unavailable individual-day raw trajectories.')
    target=OUT/'snapshot.json';temp=target.with_suffix('.json.tmp')
    temp.write_text(json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(target)
    print(json.dumps(dict(cases=len(cases),complete_trajectories=len(series),warnings=warnings,
        output_bytes=sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
