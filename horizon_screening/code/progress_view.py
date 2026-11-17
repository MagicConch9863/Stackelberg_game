"""Readable live progress, separate from machine state and historical failures."""
from collections import Counter
import time


def render_progress(tasks, rows, cfg, execution, phase='running'):
    retries=[t for t in tasks if t.get('review_mode')=='budget_retry']
    audits=[t for t in tasks if t.get('review_mode')=='certificate']
    quality=[t for t in tasks if t.get('review_mode')=='quality_review']
    running=[r for r in rows if r['status']=='running']
    counts=Counter(t['status'] for t in retries)
    completed={t.get('source_case',t['name']) for t in tasks
               if t['status']=='completed' and (not t.get('review') or t.get('review_mode')=='completion')
               and t.get('horizon')!=32}
    budgets='/'.join(map(str,execution.get('mpc_time_limits_s',cfg['time_limits_s'])))
    result=['Aktualisiert: '+time.strftime('%Y-%m-%d %H:%M:%S'),
        'Aktuell: Kontinuierliche Neuberechnung von Fällen mit hohem MIP-Gap.',
        'Parameter dieser Runde: AR(1) rho=0.9; Seed S1 (20250903).',
        f'Aktive Fälle: {len(running)}/{cfg["parallel_cases"]}',
        f'Neuberechnungen ({len(retries)} Fälle): fertig {counts["completed"]} | aktiv {counts["running"]} | '
        f'wartend {counts["pending"]} | fehlgeschlagen {counts["failed"]}',
        f'MPC-Budget: {budgets} s; ab Stufe 2 genügt eine gültige zulässige Lösung.',
        f'OL: {execution.get("ol_window_s",3600)} s je Fenster; Neuberechnung mit bis zu {cfg["threads_per_case"]} Gurobi-Threads je Fall.',
        f'Vollständige Ausgangstrajektorien: {len(completed)}/96 (2/4/6h MPC und 24h OL); 8h MPC pausiert.',
        f'Frühere Schrankenprüfung: {sum(t["status"]=="completed" for t in audits)}/{len(audits)} Fälle geprüft; '
        f'{sum(t["status"]=="running" for t in audits)} weiterhin aktiv.',
        '', 'LAUFENDE FÄLLE']
    if execution.get('quality_followup',{}).get('enabled'):
        quality_policy=execution['quality_followup']
        quality_budgets='/'.join(map(str,quality_policy.get('mpc_time_limits_s',[])))
        quality_ol=quality_policy.get('ol_time_limit_s',execution.get('ol_window_s',3600))
        qc=Counter(t['status'] for t in quality)
        errors=sum(bool(t.get('quality_followup_error')) for t in tasks)
        result[9:9]=[
            'Automatischer Nachlauf: Qualitätsprüfung nach Abschluss der Neuberechnungsrunde; neueste 2/4/6h- und OL-Ergebnisse.',
            f'Qualitätsprüfungen: fertig {qc["completed"]} | aktiv {qc["running"]} | wartend {qc["pending"]} | nicht abgeschlossen {qc["failed"]}; Auswahlfehler {errors}.',
            f'Strenge Ziele: Gap >20% auf <=10%; Gap >10% bis 20% auf <=2%. MPC {quality_budgets} s; OL {quality_ol} s.']
        if quality_policy.get('automatic_retry_unqualified'):
            result[1]='Aktuell: Strenge Qualitätsprüfung mit automatischer Wiederholung nicht bestandener Fälle.'
            result[5]=f'Aktuelles MPC-Prüfbudget: {quality_budgets} s; gültige Lösung und Ziel-Gap erforderlich.'
            result.insert(12,'Nicht bestandene Qualitätsprüfungen werden automatisch erneut eingereiht; Checkpoint und MIP-Start bleiben erhalten.')
    if phase not in ('running','completed','completed_with_failures'):
        result.insert(1,'Scheduler wird aktualisiert; bereits gestartete Solver laufen weiter.')
    by_name={t['name']:t for t in tasks}
    seasons={'winter':'Winter','spring':'Frühling','summer':'Sommer','autumn':'Herbst'}
    season_order={'winter':0,'spring':1,'summer':2,'autumn':3}
    scenario_order={'perfect':0,'iid':1,'ar1':2}
    ordered=sorted(running,key=lambda r:(by_name[r['name']].get('review_mode')!='budget_retry',
        r.get('method',''),by_name[r['name']].get('objective',''),season_order.get(r['season'],9),
        scenario_order.get(r['scenario'],9)))
    for index,row in enumerate(ordered,1):
        task=by_name[row['name']]
        kind=('Schrankenprüfung' if task.get('review_mode')=='certificate' else
              'Qualitätsprüfung' if task.get('review_mode')=='quality_review' else 'Neuberechnung')
        noise={'perfect':'Perfect','iid':'IID','ar1':'AR(1), rho=0.9'}.get(row['scenario'],row['scenario'])
        objective={'p2v_ramp':'P2V+Ramp','peak_ramp':'Peak+Ramp'}.get(task.get('objective'),'?')
        horizon=f'{task["horizon"]*.25:g}h'
        result.append(f'{index:02d} {seasons.get(row["season"],row["season"])} | {noise} | {objective} | '
                      f'{row["method"]} {horizon} | {row["seed_label"]} | {kind}')
        if task.get('review_mode')=='quality_review':
            result[-1]+=f' | automatische Runde {task.get("automatic_round",0)}'
        step=row['current_step']
        if kind=='Schrankenprüfung':
            detail=f'Geprüfte Fenster: {step}/{row["total_steps"]}'
        else:
            detail=f'Schritt {step}/304; Auswertungszeitraum: {max(0,min(288,step-16))}/288 Schritte abgeschlossen'
        if row.get('decision_time'):
            detail+='; Zeitpunkt '+row['decision_time']
        if row.get('display_status')=='STARTING':
            detail+='; Initialisierung läuft'
        result.append('   '+detail)
    if not running:
        result.append('Derzeit keine aktiven Solver.')
    pending=Counter((t['method'],t['horizon']) for t in retries+quality if t['status']=='pending')
    result+=['','WARTESCHLANGE (automatischer Start bei freiem Platz)']
    result.extend(f'{method} {h*.25:g}h: {count} Fälle'
                  for (method,h),count in sorted(pending.items()))
    if not pending:
        result.append('Keine weiteren Fälle in dieser Runde.')
    failed=[t for t in retries+quality if t['status']=='failed']
    if failed:
        result+=['','FEHLGESCHLAGENE FÄLLE DIESER RUNDE']
        result.extend(t['name'] for t in failed)
    result+=['','304 Schritte = 16 Anlaufschritte + 288 Auswertungsschritte; Prüfungen zählen Fenster.',
             'Solver-Logs: horizon_screening/logs/; Ergebnisse: horizon_screening/review/budget_retry/; Qualitätsnachlauf: horizon_screening/review/quality/']
    return result
