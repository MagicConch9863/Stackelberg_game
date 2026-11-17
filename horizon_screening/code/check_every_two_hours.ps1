$ErrorActionPreference='Stop'
$priorityRoot = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'study_queue'
if (Test-Path -LiteralPath (Join-Path $priorityRoot 'status/ACTIVE')) {
    & (Join-Path $priorityRoot 'code/check_every_two_hours.ps1')
    exit 0
}
$base=Split-Path -Parent $PSScriptRoot
$status=Join-Path $base 'status'
if(Test-Path -LiteralPath (Join-Path $status 'STOP')){exit 0}
$statePath=Join-Path $status 'coordinator.json'
$coordinator=Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
$process=Get-Process -Id $coordinator.pid -ErrorAction SilentlyContinue
$alive=$process -and ($process.StartTime.ToFileTimeUtc() -eq $coordinator.process_created)
if(!$alive -and $coordinator.status -notin @('completed','completed_with_failures')) {
    & (Join-Path $PSScriptRoot 'guard.ps1')
}
$snapshot=Get-Content -LiteralPath (Join-Path $status 'suite_status.json') -Raw | ConvertFrom-Json
$lines=@("checked: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
    "coordinator_alive: $([bool]$alive)",
    'Order: finish MPC6h -> failed/high-gap continuous reviews; MPC8h ON HOLD',
    "completed: $($snapshot.counts.completed) | running: $($snapshot.counts.running) | pending: $($snapshot.counts.pending) | failed: $($snapshot.counts.failed)")
$live=0
foreach($case in $snapshot.cases) {
    if($case.status -notin @('running','failed')){continue}
    $path=Join-Path $case.output 'progress.json'
    $progress=if(Test-Path -LiteralPath $path){Get-Content -LiteralPath $path -Raw|ConvertFrom-Json}else{$null}
    if($case.status -eq 'running') {
        $worker=Get-Process -Id $case.pid -ErrorAction SilentlyContinue
        $workerAlive=$worker -and ($worker.StartTime.ToFileTimeUtc() -eq $case.process_created)
        if($workerAlive){$live++}
        $budget=if($case.ol_time_limit_s){$case.ol_time_limit_s}else{3600}
        $lines+="$($case.name) | alive=$([bool]$workerAlive) | step=$($progress.current_step)/304 | OL_limit=$budget"
    }else{
        $lines+="FAILED $($case.name) | checkpoint retained"
    }
}
$lines+="verified_live_workers: $live/$($coordinator.parallel_cases)"
$text=($lines -join "`r`n")+"`r`n"
$encoding=New-Object Text.UTF8Encoding($false)
[IO.File]::WriteAllText((Join-Path $status 'two_hour_check.log'),$text,$encoding)
[IO.File]::AppendAllText((Join-Path $status 'two_hour_history.log'),$text+"`r`n",$encoding)
