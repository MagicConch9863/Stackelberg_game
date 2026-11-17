$ErrorActionPreference='Stop'
$batch=Split-Path -Parent $PSScriptRoot
$statePath=Join-Path $batch 'status/suite_status.json'
$suite=Get-Content -LiteralPath $statePath -Raw|ConvertFrom-Json
$originals=@($suite.cases|Where-Object {-not $_.review})
$legacy=@($suite.cases|Where-Object review)
foreach($t in $legacy){$t.status='paused';$t|Add-Member review_mode 'legacy' -Force}
function Get-SavedStep($folder){
    $path=Join-Path $folder 'checkpoint.json'
    if(-not(Test-Path -LiteralPath $path)){return -1}
    $reader=[System.IO.File]::OpenText($path)
    try{$buffer=New-Object char[] 1024;$length=$reader.Read($buffer,0,1024);$header=New-Object string($buffer,0,$length)}finally{$reader.Dispose()}
    if($header -match '"next_step"\s*:\s*(\d+)'){return [int]$Matches[1]}
    return -1
}
function Copy-Task($task){return ($task|ConvertTo-Json -Depth 25|ConvertFrom-Json)}
function Reset-Task($task){
    $task.status='pending';$task.retries=0;$task.startup_deferrals=2;$task.retry_after=0
    foreach($key in @('pid','process_created','started_at','finished_at','first_started_timestamp','exit_code','exit_history','review_scanned','certificate_scanned')){$task.PSObject.Properties.Remove($key)}
}
$new=@();$complete=0;$missing=0;$windows=0
foreach($original in $originals){
    if($original.horizon -eq 32){$original.status='paused';continue}
    $choices=@($original)+@($legacy|Where-Object {$_.name -eq ($original.name+'_review') -or $_.name -eq ($original.name+'_review_quality2')})
    $candidates=@(foreach($t in $choices){$step=Get-SavedStep $t.output;if($step -ge 16){[PSCustomObject]@{Task=$t;Step=$step;Modified=(Get-Item -LiteralPath (Join-Path $t.output 'checkpoint.json')).LastWriteTimeUtc}}})
    if(-not $candidates.Count){throw "Missing checkpoint: $($original.name)"}
    $selected=$candidates|Sort-Object @{Expression='Step';Descending=$true},@{Expression='Modified';Descending=$true}|Select-Object -First 1
    $source=$selected.Task.output
    $full=$selected.Step -eq 304 -and (Test-Path (Join-Path $source 'metrics.json')) -and (Test-Path (Join-Path $source 'official_timeseries.csv'))
    if($full){$full=@(Import-Csv -LiteralPath (Join-Path $source 'official_timeseries.csv')).Count -eq 288}
    if($full){
        $complete++
        $selected.Task.status='completed'
        if($selected.Task.review){$selected.Task|Add-Member review_mode 'completion' -Force;$selected.Task|Add-Member source_case $original.name -Force;$selected.Task|Add-Member certificate_scanned $true -Force}
        $raw=Get-Content -LiteralPath (Join-Path $source 'solver_log.json') -Raw|ConvertFrom-Json
        $targets=@($raw|Where-Object {$_.origin -ge 16 -and $_.gap -gt .200000001})
        if(-not $targets.Count){continue}
        $task=Copy-Task $original
        Reset-Task $task
        $task.name=$original.name+'_certificate'
        $task.output=Join-Path (Join-Path $batch 'review/certificates') $original.name
        $task|Add-Member review_mode 'certificate' -Force
        $windows+=$targets.Count
    }else{
        $missing++
        $destination=Join-Path (Join-Path $batch 'review/completion') $original.name
        New-Item -ItemType Directory -Path $destination -Force|Out-Null
        foreach($file in @('manifest.json','checkpoint.json','timeseries.csv','rolling_forecasts.csv','solver_log.json','progress.json')){
            $from=Join-Path $source $file
            if(Test-Path -LiteralPath $from){Copy-Item -LiteralPath $from -Destination (Join-Path $destination $file)}
        }
        $plan=[ordered]@{source=$source;replay_from=$selected.Step;targets=@{};policy='deadline48_continue_best_saved_checkpoint';continuous_replay=$true;original_results_unchanged=$true}
        [System.IO.File]::WriteAllText((Join-Path $destination 'review_plan.json'),($plan|ConvertTo-Json -Depth 5),(New-Object System.Text.UTF8Encoding($false)))
        $task=Copy-Task $original
        Reset-Task $task
        $task.name=$original.name+'_deadline_complete'
        $task.output=$destination
        $task|Add-Member review_mode 'completion' -Force
    }
    $task|Add-Member review $true -Force
    $task|Add-Member review_source $source -Force
    $task|Add-Member source_case $original.name -Force
    $task|Add-Member review_failed_origin $null -Force
    $task|Add-Member priority_gap_gt20 $true -Force
    $new+=$task
}
$suite.cases=$originals+$legacy+$new
[System.IO.File]::WriteAllText($statePath,($suite|ConvertTo-Json -Depth 30),(New-Object System.Text.UTF8Encoding($false)))
[PSCustomObject]@{Complete_trajectories=$complete;Missing_trajectories=$missing;Certificate_cases=@($new|Where-Object review_mode -eq certificate).Count;Certificate_windows=$windows;New_tasks=$new.Count}|Format-List
