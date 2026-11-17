$ErrorActionPreference = 'Stop'
$batchRoot = Split-Path -Parent $PSScriptRoot
$priorityRoot = Join-Path (Split-Path -Parent $batchRoot) 'study_queue'
if (Test-Path -LiteralPath (Join-Path $priorityRoot 'status/ACTIVE')) {
    & (Join-Path $priorityRoot 'code/guard.ps1')
    exit 0
}
$statusRoot = Join-Path $batchRoot 'status'
if (Test-Path -LiteralPath (Join-Path $statusRoot 'STOP')) { exit 0 }
$coordinatorPath = Join-Path $statusRoot 'coordinator.json'
$scriptPath = Join-Path $PSScriptRoot 'run_suite.py'
if (Test-Path -LiteralPath $coordinatorPath) {
    $state = Get-Content -LiteralPath $coordinatorPath -Raw | ConvertFrom-Json
    if ($state.status -eq 'completed') { exit 0 }
    if ($state.status -eq 'completed_with_failures') {
        $policyPath = Join-Path $statusRoot 'deadline_policy.json'
        if (!(Test-Path -LiteralPath $policyPath)) { exit 0 }
        $policy = Get-Content -LiteralPath $policyPath -Raw | ConvertFrom-Json
        if (!$policy.quality_followup.automatic_retry_unqualified) { exit 0 }
        $suite = Get-Content -LiteralPath (Join-Path $statusRoot 'suite_status.json') -Raw | ConvertFrom-Json
        $remaining = @($suite.cases | Where-Object {
            $_.review_mode -eq 'quality_review' -and $_.status -eq 'failed' -and !$_.automatic_retry_blocked
        })
        if (!$remaining.Count) { exit 0 }
    }
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId=$($state.pid)" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like "*$scriptPath*") { exit 0 }
}
$pythonPath = 'C:\Program Files\Python311\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$bootstrap = "import pathlib,runpy,site,sys;site.sethelper();entry=sys.argv.pop(1);sys.path.insert(0,str(pathlib.Path(entry).parent));sys.argv[0]=entry;runpy.run_path(entry,run_name='__main__')"
$arguments = @('-S', '-u', '-c', ('"' + $bootstrap + '"'), ('"' + $scriptPath + '"'))
$startInfo = New-Object System.Diagnostics.ProcessStartInfo
$startInfo.FileName = $pythonPath
$startInfo.Arguments = $arguments -join ' '
$startInfo.WorkingDirectory = Split-Path -Parent $batchRoot
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$startInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
$worker = [System.Diagnostics.Process]::Start($startInfo)
Add-Content -LiteralPath (Join-Path $statusRoot 'monitor.log') -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') started coordinator pid=$($worker.Id)"
