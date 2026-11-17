param([switch]$Prepare)
$ErrorActionPreference = 'Stop'
$batchRoot = Split-Path -Parent $PSScriptRoot
$workspace = Split-Path -Parent $batchRoot
$statusRoot = Join-Path $batchRoot 'status'
if (!$Prepare -and (Test-Path -LiteralPath (Join-Path $statusRoot 'STOP'))) { exit 0 }
$scriptPath = Join-Path $PSScriptRoot 'run_queue.py'
if ($Prepare) { $scriptPath = Join-Path $PSScriptRoot 'prepare.py' }
if (!$Prepare -and (Test-Path -LiteralPath (Join-Path $statusRoot 'launching.json'))) {
    $launch = Get-Content -LiteralPath (Join-Path $statusRoot 'launching.json') -Raw | ConvertFrom-Json
    $launchProcess = Get-Process -Id $launch.pid -ErrorAction SilentlyContinue
    if ($launchProcess -and $launchProcess.StartTime.ToFileTimeUtc() -eq $launch.process_created) { exit 0 }
}
if (!$Prepare -and (Test-Path -LiteralPath (Join-Path $statusRoot 'coordinator.json'))) {
    $saved = Get-Content -LiteralPath (Join-Path $statusRoot 'coordinator.json') -Raw | ConvertFrom-Json
    if ($saved.status -eq 'completed' -or $saved.status -eq 'attention') { exit 0 }
    $existing = Get-Process -Id $saved.pid -ErrorAction SilentlyContinue
    if ($existing -and $existing.StartTime.ToFileTimeUtc() -eq $saved.process_created) { exit 0 }
}
$bootstrap = "import pathlib,runpy,site,sys;site.sethelper();entry=sys.argv.pop(1);sys.path.insert(0,str(pathlib.Path(entry).parent));sys.argv[0]=entry;runpy.run_path(entry,run_name='__main__')"
$startInfo = New-Object System.Diagnostics.ProcessStartInfo
$startInfo.FileName = 'C:\Program Files\Python311\python.exe'
$startInfo.Arguments = '-S -u -c "' + $bootstrap + '" "' + $scriptPath + '"'
$startInfo.WorkingDirectory = $workspace
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$startInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
$startInfo.EnvironmentVariables['PYTHONPATH'] = Join-Path $workspace 'venv\Lib\site-packages'
$startInfo.EnvironmentVariables['PYTHONIOENCODING'] = 'utf-8'
foreach ($key in @('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')) {
    $startInfo.EnvironmentVariables[$key] = '1'
}
if ($Prepare) {
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
}
$worker = [System.Diagnostics.Process]::Start($startInfo)
if ($Prepare) {
    $output = $worker.StandardOutput.ReadToEnd()
    $errors = $worker.StandardError.ReadToEnd()
    $worker.WaitForExit()
    [System.IO.File]::WriteAllText((Join-Path $statusRoot 'prepare.log'), $output + $errors)
    Write-Output "Preparation exit=$($worker.ExitCode)"
    Write-Output $output
    Write-Output $errors
} else {
    $launch = @{pid=$worker.Id; process_created=$worker.StartTime.ToFileTimeUtc()}
    [System.IO.File]::WriteAllText((Join-Path $statusRoot 'launching.json'),($launch | ConvertTo-Json))
    Add-Content -LiteralPath (Join-Path $statusRoot 'monitor.log') -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') Koordinator gestartet: PID $($worker.Id)"
    Write-Output "Coordinator PID=$($worker.Id)"
}
