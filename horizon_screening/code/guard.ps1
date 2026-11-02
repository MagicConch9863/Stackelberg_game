$ErrorActionPreference = 'Stop'
$batchRoot = Split-Path -Parent $PSScriptRoot
$statusRoot = Join-Path $batchRoot 'status'
if (Test-Path -LiteralPath (Join-Path $statusRoot 'STOP')) { exit 0 }
$coordinatorPath = Join-Path $statusRoot 'coordinator.json'
$scriptPath = Join-Path $PSScriptRoot 'run_suite.py'
if (Test-Path -LiteralPath $coordinatorPath) {
    $state = Get-Content -LiteralPath $coordinatorPath -Raw | ConvertFrom-Json
    if ($state.status -in @('completed','completed_with_failures')) { exit 0 }
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId=$($state.pid)" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like "*$scriptPath*") { exit 0 }
}
$pythonPath = 'C:\Program Files\Python311\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$arguments = @('-u', ('"' + $scriptPath + '"'))
$startInfo = New-Object System.Diagnostics.ProcessStartInfo
$startInfo.FileName = $pythonPath
$startInfo.Arguments = $arguments -join ' '
$startInfo.WorkingDirectory = Split-Path -Parent $batchRoot
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$startInfo.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Hidden
$worker = [System.Diagnostics.Process]::Start($startInfo)
Add-Content -LiteralPath (Join-Path $statusRoot 'monitor.log') -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') started coordinator pid=$($worker.Id)"
