$ErrorActionPreference = 'Stop'
$batchRoot = Split-Path -Parent $PSScriptRoot
$horizonGuard = Join-Path (Split-Path -Parent $batchRoot) 'horizon_screening\code\guard.ps1'
if (Test-Path -LiteralPath $horizonGuard) { & $horizonGuard }
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
$launch = @{
    FilePath = $pythonPath
    ArgumentList = $arguments
    WindowStyle = 'Hidden'
    WorkingDirectory = (Split-Path -Parent $batchRoot)
    PassThru = $true
    RedirectStandardOutput = (Join-Path $statusRoot 'coordinator_stdout.log')
    RedirectStandardError = (Join-Path $statusRoot 'coordinator_stderr.log')
}
$worker = Start-Process @launch
Add-Content -LiteralPath (Join-Path $statusRoot 'monitor.log') -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') started coordinator pid=$($worker.Id)"
