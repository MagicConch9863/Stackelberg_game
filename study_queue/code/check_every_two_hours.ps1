$ErrorActionPreference = 'Stop'
$batchRoot = Split-Path -Parent $PSScriptRoot
$statusRoot = Join-Path $batchRoot 'status'
& (Join-Path $PSScriptRoot 'guard.ps1')
$brief = Join-Path $statusRoot 'progress_brief.log'
if (Test-Path -LiteralPath $brief) {
    $content = $null
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        try {
            $sharing = [System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete
            $file = [System.IO.File]::Open($brief,[System.IO.FileMode]::Open,[System.IO.FileAccess]::Read,$sharing)
            $reader = New-Object System.IO.StreamReader($file)
            try { $content = $reader.ReadToEnd() } finally { $reader.Dispose() }
            break
        } catch [System.IO.IOException] { Start-Sleep -Milliseconds 100 }
    }
    if ($null -eq $content) { throw 'Fortschrittsdatei voruebergehend gesperrt; naechste Kontrolle wird erneut lesen.' }
    $text = "Kontrolle: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`r`n" + $content
    [System.IO.File]::WriteAllText((Join-Path $statusRoot 'scheduled_two_hour_check.log'),$text)
    [System.IO.File]::AppendAllText((Join-Path $statusRoot 'scheduled_two_hour_history.log'),$text + "`r`n")
}
