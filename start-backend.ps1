param(
    [switch]$NoReload
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendRoot = Join-Path $ProjectRoot "backend"
$VenvRoot = Join-Path $BackendRoot ".venv"
$VenvPython = Join-Path $VenvRoot "Scripts\python.exe"
$ReadyMarker = Join-Path $VenvRoot ".quant-deps-ready"
$Pyproject = Join-Path $BackendRoot "pyproject.toml"

Set-Location $BackendRoot

if (-not (Test-Path $VenvPython)) {
    python -c "import sys; assert sys.version_info >= (3, 11), '需要 Python 3.11 或更高版本'"
    python -m venv $VenvRoot
    & $VenvPython -m pip install --upgrade pip
}

if (
    (-not (Test-Path $ReadyMarker)) -or
    ((Get-Item $Pyproject).LastWriteTimeUtc -gt (Get-Item $ReadyMarker).LastWriteTimeUtc)
) {
    & $VenvPython -m pip install -e ".[test]"
    Set-Content -Path $ReadyMarker -Value (Get-Date).ToUniversalTime().ToString("O")
}

$reloadArgs = @()
if (-not $NoReload) {
    $reloadArgs += "--reload"
}

Write-Host "后端地址: http://localhost:8000"
Write-Host "API 文档: http://localhost:8000/docs"
& $VenvPython -m uvicorn app.main:app --host 0.0.0.0 --port 8000 @reloadArgs
