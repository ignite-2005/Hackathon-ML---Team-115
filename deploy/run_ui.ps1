# Laptop 2: start the PathSnap evaluator UI and point it at the API on Laptop 1.
# Usage (from the repo root, in PowerShell):
#   .\deploy\run_ui.ps1 -ApiUrl "http://192.168.1.20:8000"
param(
    [string]$ApiUrl = "http://localhost:8000",
    [int]$Port = 8501
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$env:PATHSNAP_API_URL = $ApiUrl
$python = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

Write-Host "API (Laptop 1): $ApiUrl"
try {
    $h = Invoke-RestMethod "$ApiUrl/health" -TimeoutSec 5
    Write-Host "API reachable - model: $($h.model), trips: $($h.n_trips)"
} catch {
    Write-Host "WARNING: API not reachable yet at $ApiUrl (the UI will show an error until it is)."
}

& $python -m streamlit run ui/app.py --server.port $Port --browser.gatherUsageStats false
