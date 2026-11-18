# Laptop 1: start the PathSnap API (model + HMM + server-side evaluation).
# Usage (from the repo root, in PowerShell):
#   .\deploy\run_api.ps1 -TripsDir "C:\path\to\evaluation_trips"
# TripsDir = the folder holding one sub-folder per trip (<id>\<id>.track/.nodes/.arcs/.route).
param(
    [Parameter(Mandatory = $true)][string]$TripsDir,
    [int]$Port = 8000
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path $TripsDir)) { throw "Trips folder not found: $TripsDir" }
$env:PATHSNAP_TRIPS_DIR = (Resolve-Path $TripsDir).Path

$python = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

Write-Host "Trips folder : $env:PATHSNAP_TRIPS_DIR"
Write-Host "LAN addresses of this laptop (give one to Laptop 2 as http://<ip>:$Port):"
Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } |
    ForEach-Object { Write-Host "   http://$($_.IPAddress):$Port   ($($_.InterfaceAlias))" }
Write-Host "If Laptop 2 cannot connect, allow the port once (admin PowerShell):"
Write-Host "   New-NetFirewallRule -DisplayName PathSnapAPI -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow"
Write-Host ""

& $python -m uvicorn api.main:app --host 0.0.0.0 --port $Port
