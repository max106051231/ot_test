# Run in elevated PowerShell:
#   Right-click -> Run with PowerShell  (or)
#   Start-Process powershell -Verb RunAs -ArgumentList '-ExecutionPolicy Bypass -File', "$PSScriptRoot\open_firewall_2000.ps1"

$ErrorActionPreference = "Continue"
Write-Host "=== Semi-Shield: open TCP 2000 + Radmin Private ==="

Get-NetConnectionProfile |
  Where-Object { $_.InterfaceAlias -match "Radmin" } |
  ForEach-Object {
    Set-NetConnectionProfile -InterfaceIndex $_.InterfaceIndex -NetworkCategory Private
    Write-Host "[OK] $($_.InterfaceAlias) -> Private"
  }

Get-NetFirewallRule -DisplayName "Semi-Shield 2000" -ErrorAction SilentlyContinue |
  Remove-NetFirewallRule -ErrorAction SilentlyContinue

New-NetFirewallRule -DisplayName "Semi-Shield 2000" -Direction Inbound `
  -Protocol TCP -LocalPort 2000 -Action Allow -Profile Any -Enabled True | Out-Null
Write-Host "[OK] TCP 2000 allowed (all profiles)"

$py = (Get-Command python -ErrorAction SilentlyContinue)?.Source
if ($py) {
  Get-NetFirewallRule -DisplayName "Semi-Shield Python" -ErrorAction SilentlyContinue |
    Remove-NetFirewallRule -ErrorAction SilentlyContinue
  New-NetFirewallRule -DisplayName "Semi-Shield Python" -Direction Inbound `
    -Program $py -Action Allow -Profile Any -Enabled True | Out-Null
  Write-Host "[OK] Allowed $py"
}

Write-Host ""
Write-Host "Peer URL:"
Write-Host "  http://26.187.230.64:2000/"
Write-Host "  (must include :2000)"
Write-Host ""
Pause
