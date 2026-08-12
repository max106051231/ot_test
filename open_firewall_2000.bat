@echo off
REM Run as Administrator: allow TCP 2000 + set Radmin VPN to Private
echo ========================================
echo  Semi-Shield: open port 2000 for VPN
echo  Right-click this file -^> Run as administrator
echo ========================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-NetConnectionProfile | Where-Object { $_.InterfaceAlias -match 'Radmin' } | ForEach-Object { Set-NetConnectionProfile -InterfaceIndex $_.InterfaceIndex -NetworkCategory Private; Write-Host ('[OK] ' + $_.InterfaceAlias + ' -> Private') }"

netsh advfirewall firewall delete rule name="Semi-Shield 2000" >nul 2>&1
netsh advfirewall firewall add rule name="Semi-Shield 2000" dir=in action=allow protocol=TCP localport=2000 profile=any enable=yes
if errorlevel 1 (
  echo [FAIL] Please Run as administrator
  pause
  exit /b 1
)
echo [OK] TCP 2000 allowed

for /f "delims=" %%P in ('where python 2^>nul') do (
  netsh advfirewall firewall delete rule name="Semi-Shield Python" >nul 2>&1
  netsh advfirewall firewall add rule name="Semi-Shield Python" dir=in action=allow program="%%P" enable=yes profile=any
  echo [OK] Allowed %%P
  goto :after_py
)
:after_py

echo.
echo Peer should open EXACTLY:
echo   http://26.187.230.64:2000/
echo   http://26.187.230.64:2000/chat
echo   http://26.187.230.64:2000/monitor
echo.
echo Must include :2000
echo.
pause
