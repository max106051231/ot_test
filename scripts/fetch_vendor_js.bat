@echo off
REM 下載 OT.html / agent_chat 所需離線前端套件
cd /d "%~dp0"
python -c "import urllib.request; from pathlib import Path; v=Path('static/vendor'); v.mkdir(parents=True, exist_ok=True); files={'html2canvas.min.js':'https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js','chart.umd.min.js':'https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js','mermaid.min.js':'https://cdn.jsdelivr.net/npm/mermaid@10.9.0/dist/mermaid.min.js'}; [print(n, urllib.request.urlretrieve(u, v/n)[0]) for n,u in files.items()]"
echo Done. jspdf should already exist in static\vendor\
pause
