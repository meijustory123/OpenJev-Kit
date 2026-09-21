@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" start "" "%~dp0.venv\Scripts\pythonw.exe" -m scripts.launch_web --stop
exit /b 0
