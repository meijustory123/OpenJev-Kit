@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\bootstrap.ps1"
if errorlevel 1 (
    echo OpenJev setup failed. See outputs\setup for details, then double-click to retry.
    pause
    exit /b 1
)
exit /b 0
