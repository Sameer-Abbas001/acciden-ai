@echo off
cd /d "%~dp0"
title AcciDEN AI - Collision & Traffic Intelligence Hub
echo ========================================================
echo   AcciDEN AI - Neural Collision & Traffic Intelligence
echo   Starting backend server on http://127.0.0.1:8000 ...
echo ========================================================
echo.
python launch_app.py
pause
