@echo off
cd /d "%~dp0"
title AcciDEN - Updated Admin with Delete Video
set ACCIDEN_PORT=8010
echo Opening the UPDATED Admin dashboard on a separate port.
echo Use the browser tab opened by this launcher.
python launch_app.py
pause
