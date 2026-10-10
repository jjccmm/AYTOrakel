@echo off
cd /d "%~dp0"
py ayto_web.py
if errorlevel 1 pause
