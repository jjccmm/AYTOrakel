@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" ayto_data_viewer.py
) else (
  python ayto_data_viewer.py
)
