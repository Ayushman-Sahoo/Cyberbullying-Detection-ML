@echo off
cd /d "%~dp0"
python -m uvicorn cyberbullying_app:app --app-dir "%~dp0" --host 127.0.0.1 --port 8007 --reload

