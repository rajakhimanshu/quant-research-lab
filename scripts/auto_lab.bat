@echo off
REM Continuous lab runner (scripts\auto_lab.py). Started at logon by task ProofbookAutoLab
REM or by hand / Antigravity. Only one instance runs (lock file). No Cursor.
REM Pauses on research\lab\STOP. Output is appended to research\lab\auto_lab.log.

cd /d "%~dp0.."
if not exist "research\lab" mkdir "research\lab"
set LOG=research\lab\auto_lab.log
set PYTHONIOENCODING=utf-8

>> "%LOG%" echo ==== Auto lab start %DATE% %TIME% ====
.venv\Scripts\python -u scripts\auto_lab.py >> "%LOG%" 2>&1
