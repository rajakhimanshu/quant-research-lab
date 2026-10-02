@echo off
REM Weekly hypothesis intake automation.
REM Set this up in Windows Task Scheduler → weekly, your chosen day/time.
REM Action: Start a program → point at this .bat file.

cd /d "%~dp0.."

echo ============================================================
echo  Hypothesis Intake — Weekly Run
echo  %DATE% %TIME%
echo ============================================================

echo.
echo [1/2] Paper monitor (arXiv + BIS/Fed/ECB/BoE feeds)...
.venv\Scripts\python -m ats intake paper-monitor --save

echo.
echo [2/2] Groq brainstorm (all 3 categories, 5 ideas each)...
.venv\Scripts\python -m ats intake groq-brainstorm --all --n 5

echo.
echo ============================================================
echo  Done. Review new raw leads with:
echo    python -m ats intake list --status raw
echo ============================================================
