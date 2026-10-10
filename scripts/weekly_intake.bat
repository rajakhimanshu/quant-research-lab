@echo off
REM Weekly public-source intake. No LLM, no chat tokens, no tests run.
REM Scheduled by Windows Task Scheduler (task: ProofbookWeeklyIntake).
REM Output is appended to data\ideas\weekly_intake.log.

cd /d "%~dp0.."
if not exist "data\ideas" mkdir "data\ideas"
set LOG=data\ideas\weekly_intake.log

>> "%LOG%" echo ============================================================
>> "%LOG%" echo  Hypothesis Intake - Weekly Run  %DATE% %TIME%
>> "%LOG%" echo ============================================================

>> "%LOG%" echo [1/1] Paper monitor (arXiv + BIS/Fed/ECB/BoE feeds)...
.venv\Scripts\python -m ats intake paper-monitor --save >> "%LOG%" 2>&1

>> "%LOG%" echo Done. Review new raw leads with: python -m ats intake list --status raw
>> "%LOG%" echo.
