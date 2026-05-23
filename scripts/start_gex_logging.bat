@echo off
REM Run this script once daily before market open (e.g. via Windows Task Scheduler at 08:00 ET).
REM It logs a snapshot of the current GEX/levels data to logs/ for future calibration.
REM Without this data, the intraday level thresholds cannot be calibrated.
echo Running GEX snapshot logger...
cd /d C:\Users\asare\Downloads\h41_bias&regime_engine
python freeflow_logger.py
echo Done. Check logs/ for today's snapshot.
