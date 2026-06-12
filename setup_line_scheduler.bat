@echo off
chcp 65001 >nul
REM Setup Windows Task Scheduler - Daily 08:10 LINE market briefing
REM Run as Administrator

echo Creating scheduled task...
schtasks /create /tn "MarketBriefingLINE" /tr "C:\Users\user\Desktop\stock\run_market_briefing.bat" /sc DAILY /st 08:10 /f /rl HIGHEST

if %errorlevel%==0 (
    echo.
    echo [OK] Task created! Will run daily at 08:10.
    echo      Task name: MarketBriefingLINE
) else (
    echo.
    echo [FAIL] Please run this file as Administrator.
)

pause
