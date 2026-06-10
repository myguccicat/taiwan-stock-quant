@echo off
REM 一鍵設定 Windows 工作排程器 — 每日 08:10 執行美股早報 LINE 推播
REM 請「以系統管理員身份執行」此批次檔

set SCRIPT_PATH=%~dp0send_market_briefing_line.py

echo 正在安裝 requests 套件...
pip install requests --quiet

echo 正在建立排程任務...
schtasks /create /tn "美股早報LINE推播" ^
  /tr "python \"%SCRIPT_PATH%\"" ^
  /sc DAILY /st 08:10 ^
  /f ^
  /rl HIGHEST

if %errorlevel%==0 (
    echo.
    echo ✅ 排程設定完成！每天 08:10 將自動推播美股早報到您的 LINE。
    echo    任務名稱：美股早報LINE推播
    echo    腳本路徑：%SCRIPT_PATH%
) else (
    echo.
    echo ❌ 設定失敗，請確認以「系統管理員身份執行」此批次檔。
)

pause
