@echo off
cd /d "C:\Users\user\Desktop\stock"
"C:\Users\user\anaconda3\envs\django310\python.exe" send_market_briefing_line.py
"C:\Users\user\anaconda3\envs\django310\python.exe" daily_run.py
