@echo off
:: 일간 스텝 실행 헬퍼 (Windows 작업 스케줄러용)
:: 사용법: run_step.bat [collect|signal|order|eod]

setlocal
cd /d %~dp0
call conda activate quant_trading 2>nul
python scheduler\daily_runner.py --step %1 >> logs\scheduler.log 2>&1
