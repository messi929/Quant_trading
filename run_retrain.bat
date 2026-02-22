@echo off
:: 재학습 실행 헬퍼 (Windows 작업 스케줄러용)
:: 사용법: run_retrain.bat [weekly|monthly]

setlocal
cd /d %~dp0
call conda activate quant_trading 2>nul
python scheduler\retrain_runner.py --mode %1 >> logs\retrain.log 2>&1
