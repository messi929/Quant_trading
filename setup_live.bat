@echo off
setlocal EnableDelayedExpansion

echo ============================================================
echo  Alpha Signal Discovery Engine - Live Trading Setup
echo ============================================================
echo.

:: ── 1. Check conda environment ───────────────────────────────
echo [1/5] Checking conda environment...
call conda activate quant 2>nul
if errorlevel 1 (
    echo  [WARN] 'quant' environment not found. Trying quant_trading...
    call conda activate quant_trading 2>nul
    if errorlevel 1 (
        echo  [ERROR] No conda environment found.
        echo  Please run: conda create -n quant python=3.11 -y
        pause
        exit /b 1
    )
)
echo  OK: conda environment activated

:: ── 2. Install live trading packages ─────────────────────────
echo.
echo [2/5] Installing live trading packages...
pip install "python-dotenv>=1.0.0" "schedule>=1.2.0" "streamlit>=1.32.0" "plotly>=5.18.0" "requests>=2.31.0" -q
if errorlevel 1 (
    echo  [WARN] Some packages may have failed. Continuing...
)
echo  OK: packages installed

:: ── 3. Create .env file ───────────────────────────────────────
echo.
echo [3/5] Setting up credentials...
set ENV_FILE=%~dp0.env

if exist "%ENV_FILE%" (
    echo  INFO: .env file already exists. Skipping.
    goto env_done
)

echo.
echo  Enter your KIS (Korea Investment Securities) API credentials.
echo  Get them from: https://apiportal.koreainvestment.com
echo.
set /p APP_KEY=  APP_KEY:
set /p APP_SECRET=  APP_SECRET:
set /p CANO=  Account number (8 digits):
set /p ACNT_CD=  Account product code (2 digits, default=01):

if "%ACNT_CD%"=="" set ACNT_CD=01

echo.
echo  Select trading mode:
echo  1) sandbox    - Paper trading (SAFE, recommended)
echo  2) production - Live trading  (REAL ORDERS!)
set /p MODE_SEL=  Select [1/2, default=1]:
if "%MODE_SEL%"=="2" (
    set KIS_MODE=production
) else (
    set KIS_MODE=sandbox
)

(
    echo KIS_APP_KEY=%APP_KEY%
    echo KIS_APP_SECRET=%APP_SECRET%
    echo KIS_CANO=%CANO%
    echo KIS_ACNT_PRDT_CD=%ACNT_CD%
    echo KIS_MODE=%KIS_MODE%
) > "%ENV_FILE%"

echo  OK: .env created (mode: %KIS_MODE%)
:env_done

:: ── 4. Create directories ─────────────────────────────────────
echo.
echo [4/5] Creating directories...
if not exist "%~dp0data\raw"              mkdir "%~dp0data\raw"
if not exist "%~dp0data\processed"        mkdir "%~dp0data\processed"
if not exist "%~dp0saved_models\backups"  mkdir "%~dp0saved_models\backups"
if not exist "%~dp0logs"                  mkdir "%~dp0logs"
if not exist "%~dp0results"               mkdir "%~dp0results"
if not exist "%~dp0tracking"              mkdir "%~dp0tracking"
echo  OK: directories created

:: ── 5. Windows Task Scheduler (optional) ─────────────────────
echo.
echo [5/5] Task Scheduler registration...
echo.
echo  Tasks to register:
echo    AlphaSignal_DataCollect   - daily 06:00
echo    AlphaSignal_SignalGen     - daily 06:30
echo    AlphaSignal_OrderSubmit   - daily 08:50 (Monday orders only)
echo    AlphaSignal_EODRecord     - daily 16:10
echo    AlphaSignal_WeeklyRetrain - Saturday 22:00
echo.
set /p SCHED_OK=  Register tasks? [y/N]:
if /i not "%SCHED_OK%"=="y" goto sched_skip

set PROJ=%~dp0
schtasks /create /tn "AlphaSignal_DataCollect"   /tr "%PROJ%run_step.bat collect"        /sc daily  /st 06:00 /f /rl HIGHEST >nul 2>&1
schtasks /create /tn "AlphaSignal_SignalGen"      /tr "%PROJ%run_step.bat signal"         /sc daily  /st 06:30 /f /rl HIGHEST >nul 2>&1
schtasks /create /tn "AlphaSignal_OrderSubmit"    /tr "%PROJ%run_step.bat order"          /sc daily  /st 08:50 /f /rl HIGHEST >nul 2>&1
schtasks /create /tn "AlphaSignal_EODRecord"      /tr "%PROJ%run_step.bat eod"            /sc daily  /st 16:10 /f /rl HIGHEST >nul 2>&1
schtasks /create /tn "AlphaSignal_WeeklyRetrain"  /tr "%PROJ%run_retrain.bat weekly"      /sc weekly /d SAT /st 22:00 /f /rl HIGHEST >nul 2>&1
echo  OK: tasks registered
goto sched_done

:sched_skip
echo  SKIP: task scheduler not registered
echo  Manual run: python scheduler\daily_runner.py --step signal

:sched_done

:: ── Done ──────────────────────────────────────────────────────
echo.
echo ============================================================
echo  Setup complete!
echo ============================================================
echo.
echo  Next steps:
echo.
echo  1. Test signal generation (no orders):
echo     python scheduler\daily_runner.py --step signal
echo.
echo  2. Launch dashboard:
echo     streamlit run dashboard\app.py
echo.
echo  3. Run as daemon (auto schedule):
echo     python scheduler\daily_runner.py --daemon
echo.
if "%KIS_MODE%"=="production" (
    echo  WARNING: production mode - real orders will be placed!
    echo  Test thoroughly in sandbox mode first.
    echo.
)
pause
