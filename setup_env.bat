@echo off
REM ============================================
REM Quant Trading System - Environment Setup
REM RTX 4060 Ti (8GB) optimized
REM ============================================

echo ============================================
echo  Quant Trading System - Environment Setup
echo ============================================

REM Check if conda is installed
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Conda not found. Please install Miniconda first:
    echo   https://docs.anaconda.com/miniconda/
    echo.
    echo After installing, restart this script.
    pause
    exit /b 1
)

echo [1/5] Creating conda environment...
conda create -n quant python=3.11 -y
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create conda environment
    pause
    exit /b 1
)

echo [2/5] Activating environment...
call conda activate quant

echo [3/5] Installing PyTorch with CUDA 12.4...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install PyTorch
    pause
    exit /b 1
)

echo [4/5] Installing project dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo [5/5] Verifying installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo ============================================
echo  Setup complete! Activate with:
echo    conda activate quant
echo  Run with:
echo    python main.py
echo ============================================
pause
