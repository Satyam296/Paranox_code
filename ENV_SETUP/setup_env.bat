@echo off
REM Space Station Challenge - Environment Setup Script for Windows
REM This script creates a conda environment named "EDU" with all required dependencies

echo ========================================
echo Space Station Challenge - Environment Setup
echo ========================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Conda is not installed or not in PATH
    echo Please install Anaconda or Miniconda first
    pause
    exit /b 1
)

echo Creating conda environment 'EDU'...
call conda create -n EDU python=3.10 -y

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to create conda environment
    pause
    exit /b 1
)

echo.
echo Activating EDU environment...
call conda activate EDU

echo.
echo Installing PyTorch with CUDA support...
call conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

echo.
echo Installing Ultralytics YOLOv8...
call pip install ultralytics

echo.
echo Installing additional dependencies...
call pip install opencv-python
call pip install matplotlib
call pip install seaborn
call pip install pandas
call pip install numpy
call pip install pillow
call pip install pyyaml
call pip install tqdm
call pip install scikit-learn
call pip install tensorboard

echo.
echo ========================================
echo Environment setup complete!
echo ========================================
echo.
echo To activate the environment, use:
echo   conda activate EDU
echo.
echo To start training, navigate to the scripts folder and run:
echo   python train.py
echo.
pause
