@echo off
setlocal enabledelayedexpansion

:: Script builds/activates 3d_pipeline conda environment in Windows Environment

:: Define the name of the Anaconda environment
set "ENV_NAME=3d_pipeline"

:: Check if conda is installed
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo Anaconda (or Miniconda) is not installed. Please install it before running this script.
    exit /b 1
)

:: Check if the environment already exists
conda info --envs | findstr /i /c:"%ENV_NAME%" >nul 2>&1
if %errorlevel% equ 0 (
    echo "Activating existing Anaconda environment: %ENV_NAME%"
    conda activate %ENV_NAME%
) else (
    echo "Creating and activating a new Anaconda environment: %ENV_NAME%"
    conda create -y -n %ENV_NAME% python=3.9
    conda activate %ENV_NAME%
    echo "Installing pip requirements"
    pip install -r requirements.txt
    echo "Environment ready"
)

exit /b 0
