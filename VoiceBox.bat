@echo off
cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH.
    echo Please install Python from https://python.org
    pause
    exit /b
)

python -c "import numpy, scipy, sounddevice, soundfile, pygame, keyboard" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install numpy scipy sounddevice soundfile pygame keyboard
)

python voicebox.py %*
if errorlevel 1 pause
