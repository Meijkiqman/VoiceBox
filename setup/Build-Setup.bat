@echo off
cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH.
    pause
    exit /b
)

python -m pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    python -m pip install pyinstaller
)

echo Building VoiceBoxSetup.exe ...
python -m PyInstaller --onefile --console --name VoiceBoxSetup ^
    --add-data "..\voicebox.py;app" ^
    --add-data "..\controls.json;app" ^
    --add-data "..\dlc.json;app" ^
    --add-data "..\VoiceBox.bat;app" ^
    voicebox_setup.py

if errorlevel 1 (
    echo BUILD FAILED
    pause
    exit /b
)
echo.
echo Done: %~dp0dist\VoiceBoxSetup.exe
pause
