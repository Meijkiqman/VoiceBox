@echo off
rem Install the Piper neural TTS engine + six realistic English voices
rem (3 male / 3 female, US and British) into the piper\ folder.
cd /d "%~dp0"
python get_piper_voices.py
pause
