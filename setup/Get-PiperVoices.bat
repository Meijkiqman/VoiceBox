@echo off
rem Install the Piper neural TTS engine + realistic Ryan (male) and
rem Lessac (female) English voices into the piper\ folder.
cd /d "%~dp0"
python get_piper_voices.py
pause
