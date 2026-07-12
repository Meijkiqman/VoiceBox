# VoiceBox

Real-time voice changer + soundboard for Discord (or anything that takes a
microphone). One window: DSP voice effects and presets on the left, a hotkey
soundboard grid on the right, and optional AI voice conversion (RVC models
like Arthur Morgan) running on the GPU in the background.

```
real mic --> [VoiceBox: pitch/effects/soundboard | or RVC AI voice] --> VB-CABLE
                                                                          |
                                                     Discord input = "CABLE Output"
```

## Features

- **Presets** - Space Marine, Ghost, Robot, Chipmunk, Monster, Ork,
  Walkie-Talkie; every ingredient is also a manual row (pitch, robot/vocoder
  mix, helmet doubler, grit, reverb, echo, radio band-pass, bass boost).
- **Soundboard** - drop audio files (wav/flac/ogg/mp3) into `sounds/`; each
  becomes a grid tile. Keys 1-9 fire the first nine. You always hear sounds
  locally; the "To mic" toggle decides whether Discord hears them too.
  Pause freezes everything mid-clip, Stop clears it.
- **AI voice** - point `RVC_DIR` in `voicebox.py` at an RVC-beta package with
  `weights/*.pth` models; VoiceBox runs `rvc_worker.py` on RVC's own bundled
  CUDA Python and pipes the converted voice into the cable. Pick the
  character from the menu; the soundboard keeps working on top.
- **Test - hear myself** self-listen, live mic meter with peak-hold,
  keyboard + mouse + game controller navigation, remappable controls
  (`controls.json`), crash-proof against malformed config.

## Quick start

1. Install [VB-CABLE](https://vb-audio.com/Cable/) (run as admin, reboot).
2. `pip install -r requirements.txt` (or just double-click `VoiceBox.bat`,
   which installs anything missing and launches).
3. Put some sounds in `sounds/`, run `python voicebox.py`.
4. Discord -> Settings -> Voice & Video -> Input Device = **CABLE Output**.
   Also disable Noise Suppression and automatic input sensitivity there,
   or Discord's gate will chop the processed voice.

`python voicebox.py --list` prints audio devices if the auto-match fails;
device substrings are configured at the top of `voicebox.py`.

## Installer

`setup/Build-Setup.bat` builds `setup/dist/VoiceBoxSetup.exe` (PyInstaller):
a one-click bootstrapper that checks the system, silently installs Python and
the libraries if missing, downloads + silently installs VB-CABLE (one UAC
prompt), and installs VoiceBox with a Desktop shortcut. App files travel
inside the exe; `--url <zip>` downloads them instead. `--check` reports
without changing anything.

## Tests

`python tests/run_all.py` - five suites, 100+ checks, headless (no audio
hardware or window needed): DSP effects math, routing/pause/stop semantics,
preset behavior, AI worker lifecycle, and simulated keyboard/mouse UI runs.

## Project layout

```
voicebox.py       the app: audio engine, effects, soundboard, UI (pygame)
rvc_worker.py     headless RVC realtime worker (runs on RVC's runtime python)
VoiceBox.bat      run-from-source launcher (auto-installs deps)
controls.json     input bindings (delete to restore defaults)
sounds/           your soundboard clips (not committed)
assets/fonts/     bundled UI fonts (Space Grotesk, JetBrains Mono)
design/           UI skin spec (tokens + mockup) the interface is ported from
setup/            VoiceBoxSetup.exe bootstrapper source + build script
tests/            regression suites (run_all.py)
```
