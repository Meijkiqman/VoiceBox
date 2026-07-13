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
  "Save preset" snapshots your current dialing as a user preset
  (`user_presets.json` - edit it to rename) that joins the cycle.
- **Settings persist** - every slider, toggle and the chosen preset are
  saved to `settings.json` (autosaved while running) and restored on the
  next launch.
- **Soundboard** - drop audio files (wav/flac/ogg/mp3) into `sounds/`; each
  becomes a grid tile ("Rescan sounds" picks up new files without a
  restart). Keys 1-9 fire the current hotkey page; Tab / `]` / `[` or the
  PAGE chip flip pages, so every clip is reachable from the keyboard. You
  always hear sounds locally; the "To mic" toggle decides whether Discord
  hears them too. Pause freezes everything mid-clip, Stop clears it.
- **AI voice** - point `RVC_DIR` in `voicebox.py` at an RVC-beta package with
  `weights/*.pth` models; VoiceBox runs `rvc_worker.py` on RVC's own bundled
  CUDA Python and pipes the converted voice into the cable. Pick the
  character from the menu; the soundboard keeps working on top.
- **Text to speech** - type a phrase in the panel under the soundboard and
  press Enter to save it; saved phrases live in a scrollable list (click to
  speak into the mic, `x` to delete) and persist in `tts_phrases.json`.
  With "TTS voice FX" on, the speech goes through the same pitch/effect
  chain as your voice - and through the AI voice while the worker is live;
  toggle it off for clean TTS. Speech is rendered once with the Windows
  voice (SAPI) and cached in `tts_cache/`.
- **Global hotkeys** - the soundboard works while a game or Discord has
  focus: Ctrl+Alt+1-9 fire clips, Ctrl+Alt+0 stops everything, Ctrl+Alt+P
  cycles presets, Ctrl+Alt+M toggles mute. Remappable in `controls.json`
  (`"global"` section); toggleable from the SYSTEM menu.
- **Mic mute + push-to-talk** - mute from the menu, the `M` key, or the
  global hotkey; the header shows MUTED while the soundboard and TTS keep
  working. Bind a `"ptt"` key in `controls.json` for hold-to-talk (the mic
  stays muted except while the key is held).
- **Noise gate** - replaces the Discord suppression you had to turn off:
  gates room hiss ahead of the effect chain (grit/reverb amplify it
  otherwise), with hold + slow release so word tails survive. Threshold
  adjustable from the menu.
- **Record output** - one menu toggle writes the processed mix (voice +
  effects + soundboard + TTS) to `recordings/*.wav`, handy for testing
  presets or keeping funny moments.
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

Devices are picked from the DEVICES menu rows (persisted in
`settings.json`); by default VoiceBox uses the system mic and auto-finds
the cable. `python voicebox.py --list` prints audio devices if the
auto-match fails. An RVC package folder can be set with a `"rvc_dir"`
string in `settings.json` (no source edit needed).

## Installer

`setup/Build-Setup.bat` builds `setup/dist/VoiceBoxSetup.exe` (PyInstaller):
a one-click bootstrapper that checks the system, silently installs Python and
the libraries if missing, downloads + silently installs VB-CABLE (one UAC
prompt), and installs VoiceBox with a Desktop shortcut. App files travel
inside the exe; `--url <zip>` downloads them instead. `--check` reports
without changing anything.

## Tests

`python tests/run_all.py` - six suites, 100+ checks, headless (no audio
hardware or window needed): DSP effects math, routing/pause/stop semantics,
preset behavior, AI worker lifecycle, TTS phrases, and simulated
keyboard/mouse UI runs.

## Project layout

```
voicebox.py       the app: audio engine, effects, soundboard, UI (pygame)
rvc_worker.py     headless RVC realtime worker (runs on RVC's runtime python)
VoiceBox.bat      run-from-source launcher (auto-installs deps)
controls.json     input bindings (delete to restore defaults)
settings.json     your dialed-in values, restored on launch (not committed)
user_presets.json your saved presets (not committed)
sounds/           your soundboard clips (not committed)
tts_phrases.json  your saved TTS phrases (not committed)
tts_cache/        rendered TTS audio, rebuilt on demand (not committed)
assets/fonts/     bundled UI fonts (Space Grotesk, JetBrains Mono)
design/           UI skin spec (tokens + mockup) the interface is ported from
setup/            VoiceBoxSetup.exe bootstrapper source + build script
tests/            regression suites (run_all.py)
```
