# VoiceBox

Real-time voice changer + soundboard for Discord (or anything with a mic
input). Effects and presets on the left, hotkey soundboard on the right,
optional AI voice conversion (RVC) on the GPU in the background.

```
real mic --> [VoiceBox: pitch/effects/soundboard | or RVC AI voice] --> VB-CABLE
                                                                          |
                                                     Discord input = "CABLE Output"
```

## Install

1. Install [VB-CABLE](https://vb-audio.com/Cable/) (run as admin, reboot).
2. Double-click `VoiceBox.bat` - it installs what's missing and launches.
3. Discord -> Voice & Video -> Input Device = **CABLE Output** (disable
   Noise Suppression and automatic sensitivity there too).

Or by hand: `pip install -r requirements.txt`, then `python voicebox.py`.

## Features

- **Scenes** - the whole persona (effects, AI character, TTS voice) in one
  press or Ctrl+Alt+S; rename/delete right in the dropdown.
- **Presets + effects** - Space Marine, Ghost, Robot, ...; pitch, robot,
  doubler, grit, reverb, echo, radio and bass are manual rows with sliders.
  "Save preset" adds your own. Everything persists across restarts.
- **Soundboard** - drop audio files into `sounds/` (or onto the window);
  keys 1-9 fire the current page, clips are loudness-normalized on load.
- **AI voice** - RVC models (`rvc/weights/*.pth`) convert your voice live
  on an NVIDIA GPU; per-character pitch memory, optional routing through
  the effect chain.
- **Text to speech** - typed phrases speak into the mic, through the
  effects or the AI voice, with any installed OS voice.
- **Global hotkeys** - clips, stop, presets, scenes, mute and AI voice
  work while a game has focus (Ctrl+Alt+..., remappable in
  `controls.json`).
- **The rest** - mic mute + push-to-talk, noise gate, output recording to
  `recordings/`, audible state cues, HEAR self-listen, mic meter, gamepad
  navigation, resizable window.

## AI voice package

The RVC runtime + voice models (~12 GB) ship separately as a zip of the
`rvc/` folder; extract it next to `voicebox.py` so `rvc\runtime\python.exe`
exists and the AI rows appear on next launch. Without it, VoiceBox simply
runs without the AI voice.

## Installer

`setup/Build-Setup.bat` builds `VoiceBoxSetup.exe`: one click installs
Python, the libraries and VB-CABLE if missing, then VoiceBox with a
Desktop shortcut.

## Tests

`python tests/run_all.py` - headless, no audio hardware or window needed.
