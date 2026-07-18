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

- **Dashboard UI** - one self-contained card per feature (Translator,
  AI voice, My voice, Voice FX, TTS, Incoming, System) with its own
  settings and volume, plus the soundboard pinned on the right; cards
  collapse to their header (click it or press Enter, TAB cycles cards)
  and more window width means more columns.
- **Scenes** - the whole persona (effects, AI character, TTS voice) in one
  press or Ctrl+Alt+S from the strip under the header; rename/delete right
  in the dropdown.
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
- **Speech translator** - flip **Auto translate** (top of the
  Translator card) and just talk: each sentence you say is transcribed, translated (English,
  Spanish or Mandarin) and spoken into the mic in a native TTS voice - or
  in *your* RVC voice while the AI voice is live - while your raw voice
  stays off the cable. Ctrl+Alt+T does a one-shot capture instead.
  Offline after first-run downloads;
  needs `pip install -r requirements-translator.txt`.
- **Incoming translator** - captions non-English speech from the voice
  chat, in English, at the bottom of the window (optionally spoken
  aloud); English passes through uncaptioned. Route Discord's output to
  a second cable (CABLE-B) and flip "Incoming speech" on; any language
  Whisper detects gets translated, packs download on first encounter.
- **Voice harvest + retrain** - collects clean clips of your real voice
  while you play (`rvc/dataset_self/`; never while the mic is muted) and
  retrains your own RVC model from them on demand (experimental; see
  `design/VOICE_TRAINING.md`).
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

## Speech translator

`pip install -r requirements-translator.txt`, then flip the **TRANS**
row at the top of the Translator card and just talk: each sentence is
detected by its trailing silence (adaptive - it learns your mic's noise
floor, so background hum can't hold a capture open), translated, and spoken into the cable a moment
later - your raw voice never goes out while TRANS is on. Prefer it
per-sentence? Tap Ctrl+Alt+T (or the Translate row) for a one-shot
capture that sends itself the same way.
First use downloads the Whisper model and the Argos language packs (needs
internet once; everything runs locally after that). "Translate voice"
auto-picks an installed Windows voice for the target language - install
natural Spanish/Chinese voices under Settings -> Time & Language -> Speech.
With the AI voice running, translations are spoken through your RVC model
instead (note: the worker still carries your original voice while you
speak; pick a push-to-talk quiet moment, or run without the AI voice live).

**Incoming speech** (what the others say): install the A+B pack from
[VB-Audio](https://vb-audio.com/Cable/) so you have a second cable, set
Discord -> Voice & Video -> *Output* Device = **CABLE-B Input**, and pick
**CABLE-B Output** as the "Listen device" row (it auto-picks when found).
While "Incoming speech" is on, the chat is passed through to your speakers
untouched and non-English utterances are captioned in English at the
bottom of the window a few seconds later (English speech needs no caption
and gets none); "Speak incoming" reads the captions aloud too.
Uses the same translator install; language packs beyond
Norwegian/Spanish/Mandarin download automatically the first time someone
speaks them.

## Installer

`setup/Build-Setup.bat` builds `VoiceBoxSetup.exe`: one click installs
Python, the libraries and VB-CABLE if missing, then VoiceBox with a
Desktop shortcut.

## Tests

`python tests/run_all.py` - headless, no audio hardware or window needed.
