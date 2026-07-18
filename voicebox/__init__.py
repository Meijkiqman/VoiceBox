"""
VoiceBox - real-time voice changer + soundboard for Discord (or anything).

HOW IT WORKS
------------
    real mic --> [this package: pitch shift + effects + soundboard] --> VB-CABLE Input
                                                                             |
                                                          Discord input = "CABLE Output"

SETUP (Windows)
---------------
1. Install VB-CABLE:  https://vb-audio.com/Cable/  (run installer as admin, reboot).
2. pip install -r requirements.txt   (numpy scipy sounddevice soundfile pygame keyboard)
3. Put some .wav files in the ./sounds folder.
4. Run:  python voicebox.py --list      (find your device names)
5. Pick devices from the DEVICES menu rows (or edit voicebox/config.py).
6. Run:  python voicebox.py
7. In Discord: Settings -> Voice & Video -> Input Device = "CABLE Output".

macOS: use BlackHole instead of VB-CABLE.  Linux: create a null sink with
`pactl load-module module-null-sink sink_name=voicebox` and point Discord at its monitor.

CONTROLS
--------
A pygame menu window handles all input (keyboard + controller + mouse).
Inputs only fire while the VoiceBox window has focus, so typing in Discord is
safe. Bindings live in controls.json next to this file - edit to remap,
delete to restore defaults.

Mouse: hover highlights a row, click activates it, click the < > arrows to
adjust a value, scroll wheel moves the selection (or scrolls the grid when
the pointer is over it).

SOUNDBOARD
----------
Every audio file in ./sounds (wav/flac/ogg/mp3, first 64, alphabetical) gets
a button in the grid on the right - dragging a file onto the window copies
it there and rescans. Clips are peak-normalized on load. Clicking a button always plays the sound
locally so you hear it yourself; while the "To mic" toggle is on it is also
mixed into the mic channel. Pause freezes all playing sounds (both paths),
Stop clears them. Keys 1-9 trigger the first nine sounds. The Hear toggle in
the same strip mirrors the processed mix to your speakers (self-listen);
while the AI voice is live the RVC worker mirrors its converted voice to the
speakers the same way, so you hear the AI voice too.

TEXT TO SPEECH
--------------
The panel below the soundboard speaks typed phrases into the mic channel.
Type in the box (Ctrl+V pastes the clipboard), press Enter (or ADD) to
save, or Shift+Enter to speak once without saving - phrases persist in
tts_phrases.json and are synthesized once into tts_cache/ (Windows: SAPI5
plus OneCore/natural voices via PowerShell; espeak / `say` elsewhere). Click a phrase to speak it, the x on
its row deletes it. With "TTS voice FX" on (menu row or the FX chip) the
speech runs through the same pitch/effect chain as your voice - and through
the AI voice while the RVC worker is live; off = clean TTS.

SPEECH TRANSLATOR
-----------------
Tap Ctrl+Alt+T (or the Translate row) and speak Norwegian or English;
when you stop talking the capture sends itself (a second tap just cuts
it short). It is transcribed (faster-whisper), translated offline
(Argos) to English, Spanish or Mandarin, and spoken into the mic channel
with a per-language OS voice - through the effect chain, or through the
AI voice while the RVC worker is live. Your real voice is held back from
the cable while it listens. Optional install:
pip install -r requirements-translator.txt.

INCOMING SPEECH
---------------
Captions what the others say. Route Discord's output to a second virtual
cable (CABLE-B) and pick its Output side as the "Listen device" row; the
chat passes through to your speakers while non-English utterances are
captioned in English at the bottom of the window ("Speak incoming" reads
them aloud). Language packs download on first encounter per language.

VOICE HARVEST + RETRAIN
-----------------------
With "Voice harvest" on, clean clips of your raw mic speech (never while
muted) are saved to rvc/dataset_self/ as RVC training data, up to a
60-minute cap. "Retrain AI voice" trains/refreshes the MyVoice model
from them in a separate console window (experimental - the training
pieces from the full RVC-beta0717 zip must be present; see
design/VOICE_TRAINING.md).

SCENES
------
A scene is the whole persona in one row: the effect dialing, the AI
character (and whether the worker runs), its pitch and FX routing, and the
TTS voice/rate. "Save scene" snapshots the current setup into scenes.json;
the Scene row (or Ctrl+Alt+S) applies one, starting or stopping the RVC
worker to match. In the Scene and Preset dropdowns, right-click (or F2)
renames an entry in place and the x on the focused row (or Del) deletes it
- scenes and your saved presets only; the built-ins stay.

EFFECTS & PRESETS
-----------------
Pitch, robot/vocoder mix, helmet doubler, grit, reverb, echo, radio band-pass
and bass boost are individual menu rows. Numeric rows carry a draggable
slider in the middle; clicking the number itself opens a small box to type
an exact value (Enter commits, Esc cancels), and keyboard < > still steps.
The Preset row applies curated combinations (Space Marine, Ghost, ...) which
can be tweaked freely afterwards - the row shows "Custom" once any value
diverges from the applied preset. Pressing the Preset or AI character row
opens an alphabetical dropdown for direct picking. The "AI voice FX" row
routes the converted AI voice through this same chain (and the HEAR
mirror) instead of letting the worker feed the cable dry. The window
itself is resizable (drag edges, Aero snap); the soundboard pane absorbs
the extra space.

Defaults:  arrows/WASD or d-pad/left stick = navigate,  Enter/Space or A =
select,  left/right adjusts values,  1-9 = play clip,  0/Backspace or Y =
stop clips,  Esc or B = quit.
"""

import sys
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

from . import (aivoice, app, audio, config, controls, cues, dsp, harvester,
               listener, scenes, soundboard, state, trainer, translator,
               tts, ui)
from .config import *          # noqa: F401,F403 - flat API kept for tests
from .dsp import *             # noqa: F401,F403
from .soundboard import *      # noqa: F401,F403
from .state import *           # noqa: F401,F403
from .audio import *           # noqa: F401,F403
from .aivoice import *         # noqa: F401,F403
from .cues import *            # noqa: F401,F403
from .scenes import *          # noqa: F401,F403
from .tts import *             # noqa: F401,F403
from .translator import *      # noqa: F401,F403
from .listener import *        # noqa: F401,F403
from .harvester import *       # noqa: F401,F403
from .trainer import *         # noqa: F401,F403
from .controls import *        # noqa: F401,F403
from .ui import *              # noqa: F401,F403
from .app import main          # noqa: F401
