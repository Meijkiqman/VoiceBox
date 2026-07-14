"""Paths, audio constants, presets and default bindings - the edit-me module."""
from pathlib import Path

BASE_DIR      = Path(__file__).resolve().parent.parent   # the checkout root
SOUNDS_DIR    = BASE_DIR / "sounds"       # anchored: works from any cwd
CONTROLS_PATH = BASE_DIR / "controls.json"
SETTINGS_PATH = BASE_DIR / "settings.json"       # dialed-in values, restored on launch
USER_PRESETS_PATH = BASE_DIR / "user_presets.json"  # "Save preset" snapshots
RECORDINGS_DIR = BASE_DIR / "recordings"         # "Record output" wav files

TTS_PHRASES_PATH = BASE_DIR / "tts_phrases.json"  # saved TTS phrases
TTS_CACHE_DIR    = BASE_DIR / "tts_cache"         # rendered wavs, keyed by text hash
TTS_MAX_CHARS    = 200                            # per-phrase length cap

SAMPLERATE = 48000        # VB-CABLE runs at 48k by default
BLOCKSIZE  = 512          # smaller = lower latency, larger = safer. 256-1024 typical
CHANNELS   = 1            # mono processing path

# Device selection fallbacks. The DEVICES menu rows are the normal way to pick
# devices (persisted in settings.json); these constants only apply when nothing
# is selected there. Substrings are matched against device names
# (case-insensitive); use --list to see names, or set an int to force an index.
INPUT_DEVICE_MATCH   = None            # None = system default mic, or e.g. "Microphone"
OUTPUT_DEVICE_MATCH  = "CABLE Input"   # the virtual cable's INPUT side

WINDOW_SIZE = (960, 660)   # initial + minimum size; the window is resizable
MAX_CLIPS   = 64           # how many files from ./sounds get indexed

# AI voice (RVC) integration. RVC_DIR holds a trimmed RVC-beta package
# (must contain runtime\python.exe, weights\*.pth, hubert_base.pt, rmvpe.pt);
# ours lives in the rvc\ folder next to this file, so VoiceBox is
# self-contained. A "rvc_dir" string in settings.json overrides this
# constant. The AI rows only appear in the menu when the folder and at
# least one voice model exist, so machines without RVC are unaffected.
RVC_DIR = BASE_DIR / "rvc"

# Voice presets, cycled with the Preset menu row. "drive" is the grit/growl
# soft-clip amount; "robot" is the robot/vocoder mix; "reverb"/"echo"/"doubler"
# are wet mixes; "bass" is the low-shelf boost (all 0..1). "radio" is the
# walkie-talkie band-pass. Missing keys default to off.
PRESETS = [
    ("Normal",        {"semitones": 0,  "robot": 0.0, "drive": 0.0}),
    ("Chipmunk",      {"semitones": 7,  "robot": 0.0, "drive": 0.0}),
    ("Monster",       {"semitones": -6, "robot": 0.0, "drive": 0.35, "reverb": 0.3}),
    ("Robot",         {"semitones": 0,  "robot": 1.0, "drive": 0.0}),
    # (The Voicemod-recipe variant - doubler 100 / robot 49 / reverb 26 /
    # pitch -4 / bass 100 - can still be dialed in manually via the rows.)
    ("Space Marine",  {"semitones": -5, "robot": 0.0, "drive": 0.85, "reverb": 0.4}),
    ("Ork",           {"semitones": -3, "robot": 0.0, "drive": 1.0}),
    ("Ghost",         {"semitones": 2,  "robot": 0.0, "drive": 0.0,
                       "reverb": 0.85, "echo": 0.4}),
    ("Walkie-Talkie", {"semitones": 0,  "robot": 0.0, "drive": 0.25, "radio": True}),
]

DEFAULT_CONTROLS = {
    "keyboard": {
        "up":         ["up", "w"],
        "down":       ["down", "s"],
        "left":       ["left", "a"],
        "right":      ["right", "d"],
        "select":     ["return", "space"],
        "back":       ["escape"],
        "stop_clips": ["0", "backspace"],
        "mute":       ["m"],
        "page_next":  ["tab", "]"],
        "page_prev":  ["["],
        "clips":      ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
    },
    "gamepad": {
        "select":         [0],
        "back":           [1],
        "stop_clips":     [3],
        "axis_threshold": 0.5,
        "nav_cooldown":   0.22,
    },
    # System-wide hotkeys (optional `keyboard` package): the soundboard keeps
    # working while a game or Discord has focus. Names are keyboard-package
    # combos ("ctrl+alt+1"). Set "enabled" false or empty a binding to skip it.
    "global": {
        "enabled":     True,
        "clips":       ["ctrl+alt+1", "ctrl+alt+2", "ctrl+alt+3",
                        "ctrl+alt+4", "ctrl+alt+5", "ctrl+alt+6",
                        "ctrl+alt+7", "ctrl+alt+8", "ctrl+alt+9"],
        "stop_clips":  "ctrl+alt+0",
        "next_preset": "ctrl+alt+p",
        "mute":        "ctrl+alt+m",
        # push-to-talk: a single key name; while held the mic is live, on
        # release it mutes. Empty = off (mute stays a manual toggle).
        "ptt":         "",
    },
}

