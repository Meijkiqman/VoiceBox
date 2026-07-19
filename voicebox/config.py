"""Paths, audio constants, presets and default bindings - the edit-me module."""
from pathlib import Path

BASE_DIR      = Path(__file__).resolve().parent.parent   # the checkout root
SOUNDS_DIR    = BASE_DIR / "sounds"       # anchored: works from any cwd
CONTROLS_PATH = BASE_DIR / "controls.json"
SETTINGS_PATH = BASE_DIR / "settings.json"       # dialed-in values, restored on launch
USER_PRESETS_PATH = BASE_DIR / "user_presets.json"  # "Save preset" snapshots
SCENES_PATH   = BASE_DIR / "scenes.json"         # "Save scene" full-setup snapshots
RECORDINGS_DIR = BASE_DIR / "recordings"         # "Record output" wav files

# Optional neural TTS (Piper): run setup/Get-PiperVoices.bat to install the
# engine + six realistic English voices (3 male / 3 female, US + British)
# into this folder. Any extra .onnx+.onnx.json pair dropped into
# piper/voices/ appears in every voice picker as "Piper: ...".
PIPER_DIR = BASE_DIR / "piper"

TTS_PHRASES_PATH = BASE_DIR / "tts_phrases.json"  # saved TTS phrases
TTS_CACHE_DIR    = BASE_DIR / "tts_cache"         # rendered wavs, keyed by text hash
TTS_MAX_CHARS    = 200                            # per-phrase length cap

# Speech translator (optional deps: faster-whisper + argostranslate).
# Tap the hotkey / row, speak, tap again: the utterance is transcribed,
# translated and spoken into the cable in the target language's TTS voice
# (through the effects or the AI voice, like a typed phrase).
TRANS_SOURCES  = [("auto", "auto"), ("no", "Norwegian"), ("en", "English")]
TRANS_TARGETS  = [("en", "English"), ("es", "Spanish"), ("zh", "Mandarin")]
TRANS_MODEL    = "small"    # faster-whisper size; override via settings.json
                            # "trans_model" ("base" = lighter, "medium" = better)
TRANS_MAX_S    = 30.0       # capture cap per utterance, seconds
TRANS_MIN_S    = 0.4        # discard blips shorter than this
# Utterance end-pointing is adaptive: the watcher tracks the mic's noise
# floor (fast fall, slow rise) and calls speech a rise above it - a fixed
# threshold either misses quiet mics or never releases on noisy ones.
TRANS_VAD_START = 12.0      # dB above the noise floor that starts speech
TRANS_VAD_KEEP  = 6.0       # dB above the floor that keeps it going
TRANS_FLOOR_DB  = -60.0     # initial floor guess (falls to reality fast)
TRANS_FLOOR_MAX = -30.0     # the floor may never rise above this
TRANS_AUTO_STOP_S = 1.2     # trailing silence that auto-sends the capture
TRANS_IDLE_S   = 8.0        # give up when nothing was said at all

# Incoming speech translator: listens to what the OTHERS say (route Discord's
# output to a second virtual cable - e.g. CABLE-B - and pick its "Output" side
# as the Listen device), captions it in English in the window, optionally
# speaks it. Utterance segmentation knobs mirror the harvester's.
LISTEN_DEVICE_HINTS = ("cable-b output", "cable b output")  # auto-pick substrings
LISTEN_TARGET   = "en"      # incoming translations land in this language
LISTEN_THRESH_DB = -45.0    # Discord voice is compressed: gate lower than mic
LISTEN_PRE_S    = 0.3       # pre-roll kept before speech onset
LISTEN_HANG_S   = 0.6       # trailing silence that ends an utterance
LISTEN_MIN_S    = 0.8       # ignore blips shorter than this
LISTEN_MAX_S    = 15.0      # force a cut on monologues, seconds
LISTEN_CAPTION_S = 12.0     # how long a caption stays on screen
LISTEN_CAPTION_N = 3        # caption lines shown at once

# "Train new model": drop audio clips of the voice you want to clone into
# this folder - the row's file picker opens here, and training starts by
# itself once the clips are chosen.
TRAINING_DIR = BASE_DIR / "training"

# Voice harvester: collects clean speech clips from the real mic as training
# data for an RVC model of the user's own voice (rvc/dataset_self/, or
# voice_dataset/ when no RVC package is installed).
HARVEST_DIRNAME  = "dataset_self"   # under the RVC folder
HARVEST_THRESH_DB = -38.0   # block peak above this counts as speech
HARVEST_PRE_S    = 0.25     # pre-roll kept before speech onset
HARVEST_HANG_S   = 0.5      # trailing silence that ends a clip
HARVEST_MIN_S    = 2.0      # clips shorter than this are dropped
HARVEST_MAX_S    = 12.0     # clips are cut at this length
HARVEST_CAP_MIN  = 60.0     # stop collecting past this many minutes

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
CLIP_PEAK   = 0.9          # peak-normalize clips on load (0 = off); boosts
                           # are capped at 4x so quiet files don't turn to hiss

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
        "next_scene":  "ctrl+alt+s",
        "mute":        "ctrl+alt+m",
        "ai_voice":    "ctrl+alt+a",
        "translate":   "ctrl+alt+t",   # tap: start listening, tap again: speak
                                       # the translation into the mic
        # push-to-talk: a single key name; while held the mic is live, on
        # release it mutes. Empty = off (mute stays a manual toggle).
        "ptt":         "",
    },
}

