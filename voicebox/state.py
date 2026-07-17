"""Shared runtime state and its persistence (settings.json, user presets)."""
import json
import queue
import threading
from pathlib import Path

from . import soundboard     # module attr: tests swap soundboard.load_clips
from .config import (PRESETS, SAMPLERATE, SETTINGS_PATH, USER_PRESETS_PATH)
from .dsp import (BassBoost, Doubler, Echo, NoiseGate, Radio, Reverb,
                  StreamingPitchShifter)

# What survives a restart (settings.json). Ranges clamp hand-edited files.
PERSIST_FIELDS = {
    # name: (lo, hi) for numbers, bool for toggles
    "semitones":    (-12.0, 12.0),
    "robot":        (0.0, 1.0),
    "drive":        (0.0, 1.0),
    "reverb":       (0.0, 1.0),
    "echo":         (0.0, 1.0),
    "doubler":      (0.0, 1.0),
    "bass":         (0.0, 1.0),
    "voice_gain":   (0.0, 1.5),
    "clip_gain":    (0.0, 1.5),
    "tts_gain":     (0.0, 1.5),
    "tts_rate":     (-10.0, 10.0),
    "gate_db":      (-70.0, -10.0),
    "ai_pitch":     (-24.0, 24.0),
    "radio":        bool,
    "gate_on":      bool,
    "tts_fx":       bool,
    "ai_fx":        bool,
    "cues_on":      bool,
    "clips_to_mic": bool,
    "harvest_on":   bool,
    "listen_on":    bool,
    "listen_speak": bool,
    "listen_pass":  bool,
    # str = device/path name or None (None -> the defaults at the top of
    # this file). Persisting names, not indexes: indexes shift across boots.
    "input_device":  str,
    "output_device": str,
    "rvc_dir":       str,
    "tts_voice":     str,
    "trans_source":  str,      # translator: "auto" | "no" | "en" (None = auto)
    "trans_target":  str,      # translator: "en" | "es" | "zh" (None = en)
    "trans_model":   str,      # faster-whisper size (None = config default)
    "trans_voice_en": str,     # per-target TTS voice (None = auto-pick)
    "trans_voice_es": str,
    "trans_voice_zh": str,
    "listen_device": str,      # incoming-speech capture device (None = auto)
}


def load_settings(path=SETTINGS_PATH):
    """settings.json -> dict; missing/broken file -> {} (defaults win)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_settings(data, path=SETTINGS_PATH):
    try:
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass                                   # read-only install dir: run on


def load_user_presets(path=USER_PRESETS_PATH):
    """user_presets.json -> [(name, params)] in PRESETS shape; bad entries
    are dropped so a hand-edited file can't break startup."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    out = []
    if isinstance(data, list):
        for entry in data:
            if (isinstance(entry, dict) and isinstance(entry.get("name"), str)
                    and isinstance(entry.get("params"), dict)):
                out.append((entry["name"][:40], entry["params"]))
    return out


def save_user_presets(presets, path=USER_PRESETS_PATH):
    try:
        Path(path).write_text(
            json.dumps([{"name": n, "params": p} for n, p in presets],
                       indent=2),
            encoding="utf-8")
    except OSError:
        pass


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.semitones = 0.0          # UI-side display value; the shifter follows via events
        self.mic_muted = False        # silences the mic (TTS + soundboard keep working)
        self.robot = 0.0              # robot/vocoder mix, 0..1 (0 = off)
        self.robot_phase = 0.0
        self.voice_gain = 1.0
        self.clip_gain = 0.9
        self.drive = 0.0              # grit/growl soft-clip amount, 0..1
        self.reverb = 0.0             # reverb wet mix, 0..1
        self.echo = 0.0               # echo wet mix, 0..1
        self.doubler = 0.0            # helmet doubler wet mix, 0..1
        self.bass = 0.0               # low-shelf boost, 0..1
        self.radio = False            # walkie-talkie band-pass
        self.gate_on = False          # noise gate ahead of the chain
        self.gate_db = -40.0          # gate threshold (dBFS block peak)
        self.preset_idx = 0
        self.clips_to_mic = True      # soundboard also feeds the mic channel
        self.clips_paused = False     # freezes all playing sounds (both paths)
        self.tts_fx = True            # TTS through the voice chain / AI (off = clean)
        self.tts_gain = 1.0           # TTS level on the mic channel
        self.tts_voice = None         # engine voice name; None = OS default
        self.tts_rate = 0.0           # SAPI -10..10 speaking rate
        self.trans_source = None      # translator input lang; None = auto
        self.trans_target = None      # translator output lang; None = en
        self.trans_model = None       # whisper size; None = config default
        self.trans_voice_en = None    # per-target TTS voice; None = auto-pick
        self.trans_voice_es = None
        self.trans_voice_zh = None
        self.trans_hold = False       # capturing: outgoing voice path silent
        self.trans_tap = None         # Queue while capturing (raw mic blocks)
        self.harvest_on = False       # collect own-voice training clips
        self.harvest_q = None         # Queue while harvesting (raw mic blocks)
        self.listen_on = False        # incoming speech translator
        self.listen_device = None     # capture device name; None = auto (CABLE-B)
        self.listen_speak = False     # also speak captions on the speakers
        self.listen_pass = True       # pass captured audio through to speakers
        self.ai_mute = False          # AI worker owns the voice; mute ours
        self.ai_fx = False            # AI voice through the effect chain
        self.ai_pitch = 0.0           # transpose INTO the model, semitones
        self.ai_pitches = {}          # per-character memory: {stem: semitones}
        self.ai_feed = None           # AiFeed bridge while RVC is available
        self.cues_on = True           # audible blips: AI ready/died, mute
        self.cues = None              # Cues instance once the app is wired
        self.input_device = None      # device name; None = INPUT_DEVICE_MATCH
        self.output_device = None     # device name; None = OUTPUT_DEVICE_MATCH
        self.rvc_dir = None           # RVC package path; None = RVC_DIR
        self.shifter = StreamingPitchShifter(SAMPLERATE, 0.0)  # audio thread only
        self.reverb_fx = Reverb()     # effect state: audio thread only
        self.echo_fx = Echo()
        self.radio_fx = Radio()
        self.doubler_fx = Doubler()
        self.bass_fx = BassBoost()
        self.gate_fx = NoiseGate()
        self.user_presets_path = USER_PRESETS_PATH
        self.user_presets = load_user_presets(USER_PRESETS_PATH)
        self.clips, self.clip_names = soundboard.load_clips()
        self.clips_version = 0        # bumped on rescan; UI rebuilds its caches
        self.clip_page = 0            # hotkeys 1-9 fire page*9 .. page*9+8
        self.voices = []              # list of [samples, cursor]; audio thread only
        self.tts_voices = []          # list of [samples, cursor, fx]; audio thread only
        self.events = queue.Queue()   # UI thread -> audio thread
        self.status_msg = ""          # audio thread -> UI (underruns etc.)
        self.status_at = 0.0          # when status_msg was last set
        self.status_count = 0         # how many times a status fired (underrun tally)
        self.in_level = 0.0           # audio thread -> UI mic meter (block peak)
        self.monitor_q = None         # set to a Queue while self-listen is on
        self.record_q = None          # set to a Queue while recording

    def set_pitch(self, semis):
        # The shifter is owned by the audio thread; hand the change over via the
        # event queue so the callback never blocks on the UI holding the lock.
        with self.lock:
            self.semitones = max(-12, min(12, semis))
            self.events.put(("pitch", self.semitones))

    def nudge(self, attr, delta, lo=0.0, hi=1.5):
        with self.lock:
            setattr(self, attr, max(lo, min(hi, getattr(self, attr) + delta)))

    def set_val(self, attr, v, lo=0.0, hi=1.5):
        """Absolute set with clamp (sliders / typed values)."""
        with self.lock:
            setattr(self, attr, max(lo, min(hi, float(v))))

    def presets_all(self):
        """Built-in presets followed by the user's saved ones."""
        return PRESETS + self.user_presets

    def apply_preset(self, idx):
        presets = self.presets_all()
        _, p = presets[idx % len(presets)]
        with self.lock:
            self.preset_idx = idx % len(presets)
            # .get throughout: user_presets.json is hand-editable
            self.robot = float(p.get("robot", 0.0))
            self.drive = float(p.get("drive", 0.0))
            self.reverb = float(p.get("reverb", 0.0))
            self.echo = float(p.get("echo", 0.0))
            self.doubler = float(p.get("doubler", 0.0))
            self.bass = float(p.get("bass", 0.0))
            self.radio = bool(p.get("radio", False))
        self.set_pitch(p.get("semitones", 0))

    def preset_label(self):
        """Preset name while values still match it, else "Custom"."""
        presets = self.presets_all()
        name, p = presets[self.preset_idx % len(presets)]
        matches = (self.semitones == p.get("semitones", 0)
                   and self.robot == float(p.get("robot", 0.0))
                   and self.drive == p.get("drive", 0.0)
                   and self.reverb == p.get("reverb", 0.0)
                   and self.echo == p.get("echo", 0.0)
                   and self.doubler == p.get("doubler", 0.0)
                   and self.bass == p.get("bass", 0.0)
                   and self.radio == p.get("radio", False))
        return name if matches else "Custom"

    def save_user_preset(self):
        """Snapshot the current dialing as a named user preset, select it,
        and persist it. Returns the generated name."""
        with self.lock:
            params = {"semitones": self.semitones, "robot": self.robot,
                      "drive": self.drive, "reverb": self.reverb,
                      "echo": self.echo, "doubler": self.doubler,
                      "bass": self.bass, "radio": self.radio}
            taken = {n for n, _ in PRESETS} | {n for n, _ in self.user_presets}
            i = len(self.user_presets) + 1
            while f"My preset {i}" in taken:
                i += 1
            name = f"My preset {i}"
            self.user_presets.append((name, params))
            self.preset_idx = len(PRESETS) + len(self.user_presets) - 1
        save_user_presets(self.user_presets, self.user_presets_path)
        return name

    def delete_user_preset(self, i):
        """Remove user preset i (index into user_presets) and persist. The
        dialed effect values are never touched; deleting the selected preset
        just re-anchors the row (it then reads Custom, which is honest)."""
        with self.lock:
            if not (0 <= i < len(self.user_presets)):
                return False
            del self.user_presets[i]
            n = len(PRESETS)
            if self.preset_idx > n + i:
                self.preset_idx -= 1   # a later entry: its index shifted down
            elif self.preset_idx == n + i:
                self.preset_idx = 0
        save_user_presets(self.user_presets, self.user_presets_path)
        return True

    def rename_user_preset(self, i, name):
        """Rename user preset i and persist. Returns the stored name
        ('' = rejected: blank after cleanup, or i out of range)."""
        name = " ".join(str(name).split())[:40]
        with self.lock:
            if not name or not (0 <= i < len(self.user_presets)):
                return ""
            self.user_presets[i] = (name, self.user_presets[i][1])
        save_user_presets(self.user_presets, self.user_presets_path)
        return name

    def snapshot(self):
        """Persisted values -> plain dict (see PERSIST_FIELDS)."""
        with self.lock:
            data = {k: getattr(self, k) for k in PERSIST_FIELDS}
            data["preset_idx"] = self.preset_idx
            data["ai_pitches"] = dict(self.ai_pitches)
        return data

    def restore(self, data):
        """Apply a settings dict; wrong types/ranges fall back to defaults
        so a hand-edited settings.json can't break startup."""
        if not isinstance(data, dict):
            return
        semis = None
        with self.lock:
            for key, spec in PERSIST_FIELDS.items():
                if key not in data:
                    continue
                v = data[key]
                if spec is bool:
                    setattr(self, key, bool(v))
                    continue
                if spec is str:
                    setattr(self, key, v if isinstance(v, str) and v else None)
                    continue
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                lo, hi = spec
                v = max(lo, min(hi, v))
                if key == "semitones":
                    semis = v          # via set_pitch below (feeds the shifter)
                else:
                    setattr(self, key, v)
            try:
                idx = int(data.get("preset_idx", self.preset_idx))
            except (TypeError, ValueError):
                idx = self.preset_idx
            self.preset_idx = idx % len(self.presets_all())
            pitches = data.get("ai_pitches")
            if isinstance(pitches, dict):      # per-character AI pitch memory
                self.ai_pitches = {
                    str(k): int(max(-24, min(24, v)))
                    for k, v in pitches.items()
                    if isinstance(k, str) and isinstance(v, (int, float))}
        if semis is not None:
            self.set_pitch(semis)




def settings_autosave(state, stop_flag, path=SETTINGS_PATH, interval=2.0):
    """Persist changed settings every couple of seconds (daemon thread), so
    a crash or power cut loses at most one interval of dialing."""
    last = state.snapshot()
    while not stop_flag.wait(interval):
        snap = state.snapshot()
        if snap != last:
            save_settings(snap, path)
            last = snap


