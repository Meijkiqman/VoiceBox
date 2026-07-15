"""Scenes: named snapshots of the whole setup (effects + AI + TTS), so a
persona is one press instead of a row-by-row re-dial."""
import json
import time
from pathlib import Path

from .config import SCENES_PATH

# The effect dialing a scene restores, with the ranges that clamp a
# hand-edited scenes.json (same shape as state.PERSIST_FIELDS).
FX_FIELDS = {
    "semitones": (-12.0, 12.0),
    "robot":     (0.0, 1.0),
    "drive":     (0.0, 1.0),
    "reverb":    (0.0, 1.0),
    "echo":      (0.0, 1.0),
    "doubler":   (0.0, 1.0),
    "bass":      (0.0, 1.0),
    "radio":     bool,
}


def load_scenes(path=SCENES_PATH):
    """scenes.json -> [(name, params)]; bad entries are dropped so a
    hand-edited file can't break startup."""
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


def save_scenes(scenes, path=SCENES_PATH):
    try:
        Path(path).write_text(
            json.dumps([{"name": n, "params": p} for n, p in scenes],
                       indent=2),
            encoding="utf-8")
    except OSError:
        pass                                   # read-only install dir: run on


class Scenes:
    """A scene is the whole persona: the effect dialing, which AI character
    is loaded (and whether the worker runs at all), its pitch and FX
    routing, and the TTS voice/rate. Applying one starts or stops the RVC
    worker to match, so going from your own voice to a character - or back -
    is a single press or hotkey."""

    def __init__(self, state, ai=None, tts=None, path=SCENES_PATH):
        self.state = state
        self.ai = ai
        self.tts = tts
        self.path = Path(path)
        self.scenes = load_scenes(path)
        self.sel = 0                   # last applied, for cycle()
        self.applied = None            # its name, shown on the menu row

    def names(self):
        return [n for n, _ in self.scenes]

    def _report(self, msg):
        self.state.status_msg = msg
        self.state.status_at = time.time()

    def snapshot(self):
        """Current setup -> a scene params dict."""
        s = self.state
        with s.lock:
            p = {k: getattr(s, k) for k in FX_FIELDS}
            p["preset_idx"] = s.preset_idx
            p["tts_voice"] = s.tts_voice
            p["tts_rate"] = s.tts_rate
            if self.ai is not None and self.ai.available:
                p["ai_voice"] = self.ai.voice_name()
                p["ai_on"] = self.ai.proc is not None
                p["ai_pitch"] = s.ai_pitch
                p["ai_fx"] = s.ai_fx
        return p

    def save(self, name=None):
        """Snapshot the setup as a new scene and persist it. Returns the
        name (generated when none is given)."""
        params = self.snapshot()
        taken = set(self.names())
        if not name:
            i = len(self.scenes) + 1
            while f"Scene {i}" in taken:
                i += 1
            name = f"Scene {i}"
        self.scenes.append((str(name)[:40], params))
        self.sel = len(self.scenes) - 1
        self.applied = self.scenes[self.sel][0]
        save_scenes(self.scenes, self.path)
        return self.applied

    def delete(self, i):
        if not (0 <= i < len(self.scenes)):
            return
        self.scenes.pop(i)
        if self.sel > i:
            self.sel -= 1              # keep cycle() anchored on the same scene
        self.sel = min(self.sel, max(0, len(self.scenes) - 1))
        save_scenes(self.scenes, self.path)

    def rename(self, i, name):
        """Rename scene i and persist. Returns the stored name ('' = rejected:
        blank after cleanup, or i out of range)."""
        name = " ".join(str(name).split())[:40]
        if not name or not (0 <= i < len(self.scenes)):
            return ""
        old, params = self.scenes[i]
        self.scenes[i] = (name, params)
        if self.applied == old and self.sel == i:
            self.applied = name        # the menu row keeps naming this persona
        save_scenes(self.scenes, self.path)
        return name

    def _apply_fx(self, p):
        """Effect dialing, clamped. Pitch goes through set_pitch so the
        audio thread picks it up on the event queue like every other path."""
        s = self.state
        semis = None
        with s.lock:
            for key, spec in FX_FIELDS.items():
                if key not in p:
                    continue
                v = p[key]
                if spec is bool:
                    setattr(s, key, bool(v))
                    continue
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                lo, hi = spec
                v = max(lo, min(hi, v))
                if key == "semitones":
                    semis = v
                else:
                    setattr(s, key, v)
            try:                       # keeps the Preset row honest: a scene
                idx = int(p.get("preset_idx", s.preset_idx))   # saved off an
            except (TypeError, ValueError):    # untouched preset still names
                idx = s.preset_idx             # it, a tweaked one says Custom
            s.preset_idx = idx % len(s.presets_all())
        if semis is not None:
            s.set_pitch(semis)

    def _apply_ai(self, p):
        """AI character, pitch, FX routing and worker on/off. Stops first
        when the scene wants the AI off, so switching character on the way
        out doesn't spin a worker up just to kill it."""
        ai = self.ai
        if ai is None or not ai.available:
            return
        s = self.state
        want_on = bool(p.get("ai_on", False))
        if not want_on and ai.proc is not None:
            ai.stop()
        name = p.get("ai_voice")
        if isinstance(name, str):
            for i, v in enumerate(ai.voices):
                if v.stem == name:
                    ai.select(i)       # also recalls that character's pitch
                    break
        pitch = p.get("ai_pitch")      # the scene's own pitch wins over it
        if isinstance(pitch, (int, float)) and not isinstance(pitch, bool):
            with s.lock:
                s.ai_pitch = float(max(-24, min(24, int(pitch))))
            ai.set_pitch(int(s.ai_pitch))
        if "ai_fx" in p:
            fx = bool(p["ai_fx"])
            if fx != s.ai_fx:
                with s.lock:
                    s.ai_fx = fx
                ai.set_fx(fx)
        if want_on and ai.proc is None:
            ai.start()

    def _apply_tts(self, p):
        """TTS voice/rate; rendered speech is dropped only if they moved."""
        s = self.state
        changed = False
        with s.lock:
            if "tts_voice" in p:
                v = p["tts_voice"]
                v = v if isinstance(v, str) and v else None
                if v != s.tts_voice:
                    s.tts_voice = v
                    changed = True
            if "tts_rate" in p:
                try:
                    r = float(max(-10.0, min(10.0, float(p["tts_rate"]))))
                except (TypeError, ValueError):
                    r = s.tts_rate
                if r != s.tts_rate:
                    s.tts_rate = r
                    changed = True
        if changed and self.tts is not None:
            self.tts.invalidate()

    def apply(self, i):
        if not (0 <= i < len(self.scenes)):
            return
        name, p = self.scenes[i]
        self._apply_fx(p)
        self._apply_ai(p)
        self._apply_tts(p)
        self.sel = i
        self.applied = name
        self._report(f"scene: {name}")

    def cycle(self, d=1):
        """Step to the next/previous scene and apply it (the hotkey path)."""
        if self.scenes:
            self.apply((self.sel + (1 if d >= 0 else -1)) % len(self.scenes))
