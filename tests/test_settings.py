"""Settings persistence (settings.json) and user presets (user_presets.json)."""
import json
import tempfile
import threading
import time
from pathlib import Path

from _common import check, finish

import numpy as np
import voicebox

voicebox.soundboard.load_clips = lambda: ([], [])
tmpdir = Path(tempfile.mkdtemp())

# --------------------------------------------------------- snapshot/restore
state = voicebox.State()
state.set_pitch(-5)
with state.lock:
    state.drive = 0.85
    state.reverb = 0.4
    state.radio = True
    state.gate_on = True
    state.gate_db = -50.0
    state.clips_to_mic = False
snap = state.snapshot()
check("snapshot captures dialing",
      snap["semitones"] == -5 and snap["drive"] == 0.85
      and snap["radio"] is True and snap["gate_db"] == -50.0
      and snap["clips_to_mic"] is False)

fresh = voicebox.State()
fresh.restore(snap)
check("restore round-trips",
      fresh.semitones == -5 and fresh.drive == 0.85 and fresh.radio is True
      and fresh.gate_on is True and fresh.clips_to_mic is False)
ev = fresh.events.get_nowait()
check("restored pitch reaches the shifter via event", ev == ("pitch", -5))

# ------------------------------------------------------ hostile settings.json
hostile = voicebox.State()
hostile.restore({"semitones": "loud", "drive": 99, "reverb": -3,
                 "radio": 1, "gate_db": None, "preset_idx": "x",
                 "voice_gain": [1], "unknown_key": True})
check("garbage types fall back to defaults",
      hostile.semitones == 0.0 and hostile.gate_db == -40.0
      and hostile.voice_gain == 1.0)
check("out-of-range values clamp",
      hostile.drive == 1.0 and hostile.reverb == 0.0)
check("truthy int becomes bool", hostile.radio is True)
check("bad preset index ignored", hostile.preset_idx == 0)
hostile.restore(None)
hostile.restore([1, 2])
check("non-dict restore is a no-op", hostile.drive == 1.0)

# ----------------------------------------------------------- file round-trip
spath = tmpdir / "settings.json"
voicebox.save_settings(snap, spath)
check("settings file written", spath.is_file())
loaded = voicebox.load_settings(spath)
check("settings load round-trips", loaded == json.loads(json.dumps(snap)))
check("missing file -> empty dict",
      voicebox.load_settings(tmpdir / "nope.json") == {})
(tmpdir / "broken.json").write_text("{not json", encoding="utf-8")
check("broken file -> empty dict",
      voicebox.load_settings(tmpdir / "broken.json") == {})

# ------------------------------------------------------------- user presets
ppath = tmpdir / "user_presets.json"
ps = voicebox.State()
ps.user_presets_path = ppath
ps.set_pitch(3)
with ps.lock:
    ps.drive = 0.6
    ps.radio = True
name = ps.save_user_preset()
check("saved preset gets a name", name == "My preset 1")
check("saved preset is selected",
      ps.preset_idx == len(voicebox.PRESETS)
      and ps.preset_label() == "My preset 1")
check("preset file written", ppath.is_file())

reload_ = voicebox.load_user_presets(ppath)
check("user presets reload from disk",
      len(reload_) == 1 and reload_[0][0] == "My preset 1"
      and reload_[0][1]["drive"] == 0.6 and reload_[0][1]["radio"] is True)

other = voicebox.State()
other.user_presets = reload_
other.apply_preset(len(voicebox.PRESETS))
check("user preset applies like a built-in",
      other.drive == 0.6 and other.radio is True and other.semitones == 3)
check("applied user preset labels itself",
      other.preset_label() == "My preset 1")

name2 = ps.save_user_preset()
check("second save numbers up", name2 == "My preset 2")
check("preset cycle includes user presets",
      len(ps.presets_all()) == len(voicebox.PRESETS) + 2)

# malformed user preset entries are dropped / applied defensively
ppath.write_text(json.dumps([
    {"name": "ok", "params": {"drive": 0.1}},
    {"name": 5, "params": {}},                # bad name
    "junk",                                   # not a dict
    {"name": "no params"},                    # missing params
]), encoding="utf-8")
kept = voicebox.load_user_presets(ppath)
check("malformed preset entries dropped", [n for n, _ in kept] == ["ok"])
sparse = voicebox.State()
sparse.user_presets = kept
sparse.apply_preset(len(voicebox.PRESETS))    # params missing most keys
check("sparse preset params default to off",
      sparse.drive == 0.1 and sparse.robot == 0.0 and sparse.radio is False)

# restore clamps preset_idx into the combined list
wrap = voicebox.State()
wrap.user_presets = kept
wrap.restore({"preset_idx": 55})
check("preset_idx wraps into range",
      0 <= wrap.preset_idx < len(wrap.presets_all()))

# ---------------------------------------------------------------- autosave
astate = voicebox.State()
apath = tmpdir / "autosave.json"
aflag = threading.Event()
t = threading.Thread(target=voicebox.settings_autosave,
                     args=(astate, aflag, apath, 0.05), daemon=True)
t.start()
time.sleep(0.12)
check("autosave writes nothing while unchanged", not apath.is_file())
with astate.lock:
    astate.drive = 0.33
time.sleep(0.2)
aflag.set()
t.join(1.0)
check("autosave persists a change",
      apath.is_file() and voicebox.load_settings(apath)["drive"] == 0.33)

finish()
