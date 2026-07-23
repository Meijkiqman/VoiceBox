"""Auto-trainer ("Train new model"): clip import, name hygiene, the flow's
refusals, the automatic launch, and the end-of-training watcher."""
import tempfile
import threading
import time
from pathlib import Path

from _common import check, finish

import numpy as np
import voicebox

voicebox.soundboard.load_clips = lambda: ([], [])

from voicebox.state import State
from voicebox.trainer import (CHUNK_S, MIN_CLIPS, Trainer, _safe_name,
                              import_clips)

import soundfile as sf
SF_REAL = hasattr(sf, "write")         # _common stubs soundfile when missing


def wait_for(cond, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


def tone(seconds, sr, channels=1, amp=0.3):
    t = np.arange(int(seconds * sr)) / sr
    x = (amp * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    return np.stack([x] * channels, axis=1) if channels > 1 else x


# ---- name hygiene -----------------------------------------------------------
check("safe name strips punctuation", _safe_name("My Hero!") == "My_Hero")
check("safe name survives dashes", _safe_name("voice-2") == "voice-2")
check("safe name empty-safe", _safe_name(None) == "" and _safe_name("  ") == "")

# ---- clip import ------------------------------------------------------------
if SF_REAL:
    src = Path(tempfile.mkdtemp())
    ds = Path(tempfile.mkdtemp()) / "dataset_x"
    sf.write(str(src / "long.wav"), tone(30.0, 22050, channels=2), 22050)
    sf.write(str(src / "blip.wav"), tone(0.5, 22050), 22050)
    sf.write(str(src / "silent.wav"), np.zeros(22050 * 5, np.float32), 22050)
    (src / "bad.wav").write_text("not audio at all")

    kept, skipped, seconds = import_clips(sorted(src.glob("*.wav")), ds)
    # 30 s -> 12+12+6 pieces; blip too short, silent gated, bad unreadable
    check("chunks kept", kept == 3, f"kept={kept}")
    check("unreadable counted", skipped == 1, f"skipped={skipped}")
    check("duration summed", 29.0 < seconds < 31.0, f"{seconds:.1f}s")
    outs = sorted(ds.glob("*.wav"))
    check("files on disk", len(outs) == 3)
    info = sf.info(str(outs[0]))
    check("mono 16-bit at source rate",
          info.channels == 1 and info.subtype == "PCM_16"
          and info.samplerate == 22050)
    check("chunk capped", all(sf.info(str(o)).duration <= CHUNK_S + 0.1
                              for o in outs))

    # a re-import replaces the dataset instead of piling onto it
    kept2, _, _ = import_clips([src / "long.wav"], ds)
    check("reimport clears folder",
          kept2 == 3 and len(list(ds.glob("*.wav"))) == 3)
else:
    print("SKIP  clip import tests (soundfile not installed)")

# ---- refusals ---------------------------------------------------------------
tmp_rvc = Path(tempfile.mkdtemp())     # empty folder: no runtime inside
state = State()
state.rvc_dir = str(tmp_rvc)

t = Trainer(state, ai=None)
check("no runtime refuses",
      t.new_model() is False and "runtime" in state.status_msg)

(tmp_rvc / "runtime").mkdir()
(tmp_rvc / "runtime" / "python.exe").write_bytes(b"")
(tmp_rvc / "weights").mkdir()
check("fake runtime detected", t.available)


class _FakeAi:
    proc = object()                    # "worker is live"

    def rescan(self):
        self.rescanned = True


t_ai = Trainer(state, ai=_FakeAi())
check("AI live refuses",
      t_ai.new_model() is False and "GPU" in state.status_msg)

t.stage = "waiting for name..."
check("second press refuses while picking", t.new_model() is False)
t.stage = ""

# ---- the flow (dialogs stubbed out, Popen captured) -------------------------
calls = []


class _FakeProc:
    def __init__(self, cmd):
        self.cmd = cmd

    def poll(self):
        return None                    # still running

    def wait(self):
        time.sleep(60)                 # watcher thread parks until exit


voicebox.trainer.subprocess.Popen = \
    lambda cmd, **kw: calls.append(cmd) or _FakeProc(cmd)

# cancel at the name prompt
t._ask_name = lambda: None
t._new_model_flow()
check("name cancel reported",
      "canceled" in state.status_msg and t.stage == "" and not calls)

# name collision with an existing model
(tmp_rvc / "weights" / "Hero.pth").write_bytes(b"")
t._ask_name = lambda: "Hero"
t._new_model_flow()
check("existing name refused", "already exists" in state.status_msg)

# cancel at the clip picker
t._ask_name = lambda: "Nova!"
t._ask_clips = lambda: []
t._new_model_flow()
check("clip cancel reported", "no clips chosen" in state.status_msg)

if SF_REAL:
    # too little audio -> report, no launch
    t._ask_clips = lambda: [src / "long.wav"]          # 3 pieces < MIN_CLIPS
    t._new_model_flow()
    check("thin dataset refused",
          "usable" in state.status_msg and not calls)

    # enough audio -> dataset built and training launched automatically
    big = src / "big.wav"
    sf.write(str(big), tone((MIN_CLIPS + 1) * CHUNK_S, 22050), 22050)
    t._ask_clips = lambda: [big]
    t._watch = lambda proc, name: None                 # park the watcher
    t.new_model()                                      # the real row entry
    check("flow finishes", wait_for(lambda: t.stage == "" and calls))
    check("dataset built", len(list(
        (tmp_rvc / "dataset_nova").glob("*.wav"))) >= MIN_CLIPS)
    cmd = calls[-1]
    check("trains sanitized name", "Nova" in cmd and "rvc_trainer.py"
          in " ".join(str(c) for c in cmd))
    check("trains the imported dataset",
          str(tmp_rvc / "dataset_nova") in cmd)
    check("proc registered + reported",
          t.proc is not None and "training 'Nova'" in state.status_msg)
    check("row shows training", "training" in t.new_label())
else:
    print("SKIP  flow launch tests (soundfile not installed)")

# ---- the watcher ------------------------------------------------------------


class _DoneProc:
    def __init__(self, code):
        self.code = code

    def wait(self):
        return self.code


ai = _FakeAi()
ai.rescanned = False
tw = Trainer(state, ai=ai)
(tmp_rvc / "weights" / "Nova.pth").write_bytes(b"")
p = _DoneProc(0)
tw.proc = p
tw._watch(p, "Nova")
check("finish rescans + reports",
      tw.proc is None and ai.rescanned and "done" in state.status_msg)

p = _DoneProc(1)
tw.proc = p
tw._watch(p, "Nova")
check("failure reported", "failed" in state.status_msg)

p = _DoneProc(0)
tw.proc = p
tw._watch(p, "Ghosty")                 # exit 0 but no weights written
check("no-model outcome reported", "no model" in state.status_msg)

# a stale watcher (superseded run) must not touch the live slot
live = _FakeProc(["x"])
tw.proc = live
tw._watch(_DoneProc(0), "Nova")
check("stale watcher ignored", tw.proc is live)

# ---- logging ----------------------------------------------------------------
import voicebox.trainer as tr_mod

log_dir = Path(tempfile.mkdtemp()) / "logs"
tr_mod.TRAIN_LOG_DIR = log_dir

tl = Trainer(state, ai=None)
tl._open_log("Train new model")
check("log file created", tl.log_path is not None
      and tl.log_path.parent == log_dir)
check("log dir made on demand", log_dir.is_dir())

tl._report("something went wrong")
body = tl.log_path.read_text(encoding="utf-8")
check("header logged", "=== Train new model ===" in body)
check("rvc dir logged", "RVC dir" in body)
check("status messages logged", "something went wrong" in body)
check("lines timestamped", any(l[:2].isdigit() and l[2] == ":"
                               for l in body.splitlines() if l.strip()))
check("log hint names the file", tl.log_path.name in tl.log_hint())

# a refusal is logged even though the flow stops immediately
tl.ai = _FakeAi()
tl.new_model()
check("refusal logged",
      "GPU" in tl.log_path.read_text(encoding="utf-8"))
tl.ai = None

# a broken Tk dialog reads as a failure, not as a cancel
tl._ask_name = lambda: (_ for _ in ()).throw(RuntimeError("no display"))
tl._new_model_flow()
check("dialog crash caught", "no display" in state.status_msg
      and "canceled" not in state.status_msg)
check("dialog crash logged + hinted",
      "no display" in tl.log_path.read_text(encoding="utf-8")
      and tl.log_path.name in state.status_msg)

# an unwritable log directory must not break the flow
tl2 = Trainer(state, ai=None)
tr_mod.TRAIN_LOG_DIR = Path("\x00bad")     # mkdir raises on every platform
tl2._open_log("Train new model")
check("bad log dir survives", tl2.log_path is None and tl2.log_hint() == "")
tl2._report("still reported")
check("status still works without a log", state.status_msg == "still reported")
tr_mod.TRAIN_LOG_DIR = log_dir

if SF_REAL:
    # the spawned trainer is told to append to the same log
    tl3 = Trainer(state, ai=None)
    tl3._open_log("Train new model")
    tl3._ask_name = lambda: "Logged"
    tl3._ask_clips = lambda: [big]
    tl3._watch = lambda proc, name: None
    calls.clear()
    tl3._new_model_flow()
    cmd = [str(c) for c in calls[-1]]
    check("--log passed to rvc_trainer",
          "--log" in cmd and str(tl3.log_path) in cmd)
    body = tl3.log_path.read_text(encoding="utf-8")
    check("chosen clips logged", str(big) in body)
    check("import result logged", "imported" in body)
    check("launch command logged", "launching:" in body)

# ---- the menu row -----------------------------------------------------------
from voicebox.ui import Menu

menu = Menu(state, threading.Event(), trainer=t)
labels = [it.label for it in menu.items]
check("menu row present", "Train new model" in labels)
row = menu.items[labels.index("Train new model")]
check("row wired to the flow", row.select == t.new_model
      and row.value_fn == t.new_label)

finish()
