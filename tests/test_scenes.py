"""Scenes: snapshot/apply of the whole setup, persistence, AI + TTS wiring."""
import json
import tempfile
import threading
from pathlib import Path

from _common import check, finish

import numpy as np
import voicebox

voicebox.soundboard.load_clips = lambda: ([np.full(1000, 0.1, np.float32)],
                                          ["clip1"])


class FakeAI:
    """Stands in for AiVoice: records the calls a scene should make."""
    available = True

    def __init__(self, names=("ArthurMorgan", "kratos"), running=False):
        self.voices = [Path(f"{n}.pth") for n in names]
        self.sel = 0
        self.proc = "worker" if running else None
        self.started = self.stopped = 0
        self.pitches, self.fx = [], []

    def voice_name(self):
        return self.voices[self.sel].stem

    def select(self, i):
        self.sel = i

    def set_pitch(self, p):
        self.pitches.append(p)

    def set_fx(self, on):
        self.fx.append(on)

    def start(self):
        self.started += 1
        self.proc = "worker"

    def stop(self):
        self.stopped += 1
        self.proc = None


class FakeTTS:
    def __init__(self):
        self.invalidated = 0

    def invalidate(self):
        self.invalidated += 1


def fresh(ai=None, tts=None):
    state = voicebox.State()
    path = Path(tempfile.mkdtemp()) / "scenes.json"
    return state, voicebox.Scenes(state, ai, tts, path=path), path


# ----------------------------------------------------------- snapshot + save
state, sc, path = fresh(FakeAI(), FakeTTS())
check("starts with no scenes", sc.scenes == [] and sc.names() == [])
check("row shows a dash before anything is applied", sc.applied is None)

state.apply_preset([n for n, _ in voicebox.PRESETS].index("Ghost"))
with state.lock:
    state.drive = 0.75
    state.ai_pitch = 7.0
    state.ai_fx = True
    state.tts_voice = "Microsoft Jon"
    state.tts_rate = 3.0
name = sc.save()
check("save generates a name", name == "Scene 1" and sc.names() == ["Scene 1"])
p = sc.scenes[0][1]
check("scene captures the effect dialing",
      p["drive"] == 0.75 and p["echo"] == voicebox.PRESETS[
          [n for n, _ in voicebox.PRESETS].index("Ghost")][1]["echo"])
check("scene captures the AI setup",
      p["ai_voice"] == "ArthurMorgan" and p["ai_pitch"] == 7.0
      and p["ai_fx"] is True and p["ai_on"] is False)
check("scene captures the TTS setup",
      p["tts_voice"] == "Microsoft Jon" and p["tts_rate"] == 3.0)
check("scene persists on disk",
      json.loads(path.read_text(encoding="utf-8"))[0]["name"] == "Scene 1")
check("save marks the scene applied", sc.applied == "Scene 1")

sc.save("DnD ghost")
check("explicit names are kept", sc.names() == ["Scene 1", "DnD ghost"])
check("scenes reload from disk",
      voicebox.Scenes(state, None, None, path=path).names()
      == ["Scene 1", "DnD ghost"])

# --------------------------------------------------------------- apply round-trip
ai = FakeAI()
tts = FakeTTS()
state2, sc2, _ = fresh(ai, tts)
state2.apply_preset(0)                       # Normal
with state2.lock:
    state2.reverb = 0.9
    state2.radio = True
    state2.tts_rate = -4.0
state2.set_pitch(5)
while not state2.events.empty():
    state2.events.get_nowait()
sc2.save("loud")

with state2.lock:                            # dial everything away
    state2.reverb = 0.0
    state2.radio = False
    state2.tts_rate = 0.0
state2.set_pitch(0)
sc2.apply(0)
check("apply restores effect values",
      state2.reverb == 0.9 and state2.radio is True)
check("apply restores TTS rate + drops stale speech",
      state2.tts_rate == -4.0 and tts.invalidated >= 1)
pitch_evs = []
while not state2.events.empty():
    ev = state2.events.get_nowait()
    if isinstance(ev, tuple) and ev[0] == "pitch":
        pitch_evs.append(ev[1])
check("apply routes pitch through the event queue", pitch_evs[-1:] == [5.0],
      str(pitch_evs))
check("apply names the scene on the row + status line",
      sc2.applied == "loud" and state2.status_msg == "scene: loud")

# ------------------------------------------------------- AI character + worker
ai3 = FakeAI(running=False)
state3, sc3, _ = fresh(ai3, FakeTTS())
with state3.lock:
    state3.ai_pitch = 0.0
ai3.sel = 1                                  # kratos, worker off
sc3.save("kratos off")
ai3.sel = 0
with state3.lock:
    state3.ai_pitch = -9.0
sc3.apply(0)
check("apply selects the scene's character", ai3.sel == 1)
check("apply restores the character's pitch", state3.ai_pitch == 0.0)
check("scene with AI off leaves the worker down",
      ai3.proc is None and ai3.started == 0)

ai4 = FakeAI(running=True)                   # snapshot while the worker runs
state4, sc4, _ = fresh(ai4, FakeTTS())
sc4.save("arthur live")
ai4.stop()
ai4.stopped = 0
sc4.apply(0)
check("scene with AI on starts the worker",
      ai4.started == 1 and ai4.proc is not None)
check("apply pushes FX routing to the worker", ai4.fx in ([], [False]),
      str(ai4.fx))

# a scene that wants the AI off stops a running worker - without first
# spinning one up just to kill it
ai5 = FakeAI(running=False)
state5, sc5, _ = fresh(ai5, FakeTTS())
ai5.sel = 1
sc5.save("off scene")                        # ai_on False, character kratos
ai5.proc = "worker"                          # pretend it is running now
ai5.sel = 0
sc5.apply(0)
check("scene stops a running worker", ai5.proc is None and ai5.stopped == 1)
check("stopping first avoids a pointless restart", ai5.started == 0)

# --------------------------------------------------------------- cycle + delete
state6, sc6, _ = fresh(None, None)
sc6.save("a"); sc6.save("b"); sc6.save("c")
sc6.apply(0)
sc6.cycle(+1)
check("cycle steps to the next scene", sc6.applied == "b")
sc6.cycle(+1); sc6.cycle(+1)
check("cycle wraps around", sc6.applied == "a")
sc6.cycle(-1)
check("cycle steps backward", sc6.applied == "c")
sc6.delete(2)
check("delete removes + clamps the selection",
      sc6.names() == ["a", "b"] and sc6.sel <= 1)
sc6.delete(9)                                # out of range: no crash
check("out-of-range delete ignored", sc6.names() == ["a", "b"])
sc6.apply(9)
check("out-of-range apply ignored", sc6.applied == "c")

# deleting an earlier scene must not shift which scene cycle() sits on
state7, sc7, _ = fresh(None, None)
sc7.save("a"); sc7.save("b"); sc7.save("c")
sc7.apply(2)                                 # sitting on "c"
sc7.delete(0)
check("delete before the selection keeps its anchor",
      sc7.sel == 1 and sc7.names()[sc7.sel] == "c")

# ------------------------------------------------------------------- rename
state8, sc8, path8 = fresh(None, None)
sc8.save("a"); sc8.save("b")
sc8.apply(1)                                 # "b" is applied + selected
check("rename cleans and stores the name",
      sc8.rename(1, "  Ghost   mode ") == "Ghost mode"
      and sc8.names() == ["a", "Ghost mode"])
check("renaming the applied scene follows on the row",
      sc8.applied == "Ghost mode")
check("rename persists on disk",
      voicebox.Scenes(state8, None, None, path=path8).names()
      == ["a", "Ghost mode"])
check("blank rename rejected",
      sc8.rename(0, "   ") == "" and sc8.names()[0] == "a")
check("out-of-range rename rejected", sc8.rename(9, "x") == "")

empty_state, empty_sc, _ = fresh(None, None)
empty_sc.cycle(+1)                           # no scenes: must not divide by zero
check("cycle with no scenes is a no-op", empty_sc.applied is None)

# ------------------------------------------------------- hostile scenes.json
bad = Path(tempfile.mkdtemp()) / "bad.json"
bad.write_text(json.dumps([
    {"name": "ok", "params": {"reverb": 99, "semitones": "loud",
                              "radio": 1, "preset_idx": "x",
                              "ai_pitch": True, "tts_rate": None}},
    {"name": 5, "params": {}}, {"nope": 1}, "junk",
]), encoding="utf-8")
hstate = voicebox.State()
hsc = voicebox.Scenes(hstate, None, None, path=bad)
check("malformed scene entries dropped", hsc.names() == ["ok"])
hsc.apply(0)
check("hostile values clamp / fall back",
      hstate.reverb == 1.0 and hstate.semitones == 0.0
      and hstate.radio is True and hstate.preset_idx == 0
      and hstate.tts_rate == 0.0,          # None -> keeps the default
      f"reverb={hstate.reverb} semis={hstate.semitones} "
      f"radio={hstate.radio} preset={hstate.preset_idx} "
      f"rate={hstate.tts_rate}")
bad.write_text("{not json", encoding="utf-8")
check("broken scenes.json -> no scenes",
      voicebox.Scenes(hstate, None, None, path=bad).names() == [])

# --------------------------------------------------------------- menu rows
mstate = voicebox.State()
msc = voicebox.Scenes(mstate, None, None,
                      path=Path(tempfile.mkdtemp()) / "s.json")
menu = voicebox.Menu(mstate, threading.Event(), scenes=msc)
labels = [it.label for it in menu.items]
check("scene rows lead the menu", labels[:2] == ["Scene", "Save scene"])
check("no scene rows without a Scenes controller",
      "Scene" not in [it.label for it in
                      voicebox.Menu(mstate, threading.Event()).items])
srow = menu.items[0]
check("scene row shows a dash when none is applied", srow.value_fn() == "-")
next(it for it in menu.items if it.label == "Save scene").select()
check("Save scene row snapshots", msc.names() == ["Scene 1"])
check("save reports the name", "Scene 1" in mstate.status_msg)
check("scene row names the applied scene", srow.value_fn() == "Scene 1")

# ------------------------------------------- dropdown rename/delete (UI flow)
import time

import pygame

ui_state = voicebox.State()
ui_sc = voicebox.Scenes(ui_state, None, None,
                        path=Path(tempfile.mkdtemp()) / "ui.json")
ui_sc.save("alpha"); ui_sc.save("beta")
ui_sc.apply(0)
stop_flag = threading.Event()
snaps = []


def key(k):
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=k))
    pygame.event.post(pygame.event.Event(pygame.KEYUP, key=k))
    time.sleep(0.1)


def poke():
    time.sleep(0.7)
    key(pygame.K_RETURN)                     # Scene row -> dropdown opens
    key(pygame.K_F2)                         # rename the focused entry
    pygame.event.post(pygame.event.Event(pygame.TEXTINPUT, text="zulu"))
    time.sleep(0.1)
    key(pygame.K_RETURN)                     # commit the new name
    snaps.append(list(ui_sc.names()))
    key(pygame.K_ESCAPE)                     # close the picker
    key(pygame.K_RETURN)                     # reopen: still on the renamed one
    key(pygame.K_DELETE)                     # delete it
    snaps.append(list(ui_sc.names()))
    # a mouse-wheel tick also arrives as a legacy button-4/5 press on
    # Windows: it must scroll the picker, never close it
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=5,
                                         pos=(150, 138)))
    pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-1))
    time.sleep(0.1)
    key(pygame.K_RETURN)                     # picker still open: picks "beta"
    snaps.append(ui_sc.applied)
    key(pygame.K_ESCAPE)
    pygame.event.post(pygame.event.Event(pygame.QUIT))


threading.Thread(target=poke, daemon=True).start()
ui_error = []
try:
    voicebox.run_ui(ui_state, stop_flag, "dev", "", None, None, None, None,
                    None, None, None, ui_sc)
except Exception as e:
    ui_error.append(e)
check("UI dropdown rename/delete survives", not ui_error,
      repr(ui_error[0]) if ui_error else "")
check("F2 + typing renamed the scene in place",
      snaps and snaps[0] == ["zulu", "beta"], str(snaps))
check("Del removed the renamed scene",
      len(snaps) >= 2 and snaps[1] == ["beta"], str(snaps))
check("wheel-legacy button press does not close the picker",
      len(snaps) == 3 and snaps[2] == "beta", str(snaps))

finish()
