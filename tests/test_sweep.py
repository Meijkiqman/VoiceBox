"""Regression tests for the bug/performance sweep: hidden hit-rects in the
dashboard, atomic TTS cache renders, writer races, config hardening."""
import json
import math
import tempfile
import threading
import time
from pathlib import Path

from _common import check, finish, wait_ui

import numpy as np
import voicebox

voicebox.soundboard.load_clips = lambda: ([], [])
TONE = np.zeros(4800, dtype=np.float32)
tmpdir = Path(tempfile.mkdtemp())

# ------------------------------------------------- controls.json hardening
real_controls = voicebox.controls.CONTROLS_PATH
bad = tmpdir / "controls.json"
voicebox.controls.CONTROLS_PATH = bad
try:
    for label, text in (("a bare list", "[1, 2, 3]"), ("a bare string", '"x"'),
                        ("broken JSON", "{nope")):
        bad.write_text(text, encoding="utf-8")
        try:
            cfg = voicebox.load_controls()
            ok = cfg == json.loads(json.dumps(voicebox.DEFAULT_CONTROLS))
        except Exception:
            ok = False
        check(f"controls.json holding {label} falls back to defaults", ok)
finally:
    voicebox.controls.CONTROLS_PATH = real_controls

# ------------------------------------------- concurrent atomic JSON writes
target = tmpdir / "settings.json"
errors = []


def hammer(tag):
    try:
        for n in range(60):
            voicebox.persist.write_json_atomic(target, {"who": tag, "n": n})
    except Exception as e:             # pre-fix: FileNotFoundError on replace
        errors.append(e)


threads = [threading.Thread(target=hammer, args=(t,)) for t in "abcd"]
for t in threads:
    t.start()
for t in threads:
    t.join()
check("concurrent writers never collide on the temp file", not errors,
      repr(errors[:1]))
check("the surviving file is whole JSON",
      voicebox.persist.load_json(target, None) is not None)

# ------------------------------------------------ recorder writer ownership
rec = voicebox.Recorder(voicebox.State(), folder=tmpdir)


class SlowWav:
    def __init__(self):
        self.closed = False

    def write(self, block):
        pass

    def close(self):
        self.closed = True


import queue as _queue
old_file, old_stop = SlowWav(), threading.Event()
rec._stop = old_stop
old_thread = threading.Thread(target=rec._writer,
                              args=(old_file, _queue.Queue(), old_stop),
                              daemon=True)
old_thread.start()
old_stop.set()
rec._stop = threading.Event()          # a quick restart replaced the event
old_thread.join(timeout=2.0)
check("an old recorder writer still honors its own stop event",
      not old_thread.is_alive() and old_file.closed)

# ------------------------------------------------- atomic TTS cache renders
real_cache, real_synth, real_sf = (voicebox.tts.TTS_CACHE_DIR,
                                   voicebox.tts.synth_tts_wav, voicebox.tts.sf)
cache = tmpdir / "tts_cache"
voicebox.tts.TTS_CACHE_DIR = cache
seen_paths = []


class FakeSf:
    @staticmethod
    def read(path, dtype=None, always_2d=False):
        if Path(path).read_bytes() != b"whole":
            raise RuntimeError("truncated wav")
        return np.zeros((480, 1), dtype=np.float32), voicebox.SAMPLERATE


def fake_engine(text, wav_path, voice=None, rate=0):
    seen_paths.append(Path(wav_path))
    if text == "boom":
        Path(wav_path).write_bytes(b"ha")          # half-written, then dies
        raise RuntimeError("engine exploded")
    Path(wav_path).write_bytes(b"whole")


voicebox.tts.synth_tts_wav = fake_engine
voicebox.tts.sf = FakeSf
try:
    final = voicebox.tts_cache_path("hello")
    samples, wav = voicebox.tts.tts_synthesize("hello")
    check("engine renders under a private name, not the cache path",
          seen_paths and seen_paths[0] != final
          and seen_paths[0].parent == final.parent)
    check("finished render is published at the cache path",
          wav == final and final.is_file() and len(samples) == 480)
    check("no temp render left behind", list(cache.glob("*.part.wav")) == [])
    try:
        voicebox.tts.tts_synthesize("boom")
        raised = False
    except RuntimeError:
        raised = True
    check("a failed render raises and leaves no file at all",
          raised and not voicebox.tts_cache_path("boom").exists()
          and list(cache.glob("*.part.wav")) == [])
    n_calls = len(seen_paths)
    voicebox.tts.tts_synthesize("hello")
    check("second request is served from the cache", len(seen_paths) == n_calls)
    stale = cache / "dead.123-456.part.wav"
    stale.write_bytes(b"x")
    import os
    old = time.time() - 3600
    os.utime(stale, (old, old))
    fresh = cache / "live.123-456.part.wav"
    fresh.write_bytes(b"x")
    voicebox.prune_tts_cache()
    check("startup prune sweeps orphaned renders, spares live ones",
          not stale.exists() and fresh.exists())
finally:
    voicebox.tts.TTS_CACHE_DIR = real_cache
    voicebox.tts.synth_tts_wav = real_synth
    voicebox.tts.sf = real_sf

# --------------------------------------------- quick-speak renders are capped
voicebox.tts.tts_synthesize = lambda text, voice=None, rate=0: (TONE.copy(),
                                                              tmpdir / "x.wav")
bank = voicebox.TTSBank(voicebox.State(), path=tmpdir / "phrases.json")
bank.add("keeper")
wait_ui(lambda: bank.status.get("keeper") == "ready")
for n in range(20):
    bank.say(f"one-off {n}")
    wait_ui(lambda: bank.status.get(f"one-off {n}") == "ready", timeout=3.0)
loose = [t for t in bank.samples if t not in bank.phrases]
check("unsaved quick-speak renders are capped in memory", len(loose) <= 8,
      str(len(loose)))
check("the newest quick-speak render survives", "one-off 19" in bank.samples)
check("saved phrases are never trimmed", "keeper" in bank.samples)
check("trimmed entries leave no stale status/wav behind",
      set(bank.status) == set(bank.samples) == set(bank.wav_path))

# ------------------------------------------------ scenes: NaN/inf AI pitch


class FakeAi:
    available, proc, sel = True, None, 0
    voices = [Path("A.pth")]

    def __init__(self):
        self.pitches = []

    def voice_name(self):
        return "A"

    def select(self, i):
        pass

    def set_pitch(self, semis):
        self.pitches.append(semis)

    def set_fx(self, on):
        pass

    def stop(self):
        pass

    def start(self):
        pass


sstate = voicebox.State()
fai = FakeAi()
sc = voicebox.Scenes(sstate, ai=fai, path=tmpdir / "scenes.json")
for label, val in (("NaN", math.nan), ("inf", math.inf)):
    sc.scenes = [("odd", {"ai_pitch": val})]
    try:
        sc.apply(0)
        ok = -24 <= sstate.ai_pitch <= 24
    except Exception:
        ok = False
    check(f"scene with {label} AI pitch applies without crashing", ok)

# -------------------------------- voice path resumes clean after the AI mute
cstate = voicebox.State()
cstate.set_pitch(5)
cb = voicebox.make_callback(cstate)
frames = voicebox.BLOCKSIZE
out = np.zeros((frames, 1), dtype=np.float32)
loud = np.full((frames, 1), 0.5, dtype=np.float32)
for _ in range(6):
    cb(loud, out, frames, None, None)              # shifter is full of audio
cstate.ai_mute = True                              # worker owns the voice
cb(loud, out, frames, None, None)
check("shifter is emptied when the voice path goes silent",
      len(cstate.shifter.in_buf) == 0 and len(cstate.shifter.out_buf) == 0)
cstate.ai_mute = False
cb(np.zeros((frames, 1), dtype=np.float32), out, frames, None, None)
check("no stale audio replays when the voice path returns",
      float(np.abs(out).max()) < 1e-6, f"{float(np.abs(out).max()):.4f}")

# ------------------------------------- dashboard: hidden rects take no clicks
import pygame

ui_state = voicebox.State()
ui_bank = voicebox.TTSBank(ui_state, path=tmpdir / "ui_phrases.json")
for n in range(6):                     # more rows than the list can show
    ui_bank.add(f"phrase number {n}")
stop_flag = threading.Event()
dbg = voicebox.ui.ui_debug
found = {}


def poke():
    post = pygame.event.post
    labels = lambda: dbg.get("labels", [])
    voice_row = lambda: dbg.get("row_hit", {}).get(labels().index("TTS voice"))
    try:
        wait_ui(lambda: voice_row() and dbg["tts_row_hit"].get(0))
        time.sleep(0.2)
        row = voice_row().copy()
        rects = [r.copy() for r in list(dbg["tts_row_hit"].values())
                 + list(dbg["tts_del_hit"].values()) if r]
        found["overlap"] = [tuple(r) for r in rects if r.colliderect(row)]
        found["n_rects"] = len(rects)
        while not ui_state.events.empty():
            ui_state.events.get_nowait()
        # the spot where phrase 4's hidden half (and its x) used to sit
        post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1,
                                pos=(row.right - 15, row.top + 3)))
        time.sleep(0.3)
        found["phrases"] = len(ui_bank.phrases)
        found["spoke"] = not ui_state.events.empty()
        # cards scrolled up: nothing may answer from behind the scene strip
        for _ in range(12):
            post(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-1))
        time.sleep(0.6)
        pinned = {labels().index(n) for n in PINNED if n in labels()}
        top = min((r.top for d in ("row_hit", "hdr_hit", "value_hit",
                                   "tts_btn_hit", "tts_row_hit")
                   for k, r in list(dbg[d].items())
                   if r and not (d in ("row_hit", "value_hit")
                                 and k in pinned)),
                  default=10 ** 6)
        found["top"] = top
    finally:
        post(pygame.event.Event(pygame.QUIT))


# rows that live outside the scrolling pane keep their full rects
PINNED = ["Scene", "Save scene", "Clip volume", "HEAR self-listen",
          "Record output", "Global hotkeys", "Sound cues", "Input device",
          "Output device", "Rescan sounds", "Quit"]
ui_error = []
real_mouse = pygame.mouse.get_pos
threading.Thread(target=poke, daemon=True).start()
try:
    # the wheel handler reads the live pointer: park it over the cards
    pygame.mouse.get_pos = lambda: (150, 400)
    voicebox.run_ui(ui_state, stop_flag, "dev", "", None, None, None, ui_bank)
except Exception as e:
    ui_error.append(e)
finally:
    pygame.mouse.get_pos = real_mouse

check("dashboard survives the probe", not ui_error,
      repr(ui_error[0]) if ui_error else "")
check("phrase rows were on screen for the probe", found.get("n_rects", 0) > 0)
check("no phrase row/x hit-rect reaches the TTS voice row",
      found.get("overlap") == [], str(found.get("overlap")))
check("clicking the TTS voice row deletes no phrase",
      found.get("phrases") == 6, str(found.get("phrases")))
check("clicking the TTS voice row speaks no phrase", found.get("spoke") is False)
check("scrolled cards expose no hit-rect above the cards pane",
      found.get("top", 0) >= 132, str(found.get("top")))

finish()
