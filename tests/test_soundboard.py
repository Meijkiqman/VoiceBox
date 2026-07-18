"""Soundboard: routing toggle, pause/stop, local player, backlog drain, grid UI."""
import threading
import time

from _common import check, finish

import numpy as np
import voicebox


def fake_load_clips():
    clips = [np.full(4000, 0.1, dtype=np.float32) for _ in range(9)]
    return clips, [f"clip{i+1}" for i in range(9)]
voicebox.soundboard.load_clips = fake_load_clips

state = voicebox.State()
cb = voicebox.make_callback(state)
frames = voicebox.BLOCKSIZE
silent = np.zeros((frames, 1), dtype=np.float32)
out = np.zeros((frames, 1), dtype=np.float32)

def drain_ints():
    ints = []
    while not state.events.empty():
        ev = state.events.get_nowait()
        if isinstance(ev, int):
            ints.append(ev)
    return ints

# ------------------------------------------------------------- Board routing
class FakePlayer:
    def __init__(self): self.played, self.stopped, self.error = [], 0, ""
    def play(self, i): self.played.append(i)
    def stop(self): self.stopped += 1

class FakeMonitor:
    on = False

fp, fm = FakePlayer(), FakeMonitor()
b = voicebox.Board(state, fp, fm)

b.play(3)
check("play sends to mic channel when toggle on", drain_ints() == [3])
check("play always plays locally", fp.played == [3])
check("play flashes the grid tile", b.flash.get(3, 0) > time.time())

b.toggle_mic()
b.play(4)
check("to-mic off: nothing queued for the mic", drain_ints() == [])
check("to-mic off: local playback still happens", fp.played == [3, 4])
b.toggle_mic()

fm.on = True
b.play(5)
check("self-listen mirror on: local skipped (no doubling)",
      drain_ints() == [5] and fp.played == [3, 4])
fm.on = False

b.play(99)
check("out-of-range index ignored", drain_ints() == [] and fp.played == [3, 4])

# --------------------------------------------------- pause/stop, mic channel
state.events.put(1)
cb(silent, out, frames, None, None)
cur_after_one = state.voices[0][1]
check("voice starts and advances", cur_after_one == frames)

b.toggle_pause()
cb(silent, out, frames, None, None)
check("paused: output silent", np.abs(out).max() == 0.0)
check("paused: cursor frozen", state.voices[0][1] == cur_after_one)

b.toggle_pause()
cb(silent, out, frames, None, None)
check("resumed: cursor advances again", state.voices[0][1] == cur_after_one + frames)

b.toggle_pause()
b.stop()
cb(silent, out, frames, None, None)
check("stop clears mic voices, tells player, un-pauses",
      len(state.voices) == 0 and fp.stopped == 1 and not state.clips_paused)

# ------------------------------------------------------------- hotkey pages
check("nine clips are one page", b.page_count() == 1)
b.play_hot(2)
check("page 0: slot plays the clip directly", drain_ints() == [2])

state.clips = [np.full(4000, 0.1, dtype=np.float32) for _ in range(21)]
state.clip_names = [f"clip{i+1}" for i in range(21)]
check("21 clips are three pages", b.page_count() == 3)
check("page steps forward", b.set_page(+1) == 1)
b.play_hot(2)
check("page 1: slot 2 fires clip 11", drain_ints() == [11])
check("page wraps around", b.set_page(+1) == 2 and b.set_page(+1) == 0)
check("page steps backward with wrap", b.set_page(-1) == 2)
b.play_hot(8)
check("last page: overhanging slot is ignored", drain_ints() == [])
b.play_hot(2)
check("last page: valid slot fires", drain_ints() == [20])

# ------------------------------------------------------------------- rescan
seen_version = state.clips_version
new_clips = ([np.full(2000, 0.2, dtype=np.float32) for _ in range(4)],
             [f"new{i+1}" for i in range(4)])
old_load = voicebox.soundboard.load_clips
voicebox.soundboard.load_clips = lambda: new_clips
b.rescan()
voicebox.soundboard.load_clips = old_load
check("rescan swaps in the new clips",
      len(state.clips) == 4 and state.clip_names == ["new1", "new2",
                                                     "new3", "new4"])
check("rescan bumps the UI version", state.clips_version == seen_version + 1)
check("rescan clamps the page", state.clip_page == 0)
check("rescan announces the count", "4 sound(s)" in state.status_msg)
b.play_hot(1)
check("hotkeys hit the fresh list", drain_ints() == [1])

# ------------------------------------- fallback Test start drains the backlog
# With no main stream and Test off, nothing consumes state.events; opening the
# fallback stream must clear queued clicks instead of firing them all at once.
class FakeStream:
    def __init__(self, *a, **k): pass
    def start(self): pass
    def close(self): pass

drain_ints()
state.set_pitch(-3)
state.events.put(0); state.events.put(1); state.events.put(2)
real_stream = voicebox.sd.Stream
voicebox.sd.Stream = FakeStream
try:
    m = voicebox.Monitor(state, has_main_stream=False)
    m.toggle()
finally:
    voicebox.sd.Stream = real_stream
left = []
while not state.events.empty():
    left.append(state.events.get_nowait())
check("fallback test-start drains clip backlog, keeps pitch",
      left == [("pitch", -3)], str(left))
state.set_pitch(0)
drain_ints()

# ------------------------------------------------------- LocalPlayer callback
lp = voicebox.LocalPlayer(state)
out1d = np.zeros((frames, 1), dtype=np.float32)
lp.events.put(0)
lp._callback(out1d, frames, None, None)
check("local callback mixes clip", np.abs(out1d).max() > 0.05)
cur = lp.voices[0][1]
with state.lock:
    state.clips_paused = True
lp._callback(out1d, frames, None, None)
check("local callback honors pause",
      np.abs(out1d).max() == 0.0 and lp.voices[0][1] == cur)
with state.lock:
    state.clips_paused = False
lp.events.put("stop")
lp._callback(out1d, frames, None, None)
check("local callback honors stop", len(lp.voices) == 0)

prev_count = state.status_count
lp._callback(out1d, frames, None, "output underflow")
check("local player reports stream status",
      state.status_msg.startswith("speakers:")
      and state.status_count == prev_count + 1)

# ------------------------------------------------- loudness normalization
import tempfile
from pathlib import Path

import soundfile as sf

norm_dir = Path(tempfile.mkdtemp())
t = np.arange(4800) / 48000
sf.write(str(norm_dir / "loud.wav"),
         (0.98 * np.sin(2 * np.pi * 220 * t)).astype(np.float32),
         48000, subtype="FLOAT")
sf.write(str(norm_dir / "quiet.wav"),
         (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32),
         48000, subtype="FLOAT")
old_dir = voicebox.soundboard.SOUNDS_DIR
voicebox.soundboard.SOUNDS_DIR = norm_dir
n_clips, n_names = voicebox.load_clips()         # the unpatched original
voicebox.soundboard.SOUNDS_DIR = old_dir
by_name = dict(zip(n_names, n_clips))
check("loud clips normalized down to the target peak",
      abs(float(np.abs(by_name["loud"]).max()) - 0.9) < 0.02,
      f"peak={float(np.abs(by_name['loud']).max()):.3f}")
check("quiet clips boosted at most 4x",
      abs(float(np.abs(by_name["quiet"]).max()) - 0.2) < 0.02,
      f"peak={float(np.abs(by_name['quiet']).max()):.3f}")

# ------------------------------------------------------------- grid UI smoke
import pygame


def inject(ev):
    """Hand a synthetic event to run_ui's main-thread hook -
    cross-thread pygame.event.post corrupts the SDL queue."""
    from collections import deque
    voicebox.ui.ui_debug.setdefault("inject", deque()).append(ev)
with state.lock:
    state.clips_to_mic = True
drain_ints()
stop_flag = threading.Event()
ui_board = voicebox.Board(state, None, None)     # no real streams from clicks

# drag-and-drop lands in (a stand-in for) sounds/ and triggers a rescan
drop_src = Path(tempfile.mkdtemp()) / "dropped.wav"
sf.write(str(drop_src), np.zeros(4800, np.float32), 48000)
drop_dest_dir = Path(tempfile.mkdtemp())
old_sounds_dir = voicebox.ui.SOUNDS_DIR
voicebox.ui.SOUNDS_DIR = drop_dest_dir

def ui_rect(kind, key):
    """Center of a live hit-rect from the dashboard's debug registry."""
    r = voicebox.ui.ui_debug.get(kind, {}).get(key)
    return r.center if r else (0, 0)


def poke():
    from _common import wait_ui
    wait_ui(lambda: ui_rect("grid_hit", 0) != (0, 0))
    inject(pygame.event.Event(pygame.DROPFILE, file=str(drop_src)))
    time.sleep(0.1)
    for _ in range(3):
        inject(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN))
        time.sleep(0.01)
    # grid tile 0 (SOUNDBOARD card)
    inject(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1,
                                         pos=ui_rect("grid_hit", 0)))
    time.sleep(0.05)
    inject(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3))
    time.sleep(0.05)
    # the TO MIC chip in the soundboard card header
    inject(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1,
                                         pos=ui_rect("strip_hit", "mic")))
    time.sleep(0.05)
    # second grid tile, now with to-mic off -> no mic event
    inject(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1,
                                         pos=ui_rect("grid_hit", 1)))
    time.sleep(0.05)
    inject(pygame.event.Event(pygame.QUIT))

threading.Thread(target=poke, daemon=True).start()
ui_error = []
try:
    voicebox.run_ui(state, stop_flag, "dev", "", None, ui_board)
except Exception as e:
    ui_error.append(e)
finally:
    voicebox.ui.SOUNDS_DIR = old_sounds_dir
check("two-pane UI with grid survives", not ui_error,
      repr(ui_error[0]) if ui_error else "")
check("dropped file copied into sounds/",
      (drop_dest_dir / "dropped.wav").is_file())
ints = drain_ints()
check("grid click + hotkey reached mic channel", sorted(ints) == [0, 2], str(ints))
check("strip click toggled to-mic off", state.clips_to_mic is False)
check("grid click after toggle flashed locally only",
      ui_board.flash.get(1, 0) > 0)

finish()
