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

# ------------------------------------------------------------- grid UI smoke
import pygame
with state.lock:
    state.clips_to_mic = True
drain_ints()
stop_flag = threading.Event()
ui_board = voicebox.Board(state, None, None)     # no real streams from clicks

def poke():
    time.sleep(0.7)
    for _ in range(3):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN))
        time.sleep(0.01)
    # grid button 0 lives at (400,118)+(176x56)
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(410, 130)))
    time.sleep(0.05)
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3))
    time.sleep(0.05)
    # strip "To mic" button lives at (400,70)
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(410, 80)))
    time.sleep(0.05)
    # second grid button, now with to-mic off -> no mic event
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(590, 130)))
    time.sleep(0.05)
    pygame.event.post(pygame.event.Event(pygame.QUIT))

threading.Thread(target=poke, daemon=True).start()
ui_error = []
try:
    voicebox.run_ui(state, stop_flag, "dev", "", None, ui_board)
except Exception as e:
    ui_error.append(e)
check("two-pane UI with grid survives", not ui_error,
      repr(ui_error[0]) if ui_error else "")
ints = drain_ints()
check("grid click + hotkey reached mic channel", sorted(ints) == [0, 2], str(ints))
check("strip click toggled to-mic off", state.clips_to_mic is False)
check("grid click after toggle flashed locally only",
      ui_board.flash.get(1, 0) > 0)

finish()
