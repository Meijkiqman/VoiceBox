"""Core regressions: event protocol, menu flash, held-key repeat."""
import threading
import time

from _common import check, finish

import numpy as np
import voicebox


def fake_load_clips():
    clips = [np.full(1000 + 100 * i, 0.1, dtype=np.float32) for i in range(9)]
    names = [f"clip{i+1}" for i in range(9)]
    return clips, names
voicebox.soundboard.load_clips = fake_load_clips

# ------------------------------------------------- 1. callback event protocol
state = voicebox.State()
cb = voicebox.make_callback(state)
frames = voicebox.BLOCKSIZE
indata = np.zeros((frames, 1), dtype=np.float32)
outdata = np.zeros((frames, 1), dtype=np.float32)

state.set_pitch(4)
check("set_pitch stores display value", state.semitones == 4)
check("set_pitch queues event, does not touch shifter directly",
      abs(state.shifter.ratio - 1.0) < 1e-9)
cb(indata, outdata, frames, None, None)
check("callback applies queued pitch to shifter",
      abs(state.shifter.ratio - 2 ** (4 / 12)) < 1e-9)

state.events.put(2)                       # start clip 3
cb(indata, outdata, frames, None, None)
check("clip event starts a voice", len(state.voices) == 1)
check("clip audio reaches output", np.abs(outdata).max() > 0.01)

state.events.put("stop")
cb(indata, outdata, frames, None, None)
check("stop event clears voices", len(state.voices) == 0)

# out-of-range / junk events must be ignored without error
state.events.put(99); state.events.put(("bogus",)); state.events.put(None)
cb(indata, outdata, frames, None, None)
check("junk events ignored", len(state.voices) == 0)

# output length is always exactly `frames`, with voice through the shifter
sig = (0.2 * np.sin(2 * np.pi * 220 * np.arange(frames * 20) / voicebox.SAMPLERATE)
       ).astype(np.float32)
ok_len = True
for b in range(20):
    blk = sig[b * frames:(b + 1) * frames].reshape(-1, 1)
    out = np.zeros((frames, 1), dtype=np.float32)
    cb(blk, out, frames, None, None)
    if out.shape != (frames, 1):
        ok_len = False
check("pitched stream keeps block size", ok_len)
check("pitched stream produces audio after warm-up", np.abs(out).max() > 0.001)

state.set_pitch(0)
cb(indata, outdata, frames, None, None)
check("return to passthrough", abs(state.shifter.ratio - 1.0) < 1e-9)

# ------------------------------------------------- 2. Menu flash flag
stop_flag = threading.Event()
menu = voicebox.Menu(state, stop_flag)
quit_item = next(it for it in menu.items if it.label == "Quit")
stop_item = next(it for it in menu.items if it.label == "Stop all sounds")
check("Quit item has flash disabled", quit_item.flash is False)
check("Stop-all item flashes", stop_item.flash is True and stop_item.value_fn is None)
menu.sel = menu.items.index(stop_item)
menu.on_select()
check("select flashes stop-all row", menu.flash.get(menu.sel, 0) > time.time())
menu.sel = menu.items.index(quit_item)
menu.on_select()
check("select on Quit sets stop flag, no flash",
      stop_flag.is_set() and menu.sel not in menu.flash)
stop_flag.clear()

# ------------------------------------------------- 3. headless UI smoke test
import pygame

def poke():
    time.sleep(0.7)                       # let run_ui finish pygame.init
    for _ in range(20):                   # walk past the bottom (wraps + scrolls)
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN))
        time.sleep(0.01)
    for _ in range(25):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))
        time.sleep(0.01)
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3))  # clip
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3))  # repeat: ignored
    time.sleep(0.1)
    pygame.event.post(pygame.event.Event(pygame.QUIT))

t = threading.Thread(target=poke, daemon=True)
t.start()
ui_error = []
try:
    voicebox.run_ui(state, stop_flag, "dev", "")
except Exception as e:
    ui_error.append(e)
check("UI loop with scrolling + nav survives", not ui_error,
      repr(ui_error[0]) if ui_error else "")
# the two K_3 KEYDOWNs (no KEYUP between) must have started exactly one voice
n_clip_events = 0
while not state.events.empty():
    if isinstance(state.events.get_nowait(), int):
        n_clip_events += 1
check("held-key repeat starts only one clip", n_clip_events == 1, f"got {n_clip_events}")

finish()
