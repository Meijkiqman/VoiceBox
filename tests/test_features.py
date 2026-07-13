"""Presets, grit, monitor/self-listen, mic meter, mouse navigation."""
import threading
import time

from _common import check, finish

import numpy as np
import queue
import voicebox


def fake_load_clips():
    clips = [np.full(1000, 0.1, dtype=np.float32) for _ in range(9)]
    return clips, [f"clip{i+1}" for i in range(9)]
voicebox.load_clips = fake_load_clips

state = voicebox.State()
cb = voicebox.make_callback(state)
frames = voicebox.BLOCKSIZE
silent = np.zeros((frames, 1), dtype=np.float32)
out = np.zeros((frames, 1), dtype=np.float32)

# ------------------------------------------------------------------ presets
names = [n for n, _ in voicebox.PRESETS]
check("Space Marine preset exists", "Space Marine" in names, str(names))
sm = names.index("Space Marine")
sm_cfg = voicebox.PRESETS[sm][1]
state.apply_preset(sm)
check("preset sets display pitch", state.semitones == sm_cfg["semitones"])
check("preset sets effect mixes",
      state.doubler == sm_cfg.get("doubler", 0.0)
      and state.robot == sm_cfg.get("robot", 0.0)
      and state.bass == sm_cfg.get("bass", 0.0)
      and state.drive == sm_cfg.get("drive", 0.0)
      and state.reverb == sm_cfg.get("reverb", 0.0))
cb(silent, out, frames, None, None)
check("preset pitch reaches shifter via event",
      abs(state.shifter.ratio - 2 ** (sm_cfg["semitones"] / 12)) < 1e-9)

check("preset label matches after apply", state.preset_label() == "Space Marine")
state.nudge("drive", -0.05, hi=1.0)
check("preset label goes Custom on tweak", state.preset_label() == "Custom")
state.apply_preset(sm)
check("re-apply restores preset label", state.preset_label() == "Space Marine")

state.apply_preset(sm + 1)
check("preset cycling wraps forward", state.preset_idx == (sm + 1) % len(names))
state.apply_preset(0)
cb(silent, out, frames, None, None)
check("Normal preset restores passthrough",
      abs(state.shifter.ratio - 1.0) < 1e-9 and state.drive == 0.0)

# ------------------------------------------------------------------- grit dsp
with state.lock:
    state.drive = 1.0
sine = (0.5 * np.sin(2 * np.pi * 220 * np.arange(frames) / voicebox.SAMPLERATE)
        ).astype(np.float32).reshape(-1, 1)
cb(sine, out, frames, None, None)
check("grit saturates toward full scale", np.abs(out).max() > 0.9,
      f"peak={np.abs(out).max():.3f}")
check("grit stays within clip range", np.abs(out).max() <= 1.0)
with state.lock:
    state.drive = 0.0
cb(sine, out, frames, None, None)
check("drive off = clean passthrough", abs(np.abs(out).max() - 0.5) < 0.01)

# ------------------------------------------------------------------ mic meter
check("in_level tracks block peak", abs(state.in_level - 0.5) < 0.01,
      f"{state.in_level:.3f}")

# ------------------------------------------------------------- monitor mirror
q = queue.Queue(maxsize=8)
state.monitor_q = q
cb(sine, out, frames, None, None)
check("callback mirrors block to monitor queue",
      q.qsize() == 1 and len(q.get_nowait()) == frames)
for _ in range(12):                      # overfill: producer must drop, not block
    cb(sine, out, frames, None, None)
check("full monitor queue drops instead of blocking", q.qsize() == 8)
state.monitor_q = None

# real OutputStream toggle (default output device on this machine)
m = voicebox.Monitor(state, has_main_stream=True)
m.toggle()
check("monitor toggle opens stream or reports error", m.on or bool(m.error),
      m.error)
if m.on:
    check("monitor on registers queue", state.monitor_q is not None)
    time.sleep(0.15)
    m.toggle()
    check("monitor off clears queue", state.monitor_q is None and not m.on)
m.close()

# ------------------------------------------------------------------ menu rows
stop_flag = threading.Event()
menu = voicebox.Menu(state, stop_flag, voicebox.Monitor(state, True))
labels = [it.label for it in menu.items]
check("settings rows in order",
      labels == ["Preset", "Pitch", "Robot voice", "Helmet doubler",
                 "Grit / growl", "Reverb", "Echo", "Radio voice", "Bass boost",
                 "Voice volume", "Clip volume", "TTS voice FX", "TTS volume",
                 "Test - hear myself", "Sounds to mic", "Pause sounds",
                 "Stop all sounds", "Quit"],
      str(labels))
menu_nomon = voicebox.Menu(state, stop_flag)
check("Menu without monitor omits Test row",
      all(it.label != "Test - hear myself" for it in menu_nomon.items))

# ------------------------------------------------------- UI smoke incl. mouse
import pygame
while not state.events.empty():
    state.events.get_nowait()

def poke():
    time.sleep(0.7)
    for _ in range(6):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN))
        time.sleep(0.01)
    # row 0 (Preset) sits at y 82-116 in the skinned layout (header + section)
    pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION, pos=(300, 95)))    # hover row 0
    pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-1))         # wheel down
    pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=1))          # wheel up
    pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION, pos=(300, 95)))
    time.sleep(0.05)
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(300, 95)))
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(300, 2)))  # above list: ignored
    time.sleep(0.05)
    pygame.event.post(pygame.event.Event(pygame.QUIT))

threading.Thread(target=poke, daemon=True).start()
ui_error = []
try:
    voicebox.run_ui(state, stop_flag, "dev", "", None)
except Exception as e:
    ui_error.append(e)
check("UI with mouse events survives", not ui_error,
      repr(ui_error[0]) if ui_error else "")

pitch_events = 0
while not state.events.empty():
    ev = state.events.get_nowait()
    if isinstance(ev, tuple) and ev[0] == "pitch":
        pitch_events += 1
check("mouse click activated the Preset row", pitch_events >= 1)

finish()
