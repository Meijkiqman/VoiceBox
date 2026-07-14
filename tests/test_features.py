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
                 "Sounds to mic", "Pause sounds",
                 "Stop all sounds", "Quit"],
      str(labels))
menu_nomon = voicebox.Menu(state, stop_flag)   # hear-myself lives in the strip
check("menu rows identical without monitor",
      [it.label for it in menu_nomon.items] == labels)

# ------------------------------------------------------- UI smoke incl. mouse
import pygame
while not state.events.empty():
    state.events.get_nowait()

def poke():
    time.sleep(0.7)
    for _ in range(6):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN))
        time.sleep(0.01)
    # resizable window: grow + restore must relayout without crashing
    pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE, w=1160, h=780,
                                         size=(1160, 780)))
    time.sleep(0.05)
    pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE, w=960, h=660,
                                         size=(960, 660)))
    # row 0 (Preset) sits at y 82-116 in the skinned layout (header + section)
    pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION, pos=(300, 95)))    # hover row 0
    pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-1))         # wheel down
    pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=1))          # wheel up
    pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION, pos=(300, 95)))
    time.sleep(0.05)
    # click the Preset row -> alphabetical dropdown opens under it (y 120+)
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(300, 95)))
    time.sleep(0.05)
    # first item = "Chipmunk" (alphabetical); picking it applies the preset
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(150, 138)))
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(300, 2)))  # above list: ignored
    time.sleep(0.1)
    snaps.append(state.preset_label())               # before slider tweaks
    # Reverb slider (row 5, y 292-326): track spans x 168-288; drag to the end
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(228, 309)))
    time.sleep(0.05)
    pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION, pos=(348, 309)))
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=(348, 309)))
    time.sleep(0.05)
    # Echo (row 6, y 328-362): click the number, type an exact value
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(335, 345)))
    time.sleep(0.05)
    pygame.event.post(pygame.event.Event(pygame.TEXTINPUT, text="42"))
    time.sleep(0.05)
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
    time.sleep(0.05)
    pygame.event.post(pygame.event.Event(pygame.QUIT))

snaps = []

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
check("preset dropdown pick queued a pitch event", pitch_events >= 1)
check("preset dropdown pick applied Chipmunk",
      snaps and snaps[0] == "Chipmunk", str(snaps))
check("slider drag set reverb to the far end", state.reverb == 1.0,
      str(state.reverb))
check("typed value set echo to 42%", abs(state.echo - 0.42) < 1e-9,
      str(state.echo))

# ------------------------------------------- HEAR strip toggle (self-listen)
class FakeMonitor:
    def __init__(self):
        self.on, self.error = False, ""
    def toggle(self):
        self.on = not self.on

fmon = FakeMonitor()
stop_flag = threading.Event()

def hear_button_center():
    """Replicate run_ui's strip layout math to locate the HEAR button."""
    f = pygame.font.Font(
        str(voicebox.BASE_DIR / "assets" / "fonts" / "JetBrainsMono-Bold.ttf"),
        11)
    x = 384                                                 # G_X
    for lab, active in [
            ("TO MIC: ON" if state.clips_to_mic else "TO MIC: OFF",
             state.clips_to_mic),
            ("PAUSED" if state.clips_paused else "PAUSE", state.clips_paused),
            ("STOP", False)]:
        x += f.size(lab)[0] + 24 + (12 if active else 0) + 8
    return x + 10, 62 + 15                                  # STRIP_Y + H/2

def poke_hear():
    time.sleep(0.7)
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1,
                                         pos=hear_button_center()))
    time.sleep(0.1)
    pygame.event.post(pygame.event.Event(pygame.QUIT))

threading.Thread(target=poke_hear, daemon=True).start()
ui_error = []
try:
    voicebox.run_ui(state, stop_flag, "dev", "", fmon)
except Exception as e:
    ui_error.append(e)
check("UI with HEAR strip button survives", not ui_error,
      repr(ui_error[0]) if ui_error else "")
check("HEAR strip click toggled self-listen", fmon.on is True)

finish()
