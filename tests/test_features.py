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
voicebox.soundboard.load_clips = fake_load_clips

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

# ------------------------------------------------------------------ mic mute
with state.lock:
    state.mic_muted = True
loud = np.full((frames, 1), 0.5, dtype=np.float32)
cb(loud, out, frames, None, None)
check("mute silences the mic", np.abs(out).max() == 0.0)
check("meter still reads the muted mic", state.in_level == 0.5)
state.events.put(0)                       # soundboard clip while muted
cb(loud, out, frames, None, None)
check("soundboard plays while muted", np.abs(out).max() > 0.05)
state.events.put("stop")
cb(loud, out, frames, None, None)
with state.lock:
    state.mic_muted = False
cb(loud, out, frames, None, None)
check("unmute restores the voice", np.abs(out).max() > 0.4)

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
      labels == ["Preset", "Save preset", "Pitch", "Mic", "Noise gate",
                 "Robot voice", "Helmet doubler",
                 "Grit / growl", "Reverb", "Echo", "Radio voice", "Bass boost",
                 "Voice volume", "Clip volume", "TTS voice FX", "TTS volume",
                 "HEAR self-listen", "Sound cues", "Sounds to mic",
                 "Pause sounds", "Stop all sounds", "Rescan sounds", "Quit"],
      str(labels))
menu_nomon = voicebox.Menu(state, stop_flag)   # no monitor = no HEAR row
check("menu rows identical without monitor (minus HEAR)",
      [it.label for it in menu_nomon.items]
      == [l for l in labels if l != "HEAR self-listen"])

# --------------------------------------------------------------- sound cues
class CuePlayer:
    def __init__(self): self.played = []
    def play_raw(self, s): self.played.append(s)

cue_player = CuePlayer()
state.cues = voicebox.Cues(state, cue_player)
menu.toggle_mute()                      # mute -> low blip
menu.toggle_mute()                      # live -> high blip
check("mute toggles fire sound cues", len(cue_player.played) == 2)
check("cue tones are gentle float32",
      cue_player.played[0].dtype == np.float32
      and float(np.abs(cue_player.played[0]).max()) <= 0.25)
with state.lock:
    state.cues_on = False
menu.toggle_mute()
check("Sound cues off silences the blips", len(cue_player.played) == 2)
menu.toggle_mute()                      # leave the mic live again
with state.lock:
    state.cues_on = True
state.cues = None

# ------------------------------------------------------- UI smoke incl. mouse
import pygame
while not state.events.empty():
    state.events.get_nowait()
state.user_presets = []       # dropdown order below assumes built-ins only

def ui_dbg():
    return voicebox.ui.ui_debug


def ui_row(name):
    """Center of a menu row's live rect, located by label."""
    d = ui_dbg()
    r = d["row_hit"].get(d["labels"].index(name))
    return r.center if r else (0, 0)


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
    time.sleep(0.1)
    pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION,
                                         pos=ui_row("Preset")))
    pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-1))
    pygame.event.post(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=1))
    time.sleep(0.1)
    pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION,
                                         pos=ui_row("Preset")))
    time.sleep(0.05)
    # click the Preset row -> alphabetical dropdown opens anchored to it
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1,
                                         pos=ui_row("Preset")))
    time.sleep(0.1)
    # first item = "Chipmunk" (alphabetical); picking it applies the preset
    di = ui_dbg().get("drop_info")
    if di:
        pos0 = (di["rect"].x + 30,
                di["rect"].y + di["pad"] - int(di["scroll"])
                + di["item_h"] // 2)
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                             button=1, pos=pos0))
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1,
                                         pos=(300, 2)))   # header: ignored
    time.sleep(0.1)
    snaps.append(state.preset_label())               # before slider tweaks
    # Reverb slider: jump-click the track, then drag past the right end
    tr = ui_dbg()["slider_track"].get(ui_dbg()["labels"].index("Reverb"))
    if tr:
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1,
                                             pos=tr.center))
        time.sleep(0.05)
        pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION,
                                             pos=(tr.right + 60, tr.centery)))
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1,
                                             pos=(tr.right + 60, tr.centery)))
    time.sleep(0.05)
    # Echo: click the number, type an exact value
    vr = ui_dbg()["value_hit"].get(ui_dbg()["labels"].index("Echo"))
    if vr:
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1,
                                             pos=vr.center))
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

def poke_hear():
    time.sleep(0.7)
    # HEAR lives in the SYSTEM card now; the debug registry finds its row
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1,
                                         pos=ui_row("HEAR self-listen")))
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
