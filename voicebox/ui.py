"""The VoiceBox window: settings menu, soundboard grid, TTS panel (pygame).
Skin ported from design/VoiceBox Skin.dc.html - see run_ui's docstring."""
import shutil
import time
from collections import deque
from pathlib import Path

import numpy as np

from .config import (BASE_DIR, SAMPLERATE, SOUNDS_DIR, TTS_MAX_CHARS,
                     WINDOW_SIZE)
from .controls import build_keymap, load_controls
from .soundboard import Board
from .tts import TTSBank

def get_clipboard_text():
    """OS clipboard -> str ('' when empty or unavailable). Tk ships with
    CPython and needs no visible window; created per call - paste is rare.
    Newlines/tabs collapse to spaces so a paste stays a one-line phrase."""
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        try:
            text = root.clipboard_get()
        finally:
            root.destroy()
    except Exception:
        return ""
    return " ".join(str(text).split())


class MenuItem:
    def __init__(self, label, value_fn=None, select=None, adjust=None, flash=True,
                 slider=None):
        self.label = label
        self.value_fn = value_fn      # () -> str shown on the right
        self.select = select          # on_select handler
        self.adjust = adjust          # on_left/on_right handler, adjust(delta)
        self.flash = flash            # flash the row on select (off for Quit)
        self.slider = slider          # numeric rows: (get, set, lo, hi, unit)
                                      # unit "pct" (0..hi shown as %) or "st"


class Menu:
    """Single screen. Keyboard handlers call the same on_* methods the
    controller uses, so behavior is identical across input devices."""

    def __init__(self, state, stop_flag, monitor=None, board=None, ai=None,
                 hotkeys=None, engine=None, recorder=None, tts=None,
                 scenes=None, translator=None, harvester=None, trainer=None,
                 listener=None):
        self.state = state
        self.stop_flag = stop_flag
        self.monitor = monitor
        self.board = board if board is not None else Board(state)
        self.ai = ai
        self.hotkeys = hotkeys
        self.engine = engine
        self.recorder = recorder
        self.tts = tts
        self.scenes = scenes
        self.translator = translator
        self.harvester = harvester
        self.trainer = trainer
        self.listener = listener
        self.sel = 0
        self.flash = {}               # item index -> flash-until timestamp
        s = state
        self.items = [
            MenuItem("Preset",
                     lambda: s.preset_label(),
                     select=lambda: s.apply_preset(s.preset_idx),
                     adjust=lambda d: s.apply_preset(s.preset_idx + d)),
            MenuItem("Save preset",
                     select=self._save_preset),
            MenuItem("Pitch",
                     lambda: f"{s.semitones:+.0f} st" if s.semitones else "off",
                     select=lambda: s.set_pitch(0),
                     adjust=lambda d: s.set_pitch(s.semitones + d),
                     slider=(lambda: s.semitones,
                             lambda v: s.set_pitch(int(round(v))),
                             -12, 12, "st")),
            MenuItem("Mic",
                     lambda: "MUTED" if s.mic_muted else "live",
                     select=self.toggle_mute,
                     adjust=lambda d: self.toggle_mute()),
            MenuItem("Noise gate",
                     lambda: f"{s.gate_db:.0f} dB" if s.gate_on else "off",
                     select=self._toggle_gate,
                     adjust=self._adjust_gate),
            MenuItem("Robot voice",
                     lambda: f"{s.robot:.0%}" if s.robot else "off",
                     select=self._toggle_robot,
                     adjust=lambda d: s.nudge("robot", d * 0.05, hi=1.0),
                     slider=(lambda: s.robot,
                             lambda v: s.set_val("robot", v, hi=1.0),
                             0.0, 1.0, "pct")),
            MenuItem("Helmet doubler",
                     lambda: f"{s.doubler:.0%}" if s.doubler else "off",
                     adjust=lambda d: s.nudge("doubler", d * 0.05, hi=1.0),
                     slider=(lambda: s.doubler,
                             lambda v: s.set_val("doubler", v, hi=1.0),
                             0.0, 1.0, "pct")),
            MenuItem("Grit / growl",
                     lambda: f"{s.drive:.0%}",
                     adjust=lambda d: s.nudge("drive", d * 0.05, hi=1.0),
                     slider=(lambda: s.drive,
                             lambda v: s.set_val("drive", v, hi=1.0),
                             0.0, 1.0, "pct")),
            MenuItem("Reverb",
                     lambda: f"{s.reverb:.0%}",
                     adjust=lambda d: s.nudge("reverb", d * 0.05, hi=1.0),
                     slider=(lambda: s.reverb,
                             lambda v: s.set_val("reverb", v, hi=1.0),
                             0.0, 1.0, "pct")),
            MenuItem("Echo",
                     lambda: f"{s.echo:.0%}",
                     adjust=lambda d: s.nudge("echo", d * 0.05, hi=1.0),
                     slider=(lambda: s.echo,
                             lambda v: s.set_val("echo", v, hi=1.0),
                             0.0, 1.0, "pct")),
            MenuItem("Radio voice",
                     lambda: "ON" if s.radio else "off",
                     select=self._toggle_radio,
                     adjust=lambda d: self._toggle_radio()),
            MenuItem("Bass boost",
                     lambda: f"{s.bass:.0%}" if s.bass else "off",
                     adjust=lambda d: s.nudge("bass", d * 0.05, hi=1.0),
                     slider=(lambda: s.bass,
                             lambda v: s.set_val("bass", v, hi=1.0),
                             0.0, 1.0, "pct")),
            MenuItem("Voice volume",
                     lambda: f"{s.voice_gain:.0%}",
                     adjust=lambda d: s.nudge("voice_gain", d * 0.05),
                     slider=(lambda: s.voice_gain,
                             lambda v: s.set_val("voice_gain", v),
                             0.0, 1.5, "pct")),
            MenuItem("Clip volume",
                     lambda: f"{s.clip_gain:.0%}",
                     adjust=lambda d: s.nudge("clip_gain", d * 0.05),
                     slider=(lambda: s.clip_gain,
                             lambda v: s.set_val("clip_gain", v),
                             0.0, 1.5, "pct")),
        ]
        if ai is not None and ai.available:
            self.items.append(MenuItem(
                "AI voice",
                lambda: ai.status,
                select=ai.toggle,
                adjust=lambda d: ai.toggle()))
            self.items.append(MenuItem(
                "AI character",
                lambda: ai.voice_name(),
                adjust=lambda d: ai.cycle(d)))
            self.items.append(MenuItem(
                "AI pitch",
                lambda: f"{s.ai_pitch:+.0f} st" if s.ai_pitch else "off",
                select=lambda: self._set_ai_pitch(0),
                adjust=lambda d: self._set_ai_pitch(s.ai_pitch + d),
                slider=(lambda: s.ai_pitch, self._set_ai_pitch,
                        -24, 24, "st")))
            self.items.append(MenuItem(
                "AI voice FX",
                lambda: "ON" if s.ai_fx else "off",
                select=self._toggle_ai_fx,
                adjust=lambda d: self._toggle_ai_fx()))
        self.items.append(MenuItem(
            "TTS voice FX",
            lambda: "ON" if s.tts_fx else "off",
            select=self._toggle_tts_fx,
            adjust=lambda d: self._toggle_tts_fx()))
        self.items.append(MenuItem(
            "TTS volume",
            lambda: f"{s.tts_gain:.0%}",
            adjust=lambda d: s.nudge("tts_gain", d * 0.05),
            slider=(lambda: s.tts_gain,
                    lambda v: s.set_val("tts_gain", v),
                    0.0, 1.5, "pct")))
        if tts is not None:
            tts.load_voice_names()     # listed by the time the row is reached
            self.items.append(MenuItem(
                "TTS voice",
                self._tts_voice_label,
                select=lambda: self._cycle_tts_voice(+1),
                adjust=self._cycle_tts_voice))
            self.items.append(MenuItem(
                "TTS rate",
                lambda: f"{s.tts_rate:+.0f}" if s.tts_rate else "normal",
                select=self._reset_tts_rate,
                adjust=self._adjust_tts_rate))
        if translator is not None:
            self.items.append(MenuItem(
                "Auto translate",
                lambda: "ON" if translator.auto else "off",
                select=translator.toggle_auto,
                adjust=lambda d: translator.toggle_auto()))
            self.items.append(MenuItem(
                "Translate",
                translator.row_label,
                select=translator.toggle,
                adjust=lambda d: translator.toggle()))
            self.items.append(MenuItem(
                "Translate to",
                translator.target_label,
                select=lambda: translator.cycle_target(+1),
                adjust=translator.cycle_target))
            self.items.append(MenuItem(
                "Translate from",
                translator.source_label,
                select=lambda: translator.cycle_source(+1),
                adjust=translator.cycle_source))
            self.items.append(MenuItem(
                "Translate voice",
                translator.voice_label,
                select=lambda: translator.cycle_voice(+1),
                adjust=translator.cycle_voice))
            self.items.append(MenuItem(
                "Translate volume",
                lambda: f"{s.trans_gain:.0%}",
                adjust=lambda d: s.nudge("trans_gain", d * 0.05),
                slider=(lambda: s.trans_gain,
                        lambda v: s.set_val("trans_gain", v),
                        0.0, 1.5, "pct")))
        if listener is not None:
            self.items.append(MenuItem(
                "Incoming speech",
                listener.row_label,
                select=listener.toggle,
                adjust=lambda d: listener.toggle()))
            self.items.append(MenuItem(
                "Listen device",
                listener.device_label,
                select=lambda: listener.cycle_device(+1),
                adjust=listener.cycle_device))
            self.items.append(MenuItem(
                "Speak incoming",
                lambda: "ON" if s.listen_speak else "off",
                select=self._toggle_listen_speak,
                adjust=lambda d: self._toggle_listen_speak()))
            self.items.append(MenuItem(
                "Listen passthrough",
                # honest label: the persisted preference can be ON while the
                # speaker stream failed to open (device busy/missing)
                lambda: ("failed - see status"
                         if s.listen_pass and listener.on
                         and listener.out_stream is None
                         else "ON" if s.listen_pass else "off"),
                select=self._toggle_listen_pass,
                adjust=lambda d: self._toggle_listen_pass()))
        if harvester is not None:
            self.items.append(MenuItem(
                "Voice harvest",
                lambda: "ON" if harvester.on else
                        ("full" if harvester.full else "off"),
                select=harvester.toggle,
                adjust=lambda d: harvester.toggle()))
            self.items.append(MenuItem(
                "Dataset",
                lambda: f"{harvester.minutes:.1f}/60 min"))
        if trainer is not None and trainer.available:
            self.items.append(MenuItem(
                "Retrain AI voice",
                trainer.label,
                select=trainer.launch))
        if recorder is not None:
            self.items.append(MenuItem(
                "Record output",
                self._rec_label,
                select=recorder.toggle,
                adjust=lambda d: recorder.toggle()))
        if monitor is not None:
            self.items.append(MenuItem(
                "HEAR self-listen",
                lambda: "ON" if monitor.on else "off",
                select=self._toggle_monitor,
                adjust=lambda d: self._toggle_monitor()))
        if hotkeys is not None:
            self.items.append(MenuItem(
                "Global hotkeys",
                lambda: "ON" if hotkeys.on else "off",
                select=hotkeys.toggle,
                adjust=lambda d: hotkeys.toggle()))
        self.items.append(MenuItem(
            "Sound cues",
            lambda: "ON" if s.cues_on else "off",
            select=self._toggle_cues,
            adjust=lambda d: self._toggle_cues()))
        if engine is not None:
            self.items.append(MenuItem(
                "Input device",
                lambda: engine.short_name("input"),
                select=lambda: engine.cycle("input", +1),
                adjust=lambda d: engine.cycle("input", d)))
            self.items.append(MenuItem(
                "Output device",
                lambda: engine.short_name("output"),
                select=lambda: engine.cycle("output", +1),
                adjust=lambda d: engine.cycle("output", d)))
        b = self.board
        self.items.append(MenuItem("Sounds to mic",
                                   lambda: "ON" if s.clips_to_mic else "off",
                                   select=b.toggle_mic,
                                   adjust=lambda d: b.toggle_mic()))
        self.items.append(MenuItem("Pause sounds",
                                   lambda: "PAUSED" if s.clips_paused else "off",
                                   select=b.toggle_pause,
                                   adjust=lambda d: b.toggle_pause()))
        self.items.append(MenuItem("Stop all sounds", select=b.stop))
        self.items.append(MenuItem("Rescan sounds", select=b.rescan))
        self.items.append(MenuItem("Quit", select=self.stop_flag.set, flash=False))
        if scenes is not None:
            # the whole persona in one row: first, above the pieces it sets
            self.items[0:0] = [
                MenuItem("Scene",
                         lambda: scenes.applied or "-",
                         select=lambda: scenes.cycle(+1),
                         adjust=lambda d: scenes.cycle(d)),
                MenuItem("Save scene", select=self._save_scene),
            ]

    def toggle_mute(self):
        with self.state.lock:
            self.state.mic_muted = not self.state.mic_muted
        if self.state.cues is not None:
            self.state.cues.mute(self.state.mic_muted)

    def _save_preset(self):
        name = self.state.save_user_preset()
        self.state.status_msg = \
            f"saved \"{name}\" - right-click it in the Preset list to rename"
        self.state.status_at = time.time()

    def _save_scene(self):
        name = self.scenes.save()
        self.state.status_msg = \
            f"saved \"{name}\" - right-click it in the Scene list to rename"
        self.state.status_at = time.time()

    def _toggle_gate(self):
        with self.state.lock:
            self.state.gate_on = not self.state.gate_on

    def _adjust_gate(self, d):
        """Threshold in 2 dB steps; adjusting also switches the gate on so
        the row gives audible feedback while dialing."""
        with self.state.lock:
            self.state.gate_on = True
            self.state.gate_db = max(-70.0, min(-10.0,
                                                self.state.gate_db + 2.0 * d))

    def _toggle_robot(self):
        with self.state.lock:
            self.state.robot = 0.0 if self.state.robot > 0 else 1.0

    def _toggle_radio(self):
        with self.state.lock:
            self.state.radio = not self.state.radio

    def _toggle_listen_speak(self):
        with self.state.lock:
            self.state.listen_speak = not self.state.listen_speak

    def _toggle_listen_pass(self):
        self.listener.set_passthrough(not self.state.listen_pass)

    def _toggle_tts_fx(self):
        with self.state.lock:
            self.state.tts_fx = not self.state.tts_fx

    def _toggle_ai_fx(self):
        """AI voice through the effect chain (pitch, echo, ...) on/off."""
        with self.state.lock:
            self.state.ai_fx = not self.state.ai_fx
        if self.ai is not None:
            self.ai.set_fx(self.state.ai_fx)

    def _set_ai_pitch(self, v):
        """Transpose into the RVC model; applies live to a running worker."""
        with self.state.lock:
            self.state.ai_pitch = float(max(-24, min(24, int(round(v)))))
        if self.ai is not None:
            self.ai.set_pitch(int(self.state.ai_pitch))

    def _toggle_cues(self):
        with self.state.lock:
            self.state.cues_on = not self.state.cues_on
        if self.state.cues is not None and self.state.cues_on:
            self.state.cues.mute(self.state.mic_muted)   # sample the sound

    def _rec_label(self):
        if not self.recorder.on:
            return "off"
        secs = int(time.time() - self.recorder.started_at)   # sample time once:
        return f"REC {secs // 60}:{secs % 60:02d}"           # no 0:00 at 1:00

    def _tts_voice_label(self):
        v = self.state.tts_voice
        if v is None:
            return "default"
        return v if len(v) <= 24 else v[:21] + "..."

    def _cycle_tts_voice(self, d=1):
        opts = [None] + (self.tts.voice_names or [])
        try:
            i = opts.index(self.state.tts_voice)
        except ValueError:             # saved voice no longer installed
            i = 0
        with self.state.lock:
            self.state.tts_voice = opts[(i + (1 if d >= 0 else -1)) % len(opts)]
        self.tts.invalidate()

    def _adjust_tts_rate(self, d):
        with self.state.lock:
            self.state.tts_rate = max(-10.0, min(10.0, self.state.tts_rate + d))
        self.tts.invalidate()

    def _reset_tts_rate(self):
        with self.state.lock:
            self.state.tts_rate = 0.0
        self.tts.invalidate()

    def _toggle_monitor(self):
        self.monitor.toggle()
        if self.ai is not None:        # AI live: the worker mirrors the voice
            # (unless FX routing is on - then the main mirror carries it)
            self.ai.set_monitor(self.monitor.on and not self.state.ai_fx)
        if self.monitor.error:         # surface failures in the status line
            self.state.status_msg = f"test: {self.monitor.error}"
            self.state.status_at = time.time()

    def play_clip(self, i):
        self.board.play(i)

    # --- the on_* interface (called by both keyboard and controller paths) ---
    def on_up(self):    self.sel = (self.sel - 1) % len(self.items)
    def on_down(self):  self.sel = (self.sel + 1) % len(self.items)
    def on_left(self):  self._adjust(-1)
    def on_right(self): self._adjust(+1)
    def on_back(self):  self.stop_flag.set()

    def on_select(self):
        it = self.items[self.sel]
        if it.select:
            it.select()
            if it.value_fn is None and it.flash:
                self.flash[self.sel] = time.time() + 0.25

    def _adjust(self, d):
        it = self.items[self.sel]
        if it.adjust:
            it.adjust(d)


# module-level debug registry: run_ui refreshes these hit-rect dicts every
# frame so the headless tests can aim real clicks instead of replicating
# layout math. Read-only for everyone but run_ui.
ui_debug = {}


def run_ui(state, stop_flag, dev_line, err_line="", monitor=None, board=None,
           ai=None, tts=None, hotkeys=None, engine=None, recorder=None,
           scenes=None, translator=None, harvester=None, trainer=None,
           listener=None):
    """VoiceBox dashboard, ported from design/VoiceBox Dashboard.dc.html.

    One self-contained card per feature - TRANSLATOR, AI VOICE, MY VOICE,
    VOICE, TEXT-TO-SPEECH, INCOMING TRANSLATOR, SYSTEM - in fixed 296px
    columns (page pad/gaps 12), plus the SOUNDBOARD as a pinned 320px
    column on the right and a slim SCENES strip under the header. Cards
    collapse to their 32px header (state persists); card headers are focus
    stops (TAB cycles them); every card carries its own volume slider.
    Tokens, fonts and the motion recipes (single glow, 250ms flash, 140ms
    eased scroll, easeOut(t)=1-(1-t)^2) carry over from the previous skin.
    """
    import os
    import pygame
    pygame.init()
    pygame.display.set_caption("VoiceBox")
    # headless screenshot hook (tests/dev): VOICEBOX_SHOT=path[@WxH][:frames]
    shot_path, shot_frames, shot_size = "", 0, WINDOW_SIZE
    shot = os.environ.get("VOICEBOX_SHOT", "")
    if shot:
        if ":" in shot.rpartition(".png")[2] or shot.count(":") > (1 if os.name == "nt" else 0):
            shot, _, nf = shot.rpartition(":")
            shot_frames = int(nf or 8)
        else:
            shot_frames = 8
        if "@" in shot:
            shot, _, wh = shot.partition("@")
            w_, _, h_ = wh.partition("x")
            shot_size = (int(w_), int(h_))
        shot_path = shot
    screen = pygame.display.set_mode(shot_size if shot_path else WINDOW_SIZE,
                                     pygame.RESIZABLE)
    try:                          # OS-level minimum = the design's base size
        from pygame._sdl2.video import Window as _SDLWindow
        _SDLWindow.from_display_module().minimum_size = WINDOW_SIZE
    except Exception:
        pass
    clock = pygame.time.Clock()
    pygame.key.set_repeat(320, 110)           # held arrows auto-repeat
    frame_no = 0

    cfg = load_controls()
    keymap, clipmap = build_keymap(cfg, pygame)
    pad = cfg["gamepad"]

    def _buttons(v):
        if isinstance(v, int):
            return [v]
        if isinstance(v, list):
            return [b for b in v if isinstance(b, int)]
        return []

    def _num(v, fallback):
        try:
            return float(v)
        except (TypeError, ValueError):
            return fallback

    pad_select = _buttons(pad.get("select"))
    pad_back = _buttons(pad.get("back"))
    pad_stop = _buttons(pad.get("stop_clips"))
    threshold = _num(pad.get("axis_threshold"), 0.5)
    cooldown = _num(pad.get("nav_cooldown"), 0.22)
    joy_last = 0.0
    held_keys = set()             # set_repeat re-fires KEYDOWN; track real presses

    for i in range(pygame.joystick.get_count()):
        pygame.joystick.Joystick(i).init()

    # ------------------------------------------------------ theme (tokens JSON)
    CLR = {
        "bg":          (11, 13, 16),    "paneLeft":    (13, 16, 20),
        "raisedTop":   (23, 27, 34),    "raisedBot":   (19, 22, 28),
        "hoverTop":    (29, 35, 44),    "hoverBot":    (23, 27, 34),
        "active":      (35, 43, 54),    "stroke":      (35, 42, 52),
        "strokeSoft":  (28, 34, 43),    "strokeHover": (44, 53, 66),
        "accent":      (51, 214, 255),  "accentBright": (127, 230, 255),
        "accentDim":   (42, 175, 212),  "textOnAccent": (4, 20, 26),
        "danger":      (255, 77, 94),   "warning":     (255, 177, 61),
        "success":     (61, 220, 133),  "text":        (232, 237, 242),
        "text2":       (199, 208, 218), "muted":       (154, 167, 180),
        "faint":       (92, 104, 117),  "barTrack":    (35, 42, 52),
        "barFill":     (70, 82, 95),    "meterOff":    (26, 31, 38),
        "peak":        (255, 255, 255), "scrollTrack": (22, 27, 34),
        "scrollThumb": (58, 70, 83),    "headerTop":   (20, 24, 31),
        "headerBot":   (16, 20, 26),    "footerTop":   (16, 20, 26),
        "footerBot":   (13, 16, 21),    "cardBg":      (13, 16, 20),
    }
    ACCENT_TINT = ((51, 214, 255, 26), (51, 214, 255, 13))
    DANGER_TINT = ((255, 77, 94, 20), (255, 77, 94, 10))
    WARN_TINT = ((255, 177, 61, 26), (255, 177, 61, 15))

    def mixc(a, b, t):
        return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t),
                int(a[2] + (b[2] - a[2]) * t))

    FONTS_DIR = BASE_DIR / "assets" / "fonts"

    def _font(fname, size, fallback, bold=False):
        p = FONTS_DIR / fname
        try:
            if p.is_file():
                return pygame.font.Font(str(p), size)
        except Exception:
            pass
        try:
            return pygame.font.SysFont(fallback, size, bold=bold)
        except Exception:
            return pygame.font.Font(None, size + 6)

    f_word = _font("SpaceGrotesk-Bold.ttf", 17, "bahnschrift,segoeui", True)
    f_label = _font("SpaceGrotesk-Medium.ttf", 13, "segoeui")
    f_labelF = _font("SpaceGrotesk-SemiBold.ttf", 13, "segoeui", True)
    f_tile = _font("SpaceGrotesk-SemiBold.ttf", 13, "segoeui", True)
    f_hdr = _font("JetBrainsMono-Bold.ttf", 10, "consolas", True)
    f_val = _font("JetBrainsMono-Medium.ttf", 12, "consolas")
    f_valF = _font("JetBrainsMono-Bold.ttf", 12, "consolas", True)
    f_small = _font("JetBrainsMono-Medium.ttf", 10, "consolas")
    f_badge = _font("JetBrainsMono-Bold.ttf", 10, "consolas", True)
    f_strip = _font("JetBrainsMono-Bold.ttf", 11, "consolas", True)
    f_foot = _font("JetBrainsMono-Medium.ttf", 11, "consolas")

    if board is None:
        board = Board(state)
    if tts is None:
        tts = TTSBank(state, getattr(board, "player", None), monitor, ai)
    menu = Menu(state, stop_flag, monitor, board, ai, hotkeys, engine,
                recorder, tts, scenes, translator, harvester, trainer,
                listener)
    board = menu.board
    kb_action = {a: keys for a, keys in keymap.items()}

    def key_action(key):
        for action, keys in kb_action.items():
            if key in keys:
                return action
        return None

    # ------------------------------------------------------------- caches
    text_cache = {}

    def T(fnt, s, color):
        key = (id(fnt), s, color)
        surf = text_cache.get(key)
        if surf is None:
            if len(text_cache) > 900:
                text_cache.clear()
            surf = fnt.render(s, True, color)
            text_cache[key] = surf
        return surf

    def TT(fnt, s, color, tracking):
        """Letter-spaced text (cached) - CSS letter-spacing equivalent."""
        key = ("trk", id(fnt), s, color, tracking)
        surf = text_cache.get(key)
        if surf is None:
            chars = [fnt.render(c, True, color) for c in s]
            w = sum(c.get_width() for c in chars) + tracking * max(0, len(chars) - 1)
            h = max((c.get_height() for c in chars), default=1)
            surf = pygame.Surface((max(1, w), h), pygame.SRCALPHA)
            x = 0
            for c in chars:
                surf.blit(c, (x, 0))
                x += c.get_width() + tracking
            text_cache[key] = surf
        return surf

    grad_cache = {}

    def grad(w, h, top, bot, radius=0):
        """Cached 2-stop vertical gradient, optionally rounded. RGB or RGBA."""
        key = (w, h, top, bot, radius)
        surf = grad_cache.get(key)
        if surf is None:
            if len(grad_cache) > 400:
                grad_cache.clear()
            t = top if len(top) == 4 else (*top, 255)
            b = bot if len(bot) == 4 else (*bot, 255)
            g = pygame.Surface((1, 2), pygame.SRCALPHA)
            g.set_at((0, 0), t)
            g.set_at((0, 1), b)
            surf = pygame.transform.smoothscale(g, (max(1, w), max(1, h)))
            if radius:
                mask = pygame.Surface((max(1, w), max(1, h)), pygame.SRCALPHA)
                pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(),
                                 border_radius=radius)
                surf.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
            grad_cache[key] = surf
        return surf

    glow_cache = {}
    G_PAD = 14

    def glow(w, h, color, radius):
        """The one glow recipe (focusGlow token) - soft rings, cached."""
        key = (w, h, color, radius)
        surf = glow_cache.get(key)
        if surf is None:
            if len(glow_cache) > 40:
                glow_cache.clear()
            surf = pygame.Surface((w + 2 * G_PAD, h + 2 * G_PAD), pygame.SRCALPHA)
            for e in range(G_PAD, 0, -1):
                a = int(34 * ((G_PAD - e + 1) / G_PAD) ** 2)
                pygame.draw.rect(surf, (*color, a),
                                 pygame.Rect(G_PAD - e, G_PAD - e, w + 2 * e, h + 2 * e),
                                 border_radius=radius + e)
            glow_cache[key] = surf
        return surf

    # ----------------------------------------------------- layout (tokens JSON)
    HEADER_H, FOOTER_H, SCENE_H, SYS_H = 52, 32, 40, 40
    PAGE_PAD = COL_GAP = CARD_GAP = 12
    CARD_W, SB_W = 296, 320
    CARD_HDR, PAD_X, PAD_TOP, PAD_BOT = 32, 8, 6, 8
    ROW_HGT, ROW_GAP = 34, 2
    COLS, GGAP, TILE_H = 3, 8, 62
    CHIP_H = 24
    TTS_IN_H, TTS_LIST_H = 30, 100
    TTS_ROW_H, TTS_ROW_GAP = 30, 2

    SYS_TOP = HEADER_H                 # system bar sits right under the header
    SCENE_TOP = SYS_TOP + SYS_H
    CARD_TOP = SCENE_TOP + SCENE_H
    WIN_W = WIN_H = VIEW_BOT = 0
    SB_X = 0
    sb_rect = sb_grid_rect = tts_list_rect = None
    cards_area = None
    TILE_W = 0

    idx_of = {it.label: i for i, it in enumerate(menu.items)}

    def _rows(*labels):
        return [idx_of[l] for l in labels if l in idx_of]

    def card(key, title, rows, dot=None, summary=None, dim=None, show=True):
        return {"key": key, "title": title, "rows": rows, "dot": dot,
                "summary": summary, "dim": dim, "show": show}

    CARD_DEFS = [
        card("translator", "TRANSLATOR",
             _rows("Auto translate", "Translate", "Translate from",
                   "Translate to", "Translate voice", "Translate volume"),
             dot=(lambda: translator.auto) if translator else None,
             summary=(lambda: f"{translator.source().upper()} > "
                              f"{translator.target().upper()}")
             if translator else None,
             show=translator is not None),
        card("ai", "AI VOICE",
             _rows("AI voice", "AI character", "AI pitch", "AI voice FX"),
             dot=(lambda: ai.proc is not None) if ai else None,
             summary=(lambda: ai.status.upper()) if ai else None,
             # dim only a resting card: loading/error must stay readable
             dim=(lambda: ai.proc is None and ai.status == "off")
             if ai else None,
             show=ai is not None and ai.available),
        card("myvoice", "MY VOICE",
             _rows("Voice harvest", "Dataset", "Retrain AI voice"),
             dot=(lambda: harvester.on) if harvester else None,
             summary=(lambda: "EXP"),
             show=harvester is not None),
        card("voice", "VOICE",
             _rows("Preset", "Save preset", "Pitch", "Robot voice",
                   "Helmet doubler", "Grit / growl", "Reverb", "Echo",
                   "Bass boost", "Radio voice", "Noise gate", "Mic",
                   "Voice volume"),
             dot=lambda: not state.mic_muted),
        card("tts", "TEXT-TO-SPEECH",
             _rows("TTS voice", "TTS rate", "TTS voice FX", "TTS volume"),
             summary=lambda: f"{len(tts.phrases)} PHRASES"),
        card("incoming", "INCOMING TRANSLATOR",
             _rows("Incoming speech", "Listen device", "Speak incoming",
                   "Listen passthrough"),
             dot=(lambda: listener.on) if listener else None,
             dim=(lambda: not listener.on) if listener else None,
             show=listener is not None),
    ]
    # SYSTEM is not a card: it is the chip bar under the header (the old
    # always-visible toggle buttons), drawn from these menu rows in order
    SYSTEM_BAR = [
        ("HEAR self-listen", "HEAR",
         (lambda: monitor.on) if monitor is not None else None),
        ("Auto translate", "TRANS",
         (lambda: translator.auto) if translator is not None else None),
        ("Record output", "REC",
         (lambda: recorder.on) if recorder is not None else None),
        ("Global hotkeys", "KEYS",
         (lambda: hotkeys.on) if hotkeys is not None else None),
        ("Sound cues", "CUES", lambda: state.cues_on),
        ("Input device", "IN", None),
        ("Output device", "OUT", None),
        ("Rescan sounds", "RESCAN", None),
        ("Quit", "QUIT", None),
    ]
    SYSTEM_BAR = [(idx_of[lab], lab, chip, act) for lab, chip, act in
                  SYSTEM_BAR if lab in idx_of]
    cards = [c for c in CARD_DEFS if c["show"] and c["rows"]]
    card_of_row = {i: c["key"] for c in cards for i in c["rows"]}
    collapsed = state.cards_collapsed        # persisted set of card keys
    SCENE_IDX = idx_of.get("Scene")
    SAVE_SCENE_IDX = idx_of.get("Save scene")
    CLIP_VOL_IDX = idx_of.get("Clip volume")
    # bar chips that are ALSO a card row (TRANS) stay chip-click-only; the
    # rest join row_hit/focus like any row and must survive the frame clear
    SYS_SOLO = [e for e in SYSTEM_BAR if e[0] not in card_of_row]
    KEEP_ROWS = {SCENE_IDX, SAVE_SCENE_IDX} | {e[0] for e in SYS_SOLO}

    def card_h(c):
        if c["key"] in collapsed:
            return CARD_HDR
        n = len(c["rows"])
        h = CARD_HDR + PAD_TOP + n * (ROW_HGT + ROW_GAP) - (ROW_GAP if n else 0) \
            + PAD_BOT
        if c["key"] == "tts":
            h += TTS_IN_H + 6 + TTS_LIST_H + 8
        return h

    # column layout + focus stops, rebuilt when width/collapse changes
    columns = []                  # [(x, [(card, y, h), ...]), ...]
    cards_content_h = 0
    focus_stops = []              # ("row", item_idx) | ("hdr", card_key)
    stop_of = {}                  # ("row", i)/("hdr", key) -> stop index
    stop_y = {}                   # stop index -> content y (for scrolling)
    layout_dirty = [True]

    def build_layout():
        nonlocal columns, cards_content_h, focus_stops, stop_of, stop_y
        avail = WIN_W - PAGE_PAD * 2 - SB_W - COL_GAP
        n = max(1, (avail + COL_GAP) // (CARD_W + COL_GAP))
        cols = [[] for _ in range(n)]
        heights = [0.0] * n
        for c in cards:           # fixed order, shortest column first
            j = min(range(n), key=lambda k: heights[k])
            h = card_h(c)
            cols[j].append((c, heights[j], h))
            heights[j] += h + CARD_GAP
        columns = [(PAGE_PAD + j * (CARD_W + COL_GAP), col)
                   for j, col in enumerate(cols)]
        cards_content_h = max(heights) if any(heights) else 0
        focus_stops, stop_of, stop_y = [], {}, {}

        def add(stop, y):
            stop_of[stop] = len(focus_stops)
            stop_y[len(focus_stops)] = y
            focus_stops.append(stop)
        for i, _lab, _chip, _act in SYS_SOLO:
            add(("row", i), -1)              # system bar: never scrolled
        if SCENE_IDX is not None:
            add(("row", SCENE_IDX), -1)      # scene strip: never scrolled
        if SAVE_SCENE_IDX is not None:
            add(("row", SAVE_SCENE_IDX), -1)
        for _x, col in columns:
            for c, y, h in col:
                add(("hdr", c["key"]), y)
                if c["key"] not in collapsed:
                    for k, ri in enumerate(c["rows"]):
                        add(("row", ri), y + CARD_HDR + PAD_TOP
                            + k * (ROW_HGT + ROW_GAP))
        if CLIP_VOL_IDX is not None:
            add(("row", CLIP_VOL_IDX), -1)   # pinned soundboard column
        layout_dirty[0] = False

    fsel = 0        # first stop: the scene chip when scenes exist, else the
                    # first card header

    def focus_stop():
        return focus_stops[fsel] if 0 <= fsel < len(focus_stops) else None

    def focus_sync():
        st = focus_stop()
        if st and st[0] == "row":
            menu.sel = st[1]

    def focus_move(d):
        nonlocal fsel
        if focus_stops:
            fsel = (fsel + d) % len(focus_stops)
            focus_sync()

    def focus_tab(d=1):
        """Cycle card headers directly (TAB / gamepad shoulders)."""
        nonlocal fsel
        hdrs = [k for k, s in enumerate(focus_stops) if s[0] == "hdr"]
        if not hdrs:
            return
        later = [k for k in hdrs if (k - fsel) * d > 0] if d > 0 else \
                [k for k in reversed(hdrs) if k < fsel]
        fsel = (later[0] if later else (hdrs[0] if d > 0 else hdrs[-1]))
        focus_sync()

    def focus_row(i):
        """Point focus at a row by item index (mouse hover)."""
        nonlocal fsel
        st = stop_of.get(("row", i))
        if st is not None:
            fsel = st
        menu.sel = i

    def toggle_collapse(key):
        nonlocal fsel
        if key in collapsed:
            collapsed.discard(key)
        else:
            collapsed.add(key)
        cur = focus_stop()
        build_layout()
        fsel = stop_of.get(cur, stop_of.get(("hdr", key), 0))

    def relayout():
        nonlocal WIN_W, WIN_H, VIEW_BOT, SB_X, sb_rect, cards_area, TILE_W
        WIN_W = max(screen.get_width(), WINDOW_SIZE[0])
        WIN_H = max(screen.get_height(), WINDOW_SIZE[1])
        VIEW_BOT = WIN_H - FOOTER_H
        SB_X = WIN_W - PAGE_PAD - SB_W
        sb_rect = pygame.Rect(SB_X, CARD_TOP, SB_W, VIEW_BOT - CARD_TOP - 4)
        cards_area = pygame.Rect(0, CARD_TOP, SB_X - COL_GAP // 2,
                                 VIEW_BOT - CARD_TOP)
        TILE_W = (SB_W - 2 * PAD_X - GGAP * (COLS - 1)) // COLS
        ui_debug["win"] = (WIN_W, WIN_H)   # tests wait on this after resizes
        build_layout()
        rebuild_grid()

    grid_rows, grid_content_h, clips_seen = 0, 0, -1
    clip_by_id, disp_names, clip_secs = {}, [], []

    def rebuild_grid():
        nonlocal grid_rows, grid_content_h, clips_seen
        clips_seen = state.clips_version
        grid_rows = (len(state.clips) + COLS - 1) // COLS
        grid_content_h = (grid_rows * (TILE_H + GGAP) - GGAP + 20
                          if state.clips else 0)
        clip_by_id.clear()
        clip_by_id.update({id(c): i for i, c in enumerate(state.clips)})
        disp_names.clear()
        clip_secs.clear()
        name_max = TILE_W - 16 - 18
        for nm, c in zip(state.clip_names, state.clips):
            if f_tile.render(nm, True, CLR["text"]).get_width() > name_max:
                while nm and f_tile.render(nm + "...", True,
                                           CLR["text"]).get_width() > name_max:
                    nm = nm[:-1]
                nm += "..."
            disp_names.append(nm)
            clip_secs.append(f"{len(c) / SAMPLERATE:.1f}s")

    # ----------------------------------------------------------- motion state
    cards_scroll = cards_target = 0.0
    grid_scroll = grid_target = 0.0
    tts_scroll = tts_target = 0.0
    tts_text, tts_focus = "", False
    tts_trunc = {}
    hover_mix = {}
    nudge = {"i": -1, "at": 0.0, "side": 0}
    strip_press = {}
    meter_lit = 0.0
    peak_lit, peak_at = 0.0, 0.0
    row_hit, hdr_hit, strip_hit, grid_hit, scene_hit = {}, {}, {}, {}, {}
    sys_hit = {}                  # system-bar chips, by menu item index
    tts_row_hit, tts_del_hit, tts_btn_hit = {}, {}, {}
    arrow_hit = None
    slider_hit, slider_track, value_hit = {}, {}, {}
    slider_drag = None
    edit = None
    cap_cache = {"key": None, "surf": None}
    cap_hit = None
    last_t = time.time()
    ui_debug.update(row_hit=row_hit, hdr_hit=hdr_hit, strip_hit=strip_hit,
                    grid_hit=grid_hit, scene_hit=scene_hit,
                    tts_row_hit=tts_row_hit, tts_del_hit=tts_del_hit,
                    tts_btn_hit=tts_btn_hit, slider_track=slider_track,
                    value_hit=value_hit,
                    labels=[it.label for it in menu.items], drop_info=None,
                    sys_hit=sys_hit)

    def step(cur, target, dt, dur):
        if dur <= 0:
            return float(target)
        k = min(1.0, dt / dur)
        k = k * (2.0 - k)
        v = cur + (target - cur) * k
        return float(target) if abs(target - v) < 0.4 else v

    def hover_step(key, active, dt):
        m = hover_mix.get(key, 0.0)
        m = step(m, 1.0 if active else 0.0, dt, 0.09 if active else 0.15)
        if m <= 0.002:
            hover_mix.pop(key, None)
            return 0.0
        hover_mix[key] = m
        return m

    def q8(t):
        return round(t * 8) / 8

    def row_at(pos):
        for i, r in row_hit.items():
            if r.collidepoint(pos):
                return i
        return None

    def tts_commit():
        nonlocal tts_text
        if tts.add(tts_text):
            tts_text = ""

    def tts_set_focus(on):
        nonlocal tts_focus
        if on == tts_focus:
            return
        tts_focus = on
        try:
            (pygame.key.start_text_input if on else pygame.key.stop_text_input)()
        except Exception:
            pass

    def flip_page(d):
        nonlocal grid_target
        page = board.set_page(d)
        grid_target = (page * 9 // COLS) * (TILE_H + GGAP)

    def go_left():
        menu.on_left()
        nudge.update(i=menu.sel, at=time.time(), side=-1)

    def go_right():
        menu.on_right()
        nudge.update(i=menu.sel, at=time.time(), side=+1)

    # ---- numeric rows: slider drag + click-the-number-to-type ---------------
    def slider_set_from_x(i, mx):
        tr = slider_track.get(i)
        if tr is None:
            return
        _get, set_, lo, hi, unit = menu.items[i].slider
        frac = max(0.0, min(1.0, (mx - tr.x) / max(1, tr.w)))
        v = lo + frac * (hi - lo)
        set_(int(round(v)) if unit == "st" else v)

    def edit_open(i):
        nonlocal edit
        tts_set_focus(False)
        edit = {"row": i, "text": ""}
        try:
            pygame.key.start_text_input()
        except Exception:
            pass

    def edit_close():
        nonlocal edit
        edit = None
        if not tts_focus:
            try:
                pygame.key.stop_text_input()
            except Exception:
                pass

    def edit_commit():
        if edit is not None and edit["text"].strip():
            _get, set_, lo, hi, unit = menu.items[edit["row"]].slider
            try:
                v = float(edit["text"].replace(",", ".").strip())
            except ValueError:
                v = None
            if v is not None:
                if unit == "pct":
                    v /= 100.0
                set_(max(lo, min(hi, int(round(v)) if unit == "st" else v)))
        edit_close()

    def val_style(val, focused, now):
        if val == "ON":
            return f_valF, CLR["accent"], CLR["accent"]
        if val == "PAUSED":
            return f_valF, CLR["warning"], CLR["warning"]
        if val == "MUTED":
            return f_valF, CLR["danger"], CLR["danger"]
        if val.startswith("REC "):
            return f_valF, CLR["danger"], CLR["danger"]
        if val == "error":
            return f_valF, CLR["danger"], CLR["danger"]
        if val.startswith("loading"):
            a = 0.4 + 0.6 * (0.5 + 0.5 * float(np.sin(now * 2 * np.pi / 1.2)))
            return (f_val, mixc(CLR["paneLeft"], CLR["muted"], a),
                    mixc(CLR["paneLeft"], CLR["accent"], a))
        if val == "off":
            return f_val, CLR["faint"], None
        if focused:
            return f_valF, CLR["accent"], None
        return f_val, CLR["text"], None

    def draw_slider(r, i, it, val, focused, now):
        """Numeric row: label | slider track+knob | value (click to type)."""
        get, _set, lo, hi, _unit = it.slider
        cy = r.centery
        cur = float(get())
        frac = max(0.0, min(1.0, (cur - lo) / (hi - lo) if hi != lo else 0.0))
        if edit is not None and edit["row"] == i:
            box = pygame.Rect(r.right - 8 - 54, cy - 11, 54, 22)
            screen.blit(grad(box.w, box.h, CLR["headerBot"], CLR["paneLeft"], 5),
                        box.topleft)
            pygame.draw.rect(screen, CLR["accent"], box, width=1, border_radius=5)
            txt = edit["text"]
            if txt:
                ts = T(f_valF, txt, CLR["text"])
            else:
                ts = T(f_val, val, CLR["faint"])
            tx = box.right - 6 - ts.get_width()
            screen.blit(ts, (tx, cy - ts.get_height() // 2))
            if txt and (now * 2.0) % 2 < 1:
                pygame.draw.line(screen, CLR["accent"],
                                 (box.right - 5, cy - 7), (box.right - 5, cy + 7))
            value_hit[i] = box
            val_left = box.x
        else:
            n_hot = nudge["i"] == i and (now - nudge["at"]) < 0.18
            fnt, col, _dot = val_style(val, focused, now)
            ts = T(f_valF if focused else fnt, val,
                   CLR["accentBright"] if n_hot else
                   (CLR["accent"] if focused else col))
            vr = pygame.Rect(r.right - 10 - ts.get_width(),
                             cy - ts.get_height() // 2,
                             ts.get_width(), ts.get_height())
            vh = q8(hover_step(("val", i), vr.inflate(12, 10)
                               .collidepoint(mouse_pos), dt))
            if vh > 0.3:
                pygame.draw.rect(screen, mixc(CLR["paneLeft"], CLR["accent"], 0.35),
                                 vr.inflate(12, 8), width=1, border_radius=5)
            screen.blit(ts, vr.topleft)
            value_hit[i] = vr.inflate(12, 10)
            val_left = vr.x - 6
        # slider track + knob; the 296px card leaves less room than the old
        # 370px pane, so the track starts after the label and squeezes
        # before it gives up
        lw = T(f_label, it.label, CLR["text2"]).get_width()
        tx1 = min(val_left - 12, r.right - 10 - 54 - 12)
        tx0 = max(r.x + 10 + lw + 12, tx1 - 124)
        if tx1 - tx0 < 36:
            tx0 = max(r.x + 10 + lw + 8, tx1 - 36)
        track = pygame.Rect(tx0, cy - 2, max(24, tx1 - tx0), 4)
        pygame.draw.rect(screen, CLR["barTrack"], track, border_radius=2)
        if frac > 0.01:
            pygame.draw.rect(screen, CLR["accent"] if focused else CLR["barFill"],
                             pygame.Rect(track.x, track.y,
                                         max(2, int(track.w * frac)), 4),
                             border_radius=2)
        kx = track.x + int(track.w * frac)
        kh = q8(hover_step(("knob", i),
                           slider_drag == i
                           or track.inflate(10, 14).collidepoint(mouse_pos), dt))
        pygame.draw.circle(screen, CLR["paneLeft"], (kx, cy), 7)
        pygame.draw.circle(screen,
                           mixc(CLR["accent"] if focused else CLR["muted"],
                                CLR["accentBright"], kh),
                           (kx, cy), 5)
        slider_track[i] = track
        slider_hit[i] = track.inflate(12, 16)

    def draw_value(r, i, it, val, focused, now):
        nonlocal arrow_hit
        if it.slider is not None:
            draw_slider(r, i, it, val, focused, now)
            return
        cy = r.centery
        is_pct = val.endswith("%")
        n_hot = nudge["i"] == i and (now - nudge["at"]) < 0.18
        if focused and it.adjust:
            ra = pygame.Rect(r.right - 5 - 20, cy - 11, 20, 22)
            if is_pct:
                vs = T(f_valF, val, CLR["accentBright"] if n_hot else CLR["accent"])
                vx = ra.x - 6 - 34
                screen.blit(vs, (vx + 34 - vs.get_width(), cy - vs.get_height() // 2))
                bar = pygame.Rect(vx - 7 - 56, cy - 2, 56, 4)
                pygame.draw.rect(screen, CLR["barTrack"], bar, border_radius=2)
                frac = min(1.0, float(val[:-1]) / 100.0)
                if frac > 0.01:
                    pygame.draw.rect(screen, CLR["accent"],
                                     pygame.Rect(bar.x, bar.y, max(2, int(56 * frac)), 4),
                                     border_radius=2)
                la = pygame.Rect(bar.x - 8 - 20, cy - 11, 20, 22)
            else:
                fnt, col, dot = val_style(val, True, now)
                vs = T(f_valF, val, CLR["accentBright"] if n_hot else col)
                vx = ra.x - 8 - vs.get_width()
                screen.blit(vs, (vx, cy - vs.get_height() // 2))
                if dot:
                    pygame.draw.circle(screen, dot, (vx - 10, cy), 2)
                    vx -= 14
                la = pygame.Rect(vx - 8 - 20, cy - 11, 20, 22)
            for side, arect in ((-1, la), (1, ra)):
                if n_hot and nudge["side"] == side:
                    pygame.draw.rect(screen, CLR["accent"], arect, border_radius=5)
                    ss = T(f_valF, "<" if side < 0 else ">", CLR["textOnAccent"])
                else:
                    pygame.draw.rect(screen, mixc(CLR["paneLeft"], CLR["accent"], 0.35),
                                     arect, width=1, border_radius=5)
                    ss = T(f_valF, "<" if side < 0 else ">", CLR["accent"])
                screen.blit(ss, (arect.centerx - ss.get_width() // 2,
                                 arect.centery - ss.get_height() // 2))
            arrow_hit = (i, la, ra)
        else:
            right = r.right - 10
            if is_pct:
                vs = T(f_valF if focused else f_val, val,
                       CLR["accent"] if focused else CLR["text"])
                screen.blit(vs, (right - vs.get_width(), cy - vs.get_height() // 2))
                bar = pygame.Rect(right - 34 - 7 - 56, cy - 2, 56, 4)
                pygame.draw.rect(screen, CLR["barTrack"], bar, border_radius=2)
                frac = min(1.0, float(val[:-1]) / 100.0)
                if frac > 0.01:
                    pygame.draw.rect(screen,
                                     CLR["accent"] if focused else CLR["barFill"],
                                     pygame.Rect(bar.x, bar.y, max(2, int(56 * frac)), 4),
                                     border_radius=2)
            else:
                fnt, col, dot = val_style(val, focused, now)
                vs = T(fnt, val, col)
                screen.blit(vs, (right - vs.get_width(), cy - vs.get_height() // 2))
                if dot:
                    pygame.draw.circle(screen, dot, (right - vs.get_width() - 10, cy), 2)

    def draw_tts_body(r, y):
        """TEXT-TO-SPEECH card: input box + ADD button, then the phrase list
        with its own scroll - above the voice/rate/fx/volume rows. Returns
        the y where the plain rows continue."""
        nonlocal tts_list_rect, tts_scroll, tts_target
        bx0, bx1 = r.x + PAD_X, r.right - PAD_X
        in_rect = pygame.Rect(bx0, y, bx1 - bx0 - 52, TTS_IN_H)
        add_rect = pygame.Rect(in_rect.right + 6, y,
                               bx1 - in_rect.right - 6, TTS_IN_H)
        tts_btn_hit["input"] = in_rect
        tts_btn_hit["add"] = add_rect
        hm = q8(hover_step(("tts", "input"), in_rect.collidepoint(mouse_pos), dt))
        screen.blit(grad(in_rect.w, in_rect.h, CLR["headerBot"],
                         CLR["paneLeft"], 7), in_rect.topleft)
        pygame.draw.rect(screen,
                         CLR["accent"] if tts_focus
                         else mixc(CLR["stroke"], CLR["strokeHover"], hm),
                         in_rect, width=1, border_radius=7)
        screen.set_clip(in_rect.inflate(-8, 0))
        icy = in_rect.centery
        if tts_text:
            tsurf = T(f_val, tts_text, CLR["text"])
            tx0 = in_rect.x + 10 + min(0, in_rect.w - 20 - tsurf.get_width())
            screen.blit(tsurf, (tx0, icy - tsurf.get_height() // 2))
            caret_x = tx0 + tsurf.get_width() + 2
        else:
            if not tts_focus:
                ph = T(f_val, "Type a phrase...", CLR["faint"])
                screen.blit(ph, (in_rect.x + 10, icy - ph.get_height() // 2))
            caret_x = in_rect.x + 10
        if tts_focus and (now * 2.0) % 2 < 1:
            pygame.draw.line(screen, CLR["accent"],
                             (caret_x, icy - 8), (caret_x, icy + 8))
        screen.set_clip(cards_area)
        can_add = bool(tts_text.strip())
        hm = q8(hover_step(("tts", "add"), add_rect.collidepoint(mouse_pos), dt))
        if can_add:
            pygame.draw.rect(screen, mixc(CLR["accent"], CLR["accentBright"], hm),
                             add_rect, border_radius=7)
            asurf = T(f_strip, "ADD", CLR["textOnAccent"])
        else:
            screen.blit(grad(add_rect.w, add_rect.h,
                             mixc(CLR["raisedTop"], CLR["hoverTop"], hm),
                             mixc(CLR["raisedBot"], CLR["hoverBot"], hm), 7),
                        add_rect.topleft)
            pygame.draw.rect(screen, mixc(CLR["stroke"], CLR["strokeHover"], hm),
                             add_rect, width=1, border_radius=7)
            asurf = T(f_strip, "ADD", CLR["muted"])
        screen.blit(asurf, (add_rect.centerx - asurf.get_width() // 2,
                            add_rect.centery - asurf.get_height() // 2))

        list_top = y + TTS_IN_H + 6
        tts_list_rect = pygame.Rect(bx0, list_top, bx1 - bx0, TTS_LIST_H)
        n_ph = len(tts.phrases)
        tts_content_h = (n_ph * (TTS_ROW_H + TTS_ROW_GAP) - TTS_ROW_GAP + 4
                         if n_ph else 0)
        tts_target = max(0.0, min(tts_target,
                                  max(0.0, tts_content_h - TTS_LIST_H)))
        tts_scroll = step(tts_scroll, tts_target, dt, 0.14)
        screen.set_clip(tts_list_rect.clip(cards_area))
        tts_playing = {}
        sample_row = {id(tts.samples[t]): i for i, t in enumerate(tts.phrases)
                      if t in tts.samples}
        for v in list(state.tts_voices):
            try:
                samples, cur = v[0], v[1]
            except Exception:
                continue
            ri = sample_row.get(id(samples))
            if ri is not None and len(samples):
                tts_playing[ri] = max(tts_playing.get(ri, 0.0),
                                      cur / len(samples))
        if not n_ph:
            hint = T(f_small, "(none yet - Enter saves, Shift+Enter speaks)",
                     CLR["faint"])
            screen.blit(hint, (bx0 + 2, list_top + 6))
        for i in range(n_ph):
            ry = list_top - int(tts_scroll) + i * (TTS_ROW_H + TTS_ROW_GAP)
            if ry + TTS_ROW_H < tts_list_rect.y or ry > tts_list_rect.bottom:
                continue
            text = tts.phrases[i]
            rr = pygame.Rect(bx0, ry, bx1 - bx0, TTS_ROW_H)
            fl = tts.flash.get(i, 0) - now
            f = q8(min(1.0, max(0.0, fl / 0.25)) ** 2) if fl > 0 else 0.0
            hm = q8(hover_step(("ttsrow", i),
                               rr.collidepoint(mouse_pos)
                               and tts_list_rect.collidepoint(mouse_pos), dt))
            screen.blit(grad(rr.w, TTS_ROW_H,
                             mixc(mixc(CLR["raisedTop"], CLR["hoverTop"], hm),
                                  CLR["accentDim"], f),
                             mixc(mixc(CLR["raisedBot"], CLR["hoverBot"], hm),
                                  CLR["accentDim"], f), 7),
                        rr.topleft)
            prog = tts_playing.get(i)
            bcol = mixc(CLR["stroke"], CLR["accentBright"], f)
            if prog is not None and f < 0.05:
                bcol = mixc(CLR["raisedTop"], CLR["accent"], 0.45)
            elif hm > 0.4 and f < 0.05:
                bcol = CLR["strokeHover"]
            pygame.draw.rect(screen, bcol, rr, width=1, border_radius=7)
            dr = pygame.Rect(rr.right - 6 - 18, rr.centery - 9, 18, 18)
            dh = q8(hover_step(("ttsdel", i), dr.collidepoint(mouse_pos), dt))
            pygame.draw.rect(screen, mixc(CLR["strokeHover"], CLR["danger"], dh),
                             dr, width=1, border_radius=5)
            xs = T(f_badge, "x", CLR["danger"] if dh > 0.4 else CLR["muted"])
            screen.blit(xs, (dr.centerx - xs.get_width() // 2,
                             dr.centery - xs.get_height() // 2))
            tts_del_hit[i] = dr
            st_ = tts.status.get(text, "")
            if st_ == "ready" and text in tts.samples:
                dur = T(f_small, f"{len(tts.samples[text]) / SAMPLERATE:.1f}s",
                        CLR["accent"] if prog is not None else CLR["faint"])
            elif st_ == "error":
                dur = T(f_small, "err", CLR["danger"])
            else:
                a = 0.4 + 0.6 * (0.5 + 0.5 * float(np.sin(now * 2 * np.pi / 1.2)))
                dur = T(f_small, "...", mixc(CLR["raisedBot"], CLR["muted"], q8(a)))
            screen.blit(dur, (dr.x - 6 - dur.get_width(),
                              rr.centery - dur.get_height() // 2))
            nm = tts_trunc.get(text)
            if nm is None:
                if len(tts_trunc) > 400:
                    tts_trunc.clear()
                nm, name_w = text, rr.w - 92
                if f_label.render(nm, True, CLR["text"]).get_width() > name_w:
                    while nm and f_label.render(nm + "...", True,
                                                CLR["text"]).get_width() > name_w:
                        nm = nm[:-1]
                    nm += "..."
                tts_trunc[text] = nm
            hot = prog is not None or f > 0.05
            pygame.draw.polygon(screen,
                                CLR["accent"] if hot else
                                (CLR["muted"] if hm > 0.4 else CLR["faint"]),
                                [(rr.x + 10, rr.centery - 4),
                                 (rr.x + 10, rr.centery + 4),
                                 (rr.x + 16, rr.centery)])
            ns = T(f_label, nm,
                   CLR["text"] if (hot or hm > 0.4) else CLR["text2"])
            screen.blit(ns, (rr.x + 24, rr.centery - ns.get_height() // 2))
            if prog is not None:
                pygame.draw.rect(screen, CLR["accent"],
                                 pygame.Rect(rr.x, rr.bottom - 2,
                                             max(2, int(rr.w * min(1.0, prog))), 2))
            tts_row_hit[i] = rr
        if tts_content_h > TTS_LIST_H:
            track = pygame.Rect(bx1 - 3, tts_list_rect.y + 2, 3, TTS_LIST_H - 4)
            pygame.draw.rect(screen, CLR["scrollTrack"], track, border_radius=2)
            th = max(14, int(track.h * TTS_LIST_H / tts_content_h))
            tt_y = track.y + int((track.h - th)
                                 * (tts_scroll / max(1.0, tts_content_h
                                                     - TTS_LIST_H)))
            pygame.draw.rect(screen, CLR["scrollThumb"],
                             pygame.Rect(track.x, tt_y, 3, th), border_radius=2)
        screen.set_clip(cards_area)
        return list_top + TTS_LIST_H + 8

    # ------------------------------ dropdown picker (Scene / Preset / AI voice)
    DROP_ROWS = ("Scene", "Preset", "AI character")
    drop = None

    def open_dropdown(row_idx):
        nonlocal drop
        label = menu.items[row_idx].label
        n_builtin = 0
        if label == "Preset":
            presets = state.presets_all()
            n_builtin = len(presets) - len(state.user_presets)
            entries = sorted(((nm, i) for i, (nm, _p) in enumerate(presets)),
                             key=lambda e: e[0].lower())
            items = [(nm, lambda i=i: state.apply_preset(i))
                     for nm, i in entries]
            meta = [{"orig": i, "mut": i >= n_builtin} for _nm, i in entries]
            kind = "preset"
            cur = next((k for k, (_nm, i) in enumerate(entries)
                        if i == state.preset_idx), 0)
        elif label == "AI character" and ai is not None:
            entries = sorted(((p.stem, i) for i, p in enumerate(ai.voices)),
                             key=lambda e: e[0].lower())
            items = [(nm, lambda i=i: ai.select(i)) for nm, i in entries]
            meta = [{"orig": i, "mut": False} for _nm, i in entries]
            kind = "ai"
            cur = next((k for k, (_nm, i) in enumerate(entries)
                        if i == ai.sel), 0)
        elif label == "Scene" and scenes is not None:
            if not scenes.scenes:
                state.status_msg = "no scenes yet - dial a setup, then Save scene"
                state.status_at = time.time()
                return
            entries = sorted(((nm, i) for i, nm in enumerate(scenes.names())),
                             key=lambda e: e[0].lower())
            items = [(nm, lambda i=i: scenes.apply(i)) for nm, i in entries]
            meta = [{"orig": i, "mut": True} for _nm, i in entries]
            kind = "scene"
            cur = next((k for k, (_nm, i) in enumerate(entries)
                        if i == scenes.sel), 0)
        else:
            return
        # anchored to the row's on-screen rect (cards move; rects are truth)
        anchor = row_hit.get(row_idx)
        if anchor is None:
            return
        item_h, pad = 28, 4
        hint_h = 16 if any(m["mut"] for m in meta) else 0
        want = len(items) * item_h + pad * 2 + hint_h
        w = max(240, anchor.w - 12)
        x = min(anchor.x + 6, WIN_W - PAGE_PAD - w)
        below = VIEW_BOT - 6 - (anchor.bottom + 4)
        above = anchor.y - 4 - (CARD_TOP + 6)
        if below >= min(want, 200) or below >= above:
            h, y = min(want, below), anchor.bottom + 4
        else:
            h, y = min(want, above), anchor.y - 4 - min(want, above)
        rect = pygame.Rect(x, y, w, h)
        rows_h = h - hint_h
        max_scroll = max(0, want - hint_h - rows_h)
        scroll = min(max_scroll,
                     max(0, cur * item_h + pad - (rows_h - item_h) // 2))
        drop = {"items": items, "meta": meta, "kind": kind,
                "n_builtin": n_builtin, "hint_h": hint_h,
                "rect": rect, "item_h": item_h, "pad": pad,
                "sel": cur, "cur": cur, "scroll": scroll,
                "max_scroll": max_scroll, "row": row_idx, "mouse": None,
                "edit": None, "del_hit": {}}

    def drop_pick():
        nonlocal drop
        if drop and 0 <= drop["sel"] < len(drop["items"]):
            drop["items"][drop["sel"]][1]()
            menu.flash[drop["row"]] = time.time() + 0.25
        drop = None

    def drop_scroll_to(k):
        view_h = drop["rect"].h - drop["hint_h"] - drop["pad"] * 2
        top = k * drop["item_h"]
        if top < drop["scroll"]:
            drop["scroll"] = top
        elif top + drop["item_h"] > drop["scroll"] + view_h:
            drop["scroll"] = top + drop["item_h"] - view_h

    def drop_nav(d):
        drop["sel"] = (drop["sel"] + d) % len(drop["items"])
        drop_scroll_to(drop["sel"])

    def drop_close():
        nonlocal drop
        if drop is not None and drop["edit"] is not None and not tts_focus:
            try:
                pygame.key.stop_text_input()
            except Exception:
                pass
        drop = None

    def drop_refresh(sel=None, focus_orig=None):
        nonlocal drop
        if drop is None:
            return
        row_idx, scroll = drop["row"], drop["scroll"]
        drop = None
        open_dropdown(row_idx)
        if drop is None:
            return
        if focus_orig is not None:
            sel = next((k for k, m in enumerate(drop["meta"])
                        if m["orig"] == focus_orig), sel)
        if sel is not None and drop["items"]:
            drop["sel"] = min(sel, len(drop["items"]) - 1)
        drop["scroll"] = max(0, min(drop["max_scroll"], scroll))
        drop_scroll_to(drop["sel"])

    def drop_delete(k):
        if drop is None or not (0 <= k < len(drop["meta"])):
            return
        m = drop["meta"][k]
        if not m["mut"]:
            return
        if drop["kind"] == "scene":
            scenes.delete(m["orig"])
        else:
            state.delete_user_preset(m["orig"] - drop["n_builtin"])
        drop_refresh(sel=k)

    def drop_rename_start(k):
        if drop is None or not (0 <= k < len(drop["meta"])):
            return
        if not drop["meta"][k]["mut"]:
            return
        drop["sel"] = k
        drop_scroll_to(k)
        drop["edit"] = {"i": k, "text": ""}
        try:
            pygame.key.start_text_input()
        except Exception:
            pass

    def drop_rename_end(commit):
        ed, drop["edit"] = drop["edit"], None
        if not tts_focus:
            try:
                pygame.key.stop_text_input()
            except Exception:
                pass
        if not commit or ed is None or not ed["text"].strip():
            return
        m = drop["meta"][ed["i"]]
        if drop["kind"] == "scene":
            scenes.rename(m["orig"], ed["text"])
        else:
            state.rename_user_preset(m["orig"] - drop["n_builtin"], ed["text"])
        drop_refresh(focus_orig=m["orig"])

    def drop_event(event):
        nonlocal drop
        if drop["edit"] is not None:
            if event.type == pygame.TEXTINPUT:
                drop["edit"]["text"] = (drop["edit"]["text"] + event.text)[:40]
            elif event.type == pygame.KEYDOWN:
                held_keys.add(event.key)
                if event.key == pygame.K_BACKSPACE:
                    drop["edit"]["text"] = drop["edit"]["text"][:-1]
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    drop_rename_end(True)
                elif event.key == pygame.K_ESCAPE:
                    drop_rename_end(False)
            elif (event.type == pygame.MOUSEBUTTONDOWN
                    and event.button in (1, 2, 3)):
                drop_rename_end(True)
            return
        if event.type == pygame.KEYDOWN:
            held_keys.add(event.key)
            act = key_action(event.key)
            if   act == "up":     drop_nav(-1)
            elif act == "down":   drop_nav(+1)
            elif act == "select": drop_pick()
            elif event.key == pygame.K_DELETE: drop_delete(drop["sel"])
            elif event.key == pygame.K_F2:     drop_rename_start(drop["sel"])
            elif act == "back" or event.key == pygame.K_ESCAPE:
                drop_close()
        elif event.type == pygame.MOUSEWHEEL:
            drop["scroll"] = max(0, min(drop["max_scroll"],
                                        drop["scroll"]
                                        - event.y * drop["item_h"]))
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 3):
            r = drop["rect"]
            if not r.collidepoint(event.pos):
                drop_close()
                return
            if event.button == 1:
                di = next((k for k, rr in drop["del_hit"].items()
                           if rr.collidepoint(event.pos)), None)
                if di is not None:
                    drop_delete(di)
                    return
            i = (event.pos[1] - r.y - drop["pad"]
                 + int(drop["scroll"])) // drop["item_h"]
            if 0 <= i < len(drop["items"]):
                if event.button == 3:
                    drop_rename_start(i)
                else:
                    drop["sel"] = i
                    drop_pick()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
            drop_close()
        elif event.type == pygame.JOYBUTTONDOWN:
            if   event.button in pad_select: drop_pick()
            elif event.button in pad_back:   drop_close()
        elif event.type == pygame.JOYHATMOTION and event.value != (0, 0):
            if   event.value[1] ==  1: drop_nav(-1)
            elif event.value[1] == -1: drop_nav(+1)

    def do_select():
        """Focused stop select: header = collapse, dropdown rows = picker,
        the rest act directly."""
        st = focus_stop()
        if st is not None and st[0] == "hdr":
            toggle_collapse(st[1])
            return
        if menu.items[menu.sel].label in DROP_ROWS:
            open_dropdown(menu.sel)
        else:
            menu.on_select()

    relayout()
    if SCENE_IDX is not None:      # first focus: the scene chip, like before
        fsel = stop_of.get(("row", SCENE_IDX), 0)

    # test hook: events appended here are handled on the MAIN thread each
    # frame. The headless suites use this instead of pygame.event.post -
    # posting attribute-carrying events from another thread makes pygame
    # free the event dict while SDL still holds it (use-after-free).
    inject_q = ui_debug.setdefault("inject", deque())

    def drain_inject():
        evs = []
        while True:
            try:
                evs.append(inject_q.popleft())
            except IndexError:
                return evs

    # ================================================================== loop
    while not stop_flag.is_set():
        now = time.time()
        dt = min(0.1, now - last_t)
        last_t = now

        for event in list(pygame.event.get()) + drain_inject():
            if event.type == pygame.QUIT:
                stop_flag.set()

            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h),
                                                 pygame.RESIZABLE)
                relayout()

            elif event.type == pygame.DROPFILE:
                src = Path(event.file)
                if src.is_file() and src.suffix.lower() in (
                        ".wav", ".flac", ".ogg", ".mp3"):
                    try:
                        SOUNDS_DIR.mkdir(exist_ok=True)
                        dest = SOUNDS_DIR / src.name
                        if not dest.exists():
                            shutil.copy2(src, dest)
                        board.rescan()
                    except Exception as e:
                        state.status_msg = f"drop: {e}"
                        state.status_at = time.time()
                else:
                    state.status_msg = "drop: only wav / flac / ogg / mp3"
                    state.status_at = time.time()

            elif drop is not None and event.type in (
                    pygame.KEYDOWN, pygame.KEYUP, pygame.TEXTINPUT,
                    pygame.MOUSEBUTTONDOWN, pygame.MOUSEWHEEL,
                    pygame.MOUSEMOTION, pygame.JOYBUTTONDOWN,
                    pygame.JOYHATMOTION, pygame.JOYAXISMOTION):
                if event.type == pygame.KEYUP:
                    held_keys.discard(event.key)
                else:
                    drop_event(event)

            elif event.type == pygame.JOYDEVICEADDED:
                pygame.joystick.Joystick(event.device_index).init()
            elif event.type == pygame.JOYDEVICEREMOVED:
                pass

            elif event.type == pygame.TEXTINPUT:
                if edit is not None:
                    if all(c in "0123456789.,+-" for c in event.text):
                        edit["text"] = (edit["text"] + event.text)[:6]
                elif tts_focus:
                    tts_text = (tts_text + event.text)[:TTS_MAX_CHARS]

            elif event.type == pygame.KEYDOWN and edit is not None:
                held_keys.add(event.key)
                if event.key == pygame.K_BACKSPACE:
                    edit["text"] = edit["text"][:-1]
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    edit_commit()
                elif event.key == pygame.K_ESCAPE:
                    edit_close()

            elif event.type == pygame.KEYDOWN and tts_focus:
                held_keys.add(event.key)
                if event.key == pygame.K_BACKSPACE:
                    tts_text = tts_text[:-1]
                elif (event.key == pygame.K_v
                      and getattr(event, "mod", 0) & pygame.KMOD_CTRL):
                    tts_text = (tts_text
                                + get_clipboard_text())[:TTS_MAX_CHARS]
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    if getattr(event, "mod", 0) & pygame.KMOD_SHIFT:
                        tts.say(tts_text)
                    else:
                        tts_commit()
                elif event.key == pygame.K_ESCAPE:
                    tts_set_focus(False)

            elif event.type == pygame.KEYDOWN:
                repeat = event.key in held_keys
                held_keys.add(event.key)
                if event.key in clipmap:
                    if not repeat:
                        board.play_hot(clipmap[event.key])
                    continue
                if event.key == pygame.K_TAB:
                    if not repeat:
                        focus_tab(-1 if getattr(event, "mod", 0)
                                  & pygame.KMOD_SHIFT else +1)
                    continue
                act = key_action(event.key)
                if   act == "up":         focus_move(-1)
                elif act == "down":       focus_move(+1)
                elif act == "left":       go_left()
                elif act == "right":      go_right()
                elif repeat:              pass
                elif act == "select":     do_select()
                elif act == "back":       menu.on_back()
                elif act == "stop_clips": board.stop()
                elif act == "mute":       menu.toggle_mute()
                elif act == "page_next":  flip_page(+1)
                elif act == "page_prev":  flip_page(-1)

            elif event.type == pygame.KEYUP:
                held_keys.discard(event.key)
            elif event.type == pygame.WINDOWFOCUSLOST:
                held_keys.clear()

            elif event.type == pygame.MOUSEMOTION:
                if slider_drag is not None:
                    slider_set_from_x(slider_drag, event.pos[0])
                else:
                    idx = row_at(event.pos)
                    if idx is not None:
                        focus_row(idx)
                    else:
                        hk = next((k for k, r in hdr_hit.items()
                                   if r.collidepoint(event.pos)), None)
                        if hk is not None and ("hdr", hk) in stop_of:
                            fsel = stop_of[("hdr", hk)]

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                slider_drag = None

            elif event.type == pygame.MOUSEWHEEL:
                mpos = pygame.mouse.get_pos()
                if sb_grid_rect is not None and sb_grid_rect.collidepoint(mpos):
                    grid_target -= event.y * (TILE_H + GGAP)
                elif (tts_list_rect is not None
                        and tts_list_rect.collidepoint(mpos)):
                    tts_target -= event.y * (TTS_ROW_H + TTS_ROW_GAP)
                elif cards_area.collidepoint(mpos):
                    cards_target -= event.y * (ROW_HGT + ROW_GAP) * 2

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if cap_hit is not None and cap_hit.collidepoint(event.pos):
                    continue           # caption strip covers rows: swallow
                in_r = tts_btn_hit.get("input")
                tts_set_focus(in_r is not None and in_r.collidepoint(event.pos))
                if edit is not None and not value_hit.get(
                        edit["row"],
                        pygame.Rect(0, 0, 0, 0)).collidepoint(event.pos):
                    edit_commit()
                hit = next((k for k, r in strip_hit.items()
                            if r.collidepoint(event.pos)), None)
                if hit is not None:
                    strip_press[hit] = now
                if hit == "mic":
                    board.toggle_mic()
                elif hit == "pause":
                    board.toggle_pause()
                elif hit == "stop":
                    board.stop()
                elif hit == "page":
                    flip_page(+1)
                elif (sy := next((k for k, r in sys_hit.items()
                                  if r.collidepoint(event.pos)), None)) is not None:
                    strip_press[("sys", sy)] = now
                    menu.sel = sy
                    st_ = stop_of.get(("row", sy))
                    if st_ is not None:
                        fsel = st_
                    menu.on_select()
                elif (hk := next((k for k, r in hdr_hit.items()
                                  if r.collidepoint(event.pos)), None)) is not None:
                    if ("hdr", hk) in stop_of:
                        fsel = stop_of[("hdr", hk)]
                    toggle_collapse(hk)
                elif (r := tts_btn_hit.get("add")) is not None \
                        and r.collidepoint(event.pos):
                    tts_commit()
                elif (di := next((i for i, r in tts_del_hit.items()
                                  if r.collidepoint(event.pos)), None)) is not None:
                    tts.delete(di)
                elif (ti := next((i for i, r in tts_row_hit.items()
                                  if r.collidepoint(event.pos)), None)) is not None:
                    tts.play(ti)
                elif (sb_grid_rect is not None
                        and sb_grid_rect.collidepoint(event.pos) and (
                        ci := next((c for c, r in grid_hit.items()
                                    if r.collidepoint(event.pos)), None)) is not None):
                    board.play(ci)
                elif (si := next((k for k, rr in slider_hit.items()
                                  if rr.collidepoint(event.pos)), None)) is not None:
                    focus_row(si)
                    slider_drag = si
                    slider_set_from_x(si, event.pos[0])
                elif (vi := next((k for k, rr in value_hit.items()
                                  if rr.collidepoint(event.pos)), None)) is not None:
                    focus_row(vi)
                    edit_open(vi)
                else:
                    idx = row_at(event.pos)
                    if idx is not None:
                        focus_row(idx)
                        it = menu.items[idx]
                        on_row = arrow_hit is not None and arrow_hit[0] == idx
                        if it.adjust and on_row and arrow_hit[1].collidepoint(event.pos):
                            go_left()
                        elif it.adjust and on_row and arrow_hit[2].collidepoint(event.pos):
                            go_right()
                        elif it.label in DROP_ROWS:
                            open_dropdown(idx)
                        elif it.select:
                            menu.on_select()

            elif event.type == pygame.JOYBUTTONDOWN:
                if   event.button in pad_select: do_select()
                elif event.button in pad_back:   menu.on_back()
                elif event.button in pad_stop:   board.stop()

            elif event.type == pygame.JOYHATMOTION and event.value != (0, 0):
                jnow = time.time()
                if jnow - joy_last >= cooldown:
                    joy_last = jnow
                    hx, hy = event.value
                    if   hy ==  1: focus_move(-1)
                    elif hy == -1: focus_move(+1)
                    elif hx == -1: go_left()
                    elif hx ==  1: go_right()

            elif (event.type == pygame.JOYAXISMOTION and event.axis in (0, 1)
                  and abs(event.value) > threshold):
                jnow = time.time()
                if jnow - joy_last >= cooldown:
                    joy_last = jnow
                    if event.axis == 0:
                        go_left() if event.value < 0 else go_right()
                    else:
                        focus_move(-1) if event.value < 0 else focus_move(+1)

        if state.clips_version != clips_seen:
            rebuild_grid()
        if layout_dirty[0]:
            build_layout()

        # ------------------------------------------------------------- draw
        mouse_pos = pygame.mouse.get_pos()
        screen.fill(CLR["bg"])

        # header: wordmark + segmented mic meter
        screen.blit(grad(WIN_W, HEADER_H, CLR["headerTop"], CLR["headerBot"]),
                    (0, 0))
        pygame.draw.line(screen, CLR["strokeSoft"], (0, HEADER_H - 1),
                         (WIN_W, HEADER_H - 1))
        bx = 16
        for bh in (6, 13, 9, 16):
            pygame.draw.rect(screen, CLR["accent"],
                             pygame.Rect(bx, HEADER_H // 2 + 8 - bh, 3, bh),
                             border_radius=1)
            bx += 5
        wv = TT(f_word, "VOICE", CLR["text"], 1)
        wb = TT(f_word, "BOX", CLR["accent"], 1)
        wy = (HEADER_H - wv.get_height()) // 2
        screen.blit(wv, (bx + 6, wy))
        screen.blit(wb, (bx + 6 + wv.get_width(), wy))

        level = state.in_level
        db = 20.0 * float(np.log10(max(level, 1e-4)))
        seg_target = max(0.0, min(22.0, (db + 48.0) / 48.0 * 22.0))
        if seg_target > meter_lit:
            meter_lit = min(seg_target, meter_lit + dt / 0.040 * 22.0)
        else:
            meter_lit = max(seg_target, meter_lit - dt / 0.240 * 22.0)
        if meter_lit >= peak_lit:
            peak_lit, peak_at = meter_lit, now
        elif now - peak_at > 0.9:
            peak_lit = max(meter_lit, peak_lit - dt / 0.300 * 22.0)
        mx_right = WIN_W - 16
        db_s = (T(f_small, "MUTED", CLR["danger"]) if state.mic_muted
                else T(f_small, f"{max(db, -60.0):5.1f} dB", CLR["muted"]))
        screen.blit(db_s, (mx_right - db_s.get_width(),
                           (HEADER_H - db_s.get_height()) // 2))
        seg_x0 = mx_right - 52 - 10 - (22 * 7 - 2)
        lit_n, peak_n = int(meter_lit), int(peak_lit)
        for si in range(22):
            col = CLR["meterOff"]
            if si < lit_n:
                col = (CLR["success"] if si < 16 else
                       CLR["warning"] if si < 19 else CLR["danger"])
            elif si == peak_n and peak_n > lit_n:
                col = CLR["peak"]
            pygame.draw.rect(screen, col,
                             pygame.Rect(seg_x0 + si * 7, (HEADER_H - 13) // 2, 5, 13),
                             border_radius=1)
        mic_s = TT(f_hdr, "MIC", CLR["faint"], 1)
        screen.blit(mic_s, (seg_x0 - 10 - mic_s.get_width(),
                            (HEADER_H - mic_s.get_height()) // 2))

        # ------------------------------------------- system bar (chip toggles)
        sys_hit.clear()
        pygame.draw.rect(screen, CLR["headerBot"],
                         pygame.Rect(0, SYS_TOP, WIN_W, SYS_H))
        pygame.draw.line(screen, CLR["strokeSoft"], (0, SYS_TOP + SYS_H - 1),
                         (WIN_W, SYS_TOP + SYS_H - 1))
        sy_c = SYS_TOP + SYS_H // 2
        sx = PAGE_PAD
        st = focus_stop()
        for i, lab, chip, act_fn in SYSTEM_BAR:
            it = menu.items[i]
            danger = chip in ("REC", "QUIT")
            active = False
            if act_fn is not None:
                try:
                    active = bool(act_fn())
                except Exception:
                    pass
            if chip == "REC" and recorder is not None and recorder.on:
                text = menu._rec_label()               # live REC m:ss
            elif chip == "IN" and engine is not None:
                text = "IN: " + engine.short_name("input", 12)
            elif chip == "OUT" and engine is not None:
                text = "OUT: " + engine.short_name("output", 12)
            elif act_fn is not None:
                text = f"{chip}: {'ON' if active else 'OFF'}"
            else:
                text = chip
            acol = CLR["danger"] if danger and active else \
                (CLR["danger"] if chip == "QUIT" else CLR["accent"])
            tint = DANGER_TINT if danger else ACCENT_TINT
            ts0 = T(f_strip, text, acol if active else CLR["muted"])
            w = ts0.get_width() + 18 + (10 if active else 0)
            if chip == "QUIT":                         # pinned to the right
                r = pygame.Rect(WIN_W - PAGE_PAD - w, sy_c - 13, w, 26)
            else:
                r = pygame.Rect(sx, sy_c - 13, w, 26)
            hm = q8(hover_step(("sys", i), r.collidepoint(mouse_pos), dt))
            pm = max(0.0, 1.0 - (now - strip_press.get(("sys", i), 0)) / 0.08) \
                if strip_press.get(("sys", i)) else 0.0
            if active:
                screen.blit(grad(w, 26, tint[0], tint[1], 6), r.topleft)
                pygame.draw.rect(screen, mixc(CLR["bg"], acol, 0.45), r,
                                 width=1, border_radius=6)
            else:
                top = mixc(CLR["raisedTop"], CLR["hoverTop"], hm)
                bot = mixc(CLR["raisedBot"], CLR["hoverBot"], hm)
                if pm > 0:
                    top = mixc(top, CLR["active"], q8(pm))
                    bot = mixc(bot, CLR["active"], q8(pm))
                screen.blit(grad(w, 26, top, bot, 6), r.topleft)
                pygame.draw.rect(screen,
                                 mixc(CLR["stroke"], CLR["strokeHover"], hm),
                                 r, width=1, border_radius=6)
            if st == ("row", i):                       # keyboard focus ring
                screen.blit(glow(w, 26, CLR["accent"], 6),
                            (r.x - G_PAD, r.y - G_PAD))
                pygame.draw.rect(screen, CLR["accent"], r, width=1,
                                 border_radius=6)
            tx = r.x + 9
            if active:
                pygame.draw.circle(screen, acol, (tx + 2, r.centery), 2)
                tx += 10
            ts = T(f_strip, text, acol if active else
                   (CLR["text"] if hm > 0.5 or st == ("row", i)
                    else CLR["muted"]))
            screen.blit(ts, (tx, r.centery - ts.get_height() // 2))
            sys_hit[i] = r
            if i in KEEP_ROWS:
                row_hit[i] = r                         # hover/nav like a row
            if chip != "QUIT":
                sx = r.right + 6

        # -------------------------------------------------------- scene strip
        scene_hit.clear()
        pygame.draw.rect(screen, CLR["paneLeft"],
                         pygame.Rect(0, SCENE_TOP, WIN_W, SCENE_H))
        pygame.draw.line(screen, CLR["strokeSoft"],
                         (0, SCENE_TOP + SCENE_H - 1),
                         (WIN_W, SCENE_TOP + SCENE_H - 1))
        scy = SCENE_TOP + SCENE_H // 2
        hs = TT(f_hdr, "SCENES", CLR["faint"], 2)
        screen.blit(hs, (PAGE_PAD + 4, scy - hs.get_height() // 2))
        sx = PAGE_PAD + 4 + hs.get_width() + 12
        if SCENE_IDX is not None and scenes is not None:
            name = scenes.applied or "-"
            focused = focus_stop() == ("row", SCENE_IDX)
            ns = T(f_labelF if focused else f_label, name,
                   CLR["text"] if focused else CLR["text2"])
            w = ns.get_width() + 34
            r = pygame.Rect(sx, scy - 13, w, 26)
            hm = q8(hover_step(("scene", "pick"), r.collidepoint(mouse_pos), dt))
            fl = menu.flash.get(SCENE_IDX, 0) - now
            if fl > 0:
                f = q8(min(1.0, fl / 0.25) ** 2)
                screen.blit(grad(w, 26, mixc(CLR["raisedTop"], CLR["accentDim"], f),
                                 mixc(CLR["raisedBot"], CLR["accentDim"], f), 6),
                            r.topleft)
            else:
                screen.blit(grad(w, 26,
                                 mixc(CLR["raisedTop"], CLR["hoverTop"], hm),
                                 mixc(CLR["raisedBot"], CLR["hoverBot"], hm), 6),
                            r.topleft)
            if focused:
                screen.blit(glow(w, 26, CLR["accent"], 6), (r.x - G_PAD, r.y - G_PAD))
                pygame.draw.rect(screen, CLR["accent"], r, width=1, border_radius=6)
            else:
                pygame.draw.rect(screen, mixc(CLR["stroke"], CLR["strokeHover"], hm),
                                 r, width=1, border_radius=6)
            screen.blit(ns, (r.x + 10, r.centery - ns.get_height() // 2))
            ar = T(f_val, "v", CLR["faint"])
            screen.blit(ar, (r.right - 16, r.centery - ar.get_height() // 2))
            row_hit[SCENE_IDX] = r
            scene_hit["pick"] = r
            sx = r.right + 8
        if SAVE_SCENE_IDX is not None and scenes is not None:
            focused = focus_stop() == ("row", SAVE_SCENE_IDX)
            ss = T(f_strip, "SAVE SCENE", CLR["accent"] if focused else CLR["muted"])
            w = ss.get_width() + 22
            r = pygame.Rect(sx, scy - 13, w, 26)
            hm = q8(hover_step(("scene", "save"), r.collidepoint(mouse_pos), dt))
            screen.blit(grad(w, 26, mixc(CLR["raisedTop"], CLR["hoverTop"], hm),
                             mixc(CLR["raisedBot"], CLR["hoverBot"], hm), 6),
                        r.topleft)
            if focused:
                screen.blit(glow(w, 26, CLR["accent"], 6), (r.x - G_PAD, r.y - G_PAD))
                pygame.draw.rect(screen, CLR["accent"], r, width=1, border_radius=6)
            else:
                pygame.draw.rect(screen, mixc(CLR["stroke"], CLR["strokeHover"], hm),
                                 r, width=1, border_radius=6)
            screen.blit(ss, (r.x + 11, r.centery - ss.get_height() // 2))
            row_hit[SAVE_SCENE_IDX] = r
            scene_hit["save"] = r
            badge = T(f_badge, "CTRL+ALT+S", CLR["faint"])
            br = pygame.Rect(r.right + 8, scy - 9, badge.get_width() + 12, 18)
            pygame.draw.rect(screen, CLR["strokeSoft"], br, width=1,
                             border_radius=4)
            screen.blit(badge, (br.x + 6, br.centery - badge.get_height() // 2))
            hint = T(f_small, "scene = fx + ai + tts, one press", CLR["faint"])
            if br.right + 12 + hint.get_width() < SB_X - 8:
                screen.blit(hint, (br.right + 12, scy - hint.get_height() // 2))

        # ------------------------------------------------------- feature cards
        st = focus_stop()
        if st is not None and fsel in stop_y and stop_y[fsel] >= 0:
            area_h = cards_area.height
            y0, y1 = stop_y[fsel], stop_y[fsel] + ROW_HGT + CARD_GAP
            if y0 - cards_target < 6:
                cards_target = max(0.0, y0 - 6)
            elif y1 - cards_target > area_h - 6:
                cards_target = y1 - (area_h - 6)
        cards_target = max(0.0, min(cards_target,
                                    max(0.0, cards_content_h - cards_area.height + 8)))
        cards_scroll = step(cards_scroll, cards_target, dt, 0.14)

        for d_ in (row_hit, hdr_hit, slider_hit, slider_track, value_hit,
                   tts_btn_hit, tts_row_hit, tts_del_hit):
            keep_scene = {k: v for k, v in d_.items()
                          if d_ is row_hit and k in KEEP_ROWS}
            d_.clear()
            d_.update(keep_scene)
        arrow_hit = None
        tts_list_rect = None
        screen.set_clip(cards_area)
        base_y = CARD_TOP + 8 - int(cards_scroll)
        for col_x, col in columns:
            for c, cy0, ch_ in col:
                r = pygame.Rect(col_x, base_y + int(cy0), CARD_W, ch_)
                if r.bottom < CARD_TOP or r.y > VIEW_BOT:
                    hdr_hit[c["key"]] = pygame.Rect(0, -99, 0, 0)
                    continue
                is_collapsed = c["key"] in collapsed
                dim = (not is_collapsed and c["dim"] is not None
                       and c["dim"]())
                # card shell
                pygame.draw.rect(screen, CLR["cardBg"], r, border_radius=8)
                pygame.draw.rect(screen, CLR["strokeSoft"], r, width=1,
                                 border_radius=8)
                # header (a focus stop; click/enter collapses)
                hr = pygame.Rect(r.x, r.y, r.w, CARD_HDR)
                hdr_hit[c["key"]] = hr
                hdr_focused = st == ("hdr", c["key"])
                hm = q8(hover_step(("hdr", c["key"]),
                                   hr.collidepoint(mouse_pos), dt))
                if hdr_focused:
                    screen.blit(glow(hr.w, hr.h, CLR["accent"], 8),
                                (hr.x - G_PAD, hr.y - G_PAD))
                    screen.blit(grad(hr.w, hr.h, ACCENT_TINT[0], ACCENT_TINT[1], 8),
                                hr.topleft)
                    pygame.draw.rect(screen, CLR["accent"], hr, width=1,
                                     border_radius=8)
                elif hm > 0:
                    top = mixc(CLR["cardBg"], CLR["hoverTop"], hm)
                    screen.blit(grad(hr.w, hr.h, top, top, 8), hr.topleft)
                # chevron
                chx, chy = r.x + 13, r.y + CARD_HDR // 2
                ccol = CLR["accent"] if hdr_focused else CLR["faint"]
                if is_collapsed:
                    pygame.draw.polygon(screen, ccol,
                                        [(chx - 2, chy - 4), (chx - 2, chy + 4),
                                         (chx + 4, chy)])
                else:
                    pygame.draw.polygon(screen, ccol,
                                        [(chx - 4, chy - 2), (chx + 4, chy - 2),
                                         (chx, chy + 4)])
                ts = TT(f_hdr, c["title"],
                        CLR["text"] if hdr_focused else CLR["faint"], 2)
                screen.blit(ts, (r.x + 26, r.y + (CARD_HDR - ts.get_height()) // 2))
                # right side of the header: LED dot + summary
                hx_r = r.right - 10
                if c["summary"] is not None:
                    try:
                        summ = str(c["summary"]())[:26]
                    except Exception:
                        summ = ""
                    if summ:
                        sms = T(f_small, summ, CLR["muted"])
                        hx_r -= sms.get_width()
                        screen.blit(sms, (hx_r, r.y + (CARD_HDR - sms.get_height()) // 2))
                        hx_r -= 8
                if c["dot"] is not None:
                    on = False
                    try:
                        on = bool(c["dot"]())
                    except Exception:
                        pass
                    pygame.draw.circle(screen, CLR["accent"] if on else CLR["faint"],
                                       (hx_r - 3, r.y + CARD_HDR // 2), 2)
                if is_collapsed:
                    continue
                pygame.draw.line(screen, CLR["strokeSoft"],
                                 (r.x + 1, r.y + CARD_HDR - 1),
                                 (r.right - 2, r.y + CARD_HDR - 1))
                # body rows
                ry_ = r.y + CARD_HDR + PAD_TOP
                if c["key"] == "tts":
                    ry_ = draw_tts_body(r, ry_)
                for i in c["rows"]:
                    it = menu.items[i]
                    rr = pygame.Rect(r.x + PAD_X, ry_, r.w - 2 * PAD_X, ROW_HGT)
                    row_hit[i] = rr
                    focused = st == ("row", i)
                    fl = menu.flash.get(i, 0) - now
                    if focused:
                        screen.blit(glow(rr.w, ROW_HGT, CLR["accent"], 7),
                                    (rr.x - G_PAD, rr.y - G_PAD))
                        screen.blit(grad(rr.w, ROW_HGT, ACCENT_TINT[0],
                                         ACCENT_TINT[1], 7), rr.topleft)
                        pygame.draw.rect(screen, CLR["accent"], rr, width=1,
                                         border_radius=7)
                    elif fl > 0:
                        f = q8(min(1.0, fl / 0.25) ** 2)
                        screen.blit(grad(rr.w, ROW_HGT,
                                         mixc(CLR["cardBg"], CLR["accentDim"], f),
                                         mixc(CLR["cardBg"], CLR["accentDim"], f * 0.8),
                                         7), rr.topleft)
                    else:
                        hm = q8(hover_step(("row", i),
                                           rr.collidepoint(mouse_pos), dt))
                        if hm > 0:
                            top = mixc(CLR["cardBg"], CLR["hoverTop"], hm)
                            screen.blit(grad(rr.w, ROW_HGT, top, top, 7),
                                        rr.topleft)
                            if hm > 0.4:
                                pygame.draw.rect(screen, CLR["strokeHover"], rr,
                                                 width=1, border_radius=7)
                    if (it.label == "AI voice" and ai is not None
                            and ai.status == "error" and not focused):
                        screen.blit(grad(rr.w, ROW_HGT, DANGER_TINT[0],
                                         DANGER_TINT[1], 7), rr.topleft)
                        pygame.draw.rect(screen,
                                         mixc(CLR["cardBg"], CLR["danger"], 0.35),
                                         rr, width=1, border_radius=7)
                    ls = T(f_labelF if focused else f_label, it.label,
                           CLR["text"] if focused else CLR["text2"])
                    screen.blit(ls, (rr.x + 10,
                                     rr.y + (ROW_HGT - ls.get_height()) // 2))
                    if it.value_fn is not None:
                        draw_value(rr, i, it, it.value_fn(), focused, now)
                    elif it.select:
                        vs = T(f_val, "-", CLR["faint"])
                        screen.blit(vs, (rr.right - 10 - vs.get_width(),
                                         rr.y + (ROW_HGT - vs.get_height()) // 2))
                    ry_ += ROW_HGT + ROW_GAP
                if dim:
                    ov = pygame.Surface((r.w - 2, r.h - CARD_HDR - 1),
                                        pygame.SRCALPHA)
                    ov.fill((*CLR["cardBg"], 158))     # body at ~38% (design 03)
                    screen.blit(ov, (r.x + 1, r.y + CARD_HDR))
        screen.set_clip(None)
        if cards_content_h > cards_area.height:
            track = pygame.Rect(SB_X - COL_GAP + 3, CARD_TOP + 4, 3,
                                cards_area.height - 8)
            pygame.draw.rect(screen, CLR["scrollTrack"], track, border_radius=2)
            th = max(24, int(track.height * cards_area.height / cards_content_h))
            tt_y = track.y + int((track.height - th)
                                 * (cards_scroll
                                    / max(1.0, cards_content_h - cards_area.height)))
            pygame.draw.rect(screen, CLR["scrollThumb"],
                             pygame.Rect(track.x, tt_y, 3, th), border_radius=2)

        # -------------------------------------------------- soundboard column
        pygame.draw.rect(screen, CLR["cardBg"], sb_rect, border_radius=8)
        pygame.draw.rect(screen, CLR["strokeSoft"], sb_rect, width=1,
                         border_radius=8)
        sb_ts = TT(f_hdr, "SOUNDBOARD", CLR["faint"], 2)
        screen.blit(sb_ts, (sb_rect.x + 12,
                            sb_rect.y + (CARD_HDR - sb_ts.get_height()) // 2))
        cnt = T(f_small, f"{len(state.clips)} SOUNDS", CLR["faint"])
        screen.blit(cnt, (sb_rect.right - 10 - cnt.get_width(),
                          sb_rect.y + (CARD_HDR - cnt.get_height()) // 2))
        pygame.draw.line(screen, CLR["strokeSoft"],
                         (sb_rect.x + 1, sb_rect.y + CARD_HDR - 1),
                         (sb_rect.right - 2, sb_rect.y + CARD_HDR - 1))
        # chips: TO MIC / PAUSE / STOP / PG
        strip_hit.clear()
        chips = [("mic", "TO MIC", state.clips_to_mic, CLR["accent"], ACCENT_TINT),
                 ("pause", "PAUSED" if state.clips_paused else "PAUSE",
                  state.clips_paused, CLR["warning"], WARN_TINT),
                 ("stop", "STOP", False, CLR["accent"], None)]
        n_pages = board.page_count()
        if n_pages > 1:
            chips.append(("page", f"PG {state.clip_page + 1}/{n_pages}",
                          False, CLR["accent"], None))
        sx = sb_rect.x + PAD_X
        chy0 = sb_rect.y + CARD_HDR + 6
        for key, lab, active, acol, tint in chips:
            base_ts = T(f_strip, lab, acol if active else CLR["muted"])
            w = base_ts.get_width() + 18 + (10 if active else 0)
            r = pygame.Rect(sx, chy0, w, CHIP_H)
            hm = q8(hover_step(("strip2", key), r.collidepoint(mouse_pos), dt))
            pm = max(0.0, 1.0 - (now - strip_press.get(key, 0)) / 0.08) \
                if strip_press.get(key) else 0.0
            if active and tint:
                screen.blit(grad(w, CHIP_H, tint[0], tint[1], 6), r.topleft)
                pygame.draw.rect(screen, mixc(CLR["bg"], acol, 0.45), r,
                                 width=1, border_radius=6)
            else:
                top = mixc(CLR["raisedTop"], CLR["hoverTop"], hm)
                bot = mixc(CLR["raisedBot"], CLR["hoverBot"], hm)
                if pm > 0:
                    top = mixc(top, CLR["active"], q8(pm))
                    bot = mixc(bot, CLR["active"], q8(pm))
                screen.blit(grad(w, CHIP_H, top, bot, 6), r.topleft)
                pygame.draw.rect(screen, mixc(CLR["stroke"], CLR["strokeHover"], hm),
                                 r, width=1, border_radius=6)
            tx = r.x + 9
            if active:
                pygame.draw.circle(screen, acol, (tx + 2, r.centery), 2)
                tx += 10
            ts = T(f_strip, lab, acol if active else
                   (CLR["text"] if hm > 0.5 else CLR["muted"]))
            screen.blit(ts, (tx, r.centery - ts.get_height() // 2))
            strip_hit[key] = r
            sx = r.right + 6
        # clip volume row
        gy_top = chy0 + CHIP_H + 4
        if CLIP_VOL_IDX is not None:
            it = menu.items[CLIP_VOL_IDX]
            rr = pygame.Rect(sb_rect.x + PAD_X, gy_top, SB_W - 2 * PAD_X, ROW_HGT)
            row_hit[CLIP_VOL_IDX] = rr
            focused = st == ("row", CLIP_VOL_IDX)
            if focused:
                screen.blit(glow(rr.w, ROW_HGT, CLR["accent"], 7),
                            (rr.x - G_PAD, rr.y - G_PAD))
                screen.blit(grad(rr.w, ROW_HGT, ACCENT_TINT[0], ACCENT_TINT[1], 7),
                            rr.topleft)
                pygame.draw.rect(screen, CLR["accent"], rr, width=1, border_radius=7)
            ls = T(f_labelF if focused else f_label, "Clip volume",
                   CLR["text"] if focused else CLR["text2"])
            screen.blit(ls, (rr.x + 10, rr.y + (ROW_HGT - ls.get_height()) // 2))
            draw_value(rr, CLIP_VOL_IDX, it, it.value_fn(), focused, now)
            gy_top = rr.bottom + 6

        # the grid (internal scroll)
        sb_grid_rect = pygame.Rect(sb_rect.x + PAD_X, gy_top,
                                   SB_W - 2 * PAD_X, sb_rect.bottom - 24 - gy_top)
        G_X0 = sb_grid_rect.x
        grid_target = max(0.0, min(grid_target,
                                   max(0.0, grid_content_h - sb_grid_rect.height)))
        grid_scroll = step(grid_scroll, grid_target, dt, 0.14)
        screen.set_clip(sb_grid_rect)
        grid_hit.clear()
        playing = {}
        sources = [state.voices]
        if getattr(board, "player", None) is not None:
            sources.append(board.player.voices)
        for src in sources:
            for v in list(src):
                try:
                    samples, cur = v[0], v[1]
                except Exception:
                    continue
                pidx = clip_by_id.get(id(samples))
                if pidx is not None and len(samples):
                    playing[pidx] = max(playing.get(pidx, 0.0), cur / len(samples))
        if not state.clips:
            hint = T(f_small, "(no sounds - put files in ./sounds)", CLR["faint"])
            screen.blit(hint, (G_X0, sb_grid_rect.y + 8))
        gy0 = sb_grid_rect.y - int(grid_scroll)
        first_row = max(0, int(grid_scroll) // (TILE_H + GGAP))
        last_row = min(grid_rows,
                       (int(grid_scroll) + sb_grid_rect.height) // (TILE_H + GGAP) + 2)
        for ci in range(first_row * COLS, min(len(state.clips), last_row * COLS)):
            g_r, g_c = divmod(ci, COLS)
            r = pygame.Rect(G_X0 + g_c * (TILE_W + GGAP),
                            gy0 + g_r * (TILE_H + GGAP), TILE_W, TILE_H)
            fl = board.flash.get(ci, 0) - now
            f = q8(min(1.0, max(0.0, fl / 0.25)) ** 2) if fl > 0 else 0.0
            hm = q8(hover_step(("tile", ci),
                               r.collidepoint(mouse_pos)
                               and sb_grid_rect.collidepoint(mouse_pos), dt))
            prog = playing.get(ci)
            if f > 0.05:
                gsurf = glow(TILE_W, TILE_H, CLR["accent"], 8)
                gsurf.set_alpha(int(255 * f))
                screen.blit(gsurf, (r.x - G_PAD, r.y - G_PAD))
                gsurf.set_alpha(255)
            top = mixc(mixc(CLR["raisedTop"], CLR["hoverTop"], hm), CLR["accentDim"], f)
            bot = mixc(mixc(CLR["raisedBot"], CLR["hoverBot"], hm), CLR["accentDim"], f)
            screen.blit(grad(TILE_W, TILE_H, top, bot, 8), r.topleft)
            bcol = mixc(CLR["stroke"], CLR["accentBright"], f)
            if prog is not None and f < 0.05:
                bcol = mixc(CLR["raisedTop"], CLR["accent"], 0.45)
            elif hm > 0.4 and f < 0.05:
                bcol = CLR["strokeHover"]
            pygame.draw.rect(screen, bcol, r, width=1, border_radius=8)
            ns = T(f_tile, disp_names[ci], CLR["text"])
            screen.blit(ns, (r.x + 8, r.y + 7))
            if prog is not None:
                ds = T(f_small, clip_secs[ci], CLR["accent"])
                dy = r.bottom - 9 - ds.get_height()
                py_ = dy + ds.get_height() // 2
                pygame.draw.polygon(screen, CLR["accent"],
                                    [(r.x + 8, py_ - 3), (r.x + 8, py_ + 3),
                                     (r.x + 13, py_)])
                screen.blit(ds, (r.x + 17, dy))
                pygame.draw.rect(screen, CLR["accent"],
                                 pygame.Rect(r.x, r.bottom - 2,
                                             max(2, int(TILE_W * min(1.0, prog))), 2))
            else:
                ds = T(f_small, clip_secs[ci],
                       CLR["muted"] if hm > 0.5 else CLR["faint"])
                screen.blit(ds, (r.x + 8, r.bottom - 9 - ds.get_height()))
            pg0 = state.clip_page * 9
            if pg0 <= ci < pg0 + 9:
                hot = prog is not None or f > 0.05
                brect = pygame.Rect(r.right - 8 - 15, r.y + 7, 15, 15)
                pygame.draw.rect(screen,
                                 mixc(CLR["bg"], CLR["accent"], 0.45) if hot
                                 else CLR["strokeHover"],
                                 brect, width=1, border_radius=4)
                bs = T(f_badge, str(ci - pg0 + 1),
                       CLR["accent"] if hot else CLR["muted"])
                screen.blit(bs, (brect.centerx - bs.get_width() // 2,
                                 brect.centery - bs.get_height() // 2))
            grid_hit[ci] = r
        screen.blit(grad(sb_grid_rect.width, 22, (13, 16, 20, 0), (13, 16, 20, 255)),
                    (sb_grid_rect.x, sb_grid_rect.bottom - 22))
        screen.set_clip(None)
        if grid_content_h > sb_grid_rect.height:
            track = pygame.Rect(sb_rect.right - 7, sb_grid_rect.y + 2, 3,
                                sb_grid_rect.height - 4)
            pygame.draw.rect(screen, CLR["scrollTrack"], track, border_radius=2)
            th = max(24, int(track.height * sb_grid_rect.height / grid_content_h))
            tt_y = track.y + int((track.height - th)
                                 * (grid_scroll / max(1.0, grid_content_h
                                                      - sb_grid_rect.height)))
            pygame.draw.rect(screen, CLR["scrollThumb"],
                             pygame.Rect(track.x, tt_y, 3, th), border_radius=2)
        dhint = T(f_small, f"drop audio files here · 1-9 fire page "
                           f"{state.clip_page + 1}", CLR["faint"])
        screen.blit(dhint, (sb_rect.x + PAD_X,
                            sb_rect.bottom - 12 - dhint.get_height() // 2))

        # ------------------------------------------------------------ footer
        screen.blit(grad(WIN_W, FOOTER_H, CLR["footerTop"], CLR["footerBot"]),
                    (0, VIEW_BOT))
        pygame.draw.line(screen, CLR["strokeSoft"], (0, VIEW_BOT),
                         (WIN_W, VIEW_BOT))
        fy = VIEW_BOT + FOOTER_H // 2
        cur_err = engine.error if engine is not None else err_line
        cur_dev = engine.dev_line if engine is not None else dev_line
        if cur_err:
            es = T(f_foot, cur_err, CLR["danger"])
            screen.blit(es, (14, fy - es.get_height() // 2))
        elif cur_dev:
            fx = 14
            if "->" in cur_dev:
                a_, b_ = cur_dev.split("->", 1)
                parts = ((a_.strip(), CLR["muted"]), (" > ", CLR["accent"]),
                         (b_.strip(), CLR["muted"]))
            else:
                parts = ((cur_dev, CLR["muted"]),)
            for ptxt, pcol in parts:
                psur = T(f_foot, ptxt, pcol)
                screen.blit(psur, (fx, fy - psur.get_height() // 2))
                fx += psur.get_width()

        chip_up = state.status_msg and (now - state.status_at) < 4.38
        if engine is not None and engine.latency_ms and not chip_up:
            stat = f"{engine.latency_ms:.0f} ms latency"
            if state.status_count:
                stat += f" · {state.status_count} drops"
            ss = T(f_foot, stat,
                   CLR["warning"] if state.status_count else CLR["faint"])
            screen.blit(ss, (WIN_W - 14 - ss.get_width(),
                             fy - ss.get_height() // 2))

        if state.status_msg:
            t_ = now - state.status_at
            alpha = (t_ / 0.16 if t_ < 0.16 else
                     1.0 if t_ < 4.16 else
                     max(0.0, 1.0 - (t_ - 4.16) / 0.22) if t_ < 4.38 else 0.0)
            if alpha > 0:
                col = (CLR["danger"] if "error" in state.status_msg.lower()
                       else CLR["warning"])
                cs = T(f_foot, state.status_msg, col)
                cw = cs.get_width() + 30
                chip = pygame.Surface((cw, 20), pygame.SRCALPHA)
                pygame.draw.rect(chip, (*col, 26), chip.get_rect(), border_radius=6)
                pygame.draw.rect(chip, (*col, 102), chip.get_rect(), width=1,
                                 border_radius=6)
                pygame.draw.circle(chip, col, (11, 10), 2)
                chip.blit(cs, (18, (20 - cs.get_height()) // 2))
                chip.set_alpha(int(255 * alpha))
                rise = int(8 * (1.0 - min(1.0, t_ / 0.16)))
                screen.blit(chip, (WIN_W - 14 - cw, fy - 10 + rise))

        # --------------------------------------- incoming-speech caption strip
        cap_hit = None
        if listener is not None:
            cap_lines = listener.caption_lines(now)
            if cap_lines:
                key = (tuple(cap_lines), WIN_W)
                if cap_cache["key"] != key:
                    ch = f_foot.get_height() + 6
                    panel_h = 10 + ch * len(cap_lines)
                    cap = pygame.Surface((WIN_W, panel_h), pygame.SRCALPHA)
                    cap.fill((10, 13, 18, 216))
                    cy = 5
                    for ln in cap_lines:
                        tag, _, rest = ln.partition("]")
                        ts = T(f_foot, tag + "]", CLR["accent"])
                        cap.blit(ts, (14, cy))
                        cap.blit(T(f_foot, rest, CLR["text"]),
                                 (14 + ts.get_width(), cy))
                        cy += ch
                    cap_cache["key"], cap_cache["surf"] = key, cap
                cap = cap_cache["surf"]
                cap_hit = pygame.Rect(0, VIEW_BOT - cap.get_height(),
                                      WIN_W, cap.get_height())
                screen.blit(cap, cap_hit.topleft)
                pygame.draw.line(screen, CLR["strokeSoft"],
                                 cap_hit.topleft, cap_hit.topright)

        # --------------------------------------------- dropdown picker overlay
        ui_debug["drop_info"] = None if drop is None else {
            "rect": drop["rect"], "item_h": drop["item_h"],
            "pad": drop["pad"], "scroll": drop["scroll"],
            "n": len(drop["items"])}
        if drop is not None:
            r = drop["rect"]
            if (drop["edit"] is None and drop["mouse"] != mouse_pos
                    and r.collidepoint(mouse_pos)):
                mi = (mouse_pos[1] - r.y - drop["pad"]
                      + int(drop["scroll"])) // drop["item_h"]
                if 0 <= mi < len(drop["items"]):
                    drop["sel"] = mi
            drop["mouse"] = mouse_pos
            screen.blit(grad(r.w, r.h, CLR["hoverTop"], CLR["raisedBot"], 8),
                        r.topleft)
            pygame.draw.rect(screen, mixc(CLR["stroke"], CLR["accent"], 0.35),
                             r, width=1, border_radius=8)
            drop["del_hit"] = {}
            rows_clip = pygame.Rect(r.x + 1, r.y + 2, r.w - 2,
                                    r.h - 4 - drop["hint_h"])
            screen.set_clip(rows_clip)
            y0 = r.y + drop["pad"] - int(drop["scroll"])
            for i, (nm, _cb) in enumerate(drop["items"]):
                ir = pygame.Rect(r.x + 4, y0 + i * drop["item_h"],
                                 r.w - 12, drop["item_h"] - 2)
                if ir.bottom < r.y or ir.y > rows_clip.bottom:
                    continue
                ed = drop["edit"]
                if ed is not None and ed["i"] == i:
                    box = pygame.Rect(ir.x + 2, ir.centery - 11, ir.w - 4, 22)
                    screen.blit(grad(box.w, box.h, CLR["headerBot"],
                                     CLR["paneLeft"], 5), box.topleft)
                    pygame.draw.rect(screen, CLR["accent"], box, width=1,
                                     border_radius=5)
                    txt = ed["text"]
                    ts = (T(f_labelF, txt, CLR["text"]) if txt
                          else T(f_label, nm, CLR["faint"]))
                    screen.set_clip(rows_clip.clip(box.inflate(-10, 0)))
                    screen.blit(ts, (box.x + 8,
                                     box.centery - ts.get_height() // 2))
                    screen.set_clip(rows_clip)
                    if txt and (now * 2.0) % 2 < 1:
                        cx = min(box.x + 8 + ts.get_width() + 2, box.right - 6)
                        pygame.draw.line(screen, CLR["accent"],
                                         (cx, box.centery - 7),
                                         (cx, box.centery + 7))
                    continue
                if i == drop["sel"]:
                    screen.blit(grad(ir.w, ir.h, ACCENT_TINT[0],
                                     ACCENT_TINT[1], 6), ir.topleft)
                    pygame.draw.rect(screen, CLR["accent"], ir,
                                     width=1, border_radius=6)
                if i == drop["cur"]:
                    pygame.draw.circle(screen, CLR["accent"],
                                       (ir.x + 11, ir.centery), 2)
                ns = T(f_labelF if i == drop["sel"] else f_label, nm,
                       CLR["text"] if i == drop["sel"] else CLR["text2"])
                if drop["meta"][i]["mut"] and i == drop["sel"]:
                    screen.set_clip(rows_clip.clip(
                        pygame.Rect(ir.x, ir.y, ir.w - 32, ir.h)))
                    screen.blit(ns, (ir.x + 22,
                                     ir.centery - ns.get_height() // 2))
                    screen.set_clip(rows_clip)
                    dr = pygame.Rect(ir.right - 24, ir.centery - 9, 18, 18)
                    dh = q8(hover_step(("dropdel", i),
                                       dr.collidepoint(mouse_pos), dt))
                    pygame.draw.rect(screen,
                                     mixc(CLR["strokeHover"], CLR["danger"], dh),
                                     dr, width=1, border_radius=5)
                    xs = T(f_badge, "x",
                           CLR["danger"] if dh > 0.4 else CLR["muted"])
                    screen.blit(xs, (dr.centerx - xs.get_width() // 2,
                                     dr.centery - xs.get_height() // 2))
                    drop["del_hit"][i] = dr
                else:
                    screen.blit(ns, (ir.x + 22,
                                     ir.centery - ns.get_height() // 2))
            screen.set_clip(r.inflate(-2, -4))
            if drop["hint_h"]:
                hint = T(f_small, "right-click renames · x deletes",
                         CLR["faint"])
                screen.blit(hint, (r.x + 12, r.bottom - drop["hint_h"]
                                   + (drop["hint_h"] - hint.get_height()) // 2
                                   - 2))
            if drop["max_scroll"] > 0:
                rh = r.h - drop["hint_h"]
                track = pygame.Rect(r.right - 6, r.y + 4, 3, rh - 8)
                pygame.draw.rect(screen, CLR["scrollTrack"], track,
                                 border_radius=2)
                th = max(18, int(track.h * rh / (drop["max_scroll"] + rh)))
                ty = track.y + int((track.h - th)
                                   * (drop["scroll"] / drop["max_scroll"]))
                pygame.draw.rect(screen, CLR["scrollThumb"],
                                 pygame.Rect(track.x, ty, 3, th),
                                 border_radius=2)
            screen.set_clip(None)

        pygame.display.flip()
        clock.tick(30)          # 30 fps is plenty for a menu and halves GIL load
        frame_no += 1
        if shot_path and frame_no >= shot_frames:
            try:
                pygame.image.save(screen, shot_path)
            except Exception:
                pass
            stop_flag.set()

    pygame.quit()
