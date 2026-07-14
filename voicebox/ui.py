"""The VoiceBox window: settings menu, soundboard grid, TTS panel (pygame).
Skin ported from design/VoiceBox Skin.dc.html - see run_ui's docstring."""
import time

import numpy as np

from .config import BASE_DIR, SAMPLERATE, TTS_MAX_CHARS, WINDOW_SIZE
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
                 hotkeys=None, engine=None, recorder=None, tts=None):
        self.state = state
        self.stop_flag = stop_flag
        self.monitor = monitor
        self.board = board if board is not None else Board(state)
        self.ai = ai
        self.hotkeys = hotkeys
        self.engine = engine
        self.recorder = recorder
        self.tts = tts
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
        if recorder is not None:
            self.items.append(MenuItem(
                "Record output",
                lambda: (f"REC {int(time.time() - recorder.started_at) // 60}:"
                         f"{int(time.time() - recorder.started_at) % 60:02d}"
                         if recorder.on else "off"),
                select=recorder.toggle,
                adjust=lambda d: recorder.toggle()))
        if hotkeys is not None:
            self.items.append(MenuItem(
                "Global hotkeys",
                lambda: "ON" if hotkeys.on else "off",
                select=hotkeys.toggle,
                adjust=lambda d: hotkeys.toggle()))
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

    def toggle_mute(self):
        with self.state.lock:
            self.state.mic_muted = not self.state.mic_muted

    def _save_preset(self):
        name = self.state.save_user_preset()
        self.state.status_msg = f"saved \"{name}\" (edit user_presets.json to rename)"
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

    def _toggle_tts_fx(self):
        with self.state.lock:
            self.state.tts_fx = not self.state.tts_fx

    def _toggle_ai_fx(self):
        """AI voice through the effect chain (pitch, echo, ...) on/off."""
        with self.state.lock:
            self.state.ai_fx = not self.state.ai_fx
        if self.ai is not None:
            self.ai.set_fx(self.state.ai_fx)

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


def run_ui(state, stop_flag, dev_line, err_line="", monitor=None, board=None,
           ai=None, tts=None, hotkeys=None, engine=None, recorder=None):
    """VoiceBox skin, ported from design/VoiceBox Skin.dc.html.

    Faithful to the tokens JSON + motion spec in that file: Space Grotesk for
    labels / JetBrains Mono for values, cyan accent with a single glow recipe
    reserved for focus, sliding focus highlight (120ms), eased pixel scrolling
    (140ms), tile trigger flash (250ms easeIn), segmented mic meter with
    peak-hold, and toast-style status chips. easeOut(t)=1-(1-t)^2 per spec.
    """
    import pygame
    pygame.init()
    pygame.display.set_caption("VoiceBox")
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
    try:                          # OS-level minimum = the design's base size
        from pygame._sdl2.video import Window as _SDLWindow
        _SDLWindow.from_display_module().minimum_size = WINDOW_SIZE
    except Exception:
        pass
    clock = pygame.time.Clock()
    pygame.key.set_repeat(320, 110)           # held arrows auto-repeat

    cfg = load_controls()
    keymap, clipmap = build_keymap(cfg, pygame)
    pad = cfg["gamepad"]

    # controls.json is user-edited: coerce wrong-shaped values instead of
    # crashing the event loop (e.g. "select": 0 instead of [0]).
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
        "footerBot":   (13, 16, 21),
    }
    ACCENT_TINT = ((51, 214, 255, 26), (51, 214, 255, 13))
    DANGER_TINT = ((255, 77, 94, 20), (255, 77, 94, 10))
    WARN_TINT = ((255, 177, 61, 26), (255, 177, 61, 15))

    def mixc(a, b, t):
        return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t),
                int(a[2] + (b[2] - a[2]) * t))

    # fonts: bundled TTFs (assets/fonts) with system fallbacks
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
                recorder, tts)
    board = menu.board
    kb_action = {a: keys for a, keys in keymap.items()}

    def key_action(key):
        for action, keys in kb_action.items():
            if key in keys:
                return action
        return None

    # ------------------------------------------------------------- caches
    # Rendering is cached (text, gradients, glows): re-rendering every frame
    # competes with the audio callback for the GIL and can cause dropouts.
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
    HEADER_H, FOOTER_H = 52, 32
    LEFT_W = 370
    VIEW_TOP = HEADER_H
    L_X, L_RIGHT = 10, LEFT_W - 14
    L_W = L_RIGHT - L_X
    ROW_HGT, ROW_GAP = 34, 2
    HDR_FIRST, HDR_HGT = 24, 30
    LIST_PAD_TOP = 6
    G_X = LEFT_W + 14
    COLS, GGAP, TILE_H = 3, 8, 62
    STRIP_Y, STRIP_H = VIEW_TOP + 10, 30
    GRID_TOP = STRIP_Y + STRIP_H + 10
    # TTS panel: bottom strip of the right pane (header / input / phrase list)
    TTS_H = 210
    TTS_IN_H = 30
    TTS_ROW_H, TTS_ROW_GAP = 30, 2

    # window-size-dependent geometry, owned by relayout(): the window is
    # resizable (drag edges / Aero snap); the left pane keeps its width, the
    # soundboard grid and TTS panel absorb the extra space.
    WIN_W = WIN_H = 0
    VIEW_BOT = VIEW_H = G_RIGHT = TILE_W = 0
    TTS_TOP = TTS_IN_Y = TTS_LIST_TOP = 0
    LIST_RECT = TTS_LIST_RECT = GRID_RECT = None

    def relayout():
        nonlocal WIN_W, WIN_H, VIEW_BOT, VIEW_H, G_RIGHT, TILE_W, TTS_TOP, \
            TTS_IN_Y, TTS_LIST_TOP, LIST_RECT, TTS_LIST_RECT, GRID_RECT
        # layout never goes below the design's base size, even if the OS
        # ignores the window minimum (drawing past the surface just clips)
        WIN_W = max(screen.get_width(), WINDOW_SIZE[0])
        WIN_H = max(screen.get_height(), WINDOW_SIZE[1])
        VIEW_BOT = WIN_H - FOOTER_H
        VIEW_H = VIEW_BOT - VIEW_TOP
        G_RIGHT = WIN_W - 14 - 8               # 8px scroll gutter
        TILE_W = (G_RIGHT - G_X - GGAP * (COLS - 1)) // COLS
        LIST_RECT = pygame.Rect(0, VIEW_TOP, LEFT_W, VIEW_H)
        TTS_TOP = VIEW_BOT - TTS_H
        TTS_IN_Y = TTS_TOP + 30
        TTS_LIST_TOP = TTS_IN_Y + TTS_IN_H + 8
        TTS_LIST_RECT = pygame.Rect(LEFT_W + 1, TTS_LIST_TOP,
                                    WIN_W - LEFT_W - 1,
                                    VIEW_BOT - TTS_LIST_TOP)
        GRID_RECT = pygame.Rect(LEFT_W + 1, GRID_TOP, WIN_W - LEFT_W - 1,
                                TTS_TOP - GRID_TOP)
        # tile width follows the window: re-truncate + measure grid labels
        rebuild_grid()

    SECTION_OF = {
        "Preset": "VOICE", "Save preset": "VOICE", "Pitch": "VOICE",
        "Mic": "VOICE", "Noise gate": "VOICE",
        "Robot voice": "EFFECTS", "Helmet doubler": "EFFECTS",
        "Grit / growl": "EFFECTS", "Reverb": "EFFECTS", "Echo": "EFFECTS",
        "Radio voice": "EFFECTS", "Bass boost": "EFFECTS",
        "Voice volume": "EFFECTS", "Clip volume": "EFFECTS",
        "AI voice": "AI", "AI character": "AI", "AI voice FX": "AI",
        "TTS voice FX": "TTS", "TTS volume": "TTS",
        "TTS voice": "TTS", "TTS rate": "TTS",
        "Sounds to mic": "SOUNDS", "Pause sounds": "SOUNDS",
        "Stop all sounds": "SOUNDS", "Rescan sounds": "SOUNDS",
        "Record output": "SYSTEM", "Global hotkeys": "SYSTEM",
        "Input device": "DEVICES", "Output device": "DEVICES",
        "Quit": "SYSTEM",
    }
    layout, row_pos = [], {}
    y_acc, last_sec = LIST_PAD_TOP, None
    for i, it in enumerate(menu.items):
        sec = SECTION_OF.get(it.label, last_sec)
        if sec is not None and sec != last_sec:
            hh = HDR_FIRST if not layout else HDR_HGT
            layout.append(("hdr", sec, y_acc, hh))
            y_acc += hh
            last_sec = sec
        layout.append(("row", i, y_acc, ROW_HGT))
        row_pos[i] = y_acc
        y_acc += ROW_HGT + ROW_GAP
    content_h = y_acc + 20

    grid_rows, grid_content_h, clips_seen = 0, 0, -1
    clip_by_id, disp_names, clip_secs = {}, [], []

    def rebuild_grid():
        """Grid caches (truncated labels, sizes). Rebuilt after each rescan,
        so labels only re-measure when the clip list actually changed."""
        nonlocal grid_rows, grid_content_h, clips_seen
        clips_seen = state.clips_version
        grid_rows = (len(state.clips) + COLS - 1) // COLS
        grid_content_h = (grid_rows * (TILE_H + GGAP) - GGAP + 20
                          if state.clips else 0)
        clip_by_id.clear()
        clip_by_id.update({id(c): i for i, c in enumerate(state.clips)})
        disp_names.clear()
        clip_secs.clear()
        name_max = TILE_W - 20 - 22
        for nm, c in zip(state.clip_names, state.clips):
            if f_tile.render(nm, True, CLR["text"]).get_width() > name_max:
                while nm and f_tile.render(nm + "...", True,
                                           CLR["text"]).get_width() > name_max:
                    nm = nm[:-1]
                nm += "..."
            disp_names.append(nm)
            clip_secs.append(f"{len(c) / SAMPLERATE:.1f}s")
    relayout()

    # ----------------------------------------------------------- motion state
    list_scroll = list_target = 0.0
    grid_scroll = grid_target = 0.0
    tts_scroll = tts_target = 0.0
    tts_text, tts_focus = "", False
    tts_trunc = {}                # phrase -> truncated display string
    focus_y = float(row_pos.get(menu.sel, LIST_PAD_TOP))
    hover_mix = {}                # element key -> 0..1 hover blend
    nudge = {"i": -1, "at": 0.0, "side": 0}
    strip_press = {}
    meter_lit = 0.0
    peak_lit, peak_at = 0.0, 0.0
    row_hit, strip_hit, grid_hit = {}, {}, {}
    tts_row_hit, tts_del_hit, tts_btn_hit = {}, {}, {}
    arrow_hit = None              # (row, "<" rect, ">" rect) from the last draw
    slider_hit, slider_track, value_hit = {}, {}, {}   # numeric rows, per draw
    slider_drag = None            # row index while a slider knob is dragged
    edit = None                   # {"row": i, "text": str} while typing a value
    last_t = time.time()

    def step(cur, target, dt, dur):
        """One easeOut lerp step toward target (spec: easeOut = 1-(1-t)^2)."""
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
        return round(t * 8) / 8    # quantize blends so gradient cache stays small

    def row_at(pos):
        """Menu row under the mouse (uses the rects from the last draw)."""
        for i, r in row_hit.items():
            if r.collidepoint(pos):
                return i
        return None

    def tts_commit():
        nonlocal tts_text
        if tts.add(tts_text):
            tts_text = ""

    def tts_set_focus(on):
        """Textbox focus: while on, the keyboard belongs to the textbox."""
        nonlocal tts_focus
        if on == tts_focus:
            return
        tts_focus = on
        try:
            (pygame.key.start_text_input if on else pygame.key.stop_text_input)()
        except Exception:
            pass

    def flip_page(d):
        """Step the hotkey page and scroll the grid to show it."""
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
        edit = {"row": i, "text": ""}   # empty box; current value = placeholder
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
        """(font, text color, LED dot color or None) by value semantics."""
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
        # value on the right: an input box while editing, else clickable text
        if edit is not None and edit["row"] == i:
            box = pygame.Rect(r.right - 8 - 54, cy - 11, 54, 22)
            screen.blit(grad(box.w, box.h, CLR["headerBot"], CLR["paneLeft"], 5),
                        box.topleft)
            pygame.draw.rect(screen, CLR["accent"], box, width=1, border_radius=5)
            txt = edit["text"]
            if txt:
                ts = T(f_valF, txt, CLR["text"])
            else:                          # empty: current value as placeholder
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
            if vh > 0.3:                   # hint: the number is clickable
                pygame.draw.rect(screen, mixc(CLR["paneLeft"], CLR["accent"], 0.35),
                                 vr.inflate(12, 8), width=1, border_radius=5)
            screen.blit(ts, vr.topleft)
            value_hit[i] = vr.inflate(12, 10)
            val_left = vr.x - 6
        # slider track + knob in the middle of the row
        tx1 = min(val_left - 12, r.right - 10 - 54 - 12)
        tx0 = max(r.x + 148, tx1 - 124)
        track = pygame.Rect(tx0, cy - 2, tx1 - tx0, 4)
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

    # -------------------------------------- dropdown picker (Preset / AI voice)
    # Pressing the Preset or AI character row opens an alphabetical list
    # anchored to the row; while open it owns keyboard, mouse and controller.
    DROP_ROWS = ("Preset", "AI character")
    drop = None                   # dict(items, rect, sel, cur, scroll, ...) | None

    def open_dropdown(row_idx):
        nonlocal drop
        label = menu.items[row_idx].label
        if label == "Preset":
            presets = state.presets_all()      # built-ins + user presets
            entries = sorted(((nm, i) for i, (nm, _p) in enumerate(presets)),
                             key=lambda e: e[0].lower())
            items = [(nm, lambda i=i: state.apply_preset(i))
                     for nm, i in entries]
            cur = next((k for k, (_nm, i) in enumerate(entries)
                        if i == state.preset_idx), 0)
        elif ai is not None:
            entries = sorted(((p.stem, i) for i, p in enumerate(ai.voices)),
                             key=lambda e: e[0].lower())
            items = [(nm, lambda i=i: ai.select(i)) for nm, i in entries]
            cur = next((k for k, (_nm, i) in enumerate(entries)
                        if i == ai.sel), 0)
        else:
            return
        item_h, pad = 28, 4
        ry = VIEW_TOP - int(list_scroll) + row_pos[row_idx]
        want = len(items) * item_h + pad * 2
        below = VIEW_BOT - 6 - (ry + ROW_HGT + 4)
        above = ry - 4 - (VIEW_TOP + 6)
        if below >= min(want, 200) or below >= above:
            h, y = min(want, below), ry + ROW_HGT + 4
        else:
            h, y = min(want, above), ry - 4 - min(want, above)
        rect = pygame.Rect(L_X + 10, y, L_W - 20, h)
        max_scroll = max(0, want - h)
        scroll = min(max_scroll,
                     max(0, cur * item_h + pad - (h - item_h) // 2))
        drop = {"items": items, "rect": rect, "item_h": item_h, "pad": pad,
                "sel": cur, "cur": cur, "scroll": scroll,
                "max_scroll": max_scroll, "row": row_idx, "mouse": None}

    def drop_pick():
        nonlocal drop
        if drop and 0 <= drop["sel"] < len(drop["items"]):
            drop["items"][drop["sel"]][1]()
            menu.flash[drop["row"]] = time.time() + 0.25
        drop = None

    def drop_nav(d):
        drop["sel"] = (drop["sel"] + d) % len(drop["items"])
        view_h = drop["rect"].h - drop["pad"] * 2
        top = drop["sel"] * drop["item_h"]
        if top < drop["scroll"]:
            drop["scroll"] = top
        elif top + drop["item_h"] > drop["scroll"] + view_h:
            drop["scroll"] = top + drop["item_h"] - view_h

    def drop_event(event):
        """All input routes here while the picker is open."""
        nonlocal drop
        if event.type == pygame.KEYDOWN:
            held_keys.add(event.key)
            act = key_action(event.key)
            if   act == "up":     drop_nav(-1)
            elif act == "down":   drop_nav(+1)
            elif act == "select": drop_pick()
            elif act == "back" or event.key == pygame.K_ESCAPE:
                drop = None
        elif event.type == pygame.MOUSEWHEEL:
            drop["scroll"] = max(0, min(drop["max_scroll"],
                                        drop["scroll"]
                                        - event.y * drop["item_h"]))
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            r = drop["rect"]
            if r.collidepoint(event.pos):
                i = (event.pos[1] - r.y - drop["pad"]
                     + int(drop["scroll"])) // drop["item_h"]
                if 0 <= i < len(drop["items"]):
                    drop["sel"] = i
                    drop_pick()
            else:
                drop = None                # click elsewhere just closes
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drop = None
        elif event.type == pygame.JOYBUTTONDOWN:
            if   event.button in pad_select: drop_pick()
            elif event.button in pad_back:   drop = None
        elif event.type == pygame.JOYHATMOTION and event.value != (0, 0):
            if   event.value[1] ==  1: drop_nav(-1)
            elif event.value[1] == -1: drop_nav(+1)

    def do_select():
        """Row select: dropdown rows open the picker, the rest act directly."""
        if menu.items[menu.sel].label in DROP_ROWS:
            open_dropdown(menu.sel)
        else:
            menu.on_select()

    # ================================================================== loop
    while not stop_flag.is_set():
        now = time.time()
        dt = min(0.1, now - last_t)
        last_t = now

        if state.clips_version != clips_seen:      # rescan happened
            rebuild_grid()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_flag.set()

            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h),
                                                 pygame.RESIZABLE)
                relayout()

            elif drop is not None and event.type in (
                    pygame.KEYDOWN, pygame.KEYUP, pygame.MOUSEBUTTONDOWN,
                    pygame.MOUSEWHEEL, pygame.MOUSEMOTION,
                    pygame.JOYBUTTONDOWN, pygame.JOYHATMOTION,
                    pygame.JOYAXISMOTION):
                if event.type == pygame.KEYUP:
                    held_keys.discard(event.key)
                else:
                    drop_event(event)

            elif event.type == pygame.JOYDEVICEADDED:
                pygame.joystick.Joystick(event.device_index).init()
            elif event.type == pygame.JOYDEVICEREMOVED:
                pass                                   # instance dies on its own

            elif event.type == pygame.TEXTINPUT:
                if edit is not None:
                    if all(c in "0123456789.,+-" for c in event.text):
                        edit["text"] = (edit["text"] + event.text)[:6]
                elif tts_focus:
                    tts_text = (tts_text + event.text)[:TTS_MAX_CHARS]

            elif event.type == pygame.KEYDOWN and edit is not None:
                # the value box owns the keyboard (digits are clip hotkeys!)
                held_keys.add(event.key)
                if event.key == pygame.K_BACKSPACE:
                    edit["text"] = edit["text"][:-1]
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    edit_commit()
                elif event.key == pygame.K_ESCAPE:
                    edit_close()

            elif event.type == pygame.KEYDOWN and tts_focus:
                # the textbox owns the keyboard: no menu nav, no clip hotkeys
                held_keys.add(event.key)
                if event.key == pygame.K_BACKSPACE:
                    tts_text = tts_text[:-1]
                elif event.key == pygame.K_v and event.mod & pygame.KMOD_CTRL:
                    tts_text = (tts_text
                                + get_clipboard_text())[:TTS_MAX_CHARS]
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    tts_commit()
                elif event.key == pygame.K_ESCAPE:
                    tts_set_focus(False)

            elif event.type == pygame.KEYDOWN:
                # auto-repeat is only for navigation/adjust; a held clip key must
                # not stack a new copy of the clip every repeat interval
                repeat = event.key in held_keys
                held_keys.add(event.key)
                if event.key in clipmap:
                    if not repeat:                 # page-relative; play() bounds-checks
                        board.play_hot(clipmap[event.key])
                    continue
                act = key_action(event.key)
                if   act == "up":         menu.on_up()
                elif act == "down":       menu.on_down()
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
                held_keys.clear()         # KEYUPs are lost on focus change

            elif event.type == pygame.MOUSEMOTION:
                if slider_drag is not None:        # live drag: follow the mouse
                    slider_set_from_x(slider_drag, event.pos[0])
                else:
                    idx = row_at(event.pos)
                    if idx is not None:
                        menu.sel = idx

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                slider_drag = None

            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                if mx >= LEFT_W and my >= TTS_TOP:           # over the TTS panel
                    tts_target -= event.y * (TTS_ROW_H + TTS_ROW_GAP)
                elif mx >= LEFT_W:                           # over the grid pane
                    grid_target -= event.y * (TILE_H + GGAP)
                else:
                    for _ in range(abs(event.y)):
                        menu.on_up() if event.y > 0 else menu.on_down()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                in_r = tts_btn_hit.get("input")
                tts_set_focus(in_r is not None and in_r.collidepoint(event.pos))
                if edit is not None and not value_hit.get(
                        edit["row"],
                        pygame.Rect(0, 0, 0, 0)).collidepoint(event.pos):
                    edit_commit()          # clicking elsewhere confirms it
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
                elif hit == "hear":
                    menu._toggle_monitor()
                elif hit == "page":
                    flip_page(+1)
                elif (r := tts_btn_hit.get("add")) is not None \
                        and r.collidepoint(event.pos):
                    tts_commit()
                elif (r := tts_btn_hit.get("fx")) is not None \
                        and r.collidepoint(event.pos):
                    menu._toggle_tts_fx()
                elif (di := next((i for i, r in tts_del_hit.items()
                                  if r.collidepoint(event.pos)), None)) is not None:
                    tts.delete(di)
                elif (ti := next((i for i, r in tts_row_hit.items()
                                  if r.collidepoint(event.pos)), None)) is not None:
                    tts.play(ti)
                elif GRID_RECT.collidepoint(event.pos) and (
                        ci := next((c for c, r in grid_hit.items()
                                    if r.collidepoint(event.pos)), None)) is not None:
                    board.play(ci)
                elif (si := next((k for k, rr in slider_hit.items()
                                  if rr.collidepoint(event.pos)), None)) is not None:
                    menu.sel = si
                    slider_drag = si       # jump to the click, then live-drag
                    slider_set_from_x(si, event.pos[0])
                elif (vi := next((k for k, rr in value_hit.items()
                                  if rr.collidepoint(event.pos)), None)) is not None:
                    menu.sel = vi
                    edit_open(vi)          # type the number directly
                else:
                    idx = row_at(event.pos)
                    if idx is not None:
                        menu.sel = idx
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
                    if   hy ==  1: menu.on_up()
                    elif hy == -1: menu.on_down()
                    elif hx == -1: go_left()
                    elif hx ==  1: go_right()

            # left stick only (axes 0/1): filter BEFORE touching the cooldown so
            # trigger/right-stick events can't silently eat the nav timer
            elif (event.type == pygame.JOYAXISMOTION and event.axis in (0, 1)
                  and abs(event.value) > threshold):
                jnow = time.time()
                if jnow - joy_last >= cooldown:
                    joy_last = jnow
                    if event.axis == 0:
                        go_left() if event.value < 0 else go_right()
                    else:
                        menu.on_up() if event.value < 0 else menu.on_down()

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
        if seg_target > meter_lit:                     # attack 40ms / decay 240ms
            meter_lit = min(seg_target, meter_lit + dt / 0.040 * 22.0)
        else:
            meter_lit = max(seg_target, meter_lit - dt / 0.240 * 22.0)
        if meter_lit >= peak_lit:
            peak_lit, peak_at = meter_lit, now
        elif now - peak_at > 0.9:                      # hold 900ms, fall 300ms
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

        # ------------------------------------------------- left pane: settings
        pygame.draw.rect(screen, CLR["paneLeft"], LIST_RECT)
        pygame.draw.line(screen, CLR["strokeSoft"], (LEFT_W, VIEW_TOP),
                         (LEFT_W, VIEW_BOT))

        ry = row_pos.get(menu.sel, LIST_PAD_TOP)
        if ry - list_target < 6:                       # keep focus in view
            list_target = max(0.0, ry - 6)
        elif ry + ROW_HGT - list_target > VIEW_H - 6:
            list_target = ry + ROW_HGT - (VIEW_H - 6)
        list_target = max(0.0, min(list_target, max(0.0, content_h - VIEW_H)))
        list_scroll = step(list_scroll, list_target, dt, 0.14)
        focus_y = step(focus_y, float(ry), dt, 0.12)

        screen.set_clip(LIST_RECT)
        row_hit.clear()
        slider_hit.clear()
        slider_track.clear()
        value_hit.clear()
        arrow_hit = None
        base_y = VIEW_TOP - int(list_scroll)

        # sliding focus highlight: ring + tint + the one glow
        fr = pygame.Rect(L_X, base_y + int(focus_y), L_W, ROW_HGT)
        screen.blit(glow(L_W, ROW_HGT, CLR["accent"], 7), (fr.x - G_PAD, fr.y - G_PAD))
        screen.blit(grad(L_W, ROW_HGT, ACCENT_TINT[0], ACCENT_TINT[1], 7), fr.topleft)
        pygame.draw.rect(screen, CLR["accent"], fr, width=1, border_radius=7)

        for kind, data, ly, lh in layout:
            sy = base_y + ly
            if sy + lh < VIEW_TOP or sy > VIEW_BOT:
                continue
            if kind == "hdr":
                hs = TT(f_hdr, data, CLR["faint"], 2)
                ty = sy + lh - hs.get_height() - 4
                screen.blit(hs, (L_X + 4, ty))
                lyy = ty + hs.get_height() // 2
                pygame.draw.line(screen, CLR["strokeSoft"],
                                 (L_X + 4 + hs.get_width() + 8, lyy), (L_RIGHT - 4, lyy))
                continue
            i = data
            it = menu.items[i]
            r = pygame.Rect(L_X, sy, L_W, ROW_HGT)
            row_hit[i] = r
            focused = (i == menu.sel)
            fl = menu.flash.get(i, 0) - now            # select-confirm flash
            if fl > 0:
                f = q8(min(1.0, fl / 0.25) ** 2)
                screen.blit(grad(L_W, ROW_HGT,
                                 mixc(CLR["paneLeft"], CLR["accentDim"], f),
                                 mixc(CLR["paneLeft"], CLR["accentDim"], f * 0.8), 7),
                            r.topleft)
            elif not focused:
                hm = q8(hover_step(("row", i), r.collidepoint(mouse_pos), dt))
                if hm > 0:
                    top = mixc(CLR["paneLeft"], CLR["hoverTop"], hm)
                    screen.blit(grad(L_W, ROW_HGT, top, top, 7), r.topleft)
                    if hm > 0.4:
                        pygame.draw.rect(screen, CLR["strokeHover"], r,
                                         width=1, border_radius=7)
            if (it.label == "AI voice" and ai is not None
                    and ai.status == "error" and not focused):
                screen.blit(grad(L_W, ROW_HGT, DANGER_TINT[0], DANGER_TINT[1], 7),
                            r.topleft)
                pygame.draw.rect(screen, mixc(CLR["paneLeft"], CLR["danger"], 0.35),
                                 r, width=1, border_radius=7)

            ls = T(f_labelF if focused else f_label, it.label,
                   CLR["text"] if focused else CLR["text2"])
            screen.blit(ls, (r.x + 12, r.y + (ROW_HGT - ls.get_height()) // 2))
            if it.value_fn is not None:
                draw_value(r, i, it, it.value_fn(), focused, now)
            elif it.select:
                vs = T(f_val, "↵", CLR["faint"])
                screen.blit(vs, (r.right - 10 - vs.get_width(),
                                 r.y + (ROW_HGT - vs.get_height()) // 2))

        screen.blit(grad(LEFT_W - 6, 26, (13, 16, 20, 0), (13, 16, 20, 255)),
                    (0, VIEW_BOT - 26))
        if content_h > VIEW_H:
            track = pygame.Rect(LEFT_W - 5, VIEW_TOP + 4, 3, VIEW_H - 8)
            pygame.draw.rect(screen, CLR["scrollTrack"], track, border_radius=2)
            th = max(24, int(track.height * VIEW_H / content_h))
            tt_y = track.y + int((track.height - th)
                                 * (list_scroll / max(1.0, content_h - VIEW_H)))
            pygame.draw.rect(screen, CLR["scrollThumb"],
                             pygame.Rect(track.x, tt_y, 3, th), border_radius=2)
        screen.set_clip(None)

        # -------------------------------------------- right pane: control strip
        strip_hit.clear()
        sx = G_X
        strip_defs = [
            ("mic", "TO MIC: ON" if state.clips_to_mic else "TO MIC: OFF",
             state.clips_to_mic, CLR["accent"], ACCENT_TINT),
            ("pause", "PAUSED" if state.clips_paused else "PAUSE",
             state.clips_paused, CLR["warning"], WARN_TINT),
            ("stop", "STOP", False, CLR["accent"], None),
        ]
        if monitor is not None:            # self-listen toggle ("hear myself")
            strip_defs.append(
                ("hear", "HEAR: ON" if monitor.on else "HEAR: OFF",
                 monitor.on, CLR["accent"], ACCENT_TINT))
        n_pages = board.page_count()
        if n_pages > 1:                # hotkey page chip; click steps onward
            strip_defs.append(("page",
                               f"PAGE {state.clip_page + 1}/{n_pages}",
                               False, CLR["accent"], None))
        for key, lab, active, acol, tint in strip_defs:
            hm = q8(hover_step(("strip", key),
                               pygame.Rect(sx, STRIP_Y, 10, STRIP_H).collidepoint(mouse_pos)
                               or (strip_hit.get(key) or pygame.Rect(0, 0, 0, 0)
                                   ).collidepoint(mouse_pos), dt))
            base_ts = T(f_strip, lab, acol if active else
                        (CLR["text"] if hm > 0.5 else CLR["muted"]))
            w = base_ts.get_width() + 24 + (12 if active else 0)
            r = pygame.Rect(sx, STRIP_Y, w, STRIP_H)
            hm = q8(hover_step(("strip2", key), r.collidepoint(mouse_pos), dt))
            pm = max(0.0, 1.0 - (now - strip_press.get(key, 0)) / 0.08) \
                if strip_press.get(key) else 0.0
            if active and tint:
                screen.blit(grad(w, STRIP_H, tint[0], tint[1], 7), r.topleft)
                pygame.draw.rect(screen, mixc(CLR["bg"], acol, 0.45), r,
                                 width=1, border_radius=7)
            else:
                top = mixc(CLR["raisedTop"], CLR["hoverTop"], hm)
                bot = mixc(CLR["raisedBot"], CLR["hoverBot"], hm)
                if pm > 0:
                    top = mixc(top, CLR["active"], q8(pm))
                    bot = mixc(bot, CLR["active"], q8(pm))
                screen.blit(grad(w, STRIP_H, top, bot, 7), r.topleft)
                pygame.draw.rect(screen, mixc(CLR["stroke"], CLR["strokeHover"], hm),
                                 r, width=1, border_radius=7)
            tx = r.x + 12
            if active:
                pygame.draw.circle(screen, acol, (tx + 2, r.centery), 2)
                tx += 12
            ts = T(f_strip, lab, acol if active else
                   (CLR["text"] if hm > 0.5 else CLR["muted"]))
            screen.blit(ts, (tx, r.centery - ts.get_height() // 2))
            strip_hit[key] = r
            sx = r.right + 8
        cnt = T(f_small, f"{len(state.clips)} SOUNDS", CLR["faint"])
        screen.blit(cnt, (G_RIGHT - cnt.get_width(),
                          STRIP_Y + (STRIP_H - cnt.get_height()) // 2))

        # ------------------------------------------------ right pane: the grid
        grid_target = max(0.0, min(grid_target,
                                   max(0.0, grid_content_h - GRID_RECT.height)))
        grid_scroll = step(grid_scroll, grid_target, dt, 0.14)
        screen.set_clip(GRID_RECT)
        grid_hit.clear()
        playing = {}
        sources = [state.voices]
        if getattr(board, "player", None) is not None:
            sources.append(board.player.voices)
        for src in sources:
            for v in list(src):
                try:
                    samples, cur = v
                except Exception:
                    continue
                pidx = clip_by_id.get(id(samples))
                if pidx is not None and len(samples):
                    playing[pidx] = max(playing.get(pidx, 0.0), cur / len(samples))

        if not state.clips:
            hint = T(f_small, "(no sounds - put audio files in ./sounds)",
                     CLR["faint"])
            screen.blit(hint, (G_X, GRID_TOP + 8))
        gy0 = GRID_TOP - int(grid_scroll)
        first_row = max(0, int(grid_scroll) // (TILE_H + GGAP))
        last_row = min(grid_rows,
                       (int(grid_scroll) + GRID_RECT.height) // (TILE_H + GGAP) + 2)
        for ci in range(first_row * COLS, min(len(state.clips), last_row * COLS)):
            g_r, g_c = divmod(ci, COLS)
            r = pygame.Rect(G_X + g_c * (TILE_W + GGAP),
                            gy0 + g_r * (TILE_H + GGAP), TILE_W, TILE_H)
            fl = board.flash.get(ci, 0) - now
            f = q8(min(1.0, max(0.0, fl / 0.25)) ** 2) if fl > 0 else 0.0
            hm = q8(hover_step(("tile", ci),
                               r.collidepoint(mouse_pos)
                               and GRID_RECT.collidepoint(mouse_pos), dt))
            prog = playing.get(ci)
            if f > 0.05:                               # trigger flash + glow
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
            screen.blit(ns, (r.x + 10, r.y + 8))
            if prog is not None:                       # playing: ▶ + progress edge
                ds = T(f_small, clip_secs[ci], CLR["accent"])
                dy = r.bottom - 10 - ds.get_height()
                py_ = dy + ds.get_height() // 2
                pygame.draw.polygon(screen, CLR["accent"],
                                    [(r.x + 10, py_ - 3), (r.x + 10, py_ + 3),
                                     (r.x + 15, py_)])
                screen.blit(ds, (r.x + 19, dy))
                pygame.draw.rect(screen, CLR["accent"],
                                 pygame.Rect(r.x, r.bottom - 2,
                                             max(2, int(TILE_W * min(1.0, prog))), 2))
            else:
                ds = T(f_small, clip_secs[ci],
                       CLR["muted"] if hm > 0.5 else CLR["faint"])
                screen.blit(ds, (r.x + 10, r.bottom - 10 - ds.get_height()))
            pg0 = state.clip_page * 9
            if pg0 <= ci < pg0 + 9:                    # hotkey badge (this page)
                hot = prog is not None or f > 0.05
                brect = pygame.Rect(r.right - 10 - 16, r.y + 8, 16, 16)
                pygame.draw.rect(screen,
                                 mixc(CLR["bg"], CLR["accent"], 0.45) if hot
                                 else CLR["strokeHover"],
                                 brect, width=1, border_radius=4)
                bs = T(f_badge, str(ci - pg0 + 1),
                       CLR["accent"] if hot else CLR["muted"])
                screen.blit(bs, (brect.centerx - bs.get_width() // 2,
                                 brect.centery - bs.get_height() // 2))
            grid_hit[ci] = r

        screen.blit(grad(GRID_RECT.width - 8, 26, (11, 13, 16, 0), (11, 13, 16, 255)),
                    (GRID_RECT.x, GRID_RECT.bottom - 26))
        if grid_content_h > GRID_RECT.height:
            track = pygame.Rect(WIN_W - 17, GRID_TOP + 4, 3,
                                GRID_RECT.height - 8)
            pygame.draw.rect(screen, CLR["scrollTrack"], track, border_radius=2)
            th = max(24, int(track.height * GRID_RECT.height / grid_content_h))
            tt_y = track.y + int((track.height - th)
                                 * (grid_scroll / max(1.0, grid_content_h
                                                      - GRID_RECT.height)))
            pygame.draw.rect(screen, CLR["scrollThumb"],
                             pygame.Rect(track.x, tt_y, 3, th), border_radius=2)
        screen.set_clip(None)

        # ------------------------------------------------- right pane: TTS panel
        pygame.draw.line(screen, CLR["strokeSoft"], (LEFT_W + 1, TTS_TOP),
                         (WIN_W, TTS_TOP))
        tts_btn_hit.clear()
        hs = TT(f_hdr, "TEXT TO SPEECH", CLR["faint"], 2)
        screen.blit(hs, (G_X + 4, TTS_TOP + 10))
        # FX chip: same state as the "TTS voice FX" menu row; reads AI while
        # the phrase would come out in the AI voice
        fx_on = state.tts_fx
        ai_live = ai is not None and ai.proc is not None
        fx_lab = ("FX: AI" if fx_on and ai_live else
                  "FX: ON" if fx_on else "FX: OFF")
        fs = T(f_strip, fx_lab, CLR["accent"] if fx_on else CLR["muted"])
        fxr = pygame.Rect(G_RIGHT - fs.get_width() - 20 - (10 if fx_on else 0),
                          TTS_TOP + 6, fs.get_width() + 20 + (10 if fx_on else 0), 22)
        hm = q8(hover_step(("tts", "fx"), fxr.collidepoint(mouse_pos), dt))
        if fx_on:
            screen.blit(grad(fxr.w, fxr.h, ACCENT_TINT[0], ACCENT_TINT[1], 6),
                        fxr.topleft)
            pygame.draw.rect(screen, mixc(CLR["bg"], CLR["accent"], 0.45), fxr,
                             width=1, border_radius=6)
        else:
            screen.blit(grad(fxr.w, fxr.h,
                             mixc(CLR["raisedTop"], CLR["hoverTop"], hm),
                             mixc(CLR["raisedBot"], CLR["hoverBot"], hm), 6),
                        fxr.topleft)
            pygame.draw.rect(screen, mixc(CLR["stroke"], CLR["strokeHover"], hm),
                             fxr, width=1, border_radius=6)
        fx_x = fxr.x + 10
        if fx_on:
            pygame.draw.circle(screen, CLR["accent"], (fx_x + 2, fxr.centery), 2)
            fx_x += 10
        screen.blit(fs, (fx_x, fxr.centery - fs.get_height() // 2))
        tts_btn_hit["fx"] = fxr
        lyy = TTS_TOP + 10 + hs.get_height() // 2
        pygame.draw.line(screen, CLR["strokeSoft"],
                         (G_X + 4 + hs.get_width() + 8, lyy), (fxr.x - 10, lyy))

        # input box + ADD button
        in_rect = pygame.Rect(G_X, TTS_IN_Y, G_RIGHT - G_X - 66, TTS_IN_H)
        add_rect = pygame.Rect(in_rect.right + 8, TTS_IN_Y,
                               G_RIGHT - in_rect.right - 8, TTS_IN_H)
        tts_btn_hit["input"] = in_rect
        tts_btn_hit["add"] = add_rect
        hm = q8(hover_step(("tts", "input"), in_rect.collidepoint(mouse_pos), dt))
        screen.blit(grad(in_rect.w, in_rect.h, CLR["headerBot"], CLR["paneLeft"], 7),
                    in_rect.topleft)
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
                ph = T(f_val, "Type a phrase, Enter to save...", CLR["faint"])
                screen.blit(ph, (in_rect.x + 10, icy - ph.get_height() // 2))
            caret_x = in_rect.x + 10
        if tts_focus and (now * 2.0) % 2 < 1:      # blinking caret
            pygame.draw.line(screen, CLR["accent"],
                             (caret_x, icy - 8), (caret_x, icy + 8))
        screen.set_clip(None)
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

        # phrase list (scrollable; click = speak, x = delete)
        n_ph = len(tts.phrases)
        tts_content_h = (n_ph * (TTS_ROW_H + TTS_ROW_GAP) - TTS_ROW_GAP + 8
                         if n_ph else 0)
        tts_target = max(0.0, min(tts_target,
                                  max(0.0, tts_content_h - TTS_LIST_RECT.height)))
        tts_scroll = step(tts_scroll, tts_target, dt, 0.14)
        screen.set_clip(TTS_LIST_RECT)
        tts_row_hit.clear()
        tts_del_hit.clear()
        tts_playing = {}                   # row -> progress of speaking phrases
        sample_row = {id(tts.samples[t]): i for i, t in enumerate(tts.phrases)
                      if t in tts.samples}
        for v in list(state.tts_voices):
            try:
                samples, cur, _fx = v
            except Exception:
                continue
            ri = sample_row.get(id(samples))
            if ri is not None and len(samples):
                tts_playing[ri] = max(tts_playing.get(ri, 0.0), cur / len(samples))
        if not n_ph:
            hint = T(f_small, "(no phrases - type one above and press Enter)",
                     CLR["faint"])
            screen.blit(hint, (G_X, TTS_LIST_TOP + 8))
        for i in range(n_ph):
            ry = TTS_LIST_TOP - int(tts_scroll) + i * (TTS_ROW_H + TTS_ROW_GAP)
            if ry + TTS_ROW_H < TTS_LIST_RECT.y or ry > VIEW_BOT:
                continue
            text = tts.phrases[i]
            r = pygame.Rect(G_X, ry, G_RIGHT - G_X, TTS_ROW_H)
            fl = tts.flash.get(i, 0) - now
            f = q8(min(1.0, max(0.0, fl / 0.25)) ** 2) if fl > 0 else 0.0
            hm = q8(hover_step(("ttsrow", i),
                               r.collidepoint(mouse_pos)
                               and TTS_LIST_RECT.collidepoint(mouse_pos), dt))
            screen.blit(grad(r.w, TTS_ROW_H,
                             mixc(mixc(CLR["raisedTop"], CLR["hoverTop"], hm),
                                  CLR["accentDim"], f),
                             mixc(mixc(CLR["raisedBot"], CLR["hoverBot"], hm),
                                  CLR["accentDim"], f), 7),
                        r.topleft)
            prog = tts_playing.get(i)
            bcol = mixc(CLR["stroke"], CLR["accentBright"], f)
            if prog is not None and f < 0.05:
                bcol = mixc(CLR["raisedTop"], CLR["accent"], 0.45)
            elif hm > 0.4 and f < 0.05:
                bcol = CLR["strokeHover"]
            pygame.draw.rect(screen, bcol, r, width=1, border_radius=7)
            dr = pygame.Rect(r.right - 8 - 18, r.centery - 9, 18, 18)
            dh = q8(hover_step(("ttsdel", i), dr.collidepoint(mouse_pos), dt))
            pygame.draw.rect(screen, mixc(CLR["strokeHover"], CLR["danger"], dh),
                             dr, width=1, border_radius=5)
            xs = T(f_badge, "x", CLR["danger"] if dh > 0.4 else CLR["muted"])
            screen.blit(xs, (dr.centerx - xs.get_width() // 2,
                             dr.centery - xs.get_height() // 2))
            tts_del_hit[i] = dr
            st = tts.status.get(text, "")
            if st == "ready" and text in tts.samples:
                dur = T(f_small, f"{len(tts.samples[text]) / SAMPLERATE:.1f}s",
                        CLR["accent"] if prog is not None else CLR["faint"])
            elif st == "error":
                dur = T(f_small, "err", CLR["danger"])
            else:                          # synthesizing: pulse like "loading"
                a = 0.4 + 0.6 * (0.5 + 0.5 * float(np.sin(now * 2 * np.pi / 1.2)))
                dur = T(f_small, "...", mixc(CLR["raisedBot"], CLR["muted"], q8(a)))
            screen.blit(dur, (dr.x - 8 - dur.get_width(),
                              r.centery - dur.get_height() // 2))
            nm = tts_trunc.get(text)
            if nm is None:
                if len(tts_trunc) > 400:
                    tts_trunc.clear()
                nm, name_w = text, r.w - 104
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
                                [(r.x + 12, r.centery - 4),
                                 (r.x + 12, r.centery + 4), (r.x + 18, r.centery)])
            ns = T(f_label, nm,
                   CLR["text"] if (hot or hm > 0.4) else CLR["text2"])
            screen.blit(ns, (r.x + 26, r.centery - ns.get_height() // 2))
            if prog is not None:           # speaking: progress along bottom edge
                pygame.draw.rect(screen, CLR["accent"],
                                 pygame.Rect(r.x, r.bottom - 2,
                                             max(2, int(r.w * min(1.0, prog))), 2))
            tts_row_hit[i] = r
        screen.blit(grad(TTS_LIST_RECT.width - 8, 20, (11, 13, 16, 0),
                         (11, 13, 16, 255)),
                    (TTS_LIST_RECT.x, VIEW_BOT - 20))
        if tts_content_h > TTS_LIST_RECT.height:
            track = pygame.Rect(WIN_W - 17, TTS_LIST_TOP + 2, 3,
                                TTS_LIST_RECT.height - 4)
            pygame.draw.rect(screen, CLR["scrollTrack"], track, border_radius=2)
            th = max(18, int(track.height * TTS_LIST_RECT.height / tts_content_h))
            tt_y = track.y + int((track.height - th)
                                 * (tts_scroll / max(1.0, tts_content_h
                                                     - TTS_LIST_RECT.height)))
            pygame.draw.rect(screen, CLR["scrollThumb"],
                             pygame.Rect(track.x, tt_y, 3, th), border_radius=2)
        screen.set_clip(None)

        # ------------------------------------------------------------ footer
        screen.blit(grad(WIN_W, FOOTER_H, CLR["footerTop"], CLR["footerBot"]),
                    (0, VIEW_BOT))
        pygame.draw.line(screen, CLR["strokeSoft"], (0, VIEW_BOT),
                         (WIN_W, VIEW_BOT))
        fy = VIEW_BOT + FOOTER_H // 2
        # live values: the engine's device line changes when devices are
        # switched from the menu
        cur_err = engine.error if engine is not None else err_line
        cur_dev = engine.dev_line if engine is not None else dev_line
        if cur_err:
            es = T(f_foot, cur_err, CLR["danger"])
            screen.blit(es, (14, fy - es.get_height() // 2))
        elif cur_dev:
            fx = 14
            if "->" in cur_dev:
                a_, b_ = cur_dev.split("->", 1)
                parts = ((a_.strip(), CLR["muted"]), (" → ", CLR["accent"]),
                         (b_.strip(), CLR["muted"]))
            else:
                parts = ((cur_dev, CLR["muted"]),)
            for ptxt, pcol in parts:
                psur = T(f_foot, ptxt, pcol)
                screen.blit(psur, (fx, fy - psur.get_height() // 2))
                fx += psur.get_width()

        # latency + underrun tally (right side; the status chip, which uses
        # the same corner, takes precedence while it is up)
        chip_up = state.status_msg and (now - state.status_at) < 4.38
        if engine is not None and engine.latency_ms and not chip_up:
            stat = f"{engine.latency_ms:.0f} ms latency"
            if state.status_count:
                stat += f" · {state.status_count} drops"
            ss = T(f_foot, stat,
                   CLR["warning"] if state.status_count else CLR["faint"])
            screen.blit(ss, (WIN_W - 14 - ss.get_width(),
                             fy - ss.get_height() // 2))

        # status toast chip: in 160ms / hold 4s / out 220ms
        if state.status_msg:
            t_ = now - state.status_at
            alpha = (t_ / 0.16 if t_ < 0.16 else
                     1.0 if t_ < 4.16 else
                     max(0.0, 1.0 - (t_ - 4.16) / 0.22) if t_ < 4.38 else 0.0)
            if alpha > 0:
                col = (CLR["danger"] if "error" in state.status_msg.lower()
                       else CLR["warning"])
                cs = T(f_foot, f"{state.status_msg} (x{state.status_count})", col)
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

        # --------------------------------------------- dropdown picker overlay
        if drop is not None:
            r = drop["rect"]
            if drop["mouse"] != mouse_pos and r.collidepoint(mouse_pos):
                mi = (mouse_pos[1] - r.y - drop["pad"]
                      + int(drop["scroll"])) // drop["item_h"]
                if 0 <= mi < len(drop["items"]):
                    drop["sel"] = mi
            drop["mouse"] = mouse_pos
            screen.blit(grad(r.w, r.h, CLR["hoverTop"], CLR["raisedBot"], 8),
                        r.topleft)
            pygame.draw.rect(screen, mixc(CLR["stroke"], CLR["accent"], 0.35),
                             r, width=1, border_radius=8)
            screen.set_clip(r.inflate(-2, -4))
            y0 = r.y + drop["pad"] - int(drop["scroll"])
            for i, (nm, _cb) in enumerate(drop["items"]):
                ir = pygame.Rect(r.x + 4, y0 + i * drop["item_h"],
                                 r.w - 12, drop["item_h"] - 2)
                if ir.bottom < r.y or ir.y > r.bottom:
                    continue
                if i == drop["sel"]:
                    screen.blit(grad(ir.w, ir.h, ACCENT_TINT[0],
                                     ACCENT_TINT[1], 6), ir.topleft)
                    pygame.draw.rect(screen, CLR["accent"], ir,
                                     width=1, border_radius=6)
                if i == drop["cur"]:           # the currently active entry
                    pygame.draw.circle(screen, CLR["accent"],
                                       (ir.x + 11, ir.centery), 2)
                ns = T(f_labelF if i == drop["sel"] else f_label, nm,
                       CLR["text"] if i == drop["sel"] else CLR["text2"])
                screen.blit(ns, (ir.x + 22, ir.centery - ns.get_height() // 2))
            if drop["max_scroll"] > 0:
                track = pygame.Rect(r.right - 6, r.y + 4, 3, r.h - 8)
                pygame.draw.rect(screen, CLR["scrollTrack"], track,
                                 border_radius=2)
                th = max(18, int(track.h * r.h / (drop["max_scroll"] + r.h)))
                ty = track.y + int((track.h - th)
                                   * (drop["scroll"] / drop["max_scroll"]))
                pygame.draw.rect(screen, CLR["scrollThumb"],
                                 pygame.Rect(track.x, ty, 3, th),
                                 border_radius=2)
            screen.set_clip(None)

        pygame.display.flip()
        clock.tick(30)          # 30 fps is plenty for a menu and halves GIL load

    pygame.quit()

