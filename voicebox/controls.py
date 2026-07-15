"""Input bindings (controls.json) and the system-wide hotkeys."""
import json
import time

from .config import CONTROLS_PATH, DEFAULT_CONTROLS

def load_controls():
    """controls.json merged over defaults; broken/missing file -> defaults."""
    cfg = json.loads(json.dumps(DEFAULT_CONTROLS))     # deep copy
    try:
        user = json.loads(CONTROLS_PATH.read_text(encoding="utf-8"))
        for section in ("keyboard", "gamepad", "global"):
            if isinstance(user.get(section), dict):
                cfg[section].update(user[section])
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return cfg


class GlobalHotkeys:
    """System-wide hotkeys (controls.json "global" section) so the soundboard
    stays usable while a game or Discord has focus. Uses the optional
    `keyboard` package; when it is missing or lacks permission (Linux without
    root) VoiceBox degrades to window-only input instead of failing. Handlers
    fire on the keyboard package's listener thread - everything they call
    (Board.play/stop via queues, apply_preset under the state lock) is
    thread-safe already."""

    def __init__(self, state, board, cfg=None, ai=None, scenes=None):
        self.state = state
        self.board = board
        self.ai = ai                   # AiVoice, for the ai_voice binding
        self.scenes = scenes           # Scenes, for the next_scene binding
        self.cfg = (cfg or load_controls()).get("global") or {}
        self.error = ""
        self._kb = None                # the keyboard module while registered
        self._handles = []
        self._hooks = []               # PTT press/release hooks (unhook to undo)
        if self.cfg.get("enabled", True):
            self.enable()

    @property
    def on(self):
        return bool(self._handles or self._hooks)

    def _bindings(self):
        """(combo, handler) pairs from config; blank combos are skipped."""
        out = []
        clips = self.cfg.get("clips")
        if isinstance(clips, list):
            for i, combo in enumerate(clips):
                if combo:
                    out.append((str(combo),
                                lambda i=i: self.board.play_hot(i)))
        if self.cfg.get("stop_clips"):
            out.append((str(self.cfg["stop_clips"]), self.board.stop))
        if self.cfg.get("next_preset"):
            out.append((str(self.cfg["next_preset"]), self._next_preset))
        if self.cfg.get("mute"):
            out.append((str(self.cfg["mute"]), self._toggle_mute))
        if self.cfg.get("ai_voice") and self.ai is not None:
            out.append((str(self.cfg["ai_voice"]), self.ai.toggle))
        if self.cfg.get("next_scene") and self.scenes is not None:
            out.append((str(self.cfg["next_scene"]), self.scenes.cycle))
        return out

    def _next_preset(self):
        self.state.apply_preset(self.state.preset_idx + 1)

    def _toggle_mute(self):
        with self.state.lock:
            self.state.mic_muted = not self.state.mic_muted
        if self.state.cues is not None:    # manual toggle only - PTT stays
            self.state.cues.mute(self.state.mic_muted)   # silent (_set_mute)

    def _set_mute(self, muted):
        with self.state.lock:
            self.state.mic_muted = muted

    def enable(self):
        if self.on:
            return
        try:
            import keyboard                # deferred: import can itself fail
        except Exception as e:
            self.error = f"global hotkeys off: {e}"
            return
        handles, hooks = [], []
        try:
            for combo, fn in self._bindings():
                handles.append(keyboard.add_hotkey(combo, fn))
            ptt = self.cfg.get("ptt")
            if ptt:                        # hold-to-talk: live on press, muted on release
                hooks.append(keyboard.on_press_key(
                    str(ptt), lambda e: self._set_mute(False)))
                hooks.append(keyboard.on_release_key(
                    str(ptt), lambda e: self._set_mute(True)))
                self._set_mute(True)       # PTT implies mute-by-default
        except Exception as e:             # bad combo string / no permission
            for h in handles:
                try:
                    keyboard.remove_hotkey(h)
                except Exception:
                    pass
            for h in hooks:
                try:
                    keyboard.unhook(h)
                except Exception:
                    pass
            self.error = f"global hotkeys off: {e}"
            return
        self._kb = keyboard
        self._handles, self._hooks = handles, hooks
        self.error = ""

    def disable(self):
        kb, handles, hooks = self._kb, self._handles, self._hooks
        self._kb, self._handles, self._hooks = None, [], []
        for h in handles:
            try:
                kb.remove_hotkey(h)
            except Exception:
                pass
        for h in hooks:
            try:
                kb.unhook(h)
            except Exception:
                pass
        if hooks:                      # PTT armed mute-by-default; with the
            self._set_mute(False)      # hooks gone nothing could unmute it

    def toggle(self):
        self.disable() if self.on else self.enable()
        if self.error:                     # surface why it would not start
            self.state.status_msg = self.error
            self.state.status_at = time.time()

    def close(self):
        self.disable()


def build_keymap(cfg, pygame):
    """action-name -> set of pygame keycodes, plus keycode -> clip index."""
    keymap, clipmap = {}, {}
    for action, names in cfg["keyboard"].items():
        if not isinstance(names, list):
            continue
        for i, name in enumerate(names):
            try:
                code = pygame.key.key_code(str(name))
            except (ValueError, NotImplementedError):
                continue
            if action == "clips":
                clipmap[code] = i
            else:
                keymap.setdefault(action, set()).add(code)
    return keymap, clipmap


