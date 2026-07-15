"""Global hotkeys: registration from config, action routing, degradation."""
import sys
import types

from _common import check, finish

import numpy as np
import voicebox

voicebox.soundboard.load_clips = lambda: (
    [np.full(1000, 0.1, np.float32), np.full(1000, 0.2, np.float32)],
    ["clip1", "clip2"])


class FakeKeyboard(types.ModuleType):
    """Stands in for the `keyboard` package: records registrations."""
    def __init__(self):
        super().__init__("keyboard")
        self.hotkeys = {}              # handle -> (combo, fn)
        self.press_hooks = {}          # handle -> (key, fn)
        self.release_hooks = {}        # handle -> (key, fn)
        self.next_handle = 0
        self.fail_on = None            # combo that raises (permission test)

    def add_hotkey(self, combo, fn):
        if combo == self.fail_on:
            raise OSError("no permission")
        self.next_handle += 1
        self.hotkeys[self.next_handle] = (combo, fn)
        return self.next_handle

    def remove_hotkey(self, handle):
        del self.hotkeys[handle]

    def on_press_key(self, key, fn):
        self.next_handle += 1
        self.press_hooks[self.next_handle] = (key, fn)
        return self.next_handle

    def on_release_key(self, key, fn):
        self.next_handle += 1
        self.release_hooks[self.next_handle] = (key, fn)
        return self.next_handle

    def unhook(self, handle):
        self.press_hooks.pop(handle, None)
        self.release_hooks.pop(handle, None)

    def fire(self, combo):
        for c, fn in list(self.hotkeys.values()):
            if c == combo:
                fn()

    def press(self, key):
        for k, fn in list(self.press_hooks.values()):
            if k == key:
                fn(None)

    def release(self, key):
        for k, fn in list(self.release_hooks.values()):
            if k == key:
                fn(None)


def with_fake(cfg=None):
    fake = FakeKeyboard()
    sys.modules["keyboard"] = fake
    state = voicebox.State()
    board = voicebox.Board(state)
    hk = voicebox.GlobalHotkeys(state, board, cfg or {"global": {}})
    return fake, state, board, hk


# ------------------------------------------------------------- registration
cfg = {"global": {"enabled": True,
                  "clips": ["ctrl+alt+1", "ctrl+alt+2"],
                  "stop_clips": "ctrl+alt+0",
                  "next_preset": "ctrl+alt+p"}}
fake, state, board, hk = with_fake(cfg)
check("hotkeys registered on start", hk.on and len(fake.hotkeys) == 4)
check("no error on clean start", hk.error == "")

combos = sorted(c for c, _ in fake.hotkeys.values())
check("combos come from the config",
      combos == ["ctrl+alt+0", "ctrl+alt+1", "ctrl+alt+2", "ctrl+alt+p"])

# ------------------------------------------------------------ action routing
fake.fire("ctrl+alt+2")
ev = state.events.get_nowait() if not state.events.empty() else None
check("clip hotkey queues the clip", ev == 1)

state.events.put(0)                    # something to stop
fake.fire("ctrl+alt+0")
found_stop = False
while not state.events.empty():
    if state.events.get_nowait() == "stop":
        found_stop = True
check("stop hotkey stops all sounds", found_stop)

before = state.preset_idx
fake.fire("ctrl+alt+p")
check("preset hotkey cycles the preset",
      state.preset_idx == (before + 1) % len(voicebox.PRESETS))

# ------------------------------------------------------------ enable/disable
hk.toggle()
check("toggle off unregisters everything",
      not hk.on and len(fake.hotkeys) == 0)
hk.toggle()
check("toggle back on re-registers", hk.on and len(fake.hotkeys) == 4)
hk.close()
check("close unregisters", not hk.on and len(fake.hotkeys) == 0)

# --------------------------------------------------------------- mute + PTT
fake, state, board, hk = with_fake(
    {"global": {"mute": "ctrl+alt+m", "ptt": "f8"}})
check("PTT arms mute-by-default", state.mic_muted is True)
fake.press("f8")
check("PTT press goes live", state.mic_muted is False)
fake.release("f8")
check("PTT release re-mutes", state.mic_muted is True)
fake.fire("ctrl+alt+m")
check("mute hotkey toggles", state.mic_muted is False)
fake.press("f8"); fake.release("f8")   # leave the mic PTT-muted
hk.disable()
check("disable removes PTT hooks too",
      not fake.press_hooks and not fake.release_hooks)
check("disable un-mutes: no PTT key is left to go live",
      state.mic_muted is False)

# ------------------------------------------------- blank / disabled bindings
fake, state, board, hk = with_fake(
    {"global": {"enabled": True, "clips": ["ctrl+alt+1", "", None],
                "stop_clips": "", "next_preset": None}})
check("blank bindings are skipped", len(fake.hotkeys) == 1)

fake, state, board, hk = with_fake({"global": {"enabled": False,
                                               "clips": ["ctrl+alt+1"]}})
check("disabled config registers nothing", not hk.on and len(fake.hotkeys) == 0)

# ------------------------------------------------------------ AI voice hotkey
class StubAI:
    def __init__(self): self.toggles = 0
    def toggle(self): self.toggles += 1

fake = FakeKeyboard()
sys.modules["keyboard"] = fake
state = voicebox.State()
stub_ai = StubAI()
hk = voicebox.GlobalHotkeys(state, voicebox.Board(state),
                            {"global": {"ai_voice": "ctrl+alt+a"}}, ai=stub_ai)
check("AI voice hotkey registered", len(fake.hotkeys) == 1)
fake.fire("ctrl+alt+a")
fake.fire("ctrl+alt+a")
check("AI voice hotkey toggles the worker", stub_ai.toggles == 2)
hk.close()

fake = FakeKeyboard()
sys.modules["keyboard"] = fake
hk = voicebox.GlobalHotkeys(state, voicebox.Board(state),
                            {"global": {"ai_voice": "ctrl+alt+a"}})
check("AI binding skipped without AiVoice", len(fake.hotkeys) == 0)
hk.close()

# ------------------------------------------------------------- scene hotkey
class StubScenes:
    def __init__(self): self.cycled = []
    def cycle(self, d=1): self.cycled.append(d)

fake = FakeKeyboard()
sys.modules["keyboard"] = fake
stub_scenes = StubScenes()
hk = voicebox.GlobalHotkeys(state, voicebox.Board(state),
                            {"global": {"next_scene": "ctrl+alt+s"}},
                            scenes=stub_scenes)
check("scene hotkey registered", len(fake.hotkeys) == 1)
fake.fire("ctrl+alt+s")
check("scene hotkey steps to the next scene", len(stub_scenes.cycled) == 1)
hk.close()

fake = FakeKeyboard()
sys.modules["keyboard"] = fake
hk = voicebox.GlobalHotkeys(state, voicebox.Board(state),
                            {"global": {"next_scene": "ctrl+alt+s"}})
check("scene binding skipped without Scenes", len(fake.hotkeys) == 0)
hk.close()

# ------------------------------------------------------ graceful degradation
sys.modules["keyboard"] = None         # forces `import keyboard` to fail
state = voicebox.State()
hk = voicebox.GlobalHotkeys(state, voicebox.Board(state),
                            {"global": {"clips": ["ctrl+alt+1"]}})
check("missing package degrades quietly",
      not hk.on and hk.error.startswith("global hotkeys off"))
hk.toggle()
check("toggle surfaces the reason in the status line",
      state.status_msg.startswith("global hotkeys off"))

fake = FakeKeyboard()
fake.fail_on = "ctrl+alt+2"            # second registration blows up
sys.modules["keyboard"] = fake
state = voicebox.State()
hk = voicebox.GlobalHotkeys(state, voicebox.Board(state),
                            {"global": {"clips": ["ctrl+alt+1", "ctrl+alt+2"]}})
check("partial registration rolls back",
      not hk.on and len(fake.hotkeys) == 0 and hk.error != "")

# --------------------------------------------------------------- config merge
cfg = voicebox.load_controls()
check("defaults include the global section",
      isinstance(cfg.get("global"), dict) and "clips" in cfg["global"]
      and "ai_voice" in cfg["global"] and "next_scene" in cfg["global"])

# a hand-edited bare value where a list belongs must not lose the binding
import json
import tempfile
from pathlib import Path

real_path = voicebox.controls.CONTROLS_PATH
tmp_controls = Path(tempfile.mkdtemp()) / "controls.json"
tmp_controls.write_text(json.dumps(
    {"keyboard": {"select": "return"}, "gamepad": {"select": 0}}),
    encoding="utf-8")
voicebox.controls.CONTROLS_PATH = tmp_controls
cfg2 = voicebox.load_controls()
voicebox.controls.CONTROLS_PATH = real_path
check("bare string binding is wrapped, not dropped",
      cfg2["keyboard"]["select"] == ["return"]
      and cfg2["gamepad"]["select"] == [0])

del sys.modules["keyboard"]
finish()
