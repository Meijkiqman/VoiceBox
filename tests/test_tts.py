"""TTS phrases: persistence, background synthesis, playback routing, UI panel."""
import json
import tempfile
import threading
import time
from pathlib import Path

from _common import check, finish

import numpy as np
import voicebox

voicebox.load_clips = lambda: ([np.full(1000, 0.1, np.float32)], ["clip1"])

# deterministic fake synth: no speech engine, no cache files
TONE = np.full(2400, 0.2, dtype=np.float32)          # 50 ms at 48k
FAKE_WAV = Path(tempfile.mkdtemp()) / "fake.wav"
synth_calls = []


def fake_synthesize(text, voice=None, rate=0):
    synth_calls.append((text, voice, rate))
    if text == "boom":
        raise RuntimeError("engine exploded")
    return TONE.copy(), FAKE_WAV


voicebox.tts_synthesize = fake_synthesize


def wait_status(bank, text, timeout=2.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if bank.status.get(text) in ("ready", "error"):
            return bank.status[text]
        time.sleep(0.01)
    return "timeout"


def wait_event(state, timeout=2.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not state.events.empty():
            return state.events.get_nowait()
        time.sleep(0.01)
    return None


# ---------------------------------------------------------------- persistence
tmp = Path(tempfile.mkdtemp()) / "phrases.json"
state = voicebox.State()
bank = voicebox.TTSBank(state, path=tmp)
check("starts empty without a file", bank.phrases == [])
check("add saves the phrase",
      bank.add("hello world") is True and bank.phrases == ["hello world"])
check("whitespace collapsed",
      bank.add("  two \n words ") is True and bank.phrases[-1] == "two words")
check("empty rejected", bank.add("   ") is False)
check("duplicate rejected", bank.add("hello world") is False)
check("phrases persist on disk",
      json.loads(tmp.read_text(encoding="utf-8")) == ["hello world", "two words"])
check("phrases reload from disk",
      voicebox.TTSBank(state, path=tmp).phrases == ["hello world", "two words"])
bank.delete(0)
check("delete removes + persists",
      bank.phrases == ["two words"]
      and json.loads(tmp.read_text(encoding="utf-8")) == ["two words"])
bank.delete(5)                                       # out of range: no crash
check("out-of-range delete ignored", bank.phrases == ["two words"])

# ---------------------------------------------------------- synth lifecycle
check("add kicked background synth", wait_status(bank, "two words") == "ready")
check("synth result cached", len(bank.samples["two words"]) == len(TONE))
bank.add("boom")
check("failed synth -> error status", wait_status(bank, "boom") == "error")
check("synth error surfaces in status line", state.status_msg.startswith("TTS:"))
bank.delete(bank.phrases.index("boom"))

# ------------------------------------------------- routing through the chain
cb = voicebox.make_callback(state)
frames = voicebox.BLOCKSIZE
silent = np.zeros((frames, 1), dtype=np.float32)
out = np.zeros((frames, 1), dtype=np.float32)

bank.play(0)                                          # tts_fx defaults to ON
check("play of ready phrase queues a tts event", not state.events.empty())
cb(silent, out, frames, None, None)
check("fx TTS rides the voice chain", abs(np.abs(out).max() - 0.2) < 0.01,
      f"peak={np.abs(out).max():.3f}")
while np.abs(out).max() > 0:                          # drain the 50 ms phrase
    cb(silent, out, frames, None, None)

with state.lock:
    state.tts_fx = False
bank.play(0)
cb(silent, out, frames, None, None)
check("clean TTS mixes in post-chain", abs(np.abs(out).max() - 0.2) < 0.01)
while state.tts_voices:                               # drain before the next check
    cb(silent, out, frames, None, None)

# TTS volume applies on the mic channel
with state.lock:
    state.tts_gain = 0.5
bank.play(0)
cb(silent, out, frames, None, None)
check("TTS volume scales the mix", abs(np.abs(out).max() - 0.1) < 0.01)
with state.lock:
    state.tts_gain = 1.0
while state.tts_voices:
    cb(silent, out, frames, None, None)

# pause freezes, stop clears (same contract as the soundboard)
bank.play(0)
with state.lock:
    state.clips_paused = True
cb(silent, out, frames, None, None)
check("pause silences + freezes TTS",
      np.abs(out).max() == 0.0 and len(state.tts_voices) == 1
      and state.tts_voices[0][1] == 0)
with state.lock:
    state.clips_paused = False
state.events.put("stop")
cb(silent, out, frames, None, None)
check("stop clears TTS voices", state.tts_voices == [])

# fx phrase stays audible while the AI worker owns the voice path
with state.lock:
    state.tts_fx = True
    state.ai_mute = True
bank.play(0)
cb(silent, out, frames, None, None)
check("fx TTS still audible while AI owns the mic",
      abs(np.abs(out).max() - 0.2) < 0.01)
state.events.put("stop")
cb(silent, out, frames, None, None)
with state.lock:
    state.ai_mute = False

# ------------------------------------------------------- routing through AI
class FakeAI:
    def __init__(self, accept=True):
        self.proc = object()
        self.accept = accept
        self.injected = []

    def inject(self, wav_path):
        self.injected.append(str(wav_path))
        return self.accept


fake_ai = FakeAI()
bank_ai = voicebox.TTSBank(state, ai=fake_ai, path=tmp)
bank_ai.play(0)                                       # pending -> synth -> route
t0 = time.time()
while not fake_ai.injected and time.time() - t0 < 2.0:
    time.sleep(0.01)
check("fx TTS goes through the AI worker", fake_ai.injected == [str(FAKE_WAV)])
check("no mic event when the worker takes it", state.events.empty())

dead_ai = FakeAI(accept=False)
bank_dead = voicebox.TTSBank(state, ai=dead_ai, path=tmp)
bank_dead.play(0)
ev = wait_event(state)
check("worker refusal falls back to the mic event",
      dead_ai.injected and isinstance(ev, tuple) and ev[0] == "tts")

with state.lock:
    state.tts_fx = False
fake_ai.injected.clear()
bank_ai.play(0)
ev = wait_event(state)
check("fx off bypasses the AI",
      not fake_ai.injected and isinstance(ev, tuple) and ev[0] == "tts"
      and ev[2] is False)
with state.lock:
    state.tts_fx = True

# ------------------------------------------------------------- local listen
lp = voicebox.LocalPlayer(state)
lp.events.put(("raw", TONE.copy()))
out2 = np.zeros((frames, 1), dtype=np.float32)
lp._callback(out2, frames, None, None)
check("local player mixes raw TTS", np.abs(out2).max() > 0.15,
      f"peak={np.abs(out2).max():.3f}")

# --------------------------------------------------------------- menu rows
menu = voicebox.Menu(state, threading.Event())
labels = [it.label for it in menu.items]
check("TTS rows present", "TTS voice FX" in labels and "TTS volume" in labels)
check("TTS rows ordered before Sounds to mic",
      labels.index("TTS voice FX") < labels.index("Sounds to mic"))
fx_row = menu.items[labels.index("TTS voice FX")]
before = state.tts_fx
fx_row.select()
check("FX row toggles state", state.tts_fx is (not before))
fx_row.select()

# ------------------------------------------------------------- UI panel smoke
import pygame

ui_state = voicebox.State()
ui_tmp = Path(tempfile.mkdtemp()) / "phrases.json"
ui_bank = voicebox.TTSBank(ui_state, path=ui_tmp)
stop_flag = threading.Event()
snaps = []


def poke():
    time.sleep(0.7)
    post = pygame.event.post
    # click the TTS input box (spans x 384-872, y 448-478)
    post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(500, 462)))
    time.sleep(0.15)
    post(pygame.event.Event(pygame.TEXTINPUT, text="hi there"))
    # clip hotkey must NOT fire while typing
    post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_1))
    time.sleep(0.1)
    post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))   # save
    time.sleep(0.3)
    snaps.append(list(ui_bank.phrases))               # after commit
    # Escape unfocuses the box but must NOT quit the app
    post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    time.sleep(0.15)
    # click phrase row 0 (rows start at y 486) -> speak it
    post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(500, 500)))
    time.sleep(0.3)
    # click the row's x (right edge, ~x 912-930) -> delete it
    post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(920, 501)))
    time.sleep(0.15)
    snaps.append(list(ui_bank.phrases))               # after delete
    post(pygame.event.Event(pygame.QUIT))


threading.Thread(target=poke, daemon=True).start()
ui_error = []
try:
    voicebox.run_ui(ui_state, stop_flag, "dev", "", None, None, None, ui_bank)
except Exception as e:
    ui_error.append(e)
check("UI with TTS panel survives", not ui_error,
      repr(ui_error[0]) if ui_error else "")
check("typed phrase was saved via the panel", snaps and snaps[0] == ["hi there"])
check("row x deleted the phrase", len(snaps) == 2 and snaps[1] == [])

tts_events, int_events = [], []
while not ui_state.events.empty():
    ev = ui_state.events.get_nowait()
    if isinstance(ev, tuple) and ev[0] == "tts":
        tts_events.append(ev)
    elif isinstance(ev, int):
        int_events.append(ev)
check("row click spoke the phrase to the mic", len(tts_events) >= 1)
check("clip hotkeys gated while typing", int_events == [])

# ---------------------------------------------------------- voice + rate
p1 = voicebox.tts_cache_path("hello")
check("cache path is stable", p1 == voicebox.tts_cache_path("hello"))
check("cache path varies by voice",
      p1 != voicebox.tts_cache_path("hello", voice="Zira"))
check("cache path varies by rate",
      p1 != voicebox.tts_cache_path("hello", rate=3))
check("int and float rates share a key",
      voicebox.tts_cache_path("hello", rate=3)
      == voicebox.tts_cache_path("hello", rate=3.0))

vstate = voicebox.State()
with vstate.lock:
    vstate.tts_voice = "Zira"
    vstate.tts_rate = 4.0
vbank = voicebox.TTSBank(vstate, path=Path(tempfile.mkdtemp()) / "p.json")
synth_calls.clear()
vbank.add("with voice")
check("synth job passes the selected voice and rate",
      wait_status(vbank, "with voice") == "ready"
      and synth_calls == [("with voice", "Zira", 4.0)])

check("bank starts with voices unlisted", vbank.voice_names is None)
vbank.invalidate()
check("invalidate clears rendered speech, keeps phrases",
      vbank.phrases == ["with voice"] and vbank.samples == {}
      and vbank.status == {})

# menu rows cycle voice/rate and invalidate the bank
vbank.voice_names = ["Alpha", "Beta"]
vbank._voices_kicked = True                   # don't spawn the lister thread
menu2 = voicebox.Menu(vstate, threading.Event(), tts=vbank)
labels2 = [it.label for it in menu2.items]
check("TTS voice + rate rows present",
      "TTS voice" in labels2 and "TTS rate" in labels2)
with vstate.lock:
    vstate.tts_voice = None
vrow = next(it for it in menu2.items if it.label == "TTS voice")
vrow.adjust(+1)
check("voice row cycles into the list", vstate.tts_voice == "Alpha")
vrow.adjust(+1)
vrow.adjust(+1)
check("voice row wraps back to default", vstate.tts_voice is None)
check("voice label shows default", vrow.value_fn() == "default")

vbank.samples["x"] = TONE
rrow = next(it for it in menu2.items if it.label == "TTS rate")
with vstate.lock:
    vstate.tts_rate = 0.0
rrow.adjust(+1)
check("rate row steps and invalidates",
      vstate.tts_rate == 1.0 and vbank.samples == {})
for _ in range(15):
    rrow.adjust(+1)
check("rate clamps at +10", vstate.tts_rate == 10.0)
rrow.select()
check("rate select resets to normal",
      vstate.tts_rate == 0.0 and rrow.value_fn() == "normal")

finish()
