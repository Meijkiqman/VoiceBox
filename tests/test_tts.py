"""TTS phrases: persistence, background synthesis, playback routing, UI panel."""
import json
import tempfile
import threading
import time
from pathlib import Path

from _common import check, finish

import numpy as np
import voicebox

voicebox.soundboard.load_clips = lambda: ([np.full(1000, 0.1, np.float32)], ["clip1"])

# deterministic fake synth: no speech engine, no cache files
TONE = np.full(2400, 0.2, dtype=np.float32)          # 50 ms at 48k
FAKE_WAV = Path(tempfile.mkdtemp()) / "fake.wav"
synth_calls = []


def fake_synthesize(text, voice=None, rate=0):
    synth_calls.append((text, voice, rate))
    if text == "boom":
        raise RuntimeError("engine exploded")
    return TONE.copy(), FAKE_WAV


voicebox.tts.tts_synthesize = fake_synthesize


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

# -------------------------------------------------------------- quick-speak
while not state.events.empty():
    state.events.get_nowait()
check("say accepts a phrase without saving it",
      bank.say("quick one") is True and "quick one" not in bank.phrases)
check("say synthesizes on demand", wait_status(bank, "quick one") == "ready")
qev = wait_event(state)
check("say routes to the mic like a saved phrase",
      isinstance(qev, tuple) and qev[0] == "tts")
check("say of empty text rejected", bank.say("   ") is False)

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
check("TTS volume scales the mix", abs(np.abs(out).max() - 0.1) < 0.01,
      f"peak={np.abs(out).max():.3f}")
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


def inject(ev):
    """Hand a synthetic event to run_ui's main-thread hook -
    cross-thread pygame.event.post corrupts the SDL queue."""
    from collections import deque
    voicebox.ui.ui_debug.setdefault("inject", deque()).append(ev)

ui_state = voicebox.State()
ui_tmp = Path(tempfile.mkdtemp()) / "phrases.json"
ui_bank = voicebox.TTSBank(ui_state, path=ui_tmp)
stop_flag = threading.Event()
snaps = []


def ui_rect(kind, key):
    """Center of a live hit-rect from the dashboard's debug registry."""
    r = voicebox.ui.ui_debug.get(kind, {}).get(key)
    return r.center if r else (0, 0)


def poke():
    from _common import wait_ui
    wait_ui(lambda: ui_rect("tts_btn_hit", "input") != (0, 0))
    post = pygame.event.post
    # click the TTS input box (TEXT-TO-SPEECH card)
    post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1,
                            pos=ui_rect("tts_btn_hit", "input")))
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
    # click phrase row 0 -> speak it
    from _common import wait_ui as _w
    _w(lambda: ui_rect("tts_row_hit", 0) != (0, 0))
    post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1,
                            pos=ui_rect("tts_row_hit", 0)))
    time.sleep(0.3)
    # click the row's x -> delete it
    post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1,
                            pos=ui_rect("tts_del_hit", 0)))
    time.sleep(0.15)
    snaps.append(list(ui_bank.phrases))               # after delete
    # Ctrl+V pastes the (stubbed) clipboard into the box, Enter saves it
    post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1,
                            pos=ui_rect("tts_btn_hit", "input")))
    time.sleep(0.15)
    post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_v,
                            mod=pygame.KMOD_CTRL))
    time.sleep(0.1)
    post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
    time.sleep(0.3)
    snaps.append(list(ui_bank.phrases))               # after paste + save
    # Shift+Enter speaks the typed text without saving it
    post(pygame.event.Event(pygame.TEXTINPUT, text="quick ui"))
    time.sleep(0.05)
    post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN,
                            mod=pygame.KMOD_SHIFT))
    time.sleep(0.3)
    snaps.append(list(ui_bank.phrases))               # unchanged
    post(pygame.event.Event(pygame.QUIT))


real_clip = voicebox.ui.get_clipboard_text
voicebox.ui.get_clipboard_text = lambda: "pasted from clipboard"
threading.Thread(target=poke, daemon=True).start()
ui_error = []
try:
    voicebox.run_ui(ui_state, stop_flag, "dev", "", None, None, None, ui_bank)
except Exception as e:
    ui_error.append(e)
finally:
    voicebox.ui.get_clipboard_text = real_clip
check("UI with TTS panel survives", not ui_error,
      repr(ui_error[0]) if ui_error else "")
check("typed phrase was saved via the panel", snaps and snaps[0] == ["hi there"])
check("row x deleted the phrase", len(snaps) >= 2 and snaps[1] == [])
check("Ctrl+V pasted the clipboard into the box",
      len(snaps) >= 3 and snaps[2] == ["pasted from clipboard"], str(snaps))
check("Shift+Enter spoke without saving",
      len(snaps) == 4 and snaps[3] == snaps[2], str(snaps))
check("quick-speak synthesized the unsaved text",
      ui_bank.status.get("quick ui") == "ready")

tts_events, int_events = [], []
while not ui_state.events.empty():
    ev = ui_state.events.get_nowait()
    if isinstance(ev, tuple) and ev[0] == "tts":
        tts_events.append(ev)
    elif isinstance(ev, int):
        int_events.append(ev)
check("row click spoke the phrase to the mic", len(tts_events) >= 1)
check("clip hotkeys gated while typing", int_events == [])

# ----------------------------------------- voice picker dropdown (>3 options)
ui_state2 = voicebox.State()
bank2 = voicebox.TTSBank(ui_state2, path=Path(tempfile.mkdtemp()) / "p.json")
bank2.voice_names = ["Gamma", "Alpha", "Delta", "Beta"]
bank2._voices_kicked = True                  # keep the stub list as-is
stop2 = threading.Event()
sess0 = voicebox.ui.ui_debug.get("session", 0)


def poke2():
    from _common import wait_ui
    dbg = voicebox.ui.ui_debug
    # rects from the PREVIOUS run_ui session linger until this one draws:
    # wait for the fresh session before trusting any of them
    wait_ui(lambda: dbg.get("session", 0) > sess0)
    wait_ui(lambda: dbg["row_hit"].get(dbg["labels"].index("TTS voice")))
    r = dbg["row_hit"][dbg["labels"].index("TTS voice")]
    inject(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=r.center))
    wait_ui(lambda: dbg.get("drop_info"))
    di = dbg["drop_info"]
    if di:   # sorted list: default, Alpha, Beta, Delta, Gamma -> Beta row 2
        pos = (di["rect"].x + 30, di["rect"].y + di["pad"] - int(di["scroll"])
               + di["item_h"] * 2 + di["item_h"] // 2)
        inject(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos))
    wait_ui(lambda: ui_state2.tts_voice == "Beta", timeout=3.0)
    inject(pygame.event.Event(pygame.QUIT))


threading.Thread(target=poke2, daemon=True).start()
ui_error = []
try:
    voicebox.run_ui(ui_state2, stop2, "dev", "", None, None, None, bank2)
except Exception as e:
    ui_error.append(e)
check("voice picker UI survives", not ui_error,
      repr(ui_error[0]) if ui_error else "")
check(">3 options pick from a dropdown", ui_state2.tts_voice == "Beta",
      str(ui_state2.tts_voice))

# ------------------------------------------------------- Piper integration
piper_tmp = Path(tempfile.mkdtemp()) / "piper"
(piper_tmp / "voices").mkdir(parents=True)
(piper_tmp / "piper").write_bytes(b"")           # engine stand-in
(piper_tmp / "voices" / "en_US-ryan-high.onnx").write_bytes(b"x")
(piper_tmp / "voices" / "en_US-lessac-high.onnx").write_bytes(b"x")
old_piper_dir = voicebox.tts.PIPER_DIR
voicebox.tts.PIPER_DIR = piper_tmp

vm = voicebox.tts.piper_voice_map()
check("piper voices discovered",
      list(vm) == ["Piper: Lessac (en_US, high)", "Piper: Ryan (en_US, high)"],
      str(list(vm)))

piper_calls = []


def fake_piper_run(cmd, **kw):
    piper_calls.append((cmd, kw))
    import soundfile as _sf
    _sf.write(cmd[cmd.index("--output_file") + 1],
              np.full(2205, 0.2, np.float32), 22050, subtype="PCM_16")
    class R:
        returncode = 0
        stdout = stderr = ""
    return R()


old_run = voicebox.tts._piper_run
voicebox.tts._piper_run = fake_piper_run
try:
    wavp = Path(tempfile.mkdtemp()) / "piper_out.wav"
    voicebox.tts.synth_tts_wav("hello there", wavp,
                               voice="Piper: Ryan (en_US, high)", rate=10)
    cmd, kw = piper_calls[-1]
    check("piper synth routed to the engine",
          cmd[cmd.index("--model") + 1].endswith("en_US-ryan-high.onnx"))
    check("piper text over stdin", kw.get("input") == "hello there")
    check("piper rate maps to length_scale",
          cmd[cmd.index("--length_scale") + 1] == "0.500")
    check("piper wav written", wavp.is_file())
    try:
        voicebox.tts._synth_piper("x", wavp, "Piper: Gone", 0)
        check("unknown piper voice fails cleanly", False)
    except RuntimeError as e:
        check("unknown piper voice fails cleanly", "not installed" in str(e))
finally:
    voicebox.tts._piper_run = old_run
    voicebox.tts.PIPER_DIR = old_piper_dir
check("no piper folder -> no piper voices",
      voicebox.tts.piper_voice_map() == {})

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

# a synth job still in flight when the voice/rate changes must drop its
# result: it rendered the OLD voice, and marking it ready would keep
# speaking that voice until the next invalidate
gate = threading.Event()


def gated_synthesize(text, voice=None, rate=0):
    gate.wait(2.0)
    return TONE.copy(), FAKE_WAV


voicebox.tts.tts_synthesize = gated_synthesize
sbank = voicebox.TTSBank(vstate, path=Path(tempfile.mkdtemp()) / "s.json")
sbank.add("stale check")                      # job blocks on the gate
with vstate.lock:
    vstate.tts_voice = "Another Voice"        # user switches mid-render
sbank.invalidate()
gate.set()                                    # old-voice render lands now
t0 = time.time()
while time.time() - t0 < 0.5 and sbank.status.get("stale check") is None:
    time.sleep(0.01)
check("in-flight synth result dropped after invalidate",
      "stale check" not in sbank.samples
      and sbank.status.get("stale check") is None)
voicebox.tts.tts_synthesize = fake_synthesize

# invalidate() racing play() between the ready-check and routing must not
# crash the UI thread - it re-renders and auto-plays instead
gbank = voicebox.TTSBank(vstate, path=Path(tempfile.mkdtemp()) / "g.json")
gbank.add("guard me")
wait_status(gbank, "guard me")
del gbank.samples["guard me"]                 # as if invalidate() interleaved
gbank.play(0)                                 # status still says "ready"
check("play survives an invalidate mid-route (re-renders)",
      wait_status(gbank, "guard me") == "ready"
      and "guard me" in gbank.samples)

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

# ------------------------------------------------- multi-engine voice helpers
check("WinRT rate: normal maps to 1.0", voicebox.winrt_speaking_rate(0) == 1.0)
check("WinRT rate: +10 doubles the pace",
      abs(voicebox.winrt_speaking_rate(10) - 2.0) < 1e-9)
check("WinRT rate: -10 halves the pace",
      abs(voicebox.winrt_speaking_rate(-10) - 0.5) < 1e-9)
check("WinRT rate clamps to the engine's 0.5..6.0 band",
      voicebox.winrt_speaking_rate(-40) == 0.5
      and voicebox.winrt_speaking_rate(40) == 6.0)
check("voice list dedups + orders SAPI before OneCore",
      voicebox.tts._dedup(
          ["Zira", " Zira ", "", "Jon", "David", "Jon"])
      == ["Zira", "Jon", "David"])
# the Windows render script routes an exact OneCore-only name to WinRT
if hasattr(voicebox.tts, "_WIN_TTS_PS"):
    script = (voicebox.tts._WIN_TTS_PS
              .replace("__PATH__", "x").replace("__VOICE__", "Microsoft Jon")
              .replace("__RATE__", "0").replace("__WRATE__", "1.0"))
    check("render script carries both engines + the exact-match guard",
          "System.Speech" in script
          and "Windows.Media.SpeechSynthesis" in script
          and "-eq $voiceName" in script)

finish()
