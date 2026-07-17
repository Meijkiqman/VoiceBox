"""Speech translator: capture toggle, the STT->translate->TTS pipeline
(with stubbed faster-whisper/Argos), voice picking, persistence."""
import queue
import sys
import tempfile
import time
import types
from pathlib import Path

from _common import check, finish

import numpy as np

# ---- stub the heavy optional deps BEFORE the package import ----

class _Seg:
    def __init__(self, text):
        self.text = text


class _Info:
    def __init__(self, language):
        self.language = language


transcribe_calls = []


class _FakeWhisper:
    def __init__(self, *a, **k):
        pass

    def transcribe(self, audio, language=None, **k):
        transcribe_calls.append((len(audio), language))
        if language == "en":
            return [_Seg(" hello there ")], _Info("en")
        return [_Seg(" hei "), _Seg("på deg ")], _Info("no")


fw = types.ModuleType("faster_whisper")
fw.WhisperModel = _FakeWhisper
sys.modules["faster_whisper"] = fw

DIRECT = {("nb", "en"), ("en", "es"), ("en", "zh"), ("en", "nb")}


def _fake_translate(text, f, t):
    if (f, t) not in DIRECT:
        raise KeyError(f"{f}->{t}")
    return f"[{f}->{t}] {text}"


class _Pkg:
    def __init__(self, f, t):
        self.from_code, self.to_code = f, t


apkg = types.ModuleType("argostranslate.package")
apkg.get_installed_packages = lambda: [_Pkg(f, t) for f, t in DIRECT]
apkg.update_package_index = lambda: (_ for _ in ()).throw(RuntimeError("offline"))
apkg.get_available_packages = lambda: []
atrans = types.ModuleType("argostranslate.translate")
atrans.translate = _fake_translate
aroot = types.ModuleType("argostranslate")
aroot.package, aroot.translate = apkg, atrans
sys.modules["argostranslate"] = aroot
sys.modules["argostranslate.package"] = apkg
sys.modules["argostranslate.translate"] = atrans

import voicebox

voicebox.soundboard.load_clips = lambda: ([], [])

# fake TTS synth: no engine, no cache
TONE = np.full(2400, 0.2, dtype=np.float32)
FAKE_WAV = Path(tempfile.mkdtemp()) / "fake.wav"
synth_calls = []


def fake_synthesize(text, voice=None, rate=0):
    synth_calls.append((text, voice, rate))
    return TONE.copy(), FAKE_WAV


voicebox.translator.tts_synthesize = fake_synthesize

from voicebox.config import SAMPLERATE
from voicebox.state import State
from voicebox.translator import Translator, pick_voice

VOICES = ["Microsoft Zira - English (United States)",
          "Microsoft Pablo - Spanish (Spain)",
          "Microsoft Huihui - Chinese (Simplified, PRC)"]


def wait_event(state, timeout=3.0):
    try:
        return state.events.get(timeout=timeout)
    except queue.Empty:
        return None


def wait_idle(tr, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout and tr.phase:
        time.sleep(0.01)
    return not tr.phase


def speak_into(tr, state, seconds=1.0):
    """Toggle capture on, feed audio through the tap, toggle off."""
    tr.toggle()
    tap = state.trans_tap
    block = (np.random.default_rng(0)
             .normal(0, 0.1, 512).astype(np.float32))
    for _ in range(int(seconds * SAMPLERATE / 512)):
        tap.put_nowait(block)
    tr.toggle()


# ---- capture toggle wiring ----
state = State()
tr = Translator(state, voices_fn=lambda: VOICES)

tr.toggle()
check("capture sets hold", state.trans_hold is True)
check("capture sets tap", isinstance(state.trans_tap, queue.Queue))
check("row shows listening", "LISTENING" in tr.row_label())
tr.toggle()
check("stop clears hold", state.trans_hold is False)
check("stop clears tap", state.trans_tap is None)
check("empty capture reported", "heard nothing" in state.status_msg)

# ---- full pipeline: Norwegian speech -> English TTS on the mic channel ----
speak_into(tr, state)
ev = wait_event(state)
check("tts event lands", ev is not None and ev[0] == "tts")
check("event carries fx flag", ev is not None and ev[2] is True)
check("translated nb->en", "[nb->en]" in tr.last, tr.last)
check("english voice picked",
      synth_calls and synth_calls[-1][1] == VOICES[0], str(synth_calls[-1:]))
wait_idle(tr)

# ---- pivot: Norwegian -> Mandarin goes through English ----
with state.lock:
    state.trans_target = "zh"
speak_into(tr, state)
ev = wait_event(state)
check("pivot event lands", ev is not None)
check("pivoted via english", "[en->zh] [nb->en]" in tr.last, tr.last)
check("chinese voice picked",
      synth_calls and synth_calls[-1][1] == VOICES[2], str(synth_calls[-1:]))
wait_idle(tr)

# ---- same language in and out: spoken as-is, no translation ----
with state.lock:
    state.trans_source, state.trans_target = "en", "en"
speak_into(tr, state)
ev = wait_event(state)
check("same-lang event lands", ev is not None)
check("same-lang passes through", "->" not in tr.last.split("  ->  ")[0]
      and "hello there" in tr.last, tr.last)
check("whisper got language hint",
      transcribe_calls and transcribe_calls[-1][1] == "en")
wait_idle(tr)

# ---- too-short capture is dropped ----
before = state.events.qsize()
tr.toggle()
state.trans_tap.put_nowait(np.zeros(512, np.float32))
tr.toggle()
check("short capture reported", "heard nothing" in state.status_msg)
check("short capture makes no event", state.events.qsize() == before)

# ---- voice picking ----
check("explicit voice wins",
      pick_voice(VOICES, "es", VOICES[1]) == VOICES[1])
check("stale explicit falls back to hint",
      pick_voice(VOICES, "es", "Gone Voice") == VOICES[1])
check("hint match es", pick_voice(VOICES, "es") == VOICES[1])
check("hint match zh", pick_voice(VOICES, "zh") == VOICES[2])
check("no match -> default", pick_voice(["Foo"], "zh") is None)
check("no list -> default", pick_voice(None, "es") is None)

# ---- cycling writes the per-target field ----
with state.lock:
    state.trans_target = "es"
tr.cycle_voice(+1)
check("cycle voice sets es field", state.trans_voice_es == VOICES[0])
tr.cycle_target(+1)
check("cycle target moves on", state.trans_target == "zh")
tr.cycle_source(+1)
check("cycle source moves on", state.trans_source in ("no", "en", "auto"))

# ---- settings round-trip ----
with state.lock:
    state.trans_source, state.trans_target = "no", "zh"
    state.trans_voice_zh = VOICES[2]
snap = state.snapshot()
other = State()
other.restore(snap)
check("persist source", other.trans_source == "no")
check("persist target", other.trans_target == "zh")
check("persist voice", other.trans_voice_zh == VOICES[2])

finish()
