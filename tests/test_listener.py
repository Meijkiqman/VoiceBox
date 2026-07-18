"""Incoming speech translator: device guardrails, segmentation ->
caption pipeline (stubbed models), English skip, spoken captions,
on-demand pack failure, persistence."""
import queue
import sys
import tempfile
import time
import types
from pathlib import Path

from _common import check, finish

import numpy as np

# ---- stub the heavy deps BEFORE the package import (same as test_translator)

CUR = {"lang": "es", "text": " hola amigo "}


class _Seg:
    def __init__(self, text):
        self.text = text


class _Info:
    def __init__(self, language):
        self.language = language


class _FakeWhisper:
    def __init__(self, *a, **k):
        pass

    def transcribe(self, audio, language=None, **k):
        return [_Seg(CUR["text"])], _Info(language or CUR["lang"])


fw = types.ModuleType("faster_whisper")
fw.WhisperModel = _FakeWhisper
sys.modules["faster_whisper"] = fw

DIRECT = {("nb", "en"), ("en", "es"), ("en", "zh"), ("es", "en")}


def _fake_translate(text, f, t):
    if (f, t) not in DIRECT:
        raise KeyError(f"{f}->{t}")
    return f"[{f}->{t}] {text}"


class _Pkg:
    def __init__(self, f, t):
        self.from_code, self.to_code = f, t


apkg = types.ModuleType("argostranslate.package")
apkg.get_installed_packages = lambda: [_Pkg(f, t) for f, t in DIRECT]
# index reachable but empty: an unknown language is a DEFINITIVE miss
# (blacklisted), not a transient network failure
apkg.update_package_index = lambda: None
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

TONE = np.full(2400, 0.2, dtype=np.float32)
FAKE_WAV = Path(tempfile.mkdtemp()) / "fake.wav"
spoken = []


def fake_synthesize(text, voice=None, rate=0):
    spoken.append((text, voice))
    return TONE.copy(), FAKE_WAV


voicebox.listener.tts_synthesize = fake_synthesize

from voicebox.config import SAMPLERATE
from voicebox.listener import Listener
from voicebox.state import State
from voicebox.translator import Translator

BLOCK = 512


class DummyStream:
    made = []

    def __init__(self, *a, **k):
        self.kw = k
        self.closed = False
        DummyStream.made.append(self)

    def start(self):
        pass

    def close(self):
        self.closed = True


class FakePlayer:
    def __init__(self):
        self.played = []

    def play_raw(self, samples):
        self.played.append(samples)


def wait_for(cond, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


def utter(listener, seconds=2.0, amp=0.3):
    """Feed one framed utterance through the segmenter queue."""
    rng = np.random.default_rng(2)
    speech = (amp * rng.normal(0, 0.5, int(seconds * SAMPLERATE))
              .clip(-1, 1)).astype(np.float32)
    silence = np.zeros(int(1.0 * SAMPLERATE), dtype=np.float32)
    for buf in (speech, silence):
        for i in range(0, len(buf) - BLOCK + 1, BLOCK):
            listener._q.put(buf[i:i + BLOCK])


state = State()
state.listen_pass = False          # no real audio devices in CI
tr = Translator(state, voices_fn=lambda: ["Microsoft Zira - English (US)"])
player = FakePlayer()
ln = Listener(state, tr, player=player, stream_cls=DummyStream)

# no device configured and no CABLE-B installed -> refuses with guidance
ln.start()
check("no device refuses", not ln.on and "CABLE-B" in state.status_msg)

# with a (faked) resolvable device it starts
state.listen_device = "Fake Cable Out"
voicebox.listener.find_device = lambda name, kind: 3
ln.start()
check("starts with device", ln.on)
check("stream on picked device", DummyStream.made[-1].kw.get("device") == 3)
check("row shows listening", "listening" in ln.row_label())

# a Spanish utterance -> English caption with the language tag
utter(ln)
check("caption lands", wait_for(lambda: len(ln.captions) == 1),
      f"{len(ln.captions)}")
t_, lang_, text_ = ln.captions[-1]
check("caption tagged es", lang_ == "es")
check("caption translated", "[es->en]" in text_ and "hola amigo" in text_)
check("caption line format", any("[es]" in l for l in ln.caption_lines()))
check("nothing spoken by default", not player.played)

# English utterances need no translation -> no caption
CUR["lang"], CUR["text"] = "en", " already english "
utter(ln)
time.sleep(0.6)
check("english skipped", len(ln.captions) == 1)

# spoken captions: translated text goes to the local player
with state.lock:
    state.listen_speak = True
CUR["lang"], CUR["text"] = "es", " que tal "
utter(ln)
check("caption spoken", wait_for(lambda: len(player.played) == 1))
check("spoke the translation", spoken and "[es->en]" in spoken[-1][0])

# a language with no pack (and no network in the stub) -> clean error
CUR["lang"], CUR["text"] = "xx", " mystery tongue "
utter(ln)
check("no-pack reported",
      wait_for(lambda: "no translation pack for 'xx'" in state.status_msg))
n_caps = len(ln.captions)
CUR["text"] = " mystery again "
utter(ln)
time.sleep(0.6)
check("no-pack failure is remembered", len(ln.captions) == n_caps)

# caption lines are truncated for the strip
ln.captions.append((time.time(), "es", "x" * 200))
long_lines = ln.caption_lines(width=90)
check("caption truncated", long_lines[-1].endswith("...")
      and len(long_lines[-1]) <= 90)
ln.captions.pop()

# live device switch: cycle stops the old stream and opens a new one
ln._input_names = lambda: ["Fake Cable Out", "Other Device"]
n_streams = len(DummyStream.made)
old_stream = ln.stream
ln.cycle_device(+1)
check("cycle switches device", state.listen_device == "Other Device")
check("cycle reopens stream", ln.on and old_stream.closed
      and len(DummyStream.made) == n_streams + 1)

# stop + toggle persistence
ln.stop()
check("stop closes stream", not ln.on and DummyStream.made[-1].closed)
ln.toggle()
check("toggle persists on", state.listen_on is True and ln.on)
ln.toggle()
check("toggle persists off", state.listen_on is False and not ln.on)

# settings round-trip
with state.lock:
    state.listen_speak, state.listen_pass = True, False
other = State()
other.restore(state.snapshot())
check("persist listen_device", other.listen_device == "Other Device")
check("persist listen_speak", other.listen_speak is True)
check("persist listen_pass", other.listen_pass is False)

finish()
