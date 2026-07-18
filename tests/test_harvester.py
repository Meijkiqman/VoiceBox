"""Voice harvester: segmentation, quality gate, dataset cap, persistence -
and the Trainer row's refusal conditions."""
import tempfile
import time
from pathlib import Path

from _common import check, finish

import numpy as np
import voicebox

voicebox.soundboard.load_clips = lambda: ([], [])

from voicebox.config import HARVEST_CAP_MIN, SAMPLERATE
from voicebox.harvester import Harvester
from voicebox.state import State
from voicebox.trainer import Trainer

BLOCK = 512


def feed(state, samples):
    q = state.harvest_q
    for i in range(0, len(samples) - BLOCK + 1, BLOCK):
        q.put(samples[i:i + BLOCK])


def wait_for(cond, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


def speech(seconds, amp=0.3):
    rng = np.random.default_rng(1)
    return (amp * rng.normal(0, 0.5, int(seconds * SAMPLERATE))
            .clip(-1, 1)).astype(np.float32)


silence = lambda s: np.zeros(int(s * SAMPLERATE), dtype=np.float32)

tmp = Path(tempfile.mkdtemp())
state = State()
h = Harvester(state, out_dir=tmp)

check("starts empty", h.seconds == 0 and not h.on)
h.start()
check("start sets queue", state.harvest_q is not None)

# a 3 s utterance framed by silence -> exactly one kept clip
feed(state, silence(0.5))
feed(state, speech(3.0))
feed(state, silence(1.0))
check("clip saved", wait_for(lambda: h.kept == 1), f"kept={h.kept}")
wavs = list(tmp.glob("*.wav"))
check("wav on disk", len(wavs) == 1)
check("duration counted", 2.5 < h.seconds < 5.0, f"{h.seconds:.2f}s")

# too-short blip -> dropped, nothing new on disk
dropped0 = h.dropped
feed(state, speech(0.8))
feed(state, silence(1.0))
check("short clip dropped", wait_for(lambda: h.dropped == dropped0 + 1))
check("no extra wav", len(list(tmp.glob("*.wav"))) == 1)

# clipped/distorted take -> dropped
dropped0 = h.dropped
feed(state, np.sign(speech(3.0)).astype(np.float32))   # square wave at +-1.0
feed(state, silence(1.0))
check("clipped take dropped", wait_for(lambda: h.dropped == dropped0 + 1))
check("still one wav", len(list(tmp.glob("*.wav"))) == 1)

h.stop()
check("stop clears queue", state.harvest_q is None)

# toggle persists the preference
h.toggle()
check("toggle on persists", state.harvest_on is True and h.on)
h.toggle()
check("toggle off persists", state.harvest_on is False and not h.on)

# dataset cap: a full dataset refuses to start
h.seconds = HARVEST_CAP_MIN * 60 + 1
h.start()
check("full dataset refuses start", not h.on and "retrain" in state.status_msg)
check("full label", "full" in h.label())

# crossing the cap mid-session stops collection and clears the persisted flag
h2dir = Path(tempfile.mkdtemp())
h2 = Harvester(state, out_dir=h2dir)
h2.seconds = HARVEST_CAP_MIN * 60 - 3
h2.start()
with state.lock:
    state.harvest_on = True
feed(state, speech(4.0))
feed(state, silence(1.0))
check("cap auto-stops harvesting",
      wait_for(lambda: not h2.on and state.harvest_on is False))
check("cap clip still saved", len(list(h2dir.glob("*.wav"))) == 1)

# harvest_on is in the settings snapshot
with state.lock:
    state.harvest_on = True
other = State()
other.restore(state.snapshot())
check("harvest_on persisted", other.harvest_on is True)

# ---- Trainer guardrails (no RVC runtime in CI: launch must refuse) ----
state2 = State()
h2 = Harvester(state2, out_dir=tmp)


class _FakeAi:
    proc = object()               # "worker is live"


t = Trainer(state2, ai=_FakeAi(), harvester=h2)
check("trainer refuses while AI live",
      t.launch() is False and "GPU" in state2.status_msg)
t2 = Trainer(state2, ai=None, harvester=h2)
r = t2.launch()
check("trainer refuses without runtime or data", r is False)

finish()
