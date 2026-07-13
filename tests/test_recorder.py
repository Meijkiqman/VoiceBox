"""Recorder: queue-fed wav writer, callback mirroring, failure paths."""
import tempfile
import threading
import time
import types
from pathlib import Path

from _common import check, finish

import numpy as np
import voicebox

voicebox.soundboard.load_clips = lambda: ([], [])
tmpdir = Path(tempfile.mkdtemp())


class FakeWav:
    instances = []

    def __init__(self, path, mode="w", samplerate=None, channels=None,
                 subtype=None):
        if str(path).endswith("boom.wav"):
            raise OSError("disk full")
        self.path, self.samplerate = path, samplerate
        self.channels, self.subtype = channels, subtype
        self.blocks = []
        self.closed = False
        FakeWav.instances.append(self)

    def write(self, block):
        self.blocks.append(np.asarray(block).copy())

    def close(self):
        self.closed = True


real_sf = voicebox.audio.sf
voicebox.audio.sf = types.SimpleNamespace(SoundFile=FakeWav)

# ------------------------------------------------------------ start/feed/stop
state = voicebox.State()
rec = voicebox.Recorder(state, folder=tmpdir)
check("recorder starts off", not rec.on)
rec.start()
check("start arms the callback queue", rec.on and state.record_q is not None)
wav = FakeWav.instances[-1]
check("wav opened mono at the engine rate",
      wav.samplerate == voicebox.SAMPLERATE and wav.channels == 1
      and wav.subtype == "PCM_16")
check("file lands in the recordings folder",
      str(wav.path).startswith(str(tmpdir)) and str(wav.path).endswith(".wav"))

cb = voicebox.make_callback(state)
frames = voicebox.BLOCKSIZE
loud = np.full((frames, 1), 0.25, dtype=np.float32)
out = np.zeros((frames, 1), dtype=np.float32)
for _ in range(5):
    cb(loud, out, frames, None, None)
rec.stop()
check("stop disarms the queue", not rec.on and state.record_q is None)
check("writer drained every block and closed the file",
      wav.closed and len(wav.blocks) == 5
      and all(abs(b.max() - 0.25) < 1e-6 for b in wav.blocks))
check("stop announces the saved file",
      "saved" in state.status_msg and ".wav" in state.status_msg)

# ------------------------------------------------------------------- toggle
rec.toggle()
check("toggle starts a second recording",
      rec.on and FakeWav.instances[-1] is not wav)
rec.toggle()
check("toggle stops it again", not rec.on and FakeWav.instances[-1].closed)

# -------------------------------------------------------------- failure path
bad = voicebox.Recorder(state, folder=tmpdir)
real_strftime = voicebox.time.strftime
voicebox.time.strftime = lambda fmt: "boom"
bad.start()
voicebox.time.strftime = real_strftime
check("open failure stays off with an error",
      not bad.on and "disk full" in bad.error
      and state.status_msg.startswith("record:"))

# -------------------------------------------------- full queue never blocks
q = __import__("queue").Queue(maxsize=1)
state.record_q = q
q.put_nowait(np.zeros(4, np.float32))
t0 = time.time()
cb(loud, out, frames, None, None)
check("full record queue drops instead of blocking", time.time() - t0 < 0.1)
state.record_q = None

voicebox.audio.sf = real_sf
finish()
