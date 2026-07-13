"""AI voice (RVC) launcher: discovery, index matching, lifecycle, mute contract."""
import io
import queue as _queue
import tempfile
import time
from pathlib import Path

from _common import check, finish

import numpy as np
import voicebox

voicebox.soundboard.load_clips = lambda: ([np.full(4000, 0.1, np.float32)], ["clip1"])

# ------------------------------------------------------------ fake RVC folder
root = Path(tempfile.mkdtemp()) / "rvc"
(root / "runtime").mkdir(parents=True)
(root / "runtime" / "python.exe").write_bytes(b"")
(root / "weights").mkdir()
for n in ("ArthurMorgan", "kratos", "vader"):
    (root / "weights" / (n + ".pth")).write_bytes(b"x")
(root / "logs" / "ArthurMorgan").mkdir(parents=True)
idx = root / "logs" / "ArthurMorgan" / "added_IVF401_Flat_nprobe_1_ArthurMorgan_v2.index"
idx.write_bytes(b"x")

state = voicebox.State()
ai = voicebox.AiVoice(state, rvc_dir=root)
check("voices discovered alphabetically",
      [v.stem for v in ai.voices] == ["ArthurMorgan", "kratos", "vader"])
check("available with runtime + weights", ai.available is True)
check("defaults to Arthur Morgan", ai.voice_name() == "ArthurMorgan")
check("index matched by model stem", ai._index_for(ai.voices[0]) == str(idx))
check("no index -> empty string", ai._index_for(ai.voices[1]) == "")
ai.cycle(1)
check("cycle forward", ai.voice_name() == "kratos")
ai.cycle(-1)
check("cycle back", ai.voice_name() == "ArthurMorgan")

check("missing RVC dir -> unavailable",
      voicebox.AiVoice(state, rvc_dir=root / "nope").available is False)
no_runtime = Path(tempfile.mkdtemp())
(no_runtime / "weights").mkdir()
(no_runtime / "weights" / "x.pth").write_bytes(b"x")
check("weights without runtime -> unavailable",
      voicebox.AiVoice(state, rvc_dir=no_runtime).available is False)

# --------------------------------------------------- lifecycle w/ fake worker
class FakeStdout:
    """Iterable stdout that blocks like a live process pipe."""
    def __init__(self):
        self.q = _queue.Queue()
    def __iter__(self):
        return self
    def __next__(self):
        line = self.q.get()
        if line is None:
            raise StopIteration
        return line

class FakeProc:
    last = None
    def __init__(self, cmd, **kw):
        FakeProc.last = self
        self.cmd = cmd
        self.stdout = FakeStdout()
        self.terminated = False
    def terminate(self):
        self.terminated = True
        self.stdout.q.put(None)          # EOF, like a real dying process

real_popen = voicebox.subprocess.Popen
voicebox.subprocess.Popen = FakeProc
try:
    ai.start()
    check("start launches worker with model + index",
          "--pth" in FakeProc.last.cmd and "--index" in FakeProc.last.cmd
          and str(root / "weights" / "ArthurMorgan.pth") in FakeProc.last.cmd)
    check("start reports loading", ai.status == "loading...")
    check("start mutes VoiceBox voice", state.ai_mute is True)

    FakeProc.last.stdout.q.put("STATUS running sr=40000 block=14000\n")
    time.sleep(0.3)
    check("worker running -> status ON", ai.status == "ON")

    running = FakeProc.last
    ai.stop()
    time.sleep(0.3)
    check("stop terminates worker", running.terminated and ai.proc is None)
    check("stop unmutes VoiceBox voice", state.ai_mute is False)
    check("stop resets status", ai.status == "off")

    # worker dying on its own must also unmute
    ai.start()
    FakeProc.last.stdout.q.put("STATUS error no output device matching 'CABLE'\n")
    FakeProc.last.stdout.q.put(None)     # EOF: process exited
    time.sleep(0.3)
    check("worker death -> status error", ai.status == "error")
    check("worker death -> unmuted", state.ai_mute is False and ai.proc is None)
    check("worker error surfaces in status line",
          state.status_msg.startswith("AI:"))
    ai.status = "off"

    # no-index voice: --index must be absent
    ai.cycle(1)                          # kratos (proc is None, no restart)
    ai.start()
    check("no-index voice omits --index", "--index" not in FakeProc.last.cmd)
    ai.stop()
finally:
    voicebox.subprocess.Popen = real_popen

# --------------------------------------------------------- mute contract, DSP
cb = voicebox.make_callback(state)
frames = voicebox.BLOCKSIZE
sine = (0.4 * np.sin(2 * np.pi * 220 * np.arange(frames) / voicebox.SAMPLERATE)
        ).astype(np.float32).reshape(-1, 1)
out = np.zeros((frames, 1), dtype=np.float32)

with state.lock:
    state.ai_mute = True
cb(sine, out, frames, None, None)
check("ai_mute silences the voice path", np.abs(out).max() == 0.0)
state.events.put(0)                      # soundboard must still work
cb(sine, out, frames, None, None)
check("soundboard still mixes while muted", np.abs(out).max() > 0.05)
state.events.put("stop")
with state.lock:
    state.ai_mute = False
cb(sine, out, frames, None, None)
check("unmute restores the voice path", np.abs(out).max() > 0.1)

# ------------------------------------------------------------------ menu rows
import threading
menu = voicebox.Menu(state, threading.Event(), None, None, ai)
labels = [it.label for it in menu.items]
check("AI rows present when available",
      "AI voice" in labels and "AI character" in labels)
check("AI rows ordered before Sounds to mic",
      labels.index("AI voice") < labels.index("Sounds to mic"))
menu_no_ai = voicebox.Menu(state, threading.Event())
check("no AI rows without AiVoice",
      all(it.label not in ("AI voice", "AI character") for it in menu_no_ai.items))

finish()
