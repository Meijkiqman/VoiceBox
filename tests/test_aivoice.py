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
        self.stdin = io.StringIO()
        self.terminated = False
    def terminate(self):
        self.terminated = True
        self.stdout.q.put(None)          # EOF, like a real dying process

class CueRec:
    def __init__(self): self.calls = []
    def ai_ready(self): self.calls.append("ready")
    def ai_died(self): self.calls.append("died")
    def mute(self, m): self.calls.append(("mute", m))

real_popen = voicebox.subprocess.Popen
voicebox.subprocess.Popen = FakeProc
try:
    state.cues = CueRec()
    ai.start()
    check("start launches worker with model + index",
          "--pth" in FakeProc.last.cmd and "--index" in FakeProc.last.cmd
          and str(root / "weights" / "ArthurMorgan.pth") in FakeProc.last.cmd)
    check("start reports loading", ai.status == "loading...")
    check("start mutes VoiceBox voice", state.ai_mute is True)

    FakeProc.last.stdout.q.put("STATUS running sr=40000 block=14000\n")
    time.sleep(0.3)
    check("worker running -> status ON", ai.status == "ON")
    check("worker ready fires the ready cue", "ready" in state.cues.calls)

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
    check("worker death fires the died cue", "died" in state.cues.calls)
    check("clean stop fired no died cue",
          state.cues.calls.count("died") == 1)
    state.cues = None
    ai.status = "off"

    # no-index voice: --index must be absent
    ai.cycle(1)                          # kratos (proc is None, no restart)
    ai.start()
    check("no-index voice omits --index", "--index" not in FakeProc.last.cmd)
    check("self-listen off omits --monitor",
          "--monitor" not in FakeProc.last.cmd)
    ai.stop()

    # HEAR while AI live: worker gets --monitor at launch / MONITOR over stdin
    class FakeMon:
        on = True
    ai_mon = voicebox.AiVoice(state, rvc_dir=root, monitor=FakeMon())
    ai_mon.start()
    check("self-listen on adds --monitor", "--monitor" in FakeProc.last.cmd)
    ai_mon.set_monitor(False)
    check("HEAR toggle reaches the live worker",
          FakeProc.last.stdin.getvalue() == "MONITOR 0\n")
    ai_mon.stop()

    # ---- AI voice FX routing (worker -> VoiceBox effect chain) ----------
    check("fx bridge port handed to the worker",
          "--fx-port" in FakeProc.last.cmd)
    check("fx off omits --fx", "--fx" not in FakeProc.last.cmd)
    with state.lock:
        state.ai_fx = True
    ai_fx = voicebox.AiVoice(state, rvc_dir=root, monitor=FakeMon())
    ai_fx.start()
    check("fx on adds --fx at launch", "--fx" in FakeProc.last.cmd)
    check("fx on suppresses the worker-side monitor",
          "--monitor" not in FakeProc.last.cmd)
    ai_fx.set_fx(False)
    check("FX toggle reaches the live worker",
          "FX 0\n" in FakeProc.last.stdin.getvalue())
    check("self-listen handed back to the worker when FX goes off",
          FakeProc.last.stdin.getvalue().endswith("MONITOR 1\n"))
    ai_fx.stop()
    with state.lock:
        state.ai_fx = False

    # ---- AI pitch: --pitch at launch, "PITCH n" live ---------------------
    with state.lock:
        state.ai_pitch = -5.0
    ai_p = voicebox.AiVoice(state, rvc_dir=root)
    ai_p.start()
    check("AI pitch handed to the worker at launch",
          "--pitch" in FakeProc.last.cmd and "-5" in FakeProc.last.cmd)
    ai_p.set_pitch(3)
    check("live AI pitch reaches the worker",
          "PITCH 3\n" in FakeProc.last.stdin.getvalue())
    ai_p.stop()
    with state.lock:
        state.ai_pitch = 0.0
    ai_zero = voicebox.AiVoice(state, rvc_dir=root)
    ai_zero.start()
    check("zero AI pitch omits --pitch", "--pitch" not in FakeProc.last.cmd)
    ai_zero.stop()
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

# ------------------------------------- AI voice through the effect chain
class StubFeed:
    def read(self, n):
        return np.full(n, 0.25, dtype=np.float32)

state.ai_feed = StubFeed()
with state.lock:
    state.ai_mute = True
    state.ai_fx = True
cb(sine, out, frames, None, None)
check("AI voice FX feeds the chain while muted", np.abs(out).max() > 0.2,
      f"peak={np.abs(out).max():.3f}")
with state.lock:
    state.ai_fx = False
cb(sine, out, frames, None, None)
check("FX off keeps the muted voice path silent", np.abs(out).max() == 0.0)
with state.lock:
    state.ai_mute = False
state.ai_feed = None

# ---------------------------------------------- AI voice FX bridge (AiFeed)
import socket as _socket
feed = voicebox.AiFeed(state)
cli = _socket.create_connection(("127.0.0.1", feed.port), timeout=3)
cli.sendall((24000).to_bytes(4, "little"))          # half the engine rate
tone = (0.5 * np.sin(2 * np.pi * 220 * np.arange(24000) / 24000)
        ).astype(np.float32)
cli.sendall(tone.tobytes())
got, deadline = 0, time.time() + 3
while time.time() < deadline:
    with feed.lock:
        got = len(feed.buf)
    if got >= 40000:
        break
    time.sleep(0.05)
check("bridge receives + resamples to the engine rate", got >= 40000,
      f"{got} samples")
blk = feed.read(4096)
check("bridge read hands out the converted voice",
      float(np.abs(blk).max()) > 0.3, f"peak={float(np.abs(blk).max()):.3f}")
for _ in range(64):                        # drain past the end: must zero-pad
    blk = feed.read(4096)
check("bridge underrun pads with silence", float(np.abs(blk).max()) == 0.0)
cli.close()
feed.close()

# ------------------------------------------------------------------ menu rows
import threading
menu = voicebox.Menu(state, threading.Event(), None, None, ai)
labels = [it.label for it in menu.items]
check("AI rows present when available",
      "AI voice" in labels and "AI character" in labels
      and "AI pitch" in labels and "AI voice FX" in labels)
check("AI rows ordered before Sounds to mic",
      labels.index("AI voice") < labels.index("Sounds to mic"))
menu_no_ai = voicebox.Menu(state, threading.Event())
check("no AI rows without AiVoice",
      all(it.label not in ("AI voice", "AI character") for it in menu_no_ai.items))

# HEAR toggle must forward the new state to the AI worker
class RecAI:
    available = False
    def __init__(self): self.calls = []
    def set_monitor(self, on): self.calls.append(on)

class TogMon:
    def __init__(self): self.on, self.error = False, ""
    def toggle(self): self.on = not self.on

rec = RecAI()
menu_fwd = voicebox.Menu(state, threading.Event(), TogMon(), None, rec)
menu_fwd._toggle_monitor()
menu_fwd._toggle_monitor()
check("HEAR toggle forwarded to the AI worker", rec.calls == [True, False])

finish()
