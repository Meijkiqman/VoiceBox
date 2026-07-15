"""AI voice (RVC): lifecycle of the rvc_worker.py background process."""
import socket
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from .config import (BASE_DIR, INPUT_DEVICE_MATCH, OUTPUT_DEVICE_MATCH,
                     RVC_DIR, SAMPLERATE)

class _Lerp:
    """Stateful linear-interpolation resampler: phase and the last sample
    carry across chunks, so a continuous stream stays click-free."""

    def __init__(self, src, dst):
        self.step = src / dst
        self.pos = 0.0                     # read position into [tail + chunk]
        self.tail = np.zeros(1, dtype=np.float32)

    def __call__(self, x):
        if self.step == 1.0:
            return x
        data = np.concatenate([self.tail, x])
        n = int(((len(data) - 1 - 1e-6) - self.pos) / self.step) + 1
        if n <= 0:
            self.pos -= len(x)
            self.tail = data[-1:]
            return np.zeros(0, dtype=np.float32)
        idx = self.pos + np.arange(n) * self.step
        i = idx.astype(np.int64)
        f = (idx - i).astype(np.float32)
        out = data[i] * (1.0 - f) + data[np.minimum(i + 1, len(data) - 1)] * f
        self.pos = self.pos + n * self.step - (len(data) - 1)
        self.tail = data[-1:]
        return out.astype(np.float32)


class AiFeed:
    """Local bridge for the "AI voice FX" path. The worker connects here
    (handshake: 4-byte little-endian sample rate, then an endless raw
    float32 mono stream of its converted voice) and the audio callback
    pulls blocks out with read(), already resampled to the engine rate -
    so the AI voice runs through the same pitch/echo/reverb chain as the
    mic. The buffer is primed before playback starts (burst jitter margin)
    and capped so a stalled consumer can't grow the latency unbounded."""

    MAX_BUF = SAMPLERATE * 2               # ~2 s cap: drop oldest audio
    PREFILL = 4096                         # ~85 ms margin before starting

    def __init__(self, state):
        self.state = state
        self.lock = threading.Lock()
        self.buf = np.zeros(0, dtype=np.float32)
        self.primed = False
        self.connected = False
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self._stop = threading.Event()
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        """Accept loop: one worker at a time, a restart reconnects."""
        while not self._stop.is_set():
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return                     # server socket closed
            with conn:
                self._pump(conn)
            self.connected = False
            self.clear()

    def _pump(self, conn):
        try:
            hdr = b""
            while len(hdr) < 4:
                got = conn.recv(4 - len(hdr))
                if not got:
                    return
                hdr += got
            src_sr = int.from_bytes(hdr, "little")
            if not 8000 <= src_sr <= 192000:
                return
            lerp = _Lerp(src_sr, SAMPLERATE)
            self.connected = True
            pending = b""
            while not self._stop.is_set():
                data = conn.recv(65536)
                if not data:
                    return
                pending += data
                n4 = len(pending) // 4 * 4     # float32 alignment
                if not n4:
                    continue
                x = np.frombuffer(pending[:n4], dtype=np.float32)
                pending = pending[n4:]
                y = lerp(x)
                with self.lock:
                    self.buf = np.concatenate([self.buf, y])
                    if len(self.buf) > self.MAX_BUF:
                        self.buf = self.buf[-self.MAX_BUF:]
        except Exception:
            pass

    def read(self, n):
        """n engine-rate samples for the audio callback (zero-padded on
        underrun; an underrun re-primes so playback resumes with margin)."""
        with self.lock:
            if not self.primed:
                if len(self.buf) < self.PREFILL:
                    return np.zeros(n, dtype=np.float32)
                self.primed = True
            take, self.buf = self.buf[:n], self.buf[n:]
        if len(take) < n:
            self.primed = False
            take = np.concatenate([take, np.zeros(n - len(take), np.float32)])
        return take

    def clear(self):
        with self.lock:
            self.buf = np.zeros(0, dtype=np.float32)
            self.primed = False

    def close(self):
        self._stop.set()
        try:
            self.srv.close()
        except Exception:
            pass


class AiVoice:
    """AI voice changer (RVC models like Arthur Morgan) run as a background
    worker (rvc_worker.py) on RVC's own bundled Python runtime. While the
    worker is live, VoiceBox mutes its own voice path (state.ai_mute) so the
    cable carries only the converted voice - the soundboard keeps mixing."""

    def __init__(self, state, rvc_dir=None, monitor=None):
        self.state = state
        rvc_dir = rvc_dir or getattr(state, "rvc_dir", None)
        self.rvc_dir = Path(rvc_dir) if rvc_dir else RVC_DIR
        self.monitor = monitor             # self-listen: worker mirrors voice
        self.proc = None
        self.status = "off"                # off | loading... | ON | error
        self.voices = self._scan()
        self.feed = None                   # AI-voice-FX bridge (worker -> chain)
        if self.voices:
            try:
                self.feed = AiFeed(state)
            except Exception:
                self.feed = None
        state.ai_feed = self.feed          # read by the audio callback
        self.sel = 0
        for i, p in enumerate(self.voices):
            if "arthur" in p.stem.lower():  # a sensible default, partner
                self.sel = i
                break

    @property
    def available(self):
        return bool(self.voices)

    def _scan(self):
        if not (self.rvc_dir / "runtime" / "python.exe").is_file():
            return []
        weights = self.rvc_dir / "weights"
        return sorted(weights.glob("*.pth")) if weights.is_dir() else []

    def _index_for(self, pth):
        """Find the .index that belongs to a model (accent/timbre lookup)."""
        stem = pth.stem.lower()
        for folder in (self.rvc_dir / "logs", self.rvc_dir / "weights"):
            if folder.is_dir():
                for f in folder.rglob("*.index"):
                    if stem in f.name.lower():
                        return str(f)
        return ""

    def voice_name(self):
        return self.voices[self.sel].stem if self.voices else "-"

    def cycle(self, d):
        if self.voices:
            self.select((self.sel + d) % len(self.voices))

    def select(self, i):
        """Jump straight to voice i (dropdown pick); live switch restarts."""
        if not self.voices or not (0 <= i < len(self.voices)) or i == self.sel:
            return
        self.sel = i
        with self.state.lock:              # per-character pitch memory
            self.state.ai_pitch = float(
                self.state.ai_pitches.get(self.voice_name(), 0))
        if self.proc is not None:          # live switch: restart on new voice
            self.stop()                    # (start() reads the recalled pitch)
            self.start()

    def inject(self, wav_path):
        """Feed a wav into the worker's mic input ("PLAY <path>" over stdin)
        so the model converts it like speech - the TTS-through-AI path.
        Returns False when the worker can't take it (caller falls back)."""
        proc = self.proc
        if proc is None or getattr(proc, "stdin", None) is None:
            return False
        try:
            proc.stdin.write(f"PLAY {wav_path}\n")
            proc.stdin.flush()
            return True
        except Exception:
            return False

    def set_monitor(self, on):
        """Tell a live worker to mirror the converted voice to the speakers
        ("hear myself" while the AI owns the voice path). No-op when off."""
        proc = self.proc
        if proc is None or getattr(proc, "stdin", None) is None:
            return
        try:
            proc.stdin.write(f"MONITOR {1 if on else 0}\n")
            proc.stdin.flush()
        except Exception:
            pass

    def set_pitch(self, semis):
        """Live-transpose the voice going into the model (the AI pitch row).
        No worker restart needed: RVC reads the key on every inference.
        The value is remembered per character (recalled by select())."""
        semis = int(semis)
        if self.voices:
            with self.state.lock:
                if semis:
                    self.state.ai_pitches[self.voice_name()] = semis
                else:
                    self.state.ai_pitches.pop(self.voice_name(), None)
        proc = self.proc
        if proc is None or getattr(proc, "stdin", None) is None:
            return
        try:
            proc.stdin.write(f"PITCH {int(semis)}\n")
            proc.stdin.flush()
        except Exception:
            pass

    def set_fx(self, on):
        """Live-switch the routing: on = the worker streams the converted
        voice through VoiceBox's effect chain (the "AI voice FX" row), off =
        it feeds the cable directly, as before. Self-listen ownership moves
        with it: through the chain, the main HEAR mirror already carries the
        AI voice, so the worker-side mirror is released."""
        if self.feed is not None:
            self.feed.clear()              # drop audio from the old routing
        proc = self.proc
        if proc is not None and getattr(proc, "stdin", None) is not None:
            try:
                proc.stdin.write(f"FX {1 if on else 0}\n")
                proc.stdin.flush()
            except Exception:
                pass
        if self.monitor is not None and self.monitor.on:
            self.set_monitor(not on)

    def toggle(self):
        if self.proc is not None:
            self.stop()
        else:
            self.start()

    def start(self):
        if self.proc is not None or not self.voices:
            return
        pth = self.voices[self.sel]
        # the worker opens its own streams: hand it the same devices the
        # main stream uses (menu selection first, constants as fallback)
        out_match = self.state.output_device or OUTPUT_DEVICE_MATCH
        in_match = self.state.input_device or INPUT_DEVICE_MATCH
        cmd = [str(self.rvc_dir / "runtime" / "python.exe"),
               str(BASE_DIR / "rvc_worker.py"), "--pth", str(pth),
               # "" = the worker's system default; str(None) would make it
               # search for a device literally named "None"
               "--output-device",
               "" if out_match is None else str(out_match)]
        if int(self.state.ai_pitch):
            cmd += ["--pitch", str(int(self.state.ai_pitch))]
        index = self._index_for(pth)
        if index:
            cmd += ["--index", index]
        if isinstance(in_match, str) and in_match:
            cmd += ["--input-device", in_match]
        if self.feed is not None:
            cmd += ["--fx-port", str(self.feed.port)]
            if self.state.ai_fx:
                cmd += ["--fx"]            # FX routing already on at launch
        if (self.monitor is not None and self.monitor.on
                and not self.state.ai_fx):
            cmd += ["--monitor"]           # self-listen already on at launch
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=str(self.rvc_dir), text=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception as e:
            self.status = "error"
            self.state.status_msg = f"AI: {e}"
            self.state.status_at = time.time()
            return
        self.status = "loading..."
        with self.state.lock:
            self.state.ai_mute = True
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _reader(self, proc):
        """Follow one worker's stdout (also keeps its pipe from filling)."""
        try:
            for line in proc.stdout:
                line = line.strip()
                if line.startswith("STATUS running"):
                    self.status = "ON"
                    if self.state.cues is not None:
                        self.state.cues.ai_ready()
                elif line.startswith("STATUS error"):
                    self.status = "error"
                    self.state.status_msg = f"AI: {line[13:][:70]}"
                    self.state.status_at = time.time()
        except Exception:
            pass
        if proc is self.proc:              # worker died on its own
            self.proc = None
            if self.status != "error":
                self.status = "off"
            if self.state.cues is not None:
                self.state.cues.ai_died()
            with self.state.lock:
                self.state.ai_mute = False

    def stop(self):
        proc, self.proc = self.proc, None
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
        self.status = "off"
        if self.feed is not None:
            self.feed.clear()
        with self.state.lock:
            self.state.ai_mute = False

    def close(self):
        """Shutdown: stop the worker and release the FX bridge socket."""
        self.stop()
        if self.feed is not None:
            self.feed.close()


