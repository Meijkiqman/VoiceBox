"""AI voice (RVC): lifecycle of the rvc_worker.py background process."""
import subprocess
import threading
import time
from pathlib import Path

from .config import (BASE_DIR, INPUT_DEVICE_MATCH, OUTPUT_DEVICE_MATCH,
                     RVC_DIR)

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
        if self.proc is not None:          # live switch: restart on new voice
            self.stop()
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
               str(BASE_DIR / "rvc_worker.py"),
               "--pth", str(pth), "--output-device", str(out_match)]
        index = self._index_for(pth)
        if index:
            cmd += ["--index", index]
        if isinstance(in_match, str) and in_match:
            cmd += ["--input-device", in_match]
        if self.monitor is not None and self.monitor.on:
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
        with self.state.lock:
            self.state.ai_mute = False


