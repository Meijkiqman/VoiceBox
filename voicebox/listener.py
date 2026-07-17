"""Incoming speech translator: captions what the OTHERS say, in English.

Route Discord's output to a second virtual cable (e.g. VB-CABLE A+B's
"CABLE-B Input") and point the Listen device row at its "CABLE-B Output"
side. While listening is on, that stream is passed through to the default
speakers (so voice chat still sounds normal) and segmented into utterances
- the same silence-based cut the harvester uses. Each utterance goes
through the shared Whisper + Argos pipeline (any detected language ->
English), and lands as a caption line at the bottom of the window;
optionally it is also spoken on the speakers with the English TTS voice.

Argos packs for languages beyond the preinstalled set are downloaded on
first encounter (network needed once per language)."""
import queue
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd

from .audio import find_device
from .config import (BLOCKSIZE, CHANNELS, LISTEN_CAPTION_N, LISTEN_CAPTION_S,
                     LISTEN_DEVICE_HINTS, LISTEN_HANG_S, LISTEN_MAX_S,
                     LISTEN_MIN_S, LISTEN_PRE_S, LISTEN_TARGET,
                     LISTEN_THRESH_DB, SAMPLERATE)
from .tts import tts_synthesize

class Listener:
    """Owns the capture stream and its segmentation/translation threads.
    stream_cls is injectable for the headless tests."""

    def __init__(self, state, translator, player=None, stream_cls=None):
        self.state = state
        self.translator = translator
        self.player = player           # LocalPlayer, for spoken captions
        self.stream_cls = stream_cls or sd.InputStream
        self.captions = deque(maxlen=50)   # (timestamp, lang, text)
        self.error = ""
        self.phase = ""                # translation pipeline activity
        self.stream = None
        self.out_stream = None         # passthrough to the default speakers
        self.pass_q = None
        self._q = None                 # capture blocks -> segmenter
        self._jobs = None              # utterances -> translation worker
        self._stop = None
        self.device_name = ""          # resolved name, for the row

    # ---- device selection ----

    def _input_names(self):
        try:
            return [d["name"] for d in sd.query_devices()
                    if d["max_input_channels"] > 0]
        except Exception:
            return []

    def _auto_device(self):
        """A second-cable output side, if one is installed. Never the primary
        "CABLE Output" - that would capture the user's own converted voice."""
        for name in self._input_names():
            if any(h in name.lower() for h in LISTEN_DEVICE_HINTS):
                return name
        return None

    def device_options(self):
        return [None] + self._input_names()

    def cycle_device(self, d=1):
        opts = self.device_options()
        if len(opts) < 2:
            return
        try:
            i = opts.index(self.state.listen_device)
        except ValueError:
            i = 0
        with self.state.lock:
            self.state.listen_device = opts[(i + (1 if d >= 0 else -1))
                                            % len(opts)]
        if self.on:                    # live switch, same as the engine rows
            self.stop()
            self.start()

    def device_label(self, width=24):
        name = self.state.listen_device
        if name is None:
            auto = self._auto_device()
            name = f"auto ({auto})" if auto else "auto (none found)"
        return name if len(name) <= width else name[:width - 3] + "..."

    # ---- lifecycle ----

    @property
    def on(self):
        return self.stream is not None

    def row_label(self):
        if self.on:
            return self.phase or "ON - listening"
        if self.error:
            return "error - see status"
        return "off"

    def _report(self, msg):
        self.state.status_msg = msg
        self.state.status_at = time.time()

    def toggle(self):
        self.stop() if self.on else self.start()
        with self.state.lock:
            self.state.listen_on = self.on   # persisted preference

    def start(self):
        if self.on:
            return
        name = self.state.listen_device or self._auto_device()
        if name is None:
            self.error = "no capture device"
            self._report("incoming: set Discord's output to a second cable "
                         "(CABLE-B) and pick its Output side as Listen device")
            return
        try:
            dev = find_device(name, "input")
        except SystemExit:
            self.error = f"'{name}' not found"
            self._report(f"incoming: device '{name[:40]}' not found")
            return
        self._q = queue.Queue(maxsize=256)
        self._jobs = queue.Queue()
        self._stop = threading.Event()
        try:
            self.stream = self.stream_cls(
                samplerate=SAMPLERATE, blocksize=BLOCKSIZE, dtype="float32",
                channels=CHANNELS, device=dev, callback=self._capture_cb)
            self.stream.start()
        except Exception as e:
            self.stream = None
            self.error = str(e)
            self._report(f"incoming: {e}")
            return
        self.device_name = name
        self.error = ""
        threading.Thread(target=self._segmenter, daemon=True).start()
        threading.Thread(target=self._worker, daemon=True).start()
        if self.state.listen_pass:
            self._open_passthrough()
        self._report(f"incoming: listening on {name[:40]}")

    def stop(self):
        self._stop_passthrough()
        if self._stop is not None:
            self._stop.set()
        if self._q is not None:
            try:
                self._q.put_nowait(None)     # wake the segmenter
            except queue.Full:
                pass
        if self._jobs is not None:
            self._jobs.put(None)             # wake the worker
        if self.stream is not None:
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.phase = ""

    # ---- audio path ----

    def _capture_cb(self, indata, frames, time_info, status):
        block = indata[:, 0].copy()
        q = self._q
        if q is not None:
            try:
                q.put_nowait(block)
            except queue.Full:
                pass                         # translator lagging: drop
        pq = self.pass_q
        if pq is not None:
            try:
                pq.put_nowait(block)
            except queue.Full:
                pass                         # speakers lagging: drop

    def _open_passthrough(self):
        """Mirror the captured audio to the default speakers, so routing
        Discord into the cable doesn't leave the user deaf to the chat."""
        q = queue.Queue(maxsize=8)

        def cb(outdata, frames, time_info, status):
            try:
                y = q.get_nowait()
            except queue.Empty:
                y = np.zeros(frames, np.float32)
            if len(y) < frames:
                y = np.concatenate([y, np.zeros(frames - len(y), np.float32)])
            outdata[:, 0] = y[:frames]

        try:
            self.out_stream = sd.OutputStream(
                samplerate=SAMPLERATE, blocksize=BLOCKSIZE, dtype="float32",
                channels=CHANNELS, callback=cb)
            self.out_stream.start()
            self.pass_q = q
        except Exception as e:
            self.out_stream = None
            self._report(f"incoming: passthrough failed: {e}")

    def _stop_passthrough(self):
        self.pass_q = None
        if self.out_stream is not None:
            try:
                self.out_stream.close()
            except Exception:
                pass
            self.out_stream = None

    def set_passthrough(self, on):
        with self.state.lock:
            self.state.listen_pass = bool(on)
        if self.on:
            self._stop_passthrough()
            if on:
                self._open_passthrough()

    # ---- segmentation (same shape as the harvester's) ----

    def _segmenter(self):
        thresh = 10.0 ** (LISTEN_THRESH_DB / 20.0)
        pre_max = int(LISTEN_PRE_S * SAMPLERATE)
        pre, pre_len, seg, quiet = [], 0, None, 0.0
        stop, q, jobs = self._stop, self._q, self._jobs
        while not stop.is_set():
            try:
                block = q.get(timeout=0.5)
            except queue.Empty:
                continue
            if block is None:
                break
            speaking = float(np.abs(block).max()) >= thresh
            if seg is None:
                if speaking:
                    seg = pre + [block]
                    quiet = 0.0
                else:
                    pre.append(block)
                    pre_len += len(block)
                    while pre_len - len(pre[0]) >= pre_max:
                        pre_len -= len(pre[0])
                        pre.pop(0)
                continue
            seg.append(block)
            quiet = 0.0 if speaking else quiet + len(block) / SAMPLERATE
            seg_s = sum(len(b) for b in seg) / SAMPLERATE
            if quiet >= LISTEN_HANG_S or seg_s >= LISTEN_MAX_S:
                utt = np.concatenate(seg)
                pre, pre_len, seg, quiet = [], 0, None, 0.0
                if len(utt) >= LISTEN_MIN_S * SAMPLERATE:
                    jobs.put(utt)

    # ---- translation worker ----

    def _worker(self):
        stop, jobs = self._stop, self._jobs
        while not stop.is_set():
            try:
                utt = jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            if utt is None:
                break
            try:
                self.phase = "translating..."
                detected, text, out = self.translator.translate_utterance(
                    utt, LISTEN_TARGET)
                if out:                     # None = empty, or already English
                    self.captions.append((time.time(), detected, out))
                    if self.state.listen_speak and self.player is not None:
                        samples, _ = tts_synthesize(
                            out, self.translator.voice_for(LISTEN_TARGET), 0)
                        self.player.play_raw(samples)
                self.error = ""
            except Exception as e:
                self.error = str(e)[:80]
                self._report(f"incoming: {self.error}")
            finally:
                self.phase = ""

    def caption_lines(self, now=None, width=90):
        """Recent captions for the UI strip, oldest first."""
        now = now if now is not None else time.time()
        lines = [f"[{lang}]  {text}" for t, lang, text in self.captions
                 if now - t < LISTEN_CAPTION_S]
        return [ln if len(ln) <= width else ln[:width - 3] + "..."
                for ln in lines[-LISTEN_CAPTION_N:]]
