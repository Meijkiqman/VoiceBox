"""Voice harvester: collects clean speech clips from the real mic while you
use VoiceBox, as training data for an RVC model of your own voice.

While the toggle is on, the audio callback mirrors the raw mic into a queue
(same pattern as the recorder) and a worker thread segments it: a clip
starts when the block peak crosses the speech threshold (with a little
pre-roll), ends after a trailing-silence hangover, and is kept only if it
is long enough, loud enough and not clipped. Kept clips are peak-normalized
and written as 48 kHz mono 16-bit wavs into the dataset folder - exactly
what the RVC trainer wants to eat. Collection stops by itself at the
dataset cap; more hours of raw gaming audio stop helping long before that."""
import queue
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import (HARVEST_CAP_MIN, HARVEST_DIRNAME, HARVEST_HANG_S,
                     HARVEST_MAX_S, HARVEST_MIN_S, HARVEST_PRE_S,
                     HARVEST_THRESH_DB, RVC_DIR, SAMPLERATE)
from .log import get_logger
from .segment import Segmenter

_log = get_logger("harvester")


class Harvester:
    def __init__(self, state, out_dir=None):
        self.state = state
        if out_dir is None:
            rvc = Path(state.rvc_dir) if state.rvc_dir else RVC_DIR
            # keep the dataset next to the RVC package when there is one, so
            # the trainer finds it; else a visible folder in the checkout
            out_dir = (rvc / HARVEST_DIRNAME if rvc.is_dir()
                       else RVC_DIR.parent / "voice_dataset")
        self.dir = Path(out_dir)
        self.error = ""
        self.kept = 0                  # clips saved this session
        self.dropped = 0               # clips rejected this session
        self.seconds = self._scan()    # dataset total, kept fresh on save
        self._thread = None
        self._stop = None

    def _scan(self):
        """Existing dataset length in seconds (16-bit mono 48k wavs)."""
        total = 0.0
        try:
            for f in self.dir.glob("*.wav"):
                total += max(0, f.stat().st_size - 44) / (SAMPLERATE * 2)
        except OSError:
            pass
        return total

    @property
    def on(self):
        return self.state.harvest_q is not None

    @property
    def minutes(self):
        return self.seconds / 60.0

    @property
    def full(self):
        return self.minutes >= HARVEST_CAP_MIN

    def label(self):
        """The menu row's value text."""
        if self.full:
            return f"full ({self.minutes:.0f} min)"
        if self.on:
            return f"ON - {self.minutes:.1f} min"
        return f"off - {self.minutes:.1f} min" if self.seconds else "off"

    def _report(self, msg):
        self.state.report(msg)

    def toggle(self):
        self.stop() if self.on else self.start()
        with self.state.lock:
            self.state.harvest_on = self.on   # persisted: survives restarts

    def start(self):
        if self.on:
            return
        if self.full:
            self._report(f"harvest: dataset is at {HARVEST_CAP_MIN:.0f} min - "
                         "that's plenty; time to retrain")
            return
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.error = str(e)
            self._report(f"harvest: {e}")
            return
        q = queue.Queue(maxsize=256)   # ~2.7 s headroom; drop, never block
        self._stop = threading.Event()
        # stop event passed as an arg: the worker must keep honoring ITS
        # event even if a quick stop()/start() has already replaced self._stop
        self._thread = threading.Thread(target=self._worker,
                                        args=(q, self._stop), daemon=True)
        self._thread.start()
        self.state.harvest_q = q
        self.error = ""

    def stop(self):
        q = self.state.harvest_q
        self.state.harvest_q = None    # callback stops feeding immediately
        t, self._thread = self._thread, None
        if self._stop is not None:
            self._stop.set()
        if q is not None:
            try:
                q.put_nowait(None)     # unblock the worker's get()
            except queue.Full:
                pass                   # worker wakes on its own timeout
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)        # let a clip mid-write land intact

    def _worker(self, q, stop):
        seg = Segmenter(HARVEST_THRESH_DB, HARVEST_PRE_S, HARVEST_HANG_S,
                        HARVEST_MAX_S)
        while not stop.is_set():
            try:
                block = q.get(timeout=0.5)
            except queue.Empty:
                continue
            if block is None:          # stop(): flush what we were building
                break
            clip = seg.feed(block)
            if clip is None:
                continue
            self._finish(clip)
            if self.full:
                if self._stop is not stop:
                    break              # a newer session owns the toggle
                self.stop()
                with self.state.lock:
                    self.state.harvest_on = False
                self._report(f"harvest: dataset full at "
                             f"{HARVEST_CAP_MIN:.0f} min - time to retrain")
                break
        clip = seg.flush()
        if clip is not None:
            self._finish(clip)

    def _finish(self, clip):
        """Quality gate + normalize + save. Rejections are silent (a counter,
        not a status): most of them are just keyboard noise and breaths."""
        dur = len(clip) / SAMPLERATE
        peak = float(np.abs(clip).max()) if len(clip) else 0.0
        clipped = float(np.mean(np.abs(clip) > 0.985)) if len(clip) else 0.0
        if dur < HARVEST_MIN_S or peak < 0.02 or clipped > 0.001:
            self.dropped += 1
            return
        clip = clip * (0.95 / peak)    # peak-normalize; trainer re-levels anyway
        name = time.strftime("vb-%Y%m%d-%H%M%S") + f"-{self.kept:03d}.wav"
        try:
            sf.write(str(self.dir / name), clip, SAMPLERATE, subtype="PCM_16")
        except Exception as e:
            _log.warning("clip save failed: %s", e)
            self.error = str(e)
            self._report(f"harvest: {e}")
            return
        self.kept += 1
        self.seconds += dur
