"""Soundboard: clip loading from sounds/ and the Board controller shared by
the grid buttons, menu rows and hotkeys."""
import hashlib
import os
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import (CLIP_CACHE_DIRNAME, CLIP_PEAK, MAX_CLIPS, SAMPLERATE,
                     SOUNDS_DIR)
from .log import get_logger

_log = get_logger("soundboard")

# Two decoded-clip caches, both keyed by (path, mtime, size):
#  - in memory, so a rescan - every file drag-dropped onto the window
#    triggers one, on the UI thread - only decodes files that are actually
#    new or changed instead of the entire folder (a multi-second freeze at
#    64 clips);
#  - on disk (sounds/.decoded/*.npy: the ready-to-play float32 at our
#    rate), so a LAUNCH doesn't re-decode and re-resample the folder either
#    (0.77 s for 32 clips, linear in the folder). Entries whose source file
#    is gone or changed are pruned on every scan.
_clip_cache = {}


def _cache_name(key):
    path, mtime_ns, size = key
    tag = f"{path}|{mtime_ns}|{size}|{SAMPLERATE}|{CLIP_PEAK}"
    return (hashlib.sha1(tag.encode("utf-8", "surrogateescape")).hexdigest()[:20]
            + ".npy")


def _disk_get(folder, name):
    try:
        data = np.load(str(folder / name), allow_pickle=False)
        return data if data.dtype == np.float32 and data.ndim == 1 else None
    except (OSError, ValueError):          # missing, truncated, not ours
        return None


def _disk_put(folder, name, data):
    try:
        folder.mkdir(exist_ok=True)
        tmp = folder / (name + ".tmp.npy")
        np.save(str(tmp), data)
        os.replace(tmp, folder / name)     # never a half-written entry
    except OSError as e:
        _log.debug("clip cache write skipped: %s", e)


def _disk_prune(folder, keep):
    try:
        for p in folder.glob("*.npy"):
            if p.name not in keep:
                p.unlink()
    except OSError as e:
        _log.debug("clip cache prune skipped: %s", e)


def load_clips():
    clips, names = [], []
    if not SOUNDS_DIR.is_dir():
        return clips, names
    files = sorted(f for f in SOUNDS_DIR.iterdir()
                   if f.suffix.lower() in (".wav", ".flac", ".ogg", ".mp3"))
    cache_dir = SOUNDS_DIR / CLIP_CACHE_DIRNAME
    fresh, on_disk = {}, set()
    for n, f in enumerate(files):
        if len(clips) >= MAX_CLIPS:
            print(f"  soundboard full: ignoring {len(files) - n} more file(s) in sounds/")
            break
        try:
            st = f.stat()
            key = (str(f), st.st_mtime_ns, st.st_size)
            cname = _cache_name(key)
            data = _clip_cache.get(key)
            if data is None:
                data = _disk_get(cache_dir, cname)
            if data is None:
                data, sr = sf.read(str(f), dtype="float32", always_2d=True)
                data = data.mean(axis=1)           # downmix to mono
                if sr != SAMPLERATE:               # resample to our rate
                    data = resample_poly(data, SAMPLERATE,
                                         sr).astype(np.float32)
                if CLIP_PEAK:                      # tame internet-sourced levels
                    peak = float(np.abs(data).max())
                    if peak > 1e-4:
                        data = data * min(CLIP_PEAK / peak, 4.0)
                data = np.ascontiguousarray(data, dtype=np.float32)
                _disk_put(cache_dir, cname, data)
        except Exception as e:                     # one bad file must not kill startup
            print(f"  skipping {f.name}: {e}")
            _log.warning("skipping %s: %s", f.name, e)
            continue
        fresh[key] = data
        on_disk.add(cname)
        clips.append(data)
        names.append(f.stem)
    _clip_cache.clear()                            # deleted files free their RAM
    _clip_cache.update(fresh)
    if cache_dir.is_dir():
        _disk_prune(cache_dir, on_disk)
    return clips, names




class Board:
    """Soundboard control shared by the grid buttons, the menu rows and the
    1-9 hotkeys, so every input surface behaves identically."""

    def __init__(self, state, player=None, monitor=None):
        self.state = state
        self.player = player
        self.monitor = monitor
        self.flash = {}                    # clip index -> flash-until timestamp

    def play(self, i):
        if not (0 <= i < len(self.state.clips)):
            return
        to_mic = self.state.clips_to_mic
        if to_mic:
            self.state.events.put(i)
        # local listen - skipped when self-listen already mirrors the mic mix,
        # otherwise the sound would be heard doubled
        mirrored = self.monitor is not None and self.monitor.on and to_mic
        if self.player is not None and not mirrored:
            self.player.play(i)
            if self.player.error:
                self.state.report(f"speakers: {self.player.error}")
        self.flash[i] = time.time() + 0.25

    def play_hot(self, slot):
        """Hotkey slot 0-8 -> clip on the current page."""
        self.play(slot + 9 * self.state.clip_page)

    def page_count(self):
        return max(1, (len(self.state.clips) + 8) // 9)

    def set_page(self, d):
        """Step the hotkey page (wraps). Returns the new page."""
        n = self.page_count()
        with self.state.lock:
            self.state.clip_page = (self.state.clip_page + d) % n
        return self.state.clip_page

    def rescan(self):
        """Re-read sounds/ without a restart. Playing clips keep their old
        samples (the voice lists hold references); the swap is a whole-list
        assignment so the audio callback sees either the old or new list."""
        clips, names = load_clips()
        with self.state.lock:
            self.state.clips = clips
            self.state.clip_names = names
            self.state.clip_page = min(self.state.clip_page,
                                       self.page_count() - 1)
            self.state.clips_version += 1
        self.flash.clear()
        self.state.report(f"soundboard: {len(clips)} sound(s)")

    def toggle_mic(self):
        with self.state.lock:
            self.state.clips_to_mic = not self.state.clips_to_mic

    def toggle_pause(self):
        with self.state.lock:
            self.state.clips_paused = not self.state.clips_paused

    def stop(self):
        self.state.events.put("stop")
        if self.player is not None:
            self.player.stop()
        with self.state.lock:              # stop also un-pauses: clean slate
            self.state.clips_paused = False


