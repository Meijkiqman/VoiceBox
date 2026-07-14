"""Soundboard: clip loading from sounds/ and the Board controller shared by
the grid buttons, menu rows and hotkeys."""
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import CLIP_PEAK, MAX_CLIPS, SAMPLERATE, SOUNDS_DIR

def load_clips():
    clips, names = [], []
    if not SOUNDS_DIR.is_dir():
        return clips, names
    files = sorted(f for f in SOUNDS_DIR.iterdir()
                   if f.suffix.lower() in (".wav", ".flac", ".ogg", ".mp3"))
    for n, f in enumerate(files):
        if len(clips) >= MAX_CLIPS:
            print(f"  soundboard full: ignoring {len(files) - n} more file(s) in sounds/")
            break
        try:
            data, sr = sf.read(str(f), dtype="float32", always_2d=True)
        except Exception as e:                     # one bad file must not kill startup
            print(f"  skipping {f.name}: {e}")
            continue
        data = data.mean(axis=1)                   # downmix to mono
        if sr != SAMPLERATE:                       # resample to our rate
            data = resample_poly(data, SAMPLERATE, sr).astype(np.float32)
        if CLIP_PEAK:                              # tame internet-sourced levels
            peak = float(np.abs(data).max())
            if peak > 1e-4:
                data = data * min(CLIP_PEAK / peak, 4.0)
        clips.append(data)
        names.append(f.stem)
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
                self.state.status_msg = f"speakers: {self.player.error}"
                self.state.status_at = time.time()
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
        self.state.status_msg = f"soundboard: {len(clips)} sound(s)"
        self.state.status_at = time.time()

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


