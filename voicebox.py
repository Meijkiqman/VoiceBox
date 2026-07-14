"""
voicebox.py - Real-time voice changer + soundboard for Discord (or anything).

HOW IT WORKS
------------
    real mic --> [this script: pitch shift + effects + soundboard] --> VB-CABLE Input
                                                                             |
                                                          Discord input = "CABLE Output"

SETUP (Windows)
---------------
1. Install VB-CABLE:  https://vb-audio.com/Cable/  (run installer as admin, reboot).
2. pip install sounddevice soundfile numpy scipy pygame
3. Put some .wav files in the ./sounds folder.
4. Run:  python voicebox.py --list      (find your device names)
5. Edit the CONFIG block below if the auto-match doesn't find your devices.
6. Run:  python voicebox.py
7. In Discord: Settings -> Voice & Video -> Input Device = "CABLE Output".

macOS: use BlackHole instead of VB-CABLE.  Linux: create a null sink with
`pactl load-module module-null-sink sink_name=voicebox` and point Discord at its monitor.

CONTROLS
--------
A pygame menu window handles all input (keyboard + controller + mouse).
Inputs only fire while the VoiceBox window has focus, so typing in Discord is
safe. Bindings live in controls.json next to this file - edit to remap,
delete to restore defaults.

Mouse: hover highlights a row, click activates it, click the < > arrows to
adjust a value, scroll wheel moves the selection (or scrolls the grid when
the pointer is over it).

SOUNDBOARD
----------
Every audio file in ./sounds (wav/flac/ogg/mp3, first 64, alphabetical) gets
a button in the grid on the right. Clicking a button always plays the sound
locally so you hear it yourself; while the "To mic" toggle is on it is also
mixed into the mic channel. Pause freezes all playing sounds (both paths),
Stop clears them. Keys 1-9 trigger the first nine sounds. The Hear toggle in
the same strip mirrors the processed mix to your speakers (self-listen);
while the AI voice is live the RVC worker mirrors its converted voice to the
speakers the same way, so you hear the AI voice too.

TEXT TO SPEECH
--------------
The panel below the soundboard speaks typed phrases into the mic channel.
Type in the box, press Enter (or ADD) to save - phrases persist in
tts_phrases.json and are synthesized once into tts_cache/ (Windows SAPI via
PowerShell; espeak / `say` elsewhere). Click a phrase to speak it, the x on
its row deletes it. With "TTS voice FX" on (menu row or the FX chip) the
speech runs through the same pitch/effect chain as your voice - and through
the AI voice while the RVC worker is live; off = clean TTS.

EFFECTS & PRESETS
-----------------
Pitch, robot/vocoder mix, helmet doubler, grit, reverb, echo, radio band-pass
and bass boost are individual menu rows. Numeric rows carry a draggable
slider in the middle; clicking the number itself opens a small box to type
an exact value (Enter commits, Esc cancels), and keyboard < > still steps.
The Preset row applies curated combinations (Space Marine, Ghost, ...) which
can be tweaked freely afterwards - the row shows "Custom" once any value
diverges from the applied preset. Pressing the Preset or AI character row
opens an alphabetical dropdown for direct picking. The window itself is
resizable (drag edges, Aero snap); the soundboard pane absorbs the extra
space.

Defaults:  arrows/WASD or d-pad/left stick = navigate,  Enter/Space or A =
select,  left/right adjusts values,  1-9 = play clip,  0/Backspace or Y =
stop clips,  Esc or B = quit.
"""

import argparse
import hashlib
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import butter, lfilter, resample_poly

# ----------------------------------------------------------------------------- CONFIG
BASE_DIR      = Path(__file__).resolve().parent
SOUNDS_DIR    = BASE_DIR / "sounds"       # anchored: works from any cwd
CONTROLS_PATH = BASE_DIR / "controls.json"

TTS_PHRASES_PATH = BASE_DIR / "tts_phrases.json"  # saved TTS phrases
TTS_CACHE_DIR    = BASE_DIR / "tts_cache"         # rendered wavs, keyed by text hash
TTS_MAX_CHARS    = 200                            # per-phrase length cap

SAMPLERATE = 48000        # VB-CABLE runs at 48k by default
BLOCKSIZE  = 512          # smaller = lower latency, larger = safer. 256-1024 typical
CHANNELS   = 1            # mono processing path

# Device selection. Substrings are matched against device names (case-insensitive).
# Use --list to see names. Set to an int to force a specific device index instead.
INPUT_DEVICE_MATCH   = None            # None = system default mic, or e.g. "Microphone"
OUTPUT_DEVICE_MATCH  = "CABLE Input"   # the virtual cable's INPUT side

WINDOW_SIZE = (960, 660)   # initial + minimum size; the window is resizable
MAX_CLIPS   = 64           # how many files from ./sounds get indexed

# AI voice (RVC) integration. RVC_DIR holds a trimmed RVC-beta package
# (must contain runtime\python.exe, weights\*.pth, hubert_base.pt, rmvpe.pt);
# ours lives in the rvc\ folder next to this file, so VoiceBox is
# self-contained. The AI rows only appear in the menu when this folder and at
# least one voice model exist, so machines without RVC are unaffected.
RVC_DIR = BASE_DIR / "rvc"

# Voice presets, cycled with the Preset menu row. "drive" is the grit/growl
# soft-clip amount; "robot" is the robot/vocoder mix; "reverb"/"echo"/"doubler"
# are wet mixes; "bass" is the low-shelf boost (all 0..1). "radio" is the
# walkie-talkie band-pass. Missing keys default to off.
PRESETS = [
    ("Normal",        {"semitones": 0,  "robot": 0.0, "drive": 0.0}),
    ("Chipmunk",      {"semitones": 7,  "robot": 0.0, "drive": 0.0}),
    ("Monster",       {"semitones": -6, "robot": 0.0, "drive": 0.35, "reverb": 0.3}),
    ("Robot",         {"semitones": 0,  "robot": 1.0, "drive": 0.0}),
    # (The Voicemod-recipe variant - doubler 100 / robot 49 / reverb 26 /
    # pitch -4 / bass 100 - can still be dialed in manually via the rows.)
    ("Space Marine",  {"semitones": -5, "robot": 0.0, "drive": 0.85, "reverb": 0.4}),
    ("Ork",           {"semitones": -3, "robot": 0.0, "drive": 1.0}),
    ("Ghost",         {"semitones": 2,  "robot": 0.0, "drive": 0.0,
                       "reverb": 0.85, "echo": 0.4}),
    ("Walkie-Talkie", {"semitones": 0,  "robot": 0.0, "drive": 0.25, "radio": True}),
]

DEFAULT_CONTROLS = {
    "keyboard": {
        "up":         ["up", "w"],
        "down":       ["down", "s"],
        "left":       ["left", "a"],
        "right":      ["right", "d"],
        "select":     ["return", "space"],
        "back":       ["escape"],
        "stop_clips": ["0", "backspace"],
        "clips":      ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
    },
    "gamepad": {
        "select":         [0],
        "back":           [1],
        "stop_clips":     [3],
        "axis_threshold": 0.5,
        "nav_cooldown":   0.22,
    },
}

# ----------------------------------------------------------------------------- DSP
class StreamingPitchShifter:
    """Phase-vocoder pitch shifter. Equal analysis/synthesis hop so output length
    == input length (trivial streaming). Latency ~= n_fft samples."""
    def __init__(self, sr, semitones=0.0, n_fft=1024, hop=256):
        self.sr = sr; self.n_fft = n_fft; self.hop = hop
        self.win = np.hanning(n_fft).astype(np.float32)
        self.nb = n_fft // 2 + 1
        self.expected = 2 * np.pi * hop * np.arange(self.nb) / n_fft
        self.norm = np.sum(self.win ** 2) / hop
        self.ratio = 1.0
        self.reset()
        self.set_semitones(semitones)

    def reset(self):
        """Clear all streaming state (buffers + phase accumulators)."""
        self.in_buf = np.zeros(0, dtype=np.float32)
        self.out_buf = np.zeros(0, dtype=np.float32)
        self.prev_phase = np.zeros(self.nb, dtype=np.float32)
        self.sum_phase = np.zeros(self.nb, dtype=np.float32)

    def set_semitones(self, s):
        new_ratio = 2.0 ** (float(s) / 12.0)
        # Passthrough (ratio==1) bypasses the buffers entirely, so crossing that
        # boundary in either direction must reset state - otherwise stale audio
        # gets prepended and latency jumps produce a garbled blip.
        if (abs(self.ratio - 1.0) < 1e-6) != (abs(new_ratio - 1.0) < 1e-6):
            self.reset()
        self.ratio = new_ratio

    def process(self, x):
        x = np.asarray(x, dtype=np.float32).ravel()
        # ratio 1.0 -> passthrough (also avoids needless CPU)
        if abs(self.ratio - 1.0) < 1e-6:
            return x
        self.in_buf = np.concatenate([self.in_buf, x])
        produced = []
        b = np.arange(self.nb)
        while len(self.in_buf) >= self.n_fft:
            frame = self.in_buf[:self.n_fft] * self.win
            self.in_buf = self.in_buf[self.hop:]
            spec = np.fft.rfft(frame)
            mag = np.abs(spec); phase = np.angle(spec)
            dphi = phase - self.prev_phase; self.prev_phase = phase
            dphi = dphi - self.expected
            dphi = np.mod(dphi + np.pi, 2 * np.pi) - np.pi
            true_freq = self.expected + dphi
            src = b / self.ratio
            new_mag = np.interp(src, b, mag, left=0.0, right=0.0)
            new_freq = np.interp(src, b, true_freq, left=0.0, right=0.0) * self.ratio
            self.sum_phase = self.sum_phase + new_freq
            grain = np.fft.irfft(new_mag * np.exp(1j * self.sum_phase), self.n_fft)
            grain = grain.astype(np.float32) * self.win / self.norm
            if len(self.out_buf) < self.n_fft:
                self.out_buf = np.concatenate(
                    [self.out_buf, np.zeros(self.n_fft - len(self.out_buf), np.float32)])
            self.out_buf[:self.n_fft] += grain
            produced.append(self.out_buf[:self.hop].copy())
            self.out_buf = self.out_buf[self.hop:]
        return np.concatenate(produced) if produced else np.zeros(0, dtype=np.float32)


# The delay-line effects below are block-vectorized: within one call, reads
# happen before writes, which is exact for blocks up to the delay length.
# _DelayLine.process() splits longer blocks into delay-sized chunks, so any
# BLOCKSIZE works regardless of the individual delay sizes.
class _DelayLine:
    def process(self, x, *args):
        n = len(self.buf)
        if len(x) <= n:
            return self._block(x, *args)
        return np.concatenate([self._block(x[i:i + n], *args)
                               for i in range(0, len(x), n)])


class CombFilter(_DelayLine):
    """Feedback comb with a cheap 2-tap lowpass in the loop (damping)."""
    def __init__(self, delay, feedback):
        self.buf = np.zeros(delay, dtype=np.float32)
        self.pos = 0
        self.fb = feedback
        self.prev = 0.0
    def _block(self, x):
        idx = (self.pos + np.arange(len(x))) % len(self.buf)
        out = self.buf[idx]
        damped = np.empty_like(out)
        damped[0] = 0.5 * (out[0] + self.prev)
        damped[1:] = 0.5 * (out[1:] + out[:-1])
        self.prev = float(out[-1])
        self.buf[idx] = x + self.fb * damped
        self.pos = (self.pos + len(x)) % len(self.buf)
        return out


class AllpassFilter(_DelayLine):
    def __init__(self, delay, g=0.5):
        self.buf = np.zeros(delay, dtype=np.float32)
        self.pos = 0
        self.g = g
    def _block(self, x):
        idx = (self.pos + np.arange(len(x))) % len(self.buf)
        vd = self.buf[idx]
        v = x + self.g * vd
        self.buf[idx] = v
        self.pos = (self.pos + len(x)) % len(self.buf)
        return vd - self.g * v


class Reverb:
    """Schroeder reverb: 4 parallel damped combs into an allpass diffuser."""
    def __init__(self):
        self.combs = [CombFilter(1427, 0.805), CombFilter(1783, 0.827),
                      CombFilter(1987, 0.783), CombFilter(2099, 0.764)]
        self.ap = AllpassFilter(1153, 0.5)
    def process(self, x, wet):
        w = self.combs[0].process(x)
        for c in self.combs[1:]:
            w = w + c.process(x)
        w = self.ap.process(w * 0.25)
        return x * (1.0 - 0.5 * wet) + w * wet


class Echo(_DelayLine):
    """Single feedback delay (~320 ms) - classic canyon echo."""
    def __init__(self, delay=int(0.32 * SAMPLERATE), feedback=0.35):
        self.buf = np.zeros(delay, dtype=np.float32)
        self.pos = 0
        self.fb = feedback
    def _block(self, x, wet):
        idx = (self.pos + np.arange(len(x))) % len(self.buf)
        d = self.buf[idx]
        self.buf[idx] = x + self.fb * d
        self.pos = (self.pos + len(x)) % len(self.buf)
        return x + wet * d


class Radio:
    """Walkie-talkie band-pass (300-3200 Hz), stateful across blocks."""
    def __init__(self):
        nyq = SAMPLERATE / 2
        self.b, self.a = butter(2, [300 / nyq, 3200 / nyq], btype="band")
        self.zi = np.zeros(max(len(self.a), len(self.b)) - 1)
    def process(self, x):
        y, self.zi = lfilter(self.b, self.a, x, zi=self.zi)
        return y.astype(np.float32) * 1.5      # make up the band-loss level


class Doubler(_DelayLine):
    """~12 ms single-repeat full-mix delay: the in-helmet comb doubling from
    the Voicemod Space Marine recipe (Delay, mix 100 / fade 0 / time 1)."""
    def __init__(self, delay=576):
        self.buf = np.zeros(delay, dtype=np.float32)
        self.pos = 0
    def _block(self, x, wet):
        idx = (self.pos + np.arange(len(x))) % len(self.buf)
        d = self.buf[idx]
        self.buf[idx] = x                      # no feedback: one repeat only
        self.pos = (self.pos + len(x)) % len(self.buf)
        return (x + wet * d) * (1.0 / (1.0 + 0.5 * wet))   # keep level sane


class BassBoost:
    """Low-shelf boost (up to +6 dB below ~250 Hz), the recipe's EQ low gain."""
    def __init__(self):
        self.b, self.a = butter(2, 250 / (SAMPLERATE / 2))
        self.zi = np.zeros(max(len(self.a), len(self.b)) - 1)
    def process(self, x, amount):
        low, self.zi = lfilter(self.b, self.a, x, zi=self.zi)
        return x + amount * low.astype(np.float32)


# ----------------------------------------------------------------------------- STATE
class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.semitones = 0.0          # UI-side display value; the shifter follows via events
        self.robot = 0.0              # robot/vocoder mix, 0..1 (0 = off)
        self.robot_phase = 0.0
        self.voice_gain = 1.0
        self.clip_gain = 0.9
        self.drive = 0.0              # grit/growl soft-clip amount, 0..1
        self.reverb = 0.0             # reverb wet mix, 0..1
        self.echo = 0.0               # echo wet mix, 0..1
        self.doubler = 0.0            # helmet doubler wet mix, 0..1
        self.bass = 0.0               # low-shelf boost, 0..1
        self.radio = False            # walkie-talkie band-pass
        self.preset_idx = 0
        self.clips_to_mic = True      # soundboard also feeds the mic channel
        self.clips_paused = False     # freezes all playing sounds (both paths)
        self.tts_fx = True            # TTS through the voice chain / AI (off = clean)
        self.tts_gain = 1.0           # TTS level on the mic channel
        self.ai_mute = False          # AI worker owns the voice; mute ours
        self.shifter = StreamingPitchShifter(SAMPLERATE, 0.0)  # audio thread only
        self.reverb_fx = Reverb()     # effect state: audio thread only
        self.echo_fx = Echo()
        self.radio_fx = Radio()
        self.doubler_fx = Doubler()
        self.bass_fx = BassBoost()
        self.clips, self.clip_names = load_clips()
        self.voices = []              # list of [samples, cursor]; audio thread only
        self.tts_voices = []          # list of [samples, cursor, fx]; audio thread only
        self.events = queue.Queue()   # UI thread -> audio thread
        self.status_msg = ""          # audio thread -> UI (underruns etc.)
        self.status_at = 0.0          # when status_msg was last set
        self.status_count = 0         # how many times a status fired (underrun tally)
        self.in_level = 0.0           # audio thread -> UI mic meter (block peak)
        self.monitor_q = None         # set to a Queue while self-listen is on

    def set_pitch(self, semis):
        # The shifter is owned by the audio thread; hand the change over via the
        # event queue so the callback never blocks on the UI holding the lock.
        with self.lock:
            self.semitones = max(-12, min(12, semis))
            self.events.put(("pitch", self.semitones))

    def nudge(self, attr, delta, lo=0.0, hi=1.5):
        with self.lock:
            setattr(self, attr, max(lo, min(hi, getattr(self, attr) + delta)))

    def set_val(self, attr, v, lo=0.0, hi=1.5):
        """Absolute set with clamp (sliders / typed values)."""
        with self.lock:
            setattr(self, attr, max(lo, min(hi, float(v))))

    def apply_preset(self, idx):
        _, p = PRESETS[idx % len(PRESETS)]
        with self.lock:
            self.preset_idx = idx % len(PRESETS)
            self.robot = float(p["robot"])
            self.drive = p["drive"]
            self.reverb = p.get("reverb", 0.0)
            self.echo = p.get("echo", 0.0)
            self.doubler = p.get("doubler", 0.0)
            self.bass = p.get("bass", 0.0)
            self.radio = p.get("radio", False)
        self.set_pitch(p["semitones"])

    def preset_label(self):
        """Preset name while values still match it, else "Custom"."""
        name, p = PRESETS[self.preset_idx]
        matches = (self.semitones == p["semitones"]
                   and self.robot == float(p["robot"])
                   and self.drive == p["drive"]
                   and self.reverb == p.get("reverb", 0.0)
                   and self.echo == p.get("echo", 0.0)
                   and self.doubler == p.get("doubler", 0.0)
                   and self.bass == p.get("bass", 0.0)
                   and self.radio == p.get("radio", False))
        return name if matches else "Custom"


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
        clips.append(data)
        names.append(f.stem)
    return clips, names


# ----------------------------------------------------------------------------- AUDIO
def make_callback(state):
    carry = np.zeros(0, dtype=np.float32)   # over-produced samples roll forward

    def callback(indata, outdata, frames, time_info, status):
        nonlocal carry
        if status:
            state.status_msg = str(status)   # no print in the audio callback
            state.status_at = time.time()
            state.status_count += 1
        state.in_level = float(np.abs(indata[:, 0]).max())   # feeds the UI mic meter

        # Hold the lock only to copy parameters - never through the DSP, so a
        # UI-thread nudge can't stall the real-time callback.
        with state.lock:
            voice_gain, clip_gain, robot, drive, clips_paused = (
                state.voice_gain, state.clip_gain, state.robot, state.drive,
                state.clips_paused)
            reverb, echo, radio = state.reverb, state.echo, state.radio
            doubler, bass = state.doubler, state.bass
            ai_mute, tts_gain = state.ai_mute, state.tts_gain

        # apply queued UI events (audio thread owns the shifter and voice list)
        while not state.events.empty():
            ev = state.events.get_nowait()
            if ev == "stop":
                state.voices.clear()
                state.tts_voices.clear()
            elif isinstance(ev, tuple) and ev[0] == "pitch":
                state.shifter.set_semitones(ev[1])
            elif isinstance(ev, tuple) and ev[0] == "tts":
                state.tts_voices.append([ev[1], 0, bool(ev[2])])
            elif isinstance(ev, int) and 0 <= ev < len(state.clips):
                state.voices.append([state.clips[ev], 0])

        # TTS phrases: fx-tagged ones ride the mic signal itself, so the whole
        # chain (pitch, robot, reverb, ...) treats them like speech; the rest
        # (and everything while the AI owns the voice path) mixes in clean
        # after the chain, like a soundboard clip. Pause freezes the cursors.
        tts_pre = tts_post = None
        if state.tts_voices and not clips_paused:
            tts_pre = np.zeros(frames, dtype=np.float32)
            tts_post = np.zeros(frames, dtype=np.float32)
            still = []
            for samples, cur, fx in state.tts_voices:
                buf = tts_pre if (fx and not ai_mute) else tts_post
                chunk = samples[cur:cur + frames]
                buf[:len(chunk)] += chunk
                if cur + frames < len(samples):
                    still.append([samples, cur + frames, fx])
            state.tts_voices = still

        if ai_mute:
            # the RVC worker feeds the converted voice into the cable itself;
            # our own voice path stays silent so it isn't heard doubled
            y = np.zeros(frames, dtype=np.float32)
            carry = np.zeros(0, dtype=np.float32)
        else:
            x = indata[:, 0].astype(np.float32) * voice_gain
            if tts_pre is not None:
                x = x + tts_pre * tts_gain
            y = state.shifter.process(x)

            y = np.concatenate([carry, y]) if len(carry) else y
            if len(y) < frames:
                y = np.concatenate([y, np.zeros(frames - len(y), np.float32)])
                carry = np.zeros(0, dtype=np.float32)
            else:
                carry = y[frames:].copy()    # keep the remainder, never drop audio
                y = y[:frames].copy()

            # helmet doubler: short full-mix single repeat (recipe's Delay stage)
            if doubler > 1e-3:
                y = state.doubler_fx.process(y, doubler)

            # robot / vocoder: ring-mod blended by mix amount (1.0 = full robot)
            if robot > 1e-3:
                n = np.arange(frames)
                carrier = np.sin(2 * np.pi * 60.0 * (n / SAMPLERATE)
                                 + state.robot_phase)
                state.robot_phase = (state.robot_phase
                                     + 2 * np.pi * 60.0 * frames / SAMPLERATE) % (2 * np.pi)
                y = y * (1.0 - robot + robot * carrier.astype(np.float32))

            # grit / growl: soft-clip saturation (helmet-vox crunch)
            if drive > 1e-3:
                g = 1.0 + 9.0 * drive
                y = np.tanh(y * g) / float(np.tanh(g))

            # voice-only effects chain: radio band-pass -> echo -> reverb
            if radio:
                y = state.radio_fx.process(y)
            if echo > 1e-3:
                y = state.echo_fx.process(y, echo)
            if reverb > 1e-3:
                y = state.reverb_fx.process(y, reverb)
            if bass > 1e-3:                    # recipe's EQ low-gain stage
                y = state.bass_fx.process(y, bass)

        # mix active soundboard voices (paused voices keep their cursor)
        if state.voices and not clips_paused:
            still = []
            for samples, cur in state.voices:
                chunk = samples[cur:cur + frames]
                y[:len(chunk)] += chunk * clip_gain
                if cur + frames < len(samples):
                    still.append([samples, cur + frames])
            state.voices = still

        if tts_post is not None:               # clean TTS joins after the chain
            y += tts_post * tts_gain

        np.clip(y, -1.0, 1.0, out=y)          # prevent hard clipping distortion
        q = state.monitor_q                    # mirror to self-listen, if enabled
        if q is not None:
            try:
                q.put_nowait(y.copy())
            except queue.Full:
                pass                           # listener lagging: drop, never block
        outdata[:, 0] = y
    return callback


class Monitor:
    """Self-listen (the HEAR strip toggle). While the main stream is running it
    mirrors the processed mix to the default speakers. If the main stream never
    opened (e.g. virtual cable not installed yet), toggling on runs the whole
    chain as a mic -> speakers stream instead, so the voice is still testable."""

    def __init__(self, state, has_main_stream):
        self.state = state
        self.has_main = has_main_stream
        self.stream = None
        self.error = ""

    @property
    def on(self):
        return self.stream is not None

    def toggle(self):
        if self.stream is not None:            # turn off
            self.state.monitor_q = None
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None
            if not self.has_main:              # fallback stream fed the meter
                self.state.in_level = 0.0
            return
        try:
            if self.has_main:
                q = queue.Queue(maxsize=8)     # ~85 ms of audio; producer drops extras

                def cb(outdata, frames, time_info, status):
                    if status:
                        self.state.status_msg = f"test: {status}"
                        self.state.status_at = time.time()
                        self.state.status_count += 1
                    try:
                        y = q.get_nowait()
                    except queue.Empty:
                        y = np.zeros(frames, np.float32)
                    if len(y) < frames:
                        y = np.concatenate([y, np.zeros(frames - len(y), np.float32)])
                    outdata[:, 0] = y[:frames]

                self.stream = sd.OutputStream(
                    samplerate=SAMPLERATE, blocksize=BLOCKSIZE, dtype="float32",
                    channels=CHANNELS, callback=cb)
                self.stream.start()
                self.state.monitor_q = q
            else:
                # Nothing has been draining state.events while the main stream
                # was down, so a session of soundboard clicks is queued up and
                # would all fire at once. Clear the backlog, keep the pitch.
                while not self.state.events.empty():
                    try:
                        self.state.events.get_nowait()
                    except queue.Empty:
                        break
                self.state.events.put(("pitch", self.state.semitones))
                self.stream = sd.Stream(
                    samplerate=SAMPLERATE, blocksize=BLOCKSIZE, dtype="float32",
                    channels=CHANNELS, latency="high",
                    callback=make_callback(self.state))
                self.stream.start()
            self.error = ""
        except Exception as e:
            self.stream = None
            self.state.monitor_q = None
            self.error = str(e)

    def close(self):
        if self.stream is not None:
            self.toggle()


class LocalPlayer:
    """Plays soundboard clips on the default speakers, so the user always
    hears what he fires. Runs its own OutputStream (opened lazily on first
    play); the mic-channel half stays in the main stream's callback."""

    def __init__(self, state):
        self.state = state
        self.stream = None
        self.voices = []                   # [samples, cursor]; callback-owned
        self.events = queue.Queue()
        self.error = ""

    def play(self, i):
        if self._ensure():
            self.events.put(i)

    def play_raw(self, samples):
        """Queue raw samples (the TTS path) instead of a clip index."""
        if self._ensure():
            self.events.put(("raw", samples))

    def stop(self):
        self.events.put("stop")

    def _ensure(self):
        if self.stream is not None:
            return True
        try:
            self.stream = sd.OutputStream(
                samplerate=SAMPLERATE, blocksize=BLOCKSIZE, dtype="float32",
                channels=CHANNELS, callback=self._callback)
            self.stream.start()
            self.error = ""
            return True
        except Exception as e:
            self.stream = None
            self.error = str(e)
            return False

    def _callback(self, outdata, frames, time_info, status):
        state = self.state
        if status:
            state.status_msg = f"speakers: {status}"
            state.status_at = time.time()
            state.status_count += 1
        with state.lock:
            gain, paused = state.clip_gain, state.clips_paused
        while not self.events.empty():
            ev = self.events.get_nowait()
            if ev == "stop":
                self.voices.clear()
            elif isinstance(ev, tuple) and ev[0] == "raw":
                self.voices.append([ev[1], 0])
            elif isinstance(ev, int) and 0 <= ev < len(state.clips):
                self.voices.append([state.clips[ev], 0])
        y = np.zeros(frames, dtype=np.float32)
        if self.voices and not paused:
            still = []
            for samples, cur in self.voices:
                chunk = samples[cur:cur + frames]
                y[:len(chunk)] += chunk * gain
                if cur + frames < len(samples):
                    still.append([samples, cur + frames])
            self.voices = still
        np.clip(y, -1.0, 1.0, out=y)
        outdata[:, 0] = y

    def close(self):
        if self.stream is not None:
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None


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


class AiVoice:
    """AI voice changer (RVC models like Arthur Morgan) run as a background
    worker (rvc_worker.py) on RVC's own bundled Python runtime. While the
    worker is live, VoiceBox mutes its own voice path (state.ai_mute) so the
    cable carries only the converted voice - the soundboard keeps mixing."""

    def __init__(self, state, rvc_dir=None, monitor=None):
        self.state = state
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
        cmd = [str(self.rvc_dir / "runtime" / "python.exe"),
               str(BASE_DIR / "rvc_worker.py"),
               "--pth", str(pth), "--output-device", OUTPUT_DEVICE_MATCH]
        index = self._index_for(pth)
        if index:
            cmd += ["--index", index]
        if isinstance(INPUT_DEVICE_MATCH, str) and INPUT_DEVICE_MATCH:
            cmd += ["--input-device", INPUT_DEVICE_MATCH]
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


# ----------------------------------------------------------------------------- TTS
def synth_tts_wav(text, wav_path):
    """Render text to a wav file with the OS speech engine (blocking).
    Windows: SAPI via PowerShell. Fallbacks: macOS `say`, else espeak.
    The text travels over stdin so no shell-quoting issue can arise."""
    if sys.platform == "win32":
        path_lit = str(wav_path).replace("'", "''")
        cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command",
               "Add-Type -AssemblyName System.Speech; "
               "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
               f"$s.SetOutputToWaveFile('{path_lit}'); "
               "$s.Speak([Console]::In.ReadToEnd()); $s.Dispose()"]
    elif sys.platform == "darwin":
        cmd = ["say", "-o", str(wav_path), "--data-format=LEI16@22050", "-f", "-"]
    else:
        cmd = ["espeak", "-w", str(wav_path), "--stdin"]
    r = subprocess.run(cmd, input=text, text=True, capture_output=True,
                       timeout=60,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0:
        detail = (r.stderr or r.stdout or "").strip().splitlines()
        raise RuntimeError(detail[-1][:80] if detail else "speech engine failed")


def tts_synthesize(text):
    """text -> (mono float32 samples at SAMPLERATE, cached wav path).
    Synthesized once, then served from tts_cache/ across restarts."""
    TTS_CACHE_DIR.mkdir(exist_ok=True)
    wav = TTS_CACHE_DIR / (hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
                           + ".wav")
    if not wav.is_file():
        try:
            synth_tts_wav(text, wav)
        except BaseException:                  # incl. timeout: no half-written wavs
            wav.unlink(missing_ok=True)
            raise
    data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != SAMPLERATE:
        data = resample_poly(data, SAMPLERATE, sr).astype(np.float32)
    return data, wav


class TTSBank:
    """Saved text-to-speech phrases: persistence (tts_phrases.json), background
    synthesis into tts_cache/, and playback routing. With state.tts_fx on the
    rendered speech joins the mic signal, so the whole effect chain shapes it -
    and while the AI worker is live it is fed through the worker instead,
    coming out in the AI voice. With it off the phrase mixes in clean, like a
    soundboard clip. Either way it also plays on the speakers, so the user
    hears what was said."""

    def __init__(self, state, player=None, monitor=None, ai=None,
                 path=TTS_PHRASES_PATH):
        self.state = state
        self.player = player
        self.monitor = monitor
        self.ai = ai
        self.path = Path(path)
        self.phrases = self._load()
        self.samples = {}             # text -> mono 48k float32
        self.wav_path = {}            # text -> cached wav (fed to the AI worker)
        self.status = {}              # text -> "..." | "ready" | "error"
        self.flash = {}               # row index -> flash-until timestamp
        self.pending = None           # phrase to auto-play once synthesis lands

    def _load(self):
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return [str(p)[:TTS_MAX_CHARS] for p in data
                    if isinstance(p, str) and p.strip()]
        except (OSError, json.JSONDecodeError, TypeError):
            return []

    def _save(self):
        try:
            self.path.write_text(
                json.dumps(self.phrases, indent=2, ensure_ascii=False),
                encoding="utf-8")
        except OSError as e:
            self._report(f"TTS: can't save phrases: {e}")

    def _report(self, msg):
        self.state.status_msg = msg
        self.state.status_at = time.time()

    def warm(self):
        """Kick background synthesis for every phrase (startup pre-warm)."""
        for text in self.phrases:
            self.ensure(text)

    def ensure(self, text):
        if self.status.get(text) in ("...", "ready"):
            return
        self.status[text] = "..."
        threading.Thread(target=self._synth_job, args=(text,), daemon=True).start()

    def _synth_job(self, text):
        try:
            samples, wav = tts_synthesize(text)
        except Exception as e:
            self.status[text] = "error"
            if self.pending == text:
                self.pending = None
            self._report(f"TTS: {str(e)[:70]}")
            return
        self.samples[text] = samples
        self.wav_path[text] = wav
        self.status[text] = "ready"
        if self.pending == text:
            self.pending = None
            self._route(text)

    def add(self, text):
        """Save a phrase (whitespace collapsed). True = accepted."""
        text = " ".join(str(text).split())[:TTS_MAX_CHARS]
        if not text:
            return False
        if text in self.phrases:
            self._report("TTS: phrase already saved")
            return False
        self.phrases.append(text)
        self._save()
        self.ensure(text)
        return True

    def delete(self, i):
        if not (0 <= i < len(self.phrases)):
            return
        text = self.phrases.pop(i)
        self.flash.clear()             # row indices shifted
        if self.pending == text:
            self.pending = None
        self._save()
        # cache entries stay: re-adding the phrase later is instant

    def play(self, i):
        if not (0 <= i < len(self.phrases)):
            return
        text = self.phrases[i]
        self.flash[i] = time.time() + 0.25
        if self.status.get(text) != "ready":
            self.pending = text        # auto-plays when synthesis lands
            self.ensure(text)          # also retries after an earlier error
            return
        self._route(text)

    def _route(self, text):
        samples = self.samples[text]
        fx = self.state.tts_fx
        through_ai = False
        if fx and self.ai is not None and self.ai.proc is not None:
            through_ai = self.ai.inject(self.wav_path[text])
        if not through_ai:
            self.state.events.put(("tts", samples, fx))
        # local listen, like the soundboard - skipped when self-listen already
        # mirrors the mic mix (it would be heard doubled). The AI path is never
        # mirrored (the worker owns the cable), so it always plays locally.
        mirrored = (self.monitor is not None and self.monitor.on
                    and not through_ai)
        if self.player is not None and not mirrored:
            self.player.play_raw(samples)


# ----------------------------------------------------------------------------- INPUT
def load_controls():
    """controls.json merged over defaults; broken/missing file -> defaults."""
    cfg = json.loads(json.dumps(DEFAULT_CONTROLS))     # deep copy
    try:
        user = json.loads(CONTROLS_PATH.read_text(encoding="utf-8"))
        for section in ("keyboard", "gamepad"):
            if isinstance(user.get(section), dict):
                cfg[section].update(user[section])
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return cfg


def build_keymap(cfg, pygame):
    """action-name -> set of pygame keycodes, plus keycode -> clip index."""
    keymap, clipmap = {}, {}
    for action, names in cfg["keyboard"].items():
        if not isinstance(names, list):
            continue
        for i, name in enumerate(names):
            try:
                code = pygame.key.key_code(str(name))
            except (ValueError, NotImplementedError):
                continue
            if action == "clips":
                clipmap[code] = i
            else:
                keymap.setdefault(action, set()).add(code)
    return keymap, clipmap


# ----------------------------------------------------------------------------- UI
class MenuItem:
    def __init__(self, label, value_fn=None, select=None, adjust=None, flash=True,
                 slider=None):
        self.label = label
        self.value_fn = value_fn      # () -> str shown on the right
        self.select = select          # on_select handler
        self.adjust = adjust          # on_left/on_right handler, adjust(delta)
        self.flash = flash            # flash the row on select (off for Quit)
        self.slider = slider          # numeric rows: (get, set, lo, hi, unit)
                                      # unit "pct" (0..hi shown as %) or "st"


class Menu:
    """Single screen. Keyboard handlers call the same on_* methods the
    controller uses, so behavior is identical across input devices."""

    def __init__(self, state, stop_flag, monitor=None, board=None, ai=None):
        self.state = state
        self.stop_flag = stop_flag
        self.monitor = monitor
        self.board = board if board is not None else Board(state)
        self.ai = ai
        self.sel = 0
        self.flash = {}               # item index -> flash-until timestamp
        s = state
        self.items = [
            MenuItem("Preset",
                     lambda: s.preset_label(),
                     select=lambda: s.apply_preset(s.preset_idx),
                     adjust=lambda d: s.apply_preset(s.preset_idx + d)),
            MenuItem("Pitch",
                     lambda: f"{s.semitones:+.0f} st" if s.semitones else "off",
                     select=lambda: s.set_pitch(0),
                     adjust=lambda d: s.set_pitch(s.semitones + d),
                     slider=(lambda: s.semitones,
                             lambda v: s.set_pitch(int(round(v))),
                             -12, 12, "st")),
            MenuItem("Robot voice",
                     lambda: f"{s.robot:.0%}" if s.robot else "off",
                     select=self._toggle_robot,
                     adjust=lambda d: s.nudge("robot", d * 0.05, hi=1.0),
                     slider=(lambda: s.robot,
                             lambda v: s.set_val("robot", v, hi=1.0),
                             0.0, 1.0, "pct")),
            MenuItem("Helmet doubler",
                     lambda: f"{s.doubler:.0%}" if s.doubler else "off",
                     adjust=lambda d: s.nudge("doubler", d * 0.05, hi=1.0),
                     slider=(lambda: s.doubler,
                             lambda v: s.set_val("doubler", v, hi=1.0),
                             0.0, 1.0, "pct")),
            MenuItem("Grit / growl",
                     lambda: f"{s.drive:.0%}",
                     adjust=lambda d: s.nudge("drive", d * 0.05, hi=1.0),
                     slider=(lambda: s.drive,
                             lambda v: s.set_val("drive", v, hi=1.0),
                             0.0, 1.0, "pct")),
            MenuItem("Reverb",
                     lambda: f"{s.reverb:.0%}",
                     adjust=lambda d: s.nudge("reverb", d * 0.05, hi=1.0),
                     slider=(lambda: s.reverb,
                             lambda v: s.set_val("reverb", v, hi=1.0),
                             0.0, 1.0, "pct")),
            MenuItem("Echo",
                     lambda: f"{s.echo:.0%}",
                     adjust=lambda d: s.nudge("echo", d * 0.05, hi=1.0),
                     slider=(lambda: s.echo,
                             lambda v: s.set_val("echo", v, hi=1.0),
                             0.0, 1.0, "pct")),
            MenuItem("Radio voice",
                     lambda: "ON" if s.radio else "off",
                     select=self._toggle_radio,
                     adjust=lambda d: self._toggle_radio()),
            MenuItem("Bass boost",
                     lambda: f"{s.bass:.0%}" if s.bass else "off",
                     adjust=lambda d: s.nudge("bass", d * 0.05, hi=1.0),
                     slider=(lambda: s.bass,
                             lambda v: s.set_val("bass", v, hi=1.0),
                             0.0, 1.0, "pct")),
            MenuItem("Voice volume",
                     lambda: f"{s.voice_gain:.0%}",
                     adjust=lambda d: s.nudge("voice_gain", d * 0.05),
                     slider=(lambda: s.voice_gain,
                             lambda v: s.set_val("voice_gain", v),
                             0.0, 1.5, "pct")),
            MenuItem("Clip volume",
                     lambda: f"{s.clip_gain:.0%}",
                     adjust=lambda d: s.nudge("clip_gain", d * 0.05),
                     slider=(lambda: s.clip_gain,
                             lambda v: s.set_val("clip_gain", v),
                             0.0, 1.5, "pct")),
        ]
        if ai is not None and ai.available:
            self.items.append(MenuItem(
                "AI voice",
                lambda: ai.status,
                select=ai.toggle,
                adjust=lambda d: ai.toggle()))
            self.items.append(MenuItem(
                "AI character",
                lambda: ai.voice_name(),
                adjust=lambda d: ai.cycle(d)))
        self.items.append(MenuItem(
            "TTS voice FX",
            lambda: "ON" if s.tts_fx else "off",
            select=self._toggle_tts_fx,
            adjust=lambda d: self._toggle_tts_fx()))
        self.items.append(MenuItem(
            "TTS volume",
            lambda: f"{s.tts_gain:.0%}",
            adjust=lambda d: s.nudge("tts_gain", d * 0.05),
            slider=(lambda: s.tts_gain,
                    lambda v: s.set_val("tts_gain", v),
                    0.0, 1.5, "pct")))
        b = self.board
        self.items.append(MenuItem("Sounds to mic",
                                   lambda: "ON" if s.clips_to_mic else "off",
                                   select=b.toggle_mic,
                                   adjust=lambda d: b.toggle_mic()))
        self.items.append(MenuItem("Pause sounds",
                                   lambda: "PAUSED" if s.clips_paused else "off",
                                   select=b.toggle_pause,
                                   adjust=lambda d: b.toggle_pause()))
        self.items.append(MenuItem("Stop all sounds", select=b.stop))
        self.items.append(MenuItem("Quit", select=self.stop_flag.set, flash=False))

    def _toggle_robot(self):
        with self.state.lock:
            self.state.robot = 0.0 if self.state.robot > 0 else 1.0

    def _toggle_radio(self):
        with self.state.lock:
            self.state.radio = not self.state.radio

    def _toggle_tts_fx(self):
        with self.state.lock:
            self.state.tts_fx = not self.state.tts_fx

    def _toggle_monitor(self):
        self.monitor.toggle()
        if self.ai is not None:        # AI live: the worker mirrors the voice
            self.ai.set_monitor(self.monitor.on)
        if self.monitor.error:         # surface failures in the status line
            self.state.status_msg = f"test: {self.monitor.error}"
            self.state.status_at = time.time()

    def play_clip(self, i):
        self.board.play(i)

    # --- the on_* interface (called by both keyboard and controller paths) ---
    def on_up(self):    self.sel = (self.sel - 1) % len(self.items)
    def on_down(self):  self.sel = (self.sel + 1) % len(self.items)
    def on_left(self):  self._adjust(-1)
    def on_right(self): self._adjust(+1)
    def on_back(self):  self.stop_flag.set()

    def on_select(self):
        it = self.items[self.sel]
        if it.select:
            it.select()
            if it.value_fn is None and it.flash:
                self.flash[self.sel] = time.time() + 0.25

    def _adjust(self, d):
        it = self.items[self.sel]
        if it.adjust:
            it.adjust(d)


def run_ui(state, stop_flag, dev_line, err_line="", monitor=None, board=None,
           ai=None, tts=None):
    """VoiceBox skin, ported from design/VoiceBox Skin.dc.html.

    Faithful to the tokens JSON + motion spec in that file: Space Grotesk for
    labels / JetBrains Mono for values, cyan accent with a single glow recipe
    reserved for focus, sliding focus highlight (120ms), eased pixel scrolling
    (140ms), tile trigger flash (250ms easeIn), segmented mic meter with
    peak-hold, and toast-style status chips. easeOut(t)=1-(1-t)^2 per spec.
    """
    import pygame
    pygame.init()
    pygame.display.set_caption("VoiceBox")
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
    try:                          # OS-level minimum = the design's base size
        from pygame._sdl2.video import Window as _SDLWindow
        _SDLWindow.from_display_module().minimum_size = WINDOW_SIZE
    except Exception:
        pass
    clock = pygame.time.Clock()
    pygame.key.set_repeat(320, 110)           # held arrows auto-repeat

    cfg = load_controls()
    keymap, clipmap = build_keymap(cfg, pygame)
    pad = cfg["gamepad"]

    # controls.json is user-edited: coerce wrong-shaped values instead of
    # crashing the event loop (e.g. "select": 0 instead of [0]).
    def _buttons(v):
        if isinstance(v, int):
            return [v]
        if isinstance(v, list):
            return [b for b in v if isinstance(b, int)]
        return []

    def _num(v, fallback):
        try:
            return float(v)
        except (TypeError, ValueError):
            return fallback

    pad_select = _buttons(pad.get("select"))
    pad_back = _buttons(pad.get("back"))
    pad_stop = _buttons(pad.get("stop_clips"))
    threshold = _num(pad.get("axis_threshold"), 0.5)
    cooldown = _num(pad.get("nav_cooldown"), 0.22)
    joy_last = 0.0
    held_keys = set()             # set_repeat re-fires KEYDOWN; track real presses

    for i in range(pygame.joystick.get_count()):
        pygame.joystick.Joystick(i).init()

    # ------------------------------------------------------ theme (tokens JSON)
    CLR = {
        "bg":          (11, 13, 16),    "paneLeft":    (13, 16, 20),
        "raisedTop":   (23, 27, 34),    "raisedBot":   (19, 22, 28),
        "hoverTop":    (29, 35, 44),    "hoverBot":    (23, 27, 34),
        "active":      (35, 43, 54),    "stroke":      (35, 42, 52),
        "strokeSoft":  (28, 34, 43),    "strokeHover": (44, 53, 66),
        "accent":      (51, 214, 255),  "accentBright": (127, 230, 255),
        "accentDim":   (42, 175, 212),  "textOnAccent": (4, 20, 26),
        "danger":      (255, 77, 94),   "warning":     (255, 177, 61),
        "success":     (61, 220, 133),  "text":        (232, 237, 242),
        "text2":       (199, 208, 218), "muted":       (154, 167, 180),
        "faint":       (92, 104, 117),  "barTrack":    (35, 42, 52),
        "barFill":     (70, 82, 95),    "meterOff":    (26, 31, 38),
        "peak":        (255, 255, 255), "scrollTrack": (22, 27, 34),
        "scrollThumb": (58, 70, 83),    "headerTop":   (20, 24, 31),
        "headerBot":   (16, 20, 26),    "footerTop":   (16, 20, 26),
        "footerBot":   (13, 16, 21),
    }
    ACCENT_TINT = ((51, 214, 255, 26), (51, 214, 255, 13))
    DANGER_TINT = ((255, 77, 94, 20), (255, 77, 94, 10))
    WARN_TINT = ((255, 177, 61, 26), (255, 177, 61, 15))

    def mixc(a, b, t):
        return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t),
                int(a[2] + (b[2] - a[2]) * t))

    # fonts: bundled TTFs (assets/fonts) with system fallbacks
    FONTS_DIR = BASE_DIR / "assets" / "fonts"

    def _font(fname, size, fallback, bold=False):
        p = FONTS_DIR / fname
        try:
            if p.is_file():
                return pygame.font.Font(str(p), size)
        except Exception:
            pass
        try:
            return pygame.font.SysFont(fallback, size, bold=bold)
        except Exception:
            return pygame.font.Font(None, size + 6)

    f_word = _font("SpaceGrotesk-Bold.ttf", 17, "bahnschrift,segoeui", True)
    f_label = _font("SpaceGrotesk-Medium.ttf", 13, "segoeui")
    f_labelF = _font("SpaceGrotesk-SemiBold.ttf", 13, "segoeui", True)
    f_tile = _font("SpaceGrotesk-SemiBold.ttf", 13, "segoeui", True)
    f_hdr = _font("JetBrainsMono-Bold.ttf", 10, "consolas", True)
    f_val = _font("JetBrainsMono-Medium.ttf", 12, "consolas")
    f_valF = _font("JetBrainsMono-Bold.ttf", 12, "consolas", True)
    f_small = _font("JetBrainsMono-Medium.ttf", 10, "consolas")
    f_badge = _font("JetBrainsMono-Bold.ttf", 10, "consolas", True)
    f_strip = _font("JetBrainsMono-Bold.ttf", 11, "consolas", True)
    f_foot = _font("JetBrainsMono-Medium.ttf", 11, "consolas")

    menu = Menu(state, stop_flag, monitor, board, ai)
    board = menu.board
    if tts is None:
        tts = TTSBank(state, getattr(board, "player", None), monitor, ai)
    kb_action = {a: keys for a, keys in keymap.items()}

    def key_action(key):
        for action, keys in kb_action.items():
            if key in keys:
                return action
        return None

    # ------------------------------------------------------------- caches
    # Rendering is cached (text, gradients, glows): re-rendering every frame
    # competes with the audio callback for the GIL and can cause dropouts.
    text_cache = {}

    def T(fnt, s, color):
        key = (id(fnt), s, color)
        surf = text_cache.get(key)
        if surf is None:
            if len(text_cache) > 900:
                text_cache.clear()
            surf = fnt.render(s, True, color)
            text_cache[key] = surf
        return surf

    def TT(fnt, s, color, tracking):
        """Letter-spaced text (cached) - CSS letter-spacing equivalent."""
        key = ("trk", id(fnt), s, color, tracking)
        surf = text_cache.get(key)
        if surf is None:
            chars = [fnt.render(c, True, color) for c in s]
            w = sum(c.get_width() for c in chars) + tracking * max(0, len(chars) - 1)
            h = max((c.get_height() for c in chars), default=1)
            surf = pygame.Surface((max(1, w), h), pygame.SRCALPHA)
            x = 0
            for c in chars:
                surf.blit(c, (x, 0))
                x += c.get_width() + tracking
            text_cache[key] = surf
        return surf

    grad_cache = {}

    def grad(w, h, top, bot, radius=0):
        """Cached 2-stop vertical gradient, optionally rounded. RGB or RGBA."""
        key = (w, h, top, bot, radius)
        surf = grad_cache.get(key)
        if surf is None:
            if len(grad_cache) > 400:
                grad_cache.clear()
            t = top if len(top) == 4 else (*top, 255)
            b = bot if len(bot) == 4 else (*bot, 255)
            g = pygame.Surface((1, 2), pygame.SRCALPHA)
            g.set_at((0, 0), t)
            g.set_at((0, 1), b)
            surf = pygame.transform.smoothscale(g, (max(1, w), max(1, h)))
            if radius:
                mask = pygame.Surface((max(1, w), max(1, h)), pygame.SRCALPHA)
                pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(),
                                 border_radius=radius)
                surf.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
            grad_cache[key] = surf
        return surf

    glow_cache = {}
    G_PAD = 14

    def glow(w, h, color, radius):
        """The one glow recipe (focusGlow token) - soft rings, cached."""
        key = (w, h, color, radius)
        surf = glow_cache.get(key)
        if surf is None:
            if len(glow_cache) > 40:
                glow_cache.clear()
            surf = pygame.Surface((w + 2 * G_PAD, h + 2 * G_PAD), pygame.SRCALPHA)
            for e in range(G_PAD, 0, -1):
                a = int(34 * ((G_PAD - e + 1) / G_PAD) ** 2)
                pygame.draw.rect(surf, (*color, a),
                                 pygame.Rect(G_PAD - e, G_PAD - e, w + 2 * e, h + 2 * e),
                                 border_radius=radius + e)
            glow_cache[key] = surf
        return surf

    # ----------------------------------------------------- layout (tokens JSON)
    HEADER_H, FOOTER_H = 52, 32
    LEFT_W = 370
    VIEW_TOP = HEADER_H
    L_X, L_RIGHT = 10, LEFT_W - 14
    L_W = L_RIGHT - L_X
    ROW_HGT, ROW_GAP = 34, 2
    HDR_FIRST, HDR_HGT = 24, 30
    LIST_PAD_TOP = 6
    G_X = LEFT_W + 14
    COLS, GGAP, TILE_H = 3, 8, 62
    STRIP_Y, STRIP_H = VIEW_TOP + 10, 30
    GRID_TOP = STRIP_Y + STRIP_H + 10
    # TTS panel: bottom strip of the right pane (header / input / phrase list)
    TTS_H = 210
    TTS_IN_H = 30
    TTS_ROW_H, TTS_ROW_GAP = 30, 2

    # window-size-dependent geometry, owned by relayout(): the window is
    # resizable (drag edges / Aero snap); the left pane keeps its width, the
    # soundboard grid and TTS panel absorb the extra space.
    WIN_W = WIN_H = 0
    VIEW_BOT = VIEW_H = G_RIGHT = TILE_W = 0
    TTS_TOP = TTS_IN_Y = TTS_LIST_TOP = 0
    LIST_RECT = TTS_LIST_RECT = GRID_RECT = None
    disp_names = []

    def relayout():
        nonlocal WIN_W, WIN_H, VIEW_BOT, VIEW_H, G_RIGHT, TILE_W, TTS_TOP, \
            TTS_IN_Y, TTS_LIST_TOP, LIST_RECT, TTS_LIST_RECT, GRID_RECT, \
            disp_names
        # layout never goes below the design's base size, even if the OS
        # ignores the window minimum (drawing past the surface just clips)
        WIN_W = max(screen.get_width(), WINDOW_SIZE[0])
        WIN_H = max(screen.get_height(), WINDOW_SIZE[1])
        VIEW_BOT = WIN_H - FOOTER_H
        VIEW_H = VIEW_BOT - VIEW_TOP
        G_RIGHT = WIN_W - 14 - 8               # 8px scroll gutter
        TILE_W = (G_RIGHT - G_X - GGAP * (COLS - 1)) // COLS
        LIST_RECT = pygame.Rect(0, VIEW_TOP, LEFT_W, VIEW_H)
        TTS_TOP = VIEW_BOT - TTS_H
        TTS_IN_Y = TTS_TOP + 30
        TTS_LIST_TOP = TTS_IN_Y + TTS_IN_H + 8
        TTS_LIST_RECT = pygame.Rect(LEFT_W + 1, TTS_LIST_TOP,
                                    WIN_W - LEFT_W - 1,
                                    VIEW_BOT - TTS_LIST_TOP)
        GRID_RECT = pygame.Rect(LEFT_W + 1, GRID_TOP, WIN_W - LEFT_W - 1,
                                TTS_TOP - GRID_TOP)
        # tile width follows the window: re-truncate + measure grid labels
        disp_names = []
        name_max = TILE_W - 20 - 22
        for nm, _c in zip(state.clip_names, state.clips):
            if f_tile.render(nm, True, CLR["text"]).get_width() > name_max:
                while nm and f_tile.render(nm + "...", True,
                                           CLR["text"]).get_width() > name_max:
                    nm = nm[:-1]
                nm += "..."
            disp_names.append(nm)

    relayout()

    SECTION_OF = {
        "Preset": "VOICE", "Pitch": "VOICE",
        "Robot voice": "EFFECTS", "Helmet doubler": "EFFECTS",
        "Grit / growl": "EFFECTS", "Reverb": "EFFECTS", "Echo": "EFFECTS",
        "Radio voice": "EFFECTS", "Bass boost": "EFFECTS",
        "Voice volume": "EFFECTS", "Clip volume": "EFFECTS",
        "AI voice": "AI", "AI character": "AI",
        "TTS voice FX": "TTS", "TTS volume": "TTS",
        "Sounds to mic": "SOUNDS", "Pause sounds": "SOUNDS",
        "Stop all sounds": "SOUNDS",
        "Quit": "SYSTEM",
    }
    layout, row_pos = [], {}
    y_acc, last_sec = LIST_PAD_TOP, None
    for i, it in enumerate(menu.items):
        sec = SECTION_OF.get(it.label, last_sec)
        if sec is not None and sec != last_sec:
            hh = HDR_FIRST if not layout else HDR_HGT
            layout.append(("hdr", sec, y_acc, hh))
            y_acc += hh
            last_sec = sec
        layout.append(("row", i, y_acc, ROW_HGT))
        row_pos[i] = y_acc
        y_acc += ROW_HGT + ROW_GAP
    content_h = y_acc + 20

    grid_rows = (len(state.clips) + COLS - 1) // COLS
    grid_content_h = (grid_rows * (TILE_H + GGAP) - GGAP + 20) if state.clips else 0
    clip_by_id = {id(c): i for i, c in enumerate(state.clips)}
    clip_secs = [f"{len(c) / SAMPLERATE:.1f}s" for c in state.clips]

    # ----------------------------------------------------------- motion state
    list_scroll = list_target = 0.0
    grid_scroll = grid_target = 0.0
    tts_scroll = tts_target = 0.0
    tts_text, tts_focus = "", False
    tts_trunc = {}                # phrase -> truncated display string
    focus_y = float(row_pos.get(menu.sel, LIST_PAD_TOP))
    hover_mix = {}                # element key -> 0..1 hover blend
    nudge = {"i": -1, "at": 0.0, "side": 0}
    strip_press = {}
    meter_lit = 0.0
    peak_lit, peak_at = 0.0, 0.0
    row_hit, strip_hit, grid_hit = {}, {}, {}
    tts_row_hit, tts_del_hit, tts_btn_hit = {}, {}, {}
    arrow_hit = None              # (row, "<" rect, ">" rect) from the last draw
    slider_hit, slider_track, value_hit = {}, {}, {}   # numeric rows, per draw
    slider_drag = None            # row index while a slider knob is dragged
    edit = None                   # {"row": i, "text": str} while typing a value
    last_t = time.time()

    def step(cur, target, dt, dur):
        """One easeOut lerp step toward target (spec: easeOut = 1-(1-t)^2)."""
        if dur <= 0:
            return float(target)
        k = min(1.0, dt / dur)
        k = k * (2.0 - k)
        v = cur + (target - cur) * k
        return float(target) if abs(target - v) < 0.4 else v

    def hover_step(key, active, dt):
        m = hover_mix.get(key, 0.0)
        m = step(m, 1.0 if active else 0.0, dt, 0.09 if active else 0.15)
        if m <= 0.002:
            hover_mix.pop(key, None)
            return 0.0
        hover_mix[key] = m
        return m

    def q8(t):
        return round(t * 8) / 8    # quantize blends so gradient cache stays small

    def row_at(pos):
        """Menu row under the mouse (uses the rects from the last draw)."""
        for i, r in row_hit.items():
            if r.collidepoint(pos):
                return i
        return None

    def tts_commit():
        nonlocal tts_text
        if tts.add(tts_text):
            tts_text = ""

    def tts_set_focus(on):
        """Textbox focus: while on, the keyboard belongs to the textbox."""
        nonlocal tts_focus
        if on == tts_focus:
            return
        tts_focus = on
        try:
            (pygame.key.start_text_input if on else pygame.key.stop_text_input)()
        except Exception:
            pass

    def go_left():
        menu.on_left()
        nudge.update(i=menu.sel, at=time.time(), side=-1)

    def go_right():
        menu.on_right()
        nudge.update(i=menu.sel, at=time.time(), side=+1)

    # ---- numeric rows: slider drag + click-the-number-to-type ---------------
    def slider_set_from_x(i, mx):
        tr = slider_track.get(i)
        if tr is None:
            return
        _get, set_, lo, hi, unit = menu.items[i].slider
        frac = max(0.0, min(1.0, (mx - tr.x) / max(1, tr.w)))
        v = lo + frac * (hi - lo)
        set_(int(round(v)) if unit == "st" else v)

    def edit_open(i):
        nonlocal edit
        tts_set_focus(False)
        edit = {"row": i, "text": ""}   # empty box; current value = placeholder
        try:
            pygame.key.start_text_input()
        except Exception:
            pass

    def edit_close():
        nonlocal edit
        edit = None
        if not tts_focus:
            try:
                pygame.key.stop_text_input()
            except Exception:
                pass

    def edit_commit():
        if edit is not None and edit["text"].strip():
            _get, set_, lo, hi, unit = menu.items[edit["row"]].slider
            try:
                v = float(edit["text"].replace(",", ".").strip())
            except ValueError:
                v = None
            if v is not None:
                if unit == "pct":
                    v /= 100.0
                set_(max(lo, min(hi, int(round(v)) if unit == "st" else v)))
        edit_close()

    def val_style(val, focused, now):
        """(font, text color, LED dot color or None) by value semantics."""
        if val == "ON":
            return f_valF, CLR["accent"], CLR["accent"]
        if val == "PAUSED":
            return f_valF, CLR["warning"], CLR["warning"]
        if val == "error":
            return f_valF, CLR["danger"], CLR["danger"]
        if val.startswith("loading"):
            a = 0.4 + 0.6 * (0.5 + 0.5 * float(np.sin(now * 2 * np.pi / 1.2)))
            return (f_val, mixc(CLR["paneLeft"], CLR["muted"], a),
                    mixc(CLR["paneLeft"], CLR["accent"], a))
        if val == "off":
            return f_val, CLR["faint"], None
        if focused:
            return f_valF, CLR["accent"], None
        return f_val, CLR["text"], None

    def draw_slider(r, i, it, val, focused, now):
        """Numeric row: label | slider track+knob | value (click to type)."""
        get, _set, lo, hi, _unit = it.slider
        cy = r.centery
        cur = float(get())
        frac = max(0.0, min(1.0, (cur - lo) / (hi - lo) if hi != lo else 0.0))
        # value on the right: an input box while editing, else clickable text
        if edit is not None and edit["row"] == i:
            box = pygame.Rect(r.right - 8 - 54, cy - 11, 54, 22)
            screen.blit(grad(box.w, box.h, CLR["headerBot"], CLR["paneLeft"], 5),
                        box.topleft)
            pygame.draw.rect(screen, CLR["accent"], box, width=1, border_radius=5)
            txt = edit["text"]
            if txt:
                ts = T(f_valF, txt, CLR["text"])
            else:                          # empty: current value as placeholder
                ts = T(f_val, val, CLR["faint"])
            tx = box.right - 6 - ts.get_width()
            screen.blit(ts, (tx, cy - ts.get_height() // 2))
            if txt and (now * 2.0) % 2 < 1:
                pygame.draw.line(screen, CLR["accent"],
                                 (box.right - 5, cy - 7), (box.right - 5, cy + 7))
            value_hit[i] = box
            val_left = box.x
        else:
            n_hot = nudge["i"] == i and (now - nudge["at"]) < 0.18
            fnt, col, _dot = val_style(val, focused, now)
            ts = T(f_valF if focused else fnt, val,
                   CLR["accentBright"] if n_hot else
                   (CLR["accent"] if focused else col))
            vr = pygame.Rect(r.right - 10 - ts.get_width(),
                             cy - ts.get_height() // 2,
                             ts.get_width(), ts.get_height())
            vh = q8(hover_step(("val", i), vr.inflate(12, 10)
                               .collidepoint(mouse_pos), dt))
            if vh > 0.3:                   # hint: the number is clickable
                pygame.draw.rect(screen, mixc(CLR["paneLeft"], CLR["accent"], 0.35),
                                 vr.inflate(12, 8), width=1, border_radius=5)
            screen.blit(ts, vr.topleft)
            value_hit[i] = vr.inflate(12, 10)
            val_left = vr.x - 6
        # slider track + knob in the middle of the row
        tx1 = min(val_left - 12, r.right - 10 - 54 - 12)
        tx0 = max(r.x + 148, tx1 - 124)
        track = pygame.Rect(tx0, cy - 2, tx1 - tx0, 4)
        pygame.draw.rect(screen, CLR["barTrack"], track, border_radius=2)
        if frac > 0.01:
            pygame.draw.rect(screen, CLR["accent"] if focused else CLR["barFill"],
                             pygame.Rect(track.x, track.y,
                                         max(2, int(track.w * frac)), 4),
                             border_radius=2)
        kx = track.x + int(track.w * frac)
        kh = q8(hover_step(("knob", i),
                           slider_drag == i
                           or track.inflate(10, 14).collidepoint(mouse_pos), dt))
        pygame.draw.circle(screen, CLR["paneLeft"], (kx, cy), 7)
        pygame.draw.circle(screen,
                           mixc(CLR["accent"] if focused else CLR["muted"],
                                CLR["accentBright"], kh),
                           (kx, cy), 5)
        slider_track[i] = track
        slider_hit[i] = track.inflate(12, 16)

    def draw_value(r, i, it, val, focused, now):
        nonlocal arrow_hit
        if it.slider is not None:
            draw_slider(r, i, it, val, focused, now)
            return
        cy = r.centery
        is_pct = val.endswith("%")
        n_hot = nudge["i"] == i and (now - nudge["at"]) < 0.18
        if focused and it.adjust:
            ra = pygame.Rect(r.right - 5 - 20, cy - 11, 20, 22)
            if is_pct:
                vs = T(f_valF, val, CLR["accentBright"] if n_hot else CLR["accent"])
                vx = ra.x - 6 - 34
                screen.blit(vs, (vx + 34 - vs.get_width(), cy - vs.get_height() // 2))
                bar = pygame.Rect(vx - 7 - 56, cy - 2, 56, 4)
                pygame.draw.rect(screen, CLR["barTrack"], bar, border_radius=2)
                frac = min(1.0, float(val[:-1]) / 100.0)
                if frac > 0.01:
                    pygame.draw.rect(screen, CLR["accent"],
                                     pygame.Rect(bar.x, bar.y, max(2, int(56 * frac)), 4),
                                     border_radius=2)
                la = pygame.Rect(bar.x - 8 - 20, cy - 11, 20, 22)
            else:
                fnt, col, dot = val_style(val, True, now)
                vs = T(f_valF, val, CLR["accentBright"] if n_hot else col)
                vx = ra.x - 8 - vs.get_width()
                screen.blit(vs, (vx, cy - vs.get_height() // 2))
                if dot:
                    pygame.draw.circle(screen, dot, (vx - 10, cy), 2)
                    vx -= 14
                la = pygame.Rect(vx - 8 - 20, cy - 11, 20, 22)
            for side, arect in ((-1, la), (1, ra)):
                if n_hot and nudge["side"] == side:
                    pygame.draw.rect(screen, CLR["accent"], arect, border_radius=5)
                    ss = T(f_valF, "<" if side < 0 else ">", CLR["textOnAccent"])
                else:
                    pygame.draw.rect(screen, mixc(CLR["paneLeft"], CLR["accent"], 0.35),
                                     arect, width=1, border_radius=5)
                    ss = T(f_valF, "<" if side < 0 else ">", CLR["accent"])
                screen.blit(ss, (arect.centerx - ss.get_width() // 2,
                                 arect.centery - ss.get_height() // 2))
            arrow_hit = (i, la, ra)
        else:
            right = r.right - 10
            if is_pct:
                vs = T(f_valF if focused else f_val, val,
                       CLR["accent"] if focused else CLR["text"])
                screen.blit(vs, (right - vs.get_width(), cy - vs.get_height() // 2))
                bar = pygame.Rect(right - 34 - 7 - 56, cy - 2, 56, 4)
                pygame.draw.rect(screen, CLR["barTrack"], bar, border_radius=2)
                frac = min(1.0, float(val[:-1]) / 100.0)
                if frac > 0.01:
                    pygame.draw.rect(screen,
                                     CLR["accent"] if focused else CLR["barFill"],
                                     pygame.Rect(bar.x, bar.y, max(2, int(56 * frac)), 4),
                                     border_radius=2)
            else:
                fnt, col, dot = val_style(val, focused, now)
                vs = T(fnt, val, col)
                screen.blit(vs, (right - vs.get_width(), cy - vs.get_height() // 2))
                if dot:
                    pygame.draw.circle(screen, dot, (right - vs.get_width() - 10, cy), 2)

    # -------------------------------------- dropdown picker (Preset / AI voice)
    # Pressing the Preset or AI character row opens an alphabetical list
    # anchored to the row; while open it owns keyboard, mouse and controller.
    DROP_ROWS = ("Preset", "AI character")
    drop = None                   # dict(items, rect, sel, cur, scroll, ...) | None

    def open_dropdown(row_idx):
        nonlocal drop
        label = menu.items[row_idx].label
        if label == "Preset":
            entries = sorted(((nm, i) for i, (nm, _p) in enumerate(PRESETS)),
                             key=lambda e: e[0].lower())
            items = [(nm, lambda i=i: state.apply_preset(i))
                     for nm, i in entries]
            cur = next((k for k, (_nm, i) in enumerate(entries)
                        if i == state.preset_idx), 0)
        elif ai is not None:
            entries = sorted(((p.stem, i) for i, p in enumerate(ai.voices)),
                             key=lambda e: e[0].lower())
            items = [(nm, lambda i=i: ai.select(i)) for nm, i in entries]
            cur = next((k for k, (_nm, i) in enumerate(entries)
                        if i == ai.sel), 0)
        else:
            return
        item_h, pad = 28, 4
        ry = VIEW_TOP - int(list_scroll) + row_pos[row_idx]
        want = len(items) * item_h + pad * 2
        below = VIEW_BOT - 6 - (ry + ROW_HGT + 4)
        above = ry - 4 - (VIEW_TOP + 6)
        if below >= min(want, 200) or below >= above:
            h, y = min(want, below), ry + ROW_HGT + 4
        else:
            h, y = min(want, above), ry - 4 - min(want, above)
        rect = pygame.Rect(L_X + 10, y, L_W - 20, h)
        max_scroll = max(0, want - h)
        scroll = min(max_scroll,
                     max(0, cur * item_h + pad - (h - item_h) // 2))
        drop = {"items": items, "rect": rect, "item_h": item_h, "pad": pad,
                "sel": cur, "cur": cur, "scroll": scroll,
                "max_scroll": max_scroll, "row": row_idx, "mouse": None}

    def drop_pick():
        nonlocal drop
        if drop and 0 <= drop["sel"] < len(drop["items"]):
            drop["items"][drop["sel"]][1]()
            menu.flash[drop["row"]] = time.time() + 0.25
        drop = None

    def drop_nav(d):
        drop["sel"] = (drop["sel"] + d) % len(drop["items"])
        view_h = drop["rect"].h - drop["pad"] * 2
        top = drop["sel"] * drop["item_h"]
        if top < drop["scroll"]:
            drop["scroll"] = top
        elif top + drop["item_h"] > drop["scroll"] + view_h:
            drop["scroll"] = top + drop["item_h"] - view_h

    def drop_event(event):
        """All input routes here while the picker is open."""
        nonlocal drop
        if event.type == pygame.KEYDOWN:
            held_keys.add(event.key)
            act = key_action(event.key)
            if   act == "up":     drop_nav(-1)
            elif act == "down":   drop_nav(+1)
            elif act == "select": drop_pick()
            elif act == "back" or event.key == pygame.K_ESCAPE:
                drop = None
        elif event.type == pygame.MOUSEWHEEL:
            drop["scroll"] = max(0, min(drop["max_scroll"],
                                        drop["scroll"]
                                        - event.y * drop["item_h"]))
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            r = drop["rect"]
            if r.collidepoint(event.pos):
                i = (event.pos[1] - r.y - drop["pad"]
                     + int(drop["scroll"])) // drop["item_h"]
                if 0 <= i < len(drop["items"]):
                    drop["sel"] = i
                    drop_pick()
            else:
                drop = None                # click elsewhere just closes
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drop = None
        elif event.type == pygame.JOYBUTTONDOWN:
            if   event.button in pad_select: drop_pick()
            elif event.button in pad_back:   drop = None
        elif event.type == pygame.JOYHATMOTION and event.value != (0, 0):
            if   event.value[1] ==  1: drop_nav(-1)
            elif event.value[1] == -1: drop_nav(+1)

    def do_select():
        """Row select: dropdown rows open the picker, the rest act directly."""
        if menu.items[menu.sel].label in DROP_ROWS:
            open_dropdown(menu.sel)
        else:
            menu.on_select()

    # ================================================================== loop
    while not stop_flag.is_set():
        now = time.time()
        dt = min(0.1, now - last_t)
        last_t = now

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_flag.set()

            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h),
                                                 pygame.RESIZABLE)
                relayout()

            elif drop is not None and event.type in (
                    pygame.KEYDOWN, pygame.KEYUP, pygame.MOUSEBUTTONDOWN,
                    pygame.MOUSEWHEEL, pygame.MOUSEMOTION,
                    pygame.JOYBUTTONDOWN, pygame.JOYHATMOTION,
                    pygame.JOYAXISMOTION):
                if event.type == pygame.KEYUP:
                    held_keys.discard(event.key)
                else:
                    drop_event(event)

            elif event.type == pygame.JOYDEVICEADDED:
                pygame.joystick.Joystick(event.device_index).init()
            elif event.type == pygame.JOYDEVICEREMOVED:
                pass                                   # instance dies on its own

            elif event.type == pygame.TEXTINPUT:
                if edit is not None:
                    if all(c in "0123456789.,+-" for c in event.text):
                        edit["text"] = (edit["text"] + event.text)[:6]
                elif tts_focus:
                    tts_text = (tts_text + event.text)[:TTS_MAX_CHARS]

            elif event.type == pygame.KEYDOWN and edit is not None:
                # the value box owns the keyboard (digits are clip hotkeys!)
                held_keys.add(event.key)
                if event.key == pygame.K_BACKSPACE:
                    edit["text"] = edit["text"][:-1]
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    edit_commit()
                elif event.key == pygame.K_ESCAPE:
                    edit_close()

            elif event.type == pygame.KEYDOWN and tts_focus:
                # the textbox owns the keyboard: no menu nav, no clip hotkeys
                held_keys.add(event.key)
                if event.key == pygame.K_BACKSPACE:
                    tts_text = tts_text[:-1]
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    tts_commit()
                elif event.key == pygame.K_ESCAPE:
                    tts_set_focus(False)

            elif event.type == pygame.KEYDOWN:
                # auto-repeat is only for navigation/adjust; a held clip key must
                # not stack a new copy of the clip every repeat interval
                repeat = event.key in held_keys
                held_keys.add(event.key)
                if event.key in clipmap:
                    if not repeat and clipmap[event.key] < len(state.clips):
                        menu.play_clip(clipmap[event.key])
                    continue
                act = key_action(event.key)
                if   act == "up":         menu.on_up()
                elif act == "down":       menu.on_down()
                elif act == "left":       go_left()
                elif act == "right":      go_right()
                elif repeat:              pass
                elif act == "select":     do_select()
                elif act == "back":       menu.on_back()
                elif act == "stop_clips": board.stop()

            elif event.type == pygame.KEYUP:
                held_keys.discard(event.key)
            elif event.type == pygame.WINDOWFOCUSLOST:
                held_keys.clear()         # KEYUPs are lost on focus change

            elif event.type == pygame.MOUSEMOTION:
                if slider_drag is not None:        # live drag: follow the mouse
                    slider_set_from_x(slider_drag, event.pos[0])
                else:
                    idx = row_at(event.pos)
                    if idx is not None:
                        menu.sel = idx

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                slider_drag = None

            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                if mx >= LEFT_W and my >= TTS_TOP:           # over the TTS panel
                    tts_target -= event.y * (TTS_ROW_H + TTS_ROW_GAP)
                elif mx >= LEFT_W:                           # over the grid pane
                    grid_target -= event.y * (TILE_H + GGAP)
                else:
                    for _ in range(abs(event.y)):
                        menu.on_up() if event.y > 0 else menu.on_down()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                in_r = tts_btn_hit.get("input")
                tts_set_focus(in_r is not None and in_r.collidepoint(event.pos))
                if edit is not None and not value_hit.get(
                        edit["row"],
                        pygame.Rect(0, 0, 0, 0)).collidepoint(event.pos):
                    edit_commit()          # clicking elsewhere confirms it
                hit = next((k for k, r in strip_hit.items()
                            if r.collidepoint(event.pos)), None)
                if hit is not None:
                    strip_press[hit] = now
                if hit == "mic":
                    board.toggle_mic()
                elif hit == "pause":
                    board.toggle_pause()
                elif hit == "stop":
                    board.stop()
                elif hit == "hear":
                    menu._toggle_monitor()
                elif (r := tts_btn_hit.get("add")) is not None \
                        and r.collidepoint(event.pos):
                    tts_commit()
                elif (r := tts_btn_hit.get("fx")) is not None \
                        and r.collidepoint(event.pos):
                    menu._toggle_tts_fx()
                elif (di := next((i for i, r in tts_del_hit.items()
                                  if r.collidepoint(event.pos)), None)) is not None:
                    tts.delete(di)
                elif (ti := next((i for i, r in tts_row_hit.items()
                                  if r.collidepoint(event.pos)), None)) is not None:
                    tts.play(ti)
                elif GRID_RECT.collidepoint(event.pos) and (
                        ci := next((c for c, r in grid_hit.items()
                                    if r.collidepoint(event.pos)), None)) is not None:
                    board.play(ci)
                elif (si := next((k for k, rr in slider_hit.items()
                                  if rr.collidepoint(event.pos)), None)) is not None:
                    menu.sel = si
                    slider_drag = si       # jump to the click, then live-drag
                    slider_set_from_x(si, event.pos[0])
                elif (vi := next((k for k, rr in value_hit.items()
                                  if rr.collidepoint(event.pos)), None)) is not None:
                    menu.sel = vi
                    edit_open(vi)          # type the number directly
                else:
                    idx = row_at(event.pos)
                    if idx is not None:
                        menu.sel = idx
                        it = menu.items[idx]
                        on_row = arrow_hit is not None and arrow_hit[0] == idx
                        if it.adjust and on_row and arrow_hit[1].collidepoint(event.pos):
                            go_left()
                        elif it.adjust and on_row and arrow_hit[2].collidepoint(event.pos):
                            go_right()
                        elif it.label in DROP_ROWS:
                            open_dropdown(idx)
                        elif it.select:
                            menu.on_select()

            elif event.type == pygame.JOYBUTTONDOWN:
                if   event.button in pad_select: do_select()
                elif event.button in pad_back:   menu.on_back()
                elif event.button in pad_stop:   board.stop()

            elif event.type == pygame.JOYHATMOTION and event.value != (0, 0):
                jnow = time.time()
                if jnow - joy_last >= cooldown:
                    joy_last = jnow
                    hx, hy = event.value
                    if   hy ==  1: menu.on_up()
                    elif hy == -1: menu.on_down()
                    elif hx == -1: go_left()
                    elif hx ==  1: go_right()

            # left stick only (axes 0/1): filter BEFORE touching the cooldown so
            # trigger/right-stick events can't silently eat the nav timer
            elif (event.type == pygame.JOYAXISMOTION and event.axis in (0, 1)
                  and abs(event.value) > threshold):
                jnow = time.time()
                if jnow - joy_last >= cooldown:
                    joy_last = jnow
                    if event.axis == 0:
                        go_left() if event.value < 0 else go_right()
                    else:
                        menu.on_up() if event.value < 0 else menu.on_down()

        # ------------------------------------------------------------- draw
        mouse_pos = pygame.mouse.get_pos()
        screen.fill(CLR["bg"])

        # header: wordmark + segmented mic meter
        screen.blit(grad(WIN_W, HEADER_H, CLR["headerTop"], CLR["headerBot"]),
                    (0, 0))
        pygame.draw.line(screen, CLR["strokeSoft"], (0, HEADER_H - 1),
                         (WIN_W, HEADER_H - 1))
        bx = 16
        for bh in (6, 13, 9, 16):
            pygame.draw.rect(screen, CLR["accent"],
                             pygame.Rect(bx, HEADER_H // 2 + 8 - bh, 3, bh),
                             border_radius=1)
            bx += 5
        wv = TT(f_word, "VOICE", CLR["text"], 1)
        wb = TT(f_word, "BOX", CLR["accent"], 1)
        wy = (HEADER_H - wv.get_height()) // 2
        screen.blit(wv, (bx + 6, wy))
        screen.blit(wb, (bx + 6 + wv.get_width(), wy))

        level = state.in_level
        db = 20.0 * float(np.log10(max(level, 1e-4)))
        seg_target = max(0.0, min(22.0, (db + 48.0) / 48.0 * 22.0))
        if seg_target > meter_lit:                     # attack 40ms / decay 240ms
            meter_lit = min(seg_target, meter_lit + dt / 0.040 * 22.0)
        else:
            meter_lit = max(seg_target, meter_lit - dt / 0.240 * 22.0)
        if meter_lit >= peak_lit:
            peak_lit, peak_at = meter_lit, now
        elif now - peak_at > 0.9:                      # hold 900ms, fall 300ms
            peak_lit = max(meter_lit, peak_lit - dt / 0.300 * 22.0)
        mx_right = WIN_W - 16
        db_s = T(f_small, f"{max(db, -60.0):5.1f} dB", CLR["muted"])
        screen.blit(db_s, (mx_right - db_s.get_width(),
                           (HEADER_H - db_s.get_height()) // 2))
        seg_x0 = mx_right - 52 - 10 - (22 * 7 - 2)
        lit_n, peak_n = int(meter_lit), int(peak_lit)
        for si in range(22):
            col = CLR["meterOff"]
            if si < lit_n:
                col = (CLR["success"] if si < 16 else
                       CLR["warning"] if si < 19 else CLR["danger"])
            elif si == peak_n and peak_n > lit_n:
                col = CLR["peak"]
            pygame.draw.rect(screen, col,
                             pygame.Rect(seg_x0 + si * 7, (HEADER_H - 13) // 2, 5, 13),
                             border_radius=1)
        mic_s = TT(f_hdr, "MIC", CLR["faint"], 1)
        screen.blit(mic_s, (seg_x0 - 10 - mic_s.get_width(),
                            (HEADER_H - mic_s.get_height()) // 2))

        # ------------------------------------------------- left pane: settings
        pygame.draw.rect(screen, CLR["paneLeft"], LIST_RECT)
        pygame.draw.line(screen, CLR["strokeSoft"], (LEFT_W, VIEW_TOP),
                         (LEFT_W, VIEW_BOT))

        ry = row_pos.get(menu.sel, LIST_PAD_TOP)
        if ry - list_target < 6:                       # keep focus in view
            list_target = max(0.0, ry - 6)
        elif ry + ROW_HGT - list_target > VIEW_H - 6:
            list_target = ry + ROW_HGT - (VIEW_H - 6)
        list_target = max(0.0, min(list_target, max(0.0, content_h - VIEW_H)))
        list_scroll = step(list_scroll, list_target, dt, 0.14)
        focus_y = step(focus_y, float(ry), dt, 0.12)

        screen.set_clip(LIST_RECT)
        row_hit.clear()
        slider_hit.clear()
        slider_track.clear()
        value_hit.clear()
        arrow_hit = None
        base_y = VIEW_TOP - int(list_scroll)

        # sliding focus highlight: ring + tint + the one glow
        fr = pygame.Rect(L_X, base_y + int(focus_y), L_W, ROW_HGT)
        screen.blit(glow(L_W, ROW_HGT, CLR["accent"], 7), (fr.x - G_PAD, fr.y - G_PAD))
        screen.blit(grad(L_W, ROW_HGT, ACCENT_TINT[0], ACCENT_TINT[1], 7), fr.topleft)
        pygame.draw.rect(screen, CLR["accent"], fr, width=1, border_radius=7)

        for kind, data, ly, lh in layout:
            sy = base_y + ly
            if sy + lh < VIEW_TOP or sy > VIEW_BOT:
                continue
            if kind == "hdr":
                hs = TT(f_hdr, data, CLR["faint"], 2)
                ty = sy + lh - hs.get_height() - 4
                screen.blit(hs, (L_X + 4, ty))
                lyy = ty + hs.get_height() // 2
                pygame.draw.line(screen, CLR["strokeSoft"],
                                 (L_X + 4 + hs.get_width() + 8, lyy), (L_RIGHT - 4, lyy))
                continue
            i = data
            it = menu.items[i]
            r = pygame.Rect(L_X, sy, L_W, ROW_HGT)
            row_hit[i] = r
            focused = (i == menu.sel)
            fl = menu.flash.get(i, 0) - now            # select-confirm flash
            if fl > 0:
                f = q8(min(1.0, fl / 0.25) ** 2)
                screen.blit(grad(L_W, ROW_HGT,
                                 mixc(CLR["paneLeft"], CLR["accentDim"], f),
                                 mixc(CLR["paneLeft"], CLR["accentDim"], f * 0.8), 7),
                            r.topleft)
            elif not focused:
                hm = q8(hover_step(("row", i), r.collidepoint(mouse_pos), dt))
                if hm > 0:
                    top = mixc(CLR["paneLeft"], CLR["hoverTop"], hm)
                    screen.blit(grad(L_W, ROW_HGT, top, top, 7), r.topleft)
                    if hm > 0.4:
                        pygame.draw.rect(screen, CLR["strokeHover"], r,
                                         width=1, border_radius=7)
            if (it.label == "AI voice" and ai is not None
                    and ai.status == "error" and not focused):
                screen.blit(grad(L_W, ROW_HGT, DANGER_TINT[0], DANGER_TINT[1], 7),
                            r.topleft)
                pygame.draw.rect(screen, mixc(CLR["paneLeft"], CLR["danger"], 0.35),
                                 r, width=1, border_radius=7)

            ls = T(f_labelF if focused else f_label, it.label,
                   CLR["text"] if focused else CLR["text2"])
            screen.blit(ls, (r.x + 12, r.y + (ROW_HGT - ls.get_height()) // 2))
            if it.value_fn is not None:
                draw_value(r, i, it, it.value_fn(), focused, now)
            elif it.select:
                vs = T(f_val, "↵", CLR["faint"])
                screen.blit(vs, (r.right - 10 - vs.get_width(),
                                 r.y + (ROW_HGT - vs.get_height()) // 2))

        screen.blit(grad(LEFT_W - 6, 26, (13, 16, 20, 0), (13, 16, 20, 255)),
                    (0, VIEW_BOT - 26))
        if content_h > VIEW_H:
            track = pygame.Rect(LEFT_W - 5, VIEW_TOP + 4, 3, VIEW_H - 8)
            pygame.draw.rect(screen, CLR["scrollTrack"], track, border_radius=2)
            th = max(24, int(track.height * VIEW_H / content_h))
            tt_y = track.y + int((track.height - th)
                                 * (list_scroll / max(1.0, content_h - VIEW_H)))
            pygame.draw.rect(screen, CLR["scrollThumb"],
                             pygame.Rect(track.x, tt_y, 3, th), border_radius=2)
        screen.set_clip(None)

        # -------------------------------------------- right pane: control strip
        strip_hit.clear()
        sx = G_X
        strip_defs = [
            ("mic", "TO MIC: ON" if state.clips_to_mic else "TO MIC: OFF",
             state.clips_to_mic, CLR["accent"], ACCENT_TINT),
            ("pause", "PAUSED" if state.clips_paused else "PAUSE",
             state.clips_paused, CLR["warning"], WARN_TINT),
            ("stop", "STOP", False, CLR["accent"], None),
        ]
        if monitor is not None:            # self-listen toggle ("hear myself")
            strip_defs.append(
                ("hear", "HEAR: ON" if monitor.on else "HEAR: OFF",
                 monitor.on, CLR["accent"], ACCENT_TINT))
        for key, lab, active, acol, tint in strip_defs:
            hm = q8(hover_step(("strip", key),
                               pygame.Rect(sx, STRIP_Y, 10, STRIP_H).collidepoint(mouse_pos)
                               or (strip_hit.get(key) or pygame.Rect(0, 0, 0, 0)
                                   ).collidepoint(mouse_pos), dt))
            base_ts = T(f_strip, lab, acol if active else
                        (CLR["text"] if hm > 0.5 else CLR["muted"]))
            w = base_ts.get_width() + 24 + (12 if active else 0)
            r = pygame.Rect(sx, STRIP_Y, w, STRIP_H)
            hm = q8(hover_step(("strip2", key), r.collidepoint(mouse_pos), dt))
            pm = max(0.0, 1.0 - (now - strip_press.get(key, 0)) / 0.08) \
                if strip_press.get(key) else 0.0
            if active and tint:
                screen.blit(grad(w, STRIP_H, tint[0], tint[1], 7), r.topleft)
                pygame.draw.rect(screen, mixc(CLR["bg"], acol, 0.45), r,
                                 width=1, border_radius=7)
            else:
                top = mixc(CLR["raisedTop"], CLR["hoverTop"], hm)
                bot = mixc(CLR["raisedBot"], CLR["hoverBot"], hm)
                if pm > 0:
                    top = mixc(top, CLR["active"], q8(pm))
                    bot = mixc(bot, CLR["active"], q8(pm))
                screen.blit(grad(w, STRIP_H, top, bot, 7), r.topleft)
                pygame.draw.rect(screen, mixc(CLR["stroke"], CLR["strokeHover"], hm),
                                 r, width=1, border_radius=7)
            tx = r.x + 12
            if active:
                pygame.draw.circle(screen, acol, (tx + 2, r.centery), 2)
                tx += 12
            ts = T(f_strip, lab, acol if active else
                   (CLR["text"] if hm > 0.5 else CLR["muted"]))
            screen.blit(ts, (tx, r.centery - ts.get_height() // 2))
            strip_hit[key] = r
            sx = r.right + 8
        cnt = T(f_small, f"{len(state.clips)} SOUNDS", CLR["faint"])
        screen.blit(cnt, (G_RIGHT - cnt.get_width(),
                          STRIP_Y + (STRIP_H - cnt.get_height()) // 2))

        # ------------------------------------------------ right pane: the grid
        grid_target = max(0.0, min(grid_target,
                                   max(0.0, grid_content_h - GRID_RECT.height)))
        grid_scroll = step(grid_scroll, grid_target, dt, 0.14)
        screen.set_clip(GRID_RECT)
        grid_hit.clear()
        playing = {}
        sources = [state.voices]
        if getattr(board, "player", None) is not None:
            sources.append(board.player.voices)
        for src in sources:
            for v in list(src):
                try:
                    samples, cur = v
                except Exception:
                    continue
                pidx = clip_by_id.get(id(samples))
                if pidx is not None and len(samples):
                    playing[pidx] = max(playing.get(pidx, 0.0), cur / len(samples))

        if not state.clips:
            hint = T(f_small, "(no sounds - put audio files in ./sounds)",
                     CLR["faint"])
            screen.blit(hint, (G_X, GRID_TOP + 8))
        gy0 = GRID_TOP - int(grid_scroll)
        first_row = max(0, int(grid_scroll) // (TILE_H + GGAP))
        last_row = min(grid_rows,
                       (int(grid_scroll) + GRID_RECT.height) // (TILE_H + GGAP) + 2)
        for ci in range(first_row * COLS, min(len(state.clips), last_row * COLS)):
            g_r, g_c = divmod(ci, COLS)
            r = pygame.Rect(G_X + g_c * (TILE_W + GGAP),
                            gy0 + g_r * (TILE_H + GGAP), TILE_W, TILE_H)
            fl = board.flash.get(ci, 0) - now
            f = q8(min(1.0, max(0.0, fl / 0.25)) ** 2) if fl > 0 else 0.0
            hm = q8(hover_step(("tile", ci),
                               r.collidepoint(mouse_pos)
                               and GRID_RECT.collidepoint(mouse_pos), dt))
            prog = playing.get(ci)
            if f > 0.05:                               # trigger flash + glow
                gsurf = glow(TILE_W, TILE_H, CLR["accent"], 8)
                gsurf.set_alpha(int(255 * f))
                screen.blit(gsurf, (r.x - G_PAD, r.y - G_PAD))
                gsurf.set_alpha(255)
            top = mixc(mixc(CLR["raisedTop"], CLR["hoverTop"], hm), CLR["accentDim"], f)
            bot = mixc(mixc(CLR["raisedBot"], CLR["hoverBot"], hm), CLR["accentDim"], f)
            screen.blit(grad(TILE_W, TILE_H, top, bot, 8), r.topleft)
            bcol = mixc(CLR["stroke"], CLR["accentBright"], f)
            if prog is not None and f < 0.05:
                bcol = mixc(CLR["raisedTop"], CLR["accent"], 0.45)
            elif hm > 0.4 and f < 0.05:
                bcol = CLR["strokeHover"]
            pygame.draw.rect(screen, bcol, r, width=1, border_radius=8)
            ns = T(f_tile, disp_names[ci], CLR["text"])
            screen.blit(ns, (r.x + 10, r.y + 8))
            if prog is not None:                       # playing: ▶ + progress edge
                ds = T(f_small, clip_secs[ci], CLR["accent"])
                dy = r.bottom - 10 - ds.get_height()
                py_ = dy + ds.get_height() // 2
                pygame.draw.polygon(screen, CLR["accent"],
                                    [(r.x + 10, py_ - 3), (r.x + 10, py_ + 3),
                                     (r.x + 15, py_)])
                screen.blit(ds, (r.x + 19, dy))
                pygame.draw.rect(screen, CLR["accent"],
                                 pygame.Rect(r.x, r.bottom - 2,
                                             max(2, int(TILE_W * min(1.0, prog))), 2))
            else:
                ds = T(f_small, clip_secs[ci],
                       CLR["muted"] if hm > 0.5 else CLR["faint"])
                screen.blit(ds, (r.x + 10, r.bottom - 10 - ds.get_height()))
            if ci < 9:                                 # hotkey badge
                hot = prog is not None or f > 0.05
                brect = pygame.Rect(r.right - 10 - 16, r.y + 8, 16, 16)
                pygame.draw.rect(screen,
                                 mixc(CLR["bg"], CLR["accent"], 0.45) if hot
                                 else CLR["strokeHover"],
                                 brect, width=1, border_radius=4)
                bs = T(f_badge, str(ci + 1),
                       CLR["accent"] if hot else CLR["muted"])
                screen.blit(bs, (brect.centerx - bs.get_width() // 2,
                                 brect.centery - bs.get_height() // 2))
            grid_hit[ci] = r

        screen.blit(grad(GRID_RECT.width - 8, 26, (11, 13, 16, 0), (11, 13, 16, 255)),
                    (GRID_RECT.x, GRID_RECT.bottom - 26))
        if grid_content_h > GRID_RECT.height:
            track = pygame.Rect(WIN_W - 17, GRID_TOP + 4, 3,
                                GRID_RECT.height - 8)
            pygame.draw.rect(screen, CLR["scrollTrack"], track, border_radius=2)
            th = max(24, int(track.height * GRID_RECT.height / grid_content_h))
            tt_y = track.y + int((track.height - th)
                                 * (grid_scroll / max(1.0, grid_content_h
                                                      - GRID_RECT.height)))
            pygame.draw.rect(screen, CLR["scrollThumb"],
                             pygame.Rect(track.x, tt_y, 3, th), border_radius=2)
        screen.set_clip(None)

        # ------------------------------------------------- right pane: TTS panel
        pygame.draw.line(screen, CLR["strokeSoft"], (LEFT_W + 1, TTS_TOP),
                         (WIN_W, TTS_TOP))
        tts_btn_hit.clear()
        hs = TT(f_hdr, "TEXT TO SPEECH", CLR["faint"], 2)
        screen.blit(hs, (G_X + 4, TTS_TOP + 10))
        # FX chip: same state as the "TTS voice FX" menu row; reads AI while
        # the phrase would come out in the AI voice
        fx_on = state.tts_fx
        ai_live = ai is not None and ai.proc is not None
        fx_lab = ("FX: AI" if fx_on and ai_live else
                  "FX: ON" if fx_on else "FX: OFF")
        fs = T(f_strip, fx_lab, CLR["accent"] if fx_on else CLR["muted"])
        fxr = pygame.Rect(G_RIGHT - fs.get_width() - 20 - (10 if fx_on else 0),
                          TTS_TOP + 6, fs.get_width() + 20 + (10 if fx_on else 0), 22)
        hm = q8(hover_step(("tts", "fx"), fxr.collidepoint(mouse_pos), dt))
        if fx_on:
            screen.blit(grad(fxr.w, fxr.h, ACCENT_TINT[0], ACCENT_TINT[1], 6),
                        fxr.topleft)
            pygame.draw.rect(screen, mixc(CLR["bg"], CLR["accent"], 0.45), fxr,
                             width=1, border_radius=6)
        else:
            screen.blit(grad(fxr.w, fxr.h,
                             mixc(CLR["raisedTop"], CLR["hoverTop"], hm),
                             mixc(CLR["raisedBot"], CLR["hoverBot"], hm), 6),
                        fxr.topleft)
            pygame.draw.rect(screen, mixc(CLR["stroke"], CLR["strokeHover"], hm),
                             fxr, width=1, border_radius=6)
        fx_x = fxr.x + 10
        if fx_on:
            pygame.draw.circle(screen, CLR["accent"], (fx_x + 2, fxr.centery), 2)
            fx_x += 10
        screen.blit(fs, (fx_x, fxr.centery - fs.get_height() // 2))
        tts_btn_hit["fx"] = fxr
        lyy = TTS_TOP + 10 + hs.get_height() // 2
        pygame.draw.line(screen, CLR["strokeSoft"],
                         (G_X + 4 + hs.get_width() + 8, lyy), (fxr.x - 10, lyy))

        # input box + ADD button
        in_rect = pygame.Rect(G_X, TTS_IN_Y, G_RIGHT - G_X - 66, TTS_IN_H)
        add_rect = pygame.Rect(in_rect.right + 8, TTS_IN_Y,
                               G_RIGHT - in_rect.right - 8, TTS_IN_H)
        tts_btn_hit["input"] = in_rect
        tts_btn_hit["add"] = add_rect
        hm = q8(hover_step(("tts", "input"), in_rect.collidepoint(mouse_pos), dt))
        screen.blit(grad(in_rect.w, in_rect.h, CLR["headerBot"], CLR["paneLeft"], 7),
                    in_rect.topleft)
        pygame.draw.rect(screen,
                         CLR["accent"] if tts_focus
                         else mixc(CLR["stroke"], CLR["strokeHover"], hm),
                         in_rect, width=1, border_radius=7)
        screen.set_clip(in_rect.inflate(-8, 0))
        icy = in_rect.centery
        if tts_text:
            tsurf = T(f_val, tts_text, CLR["text"])
            tx0 = in_rect.x + 10 + min(0, in_rect.w - 20 - tsurf.get_width())
            screen.blit(tsurf, (tx0, icy - tsurf.get_height() // 2))
            caret_x = tx0 + tsurf.get_width() + 2
        else:
            if not tts_focus:
                ph = T(f_val, "Type a phrase, Enter to save...", CLR["faint"])
                screen.blit(ph, (in_rect.x + 10, icy - ph.get_height() // 2))
            caret_x = in_rect.x + 10
        if tts_focus and (now * 2.0) % 2 < 1:      # blinking caret
            pygame.draw.line(screen, CLR["accent"],
                             (caret_x, icy - 8), (caret_x, icy + 8))
        screen.set_clip(None)
        can_add = bool(tts_text.strip())
        hm = q8(hover_step(("tts", "add"), add_rect.collidepoint(mouse_pos), dt))
        if can_add:
            pygame.draw.rect(screen, mixc(CLR["accent"], CLR["accentBright"], hm),
                             add_rect, border_radius=7)
            asurf = T(f_strip, "ADD", CLR["textOnAccent"])
        else:
            screen.blit(grad(add_rect.w, add_rect.h,
                             mixc(CLR["raisedTop"], CLR["hoverTop"], hm),
                             mixc(CLR["raisedBot"], CLR["hoverBot"], hm), 7),
                        add_rect.topleft)
            pygame.draw.rect(screen, mixc(CLR["stroke"], CLR["strokeHover"], hm),
                             add_rect, width=1, border_radius=7)
            asurf = T(f_strip, "ADD", CLR["muted"])
        screen.blit(asurf, (add_rect.centerx - asurf.get_width() // 2,
                            add_rect.centery - asurf.get_height() // 2))

        # phrase list (scrollable; click = speak, x = delete)
        n_ph = len(tts.phrases)
        tts_content_h = (n_ph * (TTS_ROW_H + TTS_ROW_GAP) - TTS_ROW_GAP + 8
                         if n_ph else 0)
        tts_target = max(0.0, min(tts_target,
                                  max(0.0, tts_content_h - TTS_LIST_RECT.height)))
        tts_scroll = step(tts_scroll, tts_target, dt, 0.14)
        screen.set_clip(TTS_LIST_RECT)
        tts_row_hit.clear()
        tts_del_hit.clear()
        tts_playing = {}                   # row -> progress of speaking phrases
        sample_row = {id(tts.samples[t]): i for i, t in enumerate(tts.phrases)
                      if t in tts.samples}
        for v in list(state.tts_voices):
            try:
                samples, cur, _fx = v
            except Exception:
                continue
            ri = sample_row.get(id(samples))
            if ri is not None and len(samples):
                tts_playing[ri] = max(tts_playing.get(ri, 0.0), cur / len(samples))
        if not n_ph:
            hint = T(f_small, "(no phrases - type one above and press Enter)",
                     CLR["faint"])
            screen.blit(hint, (G_X, TTS_LIST_TOP + 8))
        for i in range(n_ph):
            ry = TTS_LIST_TOP - int(tts_scroll) + i * (TTS_ROW_H + TTS_ROW_GAP)
            if ry + TTS_ROW_H < TTS_LIST_RECT.y or ry > VIEW_BOT:
                continue
            text = tts.phrases[i]
            r = pygame.Rect(G_X, ry, G_RIGHT - G_X, TTS_ROW_H)
            fl = tts.flash.get(i, 0) - now
            f = q8(min(1.0, max(0.0, fl / 0.25)) ** 2) if fl > 0 else 0.0
            hm = q8(hover_step(("ttsrow", i),
                               r.collidepoint(mouse_pos)
                               and TTS_LIST_RECT.collidepoint(mouse_pos), dt))
            screen.blit(grad(r.w, TTS_ROW_H,
                             mixc(mixc(CLR["raisedTop"], CLR["hoverTop"], hm),
                                  CLR["accentDim"], f),
                             mixc(mixc(CLR["raisedBot"], CLR["hoverBot"], hm),
                                  CLR["accentDim"], f), 7),
                        r.topleft)
            prog = tts_playing.get(i)
            bcol = mixc(CLR["stroke"], CLR["accentBright"], f)
            if prog is not None and f < 0.05:
                bcol = mixc(CLR["raisedTop"], CLR["accent"], 0.45)
            elif hm > 0.4 and f < 0.05:
                bcol = CLR["strokeHover"]
            pygame.draw.rect(screen, bcol, r, width=1, border_radius=7)
            dr = pygame.Rect(r.right - 8 - 18, r.centery - 9, 18, 18)
            dh = q8(hover_step(("ttsdel", i), dr.collidepoint(mouse_pos), dt))
            pygame.draw.rect(screen, mixc(CLR["strokeHover"], CLR["danger"], dh),
                             dr, width=1, border_radius=5)
            xs = T(f_badge, "x", CLR["danger"] if dh > 0.4 else CLR["muted"])
            screen.blit(xs, (dr.centerx - xs.get_width() // 2,
                             dr.centery - xs.get_height() // 2))
            tts_del_hit[i] = dr
            st = tts.status.get(text, "")
            if st == "ready" and text in tts.samples:
                dur = T(f_small, f"{len(tts.samples[text]) / SAMPLERATE:.1f}s",
                        CLR["accent"] if prog is not None else CLR["faint"])
            elif st == "error":
                dur = T(f_small, "err", CLR["danger"])
            else:                          # synthesizing: pulse like "loading"
                a = 0.4 + 0.6 * (0.5 + 0.5 * float(np.sin(now * 2 * np.pi / 1.2)))
                dur = T(f_small, "...", mixc(CLR["raisedBot"], CLR["muted"], q8(a)))
            screen.blit(dur, (dr.x - 8 - dur.get_width(),
                              r.centery - dur.get_height() // 2))
            nm = tts_trunc.get(text)
            if nm is None:
                if len(tts_trunc) > 400:
                    tts_trunc.clear()
                nm, name_w = text, r.w - 104
                if f_label.render(nm, True, CLR["text"]).get_width() > name_w:
                    while nm and f_label.render(nm + "...", True,
                                                CLR["text"]).get_width() > name_w:
                        nm = nm[:-1]
                    nm += "..."
                tts_trunc[text] = nm
            hot = prog is not None or f > 0.05
            pygame.draw.polygon(screen,
                                CLR["accent"] if hot else
                                (CLR["muted"] if hm > 0.4 else CLR["faint"]),
                                [(r.x + 12, r.centery - 4),
                                 (r.x + 12, r.centery + 4), (r.x + 18, r.centery)])
            ns = T(f_label, nm,
                   CLR["text"] if (hot or hm > 0.4) else CLR["text2"])
            screen.blit(ns, (r.x + 26, r.centery - ns.get_height() // 2))
            if prog is not None:           # speaking: progress along bottom edge
                pygame.draw.rect(screen, CLR["accent"],
                                 pygame.Rect(r.x, r.bottom - 2,
                                             max(2, int(r.w * min(1.0, prog))), 2))
            tts_row_hit[i] = r
        screen.blit(grad(TTS_LIST_RECT.width - 8, 20, (11, 13, 16, 0),
                         (11, 13, 16, 255)),
                    (TTS_LIST_RECT.x, VIEW_BOT - 20))
        if tts_content_h > TTS_LIST_RECT.height:
            track = pygame.Rect(WIN_W - 17, TTS_LIST_TOP + 2, 3,
                                TTS_LIST_RECT.height - 4)
            pygame.draw.rect(screen, CLR["scrollTrack"], track, border_radius=2)
            th = max(18, int(track.height * TTS_LIST_RECT.height / tts_content_h))
            tt_y = track.y + int((track.height - th)
                                 * (tts_scroll / max(1.0, tts_content_h
                                                     - TTS_LIST_RECT.height)))
            pygame.draw.rect(screen, CLR["scrollThumb"],
                             pygame.Rect(track.x, tt_y, 3, th), border_radius=2)
        screen.set_clip(None)

        # ------------------------------------------------------------ footer
        screen.blit(grad(WIN_W, FOOTER_H, CLR["footerTop"], CLR["footerBot"]),
                    (0, VIEW_BOT))
        pygame.draw.line(screen, CLR["strokeSoft"], (0, VIEW_BOT),
                         (WIN_W, VIEW_BOT))
        fy = VIEW_BOT + FOOTER_H // 2
        if err_line:
            es = T(f_foot, err_line, CLR["danger"])
            screen.blit(es, (14, fy - es.get_height() // 2))
        elif dev_line:
            fx = 14
            if "->" in dev_line:
                a_, b_ = dev_line.split("->", 1)
                parts = ((a_.strip(), CLR["muted"]), (" → ", CLR["accent"]),
                         (b_.strip(), CLR["muted"]))
            else:
                parts = ((dev_line, CLR["muted"]),)
            for ptxt, pcol in parts:
                psur = T(f_foot, ptxt, pcol)
                screen.blit(psur, (fx, fy - psur.get_height() // 2))
                fx += psur.get_width()

        # status toast chip: in 160ms / hold 4s / out 220ms
        if state.status_msg:
            t_ = now - state.status_at
            alpha = (t_ / 0.16 if t_ < 0.16 else
                     1.0 if t_ < 4.16 else
                     max(0.0, 1.0 - (t_ - 4.16) / 0.22) if t_ < 4.38 else 0.0)
            if alpha > 0:
                col = (CLR["danger"] if "error" in state.status_msg.lower()
                       else CLR["warning"])
                cs = T(f_foot, f"{state.status_msg} (x{state.status_count})", col)
                cw = cs.get_width() + 30
                chip = pygame.Surface((cw, 20), pygame.SRCALPHA)
                pygame.draw.rect(chip, (*col, 26), chip.get_rect(), border_radius=6)
                pygame.draw.rect(chip, (*col, 102), chip.get_rect(), width=1,
                                 border_radius=6)
                pygame.draw.circle(chip, col, (11, 10), 2)
                chip.blit(cs, (18, (20 - cs.get_height()) // 2))
                chip.set_alpha(int(255 * alpha))
                rise = int(8 * (1.0 - min(1.0, t_ / 0.16)))
                screen.blit(chip, (WIN_W - 14 - cw, fy - 10 + rise))

        # --------------------------------------------- dropdown picker overlay
        if drop is not None:
            r = drop["rect"]
            if drop["mouse"] != mouse_pos and r.collidepoint(mouse_pos):
                mi = (mouse_pos[1] - r.y - drop["pad"]
                      + int(drop["scroll"])) // drop["item_h"]
                if 0 <= mi < len(drop["items"]):
                    drop["sel"] = mi
            drop["mouse"] = mouse_pos
            screen.blit(grad(r.w, r.h, CLR["hoverTop"], CLR["raisedBot"], 8),
                        r.topleft)
            pygame.draw.rect(screen, mixc(CLR["stroke"], CLR["accent"], 0.35),
                             r, width=1, border_radius=8)
            screen.set_clip(r.inflate(-2, -4))
            y0 = r.y + drop["pad"] - int(drop["scroll"])
            for i, (nm, _cb) in enumerate(drop["items"]):
                ir = pygame.Rect(r.x + 4, y0 + i * drop["item_h"],
                                 r.w - 12, drop["item_h"] - 2)
                if ir.bottom < r.y or ir.y > r.bottom:
                    continue
                if i == drop["sel"]:
                    screen.blit(grad(ir.w, ir.h, ACCENT_TINT[0],
                                     ACCENT_TINT[1], 6), ir.topleft)
                    pygame.draw.rect(screen, CLR["accent"], ir,
                                     width=1, border_radius=6)
                if i == drop["cur"]:           # the currently active entry
                    pygame.draw.circle(screen, CLR["accent"],
                                       (ir.x + 11, ir.centery), 2)
                ns = T(f_labelF if i == drop["sel"] else f_label, nm,
                       CLR["text"] if i == drop["sel"] else CLR["text2"])
                screen.blit(ns, (ir.x + 22, ir.centery - ns.get_height() // 2))
            if drop["max_scroll"] > 0:
                track = pygame.Rect(r.right - 6, r.y + 4, 3, r.h - 8)
                pygame.draw.rect(screen, CLR["scrollTrack"], track,
                                 border_radius=2)
                th = max(18, int(track.h * r.h / (drop["max_scroll"] + r.h)))
                ty = track.y + int((track.h - th)
                                   * (drop["scroll"] / drop["max_scroll"]))
                pygame.draw.rect(screen, CLR["scrollThumb"],
                                 pygame.Rect(track.x, ty, 3, th),
                                 border_radius=2)
            screen.set_clip(None)

        pygame.display.flip()
        clock.tick(30)          # 30 fps is plenty for a menu and halves GIL load

    pygame.quit()

# ----------------------------------------------------------------------------- UTIL
def find_device(match, kind):
    """kind: 'input' or 'output'. Returns device index or None (=default)."""
    if match is None:
        return None
    if isinstance(match, int):
        return match
    key = ("max_input_channels" if kind == "input" else "max_output_channels")
    for i, d in enumerate(sd.query_devices()):
        if match.lower() in d["name"].lower() and d[key] > 0:
            return i
    raise SystemExit(f"Could not find {kind} device matching '{match}'. Run --list.")


# ----------------------------------------------------------------------------- MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list audio devices and exit")
    args = ap.parse_args()
    if args.list:
        print(sd.query_devices()); return

    state = State()
    stop_flag = threading.Event()
    err_line = ""
    try:
        in_dev = find_device(INPUT_DEVICE_MATCH, "input")
        out_dev = find_device(OUTPUT_DEVICE_MATCH, "output")
        in_name = sd.query_devices(in_dev)["name"] if in_dev is not None else "default mic"
        out_name = sd.query_devices(out_dev)["name"] if out_dev is not None else "default out"
        dev_line = f"{in_name}  ->  {out_name}   (Discord input: CABLE Output)"
        # latency="high" buys buffering headroom: Python-side hiccups (GC, UI
        # thread holding the GIL) then cause no dropouts. Adds ~20 ms - fine
        # for voice chat, and far better than cutting out.
        stream = sd.Stream(samplerate=SAMPLERATE, blocksize=BLOCKSIZE, dtype="float32",
                           channels=CHANNELS, device=(in_dev, out_dev),
                           latency="high", callback=make_callback(state))
    except (SystemExit, Exception) as e:      # UI still opens so the error is visible
        dev_line = ""
        err_line = f"audio unavailable: {e}"
        stream = None

    monitor = Monitor(state, has_main_stream=stream is not None)
    player = LocalPlayer(state)
    board = Board(state, player, monitor)
    ai = AiVoice(state, monitor=monitor)
    tts = TTSBank(state, player, monitor, ai)
    tts.warm()                    # synthesize saved phrases in the background
    try:
        if stream:
            with stream:
                run_ui(state, stop_flag, dev_line, err_line, monitor, board,
                       ai, tts)
        else:
            run_ui(state, stop_flag, dev_line, err_line, monitor, board,
                   ai, tts)
    except KeyboardInterrupt:
        pass
    finally:
        ai.stop()
        monitor.close()
        player.close()
    print("stopped.")


if __name__ == "__main__":
    main()
