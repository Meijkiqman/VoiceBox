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
Stop clears them. Keys 1-9 trigger the first nine sounds.

EFFECTS & PRESETS
-----------------
Pitch, robot/vocoder mix, helmet doubler, grit, reverb, echo, radio band-pass
and bass boost are individual menu rows. The Preset row applies curated
combinations (Space Marine, Ghost, ...) which can be tweaked freely afterwards
- the row shows "Custom" once any value diverges from the applied preset.
"Test - hear myself" mirrors the processed mix to your speakers.

Defaults:  arrows/WASD or d-pad/left stick = navigate,  Enter/Space or A =
select,  left/right adjusts values,  1-9 = play clip,  0/Backspace or Y =
stop clips,  Esc or B = quit.
"""

import argparse
import json
import queue
import subprocess
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

SAMPLERATE = 48000        # VB-CABLE runs at 48k by default
BLOCKSIZE  = 512          # smaller = lower latency, larger = safer. 256-1024 typical
CHANNELS   = 1            # mono processing path

# Device selection. Substrings are matched against device names (case-insensitive).
# Use --list to see names. Set to an int to force a specific device index instead.
INPUT_DEVICE_MATCH   = None            # None = system default mic, or e.g. "Microphone"
OUTPUT_DEVICE_MATCH  = "CABLE Input"   # the virtual cable's INPUT side

WINDOW_SIZE = (960, 600)   # left: settings menu, right: soundboard grid
MAX_CLIPS   = 64           # how many files from ./sounds get indexed

# AI voice (RVC) integration. Point RVC_DIR at an extracted RVC-beta package
# (must contain runtime\python.exe, weights\*.pth, hubert_base.pt, rmvpe.pt).
# The AI rows only appear in the menu when this folder and at least one voice
# model exist, so machines without RVC are unaffected.
RVC_DIR = Path(r"H:\Python_Projects\DeepSpeech"
               r"\Retrieval-based-Voice-Conversion-WebUI-main\RVC-beta0717")

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
        self.ai_mute = False          # AI worker owns the voice; mute ours
        self.shifter = StreamingPitchShifter(SAMPLERATE, 0.0)  # audio thread only
        self.reverb_fx = Reverb()     # effect state: audio thread only
        self.echo_fx = Echo()
        self.radio_fx = Radio()
        self.doubler_fx = Doubler()
        self.bass_fx = BassBoost()
        self.clips, self.clip_names = load_clips()
        self.voices = []              # list of [samples, cursor]; audio thread only
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
            ai_mute = state.ai_mute

        # apply queued UI events (audio thread owns the shifter and voice list)
        while not state.events.empty():
            ev = state.events.get_nowait()
            if ev == "stop":
                state.voices.clear()
            elif isinstance(ev, tuple) and ev[0] == "pitch":
                state.shifter.set_semitones(ev[1])
            elif isinstance(ev, int) and 0 <= ev < len(state.clips):
                state.voices.append([state.clips[ev], 0])

        if ai_mute:
            # the RVC worker feeds the converted voice into the cable itself;
            # our own voice path stays silent so it isn't heard doubled
            y = np.zeros(frames, dtype=np.float32)
            carry = np.zeros(0, dtype=np.float32)
        else:
            y = state.shifter.process(indata[:, 0].astype(np.float32) * voice_gain)

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
    """Self-listen ("Test - hear myself"). While the main stream is running it
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

    def __init__(self, state, rvc_dir=None):
        self.state = state
        self.rvc_dir = Path(rvc_dir) if rvc_dir else RVC_DIR
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
        if not self.voices:
            return
        self.sel = (self.sel + d) % len(self.voices)
        if self.proc is not None:          # live switch: restart on new voice
            self.stop()
            self.start()

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
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=str(self.rvc_dir), text=True,
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
    def __init__(self, label, value_fn=None, select=None, adjust=None, flash=True):
        self.label = label
        self.value_fn = value_fn      # () -> str shown on the right
        self.select = select          # on_select handler
        self.adjust = adjust          # on_left/on_right handler, adjust(delta)
        self.flash = flash            # flash the row on select (off for Quit)


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
                     adjust=lambda d: s.set_pitch(s.semitones + d)),
            MenuItem("Robot voice",
                     lambda: f"{s.robot:.0%}" if s.robot else "off",
                     select=self._toggle_robot,
                     adjust=lambda d: s.nudge("robot", d * 0.05, hi=1.0)),
            MenuItem("Helmet doubler",
                     lambda: f"{s.doubler:.0%}" if s.doubler else "off",
                     adjust=lambda d: s.nudge("doubler", d * 0.05, hi=1.0)),
            MenuItem("Grit / growl",
                     lambda: f"{s.drive:.0%}",
                     adjust=lambda d: s.nudge("drive", d * 0.05, hi=1.0)),
            MenuItem("Reverb",
                     lambda: f"{s.reverb:.0%}",
                     adjust=lambda d: s.nudge("reverb", d * 0.05, hi=1.0)),
            MenuItem("Echo",
                     lambda: f"{s.echo:.0%}",
                     adjust=lambda d: s.nudge("echo", d * 0.05, hi=1.0)),
            MenuItem("Radio voice",
                     lambda: "ON" if s.radio else "off",
                     select=self._toggle_radio,
                     adjust=lambda d: self._toggle_radio()),
            MenuItem("Bass boost",
                     lambda: f"{s.bass:.0%}" if s.bass else "off",
                     adjust=lambda d: s.nudge("bass", d * 0.05, hi=1.0)),
            MenuItem("Voice volume",
                     lambda: f"{s.voice_gain:.0%}",
                     adjust=lambda d: s.nudge("voice_gain", d * 0.05)),
            MenuItem("Clip volume",
                     lambda: f"{s.clip_gain:.0%}",
                     adjust=lambda d: s.nudge("clip_gain", d * 0.05)),
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
        if monitor is not None:
            self.items.append(MenuItem(
                "Test - hear myself",
                lambda: "ON" if monitor.on else "off",
                select=self._toggle_monitor,
                adjust=lambda d: self._toggle_monitor()))
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

    def _toggle_monitor(self):
        self.monitor.toggle()
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


def run_ui(state, stop_flag, dev_line, err_line="", monitor=None, board=None, ai=None):
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
    screen = pygame.display.set_mode(WINDOW_SIZE)
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
    VIEW_TOP, VIEW_BOT = HEADER_H, WINDOW_SIZE[1] - FOOTER_H
    VIEW_H = VIEW_BOT - VIEW_TOP
    L_X, L_RIGHT = 10, LEFT_W - 14
    L_W = L_RIGHT - L_X
    ROW_HGT, ROW_GAP = 34, 2
    HDR_FIRST, HDR_HGT = 24, 30
    LIST_PAD_TOP = 6
    G_X = LEFT_W + 14
    G_RIGHT = WINDOW_SIZE[0] - 14 - 8          # 8px scroll gutter
    COLS, GGAP, TILE_H = 3, 8, 62
    TILE_W = (G_RIGHT - G_X - GGAP * (COLS - 1)) // COLS
    STRIP_Y, STRIP_H = VIEW_TOP + 10, 30
    GRID_TOP = STRIP_Y + STRIP_H + 10
    LIST_RECT = pygame.Rect(0, VIEW_TOP, LEFT_W, VIEW_H)
    GRID_RECT = pygame.Rect(LEFT_W + 1, GRID_TOP, WINDOW_SIZE[0] - LEFT_W - 1,
                            VIEW_BOT - GRID_TOP)

    SECTION_OF = {
        "Preset": "VOICE", "Pitch": "VOICE",
        "Robot voice": "EFFECTS", "Helmet doubler": "EFFECTS",
        "Grit / growl": "EFFECTS", "Reverb": "EFFECTS", "Echo": "EFFECTS",
        "Radio voice": "EFFECTS", "Bass boost": "EFFECTS",
        "Voice volume": "EFFECTS", "Clip volume": "EFFECTS",
        "AI voice": "AI", "AI character": "AI",
        "Sounds to mic": "SOUNDS", "Pause sounds": "SOUNDS",
        "Stop all sounds": "SOUNDS",
        "Test - hear myself": "SYSTEM", "Quit": "SYSTEM",
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

    # grid labels never change at runtime: truncate + measure them once
    disp_names, clip_secs = [], []
    name_max = TILE_W - 20 - 22
    for nm, c in zip(state.clip_names, state.clips):
        if f_tile.render(nm, True, CLR["text"]).get_width() > name_max:
            while nm and f_tile.render(nm + "...", True,
                                       CLR["text"]).get_width() > name_max:
                nm = nm[:-1]
            nm += "..."
        disp_names.append(nm)
        clip_secs.append(f"{len(c) / SAMPLERATE:.1f}s")

    # ----------------------------------------------------------- motion state
    list_scroll = list_target = 0.0
    grid_scroll = grid_target = 0.0
    focus_y = float(row_pos.get(menu.sel, LIST_PAD_TOP))
    hover_mix = {}                # element key -> 0..1 hover blend
    nudge = {"i": -1, "at": 0.0, "side": 0}
    strip_press = {}
    meter_lit = 0.0
    peak_lit, peak_at = 0.0, 0.0
    row_hit, strip_hit, grid_hit = {}, {}, {}
    arrow_hit = None              # (row, "<" rect, ">" rect) from the last draw
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

    def go_left():
        menu.on_left()
        nudge.update(i=menu.sel, at=time.time(), side=-1)

    def go_right():
        menu.on_right()
        nudge.update(i=menu.sel, at=time.time(), side=+1)

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

    def draw_value(r, i, it, val, focused, now):
        nonlocal arrow_hit
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

    # ================================================================== loop
    while not stop_flag.is_set():
        now = time.time()
        dt = min(0.1, now - last_t)
        last_t = now

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_flag.set()

            elif event.type == pygame.JOYDEVICEADDED:
                pygame.joystick.Joystick(event.device_index).init()
            elif event.type == pygame.JOYDEVICEREMOVED:
                pass                                   # instance dies on its own

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
                elif act == "select":     menu.on_select()
                elif act == "back":       menu.on_back()
                elif act == "stop_clips": board.stop()

            elif event.type == pygame.KEYUP:
                held_keys.discard(event.key)
            elif event.type == pygame.WINDOWFOCUSLOST:
                held_keys.clear()         # KEYUPs are lost on focus change

            elif event.type == pygame.MOUSEMOTION:
                idx = row_at(event.pos)
                if idx is not None:
                    menu.sel = idx

            elif event.type == pygame.MOUSEWHEEL:
                if pygame.mouse.get_pos()[0] >= LEFT_W:      # over the grid pane
                    grid_target -= event.y * (TILE_H + GGAP)
                else:
                    for _ in range(abs(event.y)):
                        menu.on_up() if event.y > 0 else menu.on_down()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
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
                elif (ci := next((c for c, r in grid_hit.items()
                                  if r.collidepoint(event.pos)), None)) is not None:
                    board.play(ci)
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
                        elif it.select:
                            menu.on_select()

            elif event.type == pygame.JOYBUTTONDOWN:
                if   event.button in pad_select: menu.on_select()
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
        screen.blit(grad(WINDOW_SIZE[0], HEADER_H, CLR["headerTop"], CLR["headerBot"]),
                    (0, 0))
        pygame.draw.line(screen, CLR["strokeSoft"], (0, HEADER_H - 1),
                         (WINDOW_SIZE[0], HEADER_H - 1))
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
        mx_right = WINDOW_SIZE[0] - 16
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
                    (GRID_RECT.x, VIEW_BOT - 26))
        if grid_content_h > GRID_RECT.height:
            track = pygame.Rect(WINDOW_SIZE[0] - 17, GRID_TOP + 4, 3,
                                GRID_RECT.height - 8)
            pygame.draw.rect(screen, CLR["scrollTrack"], track, border_radius=2)
            th = max(24, int(track.height * GRID_RECT.height / grid_content_h))
            tt_y = track.y + int((track.height - th)
                                 * (grid_scroll / max(1.0, grid_content_h
                                                      - GRID_RECT.height)))
            pygame.draw.rect(screen, CLR["scrollThumb"],
                             pygame.Rect(track.x, tt_y, 3, th), border_radius=2)
        screen.set_clip(None)

        # ------------------------------------------------------------ footer
        screen.blit(grad(WINDOW_SIZE[0], FOOTER_H, CLR["footerTop"], CLR["footerBot"]),
                    (0, VIEW_BOT))
        pygame.draw.line(screen, CLR["strokeSoft"], (0, VIEW_BOT),
                         (WINDOW_SIZE[0], VIEW_BOT))
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
                screen.blit(chip, (WINDOW_SIZE[0] - 14 - cw, fy - 10 + rise))

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
    ai = AiVoice(state)
    try:
        if stream:
            with stream:
                run_ui(state, stop_flag, dev_line, err_line, monitor, board, ai)
        else:
            run_ui(state, stop_flag, dev_line, err_line, monitor, board, ai)
    except KeyboardInterrupt:
        pass
    finally:
        ai.stop()
        monitor.close()
        player.close()
    print("stopped.")


if __name__ == "__main__":
    main()
