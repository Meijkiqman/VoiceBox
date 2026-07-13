"""DSP building blocks: pitch shifter, delay-line effects, filters, gate.
Everything here is stateful-streaming and owned by the audio thread."""
import numpy as np
from scipy.signal import butter, lfilter

from .config import SAMPLERATE

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


class NoiseGate:
    """Noise gate ahead of the effect chain, so room hiss doesn't feed the
    grit/reverb stages (which amplify it audibly). Block-level: opens fast
    when the block peak crosses the threshold, holds briefly so word tails
    survive, then releases slowly. The gain is ramped across each block so
    opening/closing never clicks."""

    def __init__(self, sr=SAMPLERATE, attack=0.005, release=0.120, hold=0.150):
        self.sr = sr
        self.attack, self.release, self.hold = attack, release, hold
        self.gain = 0.0                # start closed
        self.held = 0.0                # seconds of hold left

    def process(self, x, threshold_db):
        n = len(x)
        if not n:
            return x
        dt = n / self.sr
        if float(np.abs(x).max()) >= 10.0 ** (threshold_db / 20.0):
            target, self.held = 1.0, self.hold
        elif self.held > 0.0:
            target, self.held = 1.0, self.held - dt
        else:
            target = 0.0
        tc = self.attack if target > self.gain else self.release
        new_gain = self.gain + (target - self.gain) * min(1.0, dt / max(tc, 1e-6))
        ramp = np.linspace(self.gain, new_gain, n, endpoint=False,
                           dtype=np.float32)
        self.gain = float(new_gain)
        return x * ramp


