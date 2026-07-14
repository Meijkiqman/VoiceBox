"""Short self-heard tones for state changes you can't see while in-game:
the AI voice finishing its ~10 s load (or dying), and mic mute toggles.
Played on the local speakers only - never into the cable."""
import numpy as np

from .config import SAMPLERATE

def _tone(freqs, ms=90, amp=0.22, gap_ms=30):
    """A sequence of sine blips with click-free edges, as one buffer."""
    fade = int(SAMPLERATE * 0.005)
    gap = np.zeros(int(SAMPLERATE * gap_ms / 1000), dtype=np.float32)
    out = []
    for k, f in enumerate(freqs):
        n = int(SAMPLERATE * ms / 1000)
        t = np.arange(n) / SAMPLERATE
        y = (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)
        y[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        y[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        if k:
            out.append(gap)
        out.append(y)
    return np.concatenate(out)


class Cues:
    """Fires the blips through the LocalPlayer (speakers only). Gated by
    the "Sound cues" menu row (state.cues_on); mute cues fire on manual
    toggles only, never on push-to-talk press/release."""

    def __init__(self, state, player):
        self.state = state
        self.player = player
        self.ready = _tone((660, 880))           # AI voice loaded: rising
        self.died = _tone((440, 220), ms=120)    # AI voice died: falling
        self.muted = _tone((300,), ms=70)
        self.live = _tone((600,), ms=70)

    def _play(self, samples):
        if self.state.cues_on and self.player is not None:
            self.player.play_raw(samples)

    def ai_ready(self):
        self._play(self.ready)

    def ai_died(self):
        self._play(self.died)

    def mute(self, muted):
        self._play(self.muted if muted else self.live)
