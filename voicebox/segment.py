"""Silence-based utterance segmentation, shared by the voice harvester and
the incoming-speech listener (they used to carry two copies of it).

Feed raw blocks in order. A segment opens when a block's peak crosses the
threshold - keeping a little pre-roll so the first consonant survives -
stays open through trailing silence, and closes after `hang_s` of quiet or
at `max_s`, coming back from feed() as one array. What "too short to keep"
means is the caller's call (clip quality vs. caption noise), so no minimum
length lives here."""
import numpy as np

from .config import SAMPLERATE


class Segmenter:
    def __init__(self, thresh_db, pre_s, hang_s, max_s, sr=SAMPLERATE):
        self.thresh = 10.0 ** (thresh_db / 20.0)
        self.pre_max = int(pre_s * sr)
        self.hang_s = hang_s
        self.max_s = max_s
        self.sr = sr
        self.pre, self.pre_len = [], 0     # rolling pre-roll blocks
        self.seg, self.seg_len = None, 0   # blocks while inside speech
        self.quiet = 0.0                   # trailing silence inside seg, s

    def feed(self, block):
        """One raw block in -> a finished utterance (array) or None."""
        speaking = bool(len(block)) and float(np.abs(block).max()) >= self.thresh
        if self.seg is None:
            if speaking:
                self.seg = self.pre + [block]
                self.seg_len = self.pre_len + len(block)
                self.pre, self.pre_len, self.quiet = [], 0, 0.0
            else:
                self.pre.append(block)
                self.pre_len += len(block)
                while self.pre and self.pre_len - len(self.pre[0]) >= self.pre_max:
                    self.pre_len -= len(self.pre[0])
                    self.pre.pop(0)
            return None
        self.seg.append(block)
        self.seg_len += len(block)
        self.quiet = 0.0 if speaking else self.quiet + len(block) / self.sr
        if self.quiet >= self.hang_s or self.seg_len / self.sr >= self.max_s:
            return self.flush()
        return None

    def flush(self):
        """Close an open segment (a stop() mid-sentence hands back what was
        being built) -> array or None when nothing was open."""
        if self.seg is None:
            return None
        utt = np.concatenate(self.seg)
        self.pre, self.pre_len = [], 0
        self.seg, self.seg_len, self.quiet = None, 0, 0.0
        return utt
