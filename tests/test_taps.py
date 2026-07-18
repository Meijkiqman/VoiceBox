"""Audio-callback wiring for the translator/harvester taps: raw-mic
mirrors, mute semantics, and the trans_hold voice gate."""
import queue

from _common import check, finish

import numpy as np
import voicebox

voicebox.soundboard.load_clips = lambda: ([], [])

from voicebox.audio import make_callback
from voicebox.state import State

FRAMES = 512


def run_block(state, level=0.25):
    indata = np.full((FRAMES, 1), level, dtype=np.float32)
    outdata = np.zeros((FRAMES, 1), dtype=np.float32)
    cb(indata, outdata, FRAMES, None, None)
    return outdata[:, 0]


state = State()
cb = make_callback(state)

# baseline: mic passes through (pitch 0 = shifter bypass)
out = run_block(state)
check("mic passes through", float(np.abs(out).max()) > 0.2)

# translator tap mirrors the raw mic while set
tap = queue.Queue()
state.trans_tap = tap
run_block(state)
check("trans tap fed", tap.qsize() == 1)
block = tap.get_nowait()
check("tap is raw mic", float(block.max()) == np.float32(0.25))
state.trans_tap = None
run_block(state)
check("tap stops when cleared", tap.qsize() == 0)

# trans_hold silences the outgoing voice...
state.trans_hold = True
out = run_block(state)
check("hold silences voice", float(np.abs(out).max()) == 0.0)

# ...but fx-tagged TTS still rides the (bypassed) chain while held
state.events.put(("tts", np.full(FRAMES, 0.3, np.float32), True))
out = run_block(state)
check("tts rides chain while held", float(np.abs(out).max()) > 0.2)
state.trans_hold = False

# harvester mirror obeys mute: muted speech never reaches the dataset
hq = queue.Queue()
state.harvest_q = hq
run_block(state)
check("harvest fed while live", hq.qsize() == 1)
hq.get_nowait()
state.mic_muted = True
run_block(state)
check("harvest skipped while muted", hq.qsize() == 0)
state.mic_muted = False
state.harvest_q = None

finish()
