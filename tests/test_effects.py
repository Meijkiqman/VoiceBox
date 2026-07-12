"""DSP effects: reverb, echo, radio, doubler, bass boost, robot mix, chunking."""
from _common import check, finish

import numpy as np
import voicebox

voicebox.load_clips = lambda: ([], [])
frames = voicebox.BLOCKSIZE
zeros = np.zeros(frames, dtype=np.float32)
impulse = zeros.copy(); impulse[0] = 1.0

# ------------------------------------------------------------------- reverb
r = voicebox.Reverb()
first = r.process(impulse, 1.0)
check("reverb passes the dry impulse", abs(first[0]) > 0.4)
tail_energy = []
for _ in range(20):
    tail_energy.append(float(np.abs(r.process(zeros, 1.0)).max()))
check("reverb produces a tail", max(tail_energy[:6]) > 0.01, str(tail_energy[:6]))
check("reverb tail decays", tail_energy[-1] < max(tail_energy[:6]))
check("reverb keeps float32", first.dtype == np.float32)

# --------------------------------------------------------------------- echo
e = voicebox.Echo()
out0 = e.process(impulse, 1.0)
check("echo dry-through", out0[0] == 1.0)
spike_block, spike_val = -1, 0.0
for i in range(1, 40):                       # delay 15360 = exactly 30 blocks
    o = e.process(zeros, 1.0)
    if np.abs(o).max() > spike_val:
        spike_val, spike_block = float(np.abs(o).max()), i
check("echo repeats after ~320 ms", spike_block == 30 and spike_val > 0.9,
      f"block {spike_block} val {spike_val:.2f}")

# -------------------------------------------------------------------- radio
rd = voicebox.Radio()
dc = np.full(frames, 0.5, dtype=np.float32)
for _ in range(12):
    out_dc = rd.process(dc)
check("radio blocks DC / rumble", abs(float(out_dc.mean())) < 0.02,
      f"mean {out_dc.mean():.4f}")
rd2 = voicebox.Radio()
t = np.arange(frames * 12) / voicebox.SAMPLERATE
sine1k = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
peak = 0.0
for b in range(12):
    o = rd2.process(sine1k[b * frames:(b + 1) * frames])
    peak = max(peak, float(np.abs(o).max()))
check("radio passes speech band (1 kHz)", peak > 0.3, f"peak {peak:.2f}")
check("radio keeps float32", o.dtype == np.float32)

# ------------------------------------------------------------------ doubler
db = voicebox.Doubler()
scale = 1.0 / 1.5                            # wet=1 level compensation
o0 = db.process(impulse, 1.0)
check("doubler dry-through (scaled)", abs(o0[0] - scale) < 1e-3, f"{o0[0]:.3f}")
o1 = db.process(zeros, 1.0)
check("doubler repeats once at ~12 ms", abs(o1[576 - 512] - scale) < 1e-3,
      f"{o1[576 - 512]:.3f}")
o2 = db.process(zeros, 1.0)
check("doubler has no feedback", float(np.abs(o2).max()) < 1e-6)

# ------------------------------------- chunk-splitting (blocks > delay length)
sig = (0.3 * np.sin(2 * np.pi * 313 * np.arange(2048) / voicebox.SAMPLERATE)
       + 0.2 * np.sin(2 * np.pi * 1777 * np.arange(2048) / voicebox.SAMPLERATE)
       ).astype(np.float32)
d1, d2 = voicebox.Doubler(), voicebox.Doubler()
big = d1.process(sig, 1.0)                   # 2048 > 576-sample delay line
small = np.concatenate([d2.process(sig[i:i + 512], 1.0)
                        for i in range(0, 2048, 512)])
check("doubler exact when block exceeds its delay",
      np.allclose(big, small, atol=1e-6))
c1 = voicebox.CombFilter(1427, 0.8)
c2 = voicebox.CombFilter(1427, 0.8)
big = c1.process(sig)
small = np.concatenate([c2.process(sig[i:i + 512]) for i in range(0, 2048, 512)])
check("comb exact when block exceeds its delay",
      np.allclose(big, small, atol=1e-6))

# ---------------------------------------------------------------- bass boost
bb = voicebox.BassBoost()
t12 = np.arange(frames * 12) / voicebox.SAMPLERATE
low_in = (0.3 * np.sin(2 * np.pi * 60 * t12)).astype(np.float32)
peak_low = 0.0
for i in range(12):
    o = bb.process(low_in[i * frames:(i + 1) * frames], 1.0)
    peak_low = max(peak_low, float(np.abs(o).max()))
check("bass boost lifts lows (~2x at 60 Hz)", peak_low > 0.45, f"{peak_low:.2f}")
bb2 = voicebox.BassBoost()
high_in = (0.3 * np.sin(2 * np.pi * 5000 * t12)).astype(np.float32)
peak_high = 0.0
for i in range(12):
    o = bb2.process(high_in[i * frames:(i + 1) * frames], 1.0)
    peak_high = max(peak_high, float(np.abs(o).max()))
check("bass boost leaves highs alone", peak_high < 0.37, f"{peak_high:.2f}")

# ------------------------------------------------------------ robot as a mix
state = voicebox.State()
cb = voicebox.make_callback(state)
out = np.zeros((frames, 1), dtype=np.float32)
dc2 = np.full((frames, 1), 0.5, dtype=np.float32)
with state.lock:
    state.robot = 1.0
cb(dc2, out, frames, None, None)
check("robot 100%: full ring-mod (signal inverts)", float(out.min()) < -0.3,
      f"min {out.min():.2f}")
with state.lock:
    state.robot = 0.5
cb(dc2, out, frames, None, None)
check("robot 50%: dry half keeps signal positive", float(out.min()) > -0.01,
      f"min {out.min():.2f}")
check("robot 50%: modulation still present",
      float(out.max()) - float(out.min()) > 0.3)
with state.lock:
    state.robot = 0.0

# ------------------------------------------------------------------ presets
state = voicebox.State()
names = [n for n, _ in voicebox.PRESETS]
check("new presets present", "Ghost" in names and "Walkie-Talkie" in names,
      str(names))
state.apply_preset(names.index("Space Marine"))
check("Space Marine restored to classic settings",
      state.semitones == -5 and state.drive == 0.85 and state.reverb == 0.4
      and state.robot == 0.0 and state.doubler == 0.0 and state.bass == 0.0)
state.apply_preset(names.index("Walkie-Talkie"))
check("Walkie-Talkie enables radio", state.radio is True)
state.apply_preset(names.index("Ghost"))
check("Ghost has reverb + echo", state.reverb == 0.85 and state.echo == 0.4)
state.apply_preset(0)
check("Normal clears all effects",
      state.reverb == 0 and state.echo == 0 and state.radio is False
      and state.drive == 0)

# --------------------------------------------- integration via main callback
cb = voicebox.make_callback(state)
out = np.zeros((frames, 1), dtype=np.float32)
with state.lock:
    state.reverb = 0.8
sine = (0.4 * np.sin(2 * np.pi * 300 * np.arange(frames) / voicebox.SAMPLERATE)
        ).astype(np.float32).reshape(-1, 1)
for _ in range(4):
    cb(sine, out, frames, None, None)
silent_in = np.zeros((frames, 1), dtype=np.float32)
cb(silent_in, out, frames, None, None)
check("callback carries reverb tail into silence", np.abs(out).max() > 0.005,
      f"{np.abs(out).max():.4f}")
with state.lock:
    state.reverb = 0.0
    state.radio = True
for _ in range(4):
    cb(sine, out, frames, None, None)
check("callback radio path produces audio", np.abs(out).max() > 0.05)
check("callback output stays block-sized", out.shape == (frames, 1))

finish()
