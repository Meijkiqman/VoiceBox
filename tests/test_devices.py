"""AudioEngine: device resolution, live switching, persistence, fallbacks."""
from _common import check, finish

import numpy as np  # noqa: F401  (voicebox import needs it)
import voicebox

voicebox.soundboard.load_clips = lambda: ([], [])


class FakeStream:
    def __init__(self, **kw):
        self.kw = kw
        self.started = False
        self.closed = False
        self.latency = (0.010, 0.020)

    def start(self):
        self.started = True

    def close(self):
        self.closed = True


class FakeSD:
    """Stands in for sounddevice: fixed device list, recorded streams."""
    def __init__(self, devices=None, stream_error=None):
        self.devices = devices if devices is not None else [
            {"name": "Default Mic",
             "max_input_channels": 2, "max_output_channels": 0},
            {"name": "USB Gaming Mic",
             "max_input_channels": 1, "max_output_channels": 0},
            {"name": "CABLE Input (VB-Audio Virtual Cable)",
             "max_input_channels": 0, "max_output_channels": 2},
            {"name": "Speakers (Realtek)",
             "max_input_channels": 0, "max_output_channels": 2},
        ]
        self.streams = []
        self.stream_error = stream_error

    def query_devices(self, dev=None):
        return self.devices if dev is None else self.devices[dev]

    def Stream(self, **kw):
        if self.stream_error:
            raise RuntimeError(self.stream_error)
        s = FakeStream(**kw)
        self.streams.append(s)
        return s

    def OutputStream(self, **kw):
        return self.Stream(**kw)


real_sd = voicebox.audio.sd
voicebox.audio.sd = FakeSD()

# -------------------------------------------------------------- open/resolve
state = voicebox.State()
eng = voicebox.AudioEngine(state)
check("engine opens on defaults", eng.open() is True and eng.stream is not None)
check("stream started on the cable",
      eng.stream.started and eng.stream.kw["device"] == (None, 2))
check("dev line names both ends",
      "default mic" in eng.dev_line and "CABLE Input" in eng.dev_line)
check("engine reports round-trip latency",
      eng.latency_ms is not None and abs(eng.latency_ms - 30.0) < 0.01)

# ------------------------------------------------------------------- options
check("input options: default + input-capable devices",
      eng.options("input") == [None, "Default Mic", "USB Gaming Mic"])
check("output options: default + output-capable devices",
      eng.options("output") == [None, "CABLE Input (VB-Audio Virtual Cable)",
                                "Speakers (Realtek)"])

# ------------------------------------------------------------- live switching
first_stream = eng.stream
eng.cycle("input", +1)
check("cycle selects the first real input", state.input_device == "Default Mic")
check("cycle reopened the stream",
      first_stream.closed and eng.stream is not first_stream
      and eng.stream.kw["device"] == (0, 2))
eng.cycle("input", +1)
check("cycle steps onward", state.input_device == "USB Gaming Mic"
      and eng.stream.kw["device"] == (1, 2))
eng.cycle("input", +1)
check("cycle wraps back to default", state.input_device is None
      and eng.stream.kw["device"] == (None, 2))
eng.cycle("input", -1)
check("cycle steps backward", state.input_device == "USB Gaming Mic")

check("short name truncates for the row",
      len(eng.short_name("input")) <= 24)
with state.lock:
    state.input_device = None
eng.open()
check("default input shows as 'default'", eng.short_name("input") == "default")

# -------------------------------------------- persistence + stale device name
with state.lock:
    state.input_device = "USB Gaming Mic"
snap = state.snapshot()
check("device choice is in the settings snapshot",
      snap["input_device"] == "USB Gaming Mic")
fresh = voicebox.State()
fresh.restore(snap)
check("device choice restores", fresh.input_device == "USB Gaming Mic")

gone = voicebox.State()
gone.restore({"input_device": "Unplugged Headset"})
geng = voicebox.AudioEngine(gone)
check("stale saved device falls back to default",
      geng.open() is True and gone.input_device is None)
check("fallback is announced", "not found" in gone.status_msg)

# ------------------------------------------------------------- failure paths
voicebox.audio.sd = FakeSD(stream_error="portaudio exploded")
bad = voicebox.AudioEngine(voicebox.State())
check("stream failure keeps the UI alive",
      bad.open() is False and bad.stream is None
      and bad.error.startswith("audio unavailable"))
check("failed open clears the latency readout", bad.latency_ms is None)

voicebox.audio.sd = FakeSD(devices=[])
none = voicebox.AudioEngine(voicebox.State())
check("no devices -> error, no crash", none.open() is False)
check("no devices -> cycle is a no-op", none.cycle("input", +1) is None
      and none.stream is None)

# ------------------------------------------------------------ menu wiring
voicebox.audio.sd = FakeSD()
mstate = voicebox.State()
meng = voicebox.AudioEngine(mstate)
meng.open()
import threading
menu = voicebox.Menu(mstate, threading.Event(), engine=meng)
labels = [it.label for it in menu.items]
check("engine adds device rows",
      "Input device" in labels and "Output device" in labels)
row = next(it for it in menu.items if it.label == "Input device")
row.adjust(+1)
check("device row adjust switches the device",
      mstate.input_device == "Default Mic")


# --------------------------------------- HEAR fallback handoff on (re)open
class PickyFakeSD(FakeSD):
    """A duplex open with an explicit device fails (the cable is missing);
    the default-device fallback and mirror streams still open fine."""
    def Stream(self, **kw):
        if "device" in kw:
            raise RuntimeError("cable unplugged")
        return super().Stream(**kw)


voicebox.audio.sd = PickyFakeSD()
hstate = voicebox.State()
heng = voicebox.AudioEngine(hstate)
heng.open()                            # fails: the cable device is gone
hmon = voicebox.Monitor(hstate,
                        has_main_stream=lambda: heng.stream is not None)
heng.monitor = hmon
hmon.toggle()                          # HEAR on -> full-chain fallback
check("HEAR runs the fallback while the main stream is down",
      hmon.on and hmon.fallback and hstate.monitor_q is None)
fb1 = hmon.stream
heng.open()                            # still failing: fallback must survive
check("failed reopen restores the fallback",
      heng.stream is None and hmon.on and hmon.fallback
      and hmon.stream is not fb1 and hmon.stream.started)
fb2 = hmon.stream
voicebox.audio.sd = FakeSD()           # the cable comes back
heng.open()
check("reopen closes the fallback (no twin callbacks)", fb2.closed)
check("HEAR survives the handoff as a mirror of the new stream",
      hmon.on and not hmon.fallback and hstate.monitor_q is not None
      and heng.stream is not None and heng.stream.started)

# --------------------------------------- backlog drained when a stream opens
voicebox.audio.sd = FakeSD()
bstate = voicebox.State()
bstate.set_pitch(3)                    # a queued pitch, like restore() does
bstate.events.put(0); bstate.events.put(1); bstate.events.put(2)
beng = voicebox.AudioEngine(bstate)
beng.open()
backlog = []
while not bstate.events.empty():
    backlog.append(bstate.events.get_nowait())
check("open drains stale clip events, keeps the pitch",
      backlog == [("pitch", 3)])

voicebox.audio.sd = real_sd
finish()
