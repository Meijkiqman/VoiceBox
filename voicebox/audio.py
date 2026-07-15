"""The audio path: main-stream callback, device engine with live switching,
self-listen, the local clip player and the output recorder."""
import queue
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from .config import (BLOCKSIZE, CHANNELS, INPUT_DEVICE_MATCH,
                     OUTPUT_DEVICE_MATCH, RECORDINGS_DIR, SAMPLERATE)

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
            ai_fx = state.ai_fx
            mic_muted = state.mic_muted
            gate_on, gate_db = state.gate_on, state.gate_db

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
            elif isinstance(ev, int):
                # bind the list once: a rescan on the UI thread may swap
                # state.clips between a len() check and the index
                clips = state.clips
                if 0 <= ev < len(clips):
                    state.voices.append([clips[ev], 0])

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

        # x = the chain input; None = the voice path stays fully silent
        if ai_mute:
            # the RVC worker owns the voice. With "AI voice FX" on, its
            # converted audio comes back over the local bridge and runs
            # through the chain like mic speech; otherwise the worker feeds
            # the cable itself and our voice path stays silent (not doubled).
            feed = state.ai_feed
            if ai_fx and feed is not None:
                x = feed.read(frames) * voice_gain   # worker gates its own mic
            else:
                x = None
        elif mic_muted:
            # muted: the voice drops out but fx-tagged TTS still rides the
            # chain, so saved phrases remain usable as a stand-in voice
            x = np.zeros(frames, dtype=np.float32)
        else:
            x = indata[:, 0].astype(np.float32) * voice_gain
            if gate_on:                    # gate the mic only, never the TTS
                x = state.gate_fx.process(x, gate_db)

        if x is None:
            y = np.zeros(frames, dtype=np.float32)
            carry = np.zeros(0, dtype=np.float32)
        else:
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
        rq = state.record_q                    # mirror to the recorder, if on
        if rq is not None:
            try:
                rq.put_nowait(y.copy())
            except queue.Full:
                pass                           # writer lagging: drop, never block
        outdata[:, 0] = y
    return callback


class AudioEngine:
    """Owns the main mic -> cable stream and the device choice, so devices
    can be switched from the menu at runtime instead of editing constants.
    Selection persists by device NAME (settings.json); None means the
    defaults at the top of this file. A saved device that has vanished
    (unplugged, cable uninstalled) falls back to the default with a status
    note instead of breaking startup."""

    def __init__(self, state):
        self.state = state
        self.stream = None
        self.monitor = None            # wired by app.main: the HEAR fallback
                                       # must hand over when a stream opens
        self.error = ""
        self.dev_line = ""
        self.in_name = "default mic"   # resolved names for the UI
        self.out_name = "default out"
        self.latency_ms = None         # reported stream latency, for the footer

    def _report(self, msg):
        self.state.status_msg = msg
        self.state.status_at = time.time()

    def _resolve(self, kind):
        saved = (self.state.input_device if kind == "input"
                 else self.state.output_device)
        if saved:
            try:
                return find_device(saved, kind)
            except SystemExit:
                with self.state.lock:      # don't keep persisting a dead choice
                    setattr(self.state, kind + "_device", None)
                self._report(f"{kind} device '{saved[:40]}' not found - using default")
        fallback = INPUT_DEVICE_MATCH if kind == "input" else OUTPUT_DEVICE_MATCH
        return find_device(fallback, kind)

    def open(self):
        """(Re)open the stream on the currently selected devices."""
        self.close()
        # If self-listen is running its full-chain fallback (the main stream
        # was down), close it first: two make_callback streams would fight
        # over the event queue and the shifter state. It comes back below -
        # as a mirror of the new stream on success, as the fallback on failure.
        m = self.monitor
        was_fallback = (m is not None and m.on
                        and getattr(m, "fallback", False))
        if was_fallback:
            m.toggle()
        try:
            in_dev = self._resolve("input")
            out_dev = self._resolve("output")
            self.in_name = (sd.query_devices(in_dev)["name"]
                            if in_dev is not None else "default mic")
            self.out_name = (sd.query_devices(out_dev)["name"]
                             if out_dev is not None else "default out")
            self.dev_line = (f"{self.in_name}  ->  {self.out_name}"
                             "   (Discord input: CABLE Output)")
            # Nothing has been draining state.events while no stream was
            # running, so a session of soundboard clicks would fire all at
            # once on the new stream. Clear the backlog, keep the pitch
            # (same pattern as the self-listen fallback start).
            while not self.state.events.empty():
                try:
                    self.state.events.get_nowait()
                except queue.Empty:
                    break
            self.state.events.put(("pitch", self.state.semitones))
            # latency="high" buys buffering headroom: Python-side hiccups (GC,
            # UI thread holding the GIL) then cause no dropouts. Adds ~20 ms -
            # fine for voice chat, and far better than cutting out.
            stream = sd.Stream(samplerate=SAMPLERATE, blocksize=BLOCKSIZE,
                               dtype="float32", channels=CHANNELS,
                               device=(in_dev, out_dev), latency="high",
                               callback=make_callback(self.state))
            stream.start()
            self.stream = stream
            try:                       # duplex streams report (input, output)
                lat = stream.latency
                self.latency_ms = 1000.0 * (sum(lat)
                                            if isinstance(lat, (tuple, list))
                                            else float(lat))
            except Exception:
                self.latency_ms = None
            self.error = ""
            if was_fallback:
                m.toggle()             # HEAR stays on, now mirroring the stream
            return True
        except (SystemExit, Exception) as e:
            self.stream = None
            self.dev_line = ""
            self.latency_ms = None
            self.error = f"audio unavailable: {e}"
            if was_fallback:
                m.toggle()             # main still down: restore the fallback
            return False

    def close(self):
        if self.stream is not None:
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def options(self, kind):
        """[None, name, name, ...]: None is the "default" entry."""
        key = ("max_input_channels" if kind == "input"
               else "max_output_channels")
        try:
            names = [d["name"] for d in sd.query_devices() if d[key] > 0]
        except Exception:
            names = []
        return [None] + names

    def short_name(self, kind, width=24):
        name = self.in_name if kind == "input" else self.out_name
        saved = (self.state.input_device if kind == "input"
                 else self.state.output_device)
        if saved is None:
            name = "default" if kind == "input" else f"auto ({name[:14]})"
        return name if len(name) <= width else name[:width - 3] + "..."

    def cycle(self, kind, d):
        """Step through available devices (wrapping through "default") and
        reopen the stream on the new choice."""
        opts = self.options(kind)
        if len(opts) < 2:
            return
        saved = (self.state.input_device if kind == "input"
                 else self.state.output_device)
        try:
            i = opts.index(saved)
        except ValueError:                 # saved device vanished mid-session
            i = 0
        choice = opts[(i + (1 if d >= 0 else -1)) % len(opts)]
        with self.state.lock:
            setattr(self.state, kind + "_device", choice)
        if not self.open():
            self._report(self.error)


class Monitor:
    """Self-listen (the HEAR strip toggle). While the main stream is running it
    mirrors the processed mix to the default speakers. If the main stream never
    opened (e.g. virtual cable not installed yet), toggling on runs the whole
    chain as a mic -> speakers stream instead, so the voice is still testable."""

    def __init__(self, state, has_main_stream):
        self.state = state
        self.has_main = has_main_stream        # bool, or callable for live state
        self.stream = None
        self.fallback = False          # True while the stream is the full-chain
                                       # mic->speakers stand-in (main was down)
        self.error = ""

    def _main_up(self):
        return self.has_main() if callable(self.has_main) else self.has_main

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
            self.fallback = False
            if not self._main_up():            # fallback stream fed the meter
                self.state.in_level = 0.0
            return
        try:
            if self._main_up():
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
                self.fallback = False
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
                self.fallback = True
            self.error = ""
        except Exception as e:
            self.stream = None
            self.state.monitor_q = None
            self.fallback = False
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
            elif isinstance(ev, int):
                clips = state.clips        # bind once: rescan may swap it
                if 0 <= ev < len(clips):
                    self.voices.append([clips[ev], 0])
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


class Recorder:
    """Records the processed mix to recordings/voicebox-*.wav. The audio
    callback mirrors its output into a queue (state.record_q, same pattern
    as self-listen) and a writer thread drains it to disk, so the callback
    never touches I/O. While the AI voice is live the worker owns the cable,
    so a recording captures only VoiceBox's own mix (soundboard + TTS) -
    unless "AI voice FX" is on, which routes the converted voice through
    VoiceBox's chain, so it is captured too."""

    def __init__(self, state, folder=RECORDINGS_DIR):
        self.state = state
        self.folder = Path(folder)
        self.path = None               # current/last file
        self.started_at = 0.0
        self.error = ""
        self._thread = None
        self._stop = None

    @property
    def on(self):
        return self.state.record_q is not None

    def _report(self, msg):
        self.state.status_msg = msg
        self.state.status_at = time.time()

    def start(self):
        if self.on:
            return
        name = time.strftime("voicebox-%Y%m%d-%H%M%S") + ".wav"
        try:
            self.folder.mkdir(exist_ok=True)
            f = sf.SoundFile(str(self.folder / name), mode="w",
                             samplerate=SAMPLERATE, channels=1,
                             subtype="PCM_16")
        except Exception as e:
            self.error = str(e)
            self._report(f"record: {e}")
            return
        self.path = self.folder / name
        self.error = ""
        self.started_at = time.time()
        q = queue.Queue(maxsize=64)    # ~0.7 s of headroom for slow disks
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._writer, args=(f, q),
                                        daemon=True)
        self._thread.start()
        self.state.record_q = q

    def _writer(self, f, q):
        try:
            while not self._stop.is_set() or not q.empty():
                try:
                    f.write(q.get(timeout=0.2))
                except queue.Empty:
                    continue
        finally:
            try:
                f.close()
            except Exception:
                pass

    def stop(self):
        if not self.on:
            return
        self.state.record_q = None     # callback stops feeding first
        self._stop.set()
        self._thread.join(timeout=3.0)
        secs = time.time() - self.started_at
        self._report(f"saved {self.path.name} ({secs:.0f}s)")

    def toggle(self):
        self.stop() if self.on else self.start()

    def close(self):
        if self.on:
            self.stop()




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


