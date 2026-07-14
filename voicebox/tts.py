"""Text to speech: OS-engine rendering, the voice/rate-keyed cache, and the
saved-phrase bank."""
import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import (SAMPLERATE, TTS_CACHE_DIR, TTS_MAX_CHARS,
                     TTS_PHRASES_PATH)

def synth_tts_wav(text, wav_path, voice=None, rate=0):
    """Render text to a wav file with the OS speech engine (blocking).
    Windows: SAPI via PowerShell. Fallbacks: macOS `say`, else espeak.
    `voice` is a name substring (None = engine default); `rate` is the SAPI
    -10..10 scale, mapped to words/minute for say/espeak. The text travels
    over stdin so no shell-quoting issue can arise."""
    rate = int(max(-10, min(10, rate)))
    wpm = int(175 * 2.0 ** (rate / 10.0))      # say/espeak equivalent
    if sys.platform == "win32":
        path_lit = str(wav_path).replace("'", "''")
        sel = ""
        if voice:
            v_lit = str(voice).replace("'", "''")
            sel = ("$v = $s.GetInstalledVoices() | Where-Object "
                   f"{{ $_.VoiceInfo.Name -like '*{v_lit}*' }} "
                   "| Select-Object -First 1; "
                   "if ($v) { $s.SelectVoice($v.VoiceInfo.Name) }; ")
        cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command",
               "Add-Type -AssemblyName System.Speech; "
               "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
               f"$s.Rate = {rate}; " + sel +
               f"$s.SetOutputToWaveFile('{path_lit}'); "
               "$s.Speak([Console]::In.ReadToEnd()); $s.Dispose()"]
    elif sys.platform == "darwin":
        cmd = ["say", "-o", str(wav_path), "--data-format=LEI16@22050"]
        if voice:
            cmd += ["-v", str(voice)]
        if rate:
            cmd += ["-r", str(wpm)]
        cmd += ["-f", "-"]
    else:
        cmd = ["espeak", "-w", str(wav_path)]
        if voice:
            cmd += ["-v", str(voice)]
        if rate:
            cmd += ["-s", str(wpm)]
        cmd += ["--stdin"]
    r = subprocess.run(cmd, input=text, text=True, capture_output=True,
                       timeout=60,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0:
        detail = (r.stderr or r.stdout or "").strip().splitlines()
        raise RuntimeError(detail[-1][:80] if detail else "speech engine failed")


def list_tts_voices():
    """Installed TTS voice names, for the "TTS voice" menu row (blocking;
    runs on a background thread). Empty list on any failure."""
    kw = dict(text=True, capture_output=True, timeout=20,
              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        if sys.platform == "win32":
            r = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "Add-Type -AssemblyName System.Speech; "
                 "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
                 ".GetInstalledVoices() "
                 "| ForEach-Object { $_.VoiceInfo.Name }"], **kw)
            return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        if sys.platform == "darwin":
            r = subprocess.run(["say", "-v", "?"], **kw)
            return [ln.split()[0] for ln in r.stdout.splitlines() if ln.split()]
        r = subprocess.run(["espeak", "--voices"], **kw)
        return [ln.split()[3] for ln in r.stdout.splitlines()[1:]
                if len(ln.split()) >= 4]
    except Exception:
        return []


def tts_cache_path(text, voice=None, rate=0):
    """Cache file for a (voice, rate, text) rendering - changing the voice
    or rate must not serve stale audio, so they are part of the key."""
    key = f"{voice or ''}|{float(rate):g}|{text}"
    return TTS_CACHE_DIR / (hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
                            + ".wav")


def tts_synthesize(text, voice=None, rate=0):
    """text -> (mono float32 samples at SAMPLERATE, cached wav path).
    Synthesized once per (voice, rate, text), then served from tts_cache/
    across restarts."""
    TTS_CACHE_DIR.mkdir(exist_ok=True)
    wav = tts_cache_path(text, voice, rate)
    if not wav.is_file():
        try:
            synth_tts_wav(text, wav, voice, rate)
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
        self.voice_names = None       # installed voices; None until listed
        self._voices_kicked = False

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

    def load_voice_names(self):
        """List installed voices once, in the background (subprocess)."""
        if self._voices_kicked:
            return
        self._voices_kicked = True

        def job():
            self.voice_names = list_tts_voices()
        threading.Thread(target=job, daemon=True).start()

    def invalidate(self):
        """Voice/rate changed: rendered speech is stale. Phrases stay; the
        next play re-synthesizes (disk cache makes switching back instant)."""
        self.samples.clear()
        self.wav_path.clear()
        self.status.clear()
        self.pending = None

    def _synth_job(self, text):
        try:
            with self.state.lock:
                voice = getattr(self.state, "tts_voice", None)
                rate = getattr(self.state, "tts_rate", 0)
            samples, wav = tts_synthesize(text, voice, rate)
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

    def say(self, text):
        """Speak text once without saving it (quick-speak, Shift+Enter).
        True = accepted; synthesis may land it a moment later."""
        text = " ".join(str(text).split())[:TTS_MAX_CHARS]
        if not text:
            return False
        if self.status.get(text) == "ready":
            self._route(text)
        else:
            self.pending = text            # auto-plays when synthesis lands
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


