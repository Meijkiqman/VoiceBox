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

from .config import (PIPER_DIR, SAMPLERATE, TTS_CACHE_DIR, TTS_MAX_CHARS,
                     TTS_PHRASES_PATH)

# ---------------------------------------------------------------- Piper
# Optional neural voices: a piper/ folder (engine + voices/*.onnx) makes
# far more realistic speech than the OS engines, fully offline. Installed
# by setup/Get-PiperVoices.bat; absent = everything below returns empty.
_piper_run = subprocess.run          # indirection point for the tests


def _piper_exe():
    for name in ("piper.exe", "piper"):
        p = Path(PIPER_DIR) / name
        if p.is_file():
            return p
    return None


def piper_voice_map():
    """{display name: onnx path} for the installed Piper voices. The stem
    en_US-ryan-high becomes "Piper: Ryan (en_US, high)"."""
    if _piper_exe() is None:
        return {}
    out = {}
    for f in sorted(Path(PIPER_DIR, "voices").glob("*.onnx")):
        parts = f.stem.split("-")
        disp = (f"Piper: {parts[1].replace('_', ' ').title()} "
                f"({parts[0]}, {parts[2]})"
                if len(parts) >= 3 else f"Piper: {f.stem}")
        out[disp] = f
    return out


def _synth_piper(text, wav_path, voice, rate):
    """Render with the Piper engine (blocking). rate reuses the SAPI -10..10
    scale: each 10 steps doubles/halves the pace via length_scale."""
    onnx = piper_voice_map().get(voice)
    exe = _piper_exe()
    if onnx is None or exe is None:
        raise RuntimeError(f"Piper voice not installed: {voice[:40]}")
    ls = 2.0 ** (-float(max(-10, min(10, rate))) / 10.0)
    cmd = [str(exe), "--model", str(onnx), "--output_file", str(wav_path),
           "--length_scale", f"{ls:.3f}"]
    r = _piper_run(cmd, input=text, text=True, capture_output=True,
                   timeout=120, cwd=str(exe.parent),
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0 or not Path(wav_path).is_file():
        detail = (r.stderr or r.stdout or "").strip().splitlines()
        raise RuntimeError(detail[-1][:80] if detail else "piper failed")

def winrt_speaking_rate(rate):
    """SAPI -10..10 speaking rate -> the WinRT SpeechSynthesizer scale
    (0.5..6.0, 1.0 = normal). Matches the say/espeak wpm curve: each 10
    steps doubles/halves the pace."""
    return max(0.5, min(6.0, 2.0 ** (rate / 10.0)))


# Windows render script. One PowerShell pass picks the engine by the selected
# voice: classic SAPI5 (System.Speech) when it owns the voice or none is set,
# else the modern OneCore/WinRT engine (Windows.Media.SpeechSynthesis) - which
# is where "natural" voices and most non-English ones live and which
# System.Speech cannot see. Exact name wins; a substring is a legacy fallback.
# __PATH__/__VOICE__ are single-quote-escaped literals, __RATE__/__WRATE__ are
# numbers, and the phrase text comes over stdin (no quoting hazard).
_WIN_TTS_PS = r"""
$ErrorActionPreference = 'Stop'
$path = '__PATH__'; $rate = __RATE__; $voiceName = '__VOICE__'
$text = [Console]::In.ReadToEnd()
Add-Type -AssemblyName System.Speech
$sapi = New-Object System.Speech.Synthesis.SpeechSynthesizer
$sapiVoice = $null
if ($voiceName) {
    $sapiVoice = $sapi.GetInstalledVoices() | Where-Object {
        $_.Enabled -and $_.VoiceInfo.Name -eq $voiceName } | Select-Object -First 1
}
$winrtVoice = $null
if ($voiceName -and -not $sapiVoice) {
    try {
        [Windows.Media.SpeechSynthesis.SpeechSynthesizer,Windows.Media,ContentType=WindowsRuntime] | Out-Null
        $winrtVoice = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices | Where-Object {
            $_.DisplayName -eq $voiceName } | Select-Object -First 1
    } catch {}
}
if ($voiceName -and -not $sapiVoice -and -not $winrtVoice) {
    $sapiVoice = $sapi.GetInstalledVoices() | Where-Object {
        $_.Enabled -and $_.VoiceInfo.Name -like "*$voiceName*" } | Select-Object -First 1
}
if ($winrtVoice) {
    $sapi.Dispose()
    Add-Type -AssemblyName System.Runtime.WindowsRuntime
    [Windows.Storage.Streams.DataReader,Windows.Storage.Streams,ContentType=WindowsRuntime] | Out-Null
    $asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
        $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
        $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
    function Await($op, $t) {
        $k = $asTask.MakeGenericMethod($t).Invoke($null, @($op)); $k.Wait(-1) | Out-Null; $k.Result }
    $synth = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
    $synth.Voice = $winrtVoice
    $synth.Options.SpeakingRate = __WRATE__
    $stream = Await ($synth.SynthesizeTextToStreamAsync($text)) ([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])
    $reader = New-Object Windows.Storage.Streams.DataReader($stream)
    Await ($reader.LoadAsync([uint32]$stream.Size)) ([uint32]) | Out-Null
    $bytes = New-Object byte[] $stream.Size
    $reader.ReadBytes($bytes)
    [System.IO.File]::WriteAllBytes($path, $bytes)
    $synth.Dispose()
} else {
    if ($sapiVoice) { $sapi.SelectVoice($sapiVoice.VoiceInfo.Name) }
    $sapi.Rate = $rate
    $sapi.SetOutputToWaveFile($path)
    $sapi.Speak($text)
    $sapi.Dispose()
}
"""


def synth_tts_wav(text, wav_path, voice=None, rate=0):
    """Render text to a wav file with the OS speech engine (blocking).
    Windows: SAPI5 or, for OneCore/natural voices, WinRT - chosen by the
    selected voice (see _WIN_TTS_PS). Fallbacks: macOS `say`, else espeak.
    `voice` is a name (None = engine default); `rate` is the SAPI -10..10
    scale, mapped to words/minute for say/espeak. The text travels over
    stdin so no shell-quoting issue can arise."""
    if voice and str(voice).startswith("Piper: "):
        return _synth_piper(text, wav_path, str(voice), rate)
    rate = int(max(-10, min(10, rate)))
    wpm = int(175 * 2.0 ** (rate / 10.0))      # say/espeak equivalent
    if sys.platform == "win32":
        script = (_WIN_TTS_PS
                  .replace("__PATH__", str(wav_path).replace("'", "''"))
                  .replace("__VOICE__", str(voice or "").replace("'", "''"))
                  .replace("__RATE__", str(rate))
                  .replace("__WRATE__", f"{winrt_speaking_rate(rate):.4f}"))
        cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]
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


# Lists both engines: SAPI5 (System.Speech) first, then OneCore (WinRT) - the
# latter wrapped in try/catch so an older Windows without it just yields SAPI.
_WIN_LIST_PS = r"""
Add-Type -AssemblyName System.Speech
(New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() |
    Where-Object { $_.Enabled } | ForEach-Object { $_.VoiceInfo.Name }
try {
    [Windows.Media.SpeechSynthesis.SpeechSynthesizer,Windows.Media,ContentType=WindowsRuntime] | Out-Null
    [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices | ForEach-Object { $_.DisplayName }
} catch {}
"""


def _dedup(names):
    """Drop blanks and duplicates, preserving first-seen order (SAPI wins)."""
    out, seen = [], set()
    for n in names:
        n = n.strip()
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def list_tts_voices():
    """Installed TTS voice names, for the voice pickers (blocking; runs on
    a background thread). Piper neural voices first (they're the realistic
    ones), then the OS engines: Windows spans both classic SAPI5 and the
    OneCore voices. Empty on failure."""
    return list(piper_voice_map()) + _list_os_voices()


def _list_os_voices():
    kw = dict(text=True, capture_output=True, timeout=20,
              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        if sys.platform == "win32":
            r = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 _WIN_LIST_PS], **kw)
            return _dedup(r.stdout.splitlines())
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
        self._gen = 0                 # bumped by invalidate(): synth jobs still
                                      # in flight then drop their (stale) result

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
        self._gen += 1                 # in-flight jobs hold old-voice audio
        self.samples.clear()
        self.wav_path.clear()
        self.status.clear()
        self.pending = None

    def _synth_job(self, text):
        gen = self._gen
        try:
            with self.state.lock:
                voice = getattr(self.state, "tts_voice", None)
                rate = getattr(self.state, "tts_rate", 0)
            samples, wav = tts_synthesize(text, voice, rate)
        except Exception as e:
            if gen != self._gen:       # settings moved on mid-render: a fresh
                return                 # job owns this phrase now, stay out
            self.status[text] = "error"
            if self.pending == text:
                self.pending = None
            self._report(f"TTS: {str(e)[:70]}")
            return
        if gen != self._gen:
            # the voice/rate changed while rendering: this audio is the OLD
            # voice - marking it ready would speak the wrong voice until the
            # next invalidate. Drop it; the next play re-synthesizes.
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
        samples = self.samples.get(text)
        if samples is None:
            # invalidate() (menu row, or a scene on the hotkey thread) beat us
            # between the ready-check and here: re-render, auto-play on landing
            self.status.pop(text, None)
            self.pending = text
            self.ensure(text)
            return
        fx = self.state.tts_fx
        through_ai = False
        wav = self.wav_path.get(text)
        if (fx and wav is not None and self.ai is not None
                and self.ai.proc is not None):
            through_ai = self.ai.inject(wav)
        if not through_ai:
            self.state.events.put(("tts", samples, fx))
        # local listen, like the soundboard - skipped when self-listen already
        # mirrors the mic mix (it would be heard doubled). The AI path is never
        # mirrored (the worker owns the cable), so it always plays locally.
        mirrored = (self.monitor is not None and self.monitor.on
                    and not through_ai)
        if self.player is not None and not mirrored:
            self.player.play_raw(samples)


