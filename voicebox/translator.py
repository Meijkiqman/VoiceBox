"""Speech-to-speech translator: tap the hotkey (or the menu row), speak,
tap again - the utterance is transcribed (faster-whisper), translated
offline (Argos Translate) and spoken into the cable in the target
language's TTS voice, through the effect chain or the AI (RVC) voice
exactly like a typed TTS phrase.

The heavy dependencies are imported lazily on the worker thread, so
VoiceBox starts and runs without them; the row just reports what to
install. First use also downloads the Whisper model and the Argos language
packages (network needed once; both cache locally after that)."""
import queue
import threading
import time

import numpy as np
from scipy.signal import resample_poly

from .config import (SAMPLERATE, TRANS_MAX_S, TRANS_MIN_S, TRANS_MODEL,
                     TRANS_SOURCES, TRANS_TARGETS)
from .tts import tts_synthesize

# Whisper reports ISO codes; Argos wants "nb" for Norwegian (Bokmål - "nn"
# Nynorsk speech is close enough that routing it through nb beats failing).
ARGOS_CODE = {"no": "nb", "nn": "nb", "en": "en", "es": "es", "zh": "zh"}

# Argos package pairs the configured languages need. no->es / no->zh pivot
# through English, so nb->en plus en->target covers every combination.
ARGOS_PAIRS = [("nb", "en"), ("en", "es"), ("en", "zh")]

# Substrings that identify an installed OS voice for a target language, for
# auto-picking when the user hasn't chosen one ("Microsoft Pablo - Spanish
# (Spain)", "Microsoft Huihui - Chinese (Simplified, PRC)", ...).
VOICE_HINTS = {
    "en": ("english",),
    "es": ("spanish", "español", "espanol"),
    "zh": ("chinese", "mandarin", "taiwanese"),
}


def pick_voice(names, lang, explicit=None):
    """The TTS voice for a target language: the user's explicit choice if it
    is still installed, else the first installed voice whose name mentions
    the language, else None (= engine default)."""
    names = names or []
    if explicit and explicit in names:
        return explicit
    hints = VOICE_HINTS.get(lang, ())
    for n in names:
        low = n.lower()
        if any(h in low for h in hints):
            return n
    return None


class Translator:
    """Capture toggle + the background pipeline. All public methods are
    thread-safe (hotkey listener thread, UI thread)."""

    def __init__(self, state, player=None, monitor=None, ai=None,
                 voices_fn=None):
        self.state = state
        self.player = player
        self.monitor = monitor
        self.ai = ai
        self.capturing = False
        self.phase = ""              # "" | "loading models" | "transcribing" | ...
        self.error = ""              # sticky last failure, shown on the row
        self.last = ""               # last "heard -> said" line
        self.voices_fn = voices_fn   # () -> installed TTS voices (TTSBank's list)
        self._lock = threading.Lock()
        self._jobs = queue.Queue()
        self._whisper = None
        self._argos = None           # argostranslate.translate module when ready
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    @property
    def voice_names(self):
        return self.voices_fn() if self.voices_fn is not None else None

    # ---- configuration helpers (None-safe against settings.json) ----

    def source(self):
        v = self.state.trans_source
        return v if v in [c for c, _ in TRANS_SOURCES] else "auto"

    def target(self):
        v = self.state.trans_target
        return v if v in [c for c, _ in TRANS_TARGETS] else "en"

    def cycle_source(self, d=1):
        codes = [c for c, _ in TRANS_SOURCES]
        i = codes.index(self.source())
        with self.state.lock:
            self.state.trans_source = codes[(i + (1 if d >= 0 else -1)) % len(codes)]

    def cycle_target(self, d=1):
        codes = [c for c, _ in TRANS_TARGETS]
        i = codes.index(self.target())
        with self.state.lock:
            self.state.trans_target = codes[(i + (1 if d >= 0 else -1)) % len(codes)]

    def source_label(self):
        return dict(TRANS_SOURCES)[self.source()]

    def target_label(self):
        return dict(TRANS_TARGETS)[self.target()]

    def voice_for(self, lang=None):
        lang = lang or self.target()
        explicit = getattr(self.state, "trans_voice_" + lang, None)
        return pick_voice(self.voice_names, lang, explicit)

    def cycle_voice(self, d=1):
        """Step the current target language's voice through the installed
        list (None = auto-pick by language name)."""
        opts = [None] + (self.voice_names or [])
        attr = "trans_voice_" + self.target()
        try:
            i = opts.index(getattr(self.state, attr, None))
        except ValueError:
            i = 0
        with self.state.lock:
            setattr(self.state, attr, opts[(i + (1 if d >= 0 else -1)) % len(opts)])

    def voice_label(self):
        explicit = getattr(self.state, "trans_voice_" + self.target(), None)
        if explicit is None:
            auto = self.voice_for()
            return f"auto ({auto[:14]}...)" if auto else "auto (default)"
        return explicit if len(explicit) <= 24 else explicit[:21] + "..."

    def row_label(self):
        """The Translate row's value text."""
        if self.capturing:
            return "LISTENING - tap to speak"
        if self.phase:
            return self.phase
        if self.error:
            return "error - see status"
        return "tap and speak"

    # ---- capture ----

    def toggle(self):
        with self._lock:
            if self.capturing:
                self._stop_capture()
            else:
                self._start_capture()

    def _report(self, msg):
        self.state.status_msg = msg
        self.state.status_at = time.time()

    def _start_capture(self):
        if self.phase:                 # still working on the last utterance
            self._report("translator: busy - " + self.phase)
            return
        tap = queue.Queue(maxsize=int(TRANS_MAX_S * SAMPLERATE / 256) + 64)
        with self.state.lock:
            self.state.trans_tap = tap
            self.state.trans_hold = True   # listeners don't hear the original
        self.capturing = True
        if self.state.ai_mute:
            # the RVC worker reads the mic itself; we can't hold that back
            self._report("translating (note: AI voice still carries the original)")
        else:
            self._report(f"translating {self.source_label()} -> {self.target_label()} - speak, then tap again")

    def _stop_capture(self):
        with self.state.lock:
            tap, self.state.trans_tap = self.state.trans_tap, None
            self.state.trans_hold = False
        self.capturing = False
        blocks = []
        if tap is not None:
            while True:
                try:
                    blocks.append(tap.get_nowait())
                except queue.Empty:
                    break
        audio = (np.concatenate(blocks) if blocks
                 else np.zeros(0, dtype=np.float32))
        audio = audio[: int(TRANS_MAX_S * SAMPLERATE)]
        if len(audio) < TRANS_MIN_S * SAMPLERATE:
            self._report("translator: heard nothing")
            return
        self.phase = "working..."
        self._jobs.put(audio)

    # ---- pipeline (worker thread) ----

    def _worker(self):
        while True:
            audio = self._jobs.get()
            try:
                self._process(audio)
                self.error = ""
            except Exception as e:
                self.error = str(e)[:80]
                self._report(f"translator: {self.error}")
            finally:
                self.phase = ""

    def _ensure_models(self):
        if self._whisper is None:
            self.phase = "loading speech model..."
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise RuntimeError(
                    "needs: pip install faster-whisper argostranslate")
            name = self.state.trans_model or TRANS_MODEL
            try:
                self._whisper = WhisperModel(name, device="auto",
                                             compute_type="default")
            except Exception:
                # no CUDA runtime / half-precision unsupported: plain CPU int8
                self._whisper = WhisperModel(name, device="cpu",
                                             compute_type="int8")
        if self._argos is None:
            self.phase = "loading translation..."
            try:
                import argostranslate.package as apkg
                import argostranslate.translate as atrans
            except ImportError:
                raise RuntimeError(
                    "needs: pip install faster-whisper argostranslate")
            installed = {(p.from_code, p.to_code)
                         for p in apkg.get_installed_packages()}
            missing = [p for p in ARGOS_PAIRS if p not in installed]
            if missing:
                self.phase = "downloading language packs..."
                apkg.update_package_index()
                available = apkg.get_available_packages()
                for f, t in missing:
                    match = [p for p in available
                             if p.from_code == f and p.to_code == t]
                    if not match:
                        raise RuntimeError(f"no Argos package {f}->{t}")
                    apkg.install_from_path(match[0].download())
            self._argos = atrans

    def _translate(self, text, src, dst):
        """Argos translation; pairs without a direct package pivot through
        English (covers Norwegian -> Spanish/Mandarin)."""
        try:
            return self._argos.translate(text, src, dst)
        except Exception:
            if src == "en" or dst == "en":
                raise
        return self._argos.translate(
            self._argos.translate(text, src, "en"), "en", dst)

    def _process(self, audio):
        self._ensure_models()
        self.phase = "transcribing..."
        src_cfg = self.source()
        audio16 = resample_poly(audio, 16000, SAMPLERATE).astype(np.float32)
        segments, info = self._whisper.transcribe(
            audio16, language=None if src_cfg == "auto" else src_cfg,
            beam_size=5, vad_filter=True)
        text = " ".join(s.text.strip() for s in segments).strip()
        if not text:
            self._report("translator: heard nothing")
            return
        detected = src_cfg if src_cfg != "auto" else info.language
        src = ARGOS_CODE.get(detected)
        if src is None:
            raise RuntimeError(f"can't translate from '{detected}'")
        dst = self.target()
        if src == dst:
            out = text                 # same language: voice-over only
        else:
            self.phase = "translating..."
            out = (self._translate(text, src, dst) or "").strip()
            if not out:
                raise RuntimeError("translation came back empty")
        self.phase = "speaking..."
        samples, wav = tts_synthesize(out, self.voice_for(dst), 0)
        self.last = f"{detected}: {text}  ->  {dst}: {out}"
        self._report(self.last[:120])
        self._route(samples, wav)

    def _route(self, samples, wav):
        """Same routing contract as TTSBank._route: through the AI voice when
        it is live (the translation comes out in the user's RVC voice), else
        onto the mic channel - fx-tagged so the chain shapes it - plus the
        local speakers so the user hears what was said."""
        fx = self.state.tts_fx
        through_ai = False
        if (fx and wav is not None and self.ai is not None
                and self.ai.proc is not None):
            through_ai = self.ai.inject(wav)
        if not through_ai:
            self.state.events.put(("tts", samples, fx))
        mirrored = (self.monitor is not None and self.monitor.on
                    and not through_ai)
        if self.player is not None and not mirrored:
            self.player.play_raw(samples)
