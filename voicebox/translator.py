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

from .config import (BLOCKSIZE, SAMPLERATE, TRANS_AUTO_STOP_S,
                     TRANS_FLOOR_DB, TRANS_FLOOR_MAX, TRANS_IDLE_S,
                     TRANS_MAX_S, TRANS_MIN_S, TRANS_MODEL, TRANS_SOURCES,
                     TRANS_TARGETS, TRANS_VAD_KEEP, TRANS_VAD_START)
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
        self.auto = False            # continuous mode (the TRANS strip toggle)
        self.phase = ""              # "" | "loading models" | "transcribing" | ...
        self.error = ""              # sticky last failure, shown on the row
        self.last = ""               # last "heard -> said" line
        self.voices_fn = voices_fn   # () -> installed TTS voices (TTSBank's list)
        self._lock = threading.Lock()
        self._jobs = queue.Queue()
        self._whisper = None
        self._device = "cpu"         # where whisper actually loaded
        self._argos = None           # argostranslate.translate module when ready
        # the listener shares these models from its own thread; neither
        # whisper nor argos promises thread safety, so one lock guards both
        self.model_lock = threading.RLock()
        self._no_pack = set()        # language codes we failed to fetch packs for
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
        if self.auto:
            return "AUTO - just talk"
        if self.capturing:
            return "LISTENING - just speak"
        if self.phase:
            return self.phase
        if self.error:
            return "error - see status"
        return "Ctrl+Alt+T, speak"

    # ---- capture ----

    def toggle(self):
        """One-shot tap (hotkey/row). In auto mode it means "send now"."""
        with self._lock:
            if self.capturing:
                self._stop_capture()
            else:
                self._start_capture()

    def toggle_auto(self):
        """Continuous mode (the TRANS strip chip): while on, every utterance
        is captured, translated and spoken by itself - the cable carries
        only translations, never the raw voice."""
        with self._lock:
            self.auto = not self.auto
            with self.state.lock:
                self.state.trans_auto = self.auto   # persisted
            if self.auto and not self.capturing:
                self._start_capture()
            elif not self.auto and self.capturing:
                self._stop_capture()   # translate whatever was mid-sentence
        self._report("auto-translate ON - your voice goes out only as "
                     f"{self.target_label()} translations" if self.auto
                     else "auto-translate off")

    def _report(self, msg):
        self.state.status_msg = msg
        self.state.status_at = time.time()

    def _start_capture(self):
        if self.phase and not self.auto:   # still working on the last one
            self._report("translator: busy - " + self.phase)
            return                     # (auto mode keeps listening regardless)
        tap = queue.Queue(maxsize=int(TRANS_MAX_S * SAMPLERATE / BLOCKSIZE) + 64)
        with self.state.lock:
            self.state.trans_tap = tap
            self.state.trans_hold = True   # listeners don't hear the original
        self.capturing = True
        # hands-free: a stretch of silence after speech sends the capture by
        # itself - the second tap is only ever needed to cut a capture short
        threading.Thread(target=self._auto_stop, args=(tap,),
                         daemon=True).start()
        if self.state.ai_mute:
            # the RVC worker reads the mic itself; we can't hold that back
            self._report("translating (note: AI voice still carries the original)")
        else:
            self._report(f"translating {self.source_label()} -> "
                         f"{self.target_label()} - speak, it sends itself")

    def _auto_stop(self, tap):
        """Watch the mic meter while capturing: once speech has been heard,
        TRANS_AUTO_STOP_S of quiet ends the capture as if the user tapped
        again; TRANS_IDLE_S with no speech at all gives up (one-shot mode).

        Speech detection is adaptive: the watcher tracks the mic's noise
        floor - falling to a quieter reading instantly, rising only ~3 dB/s
        - and treats TRANS_VAD_START dB above that floor as speech onset,
        dropping below floor + TRANS_VAD_KEEP as quiet. A fixed threshold
        here either misses soft voices or, with a noisy mic, never sees
        "quiet" at all and holds the capture open forever."""
        pre_blocks = int(0.5 * SAMPLERATE / BLOCKSIZE)   # pre-roll kept in auto
        t0 = time.time()
        heard = False
        quiet_since = None
        floor = TRANS_FLOOR_DB
        while True:
            time.sleep(0.05)
            with self._lock:
                if not self.capturing or self.state.trans_tap is not tap:
                    return             # manual stop, or not our capture anymore
                now = time.time()
                lvl = 20.0 * np.log10(max(float(self.state.in_level), 1e-4))
                floor = min(TRANS_FLOOR_MAX,
                            lvl if lvl < floor else floor + 0.15)
                if lvl >= floor + (TRANS_VAD_KEEP if heard and
                                   quiet_since is None else TRANS_VAD_START):
                    heard, quiet_since = True, None
                elif heard:
                    quiet_since = quiet_since or now
                    if now - quiet_since >= TRANS_AUTO_STOP_S:
                        self._stop_capture()
                        return
                elif self.auto:
                    # continuous mode idles for free: silence never gives up,
                    # and the tap keeps only a short pre-roll so a long lull
                    # can't fill the queue and swallow the next sentence
                    t0 = now
                    while tap.qsize() > pre_blocks:
                        try:
                            tap.get_nowait()
                        except queue.Empty:
                            break
                elif now - t0 >= TRANS_IDLE_S:
                    self._stop_capture()   # one-shot tap, nothing said
                    return
                if now - t0 >= TRANS_MAX_S:
                    self._stop_capture()
                    return

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
            if not self.auto:          # in auto mode a lull is normal, not news
                self._report("translator: heard nothing")
        else:
            self.phase = "working..."
            self._jobs.put(audio)
        if self.auto:                  # continuous mode: listen right on
            self._start_capture()

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

    def warm(self):
        """Background model preload at startup - only when the optional deps
        are already installed - so the first hotkey tap doesn't sit behind a
        minutes-long load/download. Failures stay silent here; the first
        real use reports them properly."""
        def job():
            try:
                import faster_whisper              # noqa: F401
                import argostranslate.package      # noqa: F401
            except ImportError:
                return                             # feature not installed
            try:
                self._ensure_models()
            except Exception:
                pass
        threading.Thread(target=job, daemon=True).start()

    def _load_whisper(self, cpu=False):
        from faster_whisper import WhisperModel
        name = self.state.trans_model or TRANS_MODEL
        if not cpu:
            try:
                return WhisperModel(name, device="auto",
                                    compute_type="default"), "auto"
            except Exception:
                pass       # no CUDA runtime / half-precision unsupported
        return WhisperModel(name, device="cpu", compute_type="int8"), "cpu"

    def _ensure_models(self, announce=False):
        """Load whisper + argos once. Only the outgoing worker announces
        progress on self.phase (its finally clears it); the listener's
        thread must never write phase or the Translate row would report
        busy forever."""
        def note(msg):
            if announce:
                self.phase = msg
        with self.model_lock:
            if self._whisper is None:
                note("loading speech model...")
                try:
                    self._whisper, self._device = self._load_whisper()
                except ImportError:
                    raise RuntimeError(
                        "needs: pip install faster-whisper argostranslate")
            if self._argos is None:
                note("loading translation...")
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
                    note("downloading language packs...")
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

    def _ensure_pair(self, src, dst):
        """Fetch the Argos package(s) for src->dst on demand - the incoming
        listener meets languages the base install doesn't cover. Only a
        definitive "the index has no such pack" is remembered in _no_pack;
        a network hiccup stays retryable on the next utterance."""
        if src in self._no_pack:
            raise RuntimeError(f"no translation pack for '{src}'")
        import argostranslate.package as apkg
        installed = {(p.from_code, p.to_code)
                     for p in apkg.get_installed_packages()}
        if (src, dst) in installed:
            return
        need = [p for p in ((src, "en"), ("en", dst))
                if p[0] != p[1] and p not in installed]
        if not need:
            return
        try:
            apkg.update_package_index()
            available = apkg.get_available_packages()
        except Exception:
            raise RuntimeError(f"pack fetch for '{src}' failed (offline?)")
        matches = {}
        for f, t in need:
            match = [p for p in available
                     if p.from_code == f and p.to_code == t]
            if not match:
                self._no_pack.add(src)         # the index simply has no pack
                raise RuntimeError(f"no translation pack for '{src}'")
            matches[(f, t)] = match[0]
        try:
            for pack in matches.values():
                apkg.install_from_path(pack.download())
        except Exception:
            raise RuntimeError(f"pack download for '{src}' failed - will retry")

    def transcribe(self, audio48k, language=None):
        """48 kHz mono float32 -> (text, language code). Shared with the
        incoming listener; the caller holds no lock."""
        audio16 = resample_poly(audio48k, 16000,
                                SAMPLERATE).astype(np.float32)

        def run():
            # faster-whisper decodes lazily: errors surface while iterating
            # segments, so the join belongs inside the retry scope
            segments, info = self._whisper.transcribe(
                audio16, language=language, beam_size=5, vad_filter=True)
            text = " ".join(s.text.strip() for s in segments).strip()
            return text, info

        with self.model_lock:
            try:
                text, info = run()
            except Exception:
                if self._device == "cpu":
                    raise
                # GPU hiccup - typically VRAM taken by the live RVC worker.
                # Rebuild on CPU int8 once and keep going; slower beats dead.
                self._whisper, self._device = self._load_whisper(cpu=True)
                text, info = run()
        return text, (language or info.language)

    def translate_utterance(self, audio48k, dst="en"):
        """The incoming-speech path: transcribe ANY detected language and
        translate it to dst. Returns (detected, text, translated);
        translated is None when the utterance is already in dst (or empty).
        Missing language packs are fetched once on demand."""
        self._ensure_models()
        text, detected = self.transcribe(audio48k)
        if not text:
            return detected, "", None
        src = ARGOS_CODE.get(detected, detected)
        if src == dst:
            return detected, text, None
        with self.model_lock:
            self._ensure_pair(src, dst)
            out = (self._translate(text, src, dst) or "").strip()
        return detected, text, out or None

    def _process(self, audio):
        self._ensure_models(announce=True)
        self.phase = "transcribing..."
        src_cfg = self.source()
        text, detected = self.transcribe(
            audio, None if src_cfg == "auto" else src_cfg)
        if not text:
            self._report("translator: heard nothing")
            return
        src = ARGOS_CODE.get(detected)
        if src is None:
            raise RuntimeError(f"can't translate from '{detected}'")
        dst = self.target()
        if src == dst:
            out = text                 # same language: voice-over only
        else:
            self.phase = "translating..."
            with self.model_lock:
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
            self.state.events.put(("trans", samples, fx))
        mirrored = (self.monitor is not None and self.monitor.on
                    and not through_ai)
        if self.player is not None and not mirrored:
            self.player.play_raw(samples)
