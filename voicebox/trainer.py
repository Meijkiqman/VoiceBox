"""The training rows. "Retrain AI voice" refreshes the MyVoice model from
the harvested dataset; "Train new model" builds a brand-new voice from
audio clips you pick yourself: name it, choose the clips (the picker opens
in the training/ folder), and training starts by itself. Both launch
rvc_trainer.py in its own console window on the RVC runtime, so training
progress is visible and VoiceBox stays responsive. EXPERIMENTAL - see
design/VOICE_TRAINING.md."""
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from .config import BASE_DIR, RVC_DIR, TRAIN_LOG_DIR, TRAINING_DIR

TRAIN_NAME = "MyVoice"        # the experiment/model name the retrain row trains
MIN_MINUTES = 5.0             # don't bother below this much harvested data
CHUNK_S = 12.0                # imported clips are cut into pieces this long
PIECE_MIN_S = 1.0             # leftover pieces shorter than this are dropped
MIN_CLIPS = 10                # rvc_trainer refuses datasets with fewer wavs
AUDIO_TYPES = [("Audio clips", "*.wav *.flac *.ogg *.mp3 *.aiff *.aif"),
               ("All files", "*.*")]


def _safe_name(name):
    """Model names become RVC folder/file names - keep them filesystem-safe."""
    return re.sub(r"[^\w-]+", "_", (name or "").strip()).strip("_")


def import_clips(files, dataset):
    """Decode audio clips into an RVC dataset folder: mono 16-bit wavs,
    long files cut into CHUNK_S pieces (RVC's preprocess slices further on
    its own - the cut only satisfies the trainer's clip-count check).
    The folder is cleared first so the dataset is exactly this selection.
    Returns (kept, skipped, seconds)."""
    import soundfile as sf
    dataset = Path(dataset)
    dataset.mkdir(parents=True, exist_ok=True)
    for old in dataset.glob("*.wav"):
        old.unlink()
    kept, skipped, seconds = 0, 0, 0.0
    for f in files:
        f = Path(f)
        try:
            data, sr = sf.read(str(f), dtype="float32", always_2d=True)
        except Exception:
            skipped += 1
            continue
        mono = data.mean(axis=1)
        step = int(CHUNK_S * sr)
        stem = _safe_name(f.stem) or "clip"
        for i in range(0, len(mono), step):
            piece = mono[i:i + step]
            peak = float(np.abs(piece).max()) if len(piece) else 0.0
            if len(piece) < PIECE_MIN_S * sr or peak < 0.02:
                continue               # too short, or silence
            piece = piece * (0.95 / peak)
            sf.write(str(dataset / f"{stem}-{kept:04d}.wav"), piece, sr,
                     subtype="PCM_16")
            kept += 1
            seconds += len(piece) / sr
    return kept, skipped, seconds


class Trainer:
    def __init__(self, state, ai=None, harvester=None):
        self.state = state
        self.ai = ai
        self.harvester = harvester
        self.proc = None
        self.stage = ""       # new-model flow progress, shown as the row value
        self.log_path = None  # this attempt's log file (training/logs/)
        self.dialog_error = ""  # why a Tk dialog failed (vs. user cancel)

    @property
    def rvc_dir(self):
        return Path(self.state.rvc_dir) if self.state.rvc_dir else RVC_DIR

    @property
    def available(self):
        return (self.rvc_dir / "runtime" / "python.exe").is_file()

    @property
    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def label(self):
        if self.running:
            return "training... (see console)"
        return f"start ({TRAIN_NAME})"

    def new_label(self):
        if self.stage:
            return self.stage
        if self.running:
            return "training... (see console)"
        return "name it, pick clips"

    def _report(self, msg):
        """Status line + log: every message the rows show is also written
        to this attempt's log, so a refusal that flashed past is still
        readable afterwards."""
        self.state.status_msg = msg
        self.state.status_at = time.time()
        self._log(msg)

    def _log(self, msg):
        if self.log_path is None:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8",
                      errors="replace") as f:
                f.write(f"{time.strftime('%H:%M:%S')}  {msg}\n")
        except Exception:
            pass                       # logging must never break training

    def _open_log(self, what):
        """Start this attempt's log. Failing to open one is not fatal -
        self.log_path stays None and _log becomes a no-op."""
        try:
            TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
            self.log_path = (TRAIN_LOG_DIR
                             / f"train-{time.strftime('%Y%m%d-%H%M%S')}.log")
            self._log(f"=== {what} ===")
            self._log(f"VoiceBox: {BASE_DIR}")
            self._log(f"RVC dir : {self.rvc_dir}")
        except Exception:
            self.log_path = None

    def log_hint(self):
        """The log's location, for the tail of a failure message."""
        if self.log_path is None:
            return ""
        try:
            return f" - log: {self.log_path.relative_to(BASE_DIR)}"
        except ValueError:
            return f" - log: {self.log_path}"

    def _busy(self):
        """The refusals shared by both rows (one GPU, one training slot)."""
        if self.running:
            self._report("training already running - check its console window")
            return True
        if self.ai is not None and self.ai.proc is not None:
            self._report("turn off the AI voice first - training needs the GPU")
            return True
        if not self.available:
            self._report("no RVC runtime found")
            return True
        return False

    def launch(self):
        """Kick off training on the harvested dataset. Refuses while the AI
        voice is live (the GPU can't do both) or with too little data."""
        self._open_log(f"Retrain AI voice ({TRAIN_NAME})")
        if self._busy():
            return False
        mins = self.harvester.minutes if self.harvester is not None else 0.0
        if mins < MIN_MINUTES:
            self._report(f"only {mins:.1f} min of harvested voice - turn on "
                         f"'Voice harvest' and talk; {MIN_MINUTES:.0f}+ min "
                         "needed, ~30 is ideal")
            return False
        if self.harvester is not None and self.harvester.on:
            self.harvester.toggle()    # stop writing into the folder the
                                       # trainer is about to read
        dataset = (self.harvester.dir if self.harvester is not None
                   else self.rvc_dir / "dataset_self")
        if self._spawn(dataset, TRAIN_NAME):
            self._report(f"training '{TRAIN_NAME}' started in its own window "
                         "- VoiceBox stays usable (without the AI voice)")
            return True
        return False

    def new_model(self):
        """The "Train new model" row: ask for a name, pick clips, train.
        The dialogs run on a background thread so the window keeps
        rendering while they are up."""
        if self.stage:
            self._report("already setting up a model - look for a dialog "
                         "window")
            return False
        self._open_log("Train new model")
        if self._busy():
            return False
        self.stage = "see dialog..."
        threading.Thread(target=self._new_model_flow, daemon=True).start()
        return True

    def _new_model_flow(self):
        try:
            self.stage = "waiting for name..."
            self._log("asking for a model name (Tk dialog)")
            raw = self._ask_name()
            if self.dialog_error:      # Tk missing/broken: not a cancel
                self._report(f"could not open the name dialog: "
                             f"{self.dialog_error}{self.log_hint()}")
                return
            name = _safe_name(raw)
            if not name:
                self._report("new model canceled")
                return
            self._log(f"model name: {name!r} (from {raw!r})")
            if (list((self.rvc_dir / "weights").glob(f"{name}*.pth"))
                    or (self.rvc_dir / "logs" / name).is_dir()):
                self._report(f"a model called '{name}' already exists - "
                             "pick another name")
                return
            self.stage = "choosing clips..."
            self._log(f"asking for clips (picker opens in {TRAINING_DIR})")
            files = self._ask_clips()
            if self.dialog_error:      # Tk missing/broken: not a cancel
                self._report(f"could not open the clip picker: "
                             f"{self.dialog_error}{self.log_hint()}")
                return
            if not files:
                self._report("new model canceled - no clips chosen")
                return
            self._log(f"{len(files)} clip(s) chosen:")
            for f in files:
                self._log(f"    {f}")
            self.stage = "importing clips..."
            dataset = self.rvc_dir / f"dataset_{name.lower()}"
            self._log(f"importing into {dataset}")
            try:
                kept, skipped, seconds = import_clips(files, dataset)
            except Exception as e:
                self._log(traceback.format_exc())
                self._report(f"import: {e}{self.log_hint()}")
                return
            self._log(f"imported {kept} piece(s), {skipped} unreadable, "
                      f"{seconds / 60:.2f} min total")
            skip = f" ({skipped} unreadable skipped)" if skipped else ""
            if kept < MIN_CLIPS:
                self._report(f"only {seconds / 60:.1f} min of usable "
                             f"audio{skip} - need {MIN_CLIPS}+ pieces "
                             "(~2 min, 10+ is much better); add clips to "
                             "the training folder and retry")
                return
            if self.ai is not None and self.ai.proc is not None:
                self._report("turn off the AI voice first - training needs "
                             "the GPU")
                return
            if self._spawn(dataset, name):
                self._report(f"training '{name}' on {seconds / 60:.1f} min "
                             f"of clips{skip} - it runs in its own window")
        except Exception as e:                 # never lose a flow crash
            self._log(traceback.format_exc())
            self._report(f"new model: {e}{self.log_hint()}")
        finally:
            self.stage = ""

    def _ask_name(self):
        """Native name prompt. Tk ships with CPython; the root lives on
        this background thread, so pygame keeps running. A failure here
        is recorded in dialog_error - it must not read as a cancel."""
        self.dialog_error = ""
        try:
            import tkinter
            from tkinter import simpledialog
            root = tkinter.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            try:
                return simpledialog.askstring(
                    "Train new model", "Name for the new voice model:",
                    parent=root)
            finally:
                root.destroy()
        except Exception as e:
            self.dialog_error = str(e) or e.__class__.__name__
            self._log(traceback.format_exc())
            return None

    def _ask_clips(self):
        """Native multi-file picker, opening in the training/ drop folder."""
        self.dialog_error = ""
        try:
            import tkinter
            from tkinter import filedialog
            TRAINING_DIR.mkdir(parents=True, exist_ok=True)
            root = tkinter.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            try:
                return list(filedialog.askopenfilenames(
                    title="Choose training clips - training starts right "
                          "after",
                    initialdir=str(TRAINING_DIR), filetypes=AUDIO_TYPES,
                    parent=root))
            finally:
                root.destroy()
        except Exception as e:
            self.dialog_error = str(e) or e.__class__.__name__
            self._log(traceback.format_exc())
            return []

    def _spawn(self, dataset, name):
        """Start rvc_trainer.py in its own console window and watch it."""
        cmd = [str(self.rvc_dir / "runtime" / "python.exe"),
               str(BASE_DIR / "rvc_trainer.py"),
               "--dataset", str(dataset), "--name", name]
        if self.log_path is not None:      # the trainer appends its whole
            cmd += ["--log", str(self.log_path)]   # run to the same file
        flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) \
            if sys.platform == "win32" else 0
        self._log("launching: " + " ".join(cmd))
        try:
            self.proc = subprocess.Popen(cmd, cwd=str(self.rvc_dir),
                                         creationflags=flags)
        except Exception as e:
            self._log(traceback.format_exc())
            self._report(f"training: {e}{self.log_hint()}")
            return False
        threading.Thread(target=self._watch, args=(self.proc, name),
                         daemon=True).start()
        return True

    def _watch(self, proc, name):
        """Follow one training run to its end: refresh the AI character
        list so the new model shows up without a restart, and put the
        outcome in the status line (the console window is gone by then)."""
        code = proc.wait()
        if proc is not self.proc:      # a newer run took over the slot
            return
        self.proc = None
        made = list((self.rvc_dir / "weights").glob(f"{name}*.pth"))
        if code != 0:
            self._report(f"training '{name}' failed (exit {code}) - the "
                         f"reason is in the log{self.log_hint()}")
        elif not made:
            self._report(f"training '{name}' finished but wrote no model"
                         f"{self.log_hint()}")
        else:
            if self.ai is not None:
                try:
                    self.ai.rescan()
                except Exception:
                    pass
            self._report(f"training done - '{name}' is in the AI character "
                         "row")
