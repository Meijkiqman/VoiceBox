"""The "Retrain AI voice" row: launches rvc_trainer.py in its own console
window on the RVC runtime, so training progress is visible and VoiceBox
stays responsive. EXPERIMENTAL - see design/VOICE_TRAINING.md."""
import subprocess
import sys
import time
from pathlib import Path

from .config import BASE_DIR, RVC_DIR

TRAIN_NAME = "MyVoice"        # the experiment/model name the row trains
MIN_MINUTES = 5.0             # don't bother below this much data

class Trainer:
    def __init__(self, state, ai=None, harvester=None):
        self.state = state
        self.ai = ai
        self.harvester = harvester
        self.proc = None

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

    def _report(self, msg):
        self.state.status_msg = msg
        self.state.status_at = time.time()

    def launch(self):
        """Kick off training on the harvested dataset. Refuses while the AI
        voice is live (the GPU can't do both) or with too little data."""
        if self.running:
            self._report("training already running - check its console window")
            return False
        if self.ai is not None and self.ai.proc is not None:
            self._report("turn off the AI voice first - training needs the GPU")
            return False
        if not self.available:
            self._report("no RVC runtime found")
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
        cmd = [str(self.rvc_dir / "runtime" / "python.exe"),
               str(BASE_DIR / "rvc_trainer.py"),
               "--dataset", str(dataset), "--name", TRAIN_NAME]
        flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) \
            if sys.platform == "win32" else 0
        try:
            self.proc = subprocess.Popen(cmd, cwd=str(self.rvc_dir),
                                         creationflags=flags)
        except Exception as e:
            self._report(f"training: {e}")
            return False
        self._report(f"training '{TRAIN_NAME}' started in its own window - "
                     "VoiceBox stays usable (without the AI voice)")
        return True
