"""Regression tests for the second sweep: worker process-tree cleanup, dead
audio streams, hand-edited presets, and what the installer ships."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from _common import check, finish

import voicebox

voicebox.soundboard.load_clips = lambda: ([], [])
tmpdir = Path(tempfile.mkdtemp())
ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------- hand-edited user_presets.json
ppath = tmpdir / "user_presets.json"
ppath.write_text(json.dumps([
    {"name": "Broken", "params": {"semitones": "high", "robot": None,
                                  "drive": 7, "reverb": float("nan"),
                                  "echo": "0.25", "radio": 1,
                                  "mystery": [1, 2]}},
    {"name": "Fine", "params": {"semitones": -3, "drive": 0.5}},
]), encoding="utf-8")
loaded = voicebox.load_user_presets(ppath)
params = dict(loaded)["Broken"]
check("non-numeric preset values are dropped at load",
      "semitones" not in params and "robot" not in params
      and "reverb" not in params and "mystery" not in params, str(params))
check("out-of-range and string numbers are clamped/coerced",
      params.get("drive") == 1.0 and params.get("echo") == 0.25
      and params.get("radio") is True, str(params))
check("a healthy preset loads unchanged",
      dict(loaded)["Fine"] == {"semitones": -3.0, "drive": 0.5})
pstate = voicebox.State()
pstate.user_presets = loaded
try:
    pstate.apply_preset(len(voicebox.PRESETS))      # "Broken"
    label = pstate.preset_label()                   # ran every frame
    ok = label == "Broken" and pstate.drive == 1.0 and pstate.semitones == 0
except Exception as e:
    ok, label = False, repr(e)
check("a broken preset applies and labels without raising", ok, str(label))
check("scenes share the same effect spec",
      voicebox.scenes.FX_FIELDS is voicebox.state.FX_FIELDS)

# ------------------------------------------------- main stream found dead


class Stream:
    def __init__(self, active=True):
        self.active, self.closed = active, False

    def close(self):
        self.closed = True


estate = voicebox.State()
eng = voicebox.AudioEngine(estate)
check("poll with no stream is a no-op", eng.poll() is True)
s = Stream()
eng.stream, eng.dev_line, eng.latency_ms = s, "Mic  ->  CABLE", 40.0
check("a stream whose callback never ran is left alone",
      all(eng.poll() for _ in range(6)) and eng.stream is s)
estate.cb_count = 10
eng.poll()                                          # arms on a live callback
estate.cb_count = 20
check("a ticking stream stays up", eng.poll() and eng.poll() is True)
results = [eng.poll() for _ in range(3)]            # heartbeat stands still
check("a stalled callback is declared dead after 3 polls",
      results == [True, False, True] and eng.stream is None and s.closed,
      str(results))
check("the footer reports the loss instead of a healthy device line",
      "stopped" in eng.error and eng.dev_line == "" and eng.latency_ms is None)
s2 = Stream(active=False)
eng.stream, eng.error = s2, ""
check("a stream PortAudio marks inactive is closed at once",
      eng.poll() is False and s2.closed and eng.stream is None and eng.error)

# -------------------------------------------- speaker stream reopens when dead
opened = []


class OutStream(Stream):
    def __init__(self, **kw):
        super().__init__()
        opened.append(self)

    def start(self):
        pass


real_out = voicebox.audio.sd.OutputStream
voicebox.audio.sd.OutputStream = OutStream
try:
    player = voicebox.LocalPlayer(voicebox.State())
    player.play_raw([0.0])
    player.play_raw([0.0])
    check("a live speaker stream is reused", len(opened) == 1)
    opened[0].active = False                        # headphones unplugged
    player.voices = [["stale", 0]]
    player.play_raw([0.0])
    check("a dead speaker stream is closed and reopened",
          len(opened) == 2 and opened[0].closed and player.stream is opened[1])
    check("nothing queued for the dead stream replays on the new one",
          player.voices == [] and player.events.qsize() == 1)
finally:
    voicebox.audio.sd.OutputStream = real_out

# ------------------------------------- AI worker: the whole tree goes down
if sys.platform == "win32":
    script = tmpdir / "worker_like.py"
    script.write_text(
        "import time\n"
        "from multiprocessing import Manager\n"
        "if __name__ == '__main__':\n"
        "    mm = Manager()\n"                     # what rvc_for_realtime does
        "    print('ready', flush=True)\n"
        "    time.sleep(120)\n", encoding="utf-8")

    def children(pid):
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter "
             f"\"ParentProcessId={pid}\").ProcessId"],
            capture_output=True, text=True).stdout.split()
        return [int(x) for x in out]

    def alive(pid):
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True).stdout
        return str(pid) in out

    proc = subprocess.Popen([sys.executable, str(script)],
                            stdout=subprocess.PIPE, text=True)
    proc.stdout.readline()
    kids = children(proc.pid)
    astate = voicebox.State()
    ai = voicebox.AiVoice(astate, rvc_dir=tmpdir / "no_rvc")
    ai.proc = proc
    ai.stop(wait=True)
    deadline = time.time() + 5
    while time.time() < deadline and any(alive(k) for k in kids):
        time.sleep(0.2)
    left = [k for k in kids if alive(k)]
    for k in left:                                  # never leak from a test
        subprocess.run(["taskkill", "/F", "/PID", str(k)], capture_output=True)
    check("the worker had a child process to orphan", len(kids) >= 1)
    check("stop takes the worker's child processes down with it",
          left == [] and proc.poll() is not None, str(left))
    check("stop still clears the worker slot", ai.proc is None
          and ai.status == "off")

worker_src = (ROOT / "rvc_worker.py").read_text(encoding="utf-8")
check("worker leaves when its stdin pipe closes",
      "quit_ev.set()" in worker_src and "quit_ev.wait(1)" in worker_src)

# ------------------------------------------------ installer ships everything
spec = importlib.util.spec_from_file_location(
    "voicebox_setup", ROOT / "setup" / "voicebox_setup.py")
vsetup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vsetup)
bat = (ROOT / "setup" / "Build-Setup.bat").read_text(encoding="utf-8")
missing = [n for n in vsetup.APP_FILES if Path(n).name not in bat]
check("Build-Setup.bat bundles every file the installer copies",
      missing == [], str(missing))
check("every file the installer copies exists in the checkout",
      all((ROOT / n).is_file() for n in vsetup.APP_FILES))
# the app launches these by path: an install without them breaks the row
for needed in ("rvc_worker.py", "rvc_trainer.py"):
    check(f"{needed} is part of the install", needed in vsetup.APP_FILES
          and needed in bat)

src = tmpdir / "src"
dst = tmpdir / "install"
for name in vsetup.APP_FILES:
    f = src / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("new " + name, encoding="utf-8")
(src / "voicebox").mkdir()
(src / "voicebox" / "app.py").write_text("x", encoding="utf-8")
dst.mkdir()
(dst / "controls.json").write_text("my remaps", encoding="utf-8")
(dst / "voicebox.py").write_text("old", encoding="utf-8")
vsetup.INSTALL_DIR = dst
vsetup.app_source = lambda url: src
vsetup.ps = lambda *a, **k: None                    # no Desktop shortcut
vsetup.say = lambda *a, **k: None
vsetup.step_app("", False)
check("an update keeps the user's controls.json",
      (dst / "controls.json").read_text(encoding="utf-8") == "my remaps")
check("an update still replaces the app files",
      (dst / "voicebox.py").read_text(encoding="utf-8") == "new voicebox.py")
check("the Piper installer lands in setup/ where the app points",
      (dst / "setup" / "get_piper_voices.py").is_file())
fresh = tmpdir / "fresh"
vsetup.INSTALL_DIR = fresh
vsetup.step_app("", False)
check("a first install does get controls.json",
      (fresh / "controls.json").is_file())

finish()
