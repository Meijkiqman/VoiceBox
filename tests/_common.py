"""Shared test bootstrap for the VoiceBox suites.

Runs headless (SDL dummy drivers), stubs soundfile if it is not installed
(the tests monkeypatch load_clips, so the real decoder is never needed), and
puts the VoiceBox folder on sys.path so `import voicebox` works from anywhere.
"""
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

try:
    import soundfile  # noqa: F401  (real one is fine if present)
except Exception:
    _stub = types.ModuleType("soundfile")
    _stub.read = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stubbed"))
    sys.modules["soundfile"] = _stub

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The voice-list runs `espeak --voices` on a background thread; vfork-ing
# a (missing) binary while the main thread is inside SDL corrupts the
# parent heap in this container and segfaults at random later points.
# The tests never need real OS voices - stub the subprocess away globally.
try:
    import voicebox.tts as _tts_mod
    _tts_mod.list_tts_voices = lambda: []
except Exception:
    pass

FAILURES = []


def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name
          + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def wait_ui(cond, timeout=6.0, step=0.05):
    """Poll until cond() is truthy - for the run_ui poke threads, which
    must not race the render loop's first frames on a loaded machine."""
    import time
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if cond():
                return True
        except Exception:
            pass
        time.sleep(step)
    return False


def finish():
    print()
    print("ALL PASS" if not FAILURES else f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1 if FAILURES else 0)
