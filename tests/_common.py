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

FAILURES = []


def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name
          + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def finish():
    print()
    print("ALL PASS" if not FAILURES else f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1 if FAILURES else 0)
