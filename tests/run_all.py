"""Run every VoiceBox test suite; exit 1 if any fails.

Usage:  python tests/run_all.py   (from the VoiceBox folder, or anywhere)
"""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
suites = sorted(here.glob("test_*.py"))
failed = []
for s in suites:
    print(f"=== {s.name} " + "=" * max(4, 46 - len(s.name)), flush=True)
    if subprocess.run([sys.executable, str(s)]).returncode != 0:
        failed.append(s.name)
    print(flush=True)
print("ALL SUITES PASS" if not failed else f"FAILED: {', '.join(failed)}")
sys.exit(1 if failed else 0)
