"""JSON persistence helpers shared by settings, user presets, scenes and the
TTS phrase bank - one reader that treats a missing/corrupt file as "use the
default", one writer that can't leave a half-written file behind."""
import json
import os
import threading
from pathlib import Path

# Every writer of one file shares its .tmp name: the autosave thread and the
# shutdown save (or a scene saved from the hotkey thread and the UI) could
# interleave write/replace and lose the newer snapshot. Writes are tiny, so
# one process-wide lock is enough.
_write_lock = threading.Lock()


def load_json(path, default=None):
    """path -> parsed JSON; `default` when the file is missing, unreadable
    or not valid JSON (the callers validate the shape themselves)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):          # JSONDecodeError/UnicodeDecodeError
        return default


def write_json_atomic(path, obj, **dumps_kw):
    """Write JSON via a temp file + rename: a crash or power cut mid-write
    must never truncate the live file (a truncated settings.json makes
    load_settings return {} and silently resets every setting)."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(obj, indent=2, **dumps_kw)
    with _write_lock:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
