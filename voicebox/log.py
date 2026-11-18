"""The diagnostics log (voicebox.log next to the checkout).

The services swallow a lot of exceptions on purpose - audio-stream
teardown, optional features that are not installed, subprocess chatter -
and say nothing, which is right for the UI and wrong for debugging: the
one that hid the window crash for months was a swallowed AttributeError.
This gives those sites somewhere to leave a trace. Handlers are attached
by app.main via setup(); until then a NullHandler keeps imports (and the
test suites) silent."""
import logging
from logging.handlers import RotatingFileHandler

from .config import LOG_PATH

_root = logging.getLogger("voicebox")
_root.addHandler(logging.NullHandler())


def get_logger(name):
    return logging.getLogger(f"voicebox.{name}")


def setup(path=LOG_PATH, level=logging.INFO):
    """Attach the rotating file handler once (app startup). A read-only
    install dir just means no log file - never a failed launch."""
    if any(isinstance(h, RotatingFileHandler) for h in _root.handlers):
        return
    try:
        handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=2,
                                      encoding="utf-8")
    except OSError:
        return
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    _root.addHandler(handler)
    _root.setLevel(level)
