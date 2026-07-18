"""Install the Piper neural TTS engine + two realistic English voices.

Downloads into <VoiceBox>/piper/:
    piper.exe (or piper on Linux/macOS) + its espeak-ng data
    voices/en_US-ryan-high.onnx    - male, very natural
    voices/en_US-lessac-high.onnx  - female, very natural

Idempotent: anything already present is skipped. Run from anywhere:
    python setup/get_piper_voices.py
or double-click setup/Get-PiperVoices.bat on Windows.

More voices (other languages included - Spanish, Mandarin, Norwegian, ...):
browse https://huggingface.co/rhasspy/piper-voices and drop the .onnx +
.onnx.json pair into piper/voices/. They appear in every VoiceBox voice
picker as "Piper: ..." on next launch."""
import io
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PIPER = BASE / "piper"
VOICES = PIPER / "voices"

ENGINE = {
    "win32": ("https://github.com/rhasspy/piper/releases/download/"
              "2023.11.14-2/piper_windows_amd64.zip"),
    "linux": ("https://github.com/rhasspy/piper/releases/download/"
              "2023.11.14-2/piper_linux_x86_64.tar.gz"),
    "darwin": ("https://github.com/rhasspy/piper/releases/download/"
               "2023.11.14-2/piper_macos_x64.tar.gz"),
}
HF = "https://huggingface.co/rhasspy/piper-voices/resolve"
VOICE_PATHS = [
    "en/en_US/ryan/high/en_US-ryan-high.onnx",         # male, very natural
    "en/en_US/ryan/high/en_US-ryan-high.onnx.json",
    "en/en_US/lessac/high/en_US-lessac-high.onnx",     # female, very natural
    "en/en_US/lessac/high/en_US-lessac-high.onnx.json",
]
VOICE_FILES = [(f"{HF}/v1.0.0/{p}", f"{HF}/main/{p}") for p in VOICE_PATHS]


def fetch(url, label):
    print(f"  downloading {label} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "VoiceBox"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def fetch_any(urls, label):
    err = None
    for u in urls:
        try:
            return fetch(u, label)
        except Exception as e:
            err = e
    raise RuntimeError(f"{label}: {err}")


def install_engine():
    if (PIPER / "piper.exe").is_file() or (PIPER / "piper").is_file():
        print("engine: already installed")
        return
    key = ("win32" if sys.platform == "win32"
           else "darwin" if sys.platform == "darwin" else "linux")
    data = fetch(ENGINE[key], "Piper engine (~25 MB)")
    # archives carry a top-level piper/ folder; extract next to voicebox.py
    if key == "win32":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(BASE)
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
            t.extractall(BASE)
        exe = PIPER / "piper"
        if exe.is_file():
            exe.chmod(0o755)
    if not ((PIPER / "piper.exe").is_file() or (PIPER / "piper").is_file()):
        raise SystemExit("extraction did not produce piper/piper[.exe] - "
                         "delete the piper folder and re-run")
    print("engine: installed")


def install_voices():
    VOICES.mkdir(parents=True, exist_ok=True)
    failed = []
    for urls in VOICE_FILES:
        name = urls[0].rsplit("/", 1)[1]
        dest = VOICES / name
        if dest.is_file() and dest.stat().st_size > 0:
            print(f"voice:  {name} already present")
            continue
        size = "~110 MB" if name.endswith(".onnx") else "config"
        try:
            data = fetch_any(urls, f"{name} ({size})")
        except Exception as e:
            print(f"voice:  {name} FAILED ({e})")
            failed.append(name)
            continue
        dest.write_bytes(data)
        print(f"voice:  {name} installed")
    if failed:
        print("\nsome voice downloads failed - grab them manually from")
        print("  https://huggingface.co/rhasspy/piper-voices")
        print("(the .onnx AND .onnx.json pair) and put them in "
              + str(VOICES))


def main():
    print("Piper neural TTS setup -> " + str(PIPER))
    install_engine()
    install_voices()
    print("\nDone. Start VoiceBox - the voice pickers now list:")
    print("  Piper: Ryan (en_US, high)     male, natural")
    print("  Piper: Lessac (en_US, high)   female, natural")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted - re-run to resume (finished files are kept)")
    except Exception as e:
        raise SystemExit(f"setup failed: {e}")
