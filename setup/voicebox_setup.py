"""VoiceBoxSetup - one-click installer for VoiceBox.

What it does, in order:
  1. SYSTEM CHECK   Windows version, 64-bit, admin, internet, what's installed.
  2. PYTHON         If Python 3.9+ is missing: PowerShell downloads the official
                    python.org installer and runs it silently. Then pip installs
                    numpy/scipy/sounddevice/soundfile/pygame.
  3. VB-CABLE       If the virtual cable is missing: PowerShell downloads the
                    official driver pack from vb-audio.com and runs the silent
                    installer (elevated - expect one UAC prompt). Reboot after.
  4. VOICEBOX       Installs the app files to %LOCALAPPDATA%\\VoiceBox and puts
                    a shortcut on the Desktop. By default the files travel
                    inside this exe; pass --url <zip> to download them instead
                    (e.g. a GitHub archive zip once VoiceBox is pushed).

Flags:  --check (report only, change nothing)   --skip-python   --skip-driver
        --skip-app   --url <zip-with-a-VoiceBox-folder>

Build:  run Build-Setup.bat in this folder -> dist\\VoiceBoxSetup.exe
"""
import argparse
import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

PYTHON_VERSION = "3.11.9"
PYTHON_URL = (f"https://www.python.org/ftp/python/{PYTHON_VERSION}/"
              f"python-{PYTHON_VERSION}-amd64.exe")
CABLE_URL = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack45.zip"
PACKAGES = ["numpy", "scipy", "sounddevice", "soundfile", "pygame"]
APP_FILES = ["voicebox.py", "rvc_worker.py", "controls.json", "VoiceBox.bat"]
INSTALL_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "VoiceBox"


def say(msg=""):
    print(msg, flush=True)


def ps(command, capture=False):
    """Run one PowerShell command; returns CompletedProcess."""
    return subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        capture_output=capture, text=True)


def is_admin():
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def find_python():
    """Return (exe, 'x.y.z') for the newest usable Python 3.9+, or (None, None)."""
    candidates = []
    for name in ("python", "python3"):
        exe = shutil.which(name)
        if exe:
            candidates.append(exe)
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python"
    if local.is_dir():
        candidates += [str(p / "python.exe") for p in sorted(local.glob("Python3*"))]
    for exe in candidates:
        try:
            out = subprocess.run([exe, "--version"], capture_output=True, text=True,
                                 timeout=15).stdout.strip()
            ver = out.split()[-1]
            major, minor = (int(v) for v in ver.split(".")[:2])
            if (major, minor) >= (3, 9):
                return exe, ver
        except Exception:
            continue
    return None, None


def cable_installed():
    r = ps("(Get-CimInstance Win32_SoundDevice | "
           "Where-Object { $_.Name -match 'VB-Audio|CABLE' }).Name", capture=True)
    return bool(r.stdout.strip())


def internet_ok():
    r = ps("try { (Invoke-WebRequest -UseBasicParsing -Method Head "
           "-Uri 'https://www.python.org' -TimeoutSec 10).StatusCode } catch { 0 }",
           capture=True)
    return r.stdout.strip().startswith("2")


def download(url, dest):
    """Download url -> dest via PowerShell (as requested: everything web goes
    through PowerShell). Raises on failure."""
    say(f"    downloading {url}")
    r = ps(f"$ProgressPreference='SilentlyContinue'; "
           f"Invoke-WebRequest -UseBasicParsing -Uri '{url}' -OutFile '{dest}'")
    if r.returncode != 0 or not Path(dest).is_file():
        raise RuntimeError(f"download failed: {url}")
    say(f"    saved {Path(dest).name} ({Path(dest).stat().st_size // 1024} KB)")


# --------------------------------------------------------------------- steps
def step_check():
    say("== 1/4  SYSTEM CHECK " + "=" * 40)
    import platform
    win = platform.system() == "Windows"
    arch = platform.machine().lower() in ("amd64", "x86_64")
    say(f"    Windows:        {'OK  (' + platform.release() + ')' if win else 'NO - Windows only'}")
    say(f"    64-bit:         {'OK' if arch else 'NO - 64-bit required'}")
    say(f"    admin rights:   {'yes' if is_admin() else 'no (UAC prompt will appear for the driver step)'}")
    net = internet_ok()
    say(f"    internet:       {'OK' if net else 'NOT REACHABLE - downloads will fail'}")
    py_exe, py_ver = find_python()
    say(f"    python:         {py_ver + '  (' + py_exe + ')' if py_exe else 'not found - will be installed'}")
    cable = cable_installed()
    say(f"    VB-CABLE:       {'installed' if cable else 'not found - will be installed'}")
    say(f"    install target: {INSTALL_DIR}")
    if not (win and arch):
        say("\nThis system is not supported. Stopping.")
        sys.exit(1)
    return {"net": net, "python": py_exe, "cable": cable}


def step_python(report, skip):
    say("\n== 2/4  PYTHON + LIBRARIES " + "=" * 34)
    py_exe = report["python"]
    if skip:
        say("    skipped (--skip-python)")
        return py_exe
    if not py_exe:
        with tempfile.TemporaryDirectory() as td:
            installer = Path(td) / f"python-{PYTHON_VERSION}-amd64.exe"
            download(PYTHON_URL, installer)
            say("    running silent Python install (per-user, adds to PATH)...")
            r = subprocess.run([str(installer), "/quiet", "InstallAllUsers=0",
                                "PrependPath=1", "Include_launcher=1"])
            if r.returncode != 0:
                raise RuntimeError(f"Python installer exited with {r.returncode}")
        py_exe, py_ver = find_python()
        if not py_exe:
            raise RuntimeError("Python installed but not found - reboot and rerun.")
        say(f"    Python {py_ver} installed at {py_exe}")
    else:
        say(f"    Python already present: {py_exe}")
    say(f"    installing libraries: {', '.join(PACKAGES)}")
    r = subprocess.run([py_exe, "-m", "pip", "install", "--disable-pip-version-check",
                        *PACKAGES])
    if r.returncode != 0:
        raise RuntimeError("pip install failed - see output above.")
    say("    libraries OK")
    return py_exe


def step_cable(report, skip):
    say("\n== 3/4  VB-CABLE (virtual audio cable) " + "=" * 22)
    if skip:
        say("    skipped (--skip-driver)")
        return
    if report["cable"]:
        say("    already installed - nothing to do")
        return
    workdir = Path(tempfile.mkdtemp(prefix="vbcable_"))
    zpath = workdir / "VBCABLE_Driver_Pack45.zip"
    download(CABLE_URL, zpath)
    ps(f"Expand-Archive -Path '{zpath}' -DestinationPath '{workdir}' -Force")
    setup_exe = workdir / "VBCABLE_Setup_x64.exe"
    if not setup_exe.is_file():
        raise RuntimeError("VBCABLE_Setup_x64.exe missing from the driver pack")
    say("    running silent driver install (UAC prompt: click Yes)...")
    # -i = install, -h = hidden/silent; the driver installer must run elevated
    r = ps(f"Start-Process -FilePath '{setup_exe}' -ArgumentList '-i','-h' "
           f"-Verb RunAs -Wait; $LASTEXITCODE")
    if r.returncode != 0:
        raise RuntimeError("driver install was cancelled or failed")
    say("    VB-CABLE installed. IMPORTANT: reboot before first use!")


def app_source(url):
    """Folder holding the VoiceBox files: bundled with the exe by default,
    or extracted from a downloaded zip when --url is given."""
    if url:
        workdir = Path(tempfile.mkdtemp(prefix="voicebox_"))
        zpath = workdir / "voicebox.zip"
        download(url, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(workdir)
        hits = list(workdir.rglob("voicebox.py"))
        if not hits:
            raise RuntimeError("zip does not contain voicebox.py")
        return hits[0].parent
    bundled = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / "app"
    if bundled.is_dir():
        return bundled
    dev = Path(__file__).resolve().parents[1]      # running from source tree
    if (dev / "voicebox.py").is_file():
        return dev
    raise RuntimeError("no VoiceBox files found (bundle missing and no --url)")


def step_app(url, skip):
    say("\n== 4/4  VOICEBOX " + "=" * 44)
    if skip:
        say("    skipped (--skip-app)")
        return
    src = app_source(url)
    say(f"    installing from {src}")
    INSTALL_DIR.mkdir(parents=True, exist_ok=True)
    (INSTALL_DIR / "sounds").mkdir(exist_ok=True)
    copied = 0
    for name in APP_FILES:
        f = src / name
        if f.is_file():
            shutil.copy2(f, INSTALL_DIR / name)
            copied += 1
    if not copied:
        raise RuntimeError(f"no app files found in {src}")
    assets = src / "assets"
    if assets.is_dir():
        shutil.copytree(assets, INSTALL_DIR / "assets", dirs_exist_ok=True)
    say(f"    {copied} file(s) -> {INSTALL_DIR}")
    say("    creating Desktop shortcut...")
    ps("$s=(New-Object -ComObject WScript.Shell).CreateShortcut("
       "[Environment]::GetFolderPath('Desktop') + '\\VoiceBox.lnk'); "
       f"$s.TargetPath='{INSTALL_DIR / 'VoiceBox.bat'}'; "
       f"$s.WorkingDirectory='{INSTALL_DIR}'; $s.Save()")
    say("    done - double-click 'VoiceBox' on the Desktop to start")
    say(f"    AI voice (optional): extract the rvc package zip into")
    say(f"    {INSTALL_DIR}  (so it becomes {INSTALL_DIR / 'rvc'})")


def main():
    ap = argparse.ArgumentParser(description="VoiceBox one-click setup")
    ap.add_argument("--check", action="store_true", help="system check only")
    ap.add_argument("--skip-python", action="store_true")
    ap.add_argument("--skip-driver", action="store_true")
    ap.add_argument("--skip-app", action="store_true")
    ap.add_argument("--url", default="", help="zip URL to download VoiceBox from")
    args = ap.parse_args()

    say("VoiceBox Setup")
    say("--------------")
    try:
        report = step_check()
        if args.check:
            say("\n--check: report only, nothing was installed.")
            return
        if not report["net"] and not (report["python"] and report["cable"]):
            raise RuntimeError("no internet connection and components are missing")
        step_python(report, args.skip_python)
        step_cable(report, args.skip_driver)
        step_app(args.url, args.skip_app)
        say("\nAll done!" + ("" if report["cable"] else
            "  Reboot once so VB-CABLE finishes installing."))
        say("Discord: Settings -> Voice & Video -> Input Device = 'CABLE Output'.")
    except Exception as e:
        say(f"\nSETUP FAILED: {e}")
        input("\nPress Enter to close...")
        sys.exit(1)
    input("\nPress Enter to close...")


if __name__ == "__main__":
    main()
