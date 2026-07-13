"""Run VoiceBox from a checkout:  python voicebox.py [--list]

The application lives in the voicebox/ package next to this file; this
launcher only exists so `python voicebox.py` (and VoiceBox.bat, and the
installer's shortcut) keep working. `python -m voicebox` does the same.
"""
if __name__ == "__main__":
    from voicebox.app import main
    main()
