"""Application wiring: build every component and run the window."""
import argparse
import threading

import sounddevice as sd

from .aivoice import AiVoice
from .audio import AudioEngine, LocalPlayer, Monitor, Recorder
from .controls import GlobalHotkeys
from .cues import Cues
from .harvester import Harvester
from .scenes import Scenes
from .soundboard import Board
from .state import State, load_settings, save_settings, settings_autosave
from .trainer import Trainer
from .translator import Translator
from .tts import TTSBank
from .ui import run_ui

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list audio devices and exit")
    args = ap.parse_args()
    if args.list:
        print(sd.query_devices()); return

    state = State()
    state.restore(load_settings())
    stop_flag = threading.Event()
    engine = AudioEngine(state)
    engine.open()                 # failure lands in engine.error; UI still opens

    monitor = Monitor(state,
                      has_main_stream=lambda: engine.stream is not None)
    engine.monitor = monitor      # device switches hand the fallback over
    player = LocalPlayer(state)
    state.cues = Cues(state, player)
    board = Board(state, player, monitor)
    ai = AiVoice(state, monitor=monitor)
    tts = TTSBank(state, player, monitor, ai)
    tts.warm()                    # synthesize saved phrases in the background
    scenes = Scenes(state, ai, tts)
    translator = Translator(state, player=player, monitor=monitor, ai=ai,
                            voices_fn=lambda: tts.voice_names)
    harvester = Harvester(state)
    if state.harvest_on:               # persisted toggle: resume collecting
        harvester.start()
    trainer = Trainer(state, ai=ai, harvester=harvester)
    hotkeys = GlobalHotkeys(state, board, ai=ai, scenes=scenes,
                            translator=translator)
    recorder = Recorder(state)
    threading.Thread(target=settings_autosave, args=(state, stop_flag),
                     daemon=True).start()
    try:
        run_ui(state, stop_flag, engine.dev_line, engine.error, monitor,
               board, ai, tts, hotkeys, engine, recorder, scenes,
               translator, harvester, trainer)
    except KeyboardInterrupt:
        pass
    finally:
        save_settings(state.snapshot())
        recorder.close()               # before the stream: flush what's queued
        harvester.stop()
        hotkeys.close()
        ai.close()
        monitor.close()
        player.close()
        engine.close()
    print("stopped.")

