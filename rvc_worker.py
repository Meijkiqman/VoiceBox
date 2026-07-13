"""rvc_worker.py - headless real-time RVC voice conversion, driven by VoiceBox.

A faithful port of RVC-beta0717's gui_v1.py streaming loop with the GUI
stripped out: mic -> RVC model (Arthur Morgan & friends) -> virtual cable.

MUST be run with RVC's own bundled interpreter and cwd set to the RVC folder
(hubert_base.pt / rmvpe.pt / config.py are resolved from cwd):

    <RVC>\\runtime\\python.exe rvc_worker.py --pth weights\\ArthurMorgan.pth ^
        --index <path.index> --output-device "CABLE Input"

Status protocol on stdout (read by VoiceBox):  "STATUS loading",
"STATUS running ...", "STATUS error <msg>".  VoiceBox may write
"PLAY <wav path>" lines on stdin; those files are mixed into the mic input,
so the model speaks them in the AI voice (the TTS-through-AI path).
--selftest converts a few synthetic blocks and reports timing instead of
opening audio devices.
"""
import argparse
import sys


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True, help="voice model .pth")
    ap.add_argument("--index", default="", help="matching .index file (optional)")
    ap.add_argument("--pitch", type=int, default=0, help="transpose in semitones")
    ap.add_argument("--index-rate", type=float, default=0.5)
    ap.add_argument("--block", type=float, default=0.35, help="block time (s)")
    ap.add_argument("--crossfade", type=float, default=0.05)
    ap.add_argument("--extra", type=float, default=0.5, help="context length (s)")
    ap.add_argument("--threshold", type=int, default=-45, help="noise gate dB")
    ap.add_argument("--f0method", default="rmvpe",
                    choices=["pm", "harvest", "crepe", "rmvpe"])
    ap.add_argument("--input-device", default="", help="substring, empty = default mic")
    ap.add_argument("--output-device", default="CABLE Input")
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    # Everything lives under this guard: rvc_for_realtime creates a
    # multiprocessing.Manager, whose spawned child re-imports this module.
    args = parse_args()
    sys.argv = sys.argv[:1]      # RVC's config.py parses argv at import time

    import os
    sys.path.insert(0, os.getcwd())          # RVC folder (we run with cwd there)

    print("STATUS loading", flush=True)
    import time
    from multiprocessing import Queue

    import librosa
    import numpy as np
    import sounddevice as sd
    import torch
    import torch.nn.functional as F
    import torchaudio.transforms as tat

    from rvc_for_realtime import RVC

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"STATUS device {device}", flush=True)

    def find_device(match, kind):
        if not match:
            return None
        key = "max_input_channels" if kind == "input" else "max_output_channels"
        for i, d in enumerate(sd.query_devices()):
            if match.lower() in d["name"].lower() and d[key] > 0:
                return i
        print(f"STATUS error no {kind} device matching '{match}'", flush=True)
        sys.exit(1)

    index_rate = args.index_rate if args.index else 0.0
    rvc = RVC(args.pitch, args.pth, args.index, index_rate,
              1, Queue(), Queue(), device)
    if not hasattr(rvc, "tgt_sr"):           # RVC() swallows its own exceptions
        print("STATUS error voice model failed to load (see log above)", flush=True)
        sys.exit(1)
    sr = rvc.tgt_sr

    # ---- buffers, exactly as gui_v1.start_vc() sets them up -----------------
    block_frame = int(args.block * sr)
    crossfade_frame = int(min(args.crossfade, args.block) * sr)
    sola_search_frame = int(0.01 * sr)
    extra_frame = int(args.extra * sr)
    zc = sr // 100
    span = extra_frame + crossfade_frame + sola_search_frame + block_frame
    total = int(np.ceil(span / zc) * zc)
    input_wav = np.zeros(total, dtype=np.float32)
    output_wav_cache = torch.zeros(total, device=device, dtype=torch.float32)
    pitch_cache = np.zeros(total // zc, dtype="int32")
    pitchf_cache = np.zeros(total // zc, dtype="float64")
    output_wav = torch.zeros(block_frame, device=device, dtype=torch.float32)
    sola_buffer = torch.zeros(crossfade_frame, device=device, dtype=torch.float32)
    fade_in = torch.linspace(0.0, 1.0, steps=crossfade_frame,
                             device=device, dtype=torch.float32)
    fade_out = 1 - fade_in
    resampler = tat.Resample(orig_freq=sr, new_freq=16000,
                             dtype=torch.float32).to(device)
    rate1 = block_frame / span
    rate2 = (crossfade_frame + sola_search_frame + block_frame) / span

    # ---- TTS injection ------------------------------------------------------
    # VoiceBox writes "PLAY <wav>" lines on our stdin; the samples are mixed
    # into the mic signal (after the noise gate) so the model converts them
    # like normal speech. Loading/resampling happens off the audio thread.
    import queue as _queue
    import threading

    tts_q = _queue.Queue()
    tts_buf = [np.zeros(0, dtype=np.float32)]

    def _stdin_listener():
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line.startswith("PLAY "):
                    continue
                try:
                    wav, _ = librosa.load(line[5:].strip(), sr=sr, mono=True)
                    tts_q.put(np.asarray(wav, dtype=np.float32))
                except Exception as e:
                    print(f"STATUS error tts {e}", flush=True)
        except Exception:
            pass

    def process(indata):
        """One stereo block in (block_frame, 2) -> converted mono block out.
        Port of gui_v1.audio_callback (noise gate + infer + SOLA crossfade)."""
        mono = librosa.to_mono(indata.T)
        if args.threshold > -60:             # simple noise gate
            rms = librosa.feature.rms(y=mono, frame_length=2048, hop_length=1024)
            db = librosa.amplitude_to_db(rms, ref=1.0)[0]
            for i, quiet in enumerate(db < args.threshold):
                if quiet:
                    mono[i * 1024:(i + 1) * 1024] = 0
        while not tts_q.empty():             # mix queued TTS into the mic
            tts_buf[0] = np.concatenate([tts_buf[0], tts_q.get_nowait()])
        if len(tts_buf[0]):
            n = min(len(mono), len(tts_buf[0]))
            mono[:n] += tts_buf[0][:n]
            tts_buf[0] = tts_buf[0][n:]
        input_wav[:] = np.append(input_wav[block_frame:], mono)
        inp = torch.from_numpy(input_wav).to(device)
        res1 = resampler(inp)
        res2 = rvc.infer(res1, res1[-block_frame:].cpu().numpy(), rate1, rate2,
                         pitch_cache, pitchf_cache, args.f0method)
        output_wav_cache[-res2.shape[0]:] = res2
        infer_wav = output_wav_cache[-crossfade_frame - sola_search_frame - block_frame:]
        # SOLA alignment (from DDSP-SVC), verbatim from gui_v1
        cor_nom = F.conv1d(
            infer_wav[None, None, :crossfade_frame + sola_search_frame],
            sola_buffer[None, None, :])
        cor_den = torch.sqrt(F.conv1d(
            infer_wav[None, None, :crossfade_frame + sola_search_frame] ** 2,
            torch.ones(1, 1, crossfade_frame, device=device)) + 1e-8)
        sola_offset = int(torch.argmax(cor_nom[0, 0] / cor_den[0, 0]))
        output_wav[:] = infer_wav[sola_offset:sola_offset + block_frame]
        output_wav[:crossfade_frame] *= fade_in
        output_wav[:crossfade_frame] += sola_buffer[:]
        if sola_offset < sola_search_frame:
            sola_buffer[:] = infer_wav[-sola_search_frame - crossfade_frame + sola_offset:
                                       -sola_search_frame + sola_offset] * fade_out
        else:
            sola_buffer[:] = infer_wav[-crossfade_frame:] * fade_out
        return output_wav.cpu().numpy()

    if args.selftest:
        x = (0.3 * np.sin(2 * np.pi * 150 * np.arange(block_frame) / sr)
             ).astype(np.float32)
        stereo = np.stack([x, x], axis=1)
        worst = 0.0
        for i in range(4):                   # first block includes rmvpe load
            t0 = time.perf_counter()
            out = process(stereo)
            dt = time.perf_counter() - t0
            if i > 0:
                worst = max(worst, dt)
            print(f"selftest block {i}: {dt * 1000:.0f} ms, "
                  f"peak {float(np.abs(out).max()):.3f}", flush=True)
        verdict = "OK" if worst < args.block else "TOO SLOW - raise --block"
        print(f"STATUS selftest {verdict} "
              f"(worst {worst * 1000:.0f} ms vs block {args.block * 1000:.0f} ms)",
              flush=True)
        sys.exit(0)

    in_dev = find_device(args.input_device, "input")
    out_dev = find_device(args.output_device, "output")

    # First inference compiles CUDA kernels (~10 s); do it before the stream
    # opens so live audio starts clean instead of glitching through warm-up.
    print("STATUS warmup", flush=True)
    dummy = np.zeros((block_frame, 2), dtype=np.float32)
    for _ in range(2):
        process(dummy)

    def callback(indata, outdata, frames, times, status):
        try:
            out = process(indata)
            outdata[:] = np.tile(out, (2, 1)).T
        except Exception as e:
            print(f"STATUS error {e}", flush=True)
            outdata[:] = 0

    if sys.stdin is not None:                # VoiceBox feeds TTS over stdin
        threading.Thread(target=_stdin_listener, daemon=True).start()

    print(f"STATUS running sr={sr} block={block_frame} device={device}", flush=True)
    try:
        with sd.Stream(device=(in_dev, out_dev), channels=2, callback=callback,
                       blocksize=block_frame, samplerate=sr, dtype="float32"):
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
