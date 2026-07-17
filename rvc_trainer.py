"""rvc_trainer.py - train/refresh an RVC voice model from a local dataset,
driven by VoiceBox's "Retrain AI voice" row (or by hand).

EXPERIMENTAL. This orchestrates the RVC-beta0717 training pipeline the same
way the RVC WebUI's Train tab does, but headless. It MUST be run with RVC's
own bundled interpreter and cwd set to the RVC folder:

    <RVC>\\runtime\\python.exe rvc_trainer.py --dataset dataset_self --name MyVoice

Requirements inside the RVC folder (the trimmed inference-only package that
ships with VoiceBox does not include all of them - copy them in from the
full RVC-beta0717 zip if --check complains):
    trainset_preprocess_pipeline_print.py     (preprocess)
    extract_f0_print.py / extract_f0_rmvpe.py (pitch extraction)
    extract_feature_print.py                  (HuBERT features)
    train_nsf_sim_cache_sid_load_pretrain.py  (the trainer)
    configs/, logs/mute/, hubert_base.pt
    pretrained_v2/f0G40k.pth + f0D40k.pth     (v2 base models)

Steps: preprocess -> f0 -> features -> filelist/config -> train -> faiss
index. Progress prints straight to the console; VoiceBox launches this in
its own window so you can watch. Re-running with a bigger --epochs resumes
from the checkpoints in logs/<name>/ instead of starting over.

The finished model lands in weights/<name>*.pth (newest = best) and the
matching .index in logs/<name>/ - exactly where VoiceBox looks for AI
voices, so it shows up in the AI character row on next launch."""
import argparse
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

def sh(cmd):
    print("\n>>> " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode != 0:
        sys.exit(f"step failed (exit {r.returncode}): {cmd[1]}")


def newest(pattern):
    files = sorted(Path(".").glob(pattern), key=lambda f: f.stat().st_mtime)
    return files[-1] if files else None


def check_requirements(version):
    pre = "pretrained_v2" if version == "v2" else "pretrained"
    needed = [
        "trainset_preprocess_pipeline_print.py",
        "extract_feature_print.py",
        "train_nsf_sim_cache_sid_load_pretrain.py",
        "hubert_base.pt",
        f"{pre}/f0G40k.pth",
        f"{pre}/f0D40k.pth",
        "logs/mute",
        "configs",
    ]
    if not (Path("extract_f0_rmvpe.py").is_file()
            or Path("extract_f0_print.py").is_file()):
        needed.append("extract_f0_print.py")
    missing = [n for n in needed if not Path(n).exists()]
    if missing:
        print("MISSING from this RVC folder (copy from the full "
              "RVC-beta0717 zip):")
        for n in missing:
            print("   " + n)
        return False
    print("all training pieces found.")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="folder of training wavs (e.g. dataset_self)")
    ap.add_argument("--name", default="MyVoice", help="model/experiment name")
    ap.add_argument("--epochs", type=int, default=200,
                    help="total epochs; re-run with a bigger number to "
                         "continue training")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--sr", default="40k", choices=["32k", "40k", "48k"])
    ap.add_argument("--version", default="v2", choices=["v1", "v2"])
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--check", action="store_true",
                    help="only verify the training pieces exist")
    args = ap.parse_args()

    if not check_requirements(args.version):
        sys.exit(1)
    if args.check:
        return

    dataset = Path(args.dataset)
    wavs = sorted(dataset.glob("*.wav")) if dataset.is_dir() else []
    if len(wavs) < 10:
        sys.exit(f"dataset {dataset} has {len(wavs)} wavs - need at least 10 "
                 "(a few minutes of speech). Turn on 'Voice harvest' in "
                 "VoiceBox and talk for a while.")
    total_s = sum(max(0, f.stat().st_size - 44) / (48000 * 2) for f in wavs)
    print(f"dataset: {len(wavs)} clips, ~{total_s / 60:.1f} min")

    exp = Path("logs") / args.name
    exp.mkdir(parents=True, exist_ok=True)
    sr_int = {"32k": 32000, "40k": 40000, "48k": 48000}[args.sr]
    n_cpu = max(1, (os.cpu_count() or 4) // 2)

    try:
        import torch
        gpu = torch.cuda.is_available()
    except Exception:
        gpu = False
    if not gpu:
        print("WARNING: no CUDA GPU visible - training will be very slow.")
    device = "cuda:0" if gpu else "cpu"

    py = sys.executable

    # 1) preprocess: slice/normalize the dataset into 0_gt_wavs + 1_16k_wavs
    sh([py, "trainset_preprocess_pipeline_print.py", dataset, sr_int, n_cpu,
        exp, "False"])

    # 2) pitch extraction (rmvpe when the script for it exists, else harvest)
    if Path("extract_f0_rmvpe.py").is_file():
        sh([py, "extract_f0_rmvpe.py", 1, 0, 0, exp, "True" if gpu else "False"])
    else:
        sh([py, "extract_f0_print.py", exp, n_cpu, "harvest"])

    # 3) HuBERT features
    sh([py, "extract_feature_print.py", device, 1, 0, 0, exp, args.version])

    # 4) filelist + config, mirroring the WebUI's click_train()
    feat = "3_feature768" if args.version == "v2" else "3_feature256"
    names = ({f.stem for f in (exp / "0_gt_wavs").glob("*.wav")}
             & {f.stem for f in (exp / feat).glob("*.npy")}
             & {f.name[:-len(".wav.npy")] for f in (exp / "2a_f0").glob("*.wav.npy")}
             & {f.name[:-len(".wav.npy")] for f in (exp / "2b-f0nsf").glob("*.wav.npy")})
    if not names:
        sys.exit("preprocessing produced no usable segments - check the "
                 "dataset wavs (48k mono speech).")
    lines = [f"{exp}/0_gt_wavs/{n}.wav|{exp}/{feat}/{n}.npy"
             f"|{exp}/2a_f0/{n}.wav.npy|{exp}/2b-f0nsf/{n}.wav.npy|0"
             for n in sorted(names)]
    mute = Path("logs/mute")
    for _ in range(2):
        lines.append(f"{mute}/0_gt_wavs/mute{args.sr}.wav|{mute}/{feat}/mute.npy"
                     f"|{mute}/2a_f0/mute.wav.npy|{mute}/2b-f0nsf/mute.wav.npy|0")
    random.shuffle(lines)
    (exp / "filelist.txt").write_text("\n".join(lines))
    cfg = None
    for cand in (f"configs/{args.sr}_{args.version}.json",
                 f"configs/{args.version}/{args.sr}.json",
                 f"configs/{args.sr}.json"):
        if Path(cand).is_file():
            cfg = cand
            break
    if cfg is None:
        sys.exit("no matching configs/*.json for "
                 f"{args.sr} {args.version}")
    shutil.copyfile(cfg, exp / "config.json")
    print(f"filelist: {len(lines)} entries, config: {cfg}")

    # 5) train (resumes from logs/<name>/G_*.pth automatically when present)
    pre = "pretrained_v2" if args.version == "v2" else "pretrained"
    sh([py, "train_nsf_sim_cache_sid_load_pretrain.py",
        "-e", args.name, "-sr", args.sr, "-f0", 1, "-bs", args.batch,
        "-g", 0 if gpu else "", "-te", args.epochs, "-se", args.save_every,
        "-pg", f"{pre}/f0G{args.sr}.pth", "-pd", f"{pre}/f0D{args.sr}.pth",
        "-l", 1, "-c", 0, "-sw", 1, "-v", args.version])

    # 6) similarity index (what the .index file next to a model is)
    try:
        import faiss
        import numpy as np
        npys = sorted((exp / feat).glob("*.npy"))
        big = np.concatenate([np.load(str(f)) for f in npys], axis=0)
        np.random.shuffle(big)
        if big.shape[0] > 2e5:
            print(f"index: {big.shape[0]} rows is a lot - sampling 200k")
            big = big[: int(2e5)]
        dim = big.shape[1]
        n_ivf = min(int(16 * np.sqrt(big.shape[0])), big.shape[0] // 39)
        index = faiss.index_factory(dim, f"IVF{n_ivf},Flat")
        index.train(big.astype(np.float32))
        for i in range(0, big.shape[0], 8192):
            index.add(big[i:i + 8192].astype(np.float32))
        out = exp / (f"added_IVF{n_ivf}_Flat_nprobe_1_{args.name}"
                     f"_{args.version}.index")
        faiss.write_index(index, str(out))
        print(f"index written: {out}")
    except Exception as e:
        print(f"index build skipped ({e}) - the voice still works, just "
              "without the accent-lookup index.")

    w = newest(f"weights/{args.name}*.pth")
    if w:
        print(f"\nDONE. model: {w}  - it appears in VoiceBox's AI character "
              "row on next launch (or after re-selecting the AI voice).")
    else:
        print("\nTraining finished but no weights/<name>*.pth was written - "
              "open the RVC WebUI's ckpt tab to extract the small model from "
              f"{exp}/G_*.pth, or re-run with --epochs a multiple of "
              "--save-every.")


if __name__ == "__main__":
    main()
