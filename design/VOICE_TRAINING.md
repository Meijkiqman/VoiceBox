# Your own AI voice: harvest + retrain

VoiceBox can collect training data from your real mic while you play, and
retrain the RVC model of your own voice from it. The result: the speech
translator (and TTS) come out of the cable sounding like *you*.

## The loop

1. **Voice harvest** (menu row) - leave it ON while you use VoiceBox.
   Every time you speak, the raw mic signal (before any effects) is
   segmented into clips. While the mic is muted (including push-to-talk
   idle) nothing is collected - off-air speech stays off the record. Clips that are too short, too quiet or distorted
   are thrown away; the rest are normalized and saved to
   `rvc/dataset_self/` as 48 kHz mono wavs. The row shows how many minutes
   you have.
2. **Retrain AI voice** (menu row) - once you have 5+ minutes (30-45 is the
   sweet spot), press it. Training runs in its own console window on your
   GPU (`rvc_trainer.py`); the AI voice must be OFF while it runs. Expect
   roughly 20-60 minutes on a mid-range NVIDIA card.
3. The new model lands in `rvc/weights/MyVoice*.pth` with its `.index` in
   `rvc/logs/MyVoice/`. It appears in the **AI character** row on next
   launch. Keep your previous `.pth` files - if a retrain sounds worse
   (it happens), just switch back.

Collection stops by itself at 60 minutes: RVC stops improving with more
data long before that, and hours of raw gaming audio make models worse,
not better. To start a fresh dataset, delete wavs from `rvc/dataset_self/`.

## Train a brand-new voice ("Train new model")

For a voice that isn't yours - a character, a friend who donated clips -
use the **Train new model** row instead of the harvest loop:

1. Drop audio clips of the target voice into the `training/` folder
   (created next to `VoiceBox.bat`; wav/flac/ogg/mp3 - clean speech, no
   music or crowds).
2. Press **Train new model**, type a name for the model, and pick the
   clips in the file dialog (it opens in `training/`).
3. That's it - the clips are converted into an RVC dataset
   (`rvc/dataset_<name>/`, mono 16-bit wavs in <=12 s pieces) and training
   starts by itself in its own console window, exactly like a retrain.

You need at least ~2 minutes of usable audio (10 pieces) or the row tells
you to add more; 10-30 minutes is the sweet spot. When training finishes,
the model joins the **AI character** row right away - no restart. Names
collide on purpose: picking an existing model's name is refused so a new
model can't silently eat an old one.

## Training pieces (one-time setup)

The trimmed RVC package that ships with VoiceBox contains the *inference*
runtime. Training additionally needs these, all from the full
[RVC-beta0717 zip](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/releases),
copied into your `rvc/` folder:

```
trainset_preprocess_pipeline_print.py
extract_f0_print.py            (and/or extract_f0_rmvpe.py)
extract_feature_print.py
train_nsf_sim_cache_sid_load_pretrain.py
configs/                       (training configs)
logs/mute/                     (silence reference files)
pretrained_v2/f0G40k.pth       (base models training starts from)
pretrained_v2/f0D40k.pth
```

Check what's missing at any time - from inside the `rvc/` folder (the
trainer resolves everything relative to it):

```
cd rvc
runtime\python.exe ..\rvc_trainer.py --check --dataset dataset_self
```

The "Retrain AI voice" row runs the same script, so it will also tell you.

## Status: EXPERIMENTAL

The trainer drives RVC's own training scripts exactly the way the RVC
WebUI does, but RVC-beta builds vary. If a step fails, the console window
says which one; training the same dataset in the RVC WebUI's Train tab
(same folders, experiment name `MyVoice`, v2, 40k, pitch guidance on) is
the reliable fallback - the result lands in the same place and VoiceBox
picks it up the same way.

## Tips for a passable -> good model

- Talk normally, at your normal distance from the mic. The harvester's
  quality gate can't fix a clipping gain knob - if the mic meter is
  red-lining, turn the input gain down.
- Variety beats volume: normal chat, some louder moments, some quiet ones.
- Noise suppression OFF while harvesting (it smears the timbre RVC learns).
- Re-running "Retrain AI voice" continues from the last checkpoints; it
  won't start from zero each time. Pass `--epochs 300` by hand to push a
  finished run further.
