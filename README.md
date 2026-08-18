# sanskrit-tts-train

Fine-tuning [Piper](https://github.com/rhasspy/piper) (VITS) to produce a
Sanskrit text-to-speech voice that synthesizes words it never saw during
training.

The exported model is a single **61 MB ONNX file** that runs CPU-only — small
enough for a Raspberry Pi or a free-tier web host.

```bash
$ echo "धन्यवादः" | piper --model sanskrit.onnx --output_file thank_you_in_sanskrit.wav
$ aplay thank_you_in_sanskrit.wav
```

---

## Status

Working prototype. It generalizes to unseen words, but quality is uneven — see
[Known limitations](#known-limitations) before using it for anything that
teaches pronunciation.

|                 |                                  |
| --------------- | -------------------------------- |
| Training corpus | 219 utterances (~15 min)         |
| Base checkpoint | Hindi (`hi_IN`), 807 MB          |
| Trained         | 6000 epochs                      |
| Export          | `sanskrit.onnx`, 61 MB, CPU-only |
| Phonemizer      | espeak-ng, `hi` voice            |
| Sample rate     | 22050 Hz mono                    |

---

## Layout

```
sanskrit-tts-train/
├── dataset/
│   ├── h2r_audio_numbered/     # source audio, numbered to match words.txt
│   ├── wavs/                   # 219 files, resampled to 22050 Hz mono
│   ├── words.txt               # 219 lines, one utterance per line
│   └── metadata.csv            # LJSpeech format: NNN|text
├── piper/                      # cloned rhasspy/piper (see pinned commit below)
├── training/                   # output of piper_train.preprocess
│   ├── config.json             # phoneme map, sample rate, inference defaults
│   ├── dataset.jsonl           # phonemized utterances
│   ├── cache/
│   └── lightning_logs/version_0/checkpoints/
│       └── epoch=5999-step=410976.ckpt
├── hi_base.ckpt                # Hindi base checkpoint (transfer source)
├── sanskrit.onnx               # exported model
├── sanskrit.onnx.json          # inference config (copy of training/config.json)
├── test.wav
├── correct-bhavataḥ-nāma-kim.wav   # sample: good output
└── error-aham-adhyāpakaḥ-asmi.wav  # sample: poor output
```

---

## Corpus design

The 219 utterances are **not random**. They were mined from a 1000-verse
Sanskrit collection to guarantee phonetic coverage:

- **Part A (136 words)** — every vowel in independent and matra form, every
  consonant word-initially and medially, anusvāra, visarga, and ~55 conjuncts
  (क्ष ज्ञ त्र ङ्ग ञ्च ष्ण …). Four gaps the corpus lacked were filled with
  standard words: झटिति (झ), ओम् (ओ-initial), औषधम् (औ-initial), पितॄन् (ॠ).
- **Part B (83 sentences)** — 4–7 words each, for prosody. A model trained only
  on isolated words learns list-reading cadence and no sentence rhythm.

`words.txt` line _N_ corresponds to `wavs/NNN.wav`. `metadata.csv` is generated
from it:

```bash
awk '{printf "%03d|%s\n", NR, $0}' dataset/words.txt > dataset/metadata.csv
```

---

## Environment

The dependency chain is fragile. These exact versions are known to work
together; deviating from them is the most likely cause of a failed setup.

| Package           | Version        | Why pinned                                                   |
| ----------------- | -------------- | ------------------------------------------------------------ |
| Python            | 3.10.12        | 3.12+ breaks the stack                                       |
| **pip**           | **24.0**       | **24.1+ rejects pytorch-lightning 1.7's malformed metadata** |
| setuptools        | 80.10.2        | 81+ removes `pkg_resources`, which Lightning 1.7 imports     |
| torch             | 1.13.1         | piper_train requires `torch<2`                               |
| torchaudio        | 0.13.1         | must match torch                                             |
| torchvision       | 0.14.1         | must match torch                                             |
| pytorch-lightning | 1.7.7          | piper_train pins `~=1.7.0`                                   |
| torchmetrics      | 0.11.4         | later versions break Lightning 1.7                           |
| numpy             | 1.26.4         | numpy 2 breaks Lightning 1.7                                 |
| Cython            | 0.29.37        | builds `monotonic_align`                                     |
| six               | 1.17.0         | torch 1.13 tensorboard writer imports it                     |
| protobuf          | 7.35.1         |                                                              |
| tensorboard       | 2.21.0         |                                                              |
| librosa           | 0.11.0         |                                                              |
| onnxruntime       | 1.23.2         |                                                              |
| piper-phonemize   | 1.1.0          |                                                              |
| piper-tts         | 1.6.0          | inference CLI                                                |
| piper_train       | git `73c04d81` | editable install from the repo                               |
| espeak-ng         | 1.50           | system package                                               |

### Setup

```bash
sudo apt-get install -y espeak-ng espeak-ng-data build-essential

python3.10 -m venv venv
source venv/bin/activate

# CRITICAL: downgrade pip BEFORE installing piper_train
pip install "pip<24.1"

git clone https://github.com/rhasspy/piper.git
cd piper && git checkout 73c04d81d5590ecc46e522de3601ce7fb29fc2be
cd src/python && pip install -e .

pip install piper-phonemize torchmetrics==0.11.4
pip install "numpy<2" "setuptools<81" six protobuf tensorboard
pip install torch==1.13.1 torchaudio==0.13.1 torchvision==0.14.1

# Cython extension — pip install does NOT build this
./build_monotonic_align.sh
```

Verify before proceeding:

```bash
python -c "import torch, piper_train; print(torch.__version__, 'ok')"
python -c "from piper_train.vits import monotonic_align; print('monotonic_align ok')"
espeak-ng -v hi --ipa "नमस्ते"
```

### Setup errors and their causes

| Error                                                                   | Cause                                | Fix                           |
| ----------------------------------------------------------------------- | ------------------------------------ | ----------------------------- |
| `No matching distribution found for pytorch-lightning~=1.7.0`           | pip ≥ 24.1 enforces PEP 508 strictly | `pip install "pip<24.1"`      |
| `ModuleNotFoundError: pkg_resources`                                    | setuptools ≥ 81                      | `pip install "setuptools<81"` |
| `ModuleNotFoundError: six`                                              | torch 1.13 tensorboard writer        | `pip install six`             |
| `ModuleNotFoundError: piper_train.vits.monotonic_align.monotonic_align` | Cython ext not built                 | `./build_monotonic_align.sh`  |
| `praat-parselmouth` / `cp314` wheels                                    | wrong Python version                 | use 3.10                      |

---

## Reproducing the training

```bash
source venv/bin/activate

# 1. Resample source audio to 22050 Hz mono
for f in dataset/h2r_audio_numbered/*.wav; do
  ffmpeg -loglevel error -i "$f" -ar 22050 -ac 1 -sample_fmt s16 \
    "dataset/wavs/$(basename "$f")"
done

# 2. Build metadata.csv from words.txt
awk '{printf "%03d|%s\n", NR, $0}' dataset/words.txt > dataset/metadata.csv

# 3. Preprocess (phonemize + cache)
python -m piper_train.preprocess \
  --language hi \
  --input-dir dataset \
  --output-dir training \
  --dataset-format ljspeech \
  --single-speaker \
  --sample-rate 22050

# 4. Fine-tune from the Hindi base
python -m piper_train \
  --dataset-dir training \
  --accelerator gpu --devices 1 \
  --batch-size 12 \
  --validation-split 0.05 \
  --num-test-examples 3 \
  --max_epochs 6000 \
  --checkpoint-epochs 200 \
  --resume_from_checkpoint hi_base.ckpt \
  --quality medium \
  --precision 32

# 5. Export
python -m piper_train.export_onnx \
  training/lightning_logs/version_0/checkpoints/epoch=5999-step=410976.ckpt \
  sanskrit.onnx
cp training/config.json sanskrit.onnx.json
```

The Hindi base checkpoint had already trained to ~epoch 3199, so Lightning
resumes the counter there and `--max_epochs 6000` trains roughly 2800 further
epochs.

Run training under `tmux` or `screen` — it takes hours and a closed terminal
kills it.

---

## Usage

```bash
pip install piper-tts
echo "नमस्ते" | piper --model sanskrit.onnx --output_file out.wav
```

### Slowing it down

The default `length_scale: 1` is too fast for Sanskrit, which is
quantity-timed. Raise it:

```bash
echo "नमस्ते" | piper --model sanskrit.onnx --length-scale 1.4 -f out.wav
```

Or make it permanent by editing the `inference` block of `sanskrit.onnx.json`:

```json
"inference": { "noise_scale": 0.667, "length_scale": 1.4, "noise_w": 0.8 }
```

Related knobs: `--noise-scale` (pitch variation), `--noise-w-scale` (rhythm
variation), `--sentence-silence` (pause between sentences).

### Verse rhythm

Danda (।) is not treated as a break. Converting it to a comma or period before
synthesis, together with `--sentence-silence 0.6`, gives pāda-boundary pauses
and noticeably less run-on delivery.

---

## Known limitations

**Trained on synthesized audio.** The source recordings in
`dataset/h2r_audio_numbered/` came from an online Sanskrit TTS, not a human
speaker. A model cannot exceed its teacher, so this inherits that voice's
pronunciation errors. Retraining on recorded human speech is the single largest
available improvement.

**Phonemized through Hindi.** espeak-ng has no Sanskrit voice, so `hi` is used.
`config.json` confirms the phoneme inventory is generic IPA — there is no
dedicated symbol for visarga or vocalic ṛ, and Hindi conventions (schwa
deletion, ऋ → "ri", weak visarga) are applied to Sanskrit. This is the ceiling
on pronunciation accuracy and cannot be fixed by more training.

**No prosody.** Sanskrit rhythm rests on मात्रा (syllable weight) and स्वर
(pitch accent); neither survives Hindi phonemization. Output is intelligible
but metrically flat.

**Small corpus.** 219 utterances (~15 min) is below the ~30 min usually wanted
for a fine-tune. Expect uneven quality across phonemes.

**Only the final checkpoint was kept.** `--checkpoint-epochs 200` overwrote
rather than accumulated, so mid-training checkpoints are unavailable for
comparison. Pass `--save-top-k -1` if you want to keep them — with a corpus this
small the best-sounding model is often not the last epoch.

---

## Next steps

1. Retrain on human-recorded audio (largest quality gain available).
2. Test Vedic accent marks (U+0951, U+0952) to see whether espeak-hi responds.
3. Evaluate output against reference recordings using acoustic scoring rather
   than ear alone.

---

## Files not tracked in git

`hi_base.ckpt` (807 MB), `training/lightning_logs/`, `training/cache/`, and
`piper/` are large or regenerable. A suggested `.gitignore`:

```
venv/
piper/
hi_base.ckpt
training/cache/
training/lightning_logs/
dataset/wavs/
*.pyc
__pycache__/
```

Keep `dataset/words.txt`, `dataset/metadata.csv`, `sanskrit.onnx`,
`sanskrit.onnx.json`, and `training/config.json` — those are what make the run
reproducible.
