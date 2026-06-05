# `genre/` — GTZAN music-genre recognition (week 2)

The first reproduction. Two models on one dataset (GTZAN), each scored against its
source's reported number.

## Dataset: GTZAN
- 1000 clips, 10 genres × 100, 22050 Hz mono, ~30 s each.
- Tzanetakis & Cook, *Musical Genre Classification of Audio Signals*, IEEE TSAP 2002.
- Three usable representations: raw audio, the 58-feature tabular CSV (30 s **and**
  3 s segments), and Mel-spectrogram images.
- Known issues to document (not ignore): one corrupt file (`jazz.00054`), plus the
  Sturm (2013) critique — exact duplicates, mislabelings, artist repetition.

Full provenance + integrity work lives in [`data/README.md`](data/README.md) and the
emitted [`eda/`](eda/README.md) artifacts.

## The two reproductions

| Config | Model | Source | Target metric | Status |
|--------|-------|--------|---------------|--------|
| `configs/beardown.yaml` | BEARDOWN | _TBD — link repo / = INFO 510 classifier_ | _TBD_ | stub |
| `configs/transformer.yaml` | transformer / attention | _TBD — EAViT / improved-ViT / attention-CNN_ | _paper accuracy_ | stub |

Both share GTZAN ingestion, the track-level split, EDA, the trainer, and the profiler.
They differ only in `features:` and `model:` keys — that's the whole point of the
config+registry pattern.

## Open decisions wired into this folder

1. **BEARDOWN identity** → sets `configs/beardown.yaml: model` + `features` + the
   `eval` target. Until confirmed, `src/models/beardown.py` is a stub.
2. **Transformer paper** → the choice changes the feature tier:
   - EAViT (arXiv 2408.13201): 3-second segmentation, mel input.
   - improved-ViT (PLOS One, 86.8%): mel + channel attention.
   - attention-CNN (arXiv 2411.14474): spectrogram-sequence tokens + MHA.
   Set `configs/transformer.yaml: features` accordingly.
3. **Split policy** → `configs/*.yaml: split.mode` ∈ {`naive_random`, `track`,
   `artist`}. Plan: run `naive_random` (matches most papers) **and** `track` (honest),
   report the gap. The leakage rationale is in `eda/README.md`.

## Run order
```bash
python eda/run_eda.py                                   # integrity + leakage audit first
python src/train.py --config configs/beardown.yaml      # then each reproduction
python src/eval.py  --run runs/beardown/<ts>/
```
