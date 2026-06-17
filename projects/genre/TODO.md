# TODO — data loaders (leakage-safe, sampled, 70/15/15 @ 100% genre coverage)

Next build session. Goal: turn the raw GTZAN artifacts into batched tensors **without
leaking**, with random minibatch sampling and stratified 70/15/15 splits that contain
**every genre in every split**.

"Data loader" = two pieces:
1. **split logic** — which tracks go to train/val/test (`_shared/splits.py`)
2. **loader** — CSV/PNG → ids/labels → split → scale → `tf.data` (`projects/genre/src/dataio.py`)

Design spec lives in `projects/genre/DATA_PIPELINE.md`. This file is the checklist + the
invariants that must hold.

---

## Definition of done

- [ ] `python -m pytest _shared/tests/test_splits.py -q` → all green (no-overlap invariant is the load-bearing test)
- [ ] `python projects/genre/src/sanity_check.py` → prints shapes + label_map; **track split = 0 tracks shared**, **naive split = many shared** (the leakage gap, printed)
- [ ] Same harness against real data: `python projects/genre/src/sanity_check.py --data-root projects/genre/data/raw` (run from repo root)
- [ ] every split contains all 10 genres (coverage = 1.0), no resampling needed
- [ ] chosen `seed` + `ratios` recorded in `manifest.json` via `write_split_to_manifest(...)`

---

## Invariants (do NOT let these drift)

**Leakage protection — split on tracks, never on segments.**
- `track_id` is the canonical `genre.NNNNN`; all ~10 three-sec segments and the image of one
  recording collapse to it. Unit of assignment = the **track**.
- `assert_no_track_overlap(...)` runs inside `track_level_split` and **fails loud** if any
  track id lands in two splits.
- Keep `naive_random` (segment-level) alongside it — not to train on, but to **measure** how
  much GTZAN leakage inflates accuracy (the track-vs-naive delta is a result for the website).
- Fit scaler + label encoder on the **train split only** (computing μ/σ over the whole table
  before splitting is a quieter leak).

**70/15/15 @ 100% genre coverage — via stratification, not a coverage guard.**
- Split each genre's tracks 70/15/15 **independently** → every split holds all 10 genres by
  construction → coverage = 1.0, deterministically.
- This **supersedes** the old `--holdout_min_cov 0.90` resample loop. Don't reintroduce it.
- For n=100 tracks/genre the split is exactly (70,15,15); train absorbs any rounding remainder.

**Random sampling — three distinct things, keep them straight.**
1. *Split-time*: within each genre, tracks are permuted by a seeded RNG (`np.random.default_rng(seed)`)
   then sliced 70/15/15. RV = uniform permutation of tracks per stratum. Deterministic given seed.
2. *Train-time (minibatch)*: `tf.data.shuffle(buffer ≥ |train|, seed).batch(B).prefetch(AUTOTUNE)`.
   Each epoch = a fresh uniform-without-replacement pass. Balanced + stratified data ⇒ batches are
   class-balanced in expectation; no weighting needed.
3. *Weighted sampling*: NOT needed (GTZAN is balanced 100/genre). Leave as an optional future knob
   (inverse-frequency) only if segment-level imbalance ever matters — it's negligible (9 vs 10 segs).

---

## Build order

Drafts of all four files were produced this session — paste them in and verify, or rebuild
from `DATA_PIPELINE.md`. Either way the order is:

1. [ ] `_shared/splits.py` — `track_level_split` + `naive_random` + `assert_no_track_overlap` (+ `artist_split` stub). Pure numpy, no TF.
2. [ ] `_shared/tests/test_splits.py` — unit-test the no-overlap invariant, stratified ratios, determinism, partition coverage.
3. [ ] `projects/genre/src/dataio.py` — `load(representation, split_strategy, seed, ratios, standardize)`; start with `representation="tab3", split="track"`.
4. [ ] add `image` + `fused` representations (PIL load + path-resolve, dotted/undotted id fix).
5. [ ] `sanity_check.py` — one batch per rep: print shapes, confirm `label_map`, assert no track overlap, print the naive-vs-track gap.
6. [ ] record seed/ratios in `manifest.json`.

---

## Files

| path | role | state |
|------|------|-------|
| `projects/genre/DATA_PIPELINE.md` | design spec | written |
| `_shared/splits.py` | split logic (leakage guard) | draft handed off |
| `_shared/tests/test_splits.py` | invariant tests | draft handed off |
| `projects/genre/src/dataio.py` | the loader | draft handed off |
| `projects/genre/src/sanity_check.py` | end-to-end check (synthetic + `--data-root`) | draft handed off |
| `projects/genre/data/raw/` | real GTZAN (gitignored, local only) | on home machine |
| `projects/genre/data/manifest.json` | reproducibility record | exists |

---

## Open decisions (pick before/while building)

- [ ] **57 vs 58 features** — `DATA_PIPELINE.md` drops `length` → 57. Published BEARDOWN used 58 (incl. `length`). Keep 57 (length ≈ constant for 30s clips) and note the deviation, or match 58 for strict repro.
- [ ] **fused pairing** — `fused` currently joins `tab30 ⨝ image` 1:1 per track. Alternative: `tab3 ⨝ image` (image broadcast across a track's ~10 segments) if BEARDOWN's gated head wants segment-level tabular.
- [ ] **image size** — loader knob; native grey PNGs are 128×128, BEARDOWN upsamples to 224×224×1. Default 224 for repro fidelity.

---

## Environment / run

- Core (`splits.py`, the numpy path of `dataio.py`) runs **without TensorFlow**.
- TF is needed only for `to_tf_dataset(...)`. Tests + sanity run work TF-free.
- `pip install pytest` for the tests. `pip install tensorflow tensorflow-probability` when wiring the actual `tf.data` step.
- Run everything from the repo root so the `_shared` import resolves.

```bash
python -m pytest _shared/tests/test_splits.py -q
python projects/genre/src/sanity_check.py                                   # synthetic fixture
python projects/genre/src/sanity_check.py --data-root projects/genre/data/raw   # real GTZAN
```

---

## After this (not now)

Loader → `train.py` → BEARDOWN (`src/models/beardown.py` is still a stub) → k=3 CV + sweep →
emit `run.json` / `compute.json` so the Training panel has something to read.
