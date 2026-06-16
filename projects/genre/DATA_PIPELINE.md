# Data pipeline — design notes (loader · leakage · table→tensor)

Working notes for the BEARDOWN data tier. Scope: turn the GTZAN artifacts in
`data/raw/` into batched tensors for training, without leaking. Framework target
is TensorFlow (`tf.data`) since BEARDOWN is TF + TFP, but the split/normalize logic
is framework-agnostic. Build order is at the bottom.

## What's on disk (the inputs)

| artifact | shape | rows | notes |
| -------- | ----- | ---- | ----- |
| `features_30_sec.csv` | 60 cols | 1,000 | one row per track |
| `features_3_sec.csv`  | 60 cols | 9,990 | ~10 rows per track (10 tracks have 9) |
| `images_grey_scale/<genre>/<id>.png` | 128×128, mode L | 1,000 | mel spectrogram per track |

CSV columns: `filename`, `length`, 57 audio features (`chroma_stft_mean` …
`mfcc20_var`), `label`. → **57 model features** after dropping `filename`, `label`,
and `length` (per the INFO-510 convention; `length` is constant-ish and carries no
signal). 10 genres × 100 tracks, balanced.

Repaired: `jazz.00054` (wav + grey) patched in place in `raw/`. Documented-not-fixed:
10 tracks with 9 three-sec segments (genuinely short audio — not imputed), 10 hiphop
clips at 30.649 s (untrimmed). See `data/README.md`.

## 1 — Leakage prevention (the load-bearing decision)

**The risk.** The 3-sec table explodes each song into ~10 near-identical segments.
Split those 9,990 rows *randomly* and segments of the same song land in both train
and test → the model recognizes the *song*, not the *genre* → inflated accuracy.
This is the Sturm-2013 GTZAN artifact, and it's exactly the gap the INFO-510 sweep
left open (its holdout was a random row shuffle; it was safe only because it happened
to run on the 1-row-per-track 30-sec table).

**The guard — split on tracks, never on segments.**
- Derive `track_id` from `filename`: `blues.00000.3.wav` → `blues.00000`;
  image `blues00000.png` → normalize to the same id.
- `track_level_split(items, track_id_fn, seed, ratios=(0.7,0.15,0.15))`: collect the
  1,000 track ids, **stratify by genre** (each genre's 100 tracks split 70/15/15),
  then expand each track's assignment to all its rows/segments. Implement this in
  `_shared/splits.py` (currently a stub).
- **Preprocessing leaks too.** Fit the feature scaler and label encoder on the
  **train split only**, then apply to val/test. Computing mean/std over the whole
  table before splitting is a quieter form of the same leak.
- **Self-check, cheap and worth baking in:** after splitting, assert
  `set(train_tracks) ∩ set(val_tracks) ∩ set(test_tracks) == ∅`. Fail loud.

**Measure the gap, don't just avoid it.** Keep `naive_random` (segment-level) in
`splits.py` and run it alongside `track_level_split`. The accuracy delta between the
two *is a result* — the honest "how much does leakage inflate GTZAN numbers" figure.
`artist_split` is the strict ideal but needs artist metadata GTZAN doesn't ship;
leave it stubbed/aspirational.

## 2 — The data loader

One loader, parameterized by representation, so tabular / image / fused share the
split + id logic.

Responsibilities:
- read the chosen CSV (3-sec or 30-sec) and/or resolve image paths;
- attach `track_id` and integer label to every row;
- call `track_level_split` → train/val/test index sets (on track ids);
- fit scaler/encoder on train, transform all;
- emit `tf.data.Dataset` per split: `shuffle` (train only, buffer ≥ split size),
  `batch`, `prefetch(AUTOTUNE)`.

Knobs to expose (the "ancillary options"): `representation ∈ {tab30, tab3, image, fused}`,
`split_strategy ∈ {track, naive}`, `batch`, `seed`, `ratios`, `standardize on/off`,
`augment` (image only). Returning `(dataset, label_map, n, feature_cols)` keeps it
self-describing.

Filename → image path resolution must handle the dotted/undotted mismatch
(`jazz.00054` ↔ `jazz00054.png`) — normalize ids before matching, like the 510
loader's `_normalize_id`.

## 3 — Table → tensor (the concrete transforms)

**Tabular row → vector.**
`row → drop {filename, label, length} → 57 floats → z-score (train μ,σ) → float32 [57]`.
Stack the split → `X_tab : [N, 57]`. Guard NaN/inf (the EDA says zero, but assert it).

**Image → grid.**
`path → PIL open → convert("L") → /255.0 (or z-score) → [128,128] → expand → [128,128,1] float32`.
Stack → `X_img : [N, 128, 128, 1]`.

**Label → target.**
`genre str → label_map[genre] ∈ 0..9`. Keep as int for sparse CE, or `one_hot → [N,10]`.
Build `label_map` once, from the sorted genre list, and persist it (so inference agrees).

**Batch.**
`from_tensor_slices(...) → (shuffle) → batch(B) → prefetch`. Yields
`[B,57]` (tab), `[B,128,128,1]` (image), or `((img, tab), y)` (fused, BEARDOWN's
gated/concat head).

Mental model: the 3-sec table is a bag of independent 3-second "samples," each a
fixed-width feature vector; a song is just however many samples it produced (9 or 10).
Nothing is padded or truncated to a common per-track length — uniformity lives at the
*segment* level, which is already constant.

## Build order (tomorrow)

1. `_shared/splits.py`: implement `track_level_split` + `naive_random` + the
   intersection self-check. Unit-test the no-overlap invariant.
2. Loader skeleton: CSV → ids/labels → split → scaler(train) → `tf.data` (start with
   `representation=tab3`, `split=track`).
3. Add `image` + `fused` representations (path resolve + PIL load).
4. Sanity run: one batch, print shapes, confirm `label_map`, confirm no track overlap.
5. Record the chosen split seed/ratios in `manifest.json` (reproducibility).
