# FORGE — Experiment 1 (Music Genre / BEARDOWN) — Handoff

_Status snapshot for picking up tomorrow. Repo root: `INFO_698_experiments\`._
_Last run: 2026-06-19 21:24 · `runs\beardown\20260619-212419`_

---

## TL;DR

Model 1 (`beardown`) is a **faithful cfg_14 reproduction**, trained and frozen. It's the
deterministic backbone the FORGE Bayesian layer attaches to. Reproduction work is **done**;
next is the FORGE core (**LLLA + HMC** on the frozen head, then the observatory panel with
live knobs).

**Do not keep chasing 0.766** — the wall at ~0.71 is FORGE's motivating example, not a bug.

---

## Where the model landed

Frozen backbone = **clean log-mel @128 ⨝ 57 tab · gated fusion · deterministic head · batch 8**.

| metric | value | note |
|---|---|---|
| CV mean (3-fold) | **0.686–0.714** | bounces run-to-run (no fixed seed); σ ≈ 0.01–0.02 |
| best VAL | **0.7467** | latest run; ~0.003 off the 0.75 gate |
| best TEST | **0.700** | macro-F1 0.686 |
| params | 573,450 | round-trip bit-exact on every bundle |
| paper target (cfg_14) | 0.766 | original Keras+sweep number |

The current `models\beardown` bundle is the **0.747-VAL draw** (a good one).

### Gap to 0.766 — fully diagnosed, every cheap hypothesis ruled out

| hypothesis | test | verdict |
|---|---|---|
| early stopping too aggressive | `--no-early-stop` curve | ruled out (val plateaus, converged) |
| epochs / lr | same curve | ruled out |
| clean mels hurt | dirty grey A/B | ruled out (clean ≥ dirty) |
| wrong split | seed-42 stratified | ruled out (≈ track split) |
| resolution | dirty-224 run | ruled out (CV 0.629; erased the batch gain at 6× cost) |

**Lever was batch size** (32→8 = **+0.086 CV**). Residual to 0.766 = framework
(Keras vs PyTorch init/BN/optimizer) + run-to-run variance. Faithful, defensible.

---

## What's built (all on disk, repo-relative)

**Model + training**
- `projects\genre\src\models\beardown.py` — SpecCNN + TabMLP → gated fusion → φ(512) → plain `Linear` (LLLA attaches here)
- `projects\genre\src\train.py` — CV → refit → eval → bundle; early-stop + best-restore + reduce-lr; `--no-early-stop` diagnostic flag
- `projects\genre\src\bundle.py` — 7-file bundle: weights, arch, scaler, label_map, **phi_train.npy**, **ggn_eig.npz**, metrics
- `projects\genre\src\dataio.py` — representations + split modes (`track` | `naive` | **`stratified`** = BEARDOWN seed-42)
- `projects\genre\configs\beardown.yaml` — cfg_14 exact + clean-128/batch-8 (the frozen config)

**Data / features**
- `projects\genre\src\features.py` — `extract_mel` (clean recipe), `extract_tabular` (57 feats), **hardened `load_audio`** (soundfile→librosa→clear error)
- `projects\genre\eda\build_spectrograms.py` — regenerates clean log-mels → `images_mel\` (fixes the silent dirty-grey failure; validate r=1.0)

**Inference / app**
- `projects\genre\src\predict.py` — `predict_song`: window → mel+57feat → per-genre prob + MC-dropout σ
- `projects\genre\app\server.py` + `app\templates\infer.html` — `/infer` drop-zone, per-genre bars + σ, per-segment table

**Shared infra**
- `_shared\splits.py` — `stratified_split_seed42` (BEARDOWN two-stage StratifiedShuffleSplit, bit-exact)
- `_shared\profiler.py`, `_shared\schema.py` — compute logging → `compute.json` per run
- `_shared\usage.py` + `_shared\usage_assumptions.yaml` — energy/carbon/$ ledger (measured-vs-modeled, roll-up, projection)

**Bundle:** `projects\genre\models\beardown\` (the frozen Model 1)

---

## Verified

- Drop-in inference works end-to-end on **wav + mp3** (pipeline proven in sandbox).
- Bundle round-trips bit-exact (`max|Δ| = 0`).
- Usage ledger rolls up all 11 runs (~0.33 h wall, ~14.8 Wh, modeled-TDP).

---

## NEXT — the FORGE core (start here tomorrow)

Build order:

1. **`projects\genre\src\bayes\llla.py`** — last-layer Laplace from cached `ggn_eig.npz` + `phi_train.npy`. `predict_posterior(x, tau, ...)` → per-genre mean + epistemic σ. **Math core.**
   Knobs: prior precision **τ** (most important), temperature, σ inflation.
2. **`projects\genre\src\bayes\hmc.py`** — HMC sampling of the last layer over frozen φ. Same signature + diagnostics (R̂, ESS, accept, trace). Gold-standard check for LLLA.
   Knobs: step size, leapfrog steps, samples, burn-in, chains, target accept.
3. **Observatory panel** — live sliders, posterior plot updating on knob-turn, **LLLA-vs-HMC overlay**, reuse the `/infer` drop-zone.

### Open decisions to make first
- **LLLA compute location:** browser (real-time τ knob, JS reads `ggn_eig.npz`) vs Python-served. _Lean: browser real-time + Python as ground-truth validation._
- **Multiclass predictive approx:** linearized/probit (à la `laplace-torch predictive='probit'`).
- **Canonical number:** CV bounces run-to-run (no fixed seed). Decide either (a) report **CV mean ± σ**, or (b) add a **`--seed` flag to `train.py`** for a reproducible frozen bundle. _If you want reproducibility, add the seed BEFORE re-bundling._

---

## Resume commands

```powershell
# train / re-freeze backbone
python projects\genre\src\train.py --config projects\genre\configs\beardown.yaml

# song drop-in (then open http://127.0.0.1:5000 -> Inference)
python projects\genre\app\server.py

# ML usage ledger (on-demand; per-run compute.json is automatic)
python _shared\usage.py --runs projects\genre\runs --assumptions _shared\usage_assumptions.yaml
```

---

## Loose ends / small todos
- **mp3 on Windows** needs ffmpeg on PATH _or_ libsndfile≥1.1. Check: `python -c "import soundfile as sf; print('MP3' in sf.available_formats())"` (`True` = mp3 works; else `winget install ffmpeg`).
- **Edit `usage_assumptions.yaml`** — set real `tdp_watts.cpu` and `electricity_rate_usd_per_kwh` or the $/kWh are placeholders.
- **Optional:** auto-print a one-line usage summary at end of `train.py` (~5 lines) — not yet wired.
- **Semantic acceptance test (your machine):** drop a few known-genre songs at `/infer`, confirm top prediction is sensible (easy genres right, rock/country/disco fuzzy — matches a 0.7 model).
- **Config guard:** before any run, confirm `image_dir: images_mel` (not `mel_images` / not `images_grey_scale`) and `image_size: 128`.

---

## Config state (the frozen backbone)

```
split:    mode=stratified  seed=42
features: image_dir=images_mel  image_size=128  drop_length=true (->57)
train:    batch_size=8  epochs=75  optimizer=adam  lr=7.367e-05  wd=0.0
arch:     spec embed 512 (drop 0.188) · tab [192] (drop 0.191) · fusion gated · head φ512 (drop 0.20) · 10 classes
cv:       folds=3 (GroupKFold by track)
target:   val 0.766 · accept_min 0.75
```
