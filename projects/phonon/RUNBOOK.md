# FORGE · phonon · MODEL 1 — Runbook (faithful Chen et al. 2021 replication)

**What this is:** the deterministic baseline. E(3)-equivariant network (e3nn
`gate_points_2101`) predicting the 51-bin phonon DOS from crystal structure, pinned to
the authors' released reference checkpoint. Bayesian (LLLA), info-theory metrics, and
sweeps layer on top of this in models 2 and 3 — this run is the fidelity anchor.

## 0. Data (already in place)
`data/raw/` holds `phdos_e3nn_len51max1000_fwin101ord3.pkl`, `trteva_indices.pkl`
(1220/152/152), and `reference_model.torch` (authors' checkpoint, for parity checks).

## 1. Env
```bash
pip install -r requirements_phonon.txt
```
torch_cluster / torch_scatter are deliberately **not** needed (ASE builds the periodic
graph; pooling is a segment-mean). Verified on torch 2.12 / e3nn 0.6 / torch_geometric 2.8.

## 2. Smoke test (CPU, ~30 s)
```bash
python -m src.train --data-dir data/raw --limit 120 --epochs 2 --run-name smoke
```
Confirms graphs build, model forwards (2,456,312 params), loss drops, scorecard prints.

## 3. Full reproduction (GPU box — ARES / atlng01)
```bash
python -m src.train --data-dir data/raw --epochs 64 --run-name e3nn_repro --out runs/
```
~64 epochs, AdamW(lr=0.005, wd=0.05), ExponentialLR(γ=0.96), MSE. Reference wall was
~90 min on an entry GPU. Checkpoints land at `runs/e3nn_repro.torch` (state + history,
same schema as the authors' file).

## 4. Score it
```bash
python -m src.eval --data-dir data/raw --ckpt runs/e3nn_repro.torch
```
Reports the reproduction targets and writes `runs/e3nn_repro.scorecard.json`:
- `omega_bar_within_10pct`  — paper headline, **target 0.70** (Fig 2c)
- `corr_mse_natoms`         — should be ≈ 0, no size bias (Fig 2a)
- `mean_js`, `mean_emd1_cm` — info-theory diagnostics, seeded for model 3
- `mean_spectral_entropy_abs_err`

## 5. Parity check against the authors' checkpoint (optional)
`reference_model.torch` was trained with the OLD e3nn API (different state-dict layout),
so you can't `load_state_dict` it into this modern network directly. Use it for *target
numbers* (its final valid MSE ≈ 0.026, mean_abs ≈ 0.084), not weight transfer.

## Gotchas
- The network mutates `data.x` / `data.z` **in place** (embeds 118→64). Fine in a normal
  DataLoader loop (fresh batch each step). For repeated forwards on one cached input
  (e.g. the LLLA stage), **clone the batch first** or you'll double-embed (118×64 vs 64).
- `float64` throughout (`torch.set_default_dtype`) — matches the reference.
- `num_neighbors` is computed from the TRAIN split each run (≈ 40 over the full set).

## Files
```
src/data.py     CIF-lines -> ASE Atoms -> periodic graph; loader + splits + encodings
src/model.py    vendored e3nn Network + PeriodicNetwork; build_model(cfg)
src/train.py    AdamW/MSE/ExpLR loop, progress bars, checkpointing, scorecard
src/eval.py     standalone scorecard -> JSON
src/metrics.py  omega_bar, frac_within (paper); spectral_entropy/JS/EMD (model 3)
configs/e3nn_repro.yaml   exact reference kwargs
```
