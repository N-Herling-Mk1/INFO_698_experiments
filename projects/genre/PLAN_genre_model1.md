# Genre recognition — Model 1 build plan

Working plan for the faithful BEARDOWN reproduction (Model 1, the floor) plus the
machinery the FORGE panels need around it. Picked up after the data tier + Data/Model
panels are in. Naming, file paths, and the Tier R→S JSON contract are fixed below so
tomorrow is wiring, not deciding.

Locked decisions (from the signed proposal + our design pass):
- **PyTorch**, deterministic backbone, **only the final layer Bayesian**.
- Bayesian layer = **last-layer Laplace (LLLA, `laplace-torch`)** for the live path,
  **HMC/NUTS** for ground truth.
- Model 1 selection = **standard metrics** (match the source config, hit the target).
  Model 2 selection = **RRM** over the *same* sweep. (One sweep, two selections.)
- Model run names (must match the Model-panel status dots): `beardown` (M1),
  `beardown_rrm` (M2), `beardown_3sec` (M3).

---

## 0. Contracts (don't drift)

**Bundle** — what `train.py` writes so a model is born plug-in-ready for FORGE:

```
projects/genre/models/<model>/
  weights.pt          # deterministic backbone + final Linear (state_dict)
  arch.json           # layer spec -> drives the network-structure SVG + rebuild
  scaler.json         # train-fit mean/std/cols  (from dataio.Loaded.scaler)
  label_map.json      # genre -> 0..9            (inference must agree exactly)
  phi_train.npy       # cached penultimate features φ(x_train)  [N, d]
  ggn_eig.npz         # Λ (eigenvalues) + U (eigenvectors) of the last-layer GGN/Fisher
  metrics.json        # standard + reliability metrics (for the metrics wall)
```

`phi_train.npy` + `ggn_eig.npz` are the precompute that makes the LLLA knob instant
(see §Resource logging and §5). Without them the "real-time posterior" isn't real-time.

**Run record** — unchanged, already in `_shared/schema.py`:
`runs/<model>/<ts>/run.json` + `compute.json`, surfaced by `/api/runs`. The Model panel
already watches this; a model flips to "delivered" when its `run.json` lands.

**Prereq swap:** `dataio.to_tf_dataset` → `to_torch_loader` (numpy core unchanged; just
wrap `Loaded.X/.y` in a `TensorDataset`+`DataLoader`). Small, do it first.

---

## Resource logging (Feature F5) — fill-in

**Purpose (from the writeup):** turn Phase-2 infrastructure sizing from a guess into an
empirical fit. Everything FORGE deploys has a cost; F5 measures it now so the
always-on CPU box and the serverless GPU can be sized from data, not vibes.

**What it must measure — split by compute path**, because the deployed cost is dominated
by *inference*, not training:

| path | when it runs | budget (proposal) | sizes… |
|------|--------------|-------------------|--------|
| `train`        | offline, per config | — | n/a (local) |
| `llla_fit`     | once per model load | — | RAM of the always-on box |
| `llla_knob`    | per slider tick     | ≤ 1 s (ceiling 3 s) | CPU box responsiveness |
| `predict`      | per song drop       | interactive | CPU box |
| `hmc_resample` | per ground-truth job| ~1 min (ceiling 5 min) | serverless GPU |

**`_shared/profiler.py` — finish the stub.** `_Profiler` already has the skeleton; fill:
- `tick(n)`: `self._samples += n`; sample `psutil.Process().memory_info().rss` →
  `peak_rss`; if CUDA, `torch.cuda.max_memory_allocated()` → `peak_vram`.
- `record()`: wall, throughput, peak_rss_mb, peak_vram_mb (None on CPU), device,
  `n_params`, `est_flops` (set from a `profile_model(model, sample)` helper using
  `thop.profile` or a manual param count + one timed forward).
- add a lightweight `with prof.block("llla_knob"): ...` that appends
  `{label, wall_s, peak_rss_mb, peak_vram_mb, n}` to a `ComputeRecord.ops` list, so the
  four paths above are timed individually, not just the whole run.

**The cost curve (the F5 payoff).** The random sweep (§1) trains ~20 configs of *varying
size* — that spread is exactly the data to fit:

```
wall ≈ a + b·n_params + c·n_samples      (per device: cpu / cuda separately)
```

Add `CostModel.fit(records)` (least-squares over the sweep's `compute.json`s) and
`CostModel.project(n_params, n_samples)`. Emit `projects/genre/data/resource_report.json`
= per-config points + fitted coefficients + the projected always-on-CPU RAM (from
`peak_rss` of `llla_fit`+`predict`) and serverless-GPU VRAM/time (from `hmc_resample`).
That file feeds the deploy-sizing panel and the proposal's Table 6/7. **So the sweep is
also the resource-profiling campaign — one run, both deliverables.**

---

## The 8 steps → files & build order

### 1. Random sweep, standard-metric selection  *(your point 1+2)*
- `configs/beardown.yaml`: fixed bits (epochs, batch, optimizer). Add a `sweep:` block
  with the **BEARDOWN HP ranges from the writeup** (slide_7 `original_HP_ranges.png`):
  lr, dropout, conv filter counts, dense units, fusion type, etc. → *pull exact ranges
  from the report.* Anchor the search on **cfg_14** (the 0.766 config) so Model 1 is a
  faithful repro, not a fresh search.
- `src/sweep.py`: sample ~20 configs × **3-fold CV** (mirrors mk5l/mk5m). For each:
  train (profiled), compute standard metrics, write `run.json`/`compute.json`, build the
  bundle. **Select Model 1 = best by accuracy/F1** hitting **val acc ≥ 0.75**.
- Same sweep feeds Model 2 later: **select by RRM** (no retrain).

### 2. Hyperparameters from the writeup  *(point 2)*
- Single source of truth for cuts/ranges in `configs/beardown.yaml` (like HELIX's
  `config.py` pattern). Record the chosen Model-1 config into `arch.json` + `run.json`.

### 3. Song drop-in → per-genre prediction  *(point 3)*
- `src/features.py` (currently stub): implement `extract_mel` + `extract_tabular_58`
  with **the same librosa params GTZAN used** (n_mels, hop, etc.) so a dropped song lands
  in the *training distribution* — mismatch here silently wrecks scores.
- Decomposition = "testable elements": load audio (22.05 kHz mono) → window into 30-s
  segments (and 3-s for M3) → per window: mel → image tensor **+** 58 features → scale
  with the bundle's `scaler.json` → fused model → per-window softmax + epistemic σ.
- Aggregate windows → song-level per-genre scores (mean softmax) + aggregate uncertainty.
- Lives behind the proposal's uniform `/api/predict` contract; UI on the **Inference** panel
  (`/infer`, currently stub): drop zone → per-genre bars w/ uncertainty + per-segment table.

### 4. Log resource usage as it runs in FORGE  *(point 4)*
- Wrap the `predict` and LLLA paths in `prof.block(...)` (§Resource logging). Append to a
  rolling `resource_report.json`. This is the *inference-side* telemetry — the part that
  actually sizes the deployed box, distinct from the training numbers.

### 5. UI: network structure + animated posterior  *(point 5)*
- Network SVG generated from `arch.json` (same visual language as the Experiment-panel
  diagram, but reflecting the *built* model).
- **Animated LLLA posterior:** backend hands the frontend the eigenbasis (`Λ`, projected
  φ) once; the τ (prior-precision) slider recomputes predictive variance in-browser via
  `Σ_i (proj φ)²_i / (λ_i + τ)` — O(d), 60 fps, no round-trip. On screen: per-genre
  softmax bars whose error bands **breathe** as τ moves; the per-config σ histogram slides.
- Label the knob types honestly: prior/likelihood/temperature = live; architecture/feature
  knobs = "recompute" (needs a backbone re-pass), not animation; sampler knobs belong to HMC.

### 6. Yes/no: is the model at a limit?  *(point 6)*  — the diagnostician answer
Starting heuristic (refine with data), combining three signals:
- **Learning-curve slope** at 100% train (retrain at 25/50/75/100%, or reuse sweep folds).
  Still rising ⇒ more data helps. Plateaued ⇒ data-saturated.
- **Epistemic vs aleatoric** from the posterior: last-layer σ (epistemic) vs predictive
  entropy at MAP (aleatoric).
- **RRM stability axis** (`σ_fold/σ_max`).

Decision: rising curve + high epistemic ⇒ **more data**; plateaued + low epistemic +
residual error ⇒ **architecture-bound** (not a data problem); plateaued + low epistemic +
low error ⇒ **at its limit, and that's fine**. Output a one-line verdict + the lever.

### 7. HMC for the official answer  *(point 7)*
- `src/hmc.py`: NUTS over the **last-layer weights only** (low-dim → cheap), reusing
  `phi_train`. Gives the ground-truth posterior that (a) validates LLLA (hypothesis H2,
  agreement on moments/coverage) and (b) supplies the *authoritative* epistemic σ for the
  §6 verdict. This is the serverless-GPU "job" (~1 min). Render as a static overlay on the
  live LLLA bands.

### 8. Metrics wall  *(point 8)*
One panel, everything we can generate, read from the bundle's `metrics.json` + `run.json`
+ `compute.json` + `resource_report.json`:
- **Standard:** accuracy, precision/recall/F1, per-genre F1, ROC-AUC, ECE, confusion matrix.
- **Reliability:** RRM + its three axes, calibration curve, epistemic/aleatoric split.
- **Resource:** wall, peak RSS/VRAM, throughput, knob/predict latency vs budget, projected
  deploy cost (from §Resource logging).
- **Posterior:** credible bands, σ histogram, LLLA-vs-HMC agreement.

---

## Build order (so "first model done" is the shortest path)

**Day 1 — get one trained model + the contract.**
1. `dataio.to_torch_loader` (swap from TF).
2. `src/models/beardown.py` — PyTorch dual encoder (spec CNN + tab MLP) → fusion →
   final `Linear`. LLLA-ready (final layer is a plain `nn.Linear`).
3. `src/train.py` — train cfg_14 config via 3-fold CV, **profiler on from run 1**, hit
   ≥0.75, write `run.json`/`compute.json` + the bundle (incl. `phi_train`, `ggn_eig`).
4. Confirm: Model panel dot for `beardown` flips to **delivered**.

**Day 2+ —** `sweep.py` (→ resource curve + Model 2), `features.py` + `/api/predict` +
Inference panel (song drop), then LLLA animation, then HMC + the §6 verdict, then the
metrics wall.

---

## Open items to pull from the writeup before Day 1
- [ ] Exact HP ranges + the cfg_14 config (BEARDOWN report, slide_7).
- [ ] Spec-CNN layer sizes (filters/kernels/dense) and tab-MLP widths — for `arch.json`.
- [ ] Mel params used to build the GTZAN spectrograms (n_mels, n_fft, hop) — for
      `features.py` to match the training distribution.
- [ ] Confirm fusion type for Model 1 (concat vs gated) — proposal says fused; report says which.
