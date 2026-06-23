# FORGE — Phonon DoS Experiment (Phase 2) — Handoff README

**Purpose:** Pick-up notes for the FORGE phonon reproduction. Captures the target paper, the
code/data provenance (verified against the live repo), how to pull the data on a fresh machine,
and the reproduction + Bayesian-extension plan.

---

## 1. What this experiment is

FORGE's second domain experiment: reproduce a **deterministic phonon density-of-states (DoS)
regression**, then **extend it with a Bayesian layer (LLLA)** to produce calibrated uncertainty —
the gap the original paper flags but never fills.

This mirrors the `beardown` (genre) pattern: faithful deterministic baseline first, Bayesian
posterior + RRM-style reliability metrics on top.

---

## 2. The paper (reproduction target)

**Chen, Z., Andrejevic, N., Smidt, T., Ding, Z., Xu, Q., Chi, Y-T., Nguyen, Q. T., Alatas, A.,
Kong, J., Li, M.** "Direct Prediction of Phonon Density of States With Euclidean Neural Networks."
*Advanced Science* **8**, 2004214 (2021). DOI: 10.1002/advs.202004214

- arXiv: https://arxiv.org/abs/2009.05163
- Published (open access): https://onlinelibrary.wiley.com/doi/10.1002/advs.202004214
- PMC mirror: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8224435/
- Local PDF: `Advanced_Science_-_2021_-_Chen_-_Direct_Prediction_of_Phonon_Density_of_States...pdf`

### Model spec (from the paper)
- **E(3)NN** built on the L1Net block (Miller et al., arXiv:2008.08461) via the `e3nn` library.
- Crystal → periodic graph, radial cutoff **rmax = 5 Å**, periodic images included.
- Node features = mass-weighted one-hot (118-length, e.g. H = [m_H, 0, ...]). Embedding 118 → **64 scalars**.
- Conv + Gated Block layers; filters W(r) = R(|r|)·Y_lm(r̂) (learned radial × spherical harmonics).
- Final conv → sum over nodes → ReLU → normalization → **51 scalar outputs** (DoS, 0–1000 cm⁻¹, ~20 cm⁻¹ bins).
- Loss = plain **MSE**. Normalization recoverable via ∫g(ω)dω = 3N.

### Reproduction targets (numbers to hit)
- **70% of test samples within 10% relative error on average phonon frequency ω̄** (Fig 2c) — headline metric.
- No MSE↔atom-count correlation (Fig 2a); roughly balanced MSE across elements (Fig 2b).
- Worst labeled case: RuS₂, MSE ≈ 0.099.
- Alloy trick (Eq 1): two-hot composition encoding → ApB₁₋p at no extra cost (Mg₃Sb₂₍₁₋ₚ₎Bi₂ₚ demo).
- Downstream: 4,346 unseen MP structures → high-Cᵥ screening (Eq 2), <30 min on entry-level GPU.

---

## 3. Code repos

| Repo | Use it for | Notes |
|---|---|---|
| https://github.com/zhantaochen/phonondos_e3nn | **Data + splits** (and original code) | Pins ancient deps: torch 1.5.1, torch-geometric 1.5.0, specific old e3nn commit. Don't build against this. |
| https://github.com/ninarina12/phononDoS_tutorial | **Model code** | Same authors, rewritten for current e3nn (MRS 2021 tutorial). Build the Dev Container against this. |

Plan: model code from the **tutorial** repo, data/splits from the **original** repo.

---

## 4. DATA — verified contents (checked against live repo)

> All file types and shapes below were confirmed by downloading and introspecting the actual repo
> files. The `.torch` file is **NOT** the data (it's the trained model). The dataset is the **56 MB
> pickle inside the zip**, and it's fully self-contained.

### The dataset — `models/phdos_e3nn_len51max1000_fwin101ord3.zip`
Unzips to a single 56 MB pickle: `phdos_e3nn_len51max1000_fwin101ord3.pkl`
Plain Python dict, loads with `pickle.load`, **pure numpy + CIF strings — no torch/e3nn needed**:

| Key | Shape / len | Meaning |
|---|---|---|
| `material_id` | 1524 | MP IDs of training materials |
| `cif` | 1524 | crystal structures (input geometry)* |
| `phfre` | (51,) | frequency grid for 51-bin DoS (x-axis, 0–1000 cm⁻¹) |
| `phdos` | **(1524, 51)** | **training target** — 51-bin DoS the model predicts |
| `phfre_gt` | (4404,) | full-resolution DFPT frequency grid |
| `phdos_gt` | (1524, 4404) | full-resolution DFPT ground-truth DoS |

1,524 materials ≈ paper's "~1,500 crystalline solids." The 51-bin `phdos` is the Savitzky-Golay
smoothed regression target (`fwin101ord3` = window 101, order 3); `phdos_gt` is raw DFPT truth.

\* **OPEN ITEM:** confirm whether `cif` entries are raw CIF strings or pickled pymatgen `Structure`
objects — determines whether the graph-builder needs pymatgen in the loop. (Not yet checked.)

### The splits — `models/200801_trteva_indices.pkl`
List of 3 numpy index arrays. First = **1,220 train indices** (matches paper's "only 1200 examples").
Other two are test / val. ("trteva" = train / test / eval(val).)

### NOT the data (don't be fooled)
- `models/200803-1018_len51max1000_fwin101ord3_trial_run_full_data.torch` (16.8 MB) — **trained model**.
  Dict with keys `state` (e3nn state_dict), `model_kwargs` (Rs_in, Rs_out, mul, layers, max_radius,
  lmax, number_of_basis, ...), `dynamics` (20-entry training history). The `trial_run` in the name was the tell.
- `data/mp_data.csv` (77 KB) — **screening list only**: 4,347 rows of `mp-id, formula`. The 4,346
  unseen structures for the high-Cᵥ inference demo. No DoS in it.
- `models/cif_unique_files.pkl` (880 KB) — a 20,629-row lookup table (cif_name / cif_id / num_sites).
  Superset pool, not training data.

### KEY TAKEAWAY
**No Materials Project API key needed to train.** Structures + targets + splits are all bundled in
the repo. The `download_mpcifs.py` + MP-key path is *only* for the 4,346-structure screening set
(downstream high-Cᵥ demo, not the core reproduction).

---

## 5. How to get the data on the new machine

```bash
# Dataset (zip -> 56 MB pkl) and splits — these are all you need to train
curl -L -o phdos_dataset.zip \
  "https://raw.githubusercontent.com/zhantaochen/phonondos_e3nn/main/models/phdos_e3nn_len51max1000_fwin101ord3.zip"
unzip phdos_dataset.zip        # -> phdos_e3nn_len51max1000_fwin101ord3.pkl

curl -L -o trteva_indices.pkl \
  "https://raw.githubusercontent.com/zhantaochen/phonondos_e3nn/main/models/200801_trteva_indices.pkl"

# (optional) reference: original trained model, for sanity-checking your reproduction
curl -L -o reference_model.torch \
  "https://raw.githubusercontent.com/zhantaochen/phonondos_e3nn/main/models/200803-1018_len51max1000_fwin101ord3_trial_run_full_data.torch"

# (optional, downstream only) the 4,346-structure screening list
curl -L -o mp_screening.csv \
  "https://raw.githubusercontent.com/zhantaochen/phonondos_e3nn/main/data/mp_data.csv"
```

### Minimal loader
```python
import pickle, numpy as np

with open("phdos_e3nn_len51max1000_fwin101ord3.pkl", "rb") as f:
    d = pickle.load(f)             # no torch/e3nn import required

X_cif   = d["cif"]                 # 1524 structures -> build graphs
Y       = d["phdos"]               # (1524, 51) regression targets
freq    = d["phfre"]               # (51,) bin centers
Y_full  = d["phdos_gt"]            # (1524, 4404) raw DFPT truth (for uncertainty validation)

tr, te, va = pickle.load(open("trteva_indices.pkl", "rb"))   # index arrays; tr has 1220 entries
```

---

## 6. Bayesian extension plan (the FORGE contribution)

The paper's own discussion (p.5) says the model is "data-driven and probabilistic in nature" and
shouldn't be relied on "without further validation" — but it never quantifies uncertainty. That's the gap.

Why this is the *right* paper to Bayesianize:
1. **~1,200 training examples = epistemic-uncertainty regime.** Same insight as the ATLAS work
   (underdetermined weights, fixable-with-more-data). A calibrated BNN should show *high* epistemic σ
   exactly where the paper is weakest: unseen-element generalization (Fig 3, Nd/U) and
   experimental-vs-DFPT mismatch. Turns "sometimes wrong on novel chemistry" into "tells you when."
2. **LLLA insertion point.** Unlike `beardown` (classification), this is **51-output regression**.
   - Last-layer Laplace on the final `linear` weights is the natural move.
   - Watch the ∫g dω = 3N normalization: consider Bayesian head **pre-normalization**, propagate through.
   - RRM penalty vector → regression analogues: accuracy A → R² or neg-MSE on ω̄; ECE → interval
     coverage (PICP) or CRPS over the 51-bin predictive. σ and U terms transfer directly.
   - Bonus: ground truth bundled at BOTH resolutions, so you can validate that uncertainty widens
     where 51-bin smoothing discards real structure visible in `phdos_gt` (4404-pt).

**Writeup framing:** reproduce deterministic E(3)NN DoS regression → hit the 70%/10% ω̄ benchmark on
the repo's splits to establish fidelity → extend with LLLA for calibrated per-bin credible intervals
→ demonstrate uncertainty rises on unseen elements (the thing the original paper flags but doesn't deliver).

---

## 7. Next steps / open items

- [ ] Confirm `cif` entries: raw CIF strings vs pickled pymatgen `Structure` objects (graph-builder dep decision).
- [ ] Stand up Dev Container off the `phononDoS_tutorial` e3nn API.
- [ ] Wire the minimal loader above; reproduce the deterministic baseline on the repo splits.
- [ ] Verify against the bundled reference model (`*_trial_run_full_data.torch`): load its `state` +
      `model_kwargs`, reproduce its predictions before trusting your own retrain.
- [ ] Hit 70%/10% ω̄ target; log MSE-vs-atom-count and per-element MSE for parity with Figs 2a/2b.
- [ ] Design LLLA head on 51-output regression; define regression RRM vector (R²/neg-MSE, PICP/CRPS, σ, U).
- [ ] Add the Chen 2021 repo + tutorial-fork code provenance to the literature page (#literature),
      not just the paper citation.

---

## 8. Literature page note

`https://n-herling-mk1.github.io/INFO_698_documentation/#literature` — the literature section is
**client-rendered (JS-injected)**; a raw fetch returns only the nav shell + "Loading…". To review
entries programmatically, point at the source the JS pulls from (e.g. a `literature.json` / `.md` /
`.bib` in the `INFO_698_documentation` repo) rather than the rendered URL.
