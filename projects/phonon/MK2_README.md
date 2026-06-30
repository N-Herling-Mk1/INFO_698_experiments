# FORGE · phonon · MODEL 2 — info-theory loss + sweeps

**What mk2 is:** mk1's exact E(3)NN architecture trained on a *modified, information-theory-aware
objective*. Plain MSE (mk1) is blind to the frequency axis — it scores the 51 DOS bins
independently, so a peak predicted one bin low costs the same as one predicted ten bins low, and
it over-smooths sharp van Hove features. mk2 blends in a **1-D Earth Mover / Wasserstein-1 (EMD)**
term that penalises mass at the *wrong frequency* — which is exactly what the headline metric
(`omega_bar within 10%`) measures.

The modified-loss + sweep experiment, in one self-contained module.

---

## The objective

```
L = (1 - alpha) * MSE  +  alpha * DIST  +  beta * |H[p_pred] - H[p_true]|
```

| term | what | knob |
|---|---|---|
| `MSE` | mk1's bin-wise loss (fidelity anchor) | — |
| `DIST` | `emd` (default) or `js` — distributional, frequency-aware | `--alpha` |
| entropy match | `\|H[p_pred] - H[p_true]\|`, counters MSE over-smoothing | `--beta` |

- `alpha = 0` → **pure MSE.** This is the mk1 control backbone — train it explicitly so you have a
  clean A/B against the EMD backbone for FORGE.
- `alpha > 0` → the mk2 treatment.

**Why EMD is cheap & differentiable:** on an ordered axis it has a closed form,
`EMD = sum_bins |CDF_pred − CDF_true| · dx`. That's just a `cumsum` (a linear op) then `abs`/`sum`.
Gradients flow cleanly — no Sinkhorn iterations, no approximation. The torch ports of
`spectral_entropy`, `js_divergence`, `emd1d` from `src/metrics.py` live at the top of
`src/train_mk2.py`.

**Scale caveat (read this):** with EMD in bin units, a single 1-bin peak shift gives `EMD ≈ 1.0`
while `MSE ≈ 0.02` — EMD is ~50× larger in magnitude. So `alpha` is **not** a 50/50 dial; even
modest `alpha` lets EMD dominate the gradient. Don't over-interpret `alpha` — optimise
`val frac_within` (the sweep does exactly this) and watch the per-component readout printed at each
checkpoint. `--emd-units cm` rescales `dx` to cm⁻¹ if you want the term in physical units instead.

---

## Run it

Drop the file in (see below), then from `projects\phonon`:

```bash
# smoke test (CPU, ~1 min) — confirms graphs build, loss drops, scorecard prints
python -m src.train_mk2 --data-dir data/raw --limit 120 --epochs 2 --alpha 0.4 --run-name mk2_smoke

# the two backbones for the FORGE comparison (GPU box — ARES / atlng01)
python -m src.train_mk2 --data-dir data/raw --epochs 64 --alpha 0.0 --run-name mk2_mse   # control
python -m src.train_mk2 --data-dir data/raw --epochs 64 --alpha 0.4 --run-name mk2_emd   # treatment

# add the sharpness regulariser
python -m src.train_mk2 --data-dir data/raw --epochs 64 --alpha 0.4 --beta 0.02 --run-name mk2_emd_ent
```

### The "sweeps" half (Optuna)

```bash
pip install optuna
python -m src.train_mk2 --data-dir data/raw --sweep 40 --sweep-epochs 12 --limit 400 \
                        --run-name mk2_sweep
```

Searches `{alpha, beta, lr, mul, layers}`, maximising **val `frac_within`** (the paper headline).
Data + graphs are loaded **once** and reused across trials. Writes the best trial to
`runs/<run_name>.mk2_sweep.json` and prints the exact full-length retrain command for the winner.

---

## Outputs

- `runs/<run_name>.torch` — same schema as mk1 (`state` / `history` / `model_kwargs`) **plus** a
  `loss_cfg` block (`dist`, `alpha`, `beta`). Drop-in for `eval.py`, `predict.py`, and the LLLA
  bundle builder — nothing downstream needs to change.
- `history[].valid` now also carries `frac_within` and `emd_cm` per checkpoint.

---

## Why this is the experiment (FORGE tie-in)

The regulariser is the *intervention*; **FORGE is the measurement apparatus.** Train the two
backbones (`mk2_mse` vs `mk2_emd`), then run the four observatory aspects on each:

1. **eigenspectrum / τ-knob** — did the EMD term change which φ-directions are data-determined vs prior-led?
2. **data-factorization sweep** — did effective dimension / fraction-determined shift?
3. **HMC** — sanity gate (still must agree with analytic Laplace; the head is unchanged).
4. **input-feature attribution** — did the term change *which* features carry weight?

That A/B — "a frequency-aware loss reshaped the posterior geometry, and here's the observatory
measuring it" — is the mk2 story, and it reuses everything already built. The loss does **not**
touch the frozen-backbone contract: mk2 trains a new backbone, FORGE reads it afterward.

---

## Drop locations

```
projects\phonon\src\train_mk2.py     <- the ml file (this experiment)
projects\phonon\MK2_README.md        <- this readme
```

`Expand-Archive -Force` at repo root if zipped; both are path-preserving. No edits to existing
files required — `train_mk2.py` imports `src.data`, `src.model.build_model`, and `src.metrics`
exactly as `src.train` does.

---

## Resume notes (continue from home)

1. **Smoke first** on the laptop (`--limit 120 --epochs 2`) to confirm the import wiring + loss path.
2. **Two full backbones** on the GPU box: `mk2_mse` (alpha 0) and `mk2_emd` (alpha 0.4). These are the A/B.
3. **Sweep** to pick a defensible `alpha` (and check whether `beta`/`layers` help) — short trials, `--limit 400`.
4. **Score**: compare `frac_within` and `mean EMD` of `mk2_emd` vs `mk2_mse` (and vs mk1) on TEST.
   Hypothesis: EMD backbone ↑ `frac_within` and ↓ `EMD` at the cost of a slightly worse raw MSE —
   that trade is the point.
5. **Then FORGE**: build LLLA bundles off both backbones, run the four aspects, diff the eigenspectra.

Open question to settle when you pick this up: whether to keep EMD in **bin units** (current default,
clean knob) or **cm⁻¹** (`--emd-units cm`, physical but couples alpha to the 20 cm⁻¹ bin width).
Bin units are recommended for the sweep; switch to cm only for the final reported number if you want
the loss in physical units.
