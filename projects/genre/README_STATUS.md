# GENRE MODEL FAMILY — status & next steps

_Last updated: 2026-06-22 · INFO 698 capstone · FORGE genre line_

This is the "here we are / what's next" snapshot for the BEARDOWN genre-classification
family (GTZAN, 10 genres, 30-sec fused image+tabular). For the full mk2/mk2.5 decision
record see `mk2_5_LOCKIN.txt`; for the attribution design see
`FORGE_input_feature_attribution_SPEC.txt`.

---

## Where we are

### The model ladder

| Model | What it is | Selection | Status | Test acc |
|-------|------------|-----------|--------|----------|
| **mk1** `beardown` | single fixed config | accuracy | done | 0.700 |
| **mk2** `beardown_rrm` | broad 20-config sweep | variance-weighted RRM_w | **LOCKED = cfg 17** | **0.760** |
| **mk2.5** (lens, not a model) | Dirichlet/variance re-selection over mk2's sweep | — | done | — |
| **mk3** `beardown_3sec` | 3-sec clips (~10× data) | same engine | **not started** | — |

### Deployed bundles (`projects/genre/models/`)

- `beardown` — mk1
- `beardown_rrm` — **mk2, deployed = cfg 17** (img768/tab256/concat/adam, φ-dim 768). VAL 0.793 / TEST 0.760, beats mk1 by +6 test. First acceptance PASS (val ≥ 0.75).
- `beardown_rrm_cfg12` — the equal-weight RRM winner, kept as the documented contrast.
- `beardown_rrm_cfg17` — earlier test copy of cfg 17 (harmless duplicate of the deployed slot; can delete for a clean dir).

### The mk2.5 result (the real finding)

Equal-weight RRM nominally picked **cfg 12**; we deploy **cfg 17** via variance-weighted
RRM_w. The firewalled test-validation study (`batch_test.json`) then measured whether any
CV metric predicts held-out test rank (Spearman):

```
+0.685   CV accuracy (A) alone        <- the only real predictor
-0.002   RRM_w (variance, deployed)    \
-0.008   fold-stability (-s)            }  all noise
-0.022   RRM (equal-weight)            /
-0.028   epistemic (-U)               /
```

**Honest headline:** on the frozen-deterministic-backbone + post-hoc-LLLA architecture,
neither RRM reliability axis predicts generalization — U is flat across the cohort, and
s is uncorrelated with test. Reliability-aware selection gave no test-rank advantage over
plain accuracy here. This is a precise *negative result with a mechanism*: it refines
(not refutes) INFO 510 — their RRM used trained Bayesian heads where U varied; ours does
not. The deployment is still sound (cfg 17 is accuracy-strong, and accuracy is what
predicts test). Selection stayed CV-only; test was recorded post-hoc to study the metric
(firewall held).

### Code (`projects/genre/src/`)

- `rrm.py` — metric core (RRM, off-diag, RAM, Pareto, Pearson, dCor, MacKay-τ) + mk2.5
  additions (`rrm_weighted`, `variance_weights`, `dirichlet_winner_sweep`). 21/21 self-tests.
- `sweep.py` — the RRM sweep engine (reuses train.py; LLLA via llla.py).
- `promote_cfg.py` — refit any recorded config on full-train + report once.
- `batch_refit.py` — firewalled all-20 test-validation study.
- `reselect.py` — mk2.5 Dirichlet/variance re-selection lens over `sweep.json`.
- `inspect_sweep.py` — sweep.json inspector.

### Artifacts (`projects/genre/runs/beardown_rrm/<ts>/`)

`sweep.json` (the 20-config catalogue) · `reselect.json` (centroid/variance/Dirichlet) ·
`batch_test.json` (the firewalled test-validation table).

---

## FORGE integration status

FORGE currently wraps **mk1 only**. The contract is solid for expansion:

- All bundles share the same 8-file contract → FORGE wraps cfg 17 with zero changes
  (round-trip verified, max|Δ|=0).
- **Input-feature attribution is VERIFIED-amenable** on every bundle: `scaler.json`
  carries all 57 named axes (`chroma_stft_mean … mfcc20_var`) + mean/std, and
  `arch.json`+`weights.pt` rebuild the frozen backbone for the ∂φ/∂x Jacobian. Attribution
  reads `scaler.json["cols"]` and reloads the model — no re-bundling for mk1, cfg 17, or
  future mk3.

---

## Next steps (pick one to start)

### A. mk3 `beardown_3sec` — the data-limited test
The scientifically central follow-up. mk1 found "506/512 directions prior-led →
data-limited"; mk3 tests it with ~10× data (3-sec clips). **Also the natural test of the
mk2.5 finding:** does more data make U vary again, and if so does RRM_w recover
test-predictive validity?
- **Blocker:** no 3-sec fused path exists. Needs a leakage-safe spectrogram cutter —
  segment-tab ⨝ parent-track image (many-to-one), with a **track-grouped split** so all
  segments of a track stay on one side (else leakage). Design first, then `_load_fused_3sec`.

### B. Input-feature attribution — wire into FORGE across all options
Build `attribution.py` (tabular branch first: ∂σ/∂x via the chain rule, reading names
from `scaler.json["cols"]`), then a capability-gated FORGE tab. Runs on mk1 + cfg 17
with zero re-bundling. Thesis hook: *names* the latent features INFO 510's Table A3 could
only rank anonymously. See `FORGE_input_feature_attribution_SPEC.txt`.

### C. FORGE multi-model integration
Wire mk1, mk2 (cfg 17), and (later) mk3 into FORGE as Model-panel dots, plus the cfg 12
contrast. Mostly plumbing on the existing contract.

### Optional / deferred
- Writeup: mk2.5 results paragraph (the negative result + the boundary-condition framing).
- `inputs.json` manifest + `capabilities` self-declaration in `write_bundle` — only needed
  for the generic "load your model" FORGE website, not for our own models.
- Delete the duplicate `beardown_rrm_cfg17` bundle.

**Recommended order:** A (mk3) is the strongest thesis move — it closes mk1's central
data-limited question *and* tests whether mk2.5's negative result reverses with more data.
B (attribution) is the best FORGE-feature move and is independent of A.
