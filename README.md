# INFO_698_experiments — Tier R (research / reproductions)

> **This README is the bible for the research tier.** If you're confused in week 5
> about where something lives or why, the answer is here. Sub-folders have their own
> READMEs (linked inline) for depth.

This repo holds the **three reproductions** of the FORGE capstone. Each is a published
result re-run from scratch, emitting a uniform set of artifacts that the `software`
repo and the documentation site consume. It does **not** contain any web/server code —
that's Tier S (`INFO_698_software`). This tier's only job: *run compute, emit artifacts.*

The three projects (per the Gantt):

| Week | Project | Status |
|------|---------|--------|
| 2 | `projects/genre/`  — GTZAN music-genre recognition (2 models: BEARDOWN + transformer) | **active** |
| 3 | `projects/phonon/` — phonon DOS reproduction | skeleton only |
| 4 | `projects/atlas/`  — ATLAS reproduction | skeleton only |

---

## 1. Repository layout

```
INFO_698_experiments/
├── README.md                  ← you are here (the bible)
├── pyproject.toml             # dependencies (the lockfile IS the experiment)
├── Dockerfile                 # heavy image — see §3 Docker build plan
├── .devcontainer/             # VS Code "Reopen in Container" config
│   └── devcontainer.json
├── .dockerignore / .gitignore
│
├── _shared/                   # the transferable ENGINE — import, never copy
│   ├── README.md              → _shared/README.md
│   ├── schema.py              # ⭐ the artifact contract (run.json / compute.json)
│   ├── profiler.py            # compute logging → compute.json
│   ├── splits.py              # track-level split (the leakage guard)
│   └── eda.py                 # reusable EDA plotting + stats helpers
│
└── projects/                  → projects/README.md  (how to add project #2, #3)
    ├── genre/                 ← THIS WEEK
    │   ├── README.md          → projects/genre/README.md
    │   ├── project.yaml       # genealogy header (domain, source, license, paper targets)
    │   ├── data/              → projects/genre/data/README.md   ⭐ DATA AREA
    │   │   ├── raw/           #   fetched GTZAN (gitignored)
    │   │   ├── interim/       #   cached mel / 58-feat tables (gitignored)
    │   │   └── manifest.json  #   EMITTED → genealogy source of truth
    │   ├── configs/
    │   │   ├── beardown.yaml
    │   │   └── transformer.yaml
    │   ├── src/
    │   │   ├── features.py     # mel-spectrogram + 58-feature extraction
    │   │   ├── models/         #   beardown.py, transformer.py
    │   │   ├── train.py        # entrypoint: reads config → trains → emits run.json
    │   │   └── eval.py         # scorecard: our number vs the paper's target
    │   ├── eda/               → projects/genre/eda/README.md    ⭐ EDA AREA
    │   │   ├── run_eda.py      #   EMITS figures/*.png + eda_stats.json
    │   │   └── figures/
    │   └── runs/              #   EMITTED per-run artifacts (gitignored)
    │       └── <model>/<timestamp>/{run.json, compute.json, log.txt}
    ├── phonon/                # week 3 — copy the genre skeleton
    └── atlas/                 # week 4 — copy the genre skeleton
```

**Where is the data area?** → `projects/<name>/data/`. Each project owns its data.
**Where is the EDA area?** → `projects/<name>/eda/`. Each project owns its EDA.
Both are first-class, both emit artifacts the website reads. See their READMEs.

---

## 2. How a project is structured (and why it's transferable)

Every project is the **same skeleton**: `project.yaml` + `data/` + `configs/` + `src/` +
`eda/` + `runs/`. Adding project #2 (phonon) and #3 (ATLAS) means copying the skeleton
and changing three things:

1. `data/` — a new data adapter (different source, different format)
2. `src/features.py` + `src/models/` — domain-specific extraction + model
3. `configs/*.yaml` — the knobs

Everything else — the profiler, the split logic, the run.json schema, the EDA helpers —
comes from `_shared/` by **import**, not copy. That import discipline is what makes
weeks 3–4 cheap. If you find yourself copy-pasting from `genre/` into `phonon/`,
stop — that code belongs in `_shared/`.

### The run cadence
`src/train.py` reads a config, runs the experiment wrapped in the profiler, and drops
a timestamped folder under `runs/<model>/<ts>/` containing `run.json`, `compute.json`,
and `log.txt`. That folder is the **only** thing the rest of the ecosystem cares about.

---

## 3. Docker build plan  ⭐ (important — read before first run)

**Why this repo gets the heavy image:** reproduction fidelity *is* the pinned
environment. torch + librosa + audio system libs + (optionally) CUDA. The `software`
repo deliberately does NOT share this image — it never imports torch.

### 3.1 The audio trap (the #1 reproduction-killer)
`librosa` silently fails to decode GTZAN audio without system codecs. The Dockerfile
installs them up front:

```dockerfile
ARG BASE_IMAGE=python:3.12-slim          # CPU default for laptop dev/EDA/smoke runs
FROM ${BASE_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./           # lockfile FIRST → layer cache survives edits
RUN uv sync --frozen
COPY . .
```

### 3.2 One Dockerfile, two worlds (CPU ↔ CUDA)
The `BASE_IMAGE` build-arg is the switch — same recipe, same pins, no drift:

```bash
# laptop: CPU, for EDA + smoke tests
docker build -t forge-exp:cpu .

# remote GPU box (atlng01 / eepp-bigmem3): real training
docker build --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 \
             -t forge-exp:cuda .
```

### 3.3 VS Code workflow (this is the "set all the python libraries" part)
1. Open the repo folder in VS Code.
2. Command Palette → **Dev Containers: Reopen in Container**.
3. It builds the image, `postCreateCommand` runs `uv sync`, interpreter auto-selects.
4. You're now in the pinned env. Run EDA / training from the integrated terminal.

For GPU dev, uncomment `"runArgs": ["--gpus","all"]` in `.devcontainer/devcontainer.json`
(only on a box that has GPUs + the NVIDIA container toolkit).

### 3.4 Layer-caching rule
Always `COPY` the lockfile **before** the source. Editing `src/` then won't bust the
`uv sync` layer — rebuilds stay seconds, not minutes.

---

## 4. The artifact contract (what this repo emits)

Defined in [`_shared/schema.py`](_shared/schema.py). Three artifacts per run:

| File | Contains | Consumed by |
|------|----------|-------------|
| `run.json` | git SHA, config, dataset hash, per-epoch logs, final metrics | software backend, scorecard |
| `compute.json` | wall-clock, peak VRAM/RSS, throughput, est. FLOPs/params | **cost estimates**, Gantt risk, proposal |
| `eda_stats.json` + `figures/*.png` | dataset integrity, distributions, exemplars | doc-site genealogy + EDA panels |

Do not change these shapes casually — every downstream renderer depends on them.

---

## 5. Weekly workflow

```bash
# 1. fetch data (once per project) — see data/README.md
# 2. EDA — emits figures + stats the site reads
python projects/genre/eda/run_eda.py
# 3. train a reproduction — emits run.json + compute.json
python projects/genre/src/train.py --config projects/genre/configs/beardown.yaml
# 4. score it against the paper target
python projects/genre/src/eval.py --run runs/beardown/<ts>/
```

---

## 6. Open decisions (block full model fill-in, not scaffolding)

- **BEARDOWN** — link the repo/notebook, or confirm it's the INFO 510 classifier.
- **Transformer paper** — EAViT (3 s segments) / improved-ViT / attention-CNN?
- **Split policy** — naive vs track/artist-aware. Current plan: log both.

See `projects/genre/README.md` for how each decision wires into the configs.
