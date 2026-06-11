# INFO_698_experiments — FORGE reproductions (Tier R)

Three published results, each re-run from scratch and wrapped in its own small
full-stack dashboard. This repo runs compute and emits artifacts; each experiment
also carries a mini-stack that serves those artifacts to a TRON-Ares UI.

> **Why each experiment is self-contained.** We build three *vertical slices*
> first and extract the shared backend later (rule of three). You don't yet know
> the right seams between music-genre, phonon-DOS, and ATLAS reproductions, so
> committing to a single backend up front would be guessing. Each experiment owns
> its data, EDA, training, and dashboard; once all three exist, the common parts
> get promoted. Until then, **shape stays rigid, content diverges** — that's what
> keeps the eventual merge mechanical.

---

## Layout

```
INFO_698_experiments/
├── _shared/                 # the transferable engine — import, never copy
│   ├── schema.py            # ⭐ artifact contract: RunRecord / ComputeRecord
│   ├── profiler.py          # compute logging -> compute.json
│   ├── splits.py            # track-level split (leakage guard)
│   └── eda.py               # reusable EDA helpers
│
├── _app_template/           # ⭐ canonical mini-stack (copied to each project's app/)
│   ├── server.py            # Flask; auto-detects root (sibling in repo / flat in bundle)
│   ├── config.py            # phases; experiment name auto-derives from folder
│   ├── templates/index.html · static/{css,js,assets}
│   └── README.md
│
├── tools/
│   └── export_bundle.py     # app + a project's snapshots -> standalone zip (handoff)
│
├── projects/
│   ├── genre/   ← ACTIVE (BEARDOWN, GTZAN)        # EDA done; training port next
│   ├── phonon/  ← skeleton (week 3)
│   └── atlas/   ← skeleton (week 4)
│
├── Dockerfile · pyproject.toml · README.md · LICENSE · .gitignore
```

## Anatomy of one experiment

Every project is the **same skeleton** — the only divergence is content.

```
projects/<exp>/
├── project.yaml            # genealogy header (domain, source, license, targets)
├── data/
│   ├── raw/                # fetched data — GITIGNORED (never committed)
│   ├── manifest.json       # emitted genealogy (counts, checksums, known issues)
│   ├── before/eda_stats.json   # pre-fix EDA snapshot   (committed — the dashboard reads it)
│   └── after/eda_stats.json    # post-fix EDA snapshot  (committed once it exists)
├── eda/
│   ├── run_eda.py          # emits eda_stats.json + figures (per --phase)
│   └── figures/{before,after}/*.png
├── configs/                # one yaml per model/variant
├── src/
│   ├── features.py · models/ · train.py · eval.py   # the training tier
├── runs/                   # emitted run.json / compute.json — GITIGNORED
└── app/                    # copy of _app_template — serves THIS experiment's artifacts
```

### before / after — the two EDA snapshots

`before` is the **pre-fix compendium**: the honest description of the raw data
before any adjustment (missing/corrupt, type audit, per-feature stats + figures).
`after` is regenerated once the fix/imputation pass runs. The dashboard toggles
between them so a reviewer can diff exactly what the cleaning changed.

```bash
python projects/genre/eda/run_eda.py --phase before --data-root /path/to/GTZAN/Data
python projects/genre/eda/run_eda.py --phase after  --data-root /path/to/GTZAN/Data_fixed
```

## The artifact contract

| File | Emitted by | Read by |
| ---- | ---------- | ------- |
| `data/<phase>/eda_stats.json` | `eda/run_eda.py` | dashboard (Overview/Integrity/Types/Distributions) |
| `eda/figures/<phase>/*.png` | `eda/run_eda.py` | dashboard hero + gallery |
| `runs/<model>/<ts>/run.json` + `compute.json` | `src/train.py` (via `_shared/schema.py`) | dashboard `/api/runs`, scorecard |

`eda_stats.json` is schema-versioned (`schema: "eda_stats/1.x"`). Domains differ —
phonon/atlas won't have spectrograms — so the dashboard degrades gracefully on
missing blocks. Keep the core keys (`missing_corrupt`, `type_audit`, `nerd_stats`,
`figures`) stable; add domain blocks freely.

## The dashboard

One small Flask app per experiment, copied verbatim from `_app_template/`. It reads
artifacts live from disk (re-run EDA, refresh — no restart) and trains nothing.

```bash
pip install flask
python projects/genre/app/server.py        # -> http://127.0.0.1:5000
```

Routes (identical across experiments): `GET /api/config`, `/api/health`,
`/api/eda?phase=before|after`, `/api/runs`, `/figures/<phase>/<name>`.

**Cloning to a new experiment:** `cp -r _app_template projects/<exp>/app`. No edits —
`server.py` finds its own root and the experiment name derives from the folder.

**Standalone handoff:** `python tools/export_bundle.py genre` produces
`dist/genre-app.zip`, a flat self-contained copy (app + snapshots beside `server.py`)
for someone who just wants to open the dashboard. It's a build artifact, not a
second codebase.

## Framework note

The `genre` reproduction (BEARDOWN) is TensorFlow + tensorflow-probability — a
custom Flipout Bayesian head with explicit KL, plus a BayesianRidge latent pass. It
is not ported to torch; that would change the science. `pyproject.toml` reflects
this. EDA and the dashboard are light (pandas/matplotlib/flask) and need no DL stack.

## Status

| Project | Status | Notes |
| ------- | ------ | ----- |
| `genre` | active | EDA compendium done (`before`); fix pass + BEARDOWN training port next |
| `phonon` | skeleton | shape only — adapt data / eda / src |
| `atlas` | skeleton | shape only — adapt data / eda / src |

## Docker — one reproducible env, and the dashboard as a local server

Docker exists here for one reason: the heavy ML stack (TensorFlow + TFP, plus
librosa's system audio libs, plus optional CUDA) is the painful part to set up.
The dashboard is light, but it runs happily in the same image — so **"Docker" and
"a local server" are the same thing here**: flask runs *inside* the container and
its port is published to your host, so you open `http://localhost:5000` in your
normal browser while the ML environment stays sealed in the container.

```bash
make build                 # build the image once (CPU)
make up                    # genre dashboard  -> http://localhost:5000
EXP=atlas make up          # a different experiment's dashboard
make lab                   # interactive shell for EDA / training
make eda EXP=genre DATA=/data/GTZAN/Data PHASE=before
make native                # run the dashboard WITHOUT docker (just needs flask)
```

How it fits together:

- **One image, two run modes.** `docker-compose.yml` defines `dashboard` (the local
  server) and `lab` (a shell for the heavy ML side) from the *same* build — no
  second image to maintain.
- **The bind fix.** A server on `127.0.0.1` inside a container is unreachable from
  the host. `config.py` reads `FORGE_HOST`; compose sets it to `0.0.0.0`, so the
  published port works. Run natively and it defaults back to `127.0.0.1`. Same
  `server.py`, both worlds.
- **Live edits.** The repo is bind-mounted at `/workspace`; the Python env lives at
  `/opt/forge-venv` (outside the mount) so your live code edits don't shadow the
  installed dependencies — a classic compose footgun, avoided.
- **CPU or GPU.** Default base is `python:3.12-slim` (CPU). For CUDA, build with
  `--build-arg BASE_IMAGE=tensorflow/tensorflow:2.16.1-gpu` (that image already
  ships Python + CUDA + cuDNN) and uncomment the `gpu` block in compose /
  `runArgs` in the devcontainer.
- **VS Code.** `.devcontainer/` → "Reopen in Container" gives the full stack with
  ports 5000/8888 forwarded and the interpreter pointed at `/opt/forge-venv`.

You do **not** need Docker just to view a dashboard — `make native` (or
`python projects/<exp>/app/server.py`) works if you have flask. Docker is for the
reproducible heavy-ML setup; the server rides along for free.
