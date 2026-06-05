# `projects/` — the three reproductions

Each subfolder is one reproduction, all sharing an identical skeleton. `genre/` is
the reference implementation; `phonon/` and `atlas/` are built by copying it.

## Adding a new project (the week-3 / week-4 recipe)

```bash
cp -r genre/ phonon/           # copy the skeleton
rm -rf phonon/data/raw/* phonon/data/interim/* phonon/runs/*
# then edit ONLY these:
#   phonon/project.yaml        — genealogy header (domain, source, license, targets)
#   phonon/data/               — new data adapter (different source/format)
#   phonon/src/features.py     — domain feature extraction
#   phonon/src/models/         — the model(s) to reproduce
#   phonon/configs/*.yaml      — knobs
# everything else comes from _shared/ by import — do not re-implement it.
```

If you reach for code that already exists in `genre/`, that's a signal it should be
promoted into `_shared/` rather than copied. Keep projects thin.

## Skeleton (every project has exactly this)

```
<project>/
├── README.md        # project-specific notes, targets, gotchas
├── project.yaml     # genealogy header → drives the site's "project genealogy"
├── data/            # raw/ + interim/ (gitignored) + manifest.json (emitted)
├── configs/         # one yaml per model/variant
├── src/             # features.py, models/, train.py, eval.py
├── eda/             # run_eda.py → figures/ + eda_stats.json
└── runs/            # emitted per-run artifacts (gitignored)
```
