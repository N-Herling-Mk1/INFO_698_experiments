# `projects/` — the three reproductions

Each subfolder is one reproduction sharing an identical skeleton. `genre/` is the
reference; `phonon/` and `atlas/` are skeletons built by copying it.

## Add / fill a project
```bash
cp -r genre/ phonon/                       # (already done for phonon/atlas)
rm -rf phonon/data/raw/* phonon/runs/*
# fill ONLY these — everything else is structural and already matches genre:
#   project.yaml            genealogy header
#   data/                   data adapter (source/format)
#   eda/run_eda.py          domain EDA -> data/<phase>/eda_stats.json + eda/figures/<phase>/
#   src/features.py
#   src/models/<model>.py
#   configs/*.yaml
```
If you reach for code that already exists in `genre/`, promote it to `_shared/`
instead of copying. Keep projects thin.

## Skeleton (every project has exactly this)
```
<project>/
├── project.yaml
├── data/{ raw/(gitignored), manifest.json, before/eda_stats.json, after/eda_stats.json }
├── eda/{ run_eda.py, figures/{before,after}/ }
├── configs/                 # one yaml per model/variant
├── src/{ features.py, models/, train.py, eval.py }
├── runs/                    # emitted run.json/compute.json (gitignored)
└── app/                     # copy of /_app_template — serves this project's artifacts
```

`before/after` are the two EDA snapshots (pre-fix / post-fix); the dashboard
toggles between them. See the top-level README for the full contract.
