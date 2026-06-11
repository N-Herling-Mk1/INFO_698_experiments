# phonon — reproduction (skeleton)

Shape-only. Built by copying the `genre/` skeleton; fill in three things:
1. `data/` — the data adapter (source/format for this domain)
2. `eda/run_eda.py` — domain EDA (emits `data/<phase>/eda_stats.json` + `eda/figures/<phase>/`)
3. `src/features.py`, `src/models/phonon_net.py`, `configs/*.yaml` — the reproduction

Everything structural (app, before/after, schema, profiler) already matches genre.
Run the dashboard now and it will say "run run_eda.py first" until EDA exists:
```bash
python projects/phonon/app/server.py
```
