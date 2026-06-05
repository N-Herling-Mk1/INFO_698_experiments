# `eda/` — exploratory data analysis area

EDA here is mostly **integrity work**, not decoration. The pretty per-genre
spectrograms are real output (they become the website's hero visuals), but the
load-bearing job of this folder is catching the things that make a reproduction
lie to you.

## What `run_eda.py` emits
```
eda/
├── run_eda.py        # entrypoint
├── figures/          # EMITTED PNGs — consumed by the doc-site EDA panel
│   ├── class_balance.png
│   ├── feature_corr.png
│   ├── mel_exemplars.png        # one per genre — website hero row
│   └── leakage_audit.png
└── (eda_stats.json is written up to projects/genre/data/ or runs/, per schema)
```

## The integrity checklist (run, in order)
1. **Class balance** — expect 100/genre. Flags the corrupt `jazz.00054`.
2. **Audio sanity** — durations (~30 s), sample rate (22050), mono.
3. **Tabular tier** — distributions + correlation of the 58 features (30 s & 3 s).
4. **Spectrogram tier** — per-genre mel exemplars (the visuals).
5. **⭐ Leakage audit** — the one that actually matters (below).

## ⭐ The leakage trap — read this before trusting any accuracy number
GTZAN ships a 3-second-segment representation (and EAViT-style pipelines segment 30 s
into 3 s clips). If you split at the **segment** level, multiple segments of the *same
recording* land in both train and test. The model then "recognizes the recording," not
the genre, and accuracy inflates — sometimes by 10+ points. This is why some reproduced
numbers mysteriously *beat* the paper.

**Guard:** always split at the **track** level (`_shared/splits.py: track_level_split`).
`leakage_audit.png` visualizes whether any track id appears across splits — it should
be empty.

Beyond segment leakage, Sturm (2013) documents track-level duplicates and artist
repetition across genres. An **artist-aware** split is the stricter honest option.

**Plan:** emit metrics under both `naive_random` and `track` splits and headline the
gap. That gap is itself a result — and it feeds FORGE's epistemic-uncertainty story
directly (a model confident on a leaked test set is exactly the false-confidence the
posterior observatory exists to expose).
