"""GTZAN EDA compendium for the BEARDOWN reproduction.

Pre-fix integrity report: we describe the data BEFORE adjusting anything.
Emits eda/figures/*.png + projects/genre/data/eda_stats.json (site-consumable).

Three sections (per the BEARDOWN EDA spec):
  (1) MISSING / CORRUPT  — reconcile wav / grey-spectrogram / 30s-csv / 3s-csv;
                           flag corrupt + off-duration files. No fixes applied.
  (2) TYPE AUDIT         — expected vs actual dtype for every column; NaN/inf;
                           spectrogram image mode/size. Mismatches noted only.
  (3) NERD STATS         — per numeric column: mean/median/mode/std/min/max/IQR,
                           outlier count + skew; histogram + box per feature.

Plus: class_balance.png and per-genre mel exemplars (near-free, become hero art).

    python eda/run_eda.py --phase before --data-root <path to GTZAN Data dir>
"""
from __future__ import annotations
import argparse, os, sys, json, math, wave, time, contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sps
from PIL import Image


def write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, allow_nan=False)


GENRES = ["blues", "classical", "country", "disco",
          "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
EXPECT = {"sample_rate": 22050, "channels": 1, "duration_s": 30.0,
          "per_class": 100, "img_mode": "L"}
DUR_TOL = 0.5  # seconds; |dur - 30| > tol  => flagged "off-duration"
sns.set_theme(style="whitegrid")


# ----- tiny progress UI (Steve's standing pref: always show status) ----------
_T0 = time.time()
def step(msg: str) -> None:
    print(f"[eda +{time.time()-_T0:6.1f}s] {msg}", flush=True)


def _norm_id(name: str) -> str:
    s = Path(str(name)).name
    for ext in (".wav", ".png", ".au"):
        if s.lower().endswith(ext):
            s = s[: -len(ext)]
    return s.replace(".", "").replace("_", "").replace("-", "").lower()


# ============================================================ (1) MISSING/CORRUPT
def section_missing_corrupt(data_root: Path) -> dict:
    step("section 1 — missing / corrupt reconciliation")
    wav_root = data_root / "genres_original"
    grey_root = data_root / "images_grey_scale"

    wav_ids, grey_ids = set(), set()
    wav_per_class, grey_per_class = {}, {}
    off_duration, corrupt_audio = [], []

    for g in GENRES:
        wavs = sorted((wav_root / g).glob("*.wav")) if (wav_root / g).exists() else []
        greys = sorted((grey_root / g).glob("*.png")) if (grey_root / g).exists() else []
        wav_per_class[g] = len(wavs)
        grey_per_class[g] = len(greys)
        for p in wavs:
            wav_ids.add(_norm_id(p.name))
            try:
                with contextlib.closing(wave.open(str(p), "rb")) as w:
                    fr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
                dur = n / float(fr) if fr else 0.0
                if abs(dur - EXPECT["duration_s"]) > DUR_TOL:
                    off_duration.append({"id": p.stem, "genre": g,
                                         "duration_s": round(dur, 3),
                                         "sample_rate": fr, "channels": ch})
            except Exception as e:  # truncated / bad header -> corrupt
                corrupt_audio.append({"id": p.stem, "genre": g,
                                      "error": type(e).__name__,
                                      "bytes": p.stat().st_size})
        for p in greys:
            grey_ids.add(_norm_id(p.name))

    # cross-representation reconciliation — TRACK level only.
    # The 3s CSV is segment-level (blues.00000.0.wav -> blues000000) and never
    # matches a track id, so it is handled by the segment-completeness check
    # below, NOT folded into this diff.
    rep = {"wav": wav_ids, "grey_spectrogram": grey_ids}
    fp30 = data_root / "features_30_sec.csv"
    if fp30.exists():
        col = pd.read_csv(fp30, usecols=[0]).iloc[:, 0].astype(str)
        rep["features_30sec"] = set(col.map(_norm_id))

    union = set().union(*rep.values())
    present = {tag: sorted(union - ids) for tag, ids in rep.items()}  # missing per rep

    # 3s-segment completeness: each track should have 10 segments
    seg_anom = []
    fp3 = data_root / "features_3_sec.csv"
    if fp3.exists():
        f3 = pd.read_csv(fp3, usecols=[0])
        f3["track"] = f3.iloc[:, 0].astype(str).str.replace(
            r"\.(\d+)\.wav$", "", regex=True)
        seg_counts = f3.groupby("track").size()
        seg_anom = [{"track": t, "segments": int(c)}
                    for t, c in seg_counts.items() if c != 10]

    counts = {"wav_per_class": wav_per_class, "grey_per_class": grey_per_class,
              "wav_total": sum(wav_per_class.values()),
              "grey_total": sum(grey_per_class.values())}
    step(f"  wav={counts['wav_total']} grey={counts['grey_total']} "
         f"corrupt={len(corrupt_audio)} off_duration={len(off_duration)} "
         f"seg_anomalies={len(seg_anom)}")

    # auto-derived known_issues (replaces the hand-typed manifest placeholder)
    known = []
    for c in corrupt_audio:
        known.append(f"{c['genre']}.{c['id'].split('.')[-1]} audio unreadable "
                     f"({c['error']}, {c['bytes']} bytes)")
    for d in off_duration:
        known.append(f"{d['id']} off-duration ({d['duration_s']}s vs 30s)")
    for tag, miss in present.items():
        for m in miss:
            known.append(f"{m} absent from representation '{tag}'")
    for s in seg_anom:
        known.append(f"{s['track']} has {s['segments']} 3s-segments (expected 10)")

    return {
        "counts": counts,
        "corrupt_audio": corrupt_audio,
        "off_duration": off_duration,
        "missing_per_representation": present,
        "segment_anomalies": seg_anom,
        "known_issues": known,
    }


# ============================================================ (2) TYPE AUDIT
def section_type_audit(df: pd.DataFrame, data_root: Path) -> dict:
    step("section 2 — type audit (expected vs actual)")
    audit = []
    for col in df.columns:
        if col == "filename":
            expected = "string"
        elif col == "label":
            expected = "categorical(10)"
        elif col == "length":
            expected = "int"
        else:
            expected = "float"

        s = df[col]
        actual = str(s.dtype)
        coerced = pd.to_numeric(s, errors="coerce") if expected in ("float", "int") else None
        n_nan = int(coerced.isna().sum()) if coerced is not None else int(s.isna().sum())
        n_inf = int(np.isinf(coerced.to_numpy()).sum()) if coerced is not None else 0
        match = (
            (expected == "string" and pd.api.types.is_string_dtype(s)) or
            (expected == "categorical(10)" and s.nunique() == 10) or
            (expected == "int" and np.issubdtype(s.dtype, np.integer)) or
            (expected == "float" and np.issubdtype(s.dtype, np.floating))
        )
        note = ""
        if expected == "int" and np.issubdtype(s.dtype, np.floating):
            note = "stored float but integral; harmless"
            match = bool((s.dropna() % 1 == 0).all())
        if n_nan or n_inf:
            note = (note + f"; {n_nan} NaN, {n_inf} inf").strip("; ")
        if expected == "categorical(10)" and s.nunique() != 10:
            note = f"{s.nunique()} distinct labels (expected 10)"
        audit.append({"column": col, "expected": expected, "actual_dtype": actual,
                      "match": bool(match), "n_nan": n_nan, "n_inf": n_inf,
                      "note": note})

    mism = [a["column"] for a in audit if not a["match"] or a["note"]]
    step(f"  {len(audit)} columns audited; {len(mism)} with mismatch/note")

    # spectrogram image audit (sample a handful per genre)
    grey_root = data_root / "images_grey_scale"
    img_modes, img_sizes, bad_imgs = {}, {}, []
    for g in GENRES:
        for p in sorted((grey_root / g).glob("*.png"))[:5]:
            try:
                with Image.open(p) as im:
                    img_modes[im.mode] = img_modes.get(im.mode, 0) + 1
                    img_sizes[str(im.size)] = img_sizes.get(str(im.size), 0) + 1
            except Exception as e:
                bad_imgs.append({"file": str(p), "error": type(e).__name__})
    img_audit = {"expected_mode": EXPECT["img_mode"], "observed_modes": img_modes,
                 "observed_sizes": img_sizes, "unreadable": bad_imgs}
    return {"columns": audit, "n_mismatch_or_note": len(mism), "spectrograms": img_audit}


# ============================================================ (3) NERD STATS
def _numeric_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns
            if c not in ("filename", "label") and np.issubdtype(df[c].dtype, np.number)]


def _stat_block(x) -> dict:
    """The 13 descriptive stats for one numeric array (used for combined AND per-genre).
    Non-finite results (e.g. skew of a zero-variance genre slice) become None so the
    emitted JSON stays valid."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return None
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((x < lo) | (x > hi)).sum())
    within = x[(x >= lo) & (x <= hi)]
    whislo = float(within.min()) if within.size else float(x.min())
    whishi = float(within.max()) if within.size else float(x.max())
    mode_val = float(sps.mode(np.round(x, 3), keepdims=False).mode)
    blk = {
        "count": int(x.size), "mean": float(np.mean(x)),
        "median": float(np.median(x)), "mode_round3": mode_val,
        "std": float(np.std(x)), "min": float(np.min(x)), "max": float(np.max(x)),
        "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
        "whislo": whislo, "whishi": whishi,
        "n_outliers_iqr": n_out,
        "outlier_pct": round(100 * n_out / x.size, 2),
        "skew": float(sps.skew(x)),
    }
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in blk.items()}


def _dist_fig(x, title, path):
    """Histogram (kde) + box panel — the single source of truth for figure style,
    used for the combined panel AND every per-genre panel so they match exactly."""
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2),
                                 gridspec_kw={"width_ratios": [3, 1]})
    try:
        sns.histplot(x, bins=40, kde=bool(np.std(x) > 0), ax=a1, color="#1FB6C1")
    except Exception:
        sns.histplot(x, bins=40, kde=False, ax=a1, color="#1FB6C1")
    a1.set_title(title); a1.set_xlabel("")
    sns.boxplot(y=x, ax=a2, color="#F0A500", width=0.5,
                flierprops={"marker": "o", "markersize": 3,
                            "markerfacecolor": "#FF6A1A", "markeredgecolor": "none"})
    a2.set_title("box"); a2.set_ylabel("")
    fig.tight_layout(); fig.savefig(path, dpi=88); plt.close(fig)


def section_nerd_stats(df: pd.DataFrame, fig_dir: Path) -> dict:
    step("section 3 — descriptive stats + per-feature hist/box")
    cols = _numeric_cols(df)
    out, figures = {}, []
    for i, c in enumerate(cols, 1):
        x = df[c].dropna().to_numpy(dtype=float)
        out[c] = _stat_block(x)
        # same metrics, split by genre (combined stays above as out[c])
        pg = {}
        for g, sub in df.groupby("label"):
            blk = _stat_block(sub[c].to_numpy(dtype=float))
            if blk is not None:
                pg[g] = blk
        out[c]["per_genre"] = pg
        # shared bins over the combined range -> per-genre histograms are comparable
        edges = np.histogram_bin_edges(x, bins=30)
        hist_pg = {g: np.histogram(sub[c].dropna().to_numpy(dtype=float), bins=edges)[0]
                      .astype(int).tolist()
                   for g, sub in df.groupby("label")}
        out[c]["hist"] = {"edges": [round(float(e), 6) for e in edges], "per_genre": hist_pg}
        # per-feature panel: combined (unchanged style)
        fn = f"feature_{i:02d}_{c}.png"
        _dist_fig(x, f"{c} — histogram", fig_dir / fn)
        figures.append({"file": fn, "kind": "feature_dist", "feature": c,
                        "caption": f"{c}: mean={out[c]['mean']:.3g}, "
                                   f"skew={out[c]['skew']:.2f}, "
                                   f"outliers={out[c]['n_outliers_iqr']} ({out[c]['outlier_pct']}%)"})
        # one panel per genre, identical style
        gfig = {}
        for g, sub in df.groupby("label"):
            xg = sub[c].dropna().to_numpy(dtype=float)
            if xg.size == 0:
                continue
            fn_g = f"feature_{i:02d}_{c}__{g}.png"
            _dist_fig(xg, f"{c} — {g}", fig_dir / fn_g)
            gfig[g] = fn_g
        out[c]["per_genre_fig"] = gfig
        if i % 12 == 0:
            step(f"  {i}/{len(cols)} feature panels")
    return {"per_feature": out, "n_features": len(cols), "genres": list(GENRES)}, figures


# ============================================================ EXTRA FIGS
def fig_class_balance(df: pd.DataFrame, counts: dict, fig_dir: Path) -> dict:
    """Per-genre counts for BOTH representations (wav vs grey spectrogram), so the
    pre-fix imbalance (a missing grey exemplar drops a genre below 100) is visible."""
    wav = [counts["wav_per_class"].get(g, 0) for g in GENRES]
    grey = [counts["grey_per_class"].get(g, 0) for g in GENRES]
    plt.rcParams.update({"axes.edgecolor": "#1c2a3a", "text.color": "#cfe6ea",
                         "axes.labelcolor": "#cfe6ea",
                         "xtick.color": "#6f8693", "ytick.color": "#6f8693"})
    fig, ax = plt.subplots(figsize=(9, 3.4))
    fig.patch.set_facecolor("#0a0f16"); ax.set_facecolor("#0a0f16")
    x = np.arange(len(GENRES)); w = 0.4
    ax.bar(x - w/2, wav, w, label="wav (genres_original)", color="#1FB6C1")
    ax.bar(x + w/2, grey, w, label="grey spectrogram (images)", color="#F0A500")
    ax.axhline(EXPECT["per_class"], ls="--", lw=1, color="#FF6A1A", alpha=.8)
    ax.set_xticks(x); ax.set_xticklabels(GENRES, rotation=30, ha="right")
    lo = min(min(wav), min(grey)); ax.set_ylim(max(0, lo - 5), EXPECT["per_class"] + 1.3)
    ax.set_ylabel("files per genre")
    ax.set_title("Per-genre file counts by representation (dashed = expected 100)", fontsize=11)
    # annotate the first genre that is short in the grey set (pre-fix gap)
    for i, g in enumerate(GENRES):
        if grey[i] < EXPECT["per_class"]:
            ax.annotate(f"{g} grey = {grey[i]}", xy=(i + w/2, grey[i]),
                        xytext=(i - 0.3, max(0, lo - 4.4)), color="#FF6A1A",
                        fontsize=8.5, ha="left",
                        arrowprops=dict(arrowstyle="->", color="#FF6A1A", lw=1.2))
            break
    ax.legend(facecolor="#0d131c", edgecolor="#1c2a3a", labelcolor="#cfe6ea",
              fontsize=8, loc="upper right", ncol=2, framealpha=.9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(fig_dir / "class_balance.png", dpi=100,
                                    facecolor=fig.get_facecolor()); plt.close(fig)
    return {"file": "class_balance.png", "kind": "class_balance",
            "caption": "Per-genre counts for wav vs grey-spectrogram representations "
                       "(dashed = expected 100). A representation missing an exemplar "
                       "shows as a sub-100 bar."}


def _exemplar_grid(data_root: Path, fig_dir: Path, *, cmap: str, fname: str, suptitle: str):
    grey_root = data_root / "images_grey_scale"
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for ax, g in zip(axes.ravel(), GENRES):
        imgs = sorted((grey_root / g).glob("*.png"))
        if imgs:
            ax.imshow(Image.open(imgs[0]), cmap=cmap, aspect="auto")
        ax.set_title(g, fontsize=9); ax.axis("off")
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(fig_dir / fname, dpi=100); plt.close(fig)


def fig_mel_exemplars(data_root: Path, fig_dir: Path) -> list:
    """Two exemplar grids from the same source images: a true grey-scale render and a
    false-color (magma) render, so the dashboard can show both side by side."""
    _exemplar_grid(data_root, fig_dir, cmap="gray",
                   fname="mel_exemplars_grey.png",
                   suptitle="Per-genre grey-scale spectrogram exemplars")
    _exemplar_grid(data_root, fig_dir, cmap="magma",
                   fname="mel_exemplars_color.png",
                   suptitle="Per-genre spectrogram exemplars (false-color, magma)")
    return [
        {"file": "mel_exemplars_grey.png", "kind": "exemplars_grey",
         "caption": "One grey-scale spectrogram per genre (cmap=gray) — the raw representation as stored."},
        {"file": "mel_exemplars_color.png", "kind": "exemplars_color",
         "caption": "The same per-genre exemplars false-colored (cmap=magma) for readability."},
    ]


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--phase", default="before", choices=["before","after"],
                    help="EDA snapshot: before (pre-fix) | after (post-fix)")
    ap.add_argument("--data-root", default=str(here.parent / "data" / "raw"),
                    help="GTZAN Data dir (genres_original/, images_grey_scale/, *.csv)")
    ap.add_argument("--features-csv", default=None,
                    help="defaults to <data-root>/features_30_sec.csv")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    fig_dir = here / "figures" / args.phase; fig_dir.mkdir(parents=True, exist_ok=True)
    feats_csv = Path(args.features_csv) if args.features_csv else data_root / "features_30_sec.csv"
    step(f"data-root = {data_root}")

    df30 = pd.read_csv(feats_csv)
    figures = []

    missing = section_missing_corrupt(data_root)
    type_audit = section_type_audit(df30, data_root)
    nerd, feat_figs = section_nerd_stats(df30, fig_dir)
    figures += feat_figs
    figures.append(fig_class_balance(df30, missing["counts"], fig_dir))
    figures += fig_mel_exemplars(data_root, fig_dir)

    stats = {
        "schema": "eda_stats/1.0",
        "project": "genre", "model_target": "beardown",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": args.phase,
        "expectations": EXPECT,
        "missing_corrupt": missing,
        "type_audit": type_audit,
        "nerd_stats": nerd,
        "figures": figures,
    }
    out_stats = here.parent / "data" / args.phase / "eda_stats.json"
    write_json(str(out_stats), stats)
    step(f"wrote {out_stats}  ({len(figures)} figures in {fig_dir})")
    step("done.")


if __name__ == "__main__":
    main()
