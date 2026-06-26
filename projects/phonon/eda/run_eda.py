"""
phonon EDA compendium — Chen et al. (2021) DFPT phonon-DOS dataset.

Self-contained (numpy / pandas / matplotlib / scipy only — no _shared coupling),
so it runs anywhere the dataset is present. Emits the site-consumable schema the
dashboard reads (eda_stats/1.1) plus the figures the EDA panel renders.

Outputs:
  data/<phase>/eda_stats.json          (keys: missing_corrupt, type_audit, nerd_stats, figures + domain blocks)
  eda/figures/<phase>/*.png

What it does (the four things the brief asked for):
  (1) DERIVED VARIABLES — the DOS is a 51-D target on a fixed 0..1000 cm^-1 grid;
      we reduce each spectrum to physically-meaningful scalars (omega_bar, spread,
      spectral entropy, cutoff, peak, skew/kurtosis, integral) and pull composition
      scalars (n_elements, n_atoms, mean atomic mass, max Z) from the formula. These
      are the "variables" the stats/figures describe.
  (2) TYPE AUDIT       — dtype + shape of every array; confirms the frequency grid is
      shared across all 1524 materials (the key homogeneity fact); flags the one
      non-numeric column (cif text).
  (3) MISSING / CORRUPT— NaN/inf, all-zero ("empty") spectra, negative DOS, duplicate
      ids, off-grid frequency monotonicity. No fixes applied — reported only.
  (4) NERD STATS       — per derived variable: count/mean/median/mode/std/min/max/IQR,
      IQR-outlier count + skew; histogram + box per variable; plus DOS-bin box plot,
      correlation heatmap, scatter panel, element coverage, and DOS exemplars/heatmap.

    python eda/run_eda.py --phase after          # uses data/raw + data/phonon_catalog.json
"""
from __future__ import annotations
import argparse, json, math, pickle, re, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

# ---- TRON-ish but legible figure theme (light bg, Ares accents) -------------
CYAN, ORANGE, INK, GRID = "#0bb4c4", "#ff5a1f", "#0b2030", "#d7e2e6"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "figure.dpi": 110,
})

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))   # NumPy 2.x renamed trapz

_T0 = time.time()
def step(msg): print(f"[eda +{time.time()-_T0:6.1f}s] {msg}", flush=True)


# ---- standard atomic data (Z -> symbol, weight); index 0 unused -------------
_SYM = ("0 H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe "
        "Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn "
        "Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re "
        "Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm "
        "Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split()
_WT = [0, 1.008, 4.003, 6.94, 9.012, 10.81, 12.011, 14.007, 15.999, 18.998, 20.18,
       22.99, 24.305, 26.982, 28.085, 30.974, 32.06, 35.45, 39.948, 39.098, 40.078,
       44.956, 47.867, 50.942, 51.996, 54.938, 55.845, 58.933, 58.693, 63.546, 65.38,
       69.723, 72.63, 74.922, 78.971, 79.904, 83.798, 85.468, 87.62, 88.906, 91.224,
       92.906, 95.95, 98.0, 101.07, 102.906, 106.42, 107.868, 112.414, 114.818, 118.71,
       121.76, 127.6, 126.904, 131.293, 132.905, 137.327, 138.905, 140.116, 140.908,
       144.242, 145.0, 150.36, 151.964, 157.25, 158.925, 162.5, 164.93, 167.259, 168.934,
       173.045, 174.967, 178.49, 180.948, 183.84, 186.207, 190.23, 192.217, 195.084,
       196.967, 200.592, 204.38, 207.2, 208.98, 209.0, 210.0, 222.0, 223.0, 226.0, 227.0,
       232.038, 231.036, 238.029, 237.0, 244.0, 243.0, 247.0, 247.0, 251.0, 252.0, 257.0,
       258.0, 259.0, 266.0, 267.0, 268.0, 269.0, 270.0, 269.0, 278.0, 281.0, 282.0, 285.0,
       286.0, 289.0, 290.0, 293.0, 294.0, 294.0]
_Z = {s: i for i, s in enumerate(_SYM)}
_FORMULA = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(f):
    """'Na4Ag2As2' -> {'Na':4,'Ag':2,'As':2}.  'Ag1'/'Ag' both -> {'Ag':1}."""
    out = {}
    for sym, n in _FORMULA.findall(str(f)):
        if sym:
            out[sym] = out.get(sym, 0) + (int(n) if n else 1)
    return out


# ---------------------------------------------------------------- derived vars
def spectral_features(g, dos):
    """Per-material scalar reductions of a 51-bin DOS on grid g (cm^-1)."""
    s = dos.sum()
    if s <= 0:
        z = float("nan")
        return dict(omega_bar=z, spectral_spread=z, spectral_skew=z, spectral_kurtosis=z,
                    spectral_entropy=z, omega_peak=z, omega_cutoff=z, dos_peak=z,
                    dos_integral=0.0, n_active_bins=0)
    p = dos / s
    cen = float((p * g).sum())
    var = float((p * (g - cen) ** 2).sum())
    spread = math.sqrt(var) if var > 0 else 0.0
    if spread > 0:
        sk = float((p * ((g - cen) / spread) ** 3).sum())
        ku = float((p * ((g - cen) / spread) ** 4).sum())
    else:
        sk = ku = 0.0
    nz = p[p > 0]
    H = float(-(nz * np.log(nz)).sum() / math.log(len(g)))    # normalized 0..1
    thr = 0.01 * dos.max()
    active = dos > thr
    cutoff = float(g[active][-1]) if active.any() else 0.0
    return dict(omega_bar=cen, spectral_spread=spread, spectral_skew=sk,
                spectral_kurtosis=ku, spectral_entropy=H,
                omega_peak=float(g[int(dos.argmax())]), omega_cutoff=cutoff,
                dos_peak=float(dos.max()),
                dos_integral=float(_trapz(dos, g)),
                n_active_bins=int(active.sum()))


# numeric variables we profile (order = display order)
FEATURES = [
    ("omega_bar",        "mean phonon frequency  <w>  (cm-1)"),
    ("omega_peak",       "dominant peak frequency  (cm-1)"),
    ("omega_cutoff",     "spectral cutoff  w_max  (cm-1)"),
    ("spectral_spread",  "spectral spread  sigma_w  (cm-1)"),
    ("spectral_entropy", "normalized spectral entropy  (0-1)"),
    ("spectral_skew",    "spectral skewness"),
    ("spectral_kurtosis","spectral kurtosis"),
    ("dos_peak",         "peak DOS height  (norm.)"),
    ("dos_integral",     "integral g(w) dw  (norm.*cm-1)"),
    ("n_active_bins",    "active bins  (>1% of peak)"),
    ("n_elements",       "distinct elements"),
    ("n_atoms",          "atoms per formula unit"),
    ("mean_atomic_mass", "mean atomic mass  (u)"),
    ("max_Z",            "heaviest element  (Z)"),
]
SPLITS = ["train", "val", "test"]


def col_stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out = (x < lo) | (x > hi)
    try:
        mode = float(sps.mode(np.round(x, 6), keepdims=False).mode)
    except Exception:
        mode = float(np.round(np.median(x), 6))
    return dict(count=int(x.size), mean=float(x.mean()), median=float(np.median(x)),
                mode=mode, std=float(x.std(ddof=1)) if x.size > 1 else 0.0,
                min=float(x.min()), max=float(x.max()), q1=float(q1), q3=float(q3),
                iqr=float(iqr), skew=float(sps.skew(x)) if x.size > 2 else 0.0,
                n_outliers=int(out.sum()),
                outlier_frac=round(float(out.mean()), 4))


# ---------------------------------------------------------------------- figures
def savefig(fig, path):
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_hist_box(df, key, label, outdir, i):
    x = df[key].dropna().values
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.9),
                               gridspec_kw={"width_ratios": [3, 1]})
    a.hist(x, bins=40, color=CYAN, edgecolor="white", linewidth=0.4)
    a.axvline(np.median(x), color=ORANGE, lw=1.8, label=f"median {np.median(x):.3g}")
    a.set_xlabel(label); a.set_ylabel("materials"); a.legend(fontsize=8, frameon=False)
    bp = b.boxplot(x, vert=True, patch_artist=True, widths=0.6,
                   flierprops=dict(marker="o", ms=2.5, mfc=ORANGE, mec="none", alpha=0.5))
    bp["boxes"][0].set(facecolor=CYAN, alpha=0.35, edgecolor=INK)
    for med in bp["medians"]: med.set(color=ORANGE, lw=1.8)
    b.set_xticks([]); b.set_ylabel("")
    fig.suptitle(key, y=1.02, fontsize=11, fontweight="bold")
    name = f"feature_{i:02d}_{key}.png"
    savefig(fig, outdir / name); return name


def fig_split_balance(df, outdir):
    c = df["split"].value_counts().reindex(SPLITS).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    bars = ax.bar(c.index, c.values, color=[CYAN, ORANGE, "#7a8b91"], edgecolor=INK)
    for r, v in zip(bars, c.values):
        ax.text(r.get_x()+r.get_width()/2, v+8, str(v), ha="center", fontsize=9)
    ax.set_ylabel("materials"); ax.set_title("Split balance  (1220 / 152 / 152)")
    savefig(fig, outdir / "split_balance.png"); return "split_balance.png"


def fig_element_coverage(coverage, outdir, top=40):
    items = sorted(((k, v) for k, v in coverage.items() if v > 0),
                   key=lambda kv: kv[1], reverse=True)[:top]
    syms = [k for k, _ in items]; cnts = [v for _, v in items]
    npres = sum(1 for v in coverage.values() if v > 0)
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    ax.bar(range(len(syms)), cnts, color=CYAN, edgecolor=INK, linewidth=0.4)
    ax.set_xticks(range(len(syms))); ax.set_xticklabels(syms, fontsize=8, rotation=0)
    ax.set_ylabel("materials containing"); ax.set_title(f"Element coverage - top {top} of {npres} present")
    savefig(fig, outdir / "element_coverage.png"); return "element_coverage.png"


def fig_dos_binbox(g, dos, outdir):
    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    bp = ax.boxplot([dos[:, j] for j in range(dos.shape[1])], positions=g, widths=12,
                    patch_artist=True, showfliers=False)
    for box in bp["boxes"]: box.set(facecolor=CYAN, alpha=0.30, edgecolor=INK, linewidth=0.4)
    for med in bp["medians"]: med.set(color=ORANGE, lw=1.0)
    ax.set_xlim(-15, 1015); ax.set_xticks(np.arange(0, 1001, 100))
    ax.set_xlabel("frequency  w  (cm-1)"); ax.set_ylabel("DOS value  (norm.)")
    ax.set_title("DOS value distribution across the 51 frequency bins")
    savefig(fig, outdir / "dos_bin_box.png"); return "dos_bin_box.png"


def fig_dos_heatmap(g, dos, ob, outdir):
    order = np.argsort(ob)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    im = ax.imshow(dos[order], aspect="auto", cmap="magma", origin="lower",
                   extent=[g[0], g[-1], 0, dos.shape[0]], interpolation="nearest")
    ax.set_xlabel("frequency  w  (cm-1)")
    ax.set_ylabel("material  (sorted by <w> up)")
    ax.set_title("All 1524 phonon spectra")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("DOS (norm.)")
    savefig(fig, outdir / "dos_heatmap.png"); return "dos_heatmap.png"


def fig_dos_exemplars(g, dos, df, outdir):
    lo = df["omega_bar"].idxmin(); hi = df["omega_bar"].idxmax()
    mid = (df["omega_bar"] - df["omega_bar"].median()).abs().idxmin()
    widest = df["spectral_spread"].idxmax(); peaky = df["dos_peak"].idxmax()
    sharp = df["spectral_entropy"].idxmin()
    picks = [(lo, "lowest <w>"), (hi, "highest <w>"), (mid, "median <w>"),
             (widest, "widest"), (peaky, "tallest peak"), (sharp, "lowest entropy")]
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 4.6), sharex=True)
    for ax, (idx, tag) in zip(axes.ravel(), picks):
        ax.fill_between(g, dos[idx], color=CYAN, alpha=0.35)
        ax.plot(g, dos[idx], color=CYAN, lw=1.2)
        ax.set_title(f"{df.loc[idx,'formula']} . {tag}", fontsize=9)
        ax.set_xlim(0, 1000)
    for ax in axes[-1]: ax.set_xlabel("w (cm-1)")
    fig.suptitle("DOS exemplars", y=1.01, fontweight="bold")
    savefig(fig, outdir / "dos_exemplars.png"); return "dos_exemplars.png"


def fig_corr(df, outdir):
    keys = [k for k, _ in FEATURES]
    C = df[keys].corr().values
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(C, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys, fontsize=7)
    for i in range(len(keys)):
        for j in range(len(keys)):
            ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(C[i, j]) > 0.55 else INK)
    ax.set_title("Derived-variable correlation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    savefig(fig, outdir / "feature_corr.png"); return "feature_corr.png"


def fig_scatter(df, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    def sc(ax, xk, yk, xl, yl):
        ax.scatter(df[xk], df[yk], s=8, c=CYAN, alpha=0.45, edgecolors="none")
        r = df[[xk, yk]].corr().iloc[0, 1]
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(f"r = {r:+.2f}", fontsize=10)
    sc(axes[0], "mean_atomic_mass", "omega_bar", "mean atomic mass (u)", "<w> (cm-1)")
    sc(axes[1], "n_atoms", "spectral_entropy", "atoms / formula", "spectral entropy")
    sc(axes[2], "omega_bar", "spectral_spread", "<w> (cm-1)", "spread sigma_w (cm-1)")
    fig.suptitle("Variable relationships", y=1.03, fontweight="bold")
    savefig(fig, outdir / "scatter_relations.png"); return "scatter_relations.png"


# --------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parents[1]                # projects/phonon
    ap.add_argument("--phase", default="after")
    ap.add_argument("--pkl", default=str(here / "data" / "raw" /
                    "phdos_e3nn_len51max1000_fwin101ord3.pkl"))
    ap.add_argument("--catalog", default=str(here / "data" / "phonon_catalog.json"))
    ap.add_argument("--out-data", default=str(here / "data"))
    ap.add_argument("--out-figs", default=str(here / "eda" / "figures"))
    args = ap.parse_args()

    figdir = Path(args.out_figs) / args.phase
    figdir.mkdir(parents=True, exist_ok=True)
    datadir = Path(args.out_data) / args.phase
    datadir.mkdir(parents=True, exist_ok=True)

    step(f"loading {Path(args.pkl).name}")
    d = pickle.load(open(args.pkl, "rb"))
    cat = json.load(open(args.catalog))
    by_id = {str(c["id"]): c for c in cat["catalog"]}
    coverage = cat["coverage"]

    ids = np.asarray(d["material_id"])
    g = np.asarray(d["phfre"], float)
    dos = np.asarray(d["phdos"], float)
    dos_gt = np.asarray(d["phdos_gt"], float)
    N = len(ids)
    step(f"{N} materials . grid {len(g)} bins [{g.min():.0f},{g.max():.0f}] cm-1")

    # ---- (3) missing / corrupt (compute on raw arrays) ----
    empty = np.where(dos.sum(1) <= 0)[0]
    neg = np.where((dos < 0).any(1))[0]
    nan_dos = int(np.isnan(dos).sum()); inf_dos = int(np.isinf(dos).sum())
    nan_gt = int(np.isnan(dos_gt).sum()); inf_gt = int(np.isinf(dos_gt).sum())
    gt_min = float(dos_gt.min())
    dup = N - len(set(ids.tolist()))
    grid_mono = bool(np.all(np.diff(g) > 0))

    # ---- (1) derived variables per material ----
    step("deriving spectral + composition variables")
    rows = []
    for i in range(N):
        mid = str(int(ids[i]))
        c = by_id.get(mid, {})
        comp = parse_formula(c.get("formula", ""))
        masses = [_WT[_Z[s]] for s in comp if s in _Z]
        counts = [comp[s] for s in comp if s in _Z]
        zmax = max((_Z[s] for s in comp if s in _Z), default=0)
        meanm = (float(np.average(masses, weights=counts)) if masses else float("nan"))
        feat = spectral_features(g, dos[i])
        feat.update(dict(idx=i, id=mid, formula=c.get("formula", "?"),
                         split=c.get("split", "?"),
                         n_elements=len(comp), n_atoms=int(sum(comp.values())),
                         mean_atomic_mass=meanm, max_Z=int(zmax)))
        rows.append(feat)
    df = pd.DataFrame(rows)

    # ---- (2) type audit ----
    type_cols = [
        dict(name="material_id", logical="int id", dtype=str(ids.dtype), shape=list(ids.shape), note="ok"),
        dict(name="phfre", logical="freq grid (cm-1)", dtype=str(g.dtype), shape=list(g.shape),
             note="shared by all materials"),
        dict(name="phdos", logical="51-bin DOS target", dtype=str(dos.dtype), shape=list(dos.shape),
             note="peak-normalized [0,1]"),
        dict(name="phdos_gt", logical="hi-res DOS (DFPT)", dtype=str(dos_gt.dtype), shape=list(dos_gt.shape),
             note="SG-smoothed; tiny negatives"),
        dict(name="cif", logical="crystal structure", dtype="list[str]", shape=[N],
             note="non-numeric - parsed to graph at train time"),
    ]
    for k, lbl in FEATURES:
        type_cols.append(dict(name=k, logical=lbl, dtype=str(df[k].dtype),
                              shape=[N], note="derived"))
    n_note = (1 if gt_min < 0 else 0) + (1 if len(empty) else 0) + (1 if len(neg) else 0)

    # ---- figures ----
    step("figures: overview")
    figs = []
    figs.append(fig_split_balance(df, figdir))
    figs.append(fig_element_coverage(coverage, figdir))
    figs.append(fig_dos_heatmap(g, dos, df["omega_bar"].values, figdir))
    figs.append(fig_dos_exemplars(g, dos, df, figdir))
    figs.append(fig_dos_binbox(g, dos, figdir))
    figs.append(fig_corr(df, figdir))
    figs.append(fig_scatter(df, figdir))
    step("figures: per-variable hist+box")
    for i, (k, lbl) in enumerate(FEATURES, 1):
        figs.append(fig_hist_box(df, k, lbl, figdir, i))

    # ---- (4) nerd stats ----
    per_feature = {k: col_stats(df[k].values) for k, _ in FEATURES}

    # split balance + element summary (domain blocks)
    split_counts = {s: int((df["split"] == s).sum()) for s in SPLITS}
    cov_present = {k: v for k, v in coverage.items() if v > 0}
    top_elems = sorted(cov_present.items(), key=lambda kv: kv[1], reverse=True)[:12]

    stats = {
        "schema": "eda_stats/1.1",
        "project": "phonon",
        "model_target": "phonon_net",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": args.phase,
        "expectations": {
            "n_materials": 1524, "n_freq_bins": 51, "freq_range_cm": [0, 1000],
            "freq_step_cm": float(g[1] - g[0]), "grid_shared": True,
            "splits": {"train": 1220, "val": 152, "test": 152},
            "target_norm": "peak-normalized DOS (absolute scale via integral g dw = 3N)",
        },
        "missing_corrupt": {
            "counts": {"per_split": split_counts, "total": N},
            "nan_inf": {"phdos_nan": nan_dos, "phdos_inf": inf_dos,
                        "phdos_gt_nan": nan_gt, "phdos_gt_inf": inf_gt},
            "empty_spectra": {"n": int(len(empty)), "ids": [str(int(ids[i])) for i in empty[:20]]},
            "negative_dos": {"n_rows": int(len(neg)),
                             "phdos_min": float(dos.min()),
                             "phdos_gt_min": gt_min,
                             "note": "51-bin target clipped >=0; hi-res gt has SG-smoothing undershoot"},
            "duplicate_ids": int(dup),
            "grid_monotonic": grid_mono,
            "known_issues": ([] if gt_min >= 0 else
                             ["phdos_gt carries tiny negative values (Savitzky-Golay window-101 order-3 smoothing undershoot); harmless for the 51-bin target which is clipped"]),
        },
        "type_audit": {
            "columns": type_cols, "n_rows": N, "n_mismatch_or_note": int(n_note),
            "arrays": {"phdos": {"dtype": str(dos.dtype), "shape": list(dos.shape)},
                       "phdos_gt": {"dtype": str(dos_gt.dtype), "shape": list(dos_gt.shape)},
                       "phfre": {"dtype": str(g.dtype), "shape": list(g.shape)}},
            "homogeneity": "all 1524 DOS share one float64 51-bin grid; only cif is non-numeric",
        },
        "nerd_stats": {
            "per_feature": per_feature,
            "n_features": len(FEATURES),
            "feature_labels": {k: lbl for k, lbl in FEATURES},
            "splits": SPLITS,
        },
        "dos": {
            "freq_cm": g.tolist(),
            "mean_spectrum": dos.mean(0).tolist(),
            "p10": np.percentile(dos, 10, axis=0).tolist(),
            "p90": np.percentile(dos, 90, axis=0).tolist(),
        },
        "elements": {
            "n_present": len(cov_present),
            "n_absent": int(118 - len(cov_present)),
            "top": [{"symbol": s, "count": c} for s, c in top_elems],
            "coverage": cov_present,
        },
        "figures": figs,
    }

    out = datadir / "eda_stats.json"
    def _clean(o):
        if isinstance(o, float):
            return o if math.isfinite(o) else None
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        return o
    out.write_text(json.dumps(_clean(stats), indent=2, allow_nan=False), encoding="utf-8")
    step(f"wrote {out}  ({len(figs)} figures in {figdir})")
    # console digest
    print("\n  derived-variable digest (median [IQR]):")
    for k, lbl in FEATURES:
        s = per_feature[k]
        print(f"    {k:18s} {s['median']:9.3g}  [{s['q1']:.3g}, {s['q3']:.3g}]"
              f"   outliers={s['n_outliers']:3d}  skew={s['skew']:+.2f}")


if __name__ == "__main__":
    main()
