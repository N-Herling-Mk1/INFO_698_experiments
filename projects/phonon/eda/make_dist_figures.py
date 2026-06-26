"""
make_dist_figures.py — per-variable distribution figures for the EDA Distributions tab.

Standalone (numpy / pandas / matplotlib / scipy only). Re-derives the SAME 14 derived
variables as run_eda.py (identical formulas), so the figures agree with eda_stats.json
to the digit. Writes, into eda/figures/<phase>/ :

    dist_<var>.png        one dark-themed hist + KDE + boxplot per variable (14 files)
    dist_quality.json     { generated, phase, vars:{ <var>:{label,unit,stats,verdict,flags,blurb} } }

Naming is by VARIABLE NAME (dist_omega_bar.png), not a numeric index — this is what the
Distributions tab references, so the alphabetical-vs-physics ordering bug can never recur.

It does NOT touch eda_stats.json or any figure the other tabs use. Run once:

    python eda/make_dist_figures.py --phase after
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

# ----------------------------------------------------------------- dark theme
BG, PANEL = "#0e1622", "#0e1622"
CYAN, CYAN2, ORANGE, AMBER = "#2de2e6", "#6ff4f7", "#ff8a3c", "#f0a500"
TEXT, MUTED, GRID, EDGE = "#cfe6ea", "#93acb9", "#1c2a3a", "#2a3f52"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "savefig.facecolor": BG,
    "axes.edgecolor": EDGE, "axes.labelcolor": TEXT, "text.color": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "figure.dpi": 120,
})

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
_T0 = time.time()
def step(msg): print(f"[dist +{time.time()-_T0:6.1f}s] {msg}", flush=True)

# ---- atomic data (Z -> symbol, weight); index 0 unused (mirrors run_eda) ----
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
    out = {}
    for sym, n in _FORMULA.findall(str(f)):
        if sym:
            out[sym] = out.get(sym, 0) + (int(n) if n else 1)
    return out

def spectral_features(g, dos):
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
    H = float(-(nz * np.log(nz)).sum() / math.log(len(g)))
    thr = 0.01 * dos.max()
    active = dos > thr
    cutoff = float(g[active][-1]) if active.any() else 0.0
    return dict(omega_bar=cen, spectral_spread=spread, spectral_skew=sk,
                spectral_kurtosis=ku, spectral_entropy=H,
                omega_peak=float(g[int(dos.argmax())]), omega_cutoff=cutoff,
                dos_peak=float(dos.max()),
                dos_integral=float(_trapz(dos, g)),
                n_active_bins=int(active.sum()))

# (key, human label, unit) — order = display order, mirrors run_eda FEATURES
FEATURES = [
    ("omega_bar",        "mean phonon frequency \u27e8\u03c9\u27e9", "cm\u207b\u00b9"),
    ("omega_peak",       "dominant peak frequency",                  "cm\u207b\u00b9"),
    ("omega_cutoff",     "spectral cutoff \u03c9_max",               "cm\u207b\u00b9"),
    ("spectral_spread",  "spectral spread \u03c3_\u03c9",            "cm\u207b\u00b9"),
    ("spectral_entropy", "normalized spectral entropy",              "0\u20131"),
    ("spectral_skew",    "spectral skewness",                        ""),
    ("spectral_kurtosis","spectral kurtosis",                        ""),
    ("dos_peak",         "peak DOS height",                          "norm."),
    ("dos_integral",     "integral g(\u03c9) d\u03c9",               "norm.\u00b7cm\u207b\u00b9"),
    ("n_active_bins",    "active bins (>1% of peak)",                "bins"),
    ("n_elements",       "distinct elements",                        ""),
    ("n_atoms",          "atoms per formula unit",                   ""),
    ("mean_atomic_mass", "mean atomic mass",                         "u"),
    ("max_Z",            "heaviest element Z",                       ""),
]

def col_stats(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    q1, q3 = np.percentile(x, [25, 75]); iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out = (x < lo) | (x > hi)
    return dict(count=int(x.size), mean=float(x.mean()), median=float(np.median(x)),
                std=float(x.std(ddof=1)) if x.size > 1 else 0.0,
                min=float(x.min()), max=float(x.max()), q1=float(q1), q3=float(q3),
                iqr=float(iqr), skew=float(sps.skew(x)) if x.size > 2 else 0.0,
                n_outliers=int(out.sum()), outlier_frac=round(float(out.mean()), 4),
                lo_fence=float(lo), hi_fence=float(hi))

# ------------------------------------------------------------- quality verdict
def verdict_for(key, s, x, n_total, n_nan):
    """Return (verdict in {clean,minor,flag}, flags[list], blurb)."""
    flags = []
    verdict = "clean"
    uniq = int(np.unique(np.round(x, 9)).size) if x.size else 0

    # degeneracy (e.g. peak-normalized dos_peak is pinned at 1.0)
    if s is None or s["std"] == 0 or uniq <= 1:
        flags.append("no variance (degenerate)")
        verdict = "flag"
    # missingness
    if n_nan > 0:
        frac = n_nan / max(n_total, 1)
        flags.append(f"{n_nan} NaN ({frac:.1%})")
        verdict = "flag" if frac > 0.02 else (verdict if verdict == "flag" else "minor")
    if s is not None:
        # heavy outliers
        if s["outlier_frac"] > 0.05:
            flags.append(f"{s['n_outliers']} IQR-outliers ({s['outlier_frac']*100:.1f}%)")
            verdict = verdict if verdict == "flag" else "minor"
        # strong skew
        if abs(s["skew"]) > 1.5:
            flags.append(f"strong skew {s['skew']:+.1f}")
            verdict = verdict if verdict == "flag" else "minor"

    # ---- domain-aware blurb ----
    if key == "dos_peak" and (s is None or s["std"] == 0 or uniq <= 1):
        blurb = ("Peak-normalized target pins every spectrum at 1.0 \u2014 zero variance. "
                 "Carries no information as a feature; expected, not a data error.")
        return "flag", flags, blurb
    if s is None:
        return "flag", flags, "All values missing/non-finite \u2014 cannot profile."

    desc = []
    desc.append(f"median {s['median']:.3g}, IQR [{s['q1']:.3g}, {s['q3']:.3g}]")
    sk = s["skew"]
    shape = ("near-symmetric" if abs(sk) < 0.5 else
             ("right-skewed" if sk > 0 else "left-skewed"))
    desc.append(shape + f" (skew {sk:+.2f})")
    if s["n_outliers"] == 0:
        desc.append("no IQR-outliers")
    else:
        desc.append(f"{s['n_outliers']} IQR-outlier(s) ({s['outlier_frac']*100:.1f}%)")
    tail = {"clean": "Clean, well-behaved.",
            "minor": "Usable \u2014 watch the noted tail.",
            "flag":  "Needs attention."}[verdict]
    blurb = "; ".join(desc) + ". " + tail
    return verdict, flags, blurb

# --------------------------------------------------------------------- figure
def dist_figure(x, key, label, unit, s, outpath):
    """Histogram + KDE on the left, vertical boxplot on the right; dark theme."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.0),
                               gridspec_kw={"width_ratios": [3.2, 1]})
    if x.size == 0:
        a.text(0.5, 0.5, "no finite values", ha="center", va="center", color=MUTED)
        b.axis("off")
    else:
        nb = min(40, max(8, int(np.sqrt(x.size))))
        a.hist(x, bins=nb, color=CYAN, alpha=0.55, edgecolor=BG, linewidth=0.5)
        # KDE overlay (guard against zero-variance)
        if s and s["std"] > 0 and np.unique(x).size > 2:
            try:
                kde = sps.gaussian_kde(x)
                xs = np.linspace(x.min(), x.max(), 200)
                ax2 = a.twinx()
                ax2.plot(xs, kde(xs), color=CYAN2, lw=1.6)
                ax2.set_yticks([]); ax2.grid(False)
                for sp in ax2.spines.values(): sp.set_visible(False)
            except Exception:
                pass
        if s:
            a.axvspan(s["lo_fence"], s["hi_fence"], color=CYAN, alpha=0.05)
            a.axvline(s["median"], color=ORANGE, lw=1.8, label=f"median {s['median']:.3g}")
            a.axvline(s["mean"], color=AMBER, lw=1.3, ls="--", label=f"mean {s['mean']:.3g}")
            a.legend(fontsize=8, frameon=False, labelcolor=TEXT)
        xl = label + (f"  ({unit})" if unit else "")
        a.set_xlabel(xl); a.set_ylabel("materials")
        bp = b.boxplot(x, vert=True, patch_artist=True, widths=0.6,
                       flierprops=dict(marker="o", ms=2.6, mfc=ORANGE, mec="none", alpha=0.55))
        bp["boxes"][0].set(facecolor=CYAN, alpha=0.30, edgecolor=TEXT)
        for med in bp["medians"]: med.set(color=ORANGE, lw=1.8)
        for w in bp["whiskers"] + bp["caps"]: w.set(color=MUTED)
        b.set_xticks([]); b.set_ylabel("")
    fig.suptitle(key, y=1.02, color=CYAN, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------- driver
def main():
    here = Path(__file__).resolve().parents[1]                 # projects/phonon
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="after")
    ap.add_argument("--pkl", default=str(here / "data" / "raw" /
                    "phdos_e3nn_len51max1000_fwin101ord3.pkl"))
    ap.add_argument("--catalog", default=str(here / "data" / "phonon_catalog.json"))
    ap.add_argument("--out-figs", default=str(here / "eda" / "figures"))
    args = ap.parse_args()

    figdir = Path(args.out_figs) / args.phase
    figdir.mkdir(parents=True, exist_ok=True)

    step(f"loading {Path(args.pkl).name}")
    d = pickle.load(open(args.pkl, "rb"))
    cat = json.load(open(args.catalog))
    by_id = {str(c["id"]): c for c in cat["catalog"]}

    ids = np.asarray(d["material_id"])
    g = np.asarray(d["phfre"], float)
    dos = np.asarray(d["phdos"], float)
    N = len(ids)
    step(f"{N} materials . grid {len(g)} bins")

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
        feat.update(dict(n_elements=len(comp), n_atoms=int(sum(comp.values())),
                         mean_atomic_mass=meanm, max_Z=int(zmax)))
        rows.append(feat)
    df = pd.DataFrame(rows)

    step("rendering per-variable figures + quality verdicts")
    quality = {}
    for j, (key, label, unit) in enumerate(FEATURES, 1):
        col = df[key].values.astype(float)
        x = col[np.isfinite(col)]
        n_nan = int(np.isnan(col).sum())
        s = col_stats(col)
        out = figdir / f"dist_{key}.png"
        dist_figure(x, key, label, unit, s, out)
        verdict, flags, blurb = verdict_for(key, s, x, N, n_nan)
        quality[key] = {"label": label, "unit": unit, "verdict": verdict,
                        "flags": flags, "blurb": blurb,
                        "stats": (None if s is None else {
                            "median": s["median"], "mean": s["mean"], "std": s["std"],
                            "min": s["min"], "max": s["max"], "iqr": s["iqr"],
                            "skew": s["skew"], "n_outliers": s["n_outliers"],
                            "outlier_frac": s["outlier_frac"], "n_nan": n_nan})}
        step(f"  [{j:2d}/14] dist_{key}.png  -> {verdict}")

    qpath = figdir / "dist_quality.json"
    payload = {"schema": "dist_quality/1.0", "phase": args.phase,
               "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "order": [k for k, _, _ in FEATURES], "vars": quality}
    def _clean(o):
        if isinstance(o, float): return o if math.isfinite(o) else None
        if isinstance(o, dict):  return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):  return [_clean(v) for v in o]
        return o
    qpath.write_text(json.dumps(_clean(payload), indent=2, allow_nan=False), encoding="utf-8")
    step(f"wrote {qpath}  (+ 14 dist_*.png in {figdir})")

    print("\n  quality digest:")
    for k, _, _ in FEATURES:
        q = quality[k]
        print(f"    {k:18s} {q['verdict']:5s}  {q['blurb']}")


if __name__ == "__main__":
    main()
