"""
make_overview_extras.py — two Overview-tab figure fixes for the phonon EDA page.

Standalone (numpy / pandas / matplotlib / scipy only). Writes into eda/figures/<phase>/ :

  dos_exemplars.png     REGENERATED at a matched aspect ratio (1.652, == dos_heatmap)
                        so it renders the same height beside the heatmap; subplot titles
                        now use the real \u27e8\u03c9\u27e9 glyph (no more literal <w>).
  element_ptable.png    NEW periodic-table heatmap: each cell colored by how many
                        materials contain that element (magma; grey = absent).

Light theme to match the other Overview figures (heatmap / split_balance / element_coverage).
Run once, after make_dist_figures, from projects/phonon (raw pkl present in data/raw/):

    python eda/make_overview_extras.py --phase after
"""
from __future__ import annotations
import argparse, json, math, pickle, re, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import PowerNorm
import matplotlib.cm as cm

# ---- light theme, matches run_eda figures ----------------------------------
CYAN, ORANGE, INK, GRID = "#0bb4c4", "#ff5a1f", "#0b2030", "#d7e2e6"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "figure.dpi": 120,
})
_T0 = time.time()
def step(m): print(f"[ov +{time.time()-_T0:5.1f}s] {m}", flush=True)

OMEGA = "\u27e8\u03c9\u27e9"   # ⟨ω⟩

_SYM = ("0 H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe "
        "Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn "
        "Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re "
        "Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm "
        "Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split()
_Z = {s: i for i, s in enumerate(_SYM)}
_FORMULA = re.compile(r"([A-Z][a-z]?)(\d*)")

def parse_formula(f):
    out = {}
    for sym, n in _FORMULA.findall(str(f)):
        if sym:
            out[sym] = out.get(sym, 0) + (int(n) if n else 1)
    return out

def spectral_min(g, dos):
    """Just the reductions the exemplar picks need."""
    s = dos.sum()
    if s <= 0:
        z = float("nan")
        return z, z, z, 0.0
    p = dos / s
    cen = float((p * g).sum())
    var = float((p * (g - cen) ** 2).sum())
    spread = math.sqrt(var) if var > 0 else 0.0
    nz = p[p > 0]
    H = float(-(nz * np.log(nz)).sum() / math.log(len(g)))
    return cen, spread, H, float(dos.max())

# --------------------------------------------------------------- exemplars fig
def fig_exemplars(g, dos, df, outdir):
    lo = df["omega_bar"].idxmin(); hi = df["omega_bar"].idxmax()
    mid = (df["omega_bar"] - df["omega_bar"].median()).abs().idxmin()
    widest = df["spectral_spread"].idxmax(); peaky = df["dos_peak"].idxmax()
    sharp = df["spectral_entropy"].idxmin()
    picks = [(lo, f"lowest {OMEGA}"), (hi, f"highest {OMEGA}"), (mid, f"median {OMEGA}"),
             (widest, "widest"), (peaky, "tallest peak"), (sharp, "lowest entropy")]
    # aspect 9.5/5.75 = 1.652 == dos_heatmap (7.6/4.6) -> equal rendered height
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.75), sharex=True)
    for ax, (idx, tag) in zip(axes.ravel(), picks):
        ax.fill_between(g, dos[idx], color=CYAN, alpha=0.35)
        ax.plot(g, dos[idx], color=CYAN, lw=1.2)
        ax.set_title(f"{df.loc[idx,'formula']} . {tag}", fontsize=9)
        ax.set_xlim(0, 1000); ax.grid(True, color=GRID, lw=0.8)
    for ax in axes[-1]:
        ax.set_xlabel(f"\u03c9 (cm\u207b\u00b9)")
    fig.suptitle("DOS exemplars", y=0.99, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "dos_exemplars.png", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------- periodic table
# (symbol, row, col) — main block rows 1-7; f-block on rows 8.4 / 9.4 (gap below table)
_PT = [
    ("H",1,1),("He",1,18),
    ("Li",2,1),("Be",2,2),("B",2,13),("C",2,14),("N",2,15),("O",2,16),("F",2,17),("Ne",2,18),
    ("Na",3,1),("Mg",3,2),("Al",3,13),("Si",3,14),("P",3,15),("S",3,16),("Cl",3,17),("Ar",3,18),
    ("K",4,1),("Ca",4,2),("Sc",4,3),("Ti",4,4),("V",4,5),("Cr",4,6),("Mn",4,7),("Fe",4,8),
    ("Co",4,9),("Ni",4,10),("Cu",4,11),("Zn",4,12),("Ga",4,13),("Ge",4,14),("As",4,15),("Se",4,16),("Br",4,17),("Kr",4,18),
    ("Rb",5,1),("Sr",5,2),("Y",5,3),("Zr",5,4),("Nb",5,5),("Mo",5,6),("Tc",5,7),("Ru",5,8),
    ("Rh",5,9),("Pd",5,10),("Ag",5,11),("Cd",5,12),("In",5,13),("Sn",5,14),("Sb",5,15),("Te",5,16),("I",5,17),("Xe",5,18),
    ("Cs",6,1),("Ba",6,2),("Hf",6,4),("Ta",6,5),("W",6,6),("Re",6,7),("Os",6,8),("Ir",6,9),
    ("Pt",6,10),("Au",6,11),("Hg",6,12),("Tl",6,13),("Pb",6,14),("Bi",6,15),("Po",6,16),("At",6,17),("Rn",6,18),
    ("Fr",7,1),("Ra",7,2),("Rf",7,4),("Db",7,5),("Sg",7,6),("Bh",7,7),("Hs",7,8),("Mt",7,9),
    ("Ds",7,10),("Rg",7,11),("Cn",7,12),("Nh",7,13),("Fl",7,14),("Mc",7,15),("Lv",7,16),("Ts",7,17),("Og",7,18),
    ("La",8.4,3),("Ce",8.4,4),("Pr",8.4,5),("Nd",8.4,6),("Pm",8.4,7),("Sm",8.4,8),("Eu",8.4,9),("Gd",8.4,10),
    ("Tb",8.4,11),("Dy",8.4,12),("Ho",8.4,13),("Er",8.4,14),("Tm",8.4,15),("Yb",8.4,16),("Lu",8.4,17),
    ("Ac",9.4,3),("Th",9.4,4),("Pa",9.4,5),("U",9.4,6),("Np",9.4,7),("Pu",9.4,8),("Am",9.4,9),("Cm",9.4,10),
    ("Bk",9.4,11),("Cf",9.4,12),("Es",9.4,13),("Fm",9.4,14),("Md",9.4,15),("No",9.4,16),("Lr",9.4,17),
]

SPLITS = ["train", "val", "test"]

def fig_split_balance(split_counts, outdir):
    vals = [int(split_counts.get(s, 0)) for s in SPLITS]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    bars = ax.bar(SPLITS, vals, color=[CYAN, ORANGE, "#7a8b91"], edgecolor=INK)
    top = max(vals) if vals else 1
    for r, v in zip(bars, vals):
        ax.text(r.get_x()+r.get_width()/2, v + top*0.02, str(v), ha="center", fontsize=9)
    ax.set_ylim(0, top*1.14)                      # headroom so the count label isn't clipped
    ax.set_ylabel("materials"); ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_title(f"Split balance  ({vals[0]} / {vals[1]} / {vals[2]})")
    fig.tight_layout(); fig.savefig(outdir / "split_balance.png", bbox_inches="tight"); plt.close(fig)

def fig_element_coverage(coverage, outdir, top=40):
    items = sorted(((k, int(v)) for k, v in coverage.items() if int(v) > 0),
                   key=lambda kv: kv[1], reverse=True)[:top]
    syms = [k for k, _ in items]; cnts = [v for _, v in items]
    npres = sum(1 for v in coverage.values() if int(v) > 0)
    # aspect 9.5/6.13 = 1.549 == split_balance -> equal rendered height beside it
    fig, ax = plt.subplots(figsize=(9.5, 6.13))
    ax.bar(range(len(syms)), cnts, color=CYAN, edgecolor=INK, linewidth=0.4)
    ax.set_xticks(range(len(syms))); ax.set_xticklabels(syms, fontsize=9, rotation=0)
    ax.set_ylabel("materials containing"); ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_title(f"Element coverage - top {top} of {npres} present")
    fig.tight_layout(); fig.savefig(outdir / "element_coverage.png", bbox_inches="tight"); plt.close(fig)

def fig_ptable(coverage, outdir):
    cov = {k: int(v) for k, v in coverage.items() if int(v) > 0}
    vmax = max(cov.values()) if cov else 1
    norm = PowerNorm(0.6, vmin=0, vmax=vmax)
    cmap = plt.get_cmap("magma")
    fig, ax = plt.subplots(figsize=(11.0, 6.3))
    for sym, r, c in _PT:
        n = cov.get(sym, 0)
        if n > 0:
            rgba = cmap(norm(n)); lum = 0.299*rgba[0]+0.587*rgba[1]+0.114*rgba[2]
            txt = "white" if lum < 0.55 else INK; face = rgba; edge = "white"
        else:
            txt = "#9fb0b6"; face = "#eef3f5"; edge = "#dde6e9"
        ax.add_patch(Rectangle((c, -r), 0.92, 0.92, facecolor=face, edgecolor=edge, linewidth=0.6))
        ax.text(c+0.46, -r+0.60, sym, ha="center", va="center", fontsize=8.5,
                fontweight="bold", color=txt)
        if n > 0:
            ax.text(c+0.46, -r+0.26, str(n), ha="center", va="center", fontsize=6.2, color=txt)
    ax.set_xlim(0.6, 19.1); ax.set_ylim(-10.2, 0.05)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"Element coverage \u2014 periodic table  ({len(cov)} of 118 present)",
                 fontsize=12, fontweight="bold")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("materials containing element", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "element_ptable.png", bbox_inches="tight"); plt.close(fig)
    return len(cov)

# --------------------------------------------------------------------- driver
def main():
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="after")
    ap.add_argument("--pkl", default=str(here / "data" / "raw" /
                    "phdos_e3nn_len51max1000_fwin101ord3.pkl"))
    ap.add_argument("--catalog", default=str(here / "data" / "phonon_catalog.json"))
    ap.add_argument("--out-figs", default=str(here / "eda" / "figures"))
    args = ap.parse_args()
    figdir = Path(args.out_figs) / args.phase
    figdir.mkdir(parents=True, exist_ok=True)

    step("loading catalog (coverage + splits)")
    cat = json.load(open(args.catalog))
    coverage = cat.get("coverage", {})
    catalog = cat.get("catalog", [])
    split_counts = {}
    for c in catalog:
        s = c.get("split", "?"); split_counts[s] = split_counts.get(s, 0) + 1
    fig_split_balance(split_counts, figdir)
    step(f"split_balance.png  ({'/'.join(str(split_counts.get(s,0)) for s in SPLITS)}, label headroom)")
    fig_element_coverage(coverage, figdir)
    step("element_coverage.png  (taller, matched aspect 1.549)")
    n_present = fig_ptable(coverage, figdir)
    step(f"element_ptable.png  ({n_present} elements present)")

    step(f"loading {Path(args.pkl).name}")
    d = pickle.load(open(args.pkl, "rb"))
    by_id = {str(c["id"]): c for c in cat["catalog"]}
    ids = np.asarray(d["material_id"]); g = np.asarray(d["phfre"], float)
    dos = np.asarray(d["phdos"], float); N = len(ids)

    step("deriving exemplar-pick features")
    rows = []
    for i in range(N):
        cen, spread, H, peak = spectral_min(g, dos[i])
        c = by_id.get(str(int(ids[i])), {})
        rows.append(dict(omega_bar=cen, spectral_spread=spread, spectral_entropy=H,
                         dos_peak=peak, formula=c.get("formula", "?")))
    df = pd.DataFrame(rows)
    fig_exemplars(g, dos, df, figdir)
    step("dos_exemplars.png  (matched aspect 1.652)")
    step(f"done -> {figdir}")


if __name__ == "__main__":
    main()
