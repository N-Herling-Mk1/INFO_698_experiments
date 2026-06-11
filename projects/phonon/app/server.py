"""FORGE per-experiment mini-stack — backend (canonical template).

Reads the artifacts an experiment emits and serves them to the TRON-Ares
dashboard. Trains nothing. Two EDA snapshots per experiment:

    <root>/data/<phase>/eda_stats.json        phase in {before, after}
    <root>/eda/figures/<phase>/*.png

'before' is the pre-fix compendium; 'after' is regenerated after the
fix/imputation pass. The dashboard toggles between them.

Dual context — ONE file, two homes:
  • In the repo, this lives at projects/<exp>/app/ and reads its SIBLINGS
    (../data, ../eda) so the app and the training tier share one source of truth.
  • In an exported standalone bundle, server.py sits at the bundle root with
    data/ and eda/ beside it.
The root is auto-detected below, so the same file works in both.

Clone discipline: copy this folder unchanged into a new experiment. The
experiment name is auto-derived from the folder; keep route names + eda_stats.json
keys identical across genre/phonon/atlas so the eventual merge stays mechanical.

    pip install flask
    python server.py            # -> http://127.0.0.1:5000
"""
from __future__ import annotations
import json, os
from pathlib import Path
from flask import Flask, jsonify, send_file, send_from_directory, request, abort

import config

APP_DIR = Path(__file__).resolve().parent
# self-contained bundle: data/ sits beside server.py; repo: data/ is in the parent
EXP_ROOT = APP_DIR if (APP_DIR / "data").exists() else APP_DIR.parent
EXPERIMENT = config.EXPERIMENT or EXP_ROOT.name
DATA = EXP_ROOT / "data"
FIGS = EXP_ROOT / "eda" / "figures"
RUNS = EXP_ROOT / "runs"

app = Flask(__name__, static_folder="static", static_url_path="/static")


def _phase():
    p = request.args.get("phase", config.DEFAULT_PHASE)
    return p if p in config.PHASES else config.DEFAULT_PHASE


def _stats(phase):
    return DATA / phase / "eda_stats.json"


def _fig_count(phase):
    d = FIGS / phase
    return len(list(d.glob("*.png"))) if d.exists() else 0


# ---- routes (KEEP NAMES IDENTICAL ACROSS EXPERIMENTS) -----------------------
@app.get("/")
def index():
    return send_file(APP_DIR / "templates" / "index.html")


@app.get("/api/config")
def cfg():
    return jsonify(experiment=EXPERIMENT, phases=config.PHASES,
                   default_phase=config.DEFAULT_PHASE,
                   available={p: _stats(p).exists() for p in config.PHASES})


@app.get("/api/health")
def health():
    return jsonify(experiment=EXPERIMENT, ok=True,
                   figures={p: _fig_count(p) for p in config.PHASES},
                   eda_present={p: _stats(p).exists() for p in config.PHASES})


@app.get("/api/eda")
def eda():
    phase = _phase()
    fp = _stats(phase)
    if not fp.exists():
        return jsonify(error=f"no '{phase}' EDA yet — run eda/run_eda.py --phase {phase}",
                       experiment=EXPERIMENT, phase=phase), 404
    return jsonify(json.loads(fp.read_text()))


@app.get("/api/runs")
def runs():
    """Training tier (run.json / compute.json). Empty until train.py emits."""
    out = []
    if RUNS.exists():
        for rj in sorted(RUNS.glob("*/*/run.json")):
            try:
                out.append(json.loads(rj.read_text()))
            except Exception:
                pass
    return jsonify(runs=out, experiment=EXPERIMENT)


@app.get("/figures/<phase>/<path:name>")
def figures(phase, name):
    if phase not in config.PHASES:
        abort(404)
    d = FIGS / phase
    if not d.exists():
        abort(404)
    return send_from_directory(d, name)


def _banner():
    C, A, R = "\033[36m", "\033[33m", "\033[0m"
    line = "═" * 50
    print(f"{C}╔{line}╗{R}")
    print(f"{C}║  FORGE mini-stack   experiment: {A}{EXPERIMENT:<17}{C}║{R}")
    print(f"{C}╚{line}╝{R}")
    print(f"  root         : {EXP_ROOT}")
    for p in config.PHASES:
        mark = "FOUND" if _stats(p).exists() else "—    "
        print(f"  EDA [{p:<6}] : {mark}   figures: {_fig_count(p)}")
    print(f"  serving      : http://{config.HOST}:{config.PORT}\n")


if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        _banner()
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
