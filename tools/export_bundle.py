"""Export a self-contained, standalone dashboard bundle for one experiment.

The repo keeps ONE source of truth: the app reads sibling data/ + eda/. This tool
produces a flat, shippable zip where those artifacts are copied beside server.py
(the form you'd hand someone who just wants to open the dashboard). No second
codebase — the bundle is a build artifact, regenerated on demand.

    python tools/export_bundle.py genre
    python tools/export_bundle.py genre --out dist/ --phase before after
"""
from __future__ import annotations
import argparse, shutil, zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "_app_template"


def export(experiment: str, out_dir: Path, phases: list[str]) -> Path:
    exp = REPO / "projects" / experiment
    if not exp.exists():
        raise SystemExit(f"no such experiment: projects/{experiment}")

    stage = out_dir / f"{experiment}-app"
    if stage.exists():
        shutil.rmtree(stage)
    # app shell (server, config, static, templates, README)
    shutil.copytree(TEMPLATE, stage)

    # pin the experiment name so a renamed bundle folder still reports correctly
    cfg = stage / "config.py"
    cfg.write_text(cfg.read_text().replace("EXPERIMENT = None",
                                           f'EXPERIMENT = "{experiment}"'))

    # copy the EDA generator + only the requested snapshots (self-contained layout)
    (stage / "eda").mkdir(exist_ok=True)
    shutil.copy(exp / "eda" / "run_eda.py", stage / "eda" / "run_eda.py")
    for ph in phases:
        src_fig = exp / "eda" / "figures" / ph
        src_dat = exp / "data" / ph
        if src_fig.exists():
            shutil.copytree(src_fig, stage / "eda" / "figures" / ph, dirs_exist_ok=True)
        if src_dat.exists():
            shutil.copytree(src_dat, stage / "data" / ph, dirs_exist_ok=True)

    zip_path = out_dir / f"{experiment}-app.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in stage.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(out_dir))
    n = sum(1 for _ in stage.rglob("*") if _.is_file())
    print(f"[export] {experiment}: {n} files -> {zip_path}")
    return zip_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="genre | phonon | atlas")
    ap.add_argument("--out", default="dist", help="output dir (default: dist/)")
    ap.add_argument("--phase", nargs="+", default=["before", "after"],
                    help="which snapshots to include")
    args = ap.parse_args()
    out = (REPO / args.out)
    out.mkdir(parents=True, exist_ok=True)
    export(args.experiment, out, args.phase)


if __name__ == "__main__":
    main()
