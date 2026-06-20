#!/usr/bin/env python
"""Regenerate GTZAN mel spectrograms from audio with ONE recipe — the exact
features.extract_mel used at inference — into a NEW directory (default images_mel/),
leaving the Kaggle images_grey_scale/ untouched for provenance + A/B comparison.

Why: the Kaggle grey PNGs are colormapped matplotlib figures (white margins +
non-monotonic grey mapping; see EDA), so a clean drop-in mel is out-of-distribution
against them (validate_mel_recipe -> MISMATCH). Regenerating with extract_mel makes
training images and inference share one recipe -> validate passes by construction.

Recipe params come from beardown.yaml's `mel:` block (single source of truth).
Point the loader at the output via `features.image_dir` in the config.

    python projects/genre/eda/build_spectrograms.py --data-root projects/genre/data/raw
"""
from __future__ import annotations
import argparse, sys, json, time
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root
from projects.genre.src import features as F                    # noqa: E402
import yaml                                                     # noqa: E402

C, A, R = "\033[36m", "\033[33m", "\033[0m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="projects/genre/data/raw")
    ap.add_argument("--config", default="projects/genre/configs/beardown.yaml")
    ap.add_argument("--audio-dir", default="genres_original")
    ap.add_argument("--out-dir", default="images_mel", help="NEW dir; never overwrites images_grey_scale")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    mel = cfg.get("mel", {})
    image_size = cfg.get("features", {}).get("image_size", 128)
    n_mels = mel.get("n_mels", F.N_MELS); n_fft = mel.get("n_fft", F.N_FFT); hop = mel.get("hop_length", F.HOP)

    root = Path(args.data_root)
    audio = root / args.audio_dir
    out = root / args.out_dir
    if not audio.exists():
        raise FileNotFoundError(f"audio dir not found: {audio.resolve()}")
    if args.out_dir == "images_grey_scale":
        raise SystemExit("refusing to write into images_grey_scale (Kaggle originals); pick a new --out-dir")

    print(f"{C}🐻 ═══ regenerate spectrograms -> {args.out_dir}/ ═══ 🐻{R}")
    print(f"   recipe: extract_mel  n_mels={n_mels} n_fft={n_fft} hop={hop} size={image_size}")
    print(f"   originals (images_grey_scale/) left untouched\n")

    wavs = sorted(audio.rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"no .wav under {audio.resolve()} (did the audio get copied?)")
    print(f"   {len(wavs)} wavs found")

    n = 0; t0 = time.time()
    for i, w in enumerate(wavs):
        genre = w.parent.name
        undot = w.stem.replace(".", "")             # jazz.00054 -> jazz00054
        try:
            y, sr = F.load_audio(w)
            img = F.extract_mel(y, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop, image_size=image_size)
        except Exception as e:
            print(f"\n{A}   skip {w.name}: {e}{R}"); continue
        d = out / genre; d.mkdir(parents=True, exist_ok=True)
        Image.fromarray((img * 255).astype(np.uint8), mode="L").save(d / f"{undot}.png")
        n += 1
        if (i + 1) % 25 == 0 or i + 1 == len(wavs):
            bar = "█" * int(28 * (i + 1) / len(wavs))
            print(f"\r{C}   [{bar:<28}] {i+1}/{len(wavs)}  ({genre}){R}", end="", flush=True)
    print()

    manifest = {"source": "features.extract_mel (clean log-mel)", "out_dir": args.out_dir,
                "leaves_intact": "images_grey_scale (Kaggle colormapped figures)",
                "n_generated": n, "image_size": image_size, "n_mels": n_mels,
                "n_fft": n_fft, "hop_length": hop, "sample_rate": F.SR,
                "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "note": "set features.image_dir=%s in beardown.yaml to train on these." % args.out_dir}
    (out / "_recipe.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{C}   wrote {n} clean log-mel PNGs in {time.time()-t0:.1f}s -> {out}{R}")
    print(f"{A}   next: set features.image_dir: {args.out_dir} in beardown.yaml, re-validate, retrain{R}")


if __name__ == "__main__":
    main()
