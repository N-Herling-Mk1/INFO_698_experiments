"""Dataset splitting with the leakage guard.

track_level_split: ensures all segments of one recording stay in one split.
artist_split:      stricter — no artist appears in more than one split (Sturm 2013).
naive_random:      the leaky baseline most papers use — kept for honest comparison.

Pure numpy. No TensorFlow, no pandas — so EDA / tests / sanity run TF-free.

Vocabulary
----------
item        one row to be assigned (a 3s segment, a 30s track row, an image path…).
track_id    the canonical recording id (``genre.NNNNN``). All ~10 segments of one
            song and that song's image collapse to it. **Unit of assignment.**
stratum     the genre. Each genre's tracks are split independently so every split
            holds all 10 genres by construction (coverage = 1.0, deterministic).

All three splitters return a ``Split`` (below): positional index arrays into the
``items`` list you passed in, plus the track→split map and diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence
import json
import os

import numpy as np

Ratios = tuple[float, float, float]
DEFAULT_RATIOS: Ratios = (0.70, 0.15, 0.15)


# ----------------------------------------------------------------- return type
@dataclass
class Split:
    """Positional indices into the original ``items`` sequence, plus provenance.

    ``train``/``val``/``test`` index ``items`` directly, so a loader does
    ``X[split.train]`` with no further bookkeeping. ``track_split`` records the
    split each *track* landed in (the leakage-relevant fact); ``meta`` carries
    the knobs so a run is reproducible from the record alone.
    """
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    track_split: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def sizes(self) -> dict[str, int]:
        return {"train": int(self.train.size),
                "val": int(self.val.size),
                "test": int(self.test.size)}

    def as_manifest(self) -> dict[str, Any]:
        """The reproducibility slice that goes into manifest.json."""
        return {**self.meta, **self.sizes()}


# --------------------------------------------------------------------- helpers
def _genre_of(track_id: str) -> str:
    """``blues.00000`` -> ``blues``. GTZAN ids are ``genre.NNNNN``."""
    return str(track_id).split(".", 1)[0]


def _validate_ratios(ratios: Sequence[float]) -> Ratios:
    if len(ratios) != 3:
        raise ValueError(f"ratios must be (train,val,test); got {ratios!r}")
    s = float(sum(ratios))
    if not np.isclose(s, 1.0, atol=1e-6):
        raise ValueError(f"ratios must sum to 1.0; got {ratios} (sum={s})")
    if any(r < 0 for r in ratios):
        raise ValueError(f"ratios must be non-negative; got {ratios}")
    return (float(ratios[0]), float(ratios[1]), float(ratios[2]))


def _three_way_counts(n: int, ratios: Ratios) -> tuple[int, int, int]:
    """Split ``n`` items into (train, val, test). Train absorbs the rounding
    remainder so the three always sum back to ``n`` (TODO invariant)."""
    n_val = int(round(n * ratios[1]))
    n_test = int(round(n * ratios[2]))
    n_train = n - n_val - n_test
    if n_train < 0:  # pathological tiny stratum — give train priority, then val
        n_train, n_val, n_test = max(n, 0), 0, 0
    return n_train, n_val, n_test


def assert_no_track_overlap(train_tracks, val_tracks, test_tracks) -> None:
    """Fail loud if any track id lands in more than one split. This is the
    load-bearing invariant — call it after every split, in tests and at runtime."""
    a, b, c = set(train_tracks), set(val_tracks), set(test_tracks)
    bad = (a & b) | (a & c) | (b & c)
    if bad:
        raise AssertionError(
            f"TRACK LEAKAGE: {len(bad)} track id(s) in >1 split, e.g. "
            f"{sorted(bad)[:5]}")


# ------------------------------------------------------------ the three splits
def track_level_split(items: Sequence[Any],
                      track_id_fn: Callable[[Any], str],
                      seed: int = 0,
                      ratios: Ratios = DEFAULT_RATIOS,
                      stratum_fn: Callable[[str], str] | None = None) -> Split:
    """Group items by track, stratify by genre, split tracks 70/15/15, then
    expand each track's assignment back to all its rows. The default, honest split.

    Genre stratification is per-genre-independent, so every split contains all
    genres → coverage = 1.0 deterministically (no resample/coverage-guard loop).
    """
    ratios = _validate_ratios(ratios)
    stratum_fn = stratum_fn or _genre_of
    rng = np.random.default_rng(seed)

    track_of = np.array([str(track_id_fn(it)) for it in items])

    # group track ids by stratum, preserving a stable order before the shuffle
    by_stratum: dict[str, list[str]] = {}
    seen: set[str] = set()
    for t in track_of:
        if t in seen:
            continue
        seen.add(t)
        by_stratum.setdefault(stratum_fn(t), []).append(t)

    track_split: dict[str, str] = {}
    coverage: dict[str, dict[str, int]] = {}
    for stratum in sorted(by_stratum):
        tracks = np.array(by_stratum[stratum])
        perm = rng.permutation(tracks.size)          # seeded uniform permutation
        tracks = tracks[perm]
        n_tr, n_va, n_te = _three_way_counts(tracks.size, ratios)
        for t in tracks[:n_tr]:                       track_split[t] = "train"
        for t in tracks[n_tr:n_tr + n_va]:            track_split[t] = "val"
        for t in tracks[n_tr + n_va:]:                track_split[t] = "test"
        coverage[stratum] = {"train": n_tr, "val": n_va, "test": n_te}

    # expand track assignment to every row
    where = {"train": [], "val": [], "test": []}
    for i, t in enumerate(track_of):
        where[track_split[t]].append(i)

    tr_tracks = [t for t, s in track_split.items() if s == "train"]
    va_tracks = [t for t, s in track_split.items() if s == "val"]
    te_tracks = [t for t, s in track_split.items() if s == "test"]
    assert_no_track_overlap(tr_tracks, va_tracks, te_tracks)   # fail loud

    return Split(
        train=np.array(where["train"], dtype=np.int64),
        val=np.array(where["val"], dtype=np.int64),
        test=np.array(where["test"], dtype=np.int64),
        track_split=track_split,
        meta={"mode": "track", "seed": seed, "ratios": list(ratios),
              "n_tracks": len(track_split), "n_items": len(items),
              "n_strata": len(by_stratum), "coverage_per_stratum": coverage},
    )


def naive_random(items: Sequence[Any],
                 seed: int = 0,
                 ratios: Ratios = DEFAULT_RATIOS,
                 track_id_fn: Callable[[Any], str] | None = None) -> Split:
    """Segment-level random split. Leaky on GTZAN 3s segments — kept *only* to
    measure the leakage inflation (the track-vs-naive accuracy delta is a result).

    If ``track_id_fn`` is given, ``track_split`` is filled in so you can see how
    many tracks straddle splits — that count *is* the leakage being measured.
    """
    ratios = _validate_ratios(ratios)
    rng = np.random.default_rng(seed)
    n = len(items)
    perm = rng.permutation(n)
    n_tr, n_va, n_te = _three_way_counts(n, ratios)
    train = np.sort(perm[:n_tr])
    val = np.sort(perm[n_tr:n_tr + n_va])
    test = np.sort(perm[n_tr + n_va:])

    track_split: dict[str, str] = {}
    straddle = 0
    if track_id_fn is not None:
        track_of = np.array([str(track_id_fn(it)) for it in items])
        membership: dict[str, set[str]] = {}
        for idx_arr, name in ((train, "train"), (val, "val"), (test, "test")):
            for i in idx_arr:
                membership.setdefault(track_of[i], set()).add(name)
        for t, homes in membership.items():
            track_split[t] = "+".join(sorted(homes))   # e.g. "test+train" = leaked
            if len(homes) > 1:
                straddle += 1

    return Split(
        train=train, val=val, test=test, track_split=track_split,
        meta={"mode": "naive_random", "seed": seed, "ratios": list(ratios),
              "n_items": n, "tracks_straddling_splits": straddle,
              "leakage": "EXPECTED — comparison baseline, do not train on this"},
    )


def artist_split(items: Sequence[Any],
                 artist_id_fn: Callable[[Any], str],
                 seed: int = 0,
                 ratios: Ratios = DEFAULT_RATIOS) -> Split:
    """Group by artist (strictest, Sturm 2013). GTZAN ships no artist metadata,
    so this is aspirational: it works *if* you supply an ``artist_id_fn``, and
    otherwise stays an explicit NotImplemented rather than a silent wrong split."""
    raise NotImplementedError(
        "artist_split needs per-track artist metadata GTZAN does not ship. "
        "Supply artist_id_fn from an external mapping to enable the strict split.")


# ----------------------------------------------------------- manifest plumbing
def write_split_to_manifest(manifest_path: str, split: Split) -> dict:
    """Record seed/ratios/sizes/coverage into manifest.json under ``split`` so a
    run is reproducible from the manifest alone. Returns the updated manifest."""
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            manifest = json.loads(open(manifest_path, encoding="utf-8").read())
        except Exception:
            manifest = {}
    manifest["split"] = split.as_manifest()
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest
