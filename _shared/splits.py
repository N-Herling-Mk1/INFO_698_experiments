"""Dataset splitting with the leakage guard.

track_level_split: ensures all segments of one recording stay in one split.
artist_split:      stricter — no artist appears in more than one split (Sturm 2013).
naive_random:      the leaky baseline most papers use — kept for honest comparison.
"""
from __future__ import annotations


def naive_random(items, seed=0, ratios=(0.7, 0.15, 0.15)):
    """Segment-level random split. Leaky on GTZAN 3s segments — for comparison only."""
    raise NotImplementedError("TODO")


def track_level_split(items, track_id_fn, seed=0, ratios=(0.7, 0.15, 0.15)):
    """Group by track id, then split groups. The default, honest split."""
    raise NotImplementedError("TODO")


def artist_split(items, artist_id_fn, seed=0, ratios=(0.7, 0.15, 0.15)):
    """Group by artist. Strictest. Needs artist metadata (often unavailable for GTZAN)."""
    raise NotImplementedError("TODO")
