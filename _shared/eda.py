"""Reusable EDA helpers. Light deps only (numpy/pandas/matplotlib) — no torch."""
from __future__ import annotations
import json, os


def write_eda_stats(out_path: str, stats: dict) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)


def plot_class_balance(counts, out_png): raise NotImplementedError("TODO")
def plot_feature_corr(df, out_png):      raise NotImplementedError("TODO")
def plot_mel_exemplars(by_genre, out_png): raise NotImplementedError("TODO")
def plot_leakage_audit(splits, out_png): raise NotImplementedError("TODO")
