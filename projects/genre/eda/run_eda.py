"""GTZAN EDA entrypoint. Emits figures/*.png + eda_stats.json (site-consumable).
Integrity-first: class balance, audio sanity, tabular dists, mel exemplars, leakage audit.
    python projects/genre/eda/run_eda.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from _shared import eda

def main():
    out = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out, exist_ok=True)
    # TODO: load data/manifest.json + raw audio / 58-feat csv, then:
    #   eda.plot_class_balance(...)   eda.plot_feature_corr(...)
    #   eda.plot_mel_exemplars(...)   eda.plot_leakage_audit(...)  <- the load-bearing one
    eda.write_eda_stats(
        os.path.join(os.path.dirname(__file__), "..", "data", "eda_stats.json"),
        {"_note": "TODO: real stats", "schema": "eda_stats/1.0"},
    )

if __name__ == "__main__":
    main()
