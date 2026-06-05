"""Entrypoint: config -> train (profiled) -> emit run.json + compute.json.
    python src/train.py --config configs/beardown.yaml
"""
import argparse, time, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from _shared.profiler import profile_run
from _shared.schema import RunRecord

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    # TODO: load yaml, resolve model from src.models.REGISTRY, build features, train loop.
    with profile_run(device="cpu") as prof:
        pass  # TODO training loop; prof.tick(n) each step
    rec = RunRecord(project="genre", model="TODO")
    rec.compute = prof.record()
    ts = time.strftime("%Y%m%d-%H%M%S")
    rec.write(os.path.join("projects/genre/runs", rec.model, ts))

if __name__ == "__main__":
    main()
