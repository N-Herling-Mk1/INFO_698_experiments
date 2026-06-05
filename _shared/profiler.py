"""Compute logging -> ComputeRecord. Source of all cost estimates.

Usage:
    from _shared.profiler import profile_run
    with profile_run(device="cuda") as prof:
        ... train ...
        prof.tick(n_samples_processed)     # call each step/epoch
    rec = prof.record()                    # -> ComputeRecord
"""
from __future__ import annotations
import time, contextlib
from _shared.schema import ComputeRecord

# NOTE: heavy imports (torch, psutil) live inside methods so EDA can import _shared
# without a GPU stack present.


class _Profiler:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._t0 = None
        self._samples = 0
        self._peak_rss = 0.0
        self._peak_vram = None

    def tick(self, n_samples: int) -> None:
        self._samples += n_samples
        # TODO: sample psutil RSS and torch.cuda.max_memory_allocated() here

    def record(self) -> ComputeRecord:
        wall = (time.perf_counter() - self._t0) if self._t0 else 0.0
        tput = self._samples / wall if wall > 0 else 0.0
        return ComputeRecord(
            wall_seconds=wall,
            peak_rss_mb=self._peak_rss,
            peak_vram_mb=self._peak_vram,
            throughput_samples_per_s=tput,
            device=self.device,
        )


@contextlib.contextmanager
def profile_run(device: str = "cpu"):
    p = _Profiler(device)
    p._t0 = time.perf_counter()
    # TODO: print a TRON-styled live progress line / status here (Steve's standing pref)
    try:
        yield p
    finally:
        pass
