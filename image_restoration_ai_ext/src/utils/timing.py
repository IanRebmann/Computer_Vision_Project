from __future__ import annotations
import time
from contextlib import contextmanager

@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    print(f"[TIMER] {label} ...")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[TIMER] {label} done in {dt:.2f}s")
