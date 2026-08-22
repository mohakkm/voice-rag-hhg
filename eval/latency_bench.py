"""
Phase 5. Run harness.orchestrator.run_pipeline over 30-50 real test queries,
compute P50/P70/P100 PER STAGE and end-to-end. Report honestly — see
ARCHITECTURE.md "Latency — how to actually report it" before you touch this.
"""
import time
import numpy as np
from retrieval.embeddings import get_model
from retrieval.qdrant_store import get_client

_WARMED_UP = False


def warmup_runtime() -> None:
    """Load embedding model + Qdrant client exactly once per process startup."""
    global _WARMED_UP
    if _WARMED_UP:
        return

    t0 = time.perf_counter()
    get_model()
    get_client()
    warmup_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[warmup] Loaded embedding model + Qdrant client in {warmup_ms:.1f} ms")
    _WARMED_UP = True


def run_benchmark(test_queries: list[str], n_runs: int = 30):
    """
    TODO(phase-5): call run_pipeline for each query, collect latency_ms dicts,
    compute percentiles per stage. Save results to eval/results.json — this
    feeds directly into the README latency table.
    """
    warmup_runtime()
    raise NotImplementedError


def percentiles(latencies_ms: list[float]) -> dict:
    """TODO(phase-5): return {"p50":..., "p70":..., "p100":...} via np.percentile."""
    raise NotImplementedError
