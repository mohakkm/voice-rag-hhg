"""
Phase 5. Run harness.orchestrator.run_pipeline over 30-50 real test queries,
compute P50/P70/P100 PER STAGE and end-to-end. Report honestly — see
ARCHITECTURE.md "Latency — how to actually report it" before you touch this.
"""
import numpy as np


def run_benchmark(test_queries: list[str], n_runs: int = 30):
    """
    TODO(phase-5): call run_pipeline for each query, collect latency_ms dicts,
    compute percentiles per stage. Save results to eval/results.json — this
    feeds directly into the README latency table.
    """
    raise NotImplementedError


def percentiles(latencies_ms: list[float]) -> dict:
    """TODO(phase-5): return {"p50":..., "p70":..., "p100":...} via np.percentile."""
    raise NotImplementedError
