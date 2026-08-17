"""
Phase 1. Benchmark recall@k across the three chunking strategies before
picking a winner (or shipping all three as a configurable toggle).
Record results here — ARCHITECTURE.md and STATE.md both expect these numbers.
"""


def evaluate_recall_at_k(queries, ground_truth, retriever_fn, k=5):
    """TODO(phase-1): standard recall@k over a held-out query sample."""
    raise NotImplementedError
