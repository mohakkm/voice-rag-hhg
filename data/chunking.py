"""
Phase 1. Three chunking strategies per ARCHITECTURE.md — implement all
three, don't collapse to one early. Each function takes list[dict] from
ingest.py and returns list[dict]: {"chunk_id": str, "text": str, "meta": {...}}
"""


def chunk_fixed_overlap(passages, chunk_size=256, overlap=32):
    """TODO(phase-1): baseline — fixed token/word window with overlap."""
    raise NotImplementedError


def chunk_semantic(passages, embedding_fn):
    """TODO(phase-1): split on embedding-similarity breakpoints between sentences."""
    raise NotImplementedError


def chunk_metadata_aware(passages):
    """TODO(phase-1): use MSMARCO's native passage boundaries + language tag as-is."""
    raise NotImplementedError
