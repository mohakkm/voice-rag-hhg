"""
Phase 1/3. Wraps the embedding model. Self-hosted (sentence-transformers) —
no API cost, no network hop. Keep model load lazy/singleton, don't reload per call.
"""
from config import EMBEDDING_MODEL

_model = None


def get_model():
    """TODO(phase-1): lazy-load SentenceTransformer(EMBEDDING_MODEL), cache in _model."""
    raise NotImplementedError


def embed(texts: list[str]):
    """TODO(phase-1): normalize + encode. Return numpy array, one row per text."""
    raise NotImplementedError
