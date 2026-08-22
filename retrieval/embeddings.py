"""
Phase 1/3. Wraps the embedding model. Self-hosted (sentence-transformers) —
no API cost, no network hop. Keep model load lazy/singleton, don't reload per call.

Notes on multilingual-e5-large prompt format:
  - Passage text must be prefixed with "passage: " at index time.
  - Query text must be prefixed with "query: " at search time.
  embed() handles the passage prefix internally. Callers embedding a *query*
  should pass ["query: <text>"] directly (or call embed_query()).
"""
import time
import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

_model: SentenceTransformer | None = None
_model_load_count = 0
_last_model_load_ms = 0.0
_last_embed_query_timing = {
    "get_model_ms": 0.0,
    "encode_ms": 0.0,
    "total_ms": 0.0,
}


def get_model() -> SentenceTransformer:
    """Lazy-load SentenceTransformer(EMBEDDING_MODEL) and cache in the module singleton."""
    global _model, _model_load_count, _last_model_load_ms
    if _model is None:
        t0 = time.perf_counter()
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {}
        if device == "cuda":
            model_kwargs["model_kwargs"] = {"dtype": torch.float16}
        _model = SentenceTransformer(EMBEDDING_MODEL, device=device, **model_kwargs)
        _last_model_load_ms = (time.perf_counter() - t0) * 1000.0
        _model_load_count += 1
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """
    Encode *passage* texts with the e5 prefix and L2-normalize the result.

    Args:
        texts: Raw passage strings (no prefix needed — added internally).

    Returns:
        float32 numpy array of shape (len(texts), embedding_dim), L2-normalized
        so that dot-product == cosine similarity.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    prefixed = [f"passage: {t}" for t in texts]
    model = get_model()
    # Batch size of 128 is highly optimized for FP16 on the RTX 4060 GPU
    batch_size = 128 if model.device.type == "cuda" else 32
    vectors = model.encode(
        prefixed,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
        batch_size=batch_size,
    )
    return vectors.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Encode a *single query* string with the e5 query prefix.

    Returns:
        float32 numpy array of shape (1, embedding_dim), L2-normalized.
    """
    t_total = time.perf_counter()
    t0 = time.perf_counter()
    model = get_model()
    get_model_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    vector = model.encode(
        [f"query: {query}"],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    encode_ms = (time.perf_counter() - t1) * 1000.0
    _last_embed_query_timing["get_model_ms"] = get_model_ms
    _last_embed_query_timing["encode_ms"] = encode_ms
    _last_embed_query_timing["total_ms"] = (time.perf_counter() - t_total) * 1000.0
    return vector.astype(np.float32)


def get_embed_debug_state() -> dict:
    """Return singleton/debug timing state for diagnostics."""
    model = _model
    return {
        "singleton_model_loaded": model is not None,
        "model_object_id": id(model) if model is not None else None,
        "model_load_count": _model_load_count,
        "last_model_load_ms": _last_model_load_ms,
        "model_device": str(model.device) if model is not None else None,
        "last_embed_query_timing_ms": dict(_last_embed_query_timing),
    }
