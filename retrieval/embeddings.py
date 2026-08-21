"""
Phase 1/3. Wraps the embedding model. Self-hosted (sentence-transformers) —
no API cost, no network hop. Keep model load lazy/singleton, don't reload per call.

Notes on multilingual-e5-large prompt format:
  - Passage text must be prefixed with "passage: " at index time.
  - Query text must be prefixed with "query: " at search time.
  embed() handles the passage prefix internally. Callers embedding a *query*
  should pass ["query: <text>"] directly (or call embed_query()).
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazy-load SentenceTransformer(EMBEDDING_MODEL) and cache in the module singleton."""
    global _model
    if _model is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {}
        if device == "cuda":
            model_kwargs["model_kwargs"] = {"dtype": torch.float16}
        _model = SentenceTransformer(EMBEDDING_MODEL, device=device, **model_kwargs)
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
    model = get_model()
    vector = model.encode(
        [f"query: {query}"],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vector.astype(np.float32)
