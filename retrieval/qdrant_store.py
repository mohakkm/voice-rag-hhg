"""
Phase 1/3. Qdrant in embedded/local mode — no server, zero network hop.
This is the retrieval leg that has to hit the <200ms target (see ARCHITECTURE.md).
"""
import time
import numpy as np
from qdrant_client import models
from qdrant_client import QdrantClient
from config import QDRANT_PATH, TOP_K

_client: QdrantClient | None = None
_client_open_count = 0
_last_client_open_ms = 0.0
_last_search_timing = {
    "get_client_ms": 0.0,
    "query_ms": 0.0,
    "total_ms": 0.0,
}


def get_client():
    """Create embedded/local Qdrant client at QDRANT_PATH."""
    global _client, _client_open_count, _last_client_open_ms
    if _client is None:
        t0 = time.perf_counter()
        _client = QdrantClient(path=QDRANT_PATH)
        _last_client_open_ms = (time.perf_counter() - t0) * 1000.0
        _client_open_count += 1
    return _client


def _collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """Compatibility helper across qdrant-client versions."""
    try:
        return bool(client.collection_exists(collection_name=collection_name))
    except AttributeError:
        try:
            client.get_collection(collection_name=collection_name)
            return True
        except Exception:
            return False


def index_chunks(
    collection_name: str,
    chunks: list[dict],
    vectors: np.ndarray | None = None,
    batch_size: int = 256,
):
    """
    Recreate a collection and index chunk points.

    Each point payload includes chunk text + chunk metadata. Vectors can be
    provided precomputed through `vectors`; otherwise chunk texts are embedded.
    """
    if not chunks:
        raise ValueError("chunks must be non-empty")

    if vectors is None:
        from retrieval.embeddings import embed

        chunk_texts = [c["text"] for c in chunks]
        vectors = embed(chunk_texts)

    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D array, got shape={vectors.shape}")
    if vectors.shape[0] != len(chunks):
        raise ValueError(
            f"vectors row count ({vectors.shape[0]}) must match chunks ({len(chunks)})"
        )

    vector_size = int(vectors.shape[1])
    client = get_client()

    if _collection_exists(client, collection_name):
        client.delete_collection(collection_name=collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )

    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        batch_points = []

        for idx in range(start, end):
            chunk = chunks[idx]
            payload = {
                "chunk_id": chunk.get("chunk_id", str(idx)),
                "text": chunk.get("text", ""),
                **chunk.get("meta", {}),
            }
            batch_points.append(
                models.PointStruct(
                    id=idx,
                    vector=vectors[idx].tolist(),
                    payload=payload,
                )
            )

        client.upsert(collection_name=collection_name, points=batch_points, wait=True)

    count_result = client.count(collection_name=collection_name, exact=True)
    return int(count_result.count)


def search(collection_name: str, query_vector, top_k: int = TOP_K):
    """Return top_k hits with score and payload for a query vector."""
    t_total = time.perf_counter()
    t0 = time.perf_counter()
    client = get_client()
    get_client_ms = (time.perf_counter() - t0) * 1000.0
    query_vector = np.asarray(query_vector, dtype=np.float32)
    if query_vector.ndim > 1:
        if query_vector.shape[0] != 1:
            raise ValueError(f"query_vector must be 1D or (1, D), got {query_vector.shape}")
        query_vector = query_vector[0]

    t1 = time.perf_counter()
    if hasattr(client, "search"):
        hits = client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        )
    else:
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        )
        hits = response.points
    query_ms = (time.perf_counter() - t1) * 1000.0
    _last_search_timing["get_client_ms"] = get_client_ms
    _last_search_timing["query_ms"] = query_ms
    _last_search_timing["total_ms"] = (time.perf_counter() - t_total) * 1000.0

    return [
        {
            "id": hit.id,
            "score": float(hit.score),
            "payload": hit.payload or {},
        }
        for hit in hits
    ]


def get_qdrant_debug_state() -> dict:
    """Return singleton/debug timing state for diagnostics."""
    client = _client
    return {
        "singleton_client_loaded": client is not None,
        "client_object_id": id(client) if client is not None else None,
        "client_open_count": _client_open_count,
        "last_client_open_ms": _last_client_open_ms,
        "last_search_timing_ms": dict(_last_search_timing),
    }
