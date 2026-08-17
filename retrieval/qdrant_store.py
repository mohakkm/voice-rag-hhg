"""
Phase 1/3. Qdrant in embedded/local mode — no server, zero network hop.
This is the retrieval leg that has to hit the <200ms target (see ARCHITECTURE.md).
"""
from qdrant_client import QdrantClient
from config import QDRANT_PATH, TOP_K


def get_client():
    """TODO(phase-1): QdrantClient(path=QDRANT_PATH)."""
    raise NotImplementedError


def index_chunks(collection_name: str, chunks: list[dict]):
    """TODO(phase-1): create_collection + upsert points (id, vector, payload=meta+text)."""
    raise NotImplementedError


def search(collection_name: str, query_vector, top_k: int = TOP_K):
    """TODO(phase-3): return top_k hits with score + payload. This call is what you time for P50/P70/P100."""
    raise NotImplementedError
