import json
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrieval.embeddings import get_model
from retrieval.qdrant_store import search, get_client

COLLECTIONS = {
    "fixed_overlap": "fixed_overlap_chunks",
    "semantic": "semantic_chunks",
    "metadata_aware": "metadata_aware_chunks",
}


def load_sample_queries():
    docs_path = ROOT / "data" / "grouped_docs.jsonl"
    queries = []
    with docs_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            d = json.loads(line)
            queries.append(d["meta"].get("query_hi", ""))
    return queries


def embed_queries(queries):
    model = get_model()
    prefixed = [f"query: {q}" for q in queries]
    batch_size = 128 if model.device.type == "cuda" else 32
    vectors = model.encode(
        prefixed,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
        batch_size=batch_size,
    )
    return vectors.astype(np.float32)


def main():
    client = get_client()
    queries = load_sample_queries()
    query_vectors = embed_queries(queries)

    print("=== COLLECTION COUNTS ===", flush=True)
    for strategy, collection_name in COLLECTIONS.items():
        count = int(client.count(collection_name=collection_name, exact=True).count)
        print(f"{strategy:<15} -> {collection_name:<24} points={count:,}", flush=True)

    print("\n=== SEARCH BENCH (TOP-5) ===", flush=True)

    for strategy, collection_name in COLLECTIONS.items():
        rows = []
        for i, (q, qv) in enumerate(zip(queries, query_vectors), start=1):
            t0 = time.perf_counter()
            hits = search(collection_name, qv, top_k=5)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            rows.append((i, q, latency_ms, hits))

        lats = [r[2] for r in rows]
        print(f"\n[{strategy}] collection={collection_name}", flush=True)
        print(
            "search latency ms -> "
            f"avg={np.mean(lats):.2f}, p50={np.percentile(lats, 50):.2f}, "
            f"p70={np.percentile(lats, 70):.2f}, p100={np.max(lats):.2f}",
            flush=True,
        )

        for idx, q, lat_ms, hits in rows:
            print(f"\nQ{idx}: {q}", flush=True)
            print(f"latency_ms={lat_ms:.2f}", flush=True)
            for rank, hit in enumerate(hits, start=1):
                payload = hit.get("payload", {})
                chunk_id = payload.get("chunk_id", "")
                text_preview = payload.get("text", "").replace("\n", " ")[:140]
                print(
                    f"  {rank}. score={hit['score']:.4f} chunk_id={chunk_id} text='{text_preview}'",
                    flush=True,
                )


if __name__ == "__main__":
    main()
