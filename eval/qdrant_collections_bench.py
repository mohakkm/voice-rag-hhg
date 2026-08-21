import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.chunking import chunk_fixed_overlap, chunk_semantic, chunk_metadata_aware
from retrieval.embeddings import embed, get_model
from retrieval.qdrant_store import index_chunks, search, get_client


COLLECTIONS = {
    "fixed_overlap": "fixed_overlap_chunks",
    "semantic": "semantic_chunks",
    "metadata_aware": "metadata_aware_chunks",
}


def load_grouped_docs():
    docs = []
    path = ROOT / "data" / "grouped_docs.jsonl"
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs.append({"id": d["doc_id"], "text": d["text"], "meta": d["meta"]})
    return docs


def get_or_create_embeddings(texts, cache_name):
    cache_file = ROOT / "data" / f"cache_embeddings_{cache_name}.npy"
    if cache_file.exists():
        try:
            arr = np.load(cache_file)
            if arr.shape[0] == len(texts):
                print(f"Loaded cached embeddings for {cache_name}: {arr.shape}", flush=True)
                return arr.astype(np.float32)
        except Exception as e:
            print(f"Cache load failed for {cache_name}: {e}; recomputing", flush=True)

    print(f"Computing embeddings for {cache_name} ({len(texts):,} items)", flush=True)
    t0 = time.perf_counter()
    arr = embed(texts)
    np.save(cache_file, arr)
    print(f"Computed+saved {cache_name} in {time.perf_counter() - t0:.2f}s", flush=True)
    return arr.astype(np.float32)


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


def build_chunks(docs):
    return {
        "fixed_overlap": chunk_fixed_overlap(docs, chunk_size=256, overlap=32),
        "semantic": chunk_semantic(
            docs,
            lambda t: embed(t),
            threshold=0.75,
            chunk_size=256,
            min_chunk_words=20,
        ),
        "metadata_aware": chunk_metadata_aware(
            docs,
            min_chunk_words=20,
            chunk_size=256,
            overlap=32,
            max_passage_words=400,
        ),
    }


def benchmark_collection(collection_name, query_texts, query_vectors):
    rows = []
    for i, (q, qv) in enumerate(zip(query_texts, query_vectors), start=1):
        t0 = time.perf_counter()
        hits = search(collection_name, qv, top_k=5)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        rows.append({"query_idx": i, "query": q, "latency_ms": latency_ms, "hits": hits})
    return rows


def main():
    docs = load_grouped_docs()
    print(f"Loaded grouped docs: {len(docs):,}", flush=True)

    chunks_by_strategy = build_chunks(docs)

    print("\n=== INDEXING ===", flush=True)
    client = get_client()
    count_summary = {}

    for strategy, chunks in chunks_by_strategy.items():
        collection_name = COLLECTIONS[strategy]
        texts = [c["text"] for c in chunks]
        vectors = get_or_create_embeddings(texts, strategy)

        indexed_count = index_chunks(collection_name, chunks, vectors=vectors, batch_size=256)
        qdrant_count = int(client.count(collection_name=collection_name, exact=True).count)
        expected = len(chunks)
        status = "OK" if expected == indexed_count == qdrant_count else "MISMATCH"

        count_summary[strategy] = {
            "collection": collection_name,
            "expected_chunks": expected,
            "indexed_count": indexed_count,
            "qdrant_count": qdrant_count,
            "status": status,
        }

        print(
            f"{strategy:<15} -> {collection_name:<24} expected={expected:,} indexed={indexed_count:,} qdrant={qdrant_count:,} [{status}]",
            flush=True,
        )

    sample_queries = [
        docs[0]["meta"].get("query_hi", ""),
        docs[1]["meta"].get("query_hi", ""),
        docs[2]["meta"].get("query_hi", ""),
    ]
    query_vectors = embed_queries(sample_queries)

    print("\n=== SEARCH BENCH (TOP-5) ===", flush=True)
    report = {"counts": count_summary, "queries": sample_queries, "results": {}}

    for strategy, meta in count_summary.items():
        collection_name = meta["collection"]
        rows = benchmark_collection(collection_name, sample_queries, query_vectors)
        latencies = [r["latency_ms"] for r in rows]

        print(f"\n[{strategy}] collection={collection_name}", flush=True)
        print(
            "search latency ms -> "
            f"avg={np.mean(latencies):.2f}, p50={np.percentile(latencies, 50):.2f}, "
            f"p70={np.percentile(latencies, 70):.2f}, p100={np.max(latencies):.2f}",
            flush=True,
        )

        for row in rows:
            print(f"\nQ{row['query_idx']}: {row['query']}", flush=True)
            print(f"latency_ms={row['latency_ms']:.2f}", flush=True)
            for rank, hit in enumerate(row["hits"], start=1):
                payload = hit.get("payload", {})
                chunk_id = payload.get("chunk_id", "")
                text_preview = payload.get("text", "").replace("\n", " ")[:140]
                print(
                    f"  {rank}. score={hit['score']:.4f} chunk_id={chunk_id} text='{text_preview}'",
                    flush=True,
                )

        report["results"][strategy] = {
            "collection": collection_name,
            "latency_ms": {
                "avg": float(np.mean(latencies)),
                "p50": float(np.percentile(latencies, 50)),
                "p70": float(np.percentile(latencies, 70)),
                "p100": float(np.max(latencies)),
            },
            "queries": rows,
        }

    out_path = ROOT / "eval" / "qdrant_bench_report.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSaved machine-readable report to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
