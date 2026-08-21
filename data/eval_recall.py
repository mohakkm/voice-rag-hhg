import os
import sys
import json
import time
from collections import defaultdict
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# Set working directory to project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.chunking import chunk_fixed_overlap, chunk_semantic, chunk_metadata_aware
from retrieval.embeddings import get_model

# Batch query embedding helper
def embed_queries(queries: list[str]) -> np.ndarray:
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

# Batch passage embedding helper
def embed_passages(passages: list[str]) -> np.ndarray:
    from retrieval.embeddings import embed
    return embed(passages)

def get_or_create_embeddings(texts, name, embed_fn):
    cache_file = os.path.join(ROOT, "data", f"cache_embeddings_{name}.npy")
    if os.path.exists(cache_file):
        try:
            arr = np.load(cache_file)
            if arr.shape[0] == len(texts):
                print(f"Loaded cached embeddings for {name} ({arr.shape})", flush=True)
                return arr
        except Exception as e:
            print(f"Error loading cache for {name}: {e}, recomputing...", flush=True)
            
    print(f"Computing embeddings for {name} ({len(texts):,} items)...", flush=True)
    t0 = time.perf_counter()
    arr = embed_fn(texts)
    print(f"Computed in {time.perf_counter() - t0:.2f}s", flush=True)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    np.save(cache_file, arr)
    print(f"Saved embeddings for {name} to {cache_file}", flush=True)
    return arr

def precompute_covered_passages(chunks, raw_docs, min_overlap_pct=50.0):
    doc_words_cache = {}
    for doc_id, doc in raw_docs.items():
        doc_words_cache[doc_id] = doc["text"].split()

    def find_sublist(sublist, parent_list):
        n = len(parent_list)
        m = len(sublist)
        for i in range(n - m + 1):
            if parent_list[i : i + m] == sublist:
                return i
        return -1

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        doc_id = chunk_id.rsplit("_c", 1)[0]
        
        doc = raw_docs[doc_id]
        doc_words = doc_words_cache[doc_id]
        chunk_words = chunk["text"].split()
        
        start_word_idx = find_sublist(chunk_words, doc_words)
        if start_word_idx == -1:
            chunk["covered_passages"] = []
            continue
            
        end_word_idx = start_word_idx + len(chunk_words)
        
        covered = []
        for p in doc["passage_offsets"]:
            p_start = p["word_start"]
            p_end = p["word_end"]
            p_len = p_end - p_start
            
            overlap_start = max(p_start, start_word_idx)
            overlap_end = min(p_end, end_word_idx)
            overlap_len = max(0, overlap_end - overlap_start)
            
            if p_len > 0:
                pct = (overlap_len / p_len) * 100
                if pct >= min_overlap_pct:
                    covered.append(p["passage_id"])
                    
        chunk["covered_passages"] = covered

def main():
    t_start = time.perf_counter()
    
    # 1. Load data
    print("Loading grouped_docs.jsonl...", flush=True)
    docs = []
    raw_docs = {}
    docs_path = os.path.join(ROOT, "data", "grouped_docs.jsonl")
    with open(docs_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs.append({"id": d["doc_id"], "text": d["text"], "meta": d["meta"]})
            raw_docs[d["doc_id"]] = d
            
    print("Loading corpus_sample.jsonl for ground truth...", flush=True)
    ground_truth = defaultdict(list)
    sample_path = os.path.join(ROOT, "data", "corpus_sample.jsonl")
    with open(sample_path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p["meta"].get("is_selected") == 1:
                qid = p["meta"]["query_id"]
                ground_truth[qid].append(p["id"])
                
    # Filter queries that have at least one selected passage
    eval_docs = [d for d in docs if d["meta"]["query_id"] in ground_truth]
    print(f"Queries to evaluate: {len(eval_docs):,}", flush=True)
    
    # Pre-embed queries
    queries_hi = [d["meta"]["query_hi"] for d in eval_docs]
    query_embeddings = get_or_create_embeddings(queries_hi, "queries", embed_queries)
    
    strategies = {
        "fixed_overlap": lambda: chunk_fixed_overlap(docs, chunk_size=256, overlap=32),
        "semantic": lambda: chunk_semantic(docs, lambda t: embed_passages(t), threshold=0.75, chunk_size=256, min_chunk_words=20),
        "metadata_aware": lambda: chunk_metadata_aware(docs, min_chunk_words=20, chunk_size=256, overlap=32, max_passage_words=400),
    }

    # Optional: restrict to a subset of strategies via CLI args, e.g. `python eval_recall.py semantic`
    if len(sys.argv) > 1:
        requested = sys.argv[1:]
        unknown = [name for name in requested if name not in strategies]
        if unknown:
            raise ValueError(f"Unknown strategy name(s): {unknown}. Valid: {list(strategies)}")
        strategies = {name: strategies[name] for name in requested}
    
    results = {}
    
    for name, chunk_fn in strategies.items():
        print(f"\n--- Strategy: {name} ---", flush=True)
        t0 = time.perf_counter()
        
        # Generate chunks
        print(f"Generating chunks for {name}...", flush=True)
        chunks = chunk_fn()
        print(f"Total chunks generated: {len(chunks):,}", flush=True)
        
        # Precompute covered passages
        print(f"Precomputing chunk-to-passage relevance mappings...", flush=True)
        precompute_covered_passages(chunks, raw_docs)
        
        # Get chunk embeddings
        chunk_texts = [c["text"] for c in chunks]
        chunk_embeddings = get_or_create_embeddings(chunk_texts, name, embed_passages)
        
        # Brute-force numpy similarity search
        print("Running similarity search and recall evaluation...", flush=True)
        # Compute dot product similarities: shape (num_queries, num_chunks)
        sims = np.dot(query_embeddings, chunk_embeddings.T)
        
        # Find top 5 indices for all queries
        top_indices = np.argpartition(-sims, 5, axis=1)[:, :5]
        
        recalls = []
        for idx, doc in enumerate(eval_docs):
            qid = doc["meta"]["query_id"]
            gt_passages = set(ground_truth[qid])
            
            # Sort top 5 chunks by actual score
            indices = top_indices[idx]
            sorted_idx = indices[np.argsort(-sims[idx, indices])]
            
            retrieved_passages = set()
            for chunk_idx in sorted_idx:
                chunk = chunks[chunk_idx]
                retrieved_passages.update(chunk["covered_passages"])
                
            retrieved_relevant = retrieved_passages.intersection(gt_passages)
            recall = len(retrieved_relevant) / len(gt_passages) if gt_passages else 0
            recalls.append(recall)
            
        avg_recall = sum(recalls) / len(recalls) if recalls else 0
        results[name] = avg_recall
        print(f"Strategy {name} Recall@5: {avg_recall:.4f}", flush=True)
        print(f"Strategy {name} Eval finished in {time.perf_counter() - t0:.2f}s", flush=True)
        
    print("\n" + "=" * 60, flush=True)
    print("FINAL RECALL@5 COMPARISON", flush=True)
    print("=" * 60, flush=True)
    for name, recall in results.items():
        print(f"  {name:<18} : {recall:.4f}", flush=True)
    total_runtime = time.perf_counter() - t_start
    print(f"Total Eval Runtime   : {total_runtime:.2f}s ({total_runtime/60:.1f} min)", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    main()
