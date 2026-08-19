"""
Phase 1 — corpus prep (run before chunking benchmarks).

Step 1: sample N=8000 queries from the full train set (seed=42), keep only
        their passages → data/corpus_sample.jsonl

Step 2: per sampled query_id, concatenate its passages (passage-index order,
        clear separator) into a single grouped document →
        data/grouped_docs.jsonl

Usage:
    python data/build_corpus.py
"""

import json
import random
import sys
import os
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.ingest import load_hindi_passages

SAMPLE_QUERIES  = 8_000
SEED            = 42
CORPUS_OUT      = os.path.join(os.path.dirname(__file__), "corpus_sample.jsonl")
GROUPED_OUT     = os.path.join(os.path.dirname(__file__), "grouped_docs.jsonl")
# Pure whitespace: invisible after str.split(), so it never surfaces as a
# word token inside a chunk. Passage boundaries are preserved structurally
# (the gap) without polluting the text with "---" noise.
PASSAGE_SEP     = "\n\n"


# ── Step 0: ingest ──────────────────────────────────────────────────────────
print("[build_corpus] Loading train passages …", flush=True)
passages = load_hindi_passages(splits=["train"])
print(f"[build_corpus] Raw passage count: {len(passages):,}", flush=True)


# ── Step 1: sample 8 000 distinct query_ids ──────────────────────────────────
all_qids = list({p["meta"]["query_id"] for p in passages})
print(f"[build_corpus] Distinct query_ids in train: {len(all_qids):,}", flush=True)

rng = random.Random(SEED)
sampled_qids = set(rng.sample(all_qids, min(SAMPLE_QUERIES, len(all_qids))))
print(f"[build_corpus] Sampled query_ids: {len(sampled_qids):,}", flush=True)

sampled_passages = [p for p in passages if p["meta"]["query_id"] in sampled_qids]
print(f"[build_corpus] Passages kept: {len(sampled_passages):,}", flush=True)

print(f"[build_corpus] Writing {CORPUS_OUT} …", flush=True)
with open(CORPUS_OUT, "w", encoding="utf-8") as fh:
    for p in sampled_passages:
        fh.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"[build_corpus] corpus_sample.jsonl written ({len(sampled_passages):,} lines).", flush=True)


# ── Step 2: group passages per query_id into one document ───────────────────
# Bucket by (split, query_id); sort each bucket by passage index derived from id.
buckets: dict[tuple, list] = defaultdict(list)
for p in sampled_passages:
    qid  = p["meta"]["query_id"]
    splt = p["meta"]["split"]
    buckets[(splt, qid)].append(p)

# Sort each bucket by the passage index in the passage id string (e.g. p3 → 3)
def passage_index(p):
    pid = p["id"]  # e.g. "train_q123_p4"
    try:
        return int(pid.rsplit("_p", 1)[-1])
    except ValueError:
        return 0

grouped_docs = []
for (splt, qid), bucket in buckets.items():
    bucket.sort(key=passage_index)
    
    combined_parts = []
    passage_offsets = []
    current_word = 0
    
    for p in bucket:
        p_text = p["text"]
        p_len_words = len(p_text.split())
        passage_offsets.append({
            "passage_id": p["id"],
            "word_start": current_word,
            "word_end": current_word + p_len_words
        })
        combined_parts.append(p_text)
        current_word += p_len_words

    combined_text = PASSAGE_SEP.join(combined_parts)
    first_meta = bucket[0]["meta"]
    
    grouped_docs.append({
        "doc_id":          f"{splt}_q{qid}",
        "text":            combined_text,
        "passage_ids":     [p["id"] for p in bucket],
        "passage_offsets": passage_offsets,
        "meta": {
            "query_id":    qid,
            "query_hi":    first_meta.get("query_hi", ""),
            "query_en":    first_meta.get("query_en", ""),
            "source_lang": first_meta.get("source_lang", ""),
            "target_lang": first_meta.get("target_lang", ""),
            "split":       splt,
        }
    })

print(f"[build_corpus] Writing {GROUPED_OUT} …", flush=True)
with open(GROUPED_OUT, "w", encoding="utf-8") as fh:
    for doc in grouped_docs:
        fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
print(f"[build_corpus] grouped_docs.jsonl written ({len(grouped_docs):,} docs).", flush=True)


# ── Report ───────────────────────────────────────────────────────────────────
word_counts = [len(d["text"].split()) for d in grouped_docs]
avg_words   = sum(word_counts) / len(word_counts) if word_counts else 0

print("\n" + "=" * 60, flush=True)
print("REPORT", flush=True)
print("=" * 60, flush=True)
print(f"  Queries sampled          : {len(sampled_qids):,}", flush=True)
print(f"  Passages kept            : {len(sampled_passages):,}", flush=True)
print(f"  Grouped docs             : {len(grouped_docs):,}", flush=True)
print(f"  Avg words per grouped doc: {avg_words:.1f}", flush=True)
print("=" * 60, flush=True)

print("\n--- Sample grouped doc 1 ---", flush=True)
d = grouped_docs[0]
print(f"  doc_id      : {d['doc_id']}", flush=True)
print(f"  passage_ids : {d['passage_ids']}", flush=True)
print(f"  word count  : {len(d['text'].split())}", flush=True)
print(f"  text preview:\n{d['text'][:600]}", flush=True)

print("\n--- Sample grouped doc 2 ---", flush=True)
d = grouped_docs[1]
print(f"  doc_id      : {d['doc_id']}", flush=True)
print(f"  passage_ids : {d['passage_ids']}", flush=True)
print(f"  word count  : {len(d['text'].split())}", flush=True)
print(f"  text preview:\n{d['text'][:600]}", flush=True)
