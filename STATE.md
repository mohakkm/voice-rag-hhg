# HHG Task 2 — Live State

Update this every session. This is the file you paste into Cursor/any IDE chat to restore context fast — keep it current, not aspirational.

**Last updated:** Aug 22, 2026
**Current phase:** Phase 3 — Harness & Retrieval (Indexing & Qdrant Setup)

## Decisions locked (don't relitigate these mid-build)
- Language: Hindi only, `ai4bharat/MSMARCO-XI` config `"hi"`
- STT: Sarvam Saaras v3
- Embeddings: `intfloat/multilingual-e5-large` (explicit GPU/CUDA support verified)
- Vector DB: Qdrant, embedded/local mode
- Generation + guardrail checks: Claude API
- Harness: custom, no LangChain
- Frontend: Gradio → deploy on Hugging Face Spaces
- 200ms target = retrieval leg only; full pipeline latency reported honestly, broken down by stage
- Granularity / Strategies: ship all 3 strategies as a configurable toggle (fixed_overlap and semantic are statistically tied with 0.870 and 0.857 recall respectively; metadata_aware is 0.723 but included for completeness and its lowest latency profile)

## Completed
- Pull the `hi` subset of MSMARCO-XI, confirm row count and structure locally (resolved PyArrow string offset overflow crash)
- Sample N=8000 queries (seed=42) and create `data/corpus_sample.jsonl` + `data/grouped_docs.jsonl` (concatenated per-query passages) to resolve corpus scale & passage granularity issues
- Implement and verify `chunk_fixed_overlap` in `data/chunking.py` (23,989 chunks, 217.6 avg words, verified no "---" separator tokens)
- Implement `retrieval/embeddings.py` (lazy-loaded `intfloat/multilingual-e5-large`, normalized float32 vectors, e5 query/passage prefix convention applied, GPU device explicitly targeted)
- Phase 1 chunking strategies evaluation complete and validated. Final metrics on N=8000 (4,491 evaluated) queries:
  - **fixed_overlap** (chunk_size=256, overlap=32): **0.870 Recall@5**
  - **semantic** (threshold=0.75, chunk_size=256, merge_limit=20): **0.857 Recall@5** (post-regex-split fix)
  - **metadata_aware** (min_words=20, max_words=400): **0.723 Recall@5**

## In progress
- Setting up Qdrant embedded collections for indexing the 3 sets of chunks

## Blockers / resolved
- **Resolved**: Cold-start latency on first process launch was much larger than expected: embedding model load took ~6.3s and the embedded Qdrant client open took ~39.6s. Both were confirmed to be true process singletons via object-id checks, so the hit is one-time per process, not per request. `app/gradio_app.py` and `eval/latency_bench.py` now warm up once at startup (calling `get_model()` and `get_client()` before serving/benchmarking) so every fresh process pays that cost exactly once instead of on the first user query.
- **Resolved**: Chunk-splitting regex bug in semantic chunking. The original pattern `(?<=[।\.!\?])\s*` performed zero-width splits when no space followed punctuation. This cut through internal characters of abbreviation tokens (e.g. `एम.एस.एफ.सी.ए.`), inserting incorrect space delimiters on reassembly and corrupting downstream word match indices in evaluation (falsely dropping semantic recall@5 to 0.357, where 53% of chunks had mismatched indices). Corrected by using `\s+` to enforce splitting only on actual whitespace, returning semantic recall to its true value of 0.857.
- **Resolved**: Separator Bug. The initial `grouped_docs.jsonl` concatenation used `\n\n---\n\n` as a separator, causing `---` text tokens to leak into final chunks. Fixed by switching `PASSAGE_SEP` to pure whitespace `\n\n` and regenerating the grouped docs.
- **Resolved**: PyArrow offset overflow on the HF Dataset wrapping. Fixed by returning the deduplicated passages list directly.
- Sarvam free tier rate limits not yet checked against planned 30–50 query benchmark run

## Next 3 actions
1. Set up Qdrant collections layout for all 3 chunking strategies (using local/embedded Qdrant)
2. Index precomputed embeddings for all 3 chunking subsets into Qdrant collections
3. Wire basic retrieval capability in orchestrator to support configurable strategy toggling

## Team split
- ML/retrieval/harness/guardrails: [you]
- STT wiring, Gradio UI, deploy: [teammate]
(Adjust if actual split differs — keep this accurate, it's what any IDE session should assume.)
