# HHG Task 2 — Live State

Update this every session. This is the file you paste into Cursor/any IDE chat to restore context fast — keep it current, not aspirational.

**Last updated:** Aug 19, 2026
**Current phase:** Phase 1 — Data & Chunking / Embeddings

## Decisions locked (don't relitigate these mid-build)
- Language: Hindi only, `ai4bharat/MSMARCO-XI` config `"hi"`
- STT: Sarvam Saaras v3
- Embeddings: `intfloat/multilingual-e5-large` (explicit GPU/CUDA support verified)
- Vector DB: Qdrant, embedded/local mode
- Generation + guardrail checks: Claude API
- Harness: custom, no LangChain
- Frontend: Gradio → deploy on Hugging Face Spaces
- 200ms target = retrieval leg only; full pipeline latency reported honestly, broken down by stage

## Completed
- Pull the `hi` subset of MSMARCO-XI, confirm row count and structure locally (resolved PyArrow string offset overflow crash)
- Sample N=8000 queries (seed=42) and create `data/corpus_sample.jsonl` + `data/grouped_docs.jsonl` (concatenated per-query passages) to resolve corpus scale & passage granularity issues
- Implement and verify `chunk_fixed_overlap` in `data/chunking.py` (23,989 chunks, 217.6 avg words, verified no "---" separator tokens)
- Implement `retrieval/embeddings.py` (lazy-loaded `intfloat/multilingual-e5-large`, normalized float32 vectors, e5 query/passage prefix convention applied, GPU device explicitly targeted)

## In progress
- Phase 1 chunking strategies evaluation (semantic & metadata-aware)

## Blockers / resolved
- **Resolved**: Separator Bug. The initial `grouped_docs.jsonl` concatenation used `\n\n---\n\n` as a separator, causing `---` text tokens to leak into final chunks. Fixed by switching `PASSAGE_SEP` to pure whitespace `\n\n` and regenerating the grouped docs.
- **Resolved**: PyArrow offset overflow on the HF Dataset wrapping. Fixed by returning the deduplicated passages list directly.
- Sarvam free tier rate limits not yet checked against planned 30–50 query benchmark run

## Next 3 actions
1. Implement chunk_semantic in `data/chunking.py`
2. Implement chunk_metadata_aware in `data/chunking.py`
3. Qdrant indexing and retrieval setup

## Team split
- ML/retrieval/harness/guardrails: [you]
- STT wiring, Gradio UI, deploy: [teammate]
(Adjust if actual split differs — keep this accurate, it's what any IDE session should assume.)
