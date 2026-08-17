# HHG Task 2 — Live State

Update this every session. This is the file you paste into Cursor/any IDE chat to restore context fast — keep it current, not aspirational.

**Last updated:** Aug 17, 2026
**Current phase:** Phase 0 — planning locked, build not started

## Decisions locked (don't relitigate these mid-build)
- Language: Hindi only, `ai4bharat/MSMARCO-XI` config `"hi"`
- STT: Sarvam Saaras v3
- Embeddings: multilingual-e5-large or bge-m3 (pick one on first embedding test, log which)
- Vector DB: Qdrant, embedded/local mode
- Generation + guardrail checks: Claude API
- Harness: custom, no LangChain
- Frontend: Gradio → deploy on Hugging Face Spaces
- 200ms target = retrieval leg only; full pipeline latency reported honestly, broken down by stage

## Completed
- (nothing yet — architecture + checklist finalized)

## In progress
- (update as you start Phase 1)

## Blockers / open questions
- Which embedding model wins on actual Hindi recall — not decided yet, test in Phase 1
- Sarvam free tier rate limits not yet checked against planned 30–50 query benchmark run

## Next 3 actions
1. Pull the `hi` subset of MSMARCO-XI, confirm row count and structure locally
2. Stand up Qdrant in embedded mode, get one chunking strategy indexed end to end
3. Get Sarvam STT returning a transcript from a real mic input, before building anything else on top

## Team split
- ML/retrieval/harness/guardrails: [you]
- STT wiring, Gradio UI, deploy: [teammate]
(Adjust if actual split differs — keep this accurate, it's what any IDE session should assume.)
