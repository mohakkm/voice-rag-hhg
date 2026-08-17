# HHG Task 2 — Voice RAG — Architecture

Last locked: Aug 17, 2026. Change this file when a decision changes — don't let it drift from what's actually built.

## Scope
- Language: **Hindi only** (`ai4bharat/MSMARCO-XI`, config `"hi"`). Do not expand to other languages — that's scope creep, not thoroughness.
- Dataset fields per example: `query`, `answers`, `passages`, `source_lang`, `target_lang`, `meta`. Passages come pre-segmented — this matters for chunking strategy #3 below.

## Pipeline (end to end)
```
Mic audio
  → [1] Sarvam STT (Saaras v3, language_code="hi-IN")
  → [2] Guardrail: input classifier (unsafe / gibberish check)
  → [3] Embed query (multilingual-e5-large or bge-m3)
  → [4] Qdrant retrieval (top-k, embedded/local mode)
  → [5] Guardrail: grounding check (top score < threshold → refuse)
  → [6] Claude API generation (answer from retrieved context only)
  → [7] Guardrail: post-gen groundedness check (claims trace to context?)
  → [8] Structured JSON response: {answer, sources, confidence, latency_breakdown}
```
Every stage timestamped. This is what feeds the P50/P70/P100 numbers.

## Component decisions

| Component | Choice | Why |
|---|---|---|
| STT | Sarvam Saaras v3 | Purpose-built for Indian languages, sub-150ms TTFT, outperforms Scribe v2 on Hindi accuracy, handles Hinglish code-mixing. ElevenLabs is English-first — wrong tool for this dataset. |
| Embeddings | `multilingual-e5-large` or `bge-m3` | Generic English embedding models underperform badly on Devanagari script. Non-negotiable pick. |
| Vector DB | Qdrant, embedded/local mode (`qdrant-client`, no server) | Real vector DB in the stack (satisfies the brief), zero network hop keeps the retrieval leg fast. |
| Chunking | 3 strategies, benchmarked | (a) fixed-size + overlap baseline, (b) semantic/embedding-boundary splitting, (c) metadata-aware using MSMARCO's native passage boundaries + language tag. Run recall@k on all 3, keep the numbers, pick winner (or ship as a configurable toggle — stronger demo). |
| Generation | Claude API | Also used for the guardrail checks (grounding, off-topic) — one provider, less glue code. |
| Harness | Custom orchestrator, no LangChain | 2–4 day budget. A framework you have to debug under time pressure is a liability, not a shortcut. |
| Frontend | Gradio | Mic input + text output + shareable URL, minimal build time. Not Next.js — don't spend build time on frontend you're not fast at. |
| Deploy | Hugging Face Spaces | Free, gives you the "live link" requirement directly. |

## Latency — how to actually report it
The brief's 200ms target almost certainly means the **retrieval leg** (chunking + vector search), not STT+LLM generation end-to-end — that combination is seconds, not milliseconds, on any real API. Design so:
- Retrieval leg alone: target and report <200ms.
- Full pipeline (STT + retrieval + generation): report honestly, broken down by stage, P50/P70/P100 across 30–50 real test queries.
Do not fudge this. A transparent breakdown that shows you understand where time goes is stronger than a fake number.

## Guardrails — implementation notes
- **Off-topic**: reject if top retrieval similarity score < threshold (tune empirically, log the threshold you land on).
- **Unsafe/inappropriate input**: lightweight classifier pass on the STT output before it enters the pipeline.
- **Hallucination / groundedness**: post-generation check — does the answer's claims trace back to retrieved chunks? If not, return "insufficient context" instead of guessing.
- This is a scored rubric line item ("show your system knows when not to answer"). Don't shortcut it — it's cheap to build and high-signal to judges.

## Costs — target is $0
- Qdrant (embedded), embeddings (self-hosted), Gradio, HF Spaces CPU, GitHub: free, no account limits to worry about.
- Sarvam: ₹100 free credit on signup (confirm actual figure on dashboard), then ₹30/hr audio pay-as-you-go. Should not be exhausted at hackathon scale.
- Claude API: no permanent free tier, but new accounts start with a small non-expiring credit. Use Haiku 4.5 for guardrail/classification calls, Sonnet 5 only for final generation, to stay inside it. Don't re-run the latency benchmark repeatedly "just to check."
- If anything asks for a credit card before first use, stop and swap to a free alternative — nothing here should require one at this usage level.

## Repo structure (suggested)
```
/data          — ingestion + chunking scripts, recall@k eval
/stt           — Sarvam integration
/retrieval     — embeddings, Qdrant indexing, search
/harness       — orchestrator, guardrails, Claude generation calls
/eval          — latency benchmark script, P50/P70/P100 output
/app           — Gradio UI
README.md      — architecture summary + how to run
```

## Known risks (be honest about these, don't discover them on Aug 21)
- Embedding model choice is the single biggest quality lever on Hindi recall — test it early, not last.
- Qdrant embedded mode vs Qdrant Cloud: embedded is faster for latency numbers but less "production" for judges — pick embedded, mention Cloud as the scale-up path in your README.
- Sarvam free tier rate limits — check before your 30–50 query latency benchmark run, don't get throttled mid-test.
