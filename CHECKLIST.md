# HHG Task 2 — Checklist

Deadline: **Aug 22, 2026, 11:59 PM**. No resubmissions — final means final.

## ⚠️ Disqualification risk — read this first
Every team member must individually post BOTH videos on Instagram, X, and LinkedIn, tagged **#RAGInGoa**, on every platform, by every person (not one shared team post). At least 1 Instagram account must be public. This is easy to forget under build pressure — don't.

---

## Phase 1 — Data & Chunking
- [x] Pull `ai4bharat/MSMARCO-XI`, config `"hi"`
- [x] Implement chunking strategy (a): fixed-size + overlap
- [ ] Implement chunking strategy (b): semantic/embedding-boundary
- [ ] Implement chunking strategy (c): metadata-aware, native passage boundaries
- [ ] Index all 3 into Qdrant (embedded mode)
- [ ] Run recall@k eval across all 3, record numbers
- [ ] Pick winner or ship as configurable toggle

## Phase 2 — Voice Pipeline
- [ ] Sarvam API key + Saaras v3 wired, `language_code="hi-IN"`
- [ ] Mic input → transcription tested with real Hindi audio
- [ ] Handle silence / empty audio gracefully
- [ ] Handle code-mixed (Hinglish) input without crashing

## Phase 3 — Harness & Retrieval
- [ ] Orchestrator function: STT → embed → retrieve → generate → structured output
- [ ] Claude API call for answer generation, context-only prompting
- [ ] Retries + timeout handling on every external API call
- [ ] Structured JSON output: answer, sources, confidence, latency breakdown

## Phase 4 — Guardrails
- [ ] Off-topic detector (similarity threshold on retrieval score)
- [ ] Unsafe/inappropriate input classifier on STT output
- [ ] Post-generation groundedness check (answer traces to retrieved context)
- [ ] Refusal path returns "insufficient context," not a hallucinated guess
- [ ] Adversarial test pass: silence, gibberish, off-topic query, unsafe query — confirm each is handled, not crashed

## Phase 5 — Latency, UI, Deploy
- [ ] Gradio UI: mic in, transcript + answer + sources out
- [ ] Deploy to Hugging Face Spaces, confirm live link works from a cold browser
- [ ] Latency benchmark script run across 30–50 real queries
- [ ] P50 / P70 / P100 computed and reported, broken down by pipeline stage
- [ ] README written: architecture summary, how to run, latency table

## Phase 6 — Submission & Promotion
- [ ] Process video (90s) — in progress, being shot along the way
- [ ] Demo video — full pipeline, end to end, real
- [ ] Both videos posted individually by every team member on IG, X, LinkedIn with #RAGInGoa
- [ ] ≥1 public Instagram account confirmed
- [ ] GitHub repo public, clean, README complete
- [ ] Submission form filled: https://forms.gle/MNvCjcv23Hn2Eeu58
- [ ] Final QA pass on live link before submitting — no resubmissions allowed

---
Update this file as you go — check items off, don't let it go stale. Pair with STATE.md for the current live status.
