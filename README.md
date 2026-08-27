---
title: HHGOA Task2 Voice RAG
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.19.2
app_file: app/gradio_app.py
pinned: false
---

# Voice RAG — HH Goa 2026, Task 2

Voice-enabled retrieval-augmented generation over the Hindi subset of
`ai4bharat/MSMARCO-XI` (`config="hi"`).

**Pipeline:**
```
Mic audio
  → [1] Sarvam STT (Saaras v3, language_code="hi-IN")
  → [2] Guardrail: input classifier (unsafe / gibberish check)
  → [3] Embed query (multilingual-e5-large)
  → [4] Qdrant retrieval (top-k, embedded/local mode)
  → [5] Guardrail: grounding check (top score < threshold → refuse)
  → [6] Groq API — answer generation from retrieved context only
  → [7] Guardrail: post-gen groundedness check
  → [8] Structured JSON response: {answer, sources, confidence, latency_breakdown}
```

Every stage is independently timestamped. Full architecture rationale is in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| STT | Sarvam Saaras v3 (`hi-IN`) |
| Embeddings | `intfloat/multilingual-e5-large` |
| Vector DB | Qdrant, embedded/local mode (`qdrant-client`, no server process) |
| Generation | Groq API |
| Guardrails | Groq API (same provider — grounding, off-topic, unsafe input checks) |
| Harness | Custom orchestrator — no LangChain |
| Frontend | Gradio |
| Deploy | Hugging Face Spaces |

---

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # fill in SARVAM_API_KEY and GROQ_API_KEY
```

## Run

```bash
python app/gradio_app.py
```

---

## Chunking Strategies & Recall

Three chunking strategies were benchmarked on the Hindi MSMARCO-XI corpus.
Evaluation used **N=8,000 sampled queries (seed=42)**, of which **4,491 were
evaluated**. The only recall depth that was measured is **Recall@5**.

| Strategy | Parameters | Recall@5 |
|----------|-----------|----------|
| `fixed_overlap` | chunk_size=256, overlap=32 | **0.870** |
| `semantic` | threshold=0.75, chunk_size=256, merge_limit=20 | **0.857** |
| `metadata_aware` | min_words=20, max_words=400 | 0.723 |

**Notes:**
- `fixed_overlap` and `semantic` are statistically close (0.870 vs 0.857).
- `metadata_aware` scores lower (0.723) but is included for its latency
  profile and as a demonstration of metadata-boundary-aware chunking using
  MSMARCO's native passage boundaries.
- All three strategies are available as a configurable toggle in the app.
- No recall@1 or recall@3 was measured; those numbers are not reported here.

---

## Latency

Benchmark configuration:
- **15 text-only runs** (STT stage bypassed — audio not active in this benchmark
  mode, so STT columns read 0.0 ms).
- **Model warm-up:** The embedding model load and Qdrant embedded client open
  are one-time costs at process start (combined ~37,222 ms on first process
  launch). All 15 timed runs were performed **after warm-up was confirmed**.
- **Groundedness check excluded from timing:** Skipped during the benchmark
  due to Groq free-tier daily token limits; confirmed working in the live demo
  path.
- **Refused:** 2 of 15 queries were refused by the guardrail (13.3% refusal
  rate). **Errors:** 0.

### Stage-level percentiles (ms)

All values taken directly from `eval/results.json`.

| Stage | P50 (ms) | P70 (ms) | P100 (ms) |
|-------|----------|----------|-----------|
| stt | 0.00 | 0.00 | 0.00 |
| embed | 116.51 | 118.32 | 726.10 |
| retrieve | 310.13 | 319.29 | 427.59 |
| generate | 5,870.58 | 8,118.97 | 20,808.23 |
| **total** | **6,981.07** | **8,815.00** | **21,622.56** |

### Retrieval leg (embed + retrieve)

The combined retrieval leg (embed + retrieve) sits at **~426–438 ms at
P50/P70** (sum of the two measured stage percentiles above). This does **not**
meet the <200 ms retrieval target in the current benchmark. That gap is reported
transparently rather than hidden.

### Full pipeline

End-to-end latency is **generation-bound**. At P50, generation alone accounts
for ~5.87 s of the ~6.98 s total. This is expected for any LLM-backed voice
pipeline and is documented here per stage rather than collapsed into a single
number.

If lower retrieval latency is required, options are: a smaller/faster embedding
model, co-locating the Qdrant instance (Qdrant Cloud as the scale-up path), or
caching frequent queries. Reducing generation latency requires different model
or compute choices for that stage.

---

## Corpus Details

- Dataset: `ai4bharat/MSMARCO-XI`, config `"hi"` (Hindi only)
- 8,000 queries sampled (seed=42), saved to `data/corpus_sample.jsonl`
- Passages concatenated per-query to `data/grouped_docs.jsonl`
- `fixed_overlap` chunking produced **23,989 chunks** at a mean of
  **217.6 words/chunk**

---

## Current Status

See [`STATE.md`](STATE.md) for the active build phase and open tasks.
