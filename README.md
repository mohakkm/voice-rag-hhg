# Voice RAG — HH Goa 2026, Task 2

Voice-enabled RAG over the Hindi subset of `ai4bharat/MSMARCO-XI`.
Pipeline: mic → Sarvam STT → Qdrant retrieval → Claude generation, with
guardrails at every stage. Full technical decisions in `ARCHITECTURE.md`.

## Status
See `STATE.md` for current phase and `CHECKLIST.md` for the 6-phase build plan.

## Setup
```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # then fill in SARVAM_API_KEY and ANTHROPIC_API_KEY
```

## Run
```bash
python app/gradio_app.py
```
(Won't work yet — most modules are stubs. Build order is Phase 1 → 6, see CHECKLIST.md.)

## Latency

Benchmark vs demo model
- Benchmark (used by eval/latency_bench.py): `openai/gpt-oss-20b`.
- Demo / generation model (used in the live demo path): the Groq model configured via `config.GROQ_MODEL` (currently `openai/gpt-oss-120b`).

Why they differ: the benchmark temporarily points Groq-backed calls at a smaller model (`openai/gpt-oss-20b`) to avoid hitting daily token limits on larger models during free-tier testing; the demo uses whatever Groq model is available to the deployment. See `eval/latency_bench.py` for the exact override and `config.py` for the demo model setting.

Measured percentiles (ms)
- The table below is taken from the last run of `eval/latency_bench.py` (saved in `eval/results.json`) and shows P50 / P70 / P100 for each stage.

Stage | P50 (ms) | P70 (ms) | P100 (ms)
----- | -------- | -------- | ---------
stt    | 0.0      | 0.0      | 0.0
embed  | 116.5123 | 118.3229 | 726.0954
retrieve | 310.1277 | 319.2902 | 427.5882
generate | 5870.5795 | 8118.9747 | 20808.2294
total    | 6981.0688 | 8814.9964 | 21622.5631

Retrieval leg (embed + retrieve)
- Combined (embed + retrieve) P50 ≈ 426.64 ms, P70 ≈ 437.61 ms, P100 ≈ 1153.68 ms — these values are the sum of the measured embed and retrieve percentiles above.
- Conclusion: the retrieval leg does NOT meet a <200ms target in the current benchmark; the measured retrieval path is roughly 400–450 ms at P50/P70 and can spike higher at P100.

Full pipeline
- The full pipeline is generation-bound: generation dominates end-to-end latency (generate P50 ≈ 5.87s, total P50 ≈ 6.98s in this benchmark). This is consistent with any LLM-backed voice pipeline and is reported transparently here rather than forcing numbers to look artificially fast.

Notes
- `eval/latency_bench.py` temporarily overrides `config.GROQ_MODEL` with `BENCHMARK_GROQ_MODEL = "openai/gpt-oss-20b"` during timing runs; see the script for details and the recorded `eval/results.json` for the raw per-query results used to compute these percentiles.
- If you need lower retrieval latency, consider using a faster/smaller embedding model, colocating the Qdrant instance, or caching frequent queries. Reducing generation latency requires different model/compute choices for the generation stage.

See also: [ARCHITECTURE.md](/C:/Users/Mohakk/Desktop/voice-rag-hhg/ARCHITECTURE.md) for rationale on how latency is measured and why the project reports numbers this way.
