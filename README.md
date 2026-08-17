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
P50 / P70 / P100 numbers go here once `eval/latency_bench.py` runs. Retrieval leg
target: <200ms. Full pipeline (incl. STT + generation) reported honestly, not forced
under 200ms — see ARCHITECTURE.md for why.
