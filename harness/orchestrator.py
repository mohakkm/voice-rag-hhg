"""
Phase 3. The harness — structured orchestration, not a single raw prompt call.
This is what the brief means by "harness your model." Every stage timestamped
for the latency breakdown (see eval/latency_bench.py).

Pipeline: STT -> guardrail(input) -> embed -> retrieve -> guardrail(grounding)
          -> generate -> guardrail(output) -> structured response
"""
import time


def run_pipeline(audio_path: str) -> dict:
    """
    TODO(phase-3): wire stt.transcribe, retrieval.embed/search,
    harness.guardrails, harness.generate_answer together. Wrap every external
    call (Sarvam, Qdrant, Claude) with retry + timeout — see retry_with_backoff below.

    Return: {
        "transcript": str, "answer": str, "sources": list[dict],
        "confidence": float, "refused": bool, "refusal_reason": str | None,
        "latency_ms": {"stt": ..., "embed": ..., "retrieve": ..., "generate": ..., "total": ...}
    }
    """
    raise NotImplementedError


def retry_with_backoff(fn, max_retries=3, base_delay=0.5):
    """TODO(phase-3): generic retry wrapper for external API calls. Used by every stage above."""
    raise NotImplementedError
