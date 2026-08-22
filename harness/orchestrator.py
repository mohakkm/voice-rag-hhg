"""
Phase 3. Full pipeline orchestrator.

Pipeline:
  STT -> check_input_safety -> embed -> qdrant search ->
  check_grounded_in_context -> generate_answer -> check_answer_groundedness
  -> structured return

Every external call (Sarvam, Qdrant, Groq) is wrapped with a 2-attempt retry
and 1-second backoff — simple, not elaborate (Phase 3 spec).
"""
import time
import logging

logger = logging.getLogger(__name__)

COLLECTION_NAME = "metadata_aware_chunks"


def retry_with_backoff(fn, max_retries: int = 2, base_delay: float = 1.0):
    """
    Retry fn up to max_retries times with fixed base_delay between attempts.
    Raises the last exception if all attempts fail.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(base_delay)
    raise last_exc


def run_pipeline(audio_path: str) -> dict:
    """
    Run the full voice-RAG pipeline for a single audio file.

    Returns:
        {
            "transcript":      str | None,
            "answer":          str | None,
            "sources":         list[dict],
            "confidence":      float,        # top retrieval score, 0.0 on failure
            "refused":         bool,
            "refusal_reason":  str | None,
            "groq_ratelimit_headers": dict,  # from the generation call
            "latency_ms": {
                "stt": float, "embed": float, "retrieve": float,
                "generate": float, "guardrails": float, "total": float
            }
        }
    """
    from stt.sarvam_client import transcribe
    from retrieval.embeddings import embed_query, get_embed_debug_state
    from retrieval.qdrant_store import search, get_qdrant_debug_state
    from harness.generation import generate_answer
    from harness.guardrails import (
        check_input_safety,
        check_grounded_in_context,
        check_answer_groundedness,
    )

    t_total_start = time.time()
    latency = {"stt": 0.0, "embed": 0.0, "retrieve": 0.0,
               "generate": 0.0, "guardrails": 0.0, "total": 0.0}
    groq_headers: dict = {}

    def diagnostics() -> dict:
        return {
            "embedding_singleton": get_embed_debug_state(),
            "qdrant_singleton": get_qdrant_debug_state(),
        }

    # ── STT ──────────────────────────────────────────────────────────────────
    t0 = time.time()
    stt_result = retry_with_backoff(lambda: transcribe(audio_path))
    latency["stt"] = (time.time() - t0) * 1000

    if not stt_result["success"]:
        latency["total"] = (time.time() - t_total_start) * 1000
        return {
            "transcript": None, "answer": None, "sources": [],
            "confidence": 0.0, "refused": True,
            "refusal_reason": f"stt_failed: {stt_result['error']}",
            "groq_ratelimit_headers": {},
            "diagnostics": diagnostics(),
            "latency_ms": latency,
        }

    transcript = stt_result["transcript"]

    # ── Input safety guardrail ────────────────────────────────────────────────
    t0 = time.time()
    is_safe, safety_reason = retry_with_backoff(
        lambda: check_input_safety(transcript)
    )
    latency["guardrails"] += (time.time() - t0) * 1000

    if not is_safe:
        latency["total"] = (time.time() - t_total_start) * 1000
        return {
            "transcript": transcript, "answer": None, "sources": [],
            "confidence": 0.0, "refused": True,
            "refusal_reason": f"unsafe_input: {safety_reason}",
            "groq_ratelimit_headers": {},
            "diagnostics": diagnostics(),
            "latency_ms": latency,
        }

    # ── Embed query ───────────────────────────────────────────────────────────
    t0 = time.time()
    query_vector = retry_with_backoff(lambda: embed_query(transcript))
    latency["embed"] = (time.time() - t0) * 1000

    # ── Qdrant retrieval ──────────────────────────────────────────────────────
    t0 = time.time()
    hits = retry_with_backoff(lambda: search(COLLECTION_NAME, query_vector))
    latency["retrieve"] = (time.time() - t0) * 1000

    top_score = hits[0]["score"] if hits else 0.0

    # ── Retrieval groundedness guardrail ──────────────────────────────────────
    if not check_grounded_in_context(top_score):
        latency["total"] = (time.time() - t_total_start) * 1000
        return {
            "transcript": transcript, "answer": "insufficient context",
            "sources": hits, "confidence": top_score,
            "refused": True,
            "refusal_reason": f"off_topic: top_score={top_score:.3f} below threshold",
            "groq_ratelimit_headers": {},
            "diagnostics": diagnostics(),
            "latency_ms": latency,
        }

    # ── Generation ────────────────────────────────────────────────────────────
    t0 = time.time()
    answer, groq_headers = retry_with_backoff(
        lambda: generate_answer(transcript, hits)
    )
    latency["generate"] = (time.time() - t0) * 1000

    # If the model itself signals the context doesn't contain the answer,
    # treat it as a clean refusal — skip the groundedness check entirely.
    if answer.strip().lower() == "insufficient context":
        latency["total"] = (time.time() - t_total_start) * 1000
        return {
            "transcript": transcript,
            "answer": answer,
            "sources": hits,
            "confidence": top_score,
            "refused": True,
            "refusal_reason": "model_said_insufficient_context",
            "groq_ratelimit_headers": groq_headers,
            "diagnostics": diagnostics(),
            "latency_ms": latency,
        }

    # ── Answer groundedness guardrail ─────────────────────────────────────────
    t0 = time.time()
    is_grounded, grounded_reason = retry_with_backoff(
        lambda: check_answer_groundedness(answer, hits)
    )
    latency["guardrails"] += (time.time() - t0) * 1000

    latency["total"] = (time.time() - t_total_start) * 1000

    return {
        "transcript": transcript,
        "answer": answer,
        "sources": hits,
        "confidence": top_score,
        "refused": not is_grounded,
        "refusal_reason": f"ungrounded_answer: {grounded_reason}" if not is_grounded else None,
        "groq_ratelimit_headers": groq_headers,
        "diagnostics": diagnostics(),
        "latency_ms": latency,
    }


def run_pipeline_from_text(transcript: str) -> dict:
    """
    Run the retrieval + generation + guardrails stages with a pre-supplied
    transcript, skipping STT entirely. Useful for testing and benchmarking
    when audio is not required.
    """
    # Wrap inside a fake STT result and reuse the shared inner logic.
    # We monkey-patch transcribe to return the supplied text so the same
    # retry/latency bookkeeping path is exercised without touching the network.
    import stt.sarvam_client as _sc
    _orig = _sc.transcribe

    def _stub(path):
        return {"success": True, "transcript": transcript,
                "latency_ms": 0.0, "error": None}

    _sc.transcribe = _stub
    try:
        result = run_pipeline("__text_injection__")
    finally:
        _sc.transcribe = _orig

    # Overwrite the latency entry so callers know STT was skipped.
    result["latency_ms"]["stt"] = 0.0
    return result
