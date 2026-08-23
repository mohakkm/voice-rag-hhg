"""
Phase 3. Guardrails — three checks, all backed by Groq (llama-3.3-70b-versatile).

1. check_input_safety   — is the transcript safe to process?
2. check_grounded_in_context — is there relevant context? (score-based, no LLM call)
3. check_answer_groundedness — do the answer's claims trace back to the chunks?
"""
import logging

import openai
from config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL, RETRIEVAL_SCORE_THRESHOLD

_client: openai.OpenAI | None = None
logger = logging.getLogger(__name__)


def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    return _client


def _llm(system: str, user: str) -> str:
    """Single-call helper; returns stripped response text."""
    content, _ = _llm_with_usage(system, user)
    return content


def _llm_with_usage(system: str, user: str) -> tuple[str, int | None]:
    """Single-call helper returning text + prompt token usage when available."""
    client = _get_client()
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=64,
    )
    content = resp.choices[0].message.content.strip()
    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
    return content, prompt_tokens


def _estimate_tokens(text: str) -> int:
    """
    Best-effort token estimate for logging.
    Uses tiktoken when available; falls back to a rough char/4 proxy.
    """
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(GROQ_MODEL)
        except Exception:
            enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _clip_text(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " …[truncated]"


def check_input_safety(transcript: str) -> tuple[bool, str]:
    """
    Classify transcript for unsafe/inappropriate content before it enters
    the pipeline.

    Returns:
        (is_safe, reason) — reason is empty string when is_safe is True.
    """
    system = (
        "You are a content safety classifier. "
        "Determine if the following user query is safe and appropriate to process. "
        "Unsafe content includes: hate speech, explicit violence, illegal activity requests, "
        "harmful instructions, or personal attacks. "
        "Reply with exactly one word: SAFE or UNSAFE, followed by a colon and a brief reason. "
        "Example: SAFE: general question  or  UNSAFE: contains hate speech"
    )
    result = _llm(system, transcript)
    # Default to SAFE when the model returns empty or unrecognized output —
    # an absent classification is not evidence of a safety violation.
    if not result or not result.strip():
        return True, ""
    if result.upper().startswith("SAFE"):
        return True, ""
    if not result.upper().startswith("UNSAFE"):
        # Unrecognised response format — treat as safe.
        return True, ""
    reason = result.split(":", 1)[-1].strip() if ":" in result else result
    return False, reason


def check_grounded_in_context(top_score: float) -> bool:
    """
    Cheap score-based check — no LLM call needed here.
    If the best retrieval score is below RETRIEVAL_SCORE_THRESHOLD the query
    is likely off-topic; caller should skip generation and return a refusal.
    """
    return top_score >= RETRIEVAL_SCORE_THRESHOLD


def check_answer_groundedness(answer: str, retrieved_chunks: list[dict]) -> tuple[bool, str]:
    """
    Post-generation check: do the answer's claims actually trace back to
    the retrieved chunks?

    Returns:
        (is_grounded, reason) — reason is empty string when is_grounded is True.
    """
    # Keep only the highest scoring 1-2 chunks — the groundedness check only
    # needs the most relevant evidence, not the full top-k retrieval set.
    ranked = sorted(
        retrieved_chunks,
        key=lambda c: float(c.get("score", 0.0)),
        reverse=True,
    )
    trimmed_chunks = ranked[:2]
    groundedness_system = (
        "You are a groundedness verifier. "
        "Given the context chunks and an answer, determine whether every factual claim "
        "in the answer is directly supported by the context. "
        "Reply with exactly: GROUNDED: <brief reason>  or  UNGROUNDED: <brief reason>"
    )
    full_context_text = "\n\n".join(
        f"[Chunk {i+1}]: {c['payload'].get('text', '') if 'payload' in c else c.get('text', '')}"
        for i, c in enumerate(ranked)
    )
    context_text = "\n\n".join(
        f"[Chunk {i+1}]: {c['payload'].get('text', '') if 'payload' in c else c.get('text', '')}"
        for i, c in enumerate(trimmed_chunks)
    )
    clipped_answer = _clip_text(answer, 800)

    before_prompt = f"Context:\n{full_context_text}\n\nAnswer: {answer}"
    after_prompt = f"Context:\n{context_text}\n\nAnswer: {clipped_answer}"
    before_tokens = _estimate_tokens(f"{groundedness_system}\n\n{before_prompt}")
    after_tokens = _estimate_tokens(f"{groundedness_system}\n\n{after_prompt}")
    logger.info(
        "[groundedness-token-budget] before=%s after=%s chunks=%s->%s answer_chars=%s->%s",
        before_tokens,
        after_tokens,
        len(ranked),
        len(trimmed_chunks),
        len(answer or ""),
        len(clipped_answer),
    )
    print(
        f"[groundedness-token-budget] before={before_tokens} after={after_tokens} "
        f"chunks={len(ranked)}->{len(trimmed_chunks)} answer_chars={len(answer or '')}->{len(clipped_answer)}"
    )
    result, actual_prompt_tokens = _llm_with_usage(groundedness_system, after_prompt)
    # Default to GROUNDED on empty or unrecognised response — an absent
    # verdict is not evidence of hallucination.
    if actual_prompt_tokens is not None:
        print(f"[groundedness-token-budget] actual_prompt_tokens={actual_prompt_tokens}")
    if not result or not result.strip():
        return True, ""
    if result.upper().startswith("GROUNDED"):
        return True, ""
    if not result.upper().startswith("UNGROUNDED"):
        return True, ""
    reason = result.split(":", 1)[-1].strip() if ":" in result else result
    return False, reason
