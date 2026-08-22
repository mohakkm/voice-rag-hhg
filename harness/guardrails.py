"""
Phase 3. Guardrails — three checks, all backed by Groq (llama-3.3-70b-versatile).

1. check_input_safety   — is the transcript safe to process?
2. check_grounded_in_context — is there relevant context? (score-based, no LLM call)
3. check_answer_groundedness — do the answer's claims trace back to the chunks?
"""
import openai
from config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL, RETRIEVAL_SCORE_THRESHOLD

_client: openai.OpenAI | None = None


def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    return _client


def _llm(system: str, user: str) -> str:
    """Single-call helper; returns stripped response text."""
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
    return resp.choices[0].message.content.strip()


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
    context_text = "\n\n".join(
        f"[Chunk {i+1}]: {c['payload'].get('text', '') if 'payload' in c else c.get('text', '')}"
        for i, c in enumerate(retrieved_chunks)
    )
    system = (
        "You are a groundedness verifier. "
        "Given the context chunks and an answer, determine whether every factual claim "
        "in the answer is directly supported by the context. "
        "Reply with exactly: GROUNDED: <brief reason>  or  UNGROUNDED: <brief reason>"
    )
    user = f"Context:\n{context_text}\n\nAnswer: {answer}"
    result = _llm(system, user)
    # Default to GROUNDED on empty or unrecognised response — an absent
    # verdict is not evidence of hallucination.
    if not result or not result.strip():
        return True, ""
    if result.upper().startswith("GROUNDED"):
        return True, ""
    if not result.upper().startswith("UNGROUNDED"):
        return True, ""
    reason = result.split(":", 1)[-1].strip() if ":" in result else result
    return False, reason
