"""
Phase 4. Guardrails — this is a scored rubric line item, don't shortcut it.
Three checks: input (unsafe), retrieval (off-topic), output (grounded).
"""
from config import RETRIEVAL_SCORE_THRESHOLD


def check_input_safety(transcript: str) -> tuple[bool, str]:
    """
    TODO(phase-4): classify transcript for unsafe/inappropriate content before
    it enters the pipeline. Returns (is_safe, reason_if_not).
    """
    raise NotImplementedError


def check_grounded_in_context(top_score: float) -> bool:
    """
    TODO(phase-4): if top_score < RETRIEVAL_SCORE_THRESHOLD, this is off-topic —
    caller should return "insufficient context" instead of calling generation.
    """
    raise NotImplementedError


def check_answer_groundedness(answer: str, retrieved_chunks: list[dict]) -> tuple[bool, str]:
    """
    TODO(phase-4): post-generation check — does the answer's claims trace back
    to retrieved_chunks? Use a cheap Claude Haiku call, not Sonnet, for this check.
    Returns (is_grounded, reason_if_not).
    """
    raise NotImplementedError
