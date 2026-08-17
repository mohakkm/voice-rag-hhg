"""
Phase 3. Claude API wrapper for answer generation. Sonnet for the real answer,
Haiku for cheap guardrail checks (see harness/guardrails.py) — keep spend down.
"""
from config import ANTHROPIC_API_KEY


def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    """
    TODO(phase-3): prompt Claude with query + retrieved_chunks ONLY as context.
    Explicit instruction: answer only from provided context, say so if it can't.
    Model: claude-sonnet-5 (see ARCHITECTURE.md cost notes before changing this).
    """
    raise NotImplementedError
