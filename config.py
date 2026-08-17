"""
Central config loader. Every module reads settings from here —
nothing should call os.environ directly outside this file.
"""
import os
from dotenv import load_dotenv

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_data")
LANGUAGE = os.getenv("LANGUAGE", "hi-IN")

# Guardrail thresholds — tune these empirically in Phase 4, log what you land on.
RETRIEVAL_SCORE_THRESHOLD = 0.5   # below this -> off-topic refusal
TOP_K = 5

def validate():
    """Call at app startup. Fails loud if keys are missing instead of failing weird later."""
    missing = [k for k, v in {
        "SARVAM_API_KEY": SARVAM_API_KEY,
        "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
    }.items() if not v]
    if missing:
        raise RuntimeError(f"Missing required env vars: {missing}. Copy .env.example to .env and fill in.")
