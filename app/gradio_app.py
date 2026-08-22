"""
Phase 5. Gradio UI — mic input, transcript + answer + sources out.
Deploy target: Hugging Face Spaces (CPU tier, free — see ARCHITECTURE.md).
"""
import time
import gradio as gr
from config import validate
from retrieval.embeddings import get_model
from retrieval.qdrant_store import get_client

_WARMED_UP = False


def warmup_runtime() -> None:
    """Load embedding model + Qdrant client exactly once per process startup."""
    global _WARMED_UP
    if _WARMED_UP:
        return

    t0 = time.perf_counter()
    get_model()
    get_client()
    warmup_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[warmup] Loaded embedding model + Qdrant client in {warmup_ms:.1f} ms")
    _WARMED_UP = True


def build_interface():
    """
    TODO(phase-5): gr.Interface or gr.Blocks with:
    - gr.Audio(source="microphone") input
    - calls harness.orchestrator.run_pipeline
    - outputs: transcript, answer, sources (with scores), latency breakdown
    - if result["refused"], surface the refusal reason clearly, don't hide it
    """
    raise NotImplementedError


if __name__ == "__main__":
    validate()
    warmup_runtime()
    demo = build_interface()
    demo.launch()
