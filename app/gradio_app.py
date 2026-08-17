"""
Phase 5. Gradio UI — mic input, transcript + answer + sources out.
Deploy target: Hugging Face Spaces (CPU tier, free — see ARCHITECTURE.md).
"""
import gradio as gr
from config import validate


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
    demo = build_interface()
    demo.launch()
