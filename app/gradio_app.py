"""
Phase 5. Gradio UI — mic input, transcript + answer + sources out.
Deploy target: Hugging Face Spaces (CPU tier, free — see ARCHITECTURE.md).
"""
import html
import logging
import os
import tempfile
import time
import traceback
import wave

import gradio as gr
import numpy as np

from harness.orchestrator import run_pipeline
from config import validate
from retrieval.embeddings import get_model
from retrieval.qdrant_store import get_client

_WARMED_UP = False
_WARMUP_MS = 0.0

logger = logging.getLogger(__name__)

READY_CARD = """
<div style="border:2px solid #1d4ed8;border-radius:10px;padding:12px;background:#eff6ff;">
  <div style="font-weight:700;color:#1e3a8a;">🎙️ Ready</div>
  <div style="color:#1e40af;">Record or upload a Hindi question, then click Run Pipeline.</div>
</div>
"""

SUCCESS_CARD = """
<div style="border:2px solid #15803d;border-radius:10px;padding:12px;background:#ecfdf5;">
  <div style="font-weight:700;color:#166534;">✅ Answer Generated</div>
  <div style="color:#14532d;">Query passed guardrails and returned a grounded response.</div>
</div>
"""

REFUSAL_CARD_TEMPLATE = """
<div style="border:2px solid #b91c1c;border-radius:10px;padding:12px;background:#fef2f2;">
  <div style="font-weight:700;color:#991b1b;">🛑 Intentional Guardrail Refusal</div>
  <div style="color:#7f1d1d;">This is an intentional refusal decision, not a system crash.</div>
  <div style="margin-top:8px;color:#7f1d1d;"><b>Reason:</b> {reason}</div>
</div>
"""

ERROR_CARD_TEMPLATE = """
<div style="border:2px solid #854d0e;border-radius:10px;padding:12px;background:#fffbeb;">
  <div style="font-weight:700;color:#713f12;">⚠️ Pipeline Error</div>
  <div style="color:#78350f;">An unexpected runtime error occurred while processing this request.</div>
  <div style="margin-top:8px;color:#78350f;"><b>Error:</b> {error}</div>
</div>
"""


def warmup_runtime() -> float:
    """Load embedding model + Qdrant client exactly once per process startup."""
    global _WARMED_UP, _WARMUP_MS
    if _WARMED_UP:
        return _WARMUP_MS

    t0 = time.perf_counter()
    get_model()
    get_client()
    _WARMUP_MS = (time.perf_counter() - t0) * 1000.0
    print(f"[warmup] Loaded embedding model + Qdrant client in {_WARMUP_MS:.1f} ms")
    _WARMED_UP = True
    return _WARMUP_MS


def _format_sources_rows(sources: list[dict]) -> list[list]:
    rows: list[list] = []
    for idx, hit in enumerate(sources or [], start=1):
        payload = hit.get("payload", {}) if isinstance(hit, dict) else {}
        text = (payload.get("text") or hit.get("text") or "").replace("\n", " ").strip()
        if len(text) > 240:
            text = text[:237] + "..."
        rows.append([idx, round(float(hit.get("score", 0.0)), 4), text])
    return rows


def _format_latency_markdown(latency_ms: dict) -> str:
    stage_order = ("stt", "embed", "retrieve", "guardrails", "generate", "total")
    lines = ["| Stage | Latency (ms) |", "|---|---:|"]
    for stage in stage_order:
        value = latency_ms.get(stage)
        if value is None:
            continue
        lines.append(f"| {stage} | {float(value):.2f} |")
    return "\n".join(lines)


def _write_temp_wav(audio_input) -> tuple[str | None, bool]:
    """
    Convert Gradio audio input to a filesystem path accepted by run_pipeline.
    Returns (audio_path, should_delete_path).
    """
    if audio_input is None:
        return None, False

    if isinstance(audio_input, str):
        return audio_input, False

    if isinstance(audio_input, dict):
        path = audio_input.get("path")
        if path:
            return str(path), False
        return None, False

    if not isinstance(audio_input, (tuple, list)) or len(audio_input) != 2:
        raise ValueError("Unexpected audio input format from Gradio.")

    sample_rate, waveform = audio_input
    if waveform is None:
        return None, False

    arr = np.asarray(waveform)
    if arr.size == 0:
        return None, False

    if arr.ndim == 2:
        # Downmix stereo/multichannel to mono for STT ingestion.
        arr = np.mean(arr, axis=1)
    elif arr.ndim != 1:
        raise ValueError(f"Unexpected waveform dimensions: {arr.shape}")

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr, -1.0, 1.0)
        pcm = (arr * 32767.0).astype(np.int16)
    else:
        pcm = arr.astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_path = tmp.name

    with wave.open(temp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())

    return temp_path, True


def _run_from_audio(audio_input):
    audio_path = None
    should_delete_audio = False
    try:
        audio_path, should_delete_audio = _write_temp_wav(audio_input)
        if not audio_path:
            raise ValueError("Please record audio before submitting.")

        result = run_pipeline(audio_path)
        transcript = result.get("transcript") or ""
        answer = result.get("answer") or ""
        sources_rows = _format_sources_rows(result.get("sources", []))
        latency_md = _format_latency_markdown(result.get("latency_ms", {}))
        refused = bool(result.get("refused"))

        if refused:
            reason = html.escape(result.get("refusal_reason") or "Guardrail policy refusal")
            status = REFUSAL_CARD_TEMPLATE.format(reason=reason)
        else:
            status = SUCCESS_CARD

        return (
            status,
            transcript,
            answer,
            sources_rows,
            latency_md,
        )
    except Exception as exc:
        logger.exception("Pipeline crashed for audio request.")
        print(f"[gradio] Pipeline crashed for audio request: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        err = html.escape(f"{type(exc).__name__}: {exc}")
        return (
            ERROR_CARD_TEMPLATE.format(error=err),
            "",
            "",
            [],
            "",
        )
    finally:
        if should_delete_audio:
            try:
                os.remove(audio_path)
            except OSError:
                logger.warning("Failed to remove temporary audio file: %s", audio_path)


def build_interface():
    with gr.Blocks(title="Voice RAG (Hindi)") as demo:
        gr.Markdown("## Voice RAG Demo")
        gr.Markdown("Record a Hindi question using the microphone and submit.")

        with gr.Row():
            audio_in = gr.Audio(
                sources=["microphone", "upload"],
                type="numpy",
                label="Microphone Input",
            )

        submit_btn = gr.Button("Run Pipeline", variant="primary")
        clear_btn = gr.Button("Clear")

        status = gr.HTML(READY_CARD)
        transcript_out = gr.Textbox(label="Transcript", lines=3)
        answer_out = gr.Textbox(label="Answer", lines=6)
        sources_out = gr.Dataframe(
            headers=["rank", "score", "text_preview"],
            datatype=["number", "number", "str"],
            label="Retrieved Sources (with scores)",
            row_count=(0, "dynamic"),
            wrap=True,
        )
        gr.Markdown("### Latency Breakdown")
        latency_out = gr.Markdown("")

        submit_btn.click(
            fn=_run_from_audio,
            inputs=[audio_in],
            outputs=[status, transcript_out, answer_out, sources_out, latency_out],
        )
        clear_btn.click(
            fn=lambda: (READY_CARD, None, "", "", [], ""),
            inputs=[],
            outputs=[status, audio_in, transcript_out, answer_out, sources_out, latency_out],
        )

    return demo


# Warm up the runtime at module load so the first real UI query avoids cold start.
warmup_runtime()


if __name__ == "__main__":
    validate()
    demo = build_interface()
    demo.queue(max_size=1)
    demo.launch()
