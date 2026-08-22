"""
End-to-end harness test — three cases:

(a) Real Hindi question the corpus can answer (text-injected, STT skipped)
    -> grounded answer, refused=False
(b) Off-topic Hindi question via real audio (test_h3.wav: meeting agenda)
    -> refused at retrieval grounding-check, no hallucination
(c) Silent/empty audio file
    -> clean STT failure, refused=True, no crash, no traceback

Also prints Groq rate-limit headers from the generation call in case (a).

Usage:
    PYTHONPATH=. python scripts/test_harness_e2e.py
"""
import io
import json
import os
import sys
import wave

# ── ensure repo root is on the path ──────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(ROOT))

# Force UTF-8 output so Hindi text doesn't crash on Windows cp1252 console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from harness.orchestrator import run_pipeline, run_pipeline_from_text


# ── helpers ───────────────────────────────────────────────────────────────────

def make_silent_wav(path: str, duration_s: float = 1.0, sample_rate: int = 16000):
    """Write a valid but silent 16-bit mono WAV file."""
    num_samples = int(duration_s * sample_rate)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_samples)


def pretty(result: dict) -> str:
    safe = {}
    for k, v in result.items():
        if k == "sources":
            safe[k] = [
                {"score": round(h["score"], 4),
                 "text_preview": (h.get("payload", {}).get("text", "") or "")[:120]}
                for h in (v or [])
            ]
        else:
            safe[k] = v
    return json.dumps(safe, ensure_ascii=False, indent=2)


# ── test cases ────────────────────────────────────────────────────────────────

# (a) Corpus question with a literal answer in the retrieved chunks.
#     Corpus chunk: "ट्रेलब्लेज़र बसें सोमवार से शुक्रवार तक सुबह 6:30 बजे से शाम 5:30 बजे तक
#     संचालित होती हैं" (Trailblazer buses operate Mon-Fri 6:30am-5:30pm).
#     This answer is stated verbatim — no inference needed.
GROUNDED_QUERY = "ट्रेलब्लेज़र ट्रांज़िट की बसें किस समय से किस समय तक चलती हैं?"

# (b) Off-topic via real audio — test_h3.wav transcribes to a meeting agenda
#     in Hindi; unrelated to the corpus, should hit the retrieval score guard.
OFF_TOPIC_WAV = os.path.join(ROOT, "data", "test_h3.wav")

# (c) Silent WAV — must not crash, must return a clean error state.
SILENT_WAV = os.path.join(ROOT, "data", "_test_silent.wav")


def run_case_text(label: str, transcript: str):
    print("\n" + "=" * 70)
    print(f"CASE {label}")
    print(f"  transcript injected: {transcript}")
    print("=" * 70)
    try:
        result = run_pipeline_from_text(transcript)
    except Exception as exc:
        print(f"  !! UNHANDLED EXCEPTION: {type(exc).__name__}: {exc}")
        import traceback; traceback.print_exc()
        return None
    print(pretty(result))
    return result


def run_case_audio(label: str, audio_path: str):
    print("\n" + "=" * 70)
    print(f"CASE {label}")
    print(f"  audio: {os.path.basename(audio_path)}")
    print("=" * 70)
    try:
        result = run_pipeline(audio_path)
    except Exception as exc:
        print(f"  !! UNHANDLED EXCEPTION: {type(exc).__name__}: {exc}")
        import traceback; traceback.print_exc()
        return None
    print(pretty(result))
    return result


def main():
    import config
    config.validate()

    make_silent_wav(SILENT_WAV)

    result_a = run_case_text(
        "(a) grounded question [text-inject] — should answer, refused=False",
        GROUNDED_QUERY,
    )
    result_b = run_case_audio(
        "(b) off-topic audio [test_h3.wav] — should refuse at grounding-check",
        OFF_TOPIC_WAV,
    )
    result_c = run_case_audio(
        "(c) silent audio — should fail cleanly at STT, no crash",
        SILENT_WAV,
    )

    # ── Groq rate-limit headers ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GROQ RATE-LIMIT HEADERS (from case a generation call)")
    print("=" * 70)
    if result_a and result_a.get("groq_ratelimit_headers"):
        for k, v in result_a["groq_ratelimit_headers"].items():
            print(f"  {k}: {v}")
    else:
        print("  (no headers captured — generation may not have run)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for label, result, expect_refused in [
        ("(a) grounded", result_a, False),
        ("(b) off-topic", result_b, True),
        ("(c) silent",   result_c, True),
    ]:
        if result is None:
            status = "FAIL — unhandled exception"
        elif result["refused"] == expect_refused:
            status = "PASS"
        else:
            status = f"UNEXPECTED — refused={result['refused']} (expected {expect_refused})"
        print(f"  {label:25s}  {status}")

    try:
        os.remove(SILENT_WAV)
    except OSError:
        pass


if __name__ == "__main__":
    main()

