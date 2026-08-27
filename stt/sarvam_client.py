"""stt.sarvam_client
--------------------
Small wrapper around the `sarvamai` SDK to keep all Sarvam calls in one
module. The function `transcribe` is intentionally defensive: it returns a
structured dict containing `success`, `transcript`, `latency_ms` and `error`
instead of raising for common external faults (network, empty audio, etc.).

This keeps the rest of the codebase simple — callers can inspect the result
and decide whether to retry or escalate. We deliberately avoid complex
retry/backoff here; that is implemented in the orchestrator in Phase 3.
"""

# Note: in empirical tests the first Sarvam API call can be ~2-3x slower than
# steady-state (observed ~3.38s on a cold call vs ~1.2-1.8s afterwards). This
# is likely due to connection warm-up/SDK initialization and is important
# context when interpreting full-pipeline latency benchmarks.

import time
import os
from typing import Dict, Any

from config import SARVAM_API_KEY, LANGUAGE

STT_TIMEOUT_SECONDS = 15.0


def _is_timeout_error(exc: Exception) -> bool:
    """Recognize timeout exceptions across Sarvam/httpx SDK versions."""
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return "timeout" in name or "timeout" in message or "timed out" in message


def _timeout_result(start: float) -> Dict[str, Any]:
    return {
        "success": False,
        "transcript": None,
        "latency_ms": (time.time() - start) * 1000,
        "error": "stt_timeout",
    }


def transcribe(audio_path: str) -> Dict[str, Any]:
    """Transcribe an audio file using SarvamAI.

    Args:
        audio_path: Path to a local audio file (wav/mpeg/...) to send to Sarvam.

    Returns: a dict with the following keys:
        - success (bool): True when transcription succeeded and non-empty.
        - transcript (str|None): The transcribed text when success is True.
        - latency_ms (float): Round-trip latency in milliseconds for the API call.
        - error (str|None): Short error code or message when success is False.

    Behaviour:
        - Does not raise on network/API errors; returns success=False and an
          `error` string instead.
        - Detects missing files and empty/blank transcripts and reports those
          as `error` states instead of throwing.

    Note: callers who want retries should use an external retry wrapper.
    """

    start = time.time()
    if not os.path.exists(audio_path):
        return {"success": False, "transcript": None, "latency_ms": 0.0, "error": "file_not_found"}

    # Lazy import so the rest of the app can import this module even when the
    # sarvamai package is not installed (useful for static analysis / tests).
    try:
        from sarvamai import SarvamAI
    except Exception as e:  # pragma: no cover - environment may not have package
        latency_ms = (time.time() - start) * 1000
        return {"success": False, "transcript": None, "latency_ms": latency_ms, "error": f"missing_sarvamai:{e}"}

    try:
        client = SarvamAI(
            api_subscription_key=SARVAM_API_KEY,
            timeout=STT_TIMEOUT_SECONDS,
        )
        stt = client.speech_to_text

        # Quick path: some SDKs accept file_path/file/path keyword
        tried_exceptions = []
        response = None
        call_variants = [
            ("file_path", audio_path),
            ("file", audio_path),
            ("path", audio_path),
            ("audio_path", audio_path),
        ]

        for param_name, value in call_variants:
            try:
                response = stt.transcribe(**{param_name: value}, language=LANGUAGE, model="saaras:v3")
                break
            except TypeError as e:
                tried_exceptions.append(str(e))
            except Exception as e:
                if _is_timeout_error(e):
                    return _timeout_result(start)
                # Non-TypeError errors should not stop us from trying other
                # signatures; record and continue.
                tried_exceptions.append("other")

        # If the quick path didn't work, try to introspect the signature and
        # adapt to different SDK styles (keyword, file object, raw bytes, or
        # positional).
        if response is None:
            import inspect

            func = getattr(stt, "transcribe", None)
            if func is None:
                raise RuntimeError("Sarvam client has no 'transcribe' method on speech_to_text")

            try:
                sig = inspect.signature(func)
                params = [p.name for p in sig.parameters.values()]
            except Exception:
                params = []

            def build_kwargs(param_names):
                kw = {}
                if any(n in param_names for n in ("language", "lang", "language_code")):
                    if "language" in param_names:
                        kw["language"] = LANGUAGE
                    elif "lang" in param_names:
                        kw["lang"] = LANGUAGE
                    elif "language_code" in param_names:
                        kw["language_code"] = LANGUAGE
                if any(n in param_names for n in ("model", "model_name", "modelId", "model_id")):
                    if "model" in param_names:
                        kw["model"] = "saaras:v3"
                    elif "model_name" in param_names:
                        kw["model_name"] = "saaras:v3"
                    elif "model_id" in param_names:
                        kw["model_id"] = "saaras:v3"
                return kw

            file_param_candidates = ["file_path", "file", "path", "audio", "audio_path", "input"]
            for candidate in file_param_candidates:
                if candidate in params:
                    try:
                        kwargs = build_kwargs(params)
                        kwargs[candidate] = audio_path
                        response = func(**kwargs)
                        break
                    except Exception as e:
                        if _is_timeout_error(e):
                            return _timeout_result(start)
                        tried_exceptions.append(f"kw[{candidate}]: {e}")

            if response is None:
                # Try file object and raw bytes
                try:
                    with open(audio_path, "rb") as fh:
                        try:
                            kwargs = build_kwargs(params)
                            kwargs["file"] = fh
                            response = func(**kwargs)
                        except Exception as e:
                            tried_exceptions.append(f"fileobj: {e}")
                            fh.seek(0)
                            data = fh.read()
                            try:
                                kwargs = build_kwargs(params)
                                if "audio" in params:
                                    kwargs["audio"] = data
                                    response = func(**kwargs)
                                else:
                                    response = func(data)
                            except Exception as e2:
                                tried_exceptions.append(f"bytes: {e2}")
                except Exception as e:
                    tried_exceptions.append(f"open_file: {e}")

            if response is None:
                # Positional fallbacks
                try:
                    try:
                        response = func(audio_path, LANGUAGE, "saaras:v3")
                    except Exception:
                        response = func(audio_path)
                except Exception as e:
                    tried_exceptions.append(f"positional: {e}")

            if response is None:
                raise RuntimeError("Could not call Sarvam transcribe; attempts: " + " | ".join(tried_exceptions))

        # response should now be set; measure latency and extract transcript
        latency_ms = (time.time() - start) * 1000

        transcript = None
        if hasattr(response, "transcript"):
            transcript = response.transcript
        elif isinstance(response, dict):
            transcript = response.get("transcript") or response.get("text") or response.get("transcription")
        else:
            transcript = str(response)

        if not transcript or not str(transcript).strip():
            return {"success": False, "transcript": None, "latency_ms": latency_ms, "error": "empty_transcript"}

        return {"success": True, "transcript": str(transcript).strip(), "latency_ms": latency_ms, "error": None}

    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        if _is_timeout_error(e):
            return _timeout_result(start)
        return {"success": False, "transcript": None, "latency_ms": latency_ms, "error": str(e)}
