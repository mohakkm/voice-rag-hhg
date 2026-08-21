"""Quick test runner for the Sarvam STT client.

Usage:
    python scripts/test_sarvam_transcribe.py

Behaviour:
- Looks for `data/test_h1.wav`, `data/test_h2.wav`, `data/test_h3.wav` and
  transcribes them if present.
- If they are missing and both `sounddevice` and `soundfile` are installed, it
  will offer to record three short (5-8s) Hindi clips from the default
  microphone and save them under `data/`.
- Prints transcript + latency for each clip.

Note: Ensure `SARVAM_API_KEY` is set in your environment (.env) before
running. This script is intentionally simple; orchestrator handles retries.
"""

import os
import time
from pathlib import Path

from config import validate
from stt.sarvam_client import transcribe


SAMPLES = [Path("data/test_h1.wav"), Path("data/test_h2.wav"), Path("data/test_h3.wav")]


def try_record(path: Path, duration: int = 6, samplerate: int = 16000):
    try:
        import sounddevice as sd
        import soundfile as sf
    except Exception as e:
        print(f"Recording unavailable (missing packages): {e}")
        return False

    print(f"Recording {duration}s to {path} — speak now (Hindi) ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    sf.write(str(path), recording, samplerate)
    print(f"Saved {path}")
    return True


def main():
    try:
        validate()
    except Exception as e:
        print(f"Config validation failed: {e}")
        print("Set SARVAM_API_KEY in .env or environment and re-run.")
        return

    # Ensure samples exist: try to record if absent and dependencies present.
    for p in SAMPLES:
        if not p.exists():
            print(f"Missing sample: {p}")

    missing = [p for p in SAMPLES if not p.exists()]
    if missing:
        print("Attempting to record missing samples if recording deps are present...")
        for p in missing:
            ok = try_record(p)
            if not ok:
                print("Please place three short Hindi WAV files at:")
                for sp in SAMPLES:
                    print(f"  - {sp}")
                print("Aborting tests.")
                return

    # Run transcriptions
    results = []
    for p in SAMPLES:
        print(f"Transcribing {p} ...")
        start = time.time()
        out = transcribe(str(p))
        elapsed = (time.time() - start) * 1000
        # Note: transcribe already returns latency_ms (round-trip); include both.
        results.append((p, out, elapsed))

    print("\nResults:")
    for p, out, elapsed in results:
        if out.get("success"):
            print(f"{p}: OK — latency_api={out.get('latency_ms'):.1f}ms total_call={elapsed:.1f}ms")
            print(f"  Transcript: {out.get('transcript')}\n")
        else:
            print(f"{p}: FAILED — error={out.get('error')} latency_api={out.get('latency_ms'):.1f}ms total_call={elapsed:.1f}ms")


if __name__ == "__main__":
    main()
