"""
Phase 2. Sarvam Saaras v3 STT wrapper. Keep this the ONLY place that
talks to Sarvam — orchestrator.py should never import sarvamai directly.
"""
from config import SARVAM_API_KEY, LANGUAGE


def transcribe(audio_path: str) -> str:
    """
    TODO(phase-2):
    from sarvamai import SarvamAI
    client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
    response = client.speech_to_text.transcribe(file_path=audio_path, language=LANGUAGE, model="saaras:v3")
    return response.transcript

    Handle: empty/silent audio, network timeout, 429 rate limit (retry w/ backoff — see harness/orchestrator.py).
    """
    raise NotImplementedError
