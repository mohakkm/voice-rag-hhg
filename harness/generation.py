"""
Phase 3. Groq-backed answer generation.
Uses openai-compatible client pointed at Groq's API — model llama-3.3-70b-versatile.
"""
import openai
from config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL

_client: openai.OpenAI | None = None


def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    return _client


def generate_answer(query: str, retrieved_chunks: list[dict]) -> tuple[str, dict]:
    """
    Generate an answer grounded strictly in retrieved_chunks using Groq.

    The prompt explicitly instructs: answer ONLY from the provided context.
    If the context doesn't contain the answer, respond exactly with
    "insufficient context" — no guessing, no general knowledge fallback.

    Returns:
        (answer_text, headers) where headers is a dict of Groq rate-limit
        response headers (x-ratelimit-remaining-requests, etc.) for monitoring.
    """
    context_text = "\n\n".join(
        f"[Chunk {i+1}]: {c['payload'].get('text', '') if 'payload' in c else c.get('text', '')}"
        for i, c in enumerate(retrieved_chunks)
    )

    system_prompt = (
        "You are a precise question-answering assistant. "
        "Answer the user's question using ONLY the context chunks provided below. "
        "Do not use any external knowledge or make assumptions beyond what is in the context. "
        "If the context does not contain enough information to answer the question, "
        'respond with exactly the two words: "insufficient context" — nothing else.'
    )

    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"

    client = _get_client()
    response = client.chat.completions.with_raw_response.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=512,
    )

    # Extract rate-limit headers for monitoring against the 30 RPM free-tier cap.
    headers = {
        "x-ratelimit-limit-requests": response.headers.get("x-ratelimit-limit-requests"),
        "x-ratelimit-limit-tokens": response.headers.get("x-ratelimit-limit-tokens"),
        "x-ratelimit-remaining-requests": response.headers.get("x-ratelimit-remaining-requests"),
        "x-ratelimit-remaining-tokens": response.headers.get("x-ratelimit-remaining-tokens"),
        "x-ratelimit-reset-requests": response.headers.get("x-ratelimit-reset-requests"),
        "x-ratelimit-reset-tokens": response.headers.get("x-ratelimit-reset-tokens"),
    }

    parsed = response.parse()
    answer = parsed.choices[0].message.content.strip()
    return answer, headers
