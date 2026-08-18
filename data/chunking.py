"""
Phase 1. Three chunking strategies per ARCHITECTURE.md — implement all
three, don't collapse to one early. Each function takes list[dict] from
ingest.py and returns list[dict]: {"chunk_id": str, "text": str, "meta": {...}}
"""


def chunk_fixed_overlap(passages, chunk_size=256, overlap=32):
    """TODO(phase-1): baseline — fixed token/word window with overlap."""
    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("chunk_size must be strictly greater than overlap")

    for passage in passages:
        passage_id = passage.get("id", "")
        text = passage.get("text", "")
        if not text:
            continue

        meta_dict = passage.get("meta", {})
        language = meta_dict.get("target_lang", "hi")

        words = text.split()
        num_words = len(words)

        if num_words == 0:
            continue

        if num_words <= chunk_size:
            chunks.append({
                "chunk_id": f"{passage_id}_c0",
                "text": text,
                "meta": {
                    "passage_id": passage_id,
                    "language": language
                }
            })
            continue

        start_idx = 0
        chunk_idx = 0
        while start_idx < num_words:
            end_idx = min(start_idx + chunk_size, num_words)
            chunk_words = words[start_idx:end_idx]

            if not chunk_words:
                break

            chunk_text = " ".join(chunk_words)
            chunks.append({
                "chunk_id": f"{passage_id}_c{chunk_idx}",
                "text": chunk_text,
                "meta": {
                    "passage_id": passage_id,
                    "language": language
                }
            })

            if end_idx >= num_words:
                break

            start_idx += step
            chunk_idx += 1

    return chunks


def chunk_semantic(passages, embedding_fn):
    """TODO(phase-1): split on embedding-similarity breakpoints between sentences."""
    raise NotImplementedError


def chunk_metadata_aware(passages):
    """TODO(phase-1): use MSMARCO's native passage boundaries + language tag as-is."""
    raise NotImplementedError
