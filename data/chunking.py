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


import re
import numpy as np

SEMANTIC_THRESHOLD = 0.75  # tunable constant
# Post-merge pass can push chunks slightly past chunk_size=256 (observed max: 288) to prioritize merging tiny chunks.
MIN_CHUNK_WORDS = 20      # tunable constant


def chunk_semantic(
    passages,
    embedding_fn,
    threshold=SEMANTIC_THRESHOLD,
    chunk_size=256,
    min_chunk_words=MIN_CHUNK_WORDS,
):
    """
    Split on embedding-similarity breakpoints between sentences.
    
    Args:
        passages: list of dicts with keys 'id', 'text', 'meta'
        embedding_fn: callable that takes list[str] -> numpy array of embeddings (shape: N, D)
        threshold: cosine similarity threshold below which to split
        chunk_size: hard word limit per chunk
        min_chunk_words: minimum word count to keep a chunk from being merged
    """
    chunks = []
    # Splitting on Devanagari danda (।), period, question mark, exclamation, or newlines.
    # \s+ (not \s*) is required: a zero-width split lets the boundary fall *inside* a
    # whitespace-delimited token (e.g. abbreviation "एम.एस.एफ." has no spaces between the
    # periods), which fragments one token into several "sentences". Rejoining those with
    # " ".join() then inserts spaces that never existed in the source text, so the chunk's
    # word list no longer matches a contiguous slice of the document's word list downstream.
    sentence_split_regex = re.compile(r'(?<=[।\.!\?])\s+|\n+')

    total_merged_count = 0

    for passage in passages:
        passage_id = passage.get("id", "")
        text = passage.get("text", "")
        if not text:
            continue

        meta_dict = passage.get("meta", {})
        language = meta_dict.get("target_lang", "hi")

        raw_sentences = sentence_split_regex.split(text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        if not sentences:
            continue

        # Pre-split any sentence that exceeds the hard word limit to enforce chunk_size
        split_sentences = []
        for s in sentences:
            words_s = s.split()
            if len(words_s) <= chunk_size:
                split_sentences.append(s)
            else:
                for k in range(0, len(words_s), chunk_size):
                    segment = " ".join(words_s[k : k + chunk_size])
                    if segment:
                        split_sentences.append(segment)
        sentences = split_sentences

        temp_chunks = []

        if len(sentences) == 1:
            temp_chunks.append({
                "chunk_id": f"{passage_id}_c0",
                "text": sentences[0],
                "meta": {
                    "passage_id": passage_id,
                    "language": language
                }
            })
        else:
            # Batch embed all sentences of this document
            vectors = embedding_fn(sentences)

            # Compute cosine similarity between consecutive sentence embeddings
            # Since vectors are L2-normalized, cosine similarity is just the dot product
            similarities = np.sum(vectors[:-1] * vectors[1:], axis=1)

            current_chunk_sentences = [sentences[0]]
            current_chunk_word_count = len(sentences[0].split())
            chunk_idx = 0

            for j in range(1, len(sentences)):
                is_boundary = similarities[j - 1] < threshold
                sentence_word_count = len(sentences[j].split())

                if is_boundary or (current_chunk_word_count + sentence_word_count > chunk_size):
                    chunk_text = " ".join(current_chunk_sentences)
                    temp_chunks.append({
                        "chunk_id": f"{passage_id}_c{chunk_idx}",
                        "text": chunk_text,
                        "meta": {
                            "passage_id": passage_id,
                            "language": language,
                            "split_reason": "semantic" if is_boundary else "length",
                            "similarity_before_split": float(similarities[j - 1]) if is_boundary else None
                        }
                    })
                    chunk_idx += 1
                    current_chunk_sentences = [sentences[j]]
                    current_chunk_word_count = sentence_word_count
                else:
                    current_chunk_sentences.append(sentences[j])
                    current_chunk_word_count += sentence_word_count

            if current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                temp_chunks.append({
                    "chunk_id": f"{passage_id}_c{chunk_idx}",
                    "text": chunk_text,
                    "meta": {
                        "passage_id": passage_id,
                        "language": language,
                        "split_reason": "end"
                    }
                })

        # Post-process: Merge chunks smaller than min_chunk_words
        if len(temp_chunks) <= 1:
            merged = temp_chunks
        else:
            merged = []
            i = 0
            n = len(temp_chunks)
            while i < n:
                current_chunk = temp_chunks[i]
                current_wc = len(current_chunk["text"].split())

                if current_wc < min_chunk_words:
                    if i + 1 < n:
                        # Merge text into the following chunk
                        temp_chunks[i + 1]["text"] = current_chunk["text"] + " " + temp_chunks[i + 1]["text"]
                        # Preserve details about the merge
                        temp_chunks[i + 1]["meta"]["merged_prev_chunk"] = True
                    else:
                        # Last chunk: merge into preceding chunk if available
                        if merged:
                            merged[-1]["text"] = merged[-1]["text"] + " " + current_chunk["text"]
                            merged[-1]["meta"]["merged_next_chunk"] = True
                        else:
                            merged.append(current_chunk)
                    i += 1
                else:
                    merged.append(current_chunk)
                    i += 1

        total_merged_count += (len(temp_chunks) - len(merged))

        # Re-assign consecutive chunk IDs and add to global list
        for idx, chunk in enumerate(merged):
            chunk["chunk_id"] = f"{passage_id}_c{idx}"
            chunks.append(chunk)

    print(f"[chunk_semantic] Total chunks merged due to size < {min_chunk_words}: {total_merged_count:,}", flush=True)
    return chunks


def chunk_metadata_aware(
    passages, min_chunk_words=MIN_CHUNK_WORDS, chunk_size=256, overlap=32, max_passage_words=400
):
    """
    Use MSMARCO's native passage boundaries directly as chunk boundaries.
    Enforces a minimum chunk size of min_chunk_words and a maximum chunk size of max_passage_words.
    """
    chunks = []
    total_docs_merged = 0
    total_docs_split = 0

    for doc in passages:
        doc_id = doc.get("id", "")
        text = doc.get("text", "")
        meta_dict = doc.get("meta", {})
        query_id = meta_dict.get("query_id")
        language = "hin_Deva"

        raw_passages = text.split("\n\n")
        passages_list = []
        for idx, p_text in enumerate(raw_passages):
            p_text = p_text.strip()
            if p_text:
                passages_list.append({
                    "text": p_text,
                    "passage_index": idx
                })

        if not passages_list:
            continue

        # 1. Merge passages < min_chunk_words
        doc_merged = False
        if len(passages_list) <= 1:
            merged_passages = passages_list
        else:
            merged_passages = []
            i = 0
            n = len(passages_list)
            while i < n:
                current = passages_list[i]
                current_wc = len(current["text"].split())
                if current_wc < min_chunk_words:
                    if i + 1 < n:
                        passages_list[i + 1]["text"] = (
                            current["text"] + " " + passages_list[i + 1]["text"]
                        )
                        doc_merged = True
                    else:
                        if merged_passages:
                            merged_passages[-1]["text"] = (
                                merged_passages[-1]["text"] + " " + current["text"]
                            )
                            doc_merged = True
                        else:
                            merged_passages.append(current)
                    i += 1
                else:
                    merged_passages.append(current)
                    i += 1

        if doc_merged:
            total_docs_merged += 1

        # 2. Sub-split passages > max_passage_words
        doc_split = False
        final_segments = []
        for p in merged_passages:
            p_text = p["text"]
            words = p_text.split()
            p_wc = len(words)

            if p_wc > max_passage_words:
                doc_split = True
                step = chunk_size - overlap
                start_idx = 0
                sub_idx = 0
                while start_idx < p_wc:
                    end_idx = min(start_idx + chunk_size, p_wc)
                    sub_words = words[start_idx:end_idx]
                    sub_text = " ".join(sub_words)
                    final_segments.append({
                        "text": sub_text,
                        "passage_index": p["passage_index"],
                        "sub_index": sub_idx,
                    })
                    if end_idx >= p_wc:
                        break
                    start_idx += step
                    sub_idx += 1
            else:
                final_segments.append({
                    "text": p_text,
                    "passage_index": p["passage_index"],
                    "sub_index": None,
                })

        if doc_split:
            total_docs_split += 1

        # Re-assign consecutive chunk IDs and add metadata
        for idx, seg in enumerate(final_segments):
            chunk_id = f"{doc_id}_c{idx}"
            meta = {
                "query_id": query_id,
                "passage_index_within_doc": seg["passage_index"],
                "language": language,
            }
            if seg["sub_index"] is not None:
                meta["sub_index"] = seg["sub_index"]

            chunks.append({
                "chunk_id": chunk_id,
                "text": seg["text"],
                "meta": meta
            })

    print(
        f"[chunk_metadata_aware] Total docs affected by merge (<{min_chunk_words} words): {total_docs_merged:,}",
        flush=True,
    )
    print(
        f"[chunk_metadata_aware] Total docs affected by split (>{max_passage_words} words): {total_docs_split:,}",
        flush=True,
    )
    return chunks
