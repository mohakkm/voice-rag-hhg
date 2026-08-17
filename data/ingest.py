"""
Phase 1. Pull the Hindi subset of ai4bharat/MSMARCO-XI and normalize it
into a flat list of passage dicts for chunking.

Each record should end up as: {"id": str, "text": str, "meta": {...}}
meta should carry at least: source_lang, target_lang, original query_id
(needed later for strategy (c): metadata-aware chunking).
"""
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from huggingface_hub import hf_hub_download
from datasets import Dataset


def load_hindi_passages(splits: list[str] | None = None) -> list[dict]:
    """
    Load the Hindi subset of ``ai4bharat/MSMARCO-XI``, flatten every passage in
    every example into a normalised record, deduplicate on passage text, and
    return the list.

    Note:
        Because ``datasets`` 5.x built-in parquet loader hits a pyarrow bug
        (``ArrowNotImplementedError: Nested data conversions not implemented for
        chunked array outputs``) when parsing large nested structs, we download the
        parquet files via ``hf_hub_download``, read them with pandas, and load them
        into a Hugging Face ``Dataset`` using ``Dataset.from_pandas``.

    Args:
        splits: Dataset splits to load. Defaults to ``["train", "validation"]``.

    Returns:
        list of dicts with keys:
            "id"   – "<split>_q<query_id>_p<passage_idx>"
            "text" – translated Hindi passage text
            "meta" – {
                "source_lang"     : str,
                "target_lang"     : str,
                "query_id"        : int,
                "query_type"      : str,
                "query_hi"        : str,   # translated query
                "query_en"        : str,   # original English query
                "english_passage" : str,   # parallel English passage
                "is_selected"     : int,   # 1 = gold passage for this query
                "split"           : str,
            }
    """
    if splits is None:
        splits = ["train", "validation"]

    split_files = {
        "train": "train/hintrain.parquet",
        "validation": "validation/hinval.parquet"
    }

    passages: list[dict] = []
    seen_texts: set[str] = set()

    for split in splits:
        if split not in split_files:
            continue
        filename = split_files[split]
        print(f"[ingest] Downloading/loading '{filename}' using huggingface_hub...")
        try:
            file_path = hf_hub_download(
                repo_id="ai4bharat/MSMARCO-XI",
                filename=filename,
                repo_type="dataset"
            )
            print(f"[ingest] Reading {filename} with pandas...")
            df = pd.read_parquet(file_path)
            print(f"[ingest] Loading into Hugging Face Dataset ({len(df):,} rows)...")
            ds = Dataset.from_pandas(df)
        except Exception as e:
            print(f"[ingest] Error loading {split}: {e}")
            continue

        print(f"[ingest] Flattening and deduplicating split '{split}'...")
        for example in ds:
            qid = example["query_id"]
            passages_dict = example["passages"] or {}

            translated  = list(passages_dict.get("Translated_passages", []) or [])
            english     = list(passages_dict.get("English_passages",    []) or [])
            is_selected = list(passages_dict.get("is_selected",         []) or [])

            # Align lengths
            max_len = max(len(translated), len(english), len(is_selected))
            translated  += [""] * (max_len - len(translated))
            english     += [""] * (max_len - len(english))
            is_selected += [0] * (max_len - len(is_selected))

            for idx, (hi_text, en_text, sel) in enumerate(
                zip(translated, english, is_selected)
            ):
                hi_text = (hi_text or "").strip()
                if not hi_text or hi_text in seen_texts:
                    continue
                seen_texts.add(hi_text)

                passages.append({
                    "id": f"{split}_q{qid}_p{idx}",
                    "text": hi_text,
                    "meta": {
                        "source_lang":     example.get("source_lang", ""),
                        "target_lang":     example.get("target_lang", ""),
                        "query_id":        qid,
                        "query_type":      example.get("query_type", ""),
                        "query_hi":        example.get("query", ""),
                        "query_en":        example.get("Eng_Query", ""),
                        "english_passage": en_text,
                        "is_selected":     int(sel),
                        "split":           split,
                    },
                })

    print(f"[ingest] Total passages after dedup: {len(passages):,}")
    return passages


if __name__ == "__main__":
    load_hindi_passages()


