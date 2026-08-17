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

from datasets import load_dataset


def load_hindi_passages(splits: list[str] | None = None) -> list[dict]:
    """
    Load the Hindi subset of ``ai4bharat/MSMARCO-XI`` using the ``datasets`` library,
    flatten every passage in every example into a normalised record, deduplicate
    on passage text, and return the list.

    Note:
        Because ``datasets`` 5.x deprecated remote code execution (needed to resolve
        the config name ``"hi"`` in the dataset's custom loading script), we specify
        the exact Hindi parquet files directly via ``data_files``.

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

    data_files = {}
    if "train" in splits:
        data_files["train"] = "train/hintrain.parquet"
    if "validation" in splits:
        data_files["validation"] = "validation/hinval.parquet"

    print(f"[ingest] Loading dataset splits {splits} using datasets library...")
    ds = load_dataset("ai4bharat/MSMARCO-XI", data_files=data_files)

    passages: list[dict] = []
    seen_texts: set[str] = set()

    for split in splits:
        if split not in ds:
            continue
        print(f"[ingest] Flattening and deduplicating split '{split}'...")
        split_data = ds[split]
        for example in split_data:
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

