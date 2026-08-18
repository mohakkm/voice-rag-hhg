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

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from datasets import Dataset


def load_hindi_passages(splits: list[str] | None = None) -> list[dict]:
    """
    Load the Hindi subset of ``ai4bharat/MSMARCO-XI``, flatten every passage in
    every example into a normalised record, deduplicate on passage text, and
    return the list.

    Note:
        Because ``datasets`` 5.x built-in parquet loader and pyarrow's table conversion
        hit a bug (``ArrowNotImplementedError: Nested data conversions not implemented for
        chunked array outputs``) when parsing large nested structs, we download the
        parquet files via ``hf_hub_download`` and read them batch-by-batch via pyarrow's
        ``iter_batches()`` which yields contiguous record batches.

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
            print(f"[ingest] Opening {filename} with PyArrow...")
            pf = pq.ParquetFile(file_path)
            
            print(f"[ingest] Reading {filename} in contiguous batches...")
            batches = pf.iter_batches(
                batch_size=20000,
                columns=[
                    'query_id',
                    'source_lang',
                    'target_lang',
                    'query_type',
                    'query',
                    'Eng_Query',
                    'passages'
                ]
            )
            
            rows_processed = 0
            for batch in batches:
                batch_dict = batch.to_pydict()
                num_rows = len(batch_dict["query_id"])
                
                for i in range(num_rows):
                    qid = batch_dict["query_id"][i]
                    passages_struct = batch_dict["passages"][i] or {}
                    
                    translated  = list(passages_struct.get("Translated_passages", []) or [])
                    english     = list(passages_struct.get("English_passages",    []) or [])
                    is_selected = list(passages_struct.get("is_selected",         []) or [])
                    
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
                                "source_lang":     batch_dict["source_lang"][i],
                                "target_lang":     batch_dict["target_lang"][i],
                                "query_id":        qid,
                                "query_type":      batch_dict["query_type"][i],
                                "query_hi":        batch_dict["query"][i],
                                "query_en":        batch_dict["Eng_Query"][i],
                                "english_passage": en_text,
                                "is_selected":     int(sel),
                                "split":           split,
                            },
                        })
                
                rows_processed += num_rows
                if rows_processed % 100000 == 0 or rows_processed == pf.metadata.num_rows:
                    print(f"[ingest]   Processed {rows_processed:,} / {pf.metadata.num_rows:,} rows...")
                    
        except Exception as e:
            print(f"[ingest] Error loading {split}: {e}")
            continue

    print(f"[ingest] Total passages after dedup: {len(passages):,}")
    return passages


if __name__ == "__main__":
    load_hindi_passages()
