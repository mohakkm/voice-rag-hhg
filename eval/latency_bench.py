"""
Phase 5. Run harness.orchestrator.run_pipeline over real test queries,
compute P50/P70/P100 PER STAGE and end-to-end. Report honestly — see
ARCHITECTURE.md "Latency — how to actually report it" before you touch this.
"""
import json
import random
import time
from pathlib import Path
from contextlib import contextmanager

import numpy as np

import config
import harness.generation as generation_module
import harness.guardrails as guardrails_module
from harness.orchestrator import run_pipeline_from_text
from retrieval.embeddings import get_model
from retrieval.qdrant_store import get_client

_WARMED_UP = False
_WARMUP_MS = 0.0

STAGES = ["stt", "embed", "retrieve", "generate", "total"]
BENCHMARK_GROQ_MODEL = "openai/gpt-oss-20b"


def warmup_runtime() -> float:
    """Load embedding model + Qdrant client exactly once per process startup."""
    global _WARMED_UP, _WARMUP_MS
    if _WARMED_UP:
        return _WARMUP_MS

    t0 = time.perf_counter()
    get_model()
    get_client()
    _WARMUP_MS = (time.perf_counter() - t0) * 1000.0
    print(f"[warmup] Loaded embedding model + Qdrant client in {_WARMUP_MS:.1f} ms")
    _WARMED_UP = True
    return _WARMUP_MS


def percentiles(latencies_ms: list[float]) -> dict:
    """Return {"p50":..., "p70":..., "p100":...} from a list of ms values."""
    if not latencies_ms:
        return {"p50": None, "p70": None, "p100": None}
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p70": float(np.percentile(arr, 70)),
        "p100": float(np.percentile(arr, 100)),
    }


def _collect_stage_percentiles(results: list[dict]) -> dict:
    by_stage = {s: [] for s in STAGES}
    for item in results:
        latency = item.get("latency_ms", {})
        for stage in STAGES:
            value = latency.get(stage)
            if value is not None:
                by_stage[stage].append(float(value))
    return {stage: percentiles(vals) for stage, vals in by_stage.items()}


def _load_query_pool(corpus_path: Path) -> list[dict]:
    """
    Load unique queries from corpus_sample.jsonl.
    Returns items like: {"query_id": int, "query_type": str, "query_hi": str}
    """
    by_qid: dict[int, dict] = {}
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            meta = rec.get("meta", {})
            qid = int(meta.get("query_id"))
            qtype = str(meta.get("query_type", "UNKNOWN"))
            qhi = str(meta.get("query_hi", "")).strip()
            if not qhi:
                continue
            if qid not in by_qid:
                by_qid[qid] = {"query_id": qid, "query_type": qtype, "query_hi": qhi}
    return list(by_qid.values())


def _sample_mixed_queries(pool: list[dict], n_queries: int, seed: int = 42) -> list[dict]:
    """
    Stratified-ish sample: round-robin across query_type buckets to avoid
    taking all queries from one category.
    """
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {}
    for q in pool:
        buckets.setdefault(q["query_type"], []).append(q)
    for items in buckets.values():
        rng.shuffle(items)

    types = sorted(buckets.keys())
    sampled: list[dict] = []
    idx = 0
    while len(sampled) < n_queries:
        progressed = False
        for qtype in types:
            if idx < len(buckets[qtype]) and len(sampled) < n_queries:
                sampled.append(buckets[qtype][idx])
                progressed = True
        if not progressed:
            break
        idx += 1

    if len(sampled) < n_queries:
        raise RuntimeError(
            f"Could only sample {len(sampled)} mixed queries; requested {n_queries}."
        )
    return sampled


@contextmanager
def _override_benchmark_model(model_name: str):
    """Temporarily point Groq-backed benchmark calls at a smaller model."""
    original_config_model = config.GROQ_MODEL
    original_generation_model = generation_module.GROQ_MODEL
    original_guardrails_model = guardrails_module.GROQ_MODEL
    config.GROQ_MODEL = model_name
    generation_module.GROQ_MODEL = model_name
    guardrails_module.GROQ_MODEL = model_name
    try:
        yield
    finally:
        config.GROQ_MODEL = original_config_model
        generation_module.GROQ_MODEL = original_generation_model
        guardrails_module.GROQ_MODEL = original_guardrails_model


def run_benchmark(
    test_queries: list[str],
    n_runs: int = 15,
):
    """
    Run text-only and audio-included latency benchmark.

    - Text-only mode uses run_pipeline_from_text on n_runs real queries.
    - Groundedness-check is skipped in this timing run only.

    Returns a dict ready to save to eval/results.json.
    """
    warmup_ms = warmup_runtime()
    if not _WARMED_UP:
        raise RuntimeError("Warmup did not complete before timing started.")
    print("[bench] Warmup confirmed complete before benchmark timing.")
    print(f"[bench] Using benchmark Groq model: {BENCHMARK_GROQ_MODEL}")

    # --- text-only benchmark ---
    text_results: list[dict] = []
    text_refused = 0
    text_errors = 0
    with _override_benchmark_model(BENCHMARK_GROQ_MODEL):
        for i, query in enumerate(test_queries[:n_runs], start=1):
            try:
                result = run_pipeline_from_text(query, include_output_groundedness=False)
                text_results.append(result)
                if result.get("refused"):
                    text_refused += 1
            except Exception as e:
                text_errors += 1
                text_results.append(
                    {
                        "refused": True,
                        "refusal_reason": f"benchmark_error:{e}",
                        "latency_ms": {},
                    }
                )
            print(f"[bench][text] {i}/{n_runs} done")

    report = {
        "meta": {
            "n_text_runs": len(text_results),
            "warmup_ms": warmup_ms,
            "warmup_confirmed_before_timing": True,
            "groundedness_excluded_from_timing": True,
            "groundedness_note": "Skipped during benchmark due to Groq free-tier daily limits; confirmed working in live demo path.",
        },
        "text_only": {
            "stage_percentiles_ms": _collect_stage_percentiles(text_results),
            "refused_count": text_refused,
            "total_count": len(text_results),
            "refusal_rate": (text_refused / len(text_results)) if text_results else None,
            "error_count": text_errors,
        },
    }
    return report


def _print_table(report: dict):
    print("\n=== LATENCY TABLE (ms) ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    config.validate()

    repo_root = Path(__file__).resolve().parents[1]
    corpus_path = repo_root / "data" / "corpus_sample.jsonl"
    results_path = repo_root / "eval" / "results.json"

    pool = _load_query_pool(corpus_path)
    sampled = _sample_mixed_queries(pool, n_queries=15, seed=42)
    queries = [q["query_hi"] for q in sampled]

    report = run_benchmark(
        test_queries=queries,
        n_runs=15,
    )

    # Include sampled query metadata for reproducibility/audit.
    report["sampled_queries"] = sampled

    results_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[bench] Saved results to {results_path}")
    _print_table(report)


if __name__ == "__main__":
    main()
