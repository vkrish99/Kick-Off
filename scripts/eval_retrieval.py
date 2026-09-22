"""
Retrieval-quality benchmark for the Pinecone RAG pipeline.

There's no hand-labeled query set for football player lookups, so this
builds a "self-retrieval" golden set from whatever profiles are already
indexed: for each sampled player it generates a natural-language query
from their own name/team, then checks whether that player comes back in
the top-k results. If the pipeline can't retrieve a player from a query
built straight from their own profile, real user queries will do worse —
so this is a lower-bound proxy for retrieval quality, not a full
relevance judgment.

Reports Hit Rate@k and Mean Reciprocal Rank (MRR).

Usage:
    python scripts/eval_retrieval.py --sample-size 30 --k 3
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.vector_store import vector_store_service  # noqa: E402


def sample_indexed_profiles(sample_size: int) -> list[dict]:
    """Pull a scattered random sample of already-ingested profiles from Pinecone.

    Pinecone has no native "random sample" API, so this fires random unit
    vectors at the index and keeps the top match of each — a cheap way to
    get a spread of real, previously-ingested metadata to build queries from.
    """
    import random

    index = vector_store_service._get_index()
    stats = index.describe_index_stats()
    total = stats.get("total_vector_count", 0)
    if total == 0:
        raise SystemExit(
            "Pinecone index is empty — run `python scripts/load_datasets.py` first."
        )

    dim = vector_store_service._get_embedder().get_sentence_embedding_dimension()
    seen: dict[str, dict] = {}
    attempts = 0
    max_attempts = sample_size * 5
    while len(seen) < min(sample_size, total) and attempts < max_attempts:
        vec = [random.gauss(0, 1) for _ in range(dim)]
        norm = sum(v * v for v in vec) ** 0.5
        vec = [v / norm for v in vec]
        results = index.query(vector=vec, top_k=1, include_metadata=True)
        for match in results.get("matches", []):
            seen[match["id"]] = match.get("metadata", {})
        attempts += 1

    return [{"player_id": pid, **meta} for pid, meta in seen.items()]


def build_query(profile: dict) -> str:
    name = profile.get("name", "")
    team = profile.get("team", "")
    return f"Tell me about {name} who plays for {team}"


def rank_of_expected(index, embedding: list[float], k: int, expected_id: str) -> Optional[int]:
    """1-based rank of `expected_id` in the top-k results, or None if absent."""
    results = index.query(vector=embedding, top_k=k, include_metadata=True)
    ranked_ids = [m["id"] for m in results.get("matches", [])]
    if expected_id in ranked_ids:
        return ranked_ids.index(expected_id) + 1
    return None


def compute_metrics(ranks: list[Optional[int]]) -> dict:
    """Hit Rate@k and MRR from a list of per-query ranks (None = miss)."""
    total = len(ranks)
    hits = sum(1 for r in ranks if r is not None)
    hit_rate = hits / total if total else 0.0
    mrr = sum((1.0 / r) if r else 0.0 for r in ranks) / total if total else 0.0
    return {"hit_rate": hit_rate, "mrr": mrr, "hits": hits, "total": total}


def evaluate(sample_size: int, k: int) -> dict:
    profiles = sample_indexed_profiles(sample_size)
    index = vector_store_service._get_index()

    print(f"Evaluating retrieval over {len(profiles)} sampled profiles (k={k})\n")
    print(f"{'Name':<25} {'Hit':<5} {'Rank':<6} Query")
    print("-" * 100)

    ranks: list[Optional[int]] = []
    for profile in profiles:
        query = build_query(profile)
        embedding = vector_store_service._embed(query)
        rank = rank_of_expected(index, embedding, k, profile["player_id"])
        ranks.append(rank)
        name = profile.get("name", "?")
        print(f"{name:<25} {'yes' if rank else 'no':<5} {rank or '-':<6} {query}")

    metrics = compute_metrics(ranks)
    print("\n" + "=" * 40)
    print(f"Hit Rate@{k}: {metrics['hit_rate']:.1%}  ({metrics['hits']}/{metrics['total']})")
    print(f"MRR:        {metrics['mrr']:.3f}")
    print("=" * 40)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Pinecone retrieval quality.")
    parser.add_argument("--sample-size", type=int, default=30, help="Number of profiles to sample")
    parser.add_argument("--k", type=int, default=3, help="top-k for the retrieval check")
    args = parser.parse_args()
    evaluate(args.sample_size, args.k)


if __name__ == "__main__":
    main()
