"""
Tests for the retrieval-quality metrics in scripts/eval_retrieval.py.
No live Pinecone calls — just the Hit Rate@k / MRR computation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from eval_retrieval import compute_metrics, build_query


class TestComputeMetrics:
    def test_all_hits_at_rank_one(self):
        metrics = compute_metrics([1, 1, 1])
        assert metrics == {"hit_rate": 1.0, "mrr": 1.0, "hits": 3, "total": 3}

    def test_all_misses(self):
        metrics = compute_metrics([None, None])
        assert metrics == {"hit_rate": 0.0, "mrr": 0.0, "hits": 0, "total": 2}

    def test_mixed_ranks(self):
        # hits at rank 1 and rank 2, one miss
        metrics = compute_metrics([1, 2, None])
        assert metrics["hits"] == 2
        assert metrics["total"] == 3
        assert metrics["hit_rate"] == 2 / 3
        assert metrics["mrr"] == (1.0 + 0.5 + 0.0) / 3

    def test_empty_input(self):
        metrics = compute_metrics([])
        assert metrics == {"hit_rate": 0.0, "mrr": 0.0, "hits": 0, "total": 0}


class TestBuildQuery:
    def test_includes_name_and_team(self):
        query = build_query({"name": "Erling Haaland", "team": "Manchester City"})
        assert "Erling Haaland" in query
        assert "Manchester City" in query

    def test_handles_missing_fields(self):
        query = build_query({})
        assert isinstance(query, str)
