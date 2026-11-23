"""
Tests for the pre-deployment fixes: API-key auth, CORS, router parsing, RAG
grounding, vector IDs and keeping blocking work off the event loop.
"""
import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.agents.orchestrator import Orchestrator, Route, parse_route
from app.main import app
from app.services.rag import RAGService
from app.services.vector_store import VectorStoreService

client = TestClient(app)

TELEMETRY = {
    "data": [
        {
            "player_id": "p1",
            "speed": 22.5,
            "distance_covered": 10.2,
            "heart_rate": 150,
            "position_x": 10.0,
            "position_y": 20.0,
        }
    ]
}


# ---------------------------------------------------------------------------
# API-key auth on write endpoints
# ---------------------------------------------------------------------------
class TestApiKeyAuth:
    @pytest.fixture(autouse=True)
    def _api_key(self):
        with patch(
            "app.core.security.get_settings",
            return_value=SimpleNamespace(API_KEY="s3cret", MAX_QUERY_LENGTH=2000),
        ):
            yield

    def test_telemetry_without_key_is_rejected(self):
        response = client.post("/api/v1/ingest/telemetry", json=TELEMETRY)
        assert response.status_code == 401

    def test_telemetry_with_wrong_key_is_rejected(self):
        response = client.post(
            "/api/v1/ingest/telemetry", json=TELEMETRY, headers={"X-API-Key": "nope"}
        )
        assert response.status_code == 401

    @patch("app.services.cache.redis_service.enqueue_task", new_callable=AsyncMock)
    def test_telemetry_with_correct_key_is_accepted(self, _enqueue):
        response = client.post(
            "/api/v1/ingest/telemetry", json=TELEMETRY, headers={"X-API-Key": "s3cret"}
        )
        assert response.status_code == 202

    @patch("app.api.routes.rag_service")
    def test_profiles_without_key_is_rejected_before_touching_pinecone(self, mock_rag):
        response = client.post("/api/v1/ingest/profiles", json=[])
        assert response.status_code == 401
        mock_rag.ingest_profiles.assert_not_called()

    def test_read_endpoints_stay_public(self):
        assert client.get("/api/v1/config").status_code == 200


def test_wildcard_cors_does_not_allow_credentials():
    response = client.get("/health", headers={"Origin": "https://example.com"})
    assert response.headers.get("access-control-allow-origin") == "*"
    assert "access-control-allow-credentials" not in response.headers


# ---------------------------------------------------------------------------
# Router reply parsing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "reply, expected",
    [
        ("TACTICAL", Route.TACTICAL),
        ("  rag\n", Route.RAG),
        ("`SENTIMENT`", Route.SENTIMENT),
        ("FORMATION.", Route.FORMATION),
        ("NOT TACTICAL", Route.GENERAL),
        ("I think this is TACTICAL", Route.GENERAL),
        ("", Route.GENERAL),
    ],
)
def test_parse_route_needs_an_exact_token(reply, expected):
    assert parse_route(reply) is expected


class TestRouteShortcuts:
    async def test_filters_go_to_rag_without_router_call(self):
        orch = Orchestrator()
        orch._classify_route = AsyncMock()
        with patch("app.agents.orchestrator.rag_service") as mock_rag, \
             patch.object(orch, "_get_rag_chain", return_value=MagicMock()):
            mock_rag.retrieve_context = AsyncMock(return_value=[])
            _, route, _ = await orch.process_query("strikers?", filters={"position": "Striker"})
        orch._classify_route.assert_not_called()
        assert route == "RAG_NO_RESULTS"

    async def test_news_only_goes_to_sentiment_without_router_call(self):
        orch = Orchestrator()
        orch._classify_route = AsyncMock()
        with patch("app.agents.orchestrator.sentiment_agent") as mock_sentiment:
            mock_sentiment.analyze_sentiment = AsyncMock(return_value="Morale is high.")
            insight, route, _ = await orch.process_query("Morale?", news_context="Won 3-0.")
        orch._classify_route.assert_not_called()
        assert (insight, route) == ("Morale is high.", "SENTIMENT")


# ---------------------------------------------------------------------------
# RAG grounding
# ---------------------------------------------------------------------------
class TestRagAnswers:
    async def test_empty_retrieval_skips_the_llm(self):
        orch = Orchestrator()
        orch._classify_route = AsyncMock(return_value="RAG")
        chain = MagicMock()
        chain.ainvoke = AsyncMock()
        with patch("app.agents.orchestrator.rag_service") as mock_rag, \
             patch.object(orch, "_get_rag_chain", return_value=chain):
            mock_rag.retrieve_context = AsyncMock(return_value=[])
            insight, route, sources = await orch.process_query("Best young keeper?")
        chain.ainvoke.assert_not_called()
        assert route == "RAG_NO_RESULTS"
        assert "No matching players" in insight
        assert sources == []

    async def test_answer_returns_sources(self):
        orch = Orchestrator()
        orch._classify_route = AsyncMock(return_value="RAG")
        chain = MagicMock()
        chain.ainvoke = AsyncMock(return_value="Haaland is the best finisher.")
        doc = Document(
            page_content="Player: Erling Haaland",
            metadata={"player_id": 418560, "name": "Erling Haaland", "score": 0.91},
        )
        with patch("app.agents.orchestrator.rag_service") as mock_rag, \
             patch.object(orch, "_get_rag_chain", return_value=chain):
            mock_rag.retrieve_context = AsyncMock(return_value=[doc])
            _, route, sources = await orch.process_query("Best finisher?")
        assert route == "RAG"
        assert sources == [{"player_id": "418560", "name": "Erling Haaland", "score": 0.91}]

    async def test_missing_groq_key_fails_before_querying_pinecone(self):
        orch = Orchestrator()
        with patch("app.agents.orchestrator.settings") as mock_settings, \
             patch("app.agents.orchestrator.rag_service") as mock_rag:
            mock_settings.GROQ_API_KEY = None
            mock_rag.retrieve_context = AsyncMock()
            with pytest.raises(Exception) as exc_info:
                await orch.process_query("x", filters={"team": "Arsenal"})
        assert exc_info.value.status_code == 503
        mock_rag.retrieve_context.assert_not_called()

    def test_analyze_response_includes_sources(self):
        with patch(
            "app.agents.orchestrator.orchestrator.process_query",
            new_callable=AsyncMock,
            return_value=("Answer", "RAG", [{"player_id": "1", "name": "A", "score": 0.5}]),
        ):
            response = client.post("/api/v1/analyze", json={"query": "Who is A?"})
        assert response.status_code == 200
        assert response.json()["sources"] == [{"player_id": "1", "name": "A", "score": 0.5}]


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
class TestVectorStore:
    def _service(self):
        svc = VectorStoreService()
        svc._embed_batch = MagicMock(side_effect=lambda texts: [[0.1] for _ in texts])
        svc._get_index = MagicMock(return_value=MagicMock())
        return svc

    def test_documents_without_player_id_never_share_an_id(self):
        svc = self._service()
        svc.add_documents([Document(page_content="a"), Document(page_content="b")])
        svc.add_documents([Document(page_content="c")])
        ids = [
            v["id"]
            for call in svc._get_index.return_value.upsert.call_args_list
            for v in call.kwargs["vectors"]
        ]
        assert len(ids) == 3 and len(set(ids)) == 3

    def test_player_id_is_used_as_vector_id(self):
        svc = self._service()
        svc.add_documents([Document(page_content="a", metadata={"player_id": "418560"})])
        (call,) = svc._get_index.return_value.upsert.call_args_list
        assert call.kwargs["vectors"][0]["id"] == "418560"

    def test_initialize_index_checks_pinecone_once(self):
        svc = VectorStoreService()
        pc = MagicMock()
        pc.list_indexes.return_value = [SimpleNamespace(name="pitchmaster-index")]
        svc._get_pinecone = MagicMock(return_value=pc)
        with patch("app.services.vector_store.settings") as s:
            s.PINECONE_INDEX_NAME = "pitchmaster-index"
            svc.initialize_index()
            svc.initialize_index()
        pc.list_indexes.assert_called_once()

    def test_embeddings_are_normalized(self):
        svc = VectorStoreService()
        fake_model = MagicMock()
        fake_model.embed.return_value = iter([np.array([3.0, 4.0])])
        svc._embedder = fake_model
        assert svc._embed("x") == pytest.approx([0.6, 0.8])

    def test_retriever_keeps_pinecone_score(self):
        svc = VectorStoreService()
        svc._embed = MagicMock(return_value=[0.1])
        index = MagicMock()
        index.query.return_value = {
            "matches": [{"id": "1", "score": 0.87, "metadata": {"content": "c", "name": "A"}}]
        }
        svc._get_index = MagicMock(return_value=index)
        (doc,) = svc.as_retriever().invoke("q")
        assert doc.metadata["score"] == 0.87


# ---------------------------------------------------------------------------
# Blocking work runs off the event loop
# ---------------------------------------------------------------------------
async def test_slow_retrieval_does_not_block_other_requests():
    def slow_invoke(_query):
        time.sleep(0.5)  # stands in for embedding + a Pinecone round trip
        return []

    retriever = MagicMock()
    retriever.invoke.side_effect = slow_invoke

    with patch("app.services.rag.vector_store_service") as mock_vs:
        mock_vs.as_retriever.return_value = retriever
        start = time.perf_counter()
        retrieval = asyncio.create_task(RAGService().retrieve_context("q"))
        # Stands in for another request: if retrieval blocked the loop, this
        # 50 ms sleep could only finish after the 500 ms retrieval.
        await asyncio.sleep(0.05)
        other_request_latency = time.perf_counter() - start
        await retrieval

    retriever.invoke.assert_called_once_with("q")
    assert other_request_latency < 0.3
