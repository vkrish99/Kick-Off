"""
Tests for Orchestrator routing logic.
All LLM calls and downstream agents are mocked.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException
from app.agents.orchestrator import Orchestrator


@pytest.fixture
def orchestrator():
    return Orchestrator()


def _mock_router(route_token: str):
    """Return a mock _classify_route that returns the given token."""
    mock = AsyncMock(return_value=route_token)
    return mock


class TestOrchestratorRouting:
    @patch("app.agents.orchestrator.tactical_agent")
    async def test_routes_to_tactical_with_telemetry(self, mock_tactical, orchestrator):
        mock_tactical.analyze = AsyncMock(return_value="Tactical insight: player spacing is good.")
        orchestrator._classify_route = _mock_router("TACTICAL")

        insight, route = await orchestrator.process_query(
            query="What is the player spacing like?",
            telemetry_context=[{"player_id": "haaland_9", "speed": 22.0}],
        )

        assert route == "TACTICAL"
        assert insight == "Tactical insight: player spacing is good."
        mock_tactical.analyze.assert_called_once()

    @patch("app.agents.orchestrator.tactical_agent")
    async def test_tactical_falls_back_to_general_without_telemetry(self, mock_tactical, orchestrator):
        orchestrator._classify_route = _mock_router("TACTICAL")

        insight, route = await orchestrator.process_query(
            query="What is the player spacing like?",
            telemetry_context=None,
        )

        # No telemetry → returns helpful TACTICAL_NO_DATA message (not silent GENERAL)
        assert route == "TACTICAL_NO_DATA"
        assert "telemetry" in insight.lower()
        mock_tactical.analyze.assert_not_called()

    @patch("app.agents.orchestrator.formation_advisor")
    async def test_routes_to_formation_with_telemetry(self, mock_formation, orchestrator):
        mock_formation.advise = AsyncMock(return_value="Switch to 4-3-3.")
        orchestrator._classify_route = _mock_router("FORMATION")

        insight, route = await orchestrator.process_query(
            query="What formation should we use?",
            telemetry_context=[{"player_id": "p1", "fatigue_index": 0.8}],
        )

        assert route == "FORMATION"
        assert insight == "Switch to 4-3-3."
        mock_formation.advise.assert_called_once()

    @patch("app.agents.orchestrator.sentiment_agent")
    async def test_routes_to_sentiment_with_news_context(self, mock_sentiment, orchestrator):
        mock_sentiment.analyze_sentiment = AsyncMock(return_value="Player morale is high.")
        orchestrator._classify_route = _mock_router("SENTIMENT")

        insight, route = await orchestrator.process_query(
            query="How is the team morale?",
            news_context="Star player returned from injury and is fit.",
        )

        assert route == "SENTIMENT"
        assert insight == "Player morale is high."
        mock_sentiment.analyze_sentiment.assert_called_once_with(
            "Star player returned from injury and is fit.",
            "How is the team morale?",
        )

    @patch("app.agents.orchestrator.sentiment_agent")
    async def test_sentiment_falls_back_to_general_without_news(self, mock_sentiment, orchestrator):
        orchestrator._classify_route = _mock_router("SENTIMENT")

        insight, route = await orchestrator.process_query(
            query="How is the morale?",
            news_context=None,
        )

        # No news context → returns helpful SENTIMENT_NO_DATA message (not silent GENERAL)
        assert route == "SENTIMENT_NO_DATA"
        assert "news" in insight.lower() or "context" in insight.lower()
        mock_sentiment.analyze_sentiment.assert_not_called()

    async def test_general_fallback_returns_query_in_response(self, orchestrator):
        orchestrator._classify_route = _mock_router("GENERAL")

        insight, route = await orchestrator.process_query(query="Hello there!")

        assert route == "GENERAL"
        assert "Hello there!" in insight

    @patch("app.agents.orchestrator.rag_service")
    @patch("app.agents.orchestrator.settings")
    async def test_routes_to_rag_and_calls_retrieve(self, mock_settings, mock_rag, orchestrator):
        from langchain_core.documents import Document

        mock_settings.GROQ_API_KEY = "fake-key"
        mock_settings.GROQ_MODEL = "llama-3.3-70b-versatile"
        mock_rag.retrieve_context = AsyncMock(
            return_value=[Document(page_content="Haaland profile context", metadata={})]
        )
        orchestrator._classify_route = _mock_router("RAG")

        # Mock the inline chain construction in the RAG branch
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(return_value="RAG answer about Haaland")

        with patch("app.agents.orchestrator.ChatGroq") as mock_groq_cls, \
             patch("app.agents.orchestrator.ChatPromptTemplate") as mock_cpt, \
             patch("app.agents.orchestrator.StrOutputParser") as mock_parser:

            # Build: prompt | llm | parser  →  mock_chain
            mock_template = MagicMock()
            mock_intermediate = MagicMock()
            mock_cpt.from_messages.return_value = mock_template
            mock_template.__or__ = MagicMock(return_value=mock_intermediate)
            mock_intermediate.__or__ = MagicMock(return_value=mock_chain)

            insight, route = await orchestrator.process_query(
                query="What are Haaland's key stats?",
            )

        assert route == "RAG"
        assert insight == "RAG answer about Haaland"
        mock_rag.retrieve_context.assert_called_once_with("What are Haaland's key stats?")

    async def test_missing_api_key_raises_503_from_router(self, orchestrator):
        """When GROQ_API_KEY is absent, _classify_route should propagate 503."""
        # Don't mock _classify_route — let it hit _get_router_chain() with no key
        with patch("app.agents.orchestrator.settings") as mock_settings:
            mock_settings.GROQ_API_KEY = None

            with pytest.raises(HTTPException) as exc_info:
                await orchestrator.process_query(query="test")

        assert exc_info.value.status_code == 503
