"""
Unit tests for specialist agents.
All LLM calls are mocked — no real API keys required.
Uses pytest-asyncio for async test support (Python 3.14 compatible).
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# TacticalAgent
# ---------------------------------------------------------------------------
class TestTacticalAgent:
    @pytest.mark.asyncio
    @patch("app.agents.tactical.settings")
    async def test_analyze_returns_string(self, mock_settings):
        mock_settings.GROQ_API_KEY = "test-key-groq"

        from app.agents.tactical import TacticalAgent
        agent = TacticalAgent()
        # _chain is invoked via await chain.ainvoke(...) — mock the object
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(
            return_value="Player p1 is well-positioned on the right flank."
        )
        agent._chain = mock_chain

        result = await agent.analyze([{"player_id": "p1", "speed": 30}], "Is p1 fast?")
        assert isinstance(result, str)
        assert "p1" in result.lower() or len(result) > 0

    @pytest.mark.asyncio
    @patch("app.agents.tactical.settings")
    async def test_analyze_raises_503_without_key(self, mock_settings):
        mock_settings.GROQ_API_KEY = None

        from app.agents.tactical import TacticalAgent
        from fastapi import HTTPException
        agent = TacticalAgent()
        agent._chain = None  # force _get_chain() to run and raise

        with pytest.raises(HTTPException) as exc_info:
            await agent.analyze([], "test query")
        assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# SentimentAgent
# ---------------------------------------------------------------------------
class TestSentimentAgent:
    @pytest.mark.asyncio
    @patch("app.agents.sentiment.settings")
    async def test_analyze_sentiment_returns_string(self, mock_settings):
        mock_settings.GROQ_API_KEY = "test-key-groq"

        from app.agents.sentiment import SentimentAgent
        agent = SentimentAgent()
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(
            return_value="The news has a positive impact on morale."
        )
        agent._chain = mock_chain

        result = await agent.analyze_sentiment("Player scores hat-trick!", "Premier League")
        assert "morale" in result.lower()

    @pytest.mark.asyncio
    @patch("app.agents.sentiment.settings")
    async def test_analyze_sentiment_raises_503_without_key(self, mock_settings):
        mock_settings.GROQ_API_KEY = None

        from app.agents.sentiment import SentimentAgent
        from fastapi import HTTPException
        agent = SentimentAgent()
        agent._chain = None

        with pytest.raises(HTTPException) as exc_info:
            await agent.analyze_sentiment("some news", "")
        assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# FormationAdvisor
# ---------------------------------------------------------------------------
class TestFormationAdvisor:
    @pytest.mark.asyncio
    @patch("app.agents.formation.settings")
    async def test_advise_returns_string(self, mock_settings):
        mock_settings.GROQ_API_KEY = "test-key-groq"

        from app.agents.formation import FormationAdvisor
        agent = FormationAdvisor()
        mock_chain = MagicMock()
        mock_chain.ainvoke = AsyncMock(
            return_value="Recommended: 4-3-3. High energy on the wings."
        )
        agent._chain = mock_chain

        telemetry = [{"player_id": "p1", "speed": 25, "fatigue_index": 0.2}]
        result = await agent.advise(telemetry, "What formation should we use?")
        assert "4-3-3" in result

    @pytest.mark.asyncio
    @patch("app.agents.formation.settings")
    async def test_advise_raises_503_without_key(self, mock_settings):
        mock_settings.GROQ_API_KEY = None

        from app.agents.formation import FormationAdvisor
        from fastapi import HTTPException
        agent = FormationAdvisor()
        agent._chain = None

        with pytest.raises(HTTPException) as exc_info:
            await agent.advise([], "What formation?")
        assert exc_info.value.status_code == 503
