"""
Integration tests — exercise the full request pipeline end-to-end.

Strategy:
- Redis: uses the built-in in-memory fallback (no real Redis needed).
- LLM (Groq): the orchestrator chain is mocked at the chain level so no
  real API calls are made and no key is required.
- Pinecone: not exercised here (covered by vector_store unit tests).

These tests complement the unit tests in test_agents.py / test_api.py by
validating that the complete HTTP → schema → service → response path works
correctly, including the new Pydantic bounds and the /config endpoint.
"""
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from app.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# A valid telemetry item reused across tests
# ---------------------------------------------------------------------------
VALID_ITEM = {
    "player_id": "haaland_9",
    "speed": 28.4,
    "distance_covered": 9.2,
    "heart_rate": 172,
    "position_x": 85.0,
    "position_y": 34.0,
    "fatigue_index": 0.62,
}


# ===========================================================================
# Telemetry ingestion pipeline
# ===========================================================================
class TestTelemetryPipeline:
    """Validates the ingest → queue → status cycle."""

    @patch("app.services.cache.redis_service.enqueue_task", new_callable=AsyncMock)
    def test_valid_item_accepted(self, mock_enqueue):
        response = client.post(
            "/api/v1/ingest/telemetry", json={"data": [VALID_ITEM]}
        )
        assert response.status_code == 202
        assert response.json()["count"] == 1
        mock_enqueue.assert_called_once()

    def test_invalid_speed_rejected(self):
        bad = {**VALID_ITEM, "speed": 9999.0}
        response = client.post("/api/v1/ingest/telemetry", json={"data": [bad]})
        assert response.status_code == 422

    def test_negative_speed_rejected(self):
        bad = {**VALID_ITEM, "speed": -5.0}
        response = client.post("/api/v1/ingest/telemetry", json={"data": [bad]})
        assert response.status_code == 422

    def test_invalid_heart_rate_rejected(self):
        bad = {**VALID_ITEM, "heart_rate": 300}
        response = client.post("/api/v1/ingest/telemetry", json={"data": [bad]})
        assert response.status_code == 422

    def test_out_of_bounds_position_rejected(self):
        bad = {**VALID_ITEM, "position_x": 200.0}
        response = client.post("/api/v1/ingest/telemetry", json={"data": [bad]})
        assert response.status_code == 422

    def test_fatigue_above_one_rejected(self):
        bad = {**VALID_ITEM, "fatigue_index": 1.5}
        response = client.post("/api/v1/ingest/telemetry", json={"data": [bad]})
        assert response.status_code == 422

    def test_empty_batch_rejected(self):
        response = client.post("/api/v1/ingest/telemetry", json={"data": []})
        assert response.status_code == 422
        assert "empty" in response.text.lower()

    def test_oversized_batch_rejected(self):
        from app.core.config import get_settings
        limit = get_settings().MAX_TELEMETRY_BATCH
        big_batch = [VALID_ITEM] * (limit + 1)
        response = client.post(
            "/api/v1/ingest/telemetry", json={"data": big_batch}
        )
        assert response.status_code == 422
        assert "too large" in response.text.lower()


# ===========================================================================
# Analysis pipeline — LLM chain mocked at classify level
# ===========================================================================
class TestAnalysisPipeline:

    def _patch_route(self, route_token: str) -> MagicMock:
        """Return a mock for Orchestrator._classify_route that returns route_token."""
        mock = AsyncMock(return_value=route_token)
        return mock

    @patch(
        "app.agents.orchestrator.Orchestrator._classify_route",
        new_callable=AsyncMock,
        return_value="TACTICAL",
    )
    def test_tactical_without_context_returns_helpful_message(self, _):
        response = client.post(
            "/api/v1/analyze",
            json={"query": "What pressing tactic should we use?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["route_used"] == "TACTICAL_NO_DATA"
        assert "telemetry" in data["insight"].lower()

    @patch(
        "app.agents.orchestrator.Orchestrator._classify_route",
        new_callable=AsyncMock,
        return_value="FORMATION",
    )
    def test_formation_without_context_returns_helpful_message(self, _):
        response = client.post(
            "/api/v1/analyze",
            json={"query": "Should we switch to 4-3-3?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["route_used"] == "FORMATION_NO_DATA"

    @patch(
        "app.agents.orchestrator.Orchestrator._classify_route",
        new_callable=AsyncMock,
        return_value="SENTIMENT",
    )
    def test_sentiment_without_context_returns_helpful_message(self, _):
        response = client.post(
            "/api/v1/analyze",
            json={"query": "How is player morale?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["route_used"] == "SENTIMENT_NO_DATA"

    def test_query_too_long_rejected(self):
        response = client.post(
            "/api/v1/analyze",
            json={"query": "x" * 2001},
        )
        assert response.status_code == 422

    def test_empty_query_rejected(self):
        response = client.post(
            "/api/v1/analyze",
            json={"query": ""},
        )
        assert response.status_code == 422


# ===========================================================================
# /config endpoint
# ===========================================================================
class TestConfigEndpoint:

    def test_returns_pitch_dimensions(self):
        response = client.get("/api/v1/config")
        assert response.status_code == 200
        data = response.json()
        assert data["pitch_length_m"] == 105.0
        assert data["pitch_width_m"] == 68.0

    def test_returns_fatigue_threshold(self):
        response = client.get("/api/v1/config")
        assert response.status_code == 200
        data = response.json()
        assert "fatigue_high_threshold" in data
        assert 0.0 < data["fatigue_high_threshold"] <= 1.0

    def test_returns_max_batch(self):
        response = client.get("/api/v1/config")
        data = response.json()
        assert "max_telemetry_batch" in data
        assert data["max_telemetry_batch"] > 0
