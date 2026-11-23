"""
Tests for API routes.
All external I/O (Redis, OpenAI, Pinecone) is mocked.
"""
from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import AsyncMock, MagicMock, patch

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["project"] == "PitchMaster AI"
    assert "version" in data


# ---------------------------------------------------------------------------
# Telemetry Ingestion
# ---------------------------------------------------------------------------
@patch("app.services.cache.redis_service.enqueue_task", new_callable=AsyncMock)
def test_ingest_telemetry(mock_enqueue):
    payload = {
        "data": [
            {
                "player_id": "p1",
                "speed": 22.5,
                "distance_covered": 10.2,
                "heart_rate": 150,
                "position_x": 10.0,
                "position_y": 20.0,
                "fatigue_index": 0.1,
            }
        ]
    }
    response = client.post("/api/v1/ingest/telemetry", json=payload)
    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert response.json()["count"] == 1
    mock_enqueue.assert_called_once()


# ---------------------------------------------------------------------------
# Analyze endpoint
# ---------------------------------------------------------------------------
@patch("app.agents.orchestrator.orchestrator.process_query", new_callable=AsyncMock)
def test_analyze_query(mock_process):
    mock_process.return_value = ("This is a tactical insight.", "TACTICAL", [])
    payload = {"query": "How is player p1 performing?"}
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["insight"] == "This is a tactical insight."
    assert data["route_used"] == "TACTICAL"


# ---------------------------------------------------------------------------
# Queue Status endpoint
# ---------------------------------------------------------------------------
@patch("app.services.cache.redis_service.connect", new_callable=AsyncMock)
@patch("app.services.telemetry_worker.get_processed_count", return_value=42)
def test_queue_status(mock_count, mock_connect):
    # Mock the Redis client's llen method
    mock_client = MagicMock()
    mock_client.llen = AsyncMock(return_value=7)

    with patch("app.services.cache.redis_service.client", mock_client):
        response = client.get("/api/v1/queue/status")

    assert response.status_code == 200
    data = response.json()
    assert data["queue"] == "telemetry_queue"
    assert data["depth"] == 7
    assert data["processed_total"] == 42
