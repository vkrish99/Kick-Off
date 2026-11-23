"""
Tests for the API-Football player availability sync.
HTTP is served by httpx.MockTransport and Redis by the in-memory fallback —
no network or real Redis needed.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import PlayerStatus
from app.services import player_status
from app.services.api_football import ApiFootballClient, ApiFootballError
from app.services.cache import RedisService
from app.services.player_matcher import Candidate, match_player, reconcile, resolve_club

NOW = datetime(2025, 10, 3, 8, 0, tzinfo=timezone.utc)


def injury_row(pid, name, team, type_, reason, days_ago):
    return {
        "player": {"id": pid, "name": name, "type": type_, "reason": reason},
        "team": {"id": 1, "name": team},
        "fixture": {"id": pid * 100 + days_ago, "date": (NOW - timedelta(days=days_ago)).isoformat()},
    }


@pytest.fixture
def memory_redis():
    """A RedisService forced into its in-memory fallback, patched in everywhere it's used."""
    svc = RedisService()
    svc._use_memory = True
    svc._connected = True
    with (
        patch("app.services.player_status.redis_service", svc),
        patch("app.services.api_football.redis_service", svc),
    ):
        yield svc


# ---------------------------------------------------------------------------
# ID reconciliation
# ---------------------------------------------------------------------------
CITY = "Manchester City Football Club"
UNITED = "Manchester United Football Club"
SQUADS = {
    CITY: [
        Candidate("418560", "Erling Haaland", "Erling", "Haaland", CITY),
        Candidate("357565", "Rodri", "", "Rodri", CITY),
        Candidate("258004", "Rúben Dias", "Rúben", "Dias", CITY),
    ],
    UNITED: [
        Candidate("240306", "Bruno Fernandes", "Bruno", "Fernandes", UNITED),
        Candidate("999001", "Diogo Dalot", "Diogo", "Dalot", UNITED),
    ],
}


class TestMatcher:
    def test_abbreviated_first_name(self):
        assert reconcile("E. Haaland", "Manchester City", SQUADS).player_id == "418560"

    def test_full_name(self):
        assert reconcile("Bruno Fernandes", "Manchester United", SQUADS).player_id == "240306"

    def test_mononym(self):
        assert reconcile("Rodri", "Manchester City", SQUADS).player_id == "357565"

    def test_accents_are_ignored(self):
        assert reconcile("R. Dias", "Manchester City", SQUADS).player_id == "258004"

    def test_wrong_initial_is_not_matched(self):
        assert reconcile("X. Haaland", "Manchester City", SQUADS) is None

    def test_player_at_other_club_is_not_matched(self):
        assert reconcile("E. Haaland", "Manchester United", SQUADS) is None

    def test_ambiguous_surname_is_not_matched(self):
        squad = [
            Candidate("1", "Ben Silva", "Ben", "Silva", "X"),
            Candidate("2", "Bernardo Silva", "Bernardo", "Silva", "X"),
        ]
        assert match_player("B. Silva", squad) is None

    def test_club_aliases_and_legal_names(self):
        clubs = [CITY, UNITED, "Wolverhampton Wanderers Football Club", "Bayer 04 Leverkusen Fußball"]
        assert resolve_club("Manchester City", clubs) == CITY
        assert resolve_club("Manchester United", clubs) == UNITED
        assert resolve_club("Wolves", clubs) == "Wolverhampton Wanderers Football Club"
        assert resolve_club("Bayer Leverkusen", clubs) == "Bayer 04 Leverkusen Fußball"
        assert resolve_club("Arsenal", clubs) is None


# ---------------------------------------------------------------------------
# Injury rows -> statuses
# ---------------------------------------------------------------------------
class TestBuildStatuses:
    def test_keeps_latest_row_per_player(self):
        rows = [
            injury_row(1, "E. Haaland", "Manchester City", "Questionable", "Knock", days_ago=6),
            injury_row(1, "E. Haaland", "Manchester City", "Missing Fixture", "Ankle Injury", days_ago=1),
        ]
        [status] = player_status.build_statuses(rows, now=NOW)
        assert status.status == "out"
        assert status.reason == "Ankle Injury"

    def test_questionable_maps_to_doubtful(self):
        rows = [injury_row(2, "Rodri", "Manchester City", "Questionable", "Knock", days_ago=2)]
        assert player_status.build_statuses(rows, now=NOW)[0].status == "doubtful"

    def test_stale_rows_are_dropped(self):
        rows = [injury_row(3, "R. Dias", "Manchester City", "Missing Fixture", "Hamstring", days_ago=40)]
        assert player_status.build_statuses(rows, now=NOW) == []

    def test_unknown_type_and_malformed_rows_are_skipped(self):
        rows = [
            injury_row(4, "A", "B", "Something New", "?", days_ago=1),
            {"player": {"id": 5}, "fixture": {"date": "not-a-date"}},
            {},
        ]
        assert player_status.build_statuses(rows, now=NOW) == []


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------
def client_returning(body: dict, calls: list | None = None) -> ApiFootballClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        return httpx.Response(200, json=body)
    return ApiFootballClient(api_key="test-key", transport=httpx.MockTransport(handler))


class TestApiFootballClient:
    async def test_returns_response_and_sends_key(self, memory_redis):
        calls = []
        rows = [injury_row(1, "E. Haaland", "Manchester City", "Missing Fixture", "Ankle", 1)]
        api = client_returning({"errors": [], "response": rows}, calls)

        assert await api.get_injuries(39, 2025) == rows
        assert calls[0].headers["x-apisports-key"] == "test-key"
        assert calls[0].url.params["league"] == "39"

    async def test_second_call_is_served_from_cache(self, memory_redis):
        calls = []
        api = client_returning({"errors": [], "response": []}, calls)
        await api.get_injuries(39, 2025)
        await api.get_injuries(39, 2025)
        assert len(calls) == 1

    async def test_errors_in_200_body_raise(self, memory_redis):
        api = client_returning({"errors": {"plan": "Free plans do not have access to this season."}, "response": []})
        with pytest.raises(ApiFootballError, match="Free plans"):
            await api.get_injuries(39, 2025)

    async def test_missing_key_raises(self, memory_redis):
        api = ApiFootballClient(api_key="")
        api.api_key = None
        with pytest.raises(ApiFootballError, match="API_FOOTBALL_KEY"):
            await api.get_injuries(39, 2025)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def status(player_id, state="out"):
    return PlayerStatus(
        player_id=player_id, api_football_id=int(player_id), name="P" + player_id,
        team="Manchester City", status=state, reason="Injury", updated_at=NOW,
    )


class TestStorage:
    async def test_store_and_read_back(self, memory_redis):
        await player_status.store_statuses([status("1"), status("2", "doubtful")])

        assert (await player_status.get_status("1"))["status"] == "out"
        assert (await player_status.get_status("2"))["status"] == "doubtful"
        assert (await player_status.get_status("3"))["status"] == "available"
        assert len(await player_status.list_unavailable()) == 2

    async def test_unknown_before_first_sync(self, memory_redis):
        result = await player_status.get_status("1")
        assert result == {"known": False, "player_id": "1", "status": "unknown"}

    async def test_recovered_players_are_reported_as_cleared(self, memory_redis):
        await player_status.store_statuses([status("1"), status("2")])
        cleared = await player_status.store_statuses([status("2")])
        assert cleared == ["1"]
        assert [s["player_id"] for s in await player_status.list_unavailable()] == ["2"]

    async def test_unmatched_statuses_are_not_stored(self, memory_redis):
        unmatched = status("7")
        unmatched.player_id = None
        await player_status.store_statuses([unmatched])
        assert await player_status.list_unavailable() == []

    async def test_vector_metadata_update_sets_and_clears(self):
        with patch("app.services.player_status.vector_store_service") as mock_vs:
            index = mock_vs._get_index.return_value
            updated = await player_status.update_vector_metadata([status("1")], cleared=["2"])

        assert updated == 2
        calls = {c.kwargs["id"]: c.kwargs["set_metadata"]["status"] for c in index.update.call_args_list}
        assert calls == {"1": "out", "2": "available"}


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
http = TestClient(app)


@patch("app.services.player_status.get_status", new_callable=AsyncMock)
def test_player_status_route(mock_get):
    mock_get.return_value = {"known": True, "player_id": "418560", "status": "out"}
    response = http.get("/api/v1/players/418560/status")
    assert response.status_code == 200
    assert response.json()["status"] == "out"
    mock_get.assert_awaited_once_with("418560")


@patch("app.services.player_status.list_unavailable", new_callable=AsyncMock)
def test_unavailable_route_passes_team_filter(mock_list):
    mock_list.return_value = []
    response = http.get("/api/v1/players/unavailable", params={"team": "Arsenal"})
    assert response.status_code == 200
    mock_list.assert_awaited_once_with("Arsenal")


@patch("app.services.player_status.list_unavailable", new_callable=AsyncMock)
def test_unavailable_route_returns_503_when_store_down(mock_list):
    mock_list.side_effect = ConnectionError("redis down")
    assert http.get("/api/v1/players/unavailable").status_code == 503


# ---------------------------------------------------------------------------
# RAG context includes availability
# ---------------------------------------------------------------------------
def test_rag_context_mentions_availability():
    from langchain_core.documents import Document
    from app.agents.orchestrator import _with_availability

    doc = Document(page_content="Player: Erling Haaland.", metadata={"status": "out", "status_reason": "Ankle"})
    assert _with_availability(doc) == "Player: Erling Haaland. Availability: out (Ankle)."
    assert _with_availability(Document(page_content="x", metadata={})) == "x"
