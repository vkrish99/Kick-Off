"""
Player Availability Service
===========================
Turns API-Football injury rows into one current `PlayerStatus` per player,
stores them in Redis, and mirrors them onto the Pinecone profile vectors so
retrieval and the agents can see who is unavailable.

Redis layout:
  player_status:{player_id}   JSON PlayerStatus, expires after PLAYER_STATUS_TTL
  player_status:flagged       JSON list of player_ids flagged in the last sync
  player_status:last_sync     ISO timestamp of the last successful sync

Any player not flagged by the latest sync is treated as available. Players
flagged previously but not anymore are reset to "available" in Pinecone.
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone

from app.core.config import get_settings
from app.models.schemas import PlayerStatus
from app.services.cache import redis_service
from app.services.vector_store import vector_store_service

settings = get_settings()
logger = logging.getLogger(__name__)

AVAILABLE = "available"
_TYPE_TO_STATUS = {"missing fixture": "out", "questionable": "doubtful"}

STATUS_KEY = "player_status:{}"
FLAGGED_KEY = "player_status:flagged"
LAST_SYNC_KEY = "player_status:last_sync"


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def build_statuses(rows: list[dict], now: datetime | None = None) -> list[PlayerStatus]:
    """Collapse per-fixture injury rows into each player's most recent status.

    Rows whose fixture is older than PLAYER_STATUS_STALE_DAYS are dropped: a
    player last listed injured a month ago has most likely recovered.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=settings.PLAYER_STATUS_STALE_DAYS)
    latest: dict[int, tuple[datetime, dict]] = {}

    for row in rows:
        player = row.get("player") or {}
        pid = player.get("id")
        fixture_date = _parse_date((row.get("fixture") or {}).get("date"))
        if pid is None or fixture_date is None or fixture_date < cutoff:
            continue
        if pid not in latest or fixture_date > latest[pid][0]:
            latest[pid] = (fixture_date, row)

    statuses = []
    for pid, (fixture_date, row) in latest.items():
        player = row["player"]
        status = _TYPE_TO_STATUS.get((player.get("type") or "").lower())
        if status is None:
            continue
        statuses.append(PlayerStatus(
            api_football_id=pid,
            name=player.get("name") or "Unknown",
            team=(row.get("team") or {}).get("name") or "Unknown",
            status=status,
            reason=player.get("reason"),
            fixture_date=fixture_date,
            updated_at=now,
        ))
    return statuses


async def store_statuses(statuses: list[PlayerStatus]) -> list[str]:
    """Persist matched statuses; return ids that were flagged before but are now clear."""
    matched = [s for s in statuses if s.player_id]
    for s in matched:
        await redis_service.set_value(
            STATUS_KEY.format(s.player_id), s.model_dump_json(), expire=settings.PLAYER_STATUS_TTL
        )

    previous = json.loads(await redis_service.get_value(FLAGGED_KEY) or "[]")
    current = sorted({s.player_id for s in matched})
    cleared = sorted(set(previous) - set(current))

    await redis_service.set_value(FLAGGED_KEY, json.dumps(current))
    await redis_service.set_value(LAST_SYNC_KEY, datetime.now(timezone.utc).isoformat())
    return cleared


def _update_vectors(statuses: list[PlayerStatus], cleared: list[str]) -> int:
    index = vector_store_service._get_index()
    updates = [
        (s.player_id, {
            "status": s.status,
            "status_reason": s.reason or "",
            "status_updated_at": s.updated_at.isoformat(),
        })
        for s in statuses if s.player_id
    ]
    now = datetime.now(timezone.utc).isoformat()
    updates += [(pid, {"status": AVAILABLE, "status_reason": "", "status_updated_at": now}) for pid in cleared]

    updated = 0
    for pid, metadata in updates:
        try:
            index.update(id=pid, set_metadata=metadata)
            updated += 1
        except Exception as exc:  # player may simply not be ingested into the index
            logger.debug("Skipping Pinecone status update for %s: %s", pid, exc)
    return updated


async def update_vector_metadata(statuses: list[PlayerStatus], cleared: list[str]) -> int:
    """Mirror statuses onto Pinecone metadata (sync SDK, so run off the event loop)."""
    return await asyncio.to_thread(_update_vectors, statuses, cleared)


async def get_status(player_id: str) -> dict:
    """Current status for one player: flagged, available, or unknown (never synced)."""
    raw = await redis_service.get_value(STATUS_KEY.format(player_id))
    if raw:
        return {"known": True, **PlayerStatus.model_validate_json(raw).model_dump(mode="json")}
    last_sync = await redis_service.get_value(LAST_SYNC_KEY)
    if last_sync:
        return {"known": True, "player_id": player_id, "status": AVAILABLE, "updated_at": last_sync}
    return {"known": False, "player_id": player_id, "status": "unknown"}


async def list_unavailable(team: str | None = None) -> list[dict]:
    flagged = json.loads(await redis_service.get_value(FLAGGED_KEY) or "[]")
    result = []
    for pid in flagged:
        raw = await redis_service.get_value(STATUS_KEY.format(pid))
        if not raw:
            continue
        status = PlayerStatus.model_validate_json(raw)
        if team and team.lower() not in status.team.lower():
            continue
        result.append(status.model_dump(mode="json"))
    return result
