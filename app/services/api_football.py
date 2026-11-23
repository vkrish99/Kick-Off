"""
API-Football Client
===================
Thin async client for the API-Football v3 REST API (api-sports.io), used to
sync real player availability (injuries / doubts) into PitchMaster.

API-Football reports most failures (bad key, plan restrictions, exhausted
daily quota) as HTTP 200 with a non-empty `errors` field, so every response
is checked for that before its `response` payload is trusted.

Raw responses are cached in Redis so re-running a sync on the same day
doesn't burn the free tier's daily request quota.
"""
import json
import logging

import httpx

from app.core.config import get_settings
from app.services.cache import redis_service

settings = get_settings()
logger = logging.getLogger(__name__)


class ApiFootballError(RuntimeError):
    """Raised when API-Football rejects a request or reports errors in its payload."""


class ApiFootballClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,  # injectable for tests
    ):
        self.api_key = api_key or settings.API_FOOTBALL_KEY
        self.base_url = (base_url or settings.API_FOOTBALL_BASE_URL).rstrip("/")
        self.transport = transport

    async def _get(self, path: str, params: dict) -> list[dict]:
        if not self.api_key:
            raise ApiFootballError(
                "API_FOOTBALL_KEY is not configured. Get a free key at "
                "https://dashboard.api-football.com and set it in .env."
            )

        cache_key = f"api_football:{path}:{json.dumps(params, sort_keys=True)}"
        cached = await redis_service.get_value(cache_key)
        if cached:
            logger.info("API-Football cache hit: %s %s", path, params)
            return json.loads(cached)

        async with httpx.AsyncClient(
            base_url=self.base_url, timeout=20.0, transport=self.transport
        ) as client:
            resp = await client.get(path, params=params, headers={"x-apisports-key": self.api_key})

        if resp.status_code != 200:
            raise ApiFootballError(f"API-Football {path} returned HTTP {resp.status_code}")

        body = resp.json()
        errors = body.get("errors")
        # `errors` is [] on success, and either a list or a {field: message} dict on failure.
        if errors:
            messages = errors.values() if isinstance(errors, dict) else errors
            raise ApiFootballError("API-Football error: " + "; ".join(str(m) for m in messages))

        remaining = resp.headers.get("x-ratelimit-requests-remaining")
        if remaining is not None:
            logger.info("API-Football daily requests remaining: %s", remaining)

        results = body.get("response", [])
        await redis_service.set_value(
            cache_key, json.dumps(results), expire=settings.API_FOOTBALL_CACHE_TTL
        )
        return results

    async def get_injuries(self, league: int, season: int) -> list[dict]:
        """All injury/doubt rows for a league season — one row per player per fixture."""
        return await self._get("/injuries", {"league": league, "season": season})


api_football_client = ApiFootballClient()
