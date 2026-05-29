"""
Telemetry Worker
================
A long-running async coroutine that continuously pops telemetry items from the
Redis `telemetry_queue` and processes them (validates, logs, persists).

Started automatically as part of the FastAPI lifespan and runs in the background
until the application shuts down.
"""
import asyncio
import logging
from app.services.cache import redis_service
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# In-memory accumulator for quick /queue/status stats
_processed_count: int = 0


async def _write_to_influxdb(item: dict) -> None:
    """Persist a processed telemetry record to InfluxDB (no-op if not configured)."""
    if not settings.INFLUXDB_URL or not settings.INFLUXDB_TOKEN:
        return
    try:
        from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
        from influxdb_client.domain.write_precision import WritePrecision
        from influxdb_client import Point

        async with InfluxDBClientAsync(
            url=settings.INFLUXDB_URL,
            token=settings.INFLUXDB_TOKEN,
            org=settings.INFLUXDB_ORG,
        ) as client:
            write_api = client.write_api()
            point = (
                Point("player_telemetry")
                .tag("player_id", item.get("player_id", "unknown"))
                .field("speed", float(item.get("speed", 0)))
                .field("fatigue_index", float(item.get("fatigue_index", 0)))
                .field("heart_rate", int(item.get("heart_rate", 0)))
                .field("distance_covered", float(item.get("distance_covered", 0)))
            )
            await write_api.write(
                bucket=settings.INFLUXDB_BUCKET,
                record=point,
                precision=WritePrecision.SECONDS,
            )
    except Exception as exc:
        logger.warning("InfluxDB write failed (non-critical): %s", exc)


async def process_telemetry_item(item: dict) -> None:
    """Process a single telemetry record: log, alert on high fatigue, persist."""
    global _processed_count
    player_id = item.get("player_id", "unknown")
    speed = item.get("speed", 0)
    fatigue = item.get("fatigue_index", 0)

    logger.info(
        "worker player=%s speed=%.1f km/h fatigue=%.2f",
        player_id, speed, fatigue,
    )

    if fatigue is not None and fatigue > settings.FATIGUE_HIGH_THRESHOLD:
        logger.warning(
            "HIGH FATIGUE alert player=%s fatigue=%.2f — consider substitution.",
            player_id,
            fatigue,
        )

    await _write_to_influxdb(item)
    _processed_count += 1


async def telemetry_worker() -> None:
    """Main worker loop — runs until the task is cancelled on shutdown."""
    logger.info("Telemetry worker started.")
    await redis_service.connect()

    while True:
        try:
            item = await redis_service.get_task("telemetry_queue")
            if item:
                await process_telemetry_item(item)
            else:
                await asyncio.sleep(settings.WORKER_POLL_INTERVAL)
        except asyncio.CancelledError:
            logger.info("Telemetry worker shutting down gracefully.")
            break
        except Exception as exc:  # noqa: BLE001
            logger.error("Worker unexpected error: %s", exc, exc_info=True)
            await asyncio.sleep(settings.WORKER_ERROR_BACKOFF)


def get_processed_count() -> int:
    """Return the number of telemetry items processed since startup."""
    return _processed_count
