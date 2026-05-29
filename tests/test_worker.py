"""
Unit tests for the telemetry background worker.
Redis is fully mocked — no real connection needed.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_worker_processes_items():
    """Worker should pop items and increment the processed counter."""
    items = [
        {"player_id": "p1", "speed": 22.0, "fatigue_index": 0.3},
        {"player_id": "p2", "speed": 18.5, "fatigue_index": 0.6},
        None,  # Sentinel — signals queue empty → worker sleeps → we cancel
    ]
    item_iter = iter(items)

    async def mock_get_task(queue_name: str):
        try:
            return next(item_iter)
        except StopIteration:
            await asyncio.sleep(10)  # block so the task can be cancelled

    with (
        patch("app.services.telemetry_worker.redis_service.connect", new_callable=AsyncMock),
        patch("app.services.telemetry_worker.redis_service.get_task", side_effect=mock_get_task),
        patch("app.services.telemetry_worker._processed_count", 0),
    ):
        from app.services import telemetry_worker

        # Reset counter
        telemetry_worker._processed_count = 0

        worker_task = asyncio.create_task(telemetry_worker.telemetry_worker())
        await asyncio.sleep(0.3)  # Let worker drain two items + hit sleep
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

        assert telemetry_worker.get_processed_count() == 2


@pytest.mark.asyncio
async def test_worker_logs_high_fatigue(caplog):
    """Worker should emit a WARNING for fatigue > 0.75."""
    import logging

    high_fatigue_item = {"player_id": "p99", "speed": 10.0, "fatigue_index": 0.9}
    call_count = 0

    async def mock_get_task(_):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return high_fatigue_item
        await asyncio.sleep(10)

    with (
        patch("app.services.telemetry_worker.redis_service.connect", new_callable=AsyncMock),
        patch("app.services.telemetry_worker.redis_service.get_task", side_effect=mock_get_task),
    ):
        from app.services import telemetry_worker
        telemetry_worker._processed_count = 0

        with caplog.at_level(logging.WARNING, logger="app.services.telemetry_worker"):
            worker_task = asyncio.create_task(telemetry_worker.telemetry_worker())
            await asyncio.sleep(0.2)
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass

        assert any("HIGH FATIGUE" in r.message for r in caplog.records)
