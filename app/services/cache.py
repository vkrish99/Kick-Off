"""
Redis Service with in-memory fallback.
When Redis is unavailable (no Docker/Redis installed), falls back to an
in-memory asyncio.Queue so the app still works for development.
"""
import asyncio
import json
import logging
from collections import deque

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class RedisService:
    def __init__(self):
        self.redis_url = settings.REDIS_URL
        self.client = None
        self._connected = False
        self._use_memory = False
        self._memory_queues: dict[str, deque] = {}
        self._memory_kv: dict[str, str] = {}

    async def connect(self) -> None:
        if self._connected:
            return
        try:
            import redis.asyncio as redis
            self.client = redis.from_url(
                self.redis_url, encoding="utf-8", decode_responses=True
            )
            # Test the connection
            await self.client.ping()
            self._connected = True
            logger.info("✅  Connected to Redis at %s", self.redis_url)
        except (ConnectionRefusedError, OSError) as exc:
            logger.warning(
                "⚠️  Redis unavailable (%s). Using in-memory queue fallback. "
                "Telemetry will NOT persist across restarts.",
                exc,
            )
            self._use_memory = True
            self._connected = True
        except Exception as exc:
            # Catches redis.exceptions.ConnectionError, AuthenticationError, etc.
            logger.warning(
                "⚠️  Redis connection error (%s). Using in-memory queue fallback. "
                "Telemetry will NOT persist across restarts.",
                exc,
            )
            self._use_memory = True
            self._connected = True

    async def close(self) -> None:
        if self.client and not self._use_memory:
            await self.client.close()

    async def enqueue_task(self, queue_name: str, task_data: dict) -> None:
        """Push a task to a queue."""
        if not self._connected:
            await self.connect()
        if self._use_memory:
            if queue_name not in self._memory_queues:
                self._memory_queues[queue_name] = deque()
            self._memory_queues[queue_name].append(json.dumps(task_data))
            return
        await self.client.rpush(queue_name, json.dumps(task_data))

    async def get_task(self, queue_name: str) -> dict | None:
        """Pop a task from a queue."""
        if not self._connected:
            await self.connect()
        if self._use_memory:
            q = self._memory_queues.get(queue_name, deque())
            if q:
                return json.loads(q.popleft())
            await asyncio.sleep(0.1)  # simulate blpop timeout
            return None
        task = await self.client.blpop(queue_name, timeout=1)
        if task:
            return json.loads(task[1])
        return None

    async def queue_length(self, queue_name: str) -> int:
        """Get queue depth."""
        if not self._connected:
            await self.connect()
        if self._use_memory:
            return len(self._memory_queues.get(queue_name, deque()))
        return await self.client.llen(queue_name)

    async def set_value(self, key: str, value: str, expire: int = None):
        if not self._connected:
            await self.connect()
        if self._use_memory:
            self._memory_kv[key] = value
            return
        await self.client.set(key, value, ex=expire)

    async def get_value(self, key: str):
        if not self._connected:
            await self.connect()
        if self._use_memory:
            return self._memory_kv.get(key)
        return await self.client.get(key)


redis_service = RedisService()
