"""Thread-safe event bus cho SSE streaming.

Agent chạy trong threadpool (sync code gọi Ollama blocking) publish event vào
queue; FastAPI endpoint subscribe async qua `asyncio.to_thread(queue.get)`.
"""
from __future__ import annotations

import asyncio
import queue
import uuid
from typing import AsyncIterator


class EventBus:
    def __init__(self) -> None:
        self._queues: dict[str, queue.Queue] = {}

    def create_run(self) -> str:
        run_id = uuid.uuid4().hex[:8]
        self._queues[run_id] = queue.Queue()
        return run_id

    def publish(self, run_id: str, event: dict) -> None:
        q = self._queues.get(run_id)
        if q is not None:
            q.put(event)

    async def subscribe(self, run_id: str) -> AsyncIterator[dict]:
        q = self._queues.get(run_id)
        if q is None:
            return
        while True:
            event = await asyncio.to_thread(q.get)
            yield event
            if event.get("type") == "done":
                break
        # Cleanup khi client đã nhận "done"
        self._queues.pop(run_id, None)


bus = EventBus()
