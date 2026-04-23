"""Chạy agent live với emit → publish lên event bus."""
from __future__ import annotations

from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.schemas import QAExample

from .event_bus import bus


def run_live(example: QAExample, run_id: str, max_attempts: int = 3) -> None:
    """Blocking — gọi qua asyncio.to_thread(...) từ endpoint.

    Chạy cả ReAct lẫn Reflexion tuần tự. Mỗi agent phát run_end của riêng nó,
    cuối cùng publish một event {"type": "done"} để client đóng stream.
    """
    agents = [
        ("react", ReActAgent()),
        ("reflexion", ReflexionAgent(max_attempts=max_attempts)),
    ]
    try:
        for _, agent in agents:
            def emit(event: dict) -> None:
                bus.publish(run_id, event)

            record = agent.run(example, emit=emit)
            bus.publish(run_id, {
                "type": "run_end",
                "agent": agent.agent_type,
                "final_answer": record.predicted_answer,
                "correct": record.is_correct,
                "total_attempts": record.attempts,
                "total_tokens": record.token_estimate,
                "total_latency_ms": record.latency_ms,
                "failure_mode": record.failure_mode,
            })
    finally:
        bus.publish(run_id, {"type": "done"})
