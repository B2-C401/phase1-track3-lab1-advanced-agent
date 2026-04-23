"""Đọc JSONL đã lưu và chuyển thành event stream như khi chạy live."""
from __future__ import annotations

from pathlib import Path

from src.reflexion_lab.schemas import RunRecord


def load_saved_runs(run_dir: Path) -> dict[tuple[str, str], RunRecord]:
    """Trả về mapping (qid, agent_type) -> RunRecord từ thư mục output."""
    records: dict[tuple[str, str], RunRecord] = {}
    for filename, expected_agent in (
        ("react_runs.jsonl", "react"),
        ("reflexion_runs.jsonl", "reflexion"),
    ):
        path = run_dir / filename
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = RunRecord.model_validate_json(line)
            records[(record.qid, record.agent_type)] = record
    return records


def record_to_events(record: RunRecord) -> list[dict]:
    events: list[dict] = []
    for trace in record.traces:
        events.append({
            "type": "attempt_start",
            "agent": record.agent_type,
            "attempt": trace.attempt_id,
        })
        events.append({
            "type": "actor_end",
            "agent": record.agent_type,
            "attempt": trace.attempt_id,
            "answer": trace.answer,
            "tokens": trace.token_estimate,
            "latency_ms": trace.latency_ms,
        })
        events.append({
            "type": "evaluator_end",
            "agent": record.agent_type,
            "attempt": trace.attempt_id,
            "score": trace.score,
            "reason": trace.reason,
        })
        if trace.reflection:
            events.append({
                "type": "reflector_end",
                "agent": record.agent_type,
                "attempt": trace.attempt_id,
                "lesson": trace.reflection.lesson,
                "strategy": trace.reflection.next_strategy,
                "failure_reason": trace.reflection.failure_reason,
            })
        events.append({
            "type": "attempt_end",
            "agent": record.agent_type,
            "attempt": trace.attempt_id,
            "correct": trace.score == 1,
        })
    events.append({
        "type": "run_end",
        "agent": record.agent_type,
        "final_answer": record.predicted_answer,
        "correct": record.is_correct,
        "total_attempts": record.attempts,
        "total_tokens": record.token_estimate,
        "total_latency_ms": record.latency_ms,
        "failure_mode": record.failure_mode,
    })
    return events
