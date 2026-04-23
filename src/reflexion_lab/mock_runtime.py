"""Ollama-backed runtime cho Actor / Evaluator / Reflector.

Tên file giữ nguyên để không phải đổi import ở `agents.py`, nhưng phần giả
lập đã được thay bằng lời gọi HTTP tới Ollama. Mỗi hàm trả về
`(result, Usage)` để agent ghi lại token + latency thật.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")
REQUEST_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))

# Tương thích import cũ. Failure mode nay được suy ra động (xem classify_failure_mode).
FAILURE_MODE_BY_QID: dict[str, str] = {}


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _chat(system: str, user: str, *, force_json: bool = False) -> tuple[str, Usage]:
    payload: dict = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    if force_json:
        payload["format"] = "json"

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return f"[LLM_ERROR: {e}]", Usage(latency_ms=latency_ms)
    latency_ms = int((time.perf_counter() - start) * 1000)

    text = body.get("message", {}).get("content", "").strip()
    usage = Usage(
        prompt_tokens=int(body.get("prompt_eval_count", 0)),
        completion_tokens=int(body.get("eval_count", 0)),
        latency_ms=latency_ms,
    )
    return text, usage


def _format_context(example: QAExample) -> str:
    return "\n\n".join(
        f"[{i}] {chunk.title}\n{chunk.text}"
        for i, chunk in enumerate(example.context, 1)
    )


def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> tuple[str, Usage]:
    ctx = _format_context(example)
    reflection_block = ""
    if reflection_memory:
        joined = "\n\n".join(f"- {m}" for m in reflection_memory)
        reflection_block = f"\nPrevious reflections you MUST follow:\n{joined}\n"
    user = (
        f"Question: {example.question}\n\n"
        f"Context:\n{ctx}\n"
        f"{reflection_block}\n"
        f"Return ONLY the final answer on one line. No explanation."
    )
    text, usage = _chat(ACTOR_SYSTEM, user)
    first_line = next(
        (ln.strip() for ln in text.splitlines() if ln.strip()),
        text.strip(),
    )
    return first_line, usage


def evaluator(example: QAExample, answer: str) -> tuple[JudgeResult, Usage]:
    # Fast path: normalize trùng → score=1, bỏ qua LLM.
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        judge = JudgeResult(
            score=1,
            reason="Normalized predicted equals normalized gold.",
            missing_evidence=[],
            spurious_claims=[],
        )
        return judge, Usage()

    ctx = _format_context(example)
    user = (
        f"Question: {example.question}\n"
        f"Gold answer: {example.gold_answer}\n"
        f"Predicted answer: {answer}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Return JSON only."
    )
    text, usage = _chat(EVALUATOR_SYSTEM, user, force_json=True)
    try:
        data = json.loads(text)
        judge = JudgeResult(
            score=int(data.get("score", 0)),
            reason=str(data.get("reason", "")),
            missing_evidence=list(data.get("missing_evidence") or []),
            spurious_claims=list(data.get("spurious_claims") or []),
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        judge = JudgeResult(
            score=0,
            reason=f"Evaluator JSON parse failed. Raw: {text[:160]}",
            missing_evidence=[],
            spurious_claims=[],
        )
    # Safety: LLM nói đúng nhưng normalize khác → lật về 0.
    if judge.score == 1 and normalize_answer(answer) != normalize_answer(example.gold_answer):
        judge = judge.model_copy(update={
            "score": 0,
            "reason": f"LLM said correct but normalize mismatch. {judge.reason}",
        })
    return judge, usage


def reflector(
    example: QAExample,
    attempt_id: int,
    judge: JudgeResult,
) -> tuple[ReflectionEntry, Usage]:
    ctx = _format_context(example)
    missing = "; ".join(judge.missing_evidence) or "(none)"
    spurious = "; ".join(judge.spurious_claims) or "(none)"
    user = (
        f"Question: {example.question}\n"
        f"Gold answer: {example.gold_answer}\n"
        f"Attempt #{attempt_id} was WRONG.\n"
        f"Evaluator reason: {judge.reason}\n"
        f"Missing evidence: {missing}\n"
        f"Spurious claims: {spurious}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Return JSON only."
    )
    text, usage = _chat(REFLECTOR_SYSTEM, user, force_json=True)
    try:
        data = json.loads(text)
        entry = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=str(data.get("failure_reason") or judge.reason),
            lesson=str(data.get("lesson") or ""),
            next_strategy=str(data.get("next_strategy") or ""),
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        entry = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="Reflector JSON parse failed; fall back to generic lesson.",
            next_strategy="Re-read every context paragraph carefully and ground the final answer in the supporting sentence.",
        )
    return entry, usage


def classify_failure_mode(
    judge: JudgeResult,
    previous_answer: str = "",
    current_answer: str = "",
) -> str:
    """Suy ra failure_mode từ evaluator output.

    Trả về một trong: none, looping, incomplete_multi_hop, entity_drift,
    wrong_final_answer.
    """
    if judge.score == 1:
        return "none"
    if (
        previous_answer
        and normalize_answer(previous_answer) == normalize_answer(current_answer)
    ):
        return "looping"
    reason = (judge.reason or "").lower()
    if any(kw in reason for kw in ("hop", "first paragraph", "stopped", "incomplete")):
        return "incomplete_multi_hop"
    if judge.spurious_claims:
        return "entity_drift"
    return "wrong_final_answer"
