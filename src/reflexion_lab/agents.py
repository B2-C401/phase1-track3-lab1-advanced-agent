from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Literal, Optional

from .mock_runtime import (
    actor_answer,
    classify_failure_mode,
    evaluator,
    reflector,
)
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord

EmitFn = Callable[[dict], None]


@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1

    def run(self, example: QAExample, emit: Optional[EmitFn] = None) -> RunRecord:
        def _emit(event: dict) -> None:
            if emit:
                event.setdefault("agent", self.agent_type)
                emit(event)

        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        previous_answer = ""
        failure_mode = "wrong_final_answer"

        for attempt_id in range(1, self.max_attempts + 1):
            _emit({"type": "attempt_start", "attempt": attempt_id})

            answer, u_actor = actor_answer(
                example, attempt_id, self.agent_type, reflection_memory
            )
            _emit({
                "type": "actor_end",
                "attempt": attempt_id,
                "answer": answer,
                "tokens": u_actor.total_tokens,
                "latency_ms": u_actor.latency_ms,
            })

            judge, u_eval = evaluator(example, answer)
            _emit({
                "type": "evaluator_end",
                "attempt": attempt_id,
                "score": judge.score,
                "reason": judge.reason,
                "missing_evidence": judge.missing_evidence,
                "spurious_claims": judge.spurious_claims,
                "tokens": u_eval.total_tokens,
                "latency_ms": u_eval.latency_ms,
            })

            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                token_estimate=u_actor.total_tokens + u_eval.total_tokens,
                latency_ms=u_actor.latency_ms + u_eval.latency_ms,
            )
            final_answer = answer
            final_score = judge.score

            if judge.score == 1:
                _emit({"type": "attempt_end", "attempt": attempt_id, "correct": True})
                traces.append(trace)
                break

            failure_mode = classify_failure_mode(judge, previous_answer, answer)

            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection, u_refl = reflector(example, attempt_id, judge)
                _emit({
                    "type": "reflector_end",
                    "attempt": attempt_id,
                    "lesson": reflection.lesson,
                    "strategy": reflection.next_strategy,
                    "failure_reason": reflection.failure_reason,
                    "tokens": u_refl.total_tokens,
                    "latency_ms": u_refl.latency_ms,
                })
                trace.reflection = reflection
                trace.token_estimate += u_refl.total_tokens
                trace.latency_ms += u_refl.latency_ms
                reflections.append(reflection)
                reflection_memory.append(
                    f"Lesson: {reflection.lesson}\nNext strategy: {reflection.next_strategy}"
                )

            _emit({"type": "attempt_end", "attempt": attempt_id, "correct": False})
            previous_answer = answer
            traces.append(trace)

        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        if final_score == 1:
            failure_mode = "none"

        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )


class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)


class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
