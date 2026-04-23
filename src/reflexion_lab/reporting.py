from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from .schemas import ReportPayload, RunRecord

def summarize(records: list[RunRecord]) -> dict:
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[record.agent_type].append(record)
    summary: dict[str, dict] = {}
    for agent_type, rows in grouped.items():
        summary[agent_type] = {"count": len(rows), "em": round(mean(1.0 if r.is_correct else 0.0 for r in rows), 4), "avg_attempts": round(mean(r.attempts for r in rows), 4), "avg_token_estimate": round(mean(r.token_estimate for r in rows), 2), "avg_latency_ms": round(mean(r.latency_ms for r in rows), 2)}
    if "react" in summary and "reflexion" in summary:
        summary["delta_reflexion_minus_react"] = {"em_abs": round(summary["reflexion"]["em"] - summary["react"]["em"], 4), "attempts_abs": round(summary["reflexion"]["avg_attempts"] - summary["react"]["avg_attempts"], 4), "tokens_abs": round(summary["reflexion"]["avg_token_estimate"] - summary["react"]["avg_token_estimate"], 2), "latency_abs": round(summary["reflexion"]["avg_latency_ms"] - summary["react"]["avg_latency_ms"], 2)}
    return summary

def failure_breakdown(records: list[RunRecord]) -> dict:
    grouped: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        grouped[record.agent_type][record.failure_mode] += 1
        grouped["all"][record.failure_mode] += 1
    return {agent: dict(counter) for agent, counter in grouped.items()}


def _build_discussion(summary: dict, failure_modes: dict) -> str:
    react = summary.get("react", {})
    reflexion = summary.get("reflexion", {})
    delta = summary.get("delta_reflexion_minus_react", {})
    react_em = react.get("em", 0)
    reflexion_em = reflexion.get("em", 0)
    em_delta = delta.get("em_abs", 0)
    token_ratio = (
        reflexion.get("avg_token_estimate", 0) / react.get("avg_token_estimate", 1)
        if react.get("avg_token_estimate") else 0
    )
    latency_ratio = (
        reflexion.get("avg_latency_ms", 0) / react.get("avg_latency_ms", 1)
        if react.get("avg_latency_ms") else 0
    )
    react_fm = failure_modes.get("react", {})
    reflexion_fm = failure_modes.get("reflexion", {})

    # Xác định failure mode chính còn sót ở Reflexion (ngoài "none").
    remaining = {k: v for k, v in reflexion_fm.items() if k != "none"}
    top_remaining = (
        max(remaining.items(), key=lambda kv: kv[1]) if remaining else ("none", 0)
    )

    return (
        f"Reflexion đạt EM {reflexion_em:.2f} so với ReAct {react_em:.2f} "
        f"(delta {em_delta:+.2f}) trên {react.get('count', 0)} câu HotpotQA. "
        f"Cái giá phải trả: trung bình {reflexion.get('avg_attempts', 0):.2f} "
        f"attempts/câu (ReAct luôn là 1), tokens tăng {token_ratio:.2f}× và "
        f"latency tăng {latency_ratio:.2f}×. Phân tích failure_modes cho thấy "
        f"ReAct phân bổ lỗi như sau: {dict(react_fm)}, trong khi Reflexion còn "
        f"lại: {dict(reflexion_fm)}. Sau reflection, lỗi phổ biến nhất chưa xử lý "
        f"được là '{top_remaining[0]}' ({top_remaining[1]} câu) — cho thấy "
        f"evaluator đôi khi không chỉ ra đúng thiếu sót, hoặc reflector đưa ra "
        f"chiến thuật quá chung chung nên actor vẫn lặp lại lỗi. Về mặt vận hành, "
        f"với budget đủ dư, đánh đổi này hợp lý cho câu hỏi multi-hop. Với budget "
        f"khắt khe, cần adaptive_max_attempts hoặc memory_compression để giảm chi "
        f"phí. Model sử dụng là qwen2.5:3b-instruct qua Ollama; chất lượng "
        f"evaluator/reflector giới hạn bởi kích thước model."
    )


def build_report(records: list[RunRecord], dataset_name: str, mode: str = "mock") -> ReportPayload:
    examples = [{"qid": r.qid, "agent_type": r.agent_type, "gold_answer": r.gold_answer, "predicted_answer": r.predicted_answer, "is_correct": r.is_correct, "attempts": r.attempts, "failure_mode": r.failure_mode, "reflection_count": len(r.reflections)} for r in records]
    summary = summarize(records)
    failure_modes = failure_breakdown(records)
    return ReportPayload(
        meta={"dataset": dataset_name, "mode": mode, "num_records": len(records), "agents": sorted({r.agent_type for r in records})},
        summary=summary,
        failure_modes=failure_modes,
        examples=examples,
        extensions=["structured_evaluator", "reflection_memory", "benchmark_report_json"],
        discussion=_build_discussion(summary, failure_modes),
    )

def save_report(report: ReportPayload, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    json_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    s = report.summary
    react = s.get("react", {})
    reflexion = s.get("reflexion", {})
    delta = s.get("delta_reflexion_minus_react", {})
    ext_lines = "\n".join(f"- {item}" for item in report.extensions)
    md = f"""# Lab 16 Benchmark Report

## Metadata
- Dataset: {report.meta['dataset']}
- Mode: {report.meta['mode']}
- Records: {report.meta['num_records']}
- Agents: {', '.join(report.meta['agents'])}

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | {react.get('em', 0)} | {reflexion.get('em', 0)} | {delta.get('em_abs', 0)} |
| Avg attempts | {react.get('avg_attempts', 0)} | {reflexion.get('avg_attempts', 0)} | {delta.get('attempts_abs', 0)} |
| Avg token estimate | {react.get('avg_token_estimate', 0)} | {reflexion.get('avg_token_estimate', 0)} | {delta.get('tokens_abs', 0)} |
| Avg latency (ms) | {react.get('avg_latency_ms', 0)} | {reflexion.get('avg_latency_ms', 0)} | {delta.get('latency_abs', 0)} |

## Failure modes
```json
{json.dumps(report.failure_modes, indent=2)}
```

## Extensions implemented
{ext_lines}

## Discussion
{report.discussion}
"""
    md_path.write_text(md, encoding="utf-8")
    return json_path, md_path
