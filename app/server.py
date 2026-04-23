"""FastAPI server cho Reflexion UI.

Chạy:
    uvicorn app.server:app --reload --port 8000
Sau đó mở http://localhost:8000
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

# Cho phép `from src.reflexion_lab...` khi uvicorn launch từ repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reflexion_lab.utils import load_dataset  # noqa: E402

from .event_bus import bus  # noqa: E402
from .replay import load_saved_runs, record_to_events  # noqa: E402
from .runner import run_live  # noqa: E402

DATASET_PATH = Path(os.getenv("UI_DATASET", "data/hotpot_extra.json"))
RUN_DIR = Path(os.getenv("UI_RUN_DIR", "outputs/real_run"))
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Reflexion Agent UI")

# Preload
_examples = load_dataset(DATASET_PATH)
EXAMPLES_BY_QID = {ex.qid: ex for ex in _examples}
SAVED_RUNS = load_saved_runs(RUN_DIR)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/questions")
async def api_questions() -> list[dict]:
    out: list[dict] = []
    for qid, ex in EXAMPLES_BY_QID.items():
        react = SAVED_RUNS.get((qid, "react"))
        reflexion = SAVED_RUNS.get((qid, "reflexion"))
        interesting = bool(
            react
            and reflexion
            and (not react.is_correct)
            and reflexion.is_correct
        )
        out.append({
            "qid": qid,
            "question": ex.question,
            "gold": ex.gold_answer,
            "difficulty": ex.difficulty,
            "react_correct": react.is_correct if react else None,
            "reflexion_correct": reflexion.is_correct if reflexion else None,
            "reflexion_attempts": reflexion.attempts if reflexion else None,
            "interesting": interesting,
        })
    return out


@app.get("/api/trace/{qid}")
async def api_trace(qid: str) -> dict:
    if qid not in EXAMPLES_BY_QID:
        raise HTTPException(404, "unknown qid")
    result: dict = {}
    for agent in ("react", "reflexion"):
        record = SAVED_RUNS.get((qid, agent))
        if record:
            result[agent] = record_to_events(record)
    return result


@app.post("/api/replay/{qid}")
async def api_replay(qid: str, speed: float = 2.0) -> dict:
    if qid not in EXAMPLES_BY_QID:
        raise HTTPException(404, "unknown qid")
    run_id = bus.create_run()
    base_delay = 0.5 / max(speed, 0.1)

    async def playback() -> None:
        try:
            await asyncio.sleep(0.1)
            for agent in ("react", "reflexion"):
                record = SAVED_RUNS.get((qid, agent))
                if not record:
                    continue
                for event in record_to_events(record):
                    bus.publish(run_id, event)
                    # Evaluator/reflector events nghỉ lâu hơn cho dễ đọc
                    wait = base_delay
                    if event["type"] in ("reflector_end", "run_end"):
                        wait = base_delay * 1.8
                    await asyncio.sleep(wait)
        finally:
            bus.publish(run_id, {"type": "done"})

    asyncio.create_task(playback())
    return {"run_id": run_id, "mode": "replay"}


@app.post("/api/run/{qid}")
async def api_run_live(qid: str) -> dict:
    example = EXAMPLES_BY_QID.get(qid)
    if example is None:
        raise HTTPException(404, "unknown qid")
    run_id = bus.create_run()

    async def task() -> None:
        await asyncio.to_thread(run_live, example, run_id)

    asyncio.create_task(task())
    return {"run_id": run_id, "mode": "live"}


@app.get("/api/stream/{run_id}")
async def api_stream(run_id: str) -> StreamingResponse:
    async def gen():
        async for event in bus.subscribe(run_id):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.get("/api/health")
async def api_health() -> dict:
    return {
        "dataset": str(DATASET_PATH),
        "num_examples": len(EXAMPLES_BY_QID),
        "saved_runs": len(SAVED_RUNS),
    }
