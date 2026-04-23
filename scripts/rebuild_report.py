"""Rebuild report.json + report.md từ react_runs.jsonl + reflexion_runs.jsonl.

Dùng khi đã chạy benchmark tốn kém rồi nhưng logic `build_report` / discussion
thay đổi — tránh phải re-run agents.
"""
from __future__ import annotations
import sys
from pathlib import Path

# Cho phép chạy từ repo root: `python scripts/rebuild_report.py outputs/real_run`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer

from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.schemas import RunRecord

app = typer.Typer(add_completion=False)


@app.command()
def main(
    run_dir: str = "outputs/real_run",
    dataset_name: str = "hotpot_extra.json",
    mode: str = "ollama",
) -> None:
    base = Path(run_dir)
    records: list[RunRecord] = []
    for name in ["react_runs.jsonl", "reflexion_runs.jsonl"]:
        path = base / name
        if not path.exists():
            raise typer.BadParameter(f"Missing {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(RunRecord.model_validate_json(line))
    print(f"Loaded {len(records)} records from {base}")

    report = build_report(records, dataset_name=dataset_name, mode=mode)
    json_path, md_path = save_report(report, base)
    print(f"Rebuilt {json_path}")
    print(f"Rebuilt {md_path}")


if __name__ == "__main__":
    app()
