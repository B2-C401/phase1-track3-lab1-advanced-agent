from __future__ import annotations
import json
from pathlib import Path

import typer
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl

app = typer.Typer(add_completion=False)


def _run_with_progress(label: str, agent, examples: list) -> list:
    records = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(label, total=len(examples))
        for ex in examples:
            records.append(agent.run(ex))
            progress.update(task, advance=1)
    return records


@app.command()
def main(
    dataset: str = "data/hotpot_extra.json",
    out_dir: str = "outputs/real_run",
    reflexion_attempts: int = 3,
    mode: str = "ollama",
) -> None:
    examples = load_dataset(dataset)
    print(f"[cyan]Loaded {len(examples)} examples from {dataset}[/cyan]")

    react = ReActAgent()
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts)

    react_records = _run_with_progress("ReAct   ", react, examples)
    reflexion_records = _run_with_progress("Reflexion", reflexion, examples)

    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    report = build_report(all_records, dataset_name=Path(dataset).name, mode=mode)
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))


if __name__ == "__main__":
    app()
