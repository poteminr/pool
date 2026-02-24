from __future__ import annotations

import json
import os
from importlib.resources import files
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn

from pool_cli.costing import estimate_openai_cost_usd, format_cost_usd, resolve_model_pricing  # noqa: F401
from pool_cli.core import PipelineConfig, format_duration, run_pipeline, write_json_report
from pool_cli.openai_api import is_openai_available


DEFAULT_CACHE_DIR = Path.home() / ".pool" / "cache"
DEFAULT_REPORT_SNAPSHOT_NAME = "latest_report.json"
RUNTIME_DEFAULTS_RESOURCE = "runtime_defaults.json"


console = Console()


def _load_runtime_defaults() -> dict[str, object]:
    payload_text = files("pool_cli").joinpath(RUNTIME_DEFAULTS_RESOURCE).read_text(encoding="utf-8")
    return json.loads(payload_text)


def run_command(
    input_path: Path = typer.Argument(..., help="Folder with screenshots"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir", help="SQLite cache directory"),
    max_images: int | None = typer.Option(None, "--max-images", help="Optional cap for debugging"),
    save_report: Path | None = typer.Option(
        None,
        "--save-report",
        help="Optional path for JSON snapshot. Default: <cache-dir>/latest_report.json",
    ),
) -> None:
    input_path = input_path.expanduser().resolve()
    cache_dir = cache_dir.expanduser().resolve()
    report_path = (
        save_report.expanduser().resolve()
        if save_report is not None
        else (cache_dir / DEFAULT_REPORT_SNAPSHOT_NAME).resolve()
    )

    if not input_path.exists() or not input_path.is_dir():
        console.print(f"Input path is not a directory: {input_path}", style="bold red")
        raise typer.Exit(code=2)

    runtime_payload = dict(_load_runtime_defaults())
    runtime_payload["input_path"] = input_path
    runtime_payload["cache_dir"] = cache_dir
    runtime_payload["max_images"] = max_images
    config = PipelineConfig(**runtime_payload)

    if not is_openai_available():
        console.print(
            "OPENAI_API_KEY is not set. "
            "User pool discovery and action suggestions will be disabled.",
            style="yellow",
        )

    run_progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    progress_task_ids: dict[str, int] = {}
    progress_labels = {
        "embedding": "Analyzing images",
        "classifying": "Classifying into pools",
    }
    status_task_id: int | None = None

    def _progress_update(
        kind: str,
        completed: int,
        total: int,
        _usage: dict[str, int] | None,
    ) -> None:
        nonlocal status_task_id
        if total <= 0:
            return
        # Hide spinner when a real progress bar starts
        if status_task_id is not None:
            run_progress.update(status_task_id, visible=False)
            status_task_id = None
        task_id = progress_task_ids.get(kind)
        if task_id is None:
            task_id = run_progress.add_task(
                progress_labels.get(kind, kind),
                total=total,
            )
            progress_task_ids[kind] = task_id

        run_progress.update(
            task_id,
            total=total,
            completed=min(completed, total),
        )

    def _status_update(message: str) -> None:
        nonlocal status_task_id
        # Hide any previous progress bars
        for tid in progress_task_ids.values():
            run_progress.update(tid, visible=False)
        if status_task_id is not None:
            run_progress.update(status_task_id, description=message, visible=True)
        else:
            status_task_id = run_progress.add_task(message, total=None)

    with run_progress:
        result = run_pipeline(
            config=config,
            log=lambda message: None,
            progress=_progress_update,
            status=_status_update,
        )

    print_report(result.report)
    write_json_report(result.report, report_path)
    console.print(f"Saved report snapshot: {report_path}")


def print_report(report) -> None:
    console.print("pool")
    console.print(f"Analyzing your screenshots… {format_duration(report.stage_timings.get('analyzing', 0.0))}")
    console.print(f"Generating suggestions… {format_duration(report.stage_timings.get('suggesting', 0.0))}")
    usage = report.openai_usage
    if usage:
        pricing = resolve_model_pricing(model=report.model)
        est_cost_usd = estimate_openai_cost_usd(usage, pricing=pricing)
        console.print(
            "OpenAI usage: "
            f"in={int(usage.get('input_tokens', 0))}, "
            f"cached={int(usage.get('cached_input_tokens', 0))}, "
            f"out={int(usage.get('output_tokens', 0))}, "
            f"est_cost={format_cost_usd(est_cost_usd)}"
        )
    console.print("Ready.\n")

    console.print("Pools suggested\n")
    for pool in report.pools:
        if pool.match_count == 0:
            continue
        console.print(f"[bold]{pool.pool_name}[/bold] — {pool.match_count} matches")
        console.print(f"Action: \"{pool.action.title}\"" if pool.action else "Action: (none)")
        if pool.action:
            console.print(f"  {pool.action.why}")
        console.print(f"Notes: {pool.notes}")
        if pool.top_matches:
            console.print("Top matches:")
            for match in pool.top_matches[:10]:
                console.print(f"  - {Path(match.file_path).name}")
        console.print("")


def pool_entry() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    typer.run(run_command)


if __name__ == "__main__":
    pool_entry()
