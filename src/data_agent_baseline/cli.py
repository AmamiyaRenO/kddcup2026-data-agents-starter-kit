from pathlib import Path
from time import perf_counter

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import load_app_config
from data_agent_baseline.run.databao_demo import (
    create_databao_local_run_dir,
    load_databao_environment,
    run_databao_tasks,
)
from data_agent_baseline.run.runner import TaskRunArtifacts, create_run_output_dir, run_benchmark, run_single_task
from data_agent_baseline.tools.filesystem import list_context_tree
from data_agent_baseline.tools.public_eval import evaluate_public_run

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_RUNS_DIR = ARTIFACTS_DIR / "runs"
DATABAO_DEMO_RUNS_DIR = ARTIFACTS_DIR / "databao-demo"

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()


def _status_value(path: Path) -> str:
    return "present" if path.exists() else "missing"


def _format_compact_rate(completed_count: int, elapsed_seconds: float) -> str:
    if completed_count <= 0 or elapsed_seconds <= 0:
        return "rate=0.0 task/min"
    return f"rate={(completed_count / elapsed_seconds) * 60:.1f} task/min"


def _format_last_task(artifact: TaskRunArtifacts | None) -> str:
    if artifact is None:
        return "last=-"
    status = "ok" if artifact.succeeded else "fail"
    return f"last={artifact.task_id} ({status})"


def _build_compact_progress_fields(
    *,
    completed_count: int,
    succeeded_count: int,
    failed_count: int,
    task_total: int,
    max_workers: int,
    elapsed_seconds: float,
    last_artifact: TaskRunArtifacts | None,
) -> dict[str, str]:
    remaining_count = max(task_total - completed_count, 0)
    running_count = min(max_workers, remaining_count)
    queued_count = max(remaining_count - running_count, 0)
    return {
        "ok": str(succeeded_count),
        "fail": str(failed_count),
        "run": str(running_count),
        "queue": str(queued_count),
        "speed": _format_compact_rate(completed_count, elapsed_seconds),
        "last": _format_last_task(last_artifact),
    }


@app.callback()
def cli() -> None:
    """Utilities for working with the local DABench baseline project."""


@app.command()
def status(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Show the local project layout and public dataset presence."""
    app_config = load_app_config(config)
    config_path = config.resolve()
    public_dataset = DABenchPublicDataset(app_config.dataset.root_path)

    table = Table(title="DABench Baseline Status")
    table.add_column("Item")
    table.add_column("Path")
    table.add_column("State")

    table.add_row("project_root", str(PROJECT_ROOT), "ready")
    table.add_row("data_dir", str(DATA_DIR), _status_value(DATA_DIR))
    table.add_row("configs_dir", str(CONFIGS_DIR), _status_value(CONFIGS_DIR))
    table.add_row("artifacts_dir", str(ARTIFACTS_DIR), _status_value(ARTIFACTS_DIR))
    table.add_row("runs_dir", str(ARTIFACT_RUNS_DIR), _status_value(ARTIFACT_RUNS_DIR))
    table.add_row("dataset_root", str(app_config.dataset.root_path), _status_value(app_config.dataset.root_path))
    table.add_row("config_path", str(config_path), _status_value(config_path))

    console.print(table)

    if public_dataset.exists:
        console.print(f"Public tasks: {len(public_dataset.list_task_ids())}")
        counts = public_dataset.task_counts()
        if counts:
            rendered_counts = ", ".join(
                f"{difficulty}={count}" for difficulty, count in sorted(counts.items())
            )
            console.print(f"Public task counts: {rendered_counts}")


@app.command("inspect-task")
def inspect_task(
    task_id: str,
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Show task metadata and available context files."""
    app_config = load_app_config(config)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    task = dataset.get_task(task_id)
    console.print(f"Task: {task.task_id}")
    console.print(f"Difficulty: {task.difficulty}")
    console.print(f"Question: {task.question}")
    context_listing = list_context_tree(task)
    table = Table(title=f"Context Files for {task.task_id}")
    table.add_column("Path")
    table.add_column("Kind")
    table.add_column("Size")
    for entry in context_listing["entries"]:
        table.add_row(str(entry["path"]), str(entry["kind"]), str(entry["size"] or ""))
    console.print(table)


@app.command("run-task")
def run_task_command(
    task_id: str,
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
) -> None:
    """Run the ReAct baseline on one task."""
    app_config = load_app_config(config)
    try:
        _, run_output_dir = create_run_output_dir(app_config.run.output_dir, run_id=app_config.run.run_id)
    except (ValueError, FileExistsError) as exc:
        raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc
    artifacts = run_single_task(task_id=task_id, config=app_config, run_output_dir=run_output_dir)

    console.print(f"Run output: {run_output_dir}")
    console.print(f"Task output: {artifacts.task_output_dir}")
    if artifacts.prediction_csv_path is not None:
        console.print(f"Prediction CSV: {artifacts.prediction_csv_path}")
    else:
        console.print("Prediction CSV: not generated")
    if artifacts.failure_reason is not None:
        console.print(f"Failure: {artifacts.failure_reason}")


@app.command("run-benchmark")
def run_benchmark_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    limit: int | None = typer.Option(None, min=1, help="Maximum number of tasks to run."),
) -> None:
    """Run the ReAct baseline on multiple tasks from the config selection."""
    app_config = load_app_config(config)
    dataset = DABenchPublicDataset(app_config.dataset.root_path)
    task_total = len(dataset.iter_tasks())
    if limit is not None:
        task_total = min(task_total, limit)
    effective_workers = app_config.run.max_workers

    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("[green]ok={task.fields[ok]}[/green]"),
        TextColumn("[red]fail={task.fields[fail]}[/red]"),
        TextColumn("[cyan]run={task.fields[run]}[/cyan]"),
        TextColumn("[yellow]queue={task.fields[queue]}[/yellow]"),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[speed]}"),
        TextColumn("[dim]| elapsed[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]| eta[/dim]"),
        TimeRemainingColumn(),
        TextColumn("[dim]|[/dim]"),
        TextColumn("{task.fields[last]}"),
    ]
    with Progress(*progress_columns, console=console) as progress:
        progress_task_id = progress.add_task(
            "Benchmark",
            total=task_total,
            completed=0,
            **_build_compact_progress_fields(
                completed_count=0,
                succeeded_count=0,
                failed_count=0,
                task_total=task_total,
                max_workers=effective_workers,
                elapsed_seconds=0.0,
                last_artifact=None,
            ),
        )

        completion_count = 0
        succeeded_count = 0
        failed_count = 0
        start_time = perf_counter()

        def on_task_complete(artifact) -> None:
            nonlocal completion_count, succeeded_count, failed_count
            completion_count += 1
            if artifact.succeeded:
                succeeded_count += 1
            else:
                failed_count += 1
            progress.update(
                progress_task_id,
                completed=completion_count,
                description="Benchmark",
                refresh=True,
                **_build_compact_progress_fields(
                    completed_count=completion_count,
                    succeeded_count=succeeded_count,
                    failed_count=failed_count,
                    task_total=task_total,
                    max_workers=effective_workers,
                    elapsed_seconds=perf_counter() - start_time,
                    last_artifact=artifact,
                ),
            )

        try:
            run_output_dir, artifacts = run_benchmark(
                config=app_config,
                limit=limit,
                progress_callback=on_task_complete,
            )
        except (ValueError, FileExistsError) as exc:
            raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc
        progress.update(
            progress_task_id,
            completed=task_total,
            description="Benchmark",
            refresh=True,
            **_build_compact_progress_fields(
                completed_count=task_total,
                succeeded_count=succeeded_count,
                failed_count=failed_count,
                task_total=task_total,
                max_workers=effective_workers,
                elapsed_seconds=perf_counter() - start_time,
                last_artifact=artifacts[-1] if artifacts else None,
            ),
        )
    console.print(f"Run output: {run_output_dir}")
    console.print(f"Tasks attempted: {len(artifacts)}")
    console.print(f"Succeeded tasks: {sum(1 for item in artifacts if item.succeeded)}")


@app.command("run-databao-demo")
def run_databao_demo_command(
    config: Path = typer.Option(..., exists=True, dir_okay=False, help="YAML config path."),
    limit: int | None = typer.Option(None, min=1, help="Maximum number of tasks to run."),
    task_id: str | None = typer.Option(None, help="Run one specific task id, e.g. task_25."),
    difficulty: str | None = typer.Option(
        None,
        help="Run only tasks with this difficulty, e.g. easy, medium, hard, or extreme.",
    ),
    task_timeout_seconds: int | None = typer.Option(
        None,
        min=1,
        help="Override run.task_timeout_seconds for the Databao demo runner.",
    ),
    planner_mode: str | None = typer.Option(
        None,
        help="Structured planner mode: fallback, first, or off. Default is Databao-first fallback.",
    ),
    finalizer_mode: str | None = typer.Option(
        None,
        help="LLM finalizer mode: off or always. Default is off.",
    ),
) -> None:
    """Run the Databao-backed demo runner on local public tasks."""
    app_config = load_app_config(config)
    try:
        databao_env = load_databao_environment()
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc), param_hint="environment") from exc

    try:
        run_id, run_output_dir = create_databao_local_run_dir(
            DATABAO_DEMO_RUNS_DIR,
            run_id=app_config.run.run_id,
        )
    except (ValueError, FileExistsError) as exc:
        raise typer.BadParameter(str(exc), param_hint="run.run_id") from exc

    logs_dir = run_output_dir / "logs"
    effective_task_timeout_seconds = task_timeout_seconds or app_config.run.task_timeout_seconds
    console.print(f"Databao demo run: {run_id}")
    console.print(f"Input root: {app_config.dataset.root_path}")
    console.print(f"Run output: {run_output_dir}")
    console.print(f"Logs: {logs_dir}")
    console.print(f"Task timeout: {effective_task_timeout_seconds}s")
    console.print(f"Difficulty: {difficulty or 'all'}")
    console.print(f"Planner mode: {planner_mode or 'fallback'}")
    console.print(f"Finalizer mode: {finalizer_mode or 'off'}")

    artifacts = run_databao_tasks(
        input_root=app_config.dataset.root_path,
        output_root=run_output_dir,
        logs_dir=logs_dir,
        limit=limit,
        task_ids=[task_id] if task_id else None,
        difficulty=difficulty,
        task_timeout_seconds=effective_task_timeout_seconds,
        databao_env=databao_env,
        structured_planner_mode=planner_mode,
        finalizer_mode=finalizer_mode,
    )
    succeeded_count = sum(1 for artifact in artifacts if artifact.succeeded)
    prediction_written_count = sum(1 for artifact in artifacts if artifact.prediction_written)
    scorable_count = sum(1 for artifact in artifacts if artifact.scorable)
    console.print(f"Attempted tasks: {len(artifacts)}")
    console.print(f"Succeeded tasks: {succeeded_count}")
    console.print(f"Failed tasks: {len(artifacts) - succeeded_count}")
    console.print(f"Prediction written: {prediction_written_count}")
    console.print(f"Scorable tasks: {scorable_count}")
    console.print(f"Missing prediction: {len(artifacts) - prediction_written_count}")


@app.command("evaluate-public-run")
def evaluate_public_run_command(
    run_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        help="Run directory containing task outputs.",
    ),
    gold_root: Path = typer.Option(
        DATA_DIR / "public" / "output",
        exists=True,
        file_okay=False,
        help="Gold output root.",
    ),
) -> None:
    """Evaluate a public demo run with a local column-signature approximation."""
    evaluation = evaluate_public_run(run_dir, gold_root)

    table = Table(title="Public Run Evaluation")
    table.add_column("Task")
    table.add_column("Pred")
    table.add_column("Artifact")
    table.add_column("Matched")
    table.add_column("Score")
    table.add_column("Extra")
    table.add_column("Rows")
    table.add_column("Cols")
    table.add_column("Source")
    table.add_column("GuardRm")
    table.add_column("Post")
    table.add_column("Exact")
    for task in evaluation.tasks:
        removed_columns = ",".join(task.final_guard_removed_columns)
        postprocess = ",".join(task.postprocess_transformations)
        table.add_row(
            task.task_id,
            "yes" if task.prediction_written else "no",
            (
                "-"
                if task.artifact_succeeded is None
                else "ok"
                if task.artifact_succeeded
                else "fail"
            ),
            f"{task.matched_columns}/{task.gold_columns}",
            f"{task.estimated_score:.3f}",
            str(task.extra_columns),
            f"{task.prediction_rows}/{task.gold_rows}",
            f"{task.prediction_columns}/{task.gold_columns}",
            task.selected_candidate_source or "-",
            removed_columns[:40] if removed_columns else "-",
            postprocess[:40] if postprocess else "-",
            "yes" if task.exact_no_extra else "no",
        )
    console.print(table)
    console.print(
        "PER_TASK_AVERAGE_SCORE="
        f"{evaluation.per_task_average_score:.3f}"
    )
    console.print(
        "OVERALL_COLUMN_COVERAGE="
        f"{evaluation.matched_columns}/{evaluation.gold_columns}="
        f"{evaluation.overall_column_coverage:.3f}"
    )
    console.print(
        "EXACT_NO_EXTRA_TASKS="
        f"{evaluation.exact_no_extra_tasks}/{len(evaluation.tasks)}="
        f"{evaluation.exact_no_extra_rate:.3f}"
    )
    console.print(f"PREDICTION_WRITTEN={evaluation.prediction_written_count}/{len(evaluation.tasks)}")
    console.print(f"SCORABLE_TASKS={evaluation.scorable_task_count}/{len(evaluation.tasks)}")
    console.print(f"MISSING_PREDICTION={evaluation.missing_prediction_count}/{len(evaluation.tasks)}")


def main() -> None:
    app()
