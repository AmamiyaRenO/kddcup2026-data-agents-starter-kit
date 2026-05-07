from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class PublicTaskEval:
    task_id: str
    prediction_written: bool
    artifact_succeeded: bool | None
    selected_candidate_source: str | None
    final_guard_removed_columns: tuple[str, ...]
    postprocess_transformations: tuple[str, ...]
    matched_columns: int
    gold_columns: int
    extra_columns: int
    prediction_rows: int
    gold_rows: int
    prediction_columns: int
    exact_no_extra: bool
    failure_reason: str | None = None

    @property
    def coverage(self) -> float:
        if self.gold_columns == 0:
            return 0.0
        return self.matched_columns / self.gold_columns

    @property
    def estimated_score(self) -> float:
        denominator = self.gold_columns + self.extra_columns
        if denominator == 0:
            return 0.0
        return self.matched_columns / denominator

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prediction_written": self.prediction_written,
            "artifact_succeeded": self.artifact_succeeded,
            "selected_candidate_source": self.selected_candidate_source,
            "final_guard_removed_columns": list(self.final_guard_removed_columns),
            "postprocess_transformations": list(self.postprocess_transformations),
            "matched_columns": self.matched_columns,
            "gold_columns": self.gold_columns,
            "coverage": self.coverage,
            "estimated_score": self.estimated_score,
            "extra_columns": self.extra_columns,
            "prediction_rows": self.prediction_rows,
            "gold_rows": self.gold_rows,
            "prediction_columns": self.prediction_columns,
            "exact_no_extra": self.exact_no_extra,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True, slots=True)
class PublicRunEval:
    run_dir: Path
    gold_root: Path
    tasks: list[PublicTaskEval]

    @property
    def matched_columns(self) -> int:
        return sum(task.matched_columns for task in self.tasks)

    @property
    def gold_columns(self) -> int:
        return sum(task.gold_columns for task in self.tasks)

    @property
    def overall_column_coverage(self) -> float:
        if self.gold_columns == 0:
            return 0.0
        return self.matched_columns / self.gold_columns

    @property
    def exact_no_extra_tasks(self) -> int:
        return sum(1 for task in self.tasks if task.exact_no_extra)

    @property
    def exact_no_extra_rate(self) -> float:
        if not self.tasks:
            return 0.0
        return self.exact_no_extra_tasks / len(self.tasks)

    @property
    def per_task_average_score(self) -> float:
        if not self.tasks:
            return 0.0
        return sum(task.estimated_score for task in self.tasks) / len(self.tasks)

    @property
    def prediction_written_count(self) -> int:
        return sum(1 for task in self.tasks if task.prediction_written)

    @property
    def scorable_task_count(self) -> int:
        return sum(1 for task in self.tasks if task.prediction_written and task.gold_columns > 0)

    @property
    def missing_prediction_count(self) -> int:
        return len(self.tasks) - self.prediction_written_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "gold_root": str(self.gold_root),
            "task_count": len(self.tasks),
            "matched_columns": self.matched_columns,
            "gold_columns": self.gold_columns,
            "overall_column_coverage": self.overall_column_coverage,
            "per_task_average_score": self.per_task_average_score,
            "exact_no_extra_tasks": self.exact_no_extra_tasks,
            "exact_no_extra_rate": self.exact_no_extra_rate,
            "prediction_written_count": self.prediction_written_count,
            "scorable_task_count": self.scorable_task_count,
            "missing_prediction_count": self.missing_prediction_count,
            "tasks": [task.to_dict() for task in self.tasks],
        }


def normalize_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""

    try:
        number = Decimal(text.replace(",", ""))
    except InvalidOperation:
        return text

    if number == number.to_integral_value():
        return str(number.to_integral_value())
    return format(number.quantize(Decimal("0.01")).normalize(), "f")


def column_signatures(frame: pd.DataFrame) -> Counter[tuple[str, ...]]:
    return Counter(
        tuple(sorted(normalize_cell(value) for value in frame.iloc[:, column_index].tolist()))
        for column_index in range(frame.shape[1])
    )


def _artifact_selected_source(artifact_info: dict[str, Any]) -> str | None:
    source = artifact_info.get("selected_candidate_source") or artifact_info.get("candidate_source")
    return str(source) if source else None


def _artifact_final_guard_removed_columns(artifact_info: dict[str, Any]) -> tuple[str, ...]:
    guard = artifact_info.get("final_answer_guard")
    if not isinstance(guard, dict):
        return ()
    removed: list[str] = []
    for transform in guard.get("transformations", []):
        if not isinstance(transform, dict):
            continue
        for column in transform.get("removed_columns", []):
            removed.append(str(column))
    return tuple(dict.fromkeys(removed))


def _artifact_postprocess_transformations(artifact_info: dict[str, Any]) -> tuple[str, ...]:
    postprocessing = artifact_info.get("postprocessing")
    if not isinstance(postprocessing, dict):
        return ()
    kinds: list[str] = []
    for transform in postprocessing.get("transformations", []):
        if isinstance(transform, dict) and transform.get("kind"):
            kinds.append(str(transform["kind"]))
    return tuple(kinds)


def evaluate_public_task(
    prediction_path: Path,
    gold_path: Path,
    task_id: str,
    *,
    artifact_info: dict[str, Any] | None = None,
) -> PublicTaskEval:
    gold = pd.read_csv(gold_path, dtype=str, keep_default_na=False)
    prediction_written = prediction_path.exists()
    artifact_info = artifact_info or {}
    selected_source = _artifact_selected_source(artifact_info)
    final_guard_removed_columns = _artifact_final_guard_removed_columns(artifact_info)
    postprocess_transformations = _artifact_postprocess_transformations(artifact_info)
    if not prediction_written:
        return PublicTaskEval(
            task_id=task_id,
            prediction_written=False,
            artifact_succeeded=artifact_info.get("succeeded"),
            selected_candidate_source=selected_source,
            final_guard_removed_columns=final_guard_removed_columns,
            postprocess_transformations=postprocess_transformations,
            matched_columns=0,
            gold_columns=gold.shape[1],
            extra_columns=0,
            prediction_rows=0,
            gold_rows=len(gold),
            prediction_columns=0,
            exact_no_extra=False,
            failure_reason=artifact_info.get("failure_reason"),
        )

    prediction = pd.read_csv(prediction_path, dtype=str, keep_default_na=False)

    prediction_signatures = column_signatures(prediction)
    gold_signatures = column_signatures(gold)
    matched_columns = sum((prediction_signatures & gold_signatures).values())
    extra_columns = sum((prediction_signatures - gold_signatures).values())

    return PublicTaskEval(
        task_id=task_id,
        prediction_written=True,
        artifact_succeeded=artifact_info.get("succeeded"),
        selected_candidate_source=selected_source,
        final_guard_removed_columns=final_guard_removed_columns,
        postprocess_transformations=postprocess_transformations,
        matched_columns=matched_columns,
        gold_columns=gold.shape[1],
        extra_columns=extra_columns,
        prediction_rows=len(prediction),
        gold_rows=len(gold),
        prediction_columns=prediction.shape[1],
        exact_no_extra=matched_columns == gold.shape[1] and extra_columns == 0,
        failure_reason=artifact_info.get("failure_reason"),
    )


def _artifact_index(run_dir: Path) -> dict[str, dict[str, Any]]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list):
        return {}
    return {
        str(item.get("task_id")): item
        for item in tasks
        if isinstance(item, dict) and item.get("task_id")
    }


def evaluate_public_run(run_dir: Path, gold_root: Path) -> PublicRunEval:
    tasks: list[PublicTaskEval] = []
    artifact_by_task = _artifact_index(run_dir)
    task_ids = {
        path.name
        for path in run_dir.iterdir()
        if path.is_dir() and path.name.startswith("task_")
    }
    task_ids.update(artifact_by_task)
    for task_id in sorted(task_ids):
        if not task_id.startswith("task_"):
            continue
        prediction_path = run_dir / task_id / "prediction.csv"
        gold_path = gold_root / task_id / "gold.csv"
        if not gold_path.exists():
            continue
        tasks.append(
            evaluate_public_task(
                prediction_path,
                gold_path,
                task_id,
                artifact_info=artifact_by_task.get(task_id),
            )
        )

    return PublicRunEval(run_dir=run_dir, gold_root=gold_root, tasks=tasks)
