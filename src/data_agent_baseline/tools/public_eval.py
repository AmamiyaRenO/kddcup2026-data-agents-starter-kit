from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class PublicTaskEval:
    task_id: str
    matched_columns: int
    gold_columns: int
    extra_columns: int
    prediction_rows: int
    gold_rows: int
    prediction_columns: int
    exact_no_extra: bool

    @property
    def coverage(self) -> float:
        if self.gold_columns == 0:
            return 0.0
        return self.matched_columns / self.gold_columns

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "matched_columns": self.matched_columns,
            "gold_columns": self.gold_columns,
            "coverage": self.coverage,
            "extra_columns": self.extra_columns,
            "prediction_rows": self.prediction_rows,
            "gold_rows": self.gold_rows,
            "prediction_columns": self.prediction_columns,
            "exact_no_extra": self.exact_no_extra,
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "gold_root": str(self.gold_root),
            "task_count": len(self.tasks),
            "matched_columns": self.matched_columns,
            "gold_columns": self.gold_columns,
            "overall_column_coverage": self.overall_column_coverage,
            "exact_no_extra_tasks": self.exact_no_extra_tasks,
            "exact_no_extra_rate": self.exact_no_extra_rate,
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


def evaluate_public_task(prediction_path: Path, gold_path: Path, task_id: str) -> PublicTaskEval:
    prediction = pd.read_csv(prediction_path, dtype=str, keep_default_na=False)
    gold = pd.read_csv(gold_path, dtype=str, keep_default_na=False)

    prediction_signatures = column_signatures(prediction)
    gold_signatures = column_signatures(gold)
    matched_columns = sum((prediction_signatures & gold_signatures).values())
    extra_columns = sum((prediction_signatures - gold_signatures).values())

    return PublicTaskEval(
        task_id=task_id,
        matched_columns=matched_columns,
        gold_columns=gold.shape[1],
        extra_columns=extra_columns,
        prediction_rows=len(prediction),
        gold_rows=len(gold),
        prediction_columns=prediction.shape[1],
        exact_no_extra=matched_columns == gold.shape[1] and extra_columns == 0,
    )


def evaluate_public_run(run_dir: Path, gold_root: Path) -> PublicRunEval:
    tasks: list[PublicTaskEval] = []
    for task_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        if not task_dir.name.startswith("task_"):
            continue
        prediction_path = task_dir / "prediction.csv"
        gold_path = gold_root / task_dir.name / "gold.csv"
        if not prediction_path.exists() or not gold_path.exists():
            continue
        tasks.append(evaluate_public_task(prediction_path, gold_path, task_dir.name))

    return PublicRunEval(run_dir=run_dir, gold_root=gold_root, tasks=tasks)
