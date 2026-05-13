from __future__ import annotations

import json
import multiprocessing
import os
import queue
import re
import sqlite3
import traceback
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
from langchain_core.callbacks import BaseCallbackHandler
from sqlalchemy import create_engine

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.run.runner import create_run_id, resolve_run_id

MODEL_API_URL_ENV = "MODEL_API_URL"
MODEL_API_KEY_ENV = "MODEL_API_KEY"
MODEL_NAME_ENV = "MODEL_NAME"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
DATABAO_DEBUG_LOG_RAW_ENV = "DATABAO_DEBUG_LOG_RAW"
DATABAO_TASK_TIMEOUT_SECONDS_ENV = "DATABAO_TASK_TIMEOUT_SECONDS"
DATABAO_DATABAO_TIMEOUT_SECONDS_ENV = "DATABAO_DATABAO_TIMEOUT_SECONDS"
DATABAO_HEURISTIC_LEVEL_ENV = "DATABAO_HEURISTIC_LEVEL"
DATABAO_EXECUTOR_TYPE_ENV = "DATABAO_EXECUTOR_TYPE"

SOURCE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_]+")
MAX_DESCRIPTION_CHARS = 20000
MAX_CANDIDATE_PAYLOAD_ROWS = 120
MAX_CANDIDATE_CELL_CHARS = 500
MAX_CANDIDATE_PAYLOAD_CHARS = 45000
MAX_DATABAO_TOKENS = 2048
MAX_DATABAO_CODE_CHARS = 8000
DATABAO_AGENT_TIMEOUT_SECONDS = 100
MAX_FAILURE_REASON_CHARS = 1000
MAX_CONTEXT_ENRICHMENT_ROWS = 80
MAX_CONTEXT_METRIC_SUMMARIES = 8
MAX_SCHEMA_SAMPLE_ROWS = 3
MAX_SCHEMA_PROFILE_SAMPLE_ROWS = 512
RECORD_ID_PATTERN = re.compile(r"^rec[A-Za-z0-9]+$")
HEURISTIC_LEVELS = {"generic", "experimental_generic"}
DATABAO_EXECUTOR_TYPES = {"lighthouse", "lighthouse_salvage"}
_CURRENT_DIAGNOSTICS: ContextVar["TaskDiagnostics | None"] = ContextVar(
    "databao_task_diagnostics",
    default=None,
)
_CURRENT_RETRIEVED_CONTEXT: ContextVar["RetrievedContext | None"] = ContextVar(
    "databao_retrieved_context",
    default=None,
)
_CURRENT_TASK_DEADLINE: ContextVar[float | None] = ContextVar(
    "databao_task_deadline",
    default=None,
)

DISPLAY_COLUMN_PRIORITY = (
    "name",
    "title",
    "label",
    "display_name",
    "url",
    "link",
    "website",
    "text",
    "value",
    "category",
    "status",
    "description",
)

DEBUG_METADATA_COLUMNS = {
    "source_doc",
    "evidence_span",
    "confidence",
    "paragraph_index",
    "extraction_strategy",
    "strategy_kind",
    "record_id_kind",
    "debug",
    "trace",
}

HIGH_RISK_VERIFIER_KINDS = {
    "context_superlative_verification",
    "aggregate_ratio_verification",
    "superlative_verification",
}

DOCUMENT_AGENT_SAFE_BASE_COLUMNS = {
    "record_id",
    "numeric_id",
    "paragraph_id",
    "name",
    "title",
    "label",
    "description",
    "status",
    "type",
    "category",
    "format",
    "amount",
    "cost",
    "price",
    "score",
    "value",
    "height",
    "total",
    "count",
    "date",
    "year",
    "season",
    "created_at",
    "related_id",
}

MIN_SUPERLATIVE_TERMS = (
    "lowest",
    "least",
    "minimum",
    "smallest",
    "cheapest",
    "bottom",
)
MAX_SUPERLATIVE_TERMS = (
    "highest",
    "greatest",
    "largest",
    "maximum",
    "most",
    "top",
)
METRIC_TERM_ALIASES = {
    "cost": ("cost", "total_cost", "spent", "amount"),
    "price": ("price", "cost", "amount"),
    "amount": ("amount", "spent", "cost", "total"),
    "spent": ("spent", "cost", "amount"),
    "expense": ("expense", "cost", "spent"),
    "score": ("score", "points", "rating"),
    "count": ("count", "total", "number"),
    "total": ("total", "sum", "amount", "cost"),
}


@dataclass(frozen=True, slots=True)
class DatabaoEnvironment:
    model_api_url: str
    model_api_key: str
    model_name: str


@dataclass(frozen=True, slots=True)
class DatabaoTaskArtifacts:
    task_id: str
    task_output_dir: Path
    prediction_csv_path: Path | None
    log_path: Path
    succeeded: bool
    prediction_written: bool
    scorable: bool
    failure_reason: str | None
    elapsed_seconds: float
    row_count: int | None = None
    column_count: int | None = None
    candidate_source: str | None = None
    postprocessing: dict[str, Any] | None = None
    timings: dict[str, float] | None = None
    llm_calls: list[dict[str, Any]] | None = None
    route_policy: dict[str, Any] | None = None
    question_features: dict[str, Any] | None = None
    answer_contract: dict[str, Any] | None = None
    heuristic_level: str | None = None
    enabled_strategies: list[dict[str, Any]] | None = None
    applied_strategies: list[dict[str, Any]] | None = None
    retrieved_context: dict[str, Any] | None = None
    complexity_profile: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] | None = None
    candidate_scores: list[dict[str, Any]] | None = None
    selected_candidate_source: str | None = None
    final_answer_guard: dict[str, Any] | None = None
    context_payload_profile: dict[str, Any] | None = None
    databao_failure_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_output_dir": str(self.task_output_dir),
            "prediction_csv_path": str(self.prediction_csv_path) if self.prediction_csv_path else None,
            "log_path": str(self.log_path),
            "succeeded": self.succeeded,
            "prediction_written": self.prediction_written,
            "scorable": self.scorable,
            "failure_reason": self.failure_reason,
            "elapsed_seconds": self.elapsed_seconds,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "candidate_source": self.candidate_source,
            "postprocessing": self.postprocessing,
            "timings": self.timings,
            "llm_calls": self.llm_calls,
            "route_policy": self.route_policy,
            "question_features": self.question_features,
            "answer_contract": self.answer_contract,
            "heuristic_level": self.heuristic_level,
            "enabled_strategies": self.enabled_strategies or [],
            "applied_strategies": self.applied_strategies or [],
            "retrieved_context": self.retrieved_context,
            "complexity_profile": self.complexity_profile,
            "candidates": self.candidates or [],
            "candidate_scores": self.candidate_scores or [],
            "selected_candidate_source": self.selected_candidate_source,
            "final_answer_guard": self.final_answer_guard,
            "context_payload_profile": self.context_payload_profile,
            "databao_failure_type": self.databao_failure_type,
        }


AgentBuilder = Callable[[PublicTask, DatabaoEnvironment], Any]


@dataclass(frozen=True, slots=True)
class ContextTable:
    name: str
    path: str
    frame: pd.DataFrame
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class SchemaColumn:
    name: str
    dtype: str
    sample_values: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class SchemaTable:
    name: str
    path: str
    source_kind: str
    row_count: int
    columns: tuple[SchemaColumn, ...]


@dataclass(frozen=True, slots=True)
class JoinCandidate:
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    reason: str


@dataclass(frozen=True, slots=True)
class TaskContext:
    task: PublicTask
    context_tables: list[ContextTable]
    document_text: str
    schema_graph: dict[str, Any]
    context_summary: dict[str, Any] | None = None

    def table_lookup(self) -> dict[str, ContextTable]:
        return _table_lookup(self.context_tables)


@dataclass(frozen=True, slots=True)
class RetrievedContext:
    relevant_tables: tuple[str, ...]
    relevant_columns: dict[str, tuple[str, ...]]
    sample_rows: dict[str, list[dict[str, Any]]]
    candidate_join_paths: tuple[dict[str, Any], ...]
    document_snippets: tuple[dict[str, Any], ...]
    retrieval_diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "relevant_tables": list(self.relevant_tables),
            "relevant_columns": {
                table: list(columns) for table, columns in self.relevant_columns.items()
            },
            "sample_rows": self.sample_rows,
            "candidate_join_paths": list(self.candidate_join_paths),
            "document_snippets": list(self.document_snippets),
            "retrieval_diagnostics": self.retrieval_diagnostics,
        }


@dataclass(frozen=True, slots=True)
class TaskIntent:
    domain: str | None
    operation: str | None
    answer_kind: str
    target_entity: str | None
    target_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QuestionFeatures:
    asks_scalar_metric: bool
    asks_entity_or_list: bool
    asks_multi_attribute: bool
    asks_entity_plus_metric: bool
    asks_aggregation: bool
    asks_ratio_or_percentage: bool
    asks_superlative: bool
    matched_phrases: tuple[str, ...]
    strong_terms: tuple[str, ...]
    weak_terms: tuple[str, ...]
    confidence: float
    evidence: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "asks_scalar_metric": self.asks_scalar_metric,
            "asks_entity_or_list": self.asks_entity_or_list,
            "asks_multi_attribute": self.asks_multi_attribute,
            "asks_entity_plus_metric": self.asks_entity_plus_metric,
            "asks_aggregation": self.asks_aggregation,
            "asks_ratio_or_percentage": self.asks_ratio_or_percentage,
            "asks_superlative": self.asks_superlative,
            "matched_phrases": list(self.matched_phrases),
            "strong_terms": list(self.strong_terms),
            "weak_terms": list(self.weak_terms),
            "confidence": self.confidence,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class TaskComplexityProfile:
    source_count: int
    table_count: int
    db_table_count: int
    csv_table_count: int
    json_table_count: int
    document_file_count: int
    document_text_chars: int
    total_columns: int
    total_rows_sampled: int
    has_unstructured_docs: bool
    has_sqlite_db: bool
    has_multiple_sources: bool
    has_many_tables: bool
    has_join_candidates: bool
    question_token_count: int
    question_mentions_aggregation: bool
    question_mentions_superlative: bool
    question_mentions_ratio_or_percentage: bool
    question_mentions_filter_or_condition: bool
    question_mentions_multiple_attributes: bool
    estimated_context_size: int
    retrieval_confidence: float
    best_candidate_score: int | None
    best_candidate_contract_valid: bool | None
    best_candidate_empty: bool | None
    best_candidate_has_metadata_columns: bool | None
    best_candidate_shape_mismatch: bool | None
    complexity_score: int
    uncertainty_score: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_count": self.source_count,
            "table_count": self.table_count,
            "db_table_count": self.db_table_count,
            "csv_table_count": self.csv_table_count,
            "json_table_count": self.json_table_count,
            "document_file_count": self.document_file_count,
            "document_text_chars": self.document_text_chars,
            "total_columns": self.total_columns,
            "total_rows_sampled": self.total_rows_sampled,
            "has_unstructured_docs": self.has_unstructured_docs,
            "has_sqlite_db": self.has_sqlite_db,
            "has_multiple_sources": self.has_multiple_sources,
            "has_many_tables": self.has_many_tables,
            "has_join_candidates": self.has_join_candidates,
            "question_token_count": self.question_token_count,
            "question_mentions_aggregation": self.question_mentions_aggregation,
            "question_mentions_superlative": self.question_mentions_superlative,
            "question_mentions_ratio_or_percentage": self.question_mentions_ratio_or_percentage,
            "question_mentions_filter_or_condition": self.question_mentions_filter_or_condition,
            "question_mentions_multiple_attributes": self.question_mentions_multiple_attributes,
            "estimated_context_size": self.estimated_context_size,
            "retrieval_confidence": self.retrieval_confidence,
            "best_candidate_score": self.best_candidate_score,
            "best_candidate_contract_valid": self.best_candidate_contract_valid,
            "best_candidate_empty": self.best_candidate_empty,
            "best_candidate_has_metadata_columns": self.best_candidate_has_metadata_columns,
            "best_candidate_shape_mismatch": self.best_candidate_shape_mismatch,
            "complexity_score": self.complexity_score,
            "uncertainty_score": self.uncertainty_score,
        }


@dataclass(frozen=True, slots=True)
class Candidate:
    frame: pd.DataFrame
    source: str
    confidence: float
    diagnostics: dict[str, Any]
    transformations: tuple[dict[str, Any], ...] = ()
    contract_report: dict[str, Any] | None = None
    retrieval_context_used: dict[str, Any] | None = None
    elapsed_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "confidence": self.confidence,
            "row_count": len(self.frame),
            "column_count": len(self.frame.columns),
            "columns": [str(column) for column in self.frame.columns],
            "diagnostics": self.diagnostics,
            "transformations": list(self.transformations),
            "contract_report": self.contract_report,
            "retrieval_context_used": self.retrieval_context_used,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True, slots=True)
class CandidateScore:
    source: str
    score: int
    selected: bool
    reasons: tuple[str, ...]
    row_count: int
    column_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "score": self.score,
            "selected": self.selected,
            "reasons": list(self.reasons),
            "row_count": self.row_count,
            "column_count": self.column_count,
        }


@dataclass(frozen=True, slots=True)
class CandidateRankingReport:
    selected_source: str | None
    candidate_scores: tuple[CandidateScore, ...]
    rejection_reasons: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_source": self.selected_source,
            "candidate_scores": [score.to_dict() for score in self.candidate_scores],
            "rejection_reasons": self.rejection_reasons,
        }


@dataclass(frozen=True, slots=True)
class FinalAnswerShapeReport:
    transformations: tuple[dict[str, Any], ...]
    input_row_count: int
    input_column_count: int
    output_row_count: int
    output_column_count: int
    removed_rows: int
    removed_columns: tuple[str, ...]
    contract_before: dict[str, Any] | None
    contract_after: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "transformations": list(self.transformations),
            "input_row_count": self.input_row_count,
            "input_column_count": self.input_column_count,
            "output_row_count": self.output_row_count,
            "output_column_count": self.output_column_count,
            "removed_rows": self.removed_rows,
            "removed_columns": list(self.removed_columns),
            "contract_before": self.contract_before,
            "contract_after": self.contract_after,
            "metadata_columns_removed": any(
                transform.get("kind") == "metadata_column_removal"
                for transform in self.transformations
            ),
        }


@dataclass(frozen=True, slots=True)
class RouteDecision:
    route: str
    reason: str
    confidence: float
    candidate: Candidate | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "route": self.route,
            "reason": self.reason,
            "confidence": self.confidence,
        }
        if self.candidate is not None:
            payload["candidate"] = {
                "source": self.candidate.source,
                "confidence": self.candidate.confidence,
                "row_count": len(self.candidate.frame),
                "column_count": len(self.candidate.frame.columns),
                "diagnostics": self.candidate.diagnostics,
            }
        return payload


@dataclass(frozen=True, slots=True)
class AnswerContract:
    kind: str
    expected_columns: tuple[str, ...]
    max_rows: int | None
    max_columns: int | None
    allow_empty: bool
    reason: str


@dataclass(frozen=True, slots=True)
class AnswerContractReport:
    valid: bool
    should_repair: bool
    reason: str
    contract_kind: str
    expected_columns: tuple[str, ...]
    row_count: int
    column_count: int
    candidate_source: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "should_repair": self.should_repair,
            "reason": self.reason,
            "contract_kind": self.contract_kind,
            "expected_columns": list(self.expected_columns),
            "row_count": self.row_count,
            "column_count": self.column_count,
            "candidate_source": self.candidate_source,
        }


@dataclass(frozen=True, slots=True)
class ContextIdentifierMatch:
    score: float
    table: ContextTable
    id_column: str
    display_name: str
    lookup: dict[str, str]


@dataclass(frozen=True, slots=True)
class DeterministicPostprocessReport:
    applied: bool
    transformations: list[dict[str, Any]]
    failure_reason: str | None
    input_row_count: int
    input_column_count: int
    output_row_count: int
    output_column_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "transformations": self.transformations,
            "failure_reason": self.failure_reason,
            "input_row_count": self.input_row_count,
            "input_column_count": self.input_column_count,
            "output_row_count": self.output_row_count,
            "output_column_count": self.output_column_count,
        }


AnswerPostprocessor = Callable[
    [PublicTask, pd.DataFrame],
    tuple[pd.DataFrame, DeterministicPostprocessReport],
]


def load_databao_environment(env: Mapping[str, str] | None = None) -> DatabaoEnvironment:
    source = env or os.environ
    model_api_url = source.get(MODEL_API_URL_ENV, "").strip()
    model_name = source.get(MODEL_NAME_ENV, "").strip()
    missing = [
        name
        for name, value in (
            (MODEL_API_URL_ENV, model_api_url),
            (MODEL_NAME_ENV, model_name),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing required environment variables for Databao runner: "
            + ", ".join(missing)
            + "."
        )

    return DatabaoEnvironment(
        model_api_url=model_api_url,
        model_api_key=source.get(MODEL_API_KEY_ENV, "EMPTY") or "EMPTY",
        model_name=model_name,
    )


@contextmanager
def _openai_api_key_from_model_env(databao_env: DatabaoEnvironment):
    previous = os.environ.get(OPENAI_API_KEY_ENV)
    os.environ[OPENAI_API_KEY_ENV] = databao_env.model_api_key
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(OPENAI_API_KEY_ENV, None)
        else:
            os.environ[OPENAI_API_KEY_ENV] = previous


def _redact_sensitive_text(text: str, secrets: tuple[str, ...] = ()) -> str:
    message = text
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[redacted]")
    message = re.sub(r"user_[A-Za-z0-9]+", "[redacted-user]", message)
    return message


def _safe_exception_summary(exc: BaseException, secrets: tuple[str, ...] = ()) -> str:
    message = _redact_sensitive_text(str(exc), secrets)
    if len(message) > MAX_FAILURE_REASON_CHARS:
        message = message[:MAX_FAILURE_REASON_CHARS] + "...[truncated]"
    return f"{type(exc).__name__}: {message}"


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return value if value > 0 else default


def _deadline_from_start(started_at: float) -> float | None:
    timeout_seconds = _env_int(DATABAO_TASK_TIMEOUT_SECONDS_ENV, 0)
    return started_at + timeout_seconds if timeout_seconds > 0 else None


def _remaining_seconds(deadline_at: float | None) -> float | None:
    if deadline_at is None:
        return None
    return max(0.0, deadline_at - perf_counter())


def _databao_timeout_seconds(deadline_at: float | None = None) -> int:
    configured = _env_int(DATABAO_DATABAO_TIMEOUT_SECONDS_ENV, DATABAO_AGENT_TIMEOUT_SECONDS)
    if deadline_at is None:
        deadline_at = _CURRENT_TASK_DEADLINE.get()
    remaining = _remaining_seconds(deadline_at)
    if remaining is None:
        return max(1, configured)
    budgeted = max(1, int(remaining - 20))
    return max(1, min(configured, budgeted))


def _normalize_mode(
    value: str | None,
    *,
    default: str,
    allowed: set[str],
    aliases: Mapping[str, str] | None = None,
) -> str:
    aliases = aliases or {}
    raw_value = (value or default).strip().lower().replace("_", "-")
    mode = aliases.get(raw_value, raw_value)
    if mode not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise ValueError(f"Unsupported mode {value!r}. Expected one of: {allowed_values}.")
    return mode


def _heuristic_level(value: str | None = None) -> str:
    return _normalize_mode(
        value if value is not None else os.environ.get(DATABAO_HEURISTIC_LEVEL_ENV),
        default="generic",
        allowed=HEURISTIC_LEVELS,
        aliases={
            "experimental": "experimental_generic",
            "experimental-generic": "experimental_generic",
        },
    )


def _databao_executor_type(value: str | None = None) -> str:
    return _normalize_mode(
        value if value is not None else os.environ.get(DATABAO_EXECUTOR_TYPE_ENV),
        default="lighthouse_salvage",
        allowed=DATABAO_EXECUTOR_TYPES,
        aliases={"lighthouse-salvage": "lighthouse_salvage", "salvage": "lighthouse_salvage"},
    )


class StructuredPlanError(ValueError):
    """Raised when a structured query plan is not safe or executable."""


def _safe_log_value(value: Any, secrets: tuple[str, ...] = ()) -> Any:
    if isinstance(value, str):
        return _redact_sensitive_text(value, secrets)
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return {
            "type": "DataFrame",
            "row_count": len(value),
            "column_count": len(value.columns),
            "columns": [str(column) for column in value.columns],
        }
    if isinstance(value, Mapping):
        return {str(key): _safe_log_value(item, secrets) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_safe_log_value(item, secrets) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _safe_log_value(value.model_dump(mode="json"), secrets)
        except Exception:  # noqa: BLE001
            pass

    text = _redact_sensitive_text(str(value), secrets)
    if len(text) > MAX_FAILURE_REASON_CHARS:
        text = text[:MAX_FAILURE_REASON_CHARS] + "...[truncated]"
    return text


def _databao_frame_execution_payload(frame: pd.DataFrame) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    code = _text_value(frame.attrs.get("databao_code"))
    if code:
        payload["databao_code"] = code[:MAX_DATABAO_CODE_CHARS]
    for key in ("databao_submit_called", "databao_salvaged_latest_query_result"):
        if key in frame.attrs:
            payload[key] = bool(frame.attrs.get(key))
    executor_type = frame.attrs.get("databao_executor_type")
    if executor_type:
        payload["databao_executor_type"] = _text_value(executor_type)
    return payload


class TaskDiagnostics:
    def __init__(
        self,
        *,
        task: PublicTask,
        logs_dir: Path,
        databao_env: DatabaoEnvironment,
    ) -> None:
        self.task = task
        self.logs_dir = logs_dir
        self.secrets = (databao_env.model_api_key,)
        self.timings: dict[str, float] = {}
        self.llm_calls: list[dict[str, Any]] = []
        self.raw_enabled = os.environ.get(DATABAO_DEBUG_LOG_RAW_ENV, "").strip() == "1"
        self.raw_dir = logs_dir / "raw" / task.task_id
        self.progress_path = logs_dir / f"{task.task_id}.progress.json"
        self._raw_counter = 0
        self.checkpoints: dict[str, Any] = {}

    @contextmanager
    def stage(self, name: str):
        started_at = perf_counter()
        self.write_progress(active_stage=name, event="stage_start")
        try:
            yield
        finally:
            self.add_timing(name, perf_counter() - started_at)
            self.write_progress(active_stage=None, event="stage_end")

    def add_timing(self, name: str, elapsed_seconds: float) -> None:
        self.timings[name] = round(self.timings.get(name, 0.0) + elapsed_seconds, 3)

    def write_progress(
        self,
        *,
        active_stage: str | None,
        event: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_id": self.task.task_id,
            "event": event,
            "active_stage": active_stage,
            "timings": self.timings,
            "llm_call_count": len(self.llm_calls),
            "checkpoints": _safe_log_value(self.checkpoints, self.secrets),
        }
        if metadata:
            payload["metadata"] = _safe_log_value(dict(metadata), self.secrets)
        self.progress_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            errors="replace",
        )

    def checkpoint(self, name: str, **metadata: Any) -> None:
        self.checkpoints[name] = dict(metadata)
        self.write_progress(active_stage=name, event="checkpoint", metadata=metadata)

    def write_raw(self, label: str, payload: Any) -> str | None:
        if not self.raw_enabled:
            return None
        self._raw_counter += 1
        safe_label = SOURCE_NAME_PATTERN.sub("_", label).strip("_") or "raw"
        path = self.raw_dir / f"{self._raw_counter:03d}_{safe_label}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_safe_log_value(payload, self.secrets), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            errors="replace",
        )
        return str(path)

    def record_llm_call(
        self,
        stage: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        raw_request: Any | None = None,
        raw_response: Any | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "stage": stage,
            "metadata": _safe_log_value(dict(metadata or {}), self.secrets),
        }
        request_path = self.write_raw(f"{stage}_request", raw_request) if raw_request is not None else None
        response_path = self.write_raw(f"{stage}_response", raw_response) if raw_response is not None else None
        if request_path is not None:
            entry["raw_request_path"] = request_path
        if response_path is not None:
            entry["raw_response_path"] = response_path
        self.llm_calls.append(entry)


class DatabaoLangChainCallback(BaseCallbackHandler):
    def __init__(self, diagnostics: TaskDiagnostics) -> None:
        self.diagnostics = diagnostics

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[Any],
        **kwargs: Any,
    ) -> None:
        try:
            self.diagnostics.record_llm_call(
                "databao_chat_start",
                metadata={
                    "serialized_name": serialized.get("name") if isinstance(serialized, dict) else None,
                    "message_batches": len(messages),
                    "tags": kwargs.get("tags"),
                    "metadata": kwargs.get("metadata"),
                },
                raw_request={"serialized": serialized, "messages": messages, "kwargs": kwargs},
            )
        except Exception:  # noqa: BLE001
            return

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            self.diagnostics.record_llm_call(
                "databao_chat_end",
                metadata={"run_id": str(kwargs.get("run_id", ""))},
                raw_response=response,
            )
        except Exception:  # noqa: BLE001
            return

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        try:
            self.diagnostics.record_llm_call(
                "databao_chat_error",
                metadata={
                    "run_id": str(kwargs.get("run_id", "")),
                    "error": _safe_exception_summary(error, self.diagnostics.secrets),
                },
            )
        except Exception:  # noqa: BLE001
            return


def _safe_source_name(raw_name: str, used_names: set[str]) -> str:
    name = SOURCE_NAME_PATTERN.sub("_", raw_name).strip("_")
    if not name:
        name = "source"
    if not name[0].isalpha():
        name = f"source_{name}"

    candidate = name
    suffix = 2
    while candidate in used_names:
        candidate = f"{name}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _description_from_file(path: Path, root: Path, *, max_chars: int = MAX_DESCRIPTION_CHARS) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return f"Document file {_relative_posix(path, root)}:\n{text}"


def _json_value_to_dataframe(value: Any) -> pd.DataFrame | None:
    if isinstance(value, list):
        if not value:
            return pd.DataFrame()
        if all(isinstance(item, dict) for item in value):
            return pd.json_normalize(value)
        return pd.DataFrame({"value": value})

    if isinstance(value, dict):
        return pd.json_normalize(value)

    if value is None or isinstance(value, str | int | float | bool):
        return pd.DataFrame({"value": [value]})

    return None


def _json_dataframes(path: Path) -> list[tuple[str, pd.DataFrame]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames: list[tuple[str, pd.DataFrame]] = []

    if isinstance(payload, dict):
        table_name = payload.get("table")
        records = payload.get("records")
        if isinstance(table_name, str) and isinstance(records, list | dict):
            frame = _json_value_to_dataframe(records)
            if frame is not None:
                return [(table_name, frame)]

        for key, value in payload.items():
            if isinstance(value, list | dict):
                frame = _json_value_to_dataframe(value)
                if frame is not None:
                    frames.append((f"{path.stem}_{key}", frame))
        if frames:
            return frames

    frame = _json_value_to_dataframe(payload)
    if frame is not None:
        return [(path.stem, frame)]
    return []


def _context_file_paths(context_dir: Path, subdir_name: str, suffixes: set[str]) -> list[Path]:
    paths: list[Path] = []
    if context_dir.is_dir():
        for path in sorted(context_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in suffixes:
                paths.append(path)
    subdir = context_dir / subdir_name
    if subdir.is_dir():
        paths.extend(sorted(path for path in subdir.iterdir() if path.is_file() and path.suffix.lower() in suffixes))
    return paths


def _document_paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]


def _clean_document_value(value: str) -> str:
    value = re.split(r"\.\s+", value, maxsplit=1)[0]
    value = re.sub(r"\s+", " ", value).strip(" .,:;\"'`")
    value = re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.IGNORECASE)
    return value.strip()


def _document_records_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "record_id" in frame.columns:
        frame = frame.drop_duplicates(subset=["source_doc", "record_id", "paragraph_index"], keep="last")
    return frame.reset_index(drop=True)


GENERIC_NUMERIC_ATTRIBUTES = ("amount", "cost", "price", "score", "value", "height", "total", "count")
GENERIC_TEXT_ATTRIBUTES = ("name", "title", "description", "status", "type", "category", "format")
GENERIC_TIME_ATTRIBUTES = ("date", "year", "season", "created_at")


def _generic_identifier_values(paragraph: str) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value in re.findall(r"\brec[A-Za-z0-9]+\b", paragraph):
        if value not in seen:
            values.append(("record_id", value))
            seen.add(value)
    numeric_patterns = [
        r"\b(?:ID|identifier|registry id|record id|code|reference|file number)\s*:?\s*(\d+)\b",
        r"\b(\d+)\s*\((?:ID|identifier|registry id|record id|code|reference)\)",
    ]
    for pattern in numeric_patterns:
        for value in re.findall(pattern, paragraph, flags=re.IGNORECASE):
            if value not in seen:
                values.append(("numeric_id", value))
                seen.add(value)
    return values


def _extract_named_value(paragraph: str, labels: tuple[str, ...]) -> str | None:
    label_pattern = "|".join(re.escape(label) for label in labels)
    patterns = [
        rf"\b(?:{label_pattern})\b\s*(?:is|was|as|:|=|to|under|with)?\s+(?:the\s+)?([^.;,\n]+)",
        r"\b(?:called|known as|designated as|identified as|titled)\s+(?:the\s+)?([^.;,\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, paragraph, flags=re.IGNORECASE)
        if match:
            value = _clean_document_value(match.group(1))
            if value:
                return value
    return None


def _clean_identifier_display_value(value: str) -> str:
    value = _clean_document_value(value)
    value = re.sub(
        r"^(?:the\s+)?(?:academic\s+|official\s+|corresponding\s+|specialized\s+|professional\s+|"
        r"vocational\s+|interdisciplinary\s+|modern\s+|general\s+|flexible\s+|foundational\s+|"
        r"key\s+|strategic\s+|core\s+|next\s+)?"
        r"(?:record|entry|listing|profile|unit|program|track|discipline|designation)\s+"
        r"(?:for|of|as|is|was|now|to)?\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"^.*\b(?:called|known as|designated as|identified as|titled|confirmed as|updated to|"
        r"corrected to|amended to|relabeled as|recorded as|designation is|designation is now|"
        r"title is|name is)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return value.strip(" .,:;\"'`")


def _extract_display_value_for_identifier(paragraph: str, identifier_value: str) -> str | None:
    if not identifier_value:
        return None
    escaped_identifier = re.escape(identifier_value)
    id_label = r"(?:Registry ID|ID|identifier|record id|code|reference)"
    patterns = (
        rf"([^.;()\n]{{2,140}}?)\s*\(\s*{id_label}\s*:?\s*{escaped_identifier}\s*\)",
        rf"([^.;()\n]{{2,140}}?)\s*\(\s*{escaped_identifier}\s*\)",
        rf"([^.;\n]{{2,140}}?)\b{escaped_identifier}\b",
    )
    for pattern in patterns:
        match = re.search(pattern, paragraph, flags=re.IGNORECASE)
        if not match:
            continue
        value = _clean_identifier_display_value(match.group(1))
        if value and not re.fullmatch(r"(?:record|entry|id|identifier|code|reference)", value, flags=re.IGNORECASE):
            return value
    return None


def _extract_numeric_attribute(paragraph: str, attribute: str) -> float | None:
    pattern = rf"\b{re.escape(attribute)}\b[^\d.\-]{{0,40}}(-?\d+(?:\.\d+)?)"
    matches = re.findall(pattern, paragraph, flags=re.IGNORECASE)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _extract_text_attribute(paragraph: str, attribute: str) -> str | None:
    pattern = rf"\b{re.escape(attribute)}\b\s*(?:is|was|as|:|=|to|under|with)?\s+(?:the\s+)?([^.;,\n]+)"
    match = re.search(pattern, paragraph, flags=re.IGNORECASE)
    if not match:
        return None
    value = _clean_document_value(match.group(1))
    return value or None


def _extract_relation_attributes(paragraph: str) -> dict[str, str]:
    relations: dict[str, str] = {}
    for key, value in re.findall(r"\b([A-Za-z][A-Za-z0-9_]*_id)\s*:?\s*([A-Za-z0-9_.-]+)", paragraph):
        relations[key] = value.strip(".,;")
    related_values = re.findall(
        r"\b(?:related to|associated with|linked to|belongs to|connected to)\s+(?:record|reference|id|code)?\s*:?\s*([A-Za-z0-9_.-]+)",
        paragraph,
        flags=re.IGNORECASE,
    )
    if related_values:
        relations["related_id"] = related_values[-1].strip(".,;")
    return relations


def _generic_document_rows_for_path(
    path: Path,
    root: Path,
    *,
    heuristic_level: str,
) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, Any]] = []
    for paragraph_index, paragraph in enumerate(_document_paragraphs(text)):
        identifiers = _generic_identifier_values(paragraph)
        if not identifiers and heuristic_level != "experimental_generic":
            continue
        if not identifiers:
            identifiers = [("paragraph_id", f"paragraph_{paragraph_index}")]

        base: dict[str, Any] = {
            "source_doc": _relative_posix(path, root),
            "extraction_strategy": "generic_paragraph_record",
            "strategy_kind": "generic",
            "paragraph_index": paragraph_index,
            "evidence_span": paragraph[:500],
        }
        display_value = _extract_named_value(paragraph, ("name", "title"))
        if display_value is not None:
            base["name"] = display_value
        for attribute in GENERIC_TEXT_ATTRIBUTES:
            value = _extract_text_attribute(paragraph, attribute)
            if value is not None:
                base[attribute] = value
        for attribute in GENERIC_NUMERIC_ATTRIBUTES:
            value = _extract_numeric_attribute(paragraph, attribute)
            if value is not None:
                base[attribute] = value
        for attribute in GENERIC_TIME_ATTRIBUTES:
            value = _extract_text_attribute(paragraph, attribute)
            if value is not None:
                base[attribute] = value
        base.update(_extract_relation_attributes(paragraph))

        confidence = 0.35
        if any(key in base for key in ("name", "title", "status", "type", "category", "format")):
            confidence += 0.2
        if any(key in base for key in GENERIC_NUMERIC_ATTRIBUTES):
            confidence += 0.15
        if any(key.endswith("_id") or key == "related_id" for key in base):
            confidence += 0.15
        base["confidence"] = min(confidence, 0.85)

        for identifier_kind, identifier_value in identifiers:
            row = dict(base)
            row["record_id_kind"] = identifier_kind
            row["record_id"] = identifier_value
            if "name" not in row:
                display_value = _extract_display_value_for_identifier(paragraph, identifier_value)
                if display_value:
                    row["name"] = display_value
            rows.append(row)
    return rows


def document_records_for_reasoning(
    context_dir: Path,
    *,
    heuristic_level: str | None = None,
) -> list[ContextTable]:
    level = _heuristic_level(heuristic_level)
    doc_dir = context_dir / "doc"
    if not doc_dir.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in doc_dir.rglob("*.md") if item.is_file()):
        rows.extend(_generic_document_rows_for_path(path, context_dir, heuristic_level=level))
    frame = _document_records_frame(rows)
    if frame.empty:
        return []
    metadata = {
        "strategy_name": "generic_document_tables",
        "strategy_kind": "generic",
        "enabled_by": level,
        "input_document_count": len({row["source_doc"] for row in rows}),
        "extracted_row_count": len(frame),
        "extracted_columns": [str(column) for column in frame.columns],
        "confidence": float(pd.to_numeric(frame["confidence"], errors="coerce").fillna(0).mean()),
    }
    return [
        ContextTable(
            name="document_records",
            path="doc/*::generic_document_records",
            frame=frame,
            metadata=metadata,
        )
    ]


def _is_document_agent_safe_column(column_name: str) -> bool:
    lowered = column_name.lower()
    return lowered in DOCUMENT_AGENT_SAFE_BASE_COLUMNS or lowered.endswith("_id")


def _document_records_agent_frame(frame: pd.DataFrame) -> pd.DataFrame:
    selected_columns = [
        column
        for column in frame.columns
        if _is_document_agent_safe_column(str(column))
    ]
    if not selected_columns:
        return pd.DataFrame()
    output = frame.loc[:, selected_columns].copy()
    if "record_id" in output.columns and "name" in output.columns:
        scored = output.copy()
        scored["__display_score"] = scored["name"].map(_display_value_score)
        if "confidence" in scored.columns:
            scored["__confidence_score"] = pd.to_numeric(scored["confidence"], errors="coerce").fillna(0)
        else:
            scored["__confidence_score"] = 0.0
        scored = scored.sort_values(
            ["record_id", "__display_score", "__confidence_score"],
            ascending=[True, False, False],
            kind="stable",
        )
        output = scored.drop_duplicates(subset=["record_id"], keep="first").drop(
            columns=["__display_score", "__confidence_score"],
            errors="ignore",
        )
    return output.drop_duplicates().reset_index(drop=True)


def document_records_for_agent(
    context_dir: Path,
    *,
    heuristic_level: str | None = None,
) -> list[ContextTable]:
    agent_tables: list[ContextTable] = []
    for table in document_records_for_reasoning(context_dir, heuristic_level=heuristic_level):
        frame = _document_records_agent_frame(table.frame)
        if frame.empty:
            continue
        removed_columns = [
            str(column)
            for column in table.frame.columns
            if str(column) not in {str(selected) for selected in frame.columns}
        ]
        metadata = dict(table.metadata or {})
        metadata.update(
            {
                "strategy_name": "document_records_for_agent",
                "reasoning_table_name": table.name,
                "agent_safe_columns": [str(column) for column in frame.columns],
                "metadata_columns_removed": removed_columns,
            }
        )
        agent_tables.append(
            ContextTable(
                name="document_records",
                path=table.path,
                frame=frame,
                metadata=metadata,
            )
        )
    return agent_tables


def generic_document_tables(context_dir: Path, *, heuristic_level: str | None = None) -> list[ContextTable]:
    return document_records_for_agent(context_dir, heuristic_level=heuristic_level)


def _sqlite_has_tables(path: Path) -> bool:
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
        ).fetchall()
        return bool(rows)
    finally:
        conn.close()


def _retrieved_context_allows_table(
    retrieved_context: RetrievedContext | None,
    *,
    table_name: str,
    source_path: str,
) -> bool:
    if retrieved_context is None:
        return True
    relevant = {value.lower() for value in retrieved_context.relevant_tables}
    table_key = table_name.lower()
    path_key = source_path.lower()
    stem_key = Path(source_path.split("::", 1)[0]).stem.lower()
    return table_key in relevant or path_key in relevant or stem_key in relevant


def _retrieved_document_description(
    retrieved_context: RetrievedContext | None,
) -> str | None:
    if retrieved_context is None or not retrieved_context.document_snippets:
        return None
    lines = ["Retrieved document snippets for this question:"]
    for index, snippet in enumerate(retrieved_context.document_snippets, start=1):
        text = _text_value(snippet.get("text"))[:1000]
        if not text:
            continue
        lines.append(f"[{index}] {text}")
    return "\n".join(lines) if len(lines) > 1 else None


def register_context_sources(
    domain: Any,
    context_dir: Path,
    *,
    heuristic_level: str | None = None,
    retrieved_context: RetrievedContext | None = None,
    question: str | None = None,
) -> dict[str, Any]:
    del question
    effective_heuristic_level = _heuristic_level(heuristic_level)
    description_max_chars = MAX_DESCRIPTION_CHARS
    used_names: set[str] = set()
    summary: dict[str, Any] = {
        "csv_files": [],
        "json_files": [],
        "sqlite_files": [],
        "document_files": [],
        "document_materialized_tables": [],
        "registered_sources": [],
        "heuristic_level": effective_heuristic_level,
    }

    retrieved_description = _retrieved_document_description(retrieved_context)
    if retrieved_description is not None:
        domain.add_description(retrieved_description)
        summary["document_files"].append("retrieved_document_snippets")

    knowledge_path = context_dir / "knowledge.md"
    if retrieved_context is None and knowledge_path.exists():
        domain.add_description(
            _description_from_file(knowledge_path, context_dir, max_chars=description_max_chars)
        )
        summary["document_files"].append(_relative_posix(knowledge_path, context_dir))

    doc_dir = context_dir / "doc"
    if retrieved_context is None and doc_dir.is_dir():
        for path in sorted(item for item in doc_dir.rglob("*") if item.is_file()):
            domain.add_description(_description_from_file(path, context_dir, max_chars=description_max_chars))
            summary["document_files"].append(_relative_posix(path, context_dir))

    for path in _context_file_paths(context_dir, "csv", {".csv"}):
        relative_path = _relative_posix(path, context_dir)
        name = _safe_source_name(f"csv_{path.stem}", used_names)
        frame = pd.read_csv(path, low_memory=False)
        domain.add_df(frame, name=name, description=f"CSV file {relative_path}")
        summary["csv_files"].append(relative_path)
        summary["registered_sources"].append({"name": name, "kind": "csv", "rows": len(frame)})

    for path in _context_file_paths(context_dir, "json", {".json"}):
        frames = _json_dataframes(path)
        if frames:
            for hint, frame in frames:
                relative_path = _relative_posix(path, context_dir)
                name = _safe_source_name(f"json_{hint}", used_names)
                domain.add_df(
                    frame,
                    name=name,
                    description=f"JSON file {relative_path}",
                )
                summary["registered_sources"].append({"name": name, "kind": "json", "rows": len(frame)})
        else:
            if retrieved_context is None:
                domain.add_description(_description_from_file(path, context_dir, max_chars=description_max_chars))
        summary["json_files"].append(_relative_posix(path, context_dir))

    for table in generic_document_tables(context_dir, heuristic_level=effective_heuristic_level):
        agent_frame = table.frame
        name = _safe_source_name(f"doc_{table.name}", used_names)
        domain.add_df(
            agent_frame,
            name=name,
            description=f"Materialized document table {table.path}",
        )
        metadata = table.metadata or {}
        summary["document_materialized_tables"].append(
            {
                "name": table.name,
                "path": table.path,
                "rows": len(agent_frame),
                "source_rows": len(table.frame),
                "columns": [str(column) for column in table.frame.columns],
                "strategy_name": metadata.get("strategy_name", "generic_document_tables"),
                "strategy_kind": metadata.get("strategy_kind", "generic"),
                "enabled_by": metadata.get("enabled_by", effective_heuristic_level),
                "confidence": metadata.get("confidence"),
                "extracted_columns": metadata.get(
                    "extracted_columns",
                    [str(column) for column in table.frame.columns],
                ),
            }
        )
        summary["registered_sources"].append(
            {
                "name": name,
                "kind": "document_table",
                "rows": len(agent_frame),
                "source_rows": len(table.frame),
            }
        )

    for path in _context_file_paths(context_dir, "db", {".db", ".sqlite"}):
        if not _sqlite_has_tables(path):
            continue
        relative_path = _relative_posix(path, context_dir)
        name = _safe_source_name(f"db_{path.stem}", used_names)
        engine = create_engine(f"sqlite:///{path.as_posix()}")
        try:
            domain.add_db(engine, name=name, description=f"SQLite file {_relative_posix(path, context_dir)}")
        finally:
            engine.dispose()
        summary["sqlite_files"].append(_relative_posix(path, context_dir))
        summary["registered_sources"].append({"name": name, "kind": "sqlite"})

    return summary


def load_context_tables(
    context_dir: Path,
    *,
    heuristic_level: str | None = None,
    question: str | None = None,
) -> list[ContextTable]:
    del question
    effective_heuristic_level = _heuristic_level(heuristic_level)
    tables: list[ContextTable] = []

    for path in _context_file_paths(context_dir, "csv", {".csv"}):
        tables.append(
            ContextTable(
                name=path.stem,
                path=_relative_posix(path, context_dir),
                frame=pd.read_csv(path, low_memory=False),
            )
        )

    for path in _context_file_paths(context_dir, "json", {".json"}):
        for hint, frame in _json_dataframes(path):
            tables.append(
                ContextTable(
                    name=hint,
                    path=_relative_posix(path, context_dir),
                    frame=frame,
                )
            )

    for table in generic_document_tables(context_dir, heuristic_level=effective_heuristic_level):
        tables.append(table)

    for path in _context_file_paths(context_dir, "db", {".db", ".sqlite"}):
        if not _sqlite_has_tables(path):
            continue
        conn = sqlite3.connect(path)
        try:
            table_names = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
            ]
            for table_name in table_names:
                frame = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
                tables.append(
                    ContextTable(
                        name=table_name,
                        path=f"{_relative_posix(path, context_dir)}::{table_name}",
                        frame=frame,
                    )
                )
        finally:
            conn.close()

    return tables


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _looks_like_record_id(value: Any) -> bool:
    return bool(RECORD_ID_PATTERN.match(_text_value(value)))


def _column_nonempty_values(series: pd.Series) -> list[str]:
    return [text for text in (_text_value(value) for value in series.tolist()) if text]


def _column_is_record_id_like(series: pd.Series) -> bool:
    values = _column_nonempty_values(series)
    return bool(values) and all(RECORD_ID_PATTERN.match(value) for value in values)


def _identifier_column_hint(column_name: str) -> str | None:
    lowered = column_name.lower()
    if lowered.startswith("link_to_"):
        return lowered.removeprefix("link_to_").removesuffix("s")
    if lowered.endswith("_id"):
        return lowered.removesuffix("_id").removesuffix("s")
    if lowered == "id":
        return None
    return None


def _identifier_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if _is_id_like_column(str(column)):
            columns.append(str(column))
            continue
        if _column_is_record_id_like(frame[column]):
            columns.append(str(column))
    return columns


def _is_id_like_column(column: str) -> bool:
    original = str(column)
    lowered = column.lower()
    return (
        lowered == "id"
        or lowered.endswith("_id")
        or lowered.startswith("link_to_")
        or bool(re.search(r"[A-Za-z0-9]Id$", original))
        or lowered in {"identifier", "code", "reference"}
    )


def _question_asks_for_identifier(question: str) -> bool:
    return bool(re.search(r"\b(?:id|identifier|code|reference)\b", question.lower()))


def _question_asks_for_url(question: str) -> bool:
    return bool(re.search(r"\b(?:url|link|website|web site)\b", question.lower()))


def _is_display_like_column(column: str) -> bool:
    lowered = column.lower()
    aliases = _column_question_aliases(column)
    return (
        lowered in DISPLAY_COLUMN_PRIORITY
        or lowered.endswith("_name")
        or bool(aliases & {"name", "title", "label", "description", "text", "value", "url", "link"})
    )


def _display_series(table: ContextTable) -> tuple[str, pd.Series] | None:
    frame = table.frame
    lowered_to_original = {str(column).lower(): column for column in frame.columns}

    if "first_name" in lowered_to_original and "last_name" in lowered_to_original:
        first_column = lowered_to_original["first_name"]
        last_column = lowered_to_original["last_name"]
        series = (
            frame[first_column].map(_text_value).str.strip()
            + " "
            + frame[last_column].map(_text_value).str.strip()
        ).str.strip()
        return "full_name", series

    for preferred in DISPLAY_COLUMN_PRIORITY:
        if preferred in frame.columns:
            return str(preferred), frame[preferred]
        lowered = preferred.lower()
        if lowered in lowered_to_original:
            original = lowered_to_original[lowered]
            return str(original), frame[original]

    for column in frame.columns:
        lowered = str(column).lower()
        if lowered.endswith("_name") or lowered.endswith(" name"):
            return str(column), frame[column]

    display_like_columns = [
        str(column)
        for column in frame.columns
        if _is_display_like_column(str(column)) and not _is_id_like_column(str(column))
    ]
    if display_like_columns:
        return display_like_columns[0], frame[display_like_columns[0]]

    return None


def _display_value_score(value: str) -> int:
    text = _text_value(value)
    if not text:
        return -100
    words = re.findall(r"[A-Za-z0-9]+", text)
    score = 40
    if len(words) <= 4:
        score += 20
    if len(words) == 1:
        score += 10
    if len(text) > 80:
        score -= 30
    if re.search(r"\b(?:record|entry|listing|profile|unit|program|track|discipline|designation)\b", text, flags=re.I):
        score -= 12
    return score


def _make_unique_column_name(
    column_name: str,
    existing_columns: list[str],
    replaced_column: str,
) -> str:
    if column_name == replaced_column:
        return column_name
    existing = {column for column in existing_columns if column != replaced_column}
    if column_name not in existing:
        return column_name
    suffix = 2
    while f"{column_name}_{suffix}" in existing:
        suffix += 1
    return f"{column_name}_{suffix}"


def _match_context_identifier(
    *,
    context_tables: list[ContextTable],
    values: list[str],
    column_name: str,
) -> ContextIdentifierMatch | None:
    if not values:
        return None

    hint = _identifier_column_hint(column_name)
    best_match: ContextIdentifierMatch | None = None

    for table in context_tables:
        display = _display_series(table)
        if display is None:
            continue
        display_name, display_values = display

        for id_column in _identifier_columns(table.frame):
            lookup: dict[str, str] = {}
            lookup_scores: dict[str, int] = {}
            for raw_id, raw_display in zip(
                table.frame[id_column].tolist(),
                display_values.tolist(),
                strict=False,
            ):
                record_id = _text_value(raw_id)
                display_value = _text_value(raw_display)
                if record_id and display_value:
                    display_score = _display_value_score(display_value)
                    if record_id not in lookup or display_score > lookup_scores.get(record_id, -100):
                        lookup[record_id] = display_value
                        lookup_scores[record_id] = display_score

            if not lookup:
                continue
            matched_count = sum(1 for value in values if value in lookup)
            coverage = matched_count / len(values)
            if coverage < 0.8:
                continue

            score = coverage * 100
            table_hint_text = f"{table.name} {id_column}".lower()
            if hint and hint in table_hint_text:
                score += 25
            if display_name.lower().endswith("_name") or display_name.lower() == "full_name":
                score += 10

            match = ContextIdentifierMatch(
                score=score,
                table=table,
                id_column=id_column,
                display_name=display_name,
                lookup=lookup,
            )
            if best_match is None or match.score > best_match.score:
                best_match = match

    return best_match


def resolve_identifier_columns(
    task: PublicTask,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if _question_asks_for_identifier(task.question):
        return frame, []
    context_tables = load_context_tables(frame.attrs.get("context_dir", Path()))
    if not context_tables:
        return frame, []

    current_frame = frame.copy()
    transformations: list[dict[str, Any]] = []

    for column in list(current_frame.columns):
        column_name = str(column)
        series = current_frame[column]
        if not (_is_id_like_column(column_name) or _column_is_record_id_like(series)):
            continue

        values = _column_nonempty_values(series)
        if not values:
            continue

        best_match = _match_context_identifier(
            context_tables=context_tables,
            values=values,
            column_name=column_name,
        )
        if best_match is None:
            continue

        resolved = series.map(
            lambda value: best_match.lookup.get(_text_value(value), _text_value(value))
        )
        new_column_name = _make_unique_column_name(
            best_match.display_name,
            [str(existing) for existing in current_frame.columns],
            column_name,
        )
        current_frame[column] = resolved
        current_frame = current_frame.rename(columns={column: new_column_name})
        transformations.append(
            {
                "kind": "identifier_resolution",
                "source_column": column_name,
                "output_column": new_column_name,
                "lookup_table": best_match.table.path,
                "lookup_id_column": best_match.id_column,
            }
        )

    return current_frame, transformations


def _superlative_direction(question: str) -> str | None:
    lowered = question.lower()
    if any(term in lowered for term in MIN_SUPERLATIVE_TERMS):
        return "min"
    if any(term in lowered for term in MAX_SUPERLATIVE_TERMS):
        return "max"
    return None


def _preferred_metric_aliases(question: str) -> tuple[str, ...]:
    lowered = question.lower()
    aliases: list[str] = []
    for term, term_aliases in METRIC_TERM_ALIASES.items():
        if term in lowered:
            aliases.extend(term_aliases)
    return tuple(dict.fromkeys(aliases))


def _numeric_metric_columns(frame: pd.DataFrame, question: str) -> list[str]:
    aliases = _preferred_metric_aliases(question)
    candidates: list[tuple[int, str]] = []

    for column in frame.columns:
        column_name = str(column)
        lowered = column_name.lower()
        if lowered == "id" or lowered.endswith("_id") or lowered.startswith("link_to_"):
            continue

        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().sum() == 0:
            continue

        score = 0
        if aliases:
            for alias in aliases:
                if alias in lowered:
                    score += 20
        elif "total" in lowered or "amount" in lowered or "cost" in lowered:
            score += 10

        if score > 0:
            candidates.append((score, column_name))

    candidates.sort(reverse=True)
    return [column for _, column in candidates]


def _context_attribute_columns(table: ContextTable, id_column: str, display_name: str) -> list[str]:
    priority = (
        display_name,
        "display_name",
        "name",
        "title",
        "label",
        "type",
        "status",
        "date",
        "category",
        "amount",
        "cost",
        "total",
        "value",
    )
    lowered_to_original = {str(column).lower(): str(column) for column in table.frame.columns}
    selected: list[str] = []
    for column in priority:
        original = (
            column
            if column in table.frame.columns
            else lowered_to_original.get(column.lower())
        )
        if original and original not in selected:
            selected.append(original)

    for column in table.frame.columns:
        column_name = str(column)
        if column_name == id_column or column_name in selected:
            continue
        lowered = column_name.lower()
        if lowered.endswith("_id") or lowered.startswith("link_to_"):
            continue
        if len(selected) >= 8:
            break
        selected.append(column_name)
    return selected


def _context_lookup_rows(
    *,
    values: list[str],
    match: ContextIdentifierMatch,
    row_indices: list[int],
) -> list[dict[str, Any]]:
    table = match.table
    attributes = _context_attribute_columns(table, match.id_column, match.display_name)
    rows_by_id = {
        _text_value(row[match.id_column]): row
        for _, row in table.frame.iterrows()
        if _text_value(row[match.id_column])
    }
    payload: list[dict[str, Any]] = []
    for row_index, value in zip(row_indices, values, strict=False):
        if len(payload) >= MAX_CONTEXT_ENRICHMENT_ROWS:
            break
        source_row = rows_by_id.get(value)
        if source_row is None:
            continue
        payload.append(
            {
                "row_index": row_index,
                "id": value,
                "attributes": {
                    column: _json_safe_cell(source_row[column])
                    for column in attributes
                    if column in source_row.index
                },
            }
        )
    return payload


def _summarize_metric_by_key(
    *,
    frame: pd.DataFrame,
    key_column: str,
    metric_column: str,
) -> dict[str, Any]:
    summary: dict[str, float] = {}
    numeric_values = pd.to_numeric(frame[metric_column], errors="coerce")
    for key, value in zip(frame[key_column].tolist(), numeric_values.tolist(), strict=False):
        key_text = _text_value(key)
        if not key_text or pd.isna(value):
            continue
        summary[key_text] = summary.get(key_text, 0.0) + float(value)
    return summary


def _context_metric_summaries(
    *,
    context_tables: list[ContextTable],
    values: list[str],
    row_indices: list[int],
    question: str,
) -> list[dict[str, Any]]:
    candidate_values = set(values)
    summaries: list[dict[str, Any]] = []
    bridge_maps: list[tuple[ContextTable, str, str, dict[str, str]]] = []

    for table in context_tables:
        for link_column in _identifier_columns(table.frame):
            link_values = {_text_value(value) for value in table.frame[link_column].tolist()}
            if not candidate_values & link_values:
                continue

            for metric_column in _numeric_metric_columns(table.frame, question):
                summary = _summarize_metric_by_key(
                    frame=table.frame,
                    key_column=link_column,
                    metric_column=metric_column,
                )
                if any(value in summary for value in values):
                    summaries.append(
                        {
                            "source_table": table.path,
                            "link_column": link_column,
                            "metric_column": metric_column,
                            "aggregation": "sum",
                            "values": [
                                {
                                    "row_index": row_index,
                                    "id": value,
                                    "value": _json_safe_cell(summary.get(value)),
                                }
                                for row_index, value in zip(row_indices, values, strict=False)
                            ],
                        }
                    )

            for bridge_id_column in _identifier_columns(table.frame):
                if bridge_id_column == link_column:
                    continue
                bridge_map: dict[str, str] = {}
                for raw_bridge_id, raw_candidate_id in zip(
                    table.frame[bridge_id_column].tolist(),
                    table.frame[link_column].tolist(),
                    strict=False,
                ):
                    bridge_id = _text_value(raw_bridge_id)
                    candidate_id = _text_value(raw_candidate_id)
                    if bridge_id and candidate_id in candidate_values:
                        bridge_map[bridge_id] = candidate_id
                if bridge_map:
                    bridge_maps.append((table, link_column, bridge_id_column, bridge_map))

    for bridge_table, candidate_link_column, bridge_id_column, bridge_map in bridge_maps:
        bridge_ids = set(bridge_map)
        for table in context_tables:
            if table.path == bridge_table.path:
                continue
            for link_column in _identifier_columns(table.frame):
                link_values = {_text_value(value) for value in table.frame[link_column].tolist()}
                if not bridge_ids & link_values:
                    continue
                for metric_column in _numeric_metric_columns(table.frame, question):
                    by_bridge = _summarize_metric_by_key(
                        frame=table.frame,
                        key_column=link_column,
                        metric_column=metric_column,
                    )
                    by_candidate: dict[str, float] = {}
                    for bridge_id, candidate_id in bridge_map.items():
                        if bridge_id not in by_bridge:
                            continue
                        by_candidate[candidate_id] = (
                            by_candidate.get(candidate_id, 0.0) + by_bridge[bridge_id]
                        )
                    if any(value in by_candidate for value in values):
                        summaries.append(
                            {
                                "source_table": table.path,
                                "link_column": link_column,
                                "metric_column": metric_column,
                                "aggregation": "sum",
                                "via_table": bridge_table.path,
                                "via_id_column": bridge_id_column,
                                "via_candidate_column": candidate_link_column,
                                "values": [
                                    {
                                        "row_index": row_index,
                                        "id": value,
                                        "value": _json_safe_cell(by_candidate.get(value)),
                                    }
                                    for row_index, value in zip(row_indices, values, strict=False)
                                ],
                            }
                        )

    return summaries[:MAX_CONTEXT_METRIC_SUMMARIES]


def _context_enrichment_payload(task: PublicTask, frame: pd.DataFrame) -> list[dict[str, Any]]:
    context_tables = load_context_tables(task.context_dir)
    if not context_tables:
        return []

    enrichments: list[dict[str, Any]] = []
    row_indices = list(range(min(len(frame), MAX_CANDIDATE_PAYLOAD_ROWS)))
    for column_index, column in enumerate(frame.columns):
        column_name = str(column)
        row_value_pairs = [
            (row_index, _text_value(frame.iloc[row_index, column_index]))
            for row_index in row_indices
            if _text_value(frame.iloc[row_index, column_index])
        ]
        values = [value for _, value in row_value_pairs]
        value_row_indices = [row_index for row_index, _ in row_value_pairs]
        if not values or not any(_looks_like_record_id(value) for value in values):
            continue

        match = _match_context_identifier(
            context_tables=context_tables,
            values=values,
            column_name=column_name,
        )
        enrichment: dict[str, Any] = {
            "column_index": column_index,
            "column_name": column_name,
        }
        if match is not None:
            enrichment["lookup"] = {
                "table": match.table.path,
                "id_column": match.id_column,
                "display_column": match.display_name,
                "rows": _context_lookup_rows(
                    values=values,
                    match=match,
                    row_indices=value_row_indices,
                ),
            }

        metric_summaries = _context_metric_summaries(
            context_tables=context_tables,
            values=values,
            row_indices=value_row_indices,
            question=task.question,
        )
        if metric_summaries:
            enrichment["numeric_summaries"] = metric_summaries

        if "lookup" in enrichment or "numeric_summaries" in enrichment:
            enrichments.append(enrichment)

    return enrichments


def _question_requests_metric_value(question: str) -> bool:
    lowered = question.lower()
    return bool(
        re.search(r"\b(what|state|give|return|provide)\b.*\b(cost|amount|value|score|total)\b", lowered)
        or re.search(r"\b(cost|amount|value|score|total)\b.*\b(is|are)\b", lowered)
    )


def _preferred_superlative_answer_column(frame: pd.DataFrame) -> str | None:
    if len(frame.columns) <= 1:
        return None

    lowered_to_original = {str(column).lower(): str(column) for column in frame.columns}
    for preferred in DISPLAY_COLUMN_PRIORITY:
        column = lowered_to_original.get(preferred.lower())
        if column is not None:
            return column

    for column in frame.columns:
        lowered = str(column).lower()
        if lowered.endswith("_name") or lowered.endswith(" name"):
            return str(column)

    identifier_columns = _identifier_columns(frame)
    if identifier_columns:
        return identifier_columns[0]

    text_columns: list[str] = []
    for column in frame.columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().sum() == 0:
            text_columns.append(str(column))
    return text_columns[0] if len(text_columns) == 1 else None


def _question_entity_name(question: str) -> str | None:
    entity = _infer_generic_target_entity(question.lower())
    if entity is None:
        return None
    token = re.sub(r"[^a-z0-9_]+", "_", entity).strip("_")
    if not token:
        return None
    return token[:-1] if token.endswith("s") and len(token) > 1 else token


def _table_entity_candidates(table: ContextTable) -> list[str]:
    candidates: list[str] = []

    def add(value: str) -> None:
        value = value.strip().lower()
        if value and value not in candidates:
            candidates.append(value)

    table_name = table.name.lower()
    add(table_name)
    for prefix in ("csv_", "json_"):
        if table_name.startswith(prefix):
            add(table_name.removeprefix(prefix))

    for column in table.frame.columns:
        lowered = str(column).lower()
        if lowered.endswith("_id") and len(lowered) > 3:
            add(lowered.removesuffix("_id"))
    return candidates


def _table_id_column(table: ContextTable, entity: str) -> str | None:
    lowered_to_original = {str(column).lower(): str(column) for column in table.frame.columns}
    for candidate in (f"{entity}_id", "id"):
        column = lowered_to_original.get(candidate)
        if column is not None:
            return column
    return None


def _table_link_column(table: ContextTable, entity: str) -> str | None:
    lowered_to_original = {str(column).lower(): str(column) for column in table.frame.columns}
    for candidate in (f"link_to_{entity}", f"link_to_{entity}s"):
        column = lowered_to_original.get(candidate)
        if column is not None:
            return column
    return None


def _find_entity_table(tables: list[ContextTable], entity: str) -> tuple[ContextTable, str, str] | None:
    for table in tables:
        if entity not in _table_entity_candidates(table):
            continue
        id_column = _table_id_column(table, entity)
        display = _display_series(table)
        if id_column is not None and display is not None:
            display_name, _ = display
            return table, id_column, display_name

    for table in tables:
        id_column = _table_id_column(table, entity)
        display = _display_series(table)
        if id_column is not None and display is not None:
            display_name, _ = display
            return table, id_column, display_name
    return None


def _metric_entity_frame(
    tables: list[ContextTable],
    *,
    entity_table: ContextTable,
    entity_id_column: str,
    entity_name_column: str,
    metric_table: ContextTable,
    metric_column: str,
    entity: str,
) -> pd.DataFrame | None:
    metric_frame = metric_table.frame.copy()
    entity_frame = entity_table.frame[[entity_id_column, entity_name_column]].copy()
    direct_entity_link = _table_link_column(metric_table, entity)
    if direct_entity_link is not None:
        merged = metric_frame.merge(
            entity_frame,
            left_on=direct_entity_link,
            right_on=entity_id_column,
            how="inner",
        )
        return merged[[entity_name_column, metric_column]]

    for bridge_table in tables:
        if bridge_table is metric_table or bridge_table is entity_table:
            continue
        for bridge_entity in _table_entity_candidates(bridge_table):
            bridge_id_column = _table_id_column(bridge_table, bridge_entity)
            if bridge_id_column is None:
                continue
            metric_bridge_link = _table_link_column(metric_table, bridge_entity)
            bridge_entity_link = _table_link_column(bridge_table, entity)
            if metric_bridge_link is None or bridge_entity_link is None:
                continue
            bridge_frame = bridge_table.frame[[bridge_id_column, bridge_entity_link]].copy()
            merged = metric_frame.merge(
                bridge_frame,
                left_on=metric_bridge_link,
                right_on=bridge_id_column,
                how="inner",
            ).merge(
                entity_frame,
                left_on=bridge_entity_link,
                right_on=entity_id_column,
                how="inner",
            )
            return merged[[entity_name_column, metric_column]]
    return None


def _context_metric_score(table: ContextTable, metric_column: str, question: str) -> int:
    lowered_question = question.lower()
    column_name = metric_column.lower()
    aliases = _preferred_metric_aliases(question)
    score = 0
    for index, alias in enumerate(aliases):
        if column_name == alias:
            score += 100 - index
        elif alias in column_name:
            score += 60 - index

    if "cost" in lowered_question:
        if column_name == "cost":
            score += 80
    return score


def apply_context_superlative_verifier(
    task: PublicTask,
) -> tuple[pd.DataFrame | None, list[dict[str, Any]]]:
    direction = _superlative_direction(task.question)
    entity = _question_entity_name(task.question)
    if direction is None or entity is None or _question_requests_metric_value(task.question):
        return None, []

    tables = load_context_tables(task.context_dir)
    entity_match = _find_entity_table(tables, entity)
    if entity_match is None:
        return None, []
    entity_table, entity_id_column, entity_name_column = entity_match

    best_output: pd.DataFrame | None = None
    best_transform: dict[str, Any] | None = None
    best_score: int | None = None
    for metric_table in tables:
        if metric_table is entity_table:
            continue
        metric_columns = _numeric_metric_columns(metric_table.frame, task.question)
        for metric_column in metric_columns:
            metric_entity_frame = _metric_entity_frame(
                tables,
                entity_table=entity_table,
                entity_id_column=entity_id_column,
                entity_name_column=entity_name_column,
                metric_table=metric_table,
                metric_column=metric_column,
                entity=entity,
            )
            if metric_entity_frame is None or metric_entity_frame.empty:
                continue
            numeric_values = pd.to_numeric(metric_entity_frame[metric_column], errors="coerce")
            valid_values = numeric_values.dropna()
            if valid_values.empty:
                continue
            target_value = valid_values.min() if direction == "min" else valid_values.max()
            mask = numeric_values.eq(target_value)
            if not mask.any():
                continue
            output = metric_entity_frame.loc[mask, [entity_name_column]].drop_duplicates().reset_index(drop=True)
            if output.empty:
                continue
            score = _context_metric_score(metric_table, metric_column, task.question)
            if best_score is not None and score <= best_score:
                continue
            best_output = output
            best_transform = {
                "kind": "context_superlative_verification",
                "direction": direction,
                "entity": entity,
                "entity_table": entity_table.name,
                "metric_table": metric_table.name,
                "metric_column": metric_column,
                "target_value": _json_safe_cell(target_value),
                "output_rows": len(output),
                "output_column": entity_name_column,
                "metric_score": score,
            }
            best_score = score

    if best_output is None or best_transform is None:
        return None, []
    return best_output, [best_transform]


def apply_superlative_verifier(
    task: PublicTask,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    direction = _superlative_direction(task.question)
    if direction is None or len(frame) <= 1:
        return frame, []

    metric_columns = _numeric_metric_columns(frame, task.question)
    if not metric_columns:
        return frame, []

    metric_column = metric_columns[0]
    metric_values = pd.to_numeric(frame[metric_column], errors="coerce")
    valid_values = metric_values.dropna()
    if valid_values.empty:
        return frame, []

    target_value = valid_values.min() if direction == "min" else valid_values.max()
    mask = metric_values.eq(target_value)
    if not mask.any():
        return frame, []

    output = frame.loc[mask].reset_index(drop=True)
    dropped_metric = False
    selected_answer_column: str | None = None
    if len(output.columns) > 1 and not _question_requests_metric_value(task.question):
        output = output.drop(columns=[metric_column])
        dropped_metric = True
        selected_answer_column = _preferred_superlative_answer_column(output)
        if selected_answer_column is not None and len(output.columns) > 1:
            output = output[[selected_answer_column]]
    if mask.all() and not dropped_metric and selected_answer_column is None:
        return frame, []

    return output, [
        {
            "kind": "superlative_verification",
            "direction": direction,
            "metric_column": metric_column,
            "target_value": _json_safe_cell(target_value),
            "input_rows": len(frame),
            "output_rows": len(output),
            "dropped_metric_column": dropped_metric,
            "selected_answer_column": selected_answer_column,
        }
    ]


def _find_existing_columns(frame: pd.DataFrame, candidates: tuple[str, ...]) -> list[str]:
    lowered_to_original = {str(column).lower(): str(column) for column in frame.columns}
    matches: list[str] = []
    for candidate in candidates:
        column = lowered_to_original.get(candidate.lower())
        if column is not None and column not in matches:
            matches.append(column)
    return matches


def _question_target_columns(question: str, frame: pd.DataFrame) -> tuple[str, ...]:
    if len(frame.columns) <= 1:
        return ()

    lowered_question = question.lower()
    tokens = _normalized_question_tokens(question)
    compact_question = re.sub(r"[^a-z0-9]+", "", lowered_question)
    generic_answer_terms = {
        "id",
        "identifier",
        "code",
        "reference",
        "name",
        "title",
        "label",
        "description",
        "status",
        "type",
        "category",
        "date",
        "year",
        "time",
        "amount",
        "cost",
        "price",
        "score",
        "value",
        "total",
        "count",
        "phone",
        "telephone",
        "url",
        "text",
    }
    asks_identifier = _question_asks_for_identifier(question)
    asks_textual_answer = bool(re.search(r"\b(?:comment|text|body|content)\b", lowered_question))
    asks_name_answer = bool(re.search(r"\b(?:name|user|author|owner|editor|person|people)\b", lowered_question))
    asks_count_answer = _question_asks_count_metric(question)
    superlative_direction = _superlative_direction(question)
    metric_sort_terms = {"score", "cost", "price", "amount", "count", "total", "value", "views", "view"}
    scored: list[tuple[int, str]] = []
    for index, column in enumerate(frame.columns):
        column_name = str(column)
        column_lower = column_name.lower()
        aliases = _column_question_aliases(column_name)
        score = 0
        if column_lower in lowered_question:
            score += 50
        elif _is_id_like_column(column_name) and not asks_identifier:
            continue
        score += 10 * len(aliases & tokens)
        if any(alias in compact_question for alias in aliases if len(alias) > 3):
            score += 20
        if aliases & generic_answer_terms & tokens:
            score += 15
        if asks_identifier and _is_id_like_column(column_name):
            score += 40
        if asks_count_answer:
            count_aliases = aliases & {"answer", "count", "total"}
            if count_aliases or column_lower.endswith("count") or column_lower.endswith("_count"):
                score += 120
            elif _is_display_like_column(column_name) or aliases & {"name", "title", "label", "status", "type"}:
                score -= 80
        if asks_textual_answer and (
            aliases & {"comment", "text", "body", "content", "description"}
            or column_lower in {"comment", "text", "body", "content"}
        ):
            score += 80
        if asks_name_answer and (
            _is_display_like_column(column_name)
            or aliases & {"display", "displayname", "userdisplayname", "name", "owner", "editor", "author"}
        ):
            score += 35
        if (
            superlative_direction is not None
            and asks_textual_answer
            and aliases & metric_sort_terms
            and not (aliases & {"comment", "text", "body", "content"})
        ):
            score -= 60
        if score > 0:
            scored.append((score * 1000 - index, column_name))
    scored.sort(reverse=True)
    ordered = tuple(dict.fromkeys(column for _, column in scored[:3]))
    if asks_identifier and not asks_name_answer:
        id_columns = [column for _, column in scored if _is_id_like_column(column)]
        if id_columns:
            return (id_columns[0],)
    if asks_count_answer:
        count_columns = [
            column
            for _, column in scored
            if (
                _column_question_aliases(column) & {"answer", "count", "total"}
                or column.lower().endswith("count")
                or column.lower().endswith("_count")
            )
        ]
        if count_columns:
            return (count_columns[0],)
    if asks_textual_answer:
        text_columns = [
            column
            for _, column in scored
            if (
                _column_question_aliases(column) & {"text", "body", "content", "description"}
                or column.lower() in {"comment", "text", "body", "content"}
                or (
                    "comment" in _column_question_aliases(column)
                    and "count" not in _column_question_aliases(column)
                    and column.lower() not in {"commentcount", "comment_count"}
                )
            )
        ]
        if text_columns:
            return (text_columns[0],)
    return ordered


def _text_answer_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        column_name = str(column)
        column_lower = column_name.lower()
        aliases = _column_question_aliases(column_name)
        if (
            aliases & {"text", "body", "content", "description"}
            or column_lower in {"comment", "text", "body", "content"}
            or (
                "comment" in aliases
                and "count" not in aliases
                and column_lower not in {"commentcount", "comment_count"}
            )
        ):
            columns.append(column_name)
    return columns


def apply_question_column_pruner(
    task: PublicTask,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    target_columns = _question_target_columns(task.question, frame)
    if not target_columns:
        return frame, []
    if tuple(str(column) for column in frame.columns) == target_columns:
        return frame, []

    output = frame.loc[:, list(target_columns)].copy().reset_index(drop=True)
    return output, [
        {
            "kind": "question_column_pruning",
            "selected_columns": list(target_columns),
            "input_columns": [str(column) for column in frame.columns],
            "reason": "question_target_column",
        }
    ]


def apply_answer_column_verifier(
    task: PublicTask,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    output, transforms = apply_question_column_pruner(task, frame)
    for transform in transforms:
        transform["kind"] = "answer_column_verification"
    return output, transforms


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.notna().any():
            columns.append(str(column))
    return columns


def _best_numeric_column(
    columns: list[str],
    preferred_terms: tuple[str, ...],
    *,
    require_match: bool = False,
) -> str | None:
    for term in preferred_terms:
        for column in columns:
            if term in column.lower():
                return column
    if require_match:
        return None
    return columns[0] if columns else None


def _question_matched_numeric_column(
    columns: list[str],
    question: str,
    *,
    exclude: str | None = None,
    require_match: bool = False,
) -> str | None:
    lowered_question = question.lower()
    tokens = _normalized_question_tokens(question)
    scored: list[tuple[int, str]] = []
    for index, column in enumerate(columns):
        if column == exclude:
            continue
        aliases = _column_question_aliases(column)
        score = 0
        if column.lower() in lowered_question:
            score += 50
        score += 10 * len(aliases & tokens)
        if score > 0:
            scored.append((score * 1000 - index, column))
    if scored:
        scored.sort(reverse=True)
        return scored[0][1]
    if require_match:
        return None
    return next((column for column in columns if column != exclude), None)


def apply_aggregate_ratio_verifier(
    task: PublicTask,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    question = task.question.lower()
    if len(frame) != 1 or len(frame.columns) < 2:
        return frame, []

    numeric_columns = _numeric_columns(frame)
    if len(numeric_columns) < 2:
        return frame, []

    if "percentage" in question or "percent" in question:
        denominator = _best_numeric_column(
            numeric_columns,
            ("total", "all", "count", "denominator"),
            require_match=True,
        )
        numerator = _question_matched_numeric_column(
            numeric_columns,
            question,
            exclude=denominator,
            require_match=True,
        )
        if numerator is None or denominator is None:
            return frame, []
        denominator_value = pd.to_numeric(frame[denominator], errors="coerce").iloc[0]
        numerator_value = pd.to_numeric(frame[numerator], errors="coerce").iloc[0]
        if pd.isna(denominator_value) or float(denominator_value) == 0 or pd.isna(numerator_value):
            return frame, []
        value = float(numerator_value) * 100 / float(denominator_value)
        if value == 0:
            return frame, []
        output = pd.DataFrame({"percentage": [value]})
        return output, [
            {
                "kind": "aggregate_ratio_verification",
                "mode": "percentage",
                "numerator_column": numerator,
                "denominator_column": denominator,
            }
        ]

    if "compared to" in question or "how many times" in question or "ratio" in question:
        numerator = _question_matched_numeric_column(numeric_columns, question, require_match=True)
        denominator_candidates = [column for column in numeric_columns if column != numerator]
        denominator = _best_numeric_column(
            denominator_candidates,
            ("total", "all", "count", "denominator"),
            require_match=True,
        )
        if numerator is None or denominator is None:
            return frame, []
        denominator_value = pd.to_numeric(frame[denominator], errors="coerce").iloc[0]
        numerator_value = pd.to_numeric(frame[numerator], errors="coerce").iloc[0]
        if pd.isna(denominator_value) or float(denominator_value) == 0 or pd.isna(numerator_value):
            return frame, []
        value = float(numerator_value) / float(denominator_value)
        output = pd.DataFrame({"ratio": [value]})
        return output, [
            {
                "kind": "aggregate_ratio_verification",
                "mode": "ratio",
                "numerator_column": numerator,
                "denominator_column": denominator,
            }
        ]

    return frame, []


def _column_value_signature(series: pd.Series) -> tuple[str, ...]:
    return tuple(sorted(_text_value(value) for value in series.tolist()))


def _non_empty_value_count(series: pd.Series) -> int:
    return sum(1 for value in series.tolist() if _text_value(value))


def _contract_expected_frame_columns(contract: AnswerContract, frame: pd.DataFrame) -> set[str]:
    expected_lower = {column.lower() for column in contract.expected_columns}
    return {str(column) for column in frame.columns if str(column).lower() in expected_lower}


def apply_column_only_compactor(
    task: PublicTask,
    frame: pd.DataFrame,
    contract: AnswerContract,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Build a conservative column-only candidate without changing row semantics."""
    if len(frame.columns) <= 1:
        return frame, []

    features = extract_question_features(task.question)
    selected_columns = [str(column) for column in frame.columns]
    protected_columns = _contract_expected_frame_columns(contract, frame)
    removed: list[dict[str, Any]] = []

    def can_remove(column: str) -> bool:
        return column not in protected_columns

    non_empty_columns = {
        str(column): _non_empty_value_count(frame[column])
        for column in frame.columns
    }
    if any(count > 0 for count in non_empty_columns.values()):
        for column, count in list(non_empty_columns.items()):
            if count == 0 and len(selected_columns) > 1 and can_remove(column):
                selected_columns.remove(column)
                removed.append(
                    {
                        "column": column,
                        "reason": "empty_column",
                    }
                )

    signatures: dict[tuple[str, ...], str] = {}
    for column in list(selected_columns):
        signature = _column_value_signature(frame[column])
        previous = signatures.get(signature)
        if previous is not None and len(selected_columns) > 1 and can_remove(column):
            selected_columns.remove(column)
            removed.append(
                {
                    "column": column,
                    "reason": "duplicate_column_values",
                    "kept_column": previous,
                }
            )
            continue
        signatures[signature] = column

    strong_metric_request = (
        features.asks_aggregation
        or features.asks_ratio_or_percentage
        or features.asks_superlative
        or any(
            term in features.strong_terms
            for term in ("average", "avg", "mean", "sum", "total", "count", "how many")
        )
    )
    if (
        features.asks_entity_or_list
        and not features.asks_multi_attribute
        and not strong_metric_request
        and contract.kind in {"entity_list", "table", "multi_attribute", "attribute_lookup"}
        and not (contract.kind in {"two_attribute", "multi_attribute"} and len(protected_columns) >= 2)
    ):
        display_columns = [
            column
            for column in selected_columns
            if _is_display_like_column(column)
        ]
        if display_columns and len(display_columns) < len(selected_columns):
            for column in list(selected_columns):
                if column not in display_columns and len(selected_columns) > len(display_columns) and can_remove(column):
                    selected_columns.remove(column)
                    removed.append(
                        {
                            "column": column,
                            "reason": "entity_list_display_answer_present",
                            "kept_columns": display_columns,
                        }
                    )

    if contract.kind in {"entity_list", "table", "multi_attribute", "attribute_lookup"} and len(frame) > 1:
        display_columns = [
            column
            for column in selected_columns
            if _is_display_like_column(column)
        ]
        non_constant_display_columns = [
            column
            for column in display_columns
            if frame[column].map(_text_value).nunique(dropna=False) > 1
        ]
        if non_constant_display_columns:
            for column in list(display_columns):
                if (
                    column not in non_constant_display_columns
                    and len(selected_columns) > 1
                    and column in selected_columns
                    and can_remove(column)
                ):
                    selected_columns.remove(column)
                    removed.append(
                        {
                            "column": column,
                            "reason": "constant_display_helper",
                            "kept_columns": non_constant_display_columns,
                        }
                    )

    if len(selected_columns) == len(frame.columns) or not selected_columns:
        return frame, []

    output = frame.loc[:, selected_columns].copy().reset_index(drop=True)
    return output, [
        {
            "kind": "column_only_compaction",
            "selected_columns": selected_columns,
            "removed_columns": [item["column"] for item in removed],
            "removed_column_reasons": removed,
            "input_columns": [str(column) for column in frame.columns],
            "row_count_preserved": True,
            "reason": "safe_column_only_cleanup",
        }
    ]


def _approx_equal(left: float, right: float, *, rel_tol: float = 1e-6, abs_tol: float = 1e-9) -> bool:
    return abs(left - right) <= max(abs_tol, rel_tol * max(abs(left), abs(right), 1.0))


def _best_column_for_text_segment(columns: list[str], text: str) -> str | None:
    tokens = _normalized_question_tokens(text)
    compact_text = re.sub(r"[^a-z0-9]+", "", text.lower())
    scored: list[tuple[int, str]] = []
    for index, column in enumerate(columns):
        aliases = _column_question_aliases(column)
        score = 10 * len(aliases & tokens)
        if any(alias in compact_text for alias in aliases if len(alias) > 3):
            score += 20
        if score > 0:
            scored.append((score * 1000 - index, column))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1]


def apply_ratio_scale_compactor(
    task: PublicTask,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    question = task.question.lower()
    asks_percentage = "percentage" in question or "percent" in question
    asks_ratio = "compared to" in question or "how many times" in question or "ratio" in question
    if not (asks_percentage or asks_ratio):
        return frame, []
    if len(frame) != 1 or len(frame.columns) < 2:
        return frame, []

    numeric_columns = _numeric_columns(frame)
    if len(numeric_columns) < 2:
        return frame, []

    result_terms = ("ratio", "percentage", "percent", "rate", "share", "proportion")
    result_columns = [
        column
        for column in numeric_columns
        if any(term in column.lower() for term in result_terms)
    ]
    result_column = result_columns[0] if result_columns else None
    component_columns = [column for column in numeric_columns if column != result_column]
    if len(component_columns) < 2:
        return frame, []

    numerator: str | None = None
    denominator: str | None = None
    if "compared to" in question:
        before, after = question.split("compared to", 1)
        numerator = _best_column_for_text_segment(component_columns, before)
        denominator = _best_column_for_text_segment(
            [column for column in component_columns if column != numerator],
            after,
        )
    if denominator is None:
        denominator = _best_numeric_column(
            component_columns,
            ("total", "all", "denominator", "count"),
            require_match=True,
        )
    if numerator is None:
        numerator = _question_matched_numeric_column(
            component_columns,
            question,
            exclude=denominator,
            require_match=True,
        )
    if numerator is None or denominator is None:
        return frame, []

    denominator_value = pd.to_numeric(frame[denominator], errors="coerce").iloc[0]
    numerator_value = pd.to_numeric(frame[numerator], errors="coerce").iloc[0]
    if pd.isna(denominator_value) or pd.isna(numerator_value) or float(denominator_value) == 0:
        return frame, []

    ratio_value = float(numerator_value) / float(denominator_value)
    percentage_value = ratio_value * 100
    transform: dict[str, Any] = {
        "kind": "ratio_scale_compaction",
        "numerator_column": numerator,
        "denominator_column": denominator,
        "denominator_nonzero": True,
    }

    if asks_ratio:
        if result_column is not None:
            result_value = pd.to_numeric(frame[result_column], errors="coerce").iloc[0]
            if pd.isna(result_value):
                return frame, []
            result_float = float(result_value)
            if not (
                _approx_equal(result_float, ratio_value)
                or _approx_equal(result_float, percentage_value)
            ):
                return frame, []
            transform["source_result_column"] = result_column
            transform["source_result_value"] = result_float
        if ratio_value < 0:
            return frame, []
        output = pd.DataFrame({"ratio": [ratio_value]})
        transform.update(
            {
                "mode": "ratio",
                "value": ratio_value,
                "sanity_check": "numerator_divided_by_denominator",
            }
        )
        return output, [transform]

    if asks_percentage:
        if result_column is not None:
            result_value = pd.to_numeric(frame[result_column], errors="coerce").iloc[0]
            if pd.isna(result_value):
                return frame, []
            result_float = float(result_value)
            if not 0 <= result_float <= 100:
                return frame, []
            if not _approx_equal(result_float, percentage_value):
                return frame, []
            value = result_float
            transform["source_result_column"] = result_column
            transform["source_result_value"] = result_float
        else:
            value = percentage_value
            if not 0 <= value <= 100:
                return frame, []
        output = pd.DataFrame({"percentage": [value]})
        transform.update(
            {
                "mode": "percentage",
                "value": value,
                "sanity_check": "percentage_matches_components",
            }
        )
        return output, [transform]

    return frame, []


def _boolean_indicator_series(series: pd.Series) -> pd.Series | None:
    mapped = series.map(
        lambda value: (
            None
            if pd.isna(value)
            else (
                1
                if _text_value(value).strip().lower() in {"1", "true", "yes", "y"}
                else 0
                if _text_value(value).strip().lower() in {"0", "false", "no", "n"}
                else None
            )
        )
    )
    non_null = mapped.dropna()
    if non_null.empty:
        return None
    if not set(non_null.astype(int).unique()).issubset({0, 1}):
        return None
    return mapped.astype("Float64")


def apply_salvaged_boolean_percentage_compactor(
    task: PublicTask,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if not bool(frame.attrs.get("databao_salvaged_latest_query_result")):
        return frame, []
    lowered_question = task.question.lower()
    if not re.search(r"\b(?:percentage|percent|proportion|share)\b", lowered_question):
        return frame, []

    question_tokens = _normalized_question_tokens(task.question)
    negated_have = bool(
        re.search(r"\b(?:do|does|did)\s+not\s+have\b", lowered_question)
        or re.search(r"\bwithout\b", lowered_question)
    )
    positive_have = bool(re.search(r"\b(?:have|has|with)\b", lowered_question))
    if not (negated_have or positive_have):
        return frame, []

    boolean_candidates: list[tuple[str, pd.Series, int]] = []
    for column in frame.columns:
        column_name = str(column)
        if _is_id_like_column(column_name):
            continue
        aliases = _column_question_aliases(column_name)
        overlap = aliases & question_tokens
        if not overlap:
            continue
        indicator = _boolean_indicator_series(frame[column])
        if indicator is None:
            continue
        boolean_candidates.append((column_name, indicator, len(overlap)))

    if not boolean_candidates:
        return frame, []

    column_name, indicator, overlap_count = max(boolean_candidates, key=lambda item: item[2])
    valid_values = indicator.dropna()
    denominator = int(len(valid_values))
    if denominator == 0:
        return frame, []
    target_value = 0 if negated_have else 1
    numerator = int((valid_values.astype(int) == target_value).sum())
    percentage = numerator * 100.0 / denominator
    output = pd.DataFrame({"percentage": [percentage]})
    output.attrs.update(frame.attrs)
    transform = {
        "kind": "salvaged_boolean_percentage_compaction",
        "source_boolean_column": column_name,
        "target_value": target_value,
        "numerator": numerator,
        "denominator": denominator,
        "denominator_nonzero": True,
        "matched_column_alias_count": overlap_count,
        "databao_salvaged_latest_query_result": True,
        "operation": "percentage",
    }
    return output, [transform]


def apply_salvaged_detail_count_compactor(
    task: PublicTask,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if frame.empty or len(frame.columns) == 0:
        return frame, []
    if frame.attrs.get("databao_submit_called") is not False and not frame.attrs.get(
        "databao_salvaged_latest_query_result"
    ):
        return frame, []
    if not _question_asks_count_metric(task.question):
        return frame, []
    if extract_question_features(task.question).asks_ratio_or_percentage:
        return frame, []
    count_like_columns = [
        str(column)
        for column in frame.columns
        if any(term in str(column).lower() for term in ("answer", "count", "total", "number"))
    ]
    if count_like_columns and len(frame.columns) <= 2:
        return frame, []
    id_columns = [str(column) for column in frame.columns if _is_id_like_column(str(column))]
    if id_columns:
        count_value = int(frame[id_columns[0]].dropna().nunique())
        count_basis = "distinct_id"
        count_column = id_columns[0]
    else:
        count_value = int(len(frame))
        count_basis = "row_count"
        count_column = None
    output = pd.DataFrame({"count": [count_value]})
    output.attrs.update(frame.attrs)
    transform = {
        "kind": "salvaged_detail_count_compaction",
        "operation": "count",
        "count_basis": count_basis,
        "count_column": count_column,
        "input_rows": int(len(frame)),
        "input_columns": [str(column) for column in frame.columns],
    }
    return output, [transform]


GENERIC_FILTER_TOKEN_STOPWORDS = {
    "what",
    "which",
    "percentage",
    "percent",
    "proportion",
    "share",
    "ratio",
    "record",
    "records",
    "row",
    "rows",
    "entry",
    "entries",
    "table",
    "tables",
    "does",
    "did",
    "have",
    "has",
    "with",
    "without",
    "not",
    "and",
    "the",
    "this",
    "that",
    "these",
    "those",
}


def _question_filter_tokens(question: str) -> set[str]:
    return {
        token
        for token in _normalized_question_tokens(question)
        if token not in GENERIC_FILTER_TOKEN_STOPWORDS and len(token) > 2
    }


def _row_text_for_filter_match(row: pd.Series) -> str:
    parts: list[str] = []
    for column, value in row.items():
        text = _text_value(value)
        if not text:
            continue
        if _is_id_like_column(str(column)) and RECORD_ID_PATTERN.match(text):
            continue
        parts.append(text)
    return " ".join(parts)


def _question_matched_filter_frame(table: ContextTable, question_tokens: set[str]) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    if len(table.frame) == 0 or len(table.frame) > 5000 or not question_tokens:
        return table.frame.iloc[0:0].copy(), None
    matches: list[int] = []
    matched_terms: set[str] = set()
    for index, row in table.frame.iterrows():
        row_tokens = _normalized_question_tokens(_row_text_for_filter_match(row))
        overlap = question_tokens & row_tokens
        if len(overlap) >= 2:
            matches.append(index)
            matched_terms.update(overlap)
    if not matches:
        return table.frame.iloc[0:0].copy(), None
    return (
        table.frame.loc[matches].copy().reset_index(drop=True),
        {
            "filter_table": table.name,
            "filter_table_path": table.path,
            "matched_terms": sorted(matched_terms),
            "matched_rows": len(matches),
        },
    )


def _relation_column_pairs(left: pd.DataFrame, right: pd.DataFrame) -> list[tuple[str, str, int]]:
    scored_pairs: list[tuple[int, str, str, int]] = []
    left_columns = _identifier_columns(left)
    right_columns = _identifier_columns(right)
    for left_column in left_columns:
        left_values = {_text_value(value) for value in left[left_column].dropna().tolist()}
        left_values.discard("")
        if not left_values:
            continue
        for right_column in right_columns:
            right_values = {_text_value(value) for value in right[right_column].dropna().head(200000).tolist()}
            right_values.discard("")
            overlap_count = len(left_values & right_values)
            if overlap_count:
                score = overlap_count
                if left_column.lower() not in {"id", "record_id", "numeric_id"}:
                    score += 1000
                if right_column.lower() not in {"id", "record_id", "numeric_id"}:
                    score += 100
                scored_pairs.append((score, left_column, right_column, overlap_count))
    scored_pairs.sort(key=lambda item: item[0], reverse=True)
    return [(left_column, right_column, overlap_count) for _, left_column, right_column, overlap_count in scored_pairs]


def apply_salvaged_context_boolean_percentage_repair(
    task: PublicTask,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    salvaged_result = bool(frame.attrs.get("databao_salvaged_latest_query_result"))
    not_submitted = frame.attrs.get("databao_submit_called") is False
    if not (salvaged_result or not_submitted):
        return frame, []
    if len(frame) != 0 and not not_submitted:
        return frame, []
    lowered_question = task.question.lower()
    if not re.search(r"\b(?:percentage|percent|proportion|share)\b", lowered_question):
        return frame, []

    question_tokens = _question_filter_tokens(task.question)
    if len(question_tokens) < 2:
        return frame, []
    negated_have = bool(
        re.search(r"\b(?:do|does|did)\s+not\s+have\b", lowered_question)
        or re.search(r"\bwithout\b", lowered_question)
    )
    positive_have = bool(re.search(r"\b(?:have|has|with)\b", lowered_question))
    if not (negated_have or positive_have):
        return frame, []

    context_tables = load_context_tables(task.context_dir, heuristic_level=_heuristic_level())
    best: tuple[int, pd.DataFrame, dict[str, Any]] | None = None
    for filter_table in context_tables:
        filtered_frame, filter_evidence = _question_matched_filter_frame(filter_table, question_tokens)
        if filter_evidence is None or filtered_frame.empty:
            continue

        for indicator_table in context_tables:
            boolean_columns: list[tuple[str, pd.Series, int]] = []
            for column in indicator_table.frame.columns:
                column_name = str(column)
                if _is_id_like_column(column_name):
                    continue
                aliases = _column_question_aliases(column_name)
                overlap_count = len(aliases & _normalized_question_tokens(task.question))
                if overlap_count == 0:
                    continue
                indicator = _boolean_indicator_series(indicator_table.frame[column])
                if indicator is not None:
                    boolean_columns.append((column_name, indicator, overlap_count))
            if not boolean_columns:
                continue

            if filter_table.name == indicator_table.name and filter_table.path == indicator_table.path:
                joined_frame = filtered_frame
                relation_evidence: dict[str, Any] = {"relation": "same_table"}
            else:
                relation_pairs = _relation_column_pairs(filtered_frame, indicator_table.frame)
                if not relation_pairs:
                    continue
                left_column, right_column, overlap_count = relation_pairs[0]
                left_values = {_text_value(value) for value in filtered_frame[left_column].dropna().tolist()}
                left_values.discard("")
                mask = indicator_table.frame[right_column].map(lambda value: _text_value(value) in left_values)
                joined_frame = indicator_table.frame.loc[mask].copy().reset_index(drop=True)
                relation_evidence = {
                    "relation": "identifier_join",
                    "left_table": filter_table.name,
                    "left_column": left_column,
                    "right_table": indicator_table.name,
                    "right_column": right_column,
                    "matched_identifier_count": overlap_count,
                }
            if joined_frame.empty:
                continue

            for column_name, _, column_overlap in boolean_columns:
                if column_name not in joined_frame.columns:
                    continue
                indicator = _boolean_indicator_series(joined_frame[column_name])
                if indicator is None:
                    continue
                valid_values = indicator.dropna()
                denominator = int(len(valid_values))
                if denominator == 0:
                    continue
                target_value = 0 if negated_have else 1
                numerator = int((valid_values.astype(int) == target_value).sum())
                percentage = numerator * 100.0 / denominator
                output = pd.DataFrame({"percentage": [percentage]})
                output.attrs.update(frame.attrs)
                evidence = {
                    "filter_evidence": filter_evidence,
                    "relation_evidence": relation_evidence,
                    "indicator_table": indicator_table.name,
                    "indicator_column": column_name,
                    "target_value": target_value,
                    "numerator": numerator,
                    "denominator": denominator,
                    "denominator_nonzero": True,
                    "operation": "percentage",
                }
                score = int(filter_evidence["matched_rows"]) + denominator + (10 * column_overlap)
                if best is None or score > best[0]:
                    best = (score, output, evidence)

    if best is None:
        return frame, []
    _, output, evidence = best
    transform = {
        "kind": "salvaged_context_boolean_percentage_repair",
        **evidence,
    }
    return output, [transform]


def _quoted_phrases(text: str) -> tuple[str, ...]:
    phrases = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
    return tuple(match[0] or match[1] for match in phrases if match[0] or match[1])


def _display_columns_for_entity_lookup(frame: pd.DataFrame) -> list[str]:
    display_columns = _display_like_columns(frame)
    return display_columns or [str(column) for column in frame.columns if frame[column].dtype == object][:3]


def _ids_for_quoted_entity(phrase: str, context_tables: list[ContextTable]) -> list[str]:
    phrase_lower = phrase.strip().lower()
    ids: list[str] = []
    for table in context_tables:
        id_columns = _identifier_columns(table.frame)
        if not id_columns:
            continue
        for display_column in _display_columns_for_entity_lookup(table.frame):
            if display_column not in table.frame.columns:
                continue
            mask = table.frame[display_column].map(lambda value: _text_value(value).lower() == phrase_lower)
            if not mask.any():
                continue
            for id_column in id_columns:
                ids.extend(
                    _text_value(value)
                    for value in table.frame.loc[mask, id_column].tolist()
                    if _text_value(value)
                )
    return list(dict.fromkeys(ids))


def _numeric_amount_from_text(text: str) -> float | None:
    patterns = (
        r"\b(?:amount|total|allocated|budgeted|funded|provisionally budgeted|final budget)\b[^\d.\-]{0,40}(-?\d+(?:\.\d+)?)",
        r"\b(-?\d+(?:\.\d+)?)\b[^\n.;]{0,40}\b(?:amount|total|allocated|budgeted|funded)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            return float(match.group(1))
        except ValueError:
            continue
    return None


def _record_numeric_amount(record_id: str, reasoning_frame: pd.DataFrame) -> tuple[float | None, dict[str, Any]]:
    rows = reasoning_frame[reasoning_frame["record_id"].map(_text_value).eq(record_id)]
    evidence: dict[str, Any] = {"record_id": record_id, "row_count": int(len(rows))}
    if rows.empty:
        return None, evidence
    for column in ("amount", "total", "value", "cost", "price"):
        if column not in rows.columns:
            continue
        numeric = pd.to_numeric(rows[column], errors="coerce").dropna()
        positive = numeric[numeric > 0]
        if not positive.empty:
            value = float(positive.iloc[0])
            evidence.update({"amount_column": column, "amount_value": value})
            return value, evidence
    for _, row in rows.iterrows():
        text_value = _text_value(row.get("evidence_span"))
        value = _numeric_amount_from_text(text_value)
        if value is not None and value > 0:
            evidence.update({"amount_column": "evidence_span", "amount_value": value})
            return value, evidence
    return None, evidence


def _record_text_bundle(record_id: str, reasoning_frame: pd.DataFrame) -> str:
    if "record_id" not in reasoning_frame.columns:
        return ""
    rows = reasoning_frame[reasoning_frame["record_id"].astype(str) == str(record_id)]
    parts: list[str] = []
    for _, row in rows.iterrows():
        for column in ("name", "category", "status", "evidence_span"):
            value = _text_value(row.get(column))
            if value:
                parts.append(value)
    return "\n".join(parts)


def _record_question_overlap_score(
    record_id: str,
    reasoning_frame: pd.DataFrame,
    question_tokens: set[str],
) -> int:
    if not question_tokens:
        return 0
    if "record_id" not in reasoning_frame.columns:
        return 0
    rows = reasoning_frame[reasoning_frame["record_id"].astype(str) == str(record_id)]
    score = 0
    for _, row in rows.iterrows():
        for column, weight in (
            ("category", 6),
            ("status", 4),
            ("name", 3),
            ("evidence_span", 1),
        ):
            value = _text_value(row.get(column))
            if not value:
                continue
            score += weight * len(_normalized_question_tokens(value) & question_tokens)
    return score


def _record_column_overlap_score(
    record_id: str,
    reasoning_frame: pd.DataFrame,
    column_name: str,
    question_tokens: set[str],
) -> int:
    if not question_tokens or "record_id" not in reasoning_frame.columns or column_name not in reasoning_frame.columns:
        return 0
    rows = reasoning_frame[reasoning_frame["record_id"].astype(str) == str(record_id)]
    score = 0
    for _, row in rows.iterrows():
        score += len(_normalized_question_tokens(_text_value(row.get(column_name))) & question_tokens)
    return score


def _record_classification_overlap_score(
    record_id: str,
    reasoning_frame: pd.DataFrame,
    question_tokens: set[str],
) -> int:
    if not question_tokens or "record_id" not in reasoning_frame.columns:
        return 0
    rows = reasoning_frame[reasoning_frame["record_id"].astype(str) == str(record_id)]
    score = 0
    cue_pattern = re.compile(
        r"\b(?:categor(?:y|ized|ised)|classified|classification|designated|designation|allocated|assigned|corrected|reclassified)\b"
        r"[^.]{0,80}",
        flags=re.IGNORECASE,
    )
    for _, row in rows.iterrows():
        text = _text_value(row.get("evidence_span"))
        for match in cue_pattern.finditer(text):
            score += len(_normalized_question_tokens(match.group(0)) & question_tokens)
    return score


def _record_ids_related_to_entity(entity_id: str, reasoning_frame: pd.DataFrame, question_tokens: set[str]) -> list[str]:
    candidates: list[tuple[int, str]] = []
    noisy_ids = {"record", "records", "recent", "recently", "reclassified", "recurring", "reconciliation"}
    for _, row in reasoning_frame.iterrows():
        text = _text_value(row.get("evidence_span"))
        if entity_id not in text:
            continue
        row_tokens = _normalized_question_tokens(text)
        overlap = row_tokens & question_tokens
        record_ids = [
            match
            for match in re.findall(r"\brec[A-Za-z0-9]+\b", text)
            if match != entity_id and match.lower() not in noisy_ids
        ]
        for record_id in record_ids:
            candidates.append((len(overlap), record_id))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return list(dict.fromkeys(record_id for _, record_id in candidates))


def apply_quoted_entity_ratio_repair(
    task: PublicTask,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    lowered_question = task.question.lower()
    if not re.search(r"\b(?:how many times|ratio|compared to|more than)\b", lowered_question):
        return frame, []
    if len(frame) != 0 and frame.attrs.get("databao_submit_called") is not False:
        return frame, []
    phrases = _quoted_phrases(task.question)
    if len(phrases) < 2:
        return frame, []

    context_tables = load_context_tables(task.context_dir, heuristic_level=_heuristic_level())
    reasoning_tables = document_records_for_reasoning(task.context_dir, heuristic_level=_heuristic_level())
    if not reasoning_tables:
        return frame, []
    reasoning_frame = pd.concat([table.frame for table in reasoning_tables], ignore_index=True)
    if "record_id" not in reasoning_frame.columns or "evidence_span" not in reasoning_frame.columns:
        return frame, []

    question_without_quotes = re.sub(r'"[^"]+"|\'[^\']+\'', " ", task.question)
    question_tokens = _question_filter_tokens(question_without_quotes)
    resolved: list[dict[str, Any]] = []
    for phrase in phrases[:2]:
        entity_ids = _ids_for_quoted_entity(phrase, context_tables)
        if not entity_ids:
            return frame, []
        best: tuple[float, str, dict[str, Any]] | None = None
        best_evidence_score: tuple[int, int, float] | None = None
        for entity_id in entity_ids:
            for record_id in _record_ids_related_to_entity(entity_id, reasoning_frame, question_tokens):
                amount, amount_evidence = _record_numeric_amount(record_id, reasoning_frame)
                if amount is None:
                    continue
                filter_score = _record_question_overlap_score(record_id, reasoning_frame, question_tokens)
                if filter_score <= 0 and question_tokens:
                    continue
                category_score = _record_column_overlap_score(record_id, reasoning_frame, "category", question_tokens)
                classification_score = _record_classification_overlap_score(record_id, reasoning_frame, question_tokens)
                evidence_score = (category_score + classification_score, filter_score, amount)
                if best_evidence_score is None or evidence_score > best_evidence_score:
                    best = (amount, record_id, amount_evidence | {"entity_id": entity_id})
                    best_evidence_score = evidence_score
        if best is None:
            return frame, []
        amount, record_id, evidence = best
        resolved.append(
            {
                "phrase": phrase,
                "entity_ids": entity_ids,
                "record_id": record_id,
                "amount": amount,
                "evidence": evidence
                | {
                    "record_category_score": best_evidence_score[0] if best_evidence_score else 0,
                    "record_filter_score": best_evidence_score[1] if best_evidence_score else 0,
                },
            }
        )

    denominator = float(resolved[1]["amount"])
    numerator = float(resolved[0]["amount"])
    if denominator == 0:
        return frame, []
    ratio = numerator / denominator
    output = pd.DataFrame({"ratio": [ratio]})
    output.attrs.update(frame.attrs)
    transform = {
        "kind": "quoted_entity_ratio_repair",
        "operation": "ratio",
        "numerator_phrase": resolved[0]["phrase"],
        "denominator_phrase": resolved[1]["phrase"],
        "numerator_record_id": resolved[0]["record_id"],
        "denominator_record_id": resolved[1]["record_id"],
        "numerator": numerator,
        "denominator": denominator,
        "denominator_nonzero": True,
        "question_tokens": sorted(question_tokens),
        "evidence": resolved,
    }
    return output, [transform]


def postprocess_answer_table(
    task: PublicTask,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, DeterministicPostprocessReport]:
    input_row_count = len(frame)
    input_column_count = len(frame.columns)
    current_frame = frame.copy()
    current_frame.attrs["context_dir"] = task.context_dir
    transformations: list[dict[str, Any]] = []

    current_frame, resolver_transforms = resolve_identifier_columns(task, current_frame)
    transformations.extend(resolver_transforms)

    current_frame, question_column_transforms = apply_answer_column_verifier(task, current_frame)
    transformations.extend(question_column_transforms)

    return current_frame, DeterministicPostprocessReport(
        applied=bool(transformations),
        transformations=transformations,
        failure_reason=None,
        input_row_count=input_row_count,
        input_column_count=input_column_count,
        output_row_count=len(current_frame),
        output_column_count=len(current_frame.columns),
    )


def generate_verifier_candidate_frames(
    task: PublicTask,
    frame: pd.DataFrame,
) -> list[tuple[pd.DataFrame, list[dict[str, Any]], float]]:
    candidates: list[tuple[pd.DataFrame, list[dict[str, Any]], float]] = []

    context_frame, context_transforms = apply_context_superlative_verifier(task)
    if context_frame is not None and context_transforms:
        candidates.append((context_frame, context_transforms, 0.42))

    superlative_frame, superlative_transforms = apply_superlative_verifier(task, frame)
    if superlative_transforms and not superlative_frame.equals(frame):
        candidates.append((superlative_frame, superlative_transforms, 0.48))

    aggregate_frame, aggregate_transforms = apply_aggregate_ratio_verifier(task, frame)
    if aggregate_transforms and not aggregate_frame.equals(frame):
        candidates.append((aggregate_frame, aggregate_transforms, 0.46))

    ratio_frame, ratio_transforms = apply_ratio_scale_compactor(task, frame)
    if ratio_transforms and not ratio_frame.equals(frame):
        candidates.append((ratio_frame, ratio_transforms, 0.66))

    salvaged_percentage_frame, salvaged_percentage_transforms = apply_salvaged_boolean_percentage_compactor(
        task,
        frame,
    )
    if salvaged_percentage_transforms and not salvaged_percentage_frame.equals(frame):
        candidates.append((salvaged_percentage_frame, salvaged_percentage_transforms, 0.74))

    context_percentage_frame, context_percentage_transforms = apply_salvaged_context_boolean_percentage_repair(
        task,
        frame,
    )
    if context_percentage_transforms and not context_percentage_frame.equals(frame):
        candidates.append((context_percentage_frame, context_percentage_transforms, 0.68))

    detail_count_frame, detail_count_transforms = apply_salvaged_detail_count_compactor(task, frame)
    if detail_count_transforms and not detail_count_frame.equals(frame):
        candidates.append((detail_count_frame, detail_count_transforms, 0.7))

    quoted_ratio_frame, quoted_ratio_transforms = apply_quoted_entity_ratio_repair(task, frame)
    if quoted_ratio_transforms and not quoted_ratio_frame.equals(frame):
        candidates.append((quoted_ratio_frame, quoted_ratio_transforms, 0.66))

    return candidates


def _non_identifier_numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in _numeric_columns(frame)
        if not _is_id_like_column(column)
    ]


def _display_candidate_columns(frame: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in frame.columns
        if _is_display_like_column(str(column)) and not _is_id_like_column(str(column))
    ]
def _duration_seconds(value: Any) -> float | None:
    text = _text_value(value)
    if not text:
        return None
    if text.startswith("+"):
        text = text[1:]
    try:
        return float(text)
    except ValueError:
        pass
    parts = text.split(":")
    if not 2 <= len(parts) <= 3:
        return None
    try:
        numeric = [float(part) for part in parts]
    except ValueError:
        return None
    if len(numeric) == 2:
        minutes, seconds = numeric
        return minutes * 60 + seconds
    hours, minutes, seconds = numeric
    return hours * 3600 + minutes * 60 + seconds
def _table_lookup(context_tables: list[ContextTable]) -> dict[str, ContextTable]:
    lookup: dict[str, ContextTable] = {}
    for table in context_tables:
        candidates = {
            table.name,
            table.name.lower(),
            Path(table.path.split("::", 1)[0]).stem,
            Path(table.path.split("::", 1)[0]).stem.lower(),
        }
        for candidate in candidates:
            if candidate and candidate not in lookup:
                lookup[candidate] = table
    return lookup


def _resolve_table(name: Any, table_lookup: Mapping[str, ContextTable]) -> ContextTable:
    if not isinstance(name, str) or not name.strip():
        raise StructuredPlanError("table name must be a non-empty string.")
    table = table_lookup.get(name) or table_lookup.get(name.lower())
    if table is None:
        raise StructuredPlanError(f"Unknown table: {name}.")
    return table


def _resolve_column(frame: pd.DataFrame, column: Any) -> str:
    if not isinstance(column, str) or not column.strip():
        raise StructuredPlanError("column name must be a non-empty string.")
    if column in frame.columns:
        return column

    lowered = column.lower()
    lowered_matches = [str(existing) for existing in frame.columns if str(existing).lower() == lowered]
    if len(lowered_matches) == 1:
        return lowered_matches[0]

    suffix_matches = [
        str(existing)
        for existing in frame.columns
        if "." in str(existing) and str(existing).rsplit(".", 1)[-1].lower() == lowered
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    raise StructuredPlanError(f"Unknown column: {column}.")


def _resolve_alias(name: Any, aliases: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    if not isinstance(name, str) or not name.strip():
        raise StructuredPlanError("alias must be a non-empty string.")
    if name not in aliases:
        raise StructuredPlanError(f"Unknown alias: {name}.")
    return aliases[name]


def _step_alias(step: Mapping[str, Any], default: str) -> str:
    raw_alias = step.get("alias", default)
    if not isinstance(raw_alias, str) or not raw_alias.strip():
        raise StructuredPlanError("step alias must be a non-empty string.")
    return raw_alias


def _input_alias(step: Mapping[str, Any]) -> str:
    raw_alias = step.get("source", step.get("input"))
    if not isinstance(raw_alias, str) or not raw_alias.strip():
        raise StructuredPlanError("step source/input alias must be a non-empty string.")
    return raw_alias


def _comparison_mask(series: pd.Series, op: str, value: Any) -> pd.Series:
    if op in {"==", "eq", "equals"}:
        numeric_series = pd.to_numeric(series, errors="coerce")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            numeric_value = None
        if numeric_value is not None and numeric_series.notna().any():
            return numeric_series.eq(numeric_value)
        return series.map(_text_value).eq(_text_value(value))

    if op in {"!=", "ne", "not_equals"}:
        return ~_comparison_mask(series, "==", value)

    if op in {"<", "<=", ">", ">="}:
        numeric_series = pd.to_numeric(series, errors="coerce")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise StructuredPlanError(f"Operator {op} requires a numeric value.") from exc
        if op == "<":
            return numeric_series.lt(numeric_value)
        if op == "<=":
            return numeric_series.le(numeric_value)
        if op == ">":
            return numeric_series.gt(numeric_value)
        return numeric_series.ge(numeric_value)

    if op == "in":
        if not isinstance(value, list | tuple | set):
            raise StructuredPlanError("in operator requires a list value.")
        normalized_values = {_text_value(item) for item in value}
        return series.map(_text_value).isin(normalized_values)

    if op == "contains":
        return series.map(_text_value).str.contains(re.escape(_text_value(value)), case=False, na=False)

    if op == "notnull":
        return series.notna() & series.map(_text_value).ne("")

    if op == "isnull":
        return series.isna() | series.map(_text_value).eq("")

    raise StructuredPlanError(f"Unsupported filter operator: {op}.")


def _apply_filter_step(step: Mapping[str, Any], aliases: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frame = _resolve_alias(_input_alias(step), aliases)
    conditions = step.get("conditions", step.get("condition"))
    if isinstance(conditions, Mapping):
        conditions = [conditions]
    if not isinstance(conditions, list) or not conditions:
        raise StructuredPlanError("filter step requires non-empty conditions.")

    mask = pd.Series(True, index=frame.index)
    for condition in conditions:
        if not isinstance(condition, Mapping):
            raise StructuredPlanError("filter condition must be an object.")
        column = _resolve_column(frame, condition.get("column"))
        op = str(condition.get("op", "==")).strip().lower()
        mask &= _comparison_mask(frame[column], op, condition.get("value"))
    return frame.loc[mask].reset_index(drop=True)


def _apply_join_step(
    step: Mapping[str, Any],
    aliases: Mapping[str, pd.DataFrame],
    tables: Mapping[str, ContextTable],
) -> pd.DataFrame:
    left_alias = step.get("left", step.get("source", step.get("input")))
    left = _resolve_alias(left_alias, aliases)
    right_table = _resolve_table(step.get("right_table", step.get("table")), tables)
    right = right_table.frame.copy()
    left_on = _resolve_column(left, step.get("left_on", step.get("on")))
    right_on = _resolve_column(right, step.get("right_on", step.get("on")))
    how = str(step.get("how", "inner")).lower()
    if how not in {"inner", "left"}:
        raise StructuredPlanError("join how must be inner or left.")

    left_for_join = left.copy()
    overlap = {
        str(column)
        for column in left_for_join.columns
        if column != left_on and column in right.columns and column != right_on
    }
    rename_left = {column: f"{left_alias}.{column}" for column in overlap}
    if rename_left:
        left_for_join = left_for_join.rename(columns=rename_left)

    return pd.merge(
        left_for_join,
        right,
        how=how,
        left_on=left_on if left_on not in rename_left else rename_left[left_on],
        right_on=right_on,
    ).reset_index(drop=True)


def _apply_select_step(step: Mapping[str, Any], aliases: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frame = _resolve_alias(_input_alias(step), aliases)
    raw_columns = step.get("columns")
    if not isinstance(raw_columns, list) or not raw_columns:
        raise StructuredPlanError("select step requires non-empty columns.")

    output = pd.DataFrame()
    for item in raw_columns:
        if isinstance(item, Mapping):
            source_column = _resolve_column(frame, item.get("column"))
            output_name = str(item.get("as") or source_column).rsplit(".", 1)[-1]
        else:
            source_column = _resolve_column(frame, item)
            output_name = str(source_column).rsplit(".", 1)[-1]
        output[output_name] = frame[source_column].reset_index(drop=True)
    return output


def _apply_distinct_step(step: Mapping[str, Any], aliases: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frame = _resolve_alias(_input_alias(step), aliases)
    raw_columns = step.get("columns")
    if raw_columns is None:
        return frame.drop_duplicates().reset_index(drop=True)
    if not isinstance(raw_columns, list) or not raw_columns:
        raise StructuredPlanError("distinct columns must be a non-empty list when provided.")
    columns = [_resolve_column(frame, column) for column in raw_columns]
    return frame.drop_duplicates(subset=columns).reset_index(drop=True)


def _apply_sort_step(step: Mapping[str, Any], aliases: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frame = _resolve_alias(_input_alias(step), aliases)
    raw_by = step.get("by")
    if isinstance(raw_by, str):
        raw_by = [raw_by]
    if not isinstance(raw_by, list) or not raw_by:
        raise StructuredPlanError("sort step requires by columns.")
    by = [_resolve_column(frame, column) for column in raw_by]
    ascending = step.get("ascending", True)
    return frame.sort_values(by=by, ascending=ascending).reset_index(drop=True)


def _apply_limit_step(step: Mapping[str, Any], aliases: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frame = _resolve_alias(_input_alias(step), aliases)
    try:
        n = int(step.get("n", step.get("limit", 1)))
    except (TypeError, ValueError) as exc:
        raise StructuredPlanError("limit n must be an integer.") from exc
    if n < 0:
        raise StructuredPlanError("limit n must be non-negative.")
    return frame.head(n).reset_index(drop=True)


def _apply_top_k_step(step: Mapping[str, Any], aliases: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frame = _resolve_alias(_input_alias(step), aliases)
    by = _resolve_column(frame, step.get("by", step.get("column")))
    try:
        k = int(step.get("k", step.get("n", 1)))
    except (TypeError, ValueError) as exc:
        raise StructuredPlanError("top_k k must be an integer.") from exc
    if k < 0:
        raise StructuredPlanError("top_k k must be non-negative.")
    ascending = bool(step.get("ascending", False))
    sorted_frame = frame.copy()
    numeric_values = pd.to_numeric(sorted_frame[by], errors="coerce")
    if numeric_values.notna().any():
        sorted_frame = sorted_frame.assign(**{f"__sort_{by}": numeric_values})
        sorted_frame = sorted_frame.sort_values(by=f"__sort_{by}", ascending=ascending)
        sorted_frame = sorted_frame.drop(columns=[f"__sort_{by}"])
    else:
        sorted_frame = sorted_frame.sort_values(by=by, ascending=ascending)
    return sorted_frame.head(k).reset_index(drop=True)


def _apply_date_filter_step(step: Mapping[str, Any], aliases: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frame = _resolve_alias(_input_alias(step), aliases)
    column = _resolve_column(frame, step.get("column", "Date"))
    values = frame[column]
    if pd.api.types.is_numeric_dtype(values):
        date_text = values.astype("Int64").astype(str)
    else:
        date_text = values.map(_text_value)

    mask = pd.Series(True, index=frame.index)
    year = step.get("year")
    month = step.get("month")
    if year is not None:
        year_text = str(int(year))
        mask &= date_text.str.startswith(year_text, na=False)
    if month is not None:
        month_value = int(month)
        compact_month = f"{month_value:02d}"
        parsed = pd.to_datetime(date_text, errors="coerce")
        parsed_mask = parsed.dt.month.eq(month_value)
        compact_mask = date_text.str.match(r"^\d{6}$", na=False) & date_text.str[4:6].eq(compact_month)
        mask &= parsed_mask.fillna(False) | compact_mask.fillna(False)

    start = step.get("start")
    end = step.get("end")
    if start is not None or end is not None:
        parsed = pd.to_datetime(date_text, errors="coerce")
        if start is not None:
            mask &= parsed.ge(pd.to_datetime(start, errors="raise")).fillna(False)
        if end is not None:
            mask &= parsed.le(pd.to_datetime(end, errors="raise")).fillna(False)

    return frame.loc[mask].reset_index(drop=True)


def _safe_numeric_operand(frame: pd.DataFrame, operand: Any) -> pd.Series | float:
    if isinstance(operand, int | float):
        return float(operand)
    if isinstance(operand, str):
        column = _resolve_column(frame, operand)
        return pd.to_numeric(frame[column], errors="coerce")
    if isinstance(operand, Mapping):
        if "value" in operand:
            value = operand["value"]
            if not isinstance(value, int | float):
                raise StructuredPlanError("literal expression value must be numeric.")
            return float(value)
        if "column" in operand:
            column = _resolve_column(frame, operand["column"])
            return pd.to_numeric(frame[column], errors="coerce")
    raise StructuredPlanError("numeric expression operands must be columns or numeric literals.")


def _safe_binary_numeric_expression(frame: pd.DataFrame, expression: Mapping[str, Any]) -> pd.Series:
    op = str(expression.get("op", "")).lower()
    if op not in {"add", "subtract", "multiply", "divide", "ratio", "percentage"}:
        raise StructuredPlanError(f"Unsupported derive expression op: {op}.")
    left = _safe_numeric_operand(frame, expression.get("left", expression.get("numerator")))
    right = _safe_numeric_operand(frame, expression.get("right", expression.get("denominator")))
    if op == "add":
        result = left + right
    elif op == "subtract":
        result = left - right
    elif op == "multiply":
        result = left * right
    else:
        result = left / right
        if op == "percentage":
            result = result * 100
    if isinstance(result, pd.Series):
        return result
    return pd.Series([result] * len(frame), index=frame.index)


def _apply_derive_step(step: Mapping[str, Any], aliases: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frame = _resolve_alias(_input_alias(step), aliases).copy()
    expressions = step.get("columns", step.get("expressions"))
    if isinstance(expressions, Mapping):
        expressions = [expressions]
    if not isinstance(expressions, list) or not expressions:
        raise StructuredPlanError("derive step requires expression columns.")
    for expression in expressions:
        if not isinstance(expression, Mapping):
            raise StructuredPlanError("derive expression must be an object.")
        output_name = str(expression.get("as") or expression.get("name") or "").strip()
        if not output_name:
            raise StructuredPlanError("derive expression requires an output name.")
        frame[output_name] = _safe_binary_numeric_expression(frame, expression)
    return frame.reset_index(drop=True)


def _apply_ratio_or_percentage_step(
    step: Mapping[str, Any],
    aliases: Mapping[str, pd.DataFrame],
    *,
    percentage: bool,
) -> pd.DataFrame:
    frame = _resolve_alias(_input_alias(step), aliases)
    numerator = _safe_numeric_operand(frame, step.get("numerator"))
    denominator = _safe_numeric_operand(frame, step.get("denominator"))
    result = numerator / denominator
    if percentage:
        result = result * 100
    if isinstance(result, pd.Series):
        result = result.reset_index(drop=True)
    else:
        result = pd.Series([result])
    output_name = str(step.get("as") or ("percentage" if percentage else "ratio"))
    return pd.DataFrame({output_name: result})


def _apply_aggregate_step(step: Mapping[str, Any], aliases: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frame = _resolve_alias(_input_alias(step), aliases)
    raw_group_by = step.get("group_by", [])
    if isinstance(raw_group_by, str):
        raw_group_by = [raw_group_by]
    if not isinstance(raw_group_by, list):
        raise StructuredPlanError("aggregate group_by must be a list.")
    group_by = [_resolve_column(frame, column) for column in raw_group_by]

    raw_aggs = step.get("aggregations")
    if not isinstance(raw_aggs, list) or not raw_aggs:
        raise StructuredPlanError("aggregate step requires aggregations.")
    supported = {"count", "count_distinct", "sum", "mean", "average", "min", "max", "nunique"}

    if group_by:
        named_aggs: dict[str, tuple[str, str]] = {}
        for agg in raw_aggs:
            if not isinstance(agg, Mapping):
                raise StructuredPlanError("aggregation must be an object.")
            func = str(agg.get("function", "count")).lower()
            if func not in supported:
                raise StructuredPlanError(f"Unsupported aggregate function: {func}.")
            source_column = str(agg.get("column", group_by[0]))
            column = group_by[0] if source_column == "*" else _resolve_column(frame, source_column)
            output_name = str(agg.get("as") or f"{func}_{column}")
            pandas_func = "nunique" if func == "count_distinct" else "mean" if func == "average" else func
            named_aggs[output_name] = (column, pandas_func)
        return frame.groupby(group_by, dropna=False).agg(**named_aggs).reset_index()

    row: dict[str, Any] = {}
    for agg in raw_aggs:
        if not isinstance(agg, Mapping):
            raise StructuredPlanError("aggregation must be an object.")
        func = str(agg.get("function", "count")).lower()
        if func not in supported:
            raise StructuredPlanError(f"Unsupported aggregate function: {func}.")
        source_column = str(agg.get("column", "*"))
        output_name = str(agg.get("as") or f"{func}_{source_column}")
        if func == "count" and source_column == "*":
            row[output_name] = len(frame)
            continue
        series = frame[_resolve_column(frame, source_column)]
        pandas_func = "nunique" if func == "count_distinct" else "mean" if func == "average" else func
        row[output_name] = getattr(series, pandas_func)()
    return pd.DataFrame([row])


def execute_structured_plan(task: PublicTask, plan: Mapping[str, Any]) -> pd.DataFrame:
    if not isinstance(plan, Mapping):
        raise StructuredPlanError("Plan must be a JSON object.")
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        raise StructuredPlanError("Plan requires a non-empty steps list.")

    context_tables = load_context_tables(task.context_dir)
    tables = _table_lookup(context_tables)
    aliases: dict[str, pd.DataFrame] = {}
    last_alias = ""

    for index, raw_step in enumerate(steps):
        if not isinstance(raw_step, Mapping):
            raise StructuredPlanError(f"Step {index} must be an object.")
        op = str(raw_step.get("op", "")).strip().lower()
        alias = _step_alias(raw_step, f"step_{index}")

        if op == "source":
            table = _resolve_table(raw_step.get("table"), tables)
            aliases[alias] = table.frame.copy().reset_index(drop=True)
        elif op == "filter":
            aliases[alias] = _apply_filter_step(raw_step, aliases)
        elif op == "join":
            aliases[alias] = _apply_join_step(raw_step, aliases, tables)
        elif op == "select":
            aliases[alias] = _apply_select_step(raw_step, aliases)
        elif op == "distinct":
            aliases[alias] = _apply_distinct_step(raw_step, aliases)
        elif op == "sort":
            aliases[alias] = _apply_sort_step(raw_step, aliases)
        elif op == "limit":
            aliases[alias] = _apply_limit_step(raw_step, aliases)
        elif op == "top_k":
            aliases[alias] = _apply_top_k_step(raw_step, aliases)
        elif op == "date_filter":
            aliases[alias] = _apply_date_filter_step(raw_step, aliases)
        elif op == "derive":
            aliases[alias] = _apply_derive_step(raw_step, aliases)
        elif op == "ratio":
            aliases[alias] = _apply_ratio_or_percentage_step(raw_step, aliases, percentage=False)
        elif op == "percentage":
            aliases[alias] = _apply_ratio_or_percentage_step(raw_step, aliases, percentage=True)
        elif op == "aggregate":
            aliases[alias] = _apply_aggregate_step(raw_step, aliases)
        elif op == "count_distinct":
            aggregate_step = dict(raw_step)
            aggregate_step["aggregations"] = [
                {
                    "function": "count_distinct",
                    "column": raw_step.get("column"),
                    "as": raw_step.get("as", "count_distinct"),
                }
            ]
            aliases[alias] = _apply_aggregate_step(aggregate_step, aliases)
        else:
            raise StructuredPlanError(f"Unsupported plan op: {op}.")

        if not isinstance(aliases[alias], pd.DataFrame):
            raise StructuredPlanError(f"Step {index} did not produce a dataframe.")
        last_alias = alias

    output_alias = plan.get("output", last_alias)
    output = _resolve_alias(output_alias, aliases)
    return output.reset_index(drop=True)


def _context_document_text(context_dir: Path) -> str:
    max_chars = MAX_DESCRIPTION_CHARS
    pieces: list[str] = []
    knowledge_path = context_dir / "knowledge.md"
    if knowledge_path.exists():
        pieces.append(_description_from_file(knowledge_path, context_dir, max_chars=max_chars))
    doc_dir = context_dir / "doc"
    if doc_dir.is_dir():
        for path in sorted(item for item in doc_dir.rglob("*") if item.is_file()):
            pieces.append(_description_from_file(path, context_dir, max_chars=max_chars))
    text = "\n\n".join(pieces)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return text


def _table_has_columns(table: ContextTable, required_columns: tuple[str, ...]) -> bool:
    lowered = {str(column).lower() for column in table.frame.columns}
    return all(column.lower() in lowered for column in required_columns)


def _find_table_with_columns(
    context_tables: list[ContextTable],
    required_columns: tuple[str, ...],
    *,
    preferred_name: str | None = None,
) -> ContextTable | None:
    for table in context_tables:
        if preferred_name is not None and table.name.lower() != preferred_name.lower():
            continue
        if _table_has_columns(table, required_columns):
            return table
    for table in context_tables:
        if _table_has_columns(table, required_columns):
            return table
    return None


def build_task_context(
    task: PublicTask,
    *,
    context_summary: dict[str, Any] | None = None,
    heuristic_level: str | None = None,
) -> TaskContext:
    context_tables = load_context_tables(
        task.context_dir,
        heuristic_level=heuristic_level,
    )
    return TaskContext(
        task=task,
        context_tables=context_tables,
        document_text=_context_document_text(task.context_dir),
        schema_graph=build_schema_graph(context_tables),
        context_summary=context_summary,
    )


def _normalized_question_tokens(question: str) -> set[str]:
    tokens = {token for token in re.findall(r"[a-z0-9]+", question.lower()) if len(token) > 1}
    singulars = {token[:-1] for token in tokens if len(token) > 3 and token.endswith("s")}
    return tokens | singulars


def _column_question_aliases(column_name: str) -> set[str]:
    split_name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", column_name)
    lowered = split_name.lower()
    parts = {part for part in re.split(r"[^a-z0-9]+", lowered) if part}
    aliases = {lowered, re.sub(r"[^a-z0-9]+", "", lowered), *parts}
    if lowered.endswith("_id") or lowered.endswith(" id"):
        aliases.update({"id", "identifier", "reference", "code"})
    if "date" in parts or lowered.endswith("date"):
        aliases.add("date")
    if "time" in parts or lowered.endswith("time"):
        aliases.add("time")
    if "name" in parts or lowered.endswith("name"):
        aliases.add("name")
    if "url" in parts or "link" in parts or "website" in parts:
        aliases.update({"url", "link", "website"})
    if "count" in parts or lowered.endswith("count"):
        aliases.add("count")
    if lowered in {"text", "body", "content"} or parts & {"text", "body", "content"}:
        aliases.add("comment")
    return {alias for alias in aliases if alias}


def _question_referenced_columns(question: str, context_tables: list[ContextTable]) -> tuple[str, ...]:
    tokens = _normalized_question_tokens(question)
    compact_question = re.sub(r"[^a-z0-9]+", "", question.lower())
    matches: list[str] = []
    for table in context_tables:
        for column in table.frame.columns:
            column_name = str(column)
            aliases = _column_question_aliases(column_name)
            if (
                _is_id_like_column(column_name)
                and not _question_asks_for_identifier(question)
                and column_name.lower() not in question
            ):
                continue
            if column_name.lower() in question or aliases & tokens or any(alias in compact_question for alias in aliases if len(alias) > 3):
                matches.append(column_name)
    return tuple(dict.fromkeys(matches))


def _infer_generic_target_entity(question: str) -> str | None:
    match = re.search(r"\b(?:which|what|list|show|return)\s+([a-z][a-z0-9_ -]{1,40}?)(?:\s+(?:with|where|whose|that|when|for|from|has|have)\b|\?|$)", question)
    if not match:
        return None
    entity = re.sub(r"\s+", " ", match.group(1).strip().lower())
    return entity or None


STRONG_AGGREGATION_TERMS = ("average", "avg", "mean", "sum", "total", "count", "how many")
RATIO_TERMS = ("ratio", "percentage", "percent", "proportion", "share", "how many times", "compared to")
VALUE_METRIC_TERMS = ("amount", "cost", "price", "score", "points")
WEAK_METRIC_TERMS = ("number", "rate", "value", "view count")


def _schema_columns_matching_phrase(phrase: str, context_tables: list[ContextTable] | None) -> tuple[str, ...]:
    if not context_tables:
        return ()
    compact_phrase = re.sub(r"[^a-z0-9]+", "", phrase.lower())
    phrase_tokens = _normalized_question_tokens(phrase)
    matches: list[str] = []
    for table in context_tables:
        for column in table.frame.columns:
            column_name = str(column)
            aliases = _column_question_aliases(column_name)
            compact_aliases = {re.sub(r"[^a-z0-9]+", "", alias) for alias in aliases}
            if compact_phrase in compact_aliases or aliases & phrase_tokens:
                matches.append(column_name)
    return tuple(dict.fromkeys(matches))


def _add_question_evidence(
    *,
    evidence: list[dict[str, Any]],
    matched_phrases: list[str],
    strong_terms: list[str],
    weak_terms: list[str],
    phrase: str,
    kind: str,
    strength: str,
    confidence_delta: float,
    schema_columns: tuple[str, ...] = (),
) -> float:
    matched_phrases.append(phrase)
    if strength == "weak":
        weak_terms.append(phrase)
    else:
        strong_terms.append(phrase)
    evidence.append(
        {
            "kind": kind,
            "phrase": phrase,
            "strength": strength,
            "confidence_delta": confidence_delta,
            "schema_columns": list(schema_columns),
        }
    )
    return confidence_delta


def _question_looks_single_answer(question: str) -> bool:
    lowered = question.lower()
    if _superlative_direction(lowered) is not None:
        return True
    return bool(
        re.search(
            r"\b(?:single|only|exactly one|one row|one record|top\s+1|top\s+one|first row|last row)\b",
            lowered,
        )
    )


def extract_question_features(
    question: str,
    context_tables: list[ContextTable] | None = None,
) -> QuestionFeatures:
    lowered = question.lower()
    evidence: list[dict[str, Any]] = []
    matched_phrases: list[str] = []
    strong_terms: list[str] = []
    weak_terms: list[str] = []
    confidence = 0.0
    asks_aggregation = False
    asks_ratio_or_percentage = False
    asks_entity_or_list = False
    asks_multi_attribute = False
    asks_superlative = _superlative_direction(lowered) is not None

    if asks_superlative:
        confidence += _add_question_evidence(
            evidence=evidence,
            matched_phrases=matched_phrases,
            strong_terms=strong_terms,
            weak_terms=weak_terms,
            phrase="superlative",
            kind="superlative",
            strength="strong",
            confidence_delta=0.25,
        )

    for phrase in STRONG_AGGREGATION_TERMS:
        pattern = r"\bhow many\b" if phrase == "how many" else rf"\b{re.escape(phrase)}\b"
        if re.search(pattern, lowered):
            asks_aggregation = True
            confidence += _add_question_evidence(
                evidence=evidence,
                matched_phrases=matched_phrases,
                strong_terms=strong_terms,
                weak_terms=weak_terms,
                phrase=phrase,
                kind="aggregation",
                strength="strong",
                confidence_delta=0.25,
                schema_columns=_schema_columns_matching_phrase(phrase, context_tables),
            )

    for phrase in RATIO_TERMS:
        if re.search(rf"\b{re.escape(phrase)}\b", lowered):
            asks_ratio_or_percentage = True
            confidence += _add_question_evidence(
                evidence=evidence,
                matched_phrases=matched_phrases,
                strong_terms=strong_terms,
                weak_terms=weak_terms,
                phrase=phrase,
                kind="ratio_or_percentage",
                strength="strong",
                confidence_delta=0.25,
                schema_columns=_schema_columns_matching_phrase(phrase, context_tables),
            )

    for phrase in VALUE_METRIC_TERMS:
        if re.search(rf"\b{re.escape(phrase)}\b", lowered):
            confidence += _add_question_evidence(
                evidence=evidence,
                matched_phrases=matched_phrases,
                strong_terms=strong_terms,
                weak_terms=weak_terms,
                phrase=phrase,
                kind="metric_value",
                strength="medium",
                confidence_delta=0.12,
                schema_columns=_schema_columns_matching_phrase(phrase, context_tables),
            )

    for phrase in WEAK_METRIC_TERMS:
        if re.search(rf"\b{re.escape(phrase)}\b", lowered):
            schema_columns = _schema_columns_matching_phrase(phrase, context_tables)
            confidence += _add_question_evidence(
                evidence=evidence,
                matched_phrases=matched_phrases,
                strong_terms=strong_terms,
                weak_terms=weak_terms,
                phrase=phrase,
                kind="weak_metric",
                strength="weak",
                confidence_delta=0.08 if schema_columns else 0.02,
                schema_columns=schema_columns,
            )

    entity_patterns = (
        r"\b(?:list|show|return|all|each|every|rows|records|entries|matching|matched)\b",
        r"\b(?:what|which|who)\s+are\b",
        r"\b(?:what|which|who)\s+(?:[a-z0-9_ -]+s|rows|records|entries)\b",
    )
    if any(re.search(pattern, lowered) for pattern in entity_patterns):
        asks_entity_or_list = True
        confidence += _add_question_evidence(
            evidence=evidence,
            matched_phrases=matched_phrases,
            strong_terms=strong_terms,
            weak_terms=weak_terms,
            phrase="entity_or_list",
            kind="answer_shape",
            strength="medium",
            confidence_delta=0.12,
        )

    if re.search(r"\b(?:and|plus)\b", lowered) or re.search(r"\bwith\s+(?:their|its)\b", lowered):
        asks_multi_attribute = True
        confidence += _add_question_evidence(
            evidence=evidence,
            matched_phrases=matched_phrases,
            strong_terms=strong_terms,
            weak_terms=weak_terms,
            phrase="multi_attribute_connector",
            kind="answer_shape",
            strength="weak",
            confidence_delta=0.04,
        )

    has_strong_metric_signal = asks_aggregation or asks_ratio_or_percentage or any(
        entry.get("kind") == "metric_value" for entry in evidence
    )
    has_schema_backed_weak_metric = any(
        entry.get("kind") == "weak_metric" and entry.get("schema_columns")
        for entry in evidence
    )
    scalar_lead = bool(
        re.search(
            r"\bwhat\s+is\s+(?:the\s+)?(?:average|avg|mean|sum|total|count|ratio|percentage|percent)\b",
            lowered,
        )
        or re.search(r"\bhow many\b", lowered)
    )
    asks_entity_plus_metric = (
        asks_entity_or_list
        and (has_strong_metric_signal or has_schema_backed_weak_metric)
        and not scalar_lead
    )
    asks_scalar_metric = (
        ((asks_aggregation or asks_ratio_or_percentage) and not asks_entity_or_list)
        or (scalar_lead and not asks_multi_attribute)
    )
    if asks_superlative and not asks_entity_or_list:
        asks_scalar_metric = True

    return QuestionFeatures(
        asks_scalar_metric=asks_scalar_metric,
        asks_entity_or_list=asks_entity_or_list,
        asks_multi_attribute=asks_multi_attribute,
        asks_entity_plus_metric=asks_entity_plus_metric,
        asks_aggregation=asks_aggregation,
        asks_ratio_or_percentage=asks_ratio_or_percentage,
        asks_superlative=asks_superlative,
        matched_phrases=tuple(dict.fromkeys(matched_phrases)),
        strong_terms=tuple(dict.fromkeys(strong_terms)),
        weak_terms=tuple(dict.fromkeys(weak_terms)),
        confidence=round(min(confidence, 1.0), 3),
        evidence=tuple(evidence),
    )


def _question_mentions_attribute_pair(question: str) -> bool:
    return extract_question_features(question).asks_multi_attribute


def _question_mentions_metric_answer(question: str) -> bool:
    features = extract_question_features(question)
    return bool(features.strong_terms or features.weak_terms)


def _question_mentions_multiple_metrics(question: str) -> bool:
    features = extract_question_features(question)
    metric_terms = [
        entry["phrase"]
        for entry in features.evidence
        if entry.get("kind") in {"aggregation", "ratio_or_percentage", "metric_value", "weak_metric"}
        and (entry.get("strength") != "weak" or entry.get("schema_columns"))
    ]
    return len(set(metric_terms)) >= 2


def _question_mentions_entity_plus_metric(question: str) -> bool:
    return extract_question_features(question).asks_entity_plus_metric


def _display_like_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if _is_display_like_column(str(column)):
            columns.append(str(column))
    return columns


def _metric_like_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        lowered = str(column).lower()
        if lowered == "id" or lowered.endswith("_id") or lowered.startswith("link_to_"):
            continue
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            columns.append(str(column))
            continue
        sample = series.dropna().head(MAX_SCHEMA_PROFILE_SAMPLE_ROWS)
        if len(sample) and pd.to_numeric(sample, errors="coerce").notna().any():
            columns.append(str(column))
    return columns


def _retrieval_column_score(question: str, column_name: str) -> int:
    tokens = _normalized_question_tokens(question)
    compact_question = re.sub(r"[^a-z0-9]+", "", question.lower())
    aliases = _column_question_aliases(column_name)
    score = 0
    if column_name.lower() in question.lower():
        score += 60
    score += 12 * len(aliases & tokens)
    if any(alias in compact_question for alias in aliases if len(alias) > 3):
        score += 20
    if column_name.lower().endswith("_id") or column_name.lower().startswith("link_to_"):
        score += 5
    return score


def query_context_retriever(task_context: TaskContext, question: str) -> RetrievedContext:
    scored_tables: list[tuple[int, ContextTable, list[str], list[str]]] = []
    diagnostics: dict[str, Any] = {"table_scores": []}
    for table in task_context.context_tables:
        matched_columns: list[str] = []
        table_score = 0
        table_name_score = _retrieval_column_score(question, table.name)
        table_score += table_name_score
        for column in table.frame.columns:
            column_score = _retrieval_column_score(question, str(column))
            if column_score > 0:
                matched_columns.append(str(column))
                table_score += column_score
        display_columns = _display_like_columns(table.frame)
        metric_columns = _metric_like_columns(table.frame)
        table_score += 3 * len(display_columns[:2])
        table_score += 2 * len(metric_columns[:2])
        if table_score > 0:
            scored_tables.append((table_score, table, matched_columns, display_columns + metric_columns))
        diagnostics["table_scores"].append(
            {
                "table": table.name,
                "path": table.path,
                "score": table_score,
                "matched_columns": matched_columns,
            }
        )

    if not scored_tables:
        fallback_tables = sorted(task_context.context_tables, key=lambda item: len(item.frame))[:5]
        scored_tables = [
            (0, table, [], _display_like_columns(table.frame) + _metric_like_columns(table.frame))
            for table in fallback_tables
        ]
        diagnostics["fallback_reason"] = "No table/column lexical match; selected small schema slice."

    scored_tables.sort(key=lambda item: (-item[0], item[1].name))
    selected = scored_tables[:8]
    relevant_tables: list[str] = []
    relevant_columns: dict[str, tuple[str, ...]] = {}
    sample_rows: dict[str, list[dict[str, Any]]] = {}
    selected_names = {table.name for _, table, _, _ in selected}
    for _, table, matched_columns, supporting_columns in selected:
        relevant_tables.append(table.name)
        relevant_tables.append(table.path)
        columns = list(dict.fromkeys([*matched_columns, *supporting_columns, *_identifier_columns(table.frame)]))
        if not columns:
            columns = [str(column) for column in table.frame.columns[: min(5, len(table.frame.columns))]]
        relevant_columns[table.name] = tuple(columns[:12])
        sample_frame = table.frame.loc[:, [column for column in columns if column in table.frame.columns]].head(
            MAX_SCHEMA_SAMPLE_ROWS
        )
        sample_rows[table.name] = [
            {str(column): _json_safe_cell(value) for column, value in row.items()}
            for row in sample_frame.to_dict(orient="records")
        ]

    join_paths = [
        join
        for join in task_context.schema_graph.get("join_candidates", [])
        if join.get("left_table") in selected_names or join.get("right_table") in selected_names
    ][:12]

    document_snippets: list[dict[str, Any]] = []
    question_tokens = _normalized_question_tokens(question)
    for table in document_records_for_reasoning(task_context.task.context_dir):
        for _, row in table.frame.iterrows():
            text = " ".join(
                _text_value(row.get(column))
                for column in ("name", "title", "description", "status", "type", "category", "evidence_span")
                if column in table.frame.columns
            )
            overlap = question_tokens & _normalized_question_tokens(text)
            if overlap:
                document_snippets.append(
                    {
                        "source_doc": row.get("source_doc"),
                        "paragraph_index": _json_safe_cell(row.get("paragraph_index")),
                        "matched_terms": sorted(overlap),
                        "text": _text_value(row.get("evidence_span"))[:1000],
                    }
                )
    document_snippets = document_snippets[:6]
    diagnostics["selected_table_count"] = len(selected)
    diagnostics["document_snippet_count"] = len(document_snippets)
    return RetrievedContext(
        relevant_tables=tuple(dict.fromkeys(relevant_tables)),
        relevant_columns=relevant_columns,
        sample_rows=sample_rows,
        candidate_join_paths=tuple(join_paths),
        document_snippets=tuple(document_snippets),
        retrieval_diagnostics=diagnostics,
    )


def _question_mentions_filter_or_condition(question: str) -> bool:
    return bool(
        re.search(
            r"\b(?:where|with|whose|having|under|over|above|below|less than|greater than|between|"
            r"before|after|equals?|group|grouped|distinct|per|by|for|condition|match|matching|selected)\b",
            question.lower(),
        )
    )


def _question_mentions_multiple_attributes(question: str) -> bool:
    features = extract_question_features(question)
    return features.asks_multi_attribute or features.asks_entity_plus_metric


def _question_asks_count_metric(question: str) -> bool:
    features = extract_question_features(question)
    if features.asks_ratio_or_percentage and "how many times" in question.lower():
        return "count" in features.strong_terms
    return any(term in features.strong_terms for term in ("how many", "count"))


def _retrieval_confidence(retrieved_context: RetrievedContext | None) -> float:
    if retrieved_context is None:
        return 0.0
    table_scores = retrieved_context.retrieval_diagnostics.get("table_scores", [])
    scores = [
        float(item.get("score", 0) or 0)
        for item in table_scores
        if isinstance(item, Mapping)
    ]
    if not scores:
        return 0.0
    return min(max(scores) / 100.0, 1.0)


def _cheap_count_fallback_candidate(
    task_context: TaskContext,
    retrieved_context: RetrievedContext | None,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    question = task_context.task.question
    lowered_question = question.lower()
    features = extract_question_features(question, task_context.context_tables)
    strong_count_evidence = any(term in features.strong_terms for term in ("how many", "count"))
    if not (features.asks_aggregation or features.asks_scalar_metric) or not strong_count_evidence:
        return None
    if _question_mentions_filter_or_condition(question):
        return None
    plain_row_count_request = bool(
        re.search(r"\bhow many\s+(?:rows|records|entries)\b", lowered_question)
        or re.search(r"\bcount\s+(?:of\s+)?(?:rows|records|entries)\b", lowered_question)
    )

    table_lookup = task_context.table_lookup()
    selected_table: ContextTable | None = None
    reason = ""
    score = 0.0

    table_scores = (
        retrieved_context.retrieval_diagnostics.get("table_scores", [])
        if retrieved_context is not None
        else []
    )
    scored_tables: list[tuple[float, str]] = []
    for item in table_scores:
        if not isinstance(item, Mapping):
            continue
        table_name = _text_value(item.get("table"))
        if not table_name or table_name not in table_lookup:
            continue
        scored_tables.append((float(item.get("score", 0) or 0), table_name))

    scored_tables.sort(key=lambda item: item[0], reverse=True)
    if scored_tables:
        top_score, top_name = scored_tables[0]
        second_score = scored_tables[1][0] if len(scored_tables) > 1 else 0.0
        if top_score >= 20 and (
            second_score <= 0
            or top_score >= second_score + 5
            or top_score >= second_score * 1.5
        ):
            selected_table = table_lookup[top_name]
            score = top_score
            reason = "single_high_retrieval_score"

    if selected_table is None and len(task_context.context_tables) == 1:
        selected_table = task_context.context_tables[0]
        reason = "single_context_table"

    if selected_table is None:
        return None
    if not plain_row_count_request and reason != "single_context_table":
        return None

    confidence = 0.22 if reason == "single_context_table" else 0.26
    if plain_row_count_request:
        confidence += 0.04

    diagnostics = {
        "kind": "cheap_count_fallback",
        "operation": "count",
        "table": selected_table.name,
        "path": selected_table.path,
        "row_count": len(selected_table.frame),
        "retrieval_score": score,
        "reason": reason,
        "question_features": features.to_dict(),
        "executed_filter": False,
        "plain_row_count_request": plain_row_count_request,
        "confidence": round(confidence, 3),
    }
    return pd.DataFrame({"count": [len(selected_table.frame)]}), diagnostics


def _candidate_shape_mismatch(contract: AnswerContract, candidate: Candidate | None) -> bool | None:
    if candidate is None:
        return None
    row_count = len(candidate.frame)
    column_count = len(candidate.frame.columns)
    return bool(
        (contract.max_rows is not None and row_count > contract.max_rows)
        or (contract.max_columns is not None and column_count > contract.max_columns)
    )


def build_complexity_profile(
    task_context: TaskContext,
    retrieved_context: RetrievedContext | None,
    best_candidate: Candidate | None = None,
) -> TaskComplexityProfile:
    tables = task_context.context_tables
    source_kinds = [_source_kind_from_context_path(table.path) for table in tables]
    db_table_count = sum(1 for kind in source_kinds if kind == "sqlite")
    csv_table_count = sum(1 for kind in source_kinds if kind == "csv")
    json_table_count = sum(1 for kind in source_kinds if kind == "json")
    document_tables = [table for table in tables if _source_kind_from_context_path(table.path) == "document_table"]
    doc_dir = task_context.task.context_dir / "doc"
    document_file_count = len(list(doc_dir.rglob("*.md"))) if doc_dir.is_dir() else 0
    if (task_context.task.context_dir / "knowledge.md").exists():
        document_file_count += 1
    document_text_chars = len(task_context.document_text)
    total_columns = sum(len(table.frame.columns) for table in tables)
    total_rows_sampled = sum(min(len(table.frame), MAX_SCHEMA_SAMPLE_ROWS) for table in tables)
    question = task_context.task.question
    question_tokens = _normalized_question_tokens(question)
    has_unstructured_docs = document_file_count > 0 or bool(document_tables)
    has_sqlite_db = db_table_count > 0
    has_multiple_sources = len({kind for kind in source_kinds if kind != "unknown"}) > 1
    has_many_tables = len(tables) >= 4
    has_join_candidates = bool(task_context.schema_graph.get("join_candidates"))
    question_mentions_aggregation = bool(re.search(r"\b(?:count|sum|total|average|avg|mean|how many)\b", question.lower()))
    question_mentions_ratio_or_percentage = bool(re.search(r"\b(?:ratio|percentage|percent|proportion)\b", question.lower()))
    question_mentions_superlative = _superlative_direction(question) is not None
    question_mentions_filter = _question_mentions_filter_or_condition(question)
    question_mentions_multi = _question_mentions_multiple_attributes(question)
    estimated_context_size = document_text_chars + total_columns * 40 + total_rows_sampled * 200
    retrieval_confidence = _retrieval_confidence(retrieved_context)

    best_report = best_candidate.contract_report if best_candidate is not None else None
    best_contract_valid = (
        bool(best_report.get("valid"))
        if isinstance(best_report, Mapping)
        else None
    )
    best_empty = len(best_candidate.frame) == 0 if best_candidate is not None else None
    best_metadata = bool(_metadata_debug_columns(best_candidate.frame)) if best_candidate is not None else None
    contract = infer_answer_contract(task_context, infer_task_intent(task_context))
    best_shape_mismatch = _candidate_shape_mismatch(contract, best_candidate)
    best_score = int(best_candidate.confidence * 10) if best_candidate is not None else None

    complexity_score = 0
    if has_unstructured_docs:
        complexity_score += 20
    if document_text_chars > 20000:
        complexity_score += 15
    if has_multiple_sources:
        complexity_score += 15
    if has_join_candidates:
        complexity_score += 15
    if has_sqlite_db:
        complexity_score += 10
    if len(tables) >= 4:
        complexity_score += 10
    if question_mentions_aggregation:
        complexity_score += 10
    if question_mentions_ratio_or_percentage:
        complexity_score += 10
    if question_mentions_superlative:
        complexity_score += 10
    if question_mentions_filter:
        complexity_score += 10
    if question_mentions_multi:
        complexity_score += 10
    if retrieval_confidence < 0.25 and estimated_context_size > 20000:
        complexity_score += 10

    uncertainty_score = 0
    if best_candidate is None:
        uncertainty_score += 30
    elif best_empty is True and not contract.allow_empty:
        uncertainty_score += 25
    if best_contract_valid is False:
        uncertainty_score += 20
    if best_shape_mismatch:
        uncertainty_score += 15
    if best_metadata:
        uncertainty_score += 15
    if best_candidate is not None:
        if contract.max_columns is not None and len(best_candidate.frame.columns) < min(contract.max_columns, 1):
            uncertainty_score += 10
        if best_candidate.confidence < 0.5:
            uncertainty_score += 10

    return TaskComplexityProfile(
        source_count=len(set(source_kinds)),
        table_count=len(tables),
        db_table_count=db_table_count,
        csv_table_count=csv_table_count,
        json_table_count=json_table_count,
        document_file_count=document_file_count,
        document_text_chars=document_text_chars,
        total_columns=total_columns,
        total_rows_sampled=total_rows_sampled,
        has_unstructured_docs=has_unstructured_docs,
        has_sqlite_db=has_sqlite_db,
        has_multiple_sources=has_multiple_sources,
        has_many_tables=has_many_tables,
        has_join_candidates=has_join_candidates,
        question_token_count=len(question_tokens),
        question_mentions_aggregation=question_mentions_aggregation,
        question_mentions_superlative=question_mentions_superlative,
        question_mentions_ratio_or_percentage=question_mentions_ratio_or_percentage,
        question_mentions_filter_or_condition=question_mentions_filter,
        question_mentions_multiple_attributes=question_mentions_multi,
        estimated_context_size=estimated_context_size,
        retrieval_confidence=retrieval_confidence,
        best_candidate_score=best_score,
        best_candidate_contract_valid=best_contract_valid,
        best_candidate_empty=best_empty,
        best_candidate_has_metadata_columns=best_metadata,
        best_candidate_shape_mismatch=best_shape_mismatch,
        complexity_score=complexity_score,
        uncertainty_score=uncertainty_score,
    )


def infer_task_intent(task_context: TaskContext) -> TaskIntent:
    question = task_context.task.question.lower()
    features = extract_question_features(task_context.task.question, task_context.context_tables)
    target_columns = _question_referenced_columns(question, task_context.context_tables)
    operation: str | None = None
    if features.asks_ratio_or_percentage and ("percentage" in features.matched_phrases or "percent" in features.matched_phrases):
        operation = "percentage"
    elif features.asks_ratio_or_percentage:
        operation = "ratio"
    elif any(term in features.strong_terms for term in ("average", "avg", "mean")):
        operation = "average"
    elif any(term in features.strong_terms for term in ("count", "how many")):
        operation = "count"
    elif features.asks_superlative:
        operation = "superlative"
    elif any(term in features.strong_terms for term in ("sum", "total")):
        operation = "sum"
    else:
        operation = "filter_select"

    target_entity = _infer_generic_target_entity(question)
    explicit_scalar_ratio_question = bool(
        re.search(r"\bwhat\s+(?:percentage|percent|proportion|share|ratio)\b", question)
    )
    answer_kind = (
        "scalar"
        if operation in {"average", "count", "sum", "ratio", "percentage"}
        and (features.asks_scalar_metric or explicit_scalar_ratio_question)
        and (features.confidence >= 0.45 or explicit_scalar_ratio_question)
        and (not features.asks_entity_plus_metric or explicit_scalar_ratio_question)
        and (operation in {"ratio", "percentage"} or not features.asks_multi_attribute)
        else "table"
    )
    return TaskIntent(
        domain=None,
        operation=operation,
        answer_kind=answer_kind,
        target_entity=target_entity,
        target_columns=tuple(dict.fromkeys(target_columns)),
    )


def infer_answer_contract(task_context: TaskContext, intent: TaskIntent) -> AnswerContract:
    question = task_context.task.question
    lowered_question = question.lower()
    features = extract_question_features(question, task_context.context_tables)
    schema_columns = [str(column) for table in task_context.context_tables for column in table.frame.columns]
    display_schema_columns = [column for column in schema_columns if _is_display_like_column(column)]
    metric_schema_columns = [
        column
        for table in task_context.context_tables
        for column in table.frame.columns
        if str(column) in _metric_like_columns(table.frame)
    ]
    target_display_columns = [column for column in intent.target_columns if _is_display_like_column(column)]
    target_metric_columns = [column for column in intent.target_columns if column in metric_schema_columns]
    has_entity_metric_schema_evidence = bool(
        (target_display_columns or display_schema_columns)
        and (target_metric_columns or metric_schema_columns)
    )
    requested_column_count = len(intent.target_columns)
    if features.asks_entity_plus_metric and has_entity_metric_schema_evidence:
        display_evidence_count = len(target_display_columns) or min(len(display_schema_columns), 2)
        metric_evidence_count = len(target_metric_columns) or min(len(metric_schema_columns), 1)
        evidence_count = display_evidence_count + metric_evidence_count
        if evidence_count < 2 and requested_column_count >= 2:
            evidence_count = requested_column_count
        requested_column_count = max(requested_column_count, evidence_count)

    def requested_max_columns(default: int | None = None) -> int | None:
        if requested_column_count > 0:
            return requested_column_count
        return default

    if (
        intent.answer_kind == "scalar"
        and (
            features.asks_scalar_metric
            or (
                intent.operation in {"ratio", "percentage"}
                and bool(re.search(r"\bwhat\s+(?:percentage|percent|proportion|share|ratio)\b", lowered_question))
            )
        )
        and (
            features.confidence >= 0.45
            or (
                intent.operation in {"ratio", "percentage"}
                and bool(re.search(r"\bwhat\s+(?:percentage|percent|proportion|share|ratio)\b", lowered_question))
            )
        )
        and (
            not features.asks_entity_or_list
            or (
                intent.operation in {"ratio", "percentage"}
                and bool(re.search(r"\bwhat\s+(?:percentage|percent|proportion|share|ratio)\b", lowered_question))
            )
        )
    ):
        return AnswerContract(
            kind="scalar",
            expected_columns=("answer",),
            max_rows=1,
            max_columns=1,
            allow_empty=False,
            reason="Question features and schema evidence indicate a scalar metric.",
        )

    if intent.target_columns:
        expected_column_count = len(intent.target_columns)
        if (
            features.asks_entity_plus_metric
            and has_entity_metric_schema_evidence
        ) or (features.asks_aggregation and expected_column_count >= 2):
            max_columns = requested_max_columns()
            kind = "multi_attribute"
        elif expected_column_count >= 2 and features.asks_multi_attribute:
            max_columns = requested_max_columns()
            kind = "two_attribute"
        else:
            max_columns = requested_max_columns()
            kind = "two_attribute" if expected_column_count == 2 else "multi_attribute"
        return AnswerContract(
            kind=kind if expected_column_count > 1 or (max_columns is not None and max_columns > 1) else "attribute_lookup",
            expected_columns=intent.target_columns,
            max_rows=1 if (features.asks_superlative and not features.asks_entity_or_list) else None,
            max_columns=max_columns,
            allow_empty=False,
            reason="Question features are backed by matched schema columns.",
        )

    if features.asks_entity_plus_metric and has_entity_metric_schema_evidence:
        return AnswerContract(
            kind="multi_attribute",
            expected_columns=(),
            max_rows=1 if (features.asks_superlative and not features.asks_entity_or_list) else None,
            max_columns=requested_max_columns(),
            allow_empty=False,
            reason="Question features and schema evidence indicate entity plus metric attributes.",
        )

    if (
        features.asks_entity_or_list
        and features.confidence >= 0.12
        and (display_schema_columns or intent.target_columns)
    ):
        return AnswerContract(
            kind="entity_list",
            expected_columns=intent.target_columns,
            max_rows=None,
            max_columns=requested_max_columns(),
            allow_empty=False,
            reason="Question features indicate an entity/list answer with schema evidence.",
        )

    return AnswerContract(
        kind="table",
        expected_columns=(),
        max_rows=None,
        max_columns=None,
        allow_empty=True,
        reason="No strict answer shape inferred.",
    )


def _candidate_looks_like_source_name(frame: pd.DataFrame) -> bool:
    if len(frame) != 1 or len(frame.columns) != 1:
        return False
    value = _text_value(frame.iloc[0, 0]).lower()
    return bool(re.fullmatch(r"(csv|json|db|doc)_[a-z0-9_]+", value))


def _metadata_debug_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        lowered = str(column).lower()
        if lowered in DEBUG_METADATA_COLUMNS:
            columns.append(str(column))
            continue
        if lowered.startswith("debug_") or lowered.startswith("evidence_"):
            columns.append(str(column))
            continue
        if lowered.endswith("_confidence"):
            columns.append(str(column))
    return columns


def _frame_has_mostly_metadata_columns(frame: pd.DataFrame) -> bool:
    if len(frame.columns) == 0:
        return False
    return len(_metadata_debug_columns(frame)) / len(frame.columns) >= 0.5


def validate_answer_contract(
    contract: AnswerContract,
    frame: pd.DataFrame,
    *,
    candidate_source: str | None,
) -> AnswerContractReport:
    row_count = len(frame)
    column_count = len(frame.columns)
    reason = "Candidate satisfies inferred answer contract."
    valid = True
    should_repair = False

    if row_count == 0 and not contract.allow_empty:
        valid = False
        should_repair = True
        reason = "Candidate is empty but the answer contract requires a value."
    elif _metadata_debug_columns(frame):
        valid = False
        should_repair = True
        reason = "Candidate includes metadata/debug columns."
    elif _frame_has_mostly_metadata_columns(frame):
        valid = False
        should_repair = True
        reason = "Candidate is mostly source/evidence/debug fields."
    elif _candidate_looks_like_source_name(frame):
        valid = False
        should_repair = True
        reason = "Candidate appears to be a table/source-name singleton rather than an answer."
    elif contract.max_rows is not None and row_count > contract.max_rows:
        valid = False
        should_repair = True
        reason = f"Candidate has {row_count} rows; expected at most {contract.max_rows}."
    elif contract.max_columns is not None and column_count > contract.max_columns:
        valid = False
        should_repair = True
        reason = f"Candidate has {column_count} columns; expected at most {contract.max_columns}."
    elif contract.expected_columns:
        lowered_columns = {str(column).lower() for column in frame.columns}
        expected_hits = [
            expected
            for expected in contract.expected_columns
            if any(expected.lower() in column for column in lowered_columns)
        ]
        if contract.kind == "attribute_lookup" and not expected_hits:
            valid = False
            should_repair = True
            reason = f"Candidate columns do not match expected answer columns {contract.expected_columns}."

    return AnswerContractReport(
        valid=valid,
        should_repair=should_repair,
        reason=reason,
        contract_kind=contract.kind,
        expected_columns=contract.expected_columns,
        row_count=row_count,
        column_count=column_count,
        candidate_source=candidate_source,
    )


def _candidate_from_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    confidence: float,
    diagnostics: dict[str, Any],
    transformations: tuple[dict[str, Any], ...] = (),
    contract_report: dict[str, Any] | None = None,
    retrieval_context_used: dict[str, Any] | None = None,
    elapsed_seconds: float | None = None,
) -> Candidate | None:
    if not isinstance(frame, pd.DataFrame) or len(frame.columns) == 0:
        return None
    return Candidate(
        frame=frame.reset_index(drop=True),
        source=source,
        confidence=confidence,
        diagnostics=diagnostics,
        transformations=transformations,
        contract_report=contract_report,
        retrieval_context_used=retrieval_context_used,
        elapsed_seconds=elapsed_seconds,
    )


def _candidate_with_contract(candidate: Candidate, contract: AnswerContract) -> Candidate:
    report = validate_answer_contract(contract, candidate.frame, candidate_source=candidate.source)
    return Candidate(
        frame=candidate.frame,
        source=candidate.source,
        confidence=candidate.confidence,
        diagnostics=candidate.diagnostics,
        transformations=candidate.transformations,
        contract_report=report.to_dict(),
        retrieval_context_used=candidate.retrieval_context_used,
        elapsed_seconds=candidate.elapsed_seconds,
    )


def _question_matched_columns(frame: pd.DataFrame, question: str) -> list[str]:
    return list(_question_target_columns(question, frame))


def _aggregate_candidate_value_is_suspicious(frame: pd.DataFrame, question: str) -> bool:
    lowered = question.lower()
    if not ("percentage" in lowered or "percent" in lowered or "ratio" in lowered or "rate" in lowered):
        return False
    if len(frame) != 1 or len(frame.columns) != 1:
        return True
    value = pd.to_numeric(frame.iloc[:, 0], errors="coerce").iloc[0]
    if pd.isna(value):
        return True
    numeric_value = float(value)
    if numeric_value == 0:
        return True
    if ("percentage" in lowered or "percent" in lowered) and not 0 <= numeric_value <= 100:
        return True
    if ("ratio" in lowered or "rate" in lowered) and numeric_value < 0:
        return True
    return False


def _select_columns_by_priority(
    frame: pd.DataFrame,
    *,
    question: str,
    contract: AnswerContract,
    max_columns: int | None,
) -> list[str]:
    existing_columns = [str(column) for column in frame.columns]
    expected = [
        column
        for column in contract.expected_columns
        if any(column.lower() == existing.lower() for existing in existing_columns)
    ]
    selected = [next(existing for existing in existing_columns if existing.lower() == column.lower()) for column in expected]
    for column in _question_matched_columns(frame, question):
        if column not in selected:
            selected.append(column)
    priority_terms = list(DISPLAY_COLUMN_PRIORITY) + [
        "amount",
        "cost",
        "price",
        "score",
        "total",
        "count",
    ]
    if _question_asks_for_identifier(question):
        priority_terms.append("id")
    lowered_to_original = {column.lower(): column for column in existing_columns}
    for term in priority_terms:
        column = lowered_to_original.get(term)
        if column is not None and column not in selected:
            selected.append(column)
    for column in existing_columns:
        if column not in selected:
            selected.append(column)
    return selected[:max_columns] if max_columns is not None else selected


def _question_looks_list_answer(question: str) -> bool:
    if _question_looks_single_answer(question):
        return False
    return extract_question_features(question).asks_entity_or_list


def final_answer_shape_guard(
    *,
    task: PublicTask,
    frame: pd.DataFrame,
    contract: AnswerContract,
    candidate_source: str | None,
) -> tuple[pd.DataFrame, FinalAnswerShapeReport]:
    input_row_count = len(frame)
    input_column_count = len(frame.columns)
    current = frame.copy()
    transformations: list[dict[str, Any]] = []
    removed_columns: list[str] = []
    removed_rows = 0
    contract_before = validate_answer_contract(
        contract,
        current,
        candidate_source=candidate_source,
    ).to_dict()

    metadata_columns = _metadata_debug_columns(current)
    if metadata_columns:
        current = current.drop(columns=metadata_columns)
        removed_columns.extend(metadata_columns)
        transformations.append(
            {
                "kind": "metadata_column_removal",
                "removed_columns": metadata_columns,
                "reason": "debug_metadata_columns",
            }
        )

    if len(current.columns) > 0:
        if not _question_asks_for_identifier(task.question):
            id_helper_columns = [
                str(column)
                for column in current.columns
                if _is_id_like_column(str(column)) and str(column) not in contract.expected_columns
            ]
            has_display_answer = any(_is_display_like_column(str(column)) for column in current.columns)
            if id_helper_columns and has_display_answer and len(id_helper_columns) < len(current.columns):
                current = current.drop(columns=id_helper_columns)
                removed_columns.extend(id_helper_columns)
                transformations.append(
                    {
                        "kind": "final_answer_shape_guard",
                        "action": "drop_id_helper_columns",
                        "removed_columns": id_helper_columns,
                        "reason": "display_answer_present_without_identifier_request",
                    }
                )

        max_rows = contract.max_rows
        max_columns = contract.max_columns
        if contract.kind == "scalar":
            max_rows = 1
            max_columns = 1

        if max_columns is not None and len(current.columns) > max_columns:
            selected_columns = _select_columns_by_priority(
                current,
                question=task.question,
                contract=contract,
                max_columns=max_columns,
            )
            pruned_columns = [str(column) for column in current.columns if str(column) not in selected_columns]
            current = current.loc[:, selected_columns]
            removed_columns.extend(pruned_columns)
            transformations.append(
                {
                    "kind": "final_answer_shape_guard",
                    "action": "prune_columns",
                    "selected_columns": selected_columns,
                    "removed_columns": pruned_columns,
                    "max_columns": max_columns,
                    "reason": f"{contract.kind}_max_columns",
                }
            )

        if max_rows is not None and len(current) > max_rows:
            original_rows = len(current)
            current = current.head(max_rows).reset_index(drop=True)
            removed_rows += max(0, original_rows - len(current))
            transformations.append(
                {
                    "kind": "final_answer_shape_guard",
                    "action": "limit_rows",
                    "max_rows": max_rows,
                    "removed_rows": max(0, original_rows - len(current)),
                    "reason": f"{contract.kind}_max_rows",
                }
            )

    contract_after = validate_answer_contract(
        contract,
        current,
        candidate_source=candidate_source,
    ).to_dict()
    return current.reset_index(drop=True), FinalAnswerShapeReport(
        transformations=tuple(transformations),
        input_row_count=input_row_count,
        input_column_count=input_column_count,
        output_row_count=len(current),
        output_column_count=len(current.columns),
        removed_rows=removed_rows,
        removed_columns=tuple(dict.fromkeys(removed_columns)),
        contract_before=contract_before,
        contract_after=contract_after,
    )


def rank_candidates(
    *,
    candidates: list[Candidate],
    contract: AnswerContract,
    question: str,
    retrieved_context: RetrievedContext | None,
) -> tuple[Candidate | None, CandidateRankingReport]:
    scored: list[tuple[int, Candidate, list[str]]] = []
    features = extract_question_features(question)
    relevant_columns = {
        column.lower()
        for columns in (retrieved_context.relevant_columns.values() if retrieved_context else [])
        for column in columns
    }
    has_valid_non_verifier_candidate = any(
        isinstance(candidate.frame, pd.DataFrame)
        and len(candidate.frame.columns) > 0
        and not any(str(transform.get("kind", "")) in HIGH_RISK_VERIFIER_KINDS for transform in candidate.transformations)
        and validate_answer_contract(contract, candidate.frame, candidate_source=candidate.source).valid
        for candidate in candidates
    )
    percentage_repair_kinds = {
        "ratio_scale_compaction",
        "salvaged_boolean_percentage_compaction",
        "salvaged_context_boolean_percentage_repair",
        "quoted_entity_ratio_repair",
    }
    has_valid_percentage_repair_candidate = any(
        isinstance(candidate.frame, pd.DataFrame)
        and len(candidate.frame.columns) > 0
        and any(str(transform.get("kind", "")) in percentage_repair_kinds for transform in candidate.transformations)
        and validate_answer_contract(contract, candidate.frame, candidate_source=candidate.source).valid
        for candidate in candidates
    )
    for candidate in candidates:
        if not isinstance(candidate.frame, pd.DataFrame) or len(candidate.frame.columns) == 0:
            scored.append((-100, candidate, ["invalid dataframe"]))
            continue
        report = validate_answer_contract(contract, candidate.frame, candidate_source=candidate.source)
        reasons: list[str] = []
        score = int(candidate.confidence * 10)
        question_expects_multirow = (
            features.asks_entity_or_list
            and contract.kind in {"entity_list", "table", "multi_attribute", "grouped_aggregation"}
            and contract.max_rows is None
        )
        if report.valid:
            score += 40
            reasons.append("contract_valid")
        else:
            score -= 60
            reasons.append(f"contract_invalid:{report.reason}")
        if contract.max_columns is not None:
            if len(candidate.frame.columns) <= contract.max_columns:
                if contract.kind in {"scalar", "entity_list", "attribute_lookup"}:
                    score += 15
                    reasons.append("contract_column_shape_ok")
                else:
                    score += 5
                    reasons.append("contract_column_shape_compatible")
            else:
                penalty = 40 if contract.kind in {"scalar", "entity_list", "attribute_lookup"} else 10
                score -= penalty
                reasons.append("contract_column_shape_mismatch")
        else:
            score += 5
            reasons.append("open_column_contract")
        if contract.max_rows is None:
            if len(candidate.frame) > 0:
                score += 5
                reasons.append("non_empty_rows")
            if question_expects_multirow:
                if len(candidate.frame) > 1:
                    score += 15
                    reasons.append("multirow_list_candidate")
                else:
                    score -= 15
                    reasons.append("single_row_for_multirow_question")
        elif len(candidate.frame) <= contract.max_rows:
            if contract.kind == "scalar" or features.asks_superlative:
                score += 15
                reasons.append("contract_row_shape_ok")
            else:
                score += 5
                reasons.append("contract_row_shape_compatible")
        else:
            score -= 30
            reasons.append("contract_row_shape_mismatch")
        metadata_columns = _metadata_debug_columns(candidate.frame)
        if metadata_columns:
            score -= 50
            reasons.append("metadata_columns_present")
        else:
            score += 20
            reasons.append("no_metadata_columns")
        if _question_matched_columns(candidate.frame, question):
            score += 15
            reasons.append("matched_schema_columns")
        candidate_columns = {str(column).lower() for column in candidate.frame.columns}
        if candidate_columns & relevant_columns:
            score += 15
            reasons.append("uses_retrieved_column")
        id_like_columns = [str(column) for column in candidate.frame.columns if _is_id_like_column(str(column))]
        display_like_columns = [
            str(column) for column in candidate.frame.columns if _is_display_like_column(str(column))
        ]
        if id_like_columns and not _question_asks_for_identifier(question):
            if len(id_like_columns) == len(candidate.frame.columns):
                score -= 30
                reasons.append("id_only_without_id_request")
            else:
                score -= 10
                reasons.append("id_helper_column_without_id_request")
                if display_like_columns:
                    score -= 15
                    reasons.append("id_helper_with_display_answer")
        if _question_asks_for_identifier(question) and not features.asks_multi_attribute:
            non_id_columns = [
                str(column)
                for column in candidate.frame.columns
                if not _is_id_like_column(str(column))
            ]
            if id_like_columns and not non_id_columns:
                score += 20
                reasons.append("explicit_identifier_answer_only")
            elif id_like_columns and non_id_columns:
                score -= 20
                reasons.append("explicit_identifier_answer_has_extra_columns")
        if display_like_columns and contract.kind in {"entity_list", "table", "attribute_lookup", "two_attribute", "multi_attribute"}:
            score += 10
            reasons.append("display_column_present")
        if bool(re.search(r"\b(?:comment|text|body|content)\b", question.lower())):
            text_answer_columns = _text_answer_columns(candidate.frame)
            if text_answer_columns and len(text_answer_columns) == len(candidate.frame.columns):
                score += 20
                reasons.append("text_answer_columns_only")
            elif text_answer_columns and len(candidate.frame.columns) > len(text_answer_columns):
                score -= 25
                reasons.append("text_answer_has_extra_columns")
        if _question_asks_for_url(question):
            if any("url" in str(column).lower() or "link" in str(column).lower() for column in candidate.frame.columns):
                score += 20
                reasons.append("url_column_present")
            else:
                score -= 20
                reasons.append("url_question_without_url_column")
        if _question_asks_count_metric(question) and (
            contract.kind == "scalar" or (features.asks_aggregation and not _question_asks_for_identifier(question))
        ):
            count_like_columns = [
                str(column)
                for column in candidate.frame.columns
                if any(term in str(column).lower() for term in ("answer", "count", "total", "number"))
            ]
            if not count_like_columns:
                score -= 20
                reasons.append("count_question_without_count_like_column")
        if any(transform.get("kind") == "identifier_resolution" for transform in candidate.transformations):
            score += 10
            reasons.append("display_id_resolution_evidence")
        if any(transform.get("kind") == "cheap_count_fallback" for transform in candidate.transformations):
            fallback = next(
                transform for transform in candidate.transformations if transform.get("kind") == "cheap_count_fallback"
            )
            if any(term in features.strong_terms for term in ("how many", "count")) and not fallback.get("executed_filter"):
                if _question_mentions_filter_or_condition(question):
                    score -= 120
                    reasons.append("count_fallback_without_required_filter")
                elif fallback.get("plain_row_count_request") or fallback.get("reason") == "single_context_table":
                    score += 2
                    reasons.append("count_operation_evidence")
                else:
                    score -= 30
                    reasons.append("weak_count_fallback_evidence")
            else:
                score -= 60
                reasons.append("weak_count_fallback_evidence")
        if any(str(transform.get("kind", "")) == "aggregate_ratio_verification" for transform in candidate.transformations):
            aggregate_transforms = [
                transform
                for transform in candidate.transformations
                if str(transform.get("kind", "")) == "aggregate_ratio_verification"
            ]
            if _aggregate_candidate_value_is_suspicious(candidate.frame, question):
                score -= 80
                reasons.append("aggregate_ratio_sanity_failed")
            if any(
                not transform.get("numerator_column") or not transform.get("denominator_column")
                for transform in aggregate_transforms
            ):
                score -= 40
                reasons.append("aggregate_ratio_components_unclear")
            if not report.valid:
                score -= 30
                reasons.append("aggregate_ratio_contract_invalid")
        if (
            any(str(transform.get("kind", "")) in HIGH_RISK_VERIFIER_KINDS for transform in candidate.transformations)
        ):
            score -= 15
            reasons.append("verifier_candidate_discount")
            if not report.valid:
                score -= 20
                reasons.append("verifier_contract_invalid")
            if not _question_matched_columns(candidate.frame, question) and not (candidate_columns & relevant_columns):
                score -= 20
                reasons.append("verifier_without_explicit_column_evidence")
            if has_valid_non_verifier_candidate:
                score -= 15
                reasons.append("valid_non_verifier_candidate_available")
        if any(
            str(transform.get("kind", ""))
            in {"context_superlative_verification", "aggregate_ratio_verification"}
            for transform in candidate.transformations
        ):
            score -= 50
            reasons.append("high_risk_verifier_discount")
        if any(str(transform.get("kind", "")) == "ratio_scale_compaction" for transform in candidate.transformations):
            if not _aggregate_candidate_value_is_suspicious(candidate.frame, question) and report.valid:
                score += 35
                reasons.append("ratio_scale_evidence")
            else:
                score -= 35
                reasons.append("ratio_scale_sanity_failed")
        if any(
            str(transform.get("kind", "")) == "salvaged_boolean_percentage_compaction"
            for transform in candidate.transformations
        ):
            if not _aggregate_candidate_value_is_suspicious(candidate.frame, question) and report.valid:
                score += 35
                reasons.append("salvaged_boolean_percentage_evidence")
            else:
                score -= 35
                reasons.append("salvaged_boolean_percentage_sanity_failed")
        if any(
            str(transform.get("kind", "")) == "salvaged_context_boolean_percentage_repair"
            for transform in candidate.transformations
        ):
            if not _aggregate_candidate_value_is_suspicious(candidate.frame, question) and report.valid:
                score += 25
                reasons.append("salvaged_context_percentage_evidence")
            else:
                score -= 45
                reasons.append("salvaged_context_percentage_sanity_failed")
        if any(
            str(transform.get("kind", "")) == "salvaged_detail_count_compaction"
            for transform in candidate.transformations
        ):
            if report.valid and not extract_question_features(question).asks_ratio_or_percentage:
                score += 35
                reasons.append("salvaged_detail_count_evidence")
            else:
                score -= 35
                reasons.append("salvaged_detail_count_sanity_failed")
        if any(str(transform.get("kind", "")) == "quoted_entity_ratio_repair" for transform in candidate.transformations):
            if not _aggregate_candidate_value_is_suspicious(candidate.frame, question) and report.valid:
                score += 30
                reasons.append("quoted_entity_ratio_evidence")
            else:
                score -= 45
                reasons.append("quoted_entity_ratio_sanity_failed")
        if (
            features.asks_ratio_or_percentage
            and has_valid_percentage_repair_candidate
            and not any(str(transform.get("kind", "")) in percentage_repair_kinds for transform in candidate.transformations)
        ):
            score -= 35
            reasons.append("percentage_question_non_percentage_candidate")
        if any(str(transform.get("kind", "")) == "column_only_compaction" for transform in candidate.transformations):
            transform = next(
                transform
                for transform in candidate.transformations
                if str(transform.get("kind", "")) == "column_only_compaction"
            )
            if (
                transform.get("row_count_preserved") is True
                and len(candidate.frame.columns) < len(transform.get("input_columns", []))
                and not metadata_columns
            ):
                score += 20
                reasons.append("safe_column_only_compaction")
        row_count = len(candidate.frame)
        column_count = len(candidate.frame.columns)
        if row_count > 1000:
            score -= 40
            reasons.append("large_raw_table_candidate")
        if row_count > 5000:
            score -= 30
            reasons.append("very_large_candidate")
        if column_count > max(contract.max_columns or 6, 6):
            score -= 15
            reasons.append("wide_candidate")
        if len(id_like_columns) >= 2 and not _question_asks_for_identifier(question):
            score -= 20
            reasons.append("many_id_helper_columns")
        if candidate.source == "cheap_count_fallback" and not any(term in features.strong_terms for term in ("how many", "count")):
            score -= 100
            reasons.append("count_fallback_without_strong_count_evidence")
        if len(candidate.frame) == 0 and not contract.allow_empty:
            score -= 100
            reasons.append("empty_required_answer")
        scored.append((score, candidate, reasons))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = scored[0][1] if scored else None
    score_entries = tuple(
        CandidateScore(
            source=candidate.source,
            score=score,
            selected=(selected is candidate),
            reasons=tuple(reasons),
            row_count=len(candidate.frame) if isinstance(candidate.frame, pd.DataFrame) else 0,
            column_count=len(candidate.frame.columns) if isinstance(candidate.frame, pd.DataFrame) else 0,
        )
        for score, candidate, reasons in scored
    )
    rejection_reasons = {
        candidate.source: reasons
        for _, candidate, reasons in scored
        if selected is not candidate
    }
    return selected, CandidateRankingReport(
        selected_source=selected.source if selected is not None else None,
        candidate_scores=score_entries,
        rejection_reasons=rejection_reasons,
    )


def _optional_column(frame: pd.DataFrame, column_name: str) -> str | None:
    lowered = column_name.lower()
    for column in frame.columns:
        if str(column).lower() == lowered:
            return str(column)
    return None


def _source_kind_from_context_path(path: str) -> str:
    if path.startswith("doc/") or path.startswith("doc/*::") or path.endswith("::materialized") or "/doc/" in f"/{path}":
        return "document_table"
    if "::" in path or path.lower().endswith((".db", ".sqlite")):
        return "sqlite"
    suffix = Path(path.split("::", 1)[0]).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    return "unknown"


def _column_key_variants(column_name: str) -> set[str]:
    lowered = column_name.lower()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    variants = {lowered, compact}
    if lowered.endswith("_id"):
        variants.add(lowered.removesuffix("_id"))
        variants.add(compact.removesuffix("id"))
    if lowered.endswith("id"):
        variants.add(lowered.removesuffix("id"))
        variants.add(compact.removesuffix("id"))
    if lowered == "id":
        variants.add("id")
    return {variant for variant in variants if variant}


def _entity_name_variants(table_name: str) -> set[str]:
    lowered = table_name.lower()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    variants = {lowered, compact}
    for value in (lowered, compact):
        if value.endswith("s") and len(value) > 1:
            variants.add(value[:-1])
    return {variant for variant in variants if variant}


def build_schema_graph(context_tables: list[ContextTable]) -> dict[str, Any]:
    tables: list[dict[str, Any]] = []
    column_index: list[tuple[str, str, set[str]]] = []
    for table in context_tables:
        columns: list[dict[str, Any]] = []
        for column in table.frame.columns:
            column_name = str(column)
            sample_values = [
                _json_safe_cell(value)
                for value in table.frame[column].dropna().head(MAX_SCHEMA_SAMPLE_ROWS).tolist()
            ]
            columns.append(
                {
                    "name": column_name,
                    "dtype": str(table.frame[column].dtype),
                    "sample_values": sample_values,
                }
            )
            column_index.append((table.name, column_name, _column_key_variants(column_name)))
        tables.append(
            {
                "name": table.name,
                "path": table.path,
                "source_kind": _source_kind_from_context_path(table.path),
                "row_count": len(table.frame),
                "columns": columns,
                "metadata": table.metadata or {},
            }
        )

    joins: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for left_index, (left_table, left_column, left_variants) in enumerate(column_index):
        for right_table, right_column, right_variants in column_index[left_index + 1 :]:
            if left_table == right_table:
                continue
            left_lower = left_column.lower()
            right_lower = right_column.lower()
            id_like = (
                left_lower == "id"
                or right_lower == "id"
                or left_lower.endswith("id")
                or right_lower.endswith("id")
            )
            if not id_like:
                continue
            shared = left_variants & right_variants
            entity_match = (
                left_lower.endswith("id")
                and right_lower == "id"
                and bool((left_variants - {"id"}) & _entity_name_variants(right_table))
            ) or (
                right_lower.endswith("id")
                and left_lower == "id"
                and bool((right_variants - {"id"}) & _entity_name_variants(left_table))
            )
            if not shared and not entity_match:
                continue
            key = (left_table, left_column, right_table, right_column)
            if key in seen:
                continue
            seen.add(key)
            joins.append(
                {
                    "left_table": left_table,
                    "left_column": left_column,
                    "right_table": right_table,
                    "right_column": right_column,
                    "reason": "shared_identifier_variant",
                }
            )
    return {"tables": tables, "join_candidates": joins}


def _numeric_equals_mask(series: pd.Series, value: Any) -> pd.Series:
    numeric_series = pd.to_numeric(series, errors="coerce")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        numeric_value = None
    if numeric_value is not None and numeric_series.notna().any():
        return numeric_series.eq(numeric_value)
    return series.map(_text_value).eq(_text_value(value))


def _extract_year(question: str) -> int | None:
    match = re.search(r"\b(19\d{2}|20\d{2})\b", question)
    return int(match.group(1)) if match else None


def _ordinal_from_question(question: str) -> int | None:
    lowered = question.lower()
    ordinal_words = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
        "seventh": 7,
        "eighth": 8,
        "ninth": 9,
        "tenth": 10,
    }
    for word, value in ordinal_words.items():
        if re.search(rf"\b{word}\b", lowered):
            return value
    digit_match = re.search(
        r"\b(?:rank(?:ed)?|position|place|finished)\s+(\d+)(?:st|nd|rd|th)?\b",
        lowered,
    )
    if digit_match:
        return int(digit_match.group(1))
    return None


def _parse_time_seconds(value: Any) -> float | None:
    text = _text_value(value)
    if not text:
        return None
    text = text.lstrip("+")
    if not re.match(r"^\d+(?::\d{1,2}){1,2}(?:\.\d+)?$", text):
        return None
    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
    except ValueError:
        return None
    return None


def _extract_time_floor_from_question(question: str) -> int | None:
    match = re.search(r"\b\d+:\d{2}(?::\d{2})?(?:\.\d+)?\b", question)
    if not match:
        return None
    seconds = _parse_time_seconds(match.group(0))
    return int(seconds) if seconds is not None else None


def try_deterministic_candidate(task_context: TaskContext, intent: TaskIntent) -> Candidate | None:
    del task_context, intent
    return None


def _make_lighthouse_salvage_executor() -> Any:
    from databao.agent.core import ExecutionResult
    from databao.agent.core.executor import OutputModalityHints
    from databao.agent.executors.lighthouse.executor import LighthouseExecutor

    class LighthouseSalvageExecutor(LighthouseExecutor):
        @staticmethod
        def _invoke_graph_sync(compiled_graph: Any, start_state: Any, **kwargs: Any) -> Any:
            config = kwargs.get("config")
            stream_kwargs = {key: value for key, value in kwargs.items() if key not in {"config", "stream", "writer"}}
            last_state = None
            try:
                for mode, chunk in compiled_graph.stream(
                    start_state,
                    stream_mode=["values"],
                    config=config,
                    **stream_kwargs,
                ):
                    if mode == "values":
                        last_state = chunk
            except Exception:
                if isinstance(last_state, dict) and last_state.get("df") is not None:
                    return last_state
                raise
            if last_state is None:
                raise RuntimeError("Graph execution produced no output state")
            return last_state

        def _get_result_with_salvage(self, state: dict[str, Any]) -> Any:
            try:
                return self._graph.get_result(state)
            except Exception:
                df = state.get("df")
                if df is None:
                    raise
                return ExecutionResult(
                    text="Salvaged the latest successful SQL result after Databao did not submit a final result.",
                    df=df,
                    code=state.get("sql", ""),
                    meta={
                        "visualization_prompt": state.get("visualization_prompt"),
                        ExecutionResult.META_MESSAGES_KEY: state.get("messages", []),
                        "submit_called": False,
                        "salvaged_latest_query_result": True,
                    },
                )

        def execute(
            self,
            opas: list[Any],
            cache: Any,
            llm_config: Any,
            agent_config: Any,
            domain: Any,
            *,
            rows_limit: int = 100,
            stream: bool = True,
            writer: Any = None,
        ) -> Any:
            del stream
            self._init_sources_from_domain(domain)
            system_prompt = self.render_system_prompt(self._duckdb_connection, domain, agent_config.recursion_limit)
            init_state = self._graph.init_state([], limit_max_rows=rows_limit)
            execution_result, _ = self._execute_core(
                opas,
                cache,
                llm_config,
                agent_config,
                domain,
                system_prompt=system_prompt,
                init_state=init_state,
                get_result=self._get_result_with_salvage,
                stream=True,
                writer=writer,
            )
            execution_result.meta[OutputModalityHints.META_KEY] = self._make_output_modality_hints(execution_result)
            return execution_result

    return LighthouseSalvageExecutor()


def choose_route_policy(
    task_context: TaskContext,
    intent: TaskIntent,
) -> RouteDecision:
    candidate = try_deterministic_candidate(task_context, intent)
    if candidate is not None:
        return RouteDecision(
            route="deterministic",
            reason="High-confidence schema/question route matched.",
            confidence=candidate.confidence,
            candidate=candidate,
        )

    return RouteDecision(
        route="databao",
        reason="No high-confidence deterministic route matched; using Databao-first baseline.",
        confidence=0.0,
    )


def build_databao_agent(task: PublicTask, databao_env: DatabaoEnvironment) -> Any:
    import databao.agent as bao

    domain = bao.domain()
    register_context_sources(
        domain,
        task.context_dir,
        retrieved_context=_CURRENT_RETRIEVED_CONTEXT.get(),
    )
    model_kwargs: dict[str, Any] = {}
    diagnostics = _CURRENT_DIAGNOSTICS.get()
    if diagnostics is not None:
        model_kwargs["callbacks"] = [DatabaoLangChainCallback(diagnostics)]
    model_kwargs["max_retries"] = 0
    llm_config = bao.LLMConfig(
        name=databao_env.model_name,
        api_base_url=databao_env.model_api_url,
        temperature=0.0,
        max_tokens=MAX_DATABAO_TOKENS,
        timeout=_databao_timeout_seconds(),
        use_responses_api=False,
        model_kwargs=model_kwargs,
    )
    executor_type = _databao_executor_type()
    data_executor = _make_lighthouse_salvage_executor() if executor_type == "lighthouse_salvage" else None
    return bao.agent(
        domain,
        name=f"databao_{task.task_id}",
        llm_config=llm_config,
        data_executor=data_executor,
        rows_limit=1000,
        stream_ask=False,
        stream_plot=False,
        auto_output_modality=False,
    )


def _build_question_prompt(task: PublicTask) -> str:
    return (
        "Use only the registered task data sources.\n"
        "Return only the final answer table.\n"
        "Do not include explanation, evidence, source, confidence, intermediate columns, or debug metadata.\n"
        "If the question asks for one value, return exactly one row and one column.\n"
        "If the question asks for matching records or a list, preserve all matching rows.\n"
        "If the question asks for a list of entities, return exactly one column.\n"
        "If the question asks for a table with multiple attributes, return only the explicitly requested attributes.\n"
        "Prefer display, name, title, or label columns for entity answers unless the question explicitly asks for ID or code.\n"
        "Never output metadata or debug columns.\n\n"
        f"Question: {task.question}"
    )


def _json_safe_cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value[:MAX_CANDIDATE_CELL_CHARS]
    if isinstance(value, bool | int | float):
        if pd.isna(value):
            return None
        return value

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    text = str(value)
    if len(text) > MAX_CANDIDATE_CELL_CHARS:
        return text[:MAX_CANDIDATE_CELL_CHARS] + "...[truncated]"
    return text


def _candidate_table_payload(frame: pd.DataFrame, task: PublicTask | None = None) -> dict[str, Any]:
    columns = [
        {
            "index": column_index,
            "name": str(column_name),
        }
        for column_index, column_name in enumerate(frame.columns)
    ]
    rows: list[dict[str, Any]] = []

    for row_index in range(min(len(frame), MAX_CANDIDATE_PAYLOAD_ROWS)):
        row_payload = {
            "row_index": row_index,
            "values": [_json_safe_cell(value) for value in frame.iloc[row_index].tolist()],
        }
        candidate_payload = {
            "columns": columns,
            "rows": rows + [row_payload],
        }
        candidate_text = json.dumps(candidate_payload, ensure_ascii=False, separators=(",", ":"))
        if rows and len(candidate_text) > MAX_CANDIDATE_PAYLOAD_CHARS:
            break
        rows.append(row_payload)

    payload: dict[str, Any] = {
        "row_count": len(frame),
        "column_count": len(frame.columns),
        "columns": columns,
        "rows_included": rows,
        "all_rows_included": len(rows) == len(frame),
    }
    if task is not None:
        context_enrichment = _context_enrichment_payload(task, frame)
        if context_enrichment:
            payload["context_enrichment"] = context_enrichment
    return payload


def _write_prediction_csv(frame: pd.DataFrame, path: Path) -> None:
    if len(frame.columns) == 0:
        raise ValueError("Databao returned a dataframe without columns.")
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    for column in output.columns:
        series = output[column]
        if not pd.api.types.is_float_dtype(series):
            continue
        non_null = series.dropna()
        if not non_null.empty and non_null.map(float.is_integer).all():
            output[column] = series.astype("Int64")
    output.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def _prediction_csv_status(path: Path) -> tuple[bool, bool, int | None, int | None]:
    if not path.exists():
        return False, False, None, None
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:  # noqa: BLE001
        return True, False, None, None
    return True, len(frame.columns) > 0, len(frame), len(frame.columns)


def _read_task_progress(logs_dir: Path, task_id: str) -> dict[str, Any] | None:
    progress_path = logs_dir / f"{task_id}.progress.json"
    if not progress_path.exists():
        return None
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _context_payload_profile(
    task_context: TaskContext | None,
    context_summary: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if task_context is None:
        return None
    tables = task_context.context_tables
    table_profiles: list[dict[str, Any]] = []
    for table in tables:
        table_profiles.append(
            {
                "name": table.name,
                "path": table.path,
                "source_kind": _source_kind_from_context_path(table.path),
                "row_count": len(table.frame),
                "column_count": len(table.frame.columns),
            }
        )
    registered_sources = (
        list(context_summary.get("registered_sources", []))
        if isinstance(context_summary, Mapping)
        else []
    )
    return {
        "registered_source_count": len(registered_sources),
        "registered_sources": registered_sources,
        "table_count": len(tables),
        "table_profiles": table_profiles,
        "total_table_rows": sum(len(table.frame) for table in tables),
        "total_table_columns": sum(len(table.frame.columns) for table in tables),
        "document_text_chars": len(task_context.document_text),
        "document_file_count": (
            len(context_summary.get("document_files", []))
            if isinstance(context_summary, Mapping)
            else None
        ),
    }


def _candidate_raw_table_diagnostics(
    frame: pd.DataFrame,
    task_context: TaskContext | None,
) -> dict[str, Any]:
    row_count = len(frame)
    column_count = len(frame.columns)
    closest_table: dict[str, Any] | None = None
    if task_context is not None:
        for table in task_context.context_tables:
            source_rows = len(table.frame)
            if source_rows == 0:
                continue
            ratio = row_count / source_rows
            distance = abs(row_count - source_rows)
            candidate = {
                "name": table.name,
                "path": table.path,
                "source_row_count": source_rows,
                "row_count_ratio": round(ratio, 4),
                "row_count_distance": distance,
            }
            if closest_table is None or distance < int(closest_table["row_count_distance"]):
                closest_table = candidate
    close_to_source = False
    if closest_table is not None:
        ratio = float(closest_table["row_count_ratio"])
        close_to_source = 0.9 <= ratio <= 1.1
    return {
        "row_count": row_count,
        "column_count": column_count,
        "wide_candidate": column_count >= 6,
        "large_candidate": row_count >= 500,
        "close_to_source_row_count": close_to_source,
        "closest_source_table": closest_table,
        "raw_table_like": bool(row_count >= 100 and (column_count >= 6 or close_to_source)),
    }


def _databao_failure_type(failure_reason: str | None) -> str | None:
    if not failure_reason:
        return None
    lowered = failure_reason.lower()
    if "graphrecursionerror" in lowered or "recursion" in lowered:
        return "graph_recursion"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if "did not return a dataframe" in lowered or "non_dataframe" in lowered:
        return "non_dataframe"
    return "exception"


def _write_task_log(
    log_path: Path,
    *,
    task: PublicTask,
    succeeded: bool,
    elapsed_seconds: float,
    prediction_written: bool | None = None,
    scorable: bool | None = None,
    context_summary: dict[str, Any] | None = None,
    candidate_source: str | None = None,
    postprocessing: dict[str, Any] | None = None,
    timings: dict[str, float] | None = None,
    llm_calls: list[dict[str, Any]] | None = None,
    route_policy: dict[str, Any] | None = None,
    question_features: dict[str, Any] | None = None,
    answer_contract: dict[str, Any] | None = None,
    heuristic_level: str | None = None,
    enabled_strategies: list[dict[str, Any]] | None = None,
    applied_strategies: list[dict[str, Any]] | None = None,
    document_extraction_used: bool | None = None,
    retrieved_context: dict[str, Any] | None = None,
    complexity_profile: dict[str, Any] | None = None,
    candidates: list[dict[str, Any]] | None = None,
    candidate_scores: list[dict[str, Any]] | None = None,
    selected_candidate_source: str | None = None,
    final_answer_guard: dict[str, Any] | None = None,
    provisional_written_stage: str | None = None,
    missing_prediction_reason: str | None = None,
    active_progress: dict[str, Any] | None = None,
    timeout_budget: dict[str, Any] | None = None,
    failure_reason: str | None = None,
    context_payload_profile: dict[str, Any] | None = None,
    databao_failure_type: str | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": task.task_id,
        "difficulty": task.difficulty,
        "succeeded": succeeded,
        "prediction_written": bool(prediction_written),
        "scorable": bool(scorable),
        "elapsed_seconds": elapsed_seconds,
        "context_summary": context_summary,
        "candidate_source": candidate_source,
        "postprocessing": postprocessing,
        "timings": timings,
        "llm_calls": llm_calls or [],
        "route_policy": route_policy,
        "question_features": question_features,
        "answer_contract": answer_contract,
        "heuristic_level": heuristic_level,
        "enabled_strategies": enabled_strategies or [],
        "applied_strategies": applied_strategies or [],
        "document_extraction_used": bool(document_extraction_used),
        "retrieved_context": retrieved_context,
        "complexity_profile": complexity_profile,
        "candidates": candidates or [],
        "candidate_scores": candidate_scores or [],
        "selected_candidate_source": selected_candidate_source,
        "final_answer_guard": final_answer_guard,
        "provisional_written_stage": provisional_written_stage,
        "missing_prediction_reason": missing_prediction_reason,
        "active_progress": active_progress,
        "timeout_budget": timeout_budget,
        "failure_reason": failure_reason,
        "context_payload_profile": context_payload_profile,
        "databao_failure_type": databao_failure_type,
    }
    log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _timeout_task_artifact(
    *,
    task: PublicTask,
    output_root: Path,
    logs_dir: Path,
    elapsed_seconds: float,
    timeout_seconds: int,
) -> DatabaoTaskArtifacts:
    task_output_dir = output_root / task.task_id
    log_path = logs_dir / f"{task.task_id}.json"
    failure_reason = (
        f"TimeoutError: task exceeded DATABAO_TASK_TIMEOUT_SECONDS/task_timeout_seconds "
        f"limit of {timeout_seconds} seconds."
    )
    timings = {"total": round(elapsed_seconds, 3), "task_timeout_seconds": timeout_seconds}
    prediction_path = task_output_dir / "prediction.csv"
    prediction_written, scorable, row_count, column_count = _prediction_csv_status(prediction_path)
    progress = _read_task_progress(logs_dir, task.task_id)
    missing_prediction_reason = None if prediction_written else "task timed out before prediction.csv was written"
    _write_task_log(
        log_path,
        task=task,
        succeeded=False,
        prediction_written=prediction_written,
        scorable=scorable,
        elapsed_seconds=round(elapsed_seconds, 3),
        timings=timings,
        heuristic_level=_heuristic_level(),
        enabled_strategies=[
            {
                "strategy_name": "generic_document_tables",
                "strategy_kind": "generic",
                "enabled_by": _heuristic_level(),
            }
        ],
        timeout_budget={"active_progress": progress},
        missing_prediction_reason=missing_prediction_reason,
        active_progress=progress,
        failure_reason=failure_reason,
        databao_failure_type="timeout",
    )
    traceback_path = logs_dir / f"{task.task_id}.traceback.txt"
    traceback_path.parent.mkdir(parents=True, exist_ok=True)
    traceback_path.write_text(failure_reason + "\n", encoding="utf-8")
    return DatabaoTaskArtifacts(
        task_id=task.task_id,
        task_output_dir=task_output_dir,
        prediction_csv_path=prediction_path if prediction_written else None,
        log_path=log_path,
        succeeded=False,
        prediction_written=prediction_written,
        scorable=scorable,
        failure_reason=failure_reason,
        elapsed_seconds=round(elapsed_seconds, 3),
        row_count=row_count,
        column_count=column_count,
        timings=timings,
        heuristic_level=_heuristic_level(),
        databao_failure_type="timeout",
    )


def _ask_databao_candidate(
    task: PublicTask,
    databao_env: DatabaoEnvironment,
    agent_builder: AgentBuilder,
    diagnostics: TaskDiagnostics,
    *,
    stage_name: str = "databao_ask",
) -> pd.DataFrame:
    with diagnostics.stage(stage_name):
        agent = agent_builder(task, databao_env)
        thread = agent.thread(stream_ask=False, stream_plot=False, auto_output_modality=False)
        thread.ask(_build_question_prompt(task), stream=False)
        frame = thread.df()
        try:
            code = thread.code()
            meta = thread.meta()
        except Exception:  # noqa: BLE001
            code = None
            meta = {}
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Databao did not return a dataframe.")
    if code:
        frame.attrs["databao_code"] = _text_value(code)
    if isinstance(meta, Mapping):
        if "submit_called" in meta:
            frame.attrs["databao_submit_called"] = bool(meta.get("submit_called"))
        if "salvaged_latest_query_result" in meta:
            frame.attrs["databao_salvaged_latest_query_result"] = bool(meta.get("salvaged_latest_query_result"))
    frame.attrs["databao_executor_type"] = _databao_executor_type()
    diagnostics.checkpoint(
        "databao_frame_received",
        row_count=len(frame),
        column_count=len(frame.columns),
        columns=[str(column) for column in frame.columns[:20]],
        **_databao_frame_execution_payload(frame),
    )
    return frame


def run_databao_task(
    *,
    task: PublicTask,
    output_root: Path,
    logs_dir: Path,
    databao_env: DatabaoEnvironment,
    agent_builder: AgentBuilder = build_databao_agent,
    answer_postprocessor: AnswerPostprocessor = postprocess_answer_table,
    **deprecated_options: Any,
) -> DatabaoTaskArtifacts:
    del deprecated_options
    started_at = perf_counter()
    deadline_at = _deadline_from_start(started_at)
    task_output_dir = output_root / task.task_id
    prediction_csv_path = task_output_dir / "prediction.csv"
    log_path = logs_dir / f"{task.task_id}.json"
    context_summary: dict[str, Any] | None = None
    candidate_source: str | None = None
    postprocess_report_payload: dict[str, Any] | None = None
    route_policy: RouteDecision | None = None
    question_features: QuestionFeatures | None = None
    answer_contract: AnswerContract | None = None
    answer_contract_report: AnswerContractReport | None = None
    task_context: TaskContext | None = None
    retrieved_context: RetrievedContext | None = None
    complexity_profile: TaskComplexityProfile | None = None
    context_payload_profile: dict[str, Any] | None = None
    ranking_report: CandidateRankingReport | None = None
    final_guard_report: FinalAnswerShapeReport | None = None
    provisional_written_stage: str | None = None
    missing_prediction_reason: str | None = None
    databao_failure_type: str | None = None
    effective_heuristic_level = _heuristic_level()
    enabled_strategies: list[dict[str, Any]] = []
    applied_strategies: list[dict[str, Any]] = []
    document_extraction_used = False
    candidates: list[Candidate] = []
    diagnostics = TaskDiagnostics(task=task, logs_dir=logs_dir, databao_env=databao_env)
    diagnostics_token = _CURRENT_DIAGNOSTICS.set(diagnostics)
    deadline_token = _CURRENT_TASK_DEADLINE.set(deadline_at)

    def timeout_budget_payload() -> dict[str, Any]:
        timeout_seconds = _env_int(DATABAO_TASK_TIMEOUT_SECONDS_ENV, 0)
        return {
            "task_timeout_seconds": timeout_seconds or None,
            "deadline_enabled": deadline_at is not None,
            "remaining_seconds": (
                round(_remaining_seconds(deadline_at) or 0.0, 3)
                if deadline_at is not None
                else None
            ),
            "databao_timeout_seconds": _databao_timeout_seconds(deadline_at),
            "configured_databao_timeout_seconds": _env_int(
                DATABAO_DATABAO_TIMEOUT_SECONDS_ENV,
                DATABAO_AGENT_TIMEOUT_SECONDS,
            ),
            "databao_executor_type": _databao_executor_type(),
            "active_progress": _read_task_progress(logs_dir, task.task_id),
        }

    def candidate_payloads() -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for candidate in candidates:
            payload = candidate.to_dict()
            payload["raw_table_diagnostics"] = _candidate_raw_table_diagnostics(candidate.frame, task_context)
            payloads.append(payload)
        return payloads

    def candidate_score_payloads() -> list[dict[str, Any]]:
        if ranking_report is None:
            return []
        return [score.to_dict() for score in ranking_report.candidate_scores]

    def record_candidate(candidate: Candidate | None) -> Candidate | None:
        if candidate is None:
            return None
        if answer_contract is None:
            candidates.append(candidate)
            return candidate
        enriched = _candidate_with_contract(candidate, answer_contract)
        candidates.append(enriched)
        return enriched

    def build_candidate_from_frame(
        frame: pd.DataFrame,
        *,
        source: str,
        confidence: float,
        diagnostics_payload: dict[str, Any],
        transformations: tuple[dict[str, Any], ...] = (),
        elapsed_seconds: float | None = None,
    ) -> Candidate | None:
        return _candidate_from_frame(
            frame,
            source=source,
            confidence=confidence,
            diagnostics=diagnostics_payload,
            transformations=transformations,
            retrieval_context_used=retrieved_context.to_dict() if retrieved_context is not None else None,
            elapsed_seconds=elapsed_seconds,
        )

    def add_postprocessed_candidate(candidate: Candidate) -> None:
        nonlocal postprocess_report_payload
        try:
            with diagnostics.stage(f"postprocess_{candidate.source}"):
                postprocessed_frame, report = answer_postprocessor(task, candidate.frame)
                if not isinstance(postprocessed_frame, pd.DataFrame):
                    raise TypeError("Answer postprocessor did not return a dataframe.")
        except Exception as exc:  # noqa: BLE001
            report = DeterministicPostprocessReport(
                applied=False,
                transformations=[],
                failure_reason=_safe_exception_summary(exc, (databao_env.model_api_key,)),
                input_row_count=len(candidate.frame),
                input_column_count=len(candidate.frame.columns),
                output_row_count=len(candidate.frame),
                output_column_count=len(candidate.frame.columns),
            )
            candidate.diagnostics.setdefault("postprocess_error", report.failure_reason)
            return

        if not report.applied and postprocessed_frame.equals(candidate.frame):
            return

        postprocess_report_payload = report.to_dict()
        for transform in report.transformations:
            applied_strategies.append(
                {
                    "strategy_name": str(transform.get("kind", "postprocess")),
                    "strategy_kind": "generic",
                    "result": transform,
                    "applied_to_final_answer": False,
                }
            )
        postprocessed_candidate = build_candidate_from_frame(
            postprocessed_frame,
            source=f"{candidate.source}_postprocessed",
            confidence=min(candidate.confidence + 0.08, 1.0),
            diagnostics_payload={"postprocess_report": report.to_dict()},
            transformations=tuple(report.transformations),
            elapsed_seconds=None,
        )
        record_candidate(postprocessed_candidate)

    def add_verifier_candidates(base_candidate: Candidate) -> None:
        for verifier_frame, transforms, confidence in generate_verifier_candidate_frames(task, base_candidate.frame):
            verifier_candidate = record_candidate(
                build_candidate_from_frame(
                    verifier_frame,
                    source=f"{base_candidate.source}_verifier",
                    confidence=confidence,
                    diagnostics_payload={"verifier_transforms": transforms},
                    transformations=tuple(transforms),
                )
            )
            if verifier_candidate is not None:
                applied_strategies.extend(
                    {
                        "strategy_name": str(transform.get("kind", "verifier")),
                        "strategy_kind": "generic",
                        "result": transform,
                        "applied_to_final_answer": False,
                    }
                    for transform in transforms
                )

    def add_column_compaction_candidate(base_candidate: Candidate) -> None:
        try:
            with diagnostics.stage(f"column_compaction_{base_candidate.source}"):
                compact_frame, transforms = apply_column_only_compactor(
                    task,
                    base_candidate.frame,
                    answer_contract,
                )
        except Exception as exc:  # noqa: BLE001
            base_candidate.diagnostics.setdefault(
                "column_compaction_error",
                _safe_exception_summary(exc, (databao_env.model_api_key,)),
            )
            return
        if not transforms or compact_frame.equals(base_candidate.frame):
            return
        compact_candidate = record_candidate(
            build_candidate_from_frame(
                compact_frame,
                source=f"{base_candidate.source}_column_compact",
                confidence=min(base_candidate.confidence + 0.04, 1.0),
                diagnostics_payload={"column_compaction_transforms": transforms},
                transformations=tuple(transforms),
            )
        )
        if compact_candidate is not None:
            applied_strategies.extend(
                {
                    "strategy_name": str(transform.get("kind", "column_only_compaction")),
                    "strategy_kind": "generic",
                    "result": transform,
                    "applied_to_final_answer": False,
                }
                for transform in transforms
            )

    def write_candidate_prediction(candidate: Candidate, *, stage: str) -> tuple[pd.DataFrame, FinalAnswerShapeReport]:
        guarded_frame, guard_report = final_answer_shape_guard(
            task=task,
            frame=candidate.frame,
            contract=answer_contract,
            candidate_source=candidate.source,
        )
        _write_prediction_csv(guarded_frame, prediction_csv_path)
        diagnostics.add_timing(f"{stage}_prediction_write", 0.0)
        return guarded_frame, guard_report

    def write_raw_provisional(frame: pd.DataFrame, *, stage: str) -> None:
        nonlocal provisional_written_stage
        _write_prediction_csv(frame, prediction_csv_path)
        provisional_written_stage = stage
        applied_strategies.append(
            {
                "strategy_name": "raw_provisional_prediction",
                "strategy_kind": "generic",
                "stage": stage,
                "row_count": len(frame),
                "column_count": len(frame.columns),
                "applied_to_final_answer": True,
            }
        )
        diagnostics.add_timing(f"{stage}_prediction_write", 0.0)

    def ask_databao_candidate() -> pd.DataFrame:
        retrieved_token = _CURRENT_RETRIEVED_CONTEXT.set(retrieved_context)
        try:
            return _ask_databao_candidate(
                task,
                databao_env,
                agent_builder,
                diagnostics,
                stage_name="databao_ask",
            )
        finally:
            _CURRENT_RETRIEVED_CONTEXT.reset(retrieved_token)

    try:
        effective_heuristic_level = _heuristic_level()
        enabled_strategies = [
            {
                "strategy_name": "generic_document_tables",
                "strategy_kind": "generic",
                "enabled_by": effective_heuristic_level,
            },
            {
                "strategy_name": "answer_contract",
                "strategy_kind": "generic",
                "enabled_by": "generic",
            },
            {
                "strategy_name": "query_context_retriever",
                "strategy_kind": "generic",
                "enabled_by": "generic",
            },
            {
                "strategy_name": "candidate_ranker",
                "strategy_kind": "generic",
                "enabled_by": "generic",
            },
            {
                "strategy_name": "complexity_profile",
                "strategy_kind": "generic",
                "enabled_by": "generic",
            },
            {
                "strategy_name": "final_answer_shape_guard",
                "strategy_kind": "generic",
                "enabled_by": "generic",
            },
            {
                "strategy_name": "deterministic_postprocess",
                "strategy_kind": "generic",
                "enabled_by": "generic",
            },
        ]

        class _SummaryDomain:
            def add_db(self, *args, **kwargs):
                del args, kwargs

            def add_df(self, *args, **kwargs):
                del args, kwargs

            def add_description(self, *args, **kwargs):
                del args, kwargs

        with diagnostics.stage("context_load"):
            context_summary = register_context_sources(
                _SummaryDomain(),
                task.context_dir,
                heuristic_level=effective_heuristic_level,
            )
            task_context = build_task_context(
                task,
                context_summary=context_summary,
                heuristic_level=effective_heuristic_level,
            )
            context_payload_profile = _context_payload_profile(task_context, context_summary)
            diagnostics.checkpoint("context_loaded", context_payload_profile=context_payload_profile)
            question_features = extract_question_features(task.question, task_context.context_tables)
            task_intent = infer_task_intent(task_context)
            answer_contract = infer_answer_contract(task_context, task_intent)
            retrieved_context = query_context_retriever(task_context, task.question)
            for strategy in context_summary.get("document_materialized_tables", []):
                document_extraction_used = document_extraction_used or int(strategy.get("rows", 0) or 0) > 0
                applied_strategies.append(
                    {
                        "strategy_name": strategy.get("strategy_name", "generic_document_tables"),
                        "strategy_kind": strategy.get("strategy_kind", "generic"),
                        "enabled_by": strategy.get("enabled_by", effective_heuristic_level),
                        "input_document": strategy.get("path"),
                        "extracted_row_count": strategy.get("rows"),
                        "extracted_columns": strategy.get("extracted_columns") or strategy.get("columns"),
                        "confidence": strategy.get("confidence"),
                        "applied_to_final_answer": False,
                    }
                )
            applied_strategies.append(
                {
                    "strategy_name": "question_feature_extraction",
                    "strategy_kind": "generic",
                    "result": question_features.to_dict(),
                    "applied_to_final_answer": False,
                }
            )
            applied_strategies.append(
                {
                    "strategy_name": "query_context_retriever",
                    "strategy_kind": "generic",
                    "selected_table_count": len(retrieved_context.relevant_tables),
                    "document_snippet_count": len(retrieved_context.document_snippets),
                    "applied_to_final_answer": False,
                }
            )

        with _openai_api_key_from_model_env(databao_env):
            route_policy = RouteDecision(
                route="databao",
                reason="Databao-first baseline; experimental rescue/generator branches are not active.",
                confidence=0.0,
            )
            applied_strategies.append(
                {
                    "strategy_name": "route_policy",
                    "strategy_kind": "generic",
                    "route": route_policy.route,
                    "reason": route_policy.reason,
                    "confidence": route_policy.confidence,
                }
            )
            databao_started_at = perf_counter()
            try:
                databao_frame = ask_databao_candidate()
                with diagnostics.stage("raw_provisional_csv_write"):
                    write_raw_provisional(databao_frame, stage="databao_raw_provisional")
            except Exception as exc:
                databao_failure_reason = _safe_exception_summary(exc, (databao_env.model_api_key,))
                databao_failure_type = _databao_failure_type(databao_failure_reason)
                complexity_profile = build_complexity_profile(
                    task_context,
                    retrieved_context,
                    best_candidate=None,
                )
                fallback = _cheap_count_fallback_candidate(task_context, retrieved_context)
                if fallback is not None:
                    fallback_frame, fallback_transform = fallback
                    fallback_candidate = record_candidate(
                        build_candidate_from_frame(
                            fallback_frame,
                            source="cheap_count_fallback",
                            confidence=float(fallback_transform.get("confidence", 0.2) or 0.2),
                            diagnostics_payload={
                                "databao_failure_reason": databao_failure_reason,
                                "fallback": fallback_transform,
                            },
                            transformations=(fallback_transform,),
                        )
                    )
                    if fallback_candidate is not None:
                        applied_strategies.append(
                            {
                                "strategy_name": "cheap_count_fallback",
                                "strategy_kind": "generic",
                                "result": fallback_transform,
                                "applied_to_final_answer": False,
                            }
                        )
                if not candidates:
                    repair_seed = pd.DataFrame({"_databao_failure": []})
                    repair_seed.attrs["databao_salvaged_latest_query_result"] = True
                    context_percentage_frame, context_percentage_transforms = (
                        apply_salvaged_context_boolean_percentage_repair(task, repair_seed)
                    )
                    if context_percentage_transforms and not context_percentage_frame.equals(repair_seed):
                        repair_candidate = record_candidate(
                            build_candidate_from_frame(
                                context_percentage_frame,
                                source="databao_failure_context_percentage_repair",
                                confidence=0.52,
                                diagnostics_payload={
                                    "databao_failure_reason": databao_failure_reason,
                                    "repair_transforms": context_percentage_transforms,
                                },
                                transformations=tuple(context_percentage_transforms),
                            )
                        )
                        if repair_candidate is not None:
                            applied_strategies.extend(
                                {
                                    "strategy_name": str(transform.get("kind", "context_percentage_repair")),
                                    "strategy_kind": "generic",
                                    "result": transform,
                                    "applied_to_final_answer": False,
                                }
                                for transform in context_percentage_transforms
                            )
                if not candidates:
                    repair_seed = pd.DataFrame({"_databao_failure": []})
                    repair_seed.attrs["databao_salvaged_latest_query_result"] = True
                    quoted_ratio_frame, quoted_ratio_transforms = apply_quoted_entity_ratio_repair(task, repair_seed)
                    if quoted_ratio_transforms and not quoted_ratio_frame.equals(repair_seed):
                        repair_candidate = record_candidate(
                            build_candidate_from_frame(
                                quoted_ratio_frame,
                                source="databao_failure_quoted_ratio_repair",
                                confidence=0.5,
                                diagnostics_payload={
                                    "databao_failure_reason": databao_failure_reason,
                                    "repair_transforms": quoted_ratio_transforms,
                                },
                                transformations=tuple(quoted_ratio_transforms),
                            )
                        )
                        if repair_candidate is not None:
                            applied_strategies.extend(
                                {
                                    "strategy_name": str(transform.get("kind", "quoted_ratio_repair")),
                                    "strategy_kind": "generic",
                                    "result": transform,
                                    "applied_to_final_answer": False,
                                }
                                for transform in quoted_ratio_transforms
                            )
                if not candidates:
                    missing_prediction_reason = (
                        "Databao failed and cheap count fallback/context repairs did not produce a dataframe."
                    )
                    raise RuntimeError(
                        f"{missing_prediction_reason} Databao: {databao_failure_reason}"
                    ) from exc
            else:
                databao_candidate = record_candidate(
                    build_candidate_from_frame(
                        databao_frame,
                        source="databao_raw",
                        confidence=0.55,
                        diagnostics_payload={
                            "stage": "databao_ask",
                            **_databao_frame_execution_payload(databao_frame),
                        },
                        elapsed_seconds=round(perf_counter() - databao_started_at, 3),
                    )
                )
                if databao_candidate is not None:
                    add_postprocessed_candidate(databao_candidate)
                    selected_candidate, ranking_report = rank_candidates(
                        candidates=candidates,
                        contract=answer_contract,
                        question=task.question,
                        retrieved_context=retrieved_context,
                    )
                    if selected_candidate is not None:
                        with diagnostics.stage("provisional_csv_write"):
                            _, provisional_guard_report = write_candidate_prediction(
                                selected_candidate,
                                stage="provisional",
                        )
                        final_guard_report = provisional_guard_report
                        add_column_compaction_candidate(selected_candidate)
                        add_verifier_candidates(selected_candidate)

            selected_candidate, ranking_report = rank_candidates(
                candidates=candidates,
                contract=answer_contract,
                question=task.question,
                retrieved_context=retrieved_context,
            )
            complexity_profile = build_complexity_profile(
                task_context,
                retrieved_context,
                best_candidate=selected_candidate,
            )

        selected_candidate, ranking_report = rank_candidates(
            candidates=candidates,
            contract=answer_contract,
            question=task.question,
            retrieved_context=retrieved_context,
        )
        if selected_candidate is None:
            raise ValueError("No candidate dataframe was produced.")

        candidate_source = selected_candidate.source
        frame = selected_candidate.frame
        with diagnostics.stage("answer_contract"):
            answer_contract_report = validate_answer_contract(
                answer_contract,
                frame,
                candidate_source=candidate_source,
            )
            applied_strategies.append(
                {
                    "strategy_name": "answer_contract",
                    "strategy_kind": "generic",
                    "result": answer_contract_report.to_dict(),
                    "applied_to_final_answer": answer_contract_report.valid,
                }
            )

        with diagnostics.stage("final_answer_shape_guard"):
            frame, final_guard_report = final_answer_shape_guard(
                task=task,
                frame=frame,
                contract=answer_contract,
                candidate_source=candidate_source,
            )
            answer_contract_report = validate_answer_contract(
                answer_contract,
                frame,
                candidate_source=candidate_source,
            )
            for transform in final_guard_report.transformations:
                applied_strategies.append(
                    {
                        "strategy_name": str(transform.get("kind", "final_answer_shape_guard")),
                        "strategy_kind": "generic",
                        "result": transform,
                        "applied_to_final_answer": True,
                    }
                )
        with diagnostics.stage("csv_write"):
            _write_prediction_csv(frame, prediction_csv_path)

        elapsed_seconds = round(perf_counter() - started_at, 3)
        diagnostics.timings["total"] = elapsed_seconds
        diagnostics.checkpoint(
            "artifact_ready",
            prediction_written=True,
            row_count=len(frame),
            column_count=len(frame.columns),
            candidate_source=candidate_source,
            elapsed_seconds=elapsed_seconds,
        )
        _write_task_log(
            log_path,
            task=task,
            succeeded=True,
            prediction_written=True,
            scorable=True,
            elapsed_seconds=elapsed_seconds,
            context_summary=context_summary,
            candidate_source=candidate_source,
            postprocessing=postprocess_report_payload,
            timings=diagnostics.timings,
            llm_calls=diagnostics.llm_calls,
            route_policy=route_policy.to_dict() if route_policy else None,
            question_features=question_features.to_dict() if question_features else None,
            answer_contract=answer_contract_report.to_dict() if answer_contract_report else None,
            heuristic_level=effective_heuristic_level,
            enabled_strategies=enabled_strategies,
            applied_strategies=applied_strategies,
            document_extraction_used=document_extraction_used,
            retrieved_context=retrieved_context.to_dict() if retrieved_context else None,
            complexity_profile=complexity_profile.to_dict() if complexity_profile else None,
            candidates=candidate_payloads(),
            candidate_scores=candidate_score_payloads(),
            selected_candidate_source=candidate_source,
            final_answer_guard=final_guard_report.to_dict() if final_guard_report else None,
            provisional_written_stage=provisional_written_stage,
            missing_prediction_reason=missing_prediction_reason,
            active_progress=_read_task_progress(logs_dir, task.task_id),
            timeout_budget=timeout_budget_payload(),
            context_payload_profile=context_payload_profile,
            databao_failure_type=databao_failure_type,
        )
        return DatabaoTaskArtifacts(
            task_id=task.task_id,
            task_output_dir=task_output_dir,
            prediction_csv_path=prediction_csv_path,
            log_path=log_path,
            succeeded=True,
            prediction_written=True,
            scorable=True,
            failure_reason=None,
            elapsed_seconds=elapsed_seconds,
            row_count=len(frame),
            column_count=len(frame.columns),
            candidate_source=candidate_source,
            postprocessing=postprocess_report_payload,
            timings=diagnostics.timings,
            llm_calls=diagnostics.llm_calls,
            route_policy=route_policy.to_dict() if route_policy else None,
            question_features=question_features.to_dict() if question_features else None,
            answer_contract=answer_contract_report.to_dict() if answer_contract_report else None,
            heuristic_level=effective_heuristic_level,
            enabled_strategies=enabled_strategies,
            applied_strategies=applied_strategies,
            retrieved_context=retrieved_context.to_dict() if retrieved_context else None,
            candidates=candidate_payloads(),
            candidate_scores=candidate_score_payloads(),
            selected_candidate_source=candidate_source,
            final_answer_guard=final_guard_report.to_dict() if final_guard_report else None,
            context_payload_profile=context_payload_profile,
            databao_failure_type=databao_failure_type,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed_seconds = round(perf_counter() - started_at, 3)
        diagnostics.timings["total"] = elapsed_seconds
        failure_reason = _safe_exception_summary(exc, (databao_env.model_api_key,))
        _write_task_log(
            log_path,
            task=task,
            succeeded=False,
            prediction_written=_prediction_csv_status(prediction_csv_path)[0],
            scorable=_prediction_csv_status(prediction_csv_path)[1],
            elapsed_seconds=elapsed_seconds,
            context_summary=context_summary,
            candidate_source=candidate_source,
            postprocessing=postprocess_report_payload,
            timings=diagnostics.timings,
            llm_calls=diagnostics.llm_calls,
            route_policy=route_policy.to_dict() if route_policy else None,
            question_features=question_features.to_dict() if question_features else None,
            answer_contract=answer_contract_report.to_dict() if answer_contract_report else None,
            heuristic_level=effective_heuristic_level,
            enabled_strategies=enabled_strategies,
            applied_strategies=applied_strategies,
            document_extraction_used=document_extraction_used,
            retrieved_context=retrieved_context.to_dict() if retrieved_context else None,
            complexity_profile=complexity_profile.to_dict() if complexity_profile else None,
            candidates=candidate_payloads(),
            candidate_scores=candidate_score_payloads(),
            selected_candidate_source=candidate_source,
            final_answer_guard=final_guard_report.to_dict() if final_guard_report else None,
            provisional_written_stage=provisional_written_stage,
            missing_prediction_reason=missing_prediction_reason,
            active_progress=_read_task_progress(logs_dir, task.task_id),
            timeout_budget=timeout_budget_payload(),
            failure_reason=failure_reason,
            context_payload_profile=context_payload_profile,
            databao_failure_type=databao_failure_type or _databao_failure_type(failure_reason),
        )
        traceback_path = logs_dir / f"{task.task_id}.traceback.txt"
        traceback_path.write_text(
            _redact_sensitive_text(traceback.format_exc(), (databao_env.model_api_key,)),
            encoding="utf-8",
            errors="replace",
        )
        prediction_written, scorable, predicted_rows, predicted_columns = _prediction_csv_status(prediction_csv_path)
        return DatabaoTaskArtifacts(
            task_id=task.task_id,
            task_output_dir=task_output_dir,
            prediction_csv_path=prediction_csv_path if prediction_written else None,
            log_path=log_path,
            succeeded=False,
            prediction_written=prediction_written,
            scorable=scorable,
            failure_reason=failure_reason,
            elapsed_seconds=elapsed_seconds,
            row_count=predicted_rows,
            column_count=predicted_columns,
            candidate_source=candidate_source,
            postprocessing=postprocess_report_payload,
            timings=diagnostics.timings,
            llm_calls=diagnostics.llm_calls,
            route_policy=route_policy.to_dict() if route_policy else None,
            question_features=question_features.to_dict() if question_features else None,
            answer_contract=answer_contract_report.to_dict() if answer_contract_report else None,
            heuristic_level=effective_heuristic_level,
            enabled_strategies=enabled_strategies,
            applied_strategies=applied_strategies,
            retrieved_context=retrieved_context.to_dict() if retrieved_context else None,
            complexity_profile=complexity_profile.to_dict() if complexity_profile else None,
            candidates=candidate_payloads(),
            candidate_scores=candidate_score_payloads(),
            selected_candidate_source=candidate_source,
            final_answer_guard=final_guard_report.to_dict() if final_guard_report else None,
            context_payload_profile=context_payload_profile,
            databao_failure_type=databao_failure_type or _databao_failure_type(failure_reason),
        )
    finally:
        _CURRENT_DIAGNOSTICS.reset(diagnostics_token)
        _CURRENT_TASK_DEADLINE.reset(deadline_token)


def create_databao_local_run_dir(output_base: Path, *, run_id: str | None = None) -> tuple[str, Path]:
    effective_run_id = resolve_run_id(run_id) if run_id is not None else create_run_id()
    run_output_dir = output_base / effective_run_id
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return effective_run_id, run_output_dir


def _run_databao_task_child(
    result_queue: multiprocessing.Queue,
    task: PublicTask,
    output_root: Path,
    logs_dir: Path,
    databao_env: DatabaoEnvironment,
    timeout_seconds: int,
) -> None:
    previous_timeout = os.environ.get(DATABAO_TASK_TIMEOUT_SECONDS_ENV)
    os.environ[DATABAO_TASK_TIMEOUT_SECONDS_ENV] = str(timeout_seconds)
    try:
        artifact = run_databao_task(
            task=task,
            output_root=output_root,
            logs_dir=logs_dir,
            databao_env=databao_env,
        )
    finally:
        if previous_timeout is None:
            os.environ.pop(DATABAO_TASK_TIMEOUT_SECONDS_ENV, None)
        else:
            os.environ[DATABAO_TASK_TIMEOUT_SECONDS_ENV] = previous_timeout
    result_queue.put(artifact)


def _run_databao_task_with_timeout(
    *,
    task: PublicTask,
    output_root: Path,
    logs_dir: Path,
    databao_env: DatabaoEnvironment,
    timeout_seconds: int,
    **deprecated_options: Any,
) -> DatabaoTaskArtifacts:
    del deprecated_options
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_run_databao_task_child,
        args=(
            result_queue,
            task,
            output_root,
            logs_dir,
            databao_env,
            timeout_seconds,
        ),
        daemon=False,
    )
    started_at = perf_counter()
    process.start()
    deadline_at = started_at + timeout_seconds
    while True:
        try:
            artifact = result_queue.get(timeout=0.2)
        except queue.Empty:
            artifact = None
        if artifact is not None:
            process.join(2)
            if process.is_alive():
                process.terminate()
                process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join(5)
            return artifact

        elapsed_seconds = perf_counter() - started_at
        if not process.is_alive():
            break
        if perf_counter() >= deadline_at:
            process.terminate()
            process.join(10)
            if process.is_alive():
                process.kill()
                process.join(10)
            return _timeout_task_artifact(
                task=task,
                output_root=output_root,
                logs_dir=logs_dir,
                elapsed_seconds=elapsed_seconds,
                timeout_seconds=timeout_seconds,
            )

    try:
        return result_queue.get_nowait()
    except queue.Empty:
        failure_reason = f"RuntimeError: task process exited with code {process.exitcode} without an artifact."
        elapsed_seconds = round(elapsed_seconds, 3)
        prediction_path = output_root / task.task_id / "prediction.csv"
        prediction_written, scorable, row_count, column_count = _prediction_csv_status(prediction_path)
        progress = _read_task_progress(logs_dir, task.task_id)
        _write_task_log(
            logs_dir / f"{task.task_id}.json",
            task=task,
            succeeded=False,
            prediction_written=prediction_written,
            scorable=scorable,
            elapsed_seconds=elapsed_seconds,
            timings={"total": elapsed_seconds},
            timeout_budget={"active_progress": progress},
            missing_prediction_reason=(
                None
                if prediction_written
                else "task process exited before prediction.csv was written"
            ),
            active_progress=progress,
            failure_reason=failure_reason,
        )
        return DatabaoTaskArtifacts(
            task_id=task.task_id,
            task_output_dir=output_root / task.task_id,
            prediction_csv_path=prediction_path if prediction_written else None,
            log_path=logs_dir / f"{task.task_id}.json",
            succeeded=False,
            prediction_written=prediction_written,
            scorable=scorable,
            failure_reason=failure_reason,
            elapsed_seconds=elapsed_seconds,
            row_count=row_count,
            column_count=column_count,
            timings={"total": elapsed_seconds},
        )


def _can_run_task_in_child_process(
    *,
    agent_builder: AgentBuilder,
    answer_postprocessor: AnswerPostprocessor,
) -> bool:
    return (
        agent_builder is build_databao_agent
        and answer_postprocessor is postprocess_answer_table
    )


def run_databao_tasks(
    *,
    input_root: Path,
    output_root: Path,
    logs_dir: Path,
    limit: int | None = None,
    task_ids: list[str] | None = None,
    difficulty: str | None = None,
    task_timeout_seconds: int | None = None,
    databao_env: DatabaoEnvironment | None = None,
    agent_builder: AgentBuilder = build_databao_agent,
    answer_postprocessor: AnswerPostprocessor = postprocess_answer_table,
    **deprecated_options: Any,
) -> list[DatabaoTaskArtifacts]:
    del deprecated_options
    effective_env = databao_env or load_databao_environment()
    dataset = DABenchPublicDataset(input_root)
    tasks = dataset.iter_tasks(task_ids=task_ids, difficulty=difficulty)
    if limit is not None:
        tasks = tasks[:limit]

    timeout_seconds = task_timeout_seconds
    if timeout_seconds is None:
        timeout_seconds = _env_int(DATABAO_TASK_TIMEOUT_SECONDS_ENV, 0)
    use_child_process = timeout_seconds > 0 and _can_run_task_in_child_process(
        agent_builder=agent_builder,
        answer_postprocessor=answer_postprocessor,
    )

    artifacts: list[DatabaoTaskArtifacts] = []
    for task in tasks:
        if use_child_process:
            artifact = _run_databao_task_with_timeout(
                task=task,
                output_root=output_root,
                logs_dir=logs_dir,
                databao_env=effective_env,
                timeout_seconds=timeout_seconds,
            )
        else:
            artifact = run_databao_task(
                task=task,
                output_root=output_root,
                logs_dir=logs_dir,
                databao_env=effective_env,
                agent_builder=agent_builder,
                answer_postprocessor=answer_postprocessor,
            )
        artifacts.append(artifact)

    summary_path = output_root / "summary.json"
    attempted_task_count = len(artifacts)
    succeeded_task_count = sum(1 for artifact in artifacts if artifact.succeeded)
    prediction_written_count = sum(1 for artifact in artifacts if artifact.prediction_written)
    scorable_task_count = sum(1 for artifact in artifacts if artifact.scorable)
    summary_path.write_text(
        json.dumps(
            {
                "task_count": attempted_task_count,
                "attempted_task_count": attempted_task_count,
                "succeeded_task_count": succeeded_task_count,
                "failed_task_count": attempted_task_count - succeeded_task_count,
                "prediction_written_count": prediction_written_count,
                "scorable_task_count": scorable_task_count,
                "missing_prediction_count": attempted_task_count - prediction_written_count,
                "tasks": [artifact.to_dict() for artifact in artifacts],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return artifacts
