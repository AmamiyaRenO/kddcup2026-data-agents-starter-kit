import json
import os
import re
from typing import Annotated, Any, Literal

import pandas as pd
from duckdb import DuckDBPyConnection
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.constants import END, START
from langgraph.graph import add_messages
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.prebuilt import InjectedState
from typing_extensions import TypedDict

from databao.agent.configs import llm
from databao.agent.configs.agent import AgentConfig
from databao.agent.configs.llm import LLMConfig
from databao.agent.core import Domain, ExecutionResult
from databao.agent.executors.langchain_tools import make_search_context_tool
from databao.agent.executors.llm import chat, model_bind_tools
from databao.agent.executors.utils import exception_to_string
from databao.agent.executors.utils import run_sql_query as _run_sql_query

RUN_SQL_QUERY_TOOL_DESCRIPTION = """\
Run a SELECT SQL query in the database. Returns the first 12 rows in csv format.

Args:
    sql: SQL query
"""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query_ids: dict[str, ToolMessage]
    sql: str | None
    df: pd.DataFrame | None
    last_non_empty_query_id: str | None
    last_non_empty_sql: str | None
    last_non_empty_df: pd.DataFrame | None
    submit_critiques: list[dict[str, Any]]
    visualization_prompt: str | None
    ready_for_user: bool
    limit_max_rows: int | None
    no_submit_retry_count: int


def get_query_ids_mapping(messages: list[BaseMessage]) -> dict[str, ToolMessage]:
    query_ids = {}
    for message in messages:
        if isinstance(message, ToolMessage) and isinstance(message.artifact, dict) and "query_id" in message.artifact:
            query_ids[message.artifact["query_id"]] = message
    return query_ids


def _submit_critique_mode() -> str:
    value = os.environ.get("DATABAO_INTERNAL_SUBMIT_CRITIQUE_MODE", "shadow").strip().lower()
    return value if value in {"off", "shadow", "reject"} else "shadow"


def _grounding_reject_enabled() -> bool:
    return os.environ.get("DATABAO_INTERNAL_GROUNDING_REJECT", "0").strip().lower() in {"1", "true", "yes", "on"}


def _anchored_lookup_reject_enabled() -> bool:
    return os.environ.get("DATABAO_INTERNAL_ANCHORED_LOOKUP_REJECT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _finality_salvage_profile_enabled() -> bool:
    return os.environ.get("DATABAO_FINALITY_SALVAGE_PROFILE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _soft_p0_profile_enabled() -> bool:
    return os.environ.get("DATABAO_SOFT_P0_PROFILE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _p0_gates_enabled() -> bool:
    value = os.environ.get("DATABAO_INTERNAL_P0_GATES", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _p0_max_rejections() -> int:
    raw = os.environ.get("DATABAO_INTERNAL_P0_MAX_REJECTIONS", "3").strip()
    try:
        value = int(raw)
    except ValueError:
        return 3
    return max(0, value)


def _no_submit_finality_retry_enabled() -> bool:
    value = os.environ.get("DATABAO_INTERNAL_NO_SUBMIT_FINALITY_RETRY", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _no_submit_finality_max_retries() -> int:
    default = 2 if _finality_salvage_profile_enabled() else 1
    raw = os.environ.get("DATABAO_INTERNAL_NO_SUBMIT_FINALITY_MAX_RETRIES")
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        return default
    return max(0, value)


P0_REJECT_FLAGS = frozenset(
    {
        "empty_submit_result",
        "numeric_identifier_filter_needs_grounding",
        "blank_display_needs_join",
        "time_literal_precision_needs_rounding",
        "whole_second_time_limit_may_drop_matches",
    }
)


def _p0_reject_flags() -> frozenset[str]:
    if _soft_p0_profile_enabled():
        return frozenset({"empty_submit_result"})
    return P0_REJECT_FLAGS


def _latest_user_question(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = str(message.content)
            if content.lstrip().startswith("You ended without calling submit_result."):
                continue
            # Strip system instructions; the actual question is the trailing
            # text after the last "Question:" marker. The prefix often contains
            # words like "display", "name", "title", "ratio" that would otherwise
            # spuriously trigger question-keyword critiques.
            marker = "Question:"
            idx = content.rfind(marker)
            if idx >= 0:
                return content[idx + len(marker):].strip()
            return content
    return ""


def _sql_operations(sql: str | None) -> dict[str, bool]:
    compact = re.sub(r"\s+", " ", (sql or "").lower()).strip()
    return {
        "filter": bool(re.search(r"\bwhere\b|\bhaving\b", compact)),
        "groupby": bool(re.search(r"\bgroup\s+by\b", compact)),
        "aggregate": bool(re.search(r"\b(?:count|sum|avg|average|min|max)\s*\(", compact)),
        "ratio": "/" in compact and bool(re.search(r"\b(?:count|sum|avg|average)\s*\(", compact)),
        "order_by": bool(re.search(r"\border\s+by\b", compact)),
        "limit": bool(re.search(r"\blimit\s+\d+\b", compact)),
    }


def _sql_filter_equalities(sql: str | None) -> list[dict[str, str]]:
    text = sql or ""
    if not text:
        return []
    equalities: list[dict[str, str]] = []
    pattern = re.compile(
        r"(?P<column>(?:[A-Za-z_][A-Za-z0-9_]*\.)?[`\"]?[A-Za-z_][A-Za-z0-9_]*[`\"]?)\s*=\s*"
        r"(?P<value>'[^']*'|\"[^\"]*\"|-?\d+(?:\.\d+)?)",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        column = match.group("column").strip("`\"")
        value = match.group("value").strip()
        if "." in column:
            column = column.split(".")[-1].strip("`\"")
        equalities.append(
            {
                "column": column,
                "value": value.strip("'\""),
                "raw_value": value,
                "value_kind": "numeric" if re.fullmatch(r"-?\d+(?:\.\d+)?", value) else "text",
            }
        )
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for equality in equalities:
        key = (equality["column"].lower(), equality["value"], equality["value_kind"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(equality)
    return deduped


def _question_numeric_groundings(question: str) -> set[str]:
    lowered = question.lower()
    numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", lowered))
    ordinal_values = {
        "first": "1",
        "winner": "1",
        "champion": "1",
        "top": "1",
        "second": "2",
        "runner up": "2",
        "runner-up": "2",
        "third": "3",
        "fourth": "4",
        "fifth": "5",
        "sixth": "6",
        "seventh": "7",
        "eighth": "8",
        "ninth": "9",
        "tenth": "10",
    }
    for phrase, value in ordinal_values.items():
        if re.search(rf"\b{re.escape(phrase)}\b", lowered):
            numbers.add(value)
    ordinal_match = re.findall(r"\b(\d+)(?:st|nd|rd|th)\b", lowered)
    numbers.update(ordinal_match)
    return numbers


def _grounding_flags(question: str, sql: str | None) -> tuple[list[str], list[str], list[dict[str, str]]]:
    equalities = _sql_filter_equalities(sql)
    if not equalities:
        return [], [], []
    question_numbers = _question_numeric_groundings(question)
    flags: list[str] = []
    suggestions: list[str] = []
    evidence: list[dict[str, str]] = []
    metric_terms = (
        "amount",
        "cost",
        "price",
        "score",
        "points",
        "weight",
        "height",
        "time",
        "date",
        "year",
        "month",
        "day",
    )
    for equality in equalities:
        column = equality["column"]
        value = equality["value"]
        value_kind = equality["value_kind"]
        lowered_column = column.lower()
        if value_kind != "numeric":
            continue
        normalized_value = value[:-2] if value.endswith(".0") else value
        if normalized_value in question_numbers:
            continue
        if lowered_column.endswith("id") or lowered_column == "id":
            flags.append("numeric_identifier_filter_needs_grounding")
            suggestions.append(
                f"Validate `{column} = {value}` by joining or inspecting the table that maps IDs to names/descriptions from the question."
            )
            evidence.append(equality)
        elif not any(term in lowered_column for term in metric_terms):
            flags.append("numeric_code_filter_needs_value_grounding")
            suggestions.append(
                f"Validate `{column} = {value}` against distinct values or field definitions before submitting."
            )
            evidence.append(equality)
    return list(dict.fromkeys(flags)), list(dict.fromkeys(suggestions)), evidence


def _sql_like_filters(sql: str | None) -> list[dict[str, str]]:
    text = sql or ""
    if not text:
        return []
    pattern = re.compile(
        r"(?:lower\s*\(\s*)?"
        r"(?P<column>(?:[A-Za-z_][A-Za-z0-9_]*\.)?[`\"]?[A-Za-z_][A-Za-z0-9_]*[`\"]?)"
        r"\s*\)?\s+like\s+"
        r"(?P<quote>['\"])(?P<pattern>.*?)(?P=quote)",
        flags=re.IGNORECASE,
    )
    filters: list[dict[str, str]] = []
    for match in pattern.finditer(text):
        column = match.group("column").strip("`\"")
        if "." in column:
            column = column.split(".")[-1].strip("`\"")
        raw_pattern = match.group("pattern")
        term = raw_pattern.strip("%").strip().lower()
        if term:
            filters.append({"column": column, "term": term, "pattern": raw_pattern})
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in filters:
        key = (item["column"].lower(), item["term"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _anchored_lookup_flags(
    question: str,
    sql: str | None,
    df: pd.DataFrame | None,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    flags: list[str] = []
    suggestions: list[str] = []
    evidence: list[dict[str, Any]] = []
    lowered_question = question.lower()
    columns = [str(column) for column in df.columns] if isinstance(df, pd.DataFrame) else []
    row_count = int(len(df)) if isinstance(df, pd.DataFrame) else 0

    like_filters = _sql_like_filters(sql)
    filters_by_column: dict[str, list[str]] = {}
    for item in like_filters:
        filters_by_column.setdefault(item["column"], []).append(item["term"])
    sql_has_or = bool(re.search(r"\bor\b", (sql or ""), flags=re.IGNORECASE))
    question_looks_conjunctive = bool(
        "," in question
        or re.search(r"\b(?:and|with|include|including|contains|containing)\b", lowered_question)
    )
    for column, terms in filters_by_column.items():
        question_terms = [term for term in terms if term and term in lowered_question]
        if len(question_terms) >= 2 and sql_has_or and question_looks_conjunctive and row_count > 1:
            flags.append("disjunctive_text_filter_needs_anchored_match")
            suggestions.append(
                f"The question appears to require one record matching multiple item terms. Anchor on rows where `{column}` matches all requested terms, not any single term."
            )
            evidence.append({"kind": "text_filter", "column": column, "terms": question_terms, "row_count": row_count})

    link_columns = [column for column in columns if column.lower().startswith("link_to_")]
    has_link_identifier = bool(link_columns)
    has_display_column = any(
        re.search(r"(?:^|_)(?:name|title|label|display_name|full_name)(?:$|_)", column, flags=re.IGNORECASE)
        for column in columns
    )
    asks_display_name = bool(re.search(r"\b(?:who|name|full name|member|person|individual)\b", lowered_question))
    if has_link_identifier and asks_display_name and not has_display_column:
        link_samples: dict[str, list[str]] = {}
        if isinstance(df, pd.DataFrame):
            for column in link_columns[:3]:
                values = [str(value) for value in df[column].dropna().astype(str).unique().tolist()[:5]]
                if values:
                    link_samples[column] = values
        flags.append("linked_identifier_needs_display_lookup")
        sample_hint = ""
        if link_samples:
            sample_hint = f" Link values to resolve: {link_samples}."
        suggestions.append(
            "The result still contains a linked identifier; resolve link_to_* or *_id through the related table/document and submit the display/name value."
            + sample_hint
        )
        evidence.append({"kind": "link_resolution", "columns": columns[:20], "link_samples": link_samples})

    return list(dict.fromkeys(flags)), list(dict.fromkeys(suggestions)), evidence


def _date_filter_qualified_columns(sql: str | None) -> list[str]:
    """Return qualified column references that appear in a date-like comparison.

    Used by the empty-result probe to generate ``SELECT DISTINCT <col>`` queries
    against the same FROM clause Databao tried, so the model can see the real
    on-disk date format instead of having to guess at the literal shape.
    """

    if not sql:
        return []
    pattern = re.compile(
        r"((?:[A-Za-z_]\w*\.)?[`\"]?\w*(?:date|time|day|month|year)\w*[`\"]?)"
        r"\s*(?:[<>=!]+|between|like|in\s*\()",
        flags=re.IGNORECASE,
    )
    deduped: list[str] = []
    seen: set[str] = set()
    for match in pattern.finditer(sql):
        column = match.group(1).strip("`\"")
        key = column.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(column)
    return deduped


def _extract_from_clause(sql: str | None) -> str | None:
    if not sql:
        return None
    match = re.search(
        r"\bfrom\b(.+?)(?:\bwhere\b|\bgroup\s+by\b|\border\s+by\b|\blimit\b|;|$)",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    return match.group(1).strip()


def _build_date_probe_sql(original_sql: str, qualified_col: str) -> str | None:
    from_clause = _extract_from_clause(original_sql)
    if not from_clause:
        return None
    return f"SELECT DISTINCT {qualified_col} FROM {from_clause} LIMIT 5"


def _empty_date_filter_hint(sql: str | None) -> str | None:
    if not sql:
        return None
    lowered = sql.lower()
    if not re.search(
        r"\b(?:date|datetime|created_at|updated_at|posted_at|day|month|year|\w*date\w*|\w*time\w*)\b\s*"
        r"(?:[<>=!]+|between|in\s*\()",
        lowered,
    ):
        return None
    return (
        " Before re-submitting an empty result, inspect the filtered column's dtype and a few distinct "
        "sample values via a separate SELECT; the literal format may not match (string '2013-06-01' vs "
        "integer 20130601 vs other format)."
    )


def _blank_display_critique(
    question: str,
    df: pd.DataFrame | None,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    flags: list[str] = []
    suggestions: list[str] = []
    evidence: list[dict[str, Any]] = []
    if not isinstance(df, pd.DataFrame) or df.empty:
        return flags, suggestions, evidence
    lowered = question.lower()
    asks_display = bool(
        re.search(
            r"\b(?:who|name|names|full\s+name|display|title|user|member|person|individual|"
            r"posted\s+by|author|owner|customer)\b",
            lowered,
        )
    )
    if not asks_display:
        return flags, suggestions, evidence
    columns = [str(column) for column in df.columns]
    question_tokens = set(re.findall(r"[a-z0-9]+", lowered))
    requested_attribute_terms = {
        "number",
        "phone",
        "telephone",
        "url",
        "link",
        "website",
        "score",
        "cost",
        "amount",
        "price",
        "value",
        "count",
        "total",
        "date",
        "time",
        "text",
        "comment",
        "status",
        "type",
        "category",
    }
    for column in columns:
        split_column = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", column).lower()
        aliases = {part for part in re.split(r"[^a-z0-9]+", split_column) if part}
        aliases.add(re.sub(r"[^a-z0-9]+", "", split_column))
        if aliases & question_tokens & requested_attribute_terms:
            return flags, suggestions, evidence
    display_pat = re.compile(
        r"(?:^|_|(?<=[a-z]))(?:name|title|label|display|displayname|fullname|username|ownername|"
        r"firstname|lastname)(?:$|_|(?=[A-Z]))",
        flags=re.IGNORECASE,
    )
    id_pat = re.compile(
        r"(?:^|_|(?<=[a-z]))(?:id|userid|ownerid|authorid|editorid|"
        r"user_id|owner_id|author_id|editor_id|posted_by|postedby)(?:$|_|(?=[A-Z]))"
        r"|link_to_\w+",
        flags=re.IGNORECASE,
    )
    display_cols = [column for column in columns if display_pat.search(column)]
    id_cols = [
        column for column in columns if id_pat.search(column) and column not in display_cols
    ]

    blank_display_cols: list[str] = []
    for column in display_cols:
        try:
            series = df[column]
            sample = series.dropna().astype(str).str.strip()
            non_blank = sample[sample != ""]
            if len(non_blank) == 0:
                blank_display_cols.append(column)
        except Exception:
            continue

    def _id_samples() -> dict[str, list[str]]:
        samples: dict[str, list[str]] = {}
        for column in id_cols[:3]:
            try:
                values = [
                    str(value)
                    for value in df[column].dropna().astype(str).unique().tolist()[:5]
                ]
                if values:
                    samples[column] = values
            except Exception:
                continue
        return samples

    if blank_display_cols:
        id_samples = _id_samples()
        flags.append("blank_display_needs_join")
        sample_hint = (
            f" Identifier values available to resolve through a join: {id_samples}."
            if id_samples
            else ""
        )
        suggestions.append(
            f"The submitted result has blank display column(s) {blank_display_cols}; the question asks for a display/name. "
            f"Join the related name/user/title table to resolve the display value before submitting."
            + sample_hint
        )
        evidence.append(
            {
                "kind": "blank_display",
                "blank_columns": blank_display_cols,
                "id_columns": id_cols[:10],
            }
        )
    elif not display_cols and id_cols:
        id_samples = _id_samples()
        flags.append("blank_display_needs_join")
        suggestions.append(
            f"The question asks for a display/name but the result only contains identifier column(s) {id_cols[:5]}. "
            f"Join the lookup table to resolve a display/name column. Identifier samples: {id_samples}."
        )
        evidence.append(
            {
                "kind": "id_only_no_display",
                "id_columns": id_cols[:10],
                "id_samples": id_samples,
            }
        )
    return flags, suggestions, evidence


def _unit_time_granularity_critique(
    question: str,
    sql: str | None,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    flags: list[str] = []
    suggestions: list[str] = []
    evidence: list[dict[str, Any]] = []
    if not question or not sql:
        return flags, suggestions, evidence
    lowered_q = question.lower()
    lowered_sql = sql.lower()
    asks_monthly = bool(
        re.search(r"\b(?:per\s+month|monthly|each\s+month|average\s+monthly|by\s+month)\b", lowered_q)
    )
    asks_annual = bool(
        re.search(r"\b(?:per\s+year|annual|annually|each\s+year|yearly|by\s+year)\b", lowered_q)
    )
    sql_year_field = bool(re.search(r"\b\w*year\w*\b", lowered_sql))
    sql_month_field = bool(re.search(r"\b\w*month\w*\b", lowered_sql))
    sql_divide_12 = bool(re.search(r"/\s*12\b", lowered_sql))
    sql_times_12 = bool(re.search(r"\*\s*12\b", lowered_sql))
    question_whole_second_times = re.findall(r"\b(?:\d{1,2}:)?\d{1,2}:\d{2}\b(?!\.\d)", lowered_q)
    sql_fractional_equality_times = re.findall(
        r"(?:=|in\s*\()\s*['\"]((?:\d{1,2}:)?\d{1,2}:\d{2}\.\d+)['\"]",
        lowered_sql,
    )
    sql_has_limit = bool(re.search(r"\blimit\s+\d+\b", lowered_sql))
    sql_has_time_prefix_filter = bool(
        re.search(r"\b(?:like|ilike)\s*['\"][^'\"]*(?:\d{1,2}:)?\d{1,2}:\d{2}%['\"]", lowered_sql)
    )
    question_asks_single_ordered_row = bool(
        re.search(
            r"\b(?:first|last|top\s+1|highest|lowest|min(?:imum)?|max(?:imum)?|fastest|slowest|earliest|latest|ranked)\b",
            lowered_q,
        )
    )

    if asks_monthly and sql_year_field and not sql_month_field and not sql_divide_12:
        flags.append("unit_time_granularity_mismatch")
        suggestions.append(
            "The question asks for a monthly value but the SQL uses a year-level field without dividing by 12. "
            "Use a month-level column if available, or convert the annual value to monthly explicitly."
        )
        evidence.append({"kind": "monthly_vs_annual"})
    if asks_annual and sql_month_field and not sql_year_field and not sql_times_12:
        flags.append("unit_time_granularity_mismatch")
        suggestions.append(
            "The question asks for an annual value but the SQL uses a month-level field without aggregating to year. "
            "Sum the monthly values across the year or use an annual column."
        )
        evidence.append({"kind": "annual_vs_monthly"})
    if question_whole_second_times and sql_fractional_equality_times:
        flags.append("time_literal_precision_needs_rounding")
        suggestions.append(
            "The question gives a time only to whole-second precision, but the SQL uses equality against a "
            "fractional-second literal. Match the whole second with a prefix/range/truncation condition and "
            "include all rows in that second before submitting."
        )
        evidence.append(
            {
                "kind": "whole_second_question_vs_fractional_sql_equality",
                "question_times": question_whole_second_times[:5],
                "sql_times": sql_fractional_equality_times[:5],
            }
        )
    if question_whole_second_times and sql_has_time_prefix_filter and sql_has_limit and not question_asks_single_ordered_row:
        flags.append("whole_second_time_limit_may_drop_matches")
        suggestions.append(
            "The question gives a time only to whole-second precision and the SQL matches that whole second, "
            "but LIMIT may drop other rows in the same second. Remove the LIMIT unless the question explicitly "
            "asks for a first/top/ranked row."
        )
        evidence.append(
            {
                "kind": "whole_second_prefix_filter_with_limit",
                "question_times": question_whole_second_times[:5],
            }
        )
    return flags, suggestions, evidence


def _filter_audit_critique(
    question: str,
    sql: str | None,
    df: pd.DataFrame | None,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    flags: list[str] = []
    suggestions: list[str] = []
    evidence: list[dict[str, Any]] = []
    if not question or not sql or not isinstance(df, pd.DataFrame):
        return flags, suggestions, evidence
    row_count = int(len(df))
    if row_count <= 12:
        return flags, suggestions, evidence
    lowered_q = question.lower()
    constraint_phrases = re.findall(
        r"\b(?:more than|greater than|less than|fewer than|at least|at most|over|under|above|below|"
        r"between|after|before|equal to|equals)\s+[^,;.?]{1,60}",
        lowered_q,
    )
    if len(constraint_phrases) < 2:
        return flags, suggestions, evidence
    sql_clause_count = len(re.findall(r"\b(?:where|and|or)\b", (sql or "").lower()))
    if sql_clause_count < len(constraint_phrases):
        flags.append("filter_audit_missing_constraint")
        suggestions.append(
            f"The question lists {len(constraint_phrases)} constraint phrase(s) but the SQL only has "
            f"{sql_clause_count} WHERE/AND/OR clause(s). Audit which constraints are translated to SQL "
            f"and which remain unaccounted for before submitting detail rows."
        )
        evidence.append(
            {
                "kind": "constraint_count_mismatch",
                "question_constraints": constraint_phrases[:6],
                "where_count": sql_clause_count,
                "row_count": row_count,
            }
        )
    return flags, suggestions, evidence


def _critique_sample_rows(df: pd.DataFrame | None, *, max_rows: int = 2, max_cell_chars: int = 80) -> list[dict[str, Any]]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    samples: list[dict[str, Any]] = []
    for record in df.head(max_rows).to_dict(orient="records"):
        cleaned: dict[str, Any] = {}
        for key, value in record.items():
            try:
                is_missing = bool(pd.isna(value))
            except (TypeError, ValueError):
                is_missing = False
            if is_missing:
                cleaned[str(key)] = None
                continue
            text = str(value)
            if len(text) > max_cell_chars:
                text = text[: max_cell_chars - 1] + "…"
            cleaned[str(key)] = text
        samples.append(cleaned)
    return samples


def _critique_submit_result(
    messages: list[BaseMessage],
    query_id: str,
    sql: str | None,
    df: pd.DataFrame | None,
) -> dict[str, Any]:
    question = _latest_user_question(messages).lower()
    operations = _sql_operations(sql)
    flags: list[str] = []
    suggestions: list[str] = []
    row_count = int(len(df)) if isinstance(df, pd.DataFrame) else 0
    column_count = int(len(df.columns)) if isinstance(df, pd.DataFrame) else 0
    columns = [str(column) for column in df.columns] if isinstance(df, pd.DataFrame) else []

    asks_ratio = bool(re.search(r"\b(?:ratio|percentage|percent|proportion|share)\b", question))
    asks_aggregate = bool(re.search(r"\b(?:how many|count|average|avg|mean|sum|total)\b", question))
    asks_count = bool(re.search(r"\b(?:how many|count|number of)\b", question)) and not asks_ratio
    asks_list = bool(re.search(r"\b(?:which|list|all|records|names)\b", question))

    if df is None or df.empty:
        flags.append("empty_submit_result")
        date_hint = _empty_date_filter_hint(sql) or ""
        suggestions.append(
            "The submitted query is empty; revise filters or submit a previous non-empty query." + date_hint
        )
    if asks_ratio and not operations["ratio"]:
        numeric_columns = [
            column
            for column in columns
            if isinstance(df, pd.DataFrame) and pd.api.types.is_numeric_dtype(df[column])
        ]
        ratio_columns = [
            column
            for column in columns
            if re.search(r"(ratio|percentage|percent|proportion|share)", column, flags=re.IGNORECASE)
        ]
        if not ratio_columns or len(numeric_columns) > 1:
            flags.append("ratio_question_without_ratio_evidence")
            suggestions.append(
                "The question asks for a ratio/percentage; compute the numerator, denominator, and final ratio/percentage before submitting."
            )
    if asks_aggregate and row_count > 12 and not asks_list and not operations["aggregate"]:
        flags.append("aggregate_question_submitted_detail_rows")
        suggestions.append("The question asks for an aggregate; submit the aggregate result, not detail rows.")
    if asks_count and isinstance(df, pd.DataFrame) and not df.empty:
        count_columns = [
            column
            for column in columns
            if re.search(r"(count|number|total)", column, flags=re.IGNORECASE)
        ]
        if not count_columns:
            flags.append("count_question_without_count_result")
            suggestions.append("The question asks for a count; submit a count result rather than other statistics.")
    if operations["limit"] and not operations["order_by"]:
        flags.append("limit_without_order_by")
        suggestions.append("A limited result should include ORDER BY or another reason why these rows are final.")
    if column_count > 8 and not asks_list:
        flags.append("wide_submit_result")
        suggestions.append("The submitted table is wide; keep only columns needed to answer the question.")
    grounding_flags, grounding_suggestions, grounding_evidence = _grounding_flags(question, sql)
    flags.extend(grounding_flags)
    suggestions.extend(grounding_suggestions)
    anchored_flags, anchored_suggestions, anchored_evidence = _anchored_lookup_flags(question, sql, df)
    flags.extend(anchored_flags)
    suggestions.extend(anchored_suggestions)
    blank_display_flags, blank_display_suggestions, blank_display_evidence = _blank_display_critique(
        question, df
    )
    flags.extend(blank_display_flags)
    suggestions.extend(blank_display_suggestions)
    unit_flags, unit_suggestions, unit_evidence = _unit_time_granularity_critique(question, sql)
    flags.extend(unit_flags)
    suggestions.extend(unit_suggestions)
    filter_audit_flags, filter_audit_suggestions, filter_audit_evidence = _filter_audit_critique(
        question, sql, df
    )
    flags.extend(filter_audit_flags)
    suggestions.extend(filter_audit_suggestions)

    should_reject = any(
        flag in {
            "empty_submit_result",
            "ratio_question_without_ratio_evidence",
            "aggregate_question_submitted_detail_rows",
            "count_question_without_count_result",
        }
        for flag in flags
    )
    if _grounding_reject_enabled() and any(
        flag in {
            "numeric_identifier_filter_needs_grounding",
            "numeric_code_filter_needs_value_grounding",
        }
        for flag in flags
    ):
        should_reject = True
    if _anchored_lookup_reject_enabled() and any(
        flag in {
            "disjunctive_text_filter_needs_anchored_match",
            "linked_identifier_needs_display_lookup",
        }
        for flag in flags
    ):
        should_reject = True

    active_p0_reject_flags = _p0_reject_flags()
    p0_flags = [flag for flag in flags if flag in active_p0_reject_flags]
    p0_should_reject = bool(p0_flags) and _p0_gates_enabled()

    return {
        "query_id": query_id,
        "mode": _submit_critique_mode(),
        "flags": flags,
        "suggestions": suggestions,
        "should_reject": should_reject,
        "p0_flags": p0_flags,
        "p0_should_reject": p0_should_reject,
        "p0_reject_flags": sorted(active_p0_reject_flags),
        "p0_gates_enabled": _p0_gates_enabled(),
        "finality_salvage_profile_enabled": _finality_salvage_profile_enabled(),
        "soft_p0_profile_enabled": _soft_p0_profile_enabled(),
        "row_count": row_count,
        "column_count": column_count,
        "columns": columns[:20],
        "sample_rows": _critique_sample_rows(df),
        "sql_operations": operations,
        "sql_text": (sql or "")[:1000],
        "question_text": question[:300],
        "grounding_evidence": grounding_evidence,
        "grounding_reject_enabled": _grounding_reject_enabled(),
        "anchored_lookup_evidence": anchored_evidence,
        "anchored_lookup_reject_enabled": _anchored_lookup_reject_enabled(),
        "blank_display_evidence": blank_display_evidence,
        "unit_time_granularity_evidence": unit_evidence,
        "filter_audit_evidence": filter_audit_evidence,
    }


def _submit_critique_message(critique: dict[str, Any]) -> str:
    suggestions = critique.get("suggestions") or ["Revise the query result before submitting."]
    return "Submit critique: " + " ".join(str(item) for item in suggestions)


def _no_submit_result_context(critique: dict[str, Any]) -> str:
    parts: list[str] = []
    if critique.get("salvaged_previous_non_empty_result"):
        parts.append("The latest non-empty query result is being used because the most recent result was empty.")
    row_count = int(critique.get("row_count") or 0)
    column_count = int(critique.get("column_count") or 0)
    parts.append(f"Latest result shape: {row_count} rows x {column_count} columns.")
    columns = [str(column) for column in (critique.get("columns") or [])[:12]]
    if columns:
        parts.append("Latest result columns: " + ", ".join(f"`{column}`" for column in columns) + ".")
    sql_text = " ".join(str(critique.get("sql_text") or "").split())
    if sql_text:
        if len(sql_text) > 500:
            sql_text = sql_text[:499] + "…"
        parts.append(f"Latest SQL: {sql_text}")
    sample_rows = critique.get("sample_rows") or []
    if sample_rows:
        sample_text = json.dumps(sample_rows, ensure_ascii=False)
        if len(sample_text) > 500:
            sample_text = sample_text[:499] + "…"
        parts.append(f"Sample rows: {sample_text}")
        parts.append("Sample rows are only a preview; submit_result returns the full query result.")
    return " ".join(parts)


def _no_submit_finality_feedback_message(critique: dict[str, Any]) -> str:
    query_id = str(critique.get("query_id") or "latest")
    flags = ", ".join(str(flag) for flag in critique.get("flags", [])[:8])
    result_context = _no_submit_result_context(critique)
    if _finality_salvage_profile_enabled():
        instruction = (
            "If the latest non-empty query result plausibly answers the question, prefer submitting it now "
            "rather than restarting broad exploration. Only run one corrected SELECT query if the latest result "
            "is empty or clearly the wrong table or shape. Then call submit_result. Do not answer in prose."
        )
    else:
        instruction = (
            "If the latest query already answers the question, call submit_result with that query_id now. "
            "Otherwise run one corrected SELECT query, then call submit_result. Do not answer in prose."
        )
    return (
        "You ended without calling submit_result. "
        f"Latest query_id is `{query_id}`. "
        f"Diagnostic flags: {flags or 'no_submit_result'}. "
        f"{result_context} "
        f"{instruction}"
    )


def _state_no_submit_candidate(state: dict[str, Any]) -> tuple[str, str, pd.DataFrame | None, bool]:
    sql = state.get("sql") or ""
    df = state.get("df")
    query_id = "latest"
    salvaged_previous_non_empty = False
    if (df is None or df.empty) and state.get("last_non_empty_df") is not None:
        sql = state.get("last_non_empty_sql") or ""
        df = state.get("last_non_empty_df")
        query_id = state.get("last_non_empty_query_id") or "last_non_empty"
        salvaged_previous_non_empty = True
    return query_id, sql, df, salvaged_previous_non_empty


def _should_attempt_no_submit_finality_retry(state: dict[str, Any]) -> bool:
    if not _no_submit_finality_retry_enabled():
        return False
    if int(state.get("no_submit_retry_count") or 0) >= _no_submit_finality_max_retries():
        return False
    _, _, df, _ = _state_no_submit_candidate(state)
    return isinstance(df, pd.DataFrame)


def _no_submit_critique(
    messages: list[BaseMessage],
    query_id: str | None,
    sql: str | None,
    df: pd.DataFrame | None,
    reason: str,
) -> dict[str, Any]:
    critique = _critique_submit_result(messages, query_id or "latest", sql, df)
    flags = ["no_submit_result", *list(critique.get("flags", []))]
    suggestions = [
        "The agent ended without submitting a final query result; inspect the latest SQL and verify it is final.",
        *list(critique.get("suggestions", [])),
    ]
    return {
        **critique,
        "query_id": query_id or "latest",
        "flags": list(dict.fromkeys(flags)),
        "suggestions": list(dict.fromkeys(suggestions)),
        "should_reject": False,
        "no_submit": True,
        "no_submit_reason": reason,
    }


class ExecuteSubmit:
    """Simple graph with two tools: run_sql_query and submit_result.
    All context must be in the SystemMessage."""

    DISPLAY_ROW_LIMIT = 12
    """Max number of rows to return in SQL tool calls."""

    DISPLAY_CELL_CHAR_LIMIT = 1024
    """Max number of characters a dataframe cell can have before it is trimmed."""

    def __init__(self, connection: DuckDBPyConnection):
        self._connection = connection

    def init_state(self, messages: list[BaseMessage], *, limit_max_rows: int | None = None) -> AgentState:
        return AgentState(
            messages=messages,
            query_ids=get_query_ids_mapping(messages),
            sql=None,
            df=None,
            last_non_empty_query_id=None,
            last_non_empty_sql=None,
            last_non_empty_df=None,
            submit_critiques=[],
            visualization_prompt=None,
            ready_for_user=False,
            limit_max_rows=limit_max_rows,
            no_submit_retry_count=0,
        )

    def get_result(self, state: AgentState) -> ExecutionResult:
        last_ai_message = None
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage):
                last_ai_message = m
                break
        if last_ai_message is None:
            raise RuntimeError("No AI message found in message log")
        if len(last_ai_message.tool_calls) == 0:
            # Sometimes models don't call the submit_result tool, but we still want to return some dataframe.
            sql = state.get("sql", "")
            df = state.get("df")  # Latest df result (usually from run_sql_query)
            salvaged_previous_non_empty = False
            query_id = "latest"
            if (df is None or df.empty) and state.get("last_non_empty_df") is not None:
                sql = state.get("last_non_empty_sql", "")
                df = state.get("last_non_empty_df")
                salvaged_previous_non_empty = True
                query_id = state.get("last_non_empty_query_id") or "last_non_empty"
            submit_critiques = list(state.get("submit_critiques", []))
            submit_critiques.append(
                _no_submit_critique(state["messages"], query_id, sql, df, "last_ai_message_had_no_tool_calls")
            )
            visualization_prompt = state.get("visualization_prompt")
            result = ExecutionResult(
                text=last_ai_message.text,
                df=df,
                code=sql,
                meta={
                    "visualization_prompt": visualization_prompt,
                    ExecutionResult.META_MESSAGES_KEY: state["messages"],
                    "submit_called": False,
                    "submit_critiques": submit_critiques,
                    "salvaged_previous_non_empty_result": salvaged_previous_non_empty,
                    "salvaged_previous_non_empty_query_id": state.get("last_non_empty_query_id")
                    if salvaged_previous_non_empty
                    else None,
                },
            )
        elif len(last_ai_message.tool_calls) > 1:
            raise RuntimeError("Expected exactly one tool call in AI message")
        elif last_ai_message.tool_calls[0]["name"] != "submit_result":
            sql = state.get("sql", "")
            df = state.get("df")
            salvaged_previous_non_empty = False
            query_id = "latest"
            if (df is None or df.empty) and state.get("last_non_empty_df") is not None:
                sql = state.get("last_non_empty_sql", "")
                df = state.get("last_non_empty_df")
                salvaged_previous_non_empty = True
                query_id = state.get("last_non_empty_query_id") or "last_non_empty"
            submit_critiques = list(state.get("submit_critiques", []))
            submit_critiques.append(
                _no_submit_critique(
                    state["messages"],
                    query_id,
                    sql,
                    df,
                    f"latest_tool_call_was_{last_ai_message.tool_calls[0]['name']}",
                )
            )
            result = ExecutionResult(
                text=f"Latest tool call was {last_ai_message.tool_calls[0]['name']}; returning latest query result.",
                df=df,
                code=sql,
                meta={
                    "visualization_prompt": state.get("visualization_prompt"),
                    ExecutionResult.META_MESSAGES_KEY: state["messages"],
                    "submit_called": False,
                    "last_tool_call_was_submit": False,
                    "submit_critiques": submit_critiques,
                    "salvaged_previous_non_empty_result": salvaged_previous_non_empty,
                    "salvaged_previous_non_empty_query_id": state.get("last_non_empty_query_id")
                    if salvaged_previous_non_empty
                    else None,
                },
            )
        else:
            sql = state.get("sql", "")
            df = state.get("df")
            tool_call = last_ai_message.tool_calls[0]
            text = tool_call["args"]["result_description"]
            visualization_prompt = state.get("visualization_prompt", "")
            result = ExecutionResult(
                text=text,
                df=df,
                code=sql,
                meta={
                    "visualization_prompt": visualization_prompt,
                    ExecutionResult.META_MESSAGES_KEY: state["messages"],
                    "submit_called": True,
                    "submit_critiques": state.get("submit_critiques", []),
                },
            )
        return result

    def has_search_context_tool(self, domain: Domain) -> bool:
        return make_search_context_tool(domain) is not None

    def make_tools(self, domain: Domain, extra_tools: list[BaseTool] | None = None) -> list[BaseTool]:
        @tool(description=RUN_SQL_QUERY_TOOL_DESCRIPTION)
        def run_sql_query(sql: str, graph_state: Annotated[AgentState, InjectedState]) -> dict[str, Any]:
            return _run_sql_query(
                sql,
                con=self._connection,
                sql_row_limit=graph_state["limit_max_rows"],
                display_row_limit=self.DISPLAY_ROW_LIMIT,
                display_cell_char_limit=self.DISPLAY_CELL_CHAR_LIMIT,
            )

        @tool(parse_docstring=True)
        def submit_result(
            query_id: str,
            result_description: str,
            visualization_prompt: str,
        ) -> str:
            """
            Call this tool with the ID of the query you want to submit to the user.
            This will return control to the user and must always be the last tool call.
            The user will see the full query result, not just the first 12 rows. Returns a confirmation message.

            Args:
                query_id: The ID of the query to submit (query_ids are automatically generated when you run queries).
                result_description: A comment to a final result. This will be included in the final result.
                visualization_prompt: Optional visualization prompt. If not empty, a Vega-Lite visualization agent
                    will be asked to plot the submitted query data according to instructions in the prompt.
                    The instructions should be short and simple.
            """
            return f"Query {query_id} submitted successfully. Your response is now visible to the user."

        tools: list[BaseTool] = [run_sql_query, submit_result]
        search_context_tool = make_search_context_tool(domain)
        if search_context_tool is not None:
            tools.append(search_context_tool)

        if extra_tools:
            tools.extend(extra_tools)

        return tools

    def compile(
        self,
        model_config: LLMConfig,
        agent_config: AgentConfig,
        domain: Domain,
        extra_tools: list[BaseTool] | None = None,
    ) -> CompiledStateGraph[Any]:
        tools = self.make_tools(domain, extra_tools=extra_tools)
        llm_model = model_config.new_chat_model()

        if llm.is_openai_model(model_config.name):
            # Only OpenAI models support parallel tool calls parameter
            model_with_tools = model_bind_tools(llm_model, tools, parallel_tool_calls=agent_config.parallel_tool_calls)
        else:
            model_with_tools = model_bind_tools(llm_model, tools)

        def llm_node(state: AgentState) -> dict[str, Any]:
            messages = state["messages"]
            response = chat(messages, model_config, model_with_tools)
            return {"messages": [response[-1]]}

        def tool_executor_node(state: AgentState) -> dict[str, Any]:
            last_message = state["messages"][-1]
            tool_messages = []
            assert isinstance(last_message, AIMessage)

            tool_calls = last_message.tool_calls
            if len(tool_calls) == 0 and _should_attempt_no_submit_finality_retry(state):
                query_id, sql, df, salvaged_previous_non_empty = _state_no_submit_candidate(state)
                submit_critiques = list(state.get("submit_critiques", []))
                critique = _no_submit_critique(
                    state["messages"],
                    query_id,
                    sql,
                    df,
                    "final_ai_message_had_no_tool_calls_retry",
                )
                critique = {
                    **critique,
                    "action": "retry",
                    "retry_source": "no_submit_finality",
                    "salvaged_previous_non_empty_result": salvaged_previous_non_empty,
                }
                submit_critiques.append(critique)
                return {
                    "messages": [HumanMessage(content=_no_submit_finality_feedback_message(critique))],
                    "query_ids": dict(state.get("query_ids", {})),
                    "sql": sql,
                    "df": df,
                    "last_non_empty_query_id": state.get("last_non_empty_query_id"),
                    "last_non_empty_sql": state.get("last_non_empty_sql"),
                    "last_non_empty_df": state.get("last_non_empty_df"),
                    "submit_critiques": submit_critiques,
                    "visualization_prompt": state.get("visualization_prompt", ""),
                    "ready_for_user": False,
                    "no_submit_retry_count": int(state.get("no_submit_retry_count") or 0) + 1,
                }

            is_ready_for_user = any(tc["name"] == "submit_result" for tc in tool_calls)
            if is_ready_for_user:
                if len(tool_calls) > 1:
                    tool_messages = [
                        ToolMessage("submit_result must be the only tool call.", tool_call_id=tool_call["id"])
                        for tool_call in tool_calls
                    ]
                    return {"messages": tool_messages, "ready_for_user": False}
                else:
                    tool_call = tool_calls[0]

                    if "query_ids" not in state or len(state["query_ids"]) == 0:
                        tool_messages = [
                            ToolMessage("No queries have been executed yet.", tool_call_id=tool_call["id"])
                        ]
                        return {"messages": tool_messages, "ready_for_user": False}

                    query_id = tool_call["args"]["query_id"]
                    if query_id not in state["query_ids"]:
                        available_ids = ", ".join(state["query_ids"].keys())
                        tool_messages = [
                            ToolMessage(
                                f"Query ID {query_id} not found. Available query IDs: {available_ids}",
                                tool_call_id=tool_call["id"],
                            )
                        ]
                        return {"messages": tool_messages, "ready_for_user": False}

                    target_tool_message = state["query_ids"][query_id]
                    if target_tool_message.artifact is None or "df" not in target_tool_message.artifact:
                        tool_messages = [
                            ToolMessage(f"Query {query_id} does not have a valid result.", tool_call_id=tool_call["id"])
                        ]
                        return {"messages": tool_messages, "ready_for_user": False}

            query_ids = dict(state.get("query_ids", {}))
            sql = state.get("sql")
            df = state.get("df")
            last_non_empty_query_id = state.get("last_non_empty_query_id")
            last_non_empty_sql = state.get("last_non_empty_sql")
            last_non_empty_df = state.get("last_non_empty_df")
            submit_critiques = list(state.get("submit_critiques", []))
            visualization_prompt = state.get("visualization_prompt", "")

            message_index = len(state["messages"]) - 1

            for idx, tool_call in enumerate(tool_calls):
                name = tool_call["name"]
                args = tool_call["args"]
                tool_call_id = tool_call["id"]
                # Find the tool by name
                tool = next((t for t in tools if t.name == name), None)
                if tool is None:
                    tool_messages.append(ToolMessage(content=f"Tool {name} does not exist!", tool_call_id=tool_call_id))
                    continue

                try:
                    result = tool.invoke(args | {"graph_state": state})
                except Exception as e:
                    result = {"error": exception_to_string(e) + f"\nTool: {name}, Args: {args}"}

                content = ""
                if name == "run_sql_query":
                    sql = result.get("sql")
                    df = result.get("df")
                    # Generate query_id using message index and tool call index
                    query_id = f"{message_index}-{idx}"
                    # Override the query_id in the result
                    result["query_id"] = query_id
                    content = result.get("csv", result.get("error", ""))
                    if "csv" in result:
                        content = f"query_id='{query_id}'\n\n{content}"
                    if query_id:
                        query_ids[query_id] = ToolMessage(
                            content=content,
                            tool_call_id=tool_call_id,
                            artifact=result,
                        )
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        last_non_empty_query_id = query_id
                        last_non_empty_sql = sql
                        last_non_empty_df = df
                elif name == "submit_result":
                    content = str(result)
                    query_id = tool_call["args"]["query_id"]
                    visualization_prompt = tool_call["args"].get("visualization_prompt", "")
                    sql = state["query_ids"][query_id].artifact["sql"]
                    df = state["query_ids"][query_id].artifact["df"]
                    critique = _critique_submit_result(state["messages"], query_id, sql, df)
                    if "empty_submit_result" in critique.get("flags", []):
                        probe_results: list[dict[str, Any]] = []
                        for qualified_col in _date_filter_qualified_columns(sql)[:2]:
                            probe_sql = _build_date_probe_sql(sql, qualified_col)
                            if not probe_sql:
                                continue
                            try:
                                probe_res = _run_sql_query(
                                    probe_sql,
                                    con=self._connection,
                                    sql_row_limit=5,
                                    display_row_limit=5,
                                    display_cell_char_limit=200,
                                )
                                csv_text = (probe_res.get("csv") or "").strip()
                                if csv_text:
                                    probe_results.append({"column": qualified_col, "samples_csv": csv_text})
                                elif probe_res.get("error"):
                                    probe_results.append(
                                        {"column": qualified_col, "error": str(probe_res.get("error"))[:200]}
                                    )
                            except Exception as exc:
                                probe_results.append({"column": qualified_col, "error": str(exc)[:200]})
                        if probe_results:
                            critique = {**critique, "empty_result_probe": probe_results}
                            critique["suggestions"] = list(critique.get("suggestions", [])) + [
                                "Probe results show the ACTUAL stored date format for the filtered column(s) "
                                f"on the same FROM clause: {probe_results}. "
                                "Rewrite the date literal in WHERE to match this format, re-run the ORIGINAL "
                                "question's query via run_sql_query, then submit the new query result. "
                                "Do not submit these probe samples."
                            ]
                    submit_critiques.append(critique)
                    mode = _submit_critique_mode()
                    already_rejected = any(
                        item.get("query_id") == query_id and item.get("action") == "reject"
                        for item in submit_critiques
                    )
                    total_rejections = sum(
                        1 for item in submit_critiques if item.get("action") == "reject"
                    )
                    p0_reject = critique.get("p0_should_reject") and total_rejections < _p0_max_rejections()
                    mode_reject = mode == "reject" and critique["should_reject"]
                    if (mode_reject or p0_reject) and not already_rejected:
                        reject_source = "p0_gate" if p0_reject and not mode_reject else "mode_reject"
                        critique = {**critique, "action": "reject", "reject_source": reject_source}
                        submit_critiques[-1:] = [critique] if submit_critiques else [critique]
                        tool_messages.append(
                            ToolMessage(
                                content=_submit_critique_message(critique),
                                tool_call_id=tool_call_id,
                                artifact={"submit_critique": critique},
                            )
                        )
                        return {
                            "messages": tool_messages,
                            "query_ids": query_ids,
                            "sql": sql,
                            "df": df,
                            "last_non_empty_query_id": last_non_empty_query_id,
                            "last_non_empty_sql": last_non_empty_sql,
                            "last_non_empty_df": last_non_empty_df,
                            "submit_critiques": submit_critiques,
                            "visualization_prompt": visualization_prompt,
                            "ready_for_user": False,
                            "no_submit_retry_count": state.get("no_submit_retry_count", 0),
                        }
                else:
                    if isinstance(result, dict):
                        content = json.dumps(result, ensure_ascii=False, default=str)
                    else:
                        content = str(result)
                tool_messages.append(ToolMessage(content=content, tool_call_id=tool_call_id, artifact=result))
                if name == "submit_result":
                    return {
                        "messages": tool_messages,
                        "sql": sql,
                        "df": df,
                        "last_non_empty_query_id": last_non_empty_query_id,
                        "last_non_empty_sql": last_non_empty_sql,
                        "last_non_empty_df": last_non_empty_df,
                        "submit_critiques": submit_critiques,
                        "visualization_prompt": visualization_prompt,
                        "ready_for_user": True,
                        "no_submit_retry_count": state.get("no_submit_retry_count", 0),
                    }
            return {
                "messages": tool_messages,
                "query_ids": query_ids,
                "sql": sql,
                "df": df,
                "last_non_empty_query_id": last_non_empty_query_id,
                "last_non_empty_sql": last_non_empty_sql,
                "last_non_empty_df": last_non_empty_df,
                "submit_critiques": submit_critiques,
                "visualization_prompt": visualization_prompt,
                "ready_for_user": False,
                "no_submit_retry_count": state.get("no_submit_retry_count", 0),
            }

        def should_continue(state: AgentState) -> Literal["tool_executor", "end"]:
            # Check if there are tool calls in the last message
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return "tool_executor"
            if isinstance(last_message, AIMessage) and _should_attempt_no_submit_finality_retry(state):
                return "tool_executor"
            return "end"

        def should_finish(state: AgentState) -> Literal["llm_node", "end"]:
            # Check if we just executed submit_result - if so, end the conversation
            if state.get("ready_for_user", False):
                return "end"
            return "llm_node"

        graph = StateGraph(AgentState)
        graph.add_node("llm_node", llm_node)
        graph.add_node("tool_executor", tool_executor_node)

        graph.add_edge(START, "llm_node")
        graph.add_conditional_edges("llm_node", should_continue, {"tool_executor": "tool_executor", "end": END})
        graph.add_conditional_edges("tool_executor", should_finish, {"llm_node": "llm_node", "end": END})
        return graph.compile()
