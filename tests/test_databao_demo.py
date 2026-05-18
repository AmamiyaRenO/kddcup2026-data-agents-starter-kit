from __future__ import annotations

import json
import os
import queue
import sqlite3
import tempfile
import unittest
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import pandas as pd
from typer.testing import CliRunner

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.cli import app
from data_agent_baseline.run.databao_demo import (
    AnswerContract,
    Candidate,
    DatabaoEnvironment,
    DATABAO_DEBUG_LOG_RAW_ENV,
    DATABAO_DATABAO_TIMEOUT_SECONDS_ENV,
    DATABAO_TASK_TIMEOUT_SECONDS_ENV,
    StructuredPlanError,
    TaskDiagnostics,
    _candidate_table_payload,
    _databao_state_has_salvageable_frame,
    _databao_submit_critique,
    _databao_timeout_seconds,
    _build_question_prompt,
    _metric_like_columns,
    _run_databao_task_with_timeout,
    _timeout_task_artifact,
    apply_databao_observed_detail_aggregate_compactor,
    apply_answer_column_verifier,
    apply_aggregate_ratio_verifier,
    apply_column_only_compactor,
    apply_question_column_pruner,
    apply_ratio_scale_compactor,
    build_schema_graph,
    build_task_context,
    choose_route_policy,
    document_records_for_reasoning,
    execute_structured_plan,
    extract_question_features,
    final_answer_shape_guard,
    generate_verifier_candidate_frames,
    generic_document_tables,
    infer_answer_contract,
    infer_task_intent,
    load_databao_environment,
    load_context_tables,
    postprocess_answer_table,
    query_context_retriever,
    rank_candidates,
    register_context_sources,
    run_databao_task,
    run_databao_tasks,
    validate_answer_contract,
)
from data_agent_baseline.run.databao_vendor import ensure_vendor_databao_patches
from data_agent_baseline.tools.public_eval import evaluate_public_task


class FakeDomain:
    def __init__(self) -> None:
        self.dbs = []
        self.dfs = []
        self.descriptions = []

    def add_db(self, db, *, name=None, description=None) -> None:
        self.dbs.append({"db": db, "name": name, "description": description})

    def add_df(self, df, *, name=None, description=None) -> None:
        self.dfs.append({"df": df, "name": name, "description": description})

    def add_description(self, description) -> None:
        self.descriptions.append(description)


class FakeThread:
    def __init__(self, frame: pd.DataFrame | None) -> None:
        self.frame = frame
        self.query = None

    def ask(self, query: str, *, stream=None):
        self.query = query
        return self

    def df(self):
        return self.frame


class FakeAgent:
    def __init__(self, frame: pd.DataFrame | None) -> None:
        self.frame = frame

    def thread(self, **kwargs):
        return FakeThread(self.frame)


def make_agent_builder(frame: pd.DataFrame | None):
    def build_agent(task, databao_env):
        del task, databao_env
        return FakeAgent(frame)

    return build_agent


def no_answer_finalizer(*args, **kwargs):
    del args, kwargs
    raise AssertionError("finalizer hook should not be called by the Databao baseline")


def no_structured_planner(task, databao_env):
    del task, databao_env
    raise AssertionError("structured planner hook should not be called by the Databao baseline")


def write_task(
    root: Path,
    task_id: str = "task_1",
    question: str = "What is the answer?",
    difficulty: str = "easy",
) -> Path:
    task_dir = root / task_id
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "difficulty": difficulty,
                "question": question,
            }
        ),
        encoding="utf-8",
    )
    (context_dir / "knowledge.md").write_text("Use the available sources.", encoding="utf-8")
    return task_dir


class DatabaoDemoTests(unittest.TestCase):
    def test_databao_timeout_defaults_and_respects_task_deadline(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_databao_timeout_seconds(None), 100)

        deadline_at = perf_counter() + 150
        with patch.dict(os.environ, {DATABAO_DATABAO_TIMEOUT_SECONDS_ENV: "180"}):
            self.assertLessEqual(_databao_timeout_seconds(deadline_at), 130)
            self.assertGreaterEqual(_databao_timeout_seconds(deadline_at), 1)

    def test_load_databao_environment_requires_endpoint_and_model(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "MODEL_API_URL"):
            load_databao_environment({})

        env = load_databao_environment(
            {
                "MODEL_API_URL": "http://localhost:8080/v1",
                "MODEL_NAME": "qwen3.5-35b-a3b",
            }
        )
        self.assertEqual(env.model_api_key, "EMPTY")

    def test_register_context_sources_detects_supported_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context_dir = Path(tmp)
            (context_dir / "knowledge.md").write_text("Domain terms.", encoding="utf-8")

            csv_dir = context_dir / "csv"
            csv_dir.mkdir()
            (csv_dir / "items.csv").write_text("id,name\n1,A\n", encoding="utf-8")

            json_dir = context_dir / "json"
            json_dir.mkdir()
            (json_dir / "records.json").write_text(
                json.dumps({"rows": [{"id": 1, "value": "A"}]}),
                encoding="utf-8",
            )

            db_dir = context_dir / "db"
            db_dir.mkdir()
            db_path = db_dir / "sample.db"
            conn = sqlite3.connect(db_path)
            try:
                conn.execute("CREATE TABLE facts(id INTEGER PRIMARY KEY, name TEXT)")
                conn.execute("INSERT INTO facts(name) VALUES ('A')")
                conn.commit()
            finally:
                conn.close()

            domain = FakeDomain()
            summary = register_context_sources(domain, context_dir)

            self.assertEqual(len(domain.dbs), 1)
            self.assertEqual(len(domain.dfs), 2)
            self.assertEqual(len(domain.descriptions), 1)
            self.assertEqual(summary["csv_files"], ["csv/items.csv"])
            self.assertEqual(summary["json_files"], ["json/records.json"])
            self.assertEqual(summary["sqlite_files"], ["db/sample.db"])

    def test_build_question_prompt_is_plain_databao_adapter_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_root = Path(tmp)
            write_task(input_root, question="Which item has the highest score? Return the name.")
            task = DABenchPublicDataset(input_root).get_task("task_1")

        prompt = _build_question_prompt(task)

        self.assertIn("Return only the final answer table.", prompt)
        self.assertIn("Question: Which item has the highest score? Return the name.", prompt)
        self.assertNotIn("Context focus hints:", prompt)
        self.assertNotIn("Expected answer shape:", prompt)

    def test_metric_like_columns_use_schema_sample_not_full_object_column_scan(self) -> None:
        frame = pd.DataFrame(
            {
                "id": range(700),
                "numeric_metric": range(700),
                "numeric_text_metric": [""] + ["4"] * 699,
                "late_numeric_text": ["not numeric"] * 600 + ["9"] * 100,
            }
        )

        columns = _metric_like_columns(frame)

        self.assertIn("numeric_metric", columns)
        self.assertIn("numeric_text_metric", columns)
        self.assertNotIn("id", columns)
        self.assertNotIn("late_numeric_text", columns)

    def test_databao_submit_critique_records_shadow_risks_without_changing_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_root = Path(tmp)
            write_task(input_root, question="What percentage of rows are selected?")
            task = DABenchPublicDataset(input_root).get_task("task_1")
            frame = pd.DataFrame({"selected_count": [3]})
            frame.attrs["databao_code"] = "SELECT COUNT(*) AS selected_count FROM rows WHERE selected = 1"
            frame.attrs["databao_submit_called"] = False
            frame.attrs["databao_salvaged_latest_query_result"] = True
            frame.attrs["databao_submit_critiques"] = [
                {
                    "query_id": "latest",
                    "flags": ["no_submit_result", "numeric_code_filter_needs_value_grounding"],
                    "grounding_evidence": [
                        {
                            "column": "selected",
                            "value": "1",
                            "raw_value": "1",
                            "value_kind": "numeric",
                        }
                    ],
                    "anchored_lookup_evidence": [
                        {
                            "kind": "link_resolution",
                            "columns": ["cost", "link_to_member"],
                            "link_samples": {"link_to_member": ["recA"]},
                        }
                    ],
                    "no_submit": True,
                    "no_submit_reason": "latest_tool_call_was_run_sql_query",
                }
            ]
            contract = AnswerContract(
                kind="percentage",
                expected_columns=("percentage",),
                max_rows=1,
                max_columns=1,
                allow_empty=False,
                reason="test",
            )

            observation = _databao_submit_critique(task=task, frame=frame, answer_contract=contract)

        self.assertTrue(observation["shadow_only"])
        self.assertEqual(observation["source_kind"], "databao_salvaged_intermediate")
        self.assertIn("ratio_question_without_ratio_operation", observation["risk_flags"])
        self.assertIn("no_submit_latest_result", observation["risk_flags"])
        self.assertIn("submit_critique:no_submit_result", observation["risk_flags"])
        self.assertIn("submit_critique:numeric_code_filter_needs_value_grounding", observation["risk_flags"])
        self.assertEqual(
            observation["submit_critiques"][0]["anchored_lookup_evidence"][0]["link_samples"],
            {"link_to_member": ["recA"]},
        )
        self.assertTrue(observation["submit_critiques"][0]["no_submit"])
        self.assertEqual(observation["code_observation"]["code_kind"], "sql")

    def test_vendor_databao_patch_is_available(self) -> None:
        result = ensure_vendor_databao_patches()

        from databao.agent.executors.lighthouse import graph

        self.assertEqual(result["enabled"], "true")
        self.assertTrue(hasattr(graph, "_anchored_lookup_flags"))
        self.assertTrue(hasattr(graph, "_no_submit_critique"))
        self.assertTrue(hasattr(graph, "_should_attempt_no_submit_finality_retry"))

    def test_vendor_no_submit_finality_retry_is_one_shot_and_toggleable(self) -> None:
        from data_agent_baseline._vendor.databao_patches import lighthouse_graph as graph

        state = {
            "df": pd.DataFrame({"answer": [1]}),
            "no_submit_retry_count": 0,
        }
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DATABAO_INTERNAL_NO_SUBMIT_FINALITY_RETRY", None)
            os.environ.pop("DATABAO_INTERNAL_NO_SUBMIT_FINALITY_MAX_RETRIES", None)
            self.assertTrue(graph._should_attempt_no_submit_finality_retry(state))
            self.assertFalse(graph._should_attempt_no_submit_finality_retry({**state, "no_submit_retry_count": 1}))

        with patch.dict(os.environ, {"DATABAO_INTERNAL_NO_SUBMIT_FINALITY_RETRY": "0"}):
            self.assertFalse(graph._should_attempt_no_submit_finality_retry(state))

    def test_vendor_no_submit_finality_feedback_requires_submit(self) -> None:
        from langchain_core.messages import HumanMessage

        from data_agent_baseline._vendor.databao_patches import lighthouse_graph as graph

        message = graph._no_submit_finality_feedback_message(
            {
                "query_id": "3-0",
                "flags": ["no_submit_result", "count_question_without_count_result"],
                "suggestions": ["The question asks for a count; submit a count result."],
                "row_count": 2,
                "column_count": 1,
                "columns": ["answer"],
                "sql_text": "SELECT COUNT(*) AS answer FROM records",
                "sample_rows": [{"answer": "2"}],
            }
        )

        self.assertIn("submit_result", message)
        self.assertIn("3-0", message)
        self.assertIn("Latest result shape: 2 rows x 1 columns", message)
        self.assertIn("`answer`", message)
        self.assertIn("SELECT COUNT(*) AS answer FROM records", message)
        self.assertIn('"answer": "2"', message)
        self.assertIn("Sample rows are only a preview", message)
        self.assertIn("Do not answer in prose", message)
        self.assertNotIn("The question asks for a count", message)
        self.assertEqual(
            graph._latest_user_question(
                [
                    HumanMessage(content="Question: What is the original question?"),
                    HumanMessage(content=message),
                ]
            ),
            "What is the original question?",
        )

    def test_vendor_submit_critique_records_sample_rows_for_no_submit_feedback(self) -> None:
        from langchain_core.messages import HumanMessage

        from data_agent_baseline._vendor.databao_patches import lighthouse_graph as graph

        critique = graph._critique_submit_result(
            [HumanMessage(content="Question: What is the answer?")],
            "latest",
            "SELECT answer FROM table",
            pd.DataFrame({"answer": ["short", "x" * 120], "note": [None, ["nested"]]}),
        )

        self.assertEqual(critique["sample_rows"][0], {"answer": "short", "note": None})
        self.assertTrue(str(critique["sample_rows"][1]["answer"]).endswith("…"))
        self.assertIn("nested", str(critique["sample_rows"][1]["note"]))

    def test_vendor_time_granularity_critique_rejects_fractional_equality_for_whole_second_question(self) -> None:
        from data_agent_baseline._vendor.databao_patches import lighthouse_graph as graph

        flags, suggestions, evidence = graph._unit_time_granularity_critique(
            "Which row finished 0:01:54?",
            "SELECT number FROM qualifying WHERE q3 = '1:54.455'",
        )

        self.assertIn("time_literal_precision_needs_rounding", flags)
        self.assertTrue(any("whole-second precision" in suggestion for suggestion in suggestions))
        self.assertEqual(evidence[0]["kind"], "whole_second_question_vs_fractional_sql_equality")

    def test_vendor_time_granularity_critique_rejects_limited_whole_second_prefix_match(self) -> None:
        from data_agent_baseline._vendor.databao_patches import lighthouse_graph as graph

        flags, suggestions, evidence = graph._unit_time_granularity_critique(
            "Which rows finished 0:01:54?",
            "SELECT number FROM qualifying WHERE q3 LIKE '1:54%' ORDER BY q3 ASC LIMIT 1",
        )

        self.assertIn("whole_second_time_limit_may_drop_matches", flags)
        self.assertTrue(any("Remove the LIMIT" in suggestion for suggestion in suggestions))
        self.assertIn(
            "whole_second_prefix_filter_with_limit",
            {item["kind"] for item in evidence},
        )

    def test_vendor_time_granularity_critique_allows_ranked_limited_whole_second_prefix_match(self) -> None:
        from data_agent_baseline._vendor.databao_patches import lighthouse_graph as graph

        flags, _, _ = graph._unit_time_granularity_critique(
            "Which row ranked first among those who finished 0:01:54?",
            "SELECT number FROM qualifying WHERE q3 LIKE '1:54%' ORDER BY q3 ASC LIMIT 1",
        )

        self.assertNotIn("whole_second_time_limit_may_drop_matches", flags)

    def test_vendor_blank_display_critique_does_not_reject_requested_attribute_column(self) -> None:
        from data_agent_baseline._vendor.databao_patches import lighthouse_graph as graph

        flags, suggestions, evidence = graph._blank_display_critique(
            "What is his number of the driver who finished in Q3?",
            pd.DataFrame({"driverId": [1, 2], "q3": ["1:54.455", "1:54.781"], "number": [3, 5]}),
        )

        self.assertEqual(flags, [])
        self.assertEqual(suggestions, [])
        self.assertEqual(evidence, [])

    def test_databao_state_has_salvageable_frame_accepts_previous_non_empty_result(self) -> None:
        self.assertFalse(_databao_state_has_salvageable_frame({"df": None}))
        self.assertTrue(_databao_state_has_salvageable_frame({"df": pd.DataFrame({"answer": [1]})}))
        self.assertTrue(
            _databao_state_has_salvageable_frame(
                {"df": None, "last_non_empty_df": pd.DataFrame({"answer": [1]})}
            )
        )

    def test_json_loader_preserves_envelope_table_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root)
            context_dir = input_root / "task_1" / "context"
            json_dir = context_dir / "json"
            json_dir.mkdir()
            (json_dir / "Patient.json").write_text(
                json.dumps(
                    {
                        "table": "Patient",
                        "records": [{"ID": 1, "SEX": "F", "Diagnosis": "SLE"}],
                    }
                ),
                encoding="utf-8",
            )

            tables = load_context_tables(context_dir)

            self.assertEqual([table.name for table in tables], ["Patient"])

    def test_generic_document_loader_extracts_display_name_after_identifier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root)
            doc_dir = input_root / "task_1" / "context" / "doc"
            doc_dir.mkdir(parents=True)
            (doc_dir / "records.md").write_text(
                "The asset registered under recABC123 is Jordan Rivera. This person manages shared records.",
                encoding="utf-8",
            )

            tables = load_context_tables(input_root / "task_1" / "context")
            document_table = next(table for table in tables if table.name == "document_records")
            row = document_table.frame[document_table.frame["record_id"].eq("recABC123")].iloc[0]

            self.assertEqual(row["name"], "Jordan Rivera")

    def test_structured_plan_executes_task_11_style_join(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(
                input_root,
                task_id="task_11",
                question=(
                    "For patients with severe degree of thrombosis, list their ID, sex and "
                    "disease the patient is diagnosed with."
                ),
            )
            context_dir = input_root / "task_11" / "context"
            json_dir = context_dir / "json"
            json_dir.mkdir()
            (json_dir / "Examination.json").write_text(
                json.dumps(
                    {
                        "table": "Examination",
                        "records": [
                            {"ID": 163109, "Thrombosis": 2, "Diagnosis": "Exam diagnosis"},
                            {"ID": 2803470, "Thrombosis": 2, "Diagnosis": "Exam diagnosis"},
                            {"ID": 1430760, "Thrombosis": 2, "Diagnosis": "No patient row"},
                            {"ID": 999999, "Thrombosis": 1, "Diagnosis": "Mild"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (json_dir / "Patient.json").write_text(
                json.dumps(
                    {
                        "table": "Patient",
                        "records": [
                            {"ID": 163109, "SEX": "F", "Diagnosis": "SLE"},
                            {"ID": 2803470, "SEX": "F", "Diagnosis": "SLE"},
                            {"ID": 999999, "SEX": "M", "Diagnosis": "Other"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            task = DABenchPublicDataset(input_root).get_task("task_11")
            plan = {
                "steps": [
                    {"op": "source", "table": "Examination", "alias": "exam"},
                    {
                        "op": "filter",
                        "source": "exam",
                        "alias": "severe",
                        "conditions": [{"column": "Thrombosis", "op": "==", "value": 2}],
                    },
                    {
                        "op": "join",
                        "left": "severe",
                        "right_table": "Patient",
                        "left_on": "ID",
                        "right_on": "ID",
                        "how": "inner",
                        "alias": "joined",
                    },
                    {
                        "op": "select",
                        "source": "joined",
                        "alias": "answer",
                        "columns": ["ID", "SEX", "Diagnosis"],
                    },
                    {"op": "distinct", "source": "answer", "alias": "final"},
                ],
                "output": "final",
            }

            frame = execute_structured_plan(task, plan)

            self.assertEqual(
                frame.to_dict(orient="list"),
                {
                    "ID": [163109, 2803470],
                    "SEX": ["F", "F"],
                    "Diagnosis": ["SLE", "SLE"],
                },
            )

    def test_schema_graph_records_sqlite_tables_and_join_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root)
            context_dir = input_root / "task_1" / "context"
            csv_dir = context_dir / "csv"
            csv_dir.mkdir()
            (csv_dir / "orders.csv").write_text("customer_id,total\n1,20\n", encoding="utf-8")
            db_dir = context_dir / "db"
            db_dir.mkdir()
            conn = sqlite3.connect(db_dir / "customers.db")
            try:
                conn.execute("CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT)")
                conn.execute("INSERT INTO customers(id, name) VALUES (1, 'A')")
                conn.commit()
            finally:
                conn.close()

            graph = build_schema_graph(load_context_tables(context_dir))

            self.assertEqual(
                [(table["name"], table["source_kind"]) for table in graph["tables"]],
                [("orders", "csv"), ("customers", "sqlite")],
            )
            self.assertIn(
                {
                    "left_table": "orders",
                    "left_column": "customer_id",
                    "right_table": "customers",
                    "right_column": "id",
                    "reason": "shared_identifier_variant",
                },
                graph["join_candidates"],
            )

    def test_structured_plan_executes_sqlite_csv_json_three_table_join(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root)
            context_dir = input_root / "task_1" / "context"
            db_dir = context_dir / "db"
            csv_dir = context_dir / "csv"
            json_dir = context_dir / "json"
            db_dir.mkdir()
            csv_dir.mkdir()
            json_dir.mkdir()
            conn = sqlite3.connect(db_dir / "event.db")
            try:
                conn.execute("CREATE TABLE event(event_id TEXT PRIMARY KEY, event_name TEXT)")
                conn.execute("INSERT INTO event(event_id, event_name) VALUES ('recEvent', 'October Meeting')")
                conn.commit()
            finally:
                conn.close()
            (json_dir / "budget.json").write_text(
                json.dumps({"table": "budget", "records": [{"budget_id": "recBudget", "link_to_event": "recEvent"}]}),
                encoding="utf-8",
            )
            (csv_dir / "expense.csv").write_text(
                "expense_id,link_to_budget,cost\nrecExpense,recBudget,42.5\n",
                encoding="utf-8",
            )
            task = DABenchPublicDataset(input_root).get_task("task_1")
            plan = {
                "steps": [
                    {"op": "source", "table": "event", "alias": "event"},
                    {
                        "op": "join",
                        "left": "event",
                        "right_table": "budget",
                        "left_on": "event_id",
                        "right_on": "link_to_event",
                        "alias": "event_budget",
                    },
                    {
                        "op": "join",
                        "left": "event_budget",
                        "right_table": "expense",
                        "left_on": "budget_id",
                        "right_on": "link_to_budget",
                        "alias": "joined",
                    },
                    {
                        "op": "select",
                        "source": "joined",
                        "alias": "answer",
                        "columns": ["event_name", "cost"],
                    },
                ],
                "output": "answer",
            }

            frame = execute_structured_plan(task, plan)

            self.assertEqual(frame.to_dict(orient="list"), {"event_name": ["October Meeting"], "cost": [42.5]})

    def test_structured_plan_supports_date_filter_and_safe_ratio_percentage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root)
            context_dir = input_root / "task_1" / "context"
            csv_dir = context_dir / "csv"
            csv_dir.mkdir()
            (csv_dir / "facts.csv").write_text(
                "id,date,group,value\n1,2013-06-01,A,10\n2,2013-07-01,A,20\n3,2013-06-15,B,30\n",
                encoding="utf-8",
            )
            task = DABenchPublicDataset(input_root).get_task("task_1")
            plan = {
                "steps": [
                    {"op": "source", "table": "facts", "alias": "facts"},
                    {"op": "date_filter", "source": "facts", "alias": "june", "column": "date", "year": 2013, "month": 6},
                    {
                        "op": "aggregate",
                        "source": "june",
                        "alias": "agg",
                        "aggregations": [
                            {"function": "count_distinct", "column": "group", "as": "groups"},
                            {"function": "sum", "column": "value", "as": "total"},
                        ],
                    },
                    {"op": "percentage", "source": "agg", "alias": "answer", "numerator": "groups", "denominator": "total"},
                ],
                "output": "answer",
            }

            frame = execute_structured_plan(task, plan)

            self.assertAlmostEqual(frame["percentage"].iloc[0], 5.0)

    def test_structured_plan_rejects_unknown_safe_expression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root)
            context_dir = input_root / "task_1" / "context"
            csv_dir = context_dir / "csv"
            csv_dir.mkdir()
            (csv_dir / "facts.csv").write_text("a,b\n1,2\n", encoding="utf-8")
            task = DABenchPublicDataset(input_root).get_task("task_1")
            plan = {
                "steps": [
                    {"op": "source", "table": "facts", "alias": "facts"},
                    {
                        "op": "derive",
                        "source": "facts",
                        "alias": "bad",
                        "columns": [{"op": "python", "left": "a", "right": "b", "as": "unsafe"}],
                    },
                ],
                "output": "bad",
            }

            with self.assertRaises(StructuredPlanError):
                execute_structured_plan(task, plan)

    def test_structured_plan_rejects_invalid_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root)
            context_dir = input_root / "task_1" / "context"
            csv_dir = context_dir / "csv"
            csv_dir.mkdir()
            (csv_dir / "items.csv").write_text("id,name\n1,A\n", encoding="utf-8")
            task = DABenchPublicDataset(input_root).get_task("task_1")

            invalid_plans = [
                {"steps": [{"op": "source", "table": "missing", "alias": "a"}], "output": "a"},
                {
                    "steps": [
                        {"op": "source", "table": "items", "alias": "a"},
                        {
                            "op": "filter",
                            "source": "a",
                            "alias": "b",
                            "conditions": [{"column": "missing", "op": "==", "value": 1}],
                        },
                    ],
                    "output": "b",
                },
                {
                    "steps": [{"op": "select", "source": "missing_alias", "alias": "b", "columns": ["id"]}],
                    "output": "b",
                },
                {"steps": [{"op": "python", "alias": "a"}], "output": "a"},
            ]

            for plan in invalid_plans:
                with self.subTest(plan=plan):
                    with self.assertRaises(StructuredPlanError):
                        execute_structured_plan(task, plan)

    def test_successful_dataframe_writes_prediction_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            output_root = root / "output"
            logs_dir = root / "logs"
            write_task(input_root)
            task = DABenchPublicDataset(input_root).get_task("task_1")

            artifact = run_databao_task(
                task=task,
                output_root=output_root,
                logs_dir=logs_dir,
                databao_env=DatabaoEnvironment("http://localhost:8080/v1", "key", "model"),
                agent_builder=make_agent_builder(pd.DataFrame({"answer": ["A"]})),
                answer_finalizer=no_answer_finalizer,
                structured_planner=no_structured_planner,
            )

            self.assertTrue(artifact.succeeded)
            self.assertIsNotNone(artifact.prediction_csv_path)
            self.assertTrue((output_root / "task_1" / "prediction.csv").exists())
            self.assertEqual(
                (output_root / "task_1" / "prediction.csv").read_text(encoding="utf-8"),
                "answer\nA\n",
            )
            log_payload = json.loads((logs_dir / "task_1.json").read_text(encoding="utf-8"))
            self.assertEqual(log_payload["candidate_source"], "databao_raw")
            self.assertEqual(log_payload["selected_candidate_source"], "databao_raw")
            self.assertEqual(log_payload["heuristic_level"], "generic")
            self.assertIn("enabled_strategies", log_payload)
            self.assertIn("applied_strategies", log_payload)
            self.assertIn("retrieved_context", log_payload)
            self.assertIn("candidate_scores", log_payload)
            self.assertIn("final_answer_guard", log_payload)
            self.assertIn("document_extraction_used", log_payload)
            self.assertIn("context_payload_profile", log_payload)
            self.assertIsNone(log_payload["databao_failure_type"])
            self.assertIn("raw_table_diagnostics", log_payload["candidates"][0])
            self.assertEqual(artifact.heuristic_level, "generic")
            self.assertIsNotNone(artifact.context_payload_profile)
            self.assertIn("context_load", log_payload["timings"])
            self.assertIn("databao_ask", log_payload["timings"])
            self.assertEqual(log_payload["llm_calls"], [])
            progress_payload = json.loads((logs_dir / "task_1.progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress_payload["active_stage"], "artifact_ready")
            self.assertEqual(progress_payload["metadata"]["prediction_written"], True)

    def test_prediction_csv_writes_integer_like_floats_as_ints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prediction.csv"

            from data_agent_baseline.run.databao_demo import _write_prediction_csv

            _write_prediction_csv(pd.DataFrame({"ID": [163109.0, 2803470.0]}), path)

            self.assertEqual(path.read_text(encoding="utf-8"), "ID\n163109\n2803470\n")

    def test_databao_dataframe_writes_provisional_before_postprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            output_root = root / "output"
            logs_dir = root / "logs"
            write_task(input_root)
            task = DABenchPublicDataset(input_root).get_task("task_1")

            def failing_postprocessor(task, frame):
                del task, frame
                raise RuntimeError("postprocess failed")

            artifact = run_databao_task(
                task=task,
                output_root=output_root,
                logs_dir=logs_dir,
                databao_env=DatabaoEnvironment("http://localhost:8080/v1", "key", "model"),
                agent_builder=make_agent_builder(pd.DataFrame({"answer": ["raw"]})),
                answer_finalizer=no_answer_finalizer,
                answer_postprocessor=failing_postprocessor,
                structured_planner=no_structured_planner,
            )

            log_payload = json.loads((logs_dir / "task_1.json").read_text(encoding="utf-8"))
            self.assertTrue(artifact.prediction_written)
            self.assertTrue(artifact.scorable)
            self.assertEqual(log_payload["provisional_written_stage"], "databao_raw_provisional")
            self.assertEqual(
                (output_root / "task_1" / "prediction.csv").read_text(encoding="utf-8"),
                "answer\nraw\n",
            )

    def test_raw_provisional_survives_final_guard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            output_root = root / "output"
            logs_dir = root / "logs"
            write_task(input_root)
            task = DABenchPublicDataset(input_root).get_task("task_1")

            with patch(
                "data_agent_baseline.run.databao_demo.final_answer_shape_guard",
                side_effect=RuntimeError("guard failed"),
            ):
                artifact = run_databao_task(
                    task=task,
                    output_root=output_root,
                    logs_dir=logs_dir,
                    databao_env=DatabaoEnvironment("http://localhost:8080/v1", "key", "model"),
                    agent_builder=make_agent_builder(pd.DataFrame({"answer": ["raw"]})),
                    answer_finalizer=no_answer_finalizer,
                    structured_planner=no_structured_planner,
                )

            log_payload = json.loads((logs_dir / "task_1.json").read_text(encoding="utf-8"))
            self.assertFalse(artifact.succeeded)
            self.assertTrue(artifact.prediction_written)
            self.assertTrue(artifact.scorable)
            self.assertEqual(log_payload["provisional_written_stage"], "databao_raw_provisional")
            self.assertIn("guard failed", log_payload["failure_reason"])
            self.assertEqual(
                (output_root / "task_1" / "prediction.csv").read_text(encoding="utf-8"),
                "answer\nraw\n",
            )

    def test_runner_prioritizes_databao_before_structured_planner_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            output_root = root / "output"
            logs_dir = root / "logs"
            write_task(input_root)
            task = DABenchPublicDataset(input_root).get_task("task_1")

            def planner(task, databao_env):
                del task, databao_env
                raise AssertionError("Structured planner should not run when Databao returns a dataframe.")

            artifact = run_databao_task(
                task=task,
                output_root=output_root,
                logs_dir=logs_dir,
                databao_env=DatabaoEnvironment("http://localhost:8080/v1", "key", "model"),
                agent_builder=make_agent_builder(pd.DataFrame({"answer": ["databao"]})),
                answer_finalizer=no_answer_finalizer,
                structured_planner=planner,
            )

            log_payload = json.loads((logs_dir / "task_1.json").read_text(encoding="utf-8"))
            self.assertTrue(artifact.succeeded)
            self.assertEqual(artifact.candidate_source, "databao_raw")
            self.assertEqual(log_payload["candidate_source"], "databao_raw")
            self.assertNotIn("structured_planner", log_payload)
            self.assertEqual(
                (output_root / "task_1" / "prediction.csv").read_text(encoding="utf-8"),
                "answer\ndatabao\n",
            )

    def test_planner_first_mode_still_keeps_databao_first_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            output_root = root / "output"
            logs_dir = root / "logs"
            write_task(input_root)
            task = DABenchPublicDataset(input_root).get_task("task_1")

            def planner(task, databao_env):
                del task, databao_env
                raise AssertionError("Structured planner should not run in the Databao baseline.")

            artifact = run_databao_task(
                task=task,
                output_root=output_root,
                logs_dir=logs_dir,
                databao_env=DatabaoEnvironment("http://localhost:8080/v1", "key", "model"),
                agent_builder=make_agent_builder(pd.DataFrame({"answer": ["databao"]})),
                answer_finalizer=no_answer_finalizer,
                structured_planner=planner,
                structured_planner_mode="first",
            )

            log_payload = json.loads((logs_dir / "task_1.json").read_text(encoding="utf-8"))
            self.assertTrue(artifact.succeeded)
            self.assertEqual(artifact.candidate_source, "databao_raw")
            self.assertEqual(log_payload["candidate_source"], "databao_raw")
            self.assertNotIn("structured_planner", log_payload)
            self.assertEqual(
                (output_root / "task_1" / "prediction.csv").read_text(encoding="utf-8"),
                "answer\ndatabao\n",
            )

    def test_databao_failure_count_question_uses_narrow_count_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            output_root = root / "output"
            logs_dir = root / "logs"
            write_task(input_root, question="How many records are there?")
            context_dir = input_root / "task_1" / "context"
            csv_dir = context_dir / "csv"
            csv_dir.mkdir()
            (csv_dir / "records.csv").write_text("id,value\n1,A\n2,B\n", encoding="utf-8")
            task = DABenchPublicDataset(input_root).get_task("task_1")

            def planner(task, databao_env):
                del task, databao_env
                raise AssertionError("planner should not rescue Databao failure")

            def failing_agent_builder(task, databao_env):
                del task, databao_env
                raise TimeoutError("Databao timed out")

            artifact = run_databao_task(
                task=task,
                output_root=output_root,
                logs_dir=logs_dir,
                databao_env=DatabaoEnvironment("http://localhost:8080/v1", "key", "model"),
                agent_builder=failing_agent_builder,
                answer_finalizer=no_answer_finalizer,
                structured_planner=planner,
                structured_planner_mode="off",
            )

            log_payload = json.loads((logs_dir / "task_1.json").read_text(encoding="utf-8"))
            self.assertTrue(artifact.succeeded)
            self.assertEqual(artifact.candidate_source, "cheap_count_fallback")
            self.assertEqual(log_payload["candidate_source"], "cheap_count_fallback")
            self.assertNotIn("structured_planner", log_payload)
            self.assertEqual(
                (output_root / "task_1" / "prediction.csv").read_text(encoding="utf-8"),
                "count\n2\n",
            )

    def test_databao_failure_filtered_count_question_does_not_use_unfiltered_count_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            output_root = root / "output"
            logs_dir = root / "logs"
            write_task(input_root, question="How many records match the selected condition?")
            context_dir = input_root / "task_1" / "context"
            csv_dir = context_dir / "csv"
            csv_dir.mkdir()
            (csv_dir / "records.csv").write_text("id,value\n1,A\n2,B\n", encoding="utf-8")
            task = DABenchPublicDataset(input_root).get_task("task_1")

            def failing_agent_builder(task, databao_env):
                del task, databao_env
                raise TimeoutError("Databao timed out")

            artifact = run_databao_task(
                task=task,
                output_root=output_root,
                logs_dir=logs_dir,
                databao_env=DatabaoEnvironment("http://localhost:8080/v1", "key", "model"),
                agent_builder=failing_agent_builder,
                answer_finalizer=no_answer_finalizer,
                structured_planner=no_structured_planner,
                structured_planner_mode="off",
            )

            log_payload = json.loads((logs_dir / "task_1.json").read_text(encoding="utf-8"))
            self.assertFalse(artifact.succeeded)
            self.assertFalse(artifact.prediction_written)
            self.assertFalse((output_root / "task_1" / "prediction.csv").exists())
            self.assertIn("cheap count fallback", log_payload["missing_prediction_reason"])

    def test_databao_failure_grouped_count_question_does_not_use_unexecuted_count_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            output_root = root / "output"
            logs_dir = root / "logs"
            write_task(input_root, question="How many records per category should be returned?")
            context_dir = input_root / "task_1" / "context"
            csv_dir = context_dir / "csv"
            csv_dir.mkdir()
            (csv_dir / "records.csv").write_text("id,category\n1,A\n2,B\n", encoding="utf-8")
            task = DABenchPublicDataset(input_root).get_task("task_1")

            def failing_agent_builder(task, databao_env):
                del task, databao_env
                raise TimeoutError("Databao timed out")

            artifact = run_databao_task(
                task=task,
                output_root=output_root,
                logs_dir=logs_dir,
                databao_env=DatabaoEnvironment("http://localhost:8080/v1", "key", "model"),
                agent_builder=failing_agent_builder,
                answer_finalizer=no_answer_finalizer,
                structured_planner=no_structured_planner,
                structured_planner_mode="off",
            )

            self.assertFalse(artifact.succeeded)
            self.assertFalse(artifact.prediction_written)
            self.assertFalse((output_root / "task_1" / "prediction.csv").exists())

    def test_databao_failure_non_count_question_does_not_write_whole_table_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            output_root = root / "output"
            logs_dir = root / "logs"
            write_task(input_root, question="Which names match?")
            context_dir = input_root / "task_1" / "context"
            csv_dir = context_dir / "csv"
            csv_dir.mkdir()
            (csv_dir / "records.csv").write_text("id,name\n1,A\n2,B\n", encoding="utf-8")
            task = DABenchPublicDataset(input_root).get_task("task_1")

            def failing_agent_builder(task, databao_env):
                del task, databao_env
                raise TimeoutError("Databao timed out")

            artifact = run_databao_task(
                task=task,
                output_root=output_root,
                logs_dir=logs_dir,
                databao_env=DatabaoEnvironment("http://localhost:8080/v1", "key", "model"),
                agent_builder=failing_agent_builder,
                answer_finalizer=no_answer_finalizer,
                structured_planner=no_structured_planner,
                structured_planner_mode="off",
            )

            log_payload = json.loads((logs_dir / "task_1.json").read_text(encoding="utf-8"))
            self.assertFalse(artifact.succeeded)
            self.assertFalse(artifact.prediction_written)
            self.assertFalse((output_root / "task_1" / "prediction.csv").exists())
            self.assertIn("cheap count fallback", log_payload["missing_prediction_reason"])

    def test_run_databao_tasks_can_filter_task_ids_inline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            output_root = root / "output"
            logs_dir = root / "logs"
            write_task(input_root, task_id="task_1")
            write_task(input_root, task_id="task_2")

            artifacts = run_databao_tasks(
                input_root=input_root,
                output_root=output_root,
                logs_dir=logs_dir,
                task_ids=["task_2"],
                databao_env=DatabaoEnvironment("http://localhost:8080/v1", "key", "model"),
                agent_builder=make_agent_builder(pd.DataFrame({"answer": ["B"]})),
                answer_finalizer=no_answer_finalizer,
                structured_planner=no_structured_planner,
                task_timeout_seconds=1,
            )

            self.assertEqual([artifact.task_id for artifact in artifacts], ["task_2"])
            self.assertTrue((output_root / "task_2" / "prediction.csv").exists())
            self.assertFalse((output_root / "task_1" / "prediction.csv").exists())

    def test_run_databao_tasks_can_filter_by_difficulty_inline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            output_root = root / "output"
            logs_dir = root / "logs"
            write_task(input_root, task_id="task_330", difficulty="hard")
            write_task(input_root, task_id="task_344", difficulty="hard")
            write_task(input_root, task_id="task_418", difficulty="extreme")

            artifacts = run_databao_tasks(
                input_root=input_root,
                output_root=output_root,
                logs_dir=logs_dir,
                difficulty="hard",
                databao_env=DatabaoEnvironment("http://localhost:8080/v1", "key", "model"),
                agent_builder=make_agent_builder(pd.DataFrame({"answer": ["B"]})),
                answer_finalizer=no_answer_finalizer,
                structured_planner=no_structured_planner,
                task_timeout_seconds=1,
            )

            self.assertEqual([artifact.task_id for artifact in artifacts], ["task_330", "task_344"])
            self.assertTrue((output_root / "task_330" / "prediction.csv").exists())
            self.assertTrue((output_root / "task_344" / "prediction.csv").exists())
            self.assertFalse((output_root / "task_418" / "prediction.csv").exists())

    def test_timeout_wrapper_returns_queued_artifact_before_killing_live_child(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            output_root = root / "output"
            logs_dir = root / "logs"
            write_task(input_root)
            task = DABenchPublicDataset(input_root).get_task("task_1")
            log_path = logs_dir / "task_1.json"
            log_path.parent.mkdir(parents=True)
            log_path.write_text('{"succeeded": true}\n', encoding="utf-8")
            queued_artifact = type(
                "QueuedArtifact",
                (),
                {
                    "task_id": "task_1",
                    "succeeded": True,
                    "prediction_written": True,
                    "scorable": True,
                },
            )()

            class FakeQueue:
                def __init__(self, *args, **kwargs):
                    del args, kwargs
                    self._returned = False

                def get(self, timeout=None):
                    del timeout
                    if self._returned:
                        raise queue.Empty
                    self._returned = True
                    return queued_artifact

                def get_nowait(self):
                    raise queue.Empty

            class FakeProcess:
                terminated = False
                killed = False

                def __init__(self, *args, **kwargs):
                    del args, kwargs
                    self.exitcode = None

                def start(self):
                    return None

                def join(self, timeout=None):
                    del timeout
                    return None

                def is_alive(self):
                    return not self.terminated

                def terminate(self):
                    self.terminated = True
                    type(self).terminated = True

                def kill(self):
                    self.killed = True
                    type(self).killed = True

            class FakeContext:
                Queue = FakeQueue
                Process = FakeProcess

            with patch("data_agent_baseline.run.databao_demo.multiprocessing.get_context", return_value=FakeContext()):
                artifact = _run_databao_task_with_timeout(
                    task=task,
                    output_root=output_root,
                    logs_dir=logs_dir,
                    databao_env=DatabaoEnvironment("http://localhost:8080/v1", "key", "model"),
                    timeout_seconds=1,
                    structured_planner_mode="off",
                    finalizer_mode="off",
                )

            self.assertIs(artifact, queued_artifact)
            self.assertTrue(FakeProcess.terminated)
            self.assertEqual(json.loads(log_path.read_text(encoding="utf-8")), {"succeeded": True})

    def test_run_databao_demo_cli_passes_difficulty_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            run_root = root / "run"
            input_root.mkdir()
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "dataset:",
                        f"  root_path: {input_root.as_posix()}",
                        "agent:",
                        "  model: unused",
                        "  api_base: unused",
                        "  api_key: unused",
                        "  max_steps: 16",
                        "  temperature: 0.0",
                        "run:",
                        "  output_dir: artifacts/runs",
                        "  run_id:",
                        "  max_workers: 4",
                        "  task_timeout_seconds: 150",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            captured: dict[str, object] = {}

            def fake_run_databao_tasks(**kwargs):
                captured.update(kwargs)
                return []

            runner = CliRunner()
            with (
                patch(
                    "data_agent_baseline.cli.load_databao_environment",
                    return_value=DatabaoEnvironment("https://openrouter.ai/api/v1", "key", "model"),
                ),
                patch(
                    "data_agent_baseline.cli.create_databao_local_run_dir",
                    return_value=("runid", run_root),
                ),
                patch("data_agent_baseline.cli.run_databao_tasks", side_effect=fake_run_databao_tasks),
            ):
                result = runner.invoke(
                    app,
                    [
                        "run-databao-demo",
                        "--config",
                        str(config_path),
                        "--difficulty",
                        "hard",
                    ],
                )

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(captured["difficulty"], "hard")
            self.assertIsNone(captured["task_ids"])
            self.assertEqual(captured["task_timeout_seconds"], 150)

    def test_run_databao_demo_cli_help_does_not_expose_planner_mode(self) -> None:
        result = CliRunner().invoke(app, ["run-databao-demo", "--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertNotIn("--planner-mode", result.output)
        self.assertNotIn("--finalizer-mode", result.output)

    def test_timeout_artifact_writes_failure_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            output_root = root / "output"
            logs_dir = root / "logs"
            write_task(input_root)
            task = DABenchPublicDataset(input_root).get_task("task_1")
            logs_dir.mkdir()
            (logs_dir / "task_1.progress.json").write_text(
                json.dumps({"active_stage": "databao_ask"}),
                encoding="utf-8",
            )

            artifact = _timeout_task_artifact(
                task=task,
                output_root=output_root,
                logs_dir=logs_dir,
                elapsed_seconds=1.25,
                timeout_seconds=1,
            )

            log_payload = json.loads((logs_dir / "task_1.json").read_text(encoding="utf-8"))
            self.assertFalse(artifact.succeeded)
            self.assertIn("TimeoutError", artifact.failure_reason)
            self.assertEqual(log_payload["timings"]["task_timeout_seconds"], 1)
            self.assertEqual(log_payload["active_progress"]["active_stage"], "databao_ask")
            self.assertIn("prediction.csv", log_payload["missing_prediction_reason"])
            self.assertFalse((output_root / "task_1" / "prediction.csv").exists())

    def test_task_diagnostics_checkpoint_records_intermediate_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            logs_dir = root / "logs"
            write_task(input_root)
            task = DABenchPublicDataset(input_root).get_task("task_1")
            diagnostics = TaskDiagnostics(
                task=task,
                logs_dir=logs_dir,
                databao_env=DatabaoEnvironment("http://localhost:8080/v1", "key", "model"),
            )

            diagnostics.checkpoint("databao_frame_received", row_count=2, column_count=1)
            payload = json.loads((logs_dir / "task_1.progress.json").read_text(encoding="utf-8"))

            self.assertEqual(payload["event"], "checkpoint")
            self.assertEqual(payload["active_stage"], "databao_frame_received")
            self.assertEqual(payload["metadata"], {"row_count": 2, "column_count": 1})

    def test_non_dataframe_result_does_not_write_prediction_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            output_root = root / "output"
            logs_dir = root / "logs"
            write_task(input_root)
            task = DABenchPublicDataset(input_root).get_task("task_1")

            artifact = run_databao_task(
                task=task,
                output_root=output_root,
                logs_dir=logs_dir,
                databao_env=DatabaoEnvironment("http://localhost:8080/v1", "key", "model"),
                agent_builder=make_agent_builder(None),
                answer_finalizer=no_answer_finalizer,
                structured_planner=no_structured_planner,
            )

            self.assertFalse(artifact.succeeded)
            self.assertIsNone(artifact.prediction_csv_path)
            self.assertFalse((output_root / "task_1" / "prediction.csv").exists())
            self.assertTrue((logs_dir / "task_1.json").exists())

    def test_postprocessor_resolves_link_identifier_to_display_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(
                input_root,
                question="Which event has the lowest cost?",
            )
            context_dir = input_root / "task_1" / "context"
            json_dir = context_dir / "json"
            json_dir.mkdir()
            (json_dir / "event.json").write_text(
                json.dumps(
                    {
                        "table": "event",
                        "records": [
                            {"event_id": "recA", "event_name": "September Speaker"},
                            {"event_id": "recB", "event_name": "October Speaker"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            task = DABenchPublicDataset(input_root).get_task("task_1")

            frame, report = postprocess_answer_table(
                task,
                pd.DataFrame({"link_to_event": ["recA", "recB"]}),
            )

            self.assertTrue(report.applied)
            self.assertEqual(
                frame.to_dict(orient="list"),
                {"event_name": ["September Speaker", "October Speaker"]},
            )

    def test_superlative_verifier_generates_candidate_metric_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(
                input_root,
                question="Which event has the lowest cost?",
            )
            task = DABenchPublicDataset(input_root).get_task("task_1")

            candidates = generate_verifier_candidate_frames(
                task,
                pd.DataFrame(
                    {
                        "event_name": ["A", "B", "C"],
                        "expense_description": ["Food", "Parking", "Supplies"],
                        "total_cost": [20.0, 10.0, 30.0],
                    }
                ),
            )

            self.assertTrue(candidates)
            frame = candidates[0][0]
            self.assertEqual(frame.to_dict(orient="list"), {"event_name": ["B"]})

    def test_superlative_verifier_generates_candidate_when_rows_already_filtered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(
                input_root,
                question="Which event has the lowest cost?",
            )
            task = DABenchPublicDataset(input_root).get_task("task_1")

            candidates = generate_verifier_candidate_frames(
                task,
                pd.DataFrame(
                    {
                        "event_name": ["A", "B"],
                        "cost": [6.0, 6.0],
                    }
                ),
            )

            self.assertTrue(candidates)
            frame = candidates[0][0]
            self.assertEqual(frame.to_dict(orient="list"), {"event_name": ["A", "B"]})

    def test_context_chain_superlative_verifier_generates_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(
                input_root,
                question="Which event has the lowest cost?",
            )
            context_dir = input_root / "task_1" / "context"
            json_dir = context_dir / "json"
            csv_dir = context_dir / "csv"
            json_dir.mkdir()
            csv_dir.mkdir()
            (json_dir / "event.json").write_text(
                json.dumps(
                    {
                        "table": "event",
                        "records": [
                            {"event_id": "recA", "event_name": "Alpha", "status": "Open"},
                            {"event_id": "recB", "event_name": "Beta", "status": "Closed"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (csv_dir / "budget.csv").write_text(
                "budget_id,spent,link_to_event\nrecBudgetA,0,recA\nrecBudgetB,0,recB\n",
                encoding="utf-8",
            )
            (json_dir / "expense.json").write_text(
                json.dumps(
                    {
                        "table": "expense",
                        "records": [
                            {"expense_id": "recExpenseA", "cost": 20.0, "link_to_budget": "recBudgetA"},
                            {"expense_id": "recExpenseB", "cost": 6.0, "link_to_budget": "recBudgetB"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            task = DABenchPublicDataset(input_root).get_task("task_1")

            candidates = generate_verifier_candidate_frames(
                task,
                pd.DataFrame({"event_name": ["Alpha"], "total_cost": [20.0]}),
            )

            self.assertTrue(candidates)
            frame = candidates[0][0]
            self.assertEqual(frame.to_dict(orient="list"), {"event_name": ["Beta"]})

    def test_question_column_pruner_keeps_requested_date_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="What date were the state dues paid?")
            task = DABenchPublicDataset(input_root).get_task("task_1")

            frame, transforms = apply_question_column_pruner(
                task,
                pd.DataFrame(
                    {
                        "record_id": ["recA"],
                        "date": ["2021-05-20"],
                        "amount": [50],
                    }
                ),
            )

            self.assertEqual(frame.to_dict(orient="list"), {"date": ["2021-05-20"]})
            self.assertEqual(transforms[0]["kind"], "question_column_pruning")

    def test_question_column_pruner_keeps_requested_identifier_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="List all records by identifier over the threshold.")
            task = DABenchPublicDataset(input_root).get_task("task_1")

            frame, _ = apply_question_column_pruner(
                task,
                pd.DataFrame(
                    {
                        "record_id": [101, 102],
                        "date": ["2020-01-01", "2020-01-02"],
                        "amount": [500, 600],
                    }
                ),
            )

            self.assertEqual(frame.to_dict(orient="list"), {"record_id": [101, 102]})

    def test_aggregate_ratio_verifier_rewrites_generic_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="How many times is the selected item count compared to the total count?")
            task = DABenchPublicDataset(input_root).get_task("task_1")

            frame, transforms = apply_aggregate_ratio_verifier(
                task,
                pd.DataFrame({"item_count": [3], "total_count": [8]}),
            )

            self.assertAlmostEqual(frame.iloc[0, 0], 0.375)
            self.assertEqual(transforms[0]["mode"], "ratio")

    def test_aggregate_ratio_verifier_rewrites_percentage_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="What percentage of matching records are selected?")
            task = DABenchPublicDataset(input_root).get_task("task_1")

            frame, transforms = apply_aggregate_ratio_verifier(
                task,
                pd.DataFrame({"total_count": [23], "matching_count": [12]}),
            )

            self.assertAlmostEqual(frame.iloc[0, 0], 52.17391304347826)
            self.assertEqual(transforms[0]["mode"], "percentage")

    def test_databao_observed_detail_aggregate_counts_filtered_detail_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="How many rows match the selected condition?")
            task = DABenchPublicDataset(input_root).get_task("task_1")
            frame = pd.DataFrame({"record_id": [1, 2, 3], "status": ["A", "A", "A"]})
            frame.attrs["databao_code"] = "SELECT record_id, status FROM items WHERE status = 'A'"

            output, transforms = apply_databao_observed_detail_aggregate_compactor(task, frame)

            self.assertEqual(output.to_dict(orient="list"), {"count": [3]})
            self.assertEqual(transforms[0]["operation"], "count")

    def test_databao_observed_detail_aggregate_counts_distinct_display_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="How many entries match the selected condition?")
            task = DABenchPublicDataset(input_root).get_task("task_1")
            frame = pd.DataFrame({"name": ["A", "A", "B", "B", "C"]})
            frame.attrs["databao_code"] = "SELECT name FROM items WHERE status = 'A'"

            output, transforms = apply_databao_observed_detail_aggregate_compactor(task, frame)

            self.assertEqual(output.to_dict(orient="list"), {"count": [3]})
            self.assertEqual(transforms[0]["count_basis"], "distinct_column")

    def test_databao_observed_detail_aggregate_counts_distinct_ids_from_filtered_detail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="Among matching patients, how many have an abnormal measurement?")
            task = DABenchPublicDataset(input_root).get_task("task_1")
            frame = pd.DataFrame(
                {
                    "ID": [1, 1, 2, 3],
                    "FG": [36.1, 36.1, 43.8, 12.0],
                    "SEX": ["M", "M", "M", "M"],
                    "WBC": [7.7, 7.7, 9.5, 8.0],
                }
            )
            frame.attrs["databao_code"] = (
                "SELECT l.ID, l.FG, p.SEX, l.WBC FROM lab l JOIN patient p ON l.ID = p.ID "
                "WHERE p.SEX = 'M' AND l.WBC >= 4.5 AND (l.FG < 2.0 OR l.FG > 4.0)"
            )

            output, transforms = apply_databao_observed_detail_aggregate_compactor(task, frame)

            self.assertEqual(output.to_dict(orient="list"), {"count": [3]})
            self.assertEqual(transforms[0]["count_column"], "ID")

    def test_databao_observed_detail_aggregate_averages_filtered_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="What is the average number of bonds for the selected atoms?")
            task = DABenchPublicDataset(input_root).get_task("task_1")
            frame = pd.DataFrame({"atom_id": ["a", "b", "c"], "unique_bonds": [1, 2, 3]})
            frame.attrs["databao_code"] = (
                "SELECT atom_id, COUNT(DISTINCT bond_id) AS unique_bonds "
                "FROM connected WHERE element = 'x' GROUP BY atom_id"
            )

            output, transforms = apply_databao_observed_detail_aggregate_compactor(task, frame)

            self.assertEqual(output.to_dict(orient="list"), {"avg_unique_bonds": [2.0]})
            self.assertEqual(transforms[0]["operation"], "average")

    def test_databao_observed_detail_aggregate_keeps_existing_multi_metric_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="What are the average up votes and average age?")
            task = DABenchPublicDataset(input_root).get_task("task_1")
            frame = pd.DataFrame({"avg_up_votes": [182.2], "avg_age": [34.1]})
            frame.attrs["databao_code"] = "SELECT AVG(up_votes) AS avg_up_votes, AVG(age) AS avg_age FROM users"

            output, transforms = apply_databao_observed_detail_aggregate_compactor(task, frame)

            self.assertTrue(output.equals(frame))
            self.assertEqual(transforms, [])

    def test_databao_observed_detail_aggregate_does_not_override_identifier_questions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="Which post has the most answers count? State the post ID.")
            task = DABenchPublicDataset(input_root).get_task("task_1")
            frame = pd.DataFrame({"Id": [351], "AnswerCount": [12]})
            frame.attrs["databao_code"] = "SELECT Id, AnswerCount FROM posts ORDER BY AnswerCount DESC LIMIT 1"

            output, transforms = apply_databao_observed_detail_aggregate_compactor(task, frame)

            self.assertTrue(output.equals(frame))
            self.assertEqual(transforms, [])

    def test_answer_column_verifier_keeps_generic_requested_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"

            write_task(input_root, task_id="task_phone", question="What is the telephone number for the record?")
            phone_task = DABenchPublicDataset(input_root).get_task("task_phone")
            phone_frame, _ = apply_answer_column_verifier(
                phone_task,
                pd.DataFrame({"cds": [1], "AvgScrRead": [370], "Phone": ["(559) 248-5100"]}),
            )
            self.assertEqual(phone_frame.to_dict(orient="list"), {"Phone": ["(559) 248-5100"]})

            write_task(input_root, task_id="task_identifier", question="State the identifier for the selected row.")
            identifier_task = DABenchPublicDataset(input_root).get_task("task_identifier")
            identifier_frame, _ = apply_answer_column_verifier(
                identifier_task,
                pd.DataFrame({"record_id": [351], "metric_value": [None]}),
            )
            self.assertEqual(identifier_frame.to_dict(orient="list"), {"record_id": [351]})

            write_task(input_root, task_id="task_text", question="What is the text for the selected row?")
            text_task = DABenchPublicDataset(input_root).get_task("task_text")
            text_frame, _ = apply_answer_column_verifier(
                text_task,
                pd.DataFrame({"Id": [1], "Score": [14], "Text": ["hello"]}),
            )
            self.assertEqual(text_frame.to_dict(orient="list"), {"Text": ["hello"]})

            write_task(
                input_root,
                task_id="task_comment_superlative",
                question="Among the rows, what is the comment with the highest score?",
            )
            comment_task = DABenchPublicDataset(input_root).get_task("task_comment_superlative")
            comment_frame, _ = apply_answer_column_verifier(
                comment_task,
                pd.DataFrame(
                    {
                        "Id": [1],
                        "Score": [14],
                        "CommentCount": [1],
                        "Text": ["best comment"],
                        "UserId": [88],
                    }
                ),
            )
            self.assertEqual(comment_frame.to_dict(orient="list"), {"Text": ["best comment"]})

            write_task(
                input_root,
                task_id="task_post_identifier",
                question="Which record has the most answers count? State the record ID.",
            )
            post_id_task = DABenchPublicDataset(input_root).get_task("task_post_identifier")
            post_id_frame, _ = apply_answer_column_verifier(
                post_id_task,
                pd.DataFrame({"Id": [351], "OwnerUserId": [16], "AnswerCount": [12], "Title": [""]}),
            )
            self.assertEqual(post_id_frame.to_dict(orient="list"), {"Id": [351]})

            write_task(input_root, task_id="task_post_id", question="State the post ID.")
            post_id_task = DABenchPublicDataset(input_root).get_task("task_post_id")
            post_id_frame, _ = apply_answer_column_verifier(
                post_id_task,
                pd.DataFrame({"Id": [351], "AcceptedAnswerId": [99], "AnswerCount": [12]}),
            )
            self.assertEqual(post_id_frame.to_dict(orient="list"), {"Id": [351]})

            write_task(input_root, task_id="task_display", question="Identify the ViewCount and DisplayName.")
            views_task = DABenchPublicDataset(input_root).get_task("task_display")
            views_frame, _ = apply_answer_column_verifier(
                views_task,
                pd.DataFrame({"Title": ["A"], "ViewCount": [1708], "DisplayName": ["mbq"], "UserId": [88]}),
            )
            self.assertEqual(views_frame.to_dict(orient="list"), {"ViewCount": [1708], "DisplayName": ["mbq"]})

            write_task(
                input_root,
                task_id="task_count",
                question="How many members attended the selected event?",
            )
            count_task = DABenchPublicDataset(input_root).get_task("task_count")
            count_frame, _ = apply_answer_column_verifier(
                count_task,
                pd.DataFrame(
                    {
                        "event_id": ["recEvent"],
                        "event_name": ["Selected Event"],
                        "attendance_count": [17],
                    }
                ),
            )
            self.assertEqual(count_frame.to_dict(orient="list"), {"attendance_count": [17]})

            write_task(
                input_root,
                task_id="task_total_count",
                question="Calculate the total atoms with the selected property.",
            )
            total_count_task = DABenchPublicDataset(input_root).get_task("task_total_count")
            total_count_frame, _ = apply_answer_column_verifier(
                total_count_task,
                pd.DataFrame({"element": ["p"], "atom_count": [1]}),
            )
            self.assertEqual(total_count_frame.to_dict(orient="list"), {"atom_count": [1]})

            write_task(
                input_root,
                task_id="task_total_value",
                question="Identify the category and total value for the selected group.",
            )
            total_value_task = DABenchPublicDataset(input_root).get_task("task_total_value")
            total_value_frame, _ = apply_answer_column_verifier(
                total_value_task,
                pd.DataFrame({"category": ["A"], "total_value": [10.5]}),
            )
            self.assertEqual(total_value_frame.to_dict(orient="list"), {"category": ["A"], "total_value": [10.5]})

            write_task(input_root, task_id="task_attribute", question="Provide the eye colour of the selected record.")
            attribute_task = DABenchPublicDataset(input_root).get_task("task_attribute")
            attribute_frame, _ = apply_answer_column_verifier(
                attribute_task,
                pd.DataFrame({"name": ["A"], "full_name": ["A B"], "eye_colour": ["Brown"]}),
            )
            self.assertEqual(attribute_frame.to_dict(orient="list"), {"eye_colour": ["Brown"]})

            write_task(input_root, task_id="task_entity", question="Which record was selected?")
            entity_task = DABenchPublicDataset(input_root).get_task("task_entity")
            entity_frame, _ = apply_answer_column_verifier(
                entity_task,
                pd.DataFrame({"name": ["Selected Name"], "positionText": ["2"], "position": [2]}),
            )
            self.assertEqual(entity_frame.to_dict(orient="list"), {"name": ["Selected Name"]})

            write_task(input_root, task_id="task_entity_order", question="Which event was selected?")
            entity_order_task = DABenchPublicDataset(input_root).get_task("task_entity_order")
            entity_order_frame, _ = apply_answer_column_verifier(
                entity_order_task,
                pd.DataFrame({"positionText": ["2"], "name": ["Selected Event"], "date": ["2026-01-01"]}),
            )
            self.assertEqual(entity_order_frame.to_dict(orient="list"), {"name": ["Selected Event"]})

            write_task(
                input_root,
                task_id="task_finish_time",
                question="What's the finish time for the selected row?",
            )
            time_task = DABenchPublicDataset(input_root).get_task("task_finish_time")
            time_frame, _ = apply_answer_column_verifier(
                time_task,
                pd.DataFrame({"time": ["+16.445"], "fastestLapTime": ["1:35.123"]}),
            )
            self.assertEqual(time_frame.to_dict(orient="list"), {"time": ["+16.445"]})

            write_task(input_root, task_id="task_number", question="What is his number of the selected driver?")
            number_task = DABenchPublicDataset(input_root).get_task("task_number")
            number_frame, _ = apply_answer_column_verifier(
                number_task,
                pd.DataFrame({"number": [3], "q3": ["1:54.455"]}),
            )
            self.assertEqual(number_frame.to_dict(orient="list"), {"number": [3]})

    def test_identifier_resolution_prefers_camel_case_display_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="Name the user for the selected row.")
            context_dir = input_root / "task_1" / "context"
            json_dir = context_dir / "json"
            json_dir.mkdir()
            (json_dir / "users.json").write_text(
                json.dumps([{"Id": 88, "DisplayName": "mbq"}]),
                encoding="utf-8",
            )
            task = DABenchPublicDataset(input_root).get_task("task_1")

            frame, report = postprocess_answer_table(
                task,
                pd.DataFrame({"UserId": [88]}),
            )

        self.assertEqual(frame.to_dict(orient="list"), {"DisplayName": ["mbq"]})
        self.assertTrue(any(transform["kind"] == "identifier_resolution" for transform in report.transformations))

    def test_identifier_resolution_prefers_user_display_table_over_helper_titles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="Identify the total views on the selected post. Name the user who posted it last time.")
            context_dir = input_root / "task_1" / "context"
            json_dir = context_dir / "json"
            json_dir.mkdir()
            (json_dir / "posts.json").write_text(
                json.dumps(
                    [
                        {"Id": 10, "OwnerUserId": 37, "LastEditorUserId": 88, "Title": "Computer Game Datasets"},
                        {"Id": 11, "OwnerUserId": 37, "LastEditorUserId": 88, "Title": "Another Post"},
                    ]
                ),
                encoding="utf-8",
            )
            (json_dir / "users.json").write_text(
                json.dumps([{"Id": 37, "DisplayName": "owner"}, {"Id": 88, "DisplayName": "mbq"}]),
                encoding="utf-8",
            )
            task = DABenchPublicDataset(input_root).get_task("task_1")

            frame, report = postprocess_answer_table(
                task,
                pd.DataFrame(
                    {
                        "OwnerUserId": [37],
                        "OwnerDisplayName": [""],
                        "LastEditorUserId": [88.0],
                        "LastEditorDisplayName": [""],
                        "ViewCount": [1708],
                    }
                ),
            )

        self.assertEqual(frame.to_dict(orient="list"), {"LastEditorDisplayName": ["mbq"], "ViewCount": [1708]})
        self.assertTrue(
            any(
                transform.get("source_column") == "LastEditorUserId"
                and transform.get("lookup_table") == "json/users.json"
                for transform in report.transformations
            )
        )

    def test_identifier_resolution_rejects_conflicting_camel_case_id_hints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="Which event was selected?")
            context_dir = input_root / "task_1" / "context"
            csv_dir = context_dir / "csv"
            csv_dir.mkdir()
            (csv_dir / "standings.csv").write_text(
                "driverStandingsId,positionText\n1,first\n",
                encoding="utf-8",
            )
            (csv_dir / "events.csv").write_text(
                "eventId,name\n10,Selected Event\n",
                encoding="utf-8",
            )
            task = DABenchPublicDataset(input_root).get_task("task_1")

            frame, report = postprocess_answer_table(
                task,
                pd.DataFrame({"driverId": [1], "eventId": [10]}),
            )

        self.assertEqual(frame.to_dict(orient="list"), {"name": ["Selected Event"]})
        self.assertTrue(any(transform["kind"] == "identifier_resolution" for transform in report.transformations))
        self.assertFalse(
            any(
                transform.get("source_column") == "driverId"
                and transform.get("lookup_id_column") == "driverStandingsId"
                for transform in report.transformations
            )
        )

    def test_identifier_resolution_does_not_map_unrelated_id_families_to_users(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="Return the title and user display for the selected row.")
            context_dir = input_root / "task_1" / "context"
            json_dir = context_dir / "json"
            json_dir.mkdir()
            (json_dir / "users.json").write_text(
                json.dumps([{"Id": 1, "DisplayName": "wrong"}, {"Id": 88, "DisplayName": "mbq"}]),
                encoding="utf-8",
            )
            task = DABenchPublicDataset(input_root).get_task("task_1")

            frame, report = postprocess_answer_table(
                task,
                pd.DataFrame(
                    {
                        "Id": [1],
                        "Title": ["Existing Title"],
                        "PostHistoryTypeId": [1],
                        "UserId": [88],
                    }
                ),
            )

        self.assertNotIn("wrong", str(frame.to_dict(orient="list")))
        self.assertFalse(
            any(transform.get("source_column") == "PostHistoryTypeId" for transform in report.transformations)
        )
        self.assertTrue(any(transform.get("source_column") == "UserId" for transform in report.transformations))

    def test_postprocessor_removes_exact_duplicate_answer_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="List the school name and funding type.")
            task = DABenchPublicDataset(input_root).get_task("task_1")

            frame, report = postprocess_answer_table(
                task,
                pd.DataFrame(
                    {
                        "School Name": ["A", "A", "B"],
                        "Funding Type": ["Direct", "Direct", "Indirect"],
                    }
                ),
            )

        self.assertEqual(
            frame.to_dict(orient="list"),
            {"School Name": ["A", "B"], "Funding Type": ["Direct", "Indirect"]},
        )
        self.assertTrue(any(transform["kind"] == "duplicate_answer_row_removal" for transform in report.transformations))

    def test_ranker_prefers_display_candidate_over_id_helper_with_display(self) -> None:
        contract = AnswerContract(
            kind="multi_attribute",
            expected_columns=("DisplayName", "ViewCount"),
            max_rows=1,
            max_columns=3,
            allow_empty=False,
            reason="test",
        )
        raw = Candidate(
            frame=pd.DataFrame({"Id": [88], "DisplayName": ["mbq"], "Reputation": [14082]}),
            source="databao_raw",
            confidence=0.55,
            diagnostics={},
        )
        postprocessed = Candidate(
            frame=pd.DataFrame({"DisplayName": ["mbq"]}),
            source="databao_raw_postprocessed",
            confidence=0.63,
            diagnostics={},
            transformations=({"kind": "answer_column_verification"},),
        )

        selected, report = rank_candidates(
            candidates=[raw, postprocessed],
            contract=contract,
            question="Identify the total views and name the user.",
            retrieved_context=None,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.source, "databao_raw_postprocessed")
        raw_score = next(score for score in report.candidate_scores if score.source == "databao_raw")
        self.assertIn("id_helper_with_display_answer", raw_score.reasons)

    def test_ranker_prefers_identifier_only_candidate_for_explicit_id_question(self) -> None:
        contract = AnswerContract(
            kind="attribute_lookup",
            expected_columns=("Id",),
            max_rows=1,
            max_columns=3,
            allow_empty=False,
            reason="test",
        )
        raw = Candidate(
            frame=pd.DataFrame({"Id": [351], "AnswerCount": [12], "DisplayName": ["slashnick"]}),
            source="databao_raw",
            confidence=0.55,
            diagnostics={},
        )
        postprocessed = Candidate(
            frame=pd.DataFrame({"Id": [351]}),
            source="databao_raw_postprocessed",
            confidence=0.63,
            diagnostics={},
            transformations=({"kind": "answer_column_verification"},),
        )

        selected, report = rank_candidates(
            candidates=[raw, postprocessed],
            contract=contract,
            question="Which row has the highest count? State the ID.",
            retrieved_context=None,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.source, "databao_raw_postprocessed")
        raw_score = next(score for score in report.candidate_scores if score.source == "databao_raw")
        selected_score = next(score for score in report.candidate_scores if score.source == "databao_raw_postprocessed")
        self.assertIn("explicit_identifier_answer_has_extra_columns", raw_score.reasons)
        self.assertIn("explicit_identifier_answer_only", selected_score.reasons)

    def test_ranker_prefers_text_only_candidate_for_comment_answer(self) -> None:
        contract = AnswerContract(
            kind="attribute_lookup",
            expected_columns=("Text",),
            max_rows=1,
            max_columns=5,
            allow_empty=False,
            reason="test",
        )
        raw = Candidate(
            frame=pd.DataFrame(
                {
                    "Score": [14],
                    "Text": ["answer text"],
                    "ViewCount": [150],
                    "Title": ["source title"],
                }
            ),
            source="databao_raw",
            confidence=0.55,
            diagnostics={},
        )
        postprocessed = Candidate(
            frame=pd.DataFrame({"Text": ["answer text"]}),
            source="databao_raw_postprocessed",
            confidence=0.63,
            diagnostics={},
            transformations=({"kind": "answer_column_verification"},),
        )

        selected, report = rank_candidates(
            candidates=[raw, postprocessed],
            contract=contract,
            question="What is the comment with the highest score?",
            retrieved_context=None,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.source, "databao_raw_postprocessed")
        raw_score = next(score for score in report.candidate_scores if score.source == "databao_raw")
        selected_score = next(score for score in report.candidate_scores if score.source == "databao_raw_postprocessed")
        self.assertIn("text_answer_has_extra_columns", raw_score.reasons)
        self.assertIn("text_answer_columns_only", selected_score.reasons)

    def test_ranker_accepts_context_superlative_tie_candidate_with_metric_evidence(self) -> None:
        contract = AnswerContract(
            kind="multi_attribute",
            expected_columns=("event_name", "cost"),
            max_rows=None,
            max_columns=2,
            allow_empty=False,
            reason="test",
        )
        raw = Candidate(
            frame=pd.DataFrame({"event_name": ["October Speaker"], "cost": [6.0]}),
            source="databao_raw",
            confidence=0.55,
            diagnostics={},
        )
        verifier = Candidate(
            frame=pd.DataFrame({"event_name": ["November Speaker", "October Speaker", "September Speaker"]}),
            source="databao_raw_postprocessed_verifier",
            confidence=0.42,
            diagnostics={},
            transformations=(
                {
                    "kind": "context_superlative_verification",
                    "direction": "min",
                    "metric_column": "cost",
                    "metric_score": 180,
                    "output_rows": 3,
                },
            ),
        )

        selected, report = rank_candidates(
            candidates=[raw, verifier],
            contract=contract,
            question="Which event has the lowest cost?",
            retrieved_context=None,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.source, "databao_raw_postprocessed_verifier")
        selected_score = next(
            score for score in report.candidate_scores if score.source == "databao_raw_postprocessed_verifier"
        )
        self.assertIn("context_superlative_metric_evidence", selected_score.reasons)

    def test_root_level_csv_is_loaded_and_registered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context_dir = Path(tmp)
            (context_dir / "patient_sex.csv").write_text("ID,SEX\n1,M\n", encoding="utf-8")
            domain = FakeDomain()

            summary = register_context_sources(domain, context_dir)
            tables = load_context_tables(context_dir)

            self.assertEqual(summary["csv_files"], ["patient_sex.csv"])
            self.assertEqual([table.name for table in tables], ["patient_sex"])
            self.assertEqual(domain.dfs[0]["name"], "csv_patient_sex")

    def test_generic_document_materializer_extracts_record_like_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context_dir = Path(tmp)
            doc_dir = context_dir / "doc"
            doc_dir.mkdir()
            (doc_dir / "records.md").write_text(
                "Record Alpha (Registry ID: recAlpha) is known as North Sample. "
                "Its status is active, amount is 42.50, and it is related to recBeta.\n\n"
                "Reference code 200 is called South Sample. Its category is archived.",
                encoding="utf-8",
            )

            tables = generic_document_tables(context_dir)
            reasoning_tables = document_records_for_reasoning(context_dir)

            self.assertEqual(len(tables), 1)
            table = tables[0]
            self.assertEqual(table.name, "document_records")
            self.assertEqual(table.metadata["strategy_name"], "document_records_for_agent")
            self.assertGreaterEqual(len(table.frame), 2)
            self.assertIn("record_id", table.frame.columns)
            self.assertIn("name", table.frame.columns)
            self.assertIn("status", table.frame.columns)
            self.assertIn("amount", table.frame.columns)
            self.assertNotIn("evidence_span", table.frame.columns)
            self.assertNotIn("source_doc", table.frame.columns)
            self.assertEqual(len(reasoning_tables), 1)
            self.assertIn("evidence_span", reasoning_tables[0].frame.columns)

    def test_src_code_has_no_public_domain_memory_terms(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        banned_terms = [
            "superhero",
            "races",
            "legalities",
            "molecule",
            "Commander",
            "Club T-Shirts",
            "Australian Grand Prix",
            "publisher affiliation",
        ]
        violations: list[str] = []
        for path in (repo_root / "src").rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            lowered = text.lower()
            for term in banned_terms:
                if term.lower() in lowered:
                    violations.append(f"{path.relative_to(repo_root)} contains {term!r}")

        self.assertEqual(violations, [])

    def test_runner_prompt_has_no_scoring_aware_terms(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        text = (repo_root / "src" / "data_agent_baseline" / "run" / "databao_demo.py").read_text(
            encoding="utf-8",
            errors="replace",
        )
        banned_terms = [
            "official scoring",
            "extra columns are penalized",
            "column names ignored",
            "gold",
        ]

        violations = [term for term in banned_terms if term in text.lower()]

        self.assertEqual(violations, [])

    def test_runner_has_no_task_or_difficulty_specific_branches(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        text = (repo_root / "src" / "data_agent_baseline" / "run" / "databao_demo.py").read_text(
            encoding="utf-8",
            errors="replace",
        )
        banned_snippets = [
            "if task.task_id ==",
            "elif task.task_id ==",
            "if task_id ==",
            "elif task_id ==",
            "if task.difficulty ==",
            "elif task.difficulty ==",
            "difficulty ==",
            "difficulty !=",
            "doc_name ==",
            "file_stem ==",
            "materializers[",
        ]

        violations = [snippet for snippet in banned_snippets if snippet in text]

        self.assertEqual(violations, [])

    def test_question_features_treat_weak_metric_terms_as_schema_signals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="Return display name and view count for matching rows.")
            csv_dir = input_root / "task_1" / "context" / "csv"
            csv_dir.mkdir(parents=True)
            (csv_dir / "records.csv").write_text(
                "DisplayName,ViewCount,status\nA,12,ok\nB,10,ok\n",
                encoding="utf-8",
            )
            task = DABenchPublicDataset(input_root).get_task("task_1")
            task_context = build_task_context(task)

            weak_without_schema = extract_question_features("What is the number?")
            with_schema = extract_question_features(task.question, task_context.context_tables)

            self.assertFalse(weak_without_schema.asks_aggregation)
            self.assertIn("view count", with_schema.weak_terms)
            self.assertTrue(
                any(
                    evidence["phrase"] == "view count" and evidence["schema_columns"]
                    for evidence in with_schema.evidence
                )
            )
            self.assertTrue(with_schema.asks_multi_attribute)

    def test_answer_contract_rejects_source_name_singleton(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="How many rows are selected?")
            task = DABenchPublicDataset(input_root).get_task("task_1")
            task_context = build_task_context(task)
            intent = infer_task_intent(task_context)
            contract = infer_answer_contract(task_context, intent)

            report = validate_answer_contract(
                contract,
                pd.DataFrame({"name": ["csv_Laboratory"]}),
                candidate_source="databao",
            )

            self.assertFalse(report.valid)
            self.assertTrue(report.should_repair)

    def test_answer_contract_does_not_scalarize_ordinary_which_question(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="Which id and name values match the condition?")
            csv_dir = input_root / "task_1" / "context" / "csv"
            csv_dir.mkdir(parents=True)
            (csv_dir / "records.csv").write_text(
                "id,name,status\n1,A,ok\n2,B,ok\n",
                encoding="utf-8",
            )
            task = DABenchPublicDataset(input_root).get_task("task_1")
            task_context = build_task_context(task)
            contract = infer_answer_contract(task_context, infer_task_intent(task_context))

            self.assertIn(contract.kind, {"two_attribute", "multi_attribute"})
            self.assertIsNone(contract.max_rows)
            self.assertEqual(contract.max_columns, 2)

    def test_answer_contract_treats_view_count_as_attribute_not_count_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="Return display name and view count for matching rows.")
            csv_dir = input_root / "task_1" / "context" / "csv"
            csv_dir.mkdir(parents=True)
            (csv_dir / "records.csv").write_text(
                "DisplayName,ViewCount,status\nA,12,ok\nB,10,ok\n",
                encoding="utf-8",
            )
            task = DABenchPublicDataset(input_root).get_task("task_1")
            task_context = build_task_context(task)
            contract = infer_answer_contract(task_context, infer_task_intent(task_context))

            self.assertIn(contract.kind, {"two_attribute", "multi_attribute"})
            self.assertIsNone(contract.max_rows)
            self.assertGreaterEqual(contract.max_columns or 0, 2)

    def test_answer_contract_allows_entity_plus_metric_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="Which person has the highest total cost?")
            csv_dir = input_root / "task_1" / "context" / "csv"
            csv_dir.mkdir(parents=True)
            (csv_dir / "records.csv").write_text(
                "first_name,last_name,total_cost\nA,B,10\n",
                encoding="utf-8",
            )
            task = DABenchPublicDataset(input_root).get_task("task_1")
            task_context = build_task_context(task)
            contract = infer_answer_contract(task_context, infer_task_intent(task_context))

            self.assertEqual(contract.kind, "multi_attribute")
            self.assertIsNone(contract.max_rows)
            self.assertGreaterEqual(contract.max_columns or 0, 3)

    def test_answer_contract_does_not_fix_entity_list_columns_without_requested_column_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="Which records match?")
            csv_dir = input_root / "task_1" / "context" / "csv"
            csv_dir.mkdir(parents=True)
            (csv_dir / "records.csv").write_text(
                "id,name,status\n1,A,ok\n2,B,ok\n",
                encoding="utf-8",
            )
            task = DABenchPublicDataset(input_root).get_task("task_1")
            task_context = build_task_context(task)
            contract = infer_answer_contract(task_context, infer_task_intent(task_context))

            self.assertEqual(contract.kind, "entity_list")
            self.assertIsNone(contract.max_rows)
            self.assertIsNone(contract.max_columns)

    def test_final_answer_shape_guard_removes_metadata_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="What is the final answer?")
            task = DABenchPublicDataset(input_root).get_task("task_1")
            contract = AnswerContract(
                kind="table",
                expected_columns=(),
                max_rows=None,
                max_columns=None,
                allow_empty=False,
                reason="test",
            )

            guarded, report = final_answer_shape_guard(
                task=task,
                frame=pd.DataFrame(
                    {
                        "answer": ["A"],
                        "source_doc": ["doc.md"],
                        "evidence_span": ["debug text"],
                        "confidence": [0.9],
                    }
                ),
                contract=contract,
                candidate_source="databao_raw",
            )

            self.assertEqual(guarded.to_dict(orient="list"), {"answer": ["A"]})
            self.assertTrue(report.to_dict()["metadata_columns_removed"])

    def test_final_answer_shape_guard_removes_blank_and_duplicate_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="What is the name?")
            task = DABenchPublicDataset(input_root).get_task("task_1")
            contract = AnswerContract(
                kind="table",
                expected_columns=(),
                max_rows=None,
                max_columns=None,
                allow_empty=False,
                reason="test",
            )

            guarded, report = final_answer_shape_guard(
                task=task,
                frame=pd.DataFrame(
                    {
                        "name": ["Business"],
                        "name_2": ["Business"],
                        "title": [""],
                        "year": [None],
                    }
                ),
                contract=contract,
                candidate_source="databao_raw",
            )

            self.assertEqual(guarded.to_dict(orient="list"), {"name": ["Business"]})
            self.assertIn("name_2", report.removed_columns)
            self.assertIn("title", report.removed_columns)
            self.assertIn("year", report.removed_columns)

    def test_final_answer_shape_guard_enforces_scalar_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="How many rows match?")
            task = DABenchPublicDataset(input_root).get_task("task_1")
            contract = AnswerContract(
                kind="scalar",
                expected_columns=("answer",),
                max_rows=1,
                max_columns=1,
                allow_empty=False,
                reason="test",
            )

            guarded, _ = final_answer_shape_guard(
                task=task,
                frame=pd.DataFrame({"answer": [3, 4], "total": [10, 11], "source": ["x", "y"]}),
                contract=contract,
                candidate_source="databao_raw",
            )

            self.assertEqual(guarded.shape, (1, 1))
            self.assertEqual(guarded.to_dict(orient="list"), {"answer": [3]})

    def test_final_answer_shape_guard_does_not_infer_list_shape_without_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="List matching names.")
            task = DABenchPublicDataset(input_root).get_task("task_1")
            contract = AnswerContract(
                kind="table",
                expected_columns=(),
                max_rows=None,
                max_columns=None,
                allow_empty=False,
                reason="test",
            )

            guarded, _ = final_answer_shape_guard(
                task=task,
                frame=pd.DataFrame({"name": ["A", "B"], "score": [1, 2], "source": ["x", "y"]}),
                contract=contract,
                candidate_source="databao_raw",
            )

            self.assertEqual(
                guarded.to_dict(orient="list"),
                {"name": ["A", "B"], "score": [1, 2], "source": ["x", "y"]},
            )

    def test_final_answer_shape_guard_preserves_explicit_table_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="Return the name and status for matching rows.")
            task = DABenchPublicDataset(input_root).get_task("task_1")
            contract = AnswerContract(
                kind="attribute_lookup",
                expected_columns=("name", "status"),
                max_rows=None,
                max_columns=2,
                allow_empty=False,
                reason="test",
            )

            guarded, _ = final_answer_shape_guard(
                task=task,
                frame=pd.DataFrame(
                    {
                        "name": ["A"],
                        "status": ["active"],
                        "value": [10],
                        "confidence": [0.9],
                    }
                ),
                contract=contract,
                candidate_source="databao_raw",
            )

            self.assertEqual(guarded.to_dict(orient="list"), {"name": ["A"], "status": ["active"]})

    def test_final_answer_shape_guard_preserves_display_and_metric_for_multi_attribute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="Which person has the highest total cost?")
            task = DABenchPublicDataset(input_root).get_task("task_1")
            contract = AnswerContract(
                kind="multi_attribute",
                expected_columns=("total_cost",),
                max_rows=1,
                max_columns=3,
                allow_empty=False,
                reason="test",
            )

            guarded, _ = final_answer_shape_guard(
                task=task,
                frame=pd.DataFrame(
                    {
                        "first_name": ["A"],
                        "last_name": ["B"],
                        "total_cost": [10],
                        "helper_id": [1],
                    }
                ),
                contract=contract,
                candidate_source="databao_raw",
            )

            self.assertEqual(
                guarded.to_dict(orient="list"),
                {"total_cost": [10], "first_name": ["A"], "last_name": ["B"]},
            )

    def test_final_answer_shape_guard_preserves_multirow_attribute_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="Which id and name values match the condition?")
            task = DABenchPublicDataset(input_root).get_task("task_1")
            contract = AnswerContract(
                kind="two_attribute",
                expected_columns=("id", "name"),
                max_rows=None,
                max_columns=2,
                allow_empty=False,
                reason="test",
            )

            guarded, _ = final_answer_shape_guard(
                task=task,
                frame=pd.DataFrame(
                    {
                        "id": [1, 2],
                        "name": ["A", "B"],
                        "status": ["ok", "ok"],
                    }
                ),
                contract=contract,
                candidate_source="databao_raw",
            )

            self.assertEqual(guarded.to_dict(orient="list"), {"id": [1, 2], "name": ["A", "B"]})

    def test_final_answer_shape_guard_prefers_url_for_url_question(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="Which url should be returned?")
            task = DABenchPublicDataset(input_root).get_task("task_1")
            contract = AnswerContract(
                kind="entity_list",
                expected_columns=(),
                max_rows=None,
                max_columns=1,
                allow_empty=False,
                reason="test",
            )

            guarded, report = final_answer_shape_guard(
                task=task,
                frame=pd.DataFrame(
                    {
                        "record_id": ["r1", "r2"],
                        "url": ["https://a.example", "https://b.example"],
                    }
                ),
                contract=contract,
                candidate_source="databao_raw",
            )

            self.assertEqual(guarded.to_dict(orient="list"), {"url": ["https://a.example", "https://b.example"]})
            self.assertEqual(report.removed_rows, 0)
            self.assertEqual(report.removed_columns, ("record_id",))

    def test_final_answer_shape_guard_drops_id_helper_when_display_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="Which names match?")
            task = DABenchPublicDataset(input_root).get_task("task_1")
            contract = AnswerContract(
                kind="entity_list",
                expected_columns=(),
                max_rows=None,
                max_columns=None,
                allow_empty=False,
                reason="test",
            )

            guarded, report = final_answer_shape_guard(
                task=task,
                frame=pd.DataFrame({"rowId": [1, 2], "name": ["A", "B"]}),
                contract=contract,
                candidate_source="databao_raw",
            )

            self.assertEqual(
                guarded.to_dict(orient="list"),
                {"name": ["A", "B"]},
            )
            self.assertIn("rowId", report.removed_columns)

    def test_final_answer_shape_guard_uses_explicit_entity_list_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="List matching names.")
            task = DABenchPublicDataset(input_root).get_task("task_1")
            contract = AnswerContract(
                kind="entity_list",
                expected_columns=("name",),
                max_rows=None,
                max_columns=1,
                allow_empty=False,
                reason="test",
            )

            guarded, _ = final_answer_shape_guard(
                task=task,
                frame=pd.DataFrame({"name": ["A", "B"], "score": [1, 2], "source_doc": ["x", "y"]}),
                contract=contract,
                candidate_source="databao_raw",
            )

            self.assertEqual(guarded.to_dict(orient="list"), {"name": ["A", "B"]})

    def test_candidate_ranking_prefers_contract_valid_clean_candidate(self) -> None:
        contract = AnswerContract(
            kind="scalar",
            expected_columns=("answer",),
            max_rows=1,
            max_columns=1,
            allow_empty=False,
            reason="test",
        )
        bad = Candidate(
            frame=pd.DataFrame({"answer": [3], "source_doc": ["doc.md"]}),
            source="databao_raw",
            confidence=0.8,
            diagnostics={},
        )
        good = Candidate(
            frame=pd.DataFrame({"answer": [3]}),
            source="structured_planner",
            confidence=0.7,
            diagnostics={},
        )

        selected, report = rank_candidates(
            candidates=[bad, good],
            contract=contract,
            question="How many rows match?",
            retrieved_context=None,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.source, "structured_planner")
        self.assertEqual(report.selected_source, "structured_planner")

    def test_candidate_ranking_prefers_multirow_candidate_for_list_question(self) -> None:
        contract = AnswerContract(
            kind="entity_list",
            expected_columns=(),
            max_rows=None,
            max_columns=1,
            allow_empty=False,
            reason="test",
        )
        single = Candidate(
            frame=pd.DataFrame({"name": ["A"]}),
            source="databao_single",
            confidence=0.9,
            diagnostics={},
        )
        multi = Candidate(
            frame=pd.DataFrame({"name": ["A", "B", "C"]}),
            source="databao_multi",
            confidence=0.8,
            diagnostics={},
        )

        selected, report = rank_candidates(
            candidates=[single, multi],
            contract=contract,
            question="List matching names.",
            retrieved_context=None,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.source, "databao_multi")
        self.assertEqual(report.selected_source, "databao_multi")

    def test_candidate_ranking_penalizes_id_only_entity_answer(self) -> None:
        contract = AnswerContract(
            kind="entity_list",
            expected_columns=(),
            max_rows=None,
            max_columns=1,
            allow_empty=False,
            reason="test",
        )
        id_only = Candidate(
            frame=pd.DataFrame({"record_id": ["r1", "r2"]}),
            source="id_only",
            confidence=0.9,
            diagnostics={},
        )
        display = Candidate(
            frame=pd.DataFrame({"name": ["A", "B"]}),
            source="display",
            confidence=0.8,
            diagnostics={},
        )

        selected, report = rank_candidates(
            candidates=[id_only, display],
            contract=contract,
            question="Which names match?",
            retrieved_context=None,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.source, "display")
        self.assertEqual(report.selected_source, "display")

    def test_candidate_ranking_penalizes_raw_table_row_explosion(self) -> None:
        contract = AnswerContract(
            kind="entity_list",
            expected_columns=(),
            max_rows=None,
            max_columns=1,
            allow_empty=False,
            reason="test",
        )
        raw_table = Candidate(
            frame=pd.DataFrame(
                {
                    "row_id": range(1200),
                    "related_id": range(1200),
                    "name": [f"Name {index}" for index in range(1200)],
                    "helper_metric": range(1200),
                }
            ),
            source="databao_raw",
            confidence=0.9,
            diagnostics={},
        )
        compact = Candidate(
            frame=pd.DataFrame({"name": ["Name 1", "Name 2"]}),
            source="databao_raw_postprocessed",
            confidence=0.55,
            diagnostics={},
        )

        selected, report = rank_candidates(
            candidates=[raw_table, compact],
            contract=contract,
            question="Which names match?",
            retrieved_context=None,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.source, "databao_raw_postprocessed")
        raw_score = next(score for score in report.candidate_scores if score.source == "databao_raw")
        self.assertIn("large_raw_table_candidate", raw_score.reasons)

    def test_aggregate_ratio_verifier_requires_explicit_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="What percentage matches the condition?")
            task = DABenchPublicDataset(input_root).get_task("task_1")

            unchanged, transforms = apply_aggregate_ratio_verifier(
                task,
                pd.DataFrame({"score": [3], "points": [10]}),
            )

            self.assertEqual(transforms, [])
            self.assertEqual(unchanged.to_dict(orient="list"), {"score": [3], "points": [10]})

    def test_postprocess_resolves_numeric_id_to_display_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="Which name matches?")
            csv_dir = input_root / "task_1" / "context" / "csv"
            csv_dir.mkdir(parents=True)
            (csv_dir / "lookup.csv").write_text("raceId,name\n1,Alpha\n2,Beta\n", encoding="utf-8")
            task = DABenchPublicDataset(input_root).get_task("task_1")

            frame = pd.DataFrame({"raceId": [1, 2]})
            frame.attrs["context_dir"] = task.context_dir
            resolved, report = postprocess_answer_table(task, frame)

            self.assertEqual(resolved.to_dict(orient="list"), {"name": ["Alpha", "Beta"]})
            self.assertTrue(any(transform["kind"] == "identifier_resolution" for transform in report.transformations))

    def test_candidate_ranking_penalizes_zero_percentage_verifier_candidate(self) -> None:
        contract = AnswerContract(
            kind="scalar",
            expected_columns=("answer",),
            max_rows=1,
            max_columns=1,
            allow_empty=False,
            reason="test",
        )
        zero_percentage = Candidate(
            frame=pd.DataFrame({"percentage": [0]}),
            source="verifier",
            confidence=0.9,
            diagnostics={},
            transformations=(
                {
                    "kind": "aggregate_ratio_verification",
                    "numerator_column": "matched_count",
                    "denominator_column": "total_count",
                },
            ),
        )
        raw_answer = Candidate(
            frame=pd.DataFrame({"answer": [42]}),
            source="databao_raw",
            confidence=0.6,
            diagnostics={},
        )

        selected, report = rank_candidates(
            candidates=[zero_percentage, raw_answer],
            contract=contract,
            question="What percentage matches?",
            retrieved_context=None,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.source, "databao_raw")
        self.assertEqual(report.selected_source, "databao_raw")

    def test_candidate_ranking_does_not_overreward_valid_ratio_verifier(self) -> None:
        contract = AnswerContract(
            kind="scalar",
            expected_columns=("answer",),
            max_rows=1,
            max_columns=1,
            allow_empty=False,
            reason="test",
        )
        verifier = Candidate(
            frame=pd.DataFrame({"percentage": [50]}),
            source="databao_raw_verifier",
            confidence=0.95,
            diagnostics={},
            transformations=(
                {
                    "kind": "aggregate_ratio_verification",
                    "numerator_column": "matched_count",
                    "denominator_column": "total_count",
                },
            ),
        )
        raw_answer = Candidate(
            frame=pd.DataFrame({"answer": [42]}),
            source="databao_raw",
            confidence=0.6,
            diagnostics={},
        )

        selected, report = rank_candidates(
            candidates=[verifier, raw_answer],
            contract=contract,
            question="What percentage matches?",
            retrieved_context=None,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.source, "databao_raw")
        verifier_score = next(score for score in report.candidate_scores if score.source == "databao_raw_verifier")
        self.assertIn("high_risk_verifier_discount", verifier_score.reasons)
        self.assertIn("valid_non_verifier_candidate_available", verifier_score.reasons)

    def test_ratio_scale_compactor_converts_percentage_like_ratio_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_root = Path(tmp)
            write_task(
                input_root,
                question="How many times is the number of posts compared to votes?",
            )
            task = DABenchPublicDataset(input_root).get_task("task_1")
            frame = pd.DataFrame(
                {
                    "post_count": [3],
                    "vote_count": [8],
                    "posts_to_votes_ratio": [37.5],
                }
            )

            compact, transforms = apply_ratio_scale_compactor(task, frame)

        self.assertEqual(compact.to_dict(orient="list"), {"ratio": [0.375]})
        self.assertEqual(transforms[0]["kind"], "ratio_scale_compaction")
        self.assertEqual(transforms[0]["numerator_column"], "post_count")
        self.assertEqual(transforms[0]["denominator_column"], "vote_count")
        self.assertEqual(transforms[0]["source_result_column"], "posts_to_votes_ratio")

    def test_candidate_ranking_prefers_ratio_scale_compaction_over_components(self) -> None:
        contract = AnswerContract(
            kind="scalar",
            expected_columns=("answer",),
            max_rows=1,
            max_columns=1,
            allow_empty=False,
            reason="test",
        )
        raw = Candidate(
            frame=pd.DataFrame(
                {
                    "post_count": [3],
                    "vote_count": [8],
                    "posts_to_votes_ratio": [37.5],
                }
            ),
            source="databao_raw",
            confidence=0.55,
            diagnostics={},
        )
        compact = Candidate(
            frame=pd.DataFrame({"ratio": [0.375]}),
            source="databao_raw_ratio_compact",
            confidence=0.66,
            diagnostics={},
            transformations=(
                {
                    "kind": "ratio_scale_compaction",
                    "numerator_column": "post_count",
                    "denominator_column": "vote_count",
                    "source_result_column": "posts_to_votes_ratio",
                },
            ),
        )

        selected, report = rank_candidates(
            candidates=[raw, compact],
            contract=contract,
            question="How many times is the number of posts compared to votes?",
            retrieved_context=None,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected.source, "databao_raw_ratio_compact")
        compact_score = next(
            score for score in report.candidate_scores if score.source == "databao_raw_ratio_compact"
        )
        self.assertIn("ratio_scale_evidence", compact_score.reasons)

    def test_column_only_compactor_removes_empty_duplicate_and_constant_display_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_root = Path(tmp)
            write_task(input_root, question="Which names match the condition?")
            task = DABenchPublicDataset(input_root).get_task("task_1")
            contract = AnswerContract(
                kind="entity_list",
                expected_columns=(),
                max_rows=None,
                max_columns=None,
                allow_empty=False,
                reason="test",
            )
            frame = pd.DataFrame(
                {
                    "name": ["A", "B", "C"],
                    "name_2": ["Unused", "Unused", "Unused"],
                    "name_3": ["A", "B", "C"],
                    "debug_note": ["", "", ""],
                }
            )

            compact, transforms = apply_column_only_compactor(task, frame, contract)

        self.assertEqual(compact.to_dict(orient="list"), {"name": ["A", "B", "C"]})
        self.assertEqual(transforms[0]["kind"], "column_only_compaction")
        self.assertTrue(transforms[0]["row_count_preserved"])

    def test_column_only_compactor_preserves_multi_attribute_expected_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_root = Path(tmp)
            write_task(input_root, question="Return the reference name and website.")
            task = DABenchPublicDataset(input_root).get_task("task_1")
            contract = AnswerContract(
                kind="multi_attribute",
                expected_columns=("constructorRef", "url"),
                max_rows=1,
                max_columns=2,
                allow_empty=False,
                reason="test",
            )
            frame = pd.DataFrame({"constructorRef": ["mclaren"], "url": ["https://example.test"]})

            compact, transforms = apply_column_only_compactor(task, frame, contract)

        self.assertTrue(compact.equals(frame))
        self.assertEqual(transforms, [])

    def test_metadata_columns_never_reach_prediction_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            output_root = root / "output"
            logs_dir = root / "logs"
            write_task(input_root, question="What is the final answer?")
            task = DABenchPublicDataset(input_root).get_task("task_1")

            artifact = run_databao_task(
                task=task,
                output_root=output_root,
                logs_dir=logs_dir,
                databao_env=DatabaoEnvironment("http://localhost:8080/v1", "key", "model"),
                agent_builder=make_agent_builder(
                    pd.DataFrame(
                        {
                            "answer": ["A"],
                            "source_doc": ["doc.md"],
                            "evidence_span": ["debug"],
                            "confidence": [0.9],
                        }
                    )
                ),
                answer_finalizer=no_answer_finalizer,
                structured_planner=no_structured_planner,
                structured_planner_mode="off",
            )

            self.assertTrue(artifact.succeeded)
            self.assertEqual(
                (output_root / "task_1" / "prediction.csv").read_text(encoding="utf-8"),
                "answer\nA\n",
            )

    def test_ordinary_metadata_candidate_uses_cheap_repair_before_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            output_root = root / "output"
            logs_dir = root / "logs"
            write_task(input_root, question="What is the final answer?")
            task = DABenchPublicDataset(input_root).get_task("task_1")

            def forbidden_planner(task, databao_env):
                del task, databao_env
                raise AssertionError("ordinary metadata repair should not start planner rescue")

            artifact = run_databao_task(
                task=task,
                output_root=output_root,
                logs_dir=logs_dir,
                databao_env=DatabaoEnvironment("http://localhost:8080/v1", "key", "model"),
                agent_builder=make_agent_builder(
                    pd.DataFrame({"answer": ["databao"], "source_doc": ["doc.md"]})
                ),
                answer_finalizer=no_answer_finalizer,
                structured_planner=forbidden_planner,
            )

            log_payload = json.loads((logs_dir / "task_1.json").read_text(encoding="utf-8"))
            self.assertTrue(artifact.succeeded)
            self.assertEqual(artifact.selected_candidate_source, "databao_raw_postprocessed")
            self.assertEqual(log_payload["selected_candidate_source"], "databao_raw_postprocessed")
            self.assertEqual(
                (output_root / "task_1" / "prediction.csv").read_text(encoding="utf-8"),
                "answer\ndatabao\n",
            )

    def test_deadline_skips_planner_and_uses_best_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            output_root = root / "output"
            logs_dir = root / "logs"
            write_task(input_root, question="What is the final answer?")
            context_dir = input_root / "task_1" / "context"
            csv_dir = context_dir / "csv"
            json_dir = context_dir / "json"
            csv_dir.mkdir()
            json_dir.mkdir()
            (csv_dir / "left.csv").write_text("id,value\n1,A\n", encoding="utf-8")
            (csv_dir / "right.csv").write_text("left_id,amount\n1,10\n", encoding="utf-8")
            (json_dir / "extra.json").write_text(
                json.dumps({"table": "extra", "records": [{"id": 1, "status": "ok"}]}),
                encoding="utf-8",
            )
            task = DABenchPublicDataset(input_root).get_task("task_1")

            def forbidden_planner(task, databao_env):
                del task, databao_env
                raise AssertionError("planner should be skipped by deadline")

            with patch.dict(os.environ, {DATABAO_TASK_TIMEOUT_SECONDS_ENV: "30"}):
                artifact = run_databao_task(
                    task=task,
                    output_root=output_root,
                    logs_dir=logs_dir,
                    databao_env=DatabaoEnvironment("http://localhost:8080/v1", "key", "model"),
                    agent_builder=make_agent_builder(
                        pd.DataFrame({"answer": []})
                    ),
                    answer_finalizer=no_answer_finalizer,
                    structured_planner=forbidden_planner,
                    structured_planner_mode="fallback",
                )

            log_payload = json.loads((logs_dir / "task_1.json").read_text(encoding="utf-8"))
            self.assertTrue(artifact.succeeded)
            self.assertNotIn("planner_skipped_reason", log_payload)
            self.assertEqual(
                (output_root / "task_1" / "prediction.csv").read_text(encoding="utf-8"),
                "answer\n",
            )

    def test_query_context_retriever_selects_relevant_schema_slice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="What amount belongs to the matching customer name?")
            context_dir = input_root / "task_1" / "context"
            csv_dir = context_dir / "csv"
            csv_dir.mkdir()
            (csv_dir / "orders.csv").write_text("customer_id,amount\n1,20\n", encoding="utf-8")
            (csv_dir / "customers.csv").write_text("id,name\n1,A\n", encoding="utf-8")
            task = DABenchPublicDataset(input_root).get_task("task_1")
            task_context = build_task_context(task)

            retrieved = query_context_retriever(task_context, task.question)

            self.assertIn("orders", retrieved.relevant_columns)
            self.assertIn("customers", retrieved.relevant_columns)
            self.assertIn("amount", retrieved.relevant_columns["orders"])
            self.assertTrue(retrieved.candidate_join_paths)

    def test_route_policy_defaults_to_databao_for_low_confidence_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="What is the answer?")
            task = DABenchPublicDataset(input_root).get_task("task_1")
            task_context = build_task_context(task)
            intent = infer_task_intent(task_context)

            decision = choose_route_policy(task_context, intent)

            self.assertEqual(decision.route, "databao")
            self.assertIsNone(decision.candidate)

    def test_candidate_payload_enriches_identifier_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="Which event has the lowest cost?")
            context_dir = input_root / "task_1" / "context"
            json_dir = context_dir / "json"
            json_dir.mkdir()
            (json_dir / "event.json").write_text(
                json.dumps(
                    {
                        "table": "event",
                        "records": [
                            {
                                "event_id": "recEventA",
                                "event_name": "September Speaker",
                                "type": "Guest Speaker",
                                "status": "Closed",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            csv_dir = context_dir / "csv"
            csv_dir.mkdir()
            (csv_dir / "budget.csv").write_text(
                "budget_id,amount,link_to_event\nrecBudgetA,260,recEventA\n",
                encoding="utf-8",
            )
            (json_dir / "expense.json").write_text(
                json.dumps(
                    {
                        "table": "expense",
                        "records": [
                            {
                                "expense_id": "recExpenseA",
                                "cost": 13.45,
                                "link_to_budget": "recBudgetA",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            task = DABenchPublicDataset(input_root).get_task("task_1")

            payload = _candidate_table_payload(
                pd.DataFrame({"link_to_event": ["recEventA"]}),
                task=task,
            )

            enrichment = payload["context_enrichment"][0]
            self.assertEqual(
                enrichment["lookup"]["rows"][0]["attributes"]["event_name"],
                "September Speaker",
            )
            self.assertEqual(enrichment["lookup"]["rows"][0]["attributes"]["type"], "Guest Speaker")
            self.assertTrue(
                any(
                    summary["metric_column"] == "cost"
                    for summary in enrichment["numeric_summaries"]
                )
            )

    def test_raw_debug_logs_are_env_gated_and_redacted(self) -> None:
        previous = os.environ.pop(DATABAO_DEBUG_LOG_RAW_ENV, None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                input_root = root / "input"
                write_task(input_root)
                task = DABenchPublicDataset(input_root).get_task("task_1")
                env = DatabaoEnvironment("http://localhost:8080/v1", "secret-key", "model")
                diagnostics = TaskDiagnostics(task=task, logs_dir=root / "logs", databao_env=env)

                diagnostics.record_llm_call(
                    "probe",
                    metadata={"status": "ok"},
                    raw_request={"api_key": "secret-key"},
                )

                self.assertFalse((root / "logs" / "raw").exists())
                self.assertNotIn("raw_request_path", diagnostics.llm_calls[0])

                os.environ[DATABAO_DEBUG_LOG_RAW_ENV] = "1"
                diagnostics = TaskDiagnostics(task=task, logs_dir=root / "logs", databao_env=env)
                raw_path = diagnostics.write_raw("probe", {"api_key": "secret-key"})

                self.assertIsNotNone(raw_path)
                raw_text = Path(raw_path).read_text(encoding="utf-8")
                self.assertIn("[redacted]", raw_text)
                self.assertNotIn("secret-key", raw_text)
        finally:
            if previous is not None:
                os.environ[DATABAO_DEBUG_LOG_RAW_ENV] = previous
            else:
                os.environ.pop(DATABAO_DEBUG_LOG_RAW_ENV, None)

    def test_public_eval_ignores_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction_path = root / "prediction.csv"
            gold_path = root / "gold.csv"
            prediction_path.write_text("first,last\nAnn,W\nTyler,H\n", encoding="utf-8")
            gold_path.write_text("first,last\nTyler,H\nAnn,W\n", encoding="utf-8")

            result = evaluate_public_task(prediction_path, gold_path, "task_1")

            self.assertEqual(result.matched_columns, 2)
            self.assertTrue(result.exact_no_extra)

    def test_public_eval_reports_candidate_and_shape_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction_path = root / "prediction.csv"
            gold_path = root / "gold.csv"
            prediction_path.write_text("answer\nA\n", encoding="utf-8")
            gold_path.write_text("answer\nA\n", encoding="utf-8")

            result = evaluate_public_task(
                prediction_path,
                gold_path,
                "task_1",
                artifact_info={
                    "succeeded": True,
                    "selected_candidate_source": "databao_raw_postprocessed",
                    "postprocessing": {
                        "transformations": [
                            {"kind": "answer_column_verification"},
                        ],
                    },
                    "final_answer_guard": {
                        "transformations": [
                            {
                                "kind": "metadata_column_removal",
                                "removed_columns": ["source_doc"],
                            },
                        ],
                    },
                },
            )

            self.assertEqual(result.selected_candidate_source, "databao_raw_postprocessed")
            self.assertEqual(result.final_guard_removed_columns, ("source_doc",))
            self.assertEqual(result.postprocess_transformations, ("answer_column_verification",))


if __name__ == "__main__":
    unittest.main()
