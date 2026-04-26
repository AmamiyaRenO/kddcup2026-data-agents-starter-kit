from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from typer.testing import CliRunner

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.cli import app
from data_agent_baseline.run.databao_demo import (
    AnswerContract,
    AnswerFinalizationReport,
    Candidate,
    DatabaoEnvironment,
    DATABAO_DEBUG_LOG_RAW_ENV,
    DATABAO_TASK_TIMEOUT_SECONDS_ENV,
    StructuredPlanError,
    StructuredPlannerReport,
    TaskDiagnostics,
    _candidate_table_payload,
    _timeout_task_artifact,
    apply_aggregate_ratio_verifier,
    apply_answer_finalizer_decision,
    apply_answer_column_verifier,
    apply_question_column_pruner,
    build_schema_graph,
    build_task_context,
    choose_route_policy,
    document_records_for_reasoning,
    execute_structured_plan,
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


def no_answer_finalizer(task, frame, databao_env):
    del task, databao_env
    return frame, AnswerFinalizationReport(
        attempted=True,
        applied=False,
        mode=None,
        attempts=0,
        failure_reason=None,
        input_row_count=len(frame),
        input_column_count=len(frame.columns),
        output_row_count=len(frame),
        output_column_count=len(frame.columns),
        all_rows_included=True,
    )


def no_structured_planner(task, databao_env):
    del task, databao_env
    return None, StructuredPlannerReport(
        attempted=False,
        succeeded=False,
        attempts=0,
        failure_reason=None,
        plan=None,
        output_row_count=None,
        output_column_count=None,
        attempt_details=[],
    )


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
            self.assertEqual(artifact.heuristic_level, "generic")
            self.assertIn("context_load", log_payload["timings"])
            self.assertIn("databao_ask", log_payload["timings"])
            self.assertEqual(log_payload["llm_calls"], [])

    def test_prediction_csv_writes_integer_like_floats_as_ints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prediction.csv"

            from data_agent_baseline.run.databao_demo import _write_prediction_csv

            _write_prediction_csv(pd.DataFrame({"ID": [163109.0, 2803470.0]}), path)

            self.assertEqual(path.read_text(encoding="utf-8"), "ID\n163109\n2803470\n")

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
            self.assertFalse(log_payload["structured_planner"]["attempted"])
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
                frame = pd.DataFrame({"answer": ["planner"]})
                return frame, StructuredPlannerReport(
                    attempted=True,
                    succeeded=True,
                    attempts=1,
                    failure_reason=None,
                    plan={"steps": [], "output": "answer"},
                    output_row_count=1,
                    output_column_count=1,
                    attempt_details=[],
                )

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
            self.assertFalse(log_payload["structured_planner"]["attempted"])
            self.assertEqual(
                (output_root / "task_1" / "prediction.csv").read_text(encoding="utf-8"),
                "answer\ndatabao\n",
            )

    def test_runner_uses_structured_planner_rescue_after_complex_databao_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            output_root = root / "output"
            logs_dir = root / "logs"
            write_task(input_root)
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

            def planner(task, databao_env):
                del task, databao_env
                frame = pd.DataFrame({"answer": ["planner"]})
                return frame, StructuredPlannerReport(
                    attempted=True,
                    succeeded=True,
                    attempts=1,
                    failure_reason=None,
                    plan={"steps": [], "output": "answer"},
                    output_row_count=1,
                    output_column_count=1,
                    attempt_details=[],
                )

            artifact = run_databao_task(
                task=task,
                output_root=output_root,
                logs_dir=logs_dir,
                databao_env=DatabaoEnvironment("http://localhost:8080/v1", "key", "model"),
                agent_builder=make_agent_builder(None),
                answer_finalizer=no_answer_finalizer,
                structured_planner=planner,
                structured_planner_mode="fallback",
            )

            log_payload = json.loads((logs_dir / "task_1.json").read_text(encoding="utf-8"))
            self.assertTrue(artifact.succeeded)
            self.assertEqual(artifact.candidate_source, "structured_planner_rescue")
            self.assertEqual(log_payload["candidate_source"], "structured_planner_rescue")
            self.assertTrue(log_payload["structured_planner"]["succeeded"])
            self.assertEqual(
                (output_root / "task_1" / "prediction.csv").read_text(encoding="utf-8"),
                "answer\nplanner\n",
            )

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

    def test_timeout_artifact_writes_failure_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            output_root = root / "output"
            logs_dir = root / "logs"
            write_task(input_root)
            task = DABenchPublicDataset(input_root).get_task("task_1")

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
            self.assertFalse((output_root / "task_1" / "prediction.csv").exists())

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

    def test_answer_finalizer_select_mode_slices_rows_and_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "date_received": ["2019-10-02", "2019-09-12"],
                "amount": [50, 50],
                "source": ["Dues", "Dues"],
            }
        )

        pruned, mode = apply_answer_finalizer_decision(
            frame,
            {"mode": "select", "column_indices": [0], "row_indices": [1]},
        )

        self.assertEqual(mode, "select")
        self.assertEqual(pruned.to_dict(orient="list"), {"date_received": ["2019-09-12"]})

    def test_answer_finalizer_rewrite_mode_builds_dataframe(self) -> None:
        frame = pd.DataFrame({"candidate": ["A", "B"]})

        rewritten, mode = apply_answer_finalizer_decision(
            frame,
            {"mode": "rewrite", "columns": ["count"], "rows": [[2]]},
        )

        self.assertEqual(mode, "rewrite")
        self.assertEqual(rewritten.to_dict(orient="list"), {"count": [2]})

    def test_finalizer_failure_falls_back_to_databao_dataframe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            output_root = root / "output"
            logs_dir = root / "logs"
            write_task(input_root)
            task = DABenchPublicDataset(input_root).get_task("task_1")

            def broken_finalizer(task, frame, databao_env):
                del task, frame, databao_env
                raise RuntimeError("bad finalizer")

            artifact = run_databao_task(
                task=task,
                output_root=output_root,
                logs_dir=logs_dir,
                databao_env=DatabaoEnvironment("http://localhost:8080/v1", "key", "model"),
                agent_builder=make_agent_builder(pd.DataFrame({"answer": ["A"], "extra": [1]})),
                answer_finalizer=broken_finalizer,
                structured_planner=no_structured_planner,
                finalizer_mode="always",
            )

            log_payload = json.loads((logs_dir / "task_1.json").read_text(encoding="utf-8"))

            self.assertTrue(artifact.succeeded)
            self.assertEqual(
                (output_root / "task_1" / "prediction.csv").read_text(encoding="utf-8"),
                "answer\nA\n",
            )
            self.assertFalse(log_payload["finalization"]["applied"])
            self.assertIn("bad finalizer", log_payload["finalization"]["failure_reason"])

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
                            {"event_id": "recA", "event_name": "Alpha"},
                            {"event_id": "recB", "event_name": "Beta"},
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

            write_task(input_root, task_id="task_display", question="Identify the ViewCount and DisplayName.")
            views_task = DABenchPublicDataset(input_root).get_task("task_display")
            views_frame, _ = apply_answer_column_verifier(
                views_task,
                pd.DataFrame({"Title": ["A"], "ViewCount": [1708], "DisplayName": ["mbq"], "UserId": [88]}),
            )
            self.assertEqual(views_frame.to_dict(orient="list"), {"ViewCount": [1708], "DisplayName": ["mbq"]})

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

    def test_final_answer_shape_guard_enforces_entity_list_shape(self) -> None:
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

            self.assertEqual(guarded.to_dict(orient="list"), {"name": ["A", "B"]})

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
            self.assertIn("Structured planner skipped", log_payload["planner_skipped_reason"])
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

            decision = choose_route_policy(task_context, intent, structured_planner_mode="fallback")

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


if __name__ == "__main__":
    unittest.main()
