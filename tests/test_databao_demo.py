from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.run.databao_demo import (
    AnswerFinalizationReport,
    DatabaoEnvironment,
    DATABAO_DEBUG_LOG_RAW_ENV,
    StructuredPlanError,
    StructuredPlannerReport,
    TaskDiagnostics,
    apply_aggregate_ratio_verifier,
    apply_answer_column_verifier,
    apply_question_column_pruner,
    _candidate_table_payload,
    _timeout_task_artifact,
    apply_answer_finalizer_decision,
    build_schema_graph,
    build_task_context,
    choose_route_policy,
    execute_structured_plan,
    infer_task_intent,
    load_databao_environment,
    load_context_tables,
    plan_structured_candidate,
    postprocess_answer_table,
    register_context_sources,
    run_databao_task,
    run_databao_tasks,
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


def write_task(root: Path, task_id: str = "task_1", question: str = "What is the answer?") -> Path:
    task_dir = root / task_id
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "difficulty": "easy",
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

    def test_structured_planner_uses_schema_heuristic_for_task_11_pattern(self) -> None:
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
            (context_dir / "knowledge.md").write_text(
                "Thrombosis is an integer, with 1 being the most severe and 2 indicating severe cases.",
                encoding="utf-8",
            )
            json_dir = context_dir / "json"
            json_dir.mkdir()
            (json_dir / "Examination.json").write_text(
                json.dumps(
                    {
                        "table": "Examination",
                        "records": [
                            {"ID": 1, "Thrombosis": 1},
                            {"ID": 2, "Thrombosis": 2},
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
                            {"ID": 1, "SEX": "M", "Diagnosis": "APS"},
                            {"ID": 2, "SEX": "F", "Diagnosis": "SLE"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            task = DABenchPublicDataset(input_root).get_task("task_11")

            frame, report = plan_structured_candidate(
                task,
                DatabaoEnvironment("http://localhost:8080/v1", "key", "model"),
            )

            self.assertTrue(report.succeeded)
            self.assertEqual(report.attempt_details[0]["kind"], "schema_heuristic")
            self.assertEqual(
                frame.to_dict(orient="list"),
                {"ID": [2], "SEX": ["F"], "Diagnosis": ["SLE"]},
            )

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
            self.assertEqual(log_payload["candidate_source"], "databao")
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
            self.assertEqual(artifact.candidate_source, "databao")
            self.assertEqual(log_payload["candidate_source"], "databao")
            self.assertFalse(log_payload["structured_planner"]["attempted"])
            self.assertEqual(
                (output_root / "task_1" / "prediction.csv").read_text(encoding="utf-8"),
                "answer\ndatabao\n",
            )

    def test_runner_can_use_structured_planner_first_when_requested(self) -> None:
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

            def forbidden_agent_builder(task, databao_env):
                del task, databao_env
                raise AssertionError("Databao agent should not be built after planner success.")

            artifact = run_databao_task(
                task=task,
                output_root=output_root,
                logs_dir=logs_dir,
                databao_env=DatabaoEnvironment("http://localhost:8080/v1", "key", "model"),
                agent_builder=forbidden_agent_builder,
                answer_finalizer=no_answer_finalizer,
                structured_planner=planner,
                structured_planner_mode="first",
            )

            log_payload = json.loads((logs_dir / "task_1.json").read_text(encoding="utf-8"))
            self.assertTrue(artifact.succeeded)
            self.assertEqual(artifact.candidate_source, "structured_planner")
            self.assertEqual(log_payload["candidate_source"], "structured_planner")
            self.assertTrue(log_payload["structured_planner"]["succeeded"])
            self.assertEqual(
                (output_root / "task_1" / "prediction.csv").read_text(encoding="utf-8"),
                "answer\nplanner\n",
            )

    def test_runner_uses_structured_planner_fallback_after_databao_failure(self) -> None:
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
                agent_builder=make_agent_builder(None),
                answer_finalizer=no_answer_finalizer,
                structured_planner=planner,
            )

            log_payload = json.loads((logs_dir / "task_1.json").read_text(encoding="utf-8"))
            self.assertTrue(artifact.succeeded)
            self.assertEqual(artifact.candidate_source, "structured_planner_fallback")
            self.assertEqual(log_payload["candidate_source"], "structured_planner_fallback")
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
                "answer,extra\nA,1\n",
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

    def test_postprocessor_prunes_superlative_metric_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(
                input_root,
                question="Which event has the lowest cost?",
            )
            task = DABenchPublicDataset(input_root).get_task("task_1")

            frame, report = postprocess_answer_table(
                task,
                pd.DataFrame(
                    {
                        "event_name": ["A", "B", "C"],
                        "expense_description": ["Food", "Parking", "Supplies"],
                        "total_cost": [20.0, 10.0, 30.0],
                    }
                ),
            )

            self.assertTrue(report.applied)
            self.assertEqual(frame.to_dict(orient="list"), {"event_name": ["B"]})

    def test_postprocessor_drops_metric_column_when_superlative_rows_are_already_filtered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(
                input_root,
                question="Which event has the lowest cost?",
            )
            task = DABenchPublicDataset(input_root).get_task("task_1")

            frame, report = postprocess_answer_table(
                task,
                pd.DataFrame(
                    {
                        "event_name": ["A", "B"],
                        "cost": [6.0, 6.0],
                    }
                ),
            )

            self.assertTrue(report.applied)
            self.assertEqual(frame.to_dict(orient="list"), {"event_name": ["A", "B"]})

    def test_postprocessor_uses_context_chain_for_event_cost_superlative(self) -> None:
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

            frame, report = postprocess_answer_table(
                task,
                pd.DataFrame({"event_name": ["Alpha"], "total_cost": [20.0]}),
            )

            self.assertTrue(report.applied)
            self.assertEqual(frame.to_dict(orient="list"), {"event_name": ["Beta"]})

    def test_question_column_pruner_keeps_date_received(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="What date were the state dues paid?")
            task = DABenchPublicDataset(input_root).get_task("task_1")

            frame, transforms = apply_question_column_pruner(
                task,
                pd.DataFrame(
                    {
                        "income_id": ["recA"],
                        "date_received": ["2021-05-20"],
                        "amount": [50],
                    }
                ),
            )

            self.assertEqual(frame.to_dict(orient="list"), {"date_received": ["2021-05-20"]})
            self.assertEqual(transforms[0]["kind"], "question_column_pruning")

    def test_question_column_pruner_keeps_withdrawal_transaction_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="List all cash withdrawals over the threshold.")
            task = DABenchPublicDataset(input_root).get_task("task_1")

            frame, _ = apply_question_column_pruner(
                task,
                pd.DataFrame(
                    {
                        "trans_id": [101, 102],
                        "date": ["2020-01-01", "2020-01-02"],
                        "amount": [500, 600],
                    }
                ),
            )

            self.assertEqual(frame.to_dict(orient="list"), {"trans_id": [101, 102]})

    def test_aggregate_ratio_verifier_rewrites_post_vote_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="For the user No.24, how many times is the number of posts compared to votes?")
            task = DABenchPublicDataset(input_root).get_task("task_1")

            frame, transforms = apply_aggregate_ratio_verifier(
                task,
                pd.DataFrame({"post_count": [3], "vote_count": [8]}),
            )

            self.assertAlmostEqual(frame.iloc[0, 0], 0.375)
            self.assertEqual(transforms[0]["mode"], "ratio")

    def test_aggregate_ratio_verifier_rewrites_percentage_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="What percentage of European Grand Prix races were hosted in Germany?")
            task = DABenchPublicDataset(input_root).get_task("task_1")

            frame, transforms = apply_aggregate_ratio_verifier(
                task,
                pd.DataFrame({"total_european_gp": [23], "germany_count": [12]}),
            )

            self.assertAlmostEqual(frame.iloc[0, 0], 52.17391304347826)
            self.assertEqual(transforms[0]["mode"], "percentage")

    def test_answer_column_verifier_keeps_phone_post_text_and_view_user_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"

            write_task(input_root, task_id="task_phone", question="What is the telephone number for the school?")
            phone_task = DABenchPublicDataset(input_root).get_task("task_phone")
            phone_frame, _ = apply_answer_column_verifier(
                phone_task,
                pd.DataFrame({"cds": [1], "AvgScrRead": [370], "Phone": ["(559) 248-5100"]}),
            )
            self.assertEqual(phone_frame.to_dict(orient="list"), {"Phone": ["(559) 248-5100"]})

            write_task(input_root, task_id="task_post", question="Which post has the most answers count? State the post ID.")
            post_task = DABenchPublicDataset(input_root).get_task("task_post")
            post_frame, _ = apply_answer_column_verifier(
                post_task,
                pd.DataFrame({"Id": [351], "AnswerCount": [None]}),
            )
            self.assertEqual(post_frame.to_dict(orient="list"), {"Id": [351]})

            write_task(input_root, task_id="task_comment", question="What is the comment with the highest score?")
            comment_task = DABenchPublicDataset(input_root).get_task("task_comment")
            comment_frame, _ = apply_answer_column_verifier(
                comment_task,
                pd.DataFrame({"Id": [1], "Score": [14], "Text": ["hello"]}),
            )
            self.assertEqual(comment_frame.to_dict(orient="list"), {"Text": ["hello"]})

            write_task(input_root, task_id="task_views", question="Identify the total views. Name the user who posted it last time.")
            views_task = DABenchPublicDataset(input_root).get_task("task_views")
            views_frame, _ = apply_answer_column_verifier(
                views_task,
                pd.DataFrame({"Title": ["A"], "ViewCount": [1708], "DisplayName": ["mbq"], "UserId": [88]}),
            )
            self.assertEqual(views_frame.to_dict(orient="list"), {"ViewCount": [1708], "DisplayName": ["mbq"]})

    def test_deterministic_route_solves_superhero_average_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="What is the average weight of all female superheroes?")
            context_dir = input_root / "task_1" / "context"
            csv_dir = context_dir / "csv"
            json_dir = context_dir / "json"
            csv_dir.mkdir()
            json_dir.mkdir()
            (csv_dir / "superhero.csv").write_text(
                "id,superhero_name,gender_id,weight_kg\n1,A,2,50\n2,B,2,70\n3,C,1,200\n",
                encoding="utf-8",
            )
            (json_dir / "gender.json").write_text(
                json.dumps(
                    {
                        "table": "gender",
                        "records": [
                            {"id": 1, "gender": "Male"},
                            {"id": 2, "gender": "Female"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            task = DABenchPublicDataset(input_root).get_task("task_1")
            task_context = build_task_context(task)
            intent = infer_task_intent(task_context)
            decision = choose_route_policy(task_context, intent, structured_planner_mode="fallback")

            self.assertEqual(decision.route, "deterministic")
            self.assertIsNotNone(decision.candidate)
            self.assertEqual(decision.candidate.frame.iloc[0, 0], 60.0)

    def test_deterministic_route_solves_racing_finish_time_rank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(
                input_root,
                question="What's the finish time for the driver who ranked second in 2008's Chinese Grand Prix?",
            )
            context_dir = input_root / "task_1" / "context"
            csv_dir = context_dir / "csv"
            json_dir = context_dir / "json"
            csv_dir.mkdir()
            json_dir.mkdir()
            (csv_dir / "results.csv").write_text(
                "raceId,driverId,rank,positionOrder,time\n34,1,1,1,1:31:57.403\n34,8,2,3,+16.445\n",
                encoding="utf-8",
            )
            (json_dir / "races.json").write_text(
                json.dumps(
                    {
                        "table": "races",
                        "records": [
                            {"raceId": 34, "year": 2008, "name": "Chinese Grand Prix"},
                            {"raceId": 35, "year": 2008, "name": "Brazilian Grand Prix"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            task = DABenchPublicDataset(input_root).get_task("task_1")
            task_context = build_task_context(task)
            intent = infer_task_intent(task_context)
            decision = choose_route_policy(task_context, intent, structured_planner_mode="fallback")

            self.assertEqual(decision.route, "deterministic")
            self.assertEqual(decision.candidate.frame.to_dict(orient="list"), {"time": ["+16.445"]})

    def test_deterministic_route_solves_qualifying_q3_driver_number(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(
                input_root,
                question="What is his number of the driver who finished 0:01:54 in the Q3 of qualifying race No.903?",
            )
            context_dir = input_root / "task_1" / "context"
            csv_dir = context_dir / "csv"
            json_dir = context_dir / "json"
            csv_dir.mkdir()
            json_dir.mkdir()
            (csv_dir / "qualifying.csv").write_text(
                "raceId,driverId,number,q3\n"
                "903,817,3,1:54.455\n"
                "903,20,1,1:54.960\n"
                "903,1,44,1:53.860\n"
                "903,825,20,\n",
                encoding="utf-8",
            )
            (json_dir / "drivers.json").write_text(
                json.dumps(
                    {
                        "table": "drivers",
                        "records": [
                            {"driverId": 817, "number": "3"},
                            {"driverId": 20, "number": "5"},
                            {"driverId": 1, "number": "44"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            task = DABenchPublicDataset(input_root).get_task("task_1")
            task_context = build_task_context(task)
            intent = infer_task_intent(task_context)
            decision = choose_route_policy(task_context, intent, structured_planner_mode="fallback")

            self.assertEqual(decision.route, "deterministic")
            self.assertEqual(decision.candidate.frame.to_dict(orient="list"), {"number": ["3", "5"]})

    def test_deterministic_route_solves_driver_standings_race_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_root = root / "input"
            write_task(input_root, question="Which race was Alex Yoong in when he was in track number less than 20?")
            context_dir = input_root / "task_1" / "context"
            csv_dir = context_dir / "csv"
            json_dir = context_dir / "json"
            csv_dir.mkdir()
            json_dir.mkdir()
            (csv_dir / "driverStandings.csv").write_text(
                "raceId,driverId,position\n124,62,19\n125,62,20\n126,62,18\n",
                encoding="utf-8",
            )
            (csv_dir / "races.csv").write_text(
                "raceId,name\n124,Australian Grand Prix\n125,Malaysian Grand Prix\n126,Brazilian Grand Prix\n",
                encoding="utf-8",
            )
            (json_dir / "drivers.json").write_text(
                json.dumps(
                    {
                        "table": "drivers",
                        "records": [{"driverId": 62, "forename": "Alex", "surname": "Yoong"}],
                    }
                ),
                encoding="utf-8",
            )
            task = DABenchPublicDataset(input_root).get_task("task_1")
            task_context = build_task_context(task)
            intent = infer_task_intent(task_context)
            decision = choose_route_policy(task_context, intent, structured_planner_mode="fallback")

            self.assertEqual(decision.route, "deterministic")
            self.assertEqual(
                decision.candidate.frame.to_dict(orient="list"),
                {"name": ["Australian Grand Prix", "Brazilian Grand Prix"]},
            )

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
