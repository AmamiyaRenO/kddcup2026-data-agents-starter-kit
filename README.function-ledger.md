# Function Ledger

## Purpose

This file records local capabilities added around Databao. It intentionally tracks only generic Data
Agent layers, not benchmark-specific parsers, task-family routes, or public-set memory.

## Update Rules

- Add or update this file whenever a local helper changes behavior, routing, logging, fallback, or
  answer-shaping semantics.
- Remove deprecated helpers instead of leaving stale active entries.
- Do not record concrete public task domains, file stems, entity names, category values, or copied
  benchmark phrasing as active strategies.
- Strategy provenance should be architectural: schema loading, document ingestion, answer contract,
  verification, timeout/logging, or scoring safety.

## Function Ledger

| Capability | Function(s) | Why added | Current status |
| --- | --- | --- | --- |
| Databao-first control flow | `_structured_planner_mode`, `_finalizer_mode`, `_heuristic_level`, `_ask_databao_candidate`, `_try_structured_planner_candidate`, `choose_route_policy` | Keep Databao as the default black-box executor, write provisional predictions early, and allow only bounded generic planner rescue or explicit finalizer experiments. | Active. `DATABAO_HEURISTIC_LEVEL=generic` is default; only `experimental_generic` is also allowed. |
| Task context layer | `TaskContext`, `TaskIntent`, `Candidate`, `RouteDecision`, `build_task_context`, `infer_task_intent` | Centralize task files, table context, schema graph, document text, and coarse question intent. | Active generic infrastructure. |
| Query-time retrieval | `query_context_retriever`, `_retrieval_column_score`, `_display_like_columns`, `_metric_like_columns` | Rank question-relevant tables, columns, samples, document snippets, and join paths without hiding unretrieved tables from Databao. | Active soft-priority context signal. |
| Generic document ingestion | `generic_document_tables`, `document_records_for_agent`, `document_records_for_reasoning`, `_generic_document_rows_for_path`, `_generic_identifier_values`, `_extract_named_value`, `_extract_numeric_attribute`, `_extract_text_attribute`, `_extract_relation_attributes` | Convert record-like prose into a joinable table without selecting parsers by file stem or domain, while keeping debug metadata out of agent-visible tables. | Active. Agent-safe table plus reasoning metadata split. |
| Schema graph profiling | `ContextTable`, `build_schema_graph`, `_column_key_variants`, `_entity_name_variants`, `_source_kind_from_context_path` | Give the planner and verifiers table names, source kinds, row counts, columns, samples, metadata, and join candidates. | Active. |
| Safe structured executor | `execute_structured_plan`, `_apply_filter_step`, `_apply_join_step`, `_apply_select_step`, `_apply_aggregate_step`, `_apply_date_filter_step`, `_apply_derive_step`, `_apply_ratio_or_percentage_step` | Support table operations through a restricted JSON plan without arbitrary Python or SQL execution. | Active planner fallback. |
| Answer contract | `AnswerContract`, `AnswerContractReport`, `infer_answer_contract`, `validate_answer_contract`, `_candidate_looks_like_source_name` | Check whether a candidate looks like a final answer shape instead of a raw table, empty result, or source-name singleton. | Active logging and repair gate. |
| Complexity and rescue gating | `TaskComplexityProfile`, `build_complexity_profile`, `should_run_planner_rescue` | Decide whether a planner rescue is warranted from observable source/schema/question/candidate features instead of difficulty tags or ordinary suspicious output. | Active. Planner rescue requires high complexity, high uncertainty, enough budget, and bounded payload size. |
| Candidate generation and ranking | `rank_candidates`, `CandidateScore`, `CandidateRankingReport`, `_candidate_from_frame`, `_candidate_with_contract`, `generate_verifier_candidate_frames` | Compare Databao, postprocessed, verifier, structured, and optional finalizer candidates with a common generic evaluator. | Active. Verifiers generate ranked candidates rather than direct overrides. |
| Final answer guard | `final_answer_shape_guard`, `_metadata_debug_columns`, `_select_columns_by_priority`, `_question_looks_list_answer` | Remove debug metadata and enforce scalar/list/table answer shapes immediately before writing `prediction.csv`. | Active last-mile safety layer. |
| Identifier resolution | `resolve_identifier_columns`, `_match_context_identifier`, `_display_series`, `_identifier_columns` | Replace high-coverage identifier/link columns with generic display columns when context provides a safe mapping. | Active generic postprocess. |
| Superlative verification | `apply_superlative_verifier`, `apply_context_superlative_verifier`, `_numeric_metric_columns`, `_context_metric_score` | For min/max questions, propose verifier candidates with metric evidence without automatically overriding Databao. | Active generic candidate generator. |
| Column and aggregate verification | `apply_question_column_pruner`, `apply_answer_column_verifier`, `apply_aggregate_ratio_verifier`, `_question_target_columns`, `_question_matched_numeric_column` | Prune to question-referenced columns and propose scalar ratio or percentage candidates only when the components are clear. | Active generic postprocess/candidate generator. |
| CSV safety and local evaluation | `_write_prediction_csv`, `_prediction_csv_status`, `evaluate_public_run` | Write minimal UTF-8 predictions, preserve provisional predictions in failure artifacts, and evaluate public runs with per-task scoring diagnostics after execution. | Active. |
| Diagnostics and timeouts | `TaskDiagnostics`, `DatabaoLangChainCallback`, `_write_task_log`, `_timeout_task_artifact`, `_run_databao_task_child`, `_deadline_from_start`, `_planner_skip_reason_for_deadline` | Record phase timings, route decisions, strategy diagnostics, safe raw logs, prediction-written/scorable status, task timeout artifacts, candidate scores, deadline-aware planner skips, and failure summaries. | Active. |
| Guardrail test | `test_src_code_has_no_public_domain_memory_terms` | Prevent public-domain memory terms from re-entering production code. | Active test-only protection. |

## Removed / Not Allowed

- Domain-specific document materializers and route families were removed from active code.
- No `public_optimized` mode exists.
- New helpers must be domain-agnostic unless they live outside production code for an explicit,
  temporary analysis notebook or experiment.

## Strategy Map

| Layer | Strategy kind | Logged signal |
| --- | --- | --- |
| Context loading | generic | `context_summary`, schema graph table metadata |
| Query-time retrieval | generic | `retrieved_context`, selected tables/columns, sample rows, document snippets, join paths, retrieval confidence |
| Document extraction | generic or experimental_generic | `document_materialized_tables`, `document_extraction_used`, extracted rows/columns/confidence, metadata columns removed from agent table |
| Route choice | generic | `route_policy.route`, `route_policy.reason`, `candidate_source` |
| Complexity profiling | generic | `complexity_profile`, complexity score, uncertainty score, planner payload budget |
| Candidate validation | generic | `answer_contract.valid`, `should_repair`, row/column counts |
| Candidate ranking | generic | `candidate_scores`, `selected_candidate_source`, rejection reasons |
| Postprocess | generic | `postprocessing.transformations`, `applied_strategies` |
| Final guard | generic | `final_answer_guard.transformations`, metadata removed, contract before/after |
| Failure handling | generic | `failure_reason`, `prediction_written`, `scorable`, phase timings, timeout artifact |
