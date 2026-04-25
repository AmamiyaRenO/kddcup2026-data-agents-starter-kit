# Function Ledger for Non-Databao Enhancements

This file is a working memory ledger for helper functions added around Databao.
It is not part of the official submission package. It uses ASCII text so it stays
readable in the Windows PowerShell console used during development.

## Purpose

Record local enhancement functions and capabilities outside the `databao-agent`
package itself. Focus on helpers that affect answer tables, scoring behavior,
logging, fallback, postprocessing, routing, planner/executor behavior, or CSV
writing.

Do not list ordinary Databao wrapper plumbing unless it changes local behavior.
For example, `build_databao_agent` and the basic domain registration flow are the
Databao demo entry path, not a local algorithmic enhancement.

## Update Rules

- When a listed function is added, renamed, deleted, or materially changed,
  update this file in the same development pass.
- If a function is kept but disabled by default, mark it `Available, default off`.
- If a function is replaced but temporarily kept, mark it `Deprecated` and name
  the replacement.
- Prefer public demo task sources such as `task_11`, `task_22`, and `task_25`.
  Use `competition scoring rules` for general output-format lessons.
- Keep `README.dev.md` focused on run state and experiment results. Keep
  function-level provenance here.

## Function Ledger

| Capability | Function(s) | Inspired by | Why added | Current status |
| --- | --- | --- | --- | --- |
| Layered task model | `TaskContext`, `TaskIntent`, `Candidate`, `RouteDecision`, `build_task_context`, `infer_task_intent`, `choose_route_policy` | easy sweep failures after `20260425T082722Z` | A single global planner mode was too coarse. These structures let the runner understand schema/question intent, choose a route, and log why a candidate was selected. | Active. |
| Schema graph profiling | `SchemaColumn`, `SchemaTable`, `JoinCandidate`, `build_schema_graph` | medium sweep failures after `20260425T093007Z` | Medium tasks mix SQLite, CSV, and JSON tables. The route/planner layer needs source kind, row counts, sample values, and inferred join candidates before it can safely choose deterministic table operations. | Active shared infrastructure. |
| Databao-first control flow | `_structured_planner_mode`, `_finalizer_mode`, `_ask_databao_candidate`, `_try_structured_planner_candidate` | `task_25` | `task_25` showed Databao itself could return a useful candidate quickly, while planner/finalizer calls caused most latency. These functions make Databao the default low-confidence path and keep planner/finalizer as optional fallback/experiments. | Active. Planner default is `fallback`; finalizer default is `off`. |
| High-confidence deterministic route dispatch | `try_deterministic_candidate`, `_structured_report_from_candidate` | `task_11`, `task_67`, `task_80`, `task_86`, `task_89` | Some questions are safer as direct table operations than LLM calls. The route dispatcher allows those cases to skip Databao only when question and schema evidence are strong. | Active in default fallback mode. |
| Structured query planner/executor | `execute_structured_plan`, `plan_structured_candidate`, `_heuristic_structured_plan`, `_try_thrombosis_route` | `task_11`, medium sweep failures | Severe-thrombosis questions need stable filter/join/select logic. Medium tasks also need safe `limit`, `top_k`, `date_filter`, `derive`, `ratio`, `percentage`, and `count_distinct` operations. | Active. The LLM JSON planner remains fallback/experimental. |
| Safe expression execution | `_safe_numeric_operand`, `_safe_binary_numeric_expression`, `_apply_derive_step`, `_apply_ratio_or_percentage_step` | `task_243`, `task_303` | Ratio and percentage answers often require combining aggregate components. These helpers compute only whitelisted numeric expressions and reject unknown ops/columns. | Active in structured executor. |
| Aggregate/ratio verification | `apply_aggregate_ratio_verifier` | `task_243`, `task_303` | Databao may return component columns such as `post_count/vote_count` or `total/germany_count` instead of the final scalar. This verifier rewrites high-confidence component tables to the final answer. | Active in default postprocess. |
| Answer-column verification | `apply_answer_column_verifier` | `task_218`, `task_250`, `task_257`, `task_259` | Medium tasks often have the right value plus helper columns. This verifier keeps explicit target columns such as `Phone`, `PostId`, `Text`, or `ViewCount + DisplayName`. | Active in default postprocess. |
| Aggregate deterministic route | `_try_superhero_average_route` | `task_67` | Databao/planner produced a wrong average for female superhero weight. This route recognizes gender lookup plus metric average and computes the scalar directly. | Active for high-confidence superhero gender/weight average questions. |
| Racing deterministic routes | `_try_formula_qualifying_number_route`, `_try_formula_driver_standings_route`, `_try_formula_finish_time_route`, `_best_race_rows_from_question`, `_driver_ids_mentioned_in_question` | `task_80`, `task_86`, `task_89` | Formula 1 questions exposed repeatable raceId/driverId/rank/time joins that are easier to execute deterministically than to ask Databao to infer every time. | Active for narrow schema/question matches. |
| Medium deterministic routes | `_try_event_expense_group_sum_route`, `_try_gas_customer_consumption_route`, `_try_molecule_bond_route`, `_try_school_route`, `_try_posts_route`, `_try_superhero_percentage_route`, `_try_set_translation_count_route`, `_try_racing_percentage_route` | `task_163`, `task_169`, `task_173`, `task_180`, `task_194`, `task_196`, `task_199`, `task_200`, `task_214`, `task_218`, `task_243`, `task_249`, `task_250`, `task_257`, `task_259`, `task_283`, `task_303` | Medium tasks exposed reusable schema/question families for SQLite-backed joins, grouped sums, ratios, percentages, and display-column lookups. These routes skip Databao only on high-confidence matches. | Active for narrow schema/question matches. |
| Question-driven column pruning | `apply_question_column_pruner`, `_question_target_columns` | `task_22`, `task_38`, `task_89` | Databao often includes the right answer values plus helper columns. Official-style scoring penalizes extra columns, so clear target-column requests are pruned deterministically. | Active in default postprocess. |
| Deterministic identifier resolution | `resolve_identifier_columns` | `task_25`, `task_22` | Databao often returns record IDs such as `link_to_event`; scoring wants final answer values such as `event_name`. This resolver maps high-coverage ID columns to display fields from context. | Active in default postprocess. |
| Candidate-table superlative pruning | `apply_superlative_verifier` | `task_25` | Databao may return candidate rows plus helper metric columns. For lowest/highest questions, this verifier keeps the min/max rows and removes helper metric columns unless the question asks for the value. | Active in default postprocess. |
| Context-chain superlative verification | `apply_context_superlative_verifier` | `task_25` | Databao sometimes computes the wrong helper metric for "lowest cost". This verifier recomputes metric chains such as `expense.cost -> budget.link_to_event -> event.event_name`. | Active in default postprocess. |
| Context/schema loading | `load_context_tables`, `_json_dataframes`, `_table_entity_candidates` | `task_11`, `task_25` | `task_11` needed JSON envelope table names like `Examination` and `Patient`; `task_25` needed entity/table inference across `event`, `budget`, and `expense`. | Active shared infrastructure. |
| Logging diagnostics | `TaskDiagnostics`, `DatabaoLangChainCallback` | `task_25` | Long-running `task_25` runs needed phase timings and LangChain call metadata to separate Databao latency from planner/finalizer latency. | Active. Raw logs gated by `DATABAO_DEBUG_LOG_RAW=1`. |
| Per-task timeout artifact | `_timeout_task_artifact` | `task_25` | Early `task_25` runs could hang for many minutes. Timeout artifacts let the runner kill a child process and still write inspectable failure logs. | Active when task timeout is configured. |
| Optional LLM answer finalizer | `finalize_answer_table`, `apply_answer_finalizer_decision` | `task_22`, early `task_25` | `task_22` exposed extra-column penalties; early `task_25` needed row/column trimming before deterministic verifiers existed. | Available, default off via `DATABAO_FINALIZER_MODE=off`. |
| CSV answer safety | `_write_prediction_csv` | `task_11`, competition scoring rules | Public gold uses integer IDs, while Pandas may render integer-like floats as `163109.0`. This writer preserves clean CSV values and avoids index columns. | Active for every prediction write. |
| Local public evaluation | `evaluate_public_run` | `task_19`, competition scoring rules | The official score ignores row order and column names but penalizes extra/missing columns. The local evaluator approximates this with column multiset signatures. | Active as `dabench evaluate-public-run`. |

## Task Inspiration Map

| Source | Functions / capabilities it inspired | Lesson captured |
| --- | --- | --- |
| `task_11` | `execute_structured_plan`, `_heuristic_structured_plan`, `plan_structured_candidate`, `_try_thrombosis_route`, `_json_dataframes`, `load_context_tables`, `_write_prediction_csv` | Some tasks are better handled by deterministic table operations. JSON table names and integer-like ID formatting matter. |
| `task_19` | `evaluate_public_run` | Row order is not a scoring dependency, so local evaluation should compare column value multisets rather than raw CSV row order. |
| `task_22` | `apply_question_column_pruner`, `resolve_identifier_columns`, `finalize_answer_table`, `apply_answer_finalizer_decision` | Extra columns hurt precision. A correct value with helper/evidence columns is still not exact. |
| `task_25` | `_structured_planner_mode`, `_finalizer_mode`, `_ask_databao_candidate`, `_try_structured_planner_candidate`, `resolve_identifier_columns`, `apply_superlative_verifier`, `apply_context_superlative_verifier`, `_table_entity_candidates`, `TaskDiagnostics`, `DatabaoLangChainCallback`, `_timeout_task_artifact` | Databao should be the first low-confidence path when it is fast; slow helper LLM layers should be optional. Superlative questions need deterministic answer-table pruning and sometimes context-chain metric verification. |
| `task_38` | `apply_question_column_pruner` | Record-list questions may want only the record ID, even when Databao returns the full transaction table. |
| `task_67` | `_try_superhero_average_route`, `TaskIntent` aggregate detection | Simple lookup-filter-aggregate tasks can be more reliable as deterministic Pandas execution than LLM inference. |
| `task_80` | `_try_formula_qualifying_number_route`, `_parse_time_seconds`, `_extract_time_floor_from_question` | Time expressions such as `0:01:54` may match lap-time prefixes like `1:54.xxx`; blank Q3 values must not crash route execution. |
| `task_86` | `_try_formula_driver_standings_route`, `_driver_ids_mentioned_in_question`, `_extract_less_than_threshold` | Person-name to driverId and standings-position filters are reusable racing-domain operations. |
| `task_89` | `_try_formula_finish_time_route`, `_best_race_rows_from_question`, `apply_question_column_pruner` | "ranked second" can refer to the `rank` column, not finishing position; final output should be the requested time column only. |
| `task_163` | `_try_event_expense_group_sum_route`, `build_schema_graph` | Event/budget/expense chains need reliable multi-source joins and grouped cost sums. |
| `task_169`, `task_173`, `task_180` | `_try_gas_customer_consumption_route`, `_apply_date_filter_step` | Gas/customer tasks mix transaction rows, year-month consumption rows, and lookup tables; dates can appear as `YYYYMM` or full date strings. |
| `task_194`, `task_196`, `task_200` | `_try_molecule_bond_route` | Atom/bond questions need bridge-table joins and count/ratio aggregates over element filters. |
| `task_199`, `task_218` | `_try_school_route`, `apply_answer_column_verifier` | School tasks need SAT-score aggregation/top-k plus display attributes from separate school/FRPM tables. |
| `task_243`, `task_249`, `task_250`, `task_257`, `task_259` | `_try_posts_route`, `apply_aggregate_ratio_verifier`, `apply_answer_column_verifier` | StackExchange-style tasks need user/post/vote/comment joins, count ratios, author filters, last-editor lookup, and final target-column pruning. |
| `task_283` | `_try_superhero_percentage_route` | Lookup-table percentages can be computed deterministically from a fact table and colour lookup. |
| `task_303` | `_try_racing_percentage_route`, `apply_aggregate_ratio_verifier` | Percentage questions should return the final scalar, not numerator/denominator helper columns. |
| `competition scoring rules` | `_write_prediction_csv`, `evaluate_public_run`, deterministic postprocess functions | Final CSV should be minimal and value-correct. Column names/order are less important than column value signatures, while extra columns and duplicate rows can reduce score. |

## Maintenance Notes

- Keep this ledger behavior-level, not line-by-line. If a helper is only an
  implementation detail of a listed capability, do not add it unless it starts
  affecting decisions or scoring.
- When a new public task inspires a reusable rule, add it to both `Function
  Ledger` and `Task Inspiration Map`.
- When a rule becomes harmful, mark it `Deprecated` here before deleting it,
  then remove the entry after the code is actually removed.
