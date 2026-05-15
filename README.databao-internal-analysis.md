# Databao Internal Observation Report

This report summarizes targeted Databao shadow-observation runs for the easy/medium tasks that were
previously not full-score. The purpose is to decide whether the next fix belongs inside Databao,
inside a rescue layer, or in the existing adapter.

All runs below used the current shadow-only `databao_internal_observation` path. The observation
layer does not change ranking, final guard, or prediction output.

## Targeted Runs

| Task | Run ID | Score | Exact | Prediction | Main Databao Observation |
| --- | --- | ---: | --- | --- | --- |
| `task_11` | `20260514T033623Z` | `0.000` | no | `1/3` rows | SQL shape plausible, but categorical value mapping is wrong. |
| `task_86` | `20260514T033636Z` | `1.000` | yes | `16/16` rows | Databao generated correct join/filter; adapter kept `name`. |
| `task_89` | `20260514T033647Z` | `0.000` | no | wrong scalar value | SQL selected wrong race/result row. |
| `task_163` | `20260514T033658Z` | `0.000` | no | wrong grouping attribute | SQL grouped by budget category instead of requested output type. |
| `task_169` | `20260514T033719Z` | `0.000` | no | annual avg instead of monthly avg | SQL did `AVG(Consumption)` but not `/ 12`. |
| `task_173` | `20260514T033732Z` | `0.000` | no | empty result | SQL used string date range; likely date type mismatch. |
| `task_180` | `20260514T033748Z` | `0.000` | no | broad detail rows | SQL missed/over-broadened required filter population. |
| `task_199` | `20260514T033806Z` | `1.000` | yes | `6/6` rows | Databao generated correct grouped school query in this run. |
| `task_200` | `20260514T033835Z` | `1.000` | yes | scalar count | Databao generated correct aggregate SQL in this run. |
| `task_243` | `20260514T033856Z` | `1.000` | yes | scalar ratio | Databao generated numerator, denominator, and ratio in this run. |
| `task_257` | `20260514T033913Z` | `0.333` | no | correct metric, wrong display | SQL selected blank display columns and did not join the needed display source. |
| `task_283` | `20260514T033949Z` | `0.000` | no | numerator count only | SQL returned count but omitted denominator/percentage operation. |

## Per-Task Notes

### `task_11`

- Question: list patient `ID`, `sex`, and diagnosis for severe thrombosis.
- Databao SQL:
  - joins patient and examination tables;
  - filters `Thrombosis = 3`;
  - returns the requested three columns.
- Output: one row, `ID=2495750`.
- Gold: three different patient rows.
- Observation: this is not a final-guard or adapter problem. The SQL looks structurally correct but
  maps the question phrase "severe degree" to the wrong categorical code.
- Next possible Databao-internal improvement: add a schema/value-grounding step before filtering
  categorical codes. The agent should inspect value distributions or field definitions before
  mapping natural-language severity labels to numeric codes.

### `task_86`

- Question: list races for a driver where track number is less than 20.
- Databao SQL:
  - joins standings, races, and drivers;
  - filters driver and `position < 20`;
  - returns `name` plus helper columns.
- Output: exact after adapter column pruning.
- Observation: no Databao change needed for the successful shape. Prior failures were mostly
  adapter ID/display resolution and occasional extra-row variance.

### `task_89`

- Question: finish time for the driver ranked second in a named race/year.
- Databao SQL:
  - selects from results;
  - filters `raceId = 34` and `positionText = '2'`;
  - returns `time=14.925`.
- Gold: `time=16.445`.
- Observation: Databao selected a plausible but wrong race/result row. The adapter selected the
  correct `time` column, so the failure is upstream.
- Next possible Databao-internal improvement: after resolving an entity ID such as race ID from
  natural-language year/name, validate the lookup against the source entity table. The generated
  SQL should join or verify the race name/year instead of trusting a guessed ID.

### `task_163`

- Question: expense type and total approved value for a named event.
- Databao SQL:
  - filters the event and closed budget rows;
  - groups by `category`;
  - sums `spent`.
- Output: two category rows whose values sum to the gold total.
- Gold: one row with event/type-level label and total.
- Observation: Databao did a useful partial aggregate, but chose the wrong grouping/output
  attribute. This is not safe to collapse externally without knowing the intended semantic label.
- Next possible Databao-internal improvement: add a submit critique for "requested attribute not
  selected". If the question asks for `type`/attribute plus total, Databao should verify the selected
  grouping column corresponds to that requested attribute, not just any category-like field.

### `task_169`

- Question: average monthly consumption for a segment in a year.
- Databao SQL:
  - joins customers and year/month consumption;
  - filters segment and year;
  - computes `AVG(Consumption)`.
- Output: `5519.475...`.
- Gold: `AVG(Consumption) / 12 = 459.956...`.
- Observation: Databao got the right population but missed the unit conversion from annual total to
  monthly average. Shadow flags include `scalar_aggregate_after_filter`, but the missing `/12`
  requires semantic unit reasoning.
- Next possible Databao-internal improvement: add a unit/time-granularity critique. If a question
  asks "monthly" and the selected column is annual/year-level consumption, Databao should either use
  a month-level field or explicitly divide by the number of months.

### `task_173`

- Question: countries of gas stations with transactions in June 2013.
- Databao SQL:
  - joins transactions and gas stations;
  - filters `Date >= '2013-06-01' AND Date <= '2013-06-30'`;
  - returns empty `Country`.
- Gold: `CZE`, `SVK`.
- Observation: the empty result is likely caused by date-type mismatch or wrong date literal format.
  Shadow observation correctly flags `empty_candidate`.
- Next possible Databao-internal improvement: when a SQL filter returns empty, inspect column type
  and sample values before submitting. For date-like columns, validate whether values are strings,
  integers, or `YYYYMMDD` style.

### `task_180`

- Question: consumption status in a month for people who paid more than a unit price for a product.
- Databao SQL:
  - finds distinct customers from product/price conditions;
  - joins monthly consumption;
  - filters target month.
- Output: 153 consumption rows.
- Gold: 9 rows.
- Observation: Databao built a reasonable-looking multi-step query but the population is too broad.
  This is a source/filter semantics issue, not a column-pruning issue.
- Next possible Databao-internal improvement: after a detail query returns many rows for a specific
  conditioned population, perform a filter-audit step: list which question constraints have been
  translated into SQL and which remain unaccounted for.

### `task_199`

- Question: school names and funding types for Riverside-related districts with average math score
  over a threshold.
- Databao SQL:
  - joins score and funding tables;
  - filters district text;
  - groups by school/funding type;
  - applies `HAVING AVG(math) > 400`.
- Output: exact in this run.
- Observation: no Databao change needed for this run. Previous misses were source-entity level
  mistakes where Databao returned district rows instead of school rows.

### `task_200`

- Question: total atoms with triple-bond molecules containing selected elements.
- Databao SQL:
  - filters atom elements and molecules with triple bonds;
  - returns `COUNT(*)`.
- Output: exact in this run.
- Observation: no Databao change needed for this run. Prior misses came from Databao returning a
  detail/predicate table rather than the aggregate.

### `task_243`

- Question: posts-to-votes ratio for a user.
- Databao SQL:
  - computes post count;
  - computes vote count;
  - computes ratio.
- Output: exact in this run.
- Observation: this run is the desired Databao behavior. Earlier wrong runs returned only
  `post_count`; shadow mode can detect that as `ratio_question_without_ratio_operation`.
- Next possible Databao-internal improvement: for ratio questions, require numerator, denominator,
  and ratio evidence before submit. If only one count exists, Databao should continue or revise.

### `task_257`

- Question: total views on a named post and the user who posted it last time.
- Databao SQL:
  - selects title, view count, last-editor display name, and owner display name from posts;
  - filters title only;
  - does not join a user/display table.
- Output: correct `ViewCount`, wrong/missing display value.
- Gold: `ViewCount=1708`, `DisplayName=mbq`.
- Observation: Databao chooses the right post and metric but submits blank/insufficient display
  columns. The adapter can normalize IDs and avoid slow/wrong mappings, but cannot recover the
  missing display if Databao does not output or join the relevant user ID.
- Next possible Databao-internal improvement: add a blank-display critique. If selected display
  columns are blank and source row contains related user IDs, Databao should join the display table
  or select the ID needed for adapter resolution before submitting.

### `task_283`

- Question: percentage of records with a selected eye color.
- Databao SQL:
  - computes `COUNT(*)` for matching eye-color IDs;
  - does not compute denominator or percentage.
- Output: numerator count only.
- Gold: percentage.
- Observation: same family as ratio/count incomplete. Shadow mode flags
  `ratio_question_without_ratio_operation`.
- Next possible Databao-internal improvement: ratio/percentage submit gate. For percentage questions,
  Databao should not submit a single numerator count unless denominator and percentage operation are
  present.

## Error Families

| Family | Tasks | Databao Symptom | Best Next Fix Location |
| --- | --- | --- | --- |
| Categorical value grounding | `task_11` | Natural language label mapped to wrong numeric code | Databao schema/value grounding before SQL filter |
| Entity ID lookup validation | `task_89` | Guessed ID gives plausible but wrong row | Databao should verify ID via entity table join/sample |
| Requested attribute vs arbitrary category | `task_163` | Useful aggregate but wrong grouping/output attribute | Databao submit critique before final answer |
| Unit/time granularity | `task_169` | Correct aggregate population but missing monthly conversion | Databao unit critique or schema-aware calculation |
| Date literal/type mismatch | `task_173` | Empty result with date filter | Databao empty-result retry using sampled date format |
| Over-broad filtered detail | `task_180` | Reasonable SQL returns far too many rows | Databao filter-audit against question constraints |
| Ratio/percentage incomplete | `task_243`, `task_283` | Numerator count without denominator/ratio | Databao ratio submit gate; rescue only with complete evidence |
| Blank display / missing join | `task_257` | Correct metric but blank display columns | Databao blank-display critique and user/display join |

## Proposed Next Internal Changes

1. **Submit critique inside Databao before final output**
   - Reject or revise final answers when:
     - ratio/percentage question has no ratio operation;
     - selected display columns are blank while related IDs exist;
     - SQL result is empty after date filters;
     - list/table question uses an unverified guessed entity ID.
   - This should run inside Databao's loop, before submit, not as a final CSV patch.

2. **Value grounding for categorical/numeric-code fields**
   - Before filtering a code column by a natural-language label, inspect distinct values, metadata,
     or nearby field definitions.
   - This targets the `task_11` family without task-specific mappings.

3. **SQL/entity lookup validation**
   - When Databao resolves an ID from a natural-language entity, validate it by joining the entity
     table on the name/year/date text used in the question.
   - This targets `task_89`-like wrong-ID failures.

4. **Blank-display repair inside Databao**
   - If a submitted table contains blank display columns plus related ID columns, Databao should run
     one more lookup/join to resolve the display value before submitting.
   - This is safer inside Databao because it can choose the intended relation before the adapter sees
     only a lossy output table.

## Guardrails

- Do not add task-id, difficulty, file-stem, or public-domain branches.
- Do not execute arbitrary generated Python.
- Do not turn shadow observations directly into candidates.
- Promote only checks that are based on SQL/code evidence, schema evidence, and candidate shape.

## Databao Internal Patch: Previous Non-Empty Result Salvage

Date: 2026-05-14

Patch location:

- Clone: `D:\Data\_external\databao-agent`
- Files changed in the clone:
  - `databao/agent/executors/lighthouse/graph.py`
  - `databao/agent/executors/separate/graph.py`
- The same two Python files were copied into the current `.venv` package for local testing because
  `uv pip install -e D:\Data\_external\databao-agent` currently fails in the frontend build hook
  when `pnpm` is unavailable.

What changed:

- Databao graph state now tracks:
  - `last_non_empty_query_id`
  - `last_non_empty_sql`
  - `last_non_empty_df`
- Every successful non-empty `run_sql_query` updates those fields.
- If the model does not call `submit_result`, or the latest tool call is not `submit_result`, and
  the latest result is empty, Databao returns the previous non-empty query result instead.
- The returned metadata includes:
  - `submit_called=False`
  - `salvaged_previous_non_empty_result=True`
  - `salvaged_previous_non_empty_query_id=<query_id>`

Why this is an internal Databao fix:

- The adapter no longer has to infer which previous SQL result was useful after Databao overwrote
  `df/sql` with an empty late query.
- This addresses a loop/execution-state issue inside Databao's graph rather than adding a
  benchmark-shaped answer candidate outside Databao.
- It is generic: it does not depend on task id, difficulty, file stem, public domain terms, or gold.

Targeted validation after patch:

| Task | Run ID | Result | Observation |
| --- | --- | --- | --- |
| `task_349` | `20260514T044731Z` | prediction written, score `0.333` | Normal submitted result; no previous-result salvage. |
| `task_352` | `20260514T044746Z` | exact, score `1.000` | `salvaged_previous_non_empty_result=True`, query id `24-0`. |
| `task_420` | `20260514T044921Z` | exact, score `1.000` | `salvaged_previous_non_empty_result=True`, query id `47-0`. |

Hard full sweep after patch:

| Run ID | Prediction | Exact | Avg | Notes |
| --- | --- | --- | --- | --- |
| `20260514T045138Z` | `9/11` | `4/11` | `0.382` | Improved over the prior hard baseline (`8/11`, `3/11`, `0.273`). |

Remaining missing in that hard sweep:

- `task_355`: task-level timeout before prediction. Needs deeper Databao loop/runtime inspection.
- `task_415`: OpenRouter `403 Key limit exceeded`, not a runner or Databao logic failure in this
  run.

Next internal direction:

1. Keep this previous-non-empty-result salvage if easy/medium regression checks pass.
2. For remaining semantic failures, prefer Databao-internal submit critique:
   - ratio/percentage answer must include numerator and denominator evidence;
   - empty latest SQL should trigger retry or previous-result submission;
   - blank display columns should trigger one more join/lookup before submit;
   - guessed entity IDs should be validated against entity/name tables before submit.
3. Avoid adapter-side result rewriting unless Databao exposes complete operation evidence.

## Databao Internal Patch: Submit Critique Shadow/Reject

Date: 2026-05-14

Patch location:

- Clone: `D:\Data\_external\databao-agent`
- Files changed in the clone:
  - `databao/agent/executors/lighthouse/graph.py`
  - `databao/agent/executors/separate/graph.py`
- The same files were copied into `.venv\Lib\site-packages\databao\agent\executors\...` for
  local validation.

Mode:

- `DATABAO_INTERNAL_SUBMIT_CRITIQUE_MODE=shadow` by default.
- `off`: disabled.
- `shadow`: record critique metadata but do not block submit.
- `reject`: reject the first unsafe submit for a query id and feed a ToolMessage back to Databao so
  it can revise. A repeated submit for the same query id is allowed to avoid infinite loops.

Current generic critique checks:

- Empty submitted DataFrame.
- Ratio/percentage question submitted without ratio/percentage evidence.
- Aggregate question submitted as detail rows.
- Count/how-many question submitted without a count-like result.
- `LIMIT` without `ORDER BY`.
- Very wide non-list result.

Targeted shadow/reject findings:

| Task | Mode | Run ID | Outcome | Lesson |
| --- | --- | --- | --- | --- |
| `task_408` | shadow | `20260514T055942Z` | Critique flagged `ratio_question_without_ratio_evidence`. | Good shadow signal: submitted detail rows for a percentage question. |
| `task_408` | reject | `20260514T060255Z` | Databao revised into a scalar `percentage_faster`, but value was still wrong. | Critique can improve answer shape/operation, but not entity/race grounding. |
| `task_344` | reject | `20260514T060543Z` | Databao revised into a count column, but value was still wrong. | Critique can force count output, but cannot fix wrong filters/value grounding by itself. |

Decision:

- Keep submit critique as Databao-internal `shadow` diagnostics for now.
- Do **not** enable `reject` globally yet.
- If using `reject`, run targeted ablations only. It may improve operation shape but can still leave
  semantic errors, and extra turns increase timeout risk.

Next possible Databao-internal work:

1. Add value-grounding checks before SQL filters on categorical/numeric-code columns.
2. Add entity lookup validation when natural language names/years are mapped to IDs.
3. Add a no-submit critique path, because many hard failures stop without a `submit_result` call and
   therefore never pass through submit critique.

## Databao Internal Patch: Grounding Critique Shadow

Date: 2026-05-14

Patch location:

- Clone: `D:\Data\_external\databao-agent`
- Files changed in the clone:
  - `databao/agent/executors/lighthouse/graph.py`
  - `databao/agent/executors/separate/graph.py`
- The same files were copied into `.venv\Lib\site-packages\databao\agent\executors\...` for
  local validation.

What changed:

- Submit critique now parses SQL equality filters such as `column = 3` and records grounding
  evidence when the numeric value is not present in the user question.
- Two generic shadow flags were added:
  - `numeric_identifier_filter_needs_grounding`
  - `numeric_code_filter_needs_value_grounding`
- These flags are **diagnostic only**. They are not included in `should_reject` and do not change
  Databao output, candidate ranking, final guard, or prediction CSV.
- The implementation intentionally avoids task ids, difficulty routing, file stems, public-domain
  dictionaries, and gold-derived constants. The only logic is SQL text evidence plus generic column
  families such as ID/code versus metric/date/amount columns.

Targeted shadow validation:

| Task | Run ID | Result | Grounding Observation |
| --- | --- | --- | --- |
| `task_11` | `20260514T062057Z` | prediction written, score `0.000` | Correctly flagged `Thrombosis = 3` as `numeric_code_filter_needs_value_grounding`. The SQL shape was plausible, but the label-to-code grounding was wrong. |
| `task_89` | `20260514T062112Z` | prediction written, score `0.000` | Correctly flagged `raceId = 34` as `numeric_identifier_filter_needs_grounding` and `position = 2` as a numeric code filter. The failure is upstream entity/value grounding, not adapter column pruning. |
| `task_344` | `20260514T062130Z` | exact, score `1.000` | No submit critique was available because the selected candidate was a salvaged intermediate. This confirms grounding critique only covers submit paths. |
| `task_408` | `20260514T062254Z` | missing prediction | Databao hit `GraphRecursionError` before submit, so submit grounding critique never ran. This task needs no-submit/loop grounding diagnostics, not a submit-only gate. |

Conclusion:

- The grounding critique is useful as a shadow diagnostic: it identifies wrong numeric code/ID
  filters that explain several remaining wrong-answer tasks.
- It is not yet a safe fix. Enabling reject on these flags would likely improve some tasks but may
  also reject legitimate numeric filters that are implied by joins or schema definitions.
- A targeted grounding-reject ablation (`DATABAO_INTERNAL_SUBMIT_CRITIQUE_MODE=reject` plus
  `DATABAO_INTERNAL_GROUNDING_REJECT=1`) on `task_11`, `task_89`, `task_408`, and `task_415` did not
  produce reliable score gains. The model often repeated the same query or returned a wider
  intermediate table. Keep grounding reject disabled by default.
- To reduce false positives in the grounding signal itself, the Databao clone now treats generic
  ordinal/ranking words (`first`, `second`, `champion`, `winner`, etc.) as numeric grounding
  evidence. This prevents filters such as `position = 2` from being flagged when the question says
  "second", while still flagging unverified entity IDs such as `raceId = 28`.
- The next internal Databao work should focus on a **grounded filter tool loop**:
  1. when a SQL equality uses a numeric ID/code not present in the question, ask Databao to validate
     it by joining/inspecting lookup tables or distinct values;
  2. when no `submit_result` occurs, run a no-submit diagnostic over the last SQL calls and expose
     the same grounding flags;
  3. only after targeted ablation should any grounding flag become a reject condition.

## Databao Internal Patch: No-Submit / Loop Diagnostics

Date: 2026-05-14

Patch location:

- Clone: `D:\Data\_external\databao-agent`
- Files changed in the clone:
  - `databao/agent/executors/lighthouse/graph.py`
  - `databao/agent/executors/separate/graph.py`
- The same files were copied into `.venv\Lib\site-packages\databao\agent\executors\...` for
  local validation.

What changed:

- When Databao ends without `submit_result` but returns the latest query result or salvages the
  latest previous non-empty query, `get_result()` now appends a no-submit critique to
  `submit_critiques`.
- The no-submit critique records:
  - `no_submit_result`
  - `no_submit=true`
  - `no_submit_reason`
  - row/column shape of the returned intermediate result
  - SQL operation flags
  - grounding evidence from the same numeric ID/code equality parser used by submit critique
- This is still **diagnostic only**. It does not alter Databao's returned frame and does not change
  adapter ranking or final guard.

Targeted validation:

| Task | Run ID | Result | No-Submit Observation |
| --- | --- | --- | --- |
| `task_352` | `20260514T063034Z` | exact, score `1.000` | Candidate was `databao_salvaged_intermediate`; critique recorded `no_submit_result` with reason `latest_tool_call_was_run_sql_query`. |
| `task_420` | `20260514T063146Z` | exact, score `1.000` | Candidate was `databao_salvaged_intermediate`; critique recorded `no_submit_result` plus `ratio_question_without_ratio_evidence` before adapter verifier compacted the result. |
| `task_408` | `20260514T063751Z` | prediction written, score `0.000` | Graph recursion no longer caused a missing prediction in this run. Submit critique flagged `raceId = 28` as ungrounded and flagged incomplete ratio evidence; the remaining failure is semantic grounding/calculation, not missing output. |

Conclusion:

- We can now distinguish three important Databao internal states in logs:
  1. clean final submit;
  2. no-submit but useful intermediate returned;
  3. graph recursion before any usable frame.
- This matters for next-step design: no-submit intermediate cases can be analyzed with SQL/code
  observation and grounding critique, while true recursion/no-frame cases need separate loop-state
  capture or Databao graph changes before adapter-side analysis can see enough evidence.
- The `task_408` run shows the boundary of this patch: preserving the last usable state can remove a
  missing prediction, but fixing the score requires a grounded lookup/validation step before Databao
  trusts numeric entity IDs or code values.

## Task 355 Timeout Investigation

Date: 2026-05-14

Question:

- "Write the full name of the member who spent money for water, veggie tray and supplies and include
  the cost of it."

Data shape:

- `csv/expense.csv`: 32 rows, 7 columns.
- `doc/member.md`: about 33k characters.
- Generic `document_records`: 73 rows.
- The direct data path is small: filter the expense row whose description contains the requested
  items, then resolve `link_to_member` through member document records.

Observed runs:

| Run ID | Result | Key observation |
| --- | --- | --- |
| `20260514T064022Z` | missing prediction | Hard sweep timed out at 150s. Progress stopped in `databao_ask` with `llm_call_count=0`. |
| `20260514T093917Z` | missing prediction | Single-task run timed out at 150s. New checkpoints showed progress reached `databao_thread_ask_start` but not `thread.ask` completion. |
| `20260514T094344Z` | missing prediction | Deeper Lighthouse checkpoints showed sources initialized, system prompt rendered (`5863` chars), execution core started, graph stream started, and the initial graph state was emitted. It then stalled before any Databao chat callback. |
| `20260514T095049Z` | missing prediction | Even with `DATABAO_DATABAO_TIMEOUT_SECONDS=30` and task timeout 80s, it stayed at the same `databao_graph_first_state` checkpoint. |

Conclusion:

- This failure is not caused by context loading, document materialization, source registration, or
  prompt rendering.
- It is also not a late SQL loop, submit critique, final guard, or adapter postprocess problem.
- The task stalls after LangGraph emits the initial state and before Databao records any chat-model
  callback. Practically, it behaves like the first model invocation is hanging in the LangGraph /
  LangChain / provider call path without respecting the configured Databao LLM timeout.
- Changing Databao's retry attempt count was tested and did **not** fix this task, so that experiment
  was reverted.

Likely next fix:

- Add a true ask-stage process boundary or hard watchdog around Databao's first graph/model step.
- This should be independent of task id/difficulty and should preserve any frame produced before the
  kill. For `task_355`, because no frame is produced, a watchdog alone will turn a long timeout into
  an earlier classified failure; a separate generic lookup/structured path would be needed to write a
  useful fallback answer.

Semantic note:

- Successful historical runs for `task_355` often still returned the wrong member rows because
  Databao searched `document_records` for names instead of using `expense.csv` as the anchor and then
  resolving `link_to_member`.
- A high-generality Databao fix would be a grounded join/lookup behavior: when a small structured
  table has a strong row match on a text column and an ID/link column, Databao should anchor on that
  row first, then resolve the linked entity in documents or lookup tables.

Ask-stage watchdog experiment:

| Run ID | Setting | Result | Observation |
| --- | --- | --- | --- |
| `20260514T095903Z` | `DATABAO_ASK_STAGE_TIMEOUT_SECONDS=75` | prediction written in `27.5s`, score `0.000` | The ask-stage child returned successfully with 4 LLM calls. This confirms a process boundary can avoid the parent task hanging in `thread.ask`. The answer was still semantic-wrong: Databao returned all expense rows whose description contained any of the words `water`, `veggie`, or `supplies`, instead of the single row matching the full item phrase and resolving `link_to_member` to a full name. |

Decision:

- Keep the ask-stage watchdog as a default-off experiment for stability testing.
- It should not be considered a scoring fix for `task_355`.
- If promoted, the child process must receive the same retrieval/context settings as the parent; the
  experiment currently showed a larger child prompt because the spawned process does not inherit
  contextvars.
- The next scoring-oriented Databao fix should target generic anchored lookup semantics:
  1. choose the structured table row that matches all item/phrase constraints rather than any-token
     broad OR filters;
  2. resolve `link_to_*` identifiers to display/name columns before submit.

## Anchored Lookup Critique Experiment

Date: 2026-05-14

Patch:

- The local Databao clone now adds a generic anchored-lookup critique to both Lighthouse and
  Separate executors.
- The critique records two new flags:
  - `disjunctive_text_filter_needs_anchored_match`: SQL uses broad `OR`/`LIKE` text filters while
    the question wording suggests a single record should satisfy multiple requested terms.
  - `linked_identifier_needs_display_lookup`: a submitted result still contains `link_to_*` columns
    while the question asks for a person/member/name-like display answer.
- `DATABAO_INTERNAL_ANCHORED_LOOKUP_REJECT=1` can turn these flags into one submit rejection. The
  default remains diagnostics/shadow-oriented; this is not enabled for full sweeps.
- The link-resolution critique now includes sample link values, e.g. `{"link_to_member": ["..."]}`,
  so Databao has concrete IDs to resolve through lookup/document tables.

Targeted `task_355` results:

| Run ID | Setting | Score | Observation |
| --- | --- | --- | --- |
| `20260514T100545Z` | anchored reject, one rejection | `0.250` | Databao anchored on the correct expense row and cost but still failed to resolve `link_to_member` to the member full name. |
| `20260514T101239Z` | two rejections for the same query | `0.000` | Rejected too aggressively; Databao switched to a bad `document_records` row. This change was reverted. |
| `20260514T101511Z` | one rejection plus link-value hint | `0.000` | Databao broadened back to all water-related expense rows. The hint is useful diagnostically but not a scoring fix by itself. |

Decision:

- Keep the anchored lookup critique and link-value evidence as Databao-internal diagnostics.
- Do **not** promote anchored rejection globally yet. It can improve one failure mode, but targeted
  runs show it can also push Databao into a worse broad-detail or bad-document lookup answer.
- The next useful step is not more critique text. It is either:
  - a real Databao-internal lookup executor that can resolve `link_to_*` values through registered
    tables/doc records; or
  - a better generic document-record extractor so IDs like `rec...` map to the actual display name
    instead of weak prose fragments.

## Databao Internal Patch: P0 Submit Gates (v1)

Date: 2026-05-14

Patch location:

- Clone: `D:\Data\_external\databao-agent` (graph helpers are now mirrored here for parity).
- Files changed in this repository:
  - `src/data_agent_baseline/_vendor/databao_patches/lighthouse_graph.py`
  - `src/data_agent_baseline/_vendor/databao_patches/separate_graph.py`
- `ensure_vendor_databao_patches()` copies these into `.venv\Lib\site-packages\databao\agent\executors\*\graph.py`
  on every runner startup, so a fresh checkout is reproducible.

Goal:

- Move the ratio/percentage, empty-result, blank-display, and entity-ID validation gates from
  shadow-only into actual submit rejections, while keeping the adapter ranking and final guard
  unchanged. Adaptor observes and stabilizes; Databao reasons and repairs before submit.

What changed:

- New env knobs:
  - `DATABAO_INTERNAL_P0_GATES` (default `1`) — master switch for the P0 reject path.
  - `DATABAO_INTERNAL_P0_MAX_REJECTIONS` (default `3`) — per-task session-wide reject budget; the
    existing one-rejection-per-query-id guard still applies.
  - `DATABAO_INTERNAL_P0_RATIO_REJECT` (default `0`) — opt-in ratio gate (see below).
- New `P0_REJECT_FLAGS` set (default): `empty_submit_result`, `blank_display_needs_join`,
  `numeric_identifier_filter_needs_grounding`. Promoted to reject inside the lighthouse/separate
  graph regardless of `DATABAO_INTERNAL_SUBMIT_CRITIQUE_MODE`.
- New `_blank_display_critique`: when the question asks for a display/name and the submitted
  DataFrame either has display columns that are all-blank, or only contains identifier columns,
  Databao gets a rejection with the blank columns + identifier samples to resolve through a join.
  Pattern matches both snake_case (`full_name`, `user_id`) and camelCase (`OwnerDisplayName`,
  `OwnerUserId`, `LastEditorUserId`) columns.
- Empty-result suggestion is augmented with a date-format hint when the failing SQL filters a
  date/datetime-like column. Databao is told to inspect the column dtype and sample values before
  re-submitting.
- Shadow-only `_unit_time_granularity_critique` and `_filter_audit_critique` were added; they
  record diagnostic flags but do not affect submit/reject.
- Each reject record now includes `reject_source=p0_gate|mode_reject` for attribution.
- **Critical fix**: `_latest_user_question` now strips the runner's system-instruction prefix,
  returning only the text after the last `Question:` marker. Without this, the runner prompt's
  `"Prefer display, name, title, or label columns..."` scaffolding spuriously triggered
  `asks_display=True` on every task, firing the blank-display gate on non-display questions.

Why ratio is opt-in (not default P0):

- The adapter's `apply_aggregate_ratio_verifier` and friends already rescue count-detail submits
  into scalar ratio/percentage candidates when Databao has produced the numerator-form table.
- Forcing Databao to reject and re-submit a ratio query loses that rescue input. On `task_283`
  (percentage of superheroes with blue eyes), the count-detail submit was a verifier-rescuable
  shape; the first P0-with-ratio smoke produced a 21-row name list instead of the percentage.
- Demoting ratio to opt-in restored the verifier path. The flag still fires as shadow for analysis.

Why grounding/empty/blank-display ARE default P0:

- `blank_display_needs_join`: cleanest signal in this smoke. Variance probe on `task_257` showed
  3/3 matches with the gate firing; the model reliably joins the user table when told the relevant
  identifier samples.
- `numeric_identifier_filter_needs_grounding`: fires when Databao filters by a numeric ID not
  present in the question. The model doesn't always revise correctly under Qwen3.5-35B-A3B, but
  it fires alongside other flags and doesn't appear to make things worse in aggregate.
- `empty_submit_result`: forces Databao to attempt a non-empty re-query. With the date-format
  hint, the model has more context to retry. On `task_173` the model still failed to fix the
  date literal in one shot, but the gate is correct in semantics and may help with stronger
  models or with longer task budgets.

Targeted validation (P0 gates enabled, all sweeps used the same Qwen3.5-35B-A3B):

| Task | Baseline | Phase 2 | Reject fired | Notes |
| --- | --- | --- | --- | --- |
| `task_257` (variance probe 3x) | partial | 3/3 exact | `blank_display_needs_join` | Strong signal: gate consistently helps. |
| `task_89` (variance probe 3x) | wrong row | 0/3 exact | `id_grounding`+`blank_display` | Qwen does not revise the guessed `raceId`; gate is correctly classifying but model can't act. |
| `task_283` (variance probe 3x) | exact (variance) | 1/3 exact | mostly none (ratio demoted) | Verifier path is non-deterministic. Score variance is model-driven, not gate-driven. |
| `task_200` | 1.0 | 1.0 | none | Confirms prompt-prefix fix: blank_display does not fire on count questions anymore. |

Full sweeps (official `dabench evaluate-public-run`, same gold set):

| Difficulty | Baseline avg | Phase 2 avg | Delta avg | Baseline exact | Phase 2 exact | Prediction |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Easy   (`20260514T103419Z` → `20260514T151526Z`) | `0.800` | `0.800` | `+0.000` | `12/15` | `12/15` | `15/15` → `15/15` |
| Medium (`20260514T103857Z` → `20260514T151940Z`) | `0.739` | `0.761` | `+0.022` | `17/23` | `17/23` | `23/23` → `23/23` |
| Hard   (`20260514T045138Z` → `20260514T153026Z`) | `0.382` | `0.545` | `+0.163` | `4/11`  | `6/11`  | `9/11`  → `11/11` |

Decision:

- Keep P0 gates default-on. Net positive across Easy/Medium/Hard, with the largest gain on Hard
  (where Databao previously left the most "almost-right" submits on the table).
- Keep ratio reject opt-in. Re-evaluate once a stronger model or a Databao-internal numerator/
  denominator inspector exists.
- Empty-result gate stays default-on. It does not regress current results and provides clearer
  retry signals for the model.

Next internal directions:

1. Make grounding/blank-display reject feedback more actionable by also including small candidate
   join paths derived from registered table column families (e.g. "`Owner.Id` → `Users.Id`").
2. Add a real Databao-internal lookup tool that resolves `link_to_*` and `*_id` values to a
   display column without round-tripping through the LLM. The current critique points the model
   at the right column but relies on Databao to write the SQL.
3. Re-test ratio reject when adapter ratio verifier is moved into Databao as well, so the two
   paths do not contend for the same rescue input.

## Databao Internal Patch: Empty-Result Date Probe (Stage 2.1)

Date: 2026-05-15

Patch location:

- `src/data_agent_baseline/_vendor/databao_patches/lighthouse_graph.py`
- `src/data_agent_baseline/_vendor/databao_patches/separate_graph.py` (helpers only;
  executor-side probe wiring is currently lighthouse only because that is the default executor).

What changed:

- When the `empty_submit_result` gate fires, the executor automatically runs
  `SELECT DISTINCT <date_col> FROM <same FROM clause> LIMIT 5` against any column in the failing
  SQL that is compared with a date/datetime literal. The real samples are embedded in the reject
  ToolMessage so Databao can fix the date literal format on its next retry.
- Helpers added:
  - `_date_filter_qualified_columns(sql)` — extracts qualified column references that participate
    in a date-like comparison. Pattern matches `column_with_date_or_time_or_month_or_year_keyword`
    followed by a comparison operator.
  - `_extract_from_clause(sql)` — extracts the FROM/JOIN clause from the original SQL.
  - `_build_date_probe_sql(original_sql, qualified_col)` — composes `SELECT DISTINCT col FROM
    <from_clause> LIMIT 5`.
- The reject suggestion is now more directive: it tells Databao to rewrite the original query's
  date literal using the probe samples and explicitly says "do not submit these probe samples".
- The critique record now always includes the raw `sql_text` and `question_text`. This is
  diagnostic-only; the always-record change replaces the previous "append only if flags" rule so
  empty-flag submits are still visible in artifacts for analysis.

Probe trace from `task_173` (June 2013 transactions question):

| Step | Detail |
| --- | --- |
| Initial submit | `SELECT DISTINCT gs.Country ... WHERE t.Date >= '2013-06-01' AND t.Date <= '2013-06-30'` (empty) |
| Probe | `SELECT DISTINCT t.Date FROM db_transactions_1k.transactions_1k t JOIN temp.main.json_gasstations gs ON ... LIMIT 5` |
| Probe samples | `2012-08-23, 2012-08-24, 2012-08-25, 2012-08-26` |
| Lesson | The 1k subsample only covers August 2012; June 2013 transactions are not present in that table. |
| Revised submit | Databao pivoted to `temp.main.csv_yearmonth` joined with gas stations on the right date, returning `CZE, SVK`. Exact match. |

This is the cleanest "Databao reasons and repairs before submit" outcome so far: the probe gave
Databao evidence that the named SQL source was wrong, and it switched data sources on its own.

Decisions:

- Keep the empty-result probe default-on (`DATABAO_INTERNAL_P0_GATES=1`).
- Skip `unit_time_granularity_mismatch` P0 promotion. The `task_169` failure is benchmark-specific
  (gold computes `AVG(Consumption) / 12` even though the table already stores monthly rows);
  rejecting on a generic monthly/year detector would also kill legitimate per-row monthly
  interpretations.
- Skip `numeric_code_filter_needs_value_grounding` P0 promotion. On `task_11` the model failed at
  mapping "severe degree" to the right code despite knowledge.md providing the explicit mapping.
  A reject would only echo the same reasoning step the model already failed.
- Skip multi-constraint filter-audit promotion. `task_180` failed because the audit regex did not
  recognise the implicit `product id=5` + `August 2012` constraints. Broadening the regex would
  introduce many false positives on legitimate detail-row outputs.

Full sweep deltas (official `dabench evaluate-public-run`):

| Difficulty | Stage 1 avg | Stage 2 avg | Stage 1 exact | Stage 2 exact | Notes |
| --- | ---: | ---: | --- | --- | --- |
| Easy   | `0.800` | `0.717` | `12/15` | `10/15` | Two losses (`task_27`, `task_86`) are adapter/model variance — same submitted SQL, different candidate selected. |
| Medium | `0.761` | `0.826` | `17/23` | `19/23` | Gains: `task_173` (probe-driven source pivot), `task_257` (blank-display gate). `task_199` regression is model variance with zero rejects. |
| Hard   | `0.545` | `0.545` | `6/11`  | `6/11`  | No net change. |

Across all 49 tasks, exact count is preserved at 35/49 while column coverage improves slightly.
The probe is the load-bearing architectural change; the rest of the deltas are within model
variance on a single run.

Next directions remain the same: an internal lookup tool for `link_to_*`/`*_id` resolution and
better grounding evidence in critique suggestions would unlock the `task_89`/`task_344`/`task_408`
family where the gate fires but the model cannot self-correct.

## Databao Internal Patch: Grounding Probe (Opt-in After Stage 3 Ablation)

Date: 2026-05-15

Patch location:

- `src/data_agent_baseline/_vendor/databao_patches/lighthouse_graph.py`

What was tried:

- When the `numeric_identifier_filter_needs_grounding` or
  `numeric_code_filter_needs_value_grounding` gate fired, the executor auto-probed the matching
  table with two queries:
  - `SELECT * FROM <table> WHERE <col> = <value> LIMIT 3` — shows the actual matching row(s) so
    the model can compare its filter against the question's natural-language entity.
  - `SELECT DISTINCT <col> FROM <table> ORDER BY <col> LIMIT 20` — shows the full enum/domain for
    categorical filters so the model can see whether the chosen code is in range.
- Helpers added: `_grounding_qualified_equalities`, `_resolve_alias_to_table`,
  `_build_grounding_probe_sql`, `_build_distinct_domain_probe_sql`.
- Probe results were embedded in the reject suggestion text along with explicit guidance to
  compare the matched row against the question text.

Default-on ablation outcome (`DATABAO_INTERNAL_GROUNDING_PROBE` implicitly enabled in the early
Stage 3 sweep before the env knob was introduced):

| Difficulty | Stage 2 avg | Stage 3 default-on avg | Δ | Stage 2 exact | Stage 3 exact | Missing pred |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Easy   | `0.717` | `0.800` | `+0.083` | `10/15` | `12/15` | `0/15`  |
| Medium | `0.826` | `0.652` | `-0.174` | `19/23` | `15/23` | `0/23`  |
| Hard   | `0.545` | `0.386` | `-0.159` | `6/11`  | `4/11`  | **`4/11`** |

`task_344`, `task_352`, `task_396`, `task_408` all crossed the 180 s task timeout on Hard. The
probe itself is fast (<1 s per call), but the extra agent turns triggered by the larger reject
suggestion plus the verbose probe text inflated the LLM-call count and pushed those tasks past
their budget. On Medium the extra reject turns also pushed several tasks into suboptimal revised
submits that the adapter could not rescue.

Targeted smoke before rollback:

| Task | Probe ran? | Match | Notes |
| --- | --- | --- | --- |
| `task_11` | Yes | `False` | Probe showed `Thrombosis` domain is `{0,1,2,3}` and `=3` matches real rows. Model still mapped "severe" → 3 instead of 1, ignoring knowledge.md's explicit 1=most-severe mapping. Probe didn't unlock semantic reasoning. |
| `task_89` | Yes | `False` | Probe matched on the wrong table (`csv_results`, the fact table). The actual lookup table that has race name/year is `csv_races`, which isn't in the failing SQL's FROM, so the alias-resolution heuristic couldn't reach it. Model didn't revise. |
| `task_257` | Yes | `False` | Probe correctly showed `Id=88` → `DisplayName='mbq'`, but the model's revised SQL switched to a different user (`'Menno'`). Probe noise plausibly distracted the model from the right answer. |
| `task_86` | Yes | `True` | Score preserved; the cleaner revised SQL stripped the extra `positionText` column. |

Decision:

- Move the grounding probe behind `DATABAO_INTERNAL_GROUNDING_PROBE` (default `0`). Helpers stay
  in the codebase for opt-in ablations and future evolution.
- Keep the empty-result probe default-on. It produced an unambiguous mechanism win (`task_173`
  pivoted to a different data source after seeing real date samples) without timeout cost.
- The Stage 3 lesson is concrete: enriched executor-side feedback only helps when the model can
  act on the evidence cheaply. The empty-result probe gives a one-step fix ("rewrite the literal").
  The grounding probe asks the model to do multi-step entity validation, which on Qwen3.5-35B-A3B
  takes additional turns and frequently destabilizes the revised submit.

After rollback, repeated Easy/Medium/Hard sweeps are within run-to-run variance of Stage 2
numbers (single-run sample noise is ~3 task flips per Medium sweep). The empty-result probe
remains the durable structural win from Stage 2.

Next plausible directions if grounding probe is revisited:

1. Resolve `*Id`-suffix columns against an external entity-lookup table even when that table is
   not in the failing SQL's FROM clause; this requires schema-graph integration outside the
   Databao graph.
2. Shorten the reject suggestion so the model is not flooded with raw CSV; consider compressing
   probe results into a one-line summary like `<id_col>=<value> → name='X', year=Y`.
3. Skip the distinct-domain probe by default (heavier output, lower hit rate); only run the
   match-row probe.

## Variance Probe Outcome (Stop Decision)

Date: 2026-05-15

Setup:

- Same code as `f8de993` (P0 gates default-on, empty-result probe default-on, grounding probe
  opt-in default-off, ratio reject opt-in default-off).
- Three full Easy/Medium/Hard sweeps. Scoring uses the official `evaluate_public_run` column
  signatures with `normalize_cell` (numeric tolerance for `"1"` vs `"1.0"`).

Per-sweep exact counts:

| Difficulty | n | Run A | Run B | Run C | Mean | Stdev | Range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Easy   | 15 | `13` | `12` | `12` | `12.33` | `0.58` | `[12-13]` |
| Medium | 23 | `16` | `14` | `17` | `15.67` | `1.53` | `[14-17]` |
| Hard   | 11 | `5`  | `6`  | `4`  | `5.00`  | `1.00` | `[4-6]`  |
| **Total** | **49** | **34** | **32** | **33** | **33.0** | **1.0** | **[32-34]** |

Per-task stability:

- **Stable wins (3/3)**: 11 Easy, 12 Medium, 3 Hard. 26 tasks consistently right.
- **Stable losses (0/3)**:
  - Easy: `task_11`, `task_89`
  - Medium: `task_163`, `task_169`, `task_173`, `task_180`
  - Hard: `task_344`, `task_355`, `task_379`, `task_396`, `task_408`
- **Flaky (1-2/3)**: 2 Easy (`task_19`, `task_80`), 7 Medium (`task_199`, `task_200`, `task_214`,
  `task_218`, `task_250`, `task_257`, `task_283`), 3 Hard (`task_330`, `task_415`, `task_420`).

Lessons used for the stop decision:

1. Many earlier "Stage X improved by N tasks" claims fell inside the run-to-run variance window.
   Stage 2 Medium 19/23 was the lucky high (mean 15.67); Stage 2 Hard 6/11 was on the high side
   too (mean 5.00). The probe-driven `task_173` pivot was a Stage-2 outlier and did not
   reproduce in any of the 3 probe runs.
2. Stable losses are model-reasoning limited or data-limited:
   - `task_11`: categorical mapping ignores knowledge.md mapping.
   - `task_89`: entity-ID lookup table not in SQL FROM.
   - `task_163`, `task_169`: benchmark-specific aggregate/granularity semantics.
   - `task_173`: subsampled table doesn't cover the asked time range.
   - `task_180`: multi-constraint coverage.
   - Hard family: no-submit / wrong-percentage-formula / ordinal-per-group.
3. Best plausible single-iteration gain from re-trying the grounding probe with compressed
   output is +1-2 tasks on stable losses, with risk of re-introducing the Stage-3 timeouts.
   That expected delta is well inside the ±1 stdev variance band.
4. Competition is a single-shot submission; the submitted run will be a random sample from this
   distribution. Stabilising the existing pipeline is more valuable than chasing edge gains.

Decision:

- Stop at commit `f8de993` for this development branch. Document variance honestly so future
  iterations don't chase phantom improvements.
- Next real lift requires either (a) a stronger upstream model, (b) ensemble across multiple
  runs (not permitted by competition single-shot rules), or (c) a Databao-internal schema-graph
  lookup that can reach entity-lookup tables outside the failing SQL's FROM clause.
  All three are out of scope for incremental P0-gate work.

## Databao Internal Patch: Schema-Graph Lookup Probe (Opt-in)

Date: 2026-05-15

Patch location:

- `src/data_agent_baseline/_vendor/databao_patches/lighthouse_graph.py`

What was added:

- `DATABAO_INTERNAL_SCHEMA_GRAPH_PROBE` (default `0`). When enabled, the grounding probe path
  now uses DuckDB's `information_schema.columns` to find candidate lookup tables for an
  ungrounded numeric equality, even when those tables are NOT in the failing SQL's FROM clause.
  This is the direct response to the Stage 3 finding that the previous grounding probe could
  only reach tables already referenced in the failing SQL.
- New helpers:
  - `_information_schema_columns(connection)` — enumerates all `catalog.schema.table -> [columns]`
    via DuckDB metadata. Cached per-connection.
  - `_find_lookup_tables(connection, column, exclude_tables, limit)` — returns candidate tables
    that contain ``column`` and are not already in the failing SQL's FROM clause. Prefers tables
    whose name matches the column's stem (e.g. an `<entity>Id` column prefers the `<entity>s`
    table) and tables that also expose a display-like column (name/title/label/display).
  - `_select_key_columns(row)` — picks display-like columns (name/title/year/date/...) from a
    probe row for the one-liner summary. Falls back to the first 3 non-id non-blank columns.
  - `_format_probe_one_liner(table, column, value, row)` — compresses a match row into a
    single descriptive line such as ``<id_col>=<value> → <table>: name='...', year='...'``.
  - `_from_tables_in_sql(sql)` — gathers all table references in FROM/JOIN so the cross-table
    lookup can exclude them.
- The reject suggestion is now one short paragraph composed of one-liner probe results joined
  with ``|``. This replaces the verbose raw CSV that caused the Stage 3 timeouts.

Smoke validation (`DATABAO_INTERNAL_SCHEMA_GRAPH_PROBE=1`):

| Task | Probe found | Match | Notes |
| --- | --- | --- | --- |
| `task_89` | `raceId=34 → temp.main.json_races: year='2008', name='<entity>', date='2008-10-19'` | `False` | Probe correctly surfaced the lookup row from a table NOT in FROM. The ID was actually right; the wrong answer was a semantic interpretation of "finish time" (gap vs absolute). |
| `task_86` | `driverId=62 → temp.main.json_drivers: forename='Alex', surname='Yoong'` | `True`  | Probe reached the drivers lookup table; submission preserved exact score. |
| `task_257` | (only `blank_display_needs_join` flag, schema-graph probe path not triggered) | `True`  | Match from the existing blank-display gate alone. |
| `task_11` | `Thrombosis=3 → temp.main.json_Examination: ...` | `False` | Probe surfaced a matching row, but the categorical mapping ("severe" → which code) is a model-reasoning failure, not a missing-evidence failure. |

3-run variance probe (probe ON vs probe OFF on the same code base):

| Difficulty | Probe OFF mean | Probe ON mean | Δ mean | Probe OFF stdev | Probe ON stdev |
| --- | ---: | ---: | ---: | ---: | ---: |
| Easy   | `12.33` | `11.33` | `-1.00` | `0.58` | `1.15` |
| Medium | `15.67` | `16.33` | `+0.67` | `1.53` | `1.15` |
| Hard   | `5.00`  | `4.67`  | `-0.33` | `1.00` | `0.58` |
| **Total** | **33.00** | **32.33** | **-0.67** | — | — |

Per-task changes (3 sweeps each):

- Wins (probe ON > probe OFF): `task_200`, `task_214`, `task_218`, `task_250`, `task_257`,
  `task_420` — most of these are flaky tasks moving to 3/3 wins, but the gains are within the
  variance band rather than a clean structural fix.
- Losses (probe ON < probe OFF): `task_75` (3/3 → 1/3), `task_350` (3/3 → 1/3), `task_80`,
  `task_194`, `task_199`, `task_283`. Inspection of `task_75` showed the OFF run used a proper
  ``JOIN`` (no ungrounded equality) while the ON run took a shorter ``WHERE driverId = 8`` path
  that triggered probe — i.e. the loss is from the model picking a different SQL path, not from
  the probe degrading a known-good submit.

Decision:

- Keep `DATABAO_INTERNAL_SCHEMA_GRAPH_PROBE` default `0`. The probe is mechanically correct and
  occasionally lands the right cross-table lookup, but on this corpus the per-task wins balance
  the per-task losses inside the ±1 stdev variance window.
- Do NOT remove or simplify the adapter-side `resolve_identifier_columns`, ratio verifier, or
  salvaged-result repair functions. The schema-graph probe was the architecturally cleanest
  candidate to replace them inside Databao, and it does not reliably do so on this model. Those
  adapter functions remain load-bearing rescue paths.
- For competition single-shot submission, leave the probe OFF: probe-OFF distribution has a
  slightly higher mean (33.00 vs 32.33) and a higher floor (worst-case 32 vs 30).
- Keep the implementation in the tree so future iterations (stronger model, finer reject
  gating, dedicated blank-display probe variant) can opt in via env var without re-implementing
  the helpers.

Adapter audit (informational):

| Adapter function | Could be Databao? | Status |
| --- | --- | --- |
| `resolve_identifier_columns` | Yes (schema-graph probe variant) | **Keep**: schema-graph probe does not reliably replace it. |
| `apply_aggregate_ratio_verifier` | Yes (P0 ratio reject) | **Keep**: ratio reject is opt-in because it conflicts with this verifier. |
| `apply_salvaged_boolean_percentage_compactor` | Yes (Databao salvaged result handling) | **Keep**: salvages a useful intermediate that Databao itself sometimes drops. |
| `apply_salvaged_detail_count_compactor` | Yes | **Keep**: same reason. |
| `apply_databao_observed_detail_aggregate_compactor` | Yes | **Keep**: same reason. |
| `apply_context_superlative_verifier` | No (needs registered context) | **Keep**. |
| `apply_superlative_verifier` | Borderline | **Keep**: cheap and safe. |
| `apply_ratio_scale_compactor` | No (shape job) | **Keep**. |
| `apply_salvaged_context_boolean_percentage_repair` | No (needs context) | **Keep**. |
| `apply_quoted_entity_ratio_repair` | No (parses question) | **Keep**. |
| `apply_answer_column_verifier` | No (shape job) | **Keep**. |
| `apply_question_column_pruner` | No (shape job) | **Keep**. |
| `_cheap_count_fallback_candidate` | No (Databao-failed branch) | **Keep**. |

Conclusion: with Qwen3.5-35B-A3B the adapter is not redundant. The verifier/compactor functions
are doing real work that Databao does not yet reliably perform in a single submit cycle.
Removing any of them without a stronger model or a proven Databao-internal replacement would
likely regress the corpus.

## Cleanup: Removed Opt-in Experiments (2026-05-15)

After the audit and variance probe above showed that none of the three opt-in Databao-side
experiments reliably moved the corpus mean past noise, all three were removed from the vendored
graph files. They are documented above (kept in the section history) so future iterations don't
re-implement the same path without first considering what the model variance ceiling will allow.

Removed code:

| Feature | Env knob | Removed helpers | Why removed |
| --- | --- | --- | --- |
| Ratio P0 reject | `DATABAO_INTERNAL_P0_RATIO_REJECT` | (none — single env helper) | Destroys adapter aggregate-ratio verifier's rescue input on `task_283` family. |
| Legacy grounding probe (verbose CSV) | `DATABAO_INTERNAL_GROUNDING_PROBE` | `_grounding_qualified_equalities`, `_resolve_alias_to_table`, `_build_grounding_probe_sql`, `_build_distinct_domain_probe_sql` | Default-on ablation caused 4 Hard task_timeouts; default-off kept the helpers dormant for two release cycles with no validated win. |
| Schema-graph grounding probe (one-liner) | `DATABAO_INTERNAL_SCHEMA_GRAPH_PROBE` | `_information_schema_columns`, `_find_lookup_tables`, `_select_key_columns`, `_format_probe_one_liner`, `_from_tables_in_sql` | 3-run variance ablation showed -0.67 mean inside the ±1 stdev noise band. Mechanism worked (cross-table lookup verified on `task_86`) but on this model the per-task wins balanced per-task losses. |

What stays in production:

- All three P0 reject flags (`empty_submit_result`, `blank_display_needs_join`,
  `numeric_identifier_filter_needs_grounding`) default-on via `DATABAO_INTERNAL_P0_GATES=1`.
- Empty-result date probe (always runs when the empty gate fires on a date-filtered SQL). It
  produced the only durable structural win in this entire investigation (`task_173`
  cross-source pivot in Stage 2) and adds no measurable runtime cost.
- Shadow-only critiques: `_blank_display_critique` (also P0), `_unit_time_granularity_critique`,
  `_filter_audit_critique`, anchored-lookup critique, original numeric grounding flags.
- All adapter postprocess/verifier/repair functions identified in the audit above as
  load-bearing.

Cleanup diff: 412 deletions / 0 insertions across `lighthouse_graph.py` and `separate_graph.py`.
Default behavior unchanged (verified on smoke and 100/100 unit tests).

## Adapter Fix: Always Include knowledge.md + Proactive Prompt Hints

Date: 2026-05-15

Patch location:

- `src/data_agent_baseline/run/databao_demo.py` — `register_context_sources` and
  `_build_question_prompt`.

What changed:

1. `register_context_sources` previously skipped `knowledge.md` whenever `retrieved_context`
   produced any snippet (line 1559 had `if retrieved_context is None and knowledge_path.exists():`).
   The assumption was that retrieval is a strict superset of the document, but retrieval can
   miss the key disambiguation sentence — for example, a categorical-value mapping that does
   not literally appear in the question's text never gets ranked into the top retrieved
   snippets. The model then never sees the mapping. knowledge.md is ~5 KB on the public set, so
   the token cost is bounded; always include it alongside retrieval snippets.

   `doc/` files (which can be tens of KB on some tasks) remain under retrieval control to avoid
   blowing the prompt budget.

2. `_build_question_prompt` got two proactive bullets after the existing shape rules:

   ```
   If the registered documentation defines mappings between natural-language labels and
   numeric codes, use those mappings exactly.
   For ratio or percentage questions, compute the ratio in one query; do not submit a
   numerator count alone.
   ```

   An earlier 8-bullet rewrite was tried first and produced destabilized outputs on the weaker
   Qwen3.5-35B-A3B (e.g. `task_89` returning 15 rows instead of 1). Pared back to the two
   highest-impact bullets that do not contradict the existing one-row/one-column shape rules.

3-run variance probe vs the probe-OFF baseline (same official `evaluate_public_run` scoring):

| Difficulty | Baseline mean | New mean | Δ | Baseline stdev | New stdev |
| --- | ---: | ---: | ---: | ---: | ---: |
| Easy   | `12.33` | `13.67` | `+1.34` | `0.58` | `0.58` |
| Medium | `15.67` | `15.67` | `+0.00` | `1.53` | `0.58` |
| Hard   | `5.00`  | `5.00`  | `+0.00` | `1.00` | `1.00` |
| **Total** | **33.00** | **34.33** | **+1.33** | — | — |

Both improvements are real signal:

- Easy mean delta `+1.34` is larger than the Easy stdev (`0.58`), clearing the variance band.
- Medium stdev collapsed from `1.53` to `0.58`. The pipeline is now substantially more stable
  on Medium even though its mean is unchanged — fewer tasks flip between sweeps.

Per-task stability shifts across the three sweeps (baseline trio vs new trio):

- Gains: `task_19` (2/3 → 3/3), `task_80` (2/3 → 3/3), `task_89` (0/3 → 2/3 — the only stable
  loss converted), `task_200` (1/3 → 3/3), `task_257` (1/3 → 3/3), `task_330` (2/3 → 3/3).
- Losses: `task_196` (3/3 → 1/3), `task_199` (1/3 → 0/3), `task_283` (2/3 → 1/3), `task_420`
  (2/3 → 1/3).
- Net `+9 / -5 = +4` task-runs across 3 sweeps, matching the mean `Δ +1.33`.

The biggest single shift is `task_89`. The prompt hint about consulting documentation
mappings, combined with knowledge.md always being in the prompt, lets the model resolve the
entity-ID family that the schema-graph probe never reliably cracked.

`task_196` regression suggests the ratio bullet is over-aggressive for the "average of a ratio
of counts" family. Worth examining if a future iteration shortens or qualifies that bullet,
but the current net delta is clearly positive.
