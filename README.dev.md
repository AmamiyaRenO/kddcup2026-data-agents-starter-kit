# Databao Demo Development Notes

This file is a working memory note for local development in this starter-kit clone.
It is not part of the official submission package. Function-level provenance for local helper
layers is tracked in `README.function-ledger.md`.

## Local Setup

- Repository: `D:\Data\kddcup2026-data-agents-starter-kit`
- Public demo data: `data/public/input` and `data/public/output`
- Local config: `configs/react_baseline.local.yaml`

Use this cache workaround before `uv` commands on this machine:

```powershell
$env:Path = "C:\Users\tianj\.local\bin;$env:Path"
$env:UV_CACHE_DIR = "D:\Data\.uv-cache"
```

OpenRouter testing uses environment variables only. Do not write secrets into files:

```powershell
$env:MODEL_API_URL = "https://openrouter.ai/api/v1"
$env:MODEL_API_KEY = $env:Open_Router_Key
$env:MODEL_NAME = "qwen/qwen3.5-35b-a3b"
```

## Current Runner Shape

`src/data_agent_baseline/run/databao_demo.py` keeps the original ReAct baseline separate and adds a
Databao-backed, adapter-level data-agent workflow.

The current execution flow is:

1. Load `task.json` and `context/`.
2. Load CSV, JSON, SQLite, and generic markdown records into `ContextTable` objects.
3. Build a schema graph with table profiles, column samples, and join candidates.
4. Run `query_context_retriever()` to score relevant tables, columns, samples, snippets, and join
   paths. Retrieval is diagnostic soft priority, not access control: Databao still receives
   registered structured sources.
5. Infer an `AnswerContract` from the question and schema.
6. Run Databao first. As soon as Databao returns any DataFrame, write a raw provisional
   `prediction.csv` before postprocess, ranking, verifier, or final guard work can fail.
7. Generate additional candidates from cheap repair/verifiers. LLM planner rescue and extra
   generator experiments are not connected to the active runner.
8. Validate and rank candidates with a generic evaluator.
9. Apply `final_answer_shape_guard()` again before the final CSV write.
10. Write diagnostics, candidate scores, selected route, scorable status, timings, and safe ablation
   metadata.

Question parsing is now a signal layer rather than a decision layer. `extract_question_features()`
collects evidence such as aggregation, ratio/percentage, list/entity, multi-attribute, entity-plus-
metric, and superlative signals. Weak words such as `number`, `rate`, `value`, and `view count` only
gain real confidence when schema columns support them. `AnswerContract`, candidate ranking, and
cheap fallback decisions combine these question signals with schema evidence and candidate evidence
instead of letting a single regex change row or column limits.
List/entity and entity-plus-metric signals do not by themselves fix `max_columns`; column limits are
set only when requested columns can be inferred from question/schema evidence. If evidence is weak,
the contract stays open and the final guard only removes metadata/debug and safe helper columns.

This is intentionally not a benchmark-memory path. The runner no longer keeps production parsers or
routes for concrete public-set domains, entity names, category values, file stems, or task ids.
The only heuristic modes are:

- `DATABAO_HEURISTIC_LEVEL=generic` (default): generic loaders, schema graph, document records,
  answer contracts, verifiers, candidate ranking, and final shape guard.
- `DATABAO_HEURISTIC_LEVEL=experimental_generic`: allows slightly more aggressive generic
  paragraph extraction for records without explicit ids. It still must remain domain-agnostic.

## Candidate Workflow

Databao is treated as a black-box candidate generator. The adapter layer controls context,
validation, fallback, ranking, and output shape:

- `databao_raw`: the raw DataFrame returned by Databao.
- `databao_raw_postprocessed`: a deterministic repair candidate when generic postprocessing changes
  the raw DataFrame.
- `*_column_compact`: a conservative column-only cleanup candidate. It preserves all rows and only
  removes empty columns, duplicate-value columns, or constant display helper columns when another
  answer-like column remains.
- `cheap_count_fallback`: a low-confidence missing-prediction fallback used only after Databao
  failure when the question explicitly asks for plain row/record/entry count and a single
  high-confidence table can be selected. It is skipped for unexecuted filters, grouping, distinct,
  per/by/for clauses, temporal constraints, or other conditions.
- verifier candidates: generic superlative, context superlative, and aggregate/ratio checks. These
  are ranked like any other candidate and no longer directly override Databao.
- `ratio_scale_compaction`: a narrow ratio/percentage candidate when Databao already returned
  explicit numerator, denominator, and a result column whose scale can be verified. It does not infer
  numerator/denominator from context and does not override Databao without ranking evidence.

`succeeded` means the internal pipeline completed; `scorable` means a `prediction.csv` exists for
the evaluator. These are intentionally separate. Timeout and exception artifacts now inspect the
task output directory and preserve any provisional prediction that was already written.
The process wrapper returns a queued artifact as soon as the child reports one; lingering model
client/background threads may then be terminated without rewriting the task log as a timeout.

A Databao DataFrame is no longer considered fully correct by itself, but it is good enough to write
a raw provisional prediction immediately. Ordinary extra rows/columns are handled by cheap repair,
ranking, and the final shape guard. The planner is not started for ordinary suspicious output, and
it is not used as missing-prediction recovery in the default path.

Answer contracts should avoid over-scalarizing. Plain `what`, `which`, or `who` questions are not
single-row requests by themselves. Row limiting is reserved for scalar metrics or explicit top-one
language such as single/only/top-one/first-row/last-row/superlative wording. List/table contracts
preserve all candidate rows and only prune columns when the contract has explicit schema-backed
column evidence.

Display/entity resolution is generic and schema-driven. Entity answers prefer display/name/title/
label/url/link/text/value columns; id-like and `link_to_*` columns are treated as join helpers unless
the question explicitly asks for an id, identifier, code, or reference. URL/link questions prefer
URL-like columns over ids or numeric helper columns.

Identifier suppression now covers `id`, `*_id`, `link_to_*`, camelCase `*Id`, `identifier`, `code`,
and `reference`. If a final candidate exposes one of these columns and the question did not ask for
an identifier, postprocess tries to resolve it through context tables to a display column. If display
columns are already present, the final guard removes ID helper columns before writing CSV.

The ranker favors candidates with evidence: contract validity, schema-matched columns, retrieved
columns, display/id resolution, explicit aggregation components, and transformations that actually
performed the operation implied by the question. It no longer treats smaller row count as inherently
better when the answer contract allows multiple rows; list-like questions receive a bonus for
multi-row candidates. It penalizes debug metadata, answer-shape mismatch, empty required answers,
invalid frames, raw-table explosions, and weak verifier candidates without clear evidence. Count
questions penalize non-count-like output columns. Ratio/percentage verifier candidates require
explicit numerator and denominator components and are not emitted for zero/undefined percentages.
High-risk verifier candidates are deliberately discounted and no longer get a generic aggregation
bonus.

Questions that ask for an entity plus a metric, such as a display/name answer with a total, cost,
average, count, score, or value, use a multi-attribute contract. The final guard preserves display
and metric columns for these answers instead of shrinking them to a single scalar column.

## Context And Documents

The document ingestion layer is split:

- `document_records_for_reasoning()`: keeps full extraction metadata for diagnostics, verifier
  support, and log review.
- `document_records_for_agent()`: exposes only answer-safe columns to Databao.

Metadata/debug columns such as source document, evidence span, confidence, extraction strategy, and
paragraph index are never registered as ordinary Databao answer columns. They may appear in logs but
are removed by the final answer guard before CSV write.

`query_context_retriever()` is generic. It scores table names, column aliases, identifiers, display
columns, numeric metric columns, document record text, and join candidates. It does not select
context by file stem or domain name, and it does not hide unretrieved tables from Databao.

## Runner Baseline

The active runner is now a smaller Databao-first baseline. It loads context, asks Databao, writes a
raw provisional `prediction.csv` as soon as a DataFrame exists, then runs postprocess, candidate
ranking, final guard, diagnostics, and local evaluation metadata. LLM structured planner fallback,
LLM finalizer, semantic repair, schema hint, context budget, extra candidate generators, and failed
ablation switches are no longer connected to the main runner.

Timeout controls:

- `run-databao-demo --task-timeout-seconds N` sets a per-task process timeout.
- `DATABAO_DATABAO_TIMEOUT_SECONDS` controls the Databao/LangChain model timeout, default `100`.
  The effective Databao timeout is also capped dynamically at `remaining_task_seconds - 20`.

Timeout diagnostics preserve any already-written provisional CSV. Timeout logs include the latest
task progress payload under `timeout_budget.active_progress`, so missing-prediction tasks can be
triaged by active stage before changing model or context budgets.
Progress checkpoints now include `databao_frame_received`, `raw_provisional_csv_write`, and
`artifact_ready`, which separates model latency, CSV write failures, and child-process teardown.
Progress files also retain prior checkpoints, so timeout artifacts can still show `context_loaded`
payload data even when the latest active stage is Databao model execution.

## Removed Experiments

The failed role-evidence and generator experiments showed that generic-looking signals can still
become benchmark-shape overfitting if they drive both ranking and final-guard pruning. The mainline
runner therefore does not keep default-off experimental branches.

Removed from the active runner:

- LLM structured planner fallback and CLI `--planner-mode`.
- LLM finalizer and CLI `--finalizer-mode`.
- Semantic repair candidate flow.
- Schema/context prompt hint mode.
- Context budget compact mode.
- `DABENCH_CANDIDATE_GENERATORS`, including the LlamaIndex PandasQueryEngine adapter and temporary
  SQLite SQL generator.
- `DATABAO_LIGHTWEIGHT_RETRY`.
- `DATABAO_SAFE_POSTPROCESS_BONUS`.
- The old `DATABAO_RATIO_COMPACT_EXPERIMENT` switch. A narrower always-on
  `ratio_scale_compaction` candidate now exists only for already-computed component tables with
  complete numerator/denominator/result evidence.

Relevant baseline and negative-run notes:

- `20260509T132426Z` easy baseline: avg `0.711`, exact `10/15`, prediction `15/15`.
- `20260509T132904Z` medium baseline: avg `0.514`, exact `11/23`, prediction `23/23`.
- `20260509T134042Z` hard baseline: avg `0.273`, exact `3/11`, prediction `9/11`.
- `20260509T140121Z` hard lightweight-retry ablation: avg `0.152`, exact `1/11`,
  prediction `9/11`; removed instead of promoting.
- `20260510T150051Z`, `20260510T150514Z`, `20260510T151642Z` semantic-repair-off control:
  easy avg `0.700`, exact `10/15`, prediction `15/15`; medium avg `0.641`, exact `14/23`,
  prediction `22/23`; hard avg `0.212`, exact `2/11`, prediction `7/11`.
- Semantic repair v2 targeted safe check did not pass. Strong evidence helped `task_145` and
  `task_214`, but still selected wrong strong candidates for `task_25`, `task_243`, and `task_344`.
  Full safe sweep was skipped; semantic repair is now disconnected from the active runner and should
  not be promoted without a substantially narrower evidence model.
- Context budget compact targeted checks fixed some missing-prediction cases
  (`task_352`, `task_355`, `task_396`, and a later `task_420` run produced predictions), but full
  hard run `20260510T160349Z` regressed to avg `0.091`, exact `0/11`, prediction `8/11`; the switch
  has been removed from the active runner.
- Medium semantic-shape targeted checks: generic comment/text answer targeting fixed `task_259`
  (`20260510T162341Z` and `20260510T162938Z` exact). ID/display resolver changes did not reliably
  fix `task_257` because Databao returned semantically wrong source rows in later runs. Treat
  comment/text targeting as useful; do not infer broader semantic repair from it.
- Schema/context prompt hint targeted checks are mixed and remain off by default. The first enhanced
  pass improved explicit output-shape cases such as `task_250` in one run but failed most broader
  medium targets. A later `task_259` check showed that feeding `AnswerContract.expected_columns`
  into the Databao prompt can mislead the model toward filter/sort columns; enhanced hints now show
  only soft answer shape plus schema/question matched potential answer columns. `task_259`
  recovered to exact in run `20260511T003620Z`, while `task_250` later hit Databao
  `GraphRecursionError` with no prediction in `20260511T003720Z`. The broad switch has been removed;
  future prompt/context work should restart as a smaller targeted experiment.

- Post-cleanup hard diagnostics:
  - `20260511T052829Z` targeted `task_349` succeeded and wrote a prediction, but evaluator score was
    still `0.0`; this is a semantic-source issue, not a missing-file issue.
  - `20260511T053424Z` targeted `task_420` with a 100s task timeout still produced no prediction.
    The timeout log preserved `context_loaded`: two registered sources, a 249-row document record
    table, a 56,822-row SQLite table with 74 columns, about 20k document chars, and the latest stage
    was `databao_ask` after roughly 73s of context loading. This points at hard stability/context
    payload cost rather than final guard or postprocess swallowing an existing prediction.
  - Follow-up profiling found the main local bottleneck was not SQLite loading itself. The expensive
    step was `infer_answer_contract()`, which repeatedly called metric-column detection over full
    object columns. On `task_420`, contract inference dropped from about `70.3s` to `2.9s` after
    changing metric-column profiling to use numeric dtype checks plus a bounded non-null sample for
    object columns.
  - `20260511T054345Z` targeted `task_420` after the profiling fix succeeded and wrote a prediction.
    `context_load` dropped to `4.599s`; `databao_ask` took `117.847s`; total runtime was `128.895s`.
    Evaluator score remained `0.0`, so the fix addresses hard missing-prediction stability, not
    answer semantics.
  - Full sweep after the profiling fix:
    - Easy `20260511T054812Z`: prediction `15/15`, exact `9/15`, avg `0.667`.
    - Medium `20260511T055505Z`: prediction `23/23`, exact `14/23`, avg `0.641`.
    - Hard `20260511T060405Z`: prediction `9/11`, exact `0/11`, avg `0.045`.
    - `task_420` wrote a prediction in the hard sweep with `context_load=4.555s`; the remaining
      missing predictions were `task_352` (`GraphRecursionError`, 50 LLM calls) and `task_355`
      (Databao returned no dataframe after 50 calls / `127.168s` ask time). This suggests the next
      hard-stability target is Databao recursion/no-output behavior on small doc+table contexts, not
      local schema profiling.
  - Targeted answer-shape cleanup after comparing `20260507` vs `20260511` runs:
    - Count questions now prefer existing count-like answer columns over entity/filter evidence
      columns. `task_24` targeted run `20260511T065037Z` returned `attendance_count=17` and scored
      exact.
    - Column-only compaction now protects schema-backed expected columns in multi-attribute answers.
      `task_415` targeted run `20260511T064937Z` preserved `url,constructorRef` and scored exact.
    - `task_350` targeted run `20260511T065105Z` returned `count=7` and scored exact, but this was
      Databao returning a better candidate rather than a new deterministic counting fallback.
  - Full sweep after count/multi-attribute cleanup:
    - Easy `20260511T065626Z`: prediction `15/15`, exact `11/15`, avg `0.767`.
    - Medium `20260511T070308Z`: prediction `22/23`, exact `15/23`, avg `0.674`; the only missing
      task was `task_180`, caused by Databao `GraphRecursionError` after 50 LLM calls.
    - Hard `20260511T071557Z`: prediction `8/11`, exact `3/11`, avg `0.273`; missing tasks were
      `task_349`, `task_352`, and `task_420`, all Databao `GraphRecursionError` after 50 LLM calls.
      The average recovered to the older high hard baseline, but prediction-written stability is
      still worse than desired.

## 2026-05-14 High-Score Restore Decision

The branch was restored to the previously saved high-scoring `databao_demo.py` from
`C:\Users\tianj\Downloads\databao_demo.py` instead of continuing the conservative cleanup branch.
The restored file differs from the smaller baseline in one important way: Databao uses
`DATABAO_EXECUTOR_TYPE=lighthouse_salvage` by default. This executor keeps the latest successful
Databao/Lighthouse DataFrame when the agent fails to submit a final result, records the generated
code and submit/salvage flags on `frame.attrs`, and lets the normal Databao-first ranking path use
that salvaged table.

Temporary validation with the downloaded file before committing:

| Difficulty | Run id | Prediction | Exact | Avg |
| --- | --- | --- | --- | --- |
| easy | `20260513T153200Z` | `15/15` | `10/15` | `0.713` |
| medium | `20260513T153634Z` | `23/23` | `13/23` | `0.570` |
| hard | `20260513T154923Z` | `11/11` | `6/11` | `0.545` |

This result makes the tradeoff clear. The conservative branch kept the runner cleaner, but it lost
the most valuable hard-stability mechanism. The restored high-score version is less minimal because
it contains salvaged-result compactors and repair candidates, but those are tied to Databao's own
latest execution state rather than broad pre-Databao synthetic guessing. For the current goal, the
salvage path is worth keeping as the mainline.

The immediately preceding conservative branch is recorded as a negative experiment:

- Answer-focus column targeting, uppercase camel-case ID suppression, `Diagnosis`/`disease` aliasing,
  and duplicate answer-row removal were small, generic-looking fixes.
- Targeted `task_180` showed the ID-helper cleanup worked, but Databao still selected one extra
  semantic source row; the fix did not address the real failure source.
- Full checks on that branch produced easy `20260513T080952Z` avg `0.676`, exact `9/15`,
  prediction `15/15`; medium `20260513T081454Z` avg `0.576`, exact `13/23`, prediction `22/23`.
- Those changes should not be treated as active mainline behavior. If revisited, they should be
  reintroduced one at a time behind targeted ablation, not mixed into the restored salvage baseline.

## 2026-05-14 Easy Safe Cleanup

After restoring the salvage runner, the first easy-focused cleanup stayed inside deterministic
answer shaping rather than Databao prompt/planner changes:

- Entity display lookup now prefers stable display/name/title/label columns before status/category
  columns. This lets context superlative verification return event names instead of status values.
- Answer-column pruning now uses the leading answer phrase as a signal, so questions such as
  "provide the eye colour", "what is his number", and "what is the finish time" prefer the requested
  attribute over filter/sort/helper columns.
- Superlative metric columns are treated as sort evidence unless the question explicitly asks for
  the metric value. Context superlative candidates with strong metric evidence and tied rows can
  beat a single-row Databao candidate.

Targeted checks:

- `task_25` run `20260513T162452Z`: exact, repaired lowest-cost ties to three event names.
- `task_74` run `20260513T162519Z`: exact, selected `eye_colour`.
- `task_80` run `20260513T162857Z`: exact, selected only `number`.
- `task_89` run `20260513T162804Z`: column shape fixed to `time`, but Databao selected the wrong
  source row; this is a semantic filtering issue and was not forced by the guard.

Full easy check:

- `20260513T163105Z`: prediction `15/15`, exact `11/15`, avg `0.733`.
- Remaining easy misses should be treated as Databao semantic/source-row errors or run variance, not
  as safe final-guard targets.

## 2026-05-14 Medium Adapter Repair

The next adapter-only pass focused on cases where Databao already produced a filtered/detail table
or a nearly final identifier answer. It did not modify Databao internals, restore planner/finalizer,
or add task/domain/file-stem rules.

Changes:

- Databao generated code is parsed into a lightweight `sql_observation` diagnostic with operation
  flags for select, filter, join, group-by, aggregate, ratio, order-by, and limit.
- A narrow `databao_observed_detail_aggregate_compaction` verifier can turn Databao's own filtered
  detail table into a count or average candidate. It only fires when the SQL/code evidence shows
  filter/group/join work has already happened, and it is ranked like any other candidate.
- Count compaction now prefers distinct ID/display values when available, avoiding duplicate detail
  rows in count questions.
- Explicit identifier answers prefer primary/short identifier columns over helper relation IDs such
  as accepted/owner/editor/parent variants unless those helper roles are explicitly requested.

Targeted results:

- `task_145` run `20260513T165205Z`: exact. Databao returned filtered meeting detail; the adapter
  counted the rows.
- `task_196` run `20260513T165241Z`: exact. Databao returned per-atom bond counts; the adapter
  averaged the metric.
- `task_250` run `20260513T165425Z`: exact. The adapter selected `Id` rather than a helper answer id.
- `task_214` run `20260513T165305Z`: still wrong because Databao's filtered detail table contained
  the wrong population. This remains a source/filter semantics issue, not a safe adapter repair.

Full sweep after this pass:

- Easy `20260513T165623Z`: prediction `14/15`, exact `11/15`, avg `0.767`; `task_80` timed out with
  no prediction in this run, while previous targeted `task_80` run `20260513T162857Z` was exact.
- Medium `20260513T170308Z`: prediction `23/23`, exact `15/23`, avg `0.685`.

This is the current best evidence that the adapter can safely improve medium when it completes an
operation that Databao has already partially executed. It should not be broadened into arbitrary
context recomputation.

## 2026-05-14 Hard Salvage Report

Using hard run `20260513T154923Z` after restoring `lighthouse_salvage`:

- Overall: prediction `11/11`, exact `6/11`, avg `0.545`.
- The remaining wrong tasks split into two groups.

Adapter-safe candidates:

- `task_344`: Databao returned a filtered detail table with `ID, FG, SEX, WBC` and SQL already
  applied sex/WBC/FG filters. The question asks "how many". This is the same pattern as the medium
  observed-detail aggregate repair: count distinct IDs from a filtered detail table.
- `task_408`: Databao returned ordered race-result detail for the requested race with `position`,
  `time`, `milliseconds`, and `fastestLapTime`. The question asks a percentage speed/time delta
  between champion and last finisher. This is a possible narrow rank-delta/time-percentage repair,
  but it needs careful evidence checks because missing `milliseconds` values and `+delta` time
  notation can make the arithmetic brittle.

Not adapter-safe without deeper Databao/context changes:

- `task_355`: Databao salvaged a document-record lookup by ID and returned prose metadata such as
  `asset registered under`, not the expense/member/cost row. The issue is source selection and
  document materialization quality.
- `task_379`: Databao selected six molecule IDs and returned 4th atom elements, but gold has seven
  target elements. The candidate is shape-correct but source population is wrong.
- `task_396`: Databao queried document records by a list of names and returned `height` blanks,
  while the gold asks for a publisher percentage over height-filtered heroes. This is source
  population and document/table relation reasoning, not safe final-guard work.

Next hard experiments should therefore be:

1. Extend `databao_observed_detail_aggregate_compaction` to count distinct IDs for filtered detail
   tables when the SQL observation shows filtering/join evidence. This should target `task_344`
   without touching Databao internals.
2. Prototype a shadow-only `rank_delta_percentage` observation for ordered detail tables with
   complete first/last timing evidence. Only promote it if arithmetic evidence is explicit.
3. For `task_355/379/396`, inspect Databao internals or context registration. These should not be
   fixed with generic final-guard pruning because the returned rows are semantically wrong.

Follow-up targeted checks:

- `task_344` run `20260513T173041Z`: Databao returned a scalar `count=0`, not a filtered detail
  table. Its SQL inferred normal ranges via mean/stddev. The adapter cannot safely change this to
  the gold count without external threshold evidence, so this task moves from "adapter-safe" to
  "Databao/context semantics" for runs with scalar output. The distinct-ID count compactor still
  covers future runs where Databao returns filtered detail rows.
- `task_408` run `20260513T173249Z`: Databao queried `position <= 5 ORDER BY raceId, position
  LIMIT 50`, without constraining the requested race/year or producing first/last timing evidence.
  A rank-delta percentage repair would be unsafe on this candidate. Keep it as a Databao/source
  selection issue until the executor exposes a more relevant intermediate table.

## 2026-05-14 Adapter-Safe Easy/Medium Pass

This pass kept Databao-first behavior and did not add planner/finalizer/semantic-repair routes.
Only low-risk adapter fixes were applied:

- camelCase ID resolution now rejects conflicting target hints, so `driverId` does not resolve
  through `driverStandingsId` or unrelated same-domain helper tables;
- entity answers prefer strong display columns (`name`, `title`, `label`, `display_name`, URL/text
  family) before weak display-like fields such as ordinal/status text;
- explicit multi-attribute questions keep multiple requested columns before single-column strong
  matches can collapse the answer;
- exact duplicate answer rows are removed after postprocess;
- already-aggregated one-row multi-metric Databao SQL outputs are not compacted into a single
  average column.

Targeted checks:

- `task_86` run `20260514T021249Z`: exact, output race `name` column with all 16 rows.
- `task_249` run `20260514T014927Z`: exact, preserved both `avg_upvotes` and `avg_age`.
- `task_199` run `20260514T014834Z`: still wrong in this run because Databao selected the wrong
  school population; duplicate-row cleanup is not enough to repair source semantics.

Full sweeps after this pass:

- Easy `20260514T021353Z`: prediction `15/15`, exact `12/15`, avg `0.800`.
  - `task_86` had the right `name` column but Databao included one extra source row in this full
    run, so the remaining error is source/filter semantics rather than final-column selection.
- Medium `20260514T015641Z`: prediction `23/23`, exact `14/23`, avg `0.641`.
  - This run was lower than the prior `20260513T170308Z` medium run mostly due to Databao semantic
    variance (`task_200`, `task_243`) while `task_249` improved from `0.5` to exact.
- Hard `20260514T022038Z`: prediction `10/11`, exact `4/11`, avg `0.364`.

Current conclusion: these adapter fixes are worth keeping because they are schema/column-name
generic, improve easy/hard stability, and do not introduce public-task domains. Remaining easy and
medium failures are mostly Databao source/filter/operation mistakes, not safe final-guard work.

## Diagnostics

Each task log records:

- `heuristic_level`
- retrieved context summary
- task complexity profile
- context payload profile and Databao failure type
- candidate list and candidate scores
- selected candidate source
- `prediction_written`, `scorable`, and missing-prediction status
- answer contract report before/after final guard
- route policy
- postprocess and final answer guard transformations
- timeout budget state
- document extraction usage
- safe LLM call metadata

Raw prompts/responses are not written by default. Set `DATABAO_DEBUG_LOG_RAW=1` to write redacted
raw files under `logs/raw/task_<id>/`.

## Useful Commands

No-API verification:

```powershell
uv run python -m unittest discover tests
uv run --extra dev ruff check src tests
uv run python -m compileall src
```

Local smoke run:

```powershell
uv run dabench run-databao-demo --config configs/react_baseline.local.yaml --limit 1
```

Local public-run evaluation after a run:

```powershell
uv run dabench evaluate-public-run artifacts/databao-demo/<run_id>
```

The evaluator now prints per-task diagnostics including whether a prediction was written, artifact
success, matched/gold/predicted/extra columns, row counts, exact-no-extra status, and an estimated
per-task score. It also surfaces selected candidate source, final-guard removed columns, and
postprocess transformation names so shape/semantic failures can be traced without reading every task
log. Aggregate coverage is kept as debugging context rather than the only score signal.

## Latest Verification

- `uv run python -m unittest discover tests`: 93 tests OK.
- `uv run --extra dev ruff check src tests`: passed.
- `uv run python -m compileall src`: passed.
- `src/` banned public-domain memory term scan: no violations.

Keep this file high-level. Put function-by-function details in `README.function-ledger.md`.
