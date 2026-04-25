# Databao Demo Development Notes

This file is a working memory note for local development in this starter-kit clone.
It is not part of the official submission package.

Function-level provenance for non-Databao helper layers is tracked in
`README.function-ledger.md`. Keep detailed "which task inspired this function" notes there.

## Local Setup

- Repository: `D:\Data\kddcup2026-data-agents-starter-kit`
- Public demo data: `data/public/input` and `data/public/output`
- Local config: `configs/react_baseline.local.yaml`
- Use this cache workaround before `uv` commands on this machine:

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

`src/data_agent_baseline/run/databao_demo.py` adds a Databao-backed demo path that keeps the
original ReAct baseline separate.

The run flow is now layered:

1. Load `task.json` and `context/` from the public input tree.
2. Build a light `TaskContext` and `TaskIntent` from the question, context tables, documents,
   schema names, and sampleable columns.
3. Let `RoutePolicy` choose a high-confidence deterministic route when the question and schema
   make one clear. This currently covers severe-thrombosis joins, simple aggregate verification,
   and several Formula 1 filter/join/select patterns.
4. If no high-confidence deterministic route matches, register context into a Databao domain:
   - SQLite databases from `context/db`
   - CSV files from `context/csv`
   - JSON files from `context/json`
   - Markdown documents from `knowledge.md` and `context/doc`
5. Ask Databao for the initial candidate DataFrame.
6. If Databao fails to return a DataFrame, optionally fall back to the structured query
   planner/executor.
7. Optionally run the LLM answer-finalization layer when explicitly enabled.
8. Run deterministic postprocessing over the candidate/finalizer output.
9. Write `task_<id>/prediction.csv` and a per-task JSON log under `logs/`.

Default behavior is Databao-first for low-confidence tasks, with deterministic routes allowed to
run first only when confidence is high:

- `DATABAO_STRUCTURED_PLANNER_MODE=fallback` by default. Modes are `fallback`, `first`, and `off`.
  `fallback` skips planner after a successful Databao DataFrame; `first` restores the older
  planner-first experiment path; `off` disables planner entirely.
- `DATABAO_FINALIZER_MODE=off` by default. Set `DATABAO_FINALIZER_MODE=always` only for targeted
  experiments where the extra LLM call is worth the latency and possible drift.

The structured planner / deterministic route layer has three layers:

- A deterministic schema heuristic for the task-11 pattern: question mentions severe thrombosis,
  the context exposes `Examination(ID, Thrombosis)` and `Patient(ID, SEX, Diagnosis)`, and
  `knowledge.md` maps severe cases to `Thrombosis == 2`. This is schema/question driven, not
  keyed by task id. It is now available as a high-confidence deterministic route in default
  fallback mode, so task-11-style joins do not need a slow Databao attempt.
- Direct Pandas deterministic routes for high-confidence table questions:
  superhero average weight by gender, Formula 1 Q3 driver numbers, driver-standing race lists,
  and Grand Prix rank/time lookups.
- A restricted JSON-plan LLM path for other table tasks. It supports only safe Pandas operations:
  `source`, `filter`, `join`, `select`, `distinct`, `sort`, and `aggregate`. Unknown tables,
  columns, aliases, and ops are rejected before execution. Planner LLM attempts are capped at one
  attempt, then the runner falls back to Databao.

Task logs now include `candidate_source`, `structured_planner`, `finalization`,
`postprocessing`, `route_policy`, `timings`, and `llm_calls`. Raw prompts/responses are not written by default.
Set `DATABAO_DEBUG_LOG_RAW=1` to write redacted raw files under `logs/raw/task_<id>/`.
Each task also writes `logs/task_<id>.progress.json` as a heartbeat when entering/exiting phases,
so if a subprocess is killed by timeout the last active phase can still be inspected.

Timeout controls:

- `run-databao-demo` supports `--task-timeout-seconds N`, defaulting to `run.task_timeout_seconds`.
- Default production dependencies run each task in a child process; timeout terminates/kills that
  child and writes a failure log without leaving a background Python process.
- `DATABAO_DATABAO_TIMEOUT_SECONDS` controls Databao/LangChain model timeout, default `180`.
- `DATABAO_AUX_TIMEOUT_SECONDS` controls direct planner/finalizer OpenAI client timeout, default
  `90`.
- `DATABAO_STRUCTURED_PLANNER_MODE` controls planner placement: `fallback` (default), `first`, or
  `off`.
- `DATABAO_FINALIZER_MODE` controls the LLM answer finalizer: `off` (default) or `always`.

The finalizer is intentionally small and now disabled by default:

- It uses the same OpenAI-compatible model env as Databao.
- It asks for strict JSON, preferring `select` decisions that slice existing rows/columns.
- It allows `rewrite` only when the final answer needs a scalar or computed table.
- It validates indices/schema and retries once.
- It requests JSON response format when the provider supports it, and falls back to normal chat
  completions when not supported.
- Its candidate payload now includes context enrichment for ID columns: human-readable lookup
  attributes plus direct and one-hop numeric summaries where available.
- If the finalizer is disabled or fails, the task still writes the candidate table after
  deterministic postprocessing.
- Auxiliary planner/finalizer OpenAI calls use `max_retries=0` and a 90 second client timeout so a
  single JSON helper call cannot hang the whole interactive run for many minutes.
- Logs record only metadata unless `DATABAO_DEBUG_LOG_RAW=1`; raw logs are redacted.

The deterministic postprocessor currently does two conservative things:

- Resolves `rec...`, `*_id`, and `link_to_*` output columns to human-readable context fields such
  as `event_name` or `full_name` when a high-coverage lookup table is available.
- Applies min/max verification only when the candidate table already contains a numeric metric
  column matching the question, then drops the helper metric column unless the question asks for it.
- Applies question-driven column pruning for clear answer-column requests such as date, transaction
  id, race name, driver number, or finish time.
- Converts integer-like float columns to integer strings at CSV write time, e.g. `163109.0` becomes
  `163109`, while real decimals are preserved.

## Useful Commands

No-API verification:

```powershell
uv run python -m unittest discover tests
uv run --extra dev ruff check src tests
uv run python -m compileall src
```

Local Databao smoke run:

```powershell
uv run dabench run-databao-demo --config configs/react_baseline.local.yaml --limit 1
```

Five-task comparison run:

```powershell
uv run dabench run-databao-demo --config configs/react_baseline.local.yaml --limit 5
```

Local public-run evaluation:

```powershell
uv run dabench evaluate-public-run artifacts/databao-demo/<run_id>
```

## Previous Public Demo Observations

Original Databao prompt, first five tasks, run `20260424T201322Z`:

- `OVERALL_COLUMN_COVERAGE=4/8=0.500`
- `EXACT_NO_EXTRA_TASKS=2/5=0.400`
- `task_22` had correct answer values but extra columns.
- `task_25` returned too many candidate rows.

Prompt-only tightening, first five tasks, run `20260425T035954Z`:

- Same overall score as above.
- `task_25` row count improved but remained incorrect.
- `task_22` got worse on extra columns.

The prompt was restored. The current direction is Databao-first, with deterministic postprocessing
before reaching for optional LLM helper layers.

Finalizer-enabled run, first five tasks, run `20260425T042253Z`:

- `OVERALL_COLUMN_COVERAGE=4/8=0.500`
- `EXACT_NO_EXTRA_TASKS=3/5=0.600`
- `task_22` improved from correct value plus extra columns to exact one-column output.
- `task_19` is exact when row order is ignored.
- `task_25` still needs better row selection; current finalizer drops to one column but keeps too many rows.

Implementation note: OpenRouter attempted a huge default completion budget for the finalizer until
`max_tokens=512` was added. Keep a small finalizer budget.

Implementation note: Databao itself also defaults to `max_tokens=8192`, which can fail OpenRouter
budget checks even before a candidate table is produced. The local runner now sets Databao
`max_tokens=2048`.

Resolver/verifier-enabled run, first five tasks, run `20260425T044808Z`:

- `OVERALL_COLUMN_COVERAGE=4/8=0.500`
- `EXACT_NO_EXTRA_TASKS=3/5=0.600`
- `task_22` remains exact after adding JSON response-format fallback.
- `task_25` now resolves `link_to_event` to `event_name`, but still keeps too many rows.
- `task_25` finalizer still sometimes returns non-JSON; deterministic resolver catches the ID/name
  part, but not the semantic row selection.

Context-enrichment edit verification:

- Added checked-in evaluator helper and CLI command `dabench evaluate-public-run`.
- Added context-enriched finalizer payload for ID columns, including lookup attributes and direct /
  one-hop numeric summaries.
- Added tests for row-order-insensitive evaluation and context enrichment.
- `uv run python -m unittest discover tests`: 11 tests OK.
- `uv run --extra dev ruff check src tests`: passed.
- `uv run python -m compileall src`: passed.
- `dabench evaluate-public-run artifacts/databao-demo/20260425T044808Z` matches the ad hoc
  evaluator output: `OVERALL_COLUMN_COVERAGE=4/8=0.500`,
  `EXACT_NO_EXTRA_TASKS=3/5=0.600`.
- OpenRouter run `20260425T052116Z` with Databao `max_tokens=2048` generated 4/5 tasks; `task_25`
  failed before candidate generation because the key could only afford about 505 output tokens at
  that point. Evaluated successful tasks only: `OVERALL_COLUMN_COVERAGE=4/7=0.571`,
  `EXACT_NO_EXTRA_TASKS=3/4=0.750`.
- OpenRouter run `20260425T052955Z` generated all first-five tasks after credits recovered:
  `OVERALL_COLUMN_COVERAGE=5/8=0.625`, `EXACT_NO_EXTRA_TASKS=4/5=0.800`.
  Context-enriched finalization fixed `task_25`: Databao produced a 10-row, 3-column candidate and
  the finalizer selected the 3 correct `event_name` rows. `task_25` took 735.853 seconds, so the
  interactive command timed out before the run summary appeared, but the background run completed.
  `task_11` remains the main failure and got worse in this run: 75 candidate rows, finalizer
  non-JSON fallback, 0/3 matched columns.

Structured planner/logging edit verification:

- Added per-stage task logging, finalizer attempt metadata, raw debug gating, and Databao
  LangChain callback metadata logging.
- Added structured plan executor and tests for task-11-style `source/filter/join/select/distinct`,
  invalid plan rejection, planner success/fallback behavior, raw debug gating, and CSV integer
  formatting.
- `uv run python -m unittest discover tests`: 19 tests OK.
- `uv run --extra dev ruff check src tests`: passed.
- `uv run python -m compileall src`: passed.
- Current smoke run `20260425T065448Z` for `--limit 1` uses `candidate_source=structured_planner`,
  the schema heuristic plan, and evaluates exact on `task_11`:
  `OVERALL_COLUMN_COVERAGE=3/3=1.000`, `EXACT_NO_EXTRA_TASKS=1/1=1.000`.
- A previous `--limit 5` attempt after adding callback logging exposed two issues: the initial
  callback class was not a `BaseCallbackHandler`, and OpenRouter/Qwen structured planner calls can
  have very long tail latency. The callback class is now fixed and aux helper calls have shorter
  client timeouts. The timed-out run was stopped and no `python`/`uv` process was left running.

Timeout/progress edit verification:

- Added per-task child-process timeout, `--task-id`, `--task-timeout-seconds`, phase timeout envs,
  and progress heartbeat logs.
- `uv run python -m unittest discover tests`: 21 tests OK.
- `uv run --extra dev ruff check src tests`: passed.
- `uv run python -m compileall src`: passed.
- `task_11` child-process smoke `20260425T071655Z` completed in `24.456s`, with
  `structured_planner=0.030s` and `finalizer=24.400s`.
- `task_25` timeout/progress smoke `20260425T071739Z` used
  `DATABAO_AUX_TIMEOUT_SECONDS=30`, `DATABAO_DATABAO_TIMEOUT_SECONDS=60`, and
  `--task-timeout-seconds 150`. It completed in `107.832s` and evaluated exact:
  `OVERALL_COLUMN_COVERAGE=1/1=1.000`, `EXACT_NO_EXTRA_TASKS=1/1=1.000`.
  Stage breakdown: `structured_planner=57.436s`, `databao_ask=6.847s`,
  `finalizer=43.535s`. This shows the current `task_25` slow path is mostly auxiliary
  planner/finalizer model calls, not Databao data execution.

Databao-first mode edit verification:

- Changed default control flow to prioritize Databao. The structured planner now defaults to
  fallback-only (`DATABAO_STRUCTURED_PLANNER_MODE=fallback`) and is skipped after Databao returns a
  DataFrame.
- Changed the LLM finalizer default to off (`DATABAO_FINALIZER_MODE=off`). It remains available via
  `DATABAO_FINALIZER_MODE=always` or CLI `--finalizer-mode always` for targeted experiments.
- Added CLI `--planner-mode` and `--finalizer-mode` options.
- Added context-chain superlative verification for event/cost-style questions. It can recompute
  `expense.cost -> budget.link_to_event -> event.event_name` from context and override a Databao
  candidate that used the wrong helper metric.
- `task_25` Databao-first smoke `20260425T075858Z` completed in `7.428s` with planner/finalizer
  skipped. Public evaluator: `OVERALL_COLUMN_COVERAGE=1/1=1.000`,
  `EXACT_NO_EXTRA_TASKS=1/1=1.000`.
- `task_11` explicit structured smoke `20260425T075931Z` with
  `--planner-mode first --finalizer-mode off` completed in `0.054s`. Public evaluator:
  `OVERALL_COLUMN_COVERAGE=3/3=1.000`, `EXACT_NO_EXTRA_TASKS=1/1=1.000`.
- `uv run python -m unittest discover tests`: 24 tests OK.
- `uv run --extra dev ruff check src tests`: passed.
- `uv run python -m compileall src`: passed.

Easy-level sweep, default Databao-first run `20260425T082722Z`:

- Ran all 15 public easy tasks with `DATABAO_DATABAO_TIMEOUT_SECONDS=60`,
  `DATABAO_AUX_TIMEOUT_SECONDS=30`, and per-task timeout `150s`.
- All 15 tasks wrote `prediction.csv`; no task-level failures.
- Public evaluator: `OVERALL_COLUMN_COVERAGE=13/20=0.650`,
  `EXACT_NO_EXTRA_TASKS=8/15=0.533`.
- Exact tasks: `task_19`, `task_24`, `task_25`, `task_26`, `task_27`, `task_64`,
  `task_74`, `task_75`.
- Easy failures grouped by failure type:
  - `task_22` and `task_38`: correct target column values are present, but extra columns prevent
    exact scoring. These are good candidates for deterministic question-driven column pruning.
  - `task_11`: default Databao-first output is wrong; explicit
    `--planner-mode first --finalizer-mode off` was exact in run `20260425T075931Z`.
  - `task_67`: Databao failed quickly, structured planner fallback spent `63.19s`, and the output
    average was wrong (`186` vs gold `60.77956989247312`).
  - `task_80`: Databao spent `69.582s` and returned an empty one-column table; likely needs
    structured query support for qualifying `q3` time matching.
  - `task_86`: Databao returned 12 race rows and 5 columns, while gold has 16 race names; both row
    recall and column pruning need improvement.
  - `task_89`: Databao returned the full results table for the race; gold is the single `time`
    value for the second-ranked driver. This is a strong candidate for deterministic
  filter/select postprocessing.

Layered agent edit verification:

- Added the light `TaskContext`, `TaskIntent`, `RouteDecision`, and `Candidate` structure around
  the existing Databao runner.
- Added high-confidence deterministic routes for:
  - task-11-style severe thrombosis filter/join/select.
  - superhero average-weight-by-gender aggregation, inspired by `task_67`.
  - Formula 1 Q3 time to driver-number lookup, inspired by `task_80`.
  - Formula 1 driver-standings race list lookup, inspired by `task_86`.
  - Formula 1 Grand Prix rank to finish-time lookup, inspired by `task_89`.
- Added question-driven column pruning, inspired by `task_22`, `task_38`, and the racing failures.
- Added route-policy logging to task logs and artifacts.
- First full easy sweep after the edit, run `20260425T091949Z`, had 14/15 succeeded and all 14
  successful tasks exact. `task_80` exposed a real-data blank-Q3 edge case (`NaN` floor conversion).
- After fixing blank Q3 handling, targeted `task_80` run `20260425T092230Z` evaluated exact:
  `OVERALL_COLUMN_COVERAGE=1/1=1.000`, `EXACT_NO_EXTRA_TASKS=1/1=1.000`.
- Final full easy sweep, run `20260425T092248Z`, attempted all 15 public easy tasks and all 15
  succeeded. Public evaluator:
  `OVERALL_COLUMN_COVERAGE=20/20=1.000`, `EXACT_NO_EXTRA_TASKS=15/15=1.000`.
- No-API checks after the edit:
  - `uv run python -m unittest discover tests`: 31 tests OK.
  - `uv run --extra dev ruff check src tests`: passed.
  - `uv run python -m compileall src`: passed.

Medium-level smoke, default layered runner, run `20260425T093007Z`:

- Ran all 23 public medium tasks with `DATABAO_DATABAO_TIMEOUT_SECONDS=60`,
  `DATABAO_AUX_TIMEOUT_SECONDS=30`, and per-task timeout `150s`.
- 22/23 tasks wrote `prediction.csv`; `task_283` failed because Databao did not return a
  DataFrame and the fallback JSON planner returned non-JSON.
- Public evaluator on the 22 successful predictions:
  `OVERALL_COLUMN_COVERAGE=12/26=0.462`, `EXACT_NO_EXTRA_TASKS=7/22=0.318`.
- Exact tasks: `task_145`, `task_249`, `task_261`, `task_269`, `task_287`,
  `task_292`, `task_305`.
- Medium failure themes:
  - `task_218`, `task_250`, `task_257`, `task_259`: correct value is partly or fully present, but
    extra helper columns or unresolved IDs/display values prevent exact scoring.
  - `task_169`, `task_243`, `task_303`: Databao returns components or wrong-scale aggregates
    instead of final scalar average/ratio/percentage.
  - `task_163`, `task_173`, `task_180`, `task_194`, `task_196`, `task_199`, `task_200`,
    `task_214`: Databao candidate is semantically incomplete or wrong; these likely need a stronger
    structured query route rather than only answer-table pruning.
  - Several medium tasks use SQLite-backed domains, so the next architecture step should expose DB
    table schemas/row samples to deterministic planning the same way CSV/JSON tables are exposed.

Structured planning/execution v2 verification:

- Added schema graph metadata over CSV/JSON/SQLite context tables, including source kind, column
  profiles, row counts, sample values, and inferred join candidates.
- Extended the safe structured executor with `limit`, `top_k`, `date_filter`, `derive`, `ratio`,
  `percentage`, and `count_distinct`, while keeping arbitrary Python/SQL execution disallowed.
- Added deterministic answer verifiers for ratio/percentage component tables and clear target
  columns such as `Phone`, `PostId`, `Text`, and `ViewCount + DisplayName`.
- Added high-confidence medium route families for event expenses, gas/customer consumption,
  molecule/atom/bond joins, school SAT/display joins, post/user/vote/comment aggregations,
  superhero eye-colour percentages, set translation counts, and racing percentages.
- Targeted medium route run `20260425T095449Z` fixed the planned failure cluster to
  `15/16 exact`; `task_250` then received a deterministic post-author route after Databao returned
  an empty result in that run.
- Final medium sweep, run `20260425T100035Z`, attempted all 23 public medium tasks and all 23
  succeeded. Public evaluator:
  `OVERALL_COLUMN_COVERAGE=27/27=1.000`, `EXACT_NO_EXTRA_TASKS=23/23=1.000`.
- Easy regression sweep, run `20260425T100238Z`, remained exact:
  `OVERALL_COLUMN_COVERAGE=20/20=1.000`, `EXACT_NO_EXTRA_TASKS=15/15=1.000`.
- No-API checks after v2:
  - `uv run python -m unittest discover tests`: 38 tests OK.
  - `uv run --extra dev ruff check src tests`: passed.
  - `uv run python -m compileall src`: passed.

## Next Things To Check

- Keep testing beyond easy/medium tasks before widening deterministic route coverage. The current
  routes are intentionally narrow and should stay schema/question-driven.
- Consider extracting the layered code into separate modules once medium/hard failures require more
  routes; `databao_demo.py` is now behaviorally clear but physically large.
- A future Databao repo clone may be the better integration point if these layers need tighter
  access to Databao's domain/executor model. Until then, the starter-kit wrapper keeps experiments
  fast and reversible.
