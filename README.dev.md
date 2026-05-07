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
   paths. Retrieval is a soft priority signal, not access control: Databao still receives registered
   structured sources, while descriptions and planner payloads are focused by retrieval.
5. Infer an `AnswerContract` from the question and schema.
6. Run Databao first. As soon as Databao returns any DataFrame, write a raw provisional
   `prediction.csv` before postprocess, ranking, verifier, finalizer, or final guard work can fail.
7. Generate additional candidates from cheap repair/verifiers. Structured planner rescue is default
   off and should remain off for stability ablations unless explicitly enabled.
8. Validate and rank candidates with a generic evaluator.
9. Apply `final_answer_shape_guard()` again before the final CSV write.
10. Write diagnostics, candidate scores, selected route, scorable status, timings, and safe ablation
   metadata.

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
- `cheap_count_fallback`: a narrow missing-prediction fallback used only after Databao failure when
  the question explicitly asks for a count and a single high-confidence table can be selected.
- `structured_planner_rescue`: optional safe JSON-plan candidate only when planner mode is explicitly
  enabled and feature-based complexity/uncertainty gates are high enough.
- verifier candidates: generic superlative, context superlative, and aggregate/ratio checks. These
  are ranked like any other candidate and no longer directly override Databao.
- `*_finalizer`: optional LLM finalizer candidates only when `DATABAO_FINALIZER_MODE=always`.

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
preserve all candidate rows and only prune columns according to the contract.

Display/entity resolution is generic and schema-driven. Entity answers prefer display/name/title/
label/url/link/text/value columns; id-like and `link_to_*` columns are treated as join helpers unless
the question explicitly asks for an id, identifier, code, or reference. URL/link questions prefer
URL-like columns over ids or numeric helper columns.

Identifier suppression now covers `id`, `*_id`, `link_to_*`, camelCase `*Id`, `identifier`, `code`,
and `reference`. If a final candidate exposes one of these columns and the question did not ask for
an identifier, postprocess tries to resolve it through context tables to a display column. If display
columns are already present, the final guard removes ID helper columns before writing CSV.

The ranker favors contract-valid, metadata-free candidates that use question-matched or retrieved
columns. It no longer treats smaller row count as inherently better when the answer contract allows
multiple rows; list-like questions receive a bonus for multi-row candidates. It penalizes extra
columns, debug metadata, answer-shape mismatch, empty required answers, invalid frames, and weak
verifier candidates without clear evidence. Count questions penalize non-count-like output columns.
Ratio/percentage verifier candidates require explicit numerator and denominator components and are
not emitted for zero/undefined percentages. High-risk verifier candidates are deliberately
discounted and no longer get a generic aggregation bonus.

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

## Planner And Deadlines

The structured planner is now default off. It remains available only as an explicit bounded
feature-based rescue for ablation:

- Allowed operations: `source`, `filter`, `join`, `select`, `distinct`, `sort`, `top_k`, `limit`,
  `date_filter`, `aggregate`, `count_distinct`, `derive`, `ratio`, and `percentage`.
- Plans are JSON only and are locally validated before execution.
- No arbitrary Python, SQL strings, task-id branches, filename-specific parsers, or domain-specific
  operations are allowed.
- Planner attempts are capped at one.
- Planner rescue is considered only when `DATABAO_STRUCTURED_PLANNER_MODE` is explicitly set away
  from `off`, remaining budget is above 90 seconds, current candidate uncertainty is high,
  observable task complexity is high, and the planner payload stays under the configured size limit.
- If budget is low or the feature gates do not pass, the current best candidate is passed to the
  final answer guard.

Timeout controls:

- `run-databao-demo --task-timeout-seconds N` sets a per-task process timeout.
- `DATABAO_DATABAO_TIMEOUT_SECONDS` controls the Databao/LangChain model timeout, default `100`.
  The effective Databao timeout is also capped dynamically at `remaining_task_seconds - 20`.
- `DATABAO_AUX_TIMEOUT_SECONDS` controls direct planner/finalizer OpenAI client timeout, default
  `30`.
- `DATABAO_STRUCTURED_PLANNER_MODE` supports `off` (default), `fallback`, and legacy `first`
  (kept Databao-first by the adapter).
- `DATABAO_FINALIZER_MODE` supports `off` (default) and `always`.

Timeout diagnostics preserve any already-written provisional CSV. Timeout logs include the latest
task progress payload under `timeout_budget.active_progress`, so missing-prediction tasks can be
triaged by active stage before changing model or context budgets.
Progress checkpoints now include `databao_frame_received`, `raw_provisional_csv_write`, and
`artifact_ready`, which separates model latency, CSV write failures, and child-process teardown.

## Diagnostics

Each task log records:

- `heuristic_level`
- retrieved context summary
- task complexity profile and planner rescue gate state
- candidate list and candidate scores
- selected candidate source
- `prediction_written`, `scorable`, and missing-prediction status
- answer contract report before/after final guard
- route policy
- postprocess and final answer guard transformations
- planner skip reason when applicable
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

- `uv run python -m unittest discover tests`: 59 tests OK.
- `uv run --extra dev ruff check src tests`: passed.
- `uv run python -m compileall src`: passed.
- `src/` banned public-domain memory term scan: no violations.

Keep this file high-level. Put function-by-function details in `README.function-ledger.md`.
