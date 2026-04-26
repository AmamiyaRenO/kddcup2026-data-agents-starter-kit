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
6. Run Databao first. As soon as Databao returns any DataFrame, apply cheap postprocess plus the
   final shape guard and write a provisional `prediction.csv`.
7. Generate additional candidates from cheap repair/verifiers and, only under feature-based rescue
   gates, the structured planner.
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
- `structured_planner_rescue`: a safe JSON-plan candidate only when feature-based complexity and
  uncertainty gates are high enough and task budget remains.
- verifier candidates: generic superlative, context superlative, and aggregate/ratio checks. These
  are ranked like any other candidate and no longer directly override Databao.
- `*_finalizer`: optional LLM finalizer candidates only when `DATABAO_FINALIZER_MODE=always`.

`succeeded` means the internal pipeline completed; `scorable` means a `prediction.csv` exists for
the evaluator. These are intentionally separate. Timeout and exception artifacts now inspect the
task output directory and preserve any provisional prediction that was already written.

A Databao DataFrame is no longer considered fully correct by itself, but it is good enough to write
a provisional prediction. Ordinary extra rows/columns are handled by cheap repair, ranking, and the
final shape guard. The planner is not started for ordinary suspicious output.

The ranker favors contract-valid, compact, metadata-free candidates that use question-matched or
retrieved columns. It penalizes extra rows/columns, debug metadata, answer-shape mismatch, empty
required answers, invalid frames, and weak verifier candidates without clear evidence.

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

The structured planner remains a bounded feature-based rescue:

- Allowed operations: `source`, `filter`, `join`, `select`, `distinct`, `sort`, `top_k`, `limit`,
  `date_filter`, `aggregate`, `count_distinct`, `derive`, `ratio`, and `percentage`.
- Plans are JSON only and are locally validated before execution.
- No arbitrary Python, SQL strings, task-id branches, filename-specific parsers, or domain-specific
  operations are allowed.
- Planner attempts are capped at one.
- Planner rescue is considered only when remaining budget is above 90 seconds, current candidate
  uncertainty is high, observable task complexity is high, and the planner payload stays under the
  configured size limit.
- If budget is low or the feature gates do not pass, the current best candidate is passed to the
  final answer guard.

Timeout controls:

- `run-databao-demo --task-timeout-seconds N` sets a per-task process timeout.
- `DATABAO_DATABAO_TIMEOUT_SECONDS` controls the Databao/LangChain model timeout, default `180`.
- `DATABAO_AUX_TIMEOUT_SECONDS` controls direct planner/finalizer OpenAI client timeout, default
  `30`.
- `DATABAO_STRUCTURED_PLANNER_MODE` supports `fallback` (default), legacy `first` (kept
  Databao-first by the adapter), and `off`.
- `DATABAO_FINALIZER_MODE` supports `off` (default) and `always`.

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
per-task score. Aggregate coverage is kept as debugging context rather than the only score signal.

## Latest Verification

- `uv run python -m unittest discover tests`: 48 tests OK.
- `uv run --extra dev ruff check src tests`: passed.
- `uv run python -m compileall src`: passed.
- `src/` banned public-domain memory term scan: no violations.

Keep this file high-level. Put function-by-function details in `README.function-ledger.md`.
