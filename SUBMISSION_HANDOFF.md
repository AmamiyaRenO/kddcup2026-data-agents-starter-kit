# Submission Handoff — What the Docker Packager Needs to Know

Last verified commit: `f7208cf` on `origin/master`.

## TL;DR

```bash
# competition injects:
#   MODEL_API_URL       — internal Qwen3.5-35B-A3B endpoint
#   MODEL_API_KEY       — auth
#   MODEL_NAME=qwen3.5-35b-a3b

# the agent reads /input/, must write /output/task_<id>/prediction.csv
```

The runner currently writes to `artifacts/databao-demo/<run_id>/task_<id>/`, **not** `/output/`. The packager must redirect.

## Environment variables — defaults and what to keep / change

| Var | Default | Action for submission |
| --- | --- | --- |
| `MODEL_API_URL` / `MODEL_API_KEY` / `MODEL_NAME` | (unset locally) | **Injected by competition. Do not hard-code.** |
| `DATABAO_APPLY_VENDOR_PATCHES` | `1` (on) | **Keep on.** Without this, P0 gates / date probe / shape repairs silently disappear and score drops ~8-14 tasks per 49 sweep. |
| `DATABAO_INTERNAL_P0_GATES` | `1` (on) | **Keep on.** Drives empty/blank-display/numeric-id grounding reject paths. |
| `DATABAO_INTERNAL_SUBMIT_CRITIQUE_MODE` | `shadow` | **Keep.** Diagnostic-only by default. |
| `DATABAO_EXECUTOR_TYPE` | `lighthouse_salvage` | **Keep.** `separate` and `lighthouse` (non-salvage) are unsupported in this pipeline. |
| `DATABAO_TASK_TIMEOUT_SECONDS` | unset → uses CLI flag | Set per A-board / B-board budget (~110s for A-board). |
| `DATABAO_DATABAO_TIMEOUT_SECONDS` | `100` | Inner Databao timeout (capped by task budget). |
| `DATABAO_CONDITIONAL_RETRY` | `0` (off) | **Leave off for safe submission.** Opt-in experimental retry; on local single-sweep it gave +0.67 mean but adds ~50 s/Hard-task elapsed time which can blow the A-board 120 s/task budget. If A-board total has headroom, can flip to `1` for an ablation. |
| `DATABAO_RETRY_MAX_ATTEMPTS` | `2` | Only used when retry is on. |
| `DATABAO_RETRY_ONLY_ON_BAD_SIGNAL` | `1` | Only used when retry is on. |
| `DATABAO_RETRY_TEMPERATURES` | `0.0,0.35` | Only used when retry is on. |
| `DATABAO_INTERNAL_GROUNDING_REJECT` | `0` (off) | **Leave off.** Ablation showed regression. |
| `DATABAO_INTERNAL_P0_RATIO_REJECT` | `0` (off) | **Leave off.** Adapter's aggregate verifier already handles ratio; ablation showed this destroys the rescue input. |
| `DATABAO_INTERNAL_GROUNDING_PROBE` | `0` (off) | **Leave off.** Caused Stage 3 timeout regression. |
| `DATABAO_INTERNAL_SCHEMA_GRAPH_PROBE` | `0` (off) | **Leave off.** Net-negative in 3-run variance ablation. |

## CLI entry point

Console script: `dabench run-databao-demo --config <path> --task-timeout-seconds <N>`.
Installed via `uv pip install -e .` (or any equivalent that exposes `[project.scripts]` from `pyproject.toml`).

## Paths the packager must wire

1. **Input root** — the demo runner reads `app_config.dataset.root_path`. The shipped `configs/react_baseline.local.yaml` has `root_path: data/public/input`. For the competition image, point this at `/input` (either edit the config in-image or generate one at container start).
2. **Output root** — controlled by `DATABAO_DEMO_RUNS_DIR / <run_id>`, which resolves to `artifacts/databao-demo/<run_id>/task_<id>/prediction.csv`. The competition wants `/output/task_<id>/prediction.csv`. Two options:
   - patch `DATABAO_DEMO_RUNS_DIR` in `src/data_agent_baseline/cli.py` to `/output` and set the run ID to a fixed string (so the path becomes `/output/<run_id>/task_<id>/prediction.csv` — still wrong);
   - or run the demo, then move/symlink `artifacts/databao-demo/<run_id>/task_<id>/prediction.csv → /output/task_<id>/prediction.csv` at the end. **The second is simpler.**

If unsure, the safest end-of-container shell is roughly:

```bash
dabench run-databao-demo --config configs/react_baseline.local.yaml --task-timeout-seconds 110
last_run=$(ls -t artifacts/databao-demo | head -1)
mkdir -p /output
cp -r artifacts/databao-demo/"$last_run"/task_*/. /output/
# OR per-task:
# for d in artifacts/databao-demo/"$last_run"/task_*; do
#   tid=$(basename "$d")
#   mkdir -p /output/$tid
#   cp "$d/prediction.csv" /output/$tid/
# done
```

3. **Config file in-image** — the YAML's `agent.model / api_base / api_key` placeholders are **IGNORED** by the Databao demo runner (they're for the ReAct baseline). Do **not** hard-code `MODEL_API_KEY` into the YAML "to fix it"; the Databao runner reads it from the env var. The only field the runner actually uses is `dataset.root_path`.

## Files that must ship with the image

- `src/data_agent_baseline/` (full tree, especially `_vendor/databao_patches/`)
- `pyproject.toml`, `uv.lock`
- `configs/react_baseline.local.yaml` (or any config pointing `dataset.root_path` at `/input`)
- `data/public/` is **NOT needed at runtime**; the competition mounts `/input/` itself.

## Resource budget to target

A-board: ~60 tasks / 2 h = ~120 s/task. Hard tasks today average ~70 s without retry, ~118 s with retry. Run with retry **off** unless dry-run shows you have ≥30 % headroom on the total wall clock.

B-board: ~320 tasks / 12 h = ~135 s/task. More room, but retry still adds variance.

## Sanity check before submitting

1. Container starts, `dabench --help` works.
2. `dabench run-databao-demo --config configs/react_baseline.local.yaml --task-id task_25 --task-timeout-seconds 60` succeeds with `MODEL_API_URL` etc. injected. Verify a `prediction.csv` lands somewhere.
3. Confirm `/output/task_25/prediction.csv` exists after your post-copy step.
4. Vendor patches loaded — search the run's log JSON for `vendor_patch_status="updated"` or `"already_current"` (we log this in the initial diagnostic block).

## Things to leave alone

- Do not edit `src/data_agent_baseline/_vendor/databao_patches/*.py`. They are the live source of the P0 gate logic.
- Do not edit `tests/test_databao_demo.py` guardrails (`test_runner_has_no_task_or_difficulty_specific_branches`, `test_src_code_has_no_public_domain_memory_terms`). The agent expects these to keep failing if anyone reintroduces task-id branches or domain memory.
- Do not call `_install_vendor_databao_patches()` manually. `ensure_vendor_databao_patches()` runs once on first agent build via `build_databao_agent` and again at module load if needed.
