# Submission Handoff — What the Docker Packager Needs to Know

Last verified commit: `e5c5cd8` on `origin/master`.

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
| `DATABAO_GENERALIZATION_PROFILE` | `0` (off) | **Optional hidden-set A/B profile.** Set to `1` only for a separate submission aimed at generalization: prompt removes proactive mapping/ratio bullets, forced Databao P0 reject keeps only `empty_submit_result`, and adapter column pruning requires stronger evidence. Local public ablation: easy `0.733`, medium `0.558`, hard `0.379` with prediction `49/49`; this is not the public high-score profile. |
| `DATABAO_FINALITY_SALVAGE_PROFILE` | `0` (off) | **Optional v11 high-upside A/B profile.** Set to `1` only if testing a more submit-conservative Databao internal loop: no-submit finality retry defaults to two attempts and the feedback prefers submitting a plausible latest non-empty result over restarting broad exploration. Does not change adapter answer shaping or P0 reject flags. |
| `DATABAO_SOFT_P0_PROFILE` | `0` (off) | **Leave off unless explicitly testing high-risk generalization.** When set to `1`, forced P0 reject is limited to `empty_submit_result`. This is intentionally separate from `DATABAO_FINALITY_SALVAGE_PROFILE` because local ablations showed P0 softening can damage otherwise correct submits. |
| `DATABAO_INTERNAL_P0_GATES` | `1` (on) | **Keep on.** Drives empty/blank-display/numeric-id grounding reject paths. |
| `DATABAO_INTERNAL_NO_SUBMIT_FINALITY_RETRY` | `1` (on) | **Keep on.** ⭐ Adds a one-shot retry asking the model to call `submit_result` when it returned a DataFrame but no submit. Cuts no-submit/finality misses on Medium/Hard. Capped by `DATABAO_INTERNAL_NO_SUBMIT_FINALITY_MAX_RETRIES=1`. |
| `DATABAO_ENABLE_THINKING` | `0` (off) | **⭐ Flip to `1` on submission.** Injects `extra_body={"chat_template_kwargs":{"enable_thinking":True}}` into ChatOpenAI, matching the competition vLLM's `--reasoning-parser qwen3` contract. OpenRouter ignores it locally, so the local sweep won't show timing change — only takes effect on the dedicated vLLM. |
| `DATABAO_MAX_WORKERS` | unset → `1` | **⭐ Set `4` on submission.** Env-var fallback for parallel dispatch. Reads inside `run_databao_tasks`, so `python main.py` style entrypoints get concurrency without code changes. CLI flag `--max-workers` still wins when both are set. Threading is only activated in child-process mode (default when timeout > 0). |
| `DATABAO_INTERNAL_SUBMIT_CRITIQUE_MODE` | `shadow` | **Keep.** Diagnostic-only by default. |
| `DATABAO_EXECUTOR_TYPE` | `lighthouse_salvage` | **Keep.** `separate` and `lighthouse` (non-salvage) are unsupported in this pipeline. |
| `DATABAO_TASK_TIMEOUT_SECONDS` | unset → uses CLI flag | **Set `600`.** ⭐ Earlier 110s recommendation was wrong: A-board total wall-clock (7200s) is not the bottleneck; per-task partial credit is. 110s killed slow-Hard partial credit (cost ~−0.03 score). With `--max-workers=4`, 600s easily fits. |
| `DATABAO_DATABAO_TIMEOUT_SECONDS` | `100` | **Keep default `100`.** Inner LLM-call timeout. Capped dynamically at `task_timeout - 20s`. Earlier guidance suggested 180s as buffer for thinking-mode reasoning tokens, but the dedicated vLLM is fast enough that 100s is comfortable; raising it only adds risk of compounding into the outer task timeout. |
| `DATABAO_INTERNAL_P0_RATIO_REJECT` | `0` (off) | **Leave off.** Ablation showed this over-rejects and regresses otherwise useful Databao outputs. |
| `DATABAO_INTERNAL_GROUNDING_PROBE` | `0` (off) | **Leave off.** Caused Stage 3 timeout regression. |
| `DATABAO_INTERNAL_SCHEMA_GRAPH_PROBE` | `0` (off) | **Leave off.** Net-negative in 3-run variance ablation. |

⭐ = new since the prior submission attempt that scored 0.30 / 0.33.

## CLI / config knobs (not env vars)

These are set via CLI flags **or** the YAML config — not env vars.

| Knob | YAML key | CLI flag | Default | Action |
| --- | --- | --- | --- | --- |
| Parallel task dispatch | `run.max_workers` | `--max-workers` | `4` | **⭐ Pass `--max-workers 4` explicitly.** Rule docs recommend multi-threading; `run_databao_tasks` now uses `ThreadPoolExecutor` when `max_workers > 1`. Each task still runs in its own subprocess via `multiprocessing.spawn`, so thread layer is safe. Local sweep: 22.5 min → 9 min wall-clock with workers=4. |
| Per-task wall-clock | `run.task_timeout_seconds` | `--task-timeout-seconds` | `600` | **Pass `--task-timeout-seconds 600`** unless env var already pinned it. CLI wins if both set. |

## CLI entry point

Console script: `dabench run-databao-demo --config <path> --task-timeout-seconds <N> --max-workers <K>`.
Installed via `uv pip install -e .` (or any equivalent that exposes `[project.scripts]` from `pyproject.toml`).

## Paths the packager must wire

1. **Input root** — the demo runner reads `app_config.dataset.root_path`. The shipped `configs/react_baseline.local.yaml` has `root_path: data/public/input`. For the competition image, point this at `/input` (either edit the config in-image or generate one at container start).
2. **Output root** — controlled by `DATABAO_DEMO_RUNS_DIR / <run_id>`, which resolves to `artifacts/databao-demo/<run_id>/task_<id>/prediction.csv`. The competition wants `/output/task_<id>/prediction.csv`. Two options:
   - patch `DATABAO_DEMO_RUNS_DIR` in `src/data_agent_baseline/cli.py` to `/output` and set the run ID to a fixed string (so the path becomes `/output/<run_id>/task_<id>/prediction.csv` — still wrong);
   - or run the demo, then move/symlink `artifacts/databao-demo/<run_id>/task_<id>/prediction.csv → /output/task_<id>/prediction.csv` at the end. **The second is simpler.**

### Path A — `dabench` CLI (recommended for new images)

```bash
# ⭐ env vars
export DATABAO_ENABLE_THINKING=1

dabench run-databao-demo \
  --config configs/react_baseline.local.yaml \
  --task-timeout-seconds 600 \
  --max-workers 4

last_run=$(ls -t artifacts/databao-demo | head -1)
mkdir -p /output
for d in artifacts/databao-demo/"$last_run"/task_*; do
  tid=$(basename "$d")
  mkdir -p /output/$tid
  cp "$d/prediction.csv" /output/$tid/
done
```

### Path B — direct `python main.py` (matches v7 image layout)

The v7 image entrypoint runs `python /app/main.py`, which calls
`run_databao_tasks` directly **without** passing `--max-workers`. To
turn on concurrency for this entrypoint **without editing main.py**,
set the env var:

```dockerfile
ENV DATABAO_ENABLE_THINKING=1
ENV DATABAO_MAX_WORKERS=4
ENV DATABAO_TASK_TIMEOUT_SECONDS=600
```

`run_databao_tasks` now reads `DATABAO_MAX_WORKERS` as a fallback when
the caller did not pass `max_workers` explicitly. main.py needs **no
changes** — the env var alone activates parallel dispatch.

⚠️ Verify the v7-style image writes predictions to the
`output_root=/output` argument main.py passes in, **not** to
`artifacts/databao-demo/<run_id>/`. The post-copy step in Path A is
unnecessary here because main.py already writes directly to `/output`.

3. **Config file in-image** — the YAML's `agent.model / api_base / api_key` placeholders are **IGNORED** by the Databao demo runner (they're for the ReAct baseline). Do **not** hard-code `MODEL_API_KEY` into the YAML "to fix it"; the Databao runner reads it from the env var. The only field the runner actually uses is `dataset.root_path`.

## Files that must ship with the image

- `src/data_agent_baseline/` (full tree, especially `_vendor/databao_patches/`)
- `pyproject.toml`, `uv.lock`
- `configs/react_baseline.local.yaml` (or any config pointing `dataset.root_path` at `/input`)
- `data/public/` is **NOT needed at runtime**; the competition mounts `/input/` itself.

## Resource budget to target

Per the published rules (`dataagent.top/rules`):

| Board | Tasks | Wall-clock | Per-task average | Hard share |
| --- | ---: | ---: | ---: | ---: |
| **A-board** | 57 | 2 h (7200 s) | ~126 s | **52.6 %** (Hard + 3.5 % Extreme) |
| **B-board** | ~320 | 12 h | ~135 s | varies |

Concurrency-corrected budget for A-board with `--max-workers 4`:

- Expected wall-clock ≈ `57 tasks × ~120 s × thinking_penalty(~2×) / 4 workers ≈ 3400 s` — leaves ~3800 s headroom inside the 7200 s cap.
- The previous "120 s/task" framing assumed sequential dispatch. With workers=4 the **per-task** budget is effectively `4 × wall-clock / 57 ≈ 500 s/task` of compute. Setting `--task-timeout-seconds 600` is conservative.

Rule excerpt:
> "we recommend participants use multi-threading/multi-processing for batch parallel acceleration of different input samples (refer to the max_workers setting in the Starter Kit code)"

## Sanity check before submitting

1. Container starts, `dabench --help` works.
2. `dabench run-databao-demo --config configs/react_baseline.local.yaml --task-id task_25 --task-timeout-seconds 60` succeeds with `MODEL_API_URL` etc. injected. Verify a `prediction.csv` lands somewhere.
3. Confirm `/output/task_25/prediction.csv` exists after your post-copy step.
4. Vendor patches loaded — search the run's log JSON for `vendor_patch_status="updated"` or `"already_current"` (we log this in the initial diagnostic block).
5. ⭐ With `DATABAO_ENABLE_THINKING=1` on the A-board endpoint, a Hard sample task should show noticeably longer `databao_ask` elapsed (~2-4× vs OFF). If timing is identical, the vLLM is silently dropping the `chat_template_kwargs` and thinking is **not** active — open a question with the organizers.
6. ⭐ Stdout should show `Max workers: 4` near the run banner, confirming the CLI flag landed.

## Things to leave alone

- Do not edit `src/data_agent_baseline/_vendor/databao_patches/*.py`. They are the live source of the P0 gate logic.
- Do not edit `tests/test_databao_demo.py` guardrails (`test_runner_has_no_task_or_difficulty_specific_branches`, `test_src_code_has_no_public_domain_memory_terms`). The agent expects these to keep failing if anyone reintroduces task-id branches or domain memory.
- Do not call `_install_vendor_databao_patches()` manually. `ensure_vendor_databao_patches()` runs once on first agent build via `build_databao_agent` and again at module load if needed.
