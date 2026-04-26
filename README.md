<div align="center">

# Databao 链路运行说明

</div>

> 本仓库当前主要维护 `src/data_agent_baseline/run/databao_demo.py` 这一条 Databao-backed demo 链路。它读取 DataAgent-Bench public demo 输入，调用 Databao 生成候选答案表，再经过通用后处理、答案形状约束和本地 evaluator 产出可检查的运行结果。

## 适用范围

这份说明面向本地 Databao 链路调试，不是原始 ReAct baseline 的使用文档。常用入口是：

```powershell
uv run dabench run-databao-demo --config configs/react_baseline.local.yaml
```

当前链路默认读取：

- 输入数据：`data/public/input/`
- public gold：`data/public/output/task_<id>/gold.csv`
- 本地配置：`configs/react_baseline.local.yaml`
- 输出目录：`artifacts/databao-demo/<run_id>/`
- 核心实现：`src/data_agent_baseline/run/databao_demo.py`

## 运行前准备

在 Windows PowerShell 中，建议先设置本机 `uv` 路径和缓存目录：

```powershell
$env:Path = "C:\Users\tianj\.local\bin;$env:Path"
$env:UV_CACHE_DIR = "D:\Data\.uv-cache"
```

安装或同步依赖：

```powershell
uv sync --extra dev
```

Databao/OpenAI-compatible 模型配置通过环境变量传入，不要把真实 key 写入配置文件：

```powershell
$env:MODEL_API_URL = "https://openrouter.ai/api/v1"
$env:MODEL_API_KEY = $env:Open_Router_Key
$env:MODEL_NAME = "qwen/qwen3.5-35b-a3b"
```

建议的超时设置：

```powershell
$env:DATABAO_DATABAO_TIMEOUT_SECONDS = "60"
$env:DATABAO_AUX_TIMEOUT_SECONDS = "30"
```

## 推荐运行方式

### 单任务 smoke test

```powershell
uv run dabench run-databao-demo `
  --config configs/react_baseline.local.yaml `
  --task-id task_25 `
  --task-timeout-seconds 150
```

### 按难度运行

```powershell
uv run dabench run-databao-demo `
  --config configs/react_baseline.local.yaml `
  --difficulty easy `
  --task-timeout-seconds 150

uv run dabench run-databao-demo `
  --config configs/react_baseline.local.yaml `
  --difficulty medium `
  --task-timeout-seconds 150

uv run dabench run-databao-demo `
  --config configs/react_baseline.local.yaml `
  --difficulty hard `
  --task-timeout-seconds 150
```

### 限制任务数量

```powershell
uv run dabench run-databao-demo `
  --config configs/react_baseline.local.yaml `
  --difficulty easy `
  --limit 5
```

## 关键开关

### Structured planner

`DATABAO_STRUCTURED_PLANNER_MODE` 控制安全 JSON planner 是否参与 rescue：

- `fallback`：默认模式。Databao 先跑，只有在门控认为需要 rescue 且预算足够时才尝试 planner。
- `off`：完全关闭 planner。适合排查 Databao 原始输出、避免 planner 长尾超时。
- `first`：兼容旧实验入口，不建议作为默认评测模式。

关闭 planner：

```powershell
$env:DATABAO_STRUCTURED_PLANNER_MODE = "off"
```

注意：CLI 目前打印的 `Planner mode: fallback` 是展示值；若命令没有显式传 `--planner-mode`，实际执行仍会读取环境变量。要确认是否生效，请看任务日志里的 `structured_planner.attempted` 和 `failure_reason`。关闭成功时通常会看到：

```text
Structured planner disabled by DATABAO_STRUCTURED_PLANNER_MODE=off.
```

### Finalizer

`DATABAO_FINALIZER_MODE` 控制额外 LLM finalizer：

- `off`：默认关闭。
- `always`：总是尝试 finalizer，会增加延迟和漂移风险。

```powershell
$env:DATABAO_FINALIZER_MODE = "off"
```

### Heuristic level

`DATABAO_HEURISTIC_LEVEL` 控制通用启发式强度：

- `generic`：默认。只使用通用 schema、document records、answer contract、verifier、candidate ranking 和 final shape guard。
- `experimental_generic`：允许更激进的通用文本记录抽取，但仍不能引入 public set 的领域记忆或 task-id 分支。

```powershell
$env:DATABAO_HEURISTIC_LEVEL = "generic"
```

## Databao 链路流程

单个任务大致按以下步骤执行：

1. 读取 `task.json` 和 `context/`。
2. 加载 CSV、JSON、SQLite/DB、Markdown 文档为通用上下文表。
3. 构建 schema graph，记录表、列、样例值和候选 join 关系。
4. 运行通用 context retriever，给相关表、列、样例、文档片段和 join path 打分。
5. 从问题和 schema 推断 `AnswerContract`。
6. 注册上下文给 Databao，调用 Databao 生成原始候选 DataFrame。
7. 生成候选答案：`databao_raw`、通用后处理候选、verifier 候选，必要时再考虑 planner rescue。
8. 用通用 ranker 选择最合适候选。
9. 运行 final answer shape guard，剪掉明显不该输出的调试列、元数据列和弱相关列。
10. 写出 `prediction.csv`、任务日志、进度日志和 `summary.json`。

## 输出文件

每次 `run-databao-demo` 会写到：

```text
artifacts/databao-demo/<run_id>/
├── summary.json
├── logs/
│   ├── task_<id>.json
│   └── task_<id>.progress.json
└── task_<id>/
    └── prediction.csv
```

重点看这些字段：

- `prediction_written`：是否实际写出了 `prediction.csv`。
- `scorable`：是否可以被 public evaluator 评分。
- `succeeded`：内部 pipeline 是否成功结束。它和 `prediction_written` 不完全等价。
- `candidate_source`：最终选择的候选来源，例如 `databao_raw`、`databao_raw_postprocessed`。
- `candidate_scores`：候选排序原因。
- `postprocessing`：后处理和 final shape guard 做过哪些变换。
- `structured_planner`：planner 是否尝试、跳过或被关闭。
- `timings`：各阶段耗时。

## 本地评估

对某个 run 进行 public demo 评估：

```powershell
uv run dabench evaluate-public-run artifacts/databao-demo/<run_id>
```

评估输出包含：

- `PER_TASK_AVERAGE_SCORE`
- `OVERALL_COLUMN_COVERAGE`
- `EXACT_NO_EXTRA_TASKS`
- `PREDICTION_WRITTEN`
- `SCORABLE_TASKS`

常见不 exact 原因：

- `matched=0/N`：预测列签名没有命中 gold，通常是值或语义错。
- `extra > 0`：输出了 gold 不需要的额外列。
- `Rows` 不一致：返回过多、过少或空结果。
- `Pred=no`：没有写出 `prediction.csv`。
- `Artifact=fail` 但 `Pred=yes`：任务 summary 记录失败，但已有 provisional prediction，可以继续评估。

## 本地验证

修改代码后先跑无 API 检查：

```powershell
uv run python -m unittest discover tests
uv run --extra dev ruff check src tests
uv run python -m compileall src
```

当前测试重点覆盖：

- Databao 环境变量加载。
- 上下文源注册和通用表加载。
- schema graph 和 join candidate 构建。
- 通用后处理、答案列校验、比例/百分比 verifier。
- planner 关闭、跳过和 rescue 行为。
- timeout 后是否保留已写出的 provisional prediction。
- public evaluator 的列签名评分。

## 调试建议

1. 先用 `--task-id` 复现单个失败任务。
2. 看 `logs/task_<id>.json`，确认 `candidate_source`、`candidate_scores` 和 `postprocessing`。
3. 如果 `prediction.csv` 已写出但 summary 是失败，优先检查 `timings` 和 timeout artifact 是否覆盖了最终状态。
4. 如果 `matched=0/N`，打开 `prediction.csv` 和 gold 比较值，而不是只看列名。
5. 如果只有 extra columns，优先改 final shape guard 或 answer column verifier。
6. 如果行数差很多，优先看 Databao 原始候选和 verifier/ranker 是否有足够证据做行选择。
7. 避免加入 task-id、文件名、领域实体、public gold 记忆相关分支；生产链路只允许通用策略。

## 常用命令速查

```powershell
# 环境
$env:Path = "C:\Users\tianj\.local\bin;$env:Path"
$env:UV_CACHE_DIR = "D:\Data\.uv-cache"
$env:MODEL_API_URL = "https://openrouter.ai/api/v1"
$env:MODEL_API_KEY = $env:Open_Router_Key
$env:MODEL_NAME = "qwen/qwen3.5-35b-a3b"
$env:DATABAO_DATABAO_TIMEOUT_SECONDS = "60"
$env:DATABAO_AUX_TIMEOUT_SECONDS = "30"
$env:DATABAO_STRUCTURED_PLANNER_MODE = "off"
$env:DATABAO_FINALIZER_MODE = "off"

# 单任务
uv run dabench run-databao-demo --config configs/react_baseline.local.yaml --task-id task_25 --task-timeout-seconds 150

# 分难度
uv run dabench run-databao-demo --config configs/react_baseline.local.yaml --difficulty easy --task-timeout-seconds 150
uv run dabench run-databao-demo --config configs/react_baseline.local.yaml --difficulty medium --task-timeout-seconds 150
uv run dabench run-databao-demo --config configs/react_baseline.local.yaml --difficulty hard --task-timeout-seconds 150

# 评估
uv run dabench evaluate-public-run artifacts/databao-demo/<run_id>

# 无 API 检查
uv run python -m unittest discover tests
uv run --extra dev ruff check src tests
uv run python -m compileall src
```
