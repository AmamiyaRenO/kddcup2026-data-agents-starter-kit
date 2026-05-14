# Error Analysis

This file tracks current public-demo misses so we can choose fixes by evidence instead of adding
benchmark-shaped patches. It should be updated after meaningful full sweeps.

## Scope

Runs analyzed:

| Difficulty | Run ID | Prediction | Exact | Avg |
| --- | --- | ---: | ---: | ---: |
| easy | `20260514T021353Z` | `15/15` | `12/15` | `0.800` |
| medium | `20260514T015641Z` | `23/23` | `14/23` | `0.641` |
| easy | `20260514T103419Z` | `15/15` | `12/15` | `0.800` |
| medium | `20260514T103857Z` | `23/23` | `17/23` | `0.739` |
| hard | `20260514T105041Z` | `11/11` | `6/11` | `0.545` |

Planner, finalizer, semantic repair, and pre-Databao synthetic rescue were not active. Candidate
source was Databao-first plus the current generic adapter/postprocess layers.

Update 2026-05-14:

- A generic document-record display extractor and conservative final-guard blank/duplicate-column
  cleanup were added.
- This did not change Easy, substantially improved Medium, and lifted Hard while preserving
  `prediction.csv` stability.
- The strongest observed win was `task_349`: the correct value was already present, but extra blank
  and duplicate columns reduced the score. The new guard converted it to exact without task-specific
  logic.

## Easy Misses

| Task | Score | Prediction Summary | Gold Summary | Loss Reason | Repair Safety |
| --- | ---: | --- | --- | --- | --- |
| `task_11` | `0.000` | `ID=2495750, SEX=F, Diagnosis=SLE` | 3 patient rows: `163109`, `2803470`, `4395720` with `SEX=F`, `Diagnosis=SLE` | Databao selected the wrong thrombosis severity population. Output shape and columns are plausible, but source rows are wrong. | Not adapter-safe. Needs better Databao/table semantics for categorical severity mapping. |
| `task_86` | `0.000` | `name` column with 17 races, including extra `Japanese Grand Prix` | `name` column with 16 races, excluding the extra row | Adapter now selects the correct display column, but Databao included one extra source/filter row. Targeted run `20260514T021249Z` was exact, so this is model/source-filter variance rather than column selection. | Not safe as a final-guard rule. Future work should inspect Databao SQL/filter state when the extra row appears. |
| `task_89` | `0.000` | `time=14.925` | `time=16.445` | Databao found a plausible row and adapter selected the right `time` column, but the race/result row is wrong. | Not adapter-safe. Requires source row/race disambiguation inside Databao or a verified structured rescue. |

Easy conclusion: current remaining misses are mostly source/filter semantics. The adapter-safe column
fixes already helped `task_86` in targeted runs, but cannot safely remove an arbitrary extra row
without evidence from Databao's executed query.

## Medium Misses

| Task | Score | Prediction Summary | Gold Summary | Loss Reason | Repair Safety |
| --- | ---: | --- | --- | --- | --- |
| `task_163` | `0.000` | `category,total_approved_value`: `Food=121.14`, `Advertisement=54.25` | `type=Meeting`, total `175.39` | Predicted detail expense categories instead of the requested event/type-level total. Values sum to the gold amount, but the grouping/display attribute is wrong. | Not adapter-safe unless Databao exposes event/type evidence. Do not hard-code category collapsing. |
| `task_169` | `0.000` | `avg_avg_monthly_consumption=5519.475...` | `AVG(Consumption)/12=459.956...` | Databao computed an annual average but did not convert to monthly. | Risky adapter repair. A generic "monthly means divide by 12" rule may be valid sometimes but can overfit units/calendar assumptions. |
| `task_173` | `0.000` | Empty `Country` column | `Country`: `CZE`, `SVK` | Databao returned an empty result, likely due date filtering or table join semantics. | Not adapter-safe from empty output. Needs Databao/date filter diagnostics or structured SQL rescue with exact table mapping. |
| `task_180` | `0.000` | 153 rows of `CustomerID,Consumption` | 9 `Consumption` rows | Databao produced a broad detail table and missed one or more filters. Adapter selected the right answer column family but cannot infer the missing source filter. | Not adapter-safe. Needs query/filter repair, not final guard. |
| `task_199` | `0.000` | 2 rows: `Charter Funding Type,cname` for district-level `Riverside` | 6 school rows: `sname,Charter Funding Type` | Databao selected district/county rows rather than school rows. Duplicate-row cleanup reduced noise but also reveals the wrong entity level. | Not adapter-safe. Needs source table/entity-level correction. |
| `task_200` | `0.500` | `element=p, atom_count=1` | single count `1` | Correct count value is present, but an intermediate predicate/display column remains. | Adapter-safe candidate. A count/total question with a count-like answer column can safely drop predicate columns when the count column exists. |
| `task_243` | `0.000` | `post_count=3` | post/vote ratio `0.375` | Databao returned only numerator-like count, missing denominator/evidence for ratio. | Not safe unless denominator evidence is present in the candidate/log. Ratio repair should stay gated by numerator + denominator proof. |
| `task_257` | `0.250` | `ViewCount=1708` plus blank `LastEditorDisplayName`, blank `OwnerDisplayName` | `ViewCount=1708`, `DisplayName=mbq` | Correct metric is present, but ID/display resolution chose display columns already present in the row instead of resolving the intended user display value. | Potential adapter-safe work: improve generic multi-hop ID-to-display resolution when candidate has user IDs and context has display-name tables. Must avoid domain-specific names. |
| `task_283` | `0.000` | `percentage=31.6` | percentage `31.2` | Databao/adapter produced a close but wrong percentage, probably from a different denominator/population. | Not adapter-safe without population evidence. Do not tune rounding or constants. |

Medium conclusion: only `task_200` is clearly adapter-safe. `task_257` is a plausible generic
identifier/display-resolution improvement, but it needs careful evidence to avoid resolving through
the wrong table. The rest are Databao source/filter/operation mistakes.

## Cross-Task Failure Types

| Failure Type | Tasks | Meaning | Recommended Next Step |
| --- | --- | --- | --- |
| Wrong source/filter rows with plausible final shape | `task_11`, `task_86`, `task_89`, `task_173`, `task_180`, `task_199` | Databao produced a table that looks answer-shaped, but the underlying query/filter/entity level is wrong. | Observe Databao generated SQL/code and intermediate tables. Consider rescue only when the executed query contains enough evidence to repair safely. |
| Aggregation/unit operation incomplete | `task_163`, `task_169`, `task_243`, `task_283` | Databao performed part of the operation but missed grouping, denominator, or unit conversion. | Keep ratio/aggregation repair evidence-gated. Do not recompute from full context unless table/column mapping and numerator/denominator evidence are explicit. |
| Extra/intermediate column with correct answer value present | `task_200` | Answer value exists; adapter retained an unnecessary predicate/display column. | Safe adapter target. Prefer count-like answer column for count/total questions when present. |
| Display resolution incomplete | `task_257` | Correct metric exists but display/name column is wrong or blank. | Generic ID/display resolution target. Require high-coverage ID mapping and avoid using unrelated helper tables. |

## Candidate Fix Queue

0. **Kept high-yield generic output cleanup**
   - Trigger: final answer table contains all-blank columns or same-family duplicate columns with
     identical values, while at least one nonblank answer column remains.
   - Action: remove the blank/duplicate columns in `final_answer_shape_guard`.
   - Status: implemented and retained. Full sweeps after implementation:
     - Easy `20260514T103419Z`: avg `0.800`, exact `12/15`, prediction `15/15`.
     - Medium `20260514T103857Z`: avg `0.739`, exact `17/23`, prediction `23/23`.
     - Hard `20260514T105041Z`: avg `0.545`, exact `6/11`, prediction `11/11`.
   - Why safe: it does not change rows, compute values, infer filters, or choose new source tables.

1. **Safe adapter fix candidate: `task_200` pattern**
   - Trigger: question asks count/total, candidate contains one count-like answer column and one
     predicate/display column.
   - Action: keep the count-like column only.
   - Risk: low if limited to explicit count-like answer columns.
   - Status: implemented in the adapter. Targeted run `20260514T025829Z` did not expose the same
     repair opportunity because Databao returned an empty `element,bond_type` table with no
     count-like answer column, so this remains a conditional fix rather than a full task fix.

2. **Careful generic display-resolution study: `task_257` pattern**
   - Trigger: metric column is correct, candidate has user/entity ID columns or blank display fields,
     and context has high-confidence display mapping.
   - Action: generate a postprocessed candidate with metric + resolved display column.
   - Risk: medium. Prior bugs came from resolving IDs through helper tables with matching values but
     wrong entity families.
   - Status: partially implemented as generic ID-resolution hardening. Identifier lookup now filters
     context tables by candidate values, normalizes integer-like float IDs such as `88.0 -> 88`, fills
     blank companion display columns, and avoids mapping unrelated ID families to `users.Id`. Targeted
     run `20260514T031257Z` no longer timed out in postprocess, but Databao returned only
     `Id,DisplayName` without `ViewCount`, so the remaining loss is source/output selection.

3. **Databao SQL/code observation for source/filter misses**
   - Applies to `task_11`, `task_86`, `task_89`, `task_173`, `task_180`, `task_199`.
   - Goal: determine whether Databao's executed SQL already contains enough structure to repair, or
     whether it selected the wrong table/row population entirely.
   - Do not add hard-coded task/domain rules.

4. **Aggregation/ratio rescue remains gated**
   - Applies to `task_163`, `task_169`, `task_243`, `task_283`.
   - Only promote repair candidates when numerator, denominator, grouping key, filter evidence, and
     sanity checks are complete.
   - New observation hook: `databao_internal_observation` now records when Databao submits a scalar
     aggregate for a ratio/percentage question without a ratio operation. Targeted run
     `20260514T032735Z` on `task_243` correctly logged `ratio_question_without_ratio_operation`
     while leaving the answer unchanged.

## Guardrails

- Do not branch by task id, difficulty, file stem, public-set domain, entity, or category.
- Do not add scoring-aware prompt wording.
- Do not restore broad planner/finalizer/semantic-repair paths.
- Prefer targeted adapter fixes only when the correct answer value is already present or when
  Databao execution evidence proves the needed operation.
