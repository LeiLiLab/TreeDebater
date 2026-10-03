# Incremental rebuttal planning — process log

## Objective and authorization

- Implement a Linear incremental baseline and explicit argument correction, then Adaptive updates and tree-driven early rebuttal planning; compare before/after using `google.gemma-4-26b-a4b`.
- User approved **USD 200 total** for experiments within this task, including generation, judging, retries, and validation. This is a cumulative ceiling across runs and restarts, not a per-run allowance.
- User requested this file as the ongoing record of important settings, results, code changes, commands, and cost.
- Do not store provider credentials, authorization headers, or private keys here or in run artifacts.

## Workspace and baseline

- Repository: `/home/danqingwang/workspace/clone/TreeDebater` (contains the live web backend).
- New branch: `feat/incremental-rebuttal-planning`.
- Baseline commit: `065cc3c897c2e7d337391e465fbe2fb9a7828bac`.
- Original branch: `integrate-streaming-fixes`.
- `/home/danqingwang/workspace/debate` contains the older batch implementation and existing user edits; the implementation work is in TreeDebater.
- Pre-existing untracked paths are being preserved: `debate-app/frontend/`, `debate-app/reports/`, `debate-app/scripts/live_audio_smoke.py`, `emnlp_res/`, `plan.md`.

## Verified findings

- The web backend incrementally transcribes audio and calls `TreeDebater._analyze_statement()` while the opponent speaks. Planning, speech generation, feedback/revision, and TTS currently occur on the AI turn.
- Web input defaults: 3-second ASR batches, 60-word analysis batches, 15-second text wait, plus end-of-turn drain.
- Current tree actions cover propose/reinforce/attack/rebut; explicit revision/retraction and dependency invalidation need implementation.
- `speak()` mutates conversation/evidence state and enters TTS, so it must not be reused as a speculative background preparation operation.
- Each engine worker is serial; cancelling an asyncio waiter does not cancel its underlying model call.
- AWS Bedrock LiteLLM proxy is at `http://127.0.0.1:4000/v1`; ports 3000/8000 are not listening on `aries.cs.ucsb.edu` at the last check.
- Read-only proxy checks returned 200 for `/health/liveliness`, `/health/readiness`, and `/v1/models`. Listed aliases: `gpt-5.6-sol`, `google.gemma-4-26b-a4b`, `nvidia.nemotron-super-3-120b`. Readiness says database is not connected; these checks do not prove upstream inference works.
- Gemma is routed to Bedrock Mantle through the local proxy. No proxy configuration has been modified by this task.

## Experiment design

Keep the model, input chunks, prior debate context, evidence, answer budget, and final generation/revision path comparable. Separate the effects of state representation, correction, and update scheduling.

### 方案总表（持续维护，包含所有已尝试方案）

每次增加或修改方案都更新本表；具体配置、实验轮次、结果和失败记录追加在后面的过程日志。实现/离线验证不等同于模型评测完成。

| 方案 | 特点 | 主要取舍 | 尝试与验证状态 |
| --- | --- | --- | --- |
| Legacy (`legacy`) | 边听边维护原始论证树，结束后规划并生成反驳 | 保留论证关系；缺少显式撤回修正和提前反驳笔记 | 第一轮：12 案例 × 2 次，Gemma 评分 |
| End-of-turn (`end_of_turn`) | 收集完整发言后集中分析树和生成反驳 | 减少重复分析；工作集中在端点后 | 第一轮：12 × 2，Gemma 评分 |
| Linear (`linear`) | 不用论证树，每段输入更新自由文本反驳笔记 | 简单、较低开销；限定条件和假设容易混淆 | 第一轮 12 × 2；第二轮 10 × 1；第三轮补充 8 × 1（后两轮 GPT-5.6），本轮通过率 58.3%，等待 13.64s |
| Corrected Tree (`corrected_tree`) | 树支持说话者自己的修改/撤回，归档失效分支 | 修正旧目标；仍在端点后规划反驳 | 第一轮：12 × 2，Gemma 评分 |
| Adaptive Linear (`adaptive_linear`) | Linear 加模型门控，决定更新或等待 | 可跳过重复工作；门控本身增加调用与延迟 | 第一轮：12 × 2，Gemma 评分 |
| Tree Plan (`tree_plan`) | 可纠错论证树驱动流式反驳笔记 | 同时组织论证关系和提前准备；树与笔记维护较贵 | 第一轮：12 × 2，Gemma 评分；第三轮：8 × 1，GPT-5.6，通过率 50.0%，模拟文本等待 17.28s |
| Adaptive Tree (`adaptive_tree`) | Tree Plan 加模型门控，按需更新树和笔记 | 尝试减少更新；流程更复杂且有门控开销 | 第一轮：12 × 2，Gemma 评分 |
| Structured Linear (`structured_linear`) | 将当前观点、原文引用、范围/例外和反驳假设分开保存 | 可检查来源；结构合法不保证语义正确，本轮通过率下降 | 第二轮：10 × 1，GPT-5.6 评分；另有旧案例开发诊断 |
| Grounded Linear (`grounded_linear`) | 结构化状态加针对目标、例外和事实依据的反馈/修订 | 不新增反馈调用；减少无依据断言，但整体质量收益未证实 | 第二轮：10 × 1；第三轮：8 × 1，均 GPT-5.6；第三轮通过率 66.7%，等待 9.87s |
| Light Linear (`light_linear`) | Grounded Linear 加精确重复跳过、未完句缓冲、选择性门控 | 减少无效工作；调度的独立收益仍不确定 | 第二轮：10 × 1；第三轮：8 × 1，均 GPT-5.6；第三轮通过率 70.8%，等待 10.52s；另有旧案例单次真实 ASR/TTS 对照 |
| Grounded Tree (`grounded_tree`) | 保留论证树；反驳绑定有效节点和节点原文，利用攻击关系与未回应目标排序；加依据核对 | 让反驳可绑定树目标；比原版更快，但提取/匹配失败仍会触发原文回退，树的独立质量收益未证实 | 第三轮：8 × 1，GPT-5.6，通过率 62.5%，等待 11.37s；9 次中间状态回退，3/8 最终回退 |
| Light Tree (`light_tree`) | Grounded Tree 加重复跳过、未完句缓冲与选择性门控，结束时强制处理积压 | 本轮调用从 Grounded Tree 的 14.0 降至 13.25；延迟未进一步下降，质量仍受提取与条件覆盖限制 | 第三轮：8 × 1，GPT-5.6，通过率 62.5%，等待 11.96s；7 次中间状态回退，2/8 最终回退 |
| Flat Tree (`flat_tree`) | 修复后的同一树更新与节点来源；索引式规划和限定条件账本，但规划/输出移除祖先、回应边和结构排序 | 用于隔离显式关系信息的收益；仍可从完整发言自行推断关系，仍支付建树成本 | 第四轮：实现及离线验证完成，待评测 |
| Branch Tree (`branch_tree`) | 在 Flat Tree 的同等节点/来源基础上，使用我方质疑—对方回应的路径、已有回应和未回应分支摘要 | 让关系结构直接指导下一步反驳；响应边不代表问题已解决，额外上下文可能增加负担 | 第四轮：实现及离线验证完成，待评测 |

第一轮、第二轮的不同评分模型和案例不能直接混合排名。最新方向以论证树为主方法，Linear 用于消融比较；后续评分统一 GPT-5.6。

Evaluate targeted rebuttal quality, final-condition correctness, claim coverage, unsupported assertions, end-of-turn residual latency, and total input/output tokens and cost. Include late qualifiers, reversals, withdrawals, repeated content, and split clauses. Report measured text/planning latency separately from actual audible latency; do not describe a simulated timeline as a live audio measurement. Judges see delivered answers, not private preparation traces. Keep development cases separate from final held-out comparison.

## Budget and accounting

**Latest completed run:** the original Linear control is now complete. Approved cumulative cap remains **USD200**; cumulative provider-usage estimate **$2.88722584**, active guarded occupancy **$12.93704176**, available **$187.06295824**, zero pending requests. Historical pre-dispatch reservations total $276.9705208 and are not current occupancy. Successful requests settle at 4× verified usage; historical failed/unknown/audio bounds remain. The earlier USD320 proposal is withdrawn.

- Approved cumulative cap: **USD 200.00**.
- At creation of this log: **0 paid inference requests launched by this task; attributable experiment cost USD 0.00**. Existing unrelated proxy traffic is excluded.
- Authoritative pricing source located: <https://aws.amazon.com/bedrock/pricing/>. Search results show Gemma 4 26B A4B rates of $0.13/$0.40 and $0.16/$0.48; units, tier, and region must be confirmed before using a rate.
- Before launch: record exact settings, expected cost and conservative ceiling, then enable a persistent ledger which reserves an upper bound before dispatch, accounts for usage/errors/retries, and stops dispatch before the cumulative cap. No automatic cap increase.
- All failures and unresolved in-flight reservations must remain accounted for across restarts.

## Code changes

| File | Change | Verification |
| --- | --- | --- |
| `process.md` | Created this ongoing process record | Checked against git state and tool results |
| `src/streaming/planning.py` | Causal Linear/Adaptive policies, bounded updates, endpoint drain, authoritative transcript reconciliation | Offline policy regressions; real Gemma replay |
| `src/streaming/argument_revisions.py` | Speaker-owned revise/retract; archive and invalidate dependent attacks | Revision/retraction and wrong-speaker/ID tests |
| `src/debate_tree.py` | Persist stable node IDs and revision history | Tree and planner regressions |
| `src/ouragents.py` | Integrate speculative notes, opponent finalization, corrective extraction and final revision context | Integrated backend suite; real generation/feedback/revision replay |
| `src/agents.py`, `src/utils/model.py` | Planning configuration and optional generic model proxy routing | Existing defaults preserved; real Gemma upstream pilot |
| `src/utils/helper.py`, `src/utils/llm_schemas.py` | Opt-in correction extraction and target IDs | Extraction suite; tree snapshots from paid runs |
| `src/streaming/env.py` | Deliver streaming batches to the selected policy | Streaming regressions |
| `debate-app/backend/debate_app/engine_adapter.py`, `schemas.py` | API settings, streaming policy routing, checkpoint/restore planner state | Backend tests including speculative rollback |
| `src/streaming/experiment_client.py`, `src/utils/tool.py` | Durable shared pre-dispatch cost guard; do not swallow budget exceptions | Cap/restart/failure/shared-client/attribution tests |
| `src/scripts/benchmark_incremental_planning.py` | Frozen controlled text replay, usage attribution, blind checklist judging and resume | Development experiments and held-out run artifacts |
| `experiments/incremental_planning/{cases.json,manifest.json,summarize.py}` | Authored cases, authorization/settings/pricing, paired reporting and artifact audit | Counts/hashes; aggregate invariance check on development results |
| `tests/test_incremental_planning.py`, `tests/test_experiment_budget.py`, backend `tests/test_engine_adapter.py` | New meaningful regression coverage | Included in 188 tests + 42 subtests passing |
| `debate-app/configs/gemma-incremental.yml`, `debate-app/README.md`, `.gitignore` | Example opt-in configuration, usage documentation, keep large raw artifacts local | Configuration validation; diff review |

See chronological implementation entries below for changed files and verification.

## Command log

Commands below omit secrets and summarize repetitive read-only inspection.

```bash
# Repository selection and baseline inspection
git -C /home/danqingwang/workspace/clone/TreeDebater status --short
git -C /home/danqingwang/workspace/clone/TreeDebater branch --show-current
git -C /home/danqingwang/workspace/clone/TreeDebater log -3 --oneline

# Completed: create the implementation branch
git -C /home/danqingwang/workspace/clone/TreeDebater switch -c feat/incremental-rebuttal-planning
git -C /home/danqingwang/workspace/clone/TreeDebater rev-parse HEAD

# Proxy check (read-only; no model inference)
ss -lntp '( sport = :4000 )'
# Python urllib GET: http://127.0.0.1:4000/health/liveliness
# Python urllib GET: http://127.0.0.1:4000/health/readiness
# Python urllib GET: http://127.0.0.1:4000/v1/models
```

Files inspected: `src/agents.py`, `src/ouragents.py`, `src/debate_tree.py`, `src/utils/helper.py`, `src/utils/llm_schemas.py`, `src/utils/model.py`, streaming input/configuration code, backend engine/session/worker/config/schema code, and existing extraction/adapter tests. Read the experiment-cost-guard skill. No training, generation, TTS, or paid grading was started.

## Experiment results

Latest: the third comparison completed **40/40 fresh answers**, plus **10/10 development answers**, all judged by GPT-5.6. See “Third comparison completed” below and `tree-grounded-heldout-v1_summary.json`. Grounded/Light Tree improve the original Tree Plan pipeline on this sample, while matched Linear ablations remain competitive; tree binding has unresolved extraction failures.

Historical first round: the frozen held-out comparison completed **168/168** answers. Full results and limitations are recorded at the end of this log and in `experiments/incremental_planning/heldout-v1_summary.json`. Linear reduced simulated text-ready latency and had a higher mean automated checklist score, but **stable quality improvement is not established**: quality confidence intervals include zero and spot checks found judge inconsistencies. Do not treat these small authored-case results as live speech or standard benchmark results.

## Timeline

- 2026-10-02 02:15 EDT (06:15 UTC): branch and baseline revalidated; created this log after the user's request. Next: implementation, offline regressions, budget guard, Gemma pilot, then controlled comparison.

### Implementation pass 1 (in progress)

- Added `src/streaming/planning.py`: serial causal notes, Linear / Adaptive policies, bounded speculative updates, mandatory endpoint drain, version validation, and final-transcript replacement reconciliation. No speech/TTS/evidence commit in this component.
- Added `src/streaming/argument_revisions.py`: speaker-owned exact-target revise/retract operations; archive replaced nodes and their dependent subtrees.
- Updated `src/ouragents.py`, streaming input environment and backend adapter to route opponent batches through preparation and finalize before response generation. Added planner state and turn snapshots to backend checkpoints.
- Added `DebaterConfig.planning` and web `SessionSettings.planning`; extended extraction schema/prompt with opt-in revise/retract; persist revision history in tree JSON.
- Added optional `DEBATE_LLM_API_BASE` / `DEBATE_LLM_API_KEY` routing for main and helper model calls through the local proxy.
- Added `src/streaming/experiment_client.py`: durable SQLite reservations before every request, no client retries, cap enforcement across restart and concurrent clients, raw call artifacts, separate conservative cap accounting and reported-token cost estimates.
- Added offline planner/revision regressions and budget-guard tests.
- Test command: `python -m pytest tests/test_incremental_planning.py tests/test_tree_extraction.py tests/test_streaming_regressions.py -q` — **58 passed**. Earlier attempt with the conda debate Python could not find pytest; initial system-Python collection found a Python 3.9 annotation compatibility issue, fixed using postponed annotations. These were offline failures and incurred no inference cost.
- Confirmed official US on-demand pricing: **$0.13 / 1M input tokens, $0.40 / 1M output tokens**. Source: AWS Bedrock pricing, Google table. The guard reserves at $1 / 1M in both directions with 4x request headroom and retains reservations after completion.
- Concrete pilot manifest: `experiments/incremental_planning/manifest.json`. Up to five validation requests, expected $0.00525, conservative upper $1, included in the user's $200 task authorization. Main comparison currently estimated $1–$10 actual usage; all runs share the same $200 conservative reservation ledger.
- At this entry: no paid inference launched yet. Code integration and scientific comparison remain incomplete; do not infer quality or latency improvement from offline tests.

### First upstream validation and integrated tests

- Guarded command: `PYTHONPATH=src python` with `BudgetedClient('experiments/incremental_planning/run', label='pilot-connectivity').text('Reply with exactly: READY', max_tokens=32)`.
- Result: **READY**, 38 input / 2 output tokens. Reported-usage estimate **$0.00000574**; conservative reservation **$0.03354**. This confirms an actual upstream Gemma response, not only proxy health.
- Installed pytest in the existing conda debate environment: `/home/danqingwang/anaconda3/envs/debate/bin/python -m pip install pytest`. Installation succeeded (pytest 9.1.1) after retries against an unavailable preconfigured extra index; no model charges.
- Integrated test command: `PYTHONPATH=src:debate-app/backend /home/danqingwang/anaconda3/envs/debate/bin/python -m pytest tests debate-app/backend/tests -q` — **184 passed, 42 subtests passed**.
- Added `experiments/incremental_planning/cases.json`: two development cases and twelve held-out cases; ordinary arguments, repeated information, split clauses, late qualifiers, withdrawal and reversal. Rubrics are written before seeing model answers.
- Added `src/scripts/benchmark_incremental_planning.py`: real TreeDebater extraction, planning, answer generation and one feedback/revision pass; shared fixed prior opening; 60-second answer word target; no ASR/TTS or search. Every text-model call and blind checklist judgment uses Gemma through the shared ledger.
- Matching control: disable embedding fallback equally in all replay arms (exact extracted target matching remains). This is an explicit controlled text experiment, not an unmodified full speech-stack benchmark. The runner stores source/data hashes and refuses resume after settings/source change.
- Next development run: `dev-smoke-v1`, first development case, seven modes, one repeat; roughly 70–100 calls, expected usage cost below $0.25 and conservative reservations below $15. It is covered by the user's cumulative $200 approval. No held-out results inspected yet.

### Development smoke completed; second development run launched

Command (conda debate Python; `PYTHONPATH=src HF_HUB_OFFLINE=1`):

```bash
python src/scripts/benchmark_incremental_planning.py --run-id dev-smoke-v1 --split dev --limit 1 --repeats 1
```

Seven arms completed on `dev_car_scope`. Estimated residual **text** latency (measured calls + simulated arrivals): legacy 15.84s; corrected tree 14.25s; adaptive linear 15.72s; linear 15.04s; adaptive tree 15.70s; tree plan 11.32s; end-of-turn 13.69s. Corrected-tree extraction recorded an actual revision; legacy recorded none. **One development case is not evidence of general improvement.**

The initial judge was too permissive about generic mentions of scope, so these initial checklist scores are exploratory only. Tightened the judging instruction before held-out evaluation: every named condition needs an explicit mention or clear paraphrase; a generic word such as “narrow” does not count. No training or final-test tuning has occurred.

After smoke plus connectivity: **66 calls; 158,279 input / 20,203 output tokens; estimated actual usage $0.02865747; reserved ceiling $5.564192; no unresolved calls**.

Additional changes: fix per-job usage attribution for parallel workers; block unmetered LiteLLM fallback in replay; update parent status when its last attack is withdrawn; log unmatched correction targets; add `debate-app/configs/gemma-incremental.yml`; add backend checkpoint/restore and settings tests. Focused tests: **31 passed, 7 subtests passed**.

Second development run `dev-v2`: two cases, seven modes, one repeat, two workers. Expected usage cost below $0.50; conservative reservations below $25; same cumulative $200 ledger. Commands:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 /home/danqingwang/anaconda3/envs/debate/bin/python src/scripts/benchmark_incremental_planning.py --run-id dev-v2 --split dev --repeats 1 --workers 2 --worker-index 0 > experiments/incremental_planning/run/dev-v2-worker0.log 2>&1
PYTHONPATH=src HF_HUB_OFFLINE=1 /home/danqingwang/anaconda3/envs/debate/bin/python src/scripts/benchmark_incremental_planning.py --run-id dev-v2 --split dev --repeats 1 --workers 2 --worker-index 1 > experiments/incremental_planning/run/dev-v2-worker1.log 2>&1
```

Raw responses, generated answers, tree snapshots, policy events, timings, source/data hashes and worker completion markers are under `experiments/incremental_planning/run/`. That directory is git-ignored; summaries and this process log are retained in the branch. No services were restarted or deployed by this task.

### Development diagnosis and third development check

- `dev-v2` finished **14/14** answers. Summary: `experiments/incremental_planning/dev-v2_summary.json`, generated by `python experiments/incremental_planning/summarize.py dev-v2`.
- Across the two dev cases, checklist rates were legacy 0.667, end-of-turn 0.667, linear 0.833, corrected tree 0.667, adaptive linear 0.833, tree plan 0.667, adaptive tree 0.500. These are development diagnostics, not held-out claims.
- Inspection found a specific failure: after an opponent withdrew an exam-score claim and allowed teacher-authorized phone use, the tree correctly retracted the claim, but old notes still proposed educational use as if prohibited. Updated preparation to lead with current limits/withdrawals and newest verbatim input; newest speech explicitly overrides stale notes/tree summaries. Added reminders to avoid presenting an already-permitted exception as a new alternative. Correction extraction now explicitly preserves exemptions and permissions.
- Judge consistency issue: omission of a withdrawn claim was sometimes marked wrong and sometimes right. Clarified that omission is acceptable; attacking a withdrawn premise or treating the same allowed exemption as a competing alternative is not. Held-out evaluation still has not started.
- Cumulative after `dev-v2`: **196 requests, 476,824 input / 61,578 output tokens; usage estimate $0.08661832; conservative reservations $16.666872; zero unresolved calls**.
- Launched `dev-v3` only on the withdrawal development case, six modes (excluding end-of-turn), one repeat, two workers. Expected usage below $0.25, reservations below $15, all within the same approved task cap. Commands use the previous launcher with `--run-id dev-v3 --split dev --case-id dev_school_retract --modes legacy corrected_tree linear adaptive_linear tree_plan adaptive_tree --repeats 1 --workers 2 --worker-index 0` (and worker index 1); logs under `run/dev-v3-worker{0,1}.log`.
- Added configuration/usage documentation to `debate-app/README.md`. Summary tool computes paired differences after averaging repeats within each case, with a case-level bootstrap; it reports incomplete run counts rather than silently treating them as complete.

### Frozen implementation and held-out comparison launch

- `dev-v3` completed **6/6** answers; summary `dev-v3_summary.json`. It exposed a mistyped correction target, so added stable `Node.node_id` values, persisted in snapshots/JSON, and optional extraction `purpose.target_id`. Speaker ownership remains mandatory. An invalid ID is rejected; matching does not use an embedding guess for a destructive revision.
- Final revision previously received the draft/feedback without the original opponent's full statement. For the new corrective/early-preparation modes, it now receives that authoritative statement too, preserving late restrictions and exceptions. Legacy and end-of-turn controls retain their prior finalization prompt.
- Final pre-comparison checks: **188 passed, 42 subtests passed** with `PYTHONPATH=src:debate-app/backend /home/danqingwang/anaconda3/envs/debate/bin/python -m pytest tests debate-app/backend/tests -q`. `git diff --check` passed.
- Cumulative before main comparison: **254 calls; 623,156 input / 81,634 output tokens; usage estimate $0.11366388; conservative reservations $21.70924; no unresolved calls**.
- Frozen main run: **heldout-v1**, all 12 reserved test cases × 7 modes × 2 repeats = **168 answers**, 3 worker processes. Same Gemma model for all text components and blinded judging; main temperature 0.3, helper/judge 0; notes 700 tokens, gate 120, generation/helper maximum 1600, judge maximum 800; shared 60-second answer word target.
- Expected additional provider-usage cost **$0.50–$3.00**, conservative reservations approximately **$120–$170**. Both estimates include judging; remaining cap before dispatch is $178.29076 under the stricter retained-reservation accounting. The shared guard blocks a new call if its reservation would exceed $200, even if this leaves the comparison incomplete. Exact settings are in `manifest.json` and frozen worker metadata.
- Each supplied chunk is an analysis batch, released on a fixed 2.3-words/second simulated schedule. This is a policy replay, not a measurement with the web app's default 60-word ASR-analysis batching or an actual microphone.
- Main launch command, once per worker index 0, 1, 2:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 /home/danqingwang/anaconda3/envs/debate/bin/python src/scripts/benchmark_incremental_planning.py --run-id heldout-v1 --split test --repeats 2 --workers 3 --worker-index 0 > experiments/incremental_planning/run/heldout-v1-worker0.log 2>&1
```

Do not tune prompts on these held-out answers. Record any runtime defects and any subsequent rerun separately. The default production policy remains `legacy` pending evidence from this comparison.

### Held-out execution started

- Implementation committed as `ed5990d` (`Add incremental rebuttal policies and guarded Gemma replay`) on `feat/incremental-rebuttal-planning` before dispatch.
- Launched all three workers with the command above, substituting worker index and log suffix together: worker 0 session `18657`, worker 1 session `68881`, worker 2 session `8182`. All use the same durable `run/cost.sqlite` ledger.
- First main-run request was dispatched at **2026-10-02 06:57:17 UTC** (timestamp from the durable ledger).
- Initial progress: 3/168 answers generated and judged; all three workers advancing. Total retained reservations $24.453808 and reported-token estimate $0.12732323, including development. Three unresolved reservations at this instant correspond to the concurrently running requests; inspect their final states at completion.
- Interpretation constraint: `corrected_tree` versus `legacy` bundles explicit revision/retraction with supplying the authoritative opponent statement to final revision. It does not separately identify the contribution of those two changes. Early modes receive that final-revision context too. `tree_plan` versus `corrected_tree` and `adaptive_tree` versus `tree_plan` are the closer scheduling comparisons.
- Reporting-only update during execution: added paired component comparisons and raw artifact/error/truncation/warning audits to `experiments/incremental_planning/summarize.py`. No inference code, prompts, cases or run settings changed. Recomputed `dev-v3` and checked every pre-existing aggregate metric and bootstrap interval was unchanged. Its three truncated responses (request IDs 203, 222, 224) were preparation notes capped at 700 tokens, not final answers. Main-run truncations will also be disclosed; no output cap is raised mid-run.
- Mid-run checkpoint: **88/168 generated and judged**, 1,057 total requests including development, $0.47761392 reported-token estimate, $91.53068 retained reservations; 1,054 completed calls and three active requests, no errors. The three worker manifests were identical, with frozen source digest `abaf76db71bc8a19d817775d8d24dc20c3f1d41064dc70b11ac15e52c18ce413`. An audit of available control responses found no `revise`/`retract` actions emitted in `legacy`/`end_of_turn`; the shared schema permits those actions but those modes do not enable them.
- Timing audit: the text-ready measurement occurs at final `post_process`, before the tree methods analyze their own delivered answer. The runner already saves that full method duration too. Added a separate `residual_worker_return_mean_s` aggregate from the saved durations, including own-answer analysis; neither number includes TTS/playback. This reporting change requires no new inference and does not change the primary metric.

### Final held-out results — 2026-10-02

**Completed 168/168 answers and judgments**, covering 12 held-out authored cases, seven modes, two repeats. All three worker sessions exited with code 0, and all three completion markers exist. Main-run dispatch-to-last-response interval: **06:57:17–07:19:55 UTC**, approximately **22 minutes 38 seconds**. No paid request failed, no run restart or result-selection retry was needed, and no source/prompt/case settings changed during the comparison.

Reproduce the report without further model calls:

```bash
cd /home/danqingwang/workspace/clone/TreeDebater
python experiments/incremental_planning/summarize.py heldout-v1
```

Each row has 24 answers and 72 checklist judgments. Scores below are **Gemma judgments**, not ground-truth accuracy. Timing columns reconstruct the same serial input schedule from measured calls; neither measures audible response onset. Generation cost includes preparation, gating, final generation/feedback/revision and own-answer analysis where applicable, and excludes the separate checklist judge.

| Mode | Checklist pass | Strength / 5 | Strawman flags | Text ready, mean s | Worker return, mean s | Generation calls / answer | Generation USD / answer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy | 83.33% | 3.58 | 20.83% | 16.79 | 19.46 | 8.00 | 0.003925 |
| End-of-turn | 83.33% | 3.58 | 16.67% | 15.94 | 18.38 | 6.00 | 0.003160 |
| Linear | 88.89% | 3.92 | 12.50% | 14.25 | 14.25 | 6.00 | 0.003327 |
| Corrected tree | 81.94% | 3.67 | 25.00% | 14.40 | 16.98 | 8.00 | 0.004044 |
| Adaptive linear | 86.11% | 3.88 | 16.67% | 16.44 | 16.44 | 7.67 | 0.003382 |
| Tree early planning | 86.11% | 3.96 | 12.50% | 15.07 | 17.36 | 10.00 | 0.005235 |
| Adaptive tree | 88.89% | 3.96 | 4.17% | 15.41 | 18.04 | 11.33 | 0.005036 |

Answers averaged 129–132 words across modes. Every mode received zero unsupported-fact flags, which **does not establish factual correctness**; the spot checks below show why the evaluator should not be trusted as a fact checker.

Paired comparisons first average the two repetitions within each case, then bootstrap the 12 paired case differences with 10,000 samples. Differences are candidate minus baseline; negative latency is faster. These are descriptive, unadjusted 95% intervals on this small case set, not evidence of generalization or independent validation of the judge.

| Comparison | Checklist difference, percentage points [95% CI] | Text-ready difference, seconds [95% CI] |
| --- | ---: | ---: |
| Linear − legacy | +5.56 [−8.33, +16.67] | −2.54 [−5.73, −0.01] |
| Corrected tree − legacy | −1.39 [−15.28, +9.72] | −2.38 [−4.80, −0.51] |
| Adaptive linear − legacy | +2.78 [−12.50, +16.67] | −0.35 [−4.43, +3.09] |
| Tree early planning − legacy | +2.78 [−8.33, +12.50] | −1.72 [−4.99, +1.04] |
| Adaptive tree − legacy | +5.56 [−4.17, +15.28] | −1.37 [−4.56, +1.18] |
| Adaptive linear − linear | −2.78 [−11.11, +5.56] | +2.19 [+0.30, +4.18] |
| Tree early planning − corrected tree | +4.17 [−4.17, +12.50] | +0.67 [−1.22, +2.84] |
| Adaptive tree − tree early planning | +2.78 [−4.17, +11.11] | +0.34 [−1.02, +1.75] |

Interpretation:

- **Linear is a useful, inexpensive baseline in this replay.** Relative to legacy it uses 25% fewer generation calls, about 15.2% lower generation cost, and about 15.1% lower mean simulated text-ready latency. Its checklist mean improves, but its quality interval spans zero. The latency estimate is sensitive to case variation; one late-negation case contributes a large advantage.
- **Explicit correction works as a state operation, but that alone did not improve aggregate judged quality.** Opponent-tree history records 13 corrections in corrected-tree, 14 in tree-plan, and 12 in adaptive-tree. These counts are not correction recall rates. A real homework example successfully retracts the unsupported doubling claim, while one later correction still fails target matching.
- **Adaptive did not yield an overall efficiency win here.** Each adaptive variant chose WAIT eight times across 48 gate decisions. The extra gate calls and mandatory endpoint drain offset saved preparation. Adaptive linear is 2.19 seconds slower than Linear on average; adaptive tree uses more calls than tree-plan despite slightly lower token cost. This short three-batch design does not demonstrate that adaptive scheduling is useless on longer speech.
- **Tree early planning has not been shown to improve over the correction-only system.** Its quality difference is uncertain, and its mean residual text latency is slightly higher. Serial extraction plus preparation can consume the available listening interval.
- Keep the production default at `legacy`. The new policies are opt-in and are application-level baselines inspired by incremental listening/planning work; they do not reproduce the papers' training or KV-cache mechanisms. No model weights were trained or changed.

### Final audit and concrete limitations

- All **1,536 main-run requests** have usage records and local request/response artifacts; **zero missing artifacts, zero error/usage-missing/pending records** at completion. The cumulative ledger contains 1,790 successful requests including development and pilot. The three worker manifests match and the inference source and case hashes still match the frozen run.
- Four main-run calls reached the 700-token preparation-note cap: IDs **325, 775, 811, 1261**, all on `test_homework_evidence`. No final answer or judge response was truncated. All four retained their original outputs and charges.
- Logs contain **42 unmatched reinforce warnings**, **one unmatched revise warning**, and **two rejected ungrounded/motion-only extractions**. Mode-level counts are in `request_audit.runtime_warning_counts`. No embedding fallback was enabled to hide these issues.
- No `revise` or `retract` action was emitted in the complete legacy/end-of-turn control responses, despite the shared response schema allowing those tokens. Those modes do not apply correction actions.
- Deterministic trace spot-check selections: repeat 0 of `test_transit_scope` with legacy/tree-plan, `test_energy_reversal` with legacy/linear, `test_homework_evidence` with corrected-tree, and `test_parks_repetition` with adaptive-tree. These are inspection examples, not an independent human evaluation.
- **Judge inconsistency:** the transit legacy answer passed the full means-tested one-year-bus-pilot checklist item based only on mentioning “one-year pilot”; its corresponding tree-plan reason incorrectly claimed the answer lacked “one-year” even though the phrase appears in the answer. The energy Linear answer passed the existing-versus-new-reactor distinction without explicitly acknowledging that existing plants remain open. These judgments were preserved, not selectively corrected.
- **Unsupported factual assertions:** the parks answer asserted that even a short trial causes immediate, permanent business damage, yet received no unsupported-fact flag. Zero flags across the run therefore cannot be reported as zero hallucinations.
- The data are 12 authored English scenarios, not a public standard debate benchmark, and only two output repetitions per case were run. Generation and grading use the same Gemma model. The replay omits live ASR, default web analysis batching, TTS, playback, search, and semantic target matching. Three workers share one proxy, so call-time variation includes shared-service conditions. Stable quality improvement and actual first-audio latency improvement remain unverified.

### Final cost ledger

Rates used: standard US Bedrock Gemma input **$0.13/M**, output **$0.40/M**. Values are estimates from provider-reported token usage, **not a settled AWS invoice**. All development, grading, and pilot usage is included; no new paid cloud compute was provisioned.

| Run | Requests | Input tokens | Output tokens | Usage estimate USD | Retained conservative reservation USD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Connectivity pilot | 1 | 38 | 2 | 0.00000574 | 0.033540 |
| dev-smoke-v1 | 65 | 158,241 | 20,201 | 0.02865173 | 5.530652 |
| dev-v2 | 130 | 318,545 | 41,375 | 0.05796085 | 11.102680 |
| dev-v3 | 58 | 146,332 | 20,056 | 0.02704556 | 5.042368 |
| heldout-v1, including judging | 1,536 | 3,854,771 | 490,883 | 0.69747343 | 133.256708 |
| **Total** | **1,790** | **4,477,927** | **572,517** | **0.81113731** | **154.965948** |

Approved cumulative cap remains **$200**. The retained reservation is a deliberately conservative upper bound used by the dispatch guard, **not money billed**. All experimental workers have exited; there are no pending requests, automatic retries or remaining experiment jobs. The shared AWS proxy was left running and unchanged.

### Completion evidence

- Implementation branch: `feat/incremental-rebuttal-planning`; frozen implementation commit: `ed5990d`; baseline: `065cc3c897c2e7d337391e465fbe2fb9a7828bac`.
- Required Linear/correction/Adaptive/tree preparation paths implemented and connected to the streaming backend; configuration and checkpoint regressions included.
- Final inference-code verification before freezing: **188 tests passed, 42 subtests passed**. Reporting changes preserve the prior metrics and use already saved artifacts; no additional paid inference is necessary to regenerate summaries.
- Complete Gemma comparison: `experiments/incremental_planning/heldout-v1_summary.json`. All important settings, commands, outcomes, failures, limitations and costs are recorded here; raw artifacts remain in the git-ignored `experiments/incremental_planning/run/` directory.
- No deployment or push was performed. Pre-existing untracked user files were preserved.
- Final read-only audit passed: 168 unique results, three worker completion markers, zero unresolved calls, complete call artifacts, unchanged source/data hashes, and retained reservations below $200. Regenerating both `dev-v3` and `heldout-v1` with the final report script preserved every existing metric and paired interval; only explicitly added audit/secondary timing fields changed.

Final repository commands:

```bash
git diff --check
git add process.md experiments/incremental_planning/summarize.py experiments/incremental_planning/manifest.json experiments/incremental_planning/dev-v3_summary.json experiments/incremental_planning/heldout-v1_summary.json
git commit -m "Record frozen Gemma comparison results and cost audit"
git status --short
```

## Follow-up diagnosis and proposed improvements — 2026-10-02

Read-only inspection for the user's question about how to improve: compared repeat-0 answers, notes, judgments and timings for `test_plastics_split` and `test_water_negation` across legacy, Linear, corrected-tree and adaptive-tree, and re-read the planner update/gate/finalize paths. No new inference calls or implementation changes were made for this diagnosis.

- In the plastics Linear trace, notes correctly preserve the essential-use exemptions and two-year phase-in, but turn the opponent's promised exemption review into a speculative "continuous cycle" and "constant regulatory uncertainty". The final answer then asserts frequent rule changes as fact. This is an inference becoming an unsupported premise, not simply failure to remember the last chunk. The final answer also loses the explicit implementation period.
- The initial chunk ends with `except`; its qualification arrives in the next chunk. A semantic-completeness buffer is worth testing before generating substantive rebuttal plans for such unfinished clauses.
- The water corrected-tree answer explicitly mentions household-size adjustments and medical exemptions, yet the judge fails that recognition check because the answer criticizes them. Recognition and agreement must be distinguished in evaluation. This further limits the reliability of aggregate quality rankings.

Proposed next changes, **not yet implemented or experimentally verified**:

1. Start from Linear and replace free-form opponent-state summaries with compact structured current claims, scope, exceptions, withdrawals and verbatim source spans. Store our hypotheses separately; do not promote inferred review frequency or potential harms into opponent facts.
2. Make each rebuttal identify its current target and required assumptions. When the opponent already grants an exception, explicitly acknowledge it and critique a remaining issue such as eligibility, appeal or implementation; discard attacks requiring the exception to be absent. Use the existing feedback/revision pass for this targeted grounding check before adding another model call. Preserve relevant qualifications when shortening the final answer.
3. Separate confidently duplicated input, unfinished clauses and substantive changes in the scheduler. Skip only verifiable duplicates; buffer unfinished clauses; process corrections and uncertain pending material at the endpoint. Invoke a model gate only for ambiguous cases. Test this against always-update Linear under the same total budget.
4. Evaluate these changes individually, then in combination, with fresh held-out cases. The old 12 cases are now diagnostics and must not be presented as untouched test data after tuning on these observations. Use independent/human adjudication for scope preservation, withdrawn-target attacks and unsupported factual premises; separately measure live endpoint-to-first-audio latency.

Tree-driven planning and model training are lower priorities until these grounding and evaluation issues are addressed. This is a proposed experimental order, not a claim that structured state or a cheaper gate has already improved performance.

## Follow-up implementation authorized — 2026-10-02

The user instructed **进行**. Continue on `feat/incremental-rebuttal-planning` from `5a03348`, using the same approved **cumulative USD 200** cap and the original durable ledger. Prior usage estimate is $0.81113731; retained conservative reservations are $154.965948, leaving $45.034052 of reservation headroom. Do not reset or release the historical reservations.

Plan: compare `linear` → `structured_linear` → `grounded_linear` → `light_linear`. New policies respectively add source-anchored JSON state, targeted feedback/revision in the existing calls, and selective scheduling. All use Gemma. An independent **Nemotron 3 Super 120B** model will judge delivered answers without seeing policy labels or planning traces. Official AWS US standard rates rechecked on 2026-10-02: Gemma $0.13/$0.40 per million input/output tokens; Nemotron $0.15/$0.65. Both remain below the guard's $1/M per-direction rate before its 4x margin.

Concrete configurations and estimates are in `experiments/incremental_planning/manifest_v2.json`: up to three judge-calibration calls, two reused development cases, then 12 fresh cases × four modes × one repeat = 48 answers with two workers. Expected additional provider usage below $2; conservative reservations estimated $32–$42.50 including a $1 allowance for bounded audio validation. All dispatches remain subject to the unchanged shared $200 ceiling, including failures and retries. No additional approval is needed under the user's explicit task-wide authorization.

Implemented so far:

- `src/streaming/grounding.py`: compact current claims/limits/rebuttals; validate verbatim source spans and current target indices; distinguish assumptions from opponent facts; incomplete-clause and near-repetition heuristics.
- `src/streaming/planning.py`: three opt-in modes, checkpointable structured state, safe raw-prefix fallback for invalid output, exact adjacent-duplicate skipping, bounded incomplete-clause buffering, and model gates only for ambiguous near-repetition. Corrections/numeric changes bypass the gate; uncertain pending material still drains at the endpoint.
- `src/ouragents.py`: grounded modes replace generic audience feedback with targeted source/assumption checks, and preserve those checks during final revision. No extra model call added.
- `src/streaming/experiment_client.py`: independently priced judge support, rejecting models without verified prices. Existing durable cap mechanism and old reservations unchanged.
- Replay CLI accepts a separate case file and judge model. `cases_v2.json` contains two v1 diagnostics relabeled as development and twelve newly authored held-out cases, written before any new answers were sampled.
- Targeted verification: **34 passed, 4 subtests passed** using the conda debate Python with `PYTHONPATH=src:debate-app/backend` and pytest on `test_grounded_planning.py`, `test_incremental_planning.py`, `test_experiment_budget.py`, and backend `test_engine_adapter.py`. Full-suite validation remains pending at this entry.

### Follow-up development and evaluator correction

- Nemotron passed two simple calibration controls, saved in `judge_calibration_v2.json`. `grounded-dev-v1` then completed 8/8 answers on the two reused diagnostics, with zero invalid structured states. It used 55 calls, 141,864 input and 17,271 output tokens: estimated $0.02602880, retained reservations $5.059860. Summary: `grounded-dev-v1_summary.json`.
- Development checklist means were Linear .833, structured .333, grounded .500, light .500. These scores are unreliable: Nemotron failed the phase-in check for an answer explicitly saying “regarding the two-year phase-in,” while accepting implicit mentions elsewhere. Simple calibration did not establish general evaluator reliability. All original scores remain saved.
- Before any fresh held-out run, changed the independent judge to **GPT-5.6 Sol** via the existing `bedrock/us.openai.gpt-5.6-sol` route. Official geographic-inference short-context rates are **$4.40/M input, $22/M output**, including the regional premium: <https://docs.aws.amazon.com/en_en/bedrock/latest/userguide/model-card-openai-gpt-56-sol.html>. Its output cap is 800, reasoning effort `none`; no temperature field is sent because this Bedrock route rejects it. Two HTTP 400 calibration/parameter-check failures were retained in the ledger with their full reservations; no free retry assumption was made.
- Corrected GPT calibration passed both a critical-but-acknowledging answer and a fabricated blanket-ban answer; saved in `gpt_judge_calibration_v2.json`. This is a basic sanity check, not proof of evaluator accuracy. Generation remains Gemma throughout.
- The budget guard now uses **4 × ((request bytes + 8192) × max($1/M, model input rate) + max output tokens × max($1/M, model output rate))**. This preserves the previous reservation exactly for Gemma/Nemotron and raises it appropriately for GPT. Historical reservations and the $200 cap remain unchanged. A regression proves the expensive judge is rejected before dispatch when its own bound does not fit.
- Main follow-up configuration revised before seeing fresh outputs: **first ten prewritten test cases × four modes × one repeat = 40 answers**, two workers, GPT judge. The final two cases remain unused. This smaller run accommodates the higher judge reservations; it is not a selection based on results. Main remaining reservations estimated $28–$34, plus bounded final development/audio checks; dispatch still stops before $200.
- Based only on development traces, structured notes now prioritize current timing and coverage before speculative responses. Grounded modes use a concise final-revision prompt prioritizing faithful targets, conditional reasoning and relevant concessions, replacing the generic rhetorical revision prompt. The existing feedback and revision call count remains unchanged; integration tests check that routing.
- `grounded-dev-v2` runs the two final grounded modes on the single reused plastics diagnostic before freezing. Command: `PYTHONPATH=src HF_HUB_OFFLINE=1 /home/danqingwang/anaconda3/envs/debate/bin/python src/scripts/benchmark_incremental_planning.py --run-id grounded-dev-v2 --split dev --cases-file experiments/incremental_planning/cases_v2.json --modes grounded_linear light_linear --judge-model gpt-5.6-sol --limit 1 --repeats 1`, with output redirected to `run/grounded-dev-v2-worker0.log`.
- Audio probe prepared in `experiments/incremental_planning/audio_probe.py`: one recorded synthetic speech, actual shared Whisper transcriptions delivered on a real clock, two serial engine workers (Linear/light), and actual backend streaming TTS callbacks. It measures **server first playable audio chunk**, not browser playback/microphone transport. TTS rewrites are disabled identically, preserving Gemma for all debate text. A $1 external-audio bundle is reserved in the same ledger before any audio request, with durable 4x per-request bounds inside the bundle, a 24-request ceiling, provider-error latching, and unknown endpoints blocked. Text-model calls still reserve independently. Audio rates verified from official OpenAI Docs: [tts-1 $15/M characters](https://developers.openai.com/api/docs/models/tts-1), [whisper-1 $0.006/minute](https://developers.openai.com/api/docs/models/whisper-1). Existing audio models are preserved for comparability.

### Follow-up freeze and main launch

- Final development run `grounded-dev-v2` completed 2/2. Grounded: checklist 2/3, text-ready 14.97s, six generation calls; light: checklist 3/3, 7.97s, five calls. Both had zero strawman flags. One reused case is not evidence of general improvement. Total including GPT judgments: 13 calls, 29,713 input / 4,214 output tokens, usage estimate $0.03606839 and reservations $1.562500.
- Full suite after new modes, grounded-call integration and audio budget tests: **210 passed, 42 subtests passed**, with one existing Pydantic deprecation warning. A subsequently added short-context price-bound test also passes; GPT requests cannot silently exceed the verified 272K pricing tier. `git diff --check` passed.
- Freeze all inference prompts and code for `grounded-heldout-v1`. No further development tuning on these fresh outputs. First ten cases in `cases_v2.json`, four variants, one repeat, two workers, independent GPT judge. Exact pre-main cumulative costs are recorded in `manifest_v2.json.before_main`.
- Main command below is run once for worker 0 and once with both suffixes changed to 1:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 /home/danqingwang/anaconda3/envs/debate/bin/python src/scripts/benchmark_incremental_planning.py --run-id grounded-heldout-v1 --split test --cases-file experiments/incremental_planning/cases_v2.json --modes linear structured_linear grounded_linear light_linear --judge-model gpt-5.6-sol --limit 10 --repeats 1 --workers 2 --worker-index 0 > experiments/incremental_planning/run/grounded-heldout-v1-worker0.log 2>&1
```

- Frozen implementation committed as `6d44b01`. Worker 0 initially ran in session `61486`, worker 1 in `37761`.
- Runtime event: request **1895**, GPT judging `fresh_repair/linear/0`, returned an empty completion at its 800-token limit. Worker 0 exited with code 1 after preserving the generated answer; worker 1 continued. Recovered the response's 708 input / 800 output tokens into the ledger ($0.02071520 estimate), retaining its error state and full reservation. Added an audit marker and truncation flag to the original call artifact without altering its request or response. This repairs cost accounting, not the experiment output.
- Restarted worker 0 with the identical frozen command and `>>` log append. Resume reuses already-generated answers and completed judgments; only the missing judge call is retried. No prompt, output cap, source code, or data changed.

- Subsequent independent judge calls 1962 (`fresh_lighting/linear/0`) and 1982 (`fresh_evening_access/structured_linear/0`) also returned empty content at 800 tokens. Both worker processes exited after saving their answers. Reconciled each response's 704 input / 800 output tokens ($0.02069760 each), preserving errors/reservations. Cumulative retained reservations before restarting were $177.19833040; 17 answers and 15 judgments were saved. Resume each worker once with identical configuration, allowing one retry per missing judgment; no answer is regenerated or selected based on score.

### Judge truncation recovery protocol (generation remains frozen)

Request 1984, the same-setting retry for `fresh_lighting/linear/0`, returned truncated JSON. To avoid repeated insufficient output caps, `experiments/incremental_planning/resume_judge.py` retains successful 800-token judgments and permits one 1600-token recovery for a missing judgment after an error/truncation. Model, prompt, reasoning setting and answer stay identical. Future primary calls remain 800 tokens. The larger allowance is recorded on recovered judgments, every request remains metered, and a second recovery is refused. This is a documented evaluation-protocol amendment after observing infrastructure failures, not a claim of a completely homogeneous frozen judge configuration. Source inference and case hashes remain unchanged.

Recovery command replaces `src/scripts/benchmark_incremental_planning.py` with `experiments/incremental_planning/resume_judge.py`, keeping every CLI argument and append-only worker log the same. Worker 1 resumes first; worker 0 continues its existing primary run unless it fails.

Full local suite now passed **211 tests and 42 subtests** (`PYTHONPATH=src:debate-app/backend HF_HUB_OFFLINE=1 .../python -m pytest tests debate-app/backend/tests -q`), with the same pre-existing Pydantic warning.

- Before any audio dispatch, reduce the probe's external-audio bundle from $1 to **$0.50**, retaining the identical 4x per-request bounds, durable pre-dispatch checks, 24-request ceiling and error latch. This reduces allowed audio work, not the guard's safety margin. Three ≤120-second ASR chunks reserve $0.144; synthetic input and two roughly 60-second outputs are expected to fit the rest. Text inference remains separately metered. Added a regression showing a smaller bundle rejects the second request before dispatch when its bound cannot fit. No historical reservation is reduced or released.

### Follow-up final results and interpretation

All **40/40** unique held-out answers and judgments completed on ten fresh authored cases, four policies, one repeat. Both workers exited successfully. Generator: Gemma 4 26B A4B; independent judge: GPT-5.6 Sol. Frozen inference commit: `6d44b01`. Before any post-run code change, verified source SHA-256 `2719bc227a384624508ca5d491ffbbd0d12960cce83c811853d23f642ce8f1f0` and case SHA-256 `9503696fc7503efbbf62ef816e7a8a05ab8fbb67db58e83fb18ebde907c9a424` against the launch metadata. The final two authored cases remain unused.

| Policy | Checklist pass | Strength / 5 | Strawman flags | Unsupported-fact flags | Mean endpoint-to-text (s) | Mean generation calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear | 76.7% | 3.0 | 20% | 100% | 15.09 | 6.1 |
| Structured Linear | 60.0% | 3.1 | 30% | 90% | 12.59 | 6.1 |
| Grounded Linear | 70.0% | 3.3 | 10% | 0% | 9.46 | 6.1 |
| Light Linear | 80.0% | 3.7 | 10% | 40% | 9.09 | 5.8 |

All percentages above are **automatic judge outputs**, not verified real-world truth/error rates. A source quotation being present does not establish that the model's interpretation follows from it. Zero unsupported-fact flags for grounded answers does not establish factual correctness. Average answer lengths were 129.5, 131.6, 110.9 and 112.8 words respectively; shortening/revision differences can contribute to timing and quality differences.

Paired bootstrap intervals resample the ten cases (10,000 samples):

- Light versus Linear: checklist **+3.33 percentage points**, 95% interval **[-10.0, +16.7]**; simulated endpoint-to-text **-6.00 seconds**, interval **[-8.23, -3.80]**. Quality improvement is not established.
- Structured versus Linear: checklist **-16.7 points**, interval **[-30.0, -3.33]**; text latency **-2.50 seconds**, interval **[-4.99, -0.07]**. Structured state alone performed worse on this small set.
- Grounded versus structured: checklist **+10.0 points**, interval **[-6.67, +30.0]**; latency **-3.13 seconds**, interval **[-5.20, -0.72]**.
- Light versus grounded: checklist **+10.0 points**, interval **[-6.67, +26.7]**; latency **-0.37 seconds**, interval **[-2.27, +1.57]**. The scheduler's isolated benefit remains uncertain; the larger overall speed difference cannot all be attributed to scheduling.

The light scheduler skipped two exact duplicates and buffered one incomplete clause. Structured-state validation rejected **eight updates** (structured 2, grounded 2, light 4): three source quotes absent from the heard prefix, five invalid limits. Each safely used the raw heard prefix. These are real generation failures and remain counted; no invalid update was silently accepted or regenerated. Their presence is another reason to keep the new policies opt-in.

Judge audit: three empty-completion errors with recovered provider usage, five truncated requests total, no missing request artifacts. The same-cap retries for repair/Linear and evening-access/structured succeeded. Lighting/Linear and bikes/Linear required the single permitted 1600-token recovery; the other 38 final judgments used 800 tokens. Completed judgments were never replaced. The larger recovery cap is a material evaluation limitation, especially because both recovered answers are Linear. The summary's historical `unresolved_calls=3` means three retained non-OK request records, **not pending work**; actual pending count is zero. Spot checks of repair, lighting and evening-access judgments confirmed concrete quotation-based explanations, but no human adjudication or evaluator-accuracy claim is made.

Decision: retain **Light Linear as an opt-in candidate** for further validation. Do not promote structured-only mode or change the default policy based on this run. Grounded feedback appears useful for reducing unsupported assertions in these judgments, while stable quality gains and the scheduler's isolated gains remain unproven. No further paid tuning on the now-observed held-out set was performed.

### Actual audio probe

Command:

```bash
PYTHONPATH=src:debate-app/backend HF_HUB_OFFLINE=1 /home/danqingwang/anaconda3/envs/debate/bin/python experiments/incremental_planning/audio_probe.py > experiments/incremental_planning/run/grounded-audio-v1.log 2>&1
```

`grounded-audio-v1` completed successfully on the reused plastics diagnostic. Three synthesized MP3 segments total **22.896 seconds**. Real Whisper transcriptions were shared by both policies, released on the recording's real clock; final ASR became available **1.734 seconds after the audio endpoint**. Both backend workers generated real, decodable TTS chunks:

| Policy | Endpoint to first playable server chunk | Emitted chunks |
| --- | ---: | ---: |
| Linear | 16.050 s | 6 |
| Light Linear | 10.968 s | 5 |

The observed difference is **5.081 seconds** on one trial. This is a server callback measurement including ASR completion, remaining planning, answer generation and initial TTS. It excludes browser playback, microphone transport and network delivery to a client. It is not a statistically established production latency improvement. Both policies used the same 60-second budget, six-second initial / fifteen-second later chunk targets, no TTS model-based rewriting, unit speed and one parallel TTS job per pipeline. Audio-duration targets were not all met exactly; emitted output durations were about 58 and 46 seconds. Raw audio, transcriptions, event timestamps and configuration are saved in `run/grounded-audio-v1`; the tracked compact report is `grounded-audio-v1_summary.json`.

### Final cumulative cost and completion audit

The main follow-up used **286 text requests**, 704,326 input / 99,208 output tokens, estimated **$0.76377664**, retained reservations **$35.62105360**. This includes failed and repeated judging. The audio probe used **11 text requests plus one external-audio bundle**, estimated **$0.04616671**, retained reservations **$1.54323200**. Detailed external HTTP counts and sub-budget bounds are in its summary.

Across all prior and follow-up work: **2,162 ledger entries**, 5,386,770 input / 698,246 output text tokens, estimated attributable usage **$1.70095090**, retained conservative reservations **$199.86095600**, approved cap **$200**. Audio estimates are included in dollars but not text-token totals. The deliberately conservative reservation is not a billed charge. Two earlier HTTP400 requests lack provider usage; their full reservations remain retained, so the usage estimate does not assume those requests were free. `cost_audit_v2.json` records every run's totals. No budget increase, reservation release or ledger reset occurred.

All three final experiment processes exited with code zero, pending requests are zero, and no experiment retry/queue remains. The shared proxy was left unchanged. The small remaining reservation headroom is not authorization to reset historical spending.

After **all** inference and audio work ended, fixed the accounting defect exposed by empty completions: `BudgetedClient` now retains provider token usage and the truncation marker even when the completion content is empty. It still marks the request as an error and retains its original reservation. A regression reproduces the 708-input / 800-output failure and checks its $0.02071520 estimate without dispatching a real request. This post-run bookkeeping fix changes the current source hash but does not alter the frozen experiment's prompts or results.

Reproduction/reporting commands:

```bash
python experiments/incremental_planning/summarize.py grounded-heldout-v1
PYTHONPATH=src:debate-app/backend HF_HUB_OFFLINE=1 /home/danqingwang/anaconda3/envs/debate/bin/python -m pytest tests debate-app/backend/tests -q
git diff --check
```

Final verification: **213 tests passed, 42 subtests passed**, one existing Pydantic deprecation warning; `git diff --check` passed. All **2,162** ledger entries have artifacts. Audio bundle contains **20** external HTTP requests (17 TTS, 3 ASR), all successful; emitted durations were 57.991s for Linear and 46.240s for Light. No pending budget entries remain. No push or deployment was performed, and pre-existing untracked user files were preserved.

Final commit command:

```bash
git add process.md src/streaming/experiment_client.py tests/test_experiment_budget.py tests/test_audio_probe_budget.py experiments/incremental_planning/audio_probe.py experiments/incremental_planning/manifest_v2.json experiments/incremental_planning/summarize.py experiments/incremental_planning/resume_judge.py experiments/incremental_planning/cost_audit_v2.json experiments/incremental_planning/grounded-heldout-v1_summary.json experiments/incremental_planning/grounded-audio-v1_summary.json
git commit -m "Record grounded Linear comparison and bounded audio validation"
git status --short
```

## Tree-centered follow-up — prepared 2026-10-02 (America/New_York)

User instruction: **进行基于树的尝试** and maintain a single large **方案 / 特点 / 主要取舍** table covering every attempted variant. The consolidated table near the beginning of this file now lists all twelve policies, preserves each round's evaluator/sample context, and distinguishes implementation/offline tests from completed paid evaluations. Argument trees remain the main method; grounded/light Linear are matched ablation controls.

### New mechanisms

- **Grounded Tree** keeps both existing argument trees active. Each updated node can retain an excerpt validated against the actual speaker's transcript. Planning receives active node IDs, source spans, arguments, ancestor relationships and existing responses. Candidates prioritize unanswered branches, then attacks on our claims; this is a structural heuristic, not a claim of argument quality.
- Each proposed response selects a current node. The server validates that its quote belongs to that node and attaches the node ID and a content/ancestry/response version to the plan. Quotes from previously heard turns remain usable through node provenance. Quotes establish attribution, **not entailment or truth**.
- Revisions archive dependent branches and replace their source evidence; retractions remove active targets. Before using notes in generation, feedback or final revision, validate target versions again. Removed/changed targets invalidate the notes and fall back to the observed transcript. No speculative response is committed to the spoken tree.
- **Light Tree** adds the same adjacent-exact-duplicate skip, incomplete-clause buffer and selective near-repetition gate used by Light Linear. Pending material drains at the endpoint; tree extraction applies only the pending input. It does not promise to eliminate semantic extraction errors or guarantee less total work.
- `Node` JSON serialization now preserves source spans. Inspection also exposed that restoration inferred child speaker sides solely from alternating depth even though root-level proposals have their root speaker's side. Restoration now honors the saved `side` field; an offline regression checks root proposal and attack ownership plus provenance across round trips.
- Existing Linear grounding and old tree modes remain available. Default stays `legacy`. No deployment/push is included.

### Prepared comparison

`manifest_v3.json` and `cases_v3.json` specify five arms: **Tree Plan, Grounded Linear, Light Linear, Grounded Tree, Light Tree**. Development: two previously observed plastics/archives diagnostics × five arms = 10 answers. Held-out: eight newly authored cases × five arms = 40 answers, one repeat, two workers. Four fresh cases provide fixed preceding opponent/own speeches and test cross-turn branch withdrawal or already-answered objections; four use longer chunk sequences with partial withdrawals, duplicates and split exceptions. Cases were written before generating any third-round model output.

Every arm sees identical prior speeches. Tree arms additionally extract their structure, and those setup calls/costs are included in totals and reported separately from endpoint latency. All generation remains Gemma. **All scoring uses GPT-5.6 Sol with a uniform 1600-token cap from the first attempt**, reasoning `none`, temperature omitted. No mixed 800/1600 grading in this round. The judge sees the prior speeches as well as the current opponent turn, answer and checklist. Exact matching/no embedding fallback is shared across arms. No audio experiment is included in this batch.

Paired comparisons: Tree Plan → Grounded Tree; Grounded Tree → Light Tree; Grounded Linear ↔ Grounded Tree; Light Linear ↔ Light Tree. The last two hold grounding/scheduling families constant, but the source-linked target schema and context necessarily differ with the presence of a tree. Do not describe this as isolating an abstract graph representation from every implementation detail.

Prepared launch commands (not yet executed):

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 /home/danqingwang/anaconda3/envs/debate/bin/python src/scripts/benchmark_incremental_planning.py --run-id tree-grounded-dev-v1 --split dev --cases-file experiments/incremental_planning/cases_v3.json --modes tree_plan grounded_linear light_linear grounded_tree light_tree --judge-model gpt-5.6-sol --judge-max-tokens 1600 --cap-usd 200 --repeats 1 > experiments/incremental_planning/run/tree-grounded-dev-v1-worker0.log 2>&1
# After development verification and a source freeze, run worker 0 and worker 1:
PYTHONPATH=src HF_HUB_OFFLINE=1 /home/danqingwang/anaconda3/envs/debate/bin/python src/scripts/benchmark_incremental_planning.py --run-id tree-grounded-heldout-v1 --split test --cases-file experiments/incremental_planning/cases_v3.json --modes tree_plan grounded_linear light_linear grounded_tree light_tree --judge-model gpt-5.6-sol --judge-max-tokens 1600 --cap-usd 200 --repeats 1 --workers 2 --worker-index 0 > experiments/incremental_planning/run/tree-grounded-heldout-v1-worker0.log 2>&1
```

### Earlier budget amendment request — superseded by the re-audit below

Read-only ledger audit remains **2,162 entries, $1.70095090 estimated usage, $199.86095600 retained reservations, zero pending**, against the approved **$200 cumulative cap**. Remaining reservation headroom is **$0.139044**. A single properly reserved GPT judge call at the prepared 1600-token cap cannot fit; launching a partially generated batch would be unhelpful. No new paid request has been dispatched.

Rechecked official [AWS Gemma pricing](https://aws.amazon.com/bedrock/pricing/) ($0.13 input / $0.40 output per million) and [GPT-5.6 geographic pricing](https://docs.aws.amazon.com/en_en/bedrock/latest/userguide/model-card-openai-gpt-56-sol.html) ($4.40 / $22 per million, short context). Estimate for the 50-answer development+held-out batch, including extraction, planning, generation, existing feedback/revision and judging: **$2–$5 additional provider usage**, planning allowance **$10**; all values are USD and are not settled invoices. Retained conservative reservation allowances are **$20 development + $85 held-out + $10 diagnosed retries = $115**. The historical reservations are deliberately much larger than reported usage; none are released or reset.

Proposed new cumulative ceiling: **$320**. This allows the prepared $115 reservation allowance above the historical $199.860956, with a small margin. The shared SQLite pre-dispatch cap and 4x per-request bounds remain the stop mechanism. No automatic batch restarts; any one permitted missing-judge retry uses the same settings and retains its own reservation. Stop before any request that cannot fit. The added `--cap-usd` runner option only verifies an existing ledger cap; it cannot raise that cap.

The [experiment-cost-guard skill](/mnt/data4/danqingwang/.codex/skills/experiment-cost-guard/SKILL.md) requires: **“never increase the cap automatically”** and **“Ask for explicit approval to resume under that cap.”** The user's implementation request does not specify a higher dollar ceiling. Therefore all code/data/documentation/offline verification proceeds, while paid model evaluation awaits explicit approval of this concrete budget amendment. No change has been made to the approved cap.

### Third-round offline verification and handoff

Full local suite: **226 tests passed, 42 subtests passed**, one pre-existing Pydantic deprecation warning. New regressions exercise actual tree nodes (model calls mocked): node-specific quote attribution, wrong-speaker rejection, source/side serialization, structural target ranking, target-version invalidation after revision, removal of dependent attacks, real TreeDebater observation/planning integration, exact-duplicate skipping, split-clause buffering, final-ASR restoration, prior-turn sources, and shared prior-history/GPT-judge settings. Existing grounded feedback/revision tests now also cover both new tree modes without extra feedback calls. These prove software behavior under controlled fixtures, not generated-answer quality or a speedup.

Commands executed (no paid inference):

```bash
PYTHONPATH=src:debate-app/backend HF_HUB_OFFLINE=1 /home/danqingwang/anaconda3/envs/debate/bin/python -m pytest tests debate-app/backend/tests -q
git diff --check
git add process.md debate-app/README.md debate-app/backend/tests/test_engine_adapter.py src/debate_tree.py src/ouragents.py src/streaming/planning.py src/streaming/grounding.py src/streaming/tree_grounding.py src/streaming/argument_revisions.py src/scripts/benchmark_incremental_planning.py tests/test_grounded_tree.py tests/test_grounded_integration.py tests/test_benchmark_tree_history.py experiments/incremental_planning/cases_v3.json experiments/incremental_planning/manifest_v3.json experiments/incremental_planning/summarize.py experiments/incremental_planning/resume_judge.py
git commit -m "Add source-bound tree planning and document all experiment variants"
```

Current paid status: **not launched**, zero new ledger entries, existing cap unchanged at $200. No paid workers or retries are active. User's existing untracked files are preserved.

## Budget re-audit — 2026-10-02 (America/New_York)

User requested **重新审计** after questioning the $199.86 reservation total. The earlier request to increase the cumulative cap to $320 was caused by my overconservative accounting design: every completed request continued occupying its entire pre-dispatch bound even after its usage was known. That increase request is **withdrawn**. This audit keeps the approved cap at **$200**, does not erase previous spending, and does not launch any model experiment.

### Evidence and independently recomputed costs

Audited all **2,162 ledger entries against all 2,162 original artifacts**. Checked label/reservation identity, requested/returned model, provider token counters and their sum, stored rates where present, recorded cost, duplicate provider response IDs, and the 20 calls inside the audio bundle. No missing artifacts, duplicate response IDs, ledger/usage discrepancies or price discrepancies were found. There are **zero pending calls**.

| Component | Ledger entries | Reported/derived usage estimate USD | Historical pre-dispatch reservations USD |
| --- | ---: | ---: | ---: |
| Gemma 4 26B A4B | 2,100 | 0.96330075 | 184.28245600 |
| Nemotron 3 Super | 10 | 0.00280355 | 0.64025200 |
| GPT-5.6 Sol | 51 | 0.69427160 | 14.43824800 |
| Audio bundle: 17 TTS + 3 Whisper HTTP calls | 1 | 0.04057500 | 0.50000000 |
| **Total** | **2,162** | **1.70095090** | **199.86095600** |

Prices were rechecked against [AWS Bedrock pricing](https://aws.amazon.com/bedrock/pricing/), the [GPT-5.6 geographic model card](https://docs.aws.amazon.com/en_en/bedrock/latest/userguide/model-card-openai-gpt-56-sol.html), [OpenAI TTS-1 pricing](https://developers.openai.com/api/docs/models/tts-1), and [Whisper pricing](https://developers.openai.com/api/docs/models/whisper-1). Model rates remain Gemma $0.13/$0.40, Nemotron $0.15/$0.65, GPT $4.40/$22 per million input/output tokens; TTS $15/M characters and Whisper $0.006/minute. The existing proxy's selected model routes were checked without exposing credentials; no proxy settings were changed.

Two HTTP400 records (1848, 1849) have **unknown usage**, retaining **$0.44464640** in full. Three empty-completion errors (1895, 1962, 1982) have known usage totaling $0.06211040 but also retain their full combined **$0.85423360** reservation. The successful audio bundle retains its full **$0.50** because its accounting uses character/duration estimates rather than token-usage receipts. Unknown/failed costs are not assumed to be zero. The $1.70095090 usage figure is an estimate supported by available records, not a settled provider invoice. No billing statement was available to independently confirm upstream retries; a cost multiplier is a conservative allowance, not proof of an absolute billing bound.

### Corrected active budget accounting

Successful, nonempty text responses with matching model/rates, valid usage and consistent ledger/artifact evidence can now be settled at **4 × their verified reported cost**. This retains a margin while releasing unused worst-case headroom. Pending/error/unknown-usage requests keep their full original reservation; audio bundles also remain fully reserved. Every new request still reserves the original 4x byte/output-based bound *before* dispatch under a SQLite write transaction. Text and audio admission checks use the same active-occupancy calculation.

| Budget component | USD |
| --- | ---: |
| 2,156 successful text requests: $1.59826550 × 4 | 6.39306200 |
| All five failed requests: full original bounds | 1.29888000 |
| Audio bundle: full original bound | 0.50000000 |
| **Current conservative budget occupancy** | **8.19194200** |
| **Remaining within the unchanged $200 cap** | **191.80805800** |

The original `calls` table and all request artifacts are unchanged, verified by hashes before/after. A SQLite backup was taken before applying the audit at `run/cost-before-success-reconciliation.sqlite`. Added **2,156 append-only settlement records**, each containing its original reservation, reconciled charge, basis and artifact SHA-256; updates/deletes of settlements are blocked. Re-running reconciliation is idempotent. Historical `reserved_upper_usd` remains available as an audit total; **`accounted_exposure_usd` is the amount used for current admission**. Thus neither the original $199.86 total nor the $1.70 usage estimate is presented as the current guarded balance.

Implementation: `src/streaming/experiment_accounting.py`, client/audio admission integration, and `experiments/incremental_planning/reconcile_budget.py`. The latter performs a read-only dry run by default; `--apply` requires a clean audit with no pending work, takes the backup and appends evidence-linked settlements without modifying the cap. Full report: `experiments/incremental_planning/budget_reaudit.json`.

The prepared tree experiment allowance of $115 plus current occupancy is **$123.191942**, below $200 even before future successful requests are reconciled. `manifest_v3.json` and prepared commands now use the original $200 cap. The historical $320 proposal remains documented above as superseded; no budget increase is needed. No new paid inference was performed during this audit.

Validation: **233 tests passed, 42 subtests passed**, one existing Pydantic warning. Regression coverage includes reconciliation across restarts, preservation of pending/failed reservations, invalid or missing usage, cross-client admission while a request is pending, repeated settlement, immutable original records, tampered artifacts, old-ledger migration, audio sharing the same ledger, and unchanged cap enforcement.

Audit/verification commands:

```bash
PYTHONPATH=src /home/danqingwang/anaconda3/envs/debate/bin/python experiments/incremental_planning/reconcile_budget.py --output experiments/incremental_planning/run/budget_reaudit_dryrun.json
PYTHONPATH=src /home/danqingwang/anaconda3/envs/debate/bin/python experiments/incremental_planning/reconcile_budget.py --apply --output experiments/incremental_planning/budget_reaudit.json
PYTHONPATH=src:debate-app/backend HF_HUB_OFFLINE=1 /home/danqingwang/anaconda3/envs/debate/bin/python -m pytest tests debate-app/backend/tests -q
git diff --check
```

Re-running bulk reconciliation appended **zero** additional settlements and reproduced $8.191942 occupancy, confirming idempotence. Final hash checks again confirmed every original call row and artifact unchanged; all five error reservations and the audio bundle remain intact. `git diff --check` passed. No paid request was dispatched during the audit.

```bash
git add process.md src/streaming/experiment_client.py src/streaming/experiment_accounting.py tests/test_experiment_budget.py tests/test_experiment_reconciliation.py experiments/incremental_planning/audio_probe.py experiments/incremental_planning/reconcile_budget.py experiments/incremental_planning/budget_reaudit.json experiments/incremental_planning/manifest_v3.json
git commit -m "Reconcile verified usage without increasing the experiment budget"
```

## Tree experiment execution — 2026-10-03 UTC

User instructed **继续** after the cost re-audit. Launch the prepared 10-answer development comparison, then freeze the implementation before the 40-answer fresh comparison. Five arms: tree_plan, grounded_linear, light_linear, grounded_tree, light_tree. All judgments use GPT-5.6 Sol with 1,600 output tokens; generators and other settings remain as recorded above. Starting commit 388c386; starting verified usage estimate $1.70095090, active guarded occupancy $8.191942, no pending requests. Original cumulative USD200 cap remains unchanged. Expected additional usage $2–5, planning upper $10; pre-dispatch allowance $115 fits existing authorization. Guard runs before every request, shares one ledger across workers, and retains every failed attempt. No audio in this batch.

Development command:
```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 /home/danqingwang/anaconda3/envs/debate/bin/python src/scripts/benchmark_incremental_planning.py --run-id tree-grounded-dev-v1 --split dev --cases-file experiments/incremental_planning/cases_v3.json --modes tree_plan grounded_linear light_linear grounded_tree light_tree --judge-model gpt-5.6-sol --judge-max-tokens 1600 --cap-usd 200 --repeats 1 > experiments/incremental_planning/run/tree-grounded-dev-v1-worker0.log 2>&1
```

### Development results and final freeze

All **10/10** development answers and GPT-5.6 judgments completed at the same 1,600-token judge cap. 91 requests, 238,365 input / 28,427 output tokens, estimated additional usage **$0.17912524**, zero failed/pending requests. These are two reused diagnostics, not held-out evidence.

| Mode | Checklist | Strength | Strawman flag | Simulated residual text seconds | Generation calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tree Plan | 66.7% | 2.5 | 50% | 11.83 | 10.0 |
| Grounded Linear | 66.7% | 4.0 | 0% | 8.24 | 6.0 |
| Light Linear | 83.3% | 4.0 | 0% | 8.53 | 5.5 |
| Grounded Tree | 66.7% | 3.5 | 0% | 10.41 | 10.0 |
| Light Tree | 83.3% | 3.5 | 0% | 10.28 | 9.0 |

Development identified three valid source quotes rejected solely because Gemma labeled a timeline `phase-in` instead of `scope`. The parser now canonicalizes this specific equivalent label **after source validation**; the shared structured prompt explicitly assigns timing to `scope`. All structured arms receive the same fix. Offline replay of all 22 structured development responses accepts 19, including the three repaired records (2172, 2180, 2249); the other three ungrounded quotes remain rejected. No answer or score was rewritten or regenerated. Initial offline replay script passed the chunk list instead of its joined text and failed before any model request; corrected replay completed.

Full offline suite after this fix: **234 passed, 42 subtests passed**, one existing Pydantic warning. No further changes are planned based on fresh-case outcomes. The held-out run uses this frozen code, eight pre-authored new cases, five modes, one repetition, two workers. Four cases contain shared prior-round speeches; their setup calls count toward cost but are reported separately from live latency.

```bash
/home/danqingwang/anaconda3/envs/debate/bin/python experiments/incremental_planning/summarize.py tree-grounded-dev-v1
PYTHONPATH=src:debate-app/backend HF_HUB_OFFLINE=1 /home/danqingwang/anaconda3/envs/debate/bin/python -m pytest tests debate-app/backend/tests -q
git diff --check
# Run separately with N=0 and N=1:
PYTHONPATH=src HF_HUB_OFFLINE=1 /home/danqingwang/anaconda3/envs/debate/bin/python src/scripts/benchmark_incremental_planning.py --run-id tree-grounded-heldout-v1 --split test --cases-file experiments/incremental_planning/cases_v3.json --modes tree_plan grounded_linear light_linear grounded_tree light_tree --judge-model gpt-5.6-sol --judge-max-tokens 1600 --cap-usd 200 --repeats 1 --workers 2 --worker-index N > experiments/incremental_planning/run/tree-grounded-heldout-v1-workerN.log 2>&1
```

## Third comparison completed — tree-centered variants, 2026-10-03 UTC

**40/40 fresh answers and 10/10 development answers completed.** Generator Gemma, independent GPT-5.6 Sol judge, all 50 judgments capped at 1,600 tokens, no automatic or manual retries needed. Two held-out workers exited successfully, zero pending requests, zero missing artifacts, zero failed or truncated calls in either new run. Fresh inference source frozen at **3accc6e**, SHA-256 `2a553e68d02c45bc8f9b2ecf13932744a8f8aba8f8db905a035b3a2eebb5b92f`; cases SHA-256 `010a549169e25889a58760fa1e682974e5a712b2063c64c39f0e3466285f2b73`. Both hashes were verified after all workers exited and before the subsequent reporting-only snapshot fix.

### Fresh comparison: eight cases per mode

| 方案 | 检查项通过率 | 反驳强度 / 5 | 歪曲对手标记 | 无依据事实标记 | 模拟端点后文本等待 | 含自己的树分析的 worker 返回 | 生成调用（含历史初始化） | 每回答生成费用估算 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tree Plan | 50.0% | 3.00 | 37.5% | 87.5% | 17.28s | 20.50s | 14.00 | $0.007092 |
| Grounded Linear（消融） | 66.7% | 3.63 | 12.5% | 37.5% | 9.87s | 9.87s | 7.50 | $0.003687 |
| Light Linear（消融） | 70.8% | 3.63 | 25.0% | 37.5% | 10.52s | 10.52s | 7.13 | $0.003484 |
| Grounded Tree | 62.5% | 3.50 | 25.0% | 12.5% | 11.37s | 13.42s | 14.00 | $0.006538 |
| Light Tree | 62.5% | 3.88 | 0.0% | 25.0% | 11.96s | 14.20s | 13.25 | $0.006360 |

These are automated flags, not verified factual-error rates. Each tree mode averages one prior-context initialization request across all eight cases (two on each of four cross-turn cases); these are included in the request/cost columns. Live generation averages 13.00 / 13.00 / 12.25 calls for Tree Plan / Grounded Tree / Light Tree. Judging is excluded from per-answer generation cost and included in total run cost below. Mean answer lengths are 131.0, 104.75, 112.75, 114.38 and 108.13 words in table order: shorter outputs are a contributor to latency differences, so this is a whole-pipeline comparison rather than isolated scheduling speed. No live audio experiment was run in this round.

### Paired case comparisons

Case bootstrap: 10,000 resamples, eight cases, one generation each. Differences are candidate minus baseline; negative time means faster. These unadjusted intervals are descriptive and do not remove small-sample, judge or multiple-comparison limitations.

| Candidate − baseline | Checklist difference (95% interval), percentage points | Text latency difference (95% interval) |
| --- | ---: | ---: |
| Grounded Tree − Tree Plan | +12.5 [0.0, +29.2] | −5.91s [−7.74, −3.97] |
| Light Tree − Tree Plan | +12.5 [+4.2, +25.0] | −5.33s [−8.07, −2.64] |
| Light Tree − Grounded Tree | 0.0 [−16.7, +16.7] | +0.59s [−1.41, +2.44] |
| Grounded Tree − Grounded Linear | −4.2 [−16.7, +8.3] | +1.50s [+0.19, +3.03] |
| Light Tree − Light Linear | −8.3 [−20.8, 0.0] | +1.43s [−0.01, +2.80] |

Light Tree improves three cases over Tree Plan and ties five, giving a positive paired checklist interval in this small sample. Both grounded tree pipelines reduce simulated endpoint latency versus Tree Plan. However, **this does not establish that node binding itself is responsible**: the variants also change feedback/revision, and some answers use the raw-prefix fallback. Light scheduling saves 0.75 tree-generation calls per answer (5.4%) but has no demonstrated incremental quality or latency benefit over Grounded Tree. Tree methods do not outperform their matched Linear ablations here. Keep trees as the main research method and Linear as the ablation; do not change production defaults based on this sample.

Descriptive subgroups (four cases each): Tree Plan scores 58.3% on cross-turn cases and 41.7% on long chunks. All four grounded/light variants score 83.3% cross-turn. On long chunks, Grounded/Light Tree each score 41.7%, Grounded Linear 50.0%, Light Linear 58.3%. This suggests condition coverage during longer inputs deserves attention; these tiny subgroups are not independent validation.

### Tree diagnostics and retained failures

- 36 structured planning snapshots for Grounded Tree, with 9 rejected; 33 for Light Tree, with 7 rejected. Final usable bound state exists in 5/8 and 6/8 answers respectively. All 11 nonempty final states match their original request-time node versions; zero runtime `INVALID_TARGET` events. This validates those recorded bindings, not semantic correctness or comprehensive coverage.
- Across both tree variants, 17 planning requests were supplied an empty eligible target list. Eight rejected snapshots still invented or reused an unavailable ID; two produced an invalid rebuttal index. The remaining six rejections were source/attribution errors (Grounded Tree 1, Light Tree 5). All rejected snapshots fall back to raw heard text.
- Root-cause example, microgrid Grounded Tree: extraction request **2419** emitted `retract` then `revise` for the same node, so the revision no longer found its target. Requests **2421/2424** mislabeled opponent responses as `rebut` on our root-side claim; legacy `update_node` fallback treated them as reinforcement of our node. Speaker-ownership validation correctly refused to attach opponent source text to that node, leaving no eligible opponent targets. Planning requests **2420/2423/2426** then selected nonexistent targets and were rejected. This is an extraction/action-application limitation, not evidence that the underlying argument-tree idea is ineffective. No held-out-driven inference repair or rerun was performed.
- Exact matching produced unresolved correction warnings: Tree Plan 4, Grounded Tree 5, Light Tree 2; unmatched reinforce warnings: 6, 3, 6 respectively. One additional Light Tree extraction was skipped as ungrounded. These are warning occurrences, not independent case counts or always opponent-only events.
- Grounded/Light Linear had 6/5 rejected snapshots, including 4/3 unsupported limit labels in the tools case. The development-only `phase-in` fix did not cover every label the model might emit. Fresh results are retained; no additional permissive parsing was introduced after inspection.
- Light variants each skipped two exact duplicates and waited once for an incomplete clause; there were zero semantic-gate WAIT events. These cases mainly test cheap deterministic scheduling rather than the benefit of a learned gate.

### Judge and snapshot audit

Spot checks of shuttle, microgrid, river and pool show that several checklist failures are **omissions of one part of a compound requirement**, not direct contradictions. For example, the shuttle Light Tree answer acknowledges accessible vehicles but does not say “every run”; the pool Grounded Tree answer acknowledges unknown prices, a backup boiler and a study but does not explicitly state both withdrawal of the free-heating claim and study-before-contract sequencing. The checklist penalizes those omissions. The river Grounded/Light Tree answers focus on unresolved warning thresholds but omit monthly-versus-quarterly timing. Automatic strawman/unsupported flags also remain fallible and do not include separate flag-level rationales. All original judgments are retained.

Offline snapshot validation initially found four mismatched versions (Light Tree alerts; Grounded Tree pool, river, tools). Investigation showed the saved `before_generation` dictionaries shared mutable argument lists with the real trees: subsequent analysis of our generated speech appended to those lists before JSON persistence. Original pre-dispatch planning request artifacts are immutable and verify all 11 selected final states against their supplied targets. The results' original snapshot fields remain untouched; `tree-grounded-heldout-v1_diagnostics.json` records both checks. After all inference finished, the benchmark now deep-copies its diagnostic snapshot. A regression executes a simulated generation followed by tree mutation and verifies that the earlier snapshot stays unchanged. This changes future logging only, not the completed answers, scores, call durations or costs.

### Cost, validation and next priorities

| Component | New requests | Usage estimate USD | Active guarded occupancy attributable to run USD |
| --- | ---: | ---: | ---: |
| Tree development | 91 | 0.17912524 | 0.71650096 |
| Tree held-out | 487 | 0.83197722 | 3.32790888 |
| **New work total** | **578** | **1.01110246** | **4.04440984** |
| **All work cumulative** | **2,740 ledger entries** | **2.71205336** | **12.23635184** |

Original cumulative cap remains **$200**, leaving **$187.76364816** guarded headroom. Audit checked all 2,740 original artifacts, no issues, zero pending calls. The same five historical failed entries and $0.50 audio bundle retain full bounds; no new failures were added. Historical reservation sum $268.0831272 is an audit sum across completed requests, not current budget use. Usage remains a provider-token/rate estimate, not a settled bill. No cap increase, ledger reset, proxy change, push or deployment.

Final full suite: **235 tests passed, 42 subtests passed**, one existing Pydantic warning. Both evaluation workers exited 0. Post-run diagnosis invokes no model. Files: `tree-grounded-heldout-v1_summary.json`, `tree-grounded-heldout-v1_diagnostics.json`, `tree-grounded-dev-v1_summary.json`, `tree-dev-parser-replay.json`, `cost_audit_v3.json`, `manifest_v3.json`.

Recommended next research priorities (not launched in this batch): resolve contradictory extraction action sequences and speaker/action ownership before scheduling complexity; make relevant timing/exception coverage explicit in tree plans; then use new unseen cases and matched grounded-tree ablations to isolate the contribution of node binding. The current sample is retained as evaluation evidence, not reused as fresh validation.

```bash
PYTHONPATH=src /home/danqingwang/anaconda3/envs/debate/bin/python experiments/incremental_planning/reconcile_budget.py --output experiments/incremental_planning/cost_audit_v3.json
PYTHONPATH=src HF_HUB_OFFLINE=1 /home/danqingwang/anaconda3/envs/debate/bin/python experiments/incremental_planning/diagnose_tree_run.py tree-grounded-heldout-v1
/home/danqingwang/anaconda3/envs/debate/bin/python experiments/incremental_planning/summarize.py tree-grounded-heldout-v1
PYTHONPATH=src:debate-app/backend HF_HUB_OFFLINE=1 /home/danqingwang/anaconda3/envs/debate/bin/python -m pytest tests debate-app/backend/tests -q
git diff --check
```

Final result commit command (only task-owned files):
```bash
git add process.md src/scripts/benchmark_incremental_planning.py tests/test_benchmark_tree_history.py experiments/incremental_planning/manifest_v3.json experiments/incremental_planning/summarize.py experiments/incremental_planning/diagnose_tree_run.py experiments/incremental_planning/tree-grounded-dev-v1_summary.json experiments/incremental_planning/tree-grounded-heldout-v1_summary.json experiments/incremental_planning/tree-grounded-heldout-v1_diagnostics.json experiments/incremental_planning/cost_audit_v3.json
git commit -m "Report tree comparison and preserve diagnostic snapshots"
```

## Original Linear matched control — 2026-10-03 UTC

User requested **和linear进行对照**. The third comparison already included Grounded/Light Linear, but lacked the original free-text `linear` baseline. Add exactly eight original Linear answers on the same frozen eight cases, one repetition, two workers. Reuse completed tree/grounded/light results without rejudging or regenerating them. Same Gemma generation, GPT-5.6 Sol judge with 1,600-token cap, shared prior speeches, 60-second answer budget, helper temperature 0 and generator temperature 0.3, plan cap 700, generator cap 1,600, no ASR/TTS.

Starting source **679666e** differs from evaluated tree source 3accc6e only in deep-copying diagnostic snapshots; no inference prompt, policy, grading or timing section changed. The supplemental arm is run after viewing tree results, with no tuning; disclose temporal/provider-load confounding and do not call this new unseen validation. Manifest: `experiments/incremental_planning/manifest_linear_control.json`.

Budget skill continues under the existing USD200 authorization. Starting cumulative usage estimate $2.71205336; active guarded occupancy $12.23635184; no pending requests. Expected additional usage $0.10–$0.40, planning upper $1; pre-dispatch allowance including a bounded retry $14. Shared per-request admission guard and success settlement remain active; no spending reset or cap increase. Official AWS pricing rechecked: Gemma $0.13/$0.40 and GPT geographic short-context $4.40/$22 per million input/output tokens, unchanged. Sources: [Bedrock pricing](https://aws.amazon.com/bedrock/pricing/) and [GPT model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-sol.html).

```bash
# Run separately with N=0 and N=1:
PYTHONPATH=src HF_HUB_OFFLINE=1 /home/danqingwang/anaconda3/envs/debate/bin/python src/scripts/benchmark_incremental_planning.py --run-id tree-linear-control-v1 --split test --cases-file experiments/incremental_planning/cases_v3.json --modes linear --judge-model gpt-5.6-sol --judge-max-tokens 1600 --cap-usd 200 --repeats 1 --workers 2 --worker-index N > experiments/incremental_planning/run/tree-linear-control-v1-workerN.log 2>&1
```

### Linear control completed and compared

**8/8 original Linear answers and GPT-5.6 judgments completed**, two workers exited 0, no failures, truncations or retries. Source/cases hashes verified unchanged after completion. Supplemental source `679666e`, digest `8b86fdce3d17151c645f07334302895e4a1c897039aed62366941590331e3d44`. `compare_linear_control.py` verifies identical case IDs, repeats, generator, judge, token cap, temperatures, input schedule and concurrency; only mode lists and the documented diagnostic-only source difference are allowed. It joins the 48 results into a separate report and leaves all original results and judgments unchanged.

| Mode | Checklist pass rate | Strength / 5 | Strawman flag | Unsupported-fact flag | Simulated text wait | Generation calls | Generation usage USD/answer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Original Linear** | **58.3%** | **2.75** | **50.0%** | **100.0%** | **13.64s** | **7.50** | **0.004247** |
| Tree Plan | 50.0% | 3.00 | 37.5% | 87.5% | 17.28s | 14.00 | 0.007092 |
| Grounded Linear | 66.7% | 3.63 | 12.5% | 37.5% | 9.87s | 7.50 | 0.003687 |
| Grounded Tree | 62.5% | 3.50 | 25.0% | 12.5% | 11.37s | 14.00 | 0.006538 |
| Light Linear | 70.8% | 3.63 | 25.0% | 37.5% | 10.52s | 7.13 | 0.003484 |
| Light Tree | 62.5% | 3.88 | 0.0% | 25.0% | 11.96s | 13.25 | 0.006360 |

Original Linear averages 130.25 words. As above, the newer grounded pipelines produce shorter answers, contributing to their latency; timing is simulated residual text wait and excludes ASR/TTS. Calls include prior-history tree setup and post-answer own-tree analysis; per-answer cost excludes judges. Flags are automated judgments with small denominators, not independently established error rates.

| Candidate minus original Linear | Checklist difference, percentage points (95% case-bootstrap interval) | Text latency difference (95% interval) |
| --- | ---: | ---: |
| Tree Plan | −8.3 [−25.0, 0.0] | +3.64s [+1.22, +5.87] |
| Grounded Linear | +8.3 [0.0, +20.8] | −3.77s [−5.24, −2.25] |
| Light Linear | +12.5 [−4.2, +29.2] | −3.12s [−4.82, −1.13] |
| Grounded Tree | +4.2 [0.0, +12.5] | −2.27s [−3.75, −0.74] |
| Light Tree | +4.2 [−8.3, +16.7] | −1.68s [−3.93, +0.14] |

**Interpretation:** Grounded/Light Tree modestly exceed original Linear on checklist means, but both quality intervals touch or cross zero. Grounded Tree has lower measured residual latency in this comparison; Light Tree's latency interval crosses zero. Relative to the original Linear baseline, tree variants require about 1.77–1.87× generation calls. The matched Grounded/Light comparisons are more informative about the tree component: Grounded Tree trails Grounded Linear by 4.2 percentage points and adds 1.50s, while Light Tree trails Light Linear by 8.3 points and adds 1.43s. These matched quality intervals also include zero. Therefore **the current sample does not demonstrate an independent tree-structure advantage**. Improvements over old Tree Plan or original Linear cannot be attributed entirely to the tree; grounding feedback/revision and answer length are also changed. Keep trees as the principal method under investigation, with these Linear controls retained for subsequent experiments.

This is a supplemental matched comparison on already-used cases, with one repetition each; no new case or prompt tuning was performed. Original Linear ran in a later time block, so bootstrap intervals do not account for systematic provider-load changes. Do not describe this as a randomized six-arm simultaneous experiment or independent held-out confirmation.

Cost: **68** new requests (60 generation/planning, 8 judges), 191,497 input and 29,892 output tokens, usage estimate **$0.17517248**, active occupancy attributable to this run **$0.70068992**. Full audit of **2,808 entries/artifacts** found zero issues and zero pending requests. Cumulative usage estimate **$2.88722584**, guarded occupancy **$12.93704176**, remaining **$187.06295824** within the unchanged $200 cap. No new historical error reservations were released; no cap increase.

Validation: both workers completed; all metadata and case identities matched; frozen source/cases hashes unchanged; original tree answers/scores retained; raw request audit reports no missing/error/truncated records. Inference source is unchanged from the previously tested 235-test version; this turn adds reporting only. Reports: `tree-linear-control-v1_summary.json`, **`tree-vs-linear-v1_comparison.json`**, `cost_audit_linear_control.json`, `manifest_linear_control.json`.

```bash
/home/danqingwang/anaconda3/envs/debate/bin/python experiments/incremental_planning/summarize.py tree-linear-control-v1
/home/danqingwang/anaconda3/envs/debate/bin/python experiments/incremental_planning/compare_linear_control.py
PYTHONPATH=src /home/danqingwang/anaconda3/envs/debate/bin/python experiments/incremental_planning/reconcile_budget.py --output experiments/incremental_planning/cost_audit_linear_control.json
git diff --check
git add process.md experiments/incremental_planning/manifest_linear_control.json experiments/incremental_planning/compare_linear_control.py experiments/incremental_planning/tree-linear-control-v1_summary.json experiments/incremental_planning/tree-vs-linear-v1_comparison.json experiments/incremental_planning/cost_audit_linear_control.json
git commit -m "Compare original Linear against grounded tree variants"
```

## Tree repair and branch reasoning — 2026-10-03 UTC

User requested **修复树结构的对应问题，同时更好地利用树结构。grounded linear 本质是结构化，tree应该更具有优势**. Implement repairs and a direct test of explicit graph utility, without assuming that the richer representation must win. Existing USD200 cumulative authorization remains in force.

### Repairs and new variants

- New `src/streaming/tree_updates.py` resolves IDs against the pre-update graph and respects actual speaker ownership. A quoted replacement supersedes a simultaneous retract of the same target; a later actual withdrawal still wins. Same-text independent nodes are no longer all revised when a specific ID was given. Revisions archive dependent branches.
- Attack/rebut creates a child owned by the actual speaker when its target belongs to the other speaker. It never silently reinforces the opposite speaker's claim. Missing/invalid relationships preserve the sourced current claim as an unlinked proposal; unmatched withdrawals create no claim. Wrong-speaker corrections are rejected. Updated extraction instructions supply all active IDs/owners and require consistent relation actions. These updates apply to correction-enabled tree modes; legacy remains available for historical ablation.
- Relation labels, source excerpts and update events survive tree serialization. Binding validation still rejects fabricated or stale targets. Source attribution does not establish that an extracted paraphrase is semantically correct.
- `branch_tree` / `flat_tree` use zero-based choices over verified source-bearing nodes and candidate limits; the server copies quotes and binds IDs/versions. This removes the need for the model to reproduce long IDs or source quotations. The entire candidate-boundary ledger is retained for final drafting/feedback even if the compact plan selects fewer limits. Candidate limits are verbatim source sentences selected by temporal/scope markers, not semantic truth labels. Correction history stays separate and can be superseded by later speech.
- `branch_tree` includes ancestor paths, our latest linked objection, opponent reply, existing responses and structural response needs. Its prompt asks for a precise support/inference objection or concession plus remaining gap. `flat_tree` uses the same repairs, indexed choices, sources, boundary ledger and grounding delivery but removes explicit relation fields and ranking. Neither injects the legacy rendered tree into final speech prompts, preventing topology leakage in the flat control. Shared debate history remains available to both.
- Two reused development cases are the old microgrid/shuttle failures. Eight new cases were authored before outputs in `cases_v4.json`: solar carports, appointment portal, river gauges, refrigerated lockers, permit kiosks, harbour shuttle, temporary shade and repair van. All have fixed prior exchanges available to every arm. Four emphasize response chains; four emphasize several independent qualifications/corrections.

Offline regressions cover simultaneous/later corrections, ID-specific updates, speaker ownership, missing links, ungrounded excerpts, serialization, branch briefs, indexed choices, boundary retention, empty target sets, and absence of rendered topology in flat generation. Early offline runs caught legacy test doubles without a planner; guarded access preserves that compatibility. No paid model calls occurred during these failures.

### Planned comparison and budget

Five arms: original Linear, Grounded Linear, repaired Grounded Tree, Flat Tree, Branch Tree. Development: 2 old cases × 5 arms × 1 repetition = 10 answers, one worker. Held-out: **8 fresh cases × 5 arms × 2 repetitions = 80 answers**, two workers. Gemma generation/helper settings stay 0.3/0, plan cap 700, generation cap 1,600, answer budget 60 seconds. All GPT-5.6 Sol judgments use cap 1,600, reasoning none, temperature omitted. Exact/ID matching only; no audio, embedding, retrieval or extra paid compute. Every request including historical-tree setup, own-speech analysis, grading and failures is budgeted. Freeze after development before any new test outputs; bootstrap by case after averaging repetitions.

Starting cumulative usage estimate **$2.88722584**, active guarded occupancy **$12.93704176**, zero pending calls. Expected additional usage **$2–$4**, planning upper **$8**. Pre-dispatch allowance: dev $20 + main $120 + bounded retries $10 = **$150**, fitting current $200 cap even before new successful usage settles. Every dispatch still checks the shared active budget under SQLite transaction. Same historical failures remain reserved. Prices rechecked at [AWS Bedrock](https://aws.amazon.com/bedrock/pricing/) and [GPT geographic model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-sol.html): Gemma $0.13/$0.40 and GPT $4.40/$22 per million input/output tokens, unchanged. Manifest: `manifest_v4.json`.

```bash
PYTHONPATH=src:debate-app/backend HF_HUB_OFFLINE=1 /home/danqingwang/anaconda3/envs/debate/bin/python -m pytest tests debate-app/backend/tests -q
PYTHONPATH=src HF_HUB_OFFLINE=1 /home/danqingwang/anaconda3/envs/debate/bin/python src/scripts/benchmark_incremental_planning.py --run-id branch-dev-v1 --split dev --cases-file experiments/incremental_planning/cases_v4.json --modes linear grounded_linear grounded_tree flat_tree branch_tree --judge-model gpt-5.6-sol --judge-max-tokens 1600 --cap-usd 200 --repeats 1 > experiments/incremental_planning/run/branch-dev-v1-worker0.log 2>&1
# After development review and source freeze, run separately for N=0 and N=1:
PYTHONPATH=src HF_HUB_OFFLINE=1 /home/danqingwang/anaconda3/envs/debate/bin/python src/scripts/benchmark_incremental_planning.py --run-id branch-heldout-v1 --split test --cases-file experiments/incremental_planning/cases_v4.json --modes linear grounded_linear grounded_tree flat_tree branch_tree --judge-model gpt-5.6-sol --judge-max-tokens 1600 --cap-usd 200 --repeats 2 --workers 2 --worker-index N > experiments/incremental_planning/run/branch-heldout-v1-workerN.log 2>&1
```

Final pre-development full suite: **254 passed, 42 subtests passed**, one existing Pydantic deprecation warning. `git diff --check` passed.

### First development pass and mechanism refinement

`branch-dev-v1` completed 10/10 answers, 122 requests, 369,095 input / 31,670 output tokens, usage estimate **$0.19233635**, zero provider errors/truncations. Tree integrity audit found no wrong-speaker edges or ungrounded attached sources. The repaired Grounded Tree retained valid final bindings in both old failure cases; old deletion/ownership failures no longer occurred in these sampled runs.

| Dev arm (2 old cases) | Checklist | Text seconds | Final bound states | Rejected structured snapshots |
| --- | ---: | ---: | ---: | ---: |
| Linear | 83.3% | 12.88 | n/a | n/a |
| Grounded Linear | 83.3% | 9.45 | n/a | 2/8 |
| Grounded Tree | 83.3% | 9.65 | 2/2 | 1/8 |
| Flat Tree | 66.7% | 8.61 | 0/2 | 3/8 |
| Branch Tree | 100.0% | 11.34 | 1/2 | 6/8 |

The high Branch Tree checklist score does **not** validate its intended mechanism: strict planning validation rejected extra JSON fields, prose after JSON, or punctuation after JSON. Indexed planning now requests `BranchPlanResponse` JSON schema through the existing metered helper interface; validation remains strict. It does not silently salvage or alter the original bad records.

Inspection also found that an explicit opponent acceptance of our publication request was absent from the graph, and planning repeated the old objection. Added `concede` to the source-owned extraction schema only (legacy extraction schema unchanged). Such an edge preserves the exact acceptance and its speaker without asserting implementation is complete. Branch briefs now include other replies to the same objection, so an already-granted safeguard is visible when evaluating a sibling reply. Candidate limits also include acceptance/publication markers. Flat Tree sees the same claim/source material but no concession-edge labels or sibling structure.

New regressions check sibling concession visibility, preservation of the other speaker's claim, and typed planning output. Full suite **256 passed, 42 subtests passed**. No fresh-case model output has been generated. Add a six-answer `branch-dev-v2` check of Grounded Tree, Flat Tree and Branch Tree on the same two old development cases; this stays inside the previously recorded $20 development allowance and original $200 cumulative cap. Retain all v1 answers, judgments and diagnostics under their original source identity.

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 /home/danqingwang/anaconda3/envs/debate/bin/python experiments/incremental_planning/diagnose_branch_run.py branch-dev-v1
/home/danqingwang/anaconda3/envs/debate/bin/python experiments/incremental_planning/summarize.py branch-dev-v1
PYTHONPATH=src:debate-app/backend HF_HUB_OFFLINE=1 /home/danqingwang/anaconda3/envs/debate/bin/python -m pytest tests debate-app/backend/tests -q
PYTHONPATH=src HF_HUB_OFFLINE=1 /home/danqingwang/anaconda3/envs/debate/bin/python src/scripts/benchmark_incremental_planning.py --run-id branch-dev-v2 --split dev --cases-file experiments/incremental_planning/cases_v4.json --modes grounded_tree flat_tree branch_tree --judge-model gpt-5.6-sol --judge-max-tokens 1600 --cap-usd 200 --repeats 1 > experiments/incremental_planning/run/branch-dev-v2-worker0.log 2>&1
```
