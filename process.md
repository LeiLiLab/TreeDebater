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

| Variant | Intended purpose | Status |
| --- | --- | --- |
| Current TreeDebater | Online tree / turn-end planning reference | Held-out comparison complete |
| End-of-turn TreeDebater | Isolate benefit of online preparation | Implemented; held-out comparison complete |
| Linear incremental | Update explicit linear notes while listening | Implemented; held-out comparison complete |
| Corrected tree | Revision/retraction plus authoritative context at final revision | Implemented; held-out comparison complete |
| Adaptive linear | Compare scheduling with the same linear state | Implemented; held-out comparison complete |
| Tree early planning | Prepare tree-driven rebuttal plans before the endpoint | Implemented; held-out comparison complete |
| Adaptive tree early planning | Combine semantic update scheduling and early planning | Implemented; held-out comparison complete |

Evaluate targeted rebuttal quality, final-condition correctness, claim coverage, unsupported assertions, end-of-turn residual latency, and total input/output tokens and cost. Include late qualifiers, reversals, withdrawals, repeated content, and split clauses. Report measured text/planning latency separately from actual audible latency; do not describe a simulated timeline as a live audio measurement. Judges see delivered answers, not private preparation traces. Keep development cases separate from final held-out comparison.

## Budget and accounting

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

The frozen held-out comparison completed **168/168** answers. Full results and limitations are recorded at the end of this log and in `experiments/incremental_planning/heldout-v1_summary.json`. Linear reduced simulated text-ready latency and had a higher mean automated checklist score, but **stable quality improvement is not established**: quality confidence intervals include zero and spot checks found judge inconsistencies. Do not treat these small authored-case results as live speech or standard benchmark results.

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
