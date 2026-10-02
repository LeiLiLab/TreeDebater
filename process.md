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
| Current TreeDebater | Online tree / turn-end planning reference | Development replay complete |
| End-of-turn TreeDebater | Isolate benefit of online preparation | Implemented; development replay complete |
| Linear incremental | Update explicit linear notes while listening | Implemented; development replay complete |
| Corrected tree | Isolate revision/retraction support | Implemented; development replay complete |
| Adaptive linear | Compare scheduling with the same linear state | Implemented; development replay complete |
| Tree early planning | Prepare tree-driven rebuttal plans before the endpoint | Implemented; development replay complete |
| Adaptive tree early planning | Combine semantic update scheduling and early planning | Implemented; development replay complete |

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

Development results are recorded below. Held-out performance improvement is **unverified**.

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
