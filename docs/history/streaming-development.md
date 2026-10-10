# Historical streaming development notes

These entries describe successive experiments, including abandoned policies.
For current behavior use [the streaming guide](../../src/streaming/README.md).

### 🔊 Streaming TTS (Optional)
By default, the TTS pipeline generates audio serially and trims sentences that exceed the time budget. With `streaming_tts: true` on a debater’s config entry, a **streaming pipeline** is used instead:

- **Chunk-based processing**: the debate speech is split into paragraph-level chunks, each assigned a proportional share of the total time budget (opening: 240s, rebuttal: 240s, closing: 120s).
- **Adaptive refinement**: The shared `TIME_MODE_FOR_STATEMENT` setting selects duration estimation (`time`, `fastspeech`, or `openai`). When refinements are enabled, an LLM adjusts off-target chunks; candidate selection and playback accounting use actual synthesized audio duration. The listening preset allows up to 3 early or 10 later Gemma refinements per worker with meaning checks, including expansion of short chunks.
- **Streaming overlap**: while chunk N plays, chunk N+1 is being refined and TTS-generated, minimizing gaps.
- **No information loss**: instead of trimming sentences, text is rewritten to fit the budget.

To enable, set `streaming_tts: true` on each **debater** entry in your config (per-agent):
```yaml
debater:
  - side: for
    type: treedebater
    streaming_tts: true
  - side: against
    type: treedebater
    streaming_tts: true
```

For **Flat Tree** (`planning.mode: flat_tree`), the default is now whole-script
generation followed by streaming TTS (`streaming.output.speech_mode: full_script`).
The experimental `speech_mode: incremental` explicitly selects the older path: generate one complete argument, check its conditions,
sources and consistency with the published prefix, then publish its audio through
the existing chunk callback/file bridge. The next argument is generated after the
previous chunk is published, while the player can play that chunk.
Audio generation still requires `env.time_control: true` (or
`time_control=True` when calling the engine).

```yaml
env:
  time_control: true
debater:
  - side: against
    type: treedebater
    planning:
      mode: flat_tree
    streaming_listen: true
    streaming_tts: true
```

Published paragraphs are immutable, including queued audio. Each candidate is
reviewed by the existing audience reviewers; a rejected candidate gets at most
one repair and a second review. Malformed, incomplete or still failing reviews
stop the turn, preserving the published transcript and audio. An audio candidate
that exceeds the remaining budget can be regenerated and reviewed once; it is
never cut mid-argument. A failed partially published turn must not be retried as
a new full speech. Source/condition changes before publication invalidate that
candidate. Review judgments remain fallible; this is not a semantic correctness
guarantee.

Flat speech uses `streaming.output.first_chunk_seconds`, `later_chunk_seconds`,
`max_chunk_chars`, `max_stream_chunks`, `budget_mode`, TTS model/voice and seam
settings. Duration targets are estimates; actual audio controls the remaining
budget. `audio_duration` counts audio only; `experiment_elapsed` also counts
estimated gaps after the first publication. Playback buffering and actual browser
start times require separate telemetry. The adaptive TTS text-rewrite settings
and whole-speech `single_pass_revision` option do not bypass the per-argument
review. Turning `streaming_tts` off preserves the original whole-speech path;
other planning modes retain their existing TTS behavior.

Flat incremental runs write `*_chunks/chunk_NNN.mp3`, matching `.txt` files,
combined audio, and `speaking.json`. The JSON records the committed transcript,
completion/failure status, per-chunk preparation/publication timing, first checked
text and first audio readiness, and estimated queue gaps. These are different
metrics from the earlier text-only benchmark's full-answer waiting time.

Real-model testing found this path **not yet reliable**. After fixing conflicting
whole-speech/JSON draft instructions, eight known cases with real Gemma generation
and TTS produced audio in only 3/8 incremental attempts; all three stopped after
one paragraph, and 5/8 produced no audio. The full-script control completed 8/8.
First audio averaged 8.79s among the three audible incremental attempts versus
11.33s across all eight controls; this is not a valid overall speedup claim.
Among the three paired audible cases the difference was -0.72s (small-sample
95% bootstrap interval -4.03 to +3.81s). Independent automatic condition coverage
on spoken outputs was 3/9 versus 12/24; with silent turns counted as zero delivered
checks, delivery coverage was 3/24 versus 12/24. The first incremental paragraphs
lasted 27–47s despite the 12s target. Strict review can both miss semantic errors
and reject acceptable questions. Zero estimated queue gaps here does not establish
smooth incremental delivery: no incremental attempt published a second paragraph.
These are server audio-readiness and text-quality measurements, not browser onset,
ASR accuracy or a listening assessment. See [the full experiment record](process.md#real-speech-v2-results)
and [failure-inclusive results](experiments/incremental_planning/flat-speech-real-v2_summary.json).
The original 32-attempt protocol-failure run is also retained separately.

For a natural opening with the same audio/tail-revision overlap, use
`speech_mode: overlap_prefix`. It preserves the initial complete paragraph instead
of running the short-opening helper. Set `single_pass_revision: false` on the
DebaterConfig to finish a whole-draft revision before selecting that paragraph;
the app adapter currently sets this option internally, so a streaming YAML field
alone does not select the two-pass experimental recipe. Disable local tempo with
`first_chunk_local_tempo: false`; set both refinement counts to zero and both
`speed_adjust_min`/`speed_adjust_max` to 1 for unchanged-text, normal-speed TTS.
The [first-three-motion 4+4+2 comparison](process.md#motion-overlap-442-results)
uses these controls, equal historical contexts and separate natural/short opening
arms. Its harness manifest is the complete experimental configuration.
The completed 36-speech pilot found mean first-audio readiness of 32.01s for
natural openings and 29.97s for short openings, but natural was faster in 12 of
18 pairs; overall quality was 3.56/5 and 3.50/5. Both arms undershot every audio
target by more than 10%, so neither recipe yet provides close duration matching.
These are matched historical contexts, with local tempo and TTS rewriting off,
not full-match win rates or browser playback measurements.

The [complete Legacy timing comparison](process.md#legacy-motion-full-442) uses the
same18 contexts with Gemma TTS rewriting enabled and local tempo off. Mean first
audio is38.77s; mean opening/rebuttal/closing audio is217.64/222.35/102.15s against
240/240/120s targets. Seven of18 speeches are within±10%; all are shorter than
target. One request-limited partial attempt is preserved separately and explicitly
retried. This pilot improves duration proximity but does not reduce first-audio
waiting; no new quality judgment or browser playback measurement is claimed.

For whole-script delivery, `speech_mode: overlap_prefix` keeps the natural first
paragraph of the complete Flat draft after whole-draft audience feedback. It starts
opening TTS concurrently with revising the remaining draft. The opening stays
unchanged, including during adaptive TTS; only the remaining text can be rewritten.
An overlong or malformed opening falls back to whole-script revision before any
audio is published. A failure after publication preserves delivered audio/text.
`overlap_prefix.json` records draft, feedback, opening readiness, remaining-text
revision timing and exact publications. Both whole-script schedules honor
`single_pass_revision`: disabling it adds a whole-draft length revision and second
feedback pass before choosing the opening.

```yaml
streaming:
  output:
    speech_mode: overlap_prefix  # full_script for ordinary whole-text delivery
    budget_mode: audio_duration
    adaptive_delivery: true
    first_chunk_seconds: 8
    first_chunk_local_tempo: true
    local_tempo_min: 0.85
    local_tempo_max: 1.15
    local_tempo_deadband_seconds: 0.10
    later_chunk_seconds: 30
    allow_expansion: false
    refinement_model: google.gemma-4-26b-a4b
    early_max_refinements: 0
    max_refinements: 0  # Keep TTS text unchanged; compression remains experimental.
```

The live app can import [the Flat overlap-prefix preset](debate-app/configs/gemma-flat-overlap-prefix.yml),
which sets both `planning.mode: flat_tree` and the output options above.

To prepare the opening **during opponent input**, import the
[Flat listening-prefix preset](debate-app/configs/gemma-flat-listening-prefix.yml)
(`planning.mode: flat_tree`, `streaming.output.speech_mode: listening_prefix`).
The preset now enables local prepared-material retrieval with
`rehearsal: {enabled: true, pool_name: gemma-4-26b-a4b, mode: hybrid}`.
Both motion-specific `*_pool_for.json` and `*_pool_against.json` must exist under
`results/gemma-4-26b-a4b`; a missing pair stops preparation instead of generating
a new pool. Existing indexes in its `retrieval_indexes` directory are loaded and
the offline CPU encoder is warmed before speaking. Up to three current target
claims receive bounded recalled material in listening-body snapshots; without
an opponent target, recall supplies support for our own candidate claims.
Cache entries bind target context/version and the upcoming speech stage.
Materials remain private arguments, separate from heard speech and verified
evidence. This preset explicitly sets `claim_selection_strategy: saved_scores`:
opening preparation ranks the saved claim outlines. The default `native` strategy
retains TreeDebater's original history-aware framework selection and enabled
rehearsal-tree input, independently of `speech_mode`. Setting
`rehearsal.enabled: false` disables this integration. The engine flag is
`use_rehearsal_tree`; web-evidence retrieval (`use_retrieval`) and historical
example feedback remain disabled in this preset and the listening harness.
The [offline integration check](experiments/listening_retrieval_gemma/report.json)
loaded all four relevant indexes and recalled material for both sides from saved
v30 heard-input snapshots without network/API calls. This does not establish
quality or latency of a new complete match with retrieval enabled.
The subsequent [v31 live retest](experiments/incremental_planning/run/listening-motion-live-v31/report.json)
verified nonempty retrieval material and exchange records in all20 preparatory
body requests; its timing and content limitations are reported below.

The existing incremental planning call also returns a provisional framework:
our position, the core disagreement and any settled response directions. Once
that framework is clear, a worker writes the first paragraph and body together,
then reviews the actual first paragraph before publication. Claims and evidence
have already been selected in the common TreeDebater preparation lifecycle. Ordinary
elaboration keeps the overview. The planner nominates a rewrite only when new
scope, qualifications, withdrawals or stance changes require it. Each replacement
first paragraph passes the same publication review. Failed preparation is not
repeated for an unchanged framework.

Listening speech writers now retain Debater's conversation roles: our delivered
speeches are `assistant` messages and heard opponent speeches are separately
labelled `user` messages. A newer ASR snapshot replaces the current opponent turn;
the final instruction contains selected sources and private preparation, without
embedding the full history or repeating the latest opponent transcript. Drafts
and format repairs request one JSON speech object. Body revision requests only
the remaining spoken text. Native stage strategies remain shared, while legacy
Plan/Statement and reference-section output layouts are excluded. Speculative
revision reuse compares the role-labelled history as well as the instruction,
system prompt and writer options.

A separate worker updates the body draft from immutable input snapshots. Prefix
TTS runs independently and cannot block this worker. Version-bound prefix review,
ready-audio fallback and body-context ownership are described in the
[streaming policy](src/streaming/README.md#version-bound-first-paragraph-review). Initial
prewriting uses the upcoming stage's speech duration, including custom budgets;
body updates deduct the actual overview's draft length from that total. The old
`listening_body_words` setting remains accepted for configuration compatibility
but no longer sets the draft length. Audience feedback is
deferred until the complete input is available. After the first body preparation, `listening_body_update_words: 100`
requires at least 100 additional heard words before another preparation starts.
This counts transcript words, independently of draft phoneme/length settings;
zero restores updates on every changed snapshot. Changed openings and corrections
to previously heard text bypass the threshold. Queued work always takes the latest
snapshot, and the word baseline advances when work actually starts. Final full-input
feedback/reconciliation includes the remaining input even below the threshold.
Tree/planning updates and overview checks retain their existing cadence.
Both queues coalesce pending work. `listening_prefix_max_rewrites`
bounds semantic rewrites after the initial overview (default 2), while
`listening_prefix_max_calls` caps all speculative draft/review/repair/body calls
per opponent turn (default 48). Body work leaves eight calls reserved for late
overview changes. These limits exclude the existing planner calls and final-turn
generation/review. The legacy `listening_prefix_max_updates: N` maps to N-1
semantic rewrites; it no longer counts ordinary input updates. Speaking order,
including reversed debates, determines the upcoming stage. A rerecord freezes
and joins overview, body and prefix-audio workers and discards their results.

At our turn, the listener reconciles the final transcript. An endpoint review, when authoritative input changed,
checks the actual endorsed stance and two source questions: whether the overview can be said at this handoff stage
(`ready_to_speak`), and whether the latest input invalidates its actual premises
or promises (`latest_input`). Early handoff requires independence from unheard input.
Each request retains the complete transcript and source evidence but asks for a
compact result, not a report on every condition or sentence. Ordinary advocacy,
value judgments and announced response directions need not be proved in the overview.
Body feedback uses the existing Audience and its native audience feedback prompt:
message clarity, engagement, evidence presentation and persuasive elements. Complete-input feedback uses isolated audience conversations, with
corrections directed only to the remaining body. Matching early feedback is reused.
The listening preset explicitly selects `streaming.output.audience_feedback_mode:
compact` (at most three critical issues and 180 English words). The default `full`
mode preserves the original analysis format. Both modes use the same evaluation
criteria and honor enabled retrieval feedback; shortening feedback does not change
claim selection or disable retrieval. See [the shared capability boundaries](src/streaming/README.md#shared-treedebater-capabilities).
Early revision receives the selected supplemental evidence. A nonempty selection
triggers revision even when feedback says `No changes` and the body fits its time
budget. Matching final work reuses the selection and revision; speculative work
does not mark evidence used. Closing retains its existing no-new-evidence policy.
With `listening_single_body_revision`, `listening_stream_body_revision` (default
true) streams the one whole-body evidence rewrite into the existing adaptive TTS
pipeline. Complete paragraphs keep stable chunk boundaries; the first can enter
length refinement and playback before the rest of the revision finishes. Until
the final text arrives, the allocator reserves time for the estimated unseen
body; it then uses the exact remaining text and actual audio already delivered.
The original candidate pool, length-refinement limits, playback deadlines and
raw-audio fallback still apply. The first ready paragraph can also be synthesized
before final-input handover, but publication requires the same exact-input match.
Stream errors stop further delivery and retain the already published transcript;
there is no automatic text-stream retry. Set this option false to restore the
full-text handoff. The evidence-writing prompt also distinguishes source content
from feedback suggestions and requires source scope and qualifications to survive.
An overview rejection must identify
a concrete conflicting draft span and explain the correction; quoted spans are
validated locally. New details alone do not require a new overview, while an attack
on a withdrawn proposal or a false allegation about absent safeguards can.

Review format failures get one repair with unchanged text. If still inconclusive,
the system may generate one simpler overview and review it again; an inconclusive
check is not treated as a semantic defect or as approval to speak. Persistent review
failure stops before audio. A concrete conflict, stale target or unavailable candidate
gets a fresh overview and the same compact checks. Speculative calls still share the
existing cap. These are fallible model judgments, not proof of semantic accuracy.

The preset enables `listening_prefix_pre_synthesize`: the existing body worker
may synthesize a reviewed overview during opponent input. Cached audio must match
the final accepted text, voice and model; it is never published before the final
gate. A changed overview gets fresh TTS. The preset uses Gemma for all text
generation, planning and complete-input feedback. There
is no separate preparatory-review model override.
Preparatory review receives actual spoken history and the current transcript.
Listening preparation also keeps a compact `clash_records` view of each sourced
main debate-tree branch. Each record identifies the issue, recent speaker-owned
excerpts and reply links, historical/withdrawn positions, the latest change, and
which current claims have no recorded reply. Root revisions retain the same
issue ID; independent branches remain separate. Reply presence is not a verdict
on whether the reasoning succeeds. Private undelivered drafts are excluded.
Records are rebuilt from retained trees, so checkpoint restoration and transcript
replay cannot leave a separate ledger stale. Each record shows at most four
entries; planning, body drafting and feedback receive the three most recent
current issues, with older records available when space permits. Omission counts
and truncated-excerpt flags prevent treating this bounded view as complete history.
The full latest transcript remains authoritative. After complete ASR arrives, early
concurrent feedback captures the latest completed observer context independently
of the prefix review context. It never waits for pending analysis or reads mutable
trees. The resulting body task stays frozen through revision and publication. All branch records are saved under `clash_records` in the speech's
`listening_prefix.json` trace. This adds no separate model request.
`listening_planning_timeout_seconds: 6` bounds the optional planning request with
no retry; a timeout preserves verbatim input for final reconciliation. Its default
is zero (no extra deadline), and pre-synthesis defaults to false.

Once accepted, cached opening audio or fresh TTS runs concurrently with **all** remaining prompt setup,
whole-speech drafting, audience feedback and tail revisions. Completed body drafts seed that work; they must be reconciled with the complete final
transcript and cannot be published directly. Unfinished background work is never
awaited before the first audio. The overview remains
fixed in every revision and cannot be changed by TTS or replayed in the tail.
Existing `single_pass_revision` and stage-specific feedback policies still apply.
Speech revisions explicitly request plain text. If the model still returns a
`{"speech": "..."}` envelope, its speech text is extracted before prefix removal
and length estimation. A single leading echo of the fixed prefix is stripped;
repetitions inside the remaining speech still fail before that body is published.
The first spoken paragraph is reviewed before publication against the assigned side
and authoritative input. A matching prepared verdict is reused; new complete ASR
requires a fresh check before the opening can play. A rejected opening and its
unpublished body are regenerated together, with at most one repair. The body
uses the native whole-speech Audience feedback and revision; there is no separate
final-body gate. Semantic judgments remain fallible.

TTS length edits retain source/context binding and structural prefix guards.
They do not run a semantic reviewer. The obsolete `verify_rewrites` setting is
removed; `rewrite_audit.json` is always saved for an output directory.
Adaptive refinement uses the remaining playback time of all published audio as
an absolute deadline, including time already spent waiting for the body. It stops
waiting when workers and pending audio finish, and skips unchanged or previously attempted text
within a refinement branch before another TTS request. If body
preparation exhausts the playback buffer, raw synthesis still runs, while optional
edits are skipped. This does not guarantee gap-free playback when generation or
synthesis itself takes longer than the available audio.
The listening preset and next-run harness use `max_parallel_tts: 8` per candidate
pool. When estimated duration is outside the target range, refinement proceeds
while that candidate's TTS is running, as in the legacy pipeline. Available actual
audio durations are still used, and any completed candidate can be adopted using
its measured duration. Changed text must pass the configured semantic review
before synthesis. Prestart and normal workers share identical candidates, including
the original audio. At the playback deadline, completed original audio remains a
fallback; late edited candidates cannot replace it, and queued edits stop before
dispatch. Setting the pool size to1 retains sequential measured-audio refinement.
Parallel work may incur charges for unused candidates; the live harness continues
to reserve every request against the existing cumulative budgets.
The first speaker has no opponent input to overlap, so its opening is a cold start.
The provided preset uses 240/240/120-second budgets, a 16-second overview target,
20-second later chunk targets, normal speed and no local tempo. It enables
`allow_expansion: true` with `early_max_refinements: 3` and `max_refinements: 10`:
each adaptive worker may make up to 3 early or 10 later length edits, using Gemma for each edit and
the required meaning check. The fixed first chunk remains unchanged. Rejected
edits keep the original audio available; deadlines can prevent an expansion
from being used. Closing bodies group established clashes and the verdict into two
substantive paragraphs. Estimated seconds from `TIME_MODE_FOR_STATEMENT` use a
10% nominal tolerance; after at most two revisions, the closest format-valid
candidate may be used within 20%. Draft limits use `LENGTH_MODE_FOR_DRAFT` through
the same [shared interface](src/utils/speech_length.py). Actual
audio duration is measured separately against the 15% acceptance limit. These
settings do not guarantee that tail work finishes within the overview playback. `listening_prefix.json` records speculation, final
review, generation-start-relative audio readiness, tail work, actual audio duration
and queue-estimated gaps, plus overview decisions and body preparation. This readiness clock starts at stage generation; browser
turn-end measurements must also include listener backlog/transport time. Real model
latency was measured in a historical mixed-model six-turn match using real TTS,
server-paced playback and incremental Whisper ASR. That run used unauthorized
GPT-5.6 Sol preparatory feedback and does **not** satisfy the all-Gemma requirement.
The all-Gemma v21 retest stopped after three speeches: the third lasted200.881s
against240s, a16.3% shortfall. Expansion was disabled in that run. With expansion
and3/10 adaptive refinements enabled, the [v22 retest](experiments/incremental_planning/run/listening-motion-live-v22/report.json)
completed all six speeches. First audio took5.21–9.13s, the maximum playback gap
was0.0313s, and the maximum absolute duration deviation was2.03%. All549 text
requests used Gemma. Timing passed, but overall quality did not: the AGAINST
rebuttal reversed argument ownership before TTS, and the semantic checker accepted
an expansion that strengthened a qualified security claim. This is one fresh
motion, not a paired comparison with the66-speech Legacy benchmark.
With the final body stance/attribution gate enabled, the
[v23 retest](experiments/incremental_planning/run/listening-motion-live-v23/report.json)
stopped after four speeches because AGAINST rebuttal had a3.084s opening-to-body
playback gap (limit2s). First audio was4.39–7.94s and absolute duration errors
were at most2.25%. All451 text calls used Gemma. Four body gates passed without
repair, taking0.48–0.60s each; no stance reversal was observed, but this does not
establish detection or repair effectiveness. Unsupported implementation
assumptions and accepted TTS edits repeating adjacent sentences remain. Both
closing speeches were not run; overall timing and content acceptance failed.
The harness records timing failures and continues all six speeches;
budget, generation, body-gate, and listener failures still stop execution. A
completed run can therefore have `timing_pass: false`. With corrected playback
deadlines and8 TTS workers per candidate pool, the
[v24 retest](experiments/incremental_planning/run/listening-motion-live-v24/report.json)
completed all six speeches: first audio4.84–8.17s, maximum recorded gap0.0218s,
maximum absolute duration error2.26%. All79 joins had the next audio ready before
the preceding chunk ended; using decoded duration to correct the recorded-end
bookkeeping offset gives a maximum gap0.0229s. All479 text calls used Gemma.
Six body gates passed without repair, but manual content review still failed:
unsupported opponent assumptions, overstated efficacy, and repeated TTS
conclusions remain. This fresh match is not an isolated concurrency ablation.
Known new usage was about USD1.023; conservative incremental exposure USD4.539,
including two optional-planning timeouts retained at full reservation. No pending
calls remain. The frozen source snapshot is authoritative; unrelated rehearsal
cache edits appeared in the shared workspace during this run, with rehearsal
disabled and the streaming/TTS source unchanged.
The experimental `listening_prefix_overlap_final_update` option (default false)
allows a driver to supply a frozen, pre-reviewed overview with matching ready
audio and a completion callback for the final incoming history. It publishes that
overview while final ASR/tree/planning finishes; body generation waits for the
complete input, then performs the full-input overview review and final-body gate.
A late overview conflict or review error is logged and body delivery continues. Without ready
reviewed audio, it waits for the complete input as before. The ordinary app preset
does not enable this experiment.
The [v25 verification](experiments/incremental_planning/run/listening-motion-live-v25/report.json)
completed all six speeches and verified actual overlap at all five switches.
Switch first audio was0.187–0.241s (mean0.218s, versus6.149s in fresh-match v24),
but prefix-to-body gaps were1.804/10.042/1.411/0.002/3.246s. Two turns exceeded
the2s gap limit; overall timing acceptance failed. Final input handling plus whole
body feedback/length revisions can exceed the14–17s overview buffer. All six final
overview and body checks passed; complete ASR history was verified before body
work. Content quality still has unsupported assumptions and overstated claims.
This demonstrates earlier first audio, not a complete solution to continuous
playback or a paired performance comparison. Known new usage USD1.026;
conservative incremental exposure USD6.017, global USD217.371/250 with pending0.
Two further experimental options, both default false, are enabled in the
[v26 verification](experiments/incremental_planning/run/listening-motion-live-v26/report.json).
`listening_parallel_body_feedback` accepts a separate complete-ASR history callback
and reviews the frozen prepared body while the last tree/planning update finishes.
It reuses that review only when the full history and draft match exactly.
`listening_single_body_revision` permits at most one ordinary body rewrite and
leaves further duration adjustment to adaptive TTS; the final semantic gate and
its one permitted repair remain active. The full-input overview review retains
its existing order. All five feedback calls overlapped final analysis, hiding
0.909–2.085s, and every cached result matched the final input. Two speeches needed
no ordinary rewrite; the other four used one each. The six speech durations were
within0.805% of their targets. Maximum gaps by turn were
0.011/4.644/0.020/0.543/4.790/0.009s. The worst gap fell from10.042s in v25 to4.790s,
but two turns still exceeded2s; this fresh conversation is not a paired ablation.
Switch first audio remained0.184–0.240s. Final-input delay and claim selection or
overview review can still exhaust the prefix buffer. Manual review also found
missed opponent concessions and unsupported policy assumptions despite passing
gates, so overall acceptance remains false. All130 related offline tests passed.
Known new usage was USD1.020; conservative new exposure USD6.415, cumulative
exposure USD223.786/250 and study exposure USD132.340/160, with no pending calls.

The live benchmark now uses separate single-worker ASR and ordered analysis pools:
recognition can continue while the previous tree/planning update runs. Only played
audio is recognized, and the full mutable-state handover waits for both pools to
drain. This applies to `benchmark_listening_motion_live.py`.
The [v27 partial retest](experiments/incremental_planning/run/listening-motion-live-v27/report.json)
completed four speeches; turn five published its16.272s prefix, then stopped because
whole-speech feedback omitted the required `other_corrections` field. The body was
withheld and turn six did not start. All84 related offline tests passed, including
forced ASR/analysis overlap, ordered updates, final drain and failure propagation.
The four complete speeches had maximum ASR queue time1.74ms and maximum playback
gaps0.009/0.028/0.008/4.304s. Their input chunks happened to arrive after previous
analysis completed, so no live ASR/analysis overlap occurred; the event tests
establish the concurrency behavior. This partial fresh match is not evidence of a
causal speedup or full six-turn acceptance. Known usage USD0.817; conservative new
exposure USD3.907, cumulative227.693/250 and study136.247/160, pending0.

The experimental `listening_parallel_endpoint_revision` option now also starts
full-input overview review and body feedback/revision independently of final
analysis. Frozen values drive those calls; exact endpoint payload and final
continuation-prompt equality are required for reuse after handover. Native claim
selection/evidence bookkeeping and the final body gate remain after handover.
The [v28 six-turn retest](experiments/incremental_planning/run/listening-motion-live-v28/report.json)
verified five endpoint reviews overlapping analysis and four speculative revisions
overlapping analysis; three revisions were reused. Actual endpoint review finished
before revision began in each live turn; forced event tests establish that neither
blocks the other. Switch first audio was0.214–0.226s, all duration errors were under
0.5%, and maximum gaps were0.009/0.017/0.010/2.759/0.011/0.019s. The remaining gap
followed one discarded speculative revision: MP3 header versus decoded/processed
prefix duration changed the target from486 to487 words. Current code now uses the
published prefix duration, or defers revision if publication is not ready. All165
related tests pass, including padded-header regression cases; an offline replay
rebuilds the exact recorded final prompt. That fix has not yet been retested live.
Missing optional `other_corrections` defaults to an empty list while preserving
point corrections; malformed core reviews still fail. Manual review found missed
qualifications, overstatement and repeated endings, so overall acceptance remains
false. Known new usage was USD0.993; conservative new exposure USD7.490, cumulative
USD235.183/250 and study USD143.738/160, pending0. The frozen run snapshot preserves
the measured pre-fix code; the report explicitly separates the subsequent fix.

The [v29 retry](experiments/incremental_planning/run/listening-motion-live-v29/report.json)
completed all six turns using the published-prefix-duration fix. All three early
body revisions matched the committed prompt and were reused, with no discarded
speculation. The previously affected against rebuttal had a0.019s maximum gap.
However, the for rebuttal lacked a usable prepared opening: the draft and format
repair both retained nonempty opponent target IDs forbidden by early handoff.
The failed framework was then cached, and cold generation yielded14.847s first
audio and a6.479s gap. Other turns had gaps below0.020s; all duration errors were
under2.4%. Overall timing and manual quality acceptance still fail. This is a
fresh match, not a paired causal comparison. The prefix-metadata failure has since been fixed: draft and repair share explicit
empty-ID instructions and field-specific feedback; pure format failures can retry
once after new heard input within the existing call budget. Semantic review is
still required. The fix passes92 related offline tests, including replay of the
recorded failures, bounded recovery and prepared-audio handoff. The full live
retest is reported below.
Known usage USD0.980; conservative incremental exposure USD6.163, cumulative
USD241.346. During the run the user explicitly added USD20. After the frozen run
completed, the [budget migration](experiments/incremental_planning/budget_increase_250_to_270.json)
raised the shared global cap to USD270 and the study cap to USD180, preserving
all call/settlement rows and validating the new trigger. These are the same USD20
allocation, not separate increases. Immediately after that migration, global
headroom was USD28.654 and study headroom USD30.099, pending0. All31 related
driver/accounting tests passed.


The [v30 full retest](experiments/incremental_planning/run/listening-motion-live-v30/report.json)
completed all six turns and passed all timing thresholds: cold first audio6.240s,
five switch latencies0.205–0.262s (mean0.238s), maximum recorded gap0.230s,
and absolute duration errors below0.77%. All five prepared handoffs, endpoint
reviews and full-history feedback were reused; all four eligible early body
revisions matched the committed prompt and were reused. All six prefix drafts
returned empty target IDs; format repair/retry was therefore not exercised live.
The previously failing for rebuttal started in0.239s with a0.004s maximum gap.
The single delayed seam was the against rebuttal prefix-to-body handoff, with
body audio arriving about0.231s late. These are server-paced measurements, not
sound-card output or a paired causal comparison against v29.
Manual quality still fails due to unsupported categorical claims, incomplete
responses to privacy-preserving verification and repeated conclusions, despite
six accepted body gates. Known new usage USD1.023; conservative incremental
exposure USD6.085, including retained unknown-use reservations. Cumulative
exposure is USD247.431/270, leaving USD22.569; study exposure USD155.986/180.
Reconciliation reports no issues and no pending calls.

The [v31 full retest](experiments/incremental_planning/run/listening-motion-live-v31/report.json)
ran with100-word body-update batching, Gemma pool retrieval, the separated review
scopes and compact exchange records. All six turns passed timing: cold first
audio5.972s, five switch latencies0.216–0.235s, maximum recorded gap0.522s and
maximum absolute duration error1.436%. All five prepared opening handoffs and
endpoint reviews were reused; four eligible complete-history feedback/revision
chains were reused. All20 preparatory body requests received retrieved material
and records;12 drafts passed the length check and8 exceeded their draft budgets.
Both against-closing drafts exceeded its280 phoneme-based word-equivalent limit
(284.222 and352.889), leaving no reusable body. Its fresh body path caused the
single0.522s delayed seam. Source snapshots, decoded audio, ASR causality and
history, actual review prompts/verdicts and record payloads were checked.

All six publication gates accepted, but manual content review still fails.
Speeches now explicitly continue the decentralized-verification exchange, yet
often assert comparative weights or categorical outcomes without establishing
them, and sometimes call answered objections ignored. One feedback call identified
central storage as an added assumption but supplied the literal string `empty`
instead of a correction; the published speech retained that assumption as fact.
This fresh match changes several components and cannot isolate their effects.
Known new usage is USD0.972; conservative incremental exposure USD5.144 includes
six unknown-use planning timeout reservations. Cumulative exposure USD252.575/270
leaves USD17.425; study exposure USD161.130/180, with no pending calls or accounting
issues. See the [actual speeches](experiments/incremental_planning/run/listening-motion-live-v31/conversation.txt)
and [exchange-record snapshots](experiments/incremental_planning/run/listening-motion-live-v31/clash_records.json).

The [v32 simplification retest](experiments/incremental_planning/run/listening-motion-live-v32/report.json)
uses a shared immutable body task, retains oversized preparations for final
revision, requires explicit concrete feedback actions, and skips overwritten
legacy prompt construction and redundant rehearsal retrieval during validation.
All six turns passed timing: cold first audio7.037s, five switch latencies
0.202–0.218s, maximum recorded gap0.0153s, maximum absolute duration error2.301%.
All five prepared bodies, complete-input reviews and speculative revisions were
reused. Four selected bodies were oversized;14 oversized preparations were
retained overall. The last closing body was ready in7.605s versus14.591s in v31;
there were no audio-not-ready seams.289 unique offline tests passed.

Manual quality review still fails: raw-ID storage and universal breach scope
remain assumed, comparisons remain underdeveloped, and FOR promises but omits
inclusivity coverage. Explicit correction actions removed placeholder ambiguity
but did not fix model judgment.451 Gemma and159 TTS calls versus429/151 in v31
mean this version did not reduce call counts. Cold-start latency and maximum
duration error also increased within the accepted limits. This fresh match does
not establish a controlled causal performance improvement.
Known new usage USD1.016; conservative incremental exposure USD5.547. Global
exposure USD258.122/270 leaves USD11.878; study exposure USD166.676/180. Seven
unknown-use planning timeouts retain full reservations, with zero pending calls
and no accounting issues. See the [actual speeches](experiments/incremental_planning/run/listening-motion-live-v32/conversation.txt)
and [manual review](experiments/incremental_planning/run/listening-motion-live-v32/manual_quality_review.json).

After v32, the live harness restores the native input environment's separate
text threshold: `INPUT_CONFIG.min_text_words=100`. Recognition still uses each
played TTS chunk; complete ASR texts accumulate before one ordered tree/planning
update, with a final partial-batch drain before mutable handover. Full recognized
history is available independently of that buffer. The existing native
`StreamingInputEnv.min_text_words` mechanism was already independent; this fixes
the benchmark path that bypassed it. The next run ID is v33; no live rerun has
been launched. [Offline grouping](experiments/incremental_planning/analysis_batching_v33_offline.json)
of v32's80 ASR chunks yields26 analysis/planning triggers (5/5/5/5/3/3), with no
text omitted or duplicated.76 related offline tests pass. These are trigger
counts, not measured total model savings or new timing/quality results.

The next version restores three pieces of the existing design: selected issue
response chains carry full reply excerpts and conditions into body work; saved
Gemma group candidates retain their minimax scores and seed the opening case
through local ranking; the harness flushes short recognized text after60s as well
as at100 words or turn end. Timeout submission remains independent of an in-flight
ASR request. Final exchange changes invalidate stale speculative body work.
281 offline tests pass. The26-trigger grouping above predates timeout submission
and is not a measurement of the combined version.

**v33 live results:** six complete speeches, **5/6 timing passes; quality not passed**.
Cold first audio7.940s; five switch onsets0.208–0.223s; maximum duration error2.499%.
The final AGAINST closing has a4.257s prefix-to-body gap (target<=2s), while the
other five speeches stay below0.019s. Its first planning call timed out; the next
100-word batch produced an overview with only6.30s left for body preparation.
The in-flight body crossed handover/freeze, so final delivery drafted again and
finished body work at18.41s, after the15.57s prefix. See [failure diagnosis](experiments/incremental_planning/run/listening-motion-live-v33/last_gap_diagnosis.json).

Actual79 ASR requests became27 ordered analysis/planning batches (22 word-threshold,
5 final drains, no timeout flush). Text calls fell451→307 (-31.9%) against v32:
analysis/helper calls173→69 and TTS length/meaning calls208→168. TTS requests159→148.
Five pre-reviewed overview handoffs were reused; four eligible early body feedback/
revision results were invalidated by final input changes. Twelve oversized drafts
were retained. These are descriptive counts across fresh conversations, not a
controlled ablation. The live run did not exercise timeout flushing; offline
forced-timeout tests cover that path.

FOR rebuttal/closing now substantively address the previously omitted inclusivity
issue, but all six speeches retain major reasoning problems. AGAINST still treats
centralized storage as necessary after decentralization/minimization is supplied;
its feedback requests include that full context and its publication gate accepts.
FOR overstates prevention and safeguard sufficiency. Formal action fields and
source-bound chains do not establish semantic correctness.

Known new API usage is estimated at USD0.873; conservative incremental exposure
USD4.427 includes three unknown-use planning timeouts. Global exposure is now
USD262.549/270 (USD7.451 left), study USD171.103/180; zero pending calls, no
accounting issues. [Report](experiments/incremental_planning/run/listening-motion-live-v33/report.json),
[actual speeches](experiments/incremental_planning/run/listening-motion-live-v33/conversation.txt),
and [manual quality review](experiments/incremental_planning/run/listening-motion-live-v33/manual_quality_review.json).

After diagnosing v33's last gap, draft generation and preparatory feedback are
saved separately. Valid unfinished-review drafts survive handover, and an already
running draft can return an immutable value while final ASR/tree work finishes.
If ready at final body selection, that value enters the existing complete-input
feedback/revision/publication gates; otherwise the ordinary fallback starts
without an added wait. No new call stage or planning retry was introduced.
311 offline tests pass, including late arrival, frozen-state isolation, rejected
publication and mismatched/unavailable transfer cases. The next run is v34;
no paid rerun has been launched for this change.

Native debate-prompt reuse is now restored for overview drafting/repair,
preparatory bodies, and speculative/committed speech revision. These calls send
the exact configured debater system prompt, including instance overrides. The
original opening/rebuttal/closing strategy blocks are shared with legacy prompts;
the three assembled legacy stage prompts are unchanged. Streaming requests keep
their own output schema, remaining budget and immutable-prefix contract. Review
and TTS editing requests retain their task-specific instructions. Offline tests
inspect model-bound system/user messages and invalidate speculative revision when
its system prompt changes. The initial correction passed340 selected offline
regression tests. Subsequent live verification is recorded below; longer prompts
can affect cost and latency.

**v36 scheduling fix and six-turn live result.** Feedback and revision now finish
independently, so obsolete speculative work does not block the final body. When
complete source input matches, the current turn keeps its full-ASR task's private
plan/record hints despite later derived updates. Exact user/system prompt equality
still gates revision reuse; final source checks remain mandatory.354 offline tests
passed, followed by a complete six-turn real TTS/playback/ASR run.

| Speech | First audio, seconds | Largest playback gap, seconds | Audio duration, seconds | Timing pass |
| --- | ---: | ---: | ---: | --- |
| FOR opening |6.770|0.007|240.945|Yes|
| AGAINST opening |0.218|0.018|238.148|Yes|
| FOR rebuttal |0.199|0.008|239.607|Yes|
| AGAINST rebuttal |0.198|0.016|239.495|Yes|
| FOR closing |0.209|0.007|117.838|Yes|
| AGAINST closing |13.113|0.007|118.304|No|

Four full-ASR task bindings retained changed derived context, and all four early
feedback/revision results were reused. Actual authoring messages passed28/28 native
system/stage checks. The prior5.50/5.97s rebuttal gaps became0.008/0.016s in this
fresh conversation; different generated text and overview durations prevent a
controlled causal comparison. Last AGAINST closing had three3s planning timeouts,
no prepared framework, and a cold start after full-input handover. It exceeded the
10s first-audio target, so timing passed5/6 rather than all six.

Manual delivered-text review still fails quality: unsupported absolute effects,
centralization assumed despite an explicit decentralized proposal, unresolved
access trade-offs, and a new legal safeguard introduced in closing. Six body gates
accepted their inputs without repairs, demonstrating that gate acceptance does
not establish overall debate quality.12 artifact audits passed; no pending calls.
Known usage is estimated atUSD0.8686, withUSD7.4724 conservative new exposure
(including15 unknown-use timeouts, fully reserved). Under the approvedUSD30 increase,
global exposure is274.8053/300, study183.3598/210; global headroom25.1947 remains.
These are estimates/reservations, not a settled provider invoice.
See the [v36 report](experiments/incremental_planning/run/listening-motion-live-v36/report.json),
[complete delivered transcript](experiments/incremental_planning/run/listening-motion-live-v36/conversation.txt),
and [closing-start diagnosis](experiments/incremental_planning/run/listening-motion-live-v36/closing_start_diagnosis.json).

**v35 rerun: incomplete, with live prompt reuse verified.** The initial v34 attempt
was interrupted after actual requests exposed one remaining omission: cold
listening bodies used their own instruction rather than the legacy stage builder.
That branch now adds the shared stage strategy;88 focused tests cover it. Audio
bundles are reserved when a turn first needs them, retaining all per-request bounds;
62 guard/integration tests passed before launch.

v35 completed three turns and stopped during the fourth (14 of16 audio chunks
fully played). Both closing turns were not run. All27 actual authoring requests
carry the exact native system and stage strategy, including closing preparation.
The two openings passed timing; the FOR rebuttal had a5.50s gap and the partial
AGAINST rebuttal a5.97s gap. Early feedback/revisions were invalidated by changed
final allocation/feedback; final body/TTS missed the prefix playback deadline.
Three saved bodies were used; no live in-flight adoption occurred. Manual reading
still finds unsupported certainty and weak comparative weighing. AGAINST rebuttal
now engages decentralization directly, but this incomplete fresh conversation is
not a controlled quality comparison.

The outer stop error is generic listener failure. Evidence strongly suggests a
speculative model-budget reservation rejection: pre-reconciliation exposure was
USD269.715/270, six unknown-use planning timeouts retain USD1.379, recent draft/
feedback reservations exceed the remaining USD0.285, and no ASR/TTS or recorded
listener-analysis error occurred. The exact rejected request was not persisted;
this cause remains an inference. Future budget stops now retain their first cause
in `budget_stop.json` (25 offline tests passed after the run).

After reconciliation, v35 known usage is estimated at USD0.747 and conservative
incremental exposure USD4.369. Including the interrupted v34 attempt, this request
used an estimated USD0.851 and USD4.784 of conservative allowance. Global exposure
is USD267.333/270, with USD2.667 left; no pending requests or automatic restart.
See the [v35 report](experiments/incremental_planning/run/listening-motion-live-v35/report.json),
[delivered/partial transcript](experiments/incremental_planning/run/listening-motion-live-v35/conversation.txt)
and [pending v36 budget proposal](experiments/incremental_planning/proposal_listening-motion-live-v36.json).

The [v20 report](experiments/incremental_planning/run/listening-motion-live-v20/report.json)
records first audio of 4.40/5.75/5.88/5.44/9.19/7.28 seconds, maximum inter-chunk
gap of 0.0093 seconds, and maximum duration deviation of 11.47%. The first
opening uses a cold-start clock; subsequent turns include final ASR/planning
backlog after opponent playback ends. Source snapshots and spoken/ASR-only
history were audited. All six transcripts received manual content review, with
argument-strength limitations recorded; this is one match, not an independent
quality score or browser/microphone latency guarantee.

The 8-second opening is a word-rate target, not a guarantee of audio duration.
With `first_chunk_local_tempo: true`, the full-script/fixed-prefix pipeline measures
the decoded opening and uses local FFmpeg `atempo` before publishing it. It preserves
pitch while changing tempo, within the configured bounds (default 0.85–1.15).
The target is `first_chunk_seconds`; set 12 for a 12-second opening. This step changes
audio, not text, and makes no extra TTS request. It also applies when text rewrites
are disabled. A clamped adjustment can remain too short/long; it does not slice
speech or pad silence to force the target. Processing failures retain the original
audio. Callbacks, combined audio and remaining budgets use the adjusted duration.
The option defaults to false globally and remains false in the listening-prefix
preset.
`chunk_profile.csv` records the input duration, tempo factor, local processing time,
clamping and success/fallback status. This option does not apply to the separate
argument-at-a-time `incremental` speech mode.

To adjust an existing recording offline without overwriting the original:

```bash
PYTHONPATH=src python -m streaming.audio_tempo input.mp3 output_8s.mp3 --target-seconds 8
```

The output must be a new `.mp3` or `.wav` file. On 16 saved opening clips, 8±1s
compliance improved from 4/8 to 8/8; 12±1s compliance improved from 3/8 to 6/8.
Local filtering plus MP3 encode/decode averaged 0.12–0.14s per clip on this host.
This was offline reprocessing, not a new full-speech latency or listening evaluation.
See [the local tempo validation](process.md#local-audio-tempo-results).

The opening revision is limited to 1.5 times its target word count and must end
as a complete sentence. All synthesis/rewrite workers settle before a turn returns,
while audio callbacks publish chunks before the turn returns. This avoids background requests
outliving a speech's accounting. Adaptive splitting keeps sentences intact, so a
long sentence can exceed the target. With `allow_expansion: false`, short audio is
accepted without adding words or slowing speech to fill time. `verify_rewrites`
checks proposed length edits against the original paragraph and spoken context;
uncertain or rejected edits leave the original audio available. This model check
is fallible, and preserving the original can exceed the audio budget.
`rewrite_audit.json` records proposed edits and decisions; an accepted edit may
still be unused if it misses the delivery deadline.

The configuration above uses Gemma for both shortening and checks via
`DEBATE_LLM_API_BASE` (and optional `DEBATE_LLM_API_KEY`). Omitting
`refinement_model` retains the existing `gpt-5-mini` default. Four-arm real-model
results are recorded in `flat-full-adaptive-v1_summary.json` and
`flat-full-adaptive-v2_summary.json`. In the 16-turn safety retest both arms returned
8/8 speeches. Fixed opening reduced mean opening audio from 10.92s to 6.92s, but
mean first-audio readiness only changed from 11.47s to 11.08s (paired 95% case
interval includes zero); condition coverage was 50.0% versus 45.8%. None of these
60s speeches needed compression. See [the full comparison and limitations](process.md#whole-flat-adaptive-results);
this does not establish a general latency or quality improvement.

Four additional 30s compression probes exposed a missed condition even with
`verify_rewrites: true`: the checker accepted dropping a two-metre access
requirement and a flood-warning shutdown condition. It did reject a separate
"could" to "inevitably" change. The full-speech example and its app preset disable TTS
text rewriting (`max_refinements: 0`, `early_max_refinements: 0`); opening/tail
revision still happens before those words are published. To experiment with
compression, explicitly set those limits to 2 and 1 and keep `allow_expansion: false`
and `verify_rewrites: true`. See `flat-compression-probe-v1_summary.json`.


For the original full-script streaming TTS path, each speech produces a `*_chunks/`
directory containing per-chunk audio, text, and a `chunk_profile.csv` with timing
details. To visualize that pipeline's overlap timeline:
```bash
bash src/scripts/overlap_viz.sh "log_files/<run>_outputs/*_chunks"
```


The listening path now consumes an issue/action/weight `body_plan` from its
existing planning call throughout body drafting, feedback and revision. It
retains every promised overview axis, binds opponent targets to source versions,
and allocates the remaining body words locally. Legacy opening claim selection
whose resulting prompt was discarded is removed from this path. The initial
version invalidated reuse on final plan changes; v36 instead retains a complete-ASR
authoring task when source input matches. See the v36 live results above.

Listening planning now presents source text in readable exchanges with inline
qualifications. Exact copies in target sources, short history entries, and the
condition ledger are removed only when the same source/owner remains visible.
The canonical context and selection indices remain intact. Offline replay of26
saved v36 requests preserved source text and bindings while reducing total input
characters by7.48%; latency and model quality have not been remeasured.

The subsequent v37 live trial used this projection and a6-second planning timeout,
with no retry. All20 planning requests completed (median2.05s,max3.99s), but the
match stopped after four complete speeches: the FOR closing's final readiness
review rejected its already-playing overview and withheld the body. Only14.9s of
that closing played; AGAINST closing never started. Completed-turn timing passed,
but full-match completion and content quality did not. See the
[v37 report](experiments/incremental_planning/run/listening-motion-live-v37/report.json).

Preparation and final-input overview reviews now use the same decision rules.
An announced argument remains advocacy at either stage; an opponent's disagreement
alone cannot invalidate it. Rejections must classify a concrete defect, and a
latest-input rejection must quote the source that invalidates an actual premise
or attributed commitment. The v37 draft and objection are retained in an offline
contract regression. 136 relevant tests passed; live model consistency has not
yet been remeasured.

The v38 paid trial exercised native Audience feedback and advisory final reviews.
It stopped on a60-second TTS ReadTimeout: the audio transport latched closed,
then the harness reported a blocked dispatch. This was neither a review rejection
nor exhausted budget. Three whole speeches played; the fourth played only its
15.04s overview, and neither closing started. Only the first two turns completed
pipeline finalization. First-audio times for the three full speeches were
7.81/.20/.19s, but maximum gaps5.34/16.97/4.92s missed the2s target.
All15 planning calls completed and all3 final body reviews passed, so advisory
rejection behavior was not exercised live. Report and exact delivered text:
[v38 report](experiments/incremental_planning/run/listening-motion-live-v38/report.json),
[conversation](experiments/incremental_planning/run/listening-motion-live-v38/conversation.txt).

The v39 regeneration completed all six speeches with the same runtime configuration
and no failed requests or TTS timeouts. Audio durations were234.846/240.054/240.056/
238.825/119.202/118.929s. Four turns met the timing targets; FOR opening exceeded
both first-audio (10.358s) and gap (3.794s) limits, and AGAINST rebuttal had a3.243s
gap. The other five first-audio delays were.209–.225s. All26 planning requests
completed without timeout;3responses exceeded the requested list caps. All6final
body reviews accepted; no advisory warning occurred. Manual review still found
unsupported efficacy/implementation claims and adjacent repetition. Report and
exact delivered text: [v39 report](experiments/incremental_planning/run/listening-motion-live-v39/report.json),
[conversation](experiments/incremental_planning/run/listening-motion-live-v39/conversation.txt).

Final body review runs serially before body TTS, following the requested revert
of the parallel-review change. Review rejection/errors remain advisory and do not
trigger another repair. Saved v37/v39 opening/rebuttal feedback averaged.97/8.01s;
this is a descriptive comparison of different generated speeches, not an ablation.

Listening Audience feedback now keeps the native four criteria but requests only
up to three concrete edits in at most180English words. Each speech attempts at
most one preparatory feedback round; body drafts continue updating and final
complete-input feedback remains.171offline tests passed; live latency/quality
under this compact policy have not yet been measured.

The v40 paid trial completed all six speeches and met all timing targets with
compact Audience output, one preparatory feedback attempt per speech, and serial
final-body review. Cold first audio6.964s; subsequent handovers.193–.226s; worst
gap1.282s. Audience calls fell23->11 (preparatory17->5, complete-input6->6), while
complete-input feedback averaged7.638->2.672s versus v39. All11responses contained
<=180whitespace-separated words and <=3numbered edits. These are fresh speeches,
not a controlled comparison. One optional length-rewrite HTTP500 fell back without
interrupting playback. Manual content review still found unsupported absolutes,
implementation assumptions and local repetition. [v40 report](experiments/incremental_planning/run/listening-motion-live-v40/report.json),
[conversation](experiments/incremental_planning/run/listening-motion-live-v40/conversation.txt).

The v41 overview trial completed all six speeches with stage-specific framing,
optional brief greetings and heard-opponent attribution. Five turns met timing
targets; the cold opening had a2.244s gap (first audio5.506s), while later
handovers were.171–.233s. Closing overviews used final weighing, but the first
four retained future-roadmap phrasing and neither opening overview defined terms.
An AGAINST rebuttal overview reversed an opponent's criticism into an endorsed
claim; both overview reviews accepted it. No unpublished overview rewrite occurred
live. Sixteen artifact audits passed; manual content quality did not.
[v41 report](experiments/incremental_planning/run/listening-motion-live-v41/report.json),
[overview review](experiments/incremental_planning/run/listening-motion-live-v41/overview_manual_review.json),
[conversation](experiments/incremental_planning/run/listening-motion-live-v41/conversation.txt).


The v42 diverse-overview trial completed all six speeches and all timing targets
(cold first audio7.734s; later handovers.219–.256s; maximum gap.0282s).
First-paragraph variation remains limited: three opponent-paraphrase roadmaps,
no greeting/definition, and closely worded final-choice closings. All model reviews
accepted, but manual content review found17 concerns; quality did not pass.
All16 artifact audits passed after updating a stale wording assertion, with the
initial failure preserved. No production changes occurred during the run.
[v42 report](experiments/incremental_planning/run/listening-motion-live-v42/report.json),
[overview review](experiments/incremental_planning/run/listening-motion-live-v42/overview_manual_review.json),
[conversation](experiments/incremental_planning/run/listening-motion-live-v42/conversation.txt).


The v43 full motion02 attempt stopped at the cumulative budget reservation guard.
Three speeches completed; the fourth was fully generated but only partly played;
both closings remain unfinished. All workers stopped. After audio reconciliation,
USD2.7505 remains under the existing cap. A scoped artifact-completion plan awaits
resume approval. [v43 report](experiments/incremental_planning/run/listening-motion-live-v43/report.json),
[complete generated texts](experiments/incremental_planning/run/listening-motion-live-v43/generated_conversation.txt),
[continuation plan](experiments/incremental_planning/run/listening-motion-live-v43/continuation_plan.json).

After the user's `增加20刀预算` approval, the v43 artifact continuation raised
the cumulative cap to USD320 and the study cap to USD230, preserving all prior
costs. It reused four speeches and completed both closings: all six audio files
total 1191.795 seconds. Content quality failed: FOR closing reversed its stance,
and both closing first paragraphs are identical. The original live ASR run remains
interrupted; this continuation completes text/audio artifacts only. Additional
usage estimate USD0.11130568; conservative cumulative exposure USD297.694946304,
remaining USD22.305053696. Pending requests: zero.
[Completion report](experiments/incremental_planning/run/listening-motion-live-v43-completion-v1/report.json),
[full transcript](experiments/incremental_planning/run/listening-motion-live-v43-completion-v1/conversation.txt),
[combined audio](experiments/incremental_planning/run/listening-motion-live-v43-completion-v1/complete_motion.mp3).

The final body repair mechanism has since been restored at the user's request:
one targeted repair and recheck after a valid rejection; an inconclusive review
or failed recheck withholds body TTS and preserves emitted audio. A repair may
briefly correct a wrong stance in the spoken prefix, prioritizing the assigned
side. Stance findings with the assigned `source_side` and an empty `source_quote`
are valid; attribution and qualification source checks remain enforced. The saved
v43 finding now validates as a rejection in an offline replay. The first-paragraph
gate is unchanged. Related offline regressions:156passed in21.11s; no paid rerun.

The subsequent first-paragraph review change adds an independent stance verdict
with an expressed-side classification, exact draft quote and reason. Explicit
opposite framework labels are rejected locally before model review, and a reviewer
cannot approve a paragraph it classifies as the opposite side. Neutral openings
remain allowed; inconclusive assessments cannot authorize playback. Existing
bounded repair/review runs before first audio, including prepared handoffs. The
saved v43 wrong-stance draft is now rejected even with its original all-true
review. Related offline regressions:257passed in23.02s; no paid rerun.


### v44 fresh motion03 complete live test (2026-10-08)

Motion: Learning to be a good writer still matters in the age of AI. All six
speeches completed with real TTS, wall-clock playback and incremental ASR, using
the restored bounded final-body gate and independent first-paragraph stance check.
First audio: 7.224s cold, 0.178–0.244s for five handoffs. Decoded-duration maximum
playback gap: 0.430s; maximum absolute duration error: 0.703%. Timing passes.
All 16 artifact audits pass. Sources remained unchanged during the run.

All six delivered prefixes maintain their side. Of 18 prefix reviews, two
rejected drafts were rewritten: one clear wrong-side draft was caught; another
internally contradictory draft was rejected with an inaccurate stance rationale.
All six final bodies passed first review, so the body repair path was not exercised.
Human quality review remains a fail: later chunk adaptation introduces adjacent
repetition in three turns, and the argument repeatedly shifts from "still matters"
to necessity/centrality. No paid judge or automatic rerun followed this test.

New usage estimate $0.870036; conservative incremental exposure $3.481944;
global exposure $301.176890/$320, remaining $18.823110, pending=0.
Artifacts: experiments/incremental_planning/run/listening-motion-live-v44/
(report.txt, report.json, conversation.txt, complete_motion.mp3).


### Context-bound delivery edits after v44 (2026-10-08)

The three observed v44 adjacent repetitions came from length edits importing the
next segment's opening into the current segment. Final-body review preceded those
edits, and the old local meaning check neither received the next segment nor checked
content ownership/adjacency. Separately, speculative candidate approvals could
survive a change in the finalized preceding text.

Implemented immutable EditScope (owned source, finalized preceding text, following
source, revision). Every rewrite iteration retains the original source. The existing
optional edit review now requires independent meaning, source-scope and boundary
verdicts, with both neighbours supplied; missing fields fail closed. No similarity
threshold, automatic sentence deletion, or extra whole-speech review was introduced.
Context changes atomically obsolete edited candidates, including ready results and
pending callbacks. A normal worker can re-review one speculative proposal within its
existing refinement limit and reuse its exact audio on fresh approval. Original
source audio remains the fallback. A changed segment source retires its old pool.
Audit artifacts record the scopes, obsolete status, publication and reuse.

191 offline tests passed in20.18s, including three exact v44 fixtures, stale context
at writing/review/registration, ready and pending TTS callbacks, fresh approval with
audio reuse, deadline fallback, concurrency and complete speech integration.
The fixtures verify data flow and approval enforcement, not live semantic accuracy.
No paid calls or budget changes. Live quality/latency after this change are unmeasured.
Source-authored overview/body repetition remains a separate authoring issue.
Log: experiments/incremental_planning/delivery_edit_scope_offline_tests.log.
Implementation delta from v44: experiments/incremental_planning/delivery_edit_scope_vs_v44.patch;
new contract: src/streaming/delivery_edit.py.


### v45 same motion03 live regeneration (2026-10-08)

All six speeches completed after the delivery edit scope changes. Source snapshot,
ASR coverage/history and17 artifact audits passed. Timing passes4/6: For opening
first audio10.155s and duration202.188s (-15.755%) fail; Against opening has a3.93s
prefix-to-body gap, with body ready10.460s, prefix end13.434s and body audio ready
17.367s (first body TTS API6.722s). Other turn durations237.070,231.181,224.731,
112.160,118.267s; all five handoffs first audio0.188–0.232s. No controlled latency
attribution to the new review is claimed.

Manual reading found no newly introduced adjacent duplicate seams. Review call22087
correctly rejected importing the next segment's agency/vision claim. Of84 edit
reviews,44 were rejected; some explanatory expansions appear overstrict, and raw
fallbacks left the first speech short. Three speculative candidates were invalidated,
two proposals re-reviewed, zero reused-audio candidates actually published. All
published edits had final-context approvals. Six prefixes/bodies kept their side;
two wrong-side prefix drafts were withheld. Final-body repair was not exercised.
Motion-scope/unsupported-absolute advocacy issues remain; overall acceptance fails.

New usage estimate$0.746439; conservative incremental exposure$2.987194;
global$304.164084/$320, remaining$15.835916; pending0. No automatic retry.
Artifacts: experiments/incremental_planning/run/listening-motion-live-v45/
(report.txt, report.json, conversation.txt, complete_motion.mp3, delivery_scope_audit.json).


### v46 simplified listening validation (2026-10-08)

Extra prefix/change/endpoint reviews, preparatory body feedback, final-body
gate/repair/recheck and TTS meaning reviews no longer run. Native draft/revision
templates and closing-feedback policy are reused.320 offline tests pass.

Validation failed: original live trial completed four turns and stopped during
FOR closing after a60s Whisper timeout on37.266s audio. Five speeches had been
generated. Missing ASR was completed and AGAINST closing117.061s was generated
separately; six audio/text artifacts are available, not a completed live match.
AGAINST rebuttal's actual first paragraph endorses FOR despite correct against
framework metadata. Gaps5.79s/4.10s also fail the2s target. Local repair schema
and explicit short-paragraph length instructions were corrected during artifact
completion; original trial source and failed attempts remain preserved.

All removed phases have zero model calls;100 text requests include the interrupted
trial and artifact completion. This is not a paired comparison with v45. New
usage estimate$0.687133; conservative exposure$3.567532/$10; cumulative global
$307.731616/$320, remaining$12.268384, pending0. The next useful change is a
minimal check of the actual first paragraph's stance, not just its metadata.

Artifacts: experiments/incremental_planning/run/listening-motion-live-v46/
(report.txt, report.json, conversation.txt, complete_motion.mp3,
prefix_stance_regression.json). Rollback snapshot:
../TreeDebater_snapshots/before_legacy_simplification_20261008T102837Z/
(restore.py verifies by default; --restore restores captured source files).
