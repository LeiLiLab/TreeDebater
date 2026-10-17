# Streaming speech generation and listening

The `listening_prefix` path prepares speech while the opponent is speaking. A
reviewed first paragraph can start playback before complete ASR and final tree
analysis finish. The remaining body uses complete recognized input, a fixed
writing task, one streamed revision and the native duration/TTS loop.

## Source and validation scope

Local experiment archives and historical reports are not distributed with this
repository. Historical measurements below describe their original runs, not a
reproducibility guarantee for the current source or changing provider models.

## Listening and handoff

1. Played audio is transcribed in order. In the motion harness, recognized text
   is batched at 100 words or after a 60-second wait. Recognition continues while
   an ordered worker updates the debate tree and plan. The final partial batch
   is drained at the end of the turn.
2. Preparation uses value snapshots of the source history and plan. It drafts
   the first paragraph and provisional body together, reviews the exact first
   paragraph, and prepares its audio independently of later body updates.
3. Review approval binds the exact text and review context. Repaired or replaced
   prefixes must pass review before publication. Malformed review output has a
   bounded retry policy. A previously approved, ready prefix can be retained
   when an optional replacement is unavailable.
4. Handoff publishes ready prefix audio without waiting for final tree analysis.
   Complete-ASR feedback can begin while transferred prefix TTS is still pending.
   Audio failure releases duration waiters and fails delivery.
5. Body work uses an immutable `BodyTask` with the complete transcript and latest
   completed same-turn context. A matching transferred draft is checked again
   after ASR; unfinished drafts are not awaited and the task is not replaced once
   bound. Source, prompt and authoring options determine reuse eligibility.
6. Body revision streams complete paragraphs to native duration fitting and TTS.
   Published text/audio is immutable. Final analysis is merged after the
   corresponding source input is available; obsolete work cannot overwrite a
   newer task.

Listening-time evidence selection remains enabled. At handoff, selection reuses
unchanged, eligible prepared evidence (or the initial selected pool on cold
start). It does not make an additional endpoint evidence-selection request.
Closing uses previously discussed arguments and evidence, and skips Audience
feedback. Model reviews are fallible semantic checks, not factual certification.

## Configuration

### Streaming Vanilla

For a `type: default` Debater, set its per-player output configuration:

```yaml
streaming_output:
  speech_mode: vanilla_streaming
  tts_backend: fastspeech
  vanilla_chunk_words: 30
  vanilla_max_chunk_chars: 1000
```

This preserves the existing opening/rebuttal/closing prompts and static word
budget. Generation starts when the normal stage method is called with the
opponent history. It uses the configured model client with streaming enabled;
there is no listening-time preparation, planner, audience feedback, content
revision, duration fitting or tempo adjustment. Sentence boundaries are preferred
after 30 words, with a character limit for long sentences. These thresholds are
fixed for a run, not adapted to its remaining time. The final short tail is flushed.

A bounded queue overlaps LLM generation with a single ordered TTS worker. Each
chunk is published through `tts_chunk_callback(index, path, text, seconds)` as
soon as it is ready. A playback consumer is required for audible streaming;
ordinary batch `Env` generation only writes the artifacts. Vanilla streaming output does not require a listening-time planner.

`vanilla_streaming.json` records raw generation, committed chunk text, audio
durations, readiness times, generation finish time and signed duration error.
The module does not trim to the target: any host-level time limit remains the
host's policy and must be reported separately. No latency or quality equivalence
to existing runs is assumed. Failed generation/synthesis is not automatically
restarted, and already published chunks are retained with a failed status.

Metered harnesses can supply `_stream_completion(messages, on_text, **options)`;
that callable must synchronously deliver text deltas, account for the request,
and raise on stream failure. Non-streaming `_completion` overrides are rejected
instead of silently bypassing their accounting. The default path uses the
existing `agent.client`, with stream usage/cost accounting where supported.

### Shared settings

`streaming.config` owns validated input, output, playback and speech-budget
settings. Explicit CLI arguments take precedence over YAML and defaults.
Unknown settings are rejected. Compatibility fields such as
`listening_prefix_max_updates` and `listening_body_words` remain accepted for
existing configs; the latter no longer sets the body budget.

Select the synthesis backend once for regular chunks, prefix preparation and
body preparation (the default remains `openai`):

```yaml
streaming:
  output:
    tts_backend: fastspeech  # openai | fastspeech
```

`model` and `voice` select the OpenAI model/voice. The FastSpeech backend uses the
existing local LJSpeech checkpoint and HiFi-GAN vocoder under
`dependencies/fastspeech2`, at native speed. It lazily loads the vocoder, shares
one locked model instance with estimation, and exports MP3 through ffmpeg to
preserve the playback interface. It requires no OpenAI credentials for synthesis;
LLM rewriting retains its separately configured model and credentials.

`TIME_MODE_FOR_STATEMENT` still selects the estimator. With `fastspeech`
estimation, the historical `1.11 * seconds - 7` correction for predictions above
100 seconds applies only to OpenAI output; FastSpeech output uses raw predicted
seconds. The legacy `openai` estimator mode measures synthesized audio through
the selected output backend. Prepared audio is reusable only for the same backend.
FastSpeech does not perform provider-side speed adjustment; optional existing
local audio tempo processing remains a separate setting.

Supported modes remain separate:

- `full_script`: write the full speech before paragraph TTS.
- `overlap_prefix`: retain the natural opening paragraph while revising the tail.
- `listening_prefix`: prepare the reviewed opening and body during listening.
- `incremental`: retain the segment-oriented speaking interface.

`listening_prefix` supports `planning.mode: flat_tree`, `linear`, and
`adaptive_linear`. `overlap_prefix` and `incremental` still require `flat_tree`.
The linear variants do not analyze or update debate trees. Their existing
planning call returns private notes plus the same readiness/keep/replace
overview used by prefix preparation. Notes enter drafting as unverified writing
suggestions, separately from source evidence; the server owns the assigned side.
Malformed planning metadata disables new preparation until a valid update or
the existing endpoint draft fallback.

Linear sources are verbatim heard batches with stable location IDs and
text-dependent versions, bounded by `planning.max_tree_targets` for compatibility
with the shared source interface. Complete role-labelled history remains in
authoring requests. Final ASR replacement invalidates changed source bindings.
`adaptive_linear` WAIT and the speculative update cap retain new input without
starting new drafts from stale notes. Already running work keeps its original
snapshot. Endpoint reconciliation drains pending input even after WAIT/the cap,
and the body uses complete input through the existing feedback/revision path.
The same prefix review, audio handoff and immutable-publication rules apply.

The v70 experiment uses Gemma 4 26B A4B, `tts-1`/`echo`, Whisper, six turns and
240/240/120-second opening/rebuttal/closing budgets per side. Its first-paragraph
word target is a soft 50 words, the first-audio target is 18 seconds, and the
first body paragraph has no separate duration ceiling. These are experiment
settings, not all global `OutputConfig` defaults. Consult the baseline manifest
for the complete resolved configuration.

Draft measurement and duration estimation are shared through
`utils.speech_length` and `utils.time_estimator`. The baseline uses phonemes for
draft measurement and FastSpeech for duration estimation. Native TTS candidates
are measured against the remaining proportional audio budget. Local tempo
adjustment is optional and is disabled in the baseline.

## Modules

In `listening_prefix`, the speech author returns a private `body_plan` before
the spoken fields in the existing JSON response. This applies to the shared
opening/body draft, speculative body updates, and cold body generation; there
is no separate planning call. Plans bind to supplied node IDs and versions,
carry response actions and weights, and survive into review and publication
snapshots. Existing subsequent evidence-selection updates can consume the
completed author's plan. With `flat_tree`, the listener remains a source selector
only; linear modes additionally supply provisional private notes.
Missing or malformed plans are recorded in `body_plan_audit` as a fallback
without a model retry. Changed node versions invalidate their bound tactics.
Metadata is excluded from spoken text and audio. Extra prompt/output tokens
may affect latency; this change does not establish a measured latency benefit.

| Responsibility | Modules |
| --- | --- |
| Preparation, prefix review and handoff | `listening_prefix`, `overview_planning`, `overview_review` |
| Stable body inputs and reuse | `body_task`, `body_plan`, `body_feedback`, `body_revision`, `body_audio` |
| Listening context and sources | `planning`, `linear_listening`, `branch_planning`, `planning_view`, `listening_evidence`, `clash_records` |
| Ordered text batching | `text_batching` |
| Streamed paragraphs and audio | `revision_stream`, `tts_streaming`, `delivery_edit`, `audio_tempo` |
| Source updates and matching | `tree_updates`, `target_matching`, `grounding`, `constraint_review` |
| Persistent experiment accounting | `experiment_accounting`, `experiment_client` |

Shared authoring code lives in `utils/prompts/authoring.py`,
`utils/prompts/speech_generation.py`, `utils/prompts/speech_revision.py` and
`utils/speech_context.py`. Full role-labelled history remains authoritative;
plans, selected evidence and feedback are derived writing inputs.

## Diagnostics and failure handling

Per-chunk text/audio, `chunk_profile.csv`, `round_profile.csv` and
`listening_prefix.json` retain timing and publication evidence. Motion runs also
save `events.json`, `playback.json`, `heard.json`, `analysis_batches.json` and
`result.json`. Detailed TTS progress uses the `tts_streaming` logger at DEBUG
level rather than unconditional stdout output. Enable that logger when needed;
structured profiles remain available without console logging.

Listener shutdown drains remaining recognition and text batches. Join timeouts
and partial delivery are failures rather than successful completion. Bridges
publish chunk files with atomic replacement and only report a complete stream
when all expected chunks have been copied.

Paid experiment dispatch must use the existing durable ledger and cumulative
caps. Failed and in-flight reservations remain accounted. Existing launch
wrappers are single-use run records: do not reuse their run IDs, overwrite their
manifests, reset the ledger or treat old budget metadata as a new authorization.

## Validation and limits

The two complete v69/v70 motions finished 12/12 turns within 5% of their time
targets. The frozen baseline includes the 190-test targeted regression result.
Cleanup additionally checks the broader offline suite; historical test fixtures
must use current authoring interfaces and stage-specific prompt rules.

Known baseline limits:

- v70 cold-start first audio took 14.29 seconds, with a 0.817-second gap before
  the first body paragraph. Later turn handoffs took 0.252–0.338 seconds.
- Short initial inputs can wait 60 seconds in the configured text batcher. Three
  v70 batches therefore completed analysis about 64–66 seconds after the audio.
- Pending-prefix TTS and late-body adoption have offline concurrency coverage;
  the two complete runs did not exercise these boundary cases.
- These are server-paced experiments, not browser/microphone acceptance tests,
  paired performance ablations or a broad semantic-quality evaluation.

Run offline regression tests in the configured development environment:

```bash
PYTHONPATH=src python -m pytest -q tests
```

The core tests use fixtures and injected clients. Tests for optional local
experiment harnesses explicitly skip when those harnesses are absent; core
speech, planning and web handoff tests still run. No experiment archive, model
weights or real provider credentials are required for these offline tests.

## Web app integration

The separate `TreeDebater-app` repository defaults new sessions to `flat_tree` + `listening_prefix`,
with reviewed prefix pre-synthesis, streamed body revision and snapshot delivery.
The API validates planner/output compatibility through `validate_tree_speech_mode`
before model preparation. AI and microphone input share ordered text batching; AI
ASR uses a separate worker and only consumes browser-acknowledged audio.

At the audio boundary, the app can start the next AI turn before final ASR or
analysis completes. Its worker retains serial analysis but runs `generate_handoff`
on a separate lane, routing IPC replies by request ID. The adapter passes a
matching immutable snapshot as `listening_handoff`, complete recognized history
through `listening_recognized_input`, and the ordered analysis barrier through
`listening_input_completion`. Native snapshot delivery publishes body audio before
that barrier; mutable state is merged only after it is released. Missing snapshots
use the ordinary complete-input fallback. Turn/attempt identity checks, failure
propagation and worker termination prevent stale input from crossing handoffs.
See the Early turn handoff section of the application repository README. Set
`DEBATE_APP_ROOT` to that checkout to include `tests/test_web_handoff.py` in
engine test runs; without it, only this cross-repository module is skipped.
