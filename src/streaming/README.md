# Streaming speech generation and listening

The `listening_prefix` path prepares speech while the opponent is speaking. A
reviewed first paragraph can start playback before complete ASR and final tree
analysis finish. The remaining body uses complete recognized input, a fixed
writing task, one streamed revision and the native duration/TTS loop.

## Fixed reference and current source

The [v70-20261010 baseline](../../experiments/incremental_planning/baselines/v70-20261010/baseline.json)
contains the exact experimental source archive, settings, environment inventory,
validation results and known limits. The current source removes unused experiment
branches and console diagnostics; the baseline archive remains unchanged.

Check the archive without API calls:

```bash
python experiments/incremental_planning/baselines/v70-20261010/verify.py
```

Use `--workspace /path/to/TreeDebater` to compare a checkout with that original
snapshot. A cleaned checkout intentionally differs from the archived baseline.
The archive does not include credentials, the local API gateway or model weights.
Provider model aliases are not a guarantee of identical future model outputs.

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

| Responsibility | Modules |
| --- | --- |
| Preparation, prefix review and handoff | `listening_prefix`, `overview_planning`, `overview_review` |
| Stable body inputs and reuse | `body_task`, `body_plan`, `body_feedback`, `body_revision`, `body_audio` |
| Listening context and sources | `planning`, `branch_planning`, `planning_view`, `listening_evidence`, `clash_records` |
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
PYTHONPATH=src:debate-app/backend python -m pytest -q debate-app/backend/tests
```

[Historical design notes](../../docs/history/streaming-design-notes.md),
[development results](../../docs/history/streaming-development.md) and the
[experiment journal](../../docs/history/process.md) preserve earlier alternatives.
Their old settings and instructions are not the current runtime contract.
