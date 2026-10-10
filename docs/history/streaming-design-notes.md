# Historical streaming design notes

Superseded designs and experiment observations; not current configuration guidance.
See [the current guide](../../src/streaming/README.md).

# Streaming speech generation, playback, and listening

TreeDebater has two independently configurable streaming paths: **output** generates
speech audio paragraph by paragraph in [`../tts_streaming.py`](../tts_streaming.py),
and **input** transcribes delivered audio and updates the listener's debate tree in
[`env.py`](env.py). [`overlap.py`](overlap.py) connects them so later audio can be
prepared while earlier audio is being delivered and analyzed.

In the default `full_script` mode the initial speech is generated in full before the TTS pipeline starts. Here,
“streaming” means incremental publication of completed audio chunks. Each TTS request
returns a complete MP3 candidate; this path does not consume LLM token deltas or play
incoming audio bytes directly.

Run everything from **`TreeDebater/src`** so the `streaming` package and sibling modules (`env`, `agents`, `utils`, …) resolve correctly. Alternatively, put `src` on `PYTHONPATH` and run from the repo root.

Current planning modes are `legacy` (default), `end_of_turn`, `linear`,
`adaptive_linear`, `branch_tree`, and `flat_tree`. Current speech modes are
`full_script` (default), `overlap_prefix`, `listening_prefix`, and `incremental`.
Removed experimental modes are rejected by configuration validation. Historical
experiment records and source snapshots remain available for reproduction.

## Version-bound first-paragraph review

`listening_prefix_initial_review_enabled` defaults to `true`. Despite the legacy
setting name, each distinct opening or repaired replacement must pass semantic
review before TTS. Verdicts are cached by exact text, framework and authoritative
review context. Concurrent requests for the same version share a Future; endpoint
fallback reuses that verdict. A malformed result permits at most three review
calls total, independent of the bounded speech repair loop. A valid rejection
returns immediately. A repaired text receives its own review.

Prefix audio synthesis and body drafting run independently. Their result Futures
remain bound to the exact opening, including across freeze. Handoff prefers the
newest approved opening with completed audio, retaining older approved versions
while a replacement is being checked or synthesized. It never waits for a newer
version when a ready one is available. If there is no ready version, an already
running synthesis can still be transferred; first preparation can add latency.
Complete-ASR body feedback starts without waiting for that synthesis. Only the
revision's duration allocation waits for the published prefix's decoded duration.
A failed or mismatched transferred audio result releases duration waiters before
worker cleanup and never triggers a duplicate prefix synthesis.
An opening explicitly rejected by a later review cannot serve as a fallback.
Semantic approval is against the recorded heard-input snapshot, not unheard final
words. Changing the initial review policy adds bounded background model calls;
the existing per-turn call and rewrite caps still apply.

The observer stores its latest completed context separately from prefix review
context. Freeze stops speculative dispatch but still accepts completed same-turn,
same-stage analysis snapshots. After complete ASR arrives, delivery rechecks the
transferred body Future without blocking and adopts a newer matching draft that
is already complete. The body task then captures the latest available context;
pending analysis and unfinished drafts are never awaited. Draft,
context and prefix-review stamps are recorded separately. That task remains
immutable during feedback, revision and playback; later analysis updates cannot
replace its records mid-speech. Close stops all updates and joins audio workers.

Traces retain `initial_prefix_review` for compatibility and include `versions`
with each checked text, context stamp and verdict. `body_context_snapshot` records
the actual context captured for the body task. `listening_prefix_review_enabled`
remains the legacy repeated-review switch. Disabling both review switches retains
the explicit local-validation-only mode.

## Shared TreeDebater capabilities

TreeDebater owns selection, retrieval, audience feedback, revision and committed
debate state. Streaming schedules the first paragraph, its review and freezing,
ongoing body updates, early feedback/revision and audio delivery. The initial
listening request still writes the first paragraph and body together; later body
work and final analysis can overlap playback of the approved first paragraph.

`utils/speech_context.py` owns speaker-labelled history and material projection.
`utils/prompts/speech_generation.py`, `speech_revision.py` and
`audience_feedback.py` own each task's authoring and output contract. Native
whole-speech templates retain their original text. Body revision uses the v55 B
evidence-rewrite prompt: a concise writing task followed by the assigned context,
frozen prefix, remaining word target, full unpublished draft, feedback, allocation
and compact evidence excerpts with selection reasons. It asks for substantive
source-supported rewriting and returns only the remaining spoken body. Early and
committed body revisions use the same formatter. The old `streaming.body_revision`,
`body_feedback` and `rehearsal_selection` imports remain compatibility exports.

Drafting goes through `_prepare_stage_prompt(..., speech_snapshot=...)` and
`_get_response`, using a frozen request and the injected helper transport. Helper
transport retries and budget enforcement remain authoritative; the adapter does
not retry a rejected dispatch. Provider-reported authoring and isolated audience
costs are recorded even when a speculative result is discarded. External budget
ledgers remain responsible for complete accounting across provider retries.

Custom overrides of the native generation, feedback or revision hooks default to
the serial generation path. A subclass can explicitly set
`speculative_speech_safe = True` after ensuring its hooks consume snapshots, forward
the optional execution arguments and do not mutate debate state in background
work. This also requires isolated audience extensions to be safe. State-dependent
extensions remain usable through the serial path.

Two independent options preserve the distinction between capabilities and speed
choices:

- `DebaterConfig.claim_selection_strategy`: `native` (default) runs the original
  framework selection with history and enabled rehearsal trees; `saved_scores`
  selects the existing saved-score ranking implementation in `utils/claim_selection.py`.
- `OutputConfig.audience_feedback_mode`: `full` (default) retains comprehensive
  analysis; `compact` requests at most three critical issues and 180 English words.
  Both preserve enabled retrieval feedback and native evaluation criteria.

The listening preset and live harness explicitly choose `saved_scores` and
`compact`. App sessions accept the first option at the top level and the second
under `streaming.output`. Changing `speech_mode` alone selects neither shortcut.

Feedback reuse includes the audience configuration, feedback mode and retrieved
material. Retrieval caching follows the actual selected query, including in flat
mode. Body updates retain the word threshold and pending-input coalescing; changed
materials with unchanged text, transcript corrections or a changed prefix bypass
the threshold. Background work returns values; conversation, evidence consumption
and delivered speech are committed by the owning turn.

Offline tests cover extension hooks, immutable inputs, feedback modes, retrieval
invalidation, shared prompts, body cadence, audio handoff and application settings.
They do not establish new live latency or model-quality measurements.

Current listening publication and preparation behavior:

- `claim_generation()` prepares ranked claims, evidence and enabled rehearsal
  indexes through `TreeDebater.claim_selection()`. The live harness uses this
  same lifecycle. Repeated listening updates reuse the preparation.
- Planning, drafting and first-paragraph review use the native selected
  `evidence_pool` (top 10), rather than all `high_quality_evidence_pool` candidates.
  Raw documents are excluded from planning; shared writer sources appear once.
- With `listening_prepare_evidence=true` (default), each listening batch offers
  its response plan and unused evidence to a coalescing background worker. The
  first selection calls the original `_select_revision_evidence` on the full
  pool. Completed choices feed later body drafts and final revision. Incremental
  choices exclude both selected IDs and `used_evidence`; only novel locally
  matched needs reach the native selector, with at most
  `listening_evidence_candidates=20` candidates and short summaries of retained
  evidence. Final feedback can request an unmet need already heard earlier.
  Term matching is a routing heuristic, not semantic evidence validation; the
  final writer/reviewer still checks applicability. Changed/removed documents
  cannot be reused, and corrected listening plans retire choices whose matched
  topic disappeared. Set the option to false for the original endpoint selection.
- Evidence selection has no fixed round-count limit. Counters record work only;
  final-feedback supplements are not blocked by earlier selection counts.
  Native provider retries remain subject to their existing limits; background
  API requests still share `listening_prefix_max_calls`, and actual paid calls
  remain subject to the experiment's monetary budget guard.
  Freeze does not wait for pending selection or adopt late results. The
  `prepared_evidence` trace records candidate, reused and added IDs and timing.
- Supplemental revision evidence still uses the native selection interface.
  Its list-compatible result now retains the selector's `analysis` separately
  from unchanged source dictionaries. `utils/evidence_material.py` carries these
  annotations through listening caches and prepares local verbatim sentence
  windows for current draft passages, retaining adjacent context. Duplicate
  sources merge, and different passages receive material before repeated sources
  for one passage. Material size scales with the speech word budget; there is no
  fixed evidence-item count or extra summarization API call. Full documents stay
  in the evidence pool. Selection reasons and passage matches are explicitly
  writing hints, not factual sources; unsupported numeric metadata is omitted.
  Missing dates/credentials are not inferred. Early and committed revisions use
  the same formatter, and native revision also receives compact material. Shared
  revision rules now allow supplied findings to support individual reasoning
  steps and permit factual additions without changing the speaker's stance.
  Complete-ASR work shares its selected evidence and revision with final handoff
  only when the transcript, draft, available evidence and writer request match.
  Speculation does not update `used_evidence`; committed revision does.
- The initial first paragraph and body are written together. Under the default
  version-bound policy, each distinct draft or repaired replacement receives a
  semantic check before TTS. Rejection can regenerate the unpublished pair through
  the bounded repair path. Verdicts certify only their exact text and listening
  context; malformed verdicts permit up to three attempts on that same version.
  Malformed/truncated speech JSON uses the existing one-repair allowance and
  cannot publish partial text. Final-ASR additions are handled by the body.
- Early handoff publishes the unchanged prepared opening without waiting for
  complete ASR. `listening_recognized_input` supplies full history to body
  feedback/revision, which can overlap remaining tree analysis. Mutable player
  state remains owned by the observer until final input processing completes.
  Without a valid prepared handoff, generation still waits for complete input.
  Traces mark the opening context as `prefix_input_scope: listening_snapshot`;
  the opening may not reflect an opponent qualification in the final unheard words.
- Drafts and revisions inherit the writer model, temperature, `max_tokens`, system
  prompt and native stage strategy. Semantic review and evidence selection retain
  the configured helper model. Non-full speech modes require `flat_tree`.
- The writer no longer returns `target_ids`; target selection belongs to the
  planner. Legacy writer metadata is ignored, while actual-text review remains
  mandatory. Deferred bodies use the same citation/reference cleanup as ordinary
  TTS, including matching speculative body audio after cleanup.
- `verify_rewrites`, the unused TTS review function, final-body review module and
  constant final-body trace fields have been removed. TTS rewrite audits are
  saved whenever an output directory is provided; they are not semantic verdicts.

The v46 simplification and earlier experiments below describe historical behavior.
The v49 retrieval run exposed evidence duplication and revision reuse failures;
the v50 run records validation of the current changes under the same USD60 cap.
V50 completed five live turns; its final turn stopped on truncated model output.
An isolated closing replay exercised the format repair and generated audio, but
manual inspection found reversed stance despite an accepting semantic review.
The validation therefore remains failed; the six generated speech artifacts do
not establish a successful continuous or semantically correct match. See
`experiments/incremental_planning/listening-motion-live-v50-retrieval_validation.json`.

## End-to-end behavior

```mermaid
flowchart LR
    A[Full speech text] --> B[Paragraph splitting and refinement]
    B --> C[TTS candidates and chunk selection]
    C --> D[chunk_000.mp3, chunk_001.mp3, ...]
    D --> E[Bridge to turn watch directory]
    E --> F[Simulated playback and shared cursor]
    F --> G[Whisper on delivered audio]
    G --> H[Buffered transcript]
    H --> I[Listener debate-tree updates]
```

In an overlap turn, the speaker runs generation and TTS on a background thread.
A bridge thread publishes completed chunks into the turn's watch directory. The main
thread consumes them in index order, appends them to `continuous_audio.mp3`, and
**simulates playback by sleeping for their durations**. It advances a shared cursor
as audio is delivered; it does not send audio to a speaker device.

The listener runs on another thread and reads only through that cursor. ASR and tree
analysis are sequential within the listener thread, but overlap with speaker generation
and playback. A slow listener can accumulate a backlog. The turn waits for playback
and listener draining before returning, so the next speaker's turn starts afterward.

## Streaming output: paragraph TTS and refinement

[`Debater.post_process()`](../agents.py) calls
[`convert_text_to_speech_streaming()`](../tts_streaming.py) when `streaming_tts` is
true, `time_control` is true, and the speech has a positive time budget.

1. Strip references/citations and subtitles from the audio input, split on blank lines,
   and merge paragraphs shorter than 30 words with a neighbor.
2. Assign the current paragraph a duration target proportional to its character count
   among the remaining paragraphs, using the remaining budget.
3. Generate **chunk 0 directly**, without length refinement, so the first audio can be
   published as soon as that TTS request completes.
4. For subsequent chunks, estimate duration on CPU from word count (0.46 seconds per word). Refinement workers ask
   `gpt-5-mini` to rewrite the paragraph to a target word count, with preceding speech
   and following-paragraph context. They submit TTS candidates to a shared pool with
   up to eight concurrent TTS requests per chunk.
5. An estimate within tolerance lets a worker nominate its completed TTS candidate.
   Otherwise, after waiting up to the preceding chunk's audio duration, the pipeline
   selects the completed candidate closest to the current duration target. If none has
   finished, it waits for a result. It can make an additional TTS request with adjusted
   speed, clamped to 0.85–1.15, to improve the duration match.
6. Publish the selected chunk, account for audio duration and measured overrun in the
   remaining budget, and proceed. Finally concatenate the selected audio and return
   the **revised spoken paragraphs** plus the original references separately.

Some refinement starts early: a nonfinal paragraph two positions ahead is prepared
when it is at least twice as long as the intervening paragraph or at least 1,000
characters. With at least three paragraphs, the final paragraph also starts preparation
two positions ahead. Early and normal workers share candidates for their target chunk.

The previous chunk's duration is a refinement waiting budget, not a signal from the
actual playback cursor. Candidate completion, retries, and speed adjustment can extend
processing beyond that wait. Overall duration targets are therefore approximate.
`streaming.output.enable_early_cut` is a YAML setting (also available through the Python API), disabled by default, that splits
an oversized later paragraph into a head and tail before refinement.

Canonical output defaults live in [`config.py`](config.py) (`OutputConfig`): `tts-1`, voice `echo`, 10% duration
tolerance with a one-second floor, and a tighter 5% upper tolerance for the last chunk.
The TTS request currently slices each candidate to 4,096 characters; paragraph splitting
does not enforce that limit, so longer candidates can produce truncated audio.

## Modes and configuration

For `python -m streaming.overlap`, the flags apply per turn to the **speaker's**
`streaming_tts` and the **listener's** `streaming_listen`:

| Speaker TTS | Listener input | Behavior |
|---|---|---|
| Streaming | Streaming | Publish paragraph audio while later chunks are prepared; analyze delivered audio during playback. |
| Streaming | Batch | Publish paragraph audio during generation; regular `listen()` analyzes the returned speech text on the next turn. |
| Batch | Streaming | Finish the full MP3 first, split it into chunks, then analyze audio during simulated playback. |
| Batch | Batch | Finish the full MP3, simulate playback of split chunks, then use regular text-based listening. |

This table assumes `time_control: true`, a positive speech budget, and a TreeDebater
listener. The overlap environment bypasses its playback/listener machinery when the
listener is not a TreeDebater. With time control disabled, normal post-processing
returns text without generating the TTS audio needed by this path.

Edit [`../configs/overlap_debate_2.yml`](../configs/overlap_debate_2.yml) for a full
example. The shared settings are under `streaming`, and speech budgets are under `env`:

```yaml
env:
  time_control: true
  speech_budgets:
    opening: 240
    rebuttal: 240
    closing: 120

streaming:
  input:
    min_audio_seconds: 15
    min_text_words: 40
    poll_interval: 1
    audio_format: mp3
    max_audio_wait_seconds: 0
    max_text_wait_seconds: 0
    max_total_audio_seconds: 0
  playback:
    increment_seconds: 3
    listener_join_timeout: 300
  output:
    model: tts-1
    voice: echo
    refinement_model: gpt-5-mini
    max_refinements: 10
    early_max_refinements: 3
    max_parallel_tts: 8
    min_chunk_words: 30
    tolerance_ratio: 0.10
    last_chunk_upper_tolerance_ratio: 0.05
    min_tolerance_seconds: 1
    enable_early_cut: false
    early_cut_ratio: 1.25
    ratio_prestart_threshold: 2
    abs_prestart_chars: 1000
    speed_adjust_min: 0.85
    speed_adjust_max: 1.15
    refine_deadline_margin_seconds: 2
    speed_adjust_min_slack_seconds: 4
    max_chunk_chars: 900
    target_chunk_seconds: 40
    min_stream_chunks: 3
    max_stream_chunks: 8
    normalize_seams: true
    seam_head_ms: 60
    seam_tail_ms: 250
    seam_fade_ms: 10
  posthoc:
    split_mode: fixed
    chunk_seconds: 10
    silence_window_seconds: 0.7
```

Keep `streaming_tts` and `streaming_listen` on each debater. These flags still select
which paths run; shared output settings do not turn streaming on automatically.
`streaming.output.model` and `refinement_model` configure the direct OpenAI clients
inside the TTS pipeline, independently of each debater's `model` and `helper_model`.

Precedence is **explicit CLI override → YAML → typed defaults** in `config.py`.
For example, `--min-audio-seconds 8` overrides `streaming.input.min_audio_seconds`.
`--min-playback-increment` maps to `streaming.playback.increment_seconds`; existing
flag names continue to work. Output/refinement settings and speech budgets are configured
in YAML or through the Python API. Unknown settings, invalid types, nonpositive batch
sizes/timeouts, and inconsistent speed bounds are rejected before the debate starts.
Zero disables the optional audio/text wait and total-audio caps.

Old YAML files remain valid. Without explicit input settings, overlap mode retains
15-second/40-word batching; `streaming.env` and the listener demo retain their legacy
30-second/50-word defaults. Missing speech budgets retain 240/240/120 seconds.
The debate entrypoints save the **resolved** `streaming` section and speech budgets in
the run JSON's `config`, including CLI overrides. Watch-only mode logs the resolved
settings. The batch runner's per-case YAML files remain independent: modifying a template
affects configs generated afterward, not previously generated files.

Python TTS callers can pass `config=OutputConfig(...)` to `run_pipeline()` or
`convert_text_to_speech_streaming()`. Explicit existing keyword arguments such as
`voice` override the corresponding output config value. Config is passed to each worker;
no process-wide settings are mutated.

`streaming.env --debate` has different scheduling: it waits for speaker generation
and the full MP3, then feeds post-hoc chunks at approximately real-time intervals.
It starts an audio listener for each TreeDebater opponent independently of the
`streaming_listen` flag. Use `streaming.overlap` for live output/input overlap.

In overlap mode, `--chunk-seconds` (default 10) controls **post-hoc fixed splitting**,
not streaming TTS paragraph sizes. `--min-audio-seconds` (default 15) controls ASR
batching, `--min-text-words` (default 40) controls tree-update batching, and
`--min-playback-increment` (default 3 seconds) controls cursor advancement. Remaining
audio and text are drained at turn completion even below those batching thresholds.
`--max-audio-wait-seconds` applies to chunk-mode input; cursor mode currently uses
its audio threshold and end-of-stream drain instead.

## Output files and timing

For a speech named `{type}_{stage}_{side}`, the log session's output directory contains:

```text
{type}_{stage}_{side}.mp3
{type}_{stage}_{side}_chunks/
    chunk_000.txt             # selected revised paragraph
    chunk_000.mp3             # selected audio; subsequent indices are 001, 002, ...
    chunks_final.txt          # all selected paragraphs
    final.mp3                # combined audio
    chunk_profile.csv        # per-chunk targets, candidates, refinement and timing
    round_profile.csv        # aggregate audio, timing and remaining budget
```

The turn watch directory is `{watch_root}/{stage}_{side}/`. The bridge maps
`chunk_000.mp3` to `{side}_chunk001.mp3` and continues with one-based indices.
Playback publishes `continuous_audio.mp3` there for the cursor-gated listener.

TTS profiles describe generation/refinement timing. The overlap logs separately record
`SpeakerWorker`, `TtsChunkBridge`, `PlaybackMain`, and `StreamingInputEnv` events for
actual thread activity, chunk waits, playback, ASR, and tree updates. These are distinct
measurements: a TTS duration estimate is not observed end-to-end turn latency.

---

## Modules

| Module | Role |
|--------|------|
| **[`config.py`](config.py)** | Typed defaults, YAML validation, CLI overrides, and resolved run configuration. |
| **[`../tts_streaming.py`](../tts_streaming.py)** | Output pipeline: paragraph splitting, duration refinement, parallel TTS candidates, incremental chunk publication, combined MP3, and profiling. |
| **[`../agents.py`](../agents.py)** | `Debater.post_process()` selects streaming or batch TTS and records the returned speech text. |
| **`env.py`** | `StreamingInputEnv` (watch directory → Whisper → `TreeDebater._analyze_statement`), `StreamingDebateEnv` (full debate with a streaming listener each speech turn, post-hoc MP3 split into chunks), path helpers (`tts_outputs_dir_from_log`, …), and the main CLI (`--debate` or watch-only). |
| **`overlap.py`** | `OverlappingStreamingDebateEnv`: playback-driven main thread, optional `streaming_listen`, optional live **streaming TTS** chunk bridge, post-hoc fallback when no live chunks were copied. |
| **`bridges.py`** | **`run_live_chunk_bridge`**: poll `log_files/<N>_outputs` for stable full-speaker MP3s, split with pydub, write `{side}_chunkNNN.*` into a watch dir (for demos next to `env.py`). **`run_streaming_tts_chunk_copy_bridge`**: copy stable `chunk_NNN.*` from streaming TTS output into `{speaker_side}_chunkNNN.*` for overlap runs. |
| **`chunk_audio.py`** | Split audio with fixed duration or silence detection, **`stream_chunks_to_directory`** (real-time or burst pace), **`clear_watch_chunk_files`**, and a small CLI for one-off file simulation. |
| **`run_listen_demo.py`** | Standalone **listener + bridge** demo: start `StreamingInputEnv` on a TreeDebater side, optionally run the live MP3 bridge against `N_outputs`, or one-shot chunk a file. |

The package **`__init__.py`** does **not** eagerly import heavy modules. Use submodule imports (see below), or lazy access such as `import streaming` then `streaming.env` (same effect as `import streaming.env`).

---

## Command-line entrypoints

From `TreeDebater/src`:

```bash
# Full debate with streaming listener (non-overlap): same YAML idea as env.py
python -m streaming.env --debate --config configs/base_st_io.yml

# Watch-only: one TreeDebater listens on a directory of chunk MP3s
python -m streaming.env --config configs/base_st.yml --watch-dir /tmp/watch --debater-side for

# Overlap + playback-driven timing (see configs/overlap_debate_2.yml)
python -m streaming.overlap --config configs/overlap_debate_2.yml

# Split one file into chunks and write into a watch dir (pydub-only logic)
python -m streaming.chunk_audio --audio-file path/to/speech.mp3 --watch-dir /tmp/watch

# Live bridge from log_files/<N>_outputs + listener (run while env.py produces TTS)
python -m streaming.run_listen_demo --config configs/base_st.yml --watch-dir /tmp/watch --debater-side for
```

Use `--help` on any of the above for full flags.

---

## Python imports

Prefer explicit submodule imports:

```python
from streaming.config import OutputConfig, StreamingConfig, resolve_config
from tts_streaming import convert_text_to_speech_streaming, run_pipeline
from streaming.env import StreamingDebateEnv, StreamingInputEnv, StreamingInputConfig
from streaming.env import opponent_side, tts_outputs_dir_from_log
from streaming.overlap import OverlappingStreamingDebateEnv
from streaming.bridges import run_live_chunk_bridge, run_streaming_tts_chunk_copy_bridge
from streaming.chunk_audio import split_audio, stream_chunks_to_directory, clear_watch_chunk_files
```

TreeDebater’s `ouragents.py` uses `StreamingInputEnv` / `StreamingInputConfig` from **`streaming.env`** for `start_streaming_listen`.

---

## YAML knobs (debate configs)

These are documented more fully in the main TreeDebater README; at a glance:

- **`streaming_tts`** (speaker): use the streaming TTS pipeline; overlap mode can bridge `chunk_NNN` files when combined with `time_control` and the overlap env.
- **`streaming_listen`** (listener): in overlap runs, start `StreamingInputEnv` on a background thread and, after successful ingestion, set `tree_via_streaming` on the turn record so `listen()` can avoid duplicating full `_analyze_statement` when the tree was already updated from audio.

---

## Dependencies

- **`../tts_streaming.py`**: `openai`, `mutagen`, `pydub`, and the CPU word-rate estimator in `utils/speech_duration.py`; TTS and refinement use `OpenAI()` directly.
- **`env.py` / `overlap.py`**: `pydub`, `openai` (Whisper), `yaml`, TreeDebater `env` / agents / utils (API keys as elsewhere).
- **`chunk_audio.py`**: `pydub` only.
- **`bridges.py`**: `pydub`, `utils.tool` logger for the streaming-TTS copy bridge; live MP3 bridge prints to stdout for simple demos.

## Completion and failure handling

Stop the listener only after the producer/playback finishes. `StreamingInputEnv.stop()`
signals end of input: the listener makes a final scan, transcribes remaining audio even
below the normal duration threshold, and flushes buffered text. In cursor mode it only
reads audio through the playback cursor, including during shutdown.

`TreeDebater.stop_streaming_listen()` joins the listener and returns whether ingestion
completed successfully. A join timeout raises `TimeoutError`. Overlap turns set
`tree_via_streaming` only on successful ingestion; otherwise the next regular `listen()`
retains full-text analysis.

For the TTS copy bridge, `stop_event` means the producer has finished writing. The bridge
then makes a final copy pass and sets `drained_event` only for a fully copied contiguous
chunk sequence. A partially delivered live stream fails the turn instead of silently
omitting the remaining speech. TTS chunks, bridge copies, and continuous playback audio
are published with atomic renames. The TTS wrapper returns the revised spoken paragraphs
as the speech text, retaining the original reference section separately.

### Shared draft-length and duration settings

All speech paths use `utils/speech_length.py`, which reads the two existing
settings in `utils/constants.py` at use time:

```python
LENGTH_MODE_FOR_DRAFT = "phonemes"  # words / syllables / phonemes
TIME_MODE_FOR_STATEMENT = "fastspeech"  # time / fastspeech / openai
```

The first setting controls the measurement of unpublished drafts: prompt
instructions, overview limits and provisional body limits. Existing `max_words`
and prompt word budgets remain word-denominated; syllable/phoneme measurements
are converted using `WORDRATIO` (for example, 100 word equivalents correspond to
450 phonemes). Draft budgets are coarse conversions, not duration guarantees.

Listening prewriting uses the upcoming stage's `speech_budgets` (the app's
`budgets`), defaulting to 240/240/120 seconds for opening/rebuttal/closing. The
initial draft budget includes the overview; later body updates subtract its
measured draft length. Custom closing budgets are used directly without halving
a separate body limit. `listening_body_words` is accepted only for compatibility.
Supplemental evidence selected after complete ASR is included in early revision,
even if feedback and duration checks already pass. Final handover reuses matching
selection and revision work, and only committed work marks evidence used.

Listening authoring uses role-labelled speech history through
`HelperClient(history_messages=...)`: our delivered speech is an `assistant`
turn; an opponent turn is a `user` message headed `Opponent's ... Statement`.
`authoring_history` updates the current opponent turn with partial/corrected ASR
without duplicating it. Full history/transcripts are omitted from the final
source JSON. Draft and repair calls explicitly request JSON; body revision
explicitly requests plain speech, with the native stage strategy and meaning
constraints but no competing Plan/Statement/reference output layout. The live
harness forwards the same messages, and early revision reuse binds this history.

The second setting controls estimated seconds for body refinement, listening
body acceptance and TTS candidate estimates. `time`
uses the CPU word-rate estimate from `utils/speech_duration.py`; only this mode
may substitute an observed seconds-per-word rate. `fastspeech` uses the existing
shared FastSpeech2 duration-only wrapper, including its existing calibration,
and is never replaced by a word estimate after audio arrives. `openai` reuses the
renderer's configured model, voice and client factory (including an injected
budget transport). Long inputs are measured in bounded pieces without dropping
text. Direct `LengthEstimator("openai")` calls require an explicit
`audio_duration` callback; they cannot construct an independent unmetered client.
Use `tts_streaming.duration_estimator(output_config)` for the renderer binding.
Estimator errors retain their original cause across TTS worker boundaries.
Model weights load only when their estimator is selected.

Scalar text passed to `LengthEstimator.query_time` returns a scalar; list/tuple
input always returns a list, including empty and singleton batches. Phoneme
counting lazily reuses one G2P instance with serialized access. English phoneme
mode requires the NLTK `cmudict` and POS-tagger resources; with current NLTK,
install `averaged_perceptron_tagger_eng` as well as the legacy
`averaged_perceptron_tagger` resource expected by g2p_en. The ordinary
`Debater` and CLI `HumanDebater` entry points also use the shared draft interface.
Preparatory closing bodies use half the configured body word budget consistently
in prompt instructions, payload and validation.

Listening body checks use a 10% estimated-seconds band and at most two revisions;
the closest candidate within 20% may be retained after both miss the nominal
band. The prior separate 0.42-seconds-per-word check has been removed. Actual
TTS audio duration still controls playback accounting and the separate 15%
experiment acceptance threshold. Historical `fs` profiling columns denote the
selected duration-estimation step, not necessarily FastSpeech.

All complete-script and deferred-prefix delivery reuses TreeDebater's existing
`split_into_chunks` → `_merge_short_chunks` path. It preserves authored paragraphs,
splits oversized paragraphs at sentence boundaries, and merges short neighbours.
Deferred delivery leaves the published prefix untouched and passes the remaining
actual audio budget to the same splitter for the body. `adaptive_delivery` controls
refinement/deadline behaviour; it no longer selects a different chunking algorithm.

The live listening harness and `gemma-flat-listening-prefix.yml` use existing
settings `min_chunk_words: 50` and `max_stream_chunks: 5`; the other splitting
settings retain their baseline values (`max_chunk_chars: 900`,
`target_chunk_seconds: 40`, `min_stream_chunks: 3`). This encourages roughly five
chunks per speech without imposing an exact count. `max_stream_chunks` bounds the
fallback target when paragraph breaks are sparse; it is not a hard total-chunk cap,
and the independent prefix adds one chunk. Normal paragraph boundaries still take
priority. `first_chunk_seconds` controls prefix length, while `later_chunk_seconds`
now only applies to the separate incremental-argument speech mode.
FastSpeech still estimates duration for refinement, and decoded audio controls
accounting. Historical snapshots and audio results remain unchanged.

Refinement reserves up to `refine_deadline_margin_seconds` for delivery, capped at
half the preceding chunk's duration. Speed resynthesis requires at least
`speed_adjust_min_slack_seconds` of playback time remaining. Seam normalization runs
before duration accounting and chunk callbacks; disable it with `normalize_seams: false`.
The default duration backend is `fastspeech`, matching the baseline documented in
[`experiments/stage_summary.md`](../../experiments/stage_summary.md), and applies
consistently to these paths. Historical run manifests retain their original settings.

Optional length edits have an immutable `EditScope`: the owned original segment,
the finalized preceding text, the following source segment, and a context revision.
The original stays authoritative across repeated expansions/compressions. Neighbours
are read-only context; their propositions cannot be imported to meet a time target.
The obsolete `verify_rewrites` switch has been removed. Optional delivery edits
use the configured refinement model and do not dispatch a semantic reviewer.
Writing and synthesis registration share one frozen scope. Context changes
invalidate edited candidates; source audio remains available as a fallback.
Identical audio may be reused after rebinding the proposal to the current context.
Deadline fallback cannot select obsolete edits. `rewrite_audit.json` is always
saved with output chunks and records source scopes, proposal/audio reuse, obsolete
candidates and publication. These records do not establish semantic correctness.

`listening_parallel_endpoint_revision` (default false) starts frozen-body feedback
and revision from complete ASR while final tree/planning analysis drains. The first
paragraph must pass publication review before it can play; the shared body request
and exact revision binding still control reuse. The listener retains ownership of
mutable tree and planner state until final analysis completes.

With `listening_body_snapshot_delivery` (default true), a valid early handoff
using single-pass parallel revision and cached evidence publishes the body from
complete ASR and a fixed `BodyTask`. Body publication does not wait for tree or
planning analysis. The task freezes its history, draft, framework, derived hints,
evidence and writer settings; later derived updates apply to subsequent work.
`listening_recognized_input` must return the final immutable history. Each body
chunk checks that history still matches; a changed recognition result stops
publication. Analysis completion is reconciled after audio generation, before
conversation updates and evidence bookkeeping. A differing final analysis history
is a contract violation and fails the turn; it cannot retract audio already sent.
Traces save `body_publication_snapshot`, its hash, and publication/state-sync times.
Setting this option false retains final-analysis reconciliation before body
publication. Cold starts, nonparallel revisions and endpoint evidence supplements
also retain that path.


First-paragraph review now returns three independent checks: `stance_ok`,
`ready_to_speak_ok`, and `latest_input_ok`. `stance_assessment` records the position
actually endorsed by the spoken paragraph (`for`, `against`, `neutral`, or
`unclear`), an exact draft quotation and a short explanation. Neutral definitions,
questions and greetings remain allowed; unclear stance is inconclusive. Review
must evaluate the paragraph even when its framework label matches the assigned
side. Existing quote/qualification/attribution checks and bounded format repair
remain in place.

Before a reviewer request, a local check rejects an explicit opposite framework
label (`for`/`support` versus `against`/`oppose`, ignoring case and trailing period).
It does not infer stance from keywords in free-form prose. The validator also
rejects an opposite expressed-side assessment even if the model sets `stance_ok`
true. Both rejection paths enter the existing one-semantic-repair-and-review
flow before first audio. A handoff must retain a passed independent stance check
and cannot carry a conflicting framework. Model interpretation of actual prose
remains fallible; these are guardrails, not proof of semantic correctness.

Early-handoff overview drafting and repair share the same source-independent
instructions and an explicit `target_ids=[]` requirement. The initial JSON example
also uses an empty array; ordinary target-based overviews keep their existing ID
rules. A nonempty handoff target list produces a specific format diagnostic. IDs
are not silently removed, and every corrected text still passes the semantic gate.
A format-only preparation failure (including malformed candidate JSON) permits one
additional preparation attempt for the same framework after newly heard speech.
Repeated identical input or tree-only changes do not trigger it. If that extra
attempt fails structurally, further attempts wait for a different framework;
semantic rejection keeps the existing failed-framework behavior. The shared call
cap, semantic rewrite limit, and freeze lifecycle still bound all requests.

The v32 body path uses `BodyTask`, a frozen value snapshot of the draft, immutable
opening, full ASR history, preparation feedback and retained exchange records.
Early and committed work share parsing, content review, revision decisions and
feedback composition. Feedback reuse requires equality of the complete task;
revision reuse additionally requires the exact canonical prompt, including the
word budget derived from published decoded prefix audio. Preparation records are
hints; the complete transcript remains authoritative. Live source validation
still runs per chunk, without repeating rehearsal retrieval. Listening body
revision directly builds its dedicated prompt instead of constructing and then
overwriting legacy tree-grounding prompts.

An oversized provisional body is retained with `needs_fit=true`, receives the
same preparatory review, and must undergo final revision before the existing
publication gate. This flag is consumed by the first revision only. Invalid or
empty drafts and prefixes remain rejected. Whole-body review points require an
explicit `replace`, `remove`, `qualify` or `add` action and a concrete correction;
placeholder edits (`empty`, `none`, `N/A`, `No changes`) fail validation. A clean
review returns empty arrays. These changes preserve review counts, the100-word
preparation cadence, TTS concurrency and the four separate review scopes.

The live motion harness separately configures `InputConfig.min_text_words=100`
for tree analysis and planning. This restores the native input environment's
ASR → text buffer → observer separation: whole ASR chunks accumulate until the
threshold, or `max_text_wait_seconds=60` elapses from the first buffered ASR result,
then a single ordered analysis job consumes that text. Set the wait to0 to disable
timeout submission. A final drain
marker on the ASR executor submits any remaining short batch, and handover waits
for all analysis jobs. Complete recognized history is published as soon as all
ASR results arrive, regardless of the analysis buffer. Recognition can continue
while a previous100-word batch is analyzed. `analysis_batches.json` records batch
membership, word counts, queue/start/end times, buffered duration and flush reason
(`threshold`, `timeout`, `final`); each heard
chunk points to its analysis batch. The existing `listening_body_update_words`
setting controls a later, separate provisional-body stage.

Listening planning can now return a bounded `body_plan` in the existing planning
response: issue, response action, point and relative weight, with overview-axis
and current-target indices. The server resolves indices to promised axis text
and node ID/version/source quotes. Drafting, whole-body feedback and revision
consume the same plan. Missing promised axes get explicit fallback entries;
withdrawn/changed target bindings and replaced-overview tactics are discarded.
Word allocations use the actual remaining body budget and sum to that budget.
Plans remain private tactics, subordinate to the complete opponent transcript.

Listening delivery no longer calls legacy `_prepare_stage_prompt`/opening claim
selection and then discards its prompt. Cold starts use the existing overview
and prepared claims; legacy full-script generation retains its normal selection.
Early review/revision includes the frozen plan; final delivery resolves the latest
plan after handover. Changes invalidate reuse, while unchanged plans preserve it.
This introduces no extra model stage, but larger planning responses and invalidated
speculation may change latency/cost. Structural coverage does not prove substantive
coverage or sound reasoning; actual quality still requires a live review.


Listening exchange records now add at most four connected response chains,
prioritizing targets selected by the body plan before other current targets.
Each chain includes the source target, our connected prior point/objection,
the latest current opponent reply along that path, up to two other current
replies to the same objection, and our latest direct response when present.
Source quotes and validated claim-owned conditions are retained without the
180-character clipping used by the older chronological overview. Omitted sibling
counts are explicit; an absent edge does not establish an unanswered argument.
Private and withdrawn/dependent historical nodes cannot become current chains.
The remaining question is an instruction to assess the reasoning, not a generated
factual assertion or stored resolution. The full transcript remains authoritative.

Selected chains enter the resolved body plan and existing drafting/feedback/
revision prompts. The final body task uses the final exchange snapshot, including
records in the canonical revision allocation, so changed exchanges invalidate
old feedback and revision reuse even when speech text is unchanged. Overview
review reuse still depends on its own inputs. No extra model-call stage is added.

The live motion harness now ranks the complete saved Gemma pool's group roots by
existing minimax score before applying `claim_pool_limit`; stable ties preserve
original pool order and missing/invalid scores sort last without invented values.
The top three roots seed the main case, and retained candidates with original
scores enter private planning context. It no longer overwrites the pool with the
old flat-motion outline or uniform scores. Full retrieval pools remain unchanged,
including source members and tree material. Ranking expresses an internal
preparation preference, not evidence reliability. `claims_for/against.json` records
original pool indices, selected claims and retained candidate scores.

The harness's timed text batcher runs independently of the ASR executor, so
already recognized short text can be submitted while the next ASR call is in
flight. Submission is serialized onto the same ordered analysis worker. A timer
generation prevents cancelled callbacks from flushing newer batches; final drain
closes the buffer before analysis handover, and failure/cleanup cancels pending
submission. The60-second setting limits buffer waiting, not ASR or analysis
latency, and does not restart on each new short chunk. Native StreamingInputEnv
keeps its existing timeout mechanism. No adaptive semantic scheduling was added.


Draft handover now separates generation from preparatory feedback. A valid body
is saved immediately with `feedback=null` and `feedback_status=pending`, before
requesting the optional preparatory review. Completed feedback creates a new
value with `feedback_status=completed`; failure or turn freeze preserves the raw
draft for the final whole-input review. Missing preparatory feedback is distinct
from a clean review and never grants publication permission.

Each body job has a future containing a serialized immutable draft value. A
handoff captures that channel separately from the serializable opening snapshot,
bound to the exact stage, incoming turn, prefix and framework. A draft request
already started before freeze may complete the future afterwards, but cannot
update frozen preparation state, dispatch more speculative feedback or publish
speech. No worker mutates a late value after transfer.

Delivery checks for a ready transferred draft before starting and again after
complete ASR, immediately before binding the early body task. Later completions
cannot change that task. The legacy endpoint path also rechecks after final
analysis, before deciding whether to draft from scratch. There is no extra wait for a slow draft and no block on first audio.
Failed, malformed, unavailable or differently bound values use the existing
fallback. A matching draft still receives the existing final-history body
feedback, applicable revision, and stance/attribution/conditions publication gate.
Early speculative cache reuse retains exact task/prompt matching. Trace fields
`body_transfer` and `body_preparation_source` distinguish a saved snapshot,
in-flight transfer and cold draft; preparation events distinguish a saved but
unreviewed draft from a rejected draft.

This changes neither the100-word/60-second input schedule nor planning timeout/
retry policy. Next live harness run ID is v34; the handover changes have passed
311 offline tests but have not been run against paid providers.

Native authoring instructions are now reused explicitly. `utils/prompts/authoring.py`
provides the configured debater system prompt and the original stage strategy
blocks shared with `opening.py`, `rebuttal.py` and `closing.py`. The legacy composed
stage prompts retain their exact previous text. Strategy blocks exclude legacy
whole-speech output templates, so streaming JSON/plain-text contracts, budgets,
paragraph structure and frozen-prefix rules remain explicit at each call.

| Request | System instruction | Stage strategy |
| --- | --- | --- |
| Cold listening body | Existing conversation system message | Shared strategy for the current stage, added to its independent body instruction |
| Speculative/endpoint overview and repair | Configured debater system | Upcoming speech stage |
| Preparatory body | System captured when its worker starts | Upcoming speech stage |
| Early body revision | System captured for speculative work | Body task stage |
| Final revision and publication-gate repair | Current configured debater system | Current speech stage |
| Review/planning/TTS segment editing | Existing task-specific instructions | Existing task-specific scope |

The helper itself remains neutral; authoring call sites explicitly pass `sys`.
Custom or explicitly empty system prompts are respected. Speculative revision
reuse now requires both user-prompt and system-prompt equality. A changed system
prompt forces the existing committed revision path to regenerate.

`tests/test_debate_prompt_reuse.py` uses the real HelperClient message builder and
a fake transport to inspect the actual model-bound message structures. It covers
all three stages, overview repairs, body drafts, revision, production observation/
endpoint wiring, custom instructions, shared strategy text and review isolation.
Parallel endpoint/revision tests cover changed-system invalidation. These are
offline contract tests, not evidence of improved debate quality or provider timing.
The selected regression suite passed340 tests in30.27s, including the request-boundary
checks, streaming lifecycle, body feedback/gates, revision formatting and duration
interfaces. At that point no paid model, ASR or TTS requests had been launched.

Subsequent live audit found that the cold listening-body instruction also bypasses
the legacy stage builder. That independent entry now explicitly adds the shared
stage strategy;88 focused tests include all three stages at its main-model boundary.
v34 was interrupted and preserved; v35 then verified all27 actual authoring requests.
It completed3 turns and partially played a fourth before a stop, with no closing
speech execution. See the root README and v35 report for timing, quality and the
inferred budget-reservation stop. Budgeted harness audio bundles are now allocated
lazily per turn, and a post-run fix persists first model/audio reservation rejections
before their cause can be lost behind a generic playback error. Neither change
reduces the per-request bounds or raises an approved cap.


### v36: complete-ASR task binding and independent speculative completion

The user approved USD30 more (global270→300, study180→210), then requested
fixing obsolete-work waits and unifying authoring input before the next rerun.
The atomic budget migration preserves all call/settlement rows and verifies the
study rejection trigger; pre-run exposure is267.332867824 globally and175.887409919
for this study, with no pending requests.

Body feedback now publishes its immutable task and result independently from its
revision text. Final work skips obsolete feedback without waiting. Matching
feedback can be consumed while revision is in flight; the revision Future is
joined only after exact user/system prompt equality, including current evidence.
When complete source history, draft, prefix, framework and preparation match,
the current turn retains its complete-ASR task's plan/record hints despite later
derived tree/planning updates. Full transcript changes invalidate that binding.
Final endpoint/body gates still consume final source data. Committed continuation
uses the same explicit history source as early revision. No added model calls,
check removal, TTS speculation or lengthened overview is introduced.

354 selected offline tests passed in30.63s, including event-ordered regressions
that keep obsolete feedback/revision blocked until final body audio is emitted,
matching in-flight revision reuse, changed source/evidence/system invalidation,
and final qualification gates. compileall and diff whitespace checks passed.
This establishes scheduling and input contracts; live latency/quality remain to
be measured in the authorized fresh six-turn listening-motion-live-v36 run.


v36 live result (2026-10-07): full six-turn run exited0, all speech/audio played,
all requests ended; source snapshot still matches current source. Four full-ASR
bindings retained changed derived planning context and reused feedback/revision.
28/28 actual authoring requests had exact native system/stage blocks.12 artifact
audits passed. Timings (first/gap/audio seconds): FORopening6.770/.007/240.945,
AGAINSTopening.218/.018/238.148, FORrebuttal.199/.008/239.607,
AGAINSTrebuttal.198/.016/239.495, FORclosing.209/.007/117.838,
AGAINSTclosing13.113/.007/118.304.5/6 timing pass; duration errors within1.81%.
Final closing's three planning calls19375/19377/19380 all timed out at3s;
preparation stayed waiting_for_framework. Full-input arrival took7.4654s after
opponent endpoint, followed by5.6474s to first audio. No fallback-framework fix was
implemented or further paid retry launched. This issue is distinct from obsolete
body-revision waiting. Descriptive v35→v36 rebuttal gaps5.503→.0077 and5.972→.0164;
fresh speech/overview differences preclude a causal ablation claim.

349 Gemma requests,165 TTS and84 ASR.15 planning timeouts retainUSD3.997712 unknown-use
reservations. Known usage estimateUSD.86858081; conservative increment7.47239524.
Reconciliation passed with no issues/pending: global274.805263064/300,
study183.359805159/210; remaining global25.194736936. No settled invoice claimed.
All six body gates accepted,0 repairs;127 TTS rewrite checks (86 accepted,41 rejected),
49 changed chunks. Manual assessment of all6 fully delivered speeches remains
quality_fail: unsupported absolutes, obligatory centralization assertions despite
decentralized defense, insufficient access/retaliation weighing, new legal safeguard
in FORclosing, and AGAINSTclosing denying benefits previously conceded as marginal.
Sources: run/listening-motion-live-v36/{report.json,conversation.txt,
manual_quality_review.json,binding_audit.json,authoring_prompt_audit.json,
closing_start_diagnosis.json}; all audit scripts/logs retained under that run.

### Readable planning input (2026-10-07, offline validated)

`planning_view.readable_planning_context` projects the canonical context only for
listening planning with clash records. Exchange records hold full source quotes
and inline qualifications; tree targets keep readable claims and supplemental
material. Exact repeated target/latest-reply and prior-position/objection pairs
share one source object with a readable role flag. Covered history excerpts and
duplicate target sources/conditions are removed only with matching ownership and
status. Distinct or historical material stays visible. Source options retain
their original boundary indices beside the text, with unmatched options in
`additional_boundaries`; there is no quote dictionary for the model to resolve.

The projection deep-copies its input. The parser still binds selections against
the original canonical context. Full speech history, heard prefix, previous
state, target order/versions and source text are preserved. No new model calls or
generated summaries are introduced. Authoring/review prompts keep their existing
context, and planning without clash records retains its previous layout.

308 relevant regression tests passed. The offline audit
`experiments/incremental_planning/audit_readable_planning.py` checked all26 saved
v36 planning requests:23 projected and3 unchanged, with all source/condition,
ownership, history and option-index preservation checks passing. Total request
characters fell1,226,700→1,134,987 (7.48%); the three closing requests19375/19377/
19380 fell60,468→54,524,64,278→57,986 and65,248→59,831 respectively. Counts include
the original response-schema suffix. This is not a token, latency or model-quality
measurement. Report: `experiments/incremental_planning/run/readable-planning-context-offline-v1/report.json`.

### Six-second planning trial, v37 (2026-10-07)

The listening preset and live harness now use6s instead of3s for optional planning,
with no retry. The user requested one fresh match after the readable projection.
71 preflight tests and13 artifact audits (including planning-request inspection)
passed. All20 planning calls completed with valid ready overviews, median2.049s,
max3.993s; one exceeded3s. No list-count violations or unknown-usage requests were
observed. This is a fresh, shorter conversation than v36, so it does not isolate
the effects of timeout, input projection or service load.

Four speeches completed with first-audio times7.118/.211/.206/.204s, maximum
playback gap.0154s and duration errors below.75%. FOR closing began its reviewed
overview at.232s after the previous endpoint, but final-input review rejected
`prevents exclusion` on readiness grounds (`latest_input_ok` remained true).
The body was withheld;14.9s played and the sixth speech never started. Initial
review19659 had accepted the same text; final review19696 rejected it with a
different input context. The final judgment may be overly strict about an
announced argument and does not establish a new-input contradiction. No automatic
retry or follow-up model run was launched. Manual review also found unsupported
absolutes, incomplete weighing and repeated phrases in the completed speeches.

Usage estimateUSD.75054204; conservative budget increment3.00288816, cumulative
277.853693584/300 and study186.408235679/210, remaining global22.146306416. All
workers and requests ended; reconciliation had no issues. These are usage-based
estimates, not a settled invoice. Report, exact delivered conversation, manual
review and overview-rejection diagnosis are under
`experiments/incremental_planning/run/listening-motion-live-v37/`.

### Reuse native Audience for body feedback (2026-10-07)

Both provisional body feedback and complete-input whole-speech feedback now call
the configured `Audience.feedback`, sharing the native prompt's four evaluation
dimensions and input fields. Its output format is now compact: only Critical Issues
and Minimal Revision Suggestions, at most three concrete high-priority edits and
180 English words (120–180 when needed; shorter or No changes otherwise).
No Comprehensive Analysis, per-dimension summaries or praise is requested. These
are prompt-level output requirements, not a truncation or retry loop; configured
model, temperature and token allowance remain in effect. The adapter extracts the
minimal-revision section; a No changes section skips content-only revision.
Full raw audience responses remain available as feedback details.

Each speech's PrefixPreparation allows at most one preparatory feedback round,
including a failed attempt. Later body drafts still update on new input and may
use earlier feedback as drafting context. They carry feedback=None and
feedback_status=skipped_turn_limit, never a reused approval for their new text.
The immutable draft handoff records the same status. The complete-input review
remains independent and sees the full latest transcript; a new speech gets a new
preparatory allowance. Missing/invalid drafts do not spend a feedback attempt.

Each audience review gets an isolated conversation with its configured system
prompt, model, temperature and output allowance. Provider calls go through the
existing helper transport so speculative call limits and experiment accounting
still apply. The live harness honors the requested audience temperature. All
source history and the current provisional heard prefix are supplied with speaker
labels; the adapter handles both labelled debate history and retained role-based
speech history. Full source text is not duplicated in supplemental metadata.
The already spoken overview remains immutable. Complete-ASR task matching still
allows the final path to reuse early feedback without a second audience request.

315 related offline tests passed in31.23s, including native prompt/config checks,
concurrent conversation isolation, current-prefix ownership, feedback/revision
reuse and audio ordering. Subsequent v38/v39 measurements below used the verbose
native format. The newer compact-output/one-preparatory-round policy has only
offline validation so far; its live latency and quality remain unmeasured.

### Shared overview review standard (2026-10-07)

Preparation and endpoint review now send the same decision contract, with only
the stage marker and available input changing. Both distinguish announced
advocacy from asserted facts and attributed commitments. Assertive outcomes inside
an announced argument, such as the v37 phrase `prevents exclusion`, do not become
established facts or unconditional guarantees merely because the opponent has
finished. An opposing causal prediction alone is not an invalidating source.
Fabricated evidence, wrong stance, false attributions, invented implementation
requirements and premature opponent-specific claims remain blocking defects.

Each conflict now requires a `basis`: `stance_mismatch`, `fabricated_evidence`,
`misstated_commitment`, `unheard_input` or `invalidated_premise`. A `latest_input`
rejection requires `invalidated_premise` and a nonempty exact source quotation.
Missing/invalid classifications go through the existing bounded format repair;
they are never silently accepted. Classification and semantic correctness still
depend on the reviewer model; local checks validate the contract and quotations.

136 offline tests passed in16.96s, including identical-rule checks using the exact
v37 overview/objection, rejection validation, final-input conflict handling,
format recovery and playback overlap. Test responses are mocked: this verifies
the implemented contract and control flow, not live semantic accuracy. No paid
evaluation was run after this change.

### Advisory final reviews (2026-10-07)

After a reviewed overview starts playing, its full-input review is diagnostic.
A rejection, malformed review or provider error records a warning and continues
the body. This also applies when a parallel endpoint review is reused. The final
body review likewise records findings without an additional repair/recheck or
withholding audio. Ordinary Audience feedback/revision still precedes this review.
Warnings retain the actual failed/inconclusive verdict; they are not recorded as
passes. Both the application log and `review_warnings` in `listening_prefix.json`
capture the finding and `continue_body` action. Existing `endpoint_reviews` and
`body_reviews` keep their detailed results. The authoring prompt describes the
overview as committed, without claiming it passed the final review.

176 related offline tests passed in19.84s. Cases include sequential and parallel
late rejection, malformed responses, review timeouts, both reviews rejecting in
one speech, in-flight draft transfer, normal audience revision and preservation of
the actual warning verdicts while complete text/audio are delivered. Actual ASR
failure, missing body and input/TTS integrity checks retain their existing behavior.
No paid run was launched.

The live harness manifest also describes these advisory actions with zero final
body repairs and no recheck. Its8 offline tests passed in7.29s, bringing this
change's validation to184 tests; syntax compilation and whitespace checks passed.

### Native Audience and advisory review trial, v38 (2026-10-07)

A fresh paid match used the existing USD300 global/USD210 study caps, with36
preflight tests passing. Actual requests verified16 native Audience calls (12
provisional,4 complete-input), configured temperature1/max4096, original four
dimensions, isolated request histories and immutable-prefix guidance. All7
overview review requests used the shared contract. All3 final body reviews passed;
there were no live review warnings or extra final-review repairs. The warning
continuation branch remains established by offline fault injection, not this run.
All15 planning calls completed: median1.99s,max4.26s,zero timeout/list-count failures.

The trial stopped after a TTS request in the FOR rebuttal hit a60.073s ReadTimeout.
That bundle's transport latched closed and locally rejected14 later attempts;
its reserved request total wasUSD.43842 against aUSD1 allowance,33 of128 requests,
so neither funds nor request count was exhausted. The harness subsequently raised
`BudgetExceeded: TTS guard recorded a blocked dispatch` at turn finalization.
This error label must not be interpreted as exhausted global/study funds.

FOR opening, AGAINST opening and FOR rebuttal audio fully played for239.477,
237.584 and234.516s respectively. Only the first two pipelines finalized normally.
Their first-audio times were7.812/.195/.191s and maximum gaps5.342/16.967/4.920s;
all three missed the2s gap target. AGAINST rebuttal started its prepared overview
at.183s and played15.04s before stop; both closings were unstarted. Source/ASR
histories and audio coverage remain audited without changing raw failure statuses.

For the first three bodies, the final body became ready20.74/18.40/14.17s after
generation started. Initial body TTS API times were6.86/12.26/3.88s. Native final
Audience calls took6.64/8.78/7.71s; these and other steps contributed to gaps, but
this fresh conversation does not isolate any one change's effect. Manual review
found unsupported absolutes/implementation assumptions and a repeated conclusion.

Known usageUSD.52716898 excludes the failed request's unsettled charge. The failed
TTS bundle retains its fullUSD1 reservation. Reconciled incremental conservative
exposure isUSD2.68603592; cumulative280.539729504/300 and study189.094271599/210,
leaving19.460270496 global. All requests/workers ended; no automatic rerun.
Fourteen artifact audits passed, plus planning/stop diagnosis and manual quote
validation. Complete report, played conversation, native-review and delivery timing
audits, timeout diagnosis and source snapshot are under
`experiments/incremental_planning/run/listening-motion-live-v38/`.

### v39 regeneration (2026-10-07)

Same motion/configuration, explicitly requested fresh six-turn run. All six pipelines
and speeches completed;0failed requests,0pending requests, no TTS timeout. Timing
passed4/6turns. FOR opening:10.358s first audio,3.794s prefix/body gap; AGAINST
rebuttal:3.243s gap. Both gaps occurred at initial body audio: body readiness was
1.437/.304s after the overview ended, followed by2.357/2.940s audio work. Remaining
first-audio delays.209–.225s; both closings had gaps below.02s. Server-paced playback,
not browser/sound-card onset. Fresh conversation, not a controlled comparison.

All26planning calls completed (median2.568s,max4.136s), but3responses exceeded
requested list limits. All6final-body reviews accepted,0advisory warnings,0extra
final-review repairs. Sixteen manual content findings include unsupported absolute
claims, assumed mandatory centralization and repeated adjacent concluding text.
Native review acceptance does not establish content quality.

Usage estimateUSD.8369391; conservative incremental exposure3.3484764. Reconciled
cumulative283.902665904/300,study192.457207999/210,globalremaining16.097334096.
No unknown-usage requests in this run.14artifact audits passed plus planning,
completion/quote checks and source hashes;36same-code preflight tests reused from
v38 with explicit provenance. Artifacts: run/listening-motion-live-v39/ under
experiments/incremental_planning. No runtime source changes or automatic rerun.

### Compact Audience, serial final review: v40 (2026-10-07)

All six pipelines and speeches completed. Duration/first-audio/max-gap seconds:
FORopening240.266/6.964/.008; AGAINSTopening237.313/.206/.008;
FORrebuttal239.068/.226/.027; AGAINSTrebuttal239.378/.204/.130;
FORclosing120.161/.210/1.282; AGAINSTclosing118.940/.193/.003.
All6timing checks passed; maximum duration error1.120%. This is server-paced
playback, not browser/sound-card onset. Final body reviews stayed serial.

Audience calls11vs23in v39:5preparatory(one for each non-cold speech),6complete.
13later preparatory reviews were skipped while drafts kept updating. Full-input
feedback mean2.672svs7.638s,182.17vs657.17output tokens; ordinary revision mean
4.143svs4.124s. All11feedback outputs fit the requested <=180words and2–3numbered
edits; native model/temperature/four-dimension criteria and final-input reviews
were preserved. Fresh texts/service latency differ, so this is not an ablation.

26/26planning requests completed,0timeouts,median2.604s,max5.436s;3responses exceeded
requested list caps.11shared overview reviews,6accepted final body reviews,
0advisory warnings,0extra final-review repairs.31native authoring requests audited.
320Gemma calls,166TTS and88Whisper calls. One optional length_rewrite request
(call20468) returned HTTP500 after31.338s and existing fallback continued delivery.
Its usage is unknown andUSD.066876reservation remains.111meaning checks:
84accepted,27rejected;41distinct changed chunks. Manual quote-verified review
found18content concerns (unsupported absolute effects, design assumptions,
incomplete comparative weighing, and adjacent repeated phrasing). No paid judge.

Known usageUSD.84888092, incremental conservative exposure3.46371968; cumulative
287.366385584/300,study195.920927679/210,remaining12.633614416global. Invoice not
settled. Reconciliation applied with issues[],0pending calls,allworkers ended.
36budget/preflight tests passed7.01s,167compact/serial regressions passed22.82s.
15artifact audits passed; inherited handoff validator was updated to accept the
new skipped_turn_limit state while requiring feedback=None. Initial failed
validator result and successful rerun evidence are retained. Source hashes verified;
no runtime source changed during the run. Artifacts under
experiments/incremental_planning/run/listening-motion-live-v40/.

### Grounded opponent overviews: v41 (2026-10-08)

Current early-handoff drafting/review allows a brief account of an already heard
opponent point, preserving its scope, qualifications and argumentative role.
Empty target IDs remain required; transcripts establish attribution. New central
opponent input plus a changed framework can refresh an unpublished generic
overview within the existing two-rewrite/48-call caps. An invalidated candidate
is withdrawn even if replacement fails; withdrawal does not reset that cap.
Published text stays immutable. Definitions are optional in opening overviews;
brief greetings favor the first opening and count toward its existing budget.

The fresh six-turn v41 run completed. Duration/first-audio/max-gap seconds:
FOR opening239.475/5.506/2.244; AGAINST opening239.711/.208/.008;
FOR rebuttal239.381/.216/.007; AGAINST rebuttal239.724/.171/.018;
FOR closing120.522/.233/.007; AGAINST closing120.184/.210/.014.
Five of six turns passed timing. Cold prefix13.392s ended before body audio was
ready: body work ended18.539s after start, first body audio arrived21.102s.

One greeting, no opening-overview definitions, four future-roadmap overviews, and
two closing verdict/weighing overviews were observed. Three overviews explicitly
attributed opponent points; AGAINST rebuttal misattributed a criticized
all-or-nothing security criterion as FOR's endorsed view (draft20735/review20739).
Both preparatory and complete-input reviews accepted the error. All11 overview
reviews and6final body reviews accepted;0advisory warnings. No overview rewrite
was requested live, so bounded-update behavior is supported by offline tests,
not exercised by this run. Manual reading found13quote-verified concerns across
six speeches, including attribution, unsupported absolutes and repeated chunks.

317Gemma calls,160TTS,84Whisper;25of26planning calls completed, one6s timeout
(call20878) retained itsUSD.271576reservation and did not interrupt delivery.
Eleven Audience requests and31native authoring requests were audited. All16
artifact audits passed. Preflight24tests passed7.18s; prior overview regressions
162passed, with114targeted tests rerun after the final rewrite-cap fix. Source
snapshots verified unchanged through the run. Fresh texts are not a controlled
comparison; playback timings are server-paced, not browser/sound-card onset.

Known usageUSD.85679222; incremental conservative exposure3.69970488; global
291.066090464/300,study199.620632559/210,remaining8.933909536global.
Reconciliation issues=[],pending0; all workers ended. Invoice is not settled.
Artifacts: experiments/incremental_planning/run/listening-motion-live-v41/.

### First-paragraph choices after v41 (2026-10-08)

The overview can choose a definition, judging principle, own point, focused
question, contrast or direct response, with an optional brief greeting. These
are alternatives, not a checklist or forced rotation. Native body-point lead-ins
and action announcements no longer imply a mandatory overview sequence.
Preparation now snapshots `our_definition` when available and passes it to draft,
repair and review as private proposed scope, not opponent evidence. The motion
and actual exchange take priority over any conflicting prepared definition.
This change preserves the existing short-overview budget, update limits and
publication review. 162 offline regressions passed; the final wording adjustment
also passed the20 request-boundary checks. No paid rerun has measured diversity
under these prompts; v41's attribution-reversal failure remains unresolved.


### v42 diverse first-paragraph trial (2026-10-08)

User requested 跑motion after broadening first-paragraph choices and checking
native prompt alignment. Motion1, six fresh speeches, unchanged Gemma/TTS/Whisper,
call limits and USD300 global/USD210 study caps. Source snapshots remained
unchanged during the run. All six speeches were completely played and transcribed.
All six met timing thresholds: cold first audio7.734s, later handovers.219–.256s,
maximum single playback gap.0282s. Durations233.763,240.301,240.305,236.170,
119.993,118.475seconds.

Diversity remains limited: own-case FOR opening, three opponent-paraphrase/future
roadmaps, and two final-choice closings with similar opening words. No greeting or
definition was selected; these were optional, not pass/fail quotas. AGAINST closing
used a rhetorical question. AGAINST opening strengthened FOR necessity into an
unsupported only-way attribution. The saved FOR definition is empty; AGAINST's
private definition presupposes global identity linkage eliminating pseudonymity,
which conflicts with parts of FOR's actual proposal. Causality is not established.
Eleven overview reviews and six final-body reviews accepted all candidates with
no warnings, but manual reading recorded17 quote-verified concerns, principally
unsupported absolutes, implementation assumptions, an overstated attribution,
unresolved closing victory claims, and one repeated cross-chunk sentence.
No unpublished overview rewrite occurred, so update behavior was not exercised.

292Gemma calls,142TTS,77Whisper; all succeeded,25/25planning calls completed.
Preflight24tests passed7.49s; preceding prompt regressions162passed and final
wording request-boundary checks20passed. Sixteen artifact audits passed after
correcting one stale exact-wording assertion in the overview audit; initial failure
logs and results are preserved. This did not change production code or run data.
Known usageUSD.80793051; conservative increment3.23328204; global294.299372504/300,
study202.853914599/210,global remaining5.700627496. No pending/unknown run calls,
reconciliation issues=[],all workers ended. Costs are estimates, not settled bills.
Fresh match, not a paired causal comparison; timing is server-paced playback.
Artifacts: experiments/incremental_planning/run/listening-motion-live-v42/.


### Actual first paragraph with writer settings (2026-10-08, after v42)

Prefix drafting and bounded repairs now use the debater's configured writer model
and temperature through the existing metered helper transport. Speculative workers
snapshot these options; cold/endpoint generation reads the writer configuration.
Overview reviews retain their helper model and temperature. The live harness now
explicitly stores its existing main-writing temperature0.3 in DebaterConfig, and
its main response uses that same value, so a future run also drafts prefixes at0.3
instead of the former helper default0. The700-token prefix response allowance,
short spoken-word budget, call/repair limits and publication checks remain in place.

Draft and repair prompts ask for the actual first paragraph spoken in the debate,
allowing direct explanation of one concrete idea without summarizing the dispute
or announcing later paragraphs. Shared authoring guidance no longer reserves all
substantive development for the body; the body continues the fixed first paragraph.
Native stage strategy blocks remain shared.173 offline checks passed20.38s,
including real HelperClient request-boundary tests for speculative, cold and repair
paths, writer-option snapshot isolation, explicit temperature0, and unchanged
review settings. No paid generation has tested these changes yet; v42 is the
preceding version's result.


### First-paragraph-only replay on v42 inputs (2026-10-08)

User requested testing first-paragraph generation against existing material. Six
original v42 draft-context snapshots were replayed unchanged through the current
production prompt builder, with writer model Gemma26B and temperature0.3. Exactly
six draft calls were made, without semantic review, repair, body, TTS, ASR or fresh
planning. Each is an independent replay; later contexts retain the original v42
history, rather than incorporating these new drafts. The calls and contexts were
verified against the prepared manifest. All six local format checks passed.

All three future-roadmap sentences disappeared, replaced by direct claims or
reasoning. Three paragraphs still begin with opponent paraphrases; both closings
begin Ultimately, this debate. No greeting or definition was selected. The AGAINST
opening's only-way attribution and categorical efficacy claims remain concerns.
Thus directness improved in these samples, while structural variety remains limited.
Prompt and temperature changed together; one sample per context does not establish
which change caused the difference or a statistical diversity improvement.

Known usageUSD.01181807; conservative exposure.04727228, below the localUSD1 guard
and existing cumulative budgets. Global remaining5.653355216, no pending/errors.
Artifacts: experiments/incremental_planning/run/listening-motion-live-v42-first-paragraph-v1/
(report.json, comparison.txt, results.json, manifest.json, executed harness.py).


### Motion02 first-paragraph probe (2026-10-08)

User requested another motion. Six draft-only calls used Social media companies
should be required to label AI-generated content, current authoring rules and
Gemma temperature0.3. Each context contains only complete historical speeches
before that turn, plus saved private definitions and ranked claims. No target or
future speech, judgment, newly generated history, planner/tree, body, semantic
review, repair or audio was supplied/generated. These differ from motion01's
partial-input contexts, so this is not a controlled motion-only comparison.

All six local format checks passed, but manual quality did not. Only one paragraph
begins with an explicit opponent reference; no future roadmap, greeting or definition
was chosen. Both closings repeat Ultimately, this debate comes down to a choice.
FOR closing text supports labeling but its framework.position incorrectly says
against. AGAINST closing text endorses imperfect labeling as necessary and falsely
attributes a demand for perfect accuracy to the affirmative, reversing its stance.
Raw drafts are preserved and not repaired or presented as publication-approved.

Exactly6 successful calls21186–21191, no pending/errors. Known usageUSD.00834366,
conservative increment.03337464 under localUSD1/globalUSD300/studyUSD210 caps.
Global remaining5.619980576. Actual request/context/role/source checks passed.
Artifacts: experiments/incremental_planning/run/listening-motion-live-motion02-first-paragraph-v1/
(report.json, first_paragraphs.txt, results.json, manifest.json, runner snapshots).


### v43 full motion02 attempt: budget stop (2026-10-08)

User requested a complete motion02 run after first-paragraph-only probes. Current
writer temperature0.3/actual-first-paragraph prompt, original300/210USD caps.
Three speeches completed playback and timing checks:239.414,240.672,239.381seconds;
cold first audio6.312s and later handovers.192/.198s. The fourth speech was fully
generated (236.220s audio) but only its first4chunks/61.943s completed playback.
Neither closing turn started. A speculative FOR-closing body reservation was denied:
299.7467+0.5265 would exceed300USD, triggering the shared stop. The saved pipeline
exception Listener failed during playback is a consequence, not the root cause.

All workers ended, pending0. Reconciliation released completed audio reservations:
global297.249483584/300,study205.804025679/210,remaining2.750516416. Known run usage
.67465404USD; conservative increment2.86946416. One6-second planning timeout
(call21330) retains.169168USD unknown-use reservation. Invoices not settled.

Four spoken first paragraphs had no future roadmap, two opponent references,
no greeting/definition. FOR rebuttal exercised a grounded unpublished update.
Closing stance stability remains untested. Manual review of fully played text found
11concerns; content quality failed despite advisory model reviews. Sixteen artifact
audits passed over recorded scope; preflight24passed7.02s, preceding authoring
regressions173passed20.38s. No production source edits occurred during execution.

Saved complete four-speech generated text and reconstructed full fourth audio for
reuse; generated-but-unplayed content is distinguished from actual spoken history.
continuation_plan.json awaits explicit resume approval: reuse those four speeches,
complete both closing texts/audio only within the existing2.75USD remainder,
estimated additional.20–.80USD. This would complete artifacts, not restore the
interrupted real-time ASR benchmark. No new paid work has started after the stop.
Artifacts: experiments/incremental_planning/run/listening-motion-live-v43/.

2026-10-08: v43 text/audio continuation completed under the user's additional
USD20 approval (global320/study230), with a durable localUSD5 guard. Four saved
speeches were reused; two closings were generated from complete prior text,
without rebuilding the interrupted real-time ASR/planner state. Combined audio
1191.795s; closing durations117.373/118.738s. FOR closing reverses stance and both
closing first paragraphs are identical: quality failed. Correct side/history
inputs are verified; prefix authoring introduced the reversal and endpoint review
accepted it. Audience/final review detected it, while immutable-prefix revision
and advisory final review left it in the output. No production edit or retry.
Additional known usageUSD.11130568; conservative cumulative297.694946304/320,
remaining22.305053696,pending0. Fifteen related offline tests passed; reused hashes,
history, transcript/audio artifacts and ledger reconciliation passed. Details:
experiments/incremental_planning/run/listening-motion-live-v43-completion-v1/report.json.


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


### Historical listening simplification (v46, 2026-10-08)

This describes the v46 experiment; the current publication rules above supersede it. Listening prepares
the actual first paragraph/audio and a remaining speech draft on immutable input.
Drafting uses the native opening/rebuttal/closing templates. Complete-input
Audience feedback and ordinary revision remain; closing skips Audience feedback
as in the original TreeDebater. Exact matching complete-ASR work can be reused
while final tree/planning finishes. Original paragraph splitting, FastSpeech
estimation, parallel TTS and playback remain.

Removed from the active path: prefix semantic/change/endpoint reviews, preparatory
body feedback, final body gate/repair/recheck, and TTS meaning reviews. Local
format/length checks, a bounded format repair, stale-source checks, matching-audio
checks, and immutable published text remain. Revisions reuse post_process_prompt
and native duration fitting without extra paragraph quotas or97–100%word targets.
The TTS verify_rewrites option is retained for config compatibility but no longer
dispatches a model review. Historical review utility modules are not invoked by
this path.

Rollback snapshot: ../TreeDebater_snapshots/before_legacy_simplification_20261008T102837Z.
Its restore.py verifies by default and restores captured files with --restore;
credentials, model data, outputs and cumulative billing records remain in place.


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

### First body audio during final handover

With parallel endpoint revision enabled, its completed text now starts one
unpublished TTS request for the first body chunk while final tree/planning work
finishes. If feedback requires no revision, the unchanged body can start that
request immediately. Preparation shares native revision cleanup, paragraph
splitting and short-block merging with delivery.

Early feedback waits for an explicit first-audio duration result before starting
revision or unchanged-body synthesis. This also covers closing feedback that
finishes before the first chunk is published. The result uses the decoded audio
after seam normalization and local tempo processing; raw TTS duration metadata
does not set the remaining speech budget. First-audio failure releases the
waiting workers before executor shutdown. `prefix_duration_wait` records this
dependency in the delivery trace.

Final input reconciliation still selects the body. Only an exact match of the
final first chunk (including any early cut), voice and TTS model can reuse the
prepared request, even while it is in flight. Changed text uses normal synthesis;
a failed prepared request also falls back to normal synthesis. Existing duration
refinement and playback validation remain. All speculative requests are settled
before the speech returns, including when final handover fails. The per-speech
`first_body_audio` trace records preparation timing, matching and reuse in the
TTS candidate pool; reuse does not imply that a later duration edit selected
that same candidate for playback.

This overlaps synthesis with handover; it does not guarantee gap-free playback
when text preparation or TTS still exceeds the prefix's playback duration.

### Shared speech authoring after the v46 stance regression

The saved v46 requests locate the first reversed argument in call 22226, the
AGAINST listener's first rebuttal plan. Its assigned side was correct and its
previous state was empty. Calls 22231/22234/22239 kept that direction. Call 22227
then produced an independent first paragraph almost identical to FOR's opening,
while the body used native authoring and returned to AGAINST. The recorded
request/response fixture is `tests/fixtures/listening_v46_stance_reversal.json`.

Listening planning now selects source claims and boundaries only. Its schema
cannot return rebuttal prose or body tactics, and the server owns the assigned
position. Original sources, qualifications, exchange records and complete heard
speech remain available. Native writing receives speaker-labelled sources and
selected targets, not the listener's free-text argument plan. Historical plans
are not passed into current listening body allocations either. Non-listening
planning keeps its existing schema.

The native stage prompt now generates opening and initial body in ONE request.
Only a complete, locally valid pair can become a prepared speech. Format errors
identify the specific field. An omitted `framework.reason` is recorded locally
as an absent author explanation, preserving both speech fields and the semantic
publication review. Other structural framework errors use the existing one-repair
allowance to repair only the framework; returned speech changes are ignored.
Speech-format errors or semantic rejection still repair the unpublished pair
together. This replaces the initial separate
prefix/body calls; later source changes can still update the remaining body.
An in-flight update can replace the initial body after handover, with matching
prefix/framework and newer completion time. Published text remains immutable.
Final native feedback/revision and first-body audio pre-synthesis still apply.

This removes the observed cross-writer propagation path, not all possible model
stance errors. Offline tests validate request ownership, source preservation,
old-tactic rejection, draft sharing, updates and delivery. They do not establish
a new semantic error rate. Cold-start first audio may take longer because the
initial complete draft must arrive before its opening can play; during listening
that work overlaps the opponent's speech. Live timing/quality need measurement.
