# Streaming speech generation, playback, and listening

TreeDebater has two independently configurable streaming paths: **output** generates
speech audio paragraph by paragraph in [`../tts_streaming.py`](../tts_streaming.py),
and **input** transcribes delivered audio and updates the listener's debate tree in
[`env.py`](env.py). [`overlap.py`](overlap.py) connects them so later audio can be
prepared while earlier audio is being delivered and analyzed.

The initial speech is generated in full before the TTS pipeline starts. Here,
“streaming” means incremental publication of completed audio chunks. Each TTS request
returns a complete MP3 candidate; this path does not consume LLM token deltas or play
incoming audio bytes directly.

Run everything from **`TreeDebater/src`** so the `streaming` package and sibling modules (`env`, `agents`, `utils`, …) resolve correctly. Alternatively, put `src` on `PYTHONPATH` and run from the repo root.

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

### CPU speech-duration estimation

Statement refinement defaults to `TIME_MODE_FOR_STATEMENT = "time"` in
`utils/constants.py`. Streaming TTS uses the same 0.46-seconds-per-word heuristic
from `utils/speech_duration.py`; neither path loads FastSpeech weights or needs a
GPU for estimation. This is an approximate English speaking rate, not an audio
measurement. TTS candidate selection and playback budgets still use measured
audio duration. Historical `fs` timing fields in profiling output now measure
this CPU estimation step. Explicit `LengthEstimator("fastspeech")` remains an
optional legacy mode and lazily loads its model only when selected.

Non-adaptive delivery packs sentences when paragraph breaks are sparse or a paragraph
is long. Adaptive delivery retains its short first chunk and measured speaking rate.
Refinement reserves up to `refine_deadline_margin_seconds` for delivery, capped at
half the preceding chunk's duration. Speed resynthesis requires at least
`speed_adjust_min_slack_seconds` of playback time remaining. Seam normalization runs
before duration accounting and chunk callbacks; disable it with `normalize_seams: false`.
CPU duration estimation remains the default. Explicit FastSpeech mode uses the shared
wrapper without loading model weights in CPU-only runs.
