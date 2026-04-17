# Streaming Debate Logging Guide

This document explains the DEBUG-level logging added to analyze streaming debate performance across all operating modes.

## Four Operating Modes

The system supports 4 combinations of `streaming_tts` and `streaming_listen`:

| Mode | streaming_tts | streaming_listen | Description |
|------|---------------|------------------|-------------|
| **Full Overlap** | True | True | Speaker generates chunks in real-time, listener processes while speaking |
| **Speaker Stream** | True | False | Speaker generates chunks in real-time, listener processes after completion |
| **Listener Stream** | False | True | Speaker generates full audio then chunks, listener processes while "playing" |
| **Sequential** | False | False | Speaker generates full audio, listener processes after completion |

## Log Components and Events

### 1. Turn-Level Events

```
[Turn] turn_start stage=opening side=for t=1234567890.123
[Turn] mode_config stage=opening side=for streaming_tts=True streaming_listen=True mode=tts=stream_listen=stream t=1234567890.125
[Turn] turn_end stage=opening side=for t=1234567920.000
```

**Captures**: Turn boundaries and mode configuration

### 2. Speaker Thread Events

```
[SpeakerWorker] thread_start stage=opening side=for t=1234567890.130
[SpeakerWorker] generation_start stage=opening side=for t=1234567890.131
[SpeakerWorker] generation_end stage=opening side=for response_len=1250 t=1234567905.500
[SpeakerWorker] thread_end stage=opening side=for t=1234567925.200
```

**Captures**: LLM generation timing

#### Batch TTS Mode (streaming_tts=False)

```
[SpeakerWorker] posthoc_chunk_start mode=batch_tts mp3_path=treedebater_opening_for.mp3 t=1234567905.600
[SpeakerWorker] posthoc_chunk_split audio_duration=120.50s num_chunks=12 t=1234567905.650
[SpeakerWorker] posthoc_chunk_end num_chunks=12 stream_time=0.450s t=1234567906.050
```

**Captures**: Post-hoc chunking overhead

### 3. TTS Chunk Bridge Events (streaming_tts=True only)

```
[TtsChunkBridge] chunk_detected chunk_idx=1 size=45120 t=1234567890.145
[TtsChunkBridge] chunk_copied chunk_idx=1 detection_latency=0.805s copy_time=0.002s t=1234567890.950
```

**Captures**: Real-time chunk availability

### 4. Playback Main Thread Events

```
[PlaybackMain] playback_start side=for t=1234567890.150
[PlaybackMain] wait_chunk_start chunk_idx=1 cursor=0.00s t=1234567890.150
[PlaybackMain] wait_chunk_end chunk_idx=1 wait_time=0.850s t=1234567891.000
[PlaybackMain] chunk_assembled chunk_idx=1 duration=10.00s total_audio=10.00s t=1234567891.015
[PlaybackMain] file_write chunk_idx=1 size_sec=10.00s write_time=0.120s t=1234567891.135
[PlaybackMain] chunk_playback_start chunk_idx=1 duration=10.00s t=1234567891.136
[PlaybackMain] chunk_playback_end chunk_idx=1 cursor=10.00s t=1234567901.136
[PlaybackMain] playback_end side=for cursor=120.50s t=1234567920.000
```

**Captures**: 
- Speaker bubbles (wait_chunk gaps)
- File I/O overhead
- Chunk assembly timing

### 5. Streaming Listener Events (streaming_listen=True only)

```
[StreamingInputEnv] thread_start stage=opening statement_side=for cursor_mode=True t=1234567890.127
[StreamingInputEnv] file_read cursor=3.00s available=10.00s read_time=0.015s t=1234567894.200
[StreamingInputEnv] asr_start audio_range=0.00-3.00s t=1234567894.215
[StreamingInputEnv] asr_end audio_range=0.00-3.00s text_len=450 asr_time=0.850s t=1234567895.065
[StreamingInputEnv] tree_update_start words=75 text_preview=In this opening statement... t=1234567895.066
[StreamingInputEnv] tree_update_end words=75 update_time=0.245s t=1234567895.311
[StreamingInputEnv] wait_audio_accumulation available=2.50s need=3.00s t=1234567896.000
[StreamingInputEnv] thread_end stage=opening statement_side=for t=1234567925.123
```

**Captures**:
- ASR timing and RTF
- Tree update costs
- Audio starvation events
- Listener bubble (thread_end - playback_end)

### 6. Batch Listener Events (streaming_listen=False only)

```
[NonStreamingListener] batch_listen_start stage=opening side=against t=1234567920.100
[BatchListener] analyze_start stage=opening side=against opponent_side=for t=1234567920.200
[BatchListener] analyze_end stage=opening side=against analyze_time=2.450s t=1234567922.650
```

**Captures**: Sequential processing overhead

## Key Metrics Calculation

### Speaker Bubble (waiting for TTS chunks)
```
speaker_bubble = sum(wait_chunk_end - wait_chunk_start)
```

### Listener Bubble (post-playback processing)
```
listener_bubble = listener_thread_end - playback_end
```

### True Overlap
```
true_overlap = playback_duration - speaker_bubble
```

### Overlap Efficiency
```
overlap_efficiency = true_overlap / (playback_duration + listener_bubble)
```

### ASR Real-Time Factor
```
rtf = asr_time / audio_duration
# rtf < 1.0 means real-time capable
```

### Chunk End-to-End Latency
```
e2e_latency = chunk_playback_end - chunk_detected
```

### Bottleneck Detection
```
bottleneck = max(speaker_duration, playback_duration, listener_duration)
```

## Mode-Specific Metrics

### Full Overlap (tts=stream, listen=stream)
- Speaker bubbles
- Listener bubbles
- True overlap
- ASR RTF
- Chunk E2E latency

### Speaker Stream (tts=stream, listen=batch)
- Speaker bubbles (should be minimal)
- Batch analyze time
- No listener bubble (sequential)

### Listener Stream (tts=batch, listen=stream)
- Post-hoc chunk time
- Listener bubbles
- ASR RTF
- No speaker bubbles (batch TTS)

### Sequential (tts=batch, listen=batch)
- Post-hoc chunk time
- Batch analyze time
- No bubbles (no overlap)

## Using the Analysis Script

```bash
# Analyze any mode
python src/scripts/analyze_streaming_performance.py log_files/debate.log

# Compare modes
python src/scripts/analyze_streaming_performance.py log_files/full_overlap.log -o full.json
python src/scripts/analyze_streaming_performance.py log_files/sequential.log -o sequential.json

# Then compare JSON outputs to understand performance differences
```

## Example: Comparing Modes

**Full Overlap Mode:**
```
Mode:                tts=stream_listen=stream
Overlap efficiency:  84.5%
Speaker bubble:      8.2s (6.8% of playback)
Listener bubble:     12.3s (9.3% of listener time)
Bottleneck:          LISTENER
```

**Sequential Mode:**
```
Mode:                tts=batch_listen=batch
Post-hoc chunk time: 0.450s
Batch analyze time:  2.450s
No overlap (sequential execution)
```

The difference shows the benefit of streaming: ~96s saved through parallelization!

## Agent / LLM timing and I/O logs (TreeDebater)

These lines use a **separate format** from streaming events above. They are intended for grep and ad-hoc profiling, not for `analyze_streaming_performance.py`.

### Main debate log (`N.log`)

Single-line timing records share the prefix **`[timing]`**, then **`phase=...`**, **`duration_s=...`**, and optional context keys (`stage`, `side`, `call_id`, `pass_index`, `iteration`, …).

Examples:

```
[timing] phase=tree_debater_speak duration_s=45.2301 call_id=3 stage=rebuttal side=for
[timing] phase=main_get_response duration_s=12.1000 call_id=3 stage=rebuttal side=for
[timing] phase=revision_suggestion duration_s=8.0200 pass_index=1 add_evidence=True call_id=3 stage=rebuttal side=for
[timing] phase=length_adjust_iteration duration_s=3.1000 iteration=1 max_retry=10 call_id=3 fit_ok=False current_cost=4.2 stage=rebuttal side=for
[timing] phase=helper_client_litellm duration_s=2.5000 model=gpt-4o n_index=1 max_tokens=4096
[timing] phase=debater_litellm_completion duration_s=11.8000 stage=opening side=for model=gpt-4o retry_attempt=0
[timing] phase=tts_wall_clock duration_s=8.4000 stage=opening side=for kind=streaming_tts audio_duration_s=7.5
[timing] phase=env_stage_wall duration_s=120.0000 stage=opening motion=...
[timing] phase=evaluation_wall duration_s=90.0000 motion=...
```

**Macro phases** (env / parallel / compare / prepare scripts): `env_stage_wall`, `evaluation_wall`, `comparison_phase_wall`, `comparison_evaluation_total_wall`, `compare_env_stage_wall`, `prepare_claim_pool_wall`.

**Meso phases** (TreeDebater `speak`): `tree_debater_speak`, `main_get_response`, `revision_suggestion` (with `pass_index` / `add_evidence`), `length_adjust` blocks (outer `timed_phase` with `block=`), `post_process`, `listen_analyze_statement`, `analyze_statement`, `audience_exemplar_retrieval`, `audience_simulated_feedback_llm`, `evidence_selection_llm`, `rehearsal_retrieve_on_prepared_tree`.

**Atoms**: `helper_client_litellm`, `debater_litellm_completion`, `get_response_with_retry_llm`, `embedding_api_fetch`, `exemplar_retrieval_query_embedding`, `exemplar_retrieval_semantic_search`, `tts_wall_clock`, `tts_trim_wall_clock`.

Short metadata without full prompts: **`[timing-meta] call_id=... speak_session=...`** where **`speak_session`** is either **`default_speak`** (base :class:`~agents.Agent`) or **`tree_debater_speak`** (:class:`~ouragents.TreeDebater`). That value matches the default **`[io] phase=`** for blocks in that turn (Prompt, Response, TTS-related bodies, etc.). Submodule blocks may override **`phase=`** (e.g. **`audience_feedback`**, **`length_adjust`**) while keeping the same **`call_id`**.

### I/O log (`N_io.log`)

When prompt/response logging is enabled (default), large bodies are written to a **sibling file** next to the main log: if the main file is `log_files/14.log`, the I/O file is `log_files/14_io.log`.

- Disable I/O file (and fall back to legacy DEBUG prompts on the main log only): set environment variable **`DEBATE_LOG_PROMPTS=0`** (also accepts `false`, `no`, `off`).
- Each I/O block starts with **`[io]`**, then key/value pairs:
  - **`call_id`**: same id as **`[timing-meta]`** for that speak turn (sub-blocks under one turn share this id).
  - **`phase`**: usually the **`speak_session`** (`default_speak` / `tree_debater_speak`) or a **submodule** label (`audience_feedback`, `length_adjust`, …) when the block is not the top-level speak transcript.
  - **`title`**: what the block is (e.g. **`Prompt`**, **`Conversation-History`**, **`Audience-Feedback-Prompt`**, **`Response-After-TTS`**). This is **not** the same field as **`phase`** — use **`title`** to tell Prompt vs Response apart.
  - Optional **`stage`** / **`side`** for attribution.
- Then a separator line and the body.

When I/O logging is on, optional **`[io-ref]`** lines on the main log use **`speak_session`** / **`title`** / **`call_id`** for quick correlation without duplicating bodies.

On startup, when the I/O log is attached, the main log records: **`[timing] phase=io_log_ready io_log=...`**.

### Analyzing agent timing logs

Use **`src/scripts/analyze_agent_timing.py`** (separate from `analyze_streaming_performance.py`):

```bash
python src/scripts/analyze_agent_timing.py log_files/14.log
python src/scripts/analyze_agent_timing.py log_files/14.log --io-log log_files/14_io.log -v
python src/scripts/analyze_agent_timing.py log_files/14.log --json-out agent_timing_report.json
```

It summarizes macro vs speak-pipeline vs atom phases, groups nested phases by **`call_id`**, lists **`length_adjust_iteration`** rows, and optionally counts **`[io]`** blocks in the I/O file.

## Log File Location

Logs go to the same file as debate system logs:
- File handler: DEBUG level (captures all events)
- Console handler: INFO level (shows high-level progress only)

Large LLM prompts/responses default to **`N_io.log`** (see Agent section above) so the main file stays readable.

This keeps detailed timing data available for analysis while maintaining clean console output.
