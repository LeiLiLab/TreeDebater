# `streaming` package

This package implements **chunked audio → ASR → debate-tree updates** for TreeDebater, plus helpers that **feed chunk files into a watch directory** so the listener can process speech incrementally (similar in spirit to debate-anonymous chunked pipelines).

Run everything from **`TreeDebater/src`** so the `streaming` package and sibling modules (`env`, `agents`, `utils`, …) resolve correctly. Alternatively, put `src` on `PYTHONPATH` and run from the repo root.

---

## Modules

| Module | Role |
|--------|------|
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

# Overlap + playback-driven timing (see configs/overlap_debate.yml)
python -m streaming.overlap --config configs/overlap_debate.yml

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
- **`streaming_listen`** (listener): in overlap runs, start `StreamingInputEnv` on a background thread and set `tree_via_streaming` on the turn record so `listen()` can avoid duplicating full `_analyze_statement` when the tree was already updated from audio.

---

## Dependencies

- **`env.py` / `overlap.py`**: `pydub`, `openai` (Whisper), `yaml`, TreeDebater `env` / agents / utils (API keys as elsewhere).
- **`chunk_audio.py`**: `pydub` only.
- **`bridges.py`**: `pydub`, `utils.tool` logger for the streaming-TTS copy bridge; live MP3 bridge prints to stdout for simple demos.
