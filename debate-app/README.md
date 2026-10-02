# TreeDebater microphone app

## Incremental rebuttal preparation (experimental branch)

`planning.mode` selects `legacy` (default), `end_of_turn`, `linear`,
`corrected_tree`, `adaptive_linear`, `tree_plan`, or `adaptive_tree` in session YAML.
See [`configs/gemma-incremental.yml`](configs/gemma-incremental.yml) for an example.
For command-line debaters, put the same `planning` mapping in the debater configuration.

Linear policies maintain explicit notes without a debate tree. Tree policies keep the
existing claim graph; corrected modes support speaker-owned revision/retraction and
archive dependent attacks before invalidating them. Adaptive policies use a separate
semantic WAIT/UPDATE call and force remaining input through preparation at the endpoint.
`max_updates` limits speculative updates; endpoint draining still runs after that limit.
All modes keep final speaking under the existing turn controller. Preparation does not
consume evidence, commit assistant messages or invoke TTS. Engine checkpoints include
the speculative state so recording recovery discards abandoned work.

To use an OpenAI-compatible text proxy, set `DEBATE_LLM_API_BASE` before starting the
backend, for example `http://127.0.0.1:4000/v1`. Optionally set
`DEBATE_LLM_API_KEY` in the environment. Both main and helper text models use that
endpoint; ASR/TTS retain their separate provider configuration. Do not put credentials
in YAML. Existing sessions with no `planning` setting continue using the legacy policy.

The controlled Gemma text replay is in
[`../src/scripts/benchmark_incremental_planning.py`](../src/scripts/benchmark_incremental_planning.py).
It uses a persistent, shared cost ledger and must be run only against an approved
experiment budget. Settings, commands, results and limitations are recorded in
[`../process.md`](../process.md). Its estimated text-ready latency is **not** a
measurement of browser audio latency.

[Public demo](https://dqwang122.github.io/projects/Debate/debate-app/) ·
[Hosting, availability, and restart instructions](PUBLIC_WEBSITE.md).
The public frontend is hosted; the backend currently uses a temporary HTTPS
tunnel to this research server. The local backend supports two concurrent debates
by default; the public service requires a coordinated frontend/backend update to
enable the new session lifecycle.

A local web app for a human-versus-AI debate with microphone capture, incremental
transcription and tree updates, streaming AI audio playback, and downloadable results.
AI-versus-AI sessions are also supported. Application code and runtime data live here;
`../src` remains the debate engine.

AI-versus-AI sessions keep one prepared worker per side. The opponent transcribes
and analyzes each fully played audio chunk while the speaker continues generating
later chunks. Each worker retains its own trees across role changes. The next turn
waits for generation, browser playback, and listener draining to finish. Human
microphone input is transcribed and analyzed incrementally during capture. Transcription uses independent
ASR calls on audio batches (3 seconds by default), plus provider latency; it is not
word-by-word partial recognition. It requires valid OpenAI credentials even while
AI preparation is still running.

The browser prefetches and decodes published chunks and schedules them contiguously
on the Web Audio clock. Playback acknowledgements run separately and coalesce to
the latest delivered cursor when the network is slow. Pausing suspends the audio
clock; late generation/downloads can still cause underruns, during which the cursor
does not advance. Only audio that has played is acknowledged.

Debate trees group each AI's view by side. Click a branch heading to fold or unfold
its details, or use **Expand all** / **Collapse all**. Side, level, visit count,
status, and other short properties share a compact row. Deeper branches start
folded, and live analysis updates preserve your fold choices.
After AI speech generation and revision, analysis receives the actions from that
speech's plan. It matches them to quoted passages in the final speech, retaining
the speaker's main-claim role alongside any attack links. Planned claims omitted
from the speech are not inserted. Matching paraphrases still relies on the model;
missing or invented source quotations are rejected before tree updates.
For nodes use blue accents and Against nodes use orange, including nested attacks
and rebuttals. Side labels remain visible alongside the colors.

The transcript selector lets you revisit earlier turns during a debate. Choose a
stage and speaker to read that turn, or use **Back to live** / **Follow current turn**
to return to ongoing transcription. Recording and playback continue while you read
an earlier turn, and live updates preserve your selection.

## Run locally

Use Python **3.10 or newer**. For real debates, activate the same environment that runs
TreeDebater successfully; the app's lightweight dependencies do not install the engine's
Python dependency stack. App workers disable CUDA visibility and use the configured
API model for claim scoring (`use_rm_model=False`), so the local Llama reward
checkpoints are not loaded. Speech-duration estimation uses a CPU word-rate
heuristic and does not require FastSpeech. Node.js must be at least 22.13.

From `TreeDebater/debate-app`:

The startup script checks the selected Python version before launching the service.
If `str | None` raises `TypeError`, an older interpreter started the backend; stop
that backend and activate the debate environment. You can also set
`DEBATE_APP_PYTHON=/absolute/path/to/env/bin/python` to select it explicitly.

```bash
# In the existing debate Python environment:
python -m pip install -e ./backend
cp .env.example .env
# Fill in backend credentials in .env, or retain your existing environment variables.
./scripts/start-backend.sh
```

In a second terminal:

```bash
cd /path/to/TreeDebater/debate-app/frontend
npm ci
npm run dev -- --host 127.0.0.1 --port 3000
```

Open **http://localhost:3000**. Use localhost or HTTPS for microphone permission.
Run npm commands inside `debate-app/frontend`, where `package.json` and
`package-lock.json` live. From the `TreeDebater` repository root, you can instead use
`make app-frontend-install`, then `make app-frontend`. In another terminal with the
debate Python environment activated, use `make app-backend`.
These three Makefile commands also work directly inside `debate-app`.
The frontend Makefile targets load your nvm default automatically if Node or npm
is missing from PATH (including in a Conda shell), and check Node's minimum version.

If working on a remote machine, forward port 3000 to your local machine. The frontend
development server proxies `/api` requests, event streams, and microphone WebSockets
to the Python backend at `127.0.0.1:8000` on the server. Local development uses
Vinext’s Node runtime; Cloudflare integration is enabled only for builds to avoid
competing WebSocket handlers. Set `DEBATE_APP_PROXY_TARGET` when starting the
frontend to use a different backend address.
The Python backend binds to loopback by default. Use one API process (`--workers 1`);
session ownership and capacity accounting are scoped to one service process.

### Concurrent debates

`DEBATE_APP_MAX_ACTIVE_SESSIONS=2` allows two independent debates on the same URL.
Each session reserves a slot when created and retains it through evaluation,
worker shutdown, and background task cleanup. Creation retries reuse the original
session. A full backend returns HTTP 503 with `code: server_busy` and `Retry-After: 5`;
the visitor can retry Start debate after another session finishes. There is no queue.
`GET /api/health` reports aggregate `capacity` (`limit`, `active`, `available`).

The frontend sends an authenticated `/api/sessions/{id}/heartbeat` every 20 seconds.
Sessions abandoned for 120 seconds are stopped, including paused sessions. Refresh
or reconnect within that grace period to continue. Created sessions that never
start expire after 120 seconds even if heartbeats continue. A five-second sweep
starts cleanup; capacity is released after cleanup actually finishes. Public mode
also retains its 40-minute maximum lifetime and global daily allowance.

Finished sessions leave the in-memory registry; token-protected snapshots, results,
audio, events, and artifacts remain available from disk. The UI keeps receiving
events after debate completion until evaluation and finalization finish. Restarting
the backend still ends unfinished debates; workers cannot resume after a restart.

One active debate per controller ID discourages duplicate starts in the same tab.
Controller IDs are tab-local and can be replaced, so this is not a per-person quota.
Run one Uvicorn process and update the frontend together with the backend so browsers
send heartbeats. Two full real-provider debates still need measurement before
increasing capacity beyond two.

If pip repeatedly reports `Name or service not known`, a configured package index
cannot resolve. To use only PyPI for this command, bypass pip configuration and the
extra-index environment variable:

```bash
PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= python -m pip install --index-url https://pypi.org/simple -e ./backend
```

If dependencies are already installed in your debate environment, an offline editable
install can use them without accessing an index (requires `setuptools>=68`):

```bash
python -m pip install --no-index --no-build-isolation -e ./backend
```

This still checks runtime dependencies and fails if any are missing. Pydantic 2.8.2
is supported; an upgrade is not required. Test dependencies are optional.

No separate public deployment is performed by these commands. The frontend uses the
Sites scaffold; real TreeDebater models run in the Python worker, not Cloudflare Workers.

## First run

1. Choose your side, motion, AI model, and speaking budgets.
2. Allow/test your microphone. The meter samples it briefly, then releases the device.
3. Choose **TreeDebater** for actual ASR and model-generated speech. It uses your backend
   credentials and existing engine dependencies. Preparation may take time while claim
   pools are generated. Real mode does not fall back silently to demo mode.
4. Or choose **Demo** to verify browser transport: it records real microphone samples but
   returns a visibly synthetic transcript and test tones. It is not speech recognition.
5. Start the debate, then press **Start microphone** on your turn. **Finish turn** flushes
   the last samples and waits for transcription/analysis before the AI responds.
   When you open first, the microphone is available immediately while AI preparation
   runs in the background. Audio is saved on arrival. A separate speech-recognition worker publishes transcript
   chunks during preparation; tree analysis catches up when the AI is ready.
   If preparation fails, you can continue recording, then fix the credentials and use
   **Retry AI preparation**. The retry retains your audio and original speaking deadline.
   An AI opening still waits for preparation before speaking.
6. AI audio is queued as chunks become available. If the browser blocks audio, use
   **Start / resume audio**. Pausing freezes the playback cursor.
7. Review final statements and export results/config or the audio/artifact ZIP.

**Pause debate** sits beside **Stop debate**. Pause flushes microphone audio already
captured, stops the microphone and playback, and freezes the remaining speaking
time. **Resume debate** continues the same turn and recording attempt; an intentional
pause does not mark audio incomplete. A human who had not started recording still
chooses Start microphone after resuming. Paused state survives a browser refresh.
Already running model/TTS requests may finish in the background and their results
are retained, but the debate cannot advance until resumed. Stop remains available
while paused. A backend restart still ends the session; it does not resume workers.

## Configuration

`configs/human-vs-ai.yml` and `configs/demo.yml` are complete app examples. Advanced
settings support JSON editing of the `streaming` section and YAML/JSON import/export.
Existing engine YAML can also be imported: motion, model, supported streaming settings,
speech budgets, and speaking order are mapped into app settings. Remove CLI-only
streaming settings before importing: the app rejects `playback`, `posthoc`, and
input settings other than `min_audio_seconds`, `min_text_words`, and
`max_text_wait_seconds`. These unused options are absent from app defaults and exports.
The output settings use the shared TTS engine; `budget_mode` must be `audio_duration`.
Older exported app configs containing unsupported settings must also remove them.

Real debates reuse matching claim pools from `../results/deepseek-chat` when both
`<motion_with_underscores>_pool_for.json` and `_pool_against.json` exist (lowercase
motion, spaces replaced by underscores). Each AI loads its own side's file and
the paired opponent file through the engine's `pool_file` setting. The writing
motion uses these saved pools instead of generating fresh claims. Other motions
without a complete pair generate pools as before. This applies when a new worker
prepares; an already prepared debate retains its existing claims.

The app resolves engine streaming defaults through `../src/streaming/config.py`. App input
defaults are 3 seconds of audio for transcription, with tree analysis after 60 pending
words or 15 seconds since the first pending transcript batch. Finishing a turn
flushes remaining text regardless of these thresholds. Human capture is
mono signed 16-bit little-endian PCM, normally sent in 100 ms frames at the browser's
actual capture sample rate. Up to eight frames can be in transit at once, so uploads
can keep pace when acknowledgement latency exceeds a single frame's duration.
ASR WAV files preserve that rate; Whisper handles decoding.

The human deadline begins on backend capture authorization and includes silence and
connection interruptions, but excludes explicit debate pauses. Capture epochs map sample offsets to the calibrated server
clock. A 3-second default grace accepts delayed pre-deadline samples; it does not extend
speaking time. AI budgets target audible duration with `budget_mode: audio_duration`;
pauses and generation delays do not consume the target. The existing CLI retains
`experiment_elapsed` accounting. TTS duration is a soft target, not a hard audio cutoff.

Each turn shows its waiting duration, starting when that side's turn begins (including
initial preparation when AI opens). Human waiting ends when microphone capture is
authorized; AI waiting ends when the first TTS chunk is published for playback.
The AI measure excludes browser download/decoding time and playback permission delays.
The final duration remains visible and survives refresh. Stop or failure freezes an
unfinished wait. Explicit debate pauses are excluded from waiting time. Existing
sessions created before this feature have no waiting timestamps.

Optional environment variables:

- `DEBATE_APP_DATA`: absolute storage directory, default `debate-app/var`.
- `DEBATE_APP_ORIGINS`: comma-separated allowed browser origins.
- `DEBATE_APP_MAX_ACTIVE_SESSIONS`: positive concurrent debate limit, default `2`.
- `DEBATE_APP_RECONNECT_GRACE_SECONDS`: positive browser absence timeout, default `120`.
- `DEBATE_APP_UNSTARTED_TIMEOUT_SECONDS`: positive creation-to-start timeout, default `120`.
- `NEXT_PUBLIC_DEBATE_API`: frontend API base URL, default the browser’s current origin.
  Set it before building the frontend when using a separately hosted HTTPS API.

Model keys stay in backend environment variables or the existing engine credential
configuration. Never put keys in a `NEXT_PUBLIC_*` variable. API initialization imports
only configuration dataclasses; heavy model imports happen in each worker process.

## Reliability and recovery

- Session creation saves a random retry key and the original settings in the
  controlling tab before sending the request. If its response is lost, retry
  **Start debate**, including after reload, to retrieve the same session and token.
  **Retry start** resumes a session whose start request failed. Creation retries
  after a service restart retrieve the archived session; model runs are not resumed.
- Setup fields and imported configurations retain user edits when backend defaults
  arrive late or load again after a connection failure. Untouched fields still receive
  the backend defaults.
- SQLite stores snapshots, sequenced events, idempotent commands, and frame receipts.
  Acknowledgements follow PCM spooling and metadata persistence, independently of ASR.
- Duplicate audio frames are checked by digest. Missing frames request a resend; later
  frames cannot silently fill a discontinuity. An inconsistent resend cursor stops capture
  with an error instead of repeatedly submitting the same frame.
- The browser buffers up to 10 seconds of unacknowledged audio. Transport reconnect can
  resend buffered frames within the same capture epoch. Page reload cannot recover
  samples that existed only in browser memory.
- Microphone uploads use a bounded window of eight ordered frames. Cumulative
  acknowledgements release confirmed frames, and reconnect reconciles the saved
  epoch cursor before sending again. Responses from an old connection cannot affect
  the new one. Deadline truncation preserves the final accepted sample count while
  ignoring expected rejections of later frames already in transit.
- Resume retains the deadline, including after microphone startup fails. If finishing
  fails, **Retry finish** resends buffered frames and the same final marker without
  creating a new recording. Empty-turn recovery rejects audio that has not yet been
  transcribed.
- Re-record restores pre-turn tree state and permits one
  replacement recording by default. Processing retry restores trees and reprocesses the
  retained recording, publishing a new analysis revision.
- Recovery retains the recording's attempt ID while waiting for outstanding work
  and checkpoint restoration. A failed restore can be retried without losing the
  recording. Stop takes precedence over recovery; late results cannot replace a
  cancelled turn's state. An unexpected worker exit fails the session and retains
  its artifacts instead of offering retries against a dead process.
- Both engine trees, conversation/thoughts, and mutable preparation caches are included
  in worker checkpoints. Completed transcript ranges and analysis flags are saved with
  the session. Successful microphone analysis sets the engine's skip-duplicate marker.
- Stop bypasses the ingestion lock and terminates the isolated worker if necessary.
  A model timeout fails the session rather than reusing potentially inconsistent state.
  AI-versus-AI failures and Stop terminate both side workers and cancel the turn tasks.
- Stop pauses local playback while the request is pending. If the request fails,
  **Resume audio** can continue the queued playback; confirmed Stop disposes the player.
- SSE replay and persisted snapshots support browser refresh. Snapshot reads are
  coalesced and older responses cannot roll back the displayed state. Invalid local
  session data is cleared so setup can still load. Initial session restoration retries
  temporary failures every three seconds and blocks a competing Start until restoration
  succeeds or the saved session is confirmed unavailable. Service/worker restart does
  **not** resume a model run: interrupted sessions become failed, preserving artifacts.
- Session tokens are bearer capabilities stored in the controlling tab's session storage.
  This first version is for trusted local use, not a public authenticated multi-user service.

## Files and API

```text
backend/debate_app/      API, sessions/runner, storage, worker, engine adapter
frontend/               Sites frontend, AudioWorklet, queued audio player, browser tests
configs/                app examples
scripts/                startup helpers
var/app.sqlite3         durable state and event/command/frame ledgers (ignored by Git)
var/sessions/<id>/      resolved config, microphone PCM/WAV, AI chunks, engine artifacts
```

API documentation is available at http://localhost:8000/docs. Microphone WebSocket messages
use protocol version 1: `clock`, `epoch`, `frame`, and `finish`. Requests have a
`request_id`; successful replies are `ack`, failures are `error`. Frames carry turn,
attempt, epoch, sequence, sample offset/count, and base64 PCM. Finish markers declare the
last sequence and total samples. The server acknowledges its highest contiguous sequence.

Session creation accepts an `Idempotency-Key` header (32–128 characters) together
with `X-Controller-ID`. Reusing both headers and the same settings returns the
original session's current snapshot and token, including after restart. Use a
cryptographically random key and retain it until the returned credentials are saved;
the key is a recovery capability and must stay private. Changed settings with the
same key are rejected. Requests without a key remain supported for older API clients.

## Tests

```bash
cd backend
python -m pip install -e '.[test]'
python -m pytest -q

# Existing engine regressions, from TreeDebater root:
python -m pytest -q tests

# With frontend and backend running:
cd debate-app/frontend
npx playwright install chromium
npx playwright test
# Opt-in real speech test: uses excerpts from all six MP3s, once with the
# human For and once Against. Calls real ASR, debate, TTS, and evaluation APIs.
DEBATE_APP_SPEECH_DIR=/absolute/path/to/log_files/5_outputs \
  npx playwright test tests/recorded-speech.spec.ts
# Default: 18 seconds per human turn, 30-second speaking budgets.
# Set DEBATE_APP_SPEECH_SECONDS to change the excerpt length.
# Replay every recording in full with 240/240/120-second budgets:
DEBATE_APP_FULL_SPEECH=1 \
  DEBATE_APP_SPEECH_DIR=/absolute/path/to/log_files/5_outputs \
  npx playwright test tests/recorded-speech.spec.ts --trace off
# Allow roughly 40–50 minutes for both full debates at normal playback speed.
# DEBATE_APP_TEST_API=http://127.0.0.1:18080 routes this test to an isolated backend.
# Optional: point tests at an isolated frontend/backend pair.
# DEBATE_APP_TEST_URL=http://localhost:13000 npx playwright test
npx tsc --noEmit
npm run build
```

Backend tests use a deterministic injected engine; default browser integration uses
the isolated demo worker with Chromium's virtual microphone, without model calls.
The opt-in `recorded-speech.spec.ts` instead decodes the matching MP3 for each turn
into a browser audio stream and feeds the normal AudioWorklet/WebSocket microphone
pipeline. It checks real transcripts, complete audio processing, both human sides,
and playback of all AI responses. For opens both debates, covering human-first and
AI-first starts. Each stage loads `treedebater_<stage>_<human-side>.mp3` from the
supplied directory and plays its beginning at normal speed. It requires the engine environment and API
credentials, and spends model tokens. No hardware microphone is needed for this test.
Playwright attaches the final debate snapshots to its test results.

Audio scheduling and microphone lifecycle regressions (controlled clocks/transports,
no browser or model calls):

```bash
cd debate-app/frontend
node --test unit/*.test.mjs
```

See [DEBUG_REPORT.md](DEBUG_REPORT.md) for the end-to-end debug results and real-model
smoke-test coverage.

## Deployment boundary

For a shared deployment, host the Python service and worker resources on a machine with
the engine dependencies, persistent disk, and HTTPS. Configure the frontend API URL and
allowed origin accordingly. The development proxy does not run in production; configure
a production reverse proxy for `/api` or set `NEXT_PUBLIC_DEBATE_API`. Add user authentication, per-user ownership, quotas, and an
artifact retention/deletion policy before public exposure. A Sites frontend deployment
alone cannot run the Python/GPU backend or reach a visitor's localhost API.


To run the recorded-speech test with DeepSeek for debate text and timing refinement, set `DEBATE_APP_TEST_MODEL=deepseek-v4-flash`. Set `DEBATE_APP_TEST_MODEL_TIMEOUT=900` to give each model operation up to 900 seconds; this does not change the 240/240/120 speech budgets. The test requires an available `DEEPSEEK_API_KEY`, plus the existing ASR/TTS credentials. Use `--fully-parallel --workers=2` to exercise both human sides concurrently against one debug backend with two free slots. See [the full DeepSeek recheck](reports/recorded-deepseek-full-speech-2026-09-13.md) for observed limitations.

### Adaptive delivery (optional)

Enable a short first delivery followed by timing refinement during playback:

```yaml
streaming:
  output:
    budget_mode: audio_duration
    adaptive_delivery: true
    first_chunk_seconds: 12
    later_chunk_seconds: 30
```

Merge these settings into your imported app configuration. `adaptive_delivery` defaults to `false`; the two chunk-size settings take effect when it is enabled. Sizes are initial text-duration estimates, not hard audio limits. The live app uses `audio_duration`; standalone experiments can still choose `experiment_elapsed`, which also deducts preparation overruns.

This feature splits paragraphs at sentence boundaries where possible, including speeches returned as a single paragraph. Very long sentences are divided at word boundaries. It sends the first chunk straight to TTS, without an LLM timing rewrite or speculative refinement before that first delivery. It then measures speech rate from delivered audio and refines later chunks while playback continues. Remaining targets reflect actual audio already generated. Candidates must satisfy the current target using their synthesized duration; previously accepted prestarted chunks are rechecked when their target changes. Timing profiles report actual duration compliance, including for the unrefined first chunk.

The feature does not change stance or repair arguments. Existing refinement limits, preparation deadlines, and speed-adjustment bounds still apply, so it cannot guarantee an exact total duration or gap-free playback. A speech too short to split remains a single fast chunk. Smaller first chunks reduce initial synthesis work but also leave less playback time to prepare the next chunk.
