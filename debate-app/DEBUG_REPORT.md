# App debug report — updated 2026-09-13

## GitHub Pages app URL — 2026-09-13

Published the static interface under `/projects/Debate/debate-app/` in the
personal website repository, with a Live Debate App link on the project page.
The dedicated `build:pages` target preserves the existing Sites build. Asset,
microphone-worklet, and app-home URLs include the GitHub Pages subdirectory.
The static app reads its HTTPS backend address from `connection.json`.

Production static build and TypeScript validation passed. With
`Origin: https://dqwang122.github.io`, the backend passed CORS preflight,
SSE event delivery, microphone WebSocket clock acknowledgment, and rejection
of unauthenticated session access. Test sessions were stopped. The isolated
public backend was restarted to allow the new origin; 3000/8000 were untouched.
The free tunnel rotated during setup, and the published connection file was
updated to the new address. Permanent backend availability is still outstanding.

## Public website — 2026-09-13

[Public demo](https://treedebater-live.danqingw63871.chatgpt.site) is deployed.
[Deployment and availability notes](PUBLIC_WEBSITE.md) describe the temporary
backend tunnel, isolated data, fixed public model settings, and usage limits.

- Frontend production build and TypeScript validation passed; backend suite: 73 tests passed.
- Public page and runtime connection endpoint returned successfully without login.
- Direct HTTPS backend SSE delivered a session event; a microphone WebSocket
  acknowledged a clock request with the public page's Origin. CORS preflight passed,
  and session access without a token returned 403.
- Real DeepSeek Flash smoke test with a 10-second opening budget produced a
  127,872-byte audio/mpeg response 36.1 seconds after session start. The test was
  stopped after the first chunk; this is not a full debate or browser playback test.
- The initial same-origin hosted proxy buffered SSE. The deployed frontend now
  discovers the backend through `/api/connection` and connects directly for all
  debate traffic. That public HTTPS route passed the streaming checks above.
- Smoke sessions were stopped. Existing local services on 3000/8000 were preserved.
- Permanent backend hosting remains outstanding: the free tunnel address can
  expire/change after a few hours, and there is no automatic address update.

## Full 4/4/2-minute replay — 2026-09-13

[Full comparison with `log_files/5.log` and `5.json`](reports/recorded-full-speech-2026-09-13.md)
includes transcripts, timing tables, intermediate tree evidence, and the failure snapshot.
All six MP3s were replayed in full with 240/240/120-second budgets.

- **Human Against passed; human For failed at closing.** A final 848-sample
  fragment (19.23 ms of silence) was sent to Whisper, which requires at least
  100 ms. The turn entered recovery and prevented the last AI response. Ten of
  twelve debate turns completed; this is not a full end-to-end pass.
- ASR text differed from normalized reference text by **8.8–19.0% word edits**,
  with meaning-changing insertions and omissions propagating into tree claims.
  This is a text-comparison estimate, not a manually scored ASR benchmark.
- The AI Against opening/rebuttal remained inconsistent with its assigned stance.
  Length revision reversed an opening sentence, and the rebuttal attributed its
  own earlier arguments to the human opponent.
- Extraction retained an opponent-standard quotation as the Against speaker's
  own supporting claim, even after the complete opening was processed.
- Disk-backed attempts suffered buffer-full interruptions. RAM-backed test storage
  allowed all recordings to be captured; disk-latency causality remains unproven.

The simulator was corrected to wait for capture readiness before starting a file.
The reference used `deepseek-chat` and non-streaming TTS; the replay used the app's
`gpt-4o-mini` and streaming paths, so identical generated responses were not expected.
Judge evaluation was not validated because test cleanup interrupted its worker.

## Recorded microphone tests — 2026-09-13

Detailed results, recognized excerpts, audio durations, and rerun instructions:
[Recorded microphone acceptance test](reports/recorded-speech-2026-09-13.md).

Used 18-second excerpts from all six recordings in `log_files/5_outputs` to run
two real-model debates: human For versus AI Against, and AI For versus human
Against. Both included opening, rebuttal, and closing. The tests used an isolated
backend and preserved the existing services on ports 3000 and 8000.

- **Both automated browser flow tests passed** in 7.1 minutes. All 12 turns
  completed, all received human audio was processed without reported omissions,
  all 41 nonempty ASR batches were analyzed, and all six AI speeches played fully.
- **Side-adherence failure remains:** the AI assigned Against supported the motion
  in its opening, rebuttal, and closing. Correct side assignment and successful
  transport do not establish that generated arguments follow the assigned side.
- **Transcription quality issue remains:** independent 3-second ASR batches
  fragmented some words and phrases. Recognizable transcripts were not verbatim.
- Fixed repeated audio-context closure in the simulated microphone fixture.
  The final run had no uncaught browser errors. TypeScript checks passed.

This extends the earlier real-engine smoke test to both sides and all stages, but
uses excerpts rather than full-length speeches. Physical microphone behavior,
audible sound quality, and judge quality were not assessed in this run.

## Initial debug run — 2026-09-09

The app was exercised using a separate backend and data directory, Chromium's virtual
microphone, and the existing Python 3.10 debate environment. The existing development
server and debate data were preserved.

## Fixes

| Problem reproduced | Corrected behavior |
|---|---|
| Corrupt saved session JSON stopped setup initialization. | Validate saved credentials and clear unusable data without blocking setup. |
| Concurrent snapshot requests let a delayed response roll the UI backward. | Coalesce event-triggered reads and reject older or wrong-session snapshots. |
| A device error after authorization left a human turn without a retry control. | Resume capture using the original attempt and deadline. |
| A failed finish request permanently cached its rejected promise. | Retry finishing with the same idempotency key; reconnect and resend buffered audio when needed. |
| Stopping during a permission request could still open a microphone afterward. | Release late-arriving streams, abort pending startup, and settle outstanding microphone requests. |
| A rejected audio context could leave the microphone check's stream open. | Clean up the stream on all setup exits. |
| An inconsistent resend cursor caused repeated submission of the same frame. | Stop with an actionable error instead of an endless resend loop. |
| A rejected overlapping capture epoch closed the valid current epoch. | Validate the new epoch before changing the current one. |
| “Skip empty” could discard received, untranscribed audio. | Require received audio to be transcribed before treating a turn as empty. |
| An ASR result arriving after analysis failed could start another analysis. | Stop processing stale results while recovery is required. |
| Malformed WebSocket messages and non-object config imports caused server errors. | Return validation errors or an appropriate WebSocket close; keep valid connections usable. |
| Claim-pool exports wrote the old pool instead of the generated side's pool. | Write the correct generated data. |
| Slashes or very long debate motions produced invalid artifact paths. | Keep filenames bounded and safe, with a digest to distinguish sanitized names. |

Preparation error text now distinguishes retrying the backend/model setup from changing
the settings of a debate. Existing AI listener overlap and audio scheduling behavior
remains covered by regression tests.

## Verification

- Backend suite: **48 tests passed**.
- Existing engine suite: **48 tests passed**.
- Audio/microphone unit suite: **13 tests passed**.
- Browser suite: **15 tests passed**.
- TypeScript and production build: **passed**.
- Backend tests also passed under the Python 3.10 debate environment.

The browser suite covers full human-versus-AI and AI-versus-AI debates, live transcripts,
automatic playback, pause/reload/resume, microphone reconnect, rerecording, automatic
speaking deadlines, lost finish messages, configuration validation/import, artifact
exports, corrupt session data, snapshot ordering, permission failure, backend
unavailability, timer freezing, and a narrow screen.

## Real-engine smoke test

A bounded run with `gpt-4o-mini`, a one-claim pool, the normal 300-second model timeout,
and a 15-second speech target completed:

1. TreeDebater preparation and evidence retrieval.
2. Real speech generation and TTS publication: **one MP3 chunk, 15.816 seconds**.
3. Whisper transcription: **251 characters**.
4. Tree analysis of that transcript.

An initial 90-second preparation limit expired during evidence retrieval. Repeating
with the app's normal 300-second limit passed; the shorter limit was not used to change
application defaults.

## Limits

At the time of the initial September 9 run, the real-model test covered one AI
side's opening and its audio/analysis path. Complete
six-turn debates were tested with the demo engine. Physical microphones, audible sound
quality, and non-Chromium browsers were not verified. No public deployment was made.

## Review follow-up — 2026-09-09

The four subsequent review findings are fixed:

- Creation retries recover the same session and token using a random client key
  scoped to the controlling tab. The browser persists the key and original settings
  before sending, and saves credentials before clearing the pending request. Retries
  also work after reload or service restart; restarted sessions remain archived.
  A failed start request has a **Retry start** control.
- Recovery retains the attempt ID while waiting for transcription, analysis, and
  checkpoint restoration. A recoverable restoration error preserves the audio and
  transcripts and allows the same recovery command to be retried.
- Unexpected worker process exits close the executor and fail the session while
  preserving recordings. Recovery no longer reuses a dead worker.
- Stop takes precedence at each recovery wait. Late transcription and restoration
  results cannot replace a cancelled turn or exclude its recording.

Verification for this follow-up:

- Backend: **58 tests passed**, including creation replay after restart, restoration
  failure/retry, Stop during transcription/restoration, and actual demo worker exits.
- The same **58 tests passed** using `unittest` in the Python 3.10 debate environment
  (that environment does not have `pytest` installed).
- Frontend unit suite: **20 tests passed**, including five creation retry/storage
  regressions. TypeScript and the production build passed.
- The new creation helper and its tests pass lint. `app/page.tsx` still reports its
  existing explicit-`any`, effect/dependency, and accessibility/navigation lint errors.
- Browser integration and real-model smoke tests were not rerun in this follow-up.
  No model API calls or public deployment were made.

## UI interaction follow-up — 2026-09-09

The three UI interaction findings are fixed:

- A failed Stop request leaves paused playback available to **Resume audio**.
  Confirmed Stop disposes the player and disables playback controls. If audio
  suspension fails, the player is cleared so Resume can recreate it. The Stop request
  is sent without waiting for audio suspension.
- Saved-session restoration retries temporary failures every three seconds, retains
  the saved credentials, and prevents competing session creation while unresolved.
  Missing sessions and invalid tokens clear the saved session. Cleanup cancels retry
  timers and ignores stale responses; effect restart uses a fresh snapshot request.
- Late backend defaults update only untouched setup fields and individual stage
  budgets. User edits, imported configurations, and partially edited advanced
  streaming JSON survive delayed loading and connection retries.

Verification for this follow-up:

- Frontend unit suite: **31 tests passed**, including **11 new UI interaction tests**
  covering defaults, restoration, effect restart, and Stop/Resume behavior. These tests
  exercise the page's rendered handlers with controlled hooks, transport, timers,
  and audio contexts; they do not run in a browser.
- TypeScript and the production build passed. The new interaction test file passes
  lint. Previously reported page lint errors remain outside this follow-up.
- Backend code was unchanged. Browser integration, physical microphone/playback,
  and real-model smoke tests were not rerun in this follow-up.

## Debate tree readability — 2026-09-09

- Replaced the raw recursive field listing with collapsible branch headings and
  **Expand all** / **Collapse all** controls. AI perspectives and side roots start
  open; deeper branches start folded. Headings support keyboard activation.
- Side, level, visit count, status, and other short values share a wrapping metadata
  row that remains visible when folded. Arguments and evidence use paragraphs and
  lists, with compact indentation on narrow screens.
- Render the engine's structured root directly, avoiding its duplicate serialized
  root summary. Preserve other analysis details and full claim text when expanded.
- Keep fold choices across live revisions; reset them when switching sessions.
- Use blue for For and orange for Against across tree headers, borders, arrows,
  and side labels, with colors for light and dark themes. Nested nodes follow
  their own side; structural groups inherit the enclosing side when needed.

Verification: all **31 existing frontend unit tests**, TypeScript checks, and the
production build passed. Browser interaction and visual checks were not run for
this change.

## Empty reference headings — 2026-09-09

The results view hides a trailing reference heading when an AI statement has no
citations beneath it. This also applies to saved sessions when displayed. Populated
reference sections and human transcripts retain their content.

Verification: **31 existing frontend unit tests**, TypeScript checks, and the
production build passed. Browser checks were not run for this change.

## Microphone upload backlog — 2026-09-09

A recent 60-second opening turn retained **49.8 seconds** of audio. Its 498 frame
acknowledgements spanned 61.19 seconds, with a median gap of **126 ms**. The client
produced a frame every 100 ms but waited for each acknowledgement before sending
the next frame, so the pending queue grew until the 10-second limit stopped capture.
A separate temporary-file check found sub-millisecond median local audio flush and
SQLite commit times; this did not reproduce the live session's full backend load.

- Keep up to eight ordered microphone frames in transit, retaining their audio until
  acknowledged. The transport protocol and durable backend acknowledgement rules
  remain compatible with existing sessions.
- Reconcile the server's cumulative cursor before resuming uploads after reconnect.
  Ignore stale responses/close notifications from old sockets, and release frames
  confirmed by a later cumulative acknowledgement even if an earlier reply was lost.
- Finish sends the final marker after queued audio drains. Both partial and exact
  frame boundaries at the deadline retain the accepted samples and ignore expected
  rejections for later frames already in transit.
- Keep the bounded outage buffer. Stopping capture also stops the worklet, and late
  frame messages cannot continue filling the queue after capture has stopped.

The regression harness uses the actual microphone class and capture worklet with
controlled audio input, WebSocket responses, and a virtual clock. At the observed
126 ms acknowledgement delay, it records and uploads all **60 seconds / 600 frames**,
with less than one second queued. Additional cases cover a full upload window,
the final partial frame, reconnect/replay, both deadline boundaries, lost replies,
and a simulated outage that fills the bounded buffer. A backend WebSocket test checks
ordered byte preservation and idempotent replay with eight frames sent together.

Verification: **38 frontend tests** and **59 backend tests passed**. TypeScript,
the production build, and lint for the new transport test file passed. No physical
microphone, browser interaction, or real-model tests were run in this follow-up.
Reload the app before the next recording to use the updated uploader.

## Transcript history — 2026-09-09

The Live tab now includes a transcript turn selector with stage, speaker, and side
labels. Users can read earlier turns while the debate continues, then use **Back to
live** or **Follow current turn** to follow the active turn again. Selections persist
as the active turn advances and are scoped to the session.

Transcription progress, incomplete-recording notices, and empty states reflect the
selected turn. Saved statement text provides a fallback when segment/chunk records
are absent. The default view shows the latest available transcript after completion.

Verification: all **38 existing frontend tests**, TypeScript checks, and the
production build passed. Browser interaction checks were not run for this change.

## Slow opening/reloading on port 3000 — 2026-09-09

The listener is on **3000**, running `vinext dev`; no service was listening on
port 300. Warm local HTTP measurements (10 sequential requests per endpoint) found:

| Request | Median | Maximum |
| --- | ---: | ---: |
| Frontend `/` | 38.0 ms | 51.2 ms |
| Proxied `/api/clock` | 1.9 ms | 3.0 ms |
| Proxied `/api/config/defaults` | 6.1 ms | 7.3 ms |
| Direct backend `/api/clock` | 0.8 ms | 2.1 ms |
| Direct backend `/api/config/defaults` | 4.0 ms | 9.4 ms |

The host had 112 logical CPUs, approximately 61% CPU idle during sampling, and
ample available memory, with no active swapping observed. These measurements do
not indicate sustained server CPU, memory, or API response pressure at that time.

The development asset payload is a substantial page-loading cost. The route
imports a **4,169,980-byte Lucide bundle**, and the browser runtime loads a
**2,819,683-byte React DOM bundle**. Both return uncompressed bodies even when the
request advertises gzip/Brotli support. An HTTP crawl of reachable JavaScript/CSS
modules found about **13.9 MB across 166 URLs**; this inventory includes deferred
imports and is not a measured browser initial-load waterfall. Versioned dependency
responses have immutable cache headers, so subsequent loads can reuse them when
browser caching is enabled and dependency versions remain unchanged.

For comparison, the existing production output contains **633,317 bytes of
JavaScript across 10 files**, including deferred chunks, plus **205,776 bytes of
CSS** and **146,464 bytes of fonts**. No external asset URLs appeared in the page's
HTML. Serving the production output with compression and preserving the Python
`/api` HTTP/WebSocket proxy is the recommended improvement for regular use over
a forwarded port. The current proxy is configured under Vite's development server;
switching the launch command alone would not preserve that integration.

Conclusion: large development assets and their module requests are the strongest
identified explanation for slow opening/reloading. Remote forwarding throughput,
browser cache behavior, and JavaScript execution time were not measured, so their
individual contributions are not established. This was a read-only runtime
diagnosis; no application code or running server was changed, and no browser or
test suite was run.


## Stance prompt follow-up — 2026-09-13

Added explicit planning ownership, separated support/counterargument fields, corrected opponent-tree perspective, and passed motion/side into timing rewrites. 55 targeted tests passed. Saved-prompt replay improved planning direction but retained revision drift; see [replay findings and limitations](reports/stance-prompt-replay-2026-09-13.md).

Rechecked the stance changes with four additional live saved-prompt calls and all 62 tests under `tests/`. Tests passed, but the model still reversed a reinforce action and retained contradictory conclusions in both rewrite cases. The [updated replay report](reports/stance-prompt-replay-2026-09-13.md#independent-recheck) records the remaining failures; the prompt changes are not a complete fix.

DeepSeek key verification succeeded, including a request to `deepseek-v4-flash` (served as `deepseek-flash` / V4.1 Flash). Four saved-prompt calls improved planning ownership in this sample but retained rewrite drift. See [DeepSeek replay findings](reports/stance-deepseek-flash-2026-09-13.md). Live app model configuration is unchanged.


## Full DeepSeek recheck — 2026-09-13

Completed both human-side scenarios using the full 4 + 4 + 2 source audio. Human Against passed at the default timeout; Human For passed on a fresh retry with a 900-second model timeout after the original attempt failed. Completed AI speeches preserved their stance, but quotation fidelity, planner truncation/retry behavior, and duration undershoots remain. See the [full DeepSeek report](reports/recorded-deepseek-full-speech-2026-09-13.md) for comparison with `5.log`/`5.json`, exact timing, and preserved model IO.

## Adaptive delivery feature — 2026-09-13

Added optional `streaming.output.adaptive_delivery`: short first audio without a timing rewrite, sentence/word splitting for single-paragraph speeches, measured-rate refinement during playback, actual-duration checks, and invalidation of stale prestart acceptance. Disabled by default. See the [feature notes and validation](reports/adaptive-delivery-feature-2026-09-13.md) and README configuration example. A new live full-audio benchmark has not been run for this feature.

## Adaptive delivery latency recheck — 2026-09-13

Real DeepSeek/TTS off/on comparisons on identical saved inputs improved closing duration from 80.26 to 119.98 seconds and opening duration from 204.00 to 237.77 seconds. First-audio readiness was 5.55→2.53 seconds for closing and 2.46→2.70 seconds for opening. Potential playback gaps increased to 8.18 and 19.00 seconds respectively. These are chunk-ready scheduling estimates, not browser measurements. See [latency report, audio, and profiles](reports/adaptive-delivery-latency-2026-09-13.md). Duration control improved, but smooth playback remains unresolved.
