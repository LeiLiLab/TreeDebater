# Public TreeDebater website

## Local debugging alongside the public website

The same frontend source supports hot reload at `http://localhost:3000`.
Local edits appear there immediately; publishing to GitHub Pages is a separate step.

A `debate-debug` tmux session has two windows: `backend` on port 8000 and
`frontend` on port 3000. Attach with `tmux attach -t debate-debug`; switch windows
with Ctrl+B then N. Detach with Ctrl+B then D. These processes do not restart
automatically after a reboot.

The debug backend uses `debate-app/var/debug`, `DEBATE_APP_PUBLIC=0`, and allows
the localhost frontend origins. Public sessions remain on port 18090 with their
separate data. Debug mode still uses real provider credentials for real-model debates.

To start manually from the TreeDebater root (when these ports are free):

```bash
conda activate debate
DEBATE_APP_PUBLIC=0 DEBATE_APP_DATA="$PWD/debate-app/var/debug" \
  DEBATE_APP_ORIGINS=http://localhost:3000,http://127.0.0.1:3000 \
  python -m uvicorn debate_app.api:app --host 127.0.0.1 --port 8000 --no-access-log
```

In a second terminal, from the same root:

```bash
NEXT_PUBLIC_DEBATE_PUBLIC=false NEXT_PUBLIC_DEBATE_API= \
  NEXT_PUBLIC_DEBATE_BASE_PATH=/ DEBATE_APP_PROXY_TARGET=http://127.0.0.1:8000 \
  make app-frontend
```

With VS Code Remote SSH, forward server port 3000 using the Ports panel and open
the forwarded localhost address. Only port 3000 needs forwarding: the frontend
proxies API requests and microphone WebSockets to port 8000 on the server.

## Public deployment

Public page: https://dqwang122.github.io/projects/Debate/debate-app/

The app is a static React build published in the `gh-pages` branch of
`dqwang122/dqwang122.github.io`, under `projects/Debate/debate-app/`. The existing
Debate project page links to it. The checkout is alongside TreeDebater at
`../dqwang122.github.io`.

Build it with `npm run build:pages` from `debate-app/frontend`. Copy the contents
of `dist-pages/` into that website directory. Keep `connection.json` in that
directory with `{"api":"https://YOUR-BACKEND-HOST"}`. This contains a public
address, never a provider key. Commit and push the website's `gh-pages` branch
to publish; GitHub Pages may take several minutes to update.

GitHub Pages serves the interface and microphone worklet. The browser reads
`connection.json` and connects directly to the Python HTTPS backend. Its CORS
and WebSocket Origin allowlist includes `https://dqwang122.github.io`.
The current backend address is `https://dqwang122.com`, served through the existing
Cloudflare tunnel to port 18090. Preserve `connection.json` when copying a new build.
If the backend address changes, update that file, push it, and reload the app after
publication.

The earlier Sites deployment is described below for reference; its runtime
variable does not configure the GitHub Pages copy.

The frontend is deployed to Sites. The Python debate engine runs separately on
this research server, bound to loopback port 18090, with isolated data under
`debate-app/var/public`. The original services on ports 3000 and 8000 are unchanged.
Provider keys remain in the Python environment / existing engine configuration;
they are not included in the frontend source or deployment archive.

## Current availability

This public research demo runs on the research server. Its backend process and
Cloudflare tunnel must remain running. The public page URL stays the same, but
debates become unavailable when the server or tunnel stops. The launcher does not
configure automatic restart after a machine reboot.

For permanent availability, deploy this Python service and the engine dependencies
on an always-on host with persistent disk and a stable HTTPS endpoint supporting
SSE and WebSockets. Point `connection.json` to that endpoint and publish the file.
No frontend rebuild is required. Keep the public page origin in `DEBATE_APP_ORIGINS`.

## Public profile

The public launcher sets four concurrent slots. The general backend default remains
two; `DEBATE_APP_MAX_ACTIVE_SESSIONS` overrides either setting. Keep one Uvicorn
process because session ownership is held in process memory. Deploy the matching
frontend before restarting the backend so browsers send the required heartbeats.
See [Concurrent debates](README.md#concurrent-debates) for configuration and cleanup.

- Fixed GPT-4o-mini main, helper, and speech revision settings. Transcription
  and speech synthesis still use the existing OpenAI services.
- Opening/rebuttal at most 240 seconds, closing at most 120 seconds.
- At most 20 created sessions per UTC day (including demo sessions). The cap is
  persisted in SQLite; retries of the same creation request do not use extra slots.
- Up to four active debates, with a server-busy response when full. A slot remains
  occupied through evaluation and worker cleanup. One browser controller can hold
  one active session; this is not an authenticated per-person quota.
- Browser heartbeats run every 20 seconds. A two-minute absence (including while
  paused), two minutes without starting, or the public 40-minute lifetime ends a
  session, with up to five seconds of sweep delay. Final results remain available
  with the session token after workers and in-memory state are released.
- Existing session capabilities protect recordings and results. There are no
  visitor accounts. A saved session token grants access to that session.
- The page discloses that audio is sent to AI services and recordings/transcripts
  remain on the research server. Automatic retention/deletion is not implemented.
- These are usage limits, not a precise monetary spending cap. Model API usage
  is charged to the server's configured provider accounts.

Local app settings remain unrestricted unless `DEBATE_APP_PUBLIC=1` is set.

## Start or restore the backend

Use the existing `debate` environment, with the editable backend and engine
dependencies installed. Do not start a second process on an occupied port.

```bash
conda activate debate
bash debate-app/scripts/run-public-backend.sh
```

Keep the existing Cloudflare tunnel running with its route to `127.0.0.1:18090`.
Its credentials are managed separately and must not be added to this repository.

For a temporary fallback only, create a public connection in another terminal:

```bash
ssh -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
  -R 80:127.0.0.1:18090 nokey@localhost.run
```

Set `connection.json` to the returned `https://...` address and publish it. Keep
both processes running. A temporary tunnel restart can produce a different address.

The currently launched backend process is recorded in
`var/public/process.json`; its log is `var/public/backend.log`. Access logging
is disabled because EventSource and WebSocket URLs carry session tokens.

## Deployment details

The frontend has its own Git repository, rooted at `debate-app/frontend`, as
required by Sites. Its `.openai/hosting.json` identifies the existing site; do
not create another site when updating it. Build using Node 22:

```bash
cd debate-app/frontend
NEXT_PUBLIC_DEBATE_PUBLIC=true npm run build
```

The browser first reads `/api/connection` on the public page to discover the
runtime backend address. It then connects directly to that HTTPS origin for
REST, SSE, audio, and microphone WebSockets. CORS allows the public page origin.
The hosted same-origin proxy buffered SSE during verification, so the live
browser does not use that proxy for debate traffic.

The frontend source/archive exclude local environment files, test results,
session recordings, engine source, and API key configuration.
