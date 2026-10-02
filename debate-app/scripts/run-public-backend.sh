#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export DEBATE_APP_PUBLIC=1
export DEBATE_APP_DATA="${DEBATE_APP_DATA:-$APP_DIR/var/public}"
export DEBATE_APP_ORIGINS="${DEBATE_APP_ORIGINS:-https://treedebater-live.danqingw63871.chatgpt.site,https://dqwang122.github.io}"
export DEBATE_APP_PUBLIC_DAILY_SESSIONS="${DEBATE_APP_PUBLIC_DAILY_SESSIONS:-20}"
export DEBATE_APP_MAX_ACTIVE_SESSIONS="${DEBATE_APP_MAX_ACTIVE_SESSIONS:-4}"
cd "$APP_DIR/.."
exec "${DEBATE_APP_PYTHON:-python}" -m uvicorn debate_app.api:app \
  --host 127.0.0.1 --port "${DEBATE_APP_PUBLIC_PORT:-18090}" --no-access-log
