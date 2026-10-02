#!/usr/bin/env bash
set -euo pipefail
app_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$app_root/.env" ]]; then
  set -a
  source "$app_root/.env"
  set +a
fi
app_python="${DEBATE_APP_PYTHON:-python}"
"$app_python" - <<'PY'
import sys
if sys.version_info < (3, 10):
    sys.exit(
        "TreeDebater requires Python 3.10 or newer. "
        f"Selected: {sys.executable} (Python {sys.version.split()[0]}).\n"
        "Activate the debate environment with 'conda activate debate', or set "
        "DEBATE_APP_PYTHON to its Python executable, then restart the backend."
    )
PY
cd "$app_root/backend"
exec "$app_python" -m uvicorn debate_app.api:app --host 127.0.0.1 --port 8000 --no-access-log "$@"
