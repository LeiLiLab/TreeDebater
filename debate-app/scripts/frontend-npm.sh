#!/usr/bin/env bash
set -eo pipefail
app_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Conda/non-login shells may not initialize the user's nvm installation.
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  app_nvm_dir="${NVM_DIR:-$HOME/.nvm}"
  if [[ -s "$app_nvm_dir/nvm.sh" ]]; then
    source "$app_nvm_dir/nvm.sh" --no-use
    nvm use --silent default || {
      echo "Could not activate nvm's default Node. Install Node >=22.13 and set an nvm default." >&2
      exit 1
    }
  fi
fi
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  echo "Node.js >=22.13 and npm are required. Install them or initialize nvm, then retry." >&2
  exit 1
fi
node -e 'const [major, minor] = process.versions.node.split(".").map(Number); if (major < 22 || (major === 22 && minor < 13)) { console.error(`Node >=22.13 required; selected ${process.version} (${process.execPath}).`); process.exit(1); }'
cd "$app_root/frontend"
exec npm "$@"
