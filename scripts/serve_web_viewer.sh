#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_WEB_ROOT="${HOME}/projects/evasion-paths-experiments"
WEB_ROOT="${WEB_ROOT:-${DEFAULT_WEB_ROOT}}"

python "${REPO_ROOT}/experiments/serve_web_viewer.py" \
  --web-root "${WEB_ROOT}"
