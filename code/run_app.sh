#!/usr/bin/env bash
# Start the Shiny viewer. Reads only DR_PRECOMPUTED_DIR (default: the first /data/* folder that contains
# sessions_manifest.json, else ./data/precomputed). Usage: code/run_app.sh [port]   (default port 8080)
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${1:-${PORT:-8080}}"
if [ -z "${DR_PRECOMPUTED_DIR:-}" ] && [ -d /data ]; then
  m=$(find /data -maxdepth 3 -name sessions_manifest.json 2>/dev/null | head -1 || true)
  [ -n "$m" ] && export DR_PRECOMPUTED_DIR="$(dirname "$m")"
fi
export DR_PRECOMPUTED_DIR="${DR_PRECOMPUTED_DIR:-$PWD/data/precomputed}"
export MPLBACKEND=Agg
echo "DR_PRECOMPUTED_DIR=$DR_PRECOMPUTED_DIR  ->  http://0.0.0.0:$PORT"
exec shiny run code/app.py --host 0.0.0.0 --port "$PORT"
