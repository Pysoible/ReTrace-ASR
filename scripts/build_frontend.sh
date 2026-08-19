#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FRONTEND="$ROOT/frontend"
NODE_HOME="${RETRACE_NODE_HOME:-$ROOT/.tools/node}"

if [[ -x "$NODE_HOME/bin/npm" ]]; then
  export PATH="$NODE_HOME/bin:$PATH"
fi

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  printf '%s\n' "Node.js/npm not found. Set RETRACE_NODE_HOME to a Node.js installation." >&2
  exit 127
fi

cd "$FRONTEND"
if [[ ! -d node_modules ]]; then
  npm install
fi
npm run build
