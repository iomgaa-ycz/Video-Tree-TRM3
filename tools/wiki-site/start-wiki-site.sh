#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WIKI_DIR="$PROJECT_ROOT/research-wiki"
PID_FILE="$PROJECT_ROOT/.wiki-site/.pid"

if [ ! -d "$WIKI_DIR" ]; then
    exit 0
fi

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        exit 0
    fi
fi

cd "$PROJECT_ROOT"
if [ ! -d "tools/wiki-site/app/node_modules" ]; then
    echo "[wiki-site] Installing npm dependencies..."
    (cd tools/wiki-site/app && npm install) || true
fi

mkdir -p .wiki-site
nohup python -m tools.wiki_site.renderer "$WIKI_DIR" > .wiki-site/renderer.log 2>&1 &
echo "[wiki-site] Renderer started (PID $!), open http://localhost:8686"
