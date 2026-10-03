#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE_URL="https://raw.githubusercontent.com/bubbleee030/VLAtest/main/tools"

mkdir -p tools/static

curl -L "$BASE_URL/voice_pick_demo.py" -o tools/voice_pick_demo.classic.py
curl -L "$BASE_URL/static/index.html" -o tools/static/index.classic.html
curl -L "$BASE_URL/static/app.js" -o tools/static/app.classic.js
curl -L "$BASE_URL/static/style.css" -o tools/static/style.classic.css

echo "Fetched classic UI snapshot from VLAtest:"
echo "  tools/voice_pick_demo.classic.py"
echo "  tools/static/index.classic.html"
echo "  tools/static/app.classic.js"
echo "  tools/static/style.classic.css"
