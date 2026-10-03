#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x venv/bin/python ]]; then
  PYTHON_BIN="$ROOT_DIR/venv/bin/python"
elif [[ -x /opt/homebrew/Caskroom/miniforge/base/envs/voice_pick/bin/python ]]; then
  PYTHON_BIN="/opt/homebrew/Caskroom/miniforge/base/envs/voice_pick/bin/python"
elif command -v conda >/dev/null 2>&1 && conda info --envs 2>/dev/null | grep -q "voice_pick"; then
  PYTHON_BIN="$(conda info --base)/envs/voice_pick/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  PYTHON_BIN="python"
fi

DEFAULT_ENV_CFG=""
if [[ -f config/env/ubuntu.local.yaml ]]; then
  DEFAULT_ENV_CFG="config/env/ubuntu.local.yaml"
fi

ACTION="${1:-start}"
shift || true

PROFILE="full"
EXTRA_ARGS=()
UI_MODE=""

case "$ACTION" in
  start)
    PROFILE="${1:-full}"
    shift || true
    ;;
  full|recording|debug-modules)
    PROFILE="$ACTION"
    ACTION="start"
    ;;
  status|stop|logs)
    ;;
  *)
    PROFILE="$ACTION"
    ACTION="start"
    ;;
esac

EXTRA_ARGS+=("$@")

for ((i=0; i<${#EXTRA_ARGS[@]}; i++)); do
  if [[ "${EXTRA_ARGS[$i]}" == "--ui-mode" ]] && (( i + 1 < ${#EXTRA_ARGS[@]} )); then
    UI_MODE="${EXTRA_ARGS[$((i + 1))]}"
  fi
done

case "$ACTION" in
  start)
    echo "Voice Pick Launcher"
    echo "Python:  $PYTHON_BIN"
    echo "Profile: $PROFILE"
    if [[ -n "$UI_MODE" ]]; then
      echo "UI:      $UI_MODE"
    fi
    if [[ -n "$DEFAULT_ENV_CFG" ]]; then
      echo "Overlay: $DEFAULT_ENV_CFG"
    fi
    echo "URL:     http://0.0.0.0:8090"
    echo
    START_ARGS=()
    if [[ -n "$DEFAULT_ENV_CFG" ]]; then
      START_ARGS+=(--env-config "$DEFAULT_ENV_CFG")
    fi
    START_ARGS+=("${EXTRA_ARGS[@]}")
    if ((${#START_ARGS[@]})); then
      exec "$PYTHON_BIN" -m src.launcher start "$PROFILE" --attach "${START_ARGS[@]}"
    fi
    exec "$PYTHON_BIN" -m src.launcher start "$PROFILE" --attach
    ;;
  status)
    if ((${#EXTRA_ARGS[@]})); then
      exec "$PYTHON_BIN" -m src.launcher status "${EXTRA_ARGS[@]}"
    fi
    exec "$PYTHON_BIN" -m src.launcher status
    ;;
  stop)
    if ((${#EXTRA_ARGS[@]})); then
      exec "$PYTHON_BIN" -m src.launcher stop "${EXTRA_ARGS[@]}"
    fi
    exec "$PYTHON_BIN" -m src.launcher stop
    ;;
  logs)
    if ((${#EXTRA_ARGS[@]})); then
      exec "$PYTHON_BIN" -m src.launcher logs "${EXTRA_ARGS[@]}"
    fi
    exec "$PYTHON_BIN" -m src.launcher logs
    ;;
  *)
    echo "Usage:"
    echo "  bash scripts/run_demo.sh [full|recording|debug-modules]"
    echo "  bash scripts/run_demo.sh start [profile] [--ui-mode modern|classic]"
    echo "  bash scripts/run_demo.sh full --ui-mode classic"
    echo "  bash scripts/run_demo.sh status"
    echo "  bash scripts/run_demo.sh stop"
    echo "  bash scripts/run_demo.sh logs"
    exit 1
    ;;
esac
