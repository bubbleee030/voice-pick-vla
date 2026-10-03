#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

find_python() {
  local candidate version
  for candidate in python3 python3.12 python3.11 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then
      version="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
      if "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
      then
        echo "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

if ! PYTHON_BIN="$(find_python)"; then
  echo "[ERROR] Python 3.10+ not found."
  echo "Install it first: sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip"
  exit 1
fi

echo "================================================"
echo " Voice Pick Ubuntu Deployment"
echo "================================================"
echo "Root:   $ROOT_DIR"
echo "Python: $PYTHON_BIN"
echo

echo "[1/5] Installing Ubuntu system packages"
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  curl \
  ffmpeg \
  libgl1 \
  libglib2.0-0 \
  libgtk-3-0 \
  libusb-1.0-0 \
  libusb-1.0-0-dev \
  netcat-openbsd \
  portaudio19-dev \
  python3-pip \
  python3-venv \
  udev \
  usbutils \
  v4l-utils

echo
echo "[2/5] Creating virtual environment"
if [[ ! -d venv ]]; then
  "$PYTHON_BIN" -m venv venv
fi
VENV_PYTHON="$ROOT_DIR/venv/bin/python"
VENV_PIP="$ROOT_DIR/venv/bin/pip"

echo
echo "[3/5] Installing Python dependencies"
"$VENV_PIP" install --upgrade pip setuptools wheel
"$VENV_PIP" install -r requirements.txt

echo
echo "[4/5] Verifying core imports"
"$VENV_PYTHON" - <<'PY'
modules = [
    ("cv2", "opencv"),
    ("yaml", "pyyaml"),
    ("flask", "flask"),
    ("flask_socketio", "flask-socketio"),
    ("requests", "requests"),
    ("pyModbusTCP.client", "pyModbusTCP"),
]
optional = [
    ("pyrealsense2", "pyrealsense2"),
]
for module, label in modules:
    __import__(module)
    print(f"[OK] {label}")
for module, label in optional:
    try:
        __import__(module)
        print(f"[OK] {label}")
    except Exception as exc:
        print(f"[WARN] {label}: {exc}")
PY

echo
echo "[5/5] Preparing runtime folders and Ubuntu-local overlay"
mkdir -p \
  data/calibration \
  data/models/yolo \
  data/recordings \
  data/runtime/logs \
  data/teach_recordings

if [[ ! -f config/env/ubuntu.local.yaml && -f config/env/ubuntu.example.yaml ]]; then
  cp config/env/ubuntu.example.yaml config/env/ubuntu.local.yaml
  echo "[INFO] Created config/env/ubuntu.local.yaml from example"
fi

chmod +x scripts/run_demo.sh scripts/preflight_ubuntu.sh

echo
echo "================================================"
echo " Deployment Complete"
echo "================================================"
echo "Next:"
echo "1. Edit config/env/ubuntu.local.yaml"
echo "2. Run: bash scripts/preflight_ubuntu.sh"
echo "3. Start: bash scripts/run_demo.sh full"
echo
echo "Useful files:"
echo "- docs/UBUNTU_DEPLOY.md"
echo "- docs/CONFIG_MAP.md"
echo "- docs/TROUBLESHOOTING_UBUNTU.md"
