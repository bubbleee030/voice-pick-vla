#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_CFG="${1:-config/env/ubuntu.local.yaml}"

if [[ -x venv/bin/python ]]; then
  PYTHON_BIN="$ROOT_DIR/venv/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

echo "================================================"
echo " Voice Pick Ubuntu Preflight"
echo "================================================"
echo "Root:    $ROOT_DIR"
echo "Python:  $PYTHON_BIN"
echo "Overlay: $ENV_CFG"
echo

echo "[1/5] Python and launcher config"
"$PYTHON_BIN" - <<PY
from src.runtime_config import load_runtime_bundle
bundle = load_runtime_bundle(profile="full", env_config="$ENV_CFG")
print("[OK] launch profile:", bundle.profile_name)
print("[OK] arm endpoints:", len(bundle.demo_config.get("arm", {}).get("connections", [])))
print("[OK] gripper endpoints:", len(bundle.demo_config.get("gripper", {}).get("endpoints", [])))
print("[OK] claw source:", bundle.demo_config.get("cameras", {}).get("claw", {}).get("source"))
PY

echo
echo "[2/5] RealSense probe"
"$PYTHON_BIN" - <<'PY'
try:
    import pyrealsense2 as rs
    ctx = rs.context()
    devices = list(ctx.devices)
    if not devices:
        print("[WARN] No RealSense devices detected")
    for dev in devices:
        try:
            serial = dev.get_info(rs.camera_info.serial_number)
            name = dev.get_info(rs.camera_info.name)
            print(f"[OK] {serial} {name}")
        except Exception as exc:
            print(f"[WARN] Failed to read RealSense info: {exc}")
except Exception as exc:
    print(f"[WARN] pyrealsense2 unavailable: {exc}")
PY

echo
echo "[3/5] Video and serial devices"
if compgen -G "/dev/video*" >/dev/null; then
  ls -1 /dev/video*
else
  echo "[WARN] No /dev/video* devices found"
fi
if command -v v4l2-ctl >/dev/null 2>&1; then
  echo
  v4l2-ctl --list-devices || true
fi
if [[ -d /dev/serial/by-id ]]; then
  echo
  ls -1 /dev/serial/by-id
else
  echo "[WARN] /dev/serial/by-id not present"
fi

echo
echo "[4/5] Network reachability"
"$PYTHON_BIN" - <<PY
import socket
import urllib.parse
import yaml
from pathlib import Path

cfg_path = Path("$ENV_CFG")
raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
demo = raw.get("demo", raw)

def tcp_check(host, port, label):
    try:
        with socket.create_connection((host, int(port)), timeout=1.5):
            print(f"[OK] {label}: {host}:{port}")
    except Exception as exc:
        print(f"[WARN] {label}: {host}:{port} -> {exc}")

for idx, conn in enumerate(demo.get("arm", {}).get("connections", []), start=1):
    tcp_check(conn.get("host", "127.0.0.1"), conn.get("port", 1502), f"arm[{idx}]")

for idx, endpoint in enumerate(demo.get("gripper", {}).get("endpoints", []), start=1):
    tcp_check(endpoint.get("host", "127.0.0.1"), endpoint.get("port", 5000), f"gripper[{idx}]")

sensor = demo.get("sensor_api", {})
base_url = sensor.get("base_url", "")
if base_url:
    parsed = urllib.parse.urlparse(base_url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    tcp_check(parsed.hostname or "127.0.0.1", port, "sensor_api")
else:
    print("[WARN] sensor_api.base_url empty")
PY

echo
echo "[5/5] Launcher dry status"
"$PYTHON_BIN" -m src.launcher status || true

echo
echo "Preflight complete."
echo "If the warnings match disconnected hardware, fix config/env/ubuntu.local.yaml and rerun."
