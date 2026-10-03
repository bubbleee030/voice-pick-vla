# -*- coding: utf-8 -*-
import copy
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import serial
from flask import Flask, jsonify, request

# --- 設定區 ---
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
READ_TIMEOUT_S = 0.2
RECONNECT_INTERVAL_S = 1.5
STALE_AFTER_S = 2.0
CHANNEL_COUNT = 3

app = Flask(__name__)

serial_lock = threading.Lock()
state_lock = threading.Lock()
reader_stop = threading.Event()
ser = None  # type: Optional[serial.Serial]


def _new_channel_state(index: int) -> Dict[str, Any]:
    return {
        "label": f"s{index + 1}",
        "value": 0,
        "ok": False,
        "last_update_unix": None,
        "last_error": "no data yet",
    }


runtime_state: Dict[str, Any] = {
    "connected": False,
    "serial_port": SERIAL_PORT,
    "baud_rate": BAUD_RATE,
    "last_error": "",
    "last_raw_line": "",
    "last_read_unix": None,
    "last_good_unix": None,
    "parsed_channels": 0,
    "read_count": 0,
    "parse_fail_count": 0,
    "serial_fail_count": 0,
    "channels": [_new_channel_state(i) for i in range(CHANNEL_COUNT)],
}


def _set_serial_state(connected: bool, error: str = "") -> None:
    with state_lock:
        runtime_state["connected"] = connected
        runtime_state["last_error"] = error


def _open_serial() -> bool:
    global ser
    try:
        new_ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=READ_TIMEOUT_S)
        time.sleep(2.0)
        with serial_lock:
            ser = new_ser
        _set_serial_state(True, "")
        print(f"Arduino serial connected on {SERIAL_PORT} @ {BAUD_RATE}")
        return True
    except Exception as exc:
        _set_serial_state(False, f"open failed: {exc}")
        print(f"無法連接 Arduino，錯誤訊息：{exc}")
        return False


def _close_serial() -> None:
    global ser
    with serial_lock:
        current = ser
        ser = None
    try:
        if current is not None and current.is_open:
            current.close()
    except Exception:
        pass


def _status_from_snapshot(snapshot: Dict[str, Any]) -> str:
    channels = snapshot.get("channels", [])
    ok_count = sum(1 for ch in channels if ch.get("ok"))
    if ok_count == CHANNEL_COUNT:
        return "success"
    if ok_count > 0 or snapshot.get("connected"):
        return "degraded"
    return "error"


def _parse_sensor_line(raw_line: str) -> Tuple[List[Optional[int]], List[str]]:
    tokens = re.findall(r"-?\d+", raw_line)
    values = []  # type: List[Optional[int]]
    errors = []  # type: List[str]
    for idx in range(CHANNEL_COUNT):
        if idx >= len(tokens):
            values.append(None)
            errors.append("missing in latest frame")
            continue
        try:
            values.append(int(tokens[idx]))
            errors.append("")
        except ValueError:
            values.append(None)
            errors.append(f"invalid token: {tokens[idx]}")
    return values, errors


def _update_from_line(raw_line: str) -> None:
    now = time.time()
    values, errors = _parse_sensor_line(raw_line)
    parsed_channels = sum(1 for value in values if value is not None)
    with state_lock:
        runtime_state["last_raw_line"] = raw_line
        runtime_state["last_read_unix"] = now
        runtime_state["read_count"] += 1
        runtime_state["parsed_channels"] = parsed_channels
        if parsed_channels == 0:
            runtime_state["parse_fail_count"] += 1
            runtime_state["last_error"] = f"parse failed: {raw_line}"
        else:
            runtime_state["last_error"] = ""
            runtime_state["last_good_unix"] = now

        channels = runtime_state["channels"]
        for idx, channel in enumerate(channels):
            value = values[idx]
            if value is None:
                channel["ok"] = False
                channel["last_error"] = errors[idx]
                continue
            channel["value"] = value
            channel["ok"] = True
            channel["last_update_unix"] = now
            channel["last_error"] = ""


def _mark_serial_failure(exc: Exception) -> None:
    with state_lock:
        runtime_state["connected"] = False
        runtime_state["serial_fail_count"] += 1
        runtime_state["last_error"] = str(exc)


def _mark_stale_channels() -> None:
    now = time.time()
    with state_lock:
        for channel in runtime_state["channels"]:
            last_update = channel.get("last_update_unix")
            if not last_update:
                continue
            if now - float(last_update) > STALE_AFTER_S:
                channel["ok"] = False
                if not channel.get("last_error"):
                    channel["last_error"] = "stale"


def _reader_loop() -> None:
    global ser
    while not reader_stop.is_set():
        current = ser
        if current is None or not current.is_open:
            _close_serial()
            if not _open_serial():
                time.sleep(RECONNECT_INTERVAL_S)
                continue
            current = ser

        try:
            assert current is not None
            with serial_lock:
                raw_bytes = current.readline()
            if not raw_bytes:
                _mark_stale_channels()
                time.sleep(0.02)
                continue
            raw_line = raw_bytes.decode("utf-8", errors="ignore").strip()
            if not raw_line:
                _mark_stale_channels()
                time.sleep(0.02)
                continue
            _set_serial_state(True, "")
            _update_from_line(raw_line)
        except serial.SerialException as exc:
            _mark_serial_failure(exc)
            _close_serial()
            time.sleep(RECONNECT_INTERVAL_S)
        except Exception as exc:
            _mark_serial_failure(exc)
            time.sleep(0.05)


reader_thread = threading.Thread(target=_reader_loop, daemon=True)
reader_thread.start()


def _snapshot_state() -> Dict[str, Any]:
    with state_lock:
        snapshot = copy.deepcopy(runtime_state)
    status = _status_from_snapshot(snapshot)
    snapshot["status"] = status
    snapshot["analog_value1"] = int(snapshot["channels"][0]["value"])
    snapshot["analog_value2"] = int(snapshot["channels"][1]["value"])
    snapshot["analog_value3"] = int(snapshot["channels"][2]["value"])
    return snapshot


@app.route("/get_sensor", methods=["GET"])
def agx_get_sensor():
    snapshot = _snapshot_state()
    verbose = str(request.args.get("verbose", "0")).lower() in {"1", "true", "yes"}
    response = {
        "status": snapshot["status"],
        "connected": snapshot["connected"],
        "analog_value1": snapshot["analog_value1"],
        "analog_value2": snapshot["analog_value2"],
        "analog_value3": snapshot["analog_value3"],
    }
    if verbose:
        response.update({
            "parsed_channels": snapshot["parsed_channels"],
            "channels": snapshot["channels"],
            "last_error": snapshot["last_error"] or None,
            "last_raw_line": snapshot["last_raw_line"] or None,
            "last_read_unix": snapshot["last_read_unix"],
            "last_good_unix": snapshot["last_good_unix"],
        })
    return jsonify(response), 200


@app.route("/debug/state", methods=["GET"])
def debug_state():
    return jsonify(_snapshot_state()), 200


@app.route("/health", methods=["GET"])
def health():
    snapshot = _snapshot_state()
    return jsonify({
        "status": snapshot["status"],
        "connected": snapshot["connected"],
        "last_error": snapshot["last_error"] or None,
    }), 200


if __name__ == "__main__":
    try:
        print("AGX tactile sensor API starting...")
        app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False, threaded=True)
    finally:
        reader_stop.set()
        _close_serial()
