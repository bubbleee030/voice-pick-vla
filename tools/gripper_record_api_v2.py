from __future__ import annotations

import atexit
import csv
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

try:
    import serial
except ImportError:
    serial = None

try:
    from dynamixel_sdk import (
        GroupSyncRead,
        PacketHandler,
        PortHandler,
    )
except ImportError as exc:
    raise RuntimeError("dynamixel_sdk is required for gripper_record_CSV_API_V2.py") from exc


ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

DXL_IDS = [1, 2, 3]
ORIGIN_POS = [3072, 3072, 2048]

BAUDRATE_MOTOR = int(os.getenv("GRIPPER_MOTOR_BAUD", "57600"))
DEVICENAME_MOTOR = os.getenv(
    "GRIPPER_MOTOR_PORT",
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT6RWC9P-if00-port0",
)
BAUDRATE_SENSOR = int(os.getenv("GRIPPER_SENSOR_BAUD", "115200"))
DEVICENAME_SENSOR = os.getenv("GRIPPER_SENSOR_PORT", "/dev/ttyUSB0")
GRIPPER_SPEED = int(os.getenv("GRIPPER_STEP_TICKS", "20"))
SENSOR_FRESHNESS_TIMEOUT_S = float(os.getenv("GRIPPER_SENSOR_FRESHNESS_TIMEOUT", "0.5"))
APP_PORT = int(os.getenv("GRIPPER_V2_PORT", "5003"))
RAW_OUTPUT_ROOT = Path(os.getenv("GRIPPER_RAW_OUTPUT_ROOT", "data/agx_tactile_raw"))


@dataclass
class RecordingSession:
    session_id: str
    file_path: str
    object_name: str
    episode_index: int | None
    start_unix: float
    notes: str
    samples: int = 0


class GripperRecorderV2:
    def __init__(self):
        self.com_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.running = True
        self.shared_pos = list(ORIGIN_POS)
        self.latest_motor_pos = list(ORIGIN_POS)
        self.latest_tactile = [0, 0, 0]
        self.latest_tactile_ts_unix: float | None = None
        self.sensor_connected = False
        self.sample_valid = False
        self.sensor_last_update_unix: float | None = None
        self.sensor_disconnect_count = 0
        self.sensor_last_error = ""
        self.last_error = ""
        self.recording = False
        self.recording_session_id = ""
        self.current_session: RecordingSession | None = None
        self.csv_file = None
        self.csv_writer = None
        self.script_start_unix = time.time()
        self._last_sensor_connected = False
        self._next_sensor_reconnect_unix = 0.0

        self.port_handler = PortHandler(DEVICENAME_MOTOR)
        self.packet_handler = PacketHandler(2.0)
        if not self.port_handler.openPort() or not self.port_handler.setBaudRate(BAUDRATE_MOTOR):
            raise RuntimeError("Failed to open Dynamixel motor port or set baud rate")

        for dxl_id in DXL_IDS:
            self.packet_handler.reboot(self.port_handler, dxl_id)
        time.sleep(1.0)

        self.group_read = GroupSyncRead(self.port_handler, self.packet_handler, ADDR_PRESENT_POSITION, 4)
        for dxl_id in DXL_IDS:
            self.group_read.addParam(dxl_id)
        for dxl_id in DXL_IDS:
            self.packet_handler.write1ByteTxRx(self.port_handler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

        self.sensor_serial = None
        self._open_sensor_serial()
        self._move_to(self.shared_pos)

        self.bg_thread = threading.Thread(target=self._background_worker, daemon=True)
        self.bg_thread.start()

    def _open_sensor_serial(self) -> None:
        if serial is None:
            self.sensor_last_error = "pyserial not installed"
            self.sensor_serial = None
            self._next_sensor_reconnect_unix = time.time() + 5.0
            return
        try:
            self.sensor_serial = serial.Serial(DEVICENAME_SENSOR, BAUDRATE_SENSOR, timeout=0.05)
            time.sleep(0.01)
            self.sensor_last_error = ""
            self._next_sensor_reconnect_unix = 0.0
        except Exception as exc:
            self.sensor_serial = None
            self.sensor_last_error = str(exc)
            self._next_sensor_reconnect_unix = time.time() + 1.0

    def _close_sensor_serial(self) -> None:
        ser = self.sensor_serial
        self.sensor_serial = None
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    def _move_to(self, positions: list[int]) -> None:
        with self.com_lock:
            for idx, dxl_id in enumerate(DXL_IDS):
                self.packet_handler.write4ByteTxRx(self.port_handler, dxl_id, ADDR_GOAL_POSITION, int(positions[idx]))

    def _move_open_home_legacy_order(self) -> None:
        with self.com_lock:
            for idx, dxl_id in enumerate(DXL_IDS):
                time.sleep(0.1)
                dxl_id_temp = 4 - int(dxl_id)
                idx_temp = 2 - int(idx)
                self.packet_handler.write4ByteTxRx(
                    self.port_handler,
                    dxl_id_temp,
                    ADDR_GOAL_POSITION,
                    int(self.shared_pos[idx_temp]),
                )

    def _read_motor_positions(self) -> list[int]:
        with self.com_lock:
            self.group_read.txRxPacket()
            return [int(self.group_read.getData(dxl_id, ADDR_PRESENT_POSITION, 4)) for dxl_id in DXL_IDS]

    def _read_sensor_once(self, now_unix: float) -> None:
        if self.sensor_serial is None:
            return
        try:
            if self.sensor_serial.in_waiting <= 0:
                return
            line = self.sensor_serial.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                return
            parts = line.split()
            if len(parts) < 3:
                return
            values = [int(parts[0]), int(parts[1]), int(parts[2])]
            with self.state_lock:
                self.latest_tactile = values
                self.latest_tactile_ts_unix = now_unix
                self.sensor_last_update_unix = now_unix
                self.sensor_connected = True
                self.sample_valid = True
                self.sensor_last_error = ""
        except Exception as exc:
            with self.state_lock:
                self.sensor_last_error = str(exc)
            self._close_sensor_serial()

    def _refresh_sensor_state(self, now_unix: float) -> None:
        if self.sensor_serial is None and self.running and now_unix >= self._next_sensor_reconnect_unix:
            self._open_sensor_serial()
        last_update = self.sensor_last_update_unix
        connected = bool(last_update is not None and (now_unix - last_update) <= SENSOR_FRESHNESS_TIMEOUT_S)
        if self._last_sensor_connected and not connected:
            self.sensor_disconnect_count += 1
        self._last_sensor_connected = connected
        self.sensor_connected = connected
        self.sample_valid = connected and self.latest_tactile_ts_unix is not None

    def _write_record_row(self, now_unix: float, elapsed_s: float, positions: list[int]) -> None:
        if not self.recording or self.csv_writer is None or self.current_session is None:
            return
        tactile = list(self.latest_tactile)
        tactile_ts = self.latest_tactile_ts_unix if self.latest_tactile_ts_unix is not None else ""
        sensor_connected = int(bool(self.sensor_connected))
        sample_valid = int(bool(self.sample_valid))
        self.csv_writer.writerow(
            [
                f"{now_unix:.6f}",
                f"{elapsed_s:.6f}",
                positions[0],
                positions[1],
                positions[2],
                tactile[0],
                tactile[1],
                tactile[2],
                tactile_ts,
                sensor_connected,
                sample_valid,
            ]
        )
        if self.csv_file is not None:
            self.csv_file.flush()
        self.current_session.samples += 1

    def _background_worker(self) -> None:
        while self.running:
            now_unix = time.time()
            elapsed_s = now_unix - self.script_start_unix
            self._read_sensor_once(now_unix)
            positions = self._read_motor_positions()
            with self.state_lock:
                self.latest_motor_pos = list(positions)
                self._refresh_sensor_state(now_unix)
                self._write_record_row(now_unix, elapsed_s, positions)
            time.sleep(0.01)

    def _new_session_id(self) -> str:
        return time.strftime("%Y%m%d_%H%M%S", time.localtime()) + f"_{int(time.time() * 1000) % 1000:03d}"

    def _start_recording_locked(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.recording and self.current_session is not None:
            return {
                "ok": True,
                "status": "already_recording",
                "recording_session_id": self.current_session.session_id,
                "raw_csv_path": self.current_session.file_path,
            }

        session_id = str(payload.get("session_id") or self._new_session_id())
        object_name = str(payload.get("object_name") or payload.get("object") or "demo")
        episode_index = payload.get("episode_index")
        notes = str(payload.get("notes") or "")
        output_dir = Path(str(payload.get("output_dir") or RAW_OUTPUT_ROOT))
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = str(payload.get("filename") or "")
        if not filename:
            prefix = str(payload.get("filename_prefix") or "tactile_v2")
            episode_part = f"_ep{int(episode_index):03d}" if episode_index not in {None, ""} else ""
            filename = f"{prefix}_{object_name}{episode_part}_{session_id}.csv"

        file_path = output_dir / filename
        self.csv_file = open(file_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            [
                "timestamp_unix",
                "elapsed_s",
                "pos1",
                "pos2",
                "pos3",
                "val1",
                "val2",
                "val3",
                "tactile_timestamp_unix",
                "sensor_connected",
                "sample_valid",
            ]
        )

        self.script_start_unix = time.time()
        self.recording = True
        self.recording_session_id = session_id
        self.current_session = RecordingSession(
            session_id=session_id,
            file_path=str(file_path),
            object_name=object_name,
            episode_index=int(episode_index) if episode_index not in {None, ""} else None,
            start_unix=self.script_start_unix,
            notes=notes,
        )
        return {
            "ok": True,
            "status": "recording_started",
            "recording_session_id": session_id,
            "raw_csv_path": str(file_path),
        }

    def _stop_recording_locked(self) -> dict[str, Any]:
        session = self.current_session
        if not self.recording or session is None:
            return {"ok": True, "status": "not_recording"}
        self.recording = False
        self.recording_session_id = ""
        if self.csv_file is not None:
            self.csv_file.close()
        self.csv_file = None
        self.csv_writer = None
        self.current_session = None
        return {
            "ok": True,
            "status": "recording_stopped",
            "recording_session_id": session.session_id,
            "raw_csv_path": session.file_path,
            "samples": session.samples,
        }

    def start_recording(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.state_lock:
            return self._start_recording_locked(payload)

    def stop_recording(self) -> dict[str, Any]:
        with self.state_lock:
            return self._stop_recording_locked()

    def _apply_action_locked(self, action: str) -> None:
        if action == "v":
            for i in range(3):
                self.shared_pos[i] += GRIPPER_SPEED
            self._move_to(self.shared_pos)
        elif action == "c":
            for i in range(3):
                self.shared_pos[i] -= GRIPPER_SPEED
            self._move_to(self.shared_pos)
        elif action == "o":
            self.shared_pos = list(ORIGIN_POS)
            self._move_open_home_legacy_order()
        elif action == "1":
            self.shared_pos[0] -= GRIPPER_SPEED
            self._move_to(self.shared_pos)
        elif action == "4":
            self.shared_pos[0] += GRIPPER_SPEED
            self._move_to(self.shared_pos)
        elif action == "2":
            self.shared_pos[1] -= GRIPPER_SPEED
            self._move_to(self.shared_pos)
        elif action == "5":
            self.shared_pos[1] += GRIPPER_SPEED
            self._move_to(self.shared_pos)
        elif action == "3":
            self.shared_pos[2] -= GRIPPER_SPEED
            self._move_to(self.shared_pos)
        elif action == "6":
            self.shared_pos[2] += GRIPPER_SPEED
            self._move_to(self.shared_pos)
        elif action == "7":
            self.shared_pos[1] += GRIPPER_SPEED
            self.shared_pos[0] -= GRIPPER_SPEED
            self._move_to(self.shared_pos)

    def command(self, action: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        with self.state_lock:
            if action == "s":
                return self._start_recording_locked(payload)
            if action == "e":
                return self._stop_recording_locked()
            self._apply_action_locked(action)
            return {
                "ok": True,
                "status": "success",
                "action": action,
                "current_pos": list(self.shared_pos),
                "recording_session_id": self.recording_session_id or None,
            }

    def set_position(self, positions: list[int]) -> dict[str, Any]:
        if len(positions) != 3:
            return {"ok": False, "status": "error", "message": "positions must contain exactly 3 values"}
        target = [int(v) for v in positions]
        with self.state_lock:
            self.shared_pos = list(target)
            self._move_to(self.shared_pos)
            return {"ok": True, "status": "success", "current_pos": list(self.shared_pos)}

    def state_payload(self) -> dict[str, Any]:
        with self.state_lock:
            now_unix = time.time()
            return {
                "status": "ok",
                "api_version": "v2",
                "server_time_unix": now_unix,
                "elapsed_s": now_unix - self.script_start_unix,
                "current_pos": list(self.latest_motor_pos),
                "tactile_data": list(self.latest_tactile),
                "tactile_timestamp_unix": self.latest_tactile_ts_unix,
                "sensor_connected": bool(self.sensor_connected),
                "sample_valid": bool(self.sample_valid),
                "sensor_last_update_unix": self.sensor_last_update_unix,
                "sensor_disconnect_count": int(self.sensor_disconnect_count),
                "recording": bool(self.recording),
                "recording_session_id": self.recording_session_id or None,
                "raw_csv_path": self.current_session.file_path if self.current_session is not None else None,
                "last_error": self.last_error or self.sensor_last_error or None,
            }

    def health_payload(self) -> dict[str, Any]:
        payload = self.state_payload()
        payload.update(
            {
                "ok": True,
                "motor_port": DEVICENAME_MOTOR,
                "sensor_port": DEVICENAME_SENSOR,
                "sensor_freshness_timeout_s": SENSOR_FRESHNESS_TIMEOUT_S,
            }
        )
        return payload

    def shutdown(self) -> None:
        self.running = False
        if self.bg_thread.is_alive():
            self.bg_thread.join(timeout=1.0)
        with self.state_lock:
            self._stop_recording_locked()
        self._close_sensor_serial()
        try:
            self.port_handler.closePort()
        except Exception:
            pass


recorder = GripperRecorderV2()
atexit.register(recorder.shutdown)

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify(recorder.health_payload())


@app.route("/state", methods=["GET"])
def state():
    return jsonify(recorder.state_payload())


@app.route("/command", methods=["POST"])
def command():
    data = request.get_json(silent=True) or {}
    action = str(data.get("action") or "").strip()
    if not action:
        return jsonify({"ok": False, "status": "error", "message": "No action provided"}), 400
    result = recorder.command(action, payload=data)
    return jsonify(result)


@app.route("/set_position", methods=["POST"])
def set_position():
    data = request.get_json(silent=True) or {}
    positions = data.get("positions")
    if not isinstance(positions, list):
        return jsonify({"ok": False, "status": "error", "message": "positions list required"}), 400
    result = recorder.set_position(positions)
    status_code = 200 if result.get("ok") else 400
    return jsonify(result), status_code


@app.route("/recording/start", methods=["POST"])
def recording_start():
    data = request.get_json(silent=True) or {}
    return jsonify(recorder.start_recording(data))


@app.route("/recording/stop", methods=["POST"])
def recording_stop():
    return jsonify(recorder.stop_recording())


@app.route("/stop", methods=["GET"])
def stop_route():
    return jsonify({"ok": True, "status": "stopped", "current_pos": recorder.state_payload().get("current_pos", [])})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=APP_PORT)
