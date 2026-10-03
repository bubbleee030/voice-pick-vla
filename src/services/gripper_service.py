from __future__ import annotations

import socket
import threading
import time
from typing import Any, Callable

from src.runtime_config import module_enabled

try:
    import requests
except ImportError:
    requests = None


class GripperService:
    def __init__(self, demo_cfg: dict[str, Any], socketio=None):
        self.demo_cfg = demo_cfg
        self.socketio = socketio
        self.grip_cfg = demo_cfg.get("gripper", {})
        self.active_endpoint: dict[str, Any] | None = None
        self.api_version = "unknown"
        self.resolve_mode = ""
        self.connected = False
        self.last_error = ""
        self.last_state: dict[str, Any] = {}
        self.last_health: dict[str, Any] = {}
        self.last_recording_result: dict[str, Any] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._session = requests.Session() if requests is not None else None

    def _emit_state(self) -> None:
        if self.socketio is None:
            return
        self.socketio.emit("gripper_state", self.status())

    def _endpoints(self) -> list[dict[str, Any]]:
        endpoints = self.grip_cfg.get("endpoints", [])
        if not isinstance(endpoints, list) or not endpoints:
            host = self.grip_cfg.get("agx_ip", "127.0.0.1")
            port = int(self.grip_cfg.get("agx_port", 5002))
            return [{"host": host, "port": port, "label": f"{host}:{port}"}]
        return endpoints

    def _probe_tcp(self, host: str, port: int, timeout: float = 1.0) -> bool:
        try:
            sock = socket.socket()
            sock.settimeout(timeout)
            sock.connect((host, port))
            sock.close()
            return True
        except Exception:
            return False

    def _base_url(self, endpoint: dict[str, Any]) -> str:
        return f"http://{endpoint['host']}:{endpoint['port']}"

    @staticmethod
    def _normalize_state(data: dict[str, Any]) -> dict[str, Any]:
        payload = dict(data)
        if "tactile_data" not in payload and "tactile" in payload:
            tactile = payload.get("tactile")
            if isinstance(tactile, list):
                payload["tactile_data"] = tactile
        if "server_time_unix" not in payload:
            payload["server_time_unix"] = time.time()
        if "sensor_connected" not in payload:
            payload["sensor_connected"] = payload.get("tactile_timestamp_unix") is not None or "tactile_data" in payload
        if "sample_valid" not in payload:
            payload["sample_valid"] = bool(payload.get("sensor_connected"))
        if "api_version" not in payload:
            payload["api_version"] = "legacy"
        return payload

    def _record_success(self, endpoint: dict[str, Any], api_version: str, resolve_mode: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        self.active_endpoint = endpoint
        self.connected = True
        self.api_version = api_version
        self.resolve_mode = resolve_mode
        self.last_error = ""
        if payload is not None:
            self.last_health = payload
        return endpoint

    def _try_health(self, endpoint: dict[str, Any]) -> dict[str, Any] | None:
        base = self._base_url(endpoint)
        resp = self._session.get(f"{base}/health", timeout=1.0)
        if resp.status_code != 200:
            return None
        payload = resp.json()
        if not isinstance(payload, dict):
            payload = {}
        api_version = str(payload.get("api_version") or "v2")
        return self._record_success(endpoint, api_version=api_version, resolve_mode="health", payload=payload)

    def _try_state_probe(self, endpoint: dict[str, Any]) -> dict[str, Any] | None:
        base = self._base_url(endpoint)
        resp = self._session.get(f"{base}/state", timeout=1.0)
        if resp.status_code != 200:
            return None
        payload = resp.json()
        if not isinstance(payload, dict):
            payload = {}
        normalized = self._normalize_state(payload)
        self.last_state = normalized
        api_version = str(normalized.get("api_version") or "legacy")
        return self._record_success(endpoint, api_version=api_version, resolve_mode="state", payload=normalized)

    def resolve_endpoint(self) -> dict[str, Any] | None:
        if not module_enabled(self.demo_cfg, "gripper", default=True):
            self.connected = False
            self.active_endpoint = None
            self.last_error = "module disabled"
            return None
        if requests is None:
            self.connected = False
            self.last_error = "requests not installed"
            return None
        for endpoint in self._endpoints():
            host = str(endpoint["host"])
            port = int(endpoint["port"])
            if not self._probe_tcp(host, port):
                continue
            try:
                if self._try_health(endpoint) is not None:
                    return endpoint
            except Exception as exc:
                self.last_error = str(exc)
            try:
                if self._try_state_probe(endpoint) is not None:
                    return endpoint
            except Exception as exc:
                self.last_error = str(exc)
        self.active_endpoint = None
        self.connected = False
        self.api_version = "unknown"
        self.resolve_mode = ""
        if not self.last_error:
            self.last_error = "no gripper endpoint reachable"
        return None

    @property
    def active_base_url(self) -> str:
        if self.active_endpoint is None and self.resolve_endpoint() is None:
            return ""
        return self._base_url(self.active_endpoint)

    def get_state(self) -> dict[str, Any] | None:
        if requests is None:
            self.last_error = "requests not installed"
            return None
        base = self.active_base_url
        if not base:
            return None
        try:
            resp = self._session.get(f"{base}/state", timeout=1.0)
            resp.raise_for_status()
            data = resp.json()
            self.last_state = self._normalize_state(data if isinstance(data, dict) else {})
            self.connected = True
            self.api_version = str(self.last_state.get("api_version") or self.api_version or "legacy")
            self.last_error = ""
            return self.last_state
        except Exception as exc:
            self.connected = False
            self.last_error = str(exc)
            return None

    def command(self, action: str) -> bool:
        if requests is None:
            self.last_error = "requests not installed"
            return False
        base = self.active_base_url
        if not base:
            return False
        try:
            resp = self._session.post(f"{base}/command", json={"action": action}, timeout=2.0)
            resp.raise_for_status()
            self.last_error = ""
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self.connected = False
            return False

    def set_position(
        self,
        positions: list[int],
        mode: str | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> bool:
        if requests is None:
            self.last_error = "requests not installed"
            return False
        base = self.active_base_url
        if not base:
            return False
        move_mode = mode or self.grip_cfg.get("position_move_mode", "direct")
        target = [int(v) for v in positions]
        if move_mode == "stepped":
            return self._set_position_stepped(target, stop_requested)
        if stop_requested is not None and stop_requested():
            self._hold_measured_position()
            self.last_error = "position move stopped"
            return False
        return self._set_position_direct(target)

    def _hold_measured_position(self) -> bool:
        state = self.get_state() or {}
        current = state.get("current_pos")
        if (
            not isinstance(current, list)
            or len(current) != 3
            or not any(current)
        ):
            self.last_error = "cannot hold: no measured finger position"
            return False
        return self._set_position_direct([int(value) for value in current])

    def _set_position_stepped(
        self,
        target: list[int],
        stop_requested: Callable[[], bool] | None = None,
    ) -> bool:
        if stop_requested is not None and stop_requested():
            self._hold_measured_position()
            self.last_error = "position move stopped"
            return False
        current_state = self.get_state() or {}
        current = current_state.get("current_pos") if isinstance(current_state, dict) else None
        if not isinstance(current, list) or len(current) != 3:
            if stop_requested is not None and stop_requested():
                self._hold_measured_position()
                self.last_error = "position move stopped"
                return False
            return self._set_position_direct(target)

        current_vals = [int(v) for v in current]
        open_step = int(self.grip_cfg.get("open_step_ticks", 24))
        close_step = int(self.grip_cfg.get("close_step_ticks", 10))
        while current_vals != target:
            if stop_requested is not None and stop_requested():
                self._hold_measured_position()
                self.last_error = "position move stopped"
                return False
            next_vals: list[int] = []
            for idx, value in enumerate(current_vals):
                delta = target[idx] - value
                if delta == 0:
                    next_vals.append(value)
                    continue
                step = open_step if delta > 0 else close_step
                magnitude = min(abs(delta), step)
                next_vals.append(value + magnitude if delta > 0 else value - magnitude)
            if not self._set_position_direct(next_vals):
                return False
            current_vals = next_vals
            time.sleep(max(float(self.grip_cfg.get("open_step_delay_s", 0.015)), 0.005))
        return True

    def _set_position_direct(self, target: list[int]) -> bool:
        base = self.active_base_url
        if not base:
            return False
        try:
            resp = self._session.post(f"{base}/set_position", json={"positions": target}, timeout=2.0)
            resp.raise_for_status()
            self.last_error = ""
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self.connected = False
            return False

    def open(self) -> bool:
        return self.command(str(self.grip_cfg.get("open_command", "o")))

    def close(self) -> bool:
        return self.command(str(self.grip_cfg.get("close_command", "c")))

    # ---------------------------------------------------------------- LSTM
    # Finger-action API served by agx/lstm_gripper_service.py (ADR 0001).
    # All calls are tolerant: an unreachable AGX returns None/False and the
    # caller degrades to plain human-confirm gates.

    def _lstm_post(self, path: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any] | None:
        if requests is None:
            return None
        base = self.active_base_url
        if not base:
            return None
        try:
            resp = self._session.post(f"{base}{path}", json=payload or {}, timeout=timeout)
            data = resp.json() if resp.content else {}
            data.setdefault("ok", resp.ok)
            return data
        except Exception as exc:
            self.last_error = str(exc)
            return None

    def lstm_preload(self, object_key: str) -> dict[str, Any] | None:
        # Model load + finger homing runs ~5-10 s on the AGX — generous timeout;
        # call from a background thread (the arm's approach hides it).
        return self._lstm_post("/lstm/preload", {"object": object_key}, timeout=60.0)

    def lstm_start(self, action: str) -> dict[str, Any] | None:
        return self._lstm_post("/lstm/start", {"action": action}, timeout=5.0)

    def lstm_prepare(
        self,
        action: str,
        min_close_ticks: int = 60,
        min_close_fingers: int = 2,
    ) -> dict[str, Any] | None:
        return self._lstm_post(
            "/lstm/prepare",
            {
                "action": action,
                "min_close_ticks": int(min_close_ticks),
                "min_close_fingers": int(min_close_fingers),
            },
            timeout=5.0,
        )

    def lstm_activate(self) -> dict[str, Any] | None:
        return self._lstm_post("/lstm/activate", None, timeout=5.0)

    def lstm_stop(self) -> dict[str, Any] | None:
        return self._lstm_post("/lstm/stop", None, timeout=5.0)

    def reboot_motors(self, home: bool = True) -> dict[str, Any] | None:
        # Clear a Dynamixel fault latch (overload torque-off) without a full AGX
        # service restart. Reboot + optional home takes ~2-5 s -> generous timeout.
        return self._lstm_post("/motor/reboot", {"home": home}, timeout=15.0)

    def lstm_status(self) -> dict[str, Any] | None:
        if requests is None or not self.active_base_url:
            return None
        try:
            resp = self._session.get(f"{self.active_base_url}/lstm/status", timeout=2.0)
            return resp.json()
        except Exception as exc:
            self.last_error = str(exc)
            return None

    def start_remote_recording(self, payload: dict[str, Any]) -> dict[str, Any]:
        cfg = self.grip_cfg.get("remote_recording", {})
        if not bool(cfg.get("enabled", True)):
            return {"ok": False, "status": "disabled", "message": "remote gripper recording disabled"}
        if requests is None:
            return {"ok": False, "status": "error", "message": "requests not installed"}
        base = self.active_base_url
        if not base:
            return {"ok": False, "status": "error", "message": self.last_error or "no gripper endpoint"}
        timeout_s = float(cfg.get("start_timeout_s", 2.0))
        try:
            resp = self._session.post(f"{base}/recording/start", json=payload, timeout=timeout_s)
            if resp.status_code == 404:
                ok = self.command("s")
                result = {
                    "ok": ok,
                    "status": "recording_started_legacy" if ok else "recording_start_failed_legacy",
                    "api_version": self.api_version or "legacy",
                    "fallback_used": True,
                }
                self.last_recording_result = result
                return result
            resp.raise_for_status()
            result = resp.json()
            if not isinstance(result, dict):
                result = {"ok": True, "status": "recording_started"}
            result.setdefault("api_version", self.api_version or "legacy")
            result.setdefault("fallback_used", False)
            self.last_recording_result = result
            return result
        except Exception as exc:
            self.last_error = str(exc)
            result = {"ok": False, "status": "recording_start_failed", "message": str(exc), "api_version": self.api_version or "unknown"}
            self.last_recording_result = result
            return result

    def stop_remote_recording(self) -> dict[str, Any]:
        cfg = self.grip_cfg.get("remote_recording", {})
        if not bool(cfg.get("enabled", True)):
            return {"ok": False, "status": "disabled", "message": "remote gripper recording disabled"}
        if requests is None:
            return {"ok": False, "status": "error", "message": "requests not installed"}
        base = self.active_base_url
        if not base:
            return {"ok": False, "status": "error", "message": self.last_error or "no gripper endpoint"}
        timeout_s = float(cfg.get("stop_timeout_s", 4.0))
        try:
            resp = self._session.post(f"{base}/recording/stop", timeout=timeout_s)
            if resp.status_code == 404:
                ok = self.command("e")
                result = {
                    "ok": ok,
                    "status": "recording_stopped_legacy" if ok else "recording_stop_failed_legacy",
                    "api_version": self.api_version or "legacy",
                    "fallback_used": True,
                }
                self.last_recording_result = result
                return result
            resp.raise_for_status()
            result = resp.json()
            if not isinstance(result, dict):
                result = {"ok": True, "status": "recording_stopped"}
            result.setdefault("api_version", self.api_version or "legacy")
            result.setdefault("fallback_used", False)
            self.last_recording_result = result
            return result
        except Exception as exc:
            self.last_error = str(exc)
            result = {"ok": False, "status": "recording_stop_failed", "message": str(exc), "api_version": self.api_version or "unknown"}
            self.last_recording_result = result
            return result

    def start_monitoring(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop_monitoring(self):
        self._running = False
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass

    def _poll_loop(self):
        while self._running:
            self.get_state()
            self._emit_state()
            time.sleep(0.15)

    def status(self) -> dict[str, Any]:
        endpoint = self.active_endpoint or {}
        return {
            "connected": self.connected,
            "endpoint": endpoint,
            "api_version": self.api_version,
            "resolve_mode": self.resolve_mode or None,
            "base_url": self._base_url(endpoint) if endpoint else "",
            "last_error": self.last_error or None,
            "last_recording_result": self.last_recording_result or None,
            "state": self.last_state,
        }
