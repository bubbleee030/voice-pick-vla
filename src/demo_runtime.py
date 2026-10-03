"""
Demo realtime runtime for dual-camera preview/recording with feature toggles.

This module provides a reusable session runtime used by the Ubuntu control
backend and dataset capture service.

Design goals:
- Keep RealSense startup resilient (profile fallback, retries, hardware reset).
- Support per-feature on/off toggles for demo workflows.
- Keep output structure compatible with existing recording data when enabled.
- Reserve YOLO interface fields without running inference in v1.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise RuntimeError("pyrealsense2 is required for demo runtime") from exc

from src.controller import ArmController
from src.utils import PROJECT_ROOT

try:
    import requests
except ImportError:
    requests = None


@dataclass
class FeatureToggles:
    cam1_enabled: bool = True
    cam2_enabled: bool = True
    rgb_enabled: bool = True
    depth_enabled: bool = True
    align_enabled: bool = True
    arm_log_enabled: bool = True
    gripper_log_enabled: bool = False
    yolo_enabled: bool = False
    preview_window_enabled: bool = False
    disk_output_enabled: bool = True


@dataclass
class SessionConfig:
    object_name: str
    cam1_serial: str
    cam2_serial: str | None = None
    episode_index: int | None = None
    duration_s: float = 0.0
    fps: float = 10.0
    pose_hz: float = 10.0
    arm_monitor_hz: float = 4.0
    stream_timeout_ms: int = 10000
    warmup_frames: int = 10
    max_frame_errors: int = 30
    safe_profile: bool = True
    profile_retries_high: int = 6
    profile_retries_mid: int = 4
    start_retry_sleep: float = 0.2
    camera_reset_retries: int = 2
    camera_reset_wait: float = 4.0
    power_retries: int = 4
    power_retry_delay: float = 1.5
    allow_no_sudo: bool = False
    output_root: str = "data/recordings"
    notes: str = ""
    arm_host: str = "127.0.0.1"
    arm_port: int = 1502
    gripper_api_url: str = ""
    gripper_timeout_s: float = 0.5
    gripper_poll_hz: float = 20.0
    gripper_sync_samples: int = 6
    preview_scale: float = 1.6
    monitor_scale: float = 1.2
    preview_jpeg_quality: int = 80
    preview_target_hz: float = 12.0
    features: FeatureToggles = field(default_factory=FeatureToggles)


@dataclass
class CameraDeviceInfo:
    name: str
    serial: str
    firmware: str
    usb_type: str


@dataclass
class CameraRuntime:
    name: str
    serial: str
    device_info: CameraDeviceInfo
    pipeline: Any
    align: Any | None
    profile: dict[str, Any]


@dataclass
class ArmRuntime:
    controller: Any | None
    status: str
    pose_mm_deg: list[float] | None
    pose_ts: float
    reconnect_count: int = 0
    last_error: str | None = None


@dataclass
class GripperClient:
    base_url: str
    session: Any
    timeout_s: float
    offset_sec: float
    best_rtt_ms: float
    connected: bool
    api_version: str = "legacy"
    time_sync_supported: bool = False
    last_state: dict[str, Any] | None = None
    last_error: str | None = None
    poll_ok: int = 0
    poll_fail: int = 0


@dataclass
class RuntimeHooks:
    on_preview_jpeg: Callable[[bytes], None] | None = None
    on_telemetry: Callable[[dict[str, Any]], None] | None = None
    on_event: Callable[[str, dict[str, Any]], None] | None = None


@dataclass
class CameraFrameBundle:
    color: np.ndarray | None
    depth: np.ndarray | None


def is_running_as_root() -> bool:
    if hasattr(os, "geteuid"):
        return os.geteuid() == 0
    return False


def sudo_run_hint() -> str:
    return (
        "bash scripts/run_demo.sh recording "
        "# or use the UI Recording Console on http://<ubuntu-ip>:8090"
    )


def fmt_name(color_fmt) -> str:
    return str(color_fmt).split(".")[-1] if color_fmt is not None else "none"


def sanitize_url(url: str) -> str:
    return url.strip().rstrip("/")


def normalize_gripper_state(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    if "tactile_data" not in out and "tactile" in out:
        tactile = out.get("tactile")
        if isinstance(tactile, list):
            out["tactile_data"] = tactile
    if "api_version" not in out:
        out["api_version"] = "legacy"
    if "sensor_connected" not in out:
        out["sensor_connected"] = out.get("tactile_timestamp_unix") is not None or isinstance(out.get("tactile_data"), list)
    if "sample_valid" not in out:
        out["sample_valid"] = bool(out.get("sensor_connected"))
    return out


def convert_color(raw_color: np.ndarray, color_fmt) -> np.ndarray:
    if color_fmt == rs.format.rgb8:
        return cv2.cvtColor(raw_color, cv2.COLOR_RGB2BGR)

    if color_fmt == rs.format.yuyv:
        if raw_color.ndim == 2:
            if raw_color.dtype == np.uint16:
                raw_color = raw_color.view(np.uint8).reshape(raw_color.shape[0], raw_color.shape[1], 2)
            else:
                if raw_color.shape[1] % 2 != 0:
                    raise RuntimeError(f"Unexpected YUYV image shape: {raw_color.shape}")
                raw_color = raw_color.reshape(raw_color.shape[0], raw_color.shape[1] // 2, 2)
        elif raw_color.ndim == 3 and raw_color.shape[2] == 1:
            if raw_color.shape[1] % 2 != 0:
                raise RuntimeError(f"Unexpected YUYV image shape: {raw_color.shape}")
            raw_color = raw_color.reshape(raw_color.shape[0], raw_color.shape[1] // 2, 2)
        elif raw_color.ndim != 3 or raw_color.shape[2] != 2:
            raise RuntimeError(f"Unsupported YUYV image shape: {raw_color.shape}")
        return cv2.cvtColor(raw_color, cv2.COLOR_YUV2BGR_YUY2)

    return raw_color


def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_TURBO)


def center_depth_mm(depth: np.ndarray | None) -> float | None:
    if depth is None:
        return None
    h, w = depth.shape[:2]
    cx, cy = w // 2, h // 2
    roi = depth[max(0, cy - 2):min(h, cy + 3), max(0, cx - 2):min(w, cx + 3)]
    valid = roi[roi > 0]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))


def serialize_profile(profile: dict[str, Any] | None) -> dict[str, Any] | None:
    if profile is None:
        return None
    out: dict[str, Any] = {}
    for key, value in profile.items():
        if key == "color_fmt":
            out[key] = fmt_name(value)
        else:
            out[key] = value
    return out


def list_devices_with_retry(retries: int, delay_s: float, ctx: Any = None) -> list[CameraDeviceInfo]:
    retries = max(1, retries)
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            current_ctx = ctx if ctx is not None else rs.context()
            found: list[CameraDeviceInfo] = []
            for dev in current_ctx.query_devices():
                name = dev.get_info(rs.camera_info.name)
                serial = dev.get_info(rs.camera_info.serial_number)
                firmware = dev.get_info(rs.camera_info.firmware_version)
                usb_type = (
                    dev.get_info(rs.camera_info.usb_type_descriptor)
                    if dev.supports(rs.camera_info.usb_type_descriptor)
                    else "unknown"
                )
                found.append(CameraDeviceInfo(name=name, serial=serial, firmware=firmware, usb_type=usb_type))

            if found:
                return found

            last_error = RuntimeError("No RealSense camera detected")
            print(f"[realsense] attempt {attempt}/{retries}: no camera found")
        except Exception as exc:
            last_error = exc
            msg = str(exc)
            if "failed to set power state" in msg.lower():
                print(f"[realsense] attempt {attempt}/{retries}: failed to set power state")
            else:
                print(f"[realsense] attempt {attempt}/{retries}: device query failed: {msg}")

        if attempt < retries:
            time.sleep(max(0.1, delay_s))

    if last_error and "failed to set power state" in str(last_error).lower():
        hint_lines = [
            "RealSense failed to set power state.",
            "On macOS this commonly requires sudo with the same Python interpreter.",
            f"Recommended command: {sudo_run_hint()}",
        ]
        if not is_running_as_root():
            hint_lines.append("Current process is not running as root.")
        raise RuntimeError("\n".join(hint_lines))

    raise RuntimeError(f"Could not list RealSense devices: {last_error}")


def find_device(devices: list[CameraDeviceInfo], serial: str) -> CameraDeviceInfo:
    for dev in devices:
        if dev.serial == serial:
            return dev
    available = ", ".join(d.serial for d in devices) if devices else "none"
    raise RuntimeError(f"Serial {serial} not found. Available serials: {available}")


def hardware_reset_device(serial: str, wait_s: float, ctx: Any = None) -> bool:
    try:
        current_ctx = ctx if ctx is not None else rs.context()
        for dev in current_ctx.query_devices():
            if dev.get_info(rs.camera_info.serial_number) != serial:
                continue
            if not hasattr(dev, "hardware_reset"):
                print(f"[realsense] hardware reset not supported for {serial}")
                return False
            print(f"[realsense] hardware reset {serial} ...")
            dev.hardware_reset()
            time.sleep(max(1.0, wait_s))
            return True
        print(f"[realsense] device {serial} not found for hardware reset")
        return False
    except Exception as exc:
        print(f"[realsense] hardware reset failed for {serial}: {exc}")
        return False


def candidate_profiles(
    usb_type: str,
    safe_profile: bool,
    retries_high: int,
    retries_mid: int,
    rgb_enabled: bool,
    depth_enabled: bool,
) -> list[dict[str, Any]]:
    usb_str = (usb_type or "unknown").lower()
    is_usb2 = usb_str.startswith("2")
    high = max(1, retries_high)
    mid = max(1, retries_mid)

    if not rgb_enabled and not depth_enabled:
        raise RuntimeError("Both RGB and depth are disabled")

    if depth_enabled and rgb_enabled:
        if is_usb2:
            return [
                {"w": 424, "h": 240, "fps": 6, "color_fmt": rs.format.rgb8, "retries": high},
                {"w": 424, "h": 240, "fps": 6, "color_fmt": rs.format.yuyv, "retries": mid},
            ]

        if safe_profile:
            return [
                {"w": 424, "h": 240, "fps": 15, "color_fmt": rs.format.rgb8, "retries": high},
                {"w": 424, "h": 240, "fps": 15, "color_fmt": rs.format.yuyv, "retries": mid},
                {"w": 640, "h": 480, "fps": 15, "color_fmt": rs.format.rgb8, "retries": mid},
            ]

        return [
            {"w": 640, "h": 480, "fps": 30, "color_fmt": rs.format.rgb8, "retries": high},
            {"w": 848, "h": 480, "fps": 30, "color_fmt": rs.format.rgb8, "retries": mid},
            {"w": 640, "h": 480, "fps": 15, "color_fmt": rs.format.rgb8, "retries": mid},
            {"w": 424, "h": 240, "fps": 15, "color_fmt": rs.format.rgb8, "retries": mid},
            {"w": 424, "h": 240, "fps": 15, "color_fmt": rs.format.yuyv, "retries": mid},
        ]

    if rgb_enabled:
        if is_usb2:
            return [
                {"w": 424, "h": 240, "fps": 6, "color_fmt": rs.format.rgb8, "retries": high},
                {"w": 424, "h": 240, "fps": 6, "color_fmt": rs.format.yuyv, "retries": mid},
            ]

        if safe_profile:
            return [
                {"w": 424, "h": 240, "fps": 15, "color_fmt": rs.format.rgb8, "retries": high},
                {"w": 424, "h": 240, "fps": 15, "color_fmt": rs.format.yuyv, "retries": mid},
                {"w": 640, "h": 480, "fps": 15, "color_fmt": rs.format.rgb8, "retries": mid},
            ]

        return [
            {"w": 640, "h": 480, "fps": 30, "color_fmt": rs.format.rgb8, "retries": high},
            {"w": 848, "h": 480, "fps": 30, "color_fmt": rs.format.rgb8, "retries": mid},
            {"w": 640, "h": 480, "fps": 15, "color_fmt": rs.format.rgb8, "retries": mid},
            {"w": 424, "h": 240, "fps": 15, "color_fmt": rs.format.rgb8, "retries": mid},
            {"w": 424, "h": 240, "fps": 15, "color_fmt": rs.format.yuyv, "retries": mid},
        ]

    # depth_only
    if is_usb2:
        return [
            {"w": 424, "h": 240, "fps": 6, "color_fmt": None, "retries": high},
            {"w": 256, "h": 144, "fps": 90, "color_fmt": None, "retries": mid},
        ]

    if safe_profile:
        return [
            {"w": 424, "h": 240, "fps": 15, "color_fmt": None, "retries": high},
            {"w": 640, "h": 480, "fps": 15, "color_fmt": None, "retries": mid},
        ]

    return [
        {"w": 640, "h": 480, "fps": 30, "color_fmt": None, "retries": high},
        {"w": 848, "h": 480, "fps": 30, "color_fmt": None, "retries": mid},
        {"w": 640, "h": 480, "fps": 15, "color_fmt": None, "retries": mid},
        {"w": 424, "h": 240, "fps": 15, "color_fmt": None, "retries": mid},
    ]


def stop_camera(cam: CameraRuntime | None) -> None:
    if cam is None:
        return
    try:
        cam.pipeline.stop()
    except Exception:
        pass


def episode_has_data(episode_dir: Path) -> bool:
    if not episode_dir.exists() or not episode_dir.is_dir():
        return False
    if any((episode_dir / "cam1_rgb").glob("*.jpg")):
        return True
    if any((episode_dir / "cam1_depth").glob("*.png")):
        return True
    if any((episode_dir / "cam2_rgb").glob("*.jpg")):
        return True
    if any((episode_dir / "cam2_depth").glob("*.png")):
        return True
    if (episode_dir / "metadata.json").exists():
        return True
    traj = episode_dir / "trajectory.csv"
    if traj.exists() and traj.stat().st_size > 64:
        return True
    return False


def next_episode_index(object_dir: Path) -> int:
    max_idx = 0
    for p in object_dir.glob("episode_*"):
        tail = p.name.split("_")[-1]
        if tail.isdigit():
            if not episode_has_data(p):
                continue
            max_idx = max(max_idx, int(tail))
    return max_idx + 1


def cleanup_dir_if_exists(path: Path) -> None:
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
    except Exception:
        pass


def cleanup_empty_parents(path: Path, stop_at: Path) -> None:
    cur = path
    while cur != stop_at and cur.exists():
        try:
            cur.rmdir()
        except OSError:
            break
        cur = cur.parent


def estimate_gripper_offset(session: Any, base_url: str, timeout_s: float, samples: int) -> tuple[float, float, bool, str, bool]:
    offsets = []
    api_version = "legacy"
    state_reachable = False
    for _ in range(max(1, samples)):
        t0 = time.time()
        try:
            resp = session.get(f"{base_url}/state", timeout=timeout_s)
            t1 = time.time()
            if resp.status_code != 200:
                continue
            payload = normalize_gripper_state(resp.json())
            state_reachable = True
            api_version = str(payload.get("api_version") or api_version)
            remote_ts = payload.get("server_time_unix")
            if remote_ts is None:
                continue
            local_mid = (t0 + t1) * 0.5
            rtt_ms = (t1 - t0) * 1000.0
            offsets.append((float(remote_ts) - local_mid, rtt_ms))
        except Exception:
            continue

    if not offsets:
        if state_reachable:
            return 0.0, -1.0, True, api_version, False
        return 0.0, -1.0, False, api_version, False

    best = min(offsets, key=lambda x: x[1])
    return best[0], best[1], True, api_version, True


def connect_gripper(base_url: str, timeout_s: float, sync_samples: int) -> GripperClient | None:
    if not base_url:
        return None
    if requests is None:
        print("[gripper] requests not installed. Run: pip install requests")
        return None

    url = sanitize_url(base_url)
    session = requests.Session()
    offset_sec, best_rtt_ms, ok, api_version, time_sync_supported = estimate_gripper_offset(session, url, timeout_s, sync_samples)
    if not ok:
        print(f"[gripper] connect failed: {url}/state not reachable")
        return GripperClient(
            base_url=url,
            session=session,
            timeout_s=timeout_s,
            offset_sec=0.0,
            best_rtt_ms=-1.0,
            connected=False,
            api_version=api_version,
            time_sync_supported=False,
            last_error="state endpoint unavailable",
        )

    if time_sync_supported:
        print(f"[gripper] connected {url} | api={api_version} | clock_offset={offset_sec:+.4f}s best_rtt={best_rtt_ms:.1f}ms")
    else:
        print(f"[gripper] connected {url} | api={api_version} | no remote clock sync support")
    return GripperClient(
        base_url=url,
        session=session,
        timeout_s=timeout_s,
        offset_sec=offset_sec,
        best_rtt_ms=best_rtt_ms,
        connected=True,
        api_version=api_version,
        time_sync_supported=time_sync_supported,
    )


def poll_gripper_state(client: GripperClient) -> dict[str, Any] | None:
    try:
        t0 = time.time()
        resp = client.session.get(f"{client.base_url}/state", timeout=client.timeout_s)
        t1 = time.time()
        if resp.status_code != 200:
            client.poll_fail += 1
            client.last_error = f"HTTP {resp.status_code}"
            return None
        payload = normalize_gripper_state(resp.json())
        payload["rtt_ms"] = (t1 - t0) * 1000.0
        client.last_state = payload
        client.last_error = None
        client.poll_ok += 1
        return payload
    except Exception as exc:
        client.poll_fail += 1
        client.last_error = str(exc)
        return None


class DemoRealtimeSession:
    """Single recording session runtime with callbacks for API integration."""

    def __init__(self, config: SessionConfig, hooks: RuntimeHooks | None = None):
        self.config = config
        self.hooks = hooks or RuntimeHooks()
        self.stop_event = threading.Event()
        self._running = False
        self._lock = threading.Lock()
        self._telemetry: dict[str, Any] = {}
        self._summary: dict[str, Any] | None = None
        self._preview_canvas: np.ndarray | None = None
        self._last_preview_jpeg_ts = 0.0

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def telemetry(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._telemetry)

    @property
    def summary(self) -> dict[str, Any] | None:
        with self._lock:
            if self._summary is None:
                return None
            return dict(self._summary)

    def request_stop(self) -> None:
        self.stop_event.set()

    def _emit_event(self, event_name: str, payload: dict[str, Any]) -> None:
        if self.hooks.on_event is not None:
            try:
                self.hooks.on_event(event_name, payload)
            except Exception:
                pass

    def _emit_telemetry(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._telemetry = payload
        if self.hooks.on_telemetry is not None:
            try:
                self.hooks.on_telemetry(payload)
            except Exception:
                pass

    def _emit_preview(self, canvas: np.ndarray) -> None:
        self._preview_canvas = canvas
        if self.hooks.on_preview_jpeg is None:
            return

        hz = max(0.5, float(self.config.preview_target_hz))
        now = time.time()
        if now - self._last_preview_jpeg_ts < 1.0 / hz:
            return

        quality = int(max(30, min(95, self.config.preview_jpeg_quality)))
        try:
            canvas_to_encode = np.ascontiguousarray(canvas)
            ok, encoded = cv2.imencode(".jpg", canvas_to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        except Exception:
            return
        if not ok:
            return
        self._last_preview_jpeg_ts = now
        try:
            self.hooks.on_preview_jpeg(encoded.tobytes())
        except Exception:
            pass

    def _validate_config(self) -> None:
        cfg = self.config
        f = cfg.features

        if cfg.fps <= 0:
            raise RuntimeError("fps must be > 0")
        if cfg.pose_hz <= 0:
            raise RuntimeError("pose_hz must be > 0")
        if cfg.arm_monitor_hz <= 0:
            raise RuntimeError("arm_monitor_hz must be > 0")
        if cfg.stream_timeout_ms < 200:
            raise RuntimeError("stream_timeout_ms must be >= 200")
        if cfg.warmup_frames < 1:
            raise RuntimeError("warmup_frames must be >= 1")
        if cfg.max_frame_errors < 1:
            raise RuntimeError("max_frame_errors must be >= 1")
        if cfg.profile_retries_high < 1:
            raise RuntimeError("profile_retries_high must be >= 1")
        if cfg.profile_retries_mid < 1:
            raise RuntimeError("profile_retries_mid must be >= 1")
        if cfg.camera_reset_retries < 0:
            raise RuntimeError("camera_reset_retries must be >= 0")
        if cfg.power_retries < 1:
            raise RuntimeError("power_retries must be >= 1")
        if cfg.preview_jpeg_quality < 30 or cfg.preview_jpeg_quality > 95:
            raise RuntimeError("preview_jpeg_quality must be in [30, 95]")

        if not f.cam1_enabled and not f.cam2_enabled:
            raise RuntimeError("At least one camera must be enabled")
        if not f.rgb_enabled and not f.depth_enabled:
            raise RuntimeError("At least one of RGB/depth must be enabled")
        if f.cam2_enabled and not cfg.cam2_serial:
            raise RuntimeError("cam2 is enabled but cam2_serial is missing")
        if f.cam2_enabled and cfg.cam2_serial and cfg.cam2_serial == cfg.cam1_serial:
            raise RuntimeError("cam2_serial must be different from cam1_serial")
        if f.gripper_log_enabled and not cfg.gripper_api_url:
            raise RuntimeError("gripper_log_enabled requires gripper_api_url")

        if sys.platform == "darwin" and (not cfg.allow_no_sudo) and (not is_running_as_root()):
            raise RuntimeError(
                "macOS RealSense run requires sudo by default. "
                f"Hint: {sudo_run_hint()}"
            )

    def _start_camera(self, name: str, serial: str, devices: list[CameraDeviceInfo]) -> CameraRuntime:
        cfg = self.config
        f = cfg.features
        dev = find_device(devices, serial)
        profiles = candidate_profiles(
            usb_type=dev.usb_type,
            safe_profile=cfg.safe_profile,
            retries_high=cfg.profile_retries_high,
            retries_mid=cfg.profile_retries_mid,
            rgb_enabled=f.rgb_enabled,
            depth_enabled=f.depth_enabled,
        )

        last_error: Exception | None = None
        attempted: list[str] = []
        total_cycles = max(0, cfg.camera_reset_retries) + 1

        for cycle in range(total_cycles):
            for profile in profiles:
                retries = max(1, int(profile.get("retries", 1)))
                for attempt in range(1, retries + 1):
                    pipeline = rs.pipeline()
                    rs_cfg = rs.config()
                    rs_cfg.enable_device(serial)

                    if f.depth_enabled:
                        rs_cfg.enable_stream(rs.stream.depth, profile["w"], profile["h"], rs.format.z16, profile["fps"])
                    if f.rgb_enabled:
                        color_fmt = profile["color_fmt"] if profile["color_fmt"] is not None else rs.format.rgb8
                        rs_cfg.enable_stream(rs.stream.color, profile["w"], profile["h"], color_fmt, profile["fps"])

                    profile_label = f"{profile['w']}x{profile['h']}@{profile['fps']} {fmt_name(profile['color_fmt'])}"
                    print(
                        f"[{name}] trying {profile_label} | usb={dev.usb_type} "
                        f"attempt {attempt}/{retries} cycle {cycle + 1}/{total_cycles}"
                    )

                    pipeline_profile = None
                    try:
                        pipeline_profile = pipeline.start(rs_cfg)
                        for _ in range(max(1, cfg.warmup_frames)):
                            pipeline.wait_for_frames(timeout_ms=cfg.stream_timeout_ms)

                        test_frames = pipeline.wait_for_frames(timeout_ms=cfg.stream_timeout_ms)
                        if f.depth_enabled and not test_frames.get_depth_frame():
                            raise RuntimeError("missing depth frame after warmup")
                        if f.rgb_enabled and not test_frames.get_color_frame():
                            raise RuntimeError("missing color frame after warmup")

                        align = None
                        if f.align_enabled and f.depth_enabled and f.rgb_enabled:
                            align = rs.align(rs.stream.color)

                        active_profile = dict(profile)
                        active_profile["attempt"] = attempt
                        active_profile["reset_cycle"] = cycle

                        print(f"[{name}] active {profile_label}")
                        return CameraRuntime(
                            name=name,
                            serial=serial,
                            device_info=dev,
                            pipeline=pipeline,
                            align=align,
                            profile=active_profile,
                        )
                    except Exception as exc:
                        last_error = exc
                        attempted.append(
                            f"{profile_label} attempt {attempt}/{retries} cycle {cycle + 1}/{total_cycles}: {exc}"
                        )
                        print(f"[{name}] failed: {exc}")
                        if pipeline_profile is not None:
                            try:
                                pipeline.stop()
                            except Exception:
                                pass
                        if attempt < retries:
                            time.sleep(max(0.05, cfg.start_retry_sleep))

            if cycle < total_cycles - 1:
                print(f"[{name}] all profiles failed in cycle {cycle + 1}, trying hardware reset...")
                reset_ok = hardware_reset_device(serial, wait_s=cfg.camera_reset_wait)
                if reset_ok:
                    try:
                        devices = list_devices_with_retry(retries=3, delay_s=1.0)
                        dev = find_device(devices, serial)
                    except Exception as exc:
                        print(f"[{name}] device re-enumeration after reset failed: {exc}")

        details = "\n  - ".join(attempted[-20:]) if attempted else str(last_error)
        raise RuntimeError(
            f"Could not start {name} ({serial}): {last_error}\n"
            f"Attempt summary (last {min(len(attempted), 20)}):\n  - {details}"
        )

    def _capture_frame(self, cam: CameraRuntime) -> CameraFrameBundle:
        f = self.config.features
        frames = cam.pipeline.wait_for_frames(timeout_ms=self.config.stream_timeout_ms)
        if cam.align is not None:
            frames = cam.align.process(frames)

        color: np.ndarray | None = None
        depth: np.ndarray | None = None

        if f.rgb_enabled:
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise RuntimeError(f"{cam.name} missing color frame")
            raw_color = np.asanyarray(color_frame.get_data())
            color = convert_color(raw_color, cam.profile["color_fmt"])

        if f.depth_enabled:
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                raise RuntimeError(f"{cam.name} missing depth frame")
            depth = np.asanyarray(depth_frame.get_data())

        return CameraFrameBundle(color=color, depth=depth)

    def _make_output_dirs(self, episode_dir: Path, with_cam2: bool) -> dict[str, Path]:
        f = self.config.features

        dirs: dict[str, Path] = {
            "episode": episode_dir,
            "metadata": episode_dir / "metadata.json",
        }

        if f.rgb_enabled:
            dirs["cam1_rgb"] = episode_dir / "cam1_rgb"
            if with_cam2:
                dirs["cam2_rgb"] = episode_dir / "cam2_rgb"

        if f.depth_enabled:
            dirs["cam1_depth"] = episode_dir / "cam1_depth"
            if with_cam2:
                dirs["cam2_depth"] = episode_dir / "cam2_depth"

        if f.arm_log_enabled:
            dirs["trajectory"] = episode_dir / "trajectory.csv"

        if f.gripper_log_enabled:
            dirs["gripper_stream"] = episode_dir / "gripper_stream.csv"

        for key, path in dirs.items():
            if key in {"metadata", "trajectory", "gripper_stream"}:
                continue
            path.mkdir(parents=True, exist_ok=True)

        return dirs

    def _init_arm_runtime(self) -> ArmRuntime:
        if not self.config.features.arm_log_enabled:
            return ArmRuntime(controller=None, status="disabled", pose_mm_deg=None, pose_ts=0.0)
        return ArmRuntime(controller=None, status="not connected", pose_mm_deg=None, pose_ts=0.0)

    @staticmethod
    def _close_arm(arm_rt: ArmRuntime) -> None:
        if arm_rt.controller is None:
            return
        try:
            arm_rt.controller.disconnect()
        except Exception:
            pass
        arm_rt.controller = None

    def _ensure_arm_connected(self, arm_rt: ArmRuntime, verbose: bool = True) -> bool:
        cfg = self.config
        if not cfg.features.arm_log_enabled:
            arm_rt.status = "disabled"
            return False

        if arm_rt.controller is not None:
            arm_rt.status = "connected"
            return True

        try:
            ctrl = ArmController(host=cfg.arm_host, port=cfg.arm_port)
            if not ctrl.connect():
                arm_rt.controller = None
                arm_rt.status = "not connected"
                if verbose:
                    print(f"[arm] connect failed {cfg.arm_host}:{cfg.arm_port}")
                return False
            arm_rt.controller = ctrl
            arm_rt.status = "connected"
            arm_rt.last_error = None
            if verbose:
                print(f"[arm] connected {cfg.arm_host}:{cfg.arm_port}")
            return True
        except Exception as exc:
            arm_rt.controller = None
            arm_rt.status = "not connected"
            arm_rt.last_error = str(exc)
            if verbose:
                print(f"[arm] unavailable: {exc}")
            return False

    def _poll_arm_pose_live(self, arm_rt: ArmRuntime) -> None:
        if not self.config.features.arm_log_enabled:
            arm_rt.status = "disabled"
            return

        if not self._ensure_arm_connected(arm_rt, verbose=False):
            return

        ctrl = arm_rt.controller
        if ctrl is None:
            arm_rt.status = "not connected"
            return

        try:
            pose_raw = ctrl.read_current_pose()
        except Exception as exc:
            arm_rt.status = "read error"
            arm_rt.last_error = str(exc)
            self._close_arm(arm_rt)
            return

        if pose_raw is None or len(pose_raw) < 6:
            arm_rt.status = "connected (no pose)"
            return

        arm_rt.pose_mm_deg = [v / 1000.0 for v in pose_raw]
        arm_rt.pose_ts = time.time()
        arm_rt.status = "connected"
        arm_rt.last_error = None

    @staticmethod
    def _render_arm_monitor(arm_rt: ArmRuntime, start_wall: float) -> np.ndarray:
        panel = np.zeros((330, 520, 3), dtype=np.uint8)
        panel[:] = (20, 20, 20)

        cv2.putText(panel, "Arm Monitor", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        status_color = (0, 220, 0) if arm_rt.status.startswith("connected") else (0, 180, 255)
        cv2.putText(panel, f"Status: {arm_rt.status}", (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        y = 98
        if arm_rt.pose_mm_deg is None:
            cv2.putText(panel, "Pose: unavailable", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
            y += 28
        else:
            labels = ["X", "Y", "Z", "RX", "RY", "RZ"]
            units = ["mm", "mm", "mm", "deg", "deg", "deg"]
            for i, (label, unit) in enumerate(zip(labels, units)):
                value = arm_rt.pose_mm_deg[i]
                cv2.putText(
                    panel,
                    f"{label:>2}: {value:8.2f} {unit}",
                    (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (220, 220, 220),
                    1,
                )
                y += 26

            age_s = max(0.0, time.time() - arm_rt.pose_ts)
            cv2.putText(panel, f"Last update: {age_s:.2f}s ago", (16, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (170, 170, 170), 1)

        elapsed = max(0.0, time.time() - start_wall)
        cv2.putText(panel, f"Elapsed: {elapsed:.1f}s | q=quit", (16, 312), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (150, 255, 150), 1)

        if arm_rt.last_error:
            err = arm_rt.last_error
            if len(err) > 68:
                err = err[:65] + "..."
            cv2.putText(panel, f"Last error: {err}", (16, 286), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (120, 120, 255), 1)

        return panel

    @staticmethod
    def _build_camera_tile(bundle: CameraFrameBundle) -> np.ndarray | None:
        if bundle.color is not None and bundle.depth is not None:
            depth_vis = depth_to_colormap(bundle.depth)
            if depth_vis.shape[:2] != bundle.color.shape[:2]:
                depth_vis = cv2.resize(depth_vis, (bundle.color.shape[1], bundle.color.shape[0]))
            return np.hstack([bundle.color, depth_vis])

        if bundle.color is not None:
            return bundle.color

        if bundle.depth is not None:
            return depth_to_colormap(bundle.depth)

        return None

    def _build_preview_canvas(
        self,
        cam1_bundle: CameraFrameBundle | None,
        cam2_bundle: CameraFrameBundle | None,
        frame_idx: int,
        elapsed: float,
        cam1_depth_mm: float | None,
        cam2_depth_mm: float | None,
    ) -> np.ndarray | None:
        tiles: list[np.ndarray] = []

        if cam1_bundle is not None:
            tile = self._build_camera_tile(cam1_bundle)
            if tile is not None:
                tiles.append(tile)

        if cam2_bundle is not None:
            tile = self._build_camera_tile(cam2_bundle)
            if tile is not None:
                tiles.append(tile)

        if not tiles:
            return None

        if len(tiles) > 1:
            w = min(img.shape[1] for img in tiles)
            resized = [cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1]))) for img in tiles]
            canvas = np.vstack(resized)
        else:
            canvas = tiles[0]

        cv2.putText(
            canvas,
            f"obj={self.config.object_name} frame={frame_idx} elapsed={elapsed:.1f}s",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            canvas,
            "q/ESC quit | yolo=reserved_stub",
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

        depth_text = ""
        if cam1_depth_mm is not None:
            depth_text += f"cam1 center depth={cam1_depth_mm:.0f}mm"
        if cam2_depth_mm is not None:
            if depth_text:
                depth_text += " | "
            depth_text += f"cam2 center depth={cam2_depth_mm:.0f}mm"
        if depth_text:
            cv2.putText(canvas, depth_text, (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        return canvas

    def _update_running_flag(self, running: bool) -> None:
        with self._lock:
            self._running = running

    def run(self) -> dict[str, Any]:
        self._validate_config()
        cv2.ocl.setUseOpenCL(False)

        cfg = self.config
        f = cfg.features

        if f.preview_window_enabled and threading.current_thread() is not threading.main_thread():
            print("[preview] local preview window disabled in non-main thread; use /preview.mjpg")
            f.preview_window_enabled = False

        output_root = (PROJECT_ROOT / cfg.output_root).resolve()
        object_dir = output_root / cfg.object_name
        episode_idx = cfg.episode_index if cfg.episode_index is not None else next_episode_index(object_dir)
        final_episode_dir = object_dir / f"episode_{episode_idx:03d}"

        staging_root = output_root / ".staging" / cfg.object_name
        staging_episode_dir = staging_root / f"episode_{episode_idx:03d}_{int(time.time())}_{os.getpid()}"

        cam1: CameraRuntime | None = None
        cam2: CameraRuntime | None = None
        dirs: dict[str, Path] | None = None
        commit_target: Path | None = None
        committed_episode_idx: int | None = None

        arm_rt = self._init_arm_runtime()
        gripper_client: GripperClient | None = None
        gripper_sync_end_offset: float | None = None
        gripper_sync_end_rtt_ms: float | None = None

        frame_idx = 0
        saved_pose_rows = 0
        gripper_rows = 0
        stop_reason = "unknown"
        consecutive_frame_errors = 0
        preview_windows_initialized = False
        run_error: str | None = None
        run_failed = False

        trajectory_file = None
        trajectory_writer = None
        gripper_file = None
        gripper_writer = None

        start_wall = time.time()
        record_start_wall = start_wall
        start_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start_wall))

        self._emit_event("session_starting", {"object": cfg.object_name, "episode": episode_idx})
        self._update_running_flag(True)

        try:
            devices = list_devices_with_retry(cfg.power_retries, cfg.power_retry_delay)
            print("[realsense] connected devices:")
            for dev in devices:
                print(f"  - {dev.serial} | {dev.name} | FW {dev.firmware} | USB {dev.usb_type}")

            if f.cam1_enabled:
                cam1 = self._start_camera("cam1", cfg.cam1_serial, devices)
            if f.cam2_enabled and cfg.cam2_serial:
                if f.cam1_enabled and cam1 is not None:
                    time.sleep(2.0)
                cam2 = self._start_camera("cam2", cfg.cam2_serial, devices)

            if f.arm_log_enabled:
                self._ensure_arm_connected(arm_rt, verbose=True)

            if f.gripper_log_enabled:
                gripper_client = connect_gripper(cfg.gripper_api_url, cfg.gripper_timeout_s, cfg.gripper_sync_samples)

            if f.disk_output_enabled:
                dirs = self._make_output_dirs(staging_episode_dir, with_cam2=bool(cam2 is not None))

                if f.arm_log_enabled and "trajectory" in dirs:
                    trajectory_file = open(dirs["trajectory"], "w", newline="", encoding="utf-8")
                    trajectory_writer = csv.writer(trajectory_file)
                    trajectory_writer.writerow([
                        "timestamp_unix",
                        "elapsed_s",
                        "x_mm",
                        "y_mm",
                        "z_mm",
                        "rx_deg",
                        "ry_deg",
                        "rz_deg",
                        "valid",
                    ])

                if f.gripper_log_enabled and "gripper_stream" in dirs:
                    gripper_file = open(dirs["gripper_stream"], "w", newline="", encoding="utf-8")
                    gripper_writer = csv.writer(gripper_file)
                    gripper_writer.writerow([
                        "timestamp_unix",
                        "elapsed_s",
                        "server_time_unix",
                        "remote_elapsed_s",
                        "pos1",
                        "pos2",
                        "pos3",
                        "val1",
                        "val2",
                        "val3",
                        "tactile_timestamp_unix",
                        "sensor_connected",
                        "sample_valid",
                        "rtt_ms",
                        "clock_offset_sec",
                        "api_version",
                        "recording_session_id",
                        "status",
                        "error",
                    ])

            record_start_wall = time.time()
            start_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record_start_wall))

            print("\nDemo recording started")
            print(f"  object: {cfg.object_name}")
            print(f"  episode: {episode_idx:03d}")
            print(f"  safe_profile: {cfg.safe_profile}")
            print(f"  features: {asdict(f)}")
            if cfg.duration_s > 0:
                print(f"  duration: {cfg.duration_s:.1f}s")
            else:
                print("  duration: until Ctrl+C / stop request")

            next_frame_t = time.time()
            next_pose_t = time.time()
            next_arm_poll_t = time.time()
            next_gripper_poll_t = time.time()

            while True:
                if self.stop_event.is_set():
                    stop_reason = "external_stop"
                    break

                now = time.time()
                if now < next_frame_t:
                    time.sleep(min(0.002, next_frame_t - now))
                    continue

                frame_time = time.time()
                elapsed = frame_time - record_start_wall

                if cfg.duration_s > 0 and elapsed >= cfg.duration_s:
                    stop_reason = "duration_reached"
                    break

                if f.arm_log_enabled and frame_time >= next_arm_poll_t:
                    self._poll_arm_pose_live(arm_rt)
                    next_arm_poll_t = frame_time + (1.0 / cfg.arm_monitor_hz)

                gripper_state = None
                if f.gripper_log_enabled and gripper_client is not None and frame_time >= next_gripper_poll_t:
                    gripper_state = poll_gripper_state(gripper_client)
                    if gripper_writer is not None:
                        if gripper_state is None:
                            gripper_writer.writerow([
                                f"{frame_time:.6f}",
                                f"{elapsed:.3f}",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                0,
                                0,
                                "",
                                f"{gripper_client.offset_sec:+.6f}",
                                gripper_client.api_version,
                                "",
                                "fail",
                                gripper_client.last_error or "",
                            ])
                        else:
                            pos = gripper_state.get("current_pos", [])
                            tactile = gripper_state.get("tactile_data", [])
                            p1 = pos[0] if isinstance(pos, list) and len(pos) > 0 else ""
                            p2 = pos[1] if isinstance(pos, list) and len(pos) > 1 else ""
                            p3 = pos[2] if isinstance(pos, list) and len(pos) > 2 else ""
                            v1 = tactile[0] if isinstance(tactile, list) and len(tactile) > 0 else ""
                            v2 = tactile[1] if isinstance(tactile, list) and len(tactile) > 1 else ""
                            v3 = tactile[2] if isinstance(tactile, list) and len(tactile) > 2 else ""
                            gripper_writer.writerow([
                                f"{frame_time:.6f}",
                                f"{elapsed:.3f}",
                                gripper_state.get("server_time_unix", ""),
                                gripper_state.get("elapsed_s", ""),
                                p1,
                                p2,
                                p3,
                                v1,
                                v2,
                                v3,
                                gripper_state.get("tactile_timestamp_unix", ""),
                                int(bool(gripper_state.get("sensor_connected", False))),
                                int(bool(gripper_state.get("sample_valid", False))),
                                gripper_state.get("rtt_ms", ""),
                                f"{gripper_client.offset_sec:+.6f}",
                                gripper_state.get("api_version", gripper_client.api_version),
                                gripper_state.get("recording_session_id", ""),
                                "ok",
                                "",
                            ])
                        gripper_rows += 1
                    next_gripper_poll_t = frame_time + (1.0 / cfg.gripper_poll_hz)

                cam1_bundle = None
                cam2_bundle = None
                cam1_depth_mm = None
                cam2_depth_mm = None

                try:
                    if cam1 is not None:
                        cam1_bundle = self._capture_frame(cam1)
                        cam1_depth_mm = center_depth_mm(cam1_bundle.depth)

                    if cam2 is not None:
                        cam2_bundle = self._capture_frame(cam2)
                        cam2_depth_mm = center_depth_mm(cam2_bundle.depth)
                except Exception as exc:
                    consecutive_frame_errors += 1
                    if consecutive_frame_errors == 1 or consecutive_frame_errors % 5 == 0:
                        print(f"[frame] read error ({consecutive_frame_errors}/{cfg.max_frame_errors}): {exc}")
                    if consecutive_frame_errors >= cfg.max_frame_errors:
                        stop_reason = "too_many_frame_errors"
                        break
                    next_frame_t = frame_time + (1.0 / cfg.fps)
                    continue

                consecutive_frame_errors = 0

                if f.disk_output_enabled and dirs is not None:
                    if cam1_bundle is not None:
                        if cam1_bundle.color is not None and "cam1_rgb" in dirs:
                            ok = cv2.imwrite(
                                str(dirs["cam1_rgb"] / f"frame_{frame_idx:06d}.jpg"),
                                cam1_bundle.color,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                            )
                            if not ok:
                                raise RuntimeError("Failed to write cam1 rgb frame")
                        if cam1_bundle.depth is not None and "cam1_depth" in dirs:
                            ok = cv2.imwrite(
                                str(dirs["cam1_depth"] / f"frame_{frame_idx:06d}.png"),
                                cam1_bundle.depth,
                                [int(cv2.IMWRITE_PNG_COMPRESSION), 3],
                            )
                            if not ok:
                                raise RuntimeError("Failed to write cam1 depth frame")

                    if cam2_bundle is not None:
                        if cam2_bundle.color is not None and "cam2_rgb" in dirs:
                            ok = cv2.imwrite(
                                str(dirs["cam2_rgb"] / f"frame_{frame_idx:06d}.jpg"),
                                cam2_bundle.color,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                            )
                            if not ok:
                                raise RuntimeError("Failed to write cam2 rgb frame")
                        if cam2_bundle.depth is not None and "cam2_depth" in dirs:
                            ok = cv2.imwrite(
                                str(dirs["cam2_depth"] / f"frame_{frame_idx:06d}.png"),
                                cam2_bundle.depth,
                                [int(cv2.IMWRITE_PNG_COMPRESSION), 3],
                            )
                            if not ok:
                                raise RuntimeError("Failed to write cam2 depth frame")

                frame_idx += 1

                if f.arm_log_enabled and frame_time >= next_pose_t and trajectory_writer is not None:
                    if arm_rt.pose_mm_deg is not None:
                        pose = arm_rt.pose_mm_deg
                        valid = int(any(abs(v) > 1e-6 for v in pose[:3]))
                        trajectory_writer.writerow([
                            f"{frame_time:.6f}",
                            f"{elapsed:.3f}",
                            f"{pose[0]:.3f}",
                            f"{pose[1]:.3f}",
                            f"{pose[2]:.3f}",
                            f"{pose[3]:.3f}",
                            f"{pose[4]:.3f}",
                            f"{pose[5]:.3f}",
                            valid,
                        ])
                    else:
                        trajectory_writer.writerow([f"{frame_time:.6f}", f"{elapsed:.3f}", "", "", "", "", "", "", 0])

                    saved_pose_rows += 1
                    next_pose_t = frame_time + (1.0 / cfg.pose_hz)

                canvas = self._build_preview_canvas(
                    cam1_bundle=cam1_bundle,
                    cam2_bundle=cam2_bundle,
                    frame_idx=frame_idx,
                    elapsed=elapsed,
                    cam1_depth_mm=cam1_depth_mm,
                    cam2_depth_mm=cam2_depth_mm,
                )

                if canvas is not None:
                    self._emit_preview(canvas)

                telemetry_payload = {
                    "running": True,
                    "object": cfg.object_name,
                    "episode": episode_idx,
                    "frame_idx": frame_idx,
                    "elapsed_s": round(elapsed, 3),
                    "stop_reason": stop_reason,
                    "features": asdict(f),
                    "cameras": {
                        "cam1": {
                            "enabled": bool(cam1 is not None),
                            "serial": cam1.serial if cam1 is not None else "",
                            "depth_center_mm": cam1_depth_mm,
                        },
                        "cam2": {
                            "enabled": bool(cam2 is not None),
                            "serial": cam2.serial if cam2 is not None else "",
                            "depth_center_mm": cam2_depth_mm,
                        },
                    },
                    "arm": {
                        "enabled": f.arm_log_enabled,
                        "status": arm_rt.status,
                        "pose_mm_deg": arm_rt.pose_mm_deg,
                        "reconnect_count": arm_rt.reconnect_count,
                        "last_error": arm_rt.last_error,
                    },
                    "gripper": {
                        "enabled": f.gripper_log_enabled,
                        "connected": bool(gripper_client is not None and gripper_client.connected),
                        "poll_ok": gripper_client.poll_ok if gripper_client is not None else 0,
                        "poll_fail": gripper_client.poll_fail if gripper_client is not None else 0,
                        "last_error": gripper_client.last_error if gripper_client is not None else None,
                        "last_state": gripper_state if gripper_state is not None else (
                            gripper_client.last_state if gripper_client is not None else None
                        ),
                    },
                    "yolo": {
                        "enabled": f.yolo_enabled,
                        "mode": "reserved_stub",
                        "detections": [],
                    },
                }
                self._emit_telemetry(telemetry_payload)

                if f.preview_window_enabled:
                    try:
                        if not preview_windows_initialized:
                            cv2.namedWindow("demo_realtime", cv2.WINDOW_NORMAL)
                            cv2.namedWindow("demo_arm_monitor", cv2.WINDOW_NORMAL)
                            preview_windows_initialized = True

                        preview_canvas = canvas
                        if preview_canvas is None:
                            preview_canvas = np.zeros((360, 640, 3), dtype=np.uint8)
                            cv2.putText(preview_canvas, "No frame available", (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.8, (255, 255, 255), 2)
                        if cfg.preview_scale != 1.0:
                            preview_canvas = cv2.resize(
                                preview_canvas,
                                None,
                                fx=cfg.preview_scale,
                                fy=cfg.preview_scale,
                                interpolation=cv2.INTER_LINEAR,
                            )
                        cv2.imshow("demo_realtime", preview_canvas)

                        monitor = self._render_arm_monitor(arm_rt, record_start_wall)
                        if cfg.monitor_scale != 1.0:
                            monitor = cv2.resize(
                                monitor,
                                None,
                                fx=cfg.monitor_scale,
                                fy=cfg.monitor_scale,
                                interpolation=cv2.INTER_LINEAR,
                            )
                        cv2.imshow("demo_arm_monitor", monitor)

                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            stop_reason = "preview_quit"
                            break
                    except Exception as exc:
                        print(f"[preview] disabling local preview after OpenCV error: {exc}")
                        f.preview_window_enabled = False
                        if preview_windows_initialized:
                            try:
                                cv2.destroyAllWindows()
                            except Exception:
                                pass
                            preview_windows_initialized = False

                next_frame_t = frame_time + (1.0 / cfg.fps)

            if stop_reason == "unknown":
                stop_reason = "normal_end"

        except KeyboardInterrupt:
            stop_reason = "keyboard_interrupt"
            print("\nInterrupted by user")
        except Exception as exc:
            run_failed = True
            run_error = str(exc)
            stop_reason = "startup_failed" if frame_idx == 0 else "runtime_error"
            print(f"\n[error] {run_error}")
        finally:
            end_wall = time.time()
            duration_actual = max(0.0, end_wall - record_start_wall)

            if trajectory_file is not None:
                trajectory_file.close()
            if gripper_file is not None:
                gripper_file.close()

            stop_camera(cam1)
            stop_camera(cam2)
            self._close_arm(arm_rt)

            if gripper_client is not None and gripper_client.connected:
                offset_end, rtt_end, ok, api_version_end, time_sync_supported_end = estimate_gripper_offset(
                    gripper_client.session,
                    gripper_client.base_url,
                    gripper_client.timeout_s,
                    cfg.gripper_sync_samples,
                )
                if ok:
                    gripper_sync_end_offset = offset_end
                    gripper_sync_end_rtt_ms = rtt_end
                print(
                    "[gripper] end sync"
                    f" ok={ok}"
                    f" api_version={api_version_end}"
                    f" time_sync_supported={time_sync_supported_end}"
                    f" offset_sec={offset_end:+.6f}"
                    f" rtt_ms={rtt_end:.3f}"
                )

            if gripper_client is not None:
                try:
                    gripper_client.session.close()
                except Exception:
                    pass

            if f.preview_window_enabled:
                cv2.destroyAllWindows()

            if f.disk_output_enabled and dirs is not None and frame_idx > 0:
                final_episode_dir.parent.mkdir(parents=True, exist_ok=True)
                candidate_idx = episode_idx
                while True:
                    candidate = final_episode_dir.parent / f"episode_{candidate_idx:03d}"
                    if candidate.exists() and not episode_has_data(candidate):
                        cleanup_dir_if_exists(candidate)
                    if not candidate.exists():
                        commit_target = candidate
                        break
                    candidate_idx += 1

                shutil.move(str(dirs["episode"]), str(commit_target))
                dirs = self._make_output_dirs(commit_target, with_cam2=bool(cam2 is not None))
                committed_episode_idx = int(commit_target.name.split("_")[-1])
            elif f.disk_output_enabled and dirs is not None:
                cleanup_dir_if_exists(dirs["episode"])
                cleanup_empty_parents(staging_root, output_root)

            metadata = {
                "object": cfg.object_name,
                "episode": committed_episode_idx if committed_episode_idx is not None else episode_idx,
                "start_time": start_iso,
                "end_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(end_wall)),
                "duration_s": round(duration_actual, 3),
                "target_fps": cfg.fps,
                "pose_hz": cfg.pose_hz,
                "total_frames": frame_idx,
                "saved_pose_rows": saved_pose_rows,
                "saved_gripper_rows": gripper_rows,
                "stop_reason": stop_reason,
                "run_error": run_error,
                "features": asdict(f),
                "stream_timeout_ms": cfg.stream_timeout_ms,
                "max_frame_errors": cfg.max_frame_errors,
                "profile_retries_high": cfg.profile_retries_high,
                "profile_retries_mid": cfg.profile_retries_mid,
                "start_retry_sleep": cfg.start_retry_sleep,
                "camera_reset_retries": cfg.camera_reset_retries,
                "camera_reset_wait": cfg.camera_reset_wait,
                "cameras": {
                    "cam1": {
                        "enabled": bool(cam1 is not None),
                        "serial": cfg.cam1_serial,
                        "device_name": cam1.device_info.name if cam1 is not None else "",
                        "firmware": cam1.device_info.firmware if cam1 is not None else "",
                        "usb_type": cam1.device_info.usb_type if cam1 is not None else "",
                        "profile": serialize_profile(cam1.profile if cam1 is not None else None),
                    },
                    "cam2": {
                        "enabled": bool(cam2 is not None),
                        "serial": cfg.cam2_serial,
                        "device_name": cam2.device_info.name if cam2 is not None else "",
                        "firmware": cam2.device_info.firmware if cam2 is not None else "",
                        "usb_type": cam2.device_info.usb_type if cam2 is not None else "",
                        "profile": serialize_profile(cam2.profile if cam2 is not None else None),
                    },
                },
                "arm": {
                    "enabled": f.arm_log_enabled,
                    "host": cfg.arm_host,
                    "port": cfg.arm_port,
                    "connected": arm_rt.status.startswith("connected"),
                    "status": arm_rt.status,
                    "reconnect_count": arm_rt.reconnect_count,
                    "last_error": arm_rt.last_error,
                },
                "gripper": {
                    "enabled": f.gripper_log_enabled,
                    "api_url": sanitize_url(cfg.gripper_api_url) if cfg.gripper_api_url else "",
                    "connected": bool(gripper_client is not None and gripper_client.connected),
                    "api_version": gripper_client.api_version if gripper_client is not None else None,
                    "poll_ok": gripper_client.poll_ok if gripper_client is not None else 0,
                    "poll_fail": gripper_client.poll_fail if gripper_client is not None else 0,
                    "time_sync_supported": gripper_client.time_sync_supported if gripper_client is not None else False,
                    "clock_offset_start_sec": (
                        gripper_client.offset_sec if gripper_client is not None and gripper_client.connected else None
                    ),
                    "clock_offset_end_sec": gripper_sync_end_offset,
                    "clock_offset_end_rtt_ms": gripper_sync_end_rtt_ms,
                    "recording_session_id": (
                        gripper_client.last_state.get("recording_session_id")
                        if gripper_client is not None and isinstance(gripper_client.last_state, dict)
                        else None
                    ),
                    "sensor_connected": (
                        gripper_client.last_state.get("sensor_connected")
                        if gripper_client is not None and isinstance(gripper_client.last_state, dict)
                        else None
                    ),
                    "sensor_disconnect_count": (
                        gripper_client.last_state.get("sensor_disconnect_count")
                        if gripper_client is not None and isinstance(gripper_client.last_state, dict)
                        else None
                    ),
                    "last_error": gripper_client.last_error if gripper_client is not None else None,
                },
                "yolo": {
                    "enabled": f.yolo_enabled,
                    "mode": "reserved_stub",
                    "detections": [],
                },
                "notes": cfg.notes,
            }

            if f.disk_output_enabled and dirs is not None and frame_idx > 0 and commit_target is not None:
                with open(dirs["metadata"], "w", encoding="utf-8") as out_f:
                    json.dump(metadata, out_f, indent=2, ensure_ascii=False)

            summary = {
                "ok": not run_failed,
                "object": cfg.object_name,
                "episode": metadata["episode"],
                "frames": frame_idx,
                "duration_s": round(duration_actual, 3),
                "avg_fps": round(frame_idx / duration_actual, 3) if duration_actual > 0 and frame_idx > 0 else None,
                "stop_reason": stop_reason,
                "run_error": run_error,
                "output_dir": str(commit_target) if commit_target is not None else None,
                "metadata": metadata,
            }

            final_telemetry = {
                "running": False,
                "object": cfg.object_name,
                "episode": metadata["episode"],
                "frame_idx": frame_idx,
                "elapsed_s": round(duration_actual, 3),
                "stop_reason": stop_reason,
                "run_error": run_error,
                "summary": summary,
            }
            self._emit_telemetry(final_telemetry)

            with self._lock:
                self._summary = summary

            self._update_running_flag(False)
            self._emit_event("session_stopped", summary)

            print("\nDemo recording finished")
            if commit_target is not None:
                print(f"  output: {commit_target}")
            else:
                print("  output: not committed (no valid frames captured or output disabled)")
            print(f"  frames: {frame_idx}")
            if duration_actual > 0 and frame_idx > 0:
                print(f"  avg_fps: {frame_idx / duration_actual:.2f}")
            print(f"  stop_reason: {stop_reason}")
            if run_error:
                print(f"  error: {run_error}")

        if run_failed:
            raise RuntimeError(run_error or "runtime failed")

        assert self.summary is not None
        return self.summary
