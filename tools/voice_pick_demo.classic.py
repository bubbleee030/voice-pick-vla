"""
Voice Pick Demo Server.

Flask + SocketIO backend serving:
- Web UI with voice input, camera feeds, arm monitor, trajectory plot
- WebSocket events for real-time arm telemetry and pick control
- MJPEG camera streaming (RGB + Depth)
- Teach mode recording and replay

Usage:
    python tools/voice_pick_demo.py --host 0.0.0.0 --port 8090
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
try:
    import torch
except Exception:
    torch = None
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

YOLO_CONFIG_DIR = PROJECT_ROOT / "data" / "models" / "ultralytics"
YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit

from src.nlu import IntentParser
from src.teach_pipeline import PhaseSpecManager, TeachDatasetRecorder, infer_object_key_from_name
from src.utils import load_config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("demo")
PANEL_RECORDINGS_DIR = PROJECT_ROOT / "data" / "panel_recordings"


def _safe_file_token(value: Any, default: str = "item") -> str:
    token = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value or "").strip())
    token = token.strip("._")
    return token or default


def _resolve_torch_device(preferred: str | None = None) -> str:
    def cuda_runtime_supported() -> bool:
        if torch is None or not torch.cuda.is_available():
            return False
        try:
            major, minor = torch.cuda.get_device_capability(0)
            current = f"sm_{major}{minor}"
            supported = set(torch.cuda.get_arch_list() or [])
            if supported and current not in supported:
                log.warning(
                    "CUDA visible but current PyTorch wheel does not support %s; using CPU instead.",
                    current,
                )
                return False
        except Exception:
            return False
        return True

    candidate = str(preferred or "auto").strip().lower()
    if candidate == "auto":
        if cuda_runtime_supported():
            return "cuda:0"
        return "cpu"
    if candidate.startswith("cuda"):
        if cuda_runtime_supported():
            return candidate
        return "cpu"
    return "cpu"


def _gripper_endpoint_candidates(demo_cfg: dict) -> list[tuple[str, str]]:
    grip_cfg = demo_cfg.get("gripper", {})
    candidates: list[tuple[str, str]] = []
    seen: set[tuple[str, int]] = set()

    for idx, item in enumerate(grip_cfg.get("endpoints", [])):
        if not isinstance(item, dict):
            continue
        host = str(item.get("host", "")).strip()
        port = int(item.get("port", grip_cfg.get("agx_port", 5000)))
        if not host:
            continue
        key = (host, port)
        if key in seen:
            continue
        seen.add(key)
        label = str(item.get("label", f"gripper_{idx + 1}"))
        candidates.append((label, f"http://{host}:{port}"))

    fallback_host = str(grip_cfg.get("agx_ip", "")).strip()
    fallback_port = int(grip_cfg.get("agx_port", 5000))
    if fallback_host:
        key = (fallback_host, fallback_port)
        if key not in seen:
            candidates.append(("gripper_fallback", f"http://{fallback_host}:{fallback_port}"))
    return candidates


MAINTENANCE_POSE_LIMITS = {
    "x_min": -43264,
    "x_max": 671665,
    "y_min": -469446,
    "y_max": 469456,
    "z_min": 155572,
    "z_max": 968425,
    "rx_min": -89999,
    "rx_max": 89999,
    "ry_min": -89999,
    "ry_max": 89999,
}


def _module_cfg(demo_cfg: dict, name: str) -> dict[str, Any]:
    modules = demo_cfg.get("modules", {}) if isinstance(demo_cfg, dict) else {}
    cfg = modules.get(name, {}) if isinstance(modules, dict) else {}
    return cfg if isinstance(cfg, dict) else {}


def _module_enabled(demo_cfg: dict, name: str, default: bool = True) -> bool:
    cfg = _module_cfg(demo_cfg, name)
    value = cfg.get("enabled", default)
    return bool(value if value is not None else default)


def _module_visible(demo_cfg: dict, name: str, default: bool = True) -> bool:
    cfg = _module_cfg(demo_cfg, name)
    enabled = cfg.get("enabled", default)
    show = cfg.get("show_in_ui", enabled)
    return bool(show if show is not None else default)


def _effective_pose_limits(demo_cfg: dict) -> dict[str, int]:
    safety = demo_cfg.get("safety_boundary", {}) if isinstance(demo_cfg, dict) else {}
    pose_cfg = demo_cfg.get("pose_limits", {}) if isinstance(demo_cfg, dict) else {}
    effective: dict[str, int] = {}
    enforce_maintenance = bool(pose_cfg.get("enforce_maintenance_limits", True))

    for axis in ("x", "y", "z"):
        cfg_min = pose_cfg.get(f"{axis}_min", safety.get(f"{axis}_min"))
        cfg_max = pose_cfg.get(f"{axis}_max", safety.get(f"{axis}_max"))
        maint_min = MAINTENANCE_POSE_LIMITS.get(f"{axis}_min")
        maint_max = MAINTENANCE_POSE_LIMITS.get(f"{axis}_max")
        if cfg_min is None:
            cfg_min = maint_min
        if cfg_max is None:
            cfg_max = maint_max
        if enforce_maintenance:
            effective[f"{axis}_min"] = int(max(int(cfg_min), int(maint_min)))
            effective[f"{axis}_max"] = int(min(int(cfg_max), int(maint_max)))
        else:
            effective[f"{axis}_min"] = int(cfg_min)
            effective[f"{axis}_max"] = int(cfg_max)

    for axis in ("rx", "ry"):
        cfg_min = pose_cfg.get(f"{axis}_min", MAINTENANCE_POSE_LIMITS[f"{axis}_min"])
        cfg_max = pose_cfg.get(f"{axis}_max", MAINTENANCE_POSE_LIMITS[f"{axis}_max"])
        maint_min = MAINTENANCE_POSE_LIMITS[f"{axis}_min"]
        maint_max = MAINTENANCE_POSE_LIMITS[f"{axis}_max"]
        if enforce_maintenance:
            effective[f"{axis}_min"] = int(max(int(cfg_min), int(maint_min)))
            effective[f"{axis}_max"] = int(min(int(cfg_max), int(maint_max)))
        else:
            effective[f"{axis}_min"] = int(cfg_min)
            effective[f"{axis}_max"] = int(cfg_max)

    return effective


def _effective_claw_cfg(claw_cfg: dict[str, Any]) -> dict[str, Any]:
    effective = dict(claw_cfg or {})
    profile_name = str(effective.get("performance_profile", "balanced")).strip() or "balanced"
    profiles = effective.get("profiles", {}) if isinstance(effective.get("profiles"), dict) else {}
    profile_cfg = profiles.get(profile_name, {}) if isinstance(profiles, dict) else {}
    merged = {}
    if isinstance(profile_cfg, dict):
        merged.update(profile_cfg)

    for key in (
        "capture_w", "capture_h", "stream_w", "stream_h",
        "infer_w", "infer_h", "stream_fps", "jpeg_quality", "infer_interval_s",
    ):
        if key in effective:
            merged[key] = effective[key]

    overrides = effective.get("overrides", {})
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            merged[key] = value

    effective.update(merged)
    effective["performance_profile"] = profile_name
    return effective


def _apply_module_runtime_overrides(demo_cfg: dict) -> dict:
    cfg = copy.deepcopy(demo_cfg)
    cameras = cfg.get("cameras", {}) if isinstance(cfg.get("cameras"), dict) else {}
    for cam_key, module_name in (("cam1", "cam1"), ("cam2", "cam2"), ("claw", "claw_cam")):
        cam_cfg = cameras.get(cam_key, {})
        if isinstance(cam_cfg, dict) and not _module_enabled(cfg, module_name, cam_cfg.get("enabled", True)):
            cam_cfg["enabled"] = False
    if not _module_enabled(cfg, "gripper", cfg.get("gripper", {}).get("enabled", True)):
        cfg.setdefault("gripper", {})["enabled"] = False
    if not _module_enabled(cfg, "teach", cfg.get("teach", {}).get("enabled", True)):
        cfg.setdefault("teach", {})["enabled"] = False
    if not _module_enabled(cfg, "virtual_env", cfg.get("virtual_env", {}).get("enabled", True)):
        cfg.setdefault("virtual_env", {})["enabled"] = False
    if not _module_enabled(cfg, "sensor_chart", cfg.get("sensor_api", {}).get("enabled", True)):
        cfg.setdefault("sensor_api", {})["enabled"] = False
    return cfg

# ---------------------------------------------------------------------------
# Camera Manager
# ---------------------------------------------------------------------------
RS_AVAILABLE = False
try:
    import pyrealsense2 as rs
    RS_AVAILABLE = True
except ImportError:
    log.warning("pyrealsense2 not available - camera feeds will use placeholders")

YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception as exc:
    YOLO = None
    log.warning("ultralytics not available - claw camera YOLO disabled: %s", exc)


class SingleCameraStream:
    """One RealSense pipeline producing RGB and optionally depth frames."""

    def __init__(self, cam_id: str, serial: str, w: int, h: int, fps: int,
                 enable_depth: bool, label: str):
        self.cam_id = cam_id
        self.serial = serial
        self.w = w
        self.h = h
        self.fps = fps
        self.enable_depth = enable_depth
        self.label = label
        self.pipeline: Any = None
        self.align: Any = None
        self.running = False
        self.last_rgb: Optional[np.ndarray] = None
        self.last_depth: Optional[np.ndarray] = None
        self.last_depth_raw: Optional[np.ndarray] = None
        self.last_rgb_ts: float = 0.0
        self.last_depth_ts: float = 0.0
        self.last_error = ""
        self.lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        if not RS_AVAILABLE:
            self.last_error = "pyrealsense2 not available"
            return False
        try:
            self.last_error = ""
            self.pipeline = rs.pipeline()
            config = rs.config()
            if self.serial:
                config.enable_device(self.serial)
            config.enable_stream(rs.stream.color, self.w, self.h,
                                 rs.format.bgr8, self.fps)
            if self.enable_depth:
                config.enable_stream(rs.stream.depth, self.w, self.h,
                                     rs.format.z16, self.fps)
            self.pipeline.start(config)
            if self.enable_depth:
                self.align = rs.align(rs.stream.color)
            self.running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            log.info("%s started: serial=%s %dx%d@%dfps depth=%s",
                     self.label, self.serial or "auto", self.w, self.h,
                     self.fps, self.enable_depth)
            return True
        except Exception as e:
            self.last_error = str(e)
            log.error("%s start failed: %s", self.label, e)
            try:
                if self.pipeline:
                    self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
            self.running = False
            return False

    def _capture_loop(self):
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                if self.align:
                    frames = self.align.process(frames)
                color_frame = frames.get_color_frame()
                if color_frame:
                    with self.lock:
                        self.last_rgb = np.asanyarray(color_frame.get_data())
                        self.last_rgb_ts = time.time()
                if self.enable_depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        depth_arr = np.asanyarray(depth_frame.get_data())
                        depth_color = cv2.applyColorMap(
                            cv2.convertScaleAbs(depth_arr, alpha=0.03),
                            cv2.COLORMAP_JET)
                        with self.lock:
                            self.last_depth_raw = depth_arr
                            self.last_depth = depth_color
                            self.last_depth_ts = time.time()
                self.last_error = ""
            except Exception as exc:
                self.last_error = str(exc)
                time.sleep(0.5)

    def get_rgb_jpeg(self) -> bytes:
        with self.lock:
            frame = self.last_rgb
        if frame is None:
            frame = _placeholder(f"{self.label} RGB - No Signal", self.w, self.h)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes()

    def get_depth_jpeg(self) -> bytes:
        if not self.enable_depth:
            frame = _placeholder(f"{self.label} Depth - Disabled", self.w, self.h)
        else:
            with self.lock:
                frame = self.last_depth
            if frame is None:
                frame = _placeholder(f"{self.label} Depth - No Signal",
                                     self.w, self.h)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            rgb = None if self.last_rgb is None else self.last_rgb.copy()
            depth_raw = None if self.last_depth_raw is None else self.last_depth_raw.copy()
            depth_vis = None if self.last_depth is None else self.last_depth.copy()
            rgb_ts = float(self.last_rgb_ts or 0.0)
            depth_ts = float(self.last_depth_ts or 0.0)
        return {
            "rgb": rgb,
            "depth_raw": depth_raw,
            "depth_vis": depth_vis,
            "timestamp_unix": rgb_ts or depth_ts or time.time(),
        }

    def stop(self):
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        self.pipeline = None


class ClawCameraStream:
    """UVC claw-center camera with optional YOLO overlay."""

    def __init__(self, cfg: dict, default_w: int, default_h: int, default_fps: int):
        self.cfg = _effective_claw_cfg(cfg)
        self.label = str(self.cfg.get("label", "Claw Cam"))
        self.source = self.cfg.get("source", 15)
        self.w = int(self.cfg.get("resolution_w", default_w))
        self.h = int(self.cfg.get("resolution_h", default_h))
        self.capture_w = int(self.cfg.get("capture_w", self.w))
        self.capture_h = int(self.cfg.get("capture_h", self.h))
        self.stream_w = int(self.cfg.get("stream_w", self.w))
        self.stream_h = int(self.cfg.get("stream_h", self.h))
        self.infer_w = int(self.cfg.get("infer_w", self.stream_w))
        self.infer_h = int(self.cfg.get("infer_h", self.stream_h))
        self.fps = int(self.cfg.get("fps", default_fps))
        self.stream_fps = float(self.cfg.get("stream_fps", self.fps))
        self.jpeg_quality = int(self.cfg.get("jpeg_quality", 55))
        self.device = _resolve_torch_device(self.cfg.get("device", "auto"))
        self.enable_yolo = bool(self.cfg.get("enable_yolo", True))
        self.model_path = PROJECT_ROOT / str(
            self.cfg.get("model_path", "data/models/yolo/yolo12sbest.pt")
        )
        self.conf = float(self.cfg.get("conf", 0.5))
        self.infer_interval_s = float(self.cfg.get("infer_interval_s", 0.2))
        self.performance_profile = str(self.cfg.get("performance_profile", "balanced"))
        self.cap: Any = None
        self.running = False
        self.yolo_ready = False
        self.last_rgb: Optional[np.ndarray] = None
        self.last_rgb_raw: Optional[np.ndarray] = None
        self.last_labels: list[str] = []
        self.last_error = ""
        self.lock = threading.Lock()
        self._capture_thread: Optional[threading.Thread] = None
        self._infer_thread: Optional[threading.Thread] = None
        self._model: Any = None
        self._last_detected_labels: list[str] = []
        self._latest_frame: Optional[np.ndarray] = None
        self.last_frame_ts: float = 0.0

    def _open_capture(self):
        source = self.source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
        cap = cv2.VideoCapture(source, backend)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_h)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        return cap

    def _load_model(self) -> None:
        self.yolo_ready = False
        self._model = None
        if not self.enable_yolo:
            self.last_error = "YOLO disabled in config"
            return
        if not YOLO_AVAILABLE:
            self.last_error = "ultralytics not installed"
            return
        if not self.model_path.exists():
            self.last_error = f"YOLO model missing: {self.model_path}"
            return
        self._model = YOLO(str(self.model_path))
        if self.device != "cpu":
            try:
                self._model.to(self.device)
            except Exception as exc:
                log.warning("%s YOLO could not move to %s: %s", self.label, self.device, exc)
                self.device = "cpu"
        self.yolo_ready = True
        self.last_error = ""

    def start(self) -> bool:
        try:
            self.cap = self._open_capture()
            if not self.cap or not self.cap.isOpened():
                self.last_error = f"Cannot open UVC source {self.source}"
                return False
            self._load_model()
            self.running = True
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()
            self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
            self._infer_thread.start()
            log.info(
                "%s started: src=%s capture=%dx%d stream=%dx%d@%.1ffps device=%s yolo=%s",
                self.label,
                self.source,
                self.capture_w,
                self.capture_h,
                self.stream_w,
                self.stream_h,
                self.stream_fps,
                self.device,
                self.yolo_ready,
            )
            return True
        except Exception as exc:
            self.last_error = str(exc)
            log.error("%s start failed: %s", self.label, exc)
            return False

    def _capture_loop(self):
        while self.running:
            try:
                if not self.cap or not self.cap.isOpened():
                    time.sleep(0.2)
                    continue
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.05)
                    continue

                raw_frame = frame.copy()
                stream_frame = cv2.resize(
                    frame,
                    (self.stream_w, self.stream_h),
                    interpolation=cv2.INTER_AREA,
                )
                with self.lock:
                    self._latest_frame = stream_frame
                    self.last_rgb_raw = raw_frame
                    self.last_frame_ts = time.time()
                    if self.last_rgb is None:
                        self.last_rgb = stream_frame
            except Exception as exc:
                self.last_error = str(exc)
                time.sleep(0.2)

    def _infer_loop(self):
        while self.running:
            try:
                with self.lock:
                    frame = None if self._latest_frame is None else self._latest_frame.copy()
                if frame is None:
                    time.sleep(0.03)
                    continue

                if self._model is None:
                    with self.lock:
                        self.last_rgb = frame
                        self.last_labels = []
                    time.sleep(max(self.infer_interval_s, 0.05))
                    continue

                infer_frame = frame
                if self.infer_w > 0 and self.infer_h > 0 and (
                    self.infer_w != self.stream_w or self.infer_h != self.stream_h
                ):
                    infer_frame = cv2.resize(
                        frame,
                        (self.infer_w, self.infer_h),
                        interpolation=cv2.INTER_AREA,
                    )

                results = self._model(
                    infer_frame,
                    conf=self.conf,
                    verbose=False,
                    device=self.device,
                )
                labels: list[str] = []
                for box in results[0].boxes:
                    labels.append(self._model.names[int(box.cls[0])])

                display = results[0].plot()
                if display.shape[1] != self.stream_w or display.shape[0] != self.stream_h:
                    display = cv2.resize(
                        display,
                        (self.stream_w, self.stream_h),
                        interpolation=cv2.INTER_LINEAR,
                    )

                with self.lock:
                    self.last_rgb = display
                    self.last_labels = labels
                    self._last_detected_labels = labels
            except Exception as exc:
                self.last_error = str(exc)
            time.sleep(max(self.infer_interval_s, 0.03))

    def get_rgb_jpeg(self) -> bytes:
        with self.lock:
            frame = self.last_rgb
        if frame is None:
            frame = _placeholder(f"{self.label} RGB - No Signal", self.stream_w, self.stream_h)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return buf.tobytes()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            raw = None if self.last_rgb_raw is None else self.last_rgb_raw.copy()
            annotated = None if self.last_rgb is None else self.last_rgb.copy()
            labels = list(self.last_labels)
            timestamp_unix = float(self.last_frame_ts or 0.0)
        return {
            "rgb": raw if raw is not None else annotated,
            "annotated_rgb": annotated,
            "labels": labels,
            "timestamp_unix": timestamp_unix or time.time(),
        }

    def stop(self):
        self.running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)
        if self._infer_thread and self._infer_thread.is_alive():
            self._infer_thread.join(timeout=1.0)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass


def _placeholder(text: str, w: int = 424, h: int = 240) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    font_scale = 0.6 if len(text) > 25 else 0.8
    cv2.putText(img, text, (20, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100, 100, 100), 2)
    return img


class CameraManager:
    """Manage RealSense cameras plus the claw-center UVC camera."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.cams: dict[str, SingleCameraStream] = {}
        self.claw_cam: Optional[ClawCameraStream] = None
        w = cfg.get("resolution_w", 424)
        h = cfg.get("resolution_h", 240)
        fps = cfg.get("fps", 15)
        for cam_key in ("cam1", "cam2"):
            cam_cfg = cfg.get(cam_key, {})
            if not cam_cfg.get("enabled", False):
                continue
            self.cams[cam_key] = SingleCameraStream(
                cam_id=cam_key,
                serial=cam_cfg.get("serial", ""),
                w=w, h=h, fps=fps,
                enable_depth=cam_cfg.get("enable_depth", True),
                label=cam_cfg.get("label", cam_key),
            )
        claw_cfg = cfg.get("claw", {})
        if isinstance(claw_cfg, dict) and claw_cfg.get("enabled", False):
            self.claw_cam = ClawCameraStream(claw_cfg, w, h, fps)

    def start(self):
        delay = self.cfg.get("init_delay_s", 2.0)
        started = []
        for i, (key, cam) in enumerate(self.cams.items()):
            if i > 0 and delay > 0:
                log.info("Waiting %.1fs before starting %s (USB stability)...",
                         delay, cam.label)
                time.sleep(delay)
            ok = cam.start()
            started.append((key, ok))
        if self.claw_cam is not None:
            ok = self.claw_cam.start()
            started.append(("claw", ok))
        if not started:
            log.info("No cameras configured or pyrealsense2 unavailable")
        return started

    def get_cam(self, cam_key: str) -> Optional[Any]:
        if cam_key == "claw":
            return self.claw_cam
        return self.cams.get(cam_key)

    def get_jpeg(self, cam_key: str, stream: str) -> bytes:
        cam = self.get_cam(cam_key)
        if cam is None:
            w = self.cfg.get("resolution_w", 424)
            h = self.cfg.get("resolution_h", 240)
            frame = _placeholder(f"{cam_key} - Not Configured", w, h)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            return buf.tobytes()
        if cam_key == "claw":
            return cam.get_rgb_jpeg()
        if stream == "depth":
            return cam.get_depth_jpeg()
        return cam.get_rgb_jpeg()

    def get_frame(self, cam_key: str, stream: str) -> np.ndarray:
        """Return a BGR frame directly, avoiding JPEG encode/decode for recording."""
        cam = self.get_cam(cam_key)
        w = int(self.cfg.get("resolution_w", 424))
        h = int(self.cfg.get("resolution_h", 240))
        if cam is None:
            return _placeholder(f"{cam_key} - Not Configured", w, h)
        try:
            snapshot = cam.snapshot()
        except Exception:
            snapshot = None
        if not isinstance(snapshot, dict):
            return _placeholder(f"{cam_key} - No Snapshot", w, h)
        if cam_key == "claw":
            frame = snapshot.get("annotated_rgb")
            if frame is None:
                frame = snapshot.get("rgb")
        elif stream == "depth":
            frame = snapshot.get("depth_vis")
        else:
            frame = snapshot.get("rgb")
        if frame is None:
            label = f"{cam_key} {stream} - No Signal"
            if cam_key == "claw":
                claw = self.claw_cam
                w = int(getattr(claw, "stream_w", w))
                h = int(getattr(claw, "stream_h", h))
            return _placeholder(label, w, h)
        return frame.copy()

    def get_stream_interval(self, cam_key: str) -> float:
        if cam_key == "claw" and self.claw_cam is not None:
            fps = max(float(self.claw_cam.stream_fps), 1.0)
            return 1.0 / fps
        fps = max(float(self.cfg.get("fps", 15)), 1.0)
        return 1.0 / fps

    def get_snapshot(self, cam_key: str) -> Optional[dict[str, Any]]:
        cam = self.get_cam(cam_key)
        if cam is None or not hasattr(cam, "snapshot"):
            return None
        try:
            return cam.snapshot()
        except Exception:
            return None

    def status(self) -> dict:
        result = {}
        for key in ("cam1", "cam2"):
            cam = self.cams.get(key)
            if cam:
                result[key] = {
                    "running": cam.running,
                    "depth_enabled": cam.enable_depth,
                    "label": cam.label,
                    "serial": cam.serial,
                    "last_error": cam.last_error or None,
                }
            else:
                result[key] = {
                    "running": False,
                    "depth_enabled": False,
                    "label": key,
                    "serial": None,
                    "last_error": "not configured",
                }
        if self.claw_cam:
            result["claw"] = {
                "running": self.claw_cam.running,
                "depth_enabled": False,
                "label": self.claw_cam.label,
                "yolo_ready": self.claw_cam.yolo_ready,
                "last_labels": list(self.claw_cam.last_labels),
                "last_error": self.claw_cam.last_error or None,
                "profile": self.claw_cam.performance_profile,
                "capture_size": [self.claw_cam.capture_w, self.claw_cam.capture_h],
                "stream_size": [self.claw_cam.stream_w, self.claw_cam.stream_h],
                "infer_size": [self.claw_cam.infer_w, self.claw_cam.infer_h],
                "stream_fps": self.claw_cam.stream_fps,
                "infer_interval_s": self.claw_cam.infer_interval_s,
            }
        else:
            result["claw"] = {
                "running": False,
                "depth_enabled": False,
                "label": "claw",
                "yolo_ready": False,
                "last_labels": [],
                "last_error": None,
                "profile": None,
                "capture_size": [],
                "stream_size": [],
                "infer_size": [],
                "stream_fps": None,
                "infer_interval_s": None,
            }
        return result

    def stop(self):
        for cam in self.cams.values():
            cam.stop()
        if self.claw_cam:
            self.claw_cam.stop()

    def restart_camera(self, cam_key: str) -> dict[str, Any]:
        """Stop/start one camera stream and return fresh status."""
        cam = self.get_cam(cam_key)
        if cam is None:
            return {"ok": False, "error": f"{cam_key} is not configured", "status": self.status()}
        try:
            cam.stop()
            time.sleep(0.4)
            ok = bool(cam.start())
            status = self.status()
            err = None
            if cam_key == "claw" and self.claw_cam is not None:
                err = self.claw_cam.last_error or None
            elif hasattr(cam, "last_error"):
                err = getattr(cam, "last_error") or None
            return {"ok": ok, "error": err, "status": status}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "status": self.status()}

    def reload_config(self, cfg: dict) -> None:
        """Hot-reload lightweight camera settings. Active camera devices still need restart."""
        self.cfg = cfg or {}


class DashboardLogBuffer:
    """Small rolling buffer of frontend-visible dashboard logs for terminal-style videos."""

    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._entries: list[dict[str, Any]] = []

    def append(self, level: Any, message: Any, timestamp: Any = None) -> None:
        entry = {
            "level": str(level or "INFO").upper(),
            "message": str(message or ""),
            "timestamp": str(timestamp or time.strftime("%H:%M:%S")),
            "unix": time.time(),
        }
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self.max_entries:
                self._entries = self._entries[-self.max_entries:]

    def clear(self) -> None:
        with self._lock:
            self._entries = []

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._entries)


class BackendPanelRecorder:
    """Record dashboard-like per-panel MP4s without expensive browser DOM capture."""

    def __init__(
        self,
        cam_mgr: CameraManager,
        arm_mgr_getter: Any,
        gripper_monitor: Any,
        sensor_monitor: Any,
        log_buffer: DashboardLogBuffer,
    ):
        self.cam_mgr = cam_mgr
        self.arm_mgr_getter = arm_mgr_getter
        self.gripper_monitor = gripper_monitor
        self.sensor_monitor = sensor_monitor
        self.log_buffer = log_buffer
        self._lock = threading.Lock()
        self._active: Optional[dict[str, Any]] = None

    def start(self, session_dir: Path, session_id: str, panels: list[dict[str, Any]], fps: float) -> dict[str, Any]:
        self.stop(session_id=None, timeout_s=2.0)
        fps = max(min(float(fps or 60), 60.0), 1.0)
        clean_panels = []
        for panel in panels:
            if not isinstance(panel, dict):
                continue
            panel_id = _safe_file_token(panel.get("panel_id"), "")
            if not panel_id:
                continue
            clean_panels.append({
                "panel_id": panel_id,
                "width": max(int(panel.get("width") or 640), 160),
                "height": max(int(panel.get("height") or 360), 120),
            })
        stop_event = threading.Event()
        state = {
            "session_id": session_id,
            "session_dir": session_dir,
            "panels": clean_panels,
            "fps": fps,
            "stop_event": stop_event,
            "outputs": [],
            "pending_outputs": [],
            "errors": [],
            "started_unix": time.time(),
            "finished_unix": None,
            "frame_counts": {},
            "source_frame_counts": {},
            "arm_history": [],
            "gripper_history": [],
            "sensor_history": [],
            "state_lock": threading.Lock(),
        }
        thread = threading.Thread(target=self._run, args=(state,), daemon=True)
        state["thread"] = thread
        with self._lock:
            self._active = state
        thread.start()
        return {"ok": True, "session_id": session_id, "panels": clean_panels, "fps": fps}

    def stop(self, session_id: Optional[str], timeout_s: float = 8.0) -> Optional[dict[str, Any]]:
        with self._lock:
            state = self._active
            if state is None:
                return None
            if session_id and state.get("session_id") != session_id:
                return None
            self._active = None
        state["stop_event"].set()
        thread = state.get("thread")
        if thread and thread.is_alive():
            thread.join(timeout=timeout_s)
        state["finished_unix"] = time.time()
        self._finalize_recorded_panels(state)
        summary = {
            "session_id": state.get("session_id"),
            "started_unix": state.get("started_unix"),
            "finished_unix": state.get("finished_unix"),
            "fps": state.get("fps"),
            "outputs": state.get("outputs", []),
            "errors": state.get("errors", []),
            "mode": "backend_mp4",
        }
        session_dir = state.get("session_dir")
        if isinstance(session_dir, Path):
            with open(session_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        return summary

    def _run(self, state: dict[str, Any]) -> None:
        threads: list[threading.Thread] = []
        sampler = threading.Thread(target=self._sample_loop, args=(state,), daemon=True)
        sampler.start()
        threads.append(sampler)
        for panel in state["panels"]:
            thread = threading.Thread(target=self._record_single_panel, args=(state, panel), daemon=True)
            thread.start()
            threads.append(thread)
        state["stop_event"].wait()
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=2.0)

    def _sample_loop(self, state: dict[str, Any]) -> None:
        interval = 1.0 / float(state["fps"])
        next_tick = time.perf_counter()
        while not state["stop_event"].is_set():
            self._sample_state_histories(state)
            next_tick += interval
            self._wait_until(state["stop_event"], next_tick)

    def _record_single_panel(self, state: dict[str, Any], panel: dict[str, Any]) -> None:
        panel_id = panel["panel_id"]
        width = self._even(int(panel["width"]))
        height = self._even(int(panel["height"]))
        session_dir: Path = state["session_dir"]
        raw_path = session_dir / f"{panel_id}.raw.mp4"
        final_path = session_dir / f"{panel_id}.mp4"
        target_fps = self._target_fps_for_panel(panel_id, float(state["fps"]))
        writer = cv2.VideoWriter(
            str(raw_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(target_fps),
            (width, height),
        )
        if not writer.isOpened():
            with state["state_lock"]:
                state["errors"].append(f"{panel_id}: cannot open mp4 writer")
            return

        count = 0
        started_perf = time.perf_counter()
        interval = 1.0 / float(target_fps)
        next_tick = time.perf_counter()
        try:
            while not state["stop_event"].is_set():
                render_state = self._snapshot_render_state(state)
                frame = self._render_panel(panel_id, (width, height), render_state)
                writer.write(frame)
                count += 1
                next_tick += interval
                self._wait_until(state["stop_event"], next_tick)
        except Exception as exc:
            with state["state_lock"]:
                state["errors"].append(f"{panel_id}: {exc}")
        finally:
            try:
                writer.release()
            except Exception:
                pass
            with state["state_lock"]:
                state["frame_counts"][panel_id] = count
                state["source_frame_counts"][panel_id] = count
                duration_s = max(time.perf_counter() - started_perf, 0.001)
                effective_fps = max(count / duration_s, 0.1)
                if raw_path.exists() and raw_path.stat().st_size > 0:
                    state["pending_outputs"].append({
                        "panel_id": panel_id,
                        "raw_path": raw_path,
                        "final_path": final_path,
                        "frames": count,
                        "source_frames": count,
                        "target_fps": target_fps,
                        "effective_fps": effective_fps,
                        "duration_s": duration_s,
                        "width": width,
                        "height": height,
                    })

    def _finalize_recorded_panels(self, state: dict[str, Any]) -> None:
        pending = list(state.get("pending_outputs", []))
        for item in pending:
            raw_path = item.get("raw_path")
            final_path = item.get("final_path")
            if not isinstance(raw_path, Path) or not isinstance(final_path, Path):
                continue
            effective_fps = float(item.get("effective_fps") or item.get("target_fps") or state.get("fps") or 1.0)
            ok = self._rewrite_video_fps(raw_path, final_path, effective_fps, int(item.get("width", 0)), int(item.get("height", 0)))
            if not ok:
                try:
                    raw_path.replace(final_path)
                except Exception as exc:
                    with state["state_lock"]:
                        state["errors"].append(f"{item.get('panel_id', 'panel')}: finalize failed: {exc}")
                    continue
            else:
                try:
                    raw_path.unlink(missing_ok=True)
                except Exception:
                    pass
            if final_path.exists() and final_path.stat().st_size > 0:
                with state["state_lock"]:
                    state["outputs"].append({
                        "panel_id": item.get("panel_id"),
                        "path": str(final_path.relative_to(PROJECT_ROOT)),
                        "frames": int(item.get("frames", 0)),
                        "source_frames": int(item.get("source_frames", 0)),
                        "target_fps": float(item.get("target_fps", 0)),
                        "effective_fps": effective_fps,
                        "duration_s": float(item.get("duration_s", 0)),
                        "width": int(item.get("width", 0)),
                        "height": int(item.get("height", 0)),
                        "size_bytes": final_path.stat().st_size,
                    })

    @staticmethod
    def _rewrite_video_fps(raw_path: Path, final_path: Path, fps: float, width: int, height: int) -> bool:
        cap = cv2.VideoCapture(str(raw_path))
        if not cap.isOpened():
            return False
        try:
            if width <= 0:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            if height <= 0:
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if width <= 0 or height <= 0:
                return False
            writer = cv2.VideoWriter(
                str(final_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                max(float(fps), 0.1),
                (width, height),
            )
            if not writer.isOpened():
                return False
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    if frame.shape[1] != width or frame.shape[0] != height:
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    writer.write(frame)
            finally:
                writer.release()
            return final_path.exists() and final_path.stat().st_size > 0
        finally:
            cap.release()

    @staticmethod
    def _target_fps_for_panel(panel_id: str, requested_fps: float) -> float:
        if panel_id in {"cam1-rgb", "cam1-depth", "cam2-rgb", "cam2-depth", "claw-rgb"}:
            return min(requested_fps, 60.0)
        if panel_id in {"monitor", "virtual-env-left", "virtual-env-right", "quick"}:
            return min(requested_fps, 30.0)
        if panel_id in {"gripper", "sensor-chart", "trajectory"}:
            return min(requested_fps, 30.0)
        return min(requested_fps, 15.0)

    @staticmethod
    def _snapshot_render_state(state: dict[str, Any]) -> dict[str, Any]:
        with state["state_lock"]:
            return {
                "arm_history": list(state.get("arm_history", [])),
                "gripper_history": list(state.get("gripper_history", [])),
                "sensor_history": list(state.get("sensor_history", [])),
            }

    @staticmethod
    def _wait_until(stop_event: threading.Event, target_perf: float) -> None:
        # Windows timer granularity can make wait(0.016) behave like ~30 fps.
        # This bounded short-sleep loop keeps recording cadence closer to 60 fps.
        while not stop_event.is_set():
            remaining = target_perf - time.perf_counter()
            if remaining <= 0:
                return
            if remaining > 0.004:
                time.sleep(min(0.002, remaining - 0.002))
            else:
                time.sleep(0)

    def _sample_state_histories(self, state: dict[str, Any]) -> None:
        now = time.time()
        arm_entry = None
        gripper_entry = None
        sensor_entry = None
        arm_mgr = self.arm_mgr_getter()
        pose = getattr(arm_mgr, "current_pose_mm_deg", None)
        if pose is not None:
            try:
                arm_entry = {"t": now, "pose": [float(v) for v in pose[:6]]}
            except Exception:
                pass
        grip_state = getattr(self.gripper_monitor, "last_state", {}) or {}
        current_pos = grip_state.get("current_pos") if isinstance(grip_state, dict) else None
        if isinstance(current_pos, list) and len(current_pos) >= 3:
            try:
                g1, g2, g3 = [float(v) for v in current_pos[:3]]
                gripper_entry = {"t": now, "values": [g2, g1, g3]}
            except Exception:
                pass
        sensor_sample = getattr(self.sensor_monitor, "last_sample", {}) or {}
        if isinstance(sensor_sample, dict):
            try:
                values = [
                    float(sensor_sample.get("analog_value1", 0)),
                    float(sensor_sample.get("analog_value2", 0)),
                    float(sensor_sample.get("analog_value3", 0)),
                ]
                sensor_entry = {"t": now, "values": values}
            except Exception:
                pass
        with state["state_lock"]:
            if arm_entry is not None:
                state["arm_history"].append(arm_entry)
            if gripper_entry is not None:
                state["gripper_history"].append(gripper_entry)
            if sensor_entry is not None:
                state["sensor_history"].append(sensor_entry)
            max_points = 900
            for key in ("arm_history", "gripper_history", "sensor_history"):
                if len(state[key]) > max_points:
                    state[key] = state[key][-max_points:]

    def _render_panel(self, panel_id: str, size: tuple[int, int], state: dict[str, Any]) -> np.ndarray:
        width, height = size
        if panel_id in {"cam1-rgb", "cam1-depth", "cam2-rgb", "cam2-depth", "claw-rgb"}:
            cam_key, stream = panel_id.rsplit("-", 1)
            return self._render_camera_panel(panel_id, cam_key, stream, width, height)
        if panel_id == "logs":
            return self._render_logs_panel(width, height)
        if panel_id == "sensor-chart":
            return self._render_chart_panel("TACTILE SENSOR", state["sensor_history"], width, height)
        if panel_id == "gripper":
            return self._render_chart_panel("GRIPPER", state["gripper_history"], width, height)
        if panel_id == "monitor":
            return self._render_monitor_panel(width, height)
        if panel_id == "trajectory":
            return self._render_trajectory_panel(state["arm_history"], width, height)
        return self._render_placeholder(panel_id.replace("-", " ").upper(), width, height)

    def _render_camera_panel(self, title: str, cam_key: str, stream: str, width: int, height: int) -> np.ndarray:
        try:
            frame = self.cam_mgr.get_frame(cam_key, stream)
        except Exception:
            frame = None
        if frame is None:
            frame = _placeholder(f"{title} no frame", width, height)
        return self._compose_panel(title.upper(), cv2.resize(frame, (width, max(height - 36, 1))), width, height)

    def _render_chart_panel(self, title: str, history: list[dict[str, Any]], width: int, height: int) -> np.ndarray:
        body_h = max(height - 36, 1)
        body = np.zeros((body_h, width, 3), dtype=np.uint8)
        body[:] = (13, 21, 32)
        plot_top, plot_bottom = 12, max(body_h - 54, 24)
        plot_left, plot_right = 42, max(width - 16, 48)
        cv2.rectangle(body, (plot_left, plot_top), (plot_right, plot_bottom), (42, 58, 76), 1)
        for i in range(1, 4):
            y = plot_top + int((plot_bottom - plot_top) * i / 4)
            cv2.line(body, (plot_left, y), (plot_right, y), (28, 41, 56), 1)
        values = [item.get("values", [0, 0, 0]) for item in history[-160:]]
        flat = [float(v) for row in values for v in row[:3]]
        vmin = min(flat) if flat else 0.0
        vmax = max(flat) if flat else 1.0
        if vmax - vmin < 50:
            mid = (vmax + vmin) / 2.0
            vmin, vmax = mid - 25, mid + 25
        colors = [(82, 82, 255), (255, 141, 91), (161, 242, 69)]
        labels = ["Left", "Right", "Front"]
        if values:
            n = max(len(values) - 1, 1)
            for channel in range(3):
                pts = []
                for idx, row in enumerate(values):
                    x = plot_left + int((plot_right - plot_left) * idx / n)
                    y = plot_bottom - int((plot_bottom - plot_top) * (float(row[channel]) - vmin) / (vmax - vmin))
                    pts.append((x, y))
                for a, b in zip(pts, pts[1:]):
                    cv2.line(body, a, b, colors[channel], 2)
        latest = values[-1] if values else [0, 0, 0]
        step = max(width // 3, 1)
        for i, label in enumerate(labels):
            x = 14 + i * step
            cv2.putText(body, label, (x, body_h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (124, 145, 170), 1)
            cv2.putText(body, str(int(latest[i])), (x + 56, body_h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (222, 235, 247), 1)
        return self._compose_panel(title, body, width, height)

    def _render_monitor_panel(self, width: int, height: int) -> np.ndarray:
        body_h = max(height - 36, 1)
        body = np.zeros((body_h, width, 3), dtype=np.uint8)
        body[:] = (13, 21, 32)
        arm_mgr = self.arm_mgr_getter()
        pose = getattr(arm_mgr, "current_pose_mm_deg", None)
        labels = ["X", "Y", "Z", "RX", "RY", "RZ"]
        values = list(pose[:6]) if pose is not None else [0] * 6
        for i, (label, value) in enumerate(zip(labels, values)):
            y = 32 + i * 28
            cv2.putText(body, label, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (124, 145, 170), 1)
            cv2.putText(body, f"{float(value):.1f}", (80, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (222, 235, 247), 1)
        return self._compose_panel("ARM MONITOR", body, width, height)

    def _render_trajectory_panel(self, history: list[dict[str, Any]], width: int, height: int) -> np.ndarray:
        body_h = max(height - 36, 1)
        body = np.zeros((body_h, width, 3), dtype=np.uint8)
        body[:] = (13, 21, 32)
        pts3 = [item.get("pose", []) for item in history[-300:]]
        pts = [(float(p[0]), float(p[2])) for p in pts3 if len(p) >= 3]
        if len(pts) >= 2:
            xs, zs = zip(*pts)
            xmin, xmax = min(xs), max(xs)
            zmin, zmax = min(zs), max(zs)
            if xmax - xmin < 10:
                xmax += 5
                xmin -= 5
            if zmax - zmin < 10:
                zmax += 5
                zmin -= 5
            left, right, top, bottom = 42, width - 18, 14, body_h - 26
            mapped = []
            for x, z in pts:
                px = left + int((right - left) * (x - xmin) / (xmax - xmin))
                py = bottom - int((bottom - top) * (z - zmin) / (zmax - zmin))
                mapped.append((px, py))
            for idx, (a, b) in enumerate(zip(mapped, mapped[1:])):
                t = idx / max(len(mapped) - 2, 1)
                color = (int(255 * (1 - t)), int(180 * t), int(255 * t))
                cv2.line(body, a, b, color, 2)
            cv2.circle(body, mapped[-1], 5, (255, 255, 255), -1)
        return self._compose_panel("3D TRAJECTORY", body, width, height)

    def _render_logs_panel(self, width: int, height: int) -> np.ndarray:
        body_h = max(height - 36, 1)
        body = np.zeros((body_h, width, 3), dtype=np.uint8)
        body[:] = (5, 9, 13)
        entries = self.log_buffer.snapshot()
        font = cv2.FONT_HERSHEY_SIMPLEX
        line_h = 19
        max_lines = max((body_h - 18) // line_h, 1)
        visible = entries[-max_lines:]
        y = 20
        color_by_level = {
            "ERROR": (82, 82, 255),
            "WARN": (70, 190, 255),
            "STEP": (120, 235, 160),
            "INFO": (220, 220, 220),
            "GRIP": (255, 190, 120),
        }
        for entry in visible:
            level = str(entry.get("level", "INFO")).upper()
            ts = str(entry.get("timestamp", "--:--:--"))
            msg = str(entry.get("message", ""))
            line = f"{ts} [{level}] {msg}"
            max_chars = max(int((width - 22) / 8.4), 12)
            if len(line) > max_chars:
                line = line[: max_chars - 3] + "..."
            cv2.putText(
                body,
                line,
                (12, y),
                font,
                0.45,
                color_by_level.get(level, (205, 218, 230)),
                1,
                cv2.LINE_AA,
            )
            y += line_h
        if not visible:
            cv2.putText(body, "Waiting for dashboard logs...", (12, 32), font, 0.5, (120, 145, 165), 1, cv2.LINE_AA)
        return self._compose_panel("ARM LOGS - TERMINAL", body, width, height)

    def _render_placeholder(self, title: str, width: int, height: int) -> np.ndarray:
        body = np.zeros((max(height - 36, 1), width, 3), dtype=np.uint8)
        body[:] = (13, 21, 32)
        cv2.putText(body, "Frontend DOM panel", (18, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 202, 222), 1)
        cv2.putText(body, "Requires browser/screen recording", (18, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (124, 145, 170), 1)
        cv2.putText(body, "Backend MP4 records live data panels only.", (18, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (90, 112, 138), 1)
        return self._compose_panel(title, body, width, height)

    def _compose_panel(self, title: str, body: np.ndarray, width: int, height: int) -> np.ndarray:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (8, 18, 29)
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (48, 84, 112), 1)
        cv2.rectangle(frame, (1, 1), (width - 2, 35), (17, 34, 48), -1)
        cv2.putText(frame, title[:42], (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 202, 222), 1)
        body_resized = cv2.resize(body, (width, max(height - 36, 1)), interpolation=cv2.INTER_AREA)
        frame[36:height, 0:width] = body_resized[: height - 36, :width]
        return frame

    @staticmethod
    def _even(value: int) -> int:
        return max(2, int(value) - (int(value) % 2))


class ValidationSessionRecorder:
    """Persist one active validation session at a time."""

    def __init__(self, socketio: SocketIO, objects_cfg: dict):
        self.sio = socketio
        self.objects_cfg = objects_cfg
        self.root_dir = PROJECT_ROOT / "data" / "validation_sessions"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._current: Optional[dict[str, Any]] = None
        self._last_completed: Optional[dict[str, Any]] = None

    def default_teach_recording(self, object_key: str) -> str:
        obj_def = self.objects_cfg.get("classes", {}).get(object_key, {})
        return obj_def.get("default_teach_recording", f"{object_key}_pick_v1")

    def state_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active": self._public_session(self._current),
                "last_completed": self._last_completed,
            }

    def emit_state(self) -> None:
        self.sio.emit("validation_state", self.state_payload())

    def _public_session(self, session: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if session is None:
            return None
        return {
            "object_key": session["object_key"],
            "slot_id": session["slot_id"],
            "table_set_id": session["table_set_id"],
            "mode": session["mode"],
            "requested_text": session["requested_text"],
            "resolved_object_key": session["resolved_object_key"],
            "teach_recording_name": session["teach_recording_name"],
            "operator_result": session["operator_result"],
            "start_time": session["start_time"],
            "session_dir": session["session_dir"],
        }

    def _write_session_json(self, session: dict[str, Any]) -> None:
        payload = {
            "object_key": session["object_key"],
            "slot_id": session["slot_id"],
            "table_set_id": session["table_set_id"],
            "mode": session["mode"],
            "ready_pose": session["ready_pose"],
            "pick_pose": session["pick_pose"],
            "teach_recording_name": session["teach_recording_name"],
            "requested_text": session["requested_text"],
            "resolved_object_key": session["resolved_object_key"],
            "operator_result": session["operator_result"],
            "start_time": session["start_time"],
            "end_time": session["end_time"],
        }
        with open(session["session_json_path"], "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _write_jsonl(self, handle, payload: dict[str, Any]) -> None:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()

    def start_session(
        self,
        object_key: str,
        obj_def: dict[str, Any],
        mode: str,
        requested_text: str,
        resolved_object_key: str,
        teach_recording_name: str = "",
    ) -> tuple[bool, str]:
        with self._lock:
            if self._current is not None and self._current["operator_result"] is None:
                return False, "validation session already active; mark success/fail first"

            ts = time.time()
            slug = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
            session_dir = self.root_dir / f"{slug}_{int(ts * 1000) % 1000:03d}_{object_key}"
            session_dir.mkdir(parents=True, exist_ok=True)

            trajectory_path = session_dir / "arm_trajectory.csv"
            arm_logs_path = session_dir / "arm_logs.jsonl"
            speech_events_path = session_dir / "speech_events.jsonl"
            session_json_path = session_dir / "session.json"

            trajectory_file = open(trajectory_path, "w", newline="", encoding="utf-8")
            trajectory_writer = csv.writer(trajectory_file)
            trajectory_writer.writerow([
                "timestamp_unix",
                "x_mm",
                "y_mm",
                "z_mm",
                "rx_deg",
                "ry_deg",
                "rz_deg",
            ])

            arm_logs_file = open(arm_logs_path, "w", encoding="utf-8", buffering=1)
            speech_events_file = open(speech_events_path, "w", encoding="utf-8", buffering=1)

            fixed_poses = obj_def.get("fixed_poses", {})
            session = {
                "object_key": object_key,
                "slot_id": obj_def.get("slot_id", ""),
                "table_set_id": obj_def.get("table_set_id", ""),
                "mode": mode,
                "ready_pose": fixed_poses.get("approach", []),
                "pick_pose": fixed_poses.get("pick", []),
                "teach_recording_name": teach_recording_name,
                "requested_text": requested_text,
                "resolved_object_key": resolved_object_key,
                "operator_result": None,
                "start_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)),
                "end_time": None,
                "session_dir": str(session_dir),
                "session_json_path": session_json_path,
                "trajectory_file": trajectory_file,
                "trajectory_writer": trajectory_writer,
                "arm_logs_file": arm_logs_file,
                "speech_events_file": speech_events_file,
            }
            self._current = session
            self._write_session_json(session)

        self.emit_state()
        return True, str(session_dir)

    def log_arm_pose(self, pose_mm_deg: list[float], timestamp: float) -> None:
        with self._lock:
            session = self._current
            if session is None or session["operator_result"] is not None:
                return
            session["trajectory_writer"].writerow([
                f"{timestamp:.6f}",
                f"{pose_mm_deg[0]:.3f}",
                f"{pose_mm_deg[1]:.3f}",
                f"{pose_mm_deg[2]:.3f}",
                f"{pose_mm_deg[3]:.3f}",
                f"{pose_mm_deg[4]:.3f}",
                f"{pose_mm_deg[5]:.3f}",
            ])
            session["trajectory_file"].flush()

    def log_arm_log(self, level: str, message: str, timestamp: str) -> None:
        with self._lock:
            session = self._current
            if session is None or session["operator_result"] is not None:
                return
            self._write_jsonl(session["arm_logs_file"], {
                "timestamp": timestamp,
                "level": level,
                "message": message,
            })

    def log_speech(self, event_type: str, payload: dict[str, Any]) -> None:
        with self._lock:
            session = self._current
            if session is None or session["operator_result"] is not None:
                return
            self._write_jsonl(session["speech_events_file"], {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                "event_type": event_type,
                "payload": payload,
            })

    def mark_result(self, result: str) -> tuple[bool, str]:
        if result not in {"success", "fail"}:
            return False, "invalid validation result"

        with self._lock:
            session = self._current
            if session is None:
                return False, "no active validation session"

            session["operator_result"] = result
            session["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
            self._write_session_json(session)

            self._last_completed = {
                "object_key": session["object_key"],
                "slot_id": session["slot_id"],
                "table_set_id": session["table_set_id"],
                "mode": session["mode"],
                "requested_text": session["requested_text"],
                "resolved_object_key": session["resolved_object_key"],
                "teach_recording_name": session["teach_recording_name"],
                "operator_result": session["operator_result"],
                "start_time": session["start_time"],
                "end_time": session["end_time"],
                "session_dir": session["session_dir"],
            }

            session["trajectory_file"].close()
            session["arm_logs_file"].close()
            session["speech_events_file"].close()
            self._current = None

        self.emit_state()
        return True, result

    def shutdown(self) -> None:
        with self._lock:
            session = self._current
            if session is None:
                return

            session["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
            if session["operator_result"] is None:
                session["operator_result"] = "aborted"
            self._write_session_json(session)

            for key in ("trajectory_file", "arm_logs_file", "speech_events_file"):
                handle = session.get(key)
                if handle is None:
                    continue
                try:
                    handle.close()
                except Exception:
                    pass
            self._current = None

        self.emit_state()


class GripperMonitor:
    """Poll gripper API state for UI display only."""

    def __init__(
        self,
        demo_cfg: dict,
        socketio: SocketIO,
        digital_twin_sync: "DigitalTwinSync | None" = None,
    ):
        self.demo_cfg = demo_cfg
        self.sio = socketio
        self.digital_twin_sync = digital_twin_sync
        self.connected = False
        self.last_state: dict[str, Any] = {}
        self.last_error = ""
        self.endpoint_label = ""
        self._active_base_url: Optional[str] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _candidate_base_urls(self) -> list[tuple[str, str]]:
        candidates = _gripper_endpoint_candidates(self.demo_cfg)
        if not self._active_base_url:
            return candidates
        ordered = [(self.endpoint_label or "gripper_active", self._active_base_url)]
        ordered.extend((label, base_url) for label, base_url in candidates if base_url != self._active_base_url)
        return ordered

    def _emit_state(self) -> None:
        payload = {
            "connected": self.connected,
            "last_error": self.last_error or None,
            "endpoint_label": self.endpoint_label or None,
            "state": self.last_state,
        }
        self.sio.emit("gripper_state", payload)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def reload_config(self, demo_cfg: dict) -> None:
        was_running = self._running
        self.stop()
        self.demo_cfg = demo_cfg
        self.connected = False
        self.last_state = {}
        self.last_error = ""
        self.endpoint_label = ""
        self._active_base_url = None
        self._thread = None
        self._emit_state()
        if was_running:
            self.start()

    def _poll_loop(self) -> None:
        try:
            import requests
        except ImportError:
            self.connected = False
            self.last_error = "requests not installed"
            self._emit_state()
            return

        while self._running:
            try:
                last_error = "no gripper endpoints configured"
                for label, base_url in self._candidate_base_urls():
                    try:
                        resp = requests.get(f"{base_url}/state", timeout=1.0)
                        resp.raise_for_status()
                        data = resp.json()
                        if not isinstance(data, dict):
                            raise RuntimeError("state response is not a JSON object")
                        self.connected = True
                        self.last_state = data
                        self.last_error = ""
                        self.endpoint_label = label
                        self._active_base_url = base_url
                        break
                    except Exception as exc:
                        last_error = f"{label}: {exc}"
                else:
                    self.connected = False
                    self.endpoint_label = ""
                    self._active_base_url = None
                    self.last_error = last_error
            except Exception as exc:
                self.connected = False
                self.last_error = str(exc)
            self._emit_state()
            if self.connected and self.digital_twin_sync is not None:
                self.digital_twin_sync.queue_gripper_state(self.last_state)
            time.sleep(0.5 if self.connected else 1.5)


class SensorMonitor:
    """Poll claw tactile sensor API and stream short histories to the UI."""

    def __init__(self, demo_cfg: dict, socketio: SocketIO):
        self.demo_cfg = demo_cfg
        self.sio = socketio
        self.cfg = demo_cfg.get("sensor_api", {}) if isinstance(demo_cfg, dict) else {}
        self.enabled = bool(self.cfg.get("enabled", False))
        self.connected = False
        self.last_error = ""
        self.last_sample: dict[str, Any] = {}
        self.last_update_unix = 0.0
        self.history: list[dict[str, Any]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def state_payload(self) -> dict[str, Any]:
        status = "success" if self.connected else "error"
        if isinstance(self.last_sample, dict) and self.last_sample.get("status"):
            status = str(self.last_sample.get("status"))
        return {
            "enabled": self.enabled,
            "connected": self.connected,
            "status": status,
            "last_error": self.last_error or None,
            "last_sample": self.last_sample,
            "last_update_unix": self.last_update_unix or None,
        }

    def _emit_state(self) -> None:
        self.sio.emit("sensor_state", self.state_payload())

    def start(self) -> None:
        self._emit_state()
        if not self.enabled or self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def reload_config(self, demo_cfg: dict) -> None:
        was_running = self._running
        self.stop()
        self.demo_cfg = demo_cfg
        self.cfg = demo_cfg.get("sensor_api", {}) if isinstance(demo_cfg, dict) else {}
        self.enabled = bool(self.cfg.get("enabled", False))
        if not self.enabled:
            self.connected = False
            self.last_error = ""
        self._thread = None
        self._emit_state()
        if was_running:
            self.start()

    def _poll_loop(self) -> None:
        try:
            import requests
        except ImportError:
            self.connected = False
            self.last_error = "requests not installed"
            self._emit_state()
            return

        base_url = str(self.cfg.get("base_url", "")).rstrip("/")
        endpoint = str(self.cfg.get("endpoint", "/get_sensor"))
        poll_hz = max(float(self.cfg.get("poll_hz", 10)), 1.0)
        timeout_s = float(self.cfg.get("request_timeout_s", 1.0))
        history_points = max(int(self.cfg.get("history_points", 180)), 10)

        while self._running:
            if not base_url:
                self.connected = False
                self.last_error = "sensor_api.base_url not configured"
                self._emit_state()
                time.sleep(1.0)
                continue
            try:
                resp = requests.get(f"{base_url}{endpoint}", timeout=timeout_s)
                resp.raise_for_status()
                data = resp.json()
                status = str((data or {}).get("status", "")).lower() if isinstance(data, dict) else ""
                if not isinstance(data, dict) or status not in {"success", "degraded"}:
                    raise RuntimeError((data or {}).get("message", "invalid sensor response"))
                sample = {
                    "timestamp_unix": time.time(),
                    "analog_value1": int(data.get("analog_value1", 0)),
                    "analog_value2": int(data.get("analog_value2", 0)),
                    "analog_value3": int(data.get("analog_value3", 0)),
                    "status": status,
                    "connected": bool(data.get("connected", True)),
                    "channels": data.get("channels", []),
                    "last_error": data.get("last_error"),
                }
                self.connected = True
                self.last_error = ""
                self.last_sample = sample
                self.last_update_unix = sample["timestamp_unix"]
                self.history.append(sample)
                if len(self.history) > history_points:
                    self.history = self.history[-history_points:]
                self.sio.emit("sensor_data", {
                    "sample": sample,
                    "history": self.history,
                })
            except Exception as exc:
                self.connected = False
                self.last_error = str(exc)
            self._emit_state()
            time.sleep(1.0 / poll_hz)


# ---------------------------------------------------------------------------
# Digital Twin Sync
# ---------------------------------------------------------------------------
class DigitalTwinSync:
    """Push local arm/gripper state into the remote digital twin REST API."""

    def __init__(self, demo_cfg: dict, socketio: SocketIO):
        self.demo_cfg = demo_cfg
        self.sio = socketio
        self.virtual_cfg = demo_cfg.get("virtual_env", {}) or {}
        self.sync_cfg = self.virtual_cfg.get("sync", {}) or {}
        self.base_url = str(self.virtual_cfg.get("base_url", "")).rstrip("/")
        self.enabled = bool(
            self.virtual_cfg.get("enabled", False)
            and self.sync_cfg.get("enabled", False)
            and self.base_url
        )
        self.connected = False
        self.last_error = ""
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._pending_arm: Optional[dict[str, Any]] = None
        self._pending_gripper: Optional[dict[str, Any]] = None
        self._last_arm_sent_signature: Optional[tuple[float, float, float, float]] = None
        self._last_gripper_sent_signature: Optional[tuple[tuple[str, float, float, float], ...]] = None
        self._last_arm_sent_at = 0.0
        self._last_gripper_sent_at = 0.0
        self._last_arm_payload: Optional[dict[str, Any]] = None
        self._last_gripper_payload: Optional[dict[str, Any]] = None
        self._last_object_payload: Optional[dict[str, Any]] = None
        self._arm_sync_ok = 0
        self._arm_sync_fail = 0
        self._gripper_sync_ok = 0
        self._gripper_sync_fail = 0
        self._object_sync_ok = 0
        self._object_sync_fail = 0
        self._last_object_sync_at = 0.0
        self._arm_backoff_until = 0.0
        arm_cfg = self.sync_cfg.get("arm", {}) or {}
        self._configured_arm_offsets = list(arm_cfg.get("position_offsets_m", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])
        self._configured_yaw_offset = float(arm_cfg.get("yaw_offset_deg", 0.0))

    def state_payload(self) -> dict[str, Any]:
        arm_cfg = self.sync_cfg.get("arm", {}) or {}
        gripper_cfg = self.sync_cfg.get("gripper", {}) or {}
        return {
            "enabled": self.enabled,
            "connected": self.connected,
            "base_url": self.base_url or None,
            "last_error": self.last_error or None,
            "arm": {
                "enabled": bool(arm_cfg.get("enabled", False)),
                "mode": str(arm_cfg.get("mode", "direct_ee_move")),
                "ok_count": self._arm_sync_ok,
                "fail_count": self._arm_sync_fail,
                "last_sent_at": self._last_arm_sent_at or None,
                "last_payload": self._last_arm_payload,
                "backoff_until": self._arm_backoff_until or None,
                "calibration": {
                    "position_offsets_m": [round(float(v), 6) for v in arm_cfg.get("position_offsets_m", [])],
                    "yaw_offset_deg": round(float(arm_cfg.get("yaw_offset_deg", 0.0)), 4),
                },
            },
            "gripper": {
                "enabled": bool(gripper_cfg.get("enabled", False)),
                "ok_count": self._gripper_sync_ok,
                "fail_count": self._gripper_sync_fail,
                "last_sent_at": self._last_gripper_sent_at or None,
                "last_payload": self._last_gripper_payload,
            },
            "objects": {
                "enabled": bool((self.sync_cfg.get("objects", {}) or {}).get("enabled", False)),
                "ok_count": self._object_sync_ok,
                "fail_count": self._object_sync_fail,
                "last_sent_at": self._last_object_sync_at or None,
                "last_payload": self._last_object_payload,
            },
        }

    def emit_state(self) -> None:
        self.sio.emit("virtual_env_sync", self.state_payload())

    def start(self) -> None:
        self.emit_state()
        if not self.enabled or self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def reload_config(self, demo_cfg: dict) -> None:
        was_running = self._running
        self.stop()
        self.demo_cfg = demo_cfg
        self.virtual_cfg = demo_cfg.get("virtual_env", {}) or {}
        self.sync_cfg = self.virtual_cfg.get("sync", {}) or {}
        self.base_url = str(self.virtual_cfg.get("base_url", "")).rstrip("/")
        self.enabled = bool(
            self.virtual_cfg.get("enabled", False)
            and self.sync_cfg.get("enabled", False)
            and self.base_url
        )
        arm_cfg = self.sync_cfg.get("arm", {}) or {}
        self._configured_arm_offsets = list(arm_cfg.get("position_offsets_m", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])
        self._configured_yaw_offset = float(arm_cfg.get("yaw_offset_deg", 0.0))
        self._pending_arm = None
        self._pending_gripper = None
        self._last_arm_sent_signature = None
        self._last_gripper_sent_signature = None
        self._last_arm_payload = None
        self._last_gripper_payload = None
        self._last_object_payload = None
        self._arm_backoff_until = 0.0
        self._thread = None
        self.emit_state()
        if was_running:
            self.start()

    def queue_arm_pose(self, pose_mm_deg: list[float], pose_raw: Optional[list[int]] = None) -> None:
        payload = self._build_arm_payload(pose_mm_deg, pose_raw)
        if payload is None:
            return
        with self._lock:
            self._pending_arm = payload

    def queue_gripper_state(self, state: dict[str, Any]) -> None:
        payload = self._build_gripper_payload(state)
        if payload is None:
            return
        with self._lock:
            self._pending_gripper = payload

    def _worker_loop(self) -> None:
        try:
            import requests
        except ImportError:
            self.last_error = "requests not installed"
            self.connected = False
            self.emit_state()
            return

        session = requests.Session()
        min_interval = float(self.sync_cfg.get("min_interval_s", 0.35))

        while self._running:
            arm_payload = None
            gripper_payload = None
            now = time.time()

            with self._lock:
                if (
                    self._pending_arm is not None
                    and now - self._last_arm_sent_at >= min_interval
                    and now >= self._arm_backoff_until
                ):
                    arm_payload = self._pending_arm
                    self._pending_arm = None
                if self._pending_gripper is not None and now - self._last_gripper_sent_at >= min_interval:
                    gripper_payload = self._pending_gripper
                    self._pending_gripper = None

            if arm_payload is None and gripper_payload is None:
                time.sleep(0.02)
                continue

            if arm_payload is not None:
                self._push_arm_payload(session, arm_payload)
            if gripper_payload is not None:
                self._push_gripper_payload(session, gripper_payload)

    def _request_timeout(self) -> float:
        return float(self.sync_cfg.get("request_timeout_s", 0.8))

    def _post_json(self, session: Any, path: str, payload: dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        return session.post(url, json=payload, timeout=self._request_timeout())

    def _set_success(self) -> None:
        self.connected = True
        self.last_error = ""

    def _set_failure(self, error: str) -> None:
        self.connected = False
        self.last_error = error

    def _arm_cfg(self) -> dict[str, Any]:
        return self.sync_cfg.get("arm", {}) or {}

    def restore_arm_calibration(self) -> dict[str, Any]:
        arm_cfg = self._arm_cfg()
        arm_cfg["position_offsets_m"] = list(self._configured_arm_offsets)
        arm_cfg["yaw_offset_deg"] = self._configured_yaw_offset
        self._last_arm_sent_signature = None
        self._pending_arm = None
        self.emit_state()
        return {
            "ok": True,
            "position_offsets_m": arm_cfg.get("position_offsets_m", []),
            "yaw_offset_deg": arm_cfg.get("yaw_offset_deg", 0.0),
        }

    def fetch_remote_ee_pose(self) -> dict[str, Any]:
        import requests

        resp = requests.get(
            f"{self.base_url}/ee_pose",
            params={"blocking": "true", "timeout_s": 4},
            timeout=self._request_timeout(),
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError("remote ee_pose response is not a JSON object")
        return data

    def calibrate_from_current_pair(self, local_pose_mm_deg: list[float]) -> dict[str, Any]:
        arm_cfg = self._arm_cfg()
        if str(arm_cfg.get("mode", "direct_ee_move")).strip().lower() != "direct_ee_move":
            raise RuntimeError("current calibration button only applies to direct_ee_move mode")
        calib_cfg = arm_cfg.get("calibration", {}) or {}
        if not calib_cfg.get("enabled", False):
            raise RuntimeError("digital twin arm calibration is disabled")
        if not isinstance(local_pose_mm_deg, list) or len(local_pose_mm_deg) < 6:
            raise RuntimeError("no local arm pose available for calibration")

        remote = self.fetch_remote_ee_pose()
        result = remote.get("result", {}) if isinstance(remote, dict) else {}
        remote_xyz = result.get("ee_xyz_m") or result.get("tool_xyz_m")
        if not isinstance(remote_xyz, list) or len(remote_xyz) < 3:
            raise RuntimeError("remote ee_pose did not return ee_xyz_m")

        scale = float(arm_cfg.get("mm_to_m_scale", 0.001))
        indices = arm_cfg.get("position_indices", [0, 1, 2]) or [0, 1, 2]
        signs = arm_cfg.get("position_signs", [1.0, 1.0, 1.0]) or [1.0, 1.0, 1.0]
        new_offsets = []
        for i in range(3):
            src_idx = int(indices[i]) if i < len(indices) else i
            sign = float(signs[i]) if i < len(signs) else 1.0
            local_value_m = float(local_pose_mm_deg[src_idx]) * scale * sign
            new_offsets.append(float(remote_xyz[i]) - local_value_m)

        yaw_idx = int(arm_cfg.get("yaw_source_index", 5))
        yaw_sign = float(arm_cfg.get("yaw_sign", 1.0))
        local_yaw = float(local_pose_mm_deg[yaw_idx]) * yaw_sign
        remote_yaw = result.get("yaw_deg")
        if remote_yaw is None:
            remote_yaw = 0.0
        new_yaw_offset = float(remote_yaw) - local_yaw

        arm_cfg["position_offsets_m"] = new_offsets
        arm_cfg["yaw_offset_deg"] = new_yaw_offset
        arm_cfg["calibration_last_remote_pose"] = {
            "ee_xyz_m": [float(v) for v in remote_xyz[:3]],
            "yaw_deg": float(remote_yaw),
        }
        arm_cfg["calibration_last_local_pose_mm_deg"] = [
            float(v) for v in local_pose_mm_deg[:6]
        ]
        self._last_arm_sent_signature = None
        self._pending_arm = None
        self.emit_state()
        return {
            "ok": True,
            "remote_pose": arm_cfg["calibration_last_remote_pose"],
            "local_pose_mm_deg": arm_cfg["calibration_last_local_pose_mm_deg"],
            "position_offsets_m": [round(float(v), 6) for v in new_offsets],
            "yaw_offset_deg": round(float(new_yaw_offset), 4),
        }

    def _build_arm_payload(self, pose_mm_deg: list[float], pose_raw: Optional[list[int]] = None) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        arm_cfg = self._arm_cfg()
        if not arm_cfg.get("enabled", False):
            return None
        if not isinstance(pose_mm_deg, list) or len(pose_mm_deg) < 6:
            return None

        mode = str(arm_cfg.get("mode", "direct_ee_move")).strip().lower()
        if mode == "delta_raw":
            units = str(arm_cfg.get("delta_units", "raw_counts")).strip()
            blocking = bool(arm_cfg.get("blocking", False))
            timeout_s = float(arm_cfg.get("timeout_s", 0.25))
            include_ee_pose = bool(arm_cfg.get("include_ee_pose", False))

            if units == "raw_counts" and isinstance(pose_raw, list) and len(pose_raw) >= 6:
                source_pose = [int(v) for v in pose_raw[:6]]
                signature = tuple(source_pose[:6])
                request = {
                    "delta_pose_raw": source_pose,
                    "units": "raw_counts",
                    "blocking": blocking,
                    "timeout_s": timeout_s,
                    "include_ee_pose": include_ee_pose,
                }
            else:
                source_pose = [round(float(v), 3) for v in pose_mm_deg[:6]]
                signature = tuple(source_pose[:6])
                request = {
                    "delta_pose": source_pose,
                    "units": "mm_deg",
                    "blocking": blocking,
                    "timeout_s": timeout_s,
                    "include_ee_pose": include_ee_pose,
                }

            if self._last_arm_sent_signature is not None and self._last_arm_sent_signature == signature:
                return None

            return {
                "path": "/ee_move_delta_raw",
                "request": request,
                "signature": signature,
                "summary": {
                    "mode": mode,
                    "units": request.get("units"),
                    "source_pose": source_pose,
                },
            }

        scale = float(arm_cfg.get("mm_to_m_scale", 0.001))
        indices = arm_cfg.get("position_indices", [0, 1, 2]) or [0, 1, 2]
        signs = arm_cfg.get("position_signs", [1.0, 1.0, 1.0]) or [1.0, 1.0, 1.0]
        offsets = arm_cfg.get("position_offsets_m", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]
        yaw_idx = int(arm_cfg.get("yaw_source_index", 5))
        yaw_sign = float(arm_cfg.get("yaw_sign", 1.0))
        yaw_offset = float(arm_cfg.get("yaw_offset_deg", 0.0))
        delta_m = float(arm_cfg.get("delta_threshold_m", 0.002))
        delta_yaw = float(arm_cfg.get("delta_threshold_yaw_deg", 1.0))

        position = []
        for i in range(3):
            src_idx = int(indices[i]) if i < len(indices) else i
            sign = float(signs[i]) if i < len(signs) else 1.0
            offset = float(offsets[i]) if i < len(offsets) else 0.0
            if src_idx >= len(pose_mm_deg):
                return None
            position.append((float(pose_mm_deg[src_idx]) * scale * sign) + offset)

        yaw_deg = (float(pose_mm_deg[yaw_idx]) * yaw_sign) + yaw_offset
        signature = (
            round(position[0], 5),
            round(position[1], 5),
            round(position[2], 5),
            round(yaw_deg, 2),
        )
        if self._last_arm_sent_signature is not None:
            px, py, pz, pyaw = self._last_arm_sent_signature
            if (
                abs(signature[0] - px) < delta_m
                and abs(signature[1] - py) < delta_m
                and abs(signature[2] - pz) < delta_m
                and abs(signature[3] - pyaw) < delta_yaw
            ):
                return None

        return {
            "path": "/ee_move",
            "request": {
                "position": position,
                "yaw_deg": yaw_deg,
                "blocking": bool(arm_cfg.get("blocking", False)),
                "timeout_s": float(arm_cfg.get("timeout_s", 0.25)),
            },
            "signature": signature,
            "summary": {
                "mode": mode,
                "position_m": [round(v, 5) for v in position],
                "yaw_deg": round(yaw_deg, 2),
                "source_pose_mm_deg": [round(float(v), 3) for v in pose_mm_deg[:6]],
            },
        }

    def _build_gripper_payload(self, state: dict[str, Any]) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        gripper_cfg = self.sync_cfg.get("gripper", {}) or {}
        if not gripper_cfg.get("enabled", False):
            return None
        positions = state.get("current_pos")
        if not isinstance(positions, list) or len(positions) < 3:
            return None

        source_origin = gripper_cfg.get("source_origin", [0, 0, 0]) or [0, 0, 0]
        default_scale = float(gripper_cfg.get("default_scale_m_per_tick", 0.0002))
        requests_payload = []
        signature_rows: list[tuple[str, float, float, float]] = []

        for idx, target_cfg in enumerate(gripper_cfg.get("targets", []) or []):
            target_name = str(target_cfg.get("target", "")).strip()
            if not target_name:
                continue
            source_index = int(target_cfg.get("source_index", idx))
            if source_index >= len(positions):
                continue
            base_position = target_cfg.get("base_position_m", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]
            axis = str(target_cfg.get("axis", "x")).lower()
            axis_index = {"x": 0, "y": 1, "z": 2}.get(axis, 0)
            sign = float(target_cfg.get("sign", 1.0))
            scale = float(target_cfg.get("scale_m_per_tick", default_scale))
            origin = float(target_cfg.get("origin", source_origin[source_index] if source_index < len(source_origin) else 0.0))
            offset = (float(positions[source_index]) - origin) * scale * sign
            pose_position = [
                float(base_position[0]) if len(base_position) > 0 else 0.0,
                float(base_position[1]) if len(base_position) > 1 else 0.0,
                float(base_position[2]) if len(base_position) > 2 else 0.0,
            ]
            pose_position[axis_index] += offset
            request_payload = {
                "target": target_name,
                "position": pose_position,
            }
            if "quat_wxyz" in target_cfg:
                request_payload["quat_wxyz"] = list(target_cfg.get("quat_wxyz") or [])
            else:
                request_payload["yaw_deg"] = float(target_cfg.get("yaw_deg", 0.0))
            requests_payload.append(request_payload)
            signature_rows.append(
                (
                    target_name,
                    round(pose_position[0], 5),
                    round(pose_position[1], 5),
                    round(pose_position[2], 5),
                )
            )

        if not requests_payload:
            return None

        signature = tuple(signature_rows)
        delta_m = float(gripper_cfg.get("delta_threshold_m", 0.001))
        if self._last_gripper_sent_signature is not None and len(self._last_gripper_sent_signature) == len(signature):
            unchanged = True
            for prev, curr in zip(self._last_gripper_sent_signature, signature):
                if prev[0] != curr[0]:
                    unchanged = False
                    break
                if (
                    abs(prev[1] - curr[1]) >= delta_m
                    or abs(prev[2] - curr[2]) >= delta_m
                    or abs(prev[3] - curr[3]) >= delta_m
                ):
                    unchanged = False
                    break
            if unchanged:
                return None

        return {
            "requests": requests_payload,
            "signature": signature,
            "summary": {
                "source_positions": [int(v) for v in positions[:3]],
                "targets": [
                    {
                        "target": req["target"],
                        "position": [round(float(v), 5) for v in req["position"]],
                    }
                    for req in requests_payload
                ],
            },
        }

    def _push_arm_payload(self, session: Any, payload: dict[str, Any]) -> None:
        try:
            resp = self._post_json(session, payload.get("path", "/ee_move"), payload["request"])
            if resp.status_code == 409:
                with self._lock:
                    self._pending_arm = payload
                self._arm_backoff_until = time.time() + float(self.sync_cfg.get("conflict_retry_s", 0.75))
                self.last_error = f"arm sync waiting for remote slot ({payload.get('path', '/ee_move')})"
                self.emit_state()
                return
            resp.raise_for_status()
            self._last_arm_sent_at = time.time()
            self._arm_backoff_until = 0.0
            self._last_arm_sent_signature = payload.get("signature")
            self._last_arm_payload = payload["summary"]
            self._arm_sync_ok += 1
            self._set_success()
        except Exception as exc:
            self._arm_sync_fail += 1
            self._set_failure(f"arm sync failed: {exc}")
        self.emit_state()

    def _push_gripper_payload(self, session: Any, payload: dict[str, Any]) -> None:
        try:
            for req in payload["requests"]:
                resp = self._post_json(session, "/object_pose", req)
                resp.raise_for_status()
            self._last_gripper_sent_at = time.time()
            self._last_gripper_sent_signature = payload.get("signature")
            self._last_gripper_payload = payload["summary"]
            self._gripper_sync_ok += 1
            self._set_success()
        except Exception as exc:
            self._gripper_sync_fail += 1
            self._set_failure(f"gripper sync failed: {exc}")
        self.emit_state()

    def _objects_cfg(self) -> dict[str, Any]:
        return self.sync_cfg.get("objects", {}) or {}

    def _build_object_sync_requests(self, objects_cfg: dict[str, Any]) -> list[dict[str, Any]]:
        obj_sync_cfg = self._objects_cfg()
        if not self.enabled or not obj_sync_cfg.get("enabled", False):
            return []
        if not isinstance(objects_cfg, dict):
            return []

        classes = objects_cfg.get("classes", {}) or {}
        mappings = obj_sync_cfg.get("mappings", {}) or {}
        coordinate_frame = str(obj_sync_cfg.get("coordinate_frame", "delta_xyz_mm"))
        use_pick_pose = bool(obj_sync_cfg.get("use_pick_pose_as_object_pose", True))
        use_pick_rz_as_yaw = bool(obj_sync_cfg.get("use_pick_rz_as_yaw", True))
        default_pick_z_offset_mm = float(obj_sync_cfg.get("default_pick_z_offset_mm", 20.0))
        requests_payload: list[dict[str, Any]] = []

        for local_key, mapping in mappings.items():
            obj_def = classes.get(local_key, {}) or {}
            fixed_poses = obj_def.get("fixed_poses", {}) or {}
            pick_pose = fixed_poses.get("pick")
            if not use_pick_pose or not isinstance(pick_pose, list) or len(pick_pose) < 6:
                continue

            remote_target = ""
            mapping_cfg = mapping if isinstance(mapping, dict) else {"target": mapping}
            remote_target = str(mapping_cfg.get("target", "")).strip()
            if not remote_target:
                continue

            position = mapping_cfg.get("position")
            if isinstance(position, list) and len(position) >= 3:
                request_position = [float(position[0]), float(position[1]), float(position[2])]
                request_frame = str(mapping_cfg.get("coordinate_frame", coordinate_frame))
            else:
                pick_z_offset_mm = float(obj_def.get("pick_z_offset_mm", default_pick_z_offset_mm))
                request_position = [
                    round(float(pick_pose[0]) / 1000.0, 3),
                    round(float(pick_pose[1]) / 1000.0, 3),
                    round((float(pick_pose[2]) / 1000.0) - pick_z_offset_mm, 3),
                ]
                request_frame = coordinate_frame

            request_payload = {
                "target": remote_target,
                "position": request_position,
                "coordinate_frame": request_frame,
            }

            if "quat_wxyz" in mapping_cfg:
                request_payload["quat_wxyz"] = list(mapping_cfg.get("quat_wxyz") or [])
            else:
                yaw_deg = mapping_cfg.get("yaw_deg")
                if yaw_deg is None and use_pick_rz_as_yaw:
                    yaw_deg = float(pick_pose[5]) / 1000.0
                if yaw_deg is not None:
                    request_payload["yaw_deg"] = float(yaw_deg)

            requests_payload.append(request_payload)

        return requests_payload

    def sync_scene_objects(self, objects_cfg: dict[str, Any], *, reason: str = "manual") -> dict[str, Any]:
        obj_sync_cfg = self._objects_cfg()
        if not self.enabled or not obj_sync_cfg.get("enabled", False):
            raise RuntimeError("virtual env scene object sync is disabled")
        requests_payload = self._build_object_sync_requests(objects_cfg)
        if not requests_payload:
            raise RuntimeError("no scene object mappings produced any sync requests")

        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("requests not installed") from exc

        clear_first = bool(obj_sync_cfg.get("clear_before_sync", False))
        with requests.Session() as session:
            if clear_first:
                resp = self._post_json(session, "/clear_object_poses", {})
                resp.raise_for_status()
            for req in requests_payload:
                resp = self._post_json(session, "/object_pose", req)
                resp.raise_for_status()

        self._object_sync_ok += 1
        self._last_object_sync_at = time.time()
        self._last_object_payload = {
            "reason": reason,
            "coordinate_frame": str(obj_sync_cfg.get("coordinate_frame", "delta_xyz_mm")),
            "targets": [
                {
                    "target": req["target"],
                    "position": [round(float(v), 3) for v in req["position"]],
                    "coordinate_frame": req.get("coordinate_frame"),
                    "yaw_deg": req.get("yaw_deg"),
                }
                for req in requests_payload
            ],
        }
        self._set_success()
        self.emit_state()
        return {
            "ok": True,
            "count": len(requests_payload),
            "targets": [req["target"] for req in requests_payload],
            "clear_first": clear_first,
        }


# ---------------------------------------------------------------------------
# Arm Manager
# ---------------------------------------------------------------------------
class ArmManager:
    """Wraps ArmController with safety checks, auto-connect, 2-speed moves."""

    def __init__(
        self,
        demo_cfg: dict,
        socketio: SocketIO,
        validation_recorder: ValidationSessionRecorder | None = None,
        digital_twin_sync: DigitalTwinSync | None = None,
    ):
        self.demo_cfg = demo_cfg
        self.sio = socketio
        self.validation_recorder = validation_recorder
        self.digital_twin_sync = digital_twin_sync
        self.ctrl: Any = None
        self.connected = False
        self.connection_label = ""
        self.current_pose_mm_deg: Optional[list] = None
        self._poll_thread: Optional[threading.Thread] = None
        self._polling = False
        self._lock = threading.Lock()
        self._last_zero_pose_warn_ts = 0.0
        self._gripper_base_url: Optional[str] = None
        self._gripper_label = ""
        self._last_gripper_positions: Optional[list[int]] = None
        self._abort_event = threading.Event()

    def _emit_log(self, level: str, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.sio.emit("arm_log", {"level": level, "message": msg, "timestamp": ts})
        log.info("[ARM-%s] %s", level, msg)
        if self.validation_recorder is not None:
            self.validation_recorder.log_arm_log(level, msg, ts)

    def auto_connect(self) -> bool:
        """Try each connection in config order."""
        try:
            from src.controller import ArmController
        except ImportError:
            self._emit_log("ERROR", "pyModbusTCP not installed")
            return False

        connections = self.demo_cfg.get("arm", {}).get("connections", [])
        unit_id = self.demo_cfg.get("arm", {}).get("unit_id", 2)

        for conn in connections:
            host = conn["host"]
            port = conn["port"]
            label = conn.get("label", f"{host}:{port}")
            try:
                s = socket.socket()
                s.settimeout(1.5)
                s.connect((host, port))
                s.close()
            except Exception:
                self._emit_log("WARN", f"TCP probe failed: {label}")
                continue

            try:
                self.ctrl = ArmController(host=host, port=port, unit_id=unit_id)
                if self.ctrl.connect():
                    self.connected = True
                    self.connection_label = label
                    self._emit_log("STEP", f"Arm connected via {label}")
                    self.sio.emit("arm_status",
                                  {"connected": True, "label": label, "error": None})
                    return True
            except Exception as e:
                self._emit_log("WARN", f"Modbus connect failed ({label}): {e}")

        self._emit_log("ERROR", "All arm connections failed")
        self.sio.emit("arm_status",
                      {"connected": False, "label": "", "error": "All connections failed"})
        return False

    def disconnect(self):
        if self.ctrl:
            try:
                self.ctrl.disconnect()
            except Exception:
                pass
        self.connected = False
        self.ctrl = None

    def clear_abort(self) -> None:
        self._abort_event.clear()

    def request_abort(self) -> None:
        self._abort_event.set()

    def _raise_if_aborted(self) -> None:
        if self._abort_event.is_set():
            raise RuntimeError("operation aborted by emergency stop")

    def _power_down_servo(self, *, reason: str = "") -> None:
        """Best-effort servo OFF with a couple retries for flaky controllers."""
        if not self.connected or not self.ctrl:
            return
        last_error = None
        for _ in range(3):
            try:
                self.ctrl.servo_off()
                last_error = None
                time.sleep(0.15)
            except Exception as exc:
                last_error = exc
                time.sleep(0.1)
        if last_error is not None:
            msg = f"Servo OFF failed{': ' + reason if reason else ''}: {last_error}"
            self._emit_log("WARN", msg)

    def motion_stop(self) -> None:
        if not self.connected or not self.ctrl:
            return
        try:
            self.ctrl.motion_stop()
            self._emit_log("ERROR", "Motion STOP command sent")
        except Exception as exc:
            self._emit_log("WARN", f"Motion STOP failed: {exc}")

    def start_pose_polling(self):
        if self._polling:
            return
        self._polling = True
        hz = self.demo_cfg.get("arm", {}).get("pose_poll_hz", 4)
        self._poll_thread = threading.Thread(
            target=self._pose_poll_loop, args=(hz,), daemon=True)
        self._poll_thread.start()

    def stop_pose_polling(self):
        self._polling = False

    def _pose_poll_loop(self, hz: float):
        interval = 1.0 / max(hz, 1)
        while self._polling:
            if self.connected and self.ctrl:
                try:
                    raw = self.ctrl.read_current_pose()
                    if raw and len(raw) >= 6:
                        if abs(int(raw[0])) + abs(int(raw[1])) + abs(int(raw[2])) == 0:
                            now = time.time()
                            if now - self._last_zero_pose_warn_ts > 2.0:
                                self._last_zero_pose_warn_ts = now
                                self._emit_log("WARN", "Ignoring invalid all-zero arm pose from Modbus")
                            time.sleep(interval)
                            continue
                        mm_deg = [v / 1000.0 for v in raw]
                        with self._lock:
                            self.current_pose_mm_deg = mm_deg
                        self.sio.emit("arm_pose", {
                            "pose_mm_deg": mm_deg,
                            "raw": raw,
                            "timestamp": time.time(),
                        })
                        if self.digital_twin_sync is not None:
                            self.digital_twin_sync.queue_arm_pose(mm_deg, raw)
                        if self.validation_recorder is not None:
                            self.validation_recorder.log_arm_pose(mm_deg, time.time())
                except Exception:
                    pass
            time.sleep(interval)

    def check_safety(self, pose: list) -> tuple[bool, str]:
        """Check if pose is within safety boundary. Returns (safe, reason)."""
        limits = _effective_pose_limits(self.demo_cfg)
        labels = ["x", "y", "z"]
        if bool(self.demo_cfg.get("pose_limits", {}).get("enforce_rotation_limits", False)):
            labels.extend(["rx", "ry"])
        for i, axis in enumerate(labels):
            if i >= len(pose):
                break
            val = pose[i]
            lo = limits.get(f"{axis}_min", -999999999)
            hi = limits.get(f"{axis}_max", 999999999)
            if val < lo or val > hi:
                return False, f"{axis.upper()}={val} outside [{lo}, {hi}]"
        return True, ""

    def _ready_pose(self) -> list[int]:
        pose = self.demo_cfg.get("ready_pose", [490127, 0, 425027, 179999, 0, 0])
        return [int(v) for v in pose[:6]]

    def _move_ready_pose(
        self,
        speed: int | None = None,
        wait_seconds: float | None = None,
        label: str = "Ready",
    ) -> None:
        ready_pose = self._ready_pose()
        safe, reason = self.check_safety(ready_pose)
        if not safe:
            raise RuntimeError(f"Ready pose rejected by safety check: {reason}")
        ready_cfg = self.demo_cfg.get("ready_motion", {}) if isinstance(self.demo_cfg, dict) else {}
        if speed is None:
            speed = int(ready_cfg.get("speed_percent", self.demo_cfg.get("speed", {}).get("fast_percent", 20)))
        if wait_seconds is None:
            wait_seconds = float(ready_cfg.get("verify_timeout_s", 20.0))
        self._emit_log("STEP", f"{label} speed={speed}%")
        self.ctrl.move_to(ready_pose, speed=speed, wait=False)
        if bool(ready_cfg.get("verify_in_position", True)):
            self.ctrl.wait_until_in_position(
                timeout_s=wait_seconds,
                fallback_wait_s=wait_seconds,
                label=label.lower().replace(" ", "_"),
            )
        else:
            self.ctrl._wait_with_live_output(label.lower().replace(" ", "_"), wait_seconds)

    def _home_route_cfg(self) -> dict[str, Any]:
        return self.demo_cfg.get("home_routing", {}) if isinstance(self.demo_cfg, dict) else {}

    def _execute_home_motion(self) -> None:
        route_cfg = self._home_route_cfg()
        strategy = str(route_cfg.get("default_strategy", "native_1405")).strip().lower()
        staged_route = route_cfg.get("staged_route", [])
        speed_percent = int(route_cfg.get("speed_percent", 20))
        wait_seconds = float(route_cfg.get("wait_seconds", 20.0))
        verify_timeout_s = float(route_cfg.get("verify_timeout_s", max(wait_seconds, 20.0)))

        if strategy == "staged_route" and route_cfg.get("allow_staged_route", True):
            valid_route = isinstance(staged_route, list) and bool(staged_route)
            if valid_route:
                for idx, item in enumerate(staged_route):
                    if not isinstance(item, dict):
                        continue
                    pose = item.get("pose")
                    if not isinstance(pose, list) or len(pose) < 6:
                        continue
                    step_label = str(item.get("label", f"Home Step {idx + 1}"))
                    step_speed = int(item.get("speed_percent", speed_percent))
                    step_wait = float(item.get("wait_seconds", wait_seconds))
                    self._move_phase_pose(step_label, [int(v) for v in pose[:6]], step_speed, step_wait)
                return
            self._emit_log("WARN", "Configured staged home route is empty; fallback to native 1405")

        self._emit_log("STEP", "Home (native 1405)")
        self.ctrl.go_home_native(
            wait=True,
            wait_seconds=wait_seconds,
            verify_timeout_s=verify_timeout_s,
        )

    def home(self) -> None:
        self.clear_abort()
        if not self.connected and not self.auto_connect():
            return
        try:
            self._raise_if_aborted()
            self._emit_log("STEP", "Reset alarms")
            self.ctrl.reset_alarms()
            self._raise_if_aborted()
            self._emit_log("STEP", "Servo ON")
            self.ctrl.servo_on()
            self._raise_if_aborted()
            self._emit_log("STEP", "Home command")
            self._execute_home_motion()
            self._emit_log("STEP", "Home complete")
        finally:
            self._emit_log("STEP", "Servo OFF")
            self._power_down_servo(reason="home")

    def ready(self) -> None:
        self.clear_abort()
        if not self.connected and not self.auto_connect():
            return
        try:
            self._raise_if_aborted()
            self._emit_log("STEP", "Reset alarms")
            self.ctrl.reset_alarms()
            self._raise_if_aborted()
            self._emit_log("STEP", "Servo ON")
            self.ctrl.servo_on()
            self._raise_if_aborted()
            self._move_ready_pose(label="Ready")
            self._emit_log("STEP", "Ready complete")
        finally:
            self._emit_log("STEP", "Servo OFF")
            self._power_down_servo(reason="ready")

    def _execute_return_pose(self, return_pose: str, speed: int, wait_seconds: float) -> None:
        target = str(return_pose or "").strip().lower()
        if target == "home":
            self._execute_home_motion()
        elif target == "ready":
            self._move_ready_pose(speed=speed, wait_seconds=wait_seconds, label="Return Ready")

    def _gripper_candidates(self) -> list[tuple[str, str]]:
        candidates = _gripper_endpoint_candidates(self.demo_cfg)
        if not self._gripper_base_url:
            return candidates
        ordered = [(self._gripper_label or "gripper_active", self._gripper_base_url)]
        ordered.extend((label, base_url) for label, base_url in candidates if base_url != self._gripper_base_url)
        return ordered

    def _request_gripper(
        self,
        method: str,
        path: str,
        *,
        payload: Optional[dict[str, Any]] = None,
        timeout: float = 1.0,
    ):
        import requests

        last_error = "no gripper endpoints configured"
        for label, base_url in self._gripper_candidates():
            try:
                if method == "GET":
                    resp = requests.get(f"{base_url}{path}", timeout=timeout)
                else:
                    resp = requests.post(f"{base_url}{path}", json=payload, timeout=timeout)
                resp.raise_for_status()
                self._gripper_base_url = base_url
                self._gripper_label = label
                return resp
            except Exception as exc:
                last_error = f"{label}: {exc}"
                if base_url == self._gripper_base_url:
                    self._gripper_base_url = None
                    self._gripper_label = ""
        raise RuntimeError(last_error)

    def pick_fixed(self, object_key: str, obj_def: dict):
        """Execute pick sequence with fixed poses and 2-speed approach."""
        self.clear_abort()
        poses = obj_def.get("fixed_poses", {})
        approach = poses.get("approach")
        pick = poses.get("pick")
        if not approach or not pick:
            self._emit_log("ERROR", f"No fixed_poses defined for {object_key}")
            return

        for label, pose in [("approach", approach), ("pick", pick)]:
            safe, reason = self.check_safety(pose)
            if not safe:
                self._emit_log("ERROR", f"Safety reject ({label}): {reason}")
                return

        speed_cfg = self.demo_cfg.get("speed", {})
        fast = speed_cfg.get("fast_percent", 50)
        slow = speed_cfg.get("slow_percent", 10)
        seq_cfg = self.demo_cfg.get("pick_sequence", {})
        settle = seq_cfg.get("settle_wait_s", 2.0)
        return_pose = str(seq_cfg.get("return_pose", "ready")).strip().lower()

        if not self.connected:
            if not self.auto_connect():
                return

        try:
            self._raise_if_aborted()
            self._emit_log("STEP", "Reset alarms")
            self.ctrl.reset_alarms()
            self.sio.emit("pick_progress", {"step": 1, "total": 8, "name": "reset_alarms"})

            self._raise_if_aborted()
            self._emit_log("STEP", "Servo ON")
            self.ctrl.servo_on()
            self.sio.emit("pick_progress", {"step": 2, "total": 8, "name": "servo_on"})

            if seq_cfg.get("home_before", True):
                self._raise_if_aborted()
                self._emit_log("STEP", "Home before task")
                self._execute_home_motion()
                self.sio.emit("pick_progress", {"step": 3, "total": 8, "name": "home"})

            self._raise_if_aborted()
            self._emit_log("STEP", f"Fast approach -> {object_key} ({fast}%)")
            self.ctrl.move_to(approach, speed=fast, wait_seconds=settle)
            self.sio.emit("pick_progress", {"step": 4, "total": 8, "name": "approach"})

            self._raise_if_aborted()
            self._emit_log("STEP", f"Slow descend ({slow}%)")
            self.ctrl.move_to(pick, speed=slow, wait_seconds=settle)
            self.sio.emit("pick_progress", {"step": 5, "total": 8, "name": "descend"})

            self._raise_if_aborted()
            self._gripper_close()
            self.sio.emit("pick_progress", {"step": 6, "total": 8, "name": "grip"})

            self._raise_if_aborted()
            self._emit_log("STEP", "Lift")
            self.ctrl.move_to(approach, speed=fast // 2, wait_seconds=settle)
            self.sio.emit("pick_progress", {"step": 7, "total": 8, "name": "lift"})

            if seq_cfg.get("home_after", False):
                self._raise_if_aborted()
                self._execute_home_motion()
            else:
                self._raise_if_aborted()
                self._execute_return_pose(return_pose, max(fast // 2, 10), settle)

            self.sio.emit("pick_progress", {"step": 8, "total": 8, "name": "done"})
            self._emit_log("STEP", f"Pick complete: {object_key}")

        except Exception as e:
            self._emit_log("ERROR", f"Pick failed: {e}")
        finally:
            self._emit_log("STEP", "Servo OFF")
            self._power_down_servo(reason=f"pick_fixed {object_key}")

    def _gripper_close(self):
        """Send gripper close via AGX HTTP API."""
        self._gripper_action(
            self.demo_cfg.get("gripper", {}).get("close_command", "c"),
            self.demo_cfg.get("gripper", {}).get("close_delay_s", 1.5),
            "close",
        )

    def gripper_open(self):
        """Send gripper open via AGX HTTP API."""
        self._gripper_action(
            self.demo_cfg.get("gripper", {}).get("open_command", "o"),
            self.demo_cfg.get("gripper", {}).get("open_delay_s", 0.5),
            "open",
        )

    def gripper_stop(self):
        """Stop gripper server loop / disable gripper motors if supported."""
        grip_cfg = self.demo_cfg.get("gripper", {})
        if not grip_cfg.get("enabled", False):
            return
        self._emit_log("GRIP", "Gripper emergency stop")
        try:
            self._request_gripper("GET", "/stop", timeout=2.0)
        except Exception as e:
            self._emit_log("WARN", f"Gripper stop failed: {e}")

    def _gripper_action(self, action: str, delay_s: float = 0.0, label: str | None = None):
        """Send an arbitrary gripper API command."""
        grip_cfg = self.demo_cfg.get("gripper", {})
        if not grip_cfg.get("enabled", False):
            return
        display = label or action
        self._emit_log("GRIP", f"Gripper action {display} (cmd={action})")
        try:
            self._request_gripper("POST", "/command", payload={"action": action}, timeout=2.0)
        except Exception as e:
            self._emit_log("WARN", f"Gripper HTTP failed: {e}")
        if delay_s > 0:
            time.sleep(delay_s)

    def get_gripper_state(self, silent: bool = False) -> Optional[dict[str, Any]]:
        """Fetch current gripper state for teach snapshots and debug."""
        grip_cfg = self.demo_cfg.get("gripper", {})
        if not grip_cfg.get("enabled", False):
            return None
        try:
            resp = self._request_gripper("GET", "/state", timeout=1.0)
            data = resp.json()
            if not isinstance(data, dict):
                return None
            current_pos = data.get("current_pos")
            if isinstance(current_pos, list) and len(current_pos) == 3:
                try:
                    self._last_gripper_positions = [int(float(v)) for v in current_pos]
                except (TypeError, ValueError):
                    pass
            return data
        except Exception as e:
            if not silent:
                self._emit_log("WARN", f"Gripper state read failed: {e}")
            return None

    def _current_gripper_positions(self) -> Optional[list[int]]:
        state = self.get_gripper_state(silent=True)
        if isinstance(state, dict):
            current_pos = state.get("current_pos")
            if isinstance(current_pos, list) and len(current_pos) == 3:
                try:
                    parsed = [int(float(v)) for v in current_pos]
                    self._last_gripper_positions = parsed
                    return parsed
                except (TypeError, ValueError):
                    pass
        if isinstance(self._last_gripper_positions, list) and len(self._last_gripper_positions) == 3:
            return list(self._last_gripper_positions)
        return None

    def _gripper_set_position(
        self,
        positions: list[int],
        delay_s: float = 0.0,
        *,
        move_mode: str = "direct",
        step_ticks: int = 0,
        step_delay_s: float = 0.0,
    ) -> bool:
        """Replay gripper by absolute motor positions instead of symbolic commands."""
        grip_cfg = self.demo_cfg.get("gripper", {})
        if not grip_cfg.get("enabled", False):
            return False
        try:
            target = [int(v) for v in positions]
            mode = str(move_mode or "direct").strip().lower()
            if mode == "stepped" and int(step_ticks) > 0:
                current = self._current_gripper_positions()
                if current is not None:
                    step = max(int(step_ticks), 1)
                    delay = max(float(step_delay_s), 0.0)
                    self._emit_log("GRIP", f"Gripper stepped set_position -> {target} (step={step}, delay={delay:.3f}s)")
                    max_steps = max(1, int(max(abs(t - c) for c, t in zip(current, target)) / step) + 2)
                    for _ in range(max_steps):
                        self._raise_if_aborted()
                        next_pos = []
                        for cur, dst in zip(current, target):
                            delta = dst - cur
                            if abs(delta) <= step:
                                next_pos.append(dst)
                            else:
                                next_pos.append(cur + step if delta > 0 else cur - step)
                        self._request_gripper(
                            "POST",
                            "/set_position",
                            payload={"positions": next_pos},
                            timeout=2.0,
                        )
                        self._last_gripper_positions = list(next_pos)
                        current = next_pos
                        if current == target:
                            break
                        if delay > 0:
                            self._sleep_abortable(delay)
                else:
                    self._emit_log("GRIP", f"Gripper set_position -> {target} (direct fallback; current_pos unavailable)")
                    self._raise_if_aborted()
                    self._request_gripper(
                        "POST",
                        "/set_position",
                        payload={"positions": target},
                        timeout=2.0,
                    )
                    self._last_gripper_positions = list(target)
            else:
                self._emit_log("GRIP", f"Gripper set_position -> {target}")
                self._raise_if_aborted()
                self._request_gripper(
                    "POST",
                    "/set_position",
                    payload={"positions": target},
                    timeout=2.0,
                )
                self._last_gripper_positions = list(target)
            if delay_s > 0:
                self._sleep_abortable(delay_s)
            return True
        except Exception as e:
            if self._abort_event.is_set():
                raise
            self._emit_log("WARN", f"Gripper set_position failed: {e}")
            return False

    def _wait_for_gripper_position(
        self,
        target: list[int],
        *,
        timeout_s: float,
        tolerance_ticks: int,
        label: str,
    ) -> bool:
        deadline = time.time() + max(0.0, timeout_s)
        target = [int(v) for v in target]
        tolerance = max(0, int(tolerance_ticks))
        last_pos: Optional[list[int]] = None
        while time.time() < deadline:
            self._raise_if_aborted()
            state = self.get_gripper_state(silent=True)
            current = state.get("current_pos") if isinstance(state, dict) else None
            if isinstance(current, list) and len(current) == 3:
                try:
                    parsed = [int(float(v)) for v in current]
                except (TypeError, ValueError):
                    parsed = []
                if len(parsed) == 3:
                    last_pos = parsed
                    self._last_gripper_positions = parsed
                    if max(abs(a - b) for a, b in zip(parsed, target)) <= tolerance:
                        self._emit_log("GRIP", f"{label} reached {parsed}")
                        return True
            time.sleep(0.05)
        self._emit_log(
            "WARN",
            f"{label} did not reach target {target} within {timeout_s:.1f}s; last={last_pos}",
        )
        return False

    def _positions_from_waypoint(self, wp: dict[str, Any]) -> Optional[list[int]]:
        direct = wp.get("gripper_pos")
        if isinstance(direct, list) and len(direct) == 3:
            try:
                return [int(v) for v in direct]
            except (TypeError, ValueError):
                return None

        matched = wp.get("matched_external")
        if isinstance(matched, dict):
            row = matched.get("row")
            if isinstance(row, dict):
                values = [row.get("pos1"), row.get("pos2"), row.get("pos3")]
                if all(v is not None for v in values):
                    try:
                        return [int(float(v)) for v in values]
                    except (TypeError, ValueError):
                        return None
        return None

    def _move_phase_pose(
        self,
        label: str,
        pose: list[int],
        speed: int,
        settle_s: float,
        move_timeout_s: float,
        strategy: str = "direct",
    ) -> None:
        strategy = str(strategy or "direct").strip().lower()
        if strategy in {"axis_split", "xy_z_split", "phase_axis_split"}:
            self._move_phase_pose_axis_split(label, pose, speed, settle_s, move_timeout_s)
            return
        safe, reason = self.check_safety(pose)
        if not safe:
            raise RuntimeError(f"Safety reject ({label}): {reason}")
        self._emit_log("STEP", f"{label} speed={speed}%")
        self.ctrl.move_to(pose, speed=speed, wait=False)
        self._wait_until_in_position_abortable(
            timeout_s=move_timeout_s,
            fallback_wait_s=move_timeout_s,
            label=label.lower().replace(" ", "_"),
        )
        if settle_s > 0:
            self._sleep_abortable(settle_s)

    def _current_raw_pose(self) -> Optional[list[int]]:
        if not self.connected or not self.ctrl:
            return None
        try:
            raw = self.ctrl.read_current_pose()
        except Exception:
            raw = None
        if isinstance(raw, list) and len(raw) >= 6:
            try:
                parsed = [int(v) for v in raw[:6]]
            except (TypeError, ValueError):
                parsed = []
            if len(parsed) == 6 and (abs(parsed[0]) + abs(parsed[1]) + abs(parsed[2])) > 0:
                return parsed
        return None

    def _move_phase_pose_axis_split(
        self,
        label: str,
        pose: list[int],
        speed: int,
        settle_s: float,
        move_timeout_s: float,
    ) -> None:
        target = [int(v) for v in pose[:6]]
        current = self._current_raw_pose()
        if not current or len(current) < 6:
            self._emit_log("WARN", f"{label}: current pose unavailable, fallback to direct phase move")
            self._move_phase_pose(label, target, speed, settle_s, move_timeout_s, strategy="direct")
            return

        xy_rot_pose = [
            target[0],
            target[1],
            current[2],
            target[3],
            target[4],
            target[5],
        ]
        need_xy_rot = any(current[idx] != xy_rot_pose[idx] for idx in (0, 1, 3, 4, 5))
        need_z = current[2] != target[2]

        if need_xy_rot:
            safe, reason = self.check_safety(xy_rot_pose)
            if not safe:
                raise RuntimeError(f"Safety reject ({label} XY/ROT): {reason}")
            self._emit_log("STEP", f"{label} XY/ROT speed={speed}%")
            self.ctrl.move_to(xy_rot_pose, speed=speed, wait=False)
            self._wait_until_in_position_abortable(
                timeout_s=move_timeout_s,
                fallback_wait_s=move_timeout_s,
                label=label.lower().replace(" ", "_") + "_xyrot",
            )
            if settle_s > 0:
                self._sleep_abortable(settle_s)

        if need_z:
            safe, reason = self.check_safety(target)
            if not safe:
                raise RuntimeError(f"Safety reject ({label} Z): {reason}")
            self._emit_log("STEP", f"{label} Z speed={speed}%")
            self.ctrl.move_to(target, speed=speed, wait=False)
            self._wait_until_in_position_abortable(
                timeout_s=move_timeout_s,
                fallback_wait_s=move_timeout_s,
                label=label.lower().replace(" ", "_") + "_z",
            )
            if settle_s > 0:
                self._sleep_abortable(settle_s)

        if not need_xy_rot and not need_z:
            self._emit_log("STEP", f"{label} already in place")
            if settle_s > 0:
                self._sleep_abortable(settle_s)

    def _sleep_abortable(self, duration_s: float, interval_s: float = 0.05) -> None:
        deadline = time.time() + max(0.0, duration_s)
        while time.time() < deadline:
            self._raise_if_aborted()
            time.sleep(min(interval_s, max(0.0, deadline - time.time())))

    def _wait_until_in_position_abortable(
        self,
        timeout_s: float,
        fallback_wait_s: float,
        label: str = "move",
    ) -> None:
        """Wait for in-position while letting emergency stop break the replay thread."""
        flag_reg = getattr(self.ctrl, "regs", {}).get("in_position_flag")
        start = time.time()
        print_interval_s = 0.5
        poll_interval_s = 0.1
        last_print_s = -print_interval_s

        while time.time() - start < max(timeout_s, poll_interval_s):
            self._raise_if_aborted()
            elapsed = time.time() - start
            flag = self.ctrl.read_register(flag_reg) if flag_reg is not None else None
            if elapsed - last_print_s >= print_interval_s:
                pose = self.ctrl.read_current_pose_mm_deg()
                if flag_reg is None:
                    print(
                        f"  [{label}] {elapsed:.1f}/{timeout_s:.1f}s "
                        f"pose(mm/deg)={[round(v, 1) for v in pose]}"
                    )
                else:
                    print(
                        f"  [{label}] {elapsed:.1f}/{timeout_s:.1f}s "
                        f"inpos={flag} pose(mm/deg)={[round(v, 1) for v in pose]}"
                    )
                last_print_s = elapsed
            if flag == 1:
                return
            time.sleep(poll_interval_s)

        fallback_deadline = time.time() + min(fallback_wait_s, 2.0)
        while time.time() < fallback_deadline:
            self._raise_if_aborted()
            elapsed = time.time() - fallback_deadline + min(fallback_wait_s, 2.0)
            if elapsed - last_print_s >= print_interval_s:
                pose = self.ctrl.read_current_pose_mm_deg()
                print(
                    f"  [{label}] fallback pose(mm/deg)="
                    f"{[round(v, 1) for v in pose]}"
                )
                last_print_s = elapsed
            time.sleep(poll_interval_s)

    def _apply_gripper_phase_state(
        self,
        grip_cfg: dict[str, Any],
        state_key: str,
        *,
        fallback_command: str,
        label: str,
    ) -> None:
        mode = str(grip_cfg.get(f"{state_key}_mode", "command"))
        positions = grip_cfg.get(f"{state_key}_positions")
        if mode == "positions" and isinstance(positions, list) and len(positions) == 3:
            global_grip_cfg = self.demo_cfg.get("gripper", {})
            move_mode = str(
                grip_cfg.get(
                    "position_move_mode",
                    global_grip_cfg.get("position_move_mode", "direct"),
                )
            )
            step_ticks = int(
                grip_cfg.get(
                    f"{state_key}_step_ticks",
                    grip_cfg.get(
                        "step_ticks",
                        global_grip_cfg.get(
                            f"{state_key}_step_ticks",
                            global_grip_cfg.get("step_ticks", 0),
                        ),
                    ),
                )
            )
            step_delay_s = float(
                grip_cfg.get(
                    f"{state_key}_step_delay_s",
                    grip_cfg.get(
                        "step_delay_s",
                        global_grip_cfg.get(
                            f"{state_key}_step_delay_s",
                            global_grip_cfg.get("step_delay_s", 0.0),
                        ),
                    ),
                )
            )
            self._gripper_set_position(
                [int(v) for v in positions],
                move_mode=move_mode,
                step_ticks=step_ticks,
                step_delay_s=step_delay_s,
            )
            wait_s = float(grip_cfg.get(f"{state_key}_wait_s", 0.0))
            if wait_s > 0:
                tolerance = int(grip_cfg.get(f"{state_key}_tolerance_ticks", grip_cfg.get("tolerance_ticks", 30)))
                self._wait_for_gripper_position(
                    [int(v) for v in positions],
                    timeout_s=wait_s,
                    tolerance_ticks=tolerance,
                    label=f"phase_{state_key}",
                )
            return
        command = str(
            grip_cfg.get(
                f"{state_key}_command",
                self.demo_cfg.get("gripper", {}).get(fallback_command, "c" if state_key == "close" else "o"),
            )
        )
        self._gripper_action(command, delay_s=0.0, label=label)

    def replay_phase_recording(
        self,
        recording: dict[str, Any],
        phase_spec: dict[str, Any],
        move_strategy: str = "direct",
    ) -> None:
        self.clear_abort()
        if not phase_spec:
            self._emit_log("ERROR", "Phase replay requested but phase spec is missing")
            return

        poses = {
            "ready": phase_spec.get("ready_pose"),
            "hover": phase_spec.get("hover_pose"),
            "pregrasp": phase_spec.get("pregrasp_pose"),
            "grasp": phase_spec.get("grasp_pose"),
            "lift": phase_spec.get("lift_pose"),
        }
        for label, pose in poses.items():
            if not isinstance(pose, list) or len(pose) < 6:
                self._emit_log("ERROR", f"Phase replay missing pose: {label}")
                return

        speed_cfg = phase_spec.get("speed", {}) if isinstance(phase_spec, dict) else {}
        timing_cfg = phase_spec.get("timing", {}) if isinstance(phase_spec, dict) else {}
        grip_cfg = phase_spec.get("gripper", {}) if isinstance(phase_spec, dict) else {}
        fast = int(speed_cfg.get("fast_percent", 70))
        slow = int(speed_cfg.get("slow_percent", 25))
        pregrasp_speed = int(speed_cfg.get("pregrasp_percent", slow))
        grasp_speed = int(speed_cfg.get("grasp_percent", slow))
        lift_speed = int(speed_cfg.get("lift_percent", max(fast, 40)))
        place_speed = int(speed_cfg.get("place_percent", lift_speed))
        place_descend_speed = int(speed_cfg.get("place_descend_percent", slow))
        settle_s = float(timing_cfg.get("settle_s", 0.4))
        grip_hold_s = float(timing_cfg.get("grip_hold_s", 0.8))
        release_hold_s = float(timing_cfg.get("release_hold_s", grip_hold_s))
        prepare_open_s = float(timing_cfg.get("prepare_open_s", 0.0))
        move_timeout_s = float(timing_cfg.get("move_timeout_s", max(8.0, settle_s + 8.0)))
        return_pose = str(phase_spec.get("return_pose", "ready")).lower()
        move_strategy = str(move_strategy or phase_spec.get("move_strategy", "direct")).strip().lower()
        place_hover_pose = phase_spec.get("place_hover_pose")
        place_pose = phase_spec.get("place_pose")
        place_lift_pose = phase_spec.get("place_lift_pose", place_hover_pose)
        has_place_phase = isinstance(place_hover_pose, list) or isinstance(place_pose, list)
        if has_place_phase:
            if not isinstance(place_hover_pose, list) or len(place_hover_pose) < 6:
                self._emit_log("ERROR", "Phase replay missing pose: place_hover")
                return
            if not isinstance(place_pose, list) or len(place_pose) < 6:
                self._emit_log("ERROR", "Phase replay missing pose: place")
                return
            if not isinstance(place_lift_pose, list) or len(place_lift_pose) < 6:
                self._emit_log("ERROR", "Phase replay missing pose: place_lift")
                return

        if not self.connected:
            if not self.auto_connect():
                return

        try:
            self._emit_log(
                "STEP",
                f"Phase speeds: fast={fast}% pregrasp={pregrasp_speed}% grasp={grasp_speed}% lift={lift_speed}%"
            )
            if move_strategy in {"axis_split", "xy_z_split", "phase_axis_split"}:
                self._emit_log("STEP", "Phase move strategy: XY/ROT then Z")
            self._raise_if_aborted()
            self._emit_log("STEP", "Reset alarms")
            self.ctrl.reset_alarms()
            self._raise_if_aborted()
            self._emit_log("STEP", "Servo ON")
            self.ctrl.servo_on()

            if bool(grip_cfg.get("open_at_start", False)):
                self._raise_if_aborted()
                self._emit_log("GRIP", "Phase prepare gripper open")
                self._apply_gripper_phase_state(
                    grip_cfg,
                    "open",
                    fallback_command="open_command",
                    label="phase_open",
                )
                if prepare_open_s > 0:
                    self._sleep_abortable(prepare_open_s)

            self._raise_if_aborted()
            self._move_phase_pose("Ready", poses["ready"], fast, settle_s, move_timeout_s, strategy=move_strategy)
            self.sio.emit("pick_progress", {"step": 1, "total": 6, "name": "ready"})

            self._raise_if_aborted()
            self._move_phase_pose("Hover", poses["hover"], fast, settle_s, move_timeout_s, strategy=move_strategy)
            self.sio.emit("pick_progress", {"step": 2, "total": 6, "name": "hover"})

            self._raise_if_aborted()
            self._move_phase_pose("Pregrasp", poses["pregrasp"], pregrasp_speed, settle_s, move_timeout_s, strategy=move_strategy)
            self.sio.emit("pick_progress", {"step": 3, "total": 6, "name": "pregrasp"})

            self._raise_if_aborted()
            self._move_phase_pose("Grasp", poses["grasp"], grasp_speed, settle_s, move_timeout_s, strategy=move_strategy)
            self.sio.emit("pick_progress", {"step": 4, "total": 6, "name": "grasp"})

            self._raise_if_aborted()
            self._apply_gripper_phase_state(
                grip_cfg,
                "close",
                fallback_command="close_command",
                label="phase_close",
            )
            if grip_hold_s > 0:
                self._sleep_abortable(grip_hold_s)
            self.sio.emit("pick_progress", {"step": 5, "total": 6, "name": "grip"})

            self._raise_if_aborted()
            self._move_phase_pose("Lift", poses["lift"], lift_speed, settle_s, move_timeout_s, strategy=move_strategy)
            if has_place_phase:
                self._raise_if_aborted()
                self._move_phase_pose("Place Hover", place_hover_pose, place_speed, settle_s, move_timeout_s, strategy=move_strategy)

                self._raise_if_aborted()
                self._move_phase_pose("Place", place_pose, place_descend_speed, settle_s, move_timeout_s, strategy=move_strategy)

                self._raise_if_aborted()
                self._apply_gripper_phase_state(
                    grip_cfg,
                    "release",
                    fallback_command="open_command",
                    label="phase_release",
                )
                if release_hold_s > 0:
                    self._sleep_abortable(release_hold_s)

                self._raise_if_aborted()
                self._move_phase_pose("Place Lift", place_lift_pose, place_speed, settle_s, move_timeout_s, strategy=move_strategy)
            if return_pose in {"home", "ready"}:
                self._execute_return_pose(return_pose, lift_speed, move_timeout_s)
            self.sio.emit("pick_progress", {"step": 6, "total": 6, "name": "done"})
            self._emit_log("STEP", f"Phase replay complete: {phase_spec.get('source_recording', recording.get('name', 'unknown'))}")

        except Exception as e:
            self._emit_log("ERROR", f"Phase replay failed: {e}")
        finally:
            if grip_cfg.get("open_after_replay", False):
                try:
                    self._apply_gripper_phase_state(
                        grip_cfg,
                        "open",
                        fallback_command="open_command",
                        label="phase_open_after",
                    )
                except Exception:
                    pass
            self._emit_log("STEP", "Servo OFF")
            self._power_down_servo(reason="replay_phase_recording")

    def _replay_gripper_timeline(
        self,
        timeline: list[dict[str, Any]],
        stop_event: threading.Event,
        replay_start: float,
    ) -> None:
        if not timeline:
            return

        last_positions: Optional[list[int]] = None
        for sample in timeline:
            if stop_event.is_set():
                return

            positions = sample.get("positions")
            if not isinstance(positions, list) or len(positions) != 3:
                continue

            try:
                positions = [int(v) for v in positions]
                target_sec = max(0.0, float(sample.get("t_ms", 0)) / 1000.0)
            except (TypeError, ValueError):
                continue

            wait_sec = target_sec - (time.time() - replay_start)
            if wait_sec > 0:
                time.sleep(wait_sec)
            if stop_event.is_set():
                return
            if positions == last_positions:
                continue
            self._gripper_set_position(positions)
            last_positions = positions

    def _wait_until_replay_t(self, replay_start: float, target_ms: Any) -> None:
        """Align replay with the original recording timeline when timestamps exist."""
        try:
            target_sec = max(0.0, float(target_ms) / 1000.0)
        except (TypeError, ValueError):
            return

        wait_sec = target_sec - (time.time() - replay_start)
        if wait_sec > 0:
            time.sleep(wait_sec)

    def replay_recording(self, recording: dict):
        """Replay a teach-mode recording."""
        self.clear_abort()
        waypoints = recording.get("waypoints", [])
        teach_cfg = self.demo_cfg.get("teach", {})
        raw_return_pose = str(teach_cfg.get("raw_replay_return_pose", "ready")).strip().lower()
        if not waypoints:
            self._emit_log("ERROR", "Recording has no waypoints")
            return

        for wp in waypoints:
            safe, reason = self.check_safety(wp["pose"])
            if not safe:
                self._emit_log("ERROR", f"Safety reject in recording: {reason}")
                return

        if not self.connected:
            if not self.auto_connect():
                return

        try:
            self._raise_if_aborted()
            self._emit_log("STEP", "Reset alarms")
            self.ctrl.reset_alarms()
            self._raise_if_aborted()
            self._emit_log("STEP", "Servo ON")
            self.ctrl.servo_on()

            stop_event = threading.Event()
            timeline_thread: Optional[threading.Thread] = None
            replay_start = time.time()
            use_original_timing = all("t_ms" in wp for wp in waypoints)
            external_timeline = recording.get("external_timeline", [])
            timeline_end_sec = 0.0
            if isinstance(external_timeline, list) and external_timeline:
                try:
                    timeline_end_sec = max(
                        max(0.0, float(sample.get("t_ms", 0)) / 1000.0)
                        for sample in external_timeline
                    )
                except (TypeError, ValueError):
                    timeline_end_sec = 0.0
                self._emit_log("GRIP", f"Streaming external gripper timeline ({len(external_timeline)} samples)")
                timeline_thread = threading.Thread(
                    target=self._replay_gripper_timeline,
                    args=(external_timeline, stop_event, replay_start),
                    daemon=True,
                )
                timeline_thread.start()

            if use_original_timing:
                span_ms = waypoints[-1].get("t_ms", 0)
                self._emit_log("STEP", f"Replay follows recorded timing span ~{float(span_ms) / 1000.0:.1f}s")

            for i, wp in enumerate(waypoints):
                self._raise_if_aborted()
                if use_original_timing:
                    self._wait_until_replay_t(replay_start, wp.get("t_ms", 0))
                    self._raise_if_aborted()
                pose = wp["pose"]
                speed = wp.get("speed", 30)
                gripper = wp.get("gripper", None)
                gripper_pos = self._positions_from_waypoint(wp)
                self._emit_log("STEP",
                    f"Waypoint {i+1}/{len(waypoints)} speed={speed}%")
                self.ctrl.move_to(pose, speed=speed, wait_seconds=2.0)

                self.sio.emit("pick_progress", {
                    "step": i + 1, "total": len(waypoints),
                    "name": f"waypoint_{i+1}"
                })

                if timeline_thread is not None:
                    continue
                self._raise_if_aborted()
                if gripper_pos is not None:
                    self._gripper_set_position(gripper_pos)
                elif gripper == "close":
                    self._gripper_close()
                elif gripper == "open":
                    self.gripper_open()
                elif gripper not in {None, "", "none"}:
                    self._gripper_action(str(gripper))

            if timeline_thread is not None:
                remaining_sec = timeline_end_sec - (time.time() - replay_start)
                join_timeout = max(0.0, remaining_sec) + 2.0
                timeline_thread.join(timeout=join_timeout)

            if raw_return_pose in {"home", "ready"}:
                self._raise_if_aborted()
                self._execute_return_pose(raw_return_pose, 20, 2.0)

            self._emit_log("STEP", "Replay complete")

        except Exception as e:
            self._emit_log("ERROR", f"Replay failed: {e}")
        finally:
            if 'stop_event' in locals():
                stop_event.set()
            if 'timeline_thread' in locals() and timeline_thread is not None:
                timeline_thread.join(timeout=1.0)
            self._emit_log("STEP", "Servo OFF")
            self._power_down_servo(reason="replay_recording")


# ---------------------------------------------------------------------------
# Teach Mode
# ---------------------------------------------------------------------------
class TeachManager:
    """Record and replay arm waypoints."""

    def __init__(
        self,
        demo_cfg: dict,
        arm_mgr: ArmManager,
        cam_mgr: CameraManager,
        socketio: SocketIO,
        objects_cfg: dict[str, Any],
    ):
        self.demo_cfg = demo_cfg
        self.arm = arm_mgr
        self.cam_mgr = cam_mgr
        self.sio = socketio
        self.objects_cfg = objects_cfg
        self.save_dir = PROJECT_ROOT / demo_cfg.get("teach", {}).get(
            "save_dir", "data/teach_recordings")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.recording = False
        self.current_name = ""
        self.waypoints: list[dict] = []
        self.start_time = 0.0
        self.current_meta: dict[str, Any] = {}
        self.phase_mgr = PhaseSpecManager(demo_cfg, objects_cfg, self.save_dir)
        self.dataset_recorder = TeachDatasetRecorder(demo_cfg)

    def reload_config(self, demo_cfg: dict, objects_cfg: dict[str, Any]) -> None:
        self.demo_cfg = demo_cfg
        self.objects_cfg = objects_cfg
        self.save_dir = PROJECT_ROOT / demo_cfg.get("teach", {}).get("save_dir", "data/teach_recordings")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.phase_mgr = PhaseSpecManager(demo_cfg, objects_cfg, self.save_dir)
        self.dataset_recorder = TeachDatasetRecorder(demo_cfg)

    def start(self, name: str, session_meta: Optional[dict[str, Any]] = None):
        self.current_name = name
        self.waypoints = []
        self.start_time = time.time()
        self.recording = True
        meta = dict(session_meta or {})
        inferred_object = meta.get("operator_selected_object") or infer_object_key_from_name(name, self.objects_cfg)
        meta["operator_selected_object"] = inferred_object
        self.current_meta = meta
        episode_dir = self.dataset_recorder.start(
            recording_name=name,
            object_key=inferred_object,
            session_meta=meta,
            cam_mgr=self.cam_mgr,
            arm_mgr=self.arm,
        )
        self.sio.emit("teach_data", {"waypoints": [], "count": 0})
        self.sio.emit("arm_log", {
            "level": "STEP", "message": f"Teach recording started: {name}",
            "timestamp": time.strftime("%H:%M:%S")})
        if episode_dir is not None:
            self.sio.emit("arm_log", {
                "level": "STEP",
                "message": f"Dataset capture -> {episode_dir.relative_to(PROJECT_ROOT)}",
                "timestamp": time.strftime("%H:%M:%S"),
            })

    def save_waypoint(self, gripper: str = "none", speed: int = 30):
        if not self.recording:
            return
        pose = self.arm.current_pose_mm_deg
        if pose is None:
            self.sio.emit("arm_log", {
                "level": "WARN", "message": "Cannot save waypoint: no pose data",
                "timestamp": time.strftime("%H:%M:%S")})
            return
        raw_pose = [int(v * 1000) for v in pose]
        gripper_state = self.arm.get_gripper_state(silent=True)
        wp = {
            "t_ms": int((time.time() - self.start_time) * 1000),
            "pose": raw_pose,
            "gripper": gripper,
            "speed": speed,
        }
        if isinstance(gripper_state, dict):
            current_pos = gripper_state.get("current_pos")
            if isinstance(current_pos, list) and len(current_pos) == 3:
                try:
                    wp["gripper_pos"] = [int(v) for v in current_pos]
                except (TypeError, ValueError):
                    pass
            if "server_time_unix" in gripper_state:
                wp["gripper_server_time_unix"] = gripper_state.get("server_time_unix")
            if "tactile_data" in gripper_state:
                wp["tactile_data"] = gripper_state.get("tactile_data")
        self.waypoints.append(wp)
        self.dataset_recorder.note_waypoint(wp)
        extra = ""
        if "gripper_pos" in wp:
            extra = f" grip_pos={wp['gripper_pos']}"
        self.sio.emit("arm_log", {
            "level": "STEP",
            "message": f"Waypoint {len(self.waypoints)} saved: "
                       f"[{pose[0]:.1f}, {pose[1]:.1f}, {pose[2]:.1f}] "
                       f"gripper={gripper} speed={speed}%{extra}",
            "timestamp": time.strftime("%H:%M:%S")})
        self.sio.emit("teach_data", {
            "waypoints": self.waypoints,
            "count": len(self.waypoints)})

    def stop(self) -> Optional[str]:
        if not self.recording:
            return None
        self.recording = False
        if not self.waypoints:
            dataset_summary = self.dataset_recorder.stop(result="empty", failure_reason="no waypoints")
            self.sio.emit("arm_log", {
                "level": "WARN", "message": "No waypoints recorded",
                "timestamp": time.strftime("%H:%M:%S")})
            if dataset_summary.get("enabled"):
                self.sio.emit("arm_log", {
                    "level": "WARN",
                    "message": f"Dataset capture stopped: {dataset_summary.get('episode_dir', '--')}",
                    "timestamp": time.strftime("%H:%M:%S"),
                })
            self.sio.emit("teach_data", {"waypoints": [], "count": 0})
            return None
        dataset_summary = self.dataset_recorder.stop(result="recorded", failure_reason="")
        data = {
            "name": self.current_name,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "start_unix": self.start_time,
            "waypoints": self.waypoints,
            "artifacts": {
                "episode_dir": dataset_summary.get("episode_dir", ""),
                "metadata_path": dataset_summary.get("metadata_path", ""),
                "capture_error": dataset_summary.get("capture_error", ""),
                "auto_label": dataset_summary.get("auto_label", {}),
            },
        }
        phase_path = None
        try:
            phase_path = self.phase_mgr.save_phase_spec(self.current_name, data)
            data["artifacts"]["phase_yaml"] = str(phase_path.relative_to(PROJECT_ROOT))
        except Exception as exc:
            data["artifacts"]["phase_yaml_error"] = str(exc)
        path = self.save_dir / f"{self.current_name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self.sio.emit("arm_log", {
            "level": "STEP",
            "message": f"Recording saved: {path.name} ({len(self.waypoints)} waypoints)",
            "timestamp": time.strftime("%H:%M:%S")})
        if phase_path is not None:
            self.sio.emit("arm_log", {
                "level": "STEP",
                "message": f"Phase spec saved: {phase_path.name}",
                "timestamp": time.strftime("%H:%M:%S"),
            })
        elif data["artifacts"].get("phase_yaml_error"):
            self.sio.emit("arm_log", {
                "level": "WARN",
                "message": f"Phase spec generation failed: {data['artifacts']['phase_yaml_error']}",
                "timestamp": time.strftime("%H:%M:%S"),
            })
        if dataset_summary.get("enabled"):
            self.sio.emit("arm_log", {
                "level": "STEP",
                "message": f"Episode bundle saved: {dataset_summary.get('episode_dir', '--')}",
                "timestamp": time.strftime("%H:%M:%S"),
            })
        self.sio.emit("teach_data", {"waypoints": [], "count": 0})
        return str(path)

    def regenerate_phase(self, name: str) -> Optional[str]:
        rec = self.load_recording(name)
        if not rec:
            return None
        phase_path = self.phase_mgr.save_phase_spec(name, rec)
        artifacts = rec.get("artifacts", {}) if isinstance(rec, dict) else {}
        if not isinstance(artifacts, dict):
            artifacts = {}
        artifacts["phase_yaml"] = str(phase_path.relative_to(PROJECT_ROOT))
        rec["artifacts"] = artifacts
        path = self.save_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        return str(phase_path)

    def list_recordings(self) -> list[dict]:
        results = []
        for p in sorted(self.save_dir.glob("*.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                artifacts = data.get("artifacts", {}) if isinstance(data, dict) else {}
                waypoints = data.get("waypoints", []) if isinstance(data, dict) else []
                if not isinstance(waypoints, list) or not waypoints:
                    continue
                phase_path = self.phase_mgr.phase_path(p.stem)
                episode_dir = artifacts.get("episode_dir", "")
                episode_exists = bool(episode_dir and (PROJECT_ROOT / episode_dir).exists())
                results.append({
                    "id": p.stem,
                    "name": data.get("name", p.stem),
                    "created": data.get("created", ""),
                    "count": len(waypoints),
                    "has_external_timeline": bool(data.get("external_timeline")),
                    "has_phase_yaml": phase_path.exists(),
                    "has_episode_data": episode_exists,
                    "phase_yaml": str(phase_path.relative_to(PROJECT_ROOT)) if phase_path.exists() else "",
                    "episode_dir": episode_dir if episode_exists else "",
                })
            except Exception:
                pass
        return results

    def load_recording(self, name: str) -> Optional[dict]:
        path = self.save_dir / f"{name}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_phase_spec(self, name: str) -> Optional[dict[str, Any]]:
        spec = self.phase_mgr.load_phase_spec(name)
        if spec:
            return spec
        if name.endswith(".merged"):
            return self.phase_mgr.load_phase_spec(name[:-7])
        return None

    def shutdown(self) -> None:
        if self.recording:
            try:
                summary = self.dataset_recorder.stop(
                    result="aborted",
                    failure_reason="server shutdown",
                )
                if summary.get("enabled"):
                    self.sio.emit("arm_log", {
                        "level": "WARN",
                        "message": f"Teach dataset capture aborted during shutdown: {summary.get('episode_dir', '--')}",
                        "timestamp": time.strftime("%H:%M:%S"),
                    })
            except Exception:
                pass
        self.recording = False
        self.current_name = ""
        self.waypoints = []
        self.current_meta = {}
        self.sio.emit("teach_data", {"waypoints": [], "count": 0})


# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
def create_app(demo_cfg: dict) -> tuple[Flask, SocketIO]:
    static_dir = Path(__file__).parent / "static"
    app = Flask(__name__, static_folder=str(static_dir))
    app.config["SECRET_KEY"] = "voice-pick-demo"
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

    objects_cfg = load_config("objects.yaml")
    try:
        speech_prompts_cfg = load_config("speech_prompts.yaml") or {}
    except Exception:
        speech_prompts_cfg = {}
    try:
        nlu_aliases_cfg = load_config("nlu_aliases.yaml") or {}
    except Exception:
        nlu_aliases_cfg = {}
    nlu = IntentParser(objects_cfg=objects_cfg, aliases_cfg=nlu_aliases_cfg)
    latest_speech: dict[str, Any] = {}
    asr_lock = threading.Lock()
    asr_state: dict[str, Any] = {"engine": None, "error": ""}
    panel_recordings_lock = threading.Lock()
    dashboard_log_buffer = DashboardLogBuffer()

    cam_mgr = CameraManager(demo_cfg.get("cameras", {}))
    validation_recorder = ValidationSessionRecorder(socketio, objects_cfg)
    digital_twin_sync = DigitalTwinSync(demo_cfg, socketio)
    gripper_monitor = GripperMonitor(demo_cfg, socketio, digital_twin_sync=digital_twin_sync)
    sensor_monitor = SensorMonitor(demo_cfg, socketio)
    arm_mgr = ArmManager(
        demo_cfg,
        socketio,
        validation_recorder=validation_recorder,
        digital_twin_sync=digital_twin_sync,
    )
    teach_mgr = TeachManager(demo_cfg, arm_mgr, cam_mgr, socketio, objects_cfg)
    panel_recorder = BackendPanelRecorder(
        cam_mgr=cam_mgr,
        arm_mgr_getter=lambda: arm_mgr,
        gripper_monitor=gripper_monitor,
        sensor_monitor=sensor_monitor,
        log_buffer=dashboard_log_buffer,
    )

    def _replace_dict(target: dict[str, Any], fresh: dict[str, Any]) -> None:
        target.clear()
        target.update(fresh or {})

    def _build_safe_config_payload() -> dict[str, Any]:
        prompt_objects = speech_prompts_cfg.get("objects", {}) if isinstance(speech_prompts_cfg, dict) else {}
        active_object_id = speech_prompts_cfg.get("active_object", "") if isinstance(speech_prompts_cfg, dict) else ""
        active_object_prompt = prompt_objects.get(active_object_id, {}) if isinstance(prompt_objects, dict) else {}
        try:
            voice_cfg = load_config("voice_config.yaml") or {}
        except Exception:
            voice_cfg = {}
        asr_engine = voice_cfg.get("engine", "whisper")
        whisper_cfg = voice_cfg.get("whisper", {}) if isinstance(voice_cfg, dict) else {}
        effective_pose_limits = _effective_pose_limits(demo_cfg)
        effective_pose_limits["enforce_rotation_limits"] = bool(demo_cfg.get("pose_limits", {}).get("enforce_rotation_limits", False))
        return {
            "safety_boundary": demo_cfg.get("safety_boundary", {}),
            "pose_limits": effective_pose_limits,
            "speed": demo_cfg.get("speed", {}),
            "home_pose": demo_cfg.get("home_pose", []),
            "home_routing": demo_cfg.get("home_routing", {}),
            "modules": demo_cfg.get("modules", {}),
            "virtual_env": demo_cfg.get("virtual_env", {}),
            "trajectory": demo_cfg.get("trajectory", {}),
            "sensor_api": {
                "enabled": bool(demo_cfg.get("sensor_api", {}).get("enabled", False)),
                "history_points": int(demo_cfg.get("sensor_api", {}).get("history_points", 180)),
                "auto_range": bool(demo_cfg.get("sensor_api", {}).get("auto_range", True)),
                "auto_range_padding_ratio": float(demo_cfg.get("sensor_api", {}).get("auto_range_padding_ratio", 0.08)),
                "min_range_span": float(demo_cfg.get("sensor_api", {}).get("min_range_span", 300)),
                "y_min": int(demo_cfg.get("sensor_api", {}).get("y_min", 0)),
                "y_max": int(demo_cfg.get("sensor_api", {}).get("y_max", 1100)),
                "panel": demo_cfg.get("sensor_api", {}).get("panel", {}),
            },
            "teach": {
                "default_replay_mode": demo_cfg.get("teach", {}).get("default_replay_mode", "phase"),
                "enabled": bool(demo_cfg.get("teach", {}).get("enabled", True)),
            },
            "asr": {
                "mode": f"offline_{asr_engine}",
                "fallback_model": whisper_cfg.get("model_size", "small"),
            },
            "speech_prompt": {
                "active_object": active_object_id,
                "active_prompt": active_object_prompt,
                "objects": prompt_objects if isinstance(prompt_objects, dict) else {},
            },
            "nlu_debug": {
                "enabled": bool(nlu_aliases_cfg.get("debug", {}).get("enabled", False)) if isinstance(nlu_aliases_cfg, dict) else False,
            },
        }

    def _reload_runtime_configs() -> tuple[bool, str, list[str]]:
        nonlocal nlu
        notes: list[str] = []
        try:
            new_demo_cfg = _apply_module_runtime_overrides(load_config("demo_config.yaml"))
            new_objects_cfg = load_config("objects.yaml")
            try:
                new_speech_prompts_cfg = load_config("speech_prompts.yaml") or {}
            except Exception:
                new_speech_prompts_cfg = {}
            try:
                new_nlu_aliases_cfg = load_config("nlu_aliases.yaml") or {}
            except Exception:
                new_nlu_aliases_cfg = {}
            try:
                load_config("voice_config.yaml")
            except Exception as exc:
                return False, f"voice_config.yaml invalid: {exc}", notes

            _replace_dict(demo_cfg, new_demo_cfg)
            _replace_dict(objects_cfg, new_objects_cfg)
            _replace_dict(speech_prompts_cfg, new_speech_prompts_cfg)
            _replace_dict(nlu_aliases_cfg, new_nlu_aliases_cfg)

            nlu = IntentParser(objects_cfg=objects_cfg, aliases_cfg=nlu_aliases_cfg)
            validation_recorder.objects_cfg = objects_cfg
            arm_mgr.demo_cfg = demo_cfg
            cam_mgr.reload_config(demo_cfg.get("cameras", {}))
            gripper_monitor.reload_config(demo_cfg)
            sensor_monitor.reload_config(demo_cfg)
            digital_twin_sync.reload_config(demo_cfg)
            teach_mgr.reload_config(demo_cfg, objects_cfg)
            if bool(demo_cfg.get("virtual_env", {}).get("sync", {}).get("objects", {}).get("sync_on_reload", False)):
                try:
                    result = digital_twin_sync.sync_scene_objects(objects_cfg, reason="reload_config")
                    notes.append(f"Scene object sync reloaded ({result.get('count', 0)} targets)")
                except Exception as exc:
                    notes.append(f"Scene object sync skipped: {exc}")

            with asr_lock:
                asr_state["engine"] = None
                asr_state["error"] = ""

            notes.append("NLU / speech / demo / sensor / digital twin settings reloaded")
            notes.append("Offline ASR engine will be re-created on next offline transcription")
            notes.append("Camera source and some low-level Modbus changes may still require backend restart")

            socketio.emit("config_reloaded", {
                "ok": True,
                "message": "Runtime config reloaded",
                "notes": notes,
                "config": _build_safe_config_payload(),
            })
            socketio.emit("objects_catalog", objects_cfg)
            socketio.emit("camera_status", cam_mgr.status())
            socketio.emit("validation_state", validation_recorder.state_payload())
            socketio.emit("teach_list", teach_mgr.list_recordings() if _module_enabled(demo_cfg, "teach", demo_cfg.get("teach", {}).get("enabled", True)) else [])
            socketio.emit("gripper_state", {
                "connected": gripper_monitor.connected,
                "last_error": gripper_monitor.last_error or None,
                "state": gripper_monitor.last_state,
            })
            socketio.emit("sensor_state", sensor_monitor.state_payload())
            if sensor_monitor.history:
                socketio.emit("sensor_data", {
                    "history": sensor_monitor.history,
                    "sample": sensor_monitor.last_sample,
                })
            socketio.emit("virtual_env_sync", digital_twin_sync.state_payload())
            return True, "Runtime config reloaded", notes
        except Exception as exc:
            return False, str(exc), notes

    # --- HTTP Routes ---

    @app.route("/")
    def index():
        return send_from_directory(static_dir, "index.html")

    @app.route("/<path:filename>")
    def static_files(filename):
        return send_from_directory(static_dir, filename)

    def _mjpeg_gen(cam_key: str, stream: str):
        while True:
            frame = cam_mgr.get_jpeg(cam_key, stream)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(cam_mgr.get_stream_interval(cam_key))

    @app.route("/stream/<cam_key>/<stream_type>")
    def stream_camera(cam_key, stream_type):
        valid_standard = cam_key in ("cam1", "cam2") and stream_type in ("rgb", "depth")
        valid_claw = cam_key == "claw" and stream_type == "rgb"
        if not (valid_standard or valid_claw):
            return "Not found", 404
        return Response(_mjpeg_gen(cam_key, stream_type),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    # Legacy routes for backward compat
    @app.route("/stream/rgb")
    def stream_rgb():
        return Response(_mjpeg_gen("cam1", "rgb"),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/stream/depth")
    def stream_depth():
        return Response(_mjpeg_gen("cam1", "depth"),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/objects")
    def api_objects():
        classes = objects_cfg.get("classes", {})
        result = {}
        for key, obj in classes.items():
            result[key] = {
                "chinese": obj.get("chinese", []),
                "english": obj.get("english", []),
                "has_fixed_poses": "fixed_poses" in obj,
                "slot_id": obj.get("slot_id", ""),
                "table_set_id": obj.get("table_set_id", ""),
                "default_teach_recording": validation_recorder.default_teach_recording(key),
            }
        return jsonify(result)

    @app.route("/api/recordings")
    def api_recordings():
        if not _module_enabled(demo_cfg, "teach", demo_cfg.get("teach", {}).get("enabled", True)):
            return jsonify([])
        return jsonify(teach_mgr.list_recordings())

    @app.route("/api/teach_recordings")
    def api_teach_recordings():
        if not _module_enabled(demo_cfg, "teach", demo_cfg.get("teach", {}).get("enabled", True)):
            return jsonify([])
        return jsonify(teach_mgr.list_recordings())

    @app.route("/api/config")
    def api_config():
        return jsonify(_build_safe_config_payload())

    @app.route("/api/camera_status")
    def api_camera_status():
        return jsonify(cam_mgr.status())

    @app.route("/api/cameras/<cam_key>/restart", methods=["POST"])
    def api_camera_restart(cam_key):
        if cam_key not in ("cam1", "cam2", "claw"):
            return jsonify({"ok": False, "error": "unknown camera"}), 404
        result = cam_mgr.restart_camera(cam_key)
        socketio.emit("camera_status", result.get("status", cam_mgr.status()))
        status = 200 if result.get("ok") else 500
        return jsonify(result), status

    @app.route("/api/reload_config", methods=["POST"])
    def api_reload_config():
        ok, message, notes = _reload_runtime_configs()
        payload = {
            "ok": ok,
            "message": message,
            "notes": notes,
            "config": _build_safe_config_payload(),
        }
        status = 200 if ok else 500
        return jsonify(payload), status

    @app.route("/api/panel_recordings/upload", methods=["POST"])
    def api_panel_recordings_upload():
        upload = request.files.get("file")
        if upload is None:
            return jsonify({"ok": False, "error": "missing file upload"}), 400

        session_id = _safe_file_token(
            request.form.get("session_id"),
            time.strftime("panel_recording_%Y%m%d_%H%M%S"),
        )
        panel_id = _safe_file_token(request.form.get("panel_id"), "panel")
        raw_ext = Path(upload.filename or "").suffix.lower().lstrip(".")
        if raw_ext not in {"mp4", "webm"}:
            raw_ext = "mp4" if "mp4" in str(upload.mimetype or "") else "webm"

        session_dir = PANEL_RECORDINGS_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        video_path = session_dir / f"{panel_id}.{raw_ext}"
        upload.save(str(video_path))

        metadata = {
            "session_id": session_id,
            "panel_id": panel_id,
            "filename": video_path.name,
            "content_type": upload.mimetype,
            "size_bytes": video_path.stat().st_size if video_path.exists() else 0,
            "width": request.form.get("width", ""),
            "height": request.form.get("height", ""),
            "saved_unix": time.time(),
        }
        with open(session_dir / f"{panel_id}.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return jsonify({
            "ok": True,
            "path": str(video_path.relative_to(PROJECT_ROOT)),
            "metadata": metadata,
        })

    @app.route("/api/panel_recordings/start", methods=["POST"])
    def api_panel_recordings_start():
        payload = request.get_json(silent=True) or {}
        session_id = _safe_file_token(
            payload.get("session_id"),
            time.strftime("panel_recording_%Y%m%d_%H%M%S"),
        )
        session_dir = PANEL_RECORDINGS_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        panels = payload.get("panels", [])
        genlock = payload.get("genlock", {})
        metadata = {
            "session_id": session_id,
            "started_unix": time.time(),
            "fps": float(payload.get("fps", 60)),
            "panels": panels if isinstance(panels, list) else [],
            "genlock": genlock if isinstance(genlock, dict) else {},
        }
        with open(session_dir / "session.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        recorder_info = panel_recorder.start(session_dir, session_id, metadata["panels"], metadata["fps"])
        metadata["mode"] = "backend_mp4"
        metadata["recorder"] = recorder_info
        with open(session_dir / "session.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return jsonify({
            "ok": True,
            "session_id": session_id,
            "dir": str(session_dir.relative_to(PROJECT_ROOT)),
            "mode": "backend_mp4",
            "recorder": recorder_info,
        })

    @app.route("/api/panel_recordings/frame", methods=["POST"])
    def api_panel_recordings_frame():
        upload = request.files.get("file")
        if upload is None:
            return jsonify({"ok": False, "error": "missing frame upload"}), 400
        session_id = _safe_file_token(request.form.get("session_id"), "")
        panel_id = _safe_file_token(request.form.get("panel_id"), "")
        if not session_id or not panel_id:
            return jsonify({"ok": False, "error": "missing session_id or panel_id"}), 400
        try:
            frame_index = int(request.form.get("frame_index", "0"))
        except (TypeError, ValueError):
            frame_index = 0

        frame_dir = PANEL_RECORDINGS_DIR / session_id / panel_id / "frames"
        with panel_recordings_lock:
            frame_dir.mkdir(parents=True, exist_ok=True)
            frame_path = frame_dir / f"{frame_index:06d}.png"
            upload.save(str(frame_path))

            panel_dir = frame_dir.parent
            manifest_path = panel_dir / "frame_manifest.csv"
            panel_manifest_exists = manifest_path.exists()
            server_ts = time.time()
            row = {
                "frame_index": frame_index,
                "filename": frame_path.name,
                "path": str(frame_path.relative_to(PROJECT_ROOT)),
                "client_timestamp_unix": request.form.get("timestamp_unix", ""),
                "client_elapsed_ms": request.form.get("elapsed_ms", ""),
                "client_perf_ms": request.form.get("perf_ms", ""),
                "server_received_unix": f"{server_ts:.6f}",
                "width": request.form.get("width", ""),
                "height": request.form.get("height", ""),
            }
            with open(manifest_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "frame_index",
                        "filename",
                        "path",
                        "client_timestamp_unix",
                        "client_elapsed_ms",
                        "client_perf_ms",
                        "server_received_unix",
                        "width",
                        "height",
                    ],
                )
                if not panel_manifest_exists:
                    writer.writeheader()
                writer.writerow(row)

            genlock_path = PANEL_RECORDINGS_DIR / session_id / "genlock_frames.csv"
            genlock_exists = genlock_path.exists()
            with open(genlock_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "panel_id",
                        "frame_index",
                        "filename",
                        "path",
                        "client_timestamp_unix",
                        "client_elapsed_ms",
                        "client_perf_ms",
                        "server_received_unix",
                        "width",
                        "height",
                    ],
                )
                if not genlock_exists:
                    writer.writeheader()
                writer.writerow({"panel_id": panel_id, **row})

        return jsonify({
            "ok": True,
            "path": str(frame_path.relative_to(PROJECT_ROOT)),
        })

    @app.route("/api/panel_recordings/finish", methods=["POST"])
    def api_panel_recordings_finish():
        payload = request.get_json(silent=True) or {}
        session_id = _safe_file_token(payload.get("session_id"), "")
        if not session_id:
            return jsonify({"ok": False, "error": "missing session_id"}), 400
        session_dir = PANEL_RECORDINGS_DIR / session_id
        if not session_dir.exists():
            return jsonify({"ok": False, "error": "recording session not found"}), 404

        fps = float(payload.get("fps", 60) or 60)
        backend_summary = panel_recorder.stop(session_id=session_id, timeout_s=10.0)
        if backend_summary is not None:
            return jsonify({"ok": True, **backend_summary})

        panels = payload.get("panels", [])
        panel_meta = {}
        if isinstance(panels, list):
            for panel in panels:
                if not isinstance(panel, dict):
                    continue
                panel_id = _safe_file_token(panel.get("panel_id"), "")
                if panel_id:
                    panel_meta[panel_id] = panel

        outputs = []
        errors = []
        for panel_dir in sorted(path for path in session_dir.iterdir() if path.is_dir()):
            panel_id = panel_dir.name
            frames = sorted((panel_dir / "frames").glob("*.png"))
            if not frames:
                errors.append(f"{panel_id}: no frames")
                continue
            meta = panel_meta.get(panel_id, {})
            width = int(meta.get("width") or 0)
            height = int(meta.get("height") or 0)
            if width <= 0 or height <= 0:
                first = cv2.imread(str(frames[0]))
                if first is not None:
                    width = int(first.shape[1])
                    height = int(first.shape[0])
            outputs.append({
                "panel_id": panel_id,
                "frames_dir": str((panel_dir / "frames").relative_to(PROJECT_ROOT)),
                "frame_manifest": str((panel_dir / "frame_manifest.csv").relative_to(PROJECT_ROOT)),
                "frames": len(frames),
                "width": width,
                "height": height,
            })

        genlock_path = session_dir / "genlock_frames.csv"
        genlock_summary = {
            "path": str(genlock_path.relative_to(PROJECT_ROOT)) if genlock_path.exists() else None,
            "frames": 0,
            "client_time_range": None,
            "server_time_range": None,
        }
        if genlock_path.exists():
            client_ts: list[float] = []
            server_ts: list[float] = []
            with open(genlock_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    genlock_summary["frames"] += 1
                    try:
                        if row.get("client_timestamp_unix"):
                            client_ts.append(float(row["client_timestamp_unix"]))
                    except (TypeError, ValueError):
                        pass
                    try:
                        if row.get("server_received_unix"):
                            server_ts.append(float(row["server_received_unix"]))
                    except (TypeError, ValueError):
                        pass
            if client_ts:
                genlock_summary["client_time_range"] = [min(client_ts), max(client_ts)]
            if server_ts:
                genlock_summary["server_time_range"] = [min(server_ts), max(server_ts)]

        summary = {
            "session_id": session_id,
            "finished_unix": time.time(),
            "fps": fps,
            "outputs": outputs,
            "errors": errors,
            "genlock": genlock_summary,
        }
        with open(session_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return jsonify({"ok": True, **summary})

    @app.route("/api/virtual_env/sync_state")
    def api_virtual_env_sync_state():
        return jsonify(digital_twin_sync.state_payload())

    @app.route("/api/virtual_env/calibrate_current", methods=["POST"])
    def api_virtual_env_calibrate_current():
        local_pose = arm_mgr.current_pose_mm_deg
        if local_pose is None:
            return jsonify({"ok": False, "error": "no local arm pose available"}), 409
        try:
            result = digital_twin_sync.calibrate_from_current_pair(local_pose)
            return jsonify(result)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/api/virtual_env/reset_calibration", methods=["POST"])
    def api_virtual_env_reset_calibration():
        try:
            result = digital_twin_sync.restore_arm_calibration()
            return jsonify(result)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/api/virtual_env/sync_scene_objects", methods=["POST"])
    def api_virtual_env_sync_scene_objects():
        try:
            result = digital_twin_sync.sync_scene_objects(objects_cfg, reason="api")
            return jsonify(result)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    def get_offline_asr():
        engine = asr_state.get("engine")
        if engine is not None:
            return engine

        with asr_lock:
            engine = asr_state.get("engine")
            if engine is not None:
                return engine
            try:
                from src.asr import SpeechRecognizer

                engine = SpeechRecognizer()
                if not engine.available:
                    raise RuntimeError("Configured offline ASR engine is unavailable in the current Python environment")
                asr_state["engine"] = engine
                asr_state["error"] = ""
                return engine
            except Exception as exc:
                asr_state["engine"] = None
                asr_state["error"] = str(exc)
                log.error("Offline ASR init failed: %s", exc)
                raise

    @app.route("/api/asr/transcribe", methods=["POST"])
    def api_asr_transcribe():
        audio_file = request.files.get("audio")
        if audio_file is None:
            return jsonify({"ok": False, "error": "missing audio upload"}), 400

        wav_bytes = audio_file.read()
        if not wav_bytes:
            return jsonify({"ok": False, "error": "empty audio upload"}), 400

        lang_hint = request.form.get("lang", "").strip()
        try:
            asr = get_offline_asr()
            text = asr.transcribe_wav_bytes(wav_bytes, language=lang_hint).strip()
            return jsonify({
                "ok": True,
                "text": text,
                "lang": lang_hint or "zh-TW",
            })
        except Exception as exc:
            log.error("Offline ASR transcribe failed: %s", exc)
            return jsonify({
                "ok": False,
                "error": str(exc),
            }), 500

    # --- WebSocket Events ---

    @socketio.on("connect")
    def on_connect():
        log.info("Client connected")
        emit("arm_status", {
            "connected": arm_mgr.connected,
            "label": arm_mgr.connection_label,
            "error": None
        })
        emit("camera_status", cam_mgr.status())
        emit("teach_list", teach_mgr.list_recordings() if _module_enabled(demo_cfg, "teach", demo_cfg.get("teach", {}).get("enabled", True)) else [])
        emit("validation_state", validation_recorder.state_payload())
        emit("gripper_state", {
            "connected": gripper_monitor.connected,
            "last_error": gripper_monitor.last_error or None,
            "state": gripper_monitor.last_state,
        })
        emit("sensor_state", sensor_monitor.state_payload())
        if sensor_monitor.history:
            emit("sensor_data", {
                "history": sensor_monitor.history,
                "sample": sensor_monitor.last_sample,
            })
        emit("virtual_env_sync", digital_twin_sync.state_payload())

    @socketio.on("voice_text")
    def on_voice_text(data):
        text = data.get("text", "").strip()
        lang = data.get("lang", "zh-TW")
        if not text:
            return
        log.info("Voice input (%s): %s", lang, text)
        result = nlu.parse(text)
        disambiguation = ""
        if result.need_confirmation and result.candidates:
            disambiguation = nlu.get_disambiguation_prompt(result)
        elif result.need_confirmation:
            disambiguation = nlu.get_disambiguation_prompt(result)

        payload = {
            "intent": result.intent,
            "object": result.object_key,
            "confidence": result.confidence,
            "candidates": result.candidates,
            "need_confirm": result.need_confirmation,
            "disambiguation": disambiguation,
            "focus_keywords": nlu.extract_focus_keywords(text, result),
            "raw_text": text,
            "matched_phrase": result.matched_phrase,
            "match_source": result.match_source,
            "normalization_applied": result.normalization_applied,
            "normalized_text": result.normalized_text,
        }
        latest_speech.clear()
        latest_speech.update({
            "requested_text": text,
            "lang": lang,
            "nlu_result": payload,
        })
        emit("nlu_result", payload)

    @socketio.on("frontend_log")
    def on_frontend_log(data):
        if not isinstance(data, dict):
            return
        message = str(data.get("message", "")).strip()
        if not message:
            return
        dashboard_log_buffer.append(
            data.get("level", "INFO"),
            message[:500],
            data.get("timestamp") or time.strftime("%H:%M:%S"),
        )

    @socketio.on("frontend_log_clear")
    def on_frontend_log_clear():
        dashboard_log_buffer.clear()

    @socketio.on("confirm_pick")
    def on_confirm_pick(data):
        obj_key = data.get("object_key")
        method = data.get("method", "fixed")
        recording_name = data.get("recording_name", "")
        requested_text = data.get("requested_text", "").strip()
        source = data.get("source", "ui")

        if not obj_key:
            emit("arm_log", {
                "level": "ERROR", "message": "No object specified",
                "timestamp": time.strftime("%H:%M:%S")})
            return

        obj_def = objects_cfg.get("classes", {}).get(obj_key, {})
        if not obj_def:
            emit("arm_log", {
                "level": "ERROR", "message": f"Unknown object: {obj_key}",
                "timestamp": time.strftime("%H:%M:%S")})
            return

        if method == "teach" and not recording_name:
            recording_name = validation_recorder.default_teach_recording(obj_key)

        if method == "teach" and not _module_enabled(demo_cfg, "teach", demo_cfg.get("teach", {}).get("enabled", True)):
            emit("arm_log", {
                "level": "WARN",
                "message": "Teach module is disabled in config",
                "timestamp": time.strftime("%H:%M:%S"),
            })
            return

        if not requested_text:
            requested_text = latest_speech.get("requested_text", "") or obj_key

        ok, info = validation_recorder.start_session(
            object_key=obj_key,
            obj_def=obj_def,
            mode=method,
            requested_text=requested_text,
            resolved_object_key=obj_key,
            teach_recording_name=recording_name if method == "teach" else "",
        )
        if not ok:
            emit("arm_log", {
                "level": "WARN",
                "message": info,
                "timestamp": time.strftime("%H:%M:%S"),
            })
            emit("validation_state", validation_recorder.state_payload())
            return

        if source == "voice" and latest_speech:
            validation_recorder.log_speech("voice_text", {
                "requested_text": latest_speech.get("requested_text", ""),
                "lang": latest_speech.get("lang", ""),
            })
            validation_recorder.log_speech("nlu_result", latest_speech.get("nlu_result", {}))
        else:
            validation_recorder.log_speech("ui_request", {
                "requested_text": requested_text,
                "resolved_object_key": obj_key,
            })

        def _run():
            if method == "teach" and recording_name:
                rec = teach_mgr.load_recording(recording_name)
                if rec:
                    replay_mode = str(
                        demo_cfg.get("teach", {}).get("default_replay_mode", "phase")
                    ).strip().lower()
                    if replay_mode in {"phase", "phase_axis_split"}:
                        phase_spec = teach_mgr.load_phase_spec(recording_name)
                        if phase_spec:
                            strategy = "axis_split" if replay_mode == "phase_axis_split" else "direct"
                            arm_mgr.replay_phase_recording(rec, phase_spec, move_strategy=strategy)
                        else:
                            arm_mgr._emit_log(
                                "WARN",
                                f"Phase spec missing for {recording_name}; fallback to raw replay",
                            )
                            arm_mgr.replay_recording(rec)
                    else:
                        arm_mgr.replay_recording(rec)
                else:
                    arm_mgr._emit_log("ERROR", f"Recording not found: {recording_name}")
            else:
                arm_mgr.pick_fixed(obj_key, obj_def)

        threading.Thread(target=_run, daemon=True).start()

    @socketio.on("arm_connect")
    def on_arm_connect():
        threading.Thread(target=arm_mgr.auto_connect, daemon=True).start()

    @socketio.on("arm_home")
    def on_arm_home():
        threading.Thread(target=arm_mgr.home, daemon=True).start()

    @socketio.on("arm_ready")
    def on_arm_ready():
        threading.Thread(target=arm_mgr.ready, daemon=True).start()

    @socketio.on("arm_reset_alarms")
    def on_arm_reset_alarms():
        def _run():
            arm_mgr.clear_abort()
            if not arm_mgr.connected:
                arm_mgr.auto_connect()
            if arm_mgr.connected:
                arm_mgr._emit_log("STEP", "Reset Alarm command")
                arm_mgr.ctrl.reset_alarms()
                arm_mgr._emit_log("STEP", "Reset Alarm complete")
        threading.Thread(target=_run, daemon=True).start()

    @socketio.on("arm_emergency_stop")
    def on_arm_emergency_stop():
        arm_mgr.request_abort()

        def _run():
            arm_mgr._emit_log("ERROR", "EMERGENCY STOP triggered")
            if arm_mgr.connected and arm_mgr.ctrl:
                arm_mgr.motion_stop()
            try:
                arm_mgr.gripper_stop()
            except Exception:
                pass
            if arm_mgr.connected and arm_mgr.ctrl:
                arm_mgr._power_down_servo(reason="emergency_stop")
            arm_mgr._emit_log("ERROR", "Emergency stop complete")

        threading.Thread(target=_run, daemon=True).start()

    @socketio.on("teach_start")
    def on_teach_start(data):
        if not _module_enabled(demo_cfg, "teach", demo_cfg.get("teach", {}).get("enabled", True)):
            emit("arm_log", {
                "level": "WARN",
                "message": "Teach module is disabled",
                "timestamp": time.strftime("%H:%M:%S"),
            })
            return
        name = data.get("name", f"recording_{int(time.time())}")
        nlu_result = latest_speech.get("nlu_result", {}) if isinstance(latest_speech.get("nlu_result"), dict) else {}
        session_meta = {
            "requested_text": latest_speech.get("requested_text", ""),
            "asr_text": latest_speech.get("requested_text", ""),
            "lang": latest_speech.get("lang", ""),
            "operator_selected_object": nlu_result.get("object", "") or infer_object_key_from_name(name, objects_cfg),
        }
        teach_mgr.start(name, session_meta=session_meta)

    @socketio.on("teach_waypoint")
    def on_teach_waypoint(data):
        if not _module_enabled(demo_cfg, "teach", demo_cfg.get("teach", {}).get("enabled", True)):
            return
        gripper = data.get("gripper", "none")
        speed = data.get("speed", 30)
        teach_mgr.save_waypoint(gripper, speed)

    @socketio.on("teach_stop")
    def on_teach_stop():
        if not _module_enabled(demo_cfg, "teach", demo_cfg.get("teach", {}).get("enabled", True)):
            return
        teach_mgr.stop()
        emit("teach_list", teach_mgr.list_recordings())

    @socketio.on("teach_replay")
    def on_teach_replay(data):
        if not _module_enabled(demo_cfg, "teach", demo_cfg.get("teach", {}).get("enabled", True)):
            emit("arm_log", {
                "level": "WARN",
                "message": "Teach module is disabled",
                "timestamp": time.strftime("%H:%M:%S"),
            })
            return
        name = data.get("name", "")
        replay_mode = str(data.get("replay_mode", "raw")).strip().lower()
        rec = teach_mgr.load_recording(name)
        if rec:
            if replay_mode in {"phase", "phase_axis_split"}:
                phase_spec = teach_mgr.load_phase_spec(name)
                if phase_spec:
                    strategy = "axis_split" if replay_mode == "phase_axis_split" else "direct"
                    threading.Thread(
                        target=arm_mgr.replay_phase_recording,
                        args=(rec, phase_spec, strategy),
                        daemon=True,
                    ).start()
                else:
                    emit("arm_log", {
                        "level": "WARN",
                        "message": f"Phase spec missing for {name}; fallback to raw replay",
                        "timestamp": time.strftime("%H:%M:%S")})
                    threading.Thread(
                        target=arm_mgr.replay_recording, args=(rec,), daemon=True
                    ).start()
            else:
                threading.Thread(
                    target=arm_mgr.replay_recording, args=(rec,), daemon=True
                ).start()
        else:
            emit("arm_log", {
                "level": "ERROR",
                "message": f"Recording not found: {name}",
                "timestamp": time.strftime("%H:%M:%S")})

    @socketio.on("teach_regenerate_phase")
    def on_teach_regenerate_phase(data):
        if not _module_enabled(demo_cfg, "teach", demo_cfg.get("teach", {}).get("enabled", True)):
            return
        name = data.get("name", "")
        path = teach_mgr.regenerate_phase(name)
        if path:
            emit("arm_log", {
                "level": "STEP",
                "message": f"Phase spec regenerated: {Path(path).name}",
                "timestamp": time.strftime("%H:%M:%S")})
            emit("teach_list", teach_mgr.list_recordings())
        else:
            emit("arm_log", {
                "level": "ERROR",
                "message": f"Cannot regenerate phase spec: {name}",
                "timestamp": time.strftime("%H:%M:%S")})

    @socketio.on("validation_mark")
    def on_validation_mark(data):
        result = data.get("result", "")
        ok, info = validation_recorder.mark_result(result)
        level = "STEP" if ok else "WARN"
        emit("arm_log", {
            "level": level,
            "message": f"Validation mark: {info}",
            "timestamp": time.strftime("%H:%M:%S"),
        })
        emit("validation_state", validation_recorder.state_payload())

    # --- Startup ---
    try:
        preload_voice_cfg = load_config("voice_config.yaml") or {}
    except Exception:
        preload_voice_cfg = {}
    if preload_voice_cfg.get("preload_on_startup", True):
        try:
            log.info("Preloading offline ASR on startup...")
            preload_engine = get_offline_asr()
            if hasattr(preload_engine, "warmup"):
                log.info("Running offline ASR warmup inference...")
                preload_engine.warmup()
            log.info("Offline ASR preload complete.")
        except Exception as exc:
            log.error("Offline ASR preload failed: %s", exc)

    cam_mgr.start()
    digital_twin_sync.start()
    if bool(demo_cfg.get("virtual_env", {}).get("sync", {}).get("objects", {}).get("sync_on_start", False)):
        try:
            result = digital_twin_sync.sync_scene_objects(objects_cfg, reason="startup")
            log.info("Initial virtual env scene object sync complete (%s targets)", result.get("count", 0))
        except Exception as exc:
            log.warning("Initial virtual env scene object sync skipped: %s", exc)
    arm_mgr.start_pose_polling()
    gripper_monitor.start()
    sensor_monitor.start()

    shutdown_once = threading.Event()

    def shutdown_resources():
        if shutdown_once.is_set():
            return
        shutdown_once.set()
        log.info("Shutting down demo resources...")
        try:
            panel_recorder.stop(session_id=None, timeout_s=2.0)
        except Exception as exc:
            log.warning("Panel recorder shutdown warning: %s", exc)
        try:
            teach_mgr.shutdown()
        except Exception as exc:
            log.warning("Teach manager shutdown warning: %s", exc)
        try:
            validation_recorder.shutdown()
        except Exception as exc:
            log.warning("Validation recorder shutdown warning: %s", exc)
        try:
            gripper_monitor.stop()
        except Exception as exc:
            log.warning("Gripper monitor shutdown warning: %s", exc)
        try:
            sensor_monitor.stop()
        except Exception as exc:
            log.warning("Sensor monitor shutdown warning: %s", exc)
        try:
            digital_twin_sync.stop()
        except Exception as exc:
            log.warning("Digital twin shutdown warning: %s", exc)
        try:
            arm_mgr.request_abort()
            arm_mgr.stop_pose_polling()
            arm_mgr._power_down_servo(reason="shutdown")
            arm_mgr.disconnect()
        except Exception as exc:
            log.warning("Arm shutdown warning: %s", exc)
        try:
            cam_mgr.stop()
        except Exception as exc:
            log.warning("Camera shutdown warning: %s", exc)

    app.demo_shutdown = shutdown_resources

    return app, socketio


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Voice Pick Demo Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    demo_cfg = _apply_module_runtime_overrides(load_config("demo_config.yaml"))
    app, socketio = create_app(demo_cfg)

    def _handle_signal(signum, _frame):
        try:
            signame = signal.Signals(signum).name
        except Exception:
            signame = str(signum)
        log.warning("Received %s, shutting down...", signame)
        shutdown = getattr(app, "demo_shutdown", None)
        if callable(shutdown):
            shutdown()
        raise KeyboardInterrupt

    for signum in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
        if signum is None:
            continue
        try:
            signal.signal(signum, _handle_signal)
        except Exception:
            pass

    log.info("Starting Voice Pick Demo at http://%s:%d", args.host, args.port)
    try:
        socketio.run(app, host=args.host, port=args.port,
                     allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        log.info("Shutdown requested from terminal.")
    finally:
        shutdown = getattr(app, "demo_shutdown", None)
        if callable(shutdown):
            shutdown()
        log.info("Voice Pick Demo stopped.")


if __name__ == "__main__":
    main()
