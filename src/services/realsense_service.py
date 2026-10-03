from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.runtime_config import module_enabled

RS_AVAILABLE = False
try:
    import pyrealsense2 as rs

    RS_AVAILABLE = True
except ImportError:
    rs = None


def _placeholder(text: str, w: int = 424, h: int = 240) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (20, 22, 26)
    cv2.putText(img, text, (18, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 170), 2)
    return img


class SingleCameraStream:
    def __init__(self, cam_id: str, serial: str, w: int, h: int, fps: int, enable_depth: bool, label: str):
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
        self.last_rgb: np.ndarray | None = None
        self.last_depth: np.ndarray | None = None
        self.last_error: str | None = None
        self.lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        if not RS_AVAILABLE:
            self.last_error = "pyrealsense2 unavailable"
            return False
        self.stop()
        pipeline = None
        try:
            pipeline = rs.pipeline()
            cfg = rs.config()
            if self.serial:
                cfg.enable_device(self.serial)
            cfg.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, self.fps)
            if self.enable_depth:
                cfg.enable_stream(rs.stream.depth)
            pipeline.start(cfg)
            self.pipeline = pipeline
            if self.enable_depth:
                self.align = rs.align(rs.stream.color)
            self.running = True
            self.last_error = None
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            return True
        except Exception as exc:
            self.last_error = str(exc)
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass
            self.pipeline = None
            self.align = None
            self.running = False
            return False

    def _capture_loop(self):
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                if self.align is not None:
                    frames = self.align.process(frames)
                color_frame = frames.get_color_frame()
                if color_frame:
                    rgb = np.asanyarray(color_frame.get_data())
                    with self.lock:
                        self.last_rgb = rgb
                if self.enable_depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        depth_arr = np.asanyarray(depth_frame.get_data())
                        depth_color = cv2.applyColorMap(
                            cv2.convertScaleAbs(depth_arr, alpha=0.03),
                            cv2.COLORMAP_TURBO,
                        )
                        with self.lock:
                            self.last_depth = depth_color
            except Exception as exc:
                self.last_error = str(exc)
                time.sleep(0.5)

    def get_rgb_jpeg(self) -> bytes:
        with self.lock:
            frame = None if self.last_rgb is None else self.last_rgb.copy()
        if frame is None:
            frame = _placeholder(f"{self.label} RGB - No Signal", self.w, self.h)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        return buf.tobytes() if ok else b""

    def get_depth_jpeg(self) -> bytes:
        with self.lock:
            frame = None if self.last_depth is None else self.last_depth.copy()
        if not self.enable_depth:
            frame = _placeholder(f"{self.label} Depth - Disabled", self.w, self.h)
        elif frame is None:
            frame = _placeholder(f"{self.label} Depth - No Signal", self.w, self.h)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        return buf.tobytes() if ok else b""

    def stop(self):
        self.running = False
        pipeline, self.pipeline = self.pipeline, None
        if pipeline:
            try:
                pipeline.stop()
            except Exception:
                pass
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self.align = None
        with self.lock:
            self.last_rgb = None
            self.last_depth = None


class RealSenseService:
    def __init__(self, demo_cfg: dict[str, Any]):
        self.demo_cfg = demo_cfg
        self.camera_cfg = demo_cfg.get("cameras", {})
        self.cams: dict[str, SingleCameraStream] = {}
        self._started = False
        self._lock = threading.RLock()
        self._build_streams()

    def _build_streams(self):
        self.cams = {}
        w = int(self.camera_cfg.get("resolution_w", 424))
        h = int(self.camera_cfg.get("resolution_h", 240))
        fps = int(self.camera_cfg.get("fps", 15))
        for cam_key in ("cam1", "cam2"):
            if not module_enabled(self.demo_cfg, cam_key, default=True):
                continue
            node = self.camera_cfg.get(cam_key, {})
            if not node.get("enabled", False):
                continue
            self.cams[cam_key] = SingleCameraStream(
                cam_id=cam_key,
                serial=str(node.get("serial", "")),
                w=w,
                h=h,
                fps=fps,
                enable_depth=bool(node.get("enable_depth", True)),
                label=str(node.get("label", cam_key)),
            )

    def start(self) -> bool:
        with self._lock:
            if self._started:
                return all(cam.running for cam in self.cams.values())
            self._started = True
            init_delay = float(self.camera_cfg.get("init_delay_s", 2.0))
            attempts = max(1, int(self.camera_cfg.get("reconnect_attempts", 4)))
            retry_delay = max(0.0, float(self.camera_cfg.get("reconnect_delay_s", 1.0)))
            for index, cam in enumerate(self.cams.values()):
                if index > 0 and init_delay > 0:
                    time.sleep(init_delay)
                for attempt in range(attempts):
                    if cam.start():
                        break
                    if attempt + 1 < attempts and retry_delay > 0:
                        time.sleep(retry_delay)
            return all(cam.running for cam in self.cams.values())

    def stop(self):
        with self._lock:
            for cam in self.cams.values():
                cam.stop()
            self._started = False

    def get_jpeg(self, cam_key: str, stream: str) -> bytes:
        cam = self.cams.get(cam_key)
        w = int(self.camera_cfg.get("resolution_w", 424))
        h = int(self.camera_cfg.get("resolution_h", 240))
        if cam is None:
            frame = _placeholder(f"{cam_key} - Not Configured", w, h)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
            return buf.tobytes() if ok else b""
        return cam.get_depth_jpeg() if stream == "depth" else cam.get_rgb_jpeg()

    def status(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key in ("cam1", "cam2"):
            cam = self.cams.get(key)
            if cam is None:
                payload[key] = {"running": False, "depth_enabled": False, "label": key, "last_error": None}
            else:
                payload[key] = {
                    "running": cam.running,
                    "depth_enabled": cam.enable_depth,
                    "label": cam.label,
                    "last_error": cam.last_error,
                    "serial": cam.serial,
                }
        return payload
