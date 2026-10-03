from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.runtime_config import module_enabled

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class UVC_Camera_Thread:
    """Background UVC capture thread tuned for low latency.

    Auto-recovers from the claw's recurring USB drop (kernel: `usb 3-4.3.2
    disconnect, error -71` on the nested hub): after a full re-enumeration the
    old capture handle is permanently dead, so once reads fail for
    REOPEN_AFTER_S we release and reopen the (stable by-id) source. While the
    device is physically gone the reopen just fails and retries; the moment it
    re-enumerates, capture resumes without a manual restart.
    """

    REOPEN_AFTER_S = 4.0

    def __init__(self, source: int | str, capture_w: int, capture_h: int, fps: int):
        self.source = source
        self.capture_w = capture_w
        self.capture_h = capture_h
        self.fps = fps
        self.backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        self.actual_mode: tuple[int, int, float] | None = None
        self.mode_warning: str | None = None
        self.cap = self._open()

        self.ret, self.frame = self.cap.read()
        self.frame_ts = time.time() if self.ret else 0.0
        self.running = True
        self.last_error: str | None = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.source, self.backend)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_h)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Read back what the driver actually negotiated. If it differs from the
        # request the mode is NON-NATIVE — that is what stalls the stream and
        # spins up reopen churn. Surface it (status(), stream overlay) instead of
        # failing silently, so it can be fixed to a listed mode.
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        af = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.actual_mode = (aw, ah, af)
        if aw and ah and (aw, ah) != (self.capture_w, self.capture_h):
            self.mode_warning = (f"requested {self.capture_w}x{self.capture_h}@{self.fps} but "
                                 f"camera gave {aw}x{ah}@{af:.0f} — non-native mode, expect drops")
        else:
            self.mode_warning = None
        return cap

    def _reopen(self):
        self.last_error = "reopening camera"
        try:
            self.cap.release()
        except Exception:
            pass
        self.cap = self._open()

    def update(self):
        last_ok = time.time()
        while self.running:
            if not self.cap.isOpened():
                self.last_error = "camera not opened"
                if time.time() - last_ok > self.REOPEN_AFTER_S:
                    self._reopen()
                    last_ok = time.time()   # avoid a tight reopen loop
                time.sleep(0.25)
                continue
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.last_error = "read failed"
                if time.time() - last_ok > self.REOPEN_AFTER_S:
                    self._reopen()
                    last_ok = time.time()
                time.sleep(0.02)
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame
                self.frame_ts = time.time()
                self.last_error = None
            last_ok = time.time()

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self.lock:
            if self.frame is None:
                return bool(self.ret), None
            return bool(self.ret), self.frame.copy()

    def frame_age(self) -> float | None:
        with self.lock:
            if not self.frame_ts:
                return None
            return time.time() - self.frame_ts

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


def _placeholder(text: str, w: int, h: int) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (18, 19, 24)
    cv2.putText(img, text, (18, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 190), 2)
    return img


class ClawCameraService:
    def __init__(self, demo_cfg: dict[str, Any]):
        self.demo_cfg = demo_cfg
        self.claw_cfg = demo_cfg.get("cameras", {}).get("claw", {})
        self.worker: UVC_Camera_Thread | None = None
        self.detect_thread: threading.Thread | None = None
        self.detect_stop = threading.Event()
        self.running = False
        self.last_error: str | None = None
        self.active_profile_name = str(self.claw_cfg.get("performance_profile", "balanced"))
        self.latest_detection_stub: list[dict[str, Any]] = []
        # Normalized (0..1) xyxy boxes from the last YOLO pass + when it finished.
        # The stream draws these onto the FRESH capture; it never displays the
        # (stale) frame YOLO ran on, so display latency is decoupled from
        # inference latency.
        self.latest_boxes_norm: list[dict[str, Any]] = []
        self.latest_detect_ts: float = 0.0
        self.detection_stale_s = float(self.claw_cfg.get("detection_stale_s", 1.0))
        self._model: Any = None
        self._model_error: str | None = None
        self._lock = threading.Lock()

    def _active_profile(self) -> dict[str, Any]:
        profiles = self.claw_cfg.get("profiles", {})
        active_name = str(self.claw_cfg.get("performance_profile", "balanced"))
        base = profiles.get(active_name) or profiles.get("balanced") or {}
        override = self.claw_cfg.get("overrides", {}).get(active_name, {})
        merged = dict(base)
        merged.update(override)
        self.active_profile_name = active_name
        return merged

    def _resolve_model_path(self) -> Path:
        raw = str(self.claw_cfg.get("model_path", "data/models/yolo/yolo12sbest.pt"))
        path = Path(raw)
        if path.is_absolute():
            return path
        return (Path(__file__).resolve().parent.parent.parent / raw).resolve()

    def _ensure_model(self) -> bool:
        if not self.claw_cfg.get("enable_yolo", False):
            return False
        if self._model is not None:
            return True
        if YOLO is None:
            self._model_error = "ultralytics not installed"
            return False
        model_path = self._resolve_model_path()
        if not model_path.exists():
            self._model_error = f"model not found: {model_path}"
            return False
        try:
            self._model = YOLO(str(model_path))
            self._model_error = None
            return True
        except Exception as exc:
            self._model_error = str(exc)
            return False

    def _detect_loop(self):
        while not self.detect_stop.is_set():
            profile = self._active_profile()
            if not self._ensure_model():
                time.sleep(0.5)
                continue
            frame = self.get_latest_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            infer_w = int(profile.get("infer_w", 424))
            infer_h = int(profile.get("infer_h", 240))
            conf = float(self.claw_cfg.get("conf", 0.6))
            infer_frame = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
            try:
                results = self._model(infer_frame, conf=conf, verbose=False)
                detections = []
                boxes_norm = []
                for box in results[0].boxes:
                    cls_idx = int(box.cls[0])
                    label = self._model.names.get(cls_idx, str(cls_idx))
                    score = float(box.conf[0]) if box.conf is not None else 0.0
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                    boxes_norm.append({
                        "label": label,
                        "conf": score,
                        "x1": x1 / infer_w,
                        "y1": y1 / infer_h,
                        "x2": x2 / infer_w,
                        "y2": y2 / infer_h,
                    })
                    detections.append({"label": label, "conf": round(score, 4)})
                with self._lock:
                    self.latest_detection_stub = detections
                    self.latest_boxes_norm = boxes_norm
                    self.latest_detect_ts = time.time()
                    self.last_error = None
            except Exception as exc:
                self.last_error = str(exc)
            time.sleep(float(profile.get("infer_interval_s", 0.12)))

    def start(self):
        if self.running or not module_enabled(self.demo_cfg, "claw_cam", default=True):
            return
        if not self.claw_cfg.get("enabled", False):
            return
        profile = self._active_profile()
        source = self.claw_cfg.get("device")
        if source in (None, "", "auto"):
            source = self.claw_cfg.get("source", 0)
        try:
            self.worker = UVC_Camera_Thread(
                source=source,
                capture_w=int(profile.get("capture_w", 640)),
                capture_h=int(profile.get("capture_h", 360)),
                fps=int(profile.get("stream_fps", 20)),
            )
            self.running = True
            self.last_error = None
            self.detect_stop.clear()
            if self.claw_cfg.get("enable_yolo", False):
                self.detect_thread = threading.Thread(target=self._detect_loop, daemon=True)
                self.detect_thread.start()
        except Exception as exc:
            self.last_error = str(exc)
            self.running = False

    def stop(self):
        self.detect_stop.set()
        if self.detect_thread is not None:
            self.detect_thread.join(timeout=1.0)
        if self.worker is not None:
            self.worker.stop()
        self.detect_thread = None
        self.worker = None
        self.running = False

    def get_latest_frame(self) -> np.ndarray | None:
        if self.worker is None:
            return None
        ok, frame = self.worker.read()
        if not ok or frame is None:
            self.last_error = self.worker.last_error or "no frame"
            return None
        return frame

    def get_detections(self) -> tuple[list[dict[str, Any]], float | None]:
        """Latest normalized boxes + their age in seconds (None if never detected)."""
        with self._lock:
            boxes = list(self.latest_boxes_norm)
            ts = self.latest_detect_ts
        return boxes, (time.time() - ts) if ts else None

    def get_stream_frame(self) -> np.ndarray:
        profile = self._active_profile()
        frame = self.get_latest_frame()
        stream_w = int(profile.get("stream_w", 424))
        stream_h = int(profile.get("stream_h", 240))
        if frame is None:
            return _placeholder("Claw Cam - No Signal", stream_w, stream_h)
        boxes, det_age = self.get_detections()
        with self._lock:
            detections = list(self.latest_detection_stub)
        # Always stream the freshest capture; overlay boxes only if recent.
        resized = cv2.resize(frame, (stream_w, stream_h), interpolation=cv2.INTER_LINEAR)
        det_fresh = det_age is not None and det_age <= self.detection_stale_s
        if det_fresh:
            for b in boxes:
                p1 = (int(b["x1"] * stream_w), int(b["y1"] * stream_h))
                p2 = (int(b["x2"] * stream_w), int(b["y2"] * stream_h))
                cv2.rectangle(resized, p1, p2, (40, 255, 220), 2)
                cv2.putText(
                    resized,
                    f"{b['label']} {b['conf']:.2f}",
                    (p1[0], max(14, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (40, 255, 220),
                    1,
                )
        cv2.putText(
            resized,
            f"{self.claw_cfg.get('label', 'Claw Cam')} | {self.active_profile_name}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 255, 220),
            2,
        )
        if detections and det_fresh:
            summary = ", ".join([f"{d['label']}:{d['conf']:.2f}" for d in detections[:4]])
            cv2.putText(
                resized,
                summary[:100],
                (12, max(36, stream_h - 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 220, 120),
                1,
            )
        return resized

    def get_jpeg(self) -> bytes:
        frame = self.get_stream_frame()
        profile = self._active_profile()
        quality = int(profile.get("jpeg_quality", 50))
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return buf.tobytes() if ok else b""

    def status(self) -> dict[str, Any]:
        frame = self.get_latest_frame()
        _, det_age = self.get_detections()
        return {
            "enabled": bool(self.claw_cfg.get("enabled", False)),
            "running": self.running and frame is not None,
            "frame_age_s": self.worker.frame_age() if self.worker else None,
            "detect_age_s": det_age,
            "label": self.claw_cfg.get("label", "Claw Cam"),
            "profile": self.active_profile_name,
            "source": self.claw_cfg.get("device", self.claw_cfg.get("source", 0)),
            "capture_mode": (self.worker.actual_mode if self.worker else None),
            "mode_warning": (self.worker.mode_warning if self.worker else None),
            "last_error": self.last_error or self._model_error
                          or (self.worker.mode_warning if self.worker else None)
                          or (self.worker.last_error if self.worker else None),
            "yolo_enabled": bool(self.claw_cfg.get("enable_yolo", False)),
            "detections": self.latest_detection_stub,
        }
