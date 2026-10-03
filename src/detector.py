"""
YOLO Object Detection + RealSense Depth → 3D Position.

Usage:
    # On Mac with RealSense cameras connected:
    from src.detector import ObjectDetector
    det = ObjectDetector()
    results = det.detect_from_cameras(target_class="apple")
    # results: {"class": "apple", "confidence": 0.92, "robot_xyz": [x, y, z], ...}

For demo mode (no RealSense), can also detect from a single image file.
"""

import numpy as np
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("[WARN] ultralytics not installed. Run: pip install ultralytics")

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None
    print("[WARN] pyrealsense2 not installed. Camera functions unavailable.")

from src.utils import load_config, PROJECT_ROOT


class ObjectDetector:
    """
    Detect objects using YOLO and compute 3D position using RealSense depth.
    Supports pretrained COCO model (no fine-tuning needed to start).
    """

    def __init__(self, yolo_model: str = "yolov8s.pt", config_path: str = None):
        """
        Args:
            yolo_model: Path to YOLO weights or model name (e.g., "yolov8s.pt").
                        Will auto-download from Ultralytics if not found locally.
            config_path: Path to camera_calibration.yaml (or None for default).
        """
        if YOLO is None:
            raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")

        # Load YOLO model
        model_path = PROJECT_ROOT / "models" / "yolo" / yolo_model
        if model_path.exists():
            self.yolo = YOLO(str(model_path))
        else:
            # Auto-download pretrained model
            print(f"[detector] Downloading pretrained {yolo_model}...")
            self.yolo = YOLO(yolo_model)

        # Load configs
        self.objects_cfg = load_config("objects.yaml")
        self.cam_cfg = load_config("camera_calibration.yaml")

        # Build class-name → object-key mapping from COCO classes
        self._build_class_map()

        # RealSense runtime handles, keyed by camera id.
        self._camera_handles = {}

    def _build_class_map(self):
        """Build mapping from YOLO COCO class names to our object keys."""
        self.yolo_to_object = {}
        for obj_key, obj_def in self.objects_cfg.get("classes", {}).items():
            yolo_class = obj_def.get("yolo_class", obj_key)
            if yolo_class not in self.yolo_to_object:
                self.yolo_to_object[yolo_class] = []
            self.yolo_to_object[yolo_class].append(obj_key)

    def close(self):
        """Stop all active RealSense pipelines."""
        for _, handle in list(self._camera_handles.items()):
            pipeline = handle.get("pipeline")
            if pipeline is None:
                continue
            try:
                pipeline.stop()
            except Exception:
                pass
        self._camera_handles.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _camera_cfg(self, camera_id: int) -> dict:
        return self.cam_cfg.get(f"camera{camera_id}", {}) or {}

    def _resolve_serial(self, camera_id: int, serial_override: str = None) -> str:
        if serial_override and str(serial_override).strip():
            return str(serial_override).strip()
        cfg = self._camera_cfg(camera_id)
        return str(cfg.get("serial") or "").strip()

    def _start_camera(self, camera_id: int, serial_override: str = None) -> dict:
        if rs is None:
            raise RuntimeError("pyrealsense2 not installed")

        serial = self._resolve_serial(camera_id, serial_override)
        handle = self._camera_handles.get(camera_id)
        if handle is not None:
            if serial and handle.get("serial") and serial != handle.get("serial"):
                try:
                    handle["pipeline"].stop()
                except Exception:
                    pass
                self._camera_handles.pop(camera_id, None)
            else:
                return handle

        cfg = self._camera_cfg(camera_id)
        intr = cfg.get("intrinsics", {}) or {}
        width = int(intr.get("width") or 640)
        height = int(intr.get("height") or 480)
        fps = int(cfg.get("fps") or 30)

        pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        try:
            profile = pipeline.start(config)
            for _ in range(6):
                pipeline.wait_for_frames(timeout_ms=8000)
        except Exception as exc:
            try:
                pipeline.stop()
            except Exception:
                pass
            if serial:
                raise RuntimeError(f"Failed to start camera{camera_id} serial={serial}: {exc}") from exc
            raise RuntimeError(f"Failed to start camera{camera_id}: {exc}") from exc

        align = rs.align(rs.stream.color)
        device = profile.get_device()
        actual_serial = serial
        try:
            actual_serial = device.get_info(rs.camera_info.serial_number)
        except Exception:
            pass

        handle = {
            "pipeline": pipeline,
            "align": align,
            "serial": actual_serial,
            "camera_id": camera_id,
        }
        self._camera_handles[camera_id] = handle
        return handle

    @staticmethod
    def _frame_intrinsics(color_frame) -> dict:
        profile = color_frame.profile.as_video_stream_profile()
        intr = profile.get_intrinsics()
        return {
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "cx": float(intr.ppx),
            "cy": float(intr.ppy),
            "width": int(intr.width),
            "height": int(intr.height),
        }

    def capture_aligned(self, camera_id: int, serial_override: str = None) -> tuple:
        """Capture one aligned color/depth frame pair and runtime intrinsics."""
        handle = self._start_camera(camera_id, serial_override)
        frames = handle["pipeline"].wait_for_frames(timeout_ms=8000)
        frames = handle["align"].process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError(f"camera{camera_id} missing color/depth frame")

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        intrinsics = self._frame_intrinsics(color_frame)
        return color, depth, intrinsics, handle.get("serial", "")

    def detect_from_cameras(
        self,
        target_object: str = None,
        confidence_threshold: float = 0.3,
        camera_ids: tuple = (1, 2),
        serial_overrides: dict = None,
    ):
        """
        Capture live frames from one or more cameras and return the best detection.

        Returns:
            Best detection dict (includes robot_xyz_mm) or None.
        """
        if rs is None:
            raise RuntimeError("pyrealsense2 not installed")

        detections_all = []
        overrides = serial_overrides or {}

        for camera_id in camera_ids:
            serial_override = overrides.get(camera_id)
            try:
                color, depth, intrinsics, serial_used = self.capture_aligned(camera_id, serial_override)
            except Exception as exc:
                print(f"[detector] camera{camera_id} unavailable: {exc}")
                continue

            detections = self.detect_full(
                color_image=color,
                depth_image=depth,
                intrinsics=intrinsics,
                camera_id=camera_id,
                target_object=target_object,
                confidence_threshold=confidence_threshold,
            )
            for det in detections:
                det["camera_id"] = camera_id
                det["camera_serial"] = serial_used
            detections_all.extend(detections)

        valid = [d for d in detections_all if d.get("robot_xyz_mm") is not None]
        if not valid:
            return None

        valid.sort(key=lambda d: d["confidence"], reverse=True)
        return valid[0]

    def detect_from_image(self, image: np.ndarray, target_object: str = None,
                          confidence_threshold: float = 0.3) -> list:
        """
        Detect objects in a single RGB image (no depth/3D).

        Args:
            image: BGR numpy array.
            target_object: If set, only return detections matching this object key.
            confidence_threshold: Minimum confidence to keep.

        Returns:
            List of dicts: [{"class": "apple", "confidence": 0.92,
                             "bbox": [x1,y1,x2,y2], "center_px": (cx,cy)}, ...]
        """
        results = self.yolo(image, verbose=False)[0]
        detections = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < confidence_threshold:
                continue

            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]

            # Map COCO class to our object key
            matched_objects = self.yolo_to_object.get(cls_name, [])
            if not matched_objects:
                # Unknown class, still report it
                obj_key = cls_name
            else:
                obj_key = matched_objects[0]  # First match

            # Filter by target if specified
            if target_object and obj_key != target_object:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            detections.append({
                "class": obj_key,
                "yolo_class": cls_name,
                "confidence": conf,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "center_px": (int(cx), int(cy)),
            })

        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def detect_with_depth(self, color_image: np.ndarray, depth_image: np.ndarray,
                          intrinsics: dict, target_object: str = None,
                          confidence_threshold: float = 0.3) -> list:
        """
        Detect objects and compute 3D position in camera frame.

        Args:
            color_image: BGR numpy array.
            depth_image: Depth image as uint16 (mm units).
            intrinsics: Camera intrinsics dict with fx, fy, cx, cy.
            target_object: Filter to this object only.
            confidence_threshold: Minimum confidence.

        Returns:
            List of dicts with added "camera_xyz_m" field.
        """
        detections = self.detect_from_image(color_image, target_object, confidence_threshold)

        for det in detections:
            cx, cy = det["center_px"]
            x1, y1, x2, y2 = det["bbox"]

            # Get depth at center - use median of small region for robustness
            h, w = depth_image.shape[:2]
            rx = max(0, min(5, (x2 - x1) // 8))
            ry = max(0, min(5, (y2 - y1) // 8))
            y_lo = max(0, cy - ry)
            y_hi = min(h, cy + ry + 1)
            x_lo = max(0, cx - rx)
            x_hi = min(w, cx + rx + 1)
            depth_region = depth_image[y_lo:y_hi, x_lo:x_hi]
            valid_depths = depth_region[depth_region > 0]

            if len(valid_depths) == 0:
                det["camera_xyz_m"] = None
                det["depth_mm"] = None
                continue

            depth_mm = float(np.median(valid_depths))
            depth_m = depth_mm / 1000.0

            # Deproject pixel to 3D camera coordinates
            fx = intrinsics["fx"]
            fy = intrinsics["fy"]
            ppx = intrinsics["cx"]
            ppy = intrinsics["cy"]

            x_cam = (cx - ppx) * depth_m / fx
            y_cam = (cy - ppy) * depth_m / fy
            z_cam = depth_m

            det["camera_xyz_m"] = [float(x_cam), float(y_cam), float(z_cam)]
            det["camera_xyz_mm"] = [float(x_cam * 1000.0), float(y_cam * 1000.0), float(z_cam * 1000.0)]
            det["depth_mm"] = float(depth_mm)

        return detections

    def camera_to_robot(self, camera_xyz: list, camera_id: int = 1) -> list:
        """
        Transform 3D point from camera frame to robot base frame.

        Args:
            camera_xyz: [x, y, z] in mm, camera frame.
            camera_id: Which camera (1 or 2).

        Returns:
            [x, y, z] in mm, robot base frame.
        """
        key = f"camera{camera_id}_to_robot"
        he_cfg = self.cam_cfg.get("hand_eye", {}).get(key, {})

        if not he_cfg.get("calibrated", False):
            print(f"[WARN] Camera {camera_id} not calibrated. Using raw camera coords.")
            return camera_xyz

        matrix = np.array(he_cfg["matrix"]).reshape(4, 4)
        point = np.array([*camera_xyz, 1.0])
        robot_point = matrix @ point
        return robot_point[:3].tolist()

    def detect_full(self, color_image: np.ndarray, depth_image: np.ndarray,
                    intrinsics: dict, camera_id: int = 1,
                    target_object: str = None,
                    confidence_threshold: float = 0.3) -> list:
        """
        Full pipeline: detect → depth → camera 3D → robot 3D.

        Returns:
            List of dicts with "robot_xyz_m" and "robot_xyz_mm" fields.
        """
        detections = self.detect_with_depth(
            color_image, depth_image, intrinsics, target_object, confidence_threshold
        )

        for det in detections:
            if det.get("camera_xyz_mm") is None:
                det["robot_xyz_m"] = None
                det["robot_xyz_mm"] = None
                continue

            robot_xyz_mm = self.camera_to_robot(det["camera_xyz_mm"], camera_id)
            det["robot_xyz_mm"] = robot_xyz_mm
            det["robot_xyz_m"] = [v / 1000.0 for v in robot_xyz_mm]

        return detections

    def annotate_image(self, image: np.ndarray, detections: list) -> np.ndarray:
        """
        Draw bounding boxes and labels on an image for display/demo.

        Args:
            image: BGR numpy array.
            detections: Output from detect_from_image or detect_full.

        Returns:
            Annotated BGR image.
        """
        annotated = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            label = f"{det['class']} {conf:.2f}"

            # Color based on object class (consistent colors)
            color_hash = hash(det["class"]) % 360
            hsv = np.array([[[color_hash // 2, 200, 255]]], dtype=np.uint8)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 2)

            # Draw label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), bgr, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Draw depth/position info if available
            if det.get("robot_xyz_mm"):
                xyz = det["robot_xyz_mm"]
                pos_text = f"({xyz[0]:.0f}, {xyz[1]:.0f}, {xyz[2]:.0f})mm"
                cv2.putText(annotated, pos_text, (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1)

        return annotated
