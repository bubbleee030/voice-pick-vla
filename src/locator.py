"""Side-cam locator — stage-1 coarse (x,y) without the VLA (§3.6).

One YOLO shot on a cam1 frame taken at the ready pose, box center mapped to
table (x,y) mm through the per-object affine in
data/calibration/sidecam_locator.json (fit from recordings by
scripts/sidecam_locator_calib.py; MUST be refit when the rig moves — §3.7).

Accuracy demands are soft: the claw servo takes over from hover and sees
~0.5 m of table, so this stage only needs ±100 mm. cam2 is not used (cannot
see the flat objects). KNOWN LIMIT: butter_knife's affine has only ever seen
x≈490 — place the knife near x≈490 or add points to the refit CSV.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCATOR_JSON = PROJECT_ROOT / "data/calibration/sidecam_locator.json"
DEFAULT_MODEL = PROJECT_ROOT / "data/models/yolo/3_white_trapezoid.pt"


@dataclass
class LocateResult:
    x_mm: float
    y_mm: float
    conf: float
    box: dict[str, Any]     # normalized cam1 box {label, conf, x1, y1, x2, y2}


class SideCamLocator:
    def __init__(self, locator_json: str | Path = LOCATOR_JSON,
                 model_path: str | Path = DEFAULT_MODEL,
                 conf_floor: float = 0.25):
        self.affines: dict[str, dict[str, Any]] = json.loads(Path(locator_json).read_text())
        self.model_path = str(model_path)
        self.conf_floor = float(conf_floor)   # cam1 view is far — lower than the claw's 0.5
        self._model = None                    # lazy: YOLO import only when first used

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import YOLO  # noqa: PLC0415
            self._model = YOLO(self.model_path)
        return self._model

    def locate(self, frame_bgr: np.ndarray, object_key: str) -> LocateResult | None:
        """cam1 frame -> table (x,y) mm, or None if no confident box."""
        entry = self.affines[object_key]
        cls_name = entry["yolo_class"]
        model = self._ensure_model()
        h, w = frame_bgr.shape[:2]
        res = model(frame_bgr, conf=self.conf_floor, verbose=False)[0]
        best = None
        for b in res.boxes:
            if model.names[int(b.cls[0])] != cls_name:
                continue
            conf = float(b.conf[0])
            if best is None or conf > best[0]:
                best = (conf, [float(v) for v in b.xyxy[0]])
        if best is None:
            return None
        conf, (x1, y1, x2, y2) = best
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        W = np.array(entry["weights_3x2"], dtype=float)          # (3, 2)
        x_mm, y_mm = np.array([cx, cy, 1.0]) @ W
        return LocateResult(float(x_mm), float(y_mm), conf,
                            {"label": cls_name, "conf": conf,
                             "x1": x1 / w, "y1": y1 / h, "x2": x2 / w, "y2": y2 / h})
