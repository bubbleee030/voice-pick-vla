#!/usr/bin/env python3
"""Calibrate the claw-cam "arrived on top" detector from recorded episodes.

The claw cam is mounted OFF the tool axis, so "on top of the object" is a
specific (off-center) box position in the claw view — not the image center.
Every recorded episode ends with a vertical descent above the object, so all
claw frames after the arm first reaches the grasp (x,y) are ground-truth
"on top" views at descending heights. This script:

  1. aligns claw frames to trajectory.csv by elapsed time,
  2. labels frames on-top vs approach (xy distance to the episode's final xy),
  3. runs the 3-class YOLO on a sample of both,
  4. reports detection rates and the on-top box-center cluster per z-band,
  5. writes per-object setpoints to data/calibration/claw_arrival.json.

Run (5090 box, rdt env — needs the libstdc++ preload for cv2):
    PYTHONNOUSERSITE=1 LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6.0.34 \
    python scripts/claw_arrival_calib.py
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

import numpy as np

BUNDLE_ROOT = Path(__file__).resolve().parent.parent

# recording dir -> YOLO class name in 3_white_trapezoid.pt
CLASS_MAP = {"board": "board", "butter_knife": "knife", "trapezoid": "trapezoid_white"}


def load_traj(ep: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (elapsed_s, xyz) arrays from trajectory.csv (valid rows only)."""
    ts, xyz = [], []
    with open(ep / "trajectory.csv") as f:
        for row in csv.DictReader(f):
            if row.get("valid") != "1":
                continue
            ts.append(float(row["elapsed_s"]))
            xyz.append([float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])])
    return np.array(ts), np.array(xyz)


def frame_times(n_frames: int, duration: float) -> np.ndarray:
    """Claw frames are saved at a steady rate over the episode."""
    return np.linspace(0.0, duration, n_frames)


def analyze_episode(model, ep: Path, cls_name: str, xy_tol: float,
                    max_on: int, max_app: int, conf_floor: float) -> list[dict]:
    import cv2
    ts, xyz = load_traj(ep)
    if len(ts) < 5:
        return []
    frames = sorted((ep / "claw_rgb").glob("frame_*.jpg"))
    if not frames:
        return []
    ft = frame_times(len(frames), ts[-1])
    end_xy = xyz[-1, :2]
    d_xy = np.linalg.norm(xyz[:, :2] - end_xy, axis=1)
    on_top_t = ts[d_xy <= xy_tol]
    t_arrive = float(on_top_t[0]) if len(on_top_t) else None

    # sample frame indices: on-top spread across the window, approach before it
    samples: list[tuple[int, str]] = []
    if t_arrive is not None:
        on_idx = [i for i, t in enumerate(ft) if t >= t_arrive]
        app_idx = [i for i, t in enumerate(ft) if t < t_arrive]
        for i in np.linspace(0, len(on_idx) - 1, min(max_on, len(on_idx))).astype(int):
            samples.append((on_idx[i], "on_top"))
        for i in np.linspace(0, len(app_idx) - 1, min(max_app, len(app_idx))).astype(int):
            samples.append((app_idx[i], "approach"))

    out = []
    for fi, phase in samples:
        img = cv2.imread(str(frames[fi]))
        if img is None:
            continue
        h, w = img.shape[:2]
        res = model(img, conf=conf_floor, verbose=False)[0]
        best = None
        for b in res.boxes:
            name = model.names[int(b.cls[0])]
            if name != cls_name:
                continue
            conf = float(b.conf[0])
            if best is None or conf > best["conf"]:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                best = {"conf": conf,
                        "cx": (x1 + x2) / 2 / w, "cy": (y1 + y2) / 2 / h,
                        "bw": (x2 - x1) / w, "bh": (y2 - y1) / h,
                        "clipped": x1 <= 2 or y1 <= 2 or x2 >= w - 2 or y2 >= h - 2}
        z = float(xyz[np.argmin(np.abs(ts - ft[fi])), 2])
        out.append({"ep": ep.name, "frame": fi, "phase": phase, "z": z,
                    "det": best is not None, **(best or {})})
    return out


def zband(z: float) -> str:
    return "hover(z>300)" if z > 300 else ("mid(200-300)" if z > 200 else "low(<200)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", default="data/recordings")
    ap.add_argument("--model", default="data/models/yolo/3_white_trapezoid.pt")
    ap.add_argument("--objects", default="board,butter_knife,trapezoid")
    ap.add_argument("--xy-tol", type=float, default=8.0, help="mm: 'on top' xy radius")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-on", type=int, default=10, help="on-top samples per episode")
    ap.add_argument("--max-app", type=int, default=5, help="approach samples per episode")
    ap.add_argument("--out", default="data/calibration/claw_arrival.json")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(BUNDLE_ROOT / args.model))

    setpoints = {}
    for obj in args.objects.split(","):
        cls_name = CLASS_MAP[obj]
        rows = []
        for ep in sorted((BUNDLE_ROOT / args.recordings / obj).glob("episode_*")):
            rows += analyze_episode(model, ep, cls_name, args.xy_tol,
                                    args.max_on, args.max_app, args.conf)
        on = [r for r in rows if r["phase"] == "on_top"]
        app = [r for r in rows if r["phase"] == "approach"]
        on_det = [r for r in on if r["det"]]
        print(f"\n=== {obj} (YOLO class '{cls_name}') — {len(rows)} frames from "
              f"{len(set(r['ep'] for r in rows))} eps ===")
        print(f"  det rate: on_top {len(on_det)}/{len(on)}   "
              f"approach {sum(r['det'] for r in app)}/{len(app)}")
        for band in ("hover(z>300)", "mid(200-300)", "low(<200)"):
            sel = [r for r in on_det if zband(r["z"]) == band]
            if not sel:
                print(f"  {band:<14} no detections")
                continue
            cx = np.array([r["cx"] for r in sel]); cy = np.array([r["cy"] for r in sel])
            bw = np.array([r["bw"] for r in sel]); cf = np.array([r["conf"] for r in sel])
            clip = sum(r["clipped"] for r in sel)
            print(f"  {band:<14} n={len(sel):<3} cx={cx.mean():.3f}±{cx.std():.3f} "
                  f"cy={cy.mean():.3f}±{cy.std():.3f} w={bw.mean():.3f} "
                  f"conf={cf.mean():.2f} clipped={clip}/{len(sel)}")
        # setpoint = on-top cluster at hover band (fall back to all on-top dets)
        sel = [r for r in on_det if r["z"] > 300] or on_det
        if sel:
            cx = np.array([r["cx"] for r in sel]); cy = np.array([r["cy"] for r in sel])
            setpoints[obj] = {
                "yolo_class": cls_name,
                "setpoint_cx": round(float(cx.mean()), 4),
                "setpoint_cy": round(float(cy.mean()), 4),
                "tol_cx": round(float(max(3 * cx.std(), 0.04)), 4),
                "tol_cy": round(float(max(3 * cy.std(), 0.04)), 4),
                "n_samples": len(sel),
            }

    out_path = BUNDLE_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(setpoints, indent=2))
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
