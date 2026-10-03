#!/usr/bin/env python3
"""Calibrate the SIDE-CAM coarse locator: YOLO box center in cam1/cam2 (at the ready
pose) -> table (x, y) mm, fitted from recorded episodes (frame 0 = ready view,
trajectory end = ground-truth grasp x,y).

This is stage 1 of the classical pick pipeline ("AI-like hardcode"):
  side-cam YOLO -> rough (x,y) -> move at hover -> claw-cam servo to the on-top
  setpoint (claw_arrival.json) -> scripted descend.

Fits a per-object affine map [cx1,cy1,cx2,cy2, 1] -> (x, y) via least squares and
reports LEAVE-ONE-OUT error (honest small-sample estimate). Objects differ in height,
so per-object fits absorb the height parallax. Writes
data/calibration/sidecam_locator.json.

Run (5090, rdt env):
    PYTHONNOUSERSITE=1 LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6.0.34 \
    python scripts/sidecam_locator_calib.py
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

import numpy as np

BUNDLE_ROOT = Path(__file__).resolve().parent.parent
CLASS_MAP = {"board": "board", "butter_knife": "knife", "trapezoid": "trapezoid_white"}


def episode_gt(ep: Path) -> tuple[float, float]:
    last = None
    with open(ep / "trajectory.csv") as f:
        for row in csv.DictReader(f):
            if row.get("valid") == "1":
                last = row
    return float(last["x_mm"]), float(last["y_mm"])


def best_box(model, img, cls_name: str, conf_floor: float):
    res = model(img, conf=conf_floor, verbose=False)[0]
    best = None
    h, w = img.shape[:2]
    for b in res.boxes:
        if model.names[int(b.cls[0])] != cls_name:
            continue
        conf = float(b.conf[0])
        if best is None or conf > best[2]:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            best = ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, conf)
    return best


def fit_affine(F: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Least-squares W (5x2): [cx1,cy1,cx2,cy2,1] @ W = (x,y)."""
    A = np.hstack([F, np.ones((len(F), 1))])
    W, *_ = np.linalg.lstsq(A, Y, rcond=None)
    return W


def loo_errors(F: np.ndarray, Y: np.ndarray) -> np.ndarray:
    errs = []
    for i in range(len(F)):
        m = np.ones(len(F), bool); m[i] = False
        W = fit_affine(F[m], Y[m])
        pred = np.hstack([F[i], 1.0]) @ W
        errs.append(pred - Y[i])
    return np.array(errs)


def refit_from_points(args) -> int:
    """DEMO-ROOM REFIT: the rig moved, cam1's pose changed, every affine is stale.
    Procedure at the new room (10 min):
      1. place ONE object (trapezoid is easiest) at 4+ spread-out spots; for each:
         jog the arm directly over it (that pose's x,y = ground truth), save a cam1
         frame, and append a CSV row:  object,image_path,true_x,true_y
      2. run:  sidecam_locator_calib.py --points-csv demo_room_points.csv \
                   --transfer-old data/calibration/sidecam_locator.json
    Objects present in the CSV are refit directly. Objects NOT in the CSV get the
    camera shift TRANSFERRED: new_W = old_W + (new_W_ref - old_W_ref) of a refit
    reference object — valid because all maps share the same camera motion, and the
    claw servo mops up the residual (stage 1 only needs ~±100 mm: the claw sees
    ~0.5 m of table at hover)."""
    import cv2
    from ultralytics import YOLO
    model = YOLO(str(BUNDLE_ROOT / args.model))
    old = json.loads((BUNDLE_ROOT / args.transfer_old).read_text()) if args.transfer_old else {}

    pts: dict[str, list] = {}
    with open(args.points_csv) as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith("#"):
                continue
            obj, img_path, tx, ty = row[0].strip(), row[1].strip(), float(row[2]), float(row[3])
            img = cv2.imread(str((BUNDLE_ROOT / img_path) if not Path(img_path).is_absolute() else img_path))
            b = best_box(model, img, CLASS_MAP[obj], args.conf)
            if b is None:
                print(f"  [miss] {obj} {img_path}: no detection"); continue
            pts.setdefault(obj, []).append(([b[0], b[1]], [tx, ty]))

    out, ref = {}, None
    for obj, rows in pts.items():
        if len(rows) < 4:
            print(f"  [skip] {obj}: only {len(rows)} points (<4)"); continue
        F, Y = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
        errs = loo_errors(F, Y)
        print(f"[refit] {obj}: {len(F)} pts, LOO mae=({np.abs(errs[:,0]).mean():.0f},"
              f"{np.abs(errs[:,1]).mean():.0f})mm")
        W = fit_affine(F, Y)
        out[obj] = {"yolo_class": CLASS_MAP[obj], "weights_3x2": W.tolist(),
                    "features": "cam1_cx,cam1_cy,1 (normalized box center)",
                    "n_points": len(F), "refit": "demo-room points"}
        ref = ref or obj
    if ref and old:
        dW = np.array(out[ref]["weights_3x2"]) - np.array(old[ref]["weights_3x2"])
        for obj, entry in old.items():
            if obj in out:
                continue
            W = np.array(entry["weights_3x2"]) + dW
            out[obj] = {**entry, "weights_3x2": W.tolist(),
                        "refit": f"transferred camera shift from {ref}"}
            print(f"[transfer] {obj}: shifted by {ref}'s delta")
    out_path = BUNDLE_ROOT / args.out
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[saved] {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", default="data/recordings")
    ap.add_argument("--model", default="data/models/yolo/3_white_trapezoid.pt")
    ap.add_argument("--objects", default="board,butter_knife,trapezoid")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="data/calibration/sidecam_locator.json")
    ap.add_argument("--points-csv", default=None,
                    help="demo-room refit: CSV of object,image,true_x,true_y")
    ap.add_argument("--transfer-old", default=None,
                    help="with --points-csv: old locator json; objects not in the CSV "
                         "get the reference object's camera-shift delta")
    args = ap.parse_args()
    if args.points_csv:
        return refit_from_points(args)

    import cv2
    from ultralytics import YOLO
    model = YOLO(str(BUNDLE_ROOT / args.model))

    out = {}
    for obj in args.objects.split(","):
        cls_name = CLASS_MAP[obj]
        feats, gts, eps = [], [], []
        # cam1-only: the table is a plane at a fixed z, so one camera's box center
        # determines (x,y); cam2 often cannot see the flat objects (board/knife).
        for ep in sorted((BUNDLE_ROOT / args.recordings / obj).glob("episode_*")):
            c1 = cv2.imread(str(ep / "cam1_rgb" / "frame_000000.jpg"))
            if c1 is None:
                continue
            b1 = best_box(model, c1, cls_name, args.conf)
            if b1 is None:
                print(f"  [miss] {obj}/{ep.name}: cam1=NONE")
                continue
            feats.append([b1[0], b1[1]])
            gts.append(episode_gt(ep))
            eps.append(ep.name)
        F, Y = np.array(feats), np.array(gts)
        print(f"\n=== {obj}: {len(F)} usable episodes (of "
              f"{len(list((BUNDLE_ROOT / args.recordings / obj).glob('episode_*')))}) ===")
        if len(F) < 4:
            print("  too few points for a stable affine — needs more episodes/captures")
            continue
        errs = loo_errors(F, Y)
        ex, ey = np.abs(errs[:, 0]), np.abs(errs[:, 1])
        print(f"  LOO |err| x: mean={ex.mean():.0f} max={ex.max():.0f} mm   "
              f"y: mean={ey.mean():.0f} max={ey.max():.0f} mm")
        for name, e in zip(eps, errs):
            print(f"    {name}: err=({e[0]:+.0f},{e[1]:+.0f})")
        W = fit_affine(F, Y)
        out[obj] = {"yolo_class": cls_name, "weights_3x2": W.tolist(),
                    "features": "cam1_cx,cam1_cy,1 (normalized box center)",
                    "n_points": len(F),
                    "loo_mae_mm": [round(float(ex.mean()), 1), round(float(ey.mean()), 1)],
                    "loo_max_mm": [round(float(ex.max()), 1), round(float(ey.max()), 1)]}

    out_path = BUNDLE_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
