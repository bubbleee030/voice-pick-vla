#!/usr/bin/env python3
"""Off-rig eval of the arrival + servo module against recorded episodes (§3.2).

Every recorded episode ends with a vertical descent above the object, so claw
frames are ground-truth-labeled by the arm's xy distance to the episode's
final xy (same trick as claw_arrival_calib.py). This script:

  1. runs claw-YOLO over sampled frames of every episode,
  2. ARRIVAL: feeds each frame's boxes through ArrivalServo.decide() and
     reports recall on true on-top hover frames vs false-arrivals on frames
     still far away,
  3. JACOBIAN: least-squares fits J = d(cx,cy)/d(arm x,y) per object from
     hover-band frames (the §1.3 numbers, estimated from data),
  4. SERVO: simulates servo_step() on approach frames and checks the proposed
     move actually points toward the object (cosine vs ground truth),
  5. --write-yaml: saves the fitted jacobians into claw_servo.yaml
     (calibrated stays false — rig verification §1.3 still required).

Run:
    conda run --no-capture-output -n voice_pick python scripts/arrival_offline_eval.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

BUNDLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BUNDLE_ROOT))

from src.arrival import ArrivalServo  # noqa: E402

# recording dir -> YOLO class in 3_white_trapezoid.pt (mirror of claw_arrival_calib)
CLASS_MAP = {"board": "board", "butter_knife": "knife", "trapezoid": "trapezoid_white"}


def load_traj(ep: Path) -> tuple[np.ndarray, np.ndarray]:
    ts, xyz = [], []
    with open(ep / "trajectory.csv") as f:
        for row in csv.DictReader(f):
            if row.get("valid") != "1":
                continue
            ts.append(float(row["elapsed_s"]))
            xyz.append([float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])])
    return np.array(ts), np.array(xyz)


def collect(model, obj: str, recordings: Path, conf: float, per_ep: int) -> list[dict]:
    """One record per sampled frame: arm offset from final xy, z, best box."""
    import cv2
    cls_name = CLASS_MAP[obj]
    rows: list[dict] = []
    for ep in sorted((recordings / obj).glob("episode_*")):
        ts, xyz = load_traj(ep)
        frames = sorted((ep / "claw_rgb").glob("frame_*.jpg"))
        if len(ts) < 5 or not frames:
            continue
        ft = np.linspace(0.0, ts[-1], len(frames))
        end_xy = xyz[-1, :2]
        for fi in np.linspace(0, len(frames) - 1, min(per_ep, len(frames))).astype(int):
            img = cv2.imread(str(frames[fi]))
            if img is None:
                continue
            h, w = img.shape[:2]
            ti = int(np.argmin(np.abs(ts - ft[fi])))
            arm = xyz[ti]
            res = model(img, conf=conf, verbose=False)[0]
            boxes = []
            for b in res.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                boxes.append({
                    "label": model.names[int(b.cls[0])],
                    "conf": float(b.conf[0]),
                    "x1": x1 / w, "y1": y1 / h, "x2": x2 / w, "y2": y2 / h,
                })
            rows.append({
                "ep": ep.name, "z": float(arm[2]),
                "off_x": float(arm[0] - end_xy[0]),   # arm offset from on-top
                "off_y": float(arm[1] - end_xy[1]),
                "dist": float(np.hypot(arm[0] - end_xy[0], arm[1] - end_xy[1])),
                "boxes": boxes,
                "best": max((b for b in boxes if b["label"] == cls_name),
                            default=None, key=lambda b: b["conf"]),
            })
    return rows


def _hover_sel(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["z"] > 300 and r["best"] and r["dist"] < 120]


def fit_jacobian(groups: dict[str, list[dict]]) -> tuple[np.ndarray, float, int] | None:
    """Pooled LSQ fit of J = d(cx,cy)/d(arm x,y) across ALL objects.

    The claw cam is arm-mounted, so J is rig geometry — identical for every
    object at the same hover z. Pooling matters: single-object fits go
    degenerate when that object's episodes lack spread on an axis
    (butter_knife only ever saw x~490). Per-object intercepts are removed by
    centering each object's samples on its own means."""
    X, U, V = [], [], []
    for rows in groups.values():
        sel = _hover_sel(rows)
        if len(sel) < 6:
            continue
        off = np.array([[r["off_x"], r["off_y"]] for r in sel])
        cx = np.array([(r["best"]["x1"] + r["best"]["x2"]) / 2 for r in sel])
        cy = np.array([(r["best"]["y1"] + r["best"]["y2"]) / 2 for r in sel])
        X.append(off - off.mean(axis=0))
        U.append(cx - cx.mean())
        V.append(cy - cy.mean())
    if not X:
        return None
    X, U, V = np.vstack(X), np.concatenate(U), np.concatenate(V)
    if X[:, 0].std() < 3.0 or X[:, 1].std() < 3.0:   # need mm-scale spread per axis
        return None
    sol_u, res_u = np.linalg.lstsq(X, U, rcond=None)[:2]
    sol_v, res_v = np.linalg.lstsq(X, V, rcond=None)[:2]
    J = np.array([[sol_u[0], sol_u[1]], [sol_v[0], sol_v[1]]])
    ss = (U ** 2).sum() + (V ** 2).sum()
    r2 = 1.0 - float((res_u.sum() + res_v.sum()) / ss) if ss > 0 else 0.0
    return J, r2, len(U)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", default="data/recordings")
    ap.add_argument("--model", default="data/models/yolo/3_white_trapezoid.pt")
    ap.add_argument("--objects", default="board,butter_knife,trapezoid")
    ap.add_argument("--conf", type=float, default=0.25, help="collection floor (decide() applies its own)")
    ap.add_argument("--per-ep", type=int, default=25, help="frames sampled per episode")
    ap.add_argument("--far-mm", type=float, default=30.0, help="frames farther than this must NOT arrive")
    ap.add_argument("--write-yaml", action="store_true", help="save fitted jacobians into claw_servo.yaml")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(BUNDLE_ROOT / args.model))
    # Recordings predate the rig calibration (different setpoints/z) — eval
    # against the recordings' own geometry, not claw_servo_calibration.json.
    servo = ArrivalServo(rig_calib_json=Path("/nonexistent"))

    groups = {obj: collect(model, obj, BUNDLE_ROOT / args.recordings, args.conf, args.per_ep)
              for obj in args.objects.split(",")}

    # -- one pooled jacobian for the rig (arm-mounted cam => object-independent) --
    fitted: dict[str, list[list[float]]] = {}
    fit = fit_jacobian(groups)
    if fit:
        J, r2, n = fit
        servo._jac["default"] = J.tolist()
        fitted["default"] = [[round(float(v), 6) for v in row] for row in J.tolist()]
        inv = np.linalg.inv(J)
        print(f"pooled jacobian (n={n}, R2={r2:.3f}): "
              f"[[{J[0,0]:+.5f},{J[0,1]:+.5f}],[{J[1,0]:+.5f},{J[1,1]:+.5f}]]\n"
              f"  -> inverse (mm per normalized unit): "
              f"x:[{inv[0,0]:+.0f},{inv[0,1]:+.0f}] y:[{inv[1,0]:+.0f},{inv[1,1]:+.0f}]")
    else:
        print("pooled jacobian: not enough well-spread hover detections")

    for obj, rows in groups.items():
        servo._jac.pop(obj, None)   # eval every object on the pooled J
        hover = [r for r in rows if r["z"] > 300]
        print(f"\n=== {obj}: {len(rows)} frames "
              f"({len(hover)} hover, {sum(1 for r in rows if r['best'])} with box) ===")

        # -- arrival confusion at hover --
        on = [r for r in hover if r["dist"] <= 8.0]
        far = [r for r in hover if r["dist"] >= args.far_mm]
        on_hit = sum(servo.decide(r["boxes"], obj).arrived for r in on)
        far_hit = sum(servo.decide(r["boxes"], obj).arrived for r in far)
        print(f"  arrival: on-top {on_hit}/{len(on)} arrived   "
              f"far(>{args.far_mm:.0f}mm) {far_hit}/{len(far)} false-arrivals")

        # -- servo direction sim on hover approach frames --
        sim = [r for r in hover if args.far_mm >= r["dist"] or r["dist"] <= 120]
        sim = [r for r in sim if r["best"] and r["dist"] > 8.0]
        cos, gains = [], []
        for r in sim:
            d = servo.decide(r["boxes"], obj)
            step = servo.servo_step(d, obj)
            if step is None:
                continue
            need = np.array([-r["off_x"], -r["off_y"]])   # direction to the object
            s = np.array(step)
            cos.append(float(np.dot(s, need) / (np.linalg.norm(s) * np.linalg.norm(need) + 1e-9)))
            gains.append(float(np.linalg.norm(need) - np.linalg.norm(need - s)))
        if cos:
            print(f"  servo sim: {len(cos)} steps  cos(step,truth)={np.mean(cos):+.2f}  "
                  f"right-way {sum(c > 0 for c in cos)}/{len(cos)}  "
                  f"mean closing {np.mean(gains):+.1f}mm/step")
        else:
            print("  servo sim: no usable approach frames")

    if args.write_yaml and fitted:
        import yaml
        path = BUNDLE_ROOT / "data/calibration/claw_servo.yaml"
        cfg = yaml.safe_load(path.read_text())
        cfg["jacobian"] = fitted   # pooled default only; rig may add per-object
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        print(f"\n[saved] fitted jacobians -> {path}  (calibrated stays false; verify signs at rig)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
