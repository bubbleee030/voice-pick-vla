#!/usr/bin/env python3
"""Interactive claw-servo calibration (2026-07-14 three-object demo, §1.3).

YOU physically move the arm (teach pendant / jog UI / whatever you normally use).
This script NEVER commands motion — it only READS the arm's live pose (Modbus
polling) and the claw camera, at moments you tell it to capture. That is
deliberate: verifying "directly above" by eyeballing the claw image is circular
(the claw is off-axis/tilted — see CONTEXT.md "on-top setpoint"), so this tool
gets ground truth from YOUR physical pose instead, and does the pixel math for you.

It also does the coupling correctly: instead of one gain number per axis (which
silently breaks if the tilted claw couples x and y), it measures a full 2x2
Jacobian (world mm -> normalized pixel-center shift) from two independent probe
moves and inverts it, so the arrival/servo module gets an exact correction
matrix even if the claw's tilt mixes the axes.

FLOW
  1. Baseline  — you place the arm at a KNOWN-GOOD pick pose (a taught pose, or a
     manual jog+descend that you confirm actually grasps the object) and rise
     straight up to hover z. Press Enter. This pose is ground truth "directly
     above" — nothing here is read from the claw image to decide that.
  2. Probe X   — you jog ONLY x by roughly --jog-mm (y, z held). Press Enter.
  3. Return    — you jog back to the baseline pose. Press Enter (sanity check).
  4. Probe Y   — you jog ONLY y by roughly --jog-mm (x, z held). Press Enter.
  5. Report + optional live validation loop (jog anywhere, the script tells you
     the suggested correction, you apply it, it shows the residual error).
  6. Saves data/calibration/claw_servo_calibration.json (Jacobian + inverse +
     gains + the live-measured setpoint) and offers to update claw_arrival.json's
     setpoint for this object with the freshly measured (ground-truth) value.

Run on the laptop (needs the arm + claw camera; requires an interactive terminal):
    conda run --no-capture-output -n voice_pick python scripts/servo_calibrate.py \
        --object trapezoid

Math-only smoke test (no hardware; runs anywhere, including the 5090 dev box):
    python scripts/servo_calibrate.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BUNDLE_ROOT))

import numpy as np  # noqa: E402

CLASS_MAP = {"board": "board", "knife": "knife", "trapezoid": "trapezoid_white"}
DEFAULT_MODEL = "data/models/yolo/3_white_trapezoid.pt"
CALIB_OUT = "data/calibration/claw_servo_calibration.json"
ARRIVAL_JSON = "data/calibration/claw_arrival.json"

# "not a pure single-axis move" guard: warn if the OTHER two axes moved more than
# this, relative to the intended axis's motion (or this absolute mm, whichever larger)
IMPURITY_REL = 0.15
IMPURITY_ABS_MM = 4.0
# axis-coupling in the pixel Jacobian worth flagging to the user (off-diagonal vs
# diagonal magnitude)
COUPLING_FLAG = 0.30


# --------------------------------------------------------------------------- #
# Pure math — no hardware, fully unit-testable (see --self-test)
# --------------------------------------------------------------------------- #
def build_jacobian(d_world_x: np.ndarray, px_shift_x: np.ndarray,
                    d_world_y: np.ndarray, px_shift_y: np.ndarray) -> np.ndarray:
    """J maps a world (dx,dy) mm move to a (d_cx,d_cy) normalized pixel shift.
    d_world_x/px_shift_x come from the x-only probe (dy~0); d_world_y/px_shift_y
    from the y-only probe (dx~0). Columns of J = pixel response per unit world axis.
    """
    dx = d_world_x[0]
    dy = d_world_y[1]
    if abs(dx) < 1e-6 or abs(dy) < 1e-6:
        raise ValueError("probe move too small on its intended axis (<1e-3mm) — "
                          "jog farther and retry")
    col_x = px_shift_x / dx   # d(cx,cy) per mm of world x
    col_y = px_shift_y / dy   # d(cx,cy) per mm of world y
    return np.column_stack([col_x, col_y])  # 2x2: rows=(cx,cy), cols=(x,y)


def invert_jacobian(J: np.ndarray) -> tuple[np.ndarray, float]:
    det = float(np.linalg.det(J))
    if abs(det) < 1e-9:
        raise ValueError(f"Jacobian is singular/near-singular (det={det:.2e}) — "
                          "the two probe moves didn't separate in pixel space; "
                          "increase --jog-mm and retry")
    return np.linalg.inv(J), det


def coupling_ratios(J: np.ndarray) -> tuple[float, float]:
    """|off-diagonal| / |diagonal| for each column — how much x bleeds into cy,
    and y bleeds into cx, in the claw's tilted view."""
    cx_from_x, cy_from_x = J[0, 0], J[1, 0]
    cx_from_y, cy_from_y = J[0, 1], J[1, 1]
    cxy = abs(cy_from_x) / max(abs(cx_from_x), 1e-9)
    cyx = abs(cx_from_y) / max(abs(cy_from_y), 1e-9)
    return cxy, cyx


def check_purity(d_world: np.ndarray, axis: int, jog_mm: float) -> list[str]:
    warnings = []
    names = ["x", "y", "z"]
    intended = abs(d_world[axis])
    for i in range(3):
        if i == axis:
            continue
        off = abs(d_world[i])
        limit = max(IMPURITY_ABS_MM, IMPURITY_REL * intended)
        if off > limit:
            warnings.append(f"  [!] {names[i]} moved {off:.1f}mm during the "
                            f"{names[axis]}-only probe (limit {limit:.1f}mm) — "
                            f"not a pure move, results may be skewed")
    if intended < 0.3 * jog_mm:
        warnings.append(f"  [!] intended axis only moved {intended:.1f}mm "
                        f"(asked for ~{jog_mm:.0f}mm) — jog farther for a cleaner fit")
    return warnings


def predict_correction(J_inv: np.ndarray, cur_cxcy: np.ndarray,
                       setpoint_cxcy: np.ndarray) -> np.ndarray:
    """World (dx,dy) mm to move FROM cur TO setpoint."""
    pixel_error = setpoint_cxcy - cur_cxcy
    return J_inv @ pixel_error


# --------------------------------------------------------------------------- #
# Self-test: proves the math above without any arm/camera hardware
# --------------------------------------------------------------------------- #
def self_test() -> int:
    print("[self-test] 1/3 pure-math Jacobian round trip ...")
    # Simulate a claw tilted 20 degrees: x-move bleeds into cy, y-move into cx.
    # (tan(20deg)=0.36 > COUPLING_FLAG=0.30, so this tilt should trip the flag;
    # a real claw could be tilted more or less than this.)
    import math
    theta = math.radians(20)
    mm_per_norm = 180.0  # made up: 180mm world motion per 1.0 normalized-unit shift
    true_J = (1.0 / mm_per_norm) * np.array([
        [math.cos(theta), math.sin(theta)],
        [-math.sin(theta), math.cos(theta)],
    ])
    # PURE probes first: build_jacobian divides by the intended axis only, so it is
    # exact iff the probe is pure. Verify that exactness here with atol=1e-9.
    d_world_x = np.array([20.0, 0.0, 0.0])
    d_world_y = np.array([0.0, 22.0, 0.0])
    px_x = true_J @ np.array([d_world_x[0], d_world_x[1]])
    px_y = true_J @ np.array([d_world_y[0], d_world_y[1]])
    J = build_jacobian(d_world_x, px_x, d_world_y, px_y)
    assert np.allclose(J, true_J, atol=1e-9), f"Jacobian mismatch:\n{J}\nvs\n{true_J}"
    J_inv, det = invert_jacobian(J)
    assert abs(det) > 1e-9
    # round trip: predicted correction for a known pixel error should recover the
    # world move that caused it
    setpoint = np.array([0.50, 0.50])
    cur = setpoint - px_x  # pretend current = setpoint shifted by the x-probe's effect
    corr = predict_correction(J_inv, cur, setpoint)
    assert np.allclose(corr, [d_world_x[0], d_world_x[1]], atol=1e-6), corr
    cxy, cyx = coupling_ratios(J)
    assert cxy > COUPLING_FLAG and cyx > COUPLING_FLAG, "tilted-claw test should flag coupling"
    print(f"    OK — J=\n{J}\n    coupling cxy={cxy:.2f} cyx={cyx:.2f} (expected >{COUPLING_FLAG}, tilt is simulated)")

    print("[self-test] 2/3 purity + singular-Jacobian guards ...")
    warn = check_purity(d_world_x, axis=0, jog_mm=20.0)
    assert warn == [], f"expected no purity warnings on a pure probe, got {warn}"
    warn_bad = check_purity(np.array([20.0, 8.0, 0.5]), axis=0, jog_mm=20.0)
    assert warn_bad, "expected an impurity warning for an 8mm y-bleed on a 20mm x-probe"
    try:
        invert_jacobian(np.zeros((2, 2)))
        raise AssertionError("expected ValueError on singular Jacobian")
    except ValueError:
        pass
    print("    OK")

    print("[self-test] 3/3 YOLO model loads + detects on a bundled claw frame ...")
    try:
        import cv2
        from ultralytics import YOLO
    except Exception as e:
        print(f"    SKIP (cv2/ultralytics unavailable here: {e})")
        return 0
    model_path = BUNDLE_ROOT / DEFAULT_MODEL
    if not model_path.exists():
        print(f"    SKIP ({model_path} not found)")
        return 0
    model = YOLO(str(model_path))
    found_any = False
    for obj, cls_name in CLASS_MAP.items():
        frames = sorted((BUNDLE_ROOT / "data/recordings" / obj).glob("episode_*/claw_rgb/frame_0000*.jpg"))
        if not frames:
            continue
        img = cv2.imread(str(frames[len(frames) // 2]))
        box = detect_box(model, img, cls_name, conf_floor=0.25)
        print(f"    {obj:14s} sample frame -> {'detected conf=%.2f' % box[2] if box else 'no detection'}")
        found_any = found_any or box is not None
    if not found_any:
        print("    WARN: no detections on any sample frame — check the model/paths")
    print("\n[self-test] ALL CHECKS PASSED")
    return 0


# --------------------------------------------------------------------------- #
# Live capture helpers (hardware path)
# --------------------------------------------------------------------------- #
def detect_box(model, img, cls_name: str, conf_floor: float):
    """Return (cx, cy, conf, clipped) normalized 0..1, or None."""
    h, w = img.shape[:2]
    res = model(img, conf=conf_floor, verbose=False)[0]
    best = None
    for b in res.boxes:
        if model.names[int(b.cls[0])] != cls_name:
            continue
        conf = float(b.conf[0])
        if best is None or conf > best[2]:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            clipped = x1 <= 2 or y1 <= 2 or x2 >= w - 2 or y2 >= h - 2
            best = ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, conf, clipped)
    return best


def capture_stable(claw, model, cls_name: str, conf_floor: float, repeat: int):
    """Grab `repeat` claw frames a beat apart, average successful detections."""
    hits = []
    for i in range(repeat):
        frame = claw.get_latest_frame()
        if frame is not None:
            box = detect_box(model, frame, cls_name, conf_floor)
            if box is not None:
                hits.append(box)
        if i < repeat - 1:
            time.sleep(0.15)
    if not hits:
        return None
    cx = float(np.mean([h[0] for h in hits]))
    cy = float(np.mean([h[1] for h in hits]))
    conf = float(np.mean([h[2] for h in hits]))
    clipped = any(h[3] for h in hits)
    return {"cx": cx, "cy": cy, "conf": conf, "clipped": clipped, "n_ok": len(hits), "n_tried": repeat}


def read_pose(arm, timeout_s: float = 3.0) -> np.ndarray:
    t0 = time.time()
    while arm.current_pose_mm_deg is None and time.time() - t0 < timeout_s:
        time.sleep(0.1)
    if arm.current_pose_mm_deg is None:
        raise RuntimeError("no live arm pose (Modbus) — is the arm connected?")
    return np.array(arm.current_pose_mm_deg[:3], dtype=float)  # x,y,z mm


def prompt_capture(label: str, arm, claw, model, cls_name: str, conf_floor: float, repeat: int):
    while True:
        input(f"\n>>> {label}\n    Press Enter when the arm is in position ...")
        pose = read_pose(arm)
        det = capture_stable(claw, model, cls_name, conf_floor, repeat)
        if det is None:
            print(f"    [FAIL] no detection in {repeat} tries at conf>={conf_floor} "
                  f"— check the object is visible to the claw, or lower --conf. Retry.")
            continue
        flag = " CLIPPED" if det["clipped"] else ""
        print(f"    pose x={pose[0]:.1f} y={pose[1]:.1f} z={pose[2]:.1f}   "
              f"box cx={det['cx']:.3f} cy={det['cy']:.3f} conf={det['conf']:.2f} "
              f"({det['n_ok']}/{det['n_tried']} frames hit){flag}")
        ans = input("    keep this capture? [Enter]=yes  r=retry: ").strip().lower()
        if ans != "r":
            return pose, det


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object", choices=sorted(CLASS_MAP), default=None)
    ap.add_argument("--jog-mm", type=float, default=20.0,
                    help="suggested jog distance per probe (informational only — "
                         "the script MEASURES the actual move from the live pose)")
    ap.add_argument("--repeat", type=int, default=3,
                    help="claw frames averaged per capture (reduces detection jitter)")
    ap.add_argument("--conf", type=float, default=0.4, help="YOLO confidence floor")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=CALIB_OUT)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--self-test", action="store_true",
                    help="run the pure-math + detection checks with no arm/camera and exit")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.object:
        ap.error("--object is required unless --self-test")

    cls_name = CLASS_MAP[args.object]

    from src.runtime_config import load_runtime_bundle
    from src.services.arm_service import ArmService
    from src.services.claw_camera_service import ClawCameraService
    from ultralytics import YOLO

    env_cfg = "config/env/ubuntu.local.yaml"
    if not (BUNDLE_ROOT / env_cfg).exists():
        env_cfg = None
    bundle = load_runtime_bundle(profile=args.profile, env_config=env_cfg)
    demo_cfg, modbus_cfg = bundle.demo_config, bundle.modbus_config

    print("Bringing up claw camera + arm pose polling (READ-ONLY — this script "
          "never commands motion; you move the arm yourself) ...")
    claw = ClawCameraService(demo_cfg)
    arm = ArmService(demo_cfg, modbus_cfg, socketio=None)
    claw.start()
    if not arm.auto_connect():
        print("ERROR: arm not connected. Abort."); return 2
    arm.start_pose_polling()
    model = YOLO(str(BUNDLE_ROOT / args.model))

    print(f"\n=== Servo calibration: {args.object} (YOLO class '{cls_name}') ===")
    print("Step 1/4 — BASELINE: put the arm at a KNOWN-GOOD pick pose (taught pose, "
          "or a manual jog+descend you've confirmed actually grasps the object), "
          "then rise straight up to hover z.")
    base_pose, base_det = prompt_capture("Baseline (directly above, hover z)",
                                         arm, claw, model, cls_name, args.conf, args.repeat)

    old_setpoint = None
    arrival_path = BUNDLE_ROOT / ARRIVAL_JSON
    if arrival_path.exists():
        old = json.loads(arrival_path.read_text()).get(args.object)
        if old:
            old_setpoint = (old.get("setpoint_cx"), old.get("setpoint_cy"))
            dcx = base_det["cx"] - old_setpoint[0]
            dcy = base_det["cy"] - old_setpoint[1]
            tag = "close to" if abs(dcx) < 0.05 and abs(dcy) < 0.05 else "DIFFERS FROM"
            print(f"    recorded claw_arrival.json setpoint was ({old_setpoint[0]:.3f},"
                  f"{old_setpoint[1]:.3f}) — this live reading {tag} it "
                  f"(Δ={dcx:+.3f},{dcy:+.3f})")

    print("\nStep 2/4 — PROBE X: jog ONLY x by about "
          f"{args.jog_mm:.0f}mm (keep y, z fixed).")
    x_pose, x_det = prompt_capture("After x-only jog", arm, claw, model, cls_name,
                                   args.conf, args.repeat)
    d_world_x = x_pose - base_pose
    px_shift_x = np.array([x_det["cx"] - base_det["cx"], x_det["cy"] - base_det["cy"]])
    for w in check_purity(d_world_x, axis=0, jog_mm=args.jog_mm):
        print(w)

    print("\nStep 3/4 — RETURN: jog back to the Step 1 baseline pose.")
    back_pose, back_det = prompt_capture("Back at baseline", arm, claw, model, cls_name,
                                         args.conf, args.repeat)
    drift = float(np.linalg.norm(back_pose - base_pose))
    if drift > 5.0:
        print(f"    [!] pose is {drift:.1f}mm from the original baseline (backlash/drift) "
              f"— consider re-doing the baseline capture before continuing")

    print("\nStep 4/4 — PROBE Y: jog ONLY y by about "
          f"{args.jog_mm:.0f}mm (keep x, z fixed), from the baseline.")
    y_pose, y_det = prompt_capture("After y-only jog", arm, claw, model, cls_name,
                                   args.conf, args.repeat)
    d_world_y = y_pose - back_pose
    px_shift_y = np.array([y_det["cx"] - back_det["cx"], y_det["cy"] - back_det["cy"]])
    for w in check_purity(d_world_y, axis=1, jog_mm=args.jog_mm):
        print(w)

    try:
        J = build_jacobian(d_world_x, px_shift_x, d_world_y, px_shift_y)
        J_inv, det = invert_jacobian(J)
    except ValueError as e:
        print(f"\n[FAIL] {e}"); return 1

    cxy, cyx = coupling_ratios(J)
    print("\n" + "=" * 70)
    print("RESULT")
    print(f"  setpoint (live, ground-truth): cx={base_det['cx']:.3f} cy={base_det['cy']:.3f} "
          f"at hover z={base_pose[2]:.1f}mm")
    print(f"  gain_x (mm world / 1.0 norm-cx shift): {1/J[0,0]:.1f}"
          f"   gain_y: {1/J[1,1]:.1f}")
    print(f"  coupling: x-move bleeds into cy by {cxy*100:.0f}% of its cx effect; "
          f"y-move bleeds into cx by {cyx*100:.0f}% of its cy effect")
    if cxy > COUPLING_FLAG or cyx > COUPLING_FLAG:
        print("  -> claw tilt is SIGNIFICANT: use the full 2x2 correction matrix below, "
              "a single per-axis gain would be inaccurate.")
    else:
        print("  -> axes are largely independent; a simple per-axis gain would "
              "have been fine, but the matrix below is exact either way.")
    print(f"  Jacobian J (world mm -> norm pixel shift):\n{J}")
    print(f"  Correction matrix J_inv (pixel error -> world mm move):\n{J_inv}")
    print("=" * 70)

    # optional live validation loop
    print("\nOptional: validate the correction live. Jog the arm anywhere (object "
          "still visible), press Enter to see the suggested correction; 'q' to skip/stop.")
    setpoint = np.array([base_det["cx"], base_det["cy"]])
    while True:
        ans = input("\n[Enter]=capture+predict  q=done validating: ").strip().lower()
        if ans == "q":
            break
        pose = read_pose(arm)
        det = capture_stable(claw, model, cls_name, args.conf, args.repeat)
        if det is None:
            print("    no detection — reposition and retry"); continue
        cur = np.array([det["cx"], det["cy"]])
        corr = predict_correction(J_inv, cur, setpoint)
        err_px = float(np.linalg.norm(setpoint - cur))
        print(f"    current cx={cur[0]:.3f} cy={cur[1]:.3f}  pixel error={err_px:.3f}")
        print(f"    suggested move: x{corr[0]:+.1f}mm  y{corr[1]:+.1f}mm "
              f"-> jog that, then press Enter again to re-check")

    out = {}
    out_path = BUNDLE_ROOT / args.out
    if out_path.exists():
        out = json.loads(out_path.read_text())
    out[args.object] = {
        "yolo_class": cls_name,
        "hover_z_mm": round(float(base_pose[2]), 1),
        "setpoint_cx": round(base_det["cx"], 4),
        "setpoint_cy": round(base_det["cy"], 4),
        "jacobian_world_to_pixel": J.tolist(),
        "correction_matrix_pixel_to_world_mm": J_inv.tolist(),
        "coupling_x_into_cy": round(cxy, 3),
        "coupling_y_into_cx": round(cyx, 3),
        "probe_jog_mm_requested": args.jog_mm,
        "probe_x_actual_mm": d_world_x.tolist(),
        "probe_y_actual_mm": d_world_y.tolist(),
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {out_path}")

    if old_setpoint is not None or arrival_path.exists():
        ans = input(f"\nUpdate {ARRIVAL_JSON}'s setpoint for '{args.object}' with this "
                    f"live-measured value? [y/N]: ").strip().lower()
        if ans == "y":
            arrival = json.loads(arrival_path.read_text()) if arrival_path.exists() else {}
            entry = arrival.get(args.object, {"yolo_class": cls_name})
            entry["setpoint_cx"] = round(base_det["cx"], 4)
            entry["setpoint_cy"] = round(base_det["cy"], 4)
            entry["setpoint_source"] = "servo_calibrate.py live measurement " + time.strftime("%Y-%m-%d")
            arrival[args.object] = entry
            arrival_path.write_text(json.dumps(arrival, indent=2))
            print(f"[updated] {arrival_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
