#!/usr/bin/env python3
"""Hardware-free end-to-end test of the §3 auto-run loop.

Simulates the full flow with the REAL _auto_loop code path (dry_run pose
simulation): a fake claw camera renders YOLO boxes from the simulated arm
pose through the calibration jacobian, and a fake VLA predictor emits a
chunk pointing at the object with a deliberate residual bias (the ~10-30 mm
error the servo exists to erase). Passes when:

  1. VLA strides move the arm toward the object,
  2. condition C (model wants to stay) hands over to the servo,
  3. servo corrections converge inside the arrival tolerance,
  4. arrival triggers the descend, the pick gate blocks until gate(),
  5. release leg + gate + return complete, stage == done.

Run:
    conda run --no-capture-output -n voice_pick python scripts/test_auto_run_sim.py
"""
from __future__ import annotations

import sys
import threading
import time
import types
from pathlib import Path

import numpy as np

BUNDLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BUNDLE_ROOT))

from src.arrival import ArrivalServo  # noqa: E402
from src.services.vla_service import VLAService  # noqa: E402

OBJECT_XY = np.array([520.0, 60.0])      # where the "object" sits
# 30 mm bias reproduces the 2026-07-14 live tug-of-war: at the servo's
# converged spot the model always "wants" to move >stay_mm back toward its
# biased target — without the fine-phase latch, arrival never triggers.
VLA_BIAS = np.array([-30.0, 15.0])
HOVER_Z = 425.0                          # travel plane (rig calib 2026-07-14)
# Deterministic knobs (arrival_tol_mm through the rig jacobian ≈ 0.021/0.037).
TEST_TOLS = {"arrival_tol_mm": 12.0}


class FakeClaw:
    """Renders the object's YOLO box from the sim pose via the jacobian.
    visible_below_z simulates the board: undetectable from the travel plane,
    only visible once the arm has descended near its calibrated z.
    fov_radius_mm models the wrist cam's NARROW downward view: the object only
    appears once the arm is roughly on top (so the far VLA approach sees no box,
    matching reality — and the #4 sanity gate only arms when it truly can)."""

    def __init__(self, servo: ArrivalServo, get_pose, obj: str,
                 visible_below_z: float | None = None,
                 fov_radius_mm: float = 45.0,
                 drop_after_first_correction: bool = False,
                 never_visible: bool = False,
                 detection_conf: float = 0.88,
                 detection_age_s: float = 0.05,
                 restart_error: bool = False):
        self.servo = servo
        self.get_pose = get_pose
        self.obj = obj
        self.visible_below_z = visible_below_z
        self.fov_radius_mm = fov_radius_mm
        self.drop_after_first_correction = drop_after_first_correction
        self.never_visible = never_visible
        self.detection_conf = detection_conf
        self.detection_age_s = detection_age_s
        self.restart_error = restart_error
        self.first_visible_xy: np.ndarray | None = None
        self.stop_calls = 0
        self.start_calls = 0
        self.worker = None
        set_cx, set_cy, self.cls = servo.setpoint(obj)
        self.sp = np.array([set_cx, set_cy])

    def get_detections(self):
        pose = self.get_pose()
        if pose is None:
            return [], None
        if self.never_visible:
            return [], 0.0
        if (self.drop_after_first_correction and self.first_visible_xy is not None
                and float(np.linalg.norm(np.array(pose[:2]) - self.first_visible_xy)) > 0.5):
            return [], 0.0
        if self.visible_below_z is not None and pose[2] > self.visible_below_z:
            return [], 0.0                     # too high — YOLO can't see it
        off = np.array(pose[:2]) - OBJECT_XY   # arm offset from on-top
        if float(np.hypot(off[0], off[1])) > self.fov_radius_mm:
            return [], 0.0                     # outside the narrow wrist-cam FOV
        J = np.array(self.servo.jacobian(self.obj))
        cx, cy = self.sp + J @ off             # box drifts opposite the arm
        if not (0.05 < cx < 0.95 and 0.05 < cy < 0.95):
            return [], 0.0                     # object out of claw view
        if self.first_visible_xy is None:
            self.first_visible_xy = np.array(pose[:2], dtype=float)
        w, h = 0.28, 0.22
        return [{"label": self.cls, "conf": self.detection_conf,
                 "x1": cx - w / 2, "y1": cy - h / 2,
                 "x2": cx + w / 2, "y2": cy + h / 2}], self.detection_age_s

    def stop(self):
        self.stop_calls += 1

    def start(self):
        self.start_calls += 1
        if self.restart_error:
            raise RuntimeError("simulated claw restart failure")


def fake_predict(self, lang_emb, lang_mask, frame_history, blind_state):
    """Chunk pointing at OBJECT_XY + VLA_BIAS from the current pose."""
    pose = self._auto_pose()
    tgt = OBJECT_XY + VLA_BIAS
    chunk = np.zeros((64, 128), dtype=np.float32)
    for i in range(64):
        f = min(1.0, (i + 1) / 12.0)           # ease toward the target over the chunk
        chunk[i, 0] = pose[0] + (tgt[0] - pose[0]) * f
        chunk[i, 1] = pose[1] + (tgt[1] - pose[1]) * f
        chunk[i, 2] = HOVER_Z
    return chunk, pose


def run_scenario(obj: str, visible_below_z: float | None, grasp_z: float,
                 approach_only: bool = False, place_only: bool = False,
                 fov_radius_mm: float = 45.0, vla_sanity_mm: float = 40.0,
                 mode: str = "vla", drop_after_first_correction: bool = False,
                 never_visible: bool = False, max_holds: int = 30,
                 locator_x_mm: float = 500.0, detection_conf: float = 0.88,
                 detection_age_s: float = 0.05, restart_error: bool = False) -> str:
    """Run one full auto-loop scenario; returns the captured log text."""
    import src.arrival as arrmod
    real_servo = getattr(arrmod, "_REAL_SERVO", arrmod.ArrivalServo)
    arrmod._REAL_SERVO = real_servo
    arrmod.ArrivalServo = lambda *a, **kw: real_servo(overrides=TEST_TOLS)

    logs: list[str] = []
    vla = VLAService(socketio=None)
    vla._emit_log = lambda lvl, msg: (logs.append(f"[{lvl}] {msg}"),
                                      print(f"  [{lvl}] {msg}"))[0] and None
    vla._ensure_models = lambda ckpt: None
    vla._load_lang_embed = lambda e, i: (None, None)
    vla._predict_target = types.MethodType(fake_predict, vla)

    servo_cfg = real_servo(overrides=TEST_TOLS)
    vla.claw = FakeClaw(
        servo_cfg, lambda: vla._sim_pose, obj, visible_below_z,
        fov_radius_mm=fov_radius_mm,
        drop_after_first_correction=drop_after_first_correction,
        never_visible=never_visible,
        detection_conf=detection_conf,
        detection_age_s=detection_age_s,
        restart_error=restart_error,
    )
    if mode == "locator":
        import src.locator as locmod

        class FakeLocator:
            def locate(self, frame, object_key):
                return types.SimpleNamespace(x_mm=locator_x_mm, y_mm=60.0, conf=0.95)

        locmod.SideCamLocator = FakeLocator
        fake_cam = types.SimpleNamespace(
            last_rgb=np.ones((8, 8, 3), dtype=np.uint8),
            lock=threading.Lock(),
        )
        vla.rs = types.SimpleNamespace(cams={"cam1": fake_cam})

    run_cfg = {"hover_z_mm": HOVER_Z, "stride": 6, "max_step_mm": 60.0,
               "speed_percent": 12, "max_moves": 30, "blind_state": True,
               "seek_step_mm": 30.0, "approach_only": approach_only,
               "place_only": place_only, "vla_sanity_mm": vla_sanity_mm,
               "max_holds": max_holds,
               "per_step_timeout_s": 5.0, "motion_start_timeout_s": 1.0}
    tail_cfg = {"grasp_z_mm": grasp_z,
                "place_approach_pose": [490116, -203991, 424908, 179999, 0, 0],
                "place_z_mm": 180.0}

    t = threading.Thread(target=vla._auto_loop,
                         args=(obj, "", "", "", mode, run_cfg, tail_cfg, True))
    vla._running = True
    vla.auto_object = obj
    t.start()

    def press_gates():
        # full flow: pick-descend gate, pick gate, PLACE-descend gate, release gate
        gates = ("descend_ok", "released") if place_only else \
                ("descend_ok", "pick_done", "descend_ok", "released")
        for name in gates:
            while t.is_alive() and vla.awaiting_gate != name:
                time.sleep(0.1)
            if not t.is_alive():
                return
            time.sleep(0.3)
            print(f"  >> pressing gate '{name}'")
            vla.gate(name)

    g = threading.Thread(target=press_gates)
    g.start()
    t.join(timeout=60)
    if t.is_alive():
        logs.append("[ERROR] loop thread hung")
        vla.stop()
        t.join(timeout=5)
    g.join(timeout=5)
    if g.is_alive():
        logs.append("[ERROR] gate thread hung")
    logs.append(
        f"[TEST] claw_stop={vla.claw.stop_calls} claw_start={vla.claw.start_calls}"
    )
    return "\n".join(logs)


def main() -> int:
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"{'PASS' if cond else 'FAIL'}  {name}")
        ok &= cond

    common = [
        ("VLA strides ran", "move vla"),
        ("servo took over (condition C)", "move servo"),
        ("ARRIVED", "ARRIVED"),
        ("descend gate waited + acked", "'descend_ok' acknowledged"),
        ("descended to grasp z", "move descend: "),
        ("pick gate waited + acked", "'pick_done' acknowledged"),
        ("release leg ran", "move release_point"),
        ("release gate acked", "'released' acknowledged"),
        ("completed", "COMPLETE"),
    ]

    print("=== scenario 1: trapezoid — visible at the travel plane (z425) ===")
    text = run_scenario("trapezoid", visible_below_z=None, grasp_z=175.0)
    for name, needle in common:
        check("trapezoid: " + name, needle in text)
    check("trapezoid: no seek needed (calib z == travel z)", "move seek_z" not in text)
    check("trapezoid: no errors", "[ERROR]" not in text)

    print("\n=== scenario 2: board — INVISIBLE above z380 (the rig finding) ===")
    text = run_scenario("board", visible_below_z=380.0, grasp_z=140.0)
    for name, needle in common:
        check("board: " + name, needle in text)
    check("board: seek-descend ran to the vision plane", "move seek_z" in text)
    check("board: no errors", "[ERROR]" not in text)

    print("\n=== scenario 3: trapezoid --approach-only (§1.4 convergence test) ===")
    text = run_scenario("trapezoid", visible_below_z=None, grasp_z=175.0, approach_only=True)
    check("approach-only: ARRIVED and held", "APPROACH-ONLY: ARRIVED" in text)
    check("approach-only: NO descend", "move descend: " not in text)
    check("approach-only: NO gates", "waiting for" not in text)
    check("approach-only: no errors", "[ERROR]" not in text)

    print("\n=== scenario 4: --place-only (release-leg test) ===")
    text = run_scenario("trapezoid", visible_below_z=None, grasp_z=175.0, place_only=True)
    check("place-only: skipped approach", "move vla" not in text and "PLACE-ONLY" in text)
    check("place-only: NO pick descend", "move descend: " not in text)
    check("place-only: place-descend gated", "PLACE-descend" in text and "'descend_ok' acknowledged" in text)
    check("place-only: release_down ran", "move release_down" in text)
    check("place-only: release gate acked", "'released' acknowledged" in text)
    check("place-only: completed, no errors", "COMPLETE" in text and "[ERROR]" not in text)

    print("\n=== scenario 5: #4 YOLO sanity gate — claw sees it but VLA points far ===")
    # Wide FOV: the claw sees the object even from the start, while the VLA keeps
    # pointing ~33 mm off (VLA_BIAS). With sanity_mm below that bias the loop must
    # distrust the model and hand to the servo, yet still ARRIVE.
    text = run_scenario("trapezoid", visible_below_z=None, grasp_z=175.0,
                        fov_radius_mm=500.0, vla_sanity_mm=20.0)
    check("sanity: gate fired (handed to YOLO servo)", "改用 YOLO 伺服" in text)
    check("sanity: still ARRIVED", "ARRIVED" in text)
    check("sanity: completed, no errors", "COMPLETE" in text and "[ERROR]" not in text)

    print("\n=== scenario 6: board locator — YOLO lost after first servo correction ===")
    text = run_scenario(
        "board", visible_below_z=380.0, grasp_z=140.0, mode="locator",
        drop_after_first_correction=True, max_holds=20,
    )
    check("last-target: camera service restarted", "claw_stop=1 claw_start=1" in text)
    check("last-target: fallback activated", "using last verified servo target" in text)
    restart_at = text.index("restarting the claw camera service")
    fallback_at = text.index("using last verified servo target")
    check("last-target: waited five recovery polls", all(
        restart_at < text.index(f"box lost ({n})") < fallback_at for n in range(11, 16)
    ))
    check("last-target: correction not repeated", text.count("move servo") == 1)
    check("last-target: scripted tail continued",
          "move descend: " in text and "move release_point" in text)
    check("last-target: completed", "COMPLETE board" in text and "[ERROR]" not in text)

    print("\n=== scenario 7: low max_holds cannot preempt eligible recovery ===")
    text = run_scenario(
        "board", visible_below_z=380.0, grasp_z=140.0, mode="locator",
        drop_after_first_correction=True, max_holds=12,
    )
    check("low-max: fallback still activated after poll 15",
          "box lost (15)" in text and "using last verified servo target" in text)
    check("low-max: completed", "COMPLETE board" in text and "[ERROR]" not in text)

    print("\n=== scenario 8: failed camera restart cannot authorize fallback ===")
    text = run_scenario(
        "board", visible_below_z=380.0, grasp_z=140.0, mode="locator",
        drop_after_first_correction=True, restart_error=True, max_holds=15,
    )
    check("restart-fail: failure surfaced", "claw restart failed" in text)
    check("restart-fail: no fallback", "using last verified servo target" not in text)
    check("restart-fail: aborts without descend",
          "object never became visible" in text and "move descend: " not in text)

    print("\n=== scenario 9: capped correction is not fallback authority ===")
    text = run_scenario(
        "board", visible_below_z=380.0, grasp_z=140.0, mode="locator",
        locator_x_mm=450.0, fov_radius_mm=100.0,
        drop_after_first_correction=True, max_holds=3,
    )
    check("capped: servo moved", "move servo1" in text)
    check("capped: target not saved", "saved verified board servo target" not in text)
    check("capped: never descends", "move descend: " not in text)

    print("\n=== scenario 10: stale detection is not fallback authority ===")
    text = run_scenario(
        "board", visible_below_z=380.0, grasp_z=140.0, mode="locator",
        detection_age_s=2.0, max_holds=3,
    )
    check("stale: target not saved", "saved verified board servo target" not in text)
    check("stale: never servos or descends",
          "move servo" not in text and "move descend: " not in text)

    print("\n=== scenario 11: low-confidence detection is not fallback authority ===")
    text = run_scenario(
        "board", visible_below_z=380.0, grasp_z=140.0, mode="locator",
        detection_conf=0.4, max_holds=3,
    )
    check("low-conf: target not saved", "saved verified board servo target" not in text)
    check("low-conf: never servos or descends",
          "move servo" not in text and "move descend: " not in text)

    print("\n=== scenario 12: board VLA mode cannot use locator fallback ===")
    text = run_scenario(
        "board", visible_below_z=380.0, grasp_z=140.0,
        drop_after_first_correction=True, max_holds=3,
    )
    check("wrong-mode: servo ran but target not saved",
          "move servo" in text and "saved verified board servo target" not in text)
    check("wrong-mode: never fallbacks or descends",
          "using last verified servo target" not in text and "move descend: " not in text)

    print("\n=== scenario 13: non-board locator cannot use board fallback ===")
    text = run_scenario(
        "trapezoid", visible_below_z=None, grasp_z=175.0, mode="locator",
        drop_after_first_correction=True, max_holds=3,
    )
    check("wrong-object: servo ran but target not saved",
          "move servo" in text and "saved verified board servo target" not in text)
    check("wrong-object: never fallbacks or descends",
          "using last verified servo target" not in text and "move descend: " not in text)

    print("\n=== scenario 14: board locator — never detected, no fallback authority ===")
    text = run_scenario(
        "board", visible_below_z=380.0, grasp_z=140.0, mode="locator",
        never_visible=True, max_holds=3,
    )
    check("no-target: aborts", "object never became visible" in text)
    check("no-target: does not fallback", "using last verified servo target" not in text)
    check("no-target: never descends", "move descend: " not in text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
