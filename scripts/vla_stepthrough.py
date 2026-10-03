#!/usr/bin/env python3
"""Stage-2 CLOSED-LOOP STEP-THROUGH for a trained RDT/VLA checkpoint.

Unlike scripts/vla_observe.py (which never moves the arm), this tool DOES command
arm motion — but only one waypoint at a time, and ONLY after you approve each move
by pressing Enter. It runs the model closed-loop: predict a chunk from the current
MEASURED pose, move a few steps toward it, re-read the pose, re-predict, repeat —
the way this state-conditioned policy is meant to run. This is how we measure the
real end-to-end pick (the static observer can't, since the policy keys off state).

SAFETY (defense in depth):
  * Per-move approval: nothing moves unless you press Enter. 's'+Enter stops.
  * Safety box (one box, two triggers): predicted target outside box -> re-predict
    up to N; still outside -> STOP. Measured pose outside box -> STOP.
  * Hard z-floor: target z is clamped to >= z_floor and the run STOPS when the arm
    reaches the handoff plane (default 175 mm) — it will not drive below it.
  * Per-move distance cap: refuses any single commanded move > --max-step mm.
  * Low default speed (12%), adjustable live with +/-.
  * Orientation pinned to (180,0,0) — only x,y,z come from the model (gripper is
    the teammate program's job; rotations were never learned).

The arm is NEVER auto-homed; put it at ready yourself first. Stop the demo server
(it owns the cameras): pkill -f voice_pick_demo.py

Run (interactive terminal required for the Enter prompts):
    conda run --no-capture-output -n voice_pick python scripts/vla_stepthrough.py
    ... --speed 12 --stride 6 --z-floor 175 --box 320,640,-50,235,172,426
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BUNDLE_ROOT))

import numpy as np  # noqa: E402

from scripts.vla_box import parse_box, in_box                # noqa: E402
from src.runtime_config import load_runtime_bundle           # noqa: E402
from src.services.arm_service import ArmService              # noqa: E402
from src.services.claw_camera_service import ClawCameraService  # noqa: E402
from src.services.realsense_service import RealSenseService  # noqa: E402
from src.services import vla_service as vlamod               # noqa: E402
from src.services.vla_service import VLAService              # noqa: E402

import torch  # noqa: E402

DEFAULT_CKPT  = "data/datasets/trapezoid_full_blind_nchc_20260707_024347"
DEFAULT_EMBED = "data/vla_embed_trapezoid_en.pt"
DEFAULT_INSTR = "拿起梯形"
DEFAULT_BOX   = (320.0, 640.0, -50.0, 235.0, 172.0, 426.0)  # x,x,y,y,z,z mm


class StepThrough:
    def __init__(self, vla, arm, args):
        self.vla = vla
        self.arm = arm
        self.box = args.box
        self.stride = args.stride
        self.max_step = args.max_step
        self.z_floor = args.z_floor
        self.y_floor = args.y_floor            # y is never negative in this workspace
        self.max_retries = args.max_retries
        self.speed = args.speed
        self.max_speed = args.max_speed
        self.linear = args.linear              # MovL(302) straight-line vs MovP(301) joint-interp
        self.freeze_z = args.freeze_z          # XY-ONLY mode: ignore model z, hold a hover z
        self.blind_state = args.blind_state    # zero the state token (train/deploy parity)
        self.hover_z = args.hover_z
        self.freeze_z_val = None               # resolved at run start when freeze_z is on
        self._hist: list[tuple] = []
        self.record_dir = args.record          # log per-prediction input+output for offline debug/retrain
        self.true_xy = args.true_xy            # optional (x,y) ground truth for this placement
        self.x_cal = args.x_cal                # optional (m,b): x_target = m*pred_x + b
        self._pred_i = 0
        self._last_input = None
        self._corr: list[tuple] = []           # manual x/y nudges applied this session
        if self.record_dir:
            os.makedirs(self.record_dir, exist_ok=True)
            mani = os.path.join(self.record_dir, "manifest.csv")
            if not os.path.exists(mani):
                with open(mani, "w") as f:
                    f.write("i,wall_t,meas_x,meas_y,meas_z,tgt_x,tgt_y,tgt_z,true_x,true_y\n")

    _MODEL_BLANK = np.zeros((480, 640, 3), dtype=np.uint8)

    def _predict_chunk(self) -> np.ndarray:
        vla = self.vla
        cam1, cam2, _claw = vla._grab_frames()
        self._last_input = (cam1, cam2)        # exact side-cam input the model saw (for --record)
        # The model expects 3 image slots (config img_cond_len=4374 = 2 hist x 3
        # cams x 729), but it was TRAINED on only the 2 side cams (cam1,cam2) — the
        # processed dataset has 2 cameras, so slot 3 was a CONSTANT blank the model
        # learned to ignore. Feeding the LIVE, arm-mounted claw there at deploy is
        # OOD and leaks noise into x (ablation: up to 76mm; blanking lifts grid
        # x-corr +0.76->+0.84, y unchanged). So feed a blank for the claw slot —
        # the claw camera is NOT a model input, and its health is irrelevant.
        frames = (cam1, cam2, self._MODEL_BLANK)
        self._hist.append(frames)
        if len(self._hist) > vlamod.IMG_HISTORY:
            self._hist = self._hist[-vlamod.IMG_HISTORY:]
        while len(self._hist) < vlamod.IMG_HISTORY:
            self._hist.insert(0, self._hist[0])
        img_batch = [vlamod._prep_image(f, vla._siglip.image_processor)
                     for triplet in self._hist for f in triplet]
        img_t = torch.stack(img_batch, dim=0).to(vla._device, vlamod.INFER_DTYPE)
        with torch.no_grad():
            img_feat = vla._siglip(img_t)
            img_tokens = img_feat.reshape(1, -1, img_feat.shape[-1])
        pose = list(self.arm.current_pose_mm_deg[:6])
        state_t, mask_t = vla._pose_to_state(pose, 0.0)
        if self.blind_state:
            state_t = torch.zeros_like(state_t)   # parity with --blind_state training
        with torch.no_grad():
            actions = vla._rdt.predict_action(
                lang_tokens=self.lang_emb, lang_attn_mask=self.lang_mask,
                img_tokens=img_tokens, state_tokens=state_t, action_mask=mask_t,
                ctrl_freqs=torch.tensor([vlamod.CTRL_FREQ], device=vla._device),
            )
        return actions[0].float().cpu().numpy()

    def _measured(self) -> np.ndarray:
        return np.array(self.arm.current_pose_mm_deg[:6], dtype=float)

    def _log_correction(self, move_i: int, pred_tgt_x: float, pred_tgt_y: float,
                        dx: float, dy: float):
        """Record a manual x/y nudge. The model itself cannot learn online, but the
        session's corrections fit a deploy-side offset/affine (--x-cal style) that
        applies instantly — and they are labeled data for the next finetune."""
        self._corr.append((move_i, pred_tgt_x, pred_tgt_y, dx, dy))
        if self.record_dir:
            path = os.path.join(self.record_dir, "corrections.csv")
            new = not os.path.exists(path)
            with open(path, "a") as f:
                if new:
                    f.write("move,wall_t,pred_tgt_x,pred_tgt_y,man_dx,man_dy\n")
                f.write(f"{move_i},{time.time():.2f},{pred_tgt_x:.1f},{pred_tgt_y:.1f},"
                        f"{dx:.1f},{dy:.1f}\n")

    def _correction_summary(self):
        if not self._corr:
            return
        dxs = [c[3] for c in self._corr]; dys = [c[4] for c in self._corr]
        print(f"\n   MANUAL CORRECTIONS this session: {len(self._corr)} "
              f"(mean dx={np.mean(dxs):+.1f}mm  dy={np.mean(dys):+.1f}mm)")
        print(f"   -> next run, try: --x-cal 1,{np.mean(dxs):.0f}"
              f"{'  (y offset needs a y-cal flag if it persists)' if abs(np.mean(dys)) > 5 else ''}")

    def _stamp_truth(self):
        """Stamp the CURRENT arm pose as this run's ground-truth object x,y — jog
        the arm directly over the object, press 't'. Sets true_xy (so later records
        carry it) and writes truth.csv immediately (so predictions made BEFORE the
        stamp still pair to this run's truth — one placement per run)."""
        p = self._measured()
        self.true_xy = (float(p[0]), float(p[1]))
        print(f"   [truth] object stamped at x={p[0]:.1f} y={p[1]:.1f} z={p[2]:.1f}  (labels this run)")
        if self.record_dir:
            with open(os.path.join(self.record_dir, "truth.csv"), "a") as f:
                f.write(f"{time.time():.1f},{p[0]:.1f},{p[1]:.1f},{p[2]:.1f}\n")

    def _record(self, measured, target, chunk):
        """Log the exact model input (2 side cams) + state + full output chunk +
        chosen target, per prediction. This is the arm-room-only artifact: with it
        you can (a) prove offline==live (spot stale/blank side frames), (b) fit the
        x-cal from real placements, (c) build a corrections set to RETRAIN at home."""
        import cv2  # lazy
        self._pred_i += 1
        i, d = self._pred_i, self.record_dir
        if self._last_input is not None:
            cv2.imwrite(os.path.join(d, f"pred{i:03d}_cam1.jpg"), self._last_input[0])
            cv2.imwrite(os.path.join(d, f"pred{i:03d}_cam2.jpg"), self._last_input[1])
        np.savez(os.path.join(d, f"pred{i:03d}.npz"), state=np.asarray(measured),
                 chunk=np.asarray(chunk), target=np.asarray(target),
                 true_xy=np.asarray(self.true_xy if self.true_xy else []))
        tx, ty = (self.true_xy if self.true_xy else ("", ""))
        with open(os.path.join(d, "manifest.csv"), "a") as f:
            f.write(f"{i},{time.time():.1f},{measured[0]:.1f},{measured[1]:.1f},{measured[2]:.1f},"
                    f"{target[0]:.1f},{target[1]:.1f},{target[2]:.1f},{tx},{ty}\n")

    def _cams_live(self) -> bool:
        """Per-move gate on the SIDE cams only (cam1,cam2) — the model's ONLY real
        vision. The claw is NOT a model input (we blank its slot), so a dead claw
        is irrelevant; never block on it. If a side cam goes blank mid-run the
        model IS blind, so reopen the RealSense and re-check; return False only if
        a side cam is still blank after retries."""
        from scripts.vla_observe import cam_health
        def _blank_side():
            health = cam_health(self.vla._grab_frames())
            side = [h for h in health if h.split("[")[0] in ("cam1", "cam2")]
            return health, [h.split("[")[0] for h in side if "BLANK" in h or "NONE" in h]
        health, blank = _blank_side()
        if not blank:
            return True
        print(f"   [cam] SIDE cam {','.join(blank)} went BLANK mid-run — reopening RealSense ...")
        for _ in range(3):
            if self.vla.rs is not None:
                self.vla.rs.stop(); time.sleep(0.5); self.vla.rs.start()
            t0 = time.time()
            while time.time() - t0 < 15.0:
                health, blank = _blank_side()
                if not blank:
                    print(f"   [cam] recovered: {'  '.join(health)}")
                    return True
                time.sleep(0.5)
        print(f"   [cam] side cam NOT recovered: {'  '.join(health)}")
        return False

    def _pick_target(self, chunk: np.ndarray, cur: np.ndarray):
        """Target = chunk[stride], orientation pinned. z is held at a constant hover
        height in XY-ONLY mode, otherwise clamped to the handoff floor.
        Returns (pose_mm_deg, dist_mm, target_in_box)."""
        idx = min(self.stride, chunk.shape[0] - 1)
        pose_next, _grip = self.vla._action_to_pose(chunk[idx])   # pins (180,0,0)
        if self.x_cal is not None:
            pose_next[0] = self.x_cal[0] * pose_next[0] + self.x_cal[1]   # deploy x calibration
        if self.freeze_z_val is not None:
            pose_next[2] = self.freeze_z_val                      # XY-ONLY: ignore model z, hold hover z
        else:
            pose_next[2] = max(pose_next[2], self.z_floor)        # never below handoff plane
        # y is NEVER negative in this workspace (objects sit at y>=0). A predicted
        # y<0 is physically impossible => a misprediction; clamp to the floor so it
        # can't drive the arm the wrong way (run#2 went to y=-28 for a y=+199 obj).
        if pose_next[1] < self.y_floor:
            if pose_next[1] < self.y_floor - 15.0:
                print(f"   [y-clamp] model predicted y={pose_next[1]:.0f} (<{self.y_floor:.0f}, impossible) "
                      f"-> clamped to {self.y_floor:.0f}  (misprediction — check side cams / scene)")
            pose_next[1] = self.y_floor
        dist = float(np.linalg.norm(np.array(pose_next[:3]) - cur[:3]))
        return pose_next, dist, in_box(pose_next[:3], self.box, ignore_z=self.freeze_z_val is not None)

    def _predict_with_safety(self, cur):
        """Predict + re-predict until target is in box, up to N. Returns
        (chunk, pose_next, dist, in_box, retries)."""
        retries = 0
        chunk = self._predict_chunk()
        pose_next, dist, ok = self._pick_target(chunk, cur)
        while not ok and retries < self.max_retries:
            retries += 1
            chunk = self._predict_chunk()
            pose_next, dist, ok = self._pick_target(chunk, cur)
        return chunk, pose_next, dist, ok, retries

    def run(self, max_moves: int):
        a = self.arm
        # Defensive: clear alarms + enable torque (does NOT move the arm).
        try:
            a.ctrl.reset_alarms(); a.ctrl.servo_on()
        except Exception as e:
            print(f"[warn] servo_on/reset_alarms: {e}")

        print("\n" + "=" * 70)
        print(" CLOSED-LOOP STEP-THROUGH — arm WILL move on your Enter")
        print(f"   box x[{self.box[0]:.0f},{self.box[1]:.0f}] y[{self.box[2]:.0f},{self.box[3]:.0f}] z[{self.box[4]:.0f},{self.box[5]:.0f}]  "
              f"z-floor(stop)={self.z_floor:.0f}  stride={self.stride}  max-step={self.max_step:.0f}mm")
        print("   keys at each prompt:  [Enter]=execute move   r=re-predict   "
              "x+20 / y-15=nudge target(mm)   t=stamp TRUE obj xy   +/-=speed   "
              "l=MovP/MovL   s/q=stop")
        print(f"   move type: {'MovL (straight line, consistent speed)' if self.linear else 'MovP (joint-interp; Cartesian speed varies with pose)'}")
        cur = self._measured()
        print(f"   arm now: x={cur[0]:.1f} y={cur[1]:.1f} z={cur[2]:.1f}")
        if self.freeze_z:
            self.freeze_z_val = self.hover_z if self.hover_z is not None else float(cur[2])
            print(f"   MODE: XY-ONLY — z FROZEN at {self.freeze_z_val:.0f}mm "
                  f"(model z ignored; arm holds height, NO descent — verify x,y only)")
        if self.blind_state:
            print("   MODE: BLIND-STATE — state token zeroed (model localizes from vision only)")
        if input("\n   Type 'go' to begin (anything else aborts): ").strip().lower() != "go":
            print("   aborted — no motion sent."); return

        moves = 0
        while moves < max_moves:
            cur = self._measured()
            # Stop condition: reached the handoff plane.
            if cur[2] <= self.z_floor + 2.0:
                print(f"\n*** Reached handoff plane z={cur[2]:.1f} (floor {self.z_floor:.0f}). "
                      f"STOP for the teammate pick program. ***")
                break
            # Trigger B: measured pose left the box (x,y always; z only when not freeze-z,
            # because freeze-z pins z and the arm's physical Cartesian wobble around the
            # hover height is expected and must not hard-stop the run).
            if not in_box(cur[:3], self.box, ignore_z=self.freeze_z):
                print(f"\n*** MEASURED pose out of box: x={cur[0]:.0f} y={cur[1]:.0f} z={cur[2]:.0f}. HARD STOP. ***")
                a.motion_stop(); break

            # Never predict from a dead camera (claw USB drops mid-run).
            if not self._cams_live():
                print("\n*** camera(s) still blank after recovery — NOT predicting blind. ***")
                if input("   [Enter]=retry cameras   s=stop > ").strip().lower() in ("s", "q"):
                    break
                continue

            chunk, tgt, dist, ok, retries = self._predict_with_safety(cur)
            if self.record_dir:
                self._record(cur, tgt, chunk)

            print("\n" + "-" * 70)
            print(f" move {moves+1}/{max_moves}   speed={self.speed}%   retries={retries}")
            print(f"   MEASURED  x={cur[0]:7.1f} y={cur[1]:7.1f} z={cur[2]:7.1f}")
            # Inner prompt loop: nudges/speed changes re-prompt WITHOUT re-predicting
            # (the shown target stays the same chunk); only 'r' re-predicts.
            pred_tgt = (tgt[0], tgt[1])                # model's target before any nudge
            man_dx = man_dy = 0.0
            action = None
            while action is None:
                nudge = f"   [nudged {man_dx:+.0f},{man_dy:+.0f}]" if (man_dx or man_dy) else ""
                print(f"   TARGET    x={tgt[0]:7.1f} y={tgt[1]:7.1f} z={tgt[2]:7.1f}   "
                      f"(Δ {tgt[0]-cur[0]:+.0f},{tgt[1]-cur[1]:+.0f},{tgt[2]-cur[2]:+.0f}  |Δ|={dist:.0f}mm){nudge}")
                # Trigger A: predicted target still out of box after retries.
                if not ok:
                    print(f"   !! target OUT OF BOX after {retries} re-predicts — refusing. (r=re-predict, s=stop)")
                if dist > self.max_step:
                    print(f"   !! move {dist:.0f}mm exceeds --max-step {self.max_step:.0f}mm — refusing. "
                          f"(r=re-predict, or lower --stride)")

                key = input("   > ").strip().lower()
                if key in ("s", "q", "stop", "quit"):
                    action = "stop"
                elif key == "+":
                    self.speed = min(self.max_speed, self.speed + 5)
                    print(f"   speed -> {self.speed}%")
                elif key == "-":
                    self.speed = max(5, self.speed - 5)
                    print(f"   speed -> {self.speed}%")
                elif key == "t":
                    self._stamp_truth()                # arm is on the object -> record true x,y
                elif key == "l":
                    self.linear = not self.linear
                    print(f"   move type -> {'MovL (straight line, consistent speed)' if self.linear else 'MovP (joint-interp)'}")
                elif key == "r":
                    action = "repredict"
                elif key and re.fullmatch(r"(?:\s*[xy][+-]\d+(?:\.\d+)?)+\s*", key):
                    # manual correction, e.g. "x+20", "y-15", "x+20 y-15" (mm)
                    for axis, val in re.findall(r"([xy])([+-]\d+(?:\.\d+)?)", key):
                        d = float(val)
                        if axis == "x":
                            tgt[0] += d; man_dx += d
                        else:
                            tgt[1] += d; man_dy += d
                    dist = float(np.linalg.norm(np.array(tgt[:3]) - cur[:3]))
                    ok = in_box(tgt[:3], self.box, ignore_z=self.freeze_z_val is not None)
                elif key != "":
                    print("   (Enter=execute  r=re-predict  x+20 / y-15=nudge target(mm)  "
                          "t=truth  +/-=speed  s=stop)")
                elif not ok or dist > self.max_step:
                    print("   refused (see !! above). Not moving.")
                else:
                    action = "move"

            if action == "stop":
                print("   stopped by user."); break
            if action == "repredict":
                continue
            if man_dx or man_dy:
                self._log_correction(moves, pred_tgt[0], pred_tgt[1], man_dx, man_dy)

            raw = [int(v * 1000) for v in tgt]
            safe, reason = a.check_safety(raw)        # config-level box too
            if not safe:
                print(f"   config safety reject: {reason}. Not moving."); continue
            print(f"   moving @ {self.speed}% [{'MovL' if self.linear else 'MovP'}] ...", flush=True)
            a.ctrl.move_to(raw, speed=self.speed, wait=True, wait_seconds=3.0, linear=self.linear)
            time.sleep(0.4)                            # let pose poll settle
            moves += 1

        self._correction_summary()
        print(f"\nDone — {moves} move(s) executed. Arm left where it stopped (no auto-home).")


def main() -> int:
    ap = argparse.ArgumentParser(description="Closed-loop step-through with per-move approval (MOVES THE ARM).")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--embed", default=DEFAULT_EMBED)
    ap.add_argument("--instruction", default=DEFAULT_INSTR)
    ap.add_argument("--box", type=parse_box, default=DEFAULT_BOX)
    ap.add_argument("--stride", type=int, default=6, help="chunk steps ahead per approved move")
    ap.add_argument("--max-step", type=float, default=150.0, help="refuse any single move farther than this (mm)")
    ap.add_argument("--z-floor", type=float, default=175.0, help="handoff plane; clamp targets >= this and STOP here (mm)")
    ap.add_argument("--y-floor", type=float, default=0.0, help="y is never negative here; clamp predicted target y >= this (mm)")
    ap.add_argument("--record", default=None,
                    help="dir to log per-prediction side-cam frames + state + model output (arm-room capture for offline debug/retrain)")
    ap.add_argument("--true-xy", type=lambda s: tuple(float(v) for v in s.split(",")), default=None,
                    help="ground-truth object x,y for this run (logged with --record, for calibration)")
    ap.add_argument("--x-cal", type=lambda s: tuple(float(v) for v in s.split(",")), default=None,
                    help="apply x calibration 'm,b': x_target = m*pred_x + b (fix the ~+55mm x under-bias)")
    ap.add_argument("--freeze-z", action="store_true",
                    help="XY-ONLY: ignore the model's z, hold a constant hover z (verify x,y with no descent)")
    ap.add_argument("--blind-state", action="store_true",
                    help="Feed a ZERO state token to the model (required when the checkpoint "
                         "was trained with --blind_state; forces vision-only x,y localization).")
    ap.add_argument("--hover-z", type=float, default=None,
                    help="hover height for --freeze-z (mm); default = arm's z at start (~425 at ready)")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--speed", type=int, default=12, help="initial move speed percent")
    ap.add_argument("--max-speed", type=int, default=40, help="cap for live +/- speed changes")
    ap.add_argument("--linear", action="store_true",
                    help="use MovL (302) straight-line moves: consistent Cartesian speed "
                         "(MovP joint-interp speed varies with pose). Toggle live with 'l'.")
    ap.add_argument("--max-moves", type=int, default=40)
    ap.add_argument("--profile", default=None)
    args = ap.parse_args()

    env_cfg = "config/env/ubuntu.local.yaml"
    if not (BUNDLE_ROOT / env_cfg).exists():
        env_cfg = None
    bundle = load_runtime_bundle(profile=args.profile, env_config=env_cfg)
    demo_cfg, modbus_cfg = bundle.demo_config, bundle.modbus_config

    print("Bringing up cameras + arm ...")
    rs = RealSenseService(demo_cfg)
    claw = ClawCameraService(demo_cfg)
    arm = ArmService(demo_cfg, modbus_cfg, socketio=None)
    rs.start(); claw.start()
    if not arm.auto_connect():
        print("ERROR: arm not connected — cannot step through. Abort."); return 2
    arm.start_pose_polling()

    vla = VLAService(socketio=None, arm_service=arm, realsense_service=rs, claw_service=claw)
    from scripts.vla_observe import cam_health  # reuse the health printer

    def _side_blank():
        health = cam_health(vla._grab_frames())
        side = [h for h in health if h.split("[")[0] in ("cam1", "cam2")]
        return health, [h.split("[")[0] for h in side if "BLANK" in h or "NONE" in h]

    print("Warming up cameras ...")
    # ONLY the 2 side cams (cam1,cam2) are model inputs. The claw slot is blanked
    # for the model (it was never in the 2-camera training set), so a dead/blank
    # claw is irrelevant — do NOT gate on it. Gate on the side cams only.
    t0 = time.time()
    while time.time() - t0 < 8.0:
        c1, c2, _ = vla._grab_frames()
        if c1 is not None and c2 is not None and float(c1.std()) > 3.0 and float(c2.std()) > 3.0:
            break
        time.sleep(0.5)
    health, blank = _side_blank()
    print("Cameras:", "  ".join(health), " (claw = info only; NOT a model input)")
    for attempt in range(1, 4):
        if not blank:
            break
        print(f"*** SIDE cam {','.join(blank)} blank — reopening RealSense ({attempt}/3) ...")
        rs.stop(); time.sleep(0.5); rs.start()
        t0 = time.time()
        while time.time() - t0 < 15.0:
            health, blank = _side_blank()
            if not blank:
                break
            time.sleep(0.5)
        print("Cameras:", "  ".join(health))
    if blank:
        # A blank SIDE cam => the model IS blind (the claw can't help; it's not used).
        print(f"*** SIDE cam {','.join(blank)} BLANK/NONE after recovery — model is BLIND. REFUSING to move. ***")
        print("    The 2 RealSense side cams are the model's only vision. Check they are")
        print("    connected and not held by another process (pkill -f voice_pick_demo.py). Abort. ***")
        return 2
    # need a valid live pose before any motion math
    t0 = time.time()
    while arm.current_pose_mm_deg is None and time.time() - t0 < 5.0:
        time.sleep(0.2)
    if arm.current_pose_mm_deg is None:
        print("ERROR: no arm pose from Modbus. Abort."); return 2

    # Refuse to run on CPU: with CUDA wedged (e.g. the post-suspend nvidia_uvm bug)
    # torch silently falls back to CPU, the model still loads, and the first predict
    # after 'go' crawls for many minutes — indistinguishable from a hang.
    if not torch.cuda.is_available():
        print("*** CUDA NOT AVAILABLE — refusing to run the model on CPU (looks like a hang). ***")
        print("    Likely the post-suspend wedge. Fix:  sudo modprobe -r nvidia_uvm && sudo modprobe nvidia_uvm")
        print("    Then verify:  python -c \"import torch; print(torch.cuda.is_available())\"  and re-run. Abort.")
        return 2
    print("Loading model (bf16) ...")
    vla._ensure_models(str((BUNDLE_ROOT / args.checkpoint).resolve()))
    lang_emb, lang_mask = vla._load_lang_embed(str((BUNDLE_ROOT / args.embed).resolve()),
                                               args.instruction)
    st = StepThrough(vla, arm, args)
    st.lang_emb, st.lang_mask = lang_emb, lang_mask
    try:
        st.run(args.max_moves)
    except KeyboardInterrupt:
        print("\nInterrupted — sending motion stop.")
        try: arm.motion_stop()
        except Exception: pass
    finally:
        try:
            arm.stop_pose_polling(); rs.stop(); claw.stop()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
