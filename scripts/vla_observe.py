#!/usr/bin/env python3
"""Dry-observe usability check for a trained RDT/VLA checkpoint.

Runs the EXACT deploy inference pipeline (same SigLIP preprocessing, same
predict_action call as src/services/vla_service.py) against LIVE cameras + the
LIVE measured arm pose, but sends ZERO motion to the arm. Each cycle it predicts
the full 64-step action chunk from the (static) current pose and shows you, live:

  * measured pose vs the model's predicted next target (and the whole chunk)
  * in-box % per axis  -> catches x-collapse / y-fold instantly
  * chunk shape        -> does z descend toward the object, gripper close?
  * object-tracking     -> move the object, prediction should follow
  * raw predicted rx/ry/rz (before the (180,0,0) pin) -> is rotation garbage?
  * per-camera live/BLANK health (a blind cam => folded trajectory, NOT a verdict)

Inference runs in a BACKGROUND thread so the camera preview stays smooth even
though one prediction takes ~0.8s (x N re-predicts). A safety box (one box, two
triggers) is exercised motion-free: predicted t0 target outside box -> re-predict
up to N times, then annotate "[LIVE: STOP]"; measured pose outside box -> same.
In this dry tool both only LOG/annotate; the same logic arms the later live tool.

This tool NEVER calls move_to / go_ready / servo. Put the arm at ready yourself.

Run (use --no-capture-output or the terminal dashboard stays buffered/blank):
    conda run --no-capture-output -n voice_pick python scripts/vla_observe.py
    ... --box 320,640,-50,235,172,426 --max-retries 0   # snappier tracking
Quit: press 'q' in the OpenCV window, or Ctrl+C.
"""
from __future__ import annotations

import argparse
import csv
import threading
import sys
import time
from datetime import datetime
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BUNDLE_ROOT))

import numpy as np  # noqa: E402

from src.runtime_config import load_runtime_bundle           # noqa: E402
from src.services.arm_service import ArmService              # noqa: E402
from src.services.claw_camera_service import ClawCameraService  # noqa: E402
from src.services.realsense_service import RealSenseService  # noqa: E402
from src.services import vla_service as vlamod               # noqa: E402
from src.services.vla_service import VLAService              # noqa: E402

import torch  # noqa: E402
import cv2    # noqa: E402

DEFAULT_CKPT  = "data/datasets/trapezoid_0701/checkpoint-24000"
DEFAULT_EMBED = "data/vla_embed_trapezoid_en.pt"
DEFAULT_INSTR = "拿起梯形"
DEFAULT_BOX   = (320.0, 640.0, -50.0, 235.0, 172.0, 426.0)  # x,x,y,y,z,z mm
CSV_DIR       = BUNDLE_ROOT / "data" / "observe_logs"

CLR = "\033[2J\033[3J\033[H"   # clear screen + scrollback, cursor home


def parse_box(s: str) -> tuple[float, ...]:
    parts = [float(v) for v in s.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("box must be x_min,x_max,y_min,y_max,z_min,z_max")
    return tuple(parts)


def display_crop(bgr: np.ndarray | None) -> np.ndarray:
    """Same centre-pad + 384 resize the model sees (sans normalization), kept BGR
    for imshow. Lets you confirm the object is framed/centred."""
    if bgr is None:
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    h, w = bgr.shape[:2]
    m = max(h, w)
    sq = np.zeros((m, m, 3), dtype=bgr.dtype)
    sq[(m - h) // 2:(m - h) // 2 + h, (m - w) // 2:(m - w) // 2 + w, :] = bgr
    return cv2.resize(sq, (384, 384), interpolation=cv2.INTER_AREA)


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def in_box(xyz, box) -> bool:
    x, y, z = xyz
    return (box[0] <= x <= box[1]) and (box[2] <= y <= box[3]) and (box[4] <= z <= box[5])


def axis_inbox_pct(chunk: np.ndarray, box) -> tuple[float, float, float]:
    n = chunk.shape[0]
    px = float(((chunk[:, 0] >= box[0]) & (chunk[:, 0] <= box[1])).sum()) / n * 100
    py = float(((chunk[:, 1] >= box[2]) & (chunk[:, 1] <= box[3])).sum()) / n * 100
    pz = float(((chunk[:, 2] >= box[4]) & (chunk[:, 2] <= box[5])).sum()) / n * 100
    return px, py, pz


def cam_health(frames) -> list[str]:
    """Per-camera mean/std + a blank heuristic (std<3 => black/frozen => model is
    blind on that input). A blind cam is the documented cause of trajectory fold."""
    out = []
    for f, n in zip(frames, ("cam1", "cam2", "claw")):
        if f is None:
            out.append(f"{n}[NONE]")
            continue
        m, s = float(f.mean()), float(f.std())
        tag = "BLANK" if s < 3.0 else "live"
        out.append(f"{n}[{tag} m{m:.0f} s{s:.0f}]")
    return out


def build_preview(frames) -> np.ndarray:
    raws = [label(cv2.resize(f if f is not None else np.zeros((480, 640, 3), np.uint8),
                             (320, 240)), n)
            for f, n in zip(frames, ("cam1", "cam2", "claw"))]
    crops = [label(cv2.resize(display_crop(f), (320, 240)), f"{n} -> siglip 384")
             for f, n in zip(frames, ("cam1", "cam2", "claw"))]
    return np.vstack([np.hstack(raws), np.hstack(crops)])


def render_terminal(state: dict) -> None:
    box = state["box"]
    chunk = state["chunk"]
    meas = state["meas"]
    px, py, pz = state["inbox"]
    g0, gN = float(np.clip(chunk[0, 9], 0, 1)), float(np.clip(chunk[-1, 9], 0, 1))
    raw0 = vlamod._rot6d_to_euler(chunk[0, 3:9])
    rzs = np.array([vlamod._rot6d_to_euler(chunk[i, 3:9]) for i in range(0, chunk.shape[0], 8)])
    meas_in = in_box(meas[:3], box) if meas is not None else False

    def mark(ok): return "[ OK ]" if ok else "[!OUT]"

    lines = []
    lines.append("=" * 72)
    lines.append(f" VLA DRY-OBSERVE  (NO MOTION SENT)   cycle={state['cycle']}  "
                 f"latency={state['latency_ms']:6.0f}ms  retries={state['retries']}")
    lines.append(f" ckpt={state['ckpt_short']}")
    lines.append(f" instr={state['instr']}   dev={state['device']}  dtype={vlamod.INFER_DTYPE}")
    lines.append(f" box  x[{box[0]:.0f},{box[1]:.0f}] y[{box[2]:.0f},{box[3]:.0f}] z[{box[4]:.0f},{box[5]:.0f}] mm")
    if state.get("frames") is not None:
        lines.append(" CAMS  " + "  ".join(cam_health(state["frames"])))
    lines.append("-" * 72)
    if meas is not None:
        lines.append(f" MEASURED  x={meas[0]:8.1f} y={meas[1]:8.1f} z={meas[2]:8.1f}   "
                     f"rx={meas[3]:7.1f} ry={meas[4]:7.1f} rz={meas[5]:7.1f}  {mark(meas_in)}")
    else:
        lines.append(" MEASURED  <arm not connected>")
    lines.append(f" PRED t0   x={chunk[0,0]:8.1f} y={chunk[0,1]:8.1f} z={chunk[0,2]:8.1f}   "
                 f"grip={g0:4.2f}   {mark(state['step0_in'])}")
    lines.append(f" PRED rawR rx={raw0[0]:7.1f} ry={raw0[1]:7.1f} rz={raw0[2]:7.1f}  "
                 f"(pinned to 180,0,0 on the real arm)")
    lines.append("-" * 72)
    lines.append(" CHUNK (64 steps)")
    lines.append(f"   x  range [{chunk[:,0].min():7.1f} .. {chunk[:,0].max():7.1f}]   in-box {px:5.1f}%")
    lines.append(f"   y  range [{chunk[:,1].min():7.1f} .. {chunk[:,1].max():7.1f}]   in-box {py:5.1f}%")
    lines.append(f"   z  {chunk[0,2]:7.1f} -> {chunk[-1,2]:7.1f}  ({'DESCEND' if chunk[-1,2] < chunk[0,2] else 'rise/flat'})   in-box {pz:5.1f}%")
    lines.append(f"   grip {g0:4.2f} -> {gN:4.2f}  ({'closing' if gN > g0 else 'opening/flat'})")
    lines.append(f"   raw rz spread [{rzs[:,2].min():6.1f} .. {rzs[:,2].max():6.1f}]   "
                 f"rx spread [{rzs[:,0].min():6.1f} .. {rzs[:,0].max():6.1f}]")
    lines.append("-" * 72)
    trk = state.get("track")
    if trk is not None:
        lines.append(f" OBJECT-TRACK  d(t0 target) vs last cycle: "
                     f"|Δ|={trk['mag']:6.1f}mm  Δx={trk['dx']:6.1f} Δy={trk['dy']:6.1f} Δz={trk['dz']:6.1f}")
    else:
        lines.append(" OBJECT-TRACK  (first cycle)")
    warns = []
    blank_cams = [c.split("[")[0] for c in cam_health(state["frames"]) if "BLANK" in c or "NONE" in c] if state.get("frames") is not None else []
    if blank_cams:
        warns.append(f"{','.join(blank_cams)} BLANK -> model is BLIND, x-fold expected (not a model verdict)")
    if px < 50:
        warns.append(f"x in-box {px:.0f}% -> possible x-collapse")
    if not state["step0_in"]:
        warns.append(f"t0 target out of box after {state['retries']} re-predicts -> LIVE: STOP")
    if meas is not None and not meas_in:
        warns.append("MEASURED pose out of box -> LIVE: STOP (check ready/box)")
    if warns:
        lines.append(" \033[93mWARN: " + " | ".join(warns) + "\033[0m")
    lines.append("=" * 72)
    lines.append(" Move the object to test tracking. Press 'q' in the camera window (or Ctrl+C) to quit.")
    print(CLR + "\n".join(lines), flush=True)


class Observer:
    """Runs the inference loop in a background thread; main thread reads `latest`."""

    def __init__(self, vla, arm, box, max_retries, writer, t_start, max_steps):
        self.vla = vla
        self.arm = arm
        self.box = box
        self.max_retries = max_retries
        self.writer = writer
        self.t_start = t_start
        self.max_steps = max_steps
        self._hist: list[tuple] = []
        self.last_frames = None
        self._lock = threading.Lock()
        self.latest: dict | None = None
        self.stop_flag = False
        self.cycle = 0
        self._prev_target = None

    def _infer_once(self) -> np.ndarray:
        vla = self.vla
        frames = vla._grab_frames()
        self.last_frames = frames
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

        pose = [0.0] * 6
        if self.arm is not None and self.arm.current_pose_mm_deg is not None:
            pose = list(self.arm.current_pose_mm_deg[:6])
        state_t, mask_t = vla._pose_to_state(pose, 0.0)
        with torch.no_grad():
            actions = vla._rdt.predict_action(
                lang_tokens=self.lang_emb, lang_attn_mask=self.lang_mask,
                img_tokens=img_tokens, state_tokens=state_t, action_mask=mask_t,
                ctrl_freqs=torch.tensor([vlamod.CTRL_FREQ], device=vla._device),
            )
        return actions[0].float().cpu().numpy()

    def loop(self, meta: dict):
        while not self.stop_flag:
            t0 = time.time()
            chunk = self._infer_once()
            retries = 0
            while not in_box(chunk[0, :3], self.box) and retries < self.max_retries:
                retries += 1
                chunk = self._infer_once()
            latency_ms = (time.time() - t0) * 1000.0
            step0_in = in_box(chunk[0, :3], self.box)

            meas = (np.array(self.arm.current_pose_mm_deg[:6], dtype=float)
                    if self.arm.current_pose_mm_deg is not None else None)
            inbox = axis_inbox_pct(chunk, self.box)
            target = chunk[0, :3].copy()
            track = None
            if self._prev_target is not None:
                d = target - self._prev_target
                track = {"dx": float(d[0]), "dy": float(d[1]), "dz": float(d[2]),
                         "mag": float(np.linalg.norm(d))}
            self._prev_target = target

            raw0 = vlamod._rot6d_to_euler(chunk[0, 3:9])
            g0, gN = float(np.clip(chunk[0, 9], 0, 1)), float(np.clip(chunk[-1, 9], 0, 1))
            meas_in = bool(in_box(meas[:3], self.box)) if meas is not None else False
            self.writer.writerow([
                self.cycle, f"{time.time()-self.t_start:.3f}", f"{latency_ms:.0f}",
                retries, int(step0_in), int(meas_in),
                *([f"{v:.2f}" for v in meas] if meas is not None else [""] * 6),
                f"{target[0]:.2f}", f"{target[1]:.2f}", f"{target[2]:.2f}",
                f"{g0:.3f}", f"{gN:.3f}", f"{raw0[0]:.2f}", f"{raw0[1]:.2f}", f"{raw0[2]:.2f}",
                f"{inbox[0]:.1f}", f"{inbox[1]:.1f}", f"{inbox[2]:.1f}",
                f"{chunk[0,2]:.2f}", f"{chunk[-1,2]:.2f}",
                f"{track['mag']:.2f}" if track else "",
            ])

            with self._lock:
                self.latest = {
                    "cycle": self.cycle, "latency_ms": latency_ms, "retries": retries,
                    "box": self.box, "chunk": chunk, "meas": meas, "inbox": inbox,
                    "step0_in": step0_in, "track": track, "frames": self.last_frames,
                    **meta,
                }
            self.cycle += 1
            if self.max_steps and self.cycle >= self.max_steps:
                self.stop_flag = True

    def snapshot(self) -> dict | None:
        with self._lock:
            return self.latest


def main() -> int:
    ap = argparse.ArgumentParser(description="Dry-observe VLA usability check (no motion).")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--embed", default=DEFAULT_EMBED)
    ap.add_argument("--instruction", default=DEFAULT_INSTR)
    ap.add_argument("--box", type=parse_box, default=DEFAULT_BOX,
                    help="x_min,x_max,y_min,y_max,z_min,z_max in mm")
    ap.add_argument("--max-retries", type=int, default=3,
                    help="re-predicts when t0 target is out of box (0 = snappier tracking)")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = until quit")
    ap.add_argument("--no-window", action="store_true", help="disable OpenCV preview")
    ap.add_argument("--profile", default=None)
    args = ap.parse_args()

    env_cfg = "config/env/ubuntu.local.yaml"
    if not (BUNDLE_ROOT / env_cfg).exists():
        env_cfg = None
    bundle = load_runtime_bundle(profile=args.profile, env_config=env_cfg)
    demo_cfg, modbus_cfg = bundle.demo_config, bundle.modbus_config

    print("Bringing up cameras + arm (READ-ONLY; no motion will be sent) ...")
    rs = RealSenseService(demo_cfg)
    claw = ClawCameraService(demo_cfg)
    arm = ArmService(demo_cfg, modbus_cfg, socketio=None)
    rs.start()
    claw.start()
    if arm.auto_connect():
        arm.start_pose_polling()
        print("Arm connected (read-only polling).")
    else:
        print("WARN: arm not connected — measured pose will be blank.")

    vla = VLAService(socketio=None, arm_service=arm, realsense_service=rs, claw_service=claw)

    print("Warming up cameras ...")
    t_warm = time.time()
    while time.time() - t_warm < 6.0:
        fr = vla._grab_frames()
        if any(f is not None and float(f.std()) > 3.0 for f in fr):
            break
        time.sleep(0.5)
    health = cam_health(vla._grab_frames())
    print("Cameras:", "  ".join(health))
    if all(("BLANK" in h or "NONE" in h) for h in health):
        print("\n*** ALL CAMERAS BLANK — predictions will be BLIND (x-fold expected). ***")
        print("    Another process likely owns the cameras (demo server :8090).")
        print("    Stop it:  pkill -f voice_pick_demo.py   then re-run.\n")

    print("Loading model (bf16) ...")
    vla._ensure_models(str((BUNDLE_ROOT / args.checkpoint).resolve()))
    lang_emb, lang_mask = vla._load_lang_embed(str((BUNDLE_ROOT / args.embed).resolve()),
                                               args.instruction)

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = CSV_DIR / f"vla_observe_{datetime.now():%Y%m%d_%H%M%S}.csv"
    csv_f = open(csv_path, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow([
        "cycle", "t", "latency_ms", "retries", "step0_in_box", "meas_in_box",
        "meas_x", "meas_y", "meas_z", "meas_rx", "meas_ry", "meas_rz",
        "pred_x", "pred_y", "pred_z", "grip0", "gripN", "raw_rx", "raw_ry", "raw_rz",
        "inbox_x_pct", "inbox_y_pct", "inbox_z_pct", "z_start", "z_end", "track_mag",
    ])

    obs = Observer(vla, arm, args.box, args.max_retries, writer, time.time(), args.max_steps)
    obs.lang_emb, obs.lang_mask = lang_emb, lang_mask
    meta = {"ckpt_short": "/".join(Path(args.checkpoint).parts[-2:]),
            "instr": args.instruction, "device": str(vla._device)}

    win = "VLA observe — cam1|cam2|claw (top) / siglip crops (bottom)"
    if not args.no_window:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    worker = threading.Thread(target=obs.loop, args=(meta,), daemon=True)
    worker.start()
    print("Running. NO motion is sent. Place the trapezoid in view; move it to test tracking.\n")

    last_render = 0.0
    try:
        # Main thread = smooth display: refresh preview from FRESH frames every ~30ms,
        # overlay the latest (slow) prediction read from the worker.
        while not obs.stop_flag:
            if not args.no_window:
                frames = vla._grab_frames()
                cv2.imshow(win, build_preview(frames))
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            else:
                time.sleep(0.05)
            snap = obs.snapshot()
            if snap is not None and time.time() - last_render > 0.2:
                render_terminal(snap)
                last_render = time.time()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        obs.stop_flag = True
        worker.join(timeout=5.0)
        csv_f.close()
        if not args.no_window:
            cv2.destroyAllWindows()
        try:
            arm.stop_pose_polling()
            rs.stop()
            claw.stop()
        except Exception:
            pass
        print(f"\nCSV written: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
