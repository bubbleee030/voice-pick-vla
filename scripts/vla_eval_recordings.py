#!/usr/bin/env python3
"""Score a checkpoint's predicted (x,y) against recorded EPISODES (board /
butter_knife / trapezoid), for objects that have no live_capture eval set.

Each recording starts at the ready pose, so frame 0 of cam1/cam2 is exactly what
deploy would see before the first prediction; ground truth is the episode's final
trajectory (x,y). Blind-state parity (zeroed state), same as vla_eval_placements.

Run per object (5090, rdt env, needs the libstdc++ preload for cv2):
    python scripts/vla_eval_recordings.py \
        --checkpoint data/datasets/checkpoints/<threeobj ckpt> \
        --object board            # board | butter_knife | trapezoid
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BUNDLE_ROOT))
sys.path.insert(0, str(BUNDLE_ROOT / "scripts"))

import numpy as np  # noqa: E402
from vla_eval_placements import predict_xy  # noqa: E402  (reuses blind-state predict)

OBJ = {
    "board":        {"embed": "data/vla_embed_board_zh.pt",        "instruction": "拿起電路板"},
    "butter_knife": {"embed": "data/vla_embed_butter_knife_zh.pt", "instruction": "拿起奶油刀"},
    "trapezoid":    {"embed": "data/vla_embed_trapezoid_zh256.pt", "instruction": "拿起梯形"},
}


def episode_gt(ep: Path) -> tuple[float, float]:
    last = None
    with open(ep / "trajectory.csv") as f:
        for row in csv.DictReader(f):
            if row.get("valid") == "1":
                last = row
    return float(last["x_mm"]), float(last["y_mm"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--object", required=True, choices=sorted(OBJ))
    ap.add_argument("--recordings", default="data/recordings")
    ap.add_argument("--embed", default=None, help="default: the object's zh embed")
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--repeat", type=int, default=8,
                    help="diffusion samples averaged per episode")
    args = ap.parse_args()
    cfg = OBJ[args.object]
    embed = args.embed or cfg["embed"]

    import cv2
    from src.services import vla_service as vlamod
    from src.services.vla_service import VLAService
    vla = VLAService(socketio=None, arm_service=None, realsense_service=None, claw_service=None)
    vla._ensure_models(str((BUNDLE_ROOT / args.checkpoint).resolve()))
    lang_emb, lang_mask = vla._load_lang_embed(str((BUNDLE_ROOT / embed).resolve()),
                                               cfg["instruction"])
    vla._lang_emb, vla._lang_mask = lang_emb, lang_mask

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    rows, preds = [], []
    for ep in sorted((BUNDLE_ROOT / args.recordings / args.object).glob("episode_*")):
        c1 = cv2.imread(str(ep / "cam1_rgb" / "frame_000000.jpg"))
        c2 = cv2.imread(str(ep / "cam2_rgb" / "frame_000000.jpg"))
        if c1 is None or c2 is None:
            continue
        vla._hist = [(c1, c2, blank)] * vlamod.IMG_HISTORY
        px, py = predict_xy(vla, args.stride, args.repeat)
        tx, ty = episode_gt(ep)
        rows.append((ep.name, tx, ty, px, py, px - tx, py - ty))
        preds.append((tx, ty, px, py))

    print(f"\n[{args.object}] embed={embed} instruction={cfg['instruction']} "
          f"repeat={args.repeat}")
    print(f"{'episode':<13}{'true_x':>8}{'true_y':>8}{'pred_x':>8}{'pred_y':>8}{'err_x':>8}{'err_y':>8}")
    for k, tx, ty, px, py, ex, ey in rows:
        print(f"{k:<13}{tx:8.0f}{ty:8.0f}{px:8.0f}{py:8.0f}{ex:8.0f}{ey:8.0f}")
    if len(preds) >= 2:
        a = np.array(preds)
        mae_x, mae_y = np.mean(np.abs(a[:, 2] - a[:, 0])), np.mean(np.abs(a[:, 3] - a[:, 1]))
        cx = np.corrcoef(a[:, 0], a[:, 2])[0, 1] if np.std(a[:, 2]) > 1e-6 and np.std(a[:, 0]) > 1e-6 else 0.0
        cy = np.corrcoef(a[:, 1], a[:, 3])[0, 1] if np.std(a[:, 3]) > 1e-6 and np.std(a[:, 1]) > 1e-6 else 0.0
        print(f"\nMAE  x={mae_x:.0f}mm  y={mae_y:.0f}mm")
        print(f"corr x={cx:+.2f}  y={cy:+.2f}   (pred std: x={np.std(a[:,2]):.0f} y={np.std(a[:,3]):.0f})")
        print("NOTE: episodes are TRAINING data — this detects collapse/mixups, not "
              "generalization. Demo placement = these exact spots, so it is the demo metric.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
