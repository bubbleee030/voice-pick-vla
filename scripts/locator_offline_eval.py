#!/usr/bin/env python3
"""Off-rig eval of the side-cam locator against recorded episodes (§3.6).

Each episode's FIRST cam1 frame is taken at the ready pose (exactly how the
locator will be used live); the episode's final trajectory (x,y) is where the
object actually was. Reports per-episode locate error in mm — should
reproduce the calibration's LOO numbers (3–20 mm).

Run:
    conda run --no-capture-output -n voice_pick python scripts/locator_offline_eval.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

BUNDLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BUNDLE_ROOT))

from src.locator import SideCamLocator  # noqa: E402


def final_xy(ep: Path) -> tuple[float, float] | None:
    last = None
    with open(ep / "trajectory.csv") as f:
        for row in csv.DictReader(f):
            if row.get("valid") == "1":
                last = (float(row["x_mm"]), float(row["y_mm"]))
    return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", default="data/recordings")
    ap.add_argument("--objects", default="board,butter_knife,trapezoid")
    args = ap.parse_args()

    import cv2
    loc = SideCamLocator()

    for obj in args.objects.split(","):
        errs, misses = [], 0
        for ep in sorted((BUNDLE_ROOT / args.recordings / obj).glob("episode_*")):
            frames = sorted((ep / "cam1_rgb").glob("frame_*.jpg"))
            truth = final_xy(ep)
            if not frames or truth is None:
                continue
            img = cv2.imread(str(frames[0]))
            r = loc.locate(img, obj) if img is not None else None
            if r is None:
                misses += 1
                print(f"  {obj}/{ep.name}: NO DETECTION")
                continue
            ex, ey = r.x_mm - truth[0], r.y_mm - truth[1]
            errs.append((abs(ex), abs(ey)))
            print(f"  {obj}/{ep.name}: pred=({r.x_mm:.0f},{r.y_mm:.0f}) "
                  f"true=({truth[0]:.0f},{truth[1]:.0f}) err=({ex:+.0f},{ey:+.0f})mm conf={r.conf:.2f}")
        if errs:
            e = np.array(errs)
            print(f"[{obj}] n={len(errs)} miss={misses}  "
                  f"MAE=({e[:,0].mean():.0f},{e[:,1].mean():.0f})mm  "
                  f"max=({e[:,0].max():.0f},{e[:,1].max():.0f})mm  "
                  f"{'OK (<±100mm needed)' if e.max() < 100 else 'CHECK: outside servo catch basin'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
