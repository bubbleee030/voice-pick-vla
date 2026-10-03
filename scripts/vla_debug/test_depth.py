"""
Depth-tracking test for the retrained trapezoid checkpoint.
Default target: data/datasets/checkpoints/trapezoid_20260622/checkpoint-30000 (override via argv[1]).

The old model collapsed eef-x to ~290 regardless of the object's depth. This feeds
the model RECORDED frames + state from episodes that grasp at DIFFERENT depths
(near ~332 mm ... far ~636 mm) and reports where the model wants to go in x:
  - PRED x[0]      : immediate next x in the predicted 64-step chunk
  - PRED x@zmin    : x at the chunk's lowest z (the model's intended grasp depth)
  - PRED x max     : furthest-out x the chunk reaches
If PRED x@zmin / x-max now TRACK the episode's true grasp-x across depths, x is fixed.

Run:
  LD_LIBRARY_PATH=$RDT_ENV/lib  rdt_python  scripts/vla_debug/test_depth.py
  # or test a specific checkpoint:
  LD_LIBRARY_PATH=$RDT_ENV/lib  rdt_python  scripts/vla_debug/test_depth.py  <path/to/checkpoint-NNNN>
"""
from __future__ import annotations
import csv, glob, os, sys
import numpy as np
import cv2
import torch

BUNDLE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BUNDLE)
sys.path.insert(0, os.path.join(BUNDLE, "vla"))  # laptop: bundled deploy model code

from models.rdt_runner import RDTRunner
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from src.services.vla_service import _prep_image, _euler_to_rot6d, INFER_DTYPE
if os.environ.get("VLA_DTYPE") == "fp32":   # laptop-side dtype probe (deploy uses bf16)
    INFER_DTYPE = torch.float32

# Checkpoint to test: pass an ema dir (or a checkpoint dir; we append /ema) as argv[1].
# Default is the latest retrained trapezoid checkpoint (step 30000, final).
_DEFAULT_CKPT = os.path.join(BUNDLE, "data/datasets/checkpoints/trapezoid_20260622/checkpoint-30000/ema")
def _resolve_ckpt():
    if len(sys.argv) > 1:
        p = sys.argv[1]
        if os.path.basename(p.rstrip("/")) != "ema" and os.path.isdir(os.path.join(p, "ema")):
            p = os.path.join(p, "ema")
        return p
    return _DEFAULT_CKPT
CKPT  = _resolve_ckpt()
EMBED = os.path.join(BUNDLE, "data/datasets/trapezoid_processed/episode_0/instruction_embedding.pt")
STATE_DIM, CTRL_FREQ, IMG_HISTORY, NSAMP = 128, 6, 2, 4
EARLY_T = 4  # early timestep: arm still ~ready, object visible at its depth

EPISODES = [  # (raw episode, label) spanning depth
    ("episode_012", "near"), ("episode_010", "near-mid"),
    ("episode_001", "mid"),  ("episode_007", "mid-far"), ("episode_004", "far")]


def load_traj(ep):
    rows = []
    with open(os.path.join(ep, "trajectory.csv")) as f:
        for r in csv.DictReader(f):
            if r.get("valid", "1") in ("0", "0.0"):
                continue
            rows.append([float(r[k]) for k in ("x_mm","y_mm","z_mm","rx_deg","ry_deg","rz_deg")])
    return rows


def frame(ep, cam, idx, n_cam, n_traj):
    j = idx if n_cam == n_traj else round(idx * (n_cam - 1) / max(1, n_traj - 1))
    fs = sorted(glob.glob(os.path.join(ep, cam, "frame_*.jpg")))
    return cv2.imread(fs[min(j, len(fs) - 1)])  # BGR -> _prep_image swaps to RGB (training-faithful)


def state_tensor(pose, device):
    x, y, z, rx, ry, rz = pose
    s = np.zeros(STATE_DIM, np.float32); m = np.zeros(STATE_DIM, np.float32)
    s[0:3] = [x, y, z]; s[3:9] = _euler_to_rot6d(rx, ry, rz)
    m[0:10] = 1.0
    return (torch.from_numpy(s)[None, None].to(device, INFER_DTYPE),
            torch.from_numpy(m)[None, None].to(device, INFER_DTYPE))


def main():
    device = torch.device("cuda")
    print(f"loading {CKPT} ...")
    rdt = RDTRunner.from_pretrained(CKPT).to(INFER_DTYPE).to(device).eval()
    siglip = SiglipVisionTower("google/siglip-so400m-patch14-384").to(INFER_DTYPE).to(device).eval()
    d = torch.load(EMBED, map_location="cpu", weights_only=True)
    emb = (d if d.dim() == 3 else d.unsqueeze(0)).to(device, INFER_DTYPE)
    lmask = torch.ones(emb.shape[:2]).to(device, INFER_DTYPE)

    print(f"\n{'episode':<22}{'GT grasp-x':>11}{'state-x':>9}{'PRED x[0]':>11}{'PRED x@zmin':>13}{'PRED x-max':>11}")
    print("-" * 77)
    rows = []
    all_eps = sorted(glob.glob(os.path.join(BUNDLE, "data/recordings/trapezoid/episode_*")),
                     key=lambda p: int(p.split("_")[-1]))
    for ep in all_eps:
        epname, label = os.path.basename(ep), ""
        traj = load_traj(ep); n = len(traj)
        gt_grasp_x = traj[int(np.argmin([p[2] for p in traj]))][0]
        ncam = {c: len(glob.glob(os.path.join(ep, c, "frame_*.jpg"))) for c in ("cam1_rgb","cam2_rgb","claw_rgb")}
        t = min(EARLY_T, n - 2)
        st, sm = state_tensor(traj[t], device)
        hist = [max(0, t - 1), t]
        trip = []
        for ti in hist:
            trip += [frame(ep, "cam1_rgb", ti, ncam["cam1_rgb"], n),
                     frame(ep, "cam2_rgb", ti, ncam["cam2_rgb"], n),
                     frame(ep, "claw_rgb", ti, ncam["claw_rgb"], n)]
        batch = torch.stack([_prep_image(f, siglip.image_processor) for f in trip], 0).to(device, INFER_DTYPE)
        with torch.no_grad():
            imgtok = siglip(batch).reshape(1, -1, 1152)
        chunks = []
        for _ in range(NSAMP):
            with torch.no_grad():
                a = rdt.predict_action(lang_tokens=emb, lang_attn_mask=lmask, img_tokens=imgtok,
                                       state_tokens=st, action_mask=sm,
                                       ctrl_freqs=torch.tensor([CTRL_FREQ], device=device))
            chunks.append(a[0].float().cpu().numpy())
        ch = np.mean(chunks, 0)  # (64, 128)
        x0 = ch[0, 0]; x_zmin = ch[int(np.argmin(ch[:, 2])), 0]; xmax = ch[:, 0].max()
        rows.append((gt_grasp_x, x_zmin))
        print(f"{epname+' ('+label+')':<22}{gt_grasp_x:>11.0f}{traj[t][0]:>9.0f}{x0:>11.0f}{x_zmin:>13.0f}{xmax:>11.0f}")

    gt = np.array([r[0] for r in rows]); pr = np.array([r[1] for r in rows])
    print("-" * 77)
    print(f"GT grasp-x span : {gt.min():.0f}..{gt.max():.0f}  (range {np.ptp(gt):.0f})")
    print(f"PRED x@zmin span: {pr.min():.0f}..{pr.max():.0f}  (range {np.ptp(pr):.0f})")
    if gt.std() > 1e-6:
        print(f"correlation(GT, PRED) = {np.corrcoef(gt, pr)[0,1]:+.2f}   "
              f"(old model: PRED ~const ~290-430 regardless of GT)")


if __name__ == "__main__":
    main()
