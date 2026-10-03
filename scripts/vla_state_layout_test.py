"""
Decide whether the deploy state/action layout matches the trained RDT contract.

Hypothesis: training used the official STATE_VEC_IDX_MAPPING (eef pos -> idx
30,31,32; eef 6D angle -> 33..38; gripper_open -> 10), but deploy's
_pose_to_state/_action_to_pose use idx 0..9. If so, packing state at the OFFICIAL
indices and reading action at the OFFICIAL indices should make the predicted x
fall back into the trained range (trapezoid x in [387,619]); the deploy 0..9
layout yields the OOD x~305.

Run with the server STOPPED:
    conda run -n voice_pick python scripts/vla_state_layout_test.py --n 4
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np
import cv2
import torch

BUNDLE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BUNDLE_ROOT)
sys.path.insert(0, os.path.join(BUNDLE_ROOT, "vla"))

from models.rdt_runner import RDTRunner
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from src.services.vla_service import _prep_image, _euler_to_rot6d, INFER_DTYPE

CKPT = os.path.join(BUNDLE_ROOT, "data/datasets/checkpoints/combined_nchc_20260529_120309")
EMBED = os.path.join(BUNDLE_ROOT, "data/vla_embed_trapezoid_en.pt")
SNAPDIR = os.path.join(BUNDLE_ROOT, "data/runtime/snapshots")
STATE_DIM, CTRL_FREQ = 128, 6


def newest(cam):
    fs = sorted(glob.glob(os.path.join(SNAPDIR, f"{cam}_rgb_*.jpg")))
    return fs[-1]


def load_embed(device):
    d = torch.load(EMBED, map_location="cpu", weights_only=True)
    emb = d if d.dim() == 3 else d.unsqueeze(0)
    return emb.to(device, INFER_DTYPE), torch.ones(emb.shape[:2]).to(device, INFER_DTYPE)


def pack(pose, layout, device):
    x, y, z, rx, ry, rz = pose
    r6d = _euler_to_rot6d(rx, ry, rz)
    s = np.zeros(STATE_DIM, dtype=np.float32)
    m = np.zeros(STATE_DIM, dtype=np.float32)
    if layout == "deploy":      # idx 0..9
        s[0:3] = [x, y, z]; s[3:9] = r6d; s[9] = 0.0
        idx = list(range(10))
    else:                       # official STATE_VEC_IDX_MAPPING
        s[30:33] = [x, y, z]; s[33:39] = r6d; s[10] = 0.0
        idx = [10, 30, 31, 32, 33, 34, 35, 36, 37, 38]
    m[idx] = 1.0
    return (torch.from_numpy(s).unsqueeze(0).unsqueeze(0).to(device, INFER_DTYPE),
            torch.from_numpy(m).unsqueeze(0).unsqueeze(0).to(device, INFER_DTYPE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--pose", type=float, nargs=6, default=[490.1, 0.0, 425.0, 180.0, 0.0, 0.0])
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} pose={args.pose}")

    rdt = RDTRunner.from_pretrained(os.path.join(CKPT, "ema")).to(INFER_DTYPE).to(device).eval()
    siglip = SiglipVisionTower("google/siglip-so400m-patch14-384").to(INFER_DTYPE).to(device).eval()
    lang, lmask = load_embed(device)

    frames = [cv2.imread(newest(c)) for c in ("cam1", "cam2", "claw")]
    triplet = [_prep_image(f, siglip.image_processor) for f in frames]
    img_t = torch.stack(triplet + triplet, dim=0).to(device, INFER_DTYPE)
    with torch.no_grad():
        feat = siglip(img_t)
        imgtok = feat.reshape(1, -1, feat.shape[-1])

    for layout in ("deploy", "official"):
        st, sm = pack(args.pose, layout, device)
        xs0 = []; rows = []
        for _ in range(args.n):
            with torch.no_grad():
                a = rdt.predict_action(lang_tokens=lang, lang_attn_mask=lmask, img_tokens=imgtok,
                                       state_tokens=st, action_mask=sm,
                                       ctrl_freqs=torch.tensor([CTRL_FREQ], device=device))
            a = a[0].float().cpu().numpy()
            rows.append(a)
        a = np.stack(rows)  # (n,64,128)
        print(f"\n===== layout = {layout} =====")
        # read eef the way THIS layout would
        if layout == "deploy":
            xi, yi, zi, gi = 0, 1, 2, 9
        else:
            xi, yi, zi, gi = 30, 31, 32, 10
        print(f" read idx x={xi} y={yi} z={zi} grip={gi}")
        print(f"  step0  x={a[:,0,xi].mean():7.1f}±{a[:,0,xi].std():4.1f}  "
              f"y={a[:,0,yi].mean():7.1f}±{a[:,0,yi].std():4.1f}  "
              f"z={a[:,0,zi].mean():7.1f}±{a[:,0,zi].std():4.1f}  grip={a[:,0,gi].mean():5.2f}")
        print(f"  step0  ALSO idx0..2 = [{a[:,0,0].mean():.1f}, {a[:,0,1].mean():.1f}, {a[:,0,2].mean():.1f}]  "
              f"idx30..32 = [{a[:,0,30].mean():.1f}, {a[:,0,31].mean():.1f}, {a[:,0,32].mean():.1f}]")
        # trajectory of the eef-x for this layout over first 8 steps
        traj = a[:, :8, xi].mean(axis=0)
        print("  eef-x steps0..7:", " ".join(f"{v:.0f}" for v in traj))
        zt = a[:, :8, zi].mean(axis=0)
        print("  eef-z steps0..7:", " ".join(f"{v:.0f}" for v in zt))


if __name__ == "__main__":
    main()
