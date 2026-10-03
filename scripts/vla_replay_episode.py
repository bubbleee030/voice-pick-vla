"""
Faithful trajectory replay: feed the model a RECORDED episode's real frames +
real state at chosen timesteps and compare its predicted action chunk to the
ground-truth next poses. This is the on-box equivalent of the 5090's
debug_predict_trajectory.py — it tells us whether the model tracks x when given
genuine training-distribution inputs (vs the static ready-pose probes that
showed x collapsing to ~290).

Frames are fed training-faithfully: cv2.imread (BGR) -> _prep_image (which swaps
to RGB), matching how the dataset was built. Use --channels rgb to instead feed
cv2.cvtColor(...BGR2RGB) first, i.e. simulate the live cam.last_rgb path, to
probe channel-order sensitivity.

Run with the server STOPPED:
    conda run -n voice_pick python scripts/vla_replay_episode.py \
        --episode data/recordings/trapezoid/episode_002 --steps 0 20 50 100 150
"""
from __future__ import annotations
import argparse, csv, glob, os, sys
import numpy as np
import cv2
import torch

BUNDLE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BUNDLE_ROOT)
sys.path.insert(0, os.path.join(BUNDLE_ROOT, "vla"))

from models.rdt_runner import RDTRunner
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from src.services.vla_service import _prep_image, _euler_to_rot6d, INFER_DTYPE

CKPT = os.path.join(BUNDLE_ROOT, "data/datasets/trapezoid_0701/checkpoint-24000")
STATE_DIM, CTRL_FREQ = 128, 6
IMG_HISTORY = 2


def load_traj(ep):
    rows = []
    with open(os.path.join(ep, "trajectory.csv")) as f:
        for r in csv.DictReader(f):
            rows.append((float(r["x_mm"]), float(r["y_mm"]), float(r["z_mm"]),
                         float(r["rx_deg"]), float(r["ry_deg"]), float(r["rz_deg"])))
    return rows


def frame(ep, cam, idx, n_cam, n_traj, channels):
    # map traj index -> this cam's frame index (claw runs faster)
    j = idx if n_cam == n_traj else round(idx * (n_cam - 1) / max(1, n_traj - 1))
    fs = sorted(glob.glob(os.path.join(ep, cam, "frame_*.jpg")))
    img = cv2.imread(fs[min(j, len(fs) - 1)])           # BGR
    if channels == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # simulate live cam.last_rgb
    return img


def state_tensor(pose, device):
    x, y, z, rx, ry, rz = pose
    s = np.zeros(STATE_DIM, np.float32); m = np.zeros(STATE_DIM, np.float32)
    s[0:3] = [x, y, z]; s[3:9] = _euler_to_rot6d(rx, ry, rz); s[9] = 0.0
    m[0:10] = 1.0
    return (torch.from_numpy(s).unsqueeze(0).unsqueeze(0).to(device, INFER_DTYPE),
            torch.from_numpy(m).unsqueeze(0).unsqueeze(0).to(device, INFER_DTYPE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="data/recordings/trapezoid/episode_002")
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--embed", default="data/vla_embed_trapezoid_en.pt")
    ap.add_argument("--steps", type=int, nargs="+", default=[0, 20, 50, 100, 150])
    ap.add_argument("--channels", choices=["bgr", "rgb"], default="bgr")
    ap.add_argument("--cam-order", default="cam1_rgb,cam2_rgb,claw_rgb",
                    help="comma-sep folder order fed as the 3 cameras (test cam1<->cam2 swap)")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16",
                    help="inference dtype; deploy uses bf16, the 5090 likely fp32")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    global INFER_DTYPE
    INFER_DTYPE = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    print(f"[dtype] {INFER_DTYPE}")
    ep = os.path.join(BUNDLE_ROOT, args.episode) if not os.path.isabs(args.episode) else args.episode
    ckpt = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(BUNDLE_ROOT, args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_path = os.path.join(ckpt, "ema") if os.path.isdir(os.path.join(ckpt, "ema")) else ckpt
    print(f"[ckpt] {load_path}")
    rdt = RDTRunner.from_pretrained(load_path).to(INFER_DTYPE).to(device).eval()
    siglip = SiglipVisionTower("google/siglip-so400m-patch14-384").to(INFER_DTYPE).to(device).eval()
    d = torch.load(os.path.join(BUNDLE_ROOT, args.embed), map_location="cpu", weights_only=True)
    emb = d if d.dim() == 3 else d.unsqueeze(0)
    lang = emb.to(device, INFER_DTYPE); lmask = torch.ones(emb.shape[:2]).to(device, INFER_DTYPE)

    traj = load_traj(ep)
    n_traj = len(traj)
    cams = [c.strip() for c in args.cam_order.split(",")]   # order fed as the 3 cameras
    counts = [len(glob.glob(os.path.join(ep, c, "frame_*.jpg"))) for c in cams]
    print(f"[ep] {os.path.basename(ep)}  traj={n_traj}  cams={cams} counts={counts}  channels={args.channels}")
    print(f"{'t':>4} | {'STATE x,y,z':>22} | {'GT next x,y,z':>22} | {'PRED step0 x,y,z':>24} | {'PRED step7':>20}")

    for t in args.steps:
        if t >= n_traj - 1:
            continue
        st, sm = state_tensor(traj[t], device)
        # build 6-frame history [t-1, t]
        hist = [max(0, t - 1), t]
        triplets = []
        for ti in hist:
            for c, nc in zip(cams, counts):
                triplets.append(frame(ep, c, ti, nc, n_traj, args.channels))
        batch = torch.stack([_prep_image(f, siglip.image_processor) for f in triplets], 0).to(device, INFER_DTYPE)
        with torch.no_grad():
            feat = siglip(batch); imgtok = feat.reshape(1, -1, feat.shape[-1])
        s0 = []; s7 = []
        for _ in range(args.n):
            with torch.no_grad():
                a = rdt.predict_action(lang_tokens=lang, lang_attn_mask=lmask, img_tokens=imgtok,
                                       state_tokens=st, action_mask=sm,
                                       ctrl_freqs=torch.tensor([CTRL_FREQ], device=device))
            a = a[0].float().cpu().numpy()
            s0.append(a[0, :3]); s7.append(a[7, :3])
        s0 = np.mean(s0, 0); s7 = np.mean(s7, 0)
        gt = traj[min(t + 1, n_traj - 1)]
        print(f"{t:>4} | {traj[t][0]:6.0f},{traj[t][1]:6.0f},{traj[t][2]:6.0f}      | "
              f"{gt[0]:6.0f},{gt[1]:6.0f},{gt[2]:6.0f}      | "
              f"{s0[0]:7.0f},{s0[1]:6.0f},{s0[2]:6.0f}        | {s7[0]:6.0f},{s7[1]:6.0f},{s7[2]:6.0f}")


if __name__ == "__main__":
    main()
