"""
Controlled vision-dependence test for the combined VLA checkpoint.

Loads the model ONCE and, holding state + language fixed, predicts the action
chunk under two image conditions:
  (A) REAL  — the latest saved camera snapshots (cam1, cam2, claw)
  (B) BLANK — all-zero frames (what _grab_frames feeds when cameras are dead)

Draws N stochastic diffusion samples per condition (RDT sampling is noisy) and
reports mean +/- std of the step-0 target [x,y,z]. If REAL vs BLANK differ by
much more than the within-condition spread, the policy is genuinely using vision.

Run (server should be STOPPED so this gets the full 8 GB):
    conda run -n voice_pick python scripts/vla_vision_dependence.py --n 6
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
# reuse the EXACT deploy preprocessing + dtype + rot helper
from src.services.vla_service import _prep_image, _euler_to_rot6d, INFER_DTYPE

CKPT = os.path.join(BUNDLE_ROOT, "data/datasets/checkpoints/combined_nchc_20260529_120309")
EMBED = os.path.join(BUNDLE_ROOT, "data/vla_embed_trapezoid_en.pt")
SNAPDIR = os.path.join(BUNDLE_ROOT, "data/runtime/snapshots")
STATE_DIM, CHUNK, CTRL_FREQ = 128, 64, 6


def newest(cam: str) -> str:
    fs = sorted(glob.glob(os.path.join(SNAPDIR, f"{cam}_rgb_*.jpg")))
    if not fs:
        raise FileNotFoundError(f"no snapshot for {cam} in {SNAPDIR}")
    return fs[-1]


def load_embed(device):
    data = torch.load(EMBED, map_location="cpu", weights_only=True)
    emb = data if data.dim() == 3 else data.unsqueeze(0)
    mask = torch.ones(emb.shape[:2])
    return emb.to(device, INFER_DTYPE), mask.to(device, INFER_DTYPE)


def make_state(pose, device):
    x, y, z, rx, ry, rz = pose
    s = np.zeros(STATE_DIM, dtype=np.float32)
    s[0:3] = [x, y, z]
    s[3:9] = _euler_to_rot6d(rx, ry, rz)
    s[9] = 0.0
    m = np.zeros(STATE_DIM, dtype=np.float32)
    m[0:10] = 1.0
    return (torch.from_numpy(s).unsqueeze(0).unsqueeze(0).to(device, INFER_DTYPE),
            torch.from_numpy(m).unsqueeze(0).unsqueeze(0).to(device, INFER_DTYPE))


def img_tokens(frames_bgr, siglip, device):
    # frames_bgr: list of 3 BGR arrays [cam1,cam2,claw]; history=2 -> duplicate -> 6
    triplet = [_prep_image(f, siglip.image_processor) for f in frames_bgr]
    batch = triplet + triplet
    img_t = torch.stack(batch, dim=0).to(device, INFER_DTYPE)
    with torch.no_grad():
        feat = siglip(img_t)
        return feat.reshape(1, -1, feat.shape[-1])


def sample_xyz(rdt, lang, lmask, imgtok, state_t, smask, device, n):
    outs = []
    for _ in range(n):
        with torch.no_grad():
            a = rdt.predict_action(lang_tokens=lang, lang_attn_mask=lmask,
                                   img_tokens=imgtok, state_tokens=state_t,
                                   action_mask=smask,
                                   ctrl_freqs=torch.tensor([CTRL_FREQ], device=device))
        outs.append(a[0].float().cpu().numpy())   # (CHUNK,128)
    return np.stack(outs)   # (n,CHUNK,128)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--pose", type=float, nargs=6,
                    default=[473.8, -2.8, 308.9, 180.0, 6.46, 0.0],
                    help="x y z rx ry rz fed as state (same for both conditions)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} dtype={INFER_DTYPE}")
    rdt = RDTRunner.from_pretrained(os.path.join(CKPT, "ema")).to(INFER_DTYPE).to(device).eval()
    siglip = SiglipVisionTower("google/siglip-so400m-patch14-384").to(INFER_DTYPE).to(device).eval()
    lang, lmask = load_embed(device)
    state_t, smask = make_state(args.pose, device)
    print(f"[state] pose={args.pose}")

    real_bgr = [cv2.imread(newest(c)) for c in ("cam1", "cam2", "claw")]
    for c, im in zip(("cam1", "cam2", "claw"), real_bgr):
        print(f"[real] {c}: shape={None if im is None else im.shape} "
              f"mean={None if im is None else round(float(im.mean()),1)}")
    blank_bgr = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]

    real_tok = img_tokens(real_bgr, siglip, device)
    blank_tok = img_tokens(blank_bgr, siglip, device)

    real = sample_xyz(rdt, lang, lmask, real_tok, state_t, smask, device, args.n)
    blank = sample_xyz(rdt, lang, lmask, blank_tok, state_t, smask, device, args.n)

    def stats(arr, k):  # step-0 axis k
        v = arr[:, 0, k]
        return v.mean(), v.std()

    print(f"\n=== step-0 target  (N={args.n} samples each)  [state x={args.pose[0]:.0f} y={args.pose[1]:.0f} z={args.pose[2]:.0f}] ===")
    print(f"{'axis':>5} | {'REAL mean±std':>18} | {'BLANK mean±std':>18} | {'|Δmean|':>8}")
    for k, name in ((0, "x"), (1, "y"), (2, "z"), (9, "grip")):
        rm, rs = stats(real, k); bm, bs = stats(blank, k)
        print(f"{name:>5} | {rm:8.1f} ± {rs:5.1f}     | {bm:8.1f} ± {bs:5.1f}     | {abs(rm-bm):8.1f}")

    # gross trajectory direction over first 8 steps
    print("\n=== mean over steps 0..7 ===")
    for k, name in ((0, "x"), (1, "y"), (2, "z")):
        print(f"  {name}: REAL={real[:, :8, k].mean():7.1f}  BLANK={blank[:, :8, k].mean():7.1f}")


if __name__ == "__main__":
    main()
