"""
A/B test: is the live-deploy failure caused by a CHANNEL-ORDER (RGB vs BGR) mismatch?

Training built the dataset from cv2.imread (BGR) -> _prep_image (which does
bgr[:, :, ::-1]) -> RGB into SigLIP.  The LIVE server instead feeds cam.last_rgb
(already RGB) -> _prep_image (::-1) -> BGR into SigLIP -> colors inverted vs train.

Hold everything constant (same ckpt, frames, state, embed) and toggle ONLY how the
frame's channels are ordered before _prep_image:
  CORRECT   : pass cv2.imread BGR  -> _prep_image -> RGB    (what training saw)
  LIVE-SIM  : pre-swap to RGB first -> _prep_image -> BGR   (what the live server does)

If LIVE-SIM collapses x toward ~290 (and y up) while CORRECT tracks the object,
the channel-order double-swap in the live path is the root cause of the deploy failure.
"""
from __future__ import annotations
import csv, glob, os, sys
import numpy as np, cv2, torch

BUNDLE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BUNDLE)
sys.path.insert(0, os.environ.get("VLA_TRAIN_REPO", "../VLA"))

from models.rdt_runner import RDTRunner
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from src.services.vla_service import _prep_image, _euler_to_rot6d, INFER_DTYPE

CKPT  = os.path.join(BUNDLE, "data/datasets/checkpoints/trapezoid_20260622/checkpoint-30000/ema")
EMB   = os.environ.get("VLA_TRAIN_REPO", "../VLA") + "/data/datasets/trapezoid_processed/episode_0/instruction_embedding.pt"
STATE_DIM, CTRL_FREQ, NSAMP, EARLY_T = 128, 6, 6, 4
EPS = ["episode_016", "episode_012", "episode_004"]  # below / near / far


def load_traj(ep):
    rows = []
    with open(os.path.join(ep, "trajectory.csv")) as f:
        for r in csv.DictReader(f):
            if r.get("valid", "1") in ("0", "0.0"):
                continue
            rows.append([float(r[k]) for k in ("x_mm","y_mm","z_mm","rx_deg","ry_deg","rz_deg")])
    return rows


def frame(ep, cam, idx, n_cam, n_traj, live_sim):
    j = idx if n_cam == n_traj else round(idx * (n_cam - 1) / max(1, n_traj - 1))
    fs = sorted(glob.glob(os.path.join(ep, cam, "frame_*.jpg")))
    img = cv2.imread(fs[min(j, len(fs) - 1)])              # BGR (as training read it)
    if live_sim:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)         # simulate cam.last_rgb (already RGB)
    return img


def state_tensor(pose, device):
    x, y, z, rx, ry, rz = pose
    s = np.zeros(STATE_DIM, np.float32); m = np.zeros(STATE_DIM, np.float32)
    s[0:3] = [x, y, z]; s[3:9] = _euler_to_rot6d(rx, ry, rz)
    m[0:10] = 1.0
    return (torch.from_numpy(s)[None, None].to(device, INFER_DTYPE),
            torch.from_numpy(m)[None, None].to(device, INFER_DTYPE))


def predict(rdt, siglip, ep, emb, lmask, device, live_sim):
    traj = load_traj(ep); n = len(traj); t = min(EARLY_T, n - 2)
    ncam = {c: len(glob.glob(os.path.join(ep, c, "frame_*.jpg"))) for c in ("cam1_rgb","cam2_rgb","claw_rgb")}
    st, sm = state_tensor(traj[t], device)
    trip = []
    for ti in (max(0, t - 1), t):
        trip += [frame(ep, "cam1_rgb", ti, ncam["cam1_rgb"], n, live_sim),
                 frame(ep, "cam2_rgb", ti, ncam["cam2_rgb"], n, live_sim),
                 frame(ep, "claw_rgb", ti, ncam["claw_rgb"], n, live_sim)]
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
    ch = np.mean(chunks, 0)
    gt = traj[int(np.argmin([p[2] for p in traj]))]
    zmin = int(np.argmin(ch[:, 2]))
    return gt[0], ch[0, 0], ch[0, 1], ch[zmin, 0]   # GTx, x0, y0, x@zmin


def main():
    device = torch.device("cuda")
    rdt = RDTRunner.from_pretrained(CKPT).to(INFER_DTYPE).to(device).eval()
    siglip = SiglipVisionTower("google/siglip-so400m-patch14-384").to(INFER_DTYPE).to(device).eval()
    d = torch.load(EMB, map_location="cpu", weights_only=True)
    emb = (d if d.dim() == 3 else d.unsqueeze(0)).to(device, INFER_DTYPE)
    lmask = torch.ones(emb.shape[:2]).to(device, INFER_DTYPE)

    print(f"\n{'episode':<14}{'GTx':>6}|  {'CORRECT (train order)':>26} |  {'LIVE-SIM (inverted)':>24}")
    print(f"{'':14}{'':6}|  {'x[0]':>8}{'y[0]':>9}{'x@zmin':>9} |  {'x[0]':>8}{'y[0]':>8}{'x@zmin':>8}")
    print("-" * 78)
    for ep in EPS:
        epp = os.path.join(BUNDLE, "data/recordings/trapezoid", ep)
        gx, cx0, cy0, czm = predict(rdt, siglip, epp, emb, lmask, device, live_sim=False)
        _,  lx0, ly0, lzm = predict(rdt, siglip, epp, emb, lmask, device, live_sim=True)
        print(f"{ep:<14}{gx:>6.0f}|  {cx0:>8.0f}{cy0:>9.0f}{czm:>9.0f} |  {lx0:>8.0f}{ly0:>8.0f}{lzm:>8.0f}")
    print("-" * 78)
    print("If LIVE-SIM x collapses (~290) / y rises while CORRECT tracks GTx,")
    print("the live channel-order double-swap (RGB fed to a BGR-expecting _prep_image) is the bug.")


if __name__ == "__main__":
    main()
