"""
A/B test: is the deploy failure caused by feeding the WRONG language embedding?

Holds EVERYTHING constant (same checkpoint, same frames, same state, same model
code, same image preprocessing) and toggles ONLY the language embedding:
  TRAIN  = the embedding the checkpoint was trained on (trapezoid_processed/episode_0)
  DEPLOY = data/vla_embed_trapezoid_en.pt  (what the live server / replay feeds)

If PRED x flips from ~correct (train) to the deploy-failure range (~290-385) under
DEPLOY, the embedding mismatch is the root cause — not normalization, not the model.
"""
from __future__ import annotations
import csv, glob, os, sys
import numpy as np, cv2, torch

BUNDLE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BUNDLE)
sys.path.insert(0, os.environ.get("VLA_TRAIN_REPO", "../VLA"))  # inference code is byte-identical to bundle's vla/

from models.rdt_runner import RDTRunner
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from src.services.vla_service import _prep_image, _euler_to_rot6d, INFER_DTYPE

CKPT  = os.path.join(BUNDLE, "data/datasets/checkpoints/trapezoid_20260622/checkpoint-30000/ema")
EMB_TRAIN  = os.environ.get("VLA_TRAIN_REPO", "../VLA") + "/data/datasets/trapezoid_processed/episode_0/instruction_embedding.pt"
EMB_DEPLOY = os.path.join(BUNDLE, "data/vla_embed_trapezoid_en.pt")
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


def frame(ep, cam, idx, n_cam, n_traj):
    j = idx if n_cam == n_traj else round(idx * (n_cam - 1) / max(1, n_traj - 1))
    fs = sorted(glob.glob(os.path.join(ep, cam, "frame_*.jpg")))
    return cv2.imread(fs[min(j, len(fs) - 1)])


def state_tensor(pose, device):
    x, y, z, rx, ry, rz = pose
    s = np.zeros(STATE_DIM, np.float32); m = np.zeros(STATE_DIM, np.float32)
    s[0:3] = [x, y, z]; s[3:9] = _euler_to_rot6d(rx, ry, rz)
    m[0:10] = 1.0
    return (torch.from_numpy(s)[None, None].to(device, INFER_DTYPE),
            torch.from_numpy(m)[None, None].to(device, INFER_DTYPE))


def load_emb(path, device):
    d = torch.load(path, map_location="cpu", weights_only=True)
    emb = (d if d.dim() == 3 else d.unsqueeze(0)).to(device, INFER_DTYPE)
    return emb, torch.ones(emb.shape[:2]).to(device, INFER_DTYPE)


def predict_x(rdt, siglip, ep, emb, lmask, device):
    traj = load_traj(ep); n = len(traj); t = min(EARLY_T, n - 2)
    ncam = {c: len(glob.glob(os.path.join(ep, c, "frame_*.jpg"))) for c in ("cam1_rgb","cam2_rgb","claw_rgb")}
    st, sm = state_tensor(traj[t], device)
    trip = []
    for ti in (max(0, t - 1), t):
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
    ch = np.mean(chunks, 0)
    gt_grasp_x = traj[int(np.argmin([p[2] for p in traj]))][0]
    return gt_grasp_x, ch[0, 0], ch[int(np.argmin(ch[:, 2])), 0]


def main():
    device = torch.device("cuda")
    rdt = RDTRunner.from_pretrained(CKPT).to(INFER_DTYPE).to(device).eval()
    siglip = SiglipVisionTower("google/siglip-so400m-patch14-384").to(INFER_DTYPE).to(device).eval()
    emb_tr, m_tr = load_emb(EMB_TRAIN, device)
    emb_dp, m_dp = load_emb(EMB_DEPLOY, device)

    print(f"\n{'episode':<14}{'GT grasp-x':>11}{'TRAIN x[0]':>12}{'TRAIN x@zmin':>14}"
          f"{'DEPLOY x[0]':>13}{'DEPLOY x@zmin':>15}")
    print("-" * 79)
    for ep in EPS:
        epp = os.path.join(BUNDLE, "data/recordings/trapezoid", ep)
        gt, tx0, tzm = predict_x(rdt, siglip, epp, emb_tr, m_tr, device)
        _,  dx0, dzm = predict_x(rdt, siglip, epp, emb_dp, m_dp, device)
        print(f"{ep:<14}{gt:>11.0f}{tx0:>12.0f}{tzm:>14.0f}{dx0:>13.0f}{dzm:>15.0f}")
    print("-" * 79)
    print("If DEPLOY columns collapse (x far from GT, toward ~290-385) while TRAIN tracks GT,")
    print("the live server is feeding the WRONG language embedding -> that is the root cause.")


if __name__ == "__main__":
    main()
