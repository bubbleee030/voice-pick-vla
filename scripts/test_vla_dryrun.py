"""
Dry-run sanity check for the COMBINED VLA checkpoint (trapezoid + chopsticks).
Loads RDT + SigLIP, runs ONE inference step with dummy (zero) image inputs and
each precomputed English language embedding. Prints action shape, value ranges,
NaN count, latency and peak GPU memory. Does NOT connect to or move the arm.

Run (deploy-faithful float32):
    conda run -n voice_pick python scripts/test_vla_dryrun.py
Run in bf16 (half the VRAM — for the 8 GB laptop GPU):
    VLA_DTYPE=bf16 conda run -n voice_pick python scripts/test_vla_dryrun.py
"""
from __future__ import annotations

import os
import sys
import time

BUNDLE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BUNDLE_ROOT)
sys.path.insert(0, os.path.join(BUNDLE_ROOT, "vla"))

import numpy as np
import torch

from models.rdt_runner import RDTRunner
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower

CKPT = os.path.join(
    BUNDLE_ROOT,
    "data/datasets/checkpoints/combined_nchc_20260529_120309",
)
EMBEDS = {
    "trapezoid (拿起梯形)": os.path.join(BUNDLE_ROOT, "data/vla_embed_trapezoid_en.pt"),
    "chopsticks (拿起筷子)": os.path.join(BUNDLE_ROOT, "data/vla_embed_chopsticks_en.pt"),
}

IMG_H = IMG_W = 384
IMG_HISTORY = 2          # cam1, cam2, claw × 2 timesteps = 6 frames
STATE_DIM = 128
CHUNK_SIZE = 64
CTRL_FREQ = 6
DTYPE = torch.bfloat16 if os.environ.get("VLA_DTYPE", "float32") == "bf16" else torch.float32


def load_lang_embed(path: str, device: torch.device):
    """Accept both the new bare-tensor (1,1024,4096) format AND the old
    dict {embedding:(L,4096), attention_mask:(L,)} format. The bare-tensor
    files carry no mask, so an all-ones (1,L) mask is synthesised."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if torch.is_tensor(data):
        emb = data
        if emb.dim() == 2:
            emb = emb.unsqueeze(0)                       # (1, L, 4096)
        mask = torch.ones(emb.shape[:2])                 # (1, L) all-ones
    else:
        emb = data["embedding"].unsqueeze(0)
        mask = data["attention_mask"].unsqueeze(0)
    # mask is fed to SDPA as an additive bias -> its dtype must match the query (DTYPE).
    return emb.to(DTYPE).to(device), mask.to(DTYPE).to(device)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] torch {torch.__version__}  device={device}  dtype={DTYPE}")
    if device.type == "cuda":
        cap = "".join(map(str, torch.cuda.get_device_capability(0)))
        print(f"[env] GPU {torch.cuda.get_device_name(0)}  sm_{cap}  "
              f"total={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    ema_dir = os.path.join(CKPT, "ema")
    load_path = ema_dir if os.path.isdir(ema_dir) else CKPT
    print(f"[load] RDT from: {load_path}")
    rdt = RDTRunner.from_pretrained(load_path).to(DTYPE).to(device).eval()
    print(f"[load] RDT params: {sum(p.numel() for p in rdt.parameters())/1e9:.2f}B")

    print("[load] SigLIP …")
    siglip = SiglipVisionTower("google/siglip-so400m-patch14-384").to(DTYPE).to(device).eval()
    if device.type == "cuda":
        print(f"[mem ] after load: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # 6 zero RGB frames at 384×384 (cam1,cam2,claw × 2 timesteps).
    images = torch.zeros(6, 3, IMG_H, IMG_W, device=device, dtype=DTYPE)
    with torch.no_grad():
        img_feat = siglip(images)                                  # (6, 729, 1152)
        img_tokens = img_feat.reshape(1, -1, img_feat.shape[-1])   # (1, 4374, 1152)
    print(f"[run ] img_tokens: {tuple(img_tokens.shape)}")

    # Home-pose state: [x_mm, y_mm, z_mm, r6d×6, gripper, zeros×118].
    state = np.zeros(STATE_DIM, dtype=np.float32)
    state[0:3] = [444.0, 0.0, 744.0]
    state[3:9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]   # identity rotation in 6D form
    state[9] = 0.0
    smask = np.zeros(STATE_DIM, dtype=np.float32)
    smask[0:10] = 1.0
    state_t = torch.from_numpy(state).unsqueeze(0).unsqueeze(0).to(DTYPE).to(device)
    mask_t = torch.from_numpy(smask).unsqueeze(0).unsqueeze(0).to(DTYPE).to(device)

    overall_ok = True
    for label, path in EMBEDS.items():
        lang_emb, lang_mask = load_lang_embed(path, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            actions = rdt.predict_action(
                lang_tokens=lang_emb,
                lang_attn_mask=lang_mask,
                img_tokens=img_tokens,
                state_tokens=state_t,
                action_mask=mask_t,
                ctrl_freqs=torch.tensor([CTRL_FREQ], device=device),
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.time() - t0) * 1000.0

        a = actions[0].float().cpu().numpy()              # (chunk, 128)
        nans = int(np.isnan(a).sum())
        finite = bool(np.isfinite(a).all())
        max_pos = float(np.abs(a[:, :3]).max())
        g_min, g_max = float(a[:, 9].min()), float(a[:, 9].max())
        peak = torch.cuda.max_memory_allocated()/1e9 if device.type == "cuda" else 0.0
        shape_ok = tuple(actions.shape)[1:] == (CHUNK_SIZE, STATE_DIM)
        ok = (nans == 0) and finite and shape_ok and (max_pos < 10000)
        overall_ok &= ok

        print(f"\n[{label}]")
        print(f"    shape={tuple(actions.shape)}  NaNs={nans}  finite={finite}  "
              f"max|pos|={max_pos:.1f}mm  grip=[{g_min:.2f},{g_max:.2f}]")
        print(f"    latency={dt_ms:.0f}ms  peakGPU={peak:.2f}GB  -> {'PASS' if ok else 'FAIL'}")
        for i in range(min(3, a.shape[0])):
            print(f"    t={i}: x={a[i,0]:8.1f} y={a[i,1]:8.1f} z={a[i,2]:8.1f} grip={a[i,9]:.3f}")

    print("\n==== SMOKE TEST " + ("PASS" if overall_ok else "FAIL") + " ====")
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
