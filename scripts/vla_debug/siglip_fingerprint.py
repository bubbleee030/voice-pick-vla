"""
Deterministic vision-tower fingerprint — run on BOTH boxes and diff the numbers.

Frames + weights + dtype + code are now confirmed identical across the 5090 and the
laptop, yet predicted x differs. The only unverified stage is VISION: the SigLIP
image_processor (preprocessing) and the SigLIP tower (features). Both are deterministic
(no diffusion), so their outputs MUST match bit-for-bit across boxes if the encoder is
the same. This prints a stable fingerprint of each stage so we can pinpoint the divergence.

Stage A = post _prep_image tensor  -> isolates image_processor / transformers version
Stage B = post SigLIP tokens       -> isolates the vision tower weights/version

Run:
  LD_LIBRARY_PATH=$RDT_ENV/lib  rdt_python  scripts/vla_debug/siglip_fingerprint.py
"""
from __future__ import annotations
import glob, os, sys
import numpy as np, cv2, torch

BUNDLE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BUNDLE)
sys.path.insert(0, os.path.join(BUNDLE, "vla"))   # use the BUNDLE's tower (what deploy runs)

import transformers
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from src.services.vla_service import _prep_image, INFER_DTYPE

EP = os.path.join(BUNDLE, "data/recordings/trapezoid/episode_016")


def fp(t, name):
    f = t.float()
    print(f"  {name:22s} shape={tuple(t.shape)} dtype={t.dtype} "
          f"mean={f.mean():+.5f} std={f.std():.5f} norm={f.norm():.3f} "
          f"min={f.min():+.4f} max={f.max():+.4f}")
    flat = f.flatten()
    print(f"  {'  first8':22s} {np.array2string(flat[:8].cpu().numpy(), precision=4, floatmode='fixed')}")


def main():
    device = torch.device("cuda")
    print(f"torch={torch.__version__}  transformers={transformers.__version__}  "
          f"cuda={torch.version.cuda}  gpu={torch.cuda.get_device_name(0)}")

    # SigLIP snapshot identity (which exact HF revision is loaded)
    from huggingface_hub import try_to_load_from_cache  # noqa
    hub = os.path.expanduser("~/.cache/huggingface/hub/models--google--siglip-so400m-patch14-384")
    snaps = glob.glob(os.path.join(hub, "snapshots", "*"))
    print(f"siglip snapshot dir(s): {[os.path.basename(s) for s in snaps]}")

    try:
        siglip = SiglipVisionTower("google/siglip-so400m-patch14-384", None).to(INFER_DTYPE).to(device).eval()
    except TypeError:
        siglip = SiglipVisionTower("google/siglip-so400m-patch14-384").to(INFER_DTYPE).to(device).eval()

    # one fixed, deterministic input per camera (frame_000000)
    for cam in ("cam1_rgb", "cam2_rgb", "claw_rgb"):
        f0 = sorted(glob.glob(os.path.join(EP, cam, "frame_*.jpg")))[0]
        bgr = cv2.imread(f0)
        prepped = _prep_image(bgr, siglip.image_processor)        # Stage A
        x = prepped[None].to(device, INFER_DTYPE)
        with torch.no_grad():
            tok = siglip(x)                                       # Stage B (1,729,1152)
        print(f"\n[{cam}]  src md5-shape={bgr.shape}")
        fp(prepped, "A: post _prep_image")
        fp(tok[0], "B: post SigLIP tokens")


if __name__ == "__main__":
    main()
