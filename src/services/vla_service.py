"""VLA inference service — drives the physical arm with a trained RDT checkpoint."""
from __future__ import annotations

import sys
import threading
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from src.services.arm_service import ArmService
    from src.services.realsense_service import RealSenseService
    from src.services.claw_camera_service import ClawCameraService

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_VLA_CANDIDATES = [
    PROJECT_ROOT.parent / "VLA",   # sibling repo on 5090
    PROJECT_ROOT / "vla",          # bundled copy for laptop
]
VLA_CODE_DIR: Path | None = next((p for p in _VLA_CANDIDATES if (p / "models").exists()), None)

IMG_H = IMG_W = 384
IMG_HISTORY = 2
CHUNK_SIZE   = 64
# AGX finger open/home pose (= lstm_gripper_service HOME_POS = run_LSTM5 origin).
# Ramp here via /set_position (stepped) — NOT the 'o' jog: 'o' runs under
# state_lock+com_lock (deadlock vs the reader) and is unpaced, so it does not
# reliably reach home. set_position uses the deadlock-free write_goals path.
FINGER_HOME_POS = [3072, 3072, 2048]
STATE_DIM    = 128
CTRL_FREQ    = 6   # Hz — must match training dataset control_freq
# bf16 inference: ~3.5 GB peak vs ~6.9 GB in float32 (safer on the 8 GB 4060) and ~3x faster.
INFER_DTYPE  = torch.bfloat16


def _add_vla_to_path():
    if VLA_CODE_DIR is None:
        raise RuntimeError(
            "VLA model code not found. Expected either "
            f"{_VLA_CANDIDATES[0]} or {_VLA_CANDIDATES[1]}."
        )
    p = str(VLA_CODE_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def _prep_image(bgr: np.ndarray, processor) -> torch.Tensor:
    """Preprocess one camera frame EXACTLY as training did, so SigLIP sees the
    distribution it was trained on.

    Training chain (data/directory_vla_dataset.pad_and_resize_for_siglip +
    train/dataset.py SigLIP image_processor): centre-pad to square, resize to
    384 (LANCZOS), then processor-normalize to ~[-1, 1]. SiglipVisionTower does
    NOT normalize internally — it feeds pixel_values straight to the vision
    model — so handing it raw [0, 1] / stretched frames silently blinds the
    policy. A blind policy ignores the object and falls back to a memorised mean
    trajectory: that is the "arm folds back, Y -> ~-200mm" failure observed on
    the robot. Confirmed by VLA/eval/debug_predict_trajectory.py: real frames ->
    Y~+87, zeroed frames -> Y~-225 (mean |delta| 133mm). Verified byte-identical
    to training by VLA/eval/_verify_prep_matches_training.py (max diff 0.0)."""
    if bgr is None:
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb = bgr[:, :, ::-1].copy()
    h, w = rgb.shape[:2]
    m = max(h, w)
    square = np.zeros((m, m, 3), dtype=rgb.dtype)
    square[(m - h) // 2:(m - h) // 2 + h, (m - w) // 2:(m - w) // 2 + w, :] = rgb
    pil = Image.fromarray(square).resize((IMG_W, IMG_H), Image.LANCZOS)
    # processor returns float32 pixel_values; the loop casts the stacked batch
    # to INFER_DTYPE before SigLIP, so dtype is handled there.
    return processor.preprocess(pil, return_tensors="pt")["pixel_values"][0]


def _euler_to_rot6d(rx: float, ry: float, rz: float) -> np.ndarray:
    R = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
    return R[:, :2].T.flatten()


def _rot6d_to_euler(r6d: np.ndarray) -> np.ndarray:
    a1, a2 = r6d[0:3], r6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=1)
    return Rotation.from_matrix(R).as_euler("xyz", degrees=True)


def _claw_loss_recovery_decision(
    *,
    lost_holds: int,
    post_restart_holds: int,
    restart_succeeded: bool,
    has_verified_target: bool,
    restart_after_holds: int,
    post_restart_grace_holds: int,
) -> tuple[bool, bool, bool]:
    """Return restart, saved-target fallback, and promised-recovery decisions."""
    restart_due = (
        not restart_succeeded
        and lost_holds == restart_after_holds
    )
    fallback_ready = (
        has_verified_target
        and restart_succeeded
        and post_restart_holds >= post_restart_grace_holds
    )
    recovery_pending = (
        has_verified_target
        and (
            lost_holds < restart_after_holds
            or (
                restart_succeeded
                and post_restart_holds < post_restart_grace_holds
            )
        )
    )
    return restart_due, fallback_ready, recovery_pending


class VLAService:
    def __init__(
        self,
        socketio=None,
        arm_service: "ArmService | None" = None,
        realsense_service: "RealSenseService | None" = None,
        claw_service: "ClawCameraService | None" = None,
    ):
        self.socketio = socketio
        self.arm  = arm_service
        self.rs   = realsense_service
        self.claw = claw_service

        self._running = False
        self._thread: threading.Thread | None = None
        self._rdt    = None
        self._siglip = None
        self._device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded_ckpt: str | None = None
        self._model_lock = threading.Lock()   # serialize model loads (startup warm vs a run)

        self.status = "idle"
        self.step   = 0
        self.checkpoint_path = ""

        # --- auto-run mode state (§3 three-object demo) ---
        self.auto_stage = ""              # "" unless an auto run is active
        self.auto_object = ""
        self.awaiting_gate: str | None = None
        self._gate_ack: str | None = None
        self.grasp_accept_available = False
        self._grasp_accept_event = threading.Event()
        self.release_accept_available = False
        self._release_accept_event = threading.Event()
        self.release_home_available = False
        self._release_home_event = threading.Event()
        self._manual_release_requested = False
        self._manual_release_position: list[int] | None = None
        self._arrival: "Any" = None       # live ArrivalServo (UI tunes its knobs)
        self._sim_pose: list[float] | None = None   # dry-run simulated pose

    def _emit_log(self, level: str, msg: str):
        ts = time.strftime("%H:%M:%S")
        if self.socketio:
            self.socketio.emit("vla_log", {"level": level, "message": msg, "timestamp": ts})

    def _emit_status(self):
        if self.socketio:
            self.socketio.emit("vla_status", self.vla_status_payload())

    def vla_status_payload(self) -> dict[str, Any]:
        finger_stage_action = None
        if self.grasp_accept_available:
            finger_stage_action = "grasp"
        elif self.release_home_available:
            finger_stage_action = "release_home"
        elif self.release_accept_available:
            finger_stage_action = (
                "release_pose"
                if self._manual_release_position is not None
                else "release"
            )
        return {
            "status":     self.status,
            "step":       self.step,
            "checkpoint": self.checkpoint_path,
            "running":    self._running,
            "auto_stage": self.auto_stage,
            "auto_object": self.auto_object,
            "awaiting_gate": self.awaiting_gate,
            "grasp_accept_available": self.grasp_accept_available,
            "release_accept_available": self.release_accept_available,
            "release_home_available": self.release_home_available,
            "finger_stage_action": finger_stage_action,
        }

    @staticmethod
    def _resolve(path_str: str) -> str:
        p = Path(path_str)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return str(p)

    def start(
        self,
        checkpoint_path: str,
        embed_path: str,
        instruction: str = "拿起梯形",
        max_steps: int = 200,
        exec_steps: int = 8,
        dry_run: bool = False,
    ):
        if self._running:
            self._emit_log("WARN", "VLA already running — stop it first")
            return
        checkpoint_path = self._resolve(checkpoint_path)
        embed_path      = self._resolve(embed_path)
        self.checkpoint_path = checkpoint_path
        self._running = True
        self.step = 0
        self.status = "loading"
        self._emit_status()
        self._thread = threading.Thread(
            target=self._loop,
            args=(checkpoint_path, embed_path, instruction, max_steps, exec_steps, dry_run),
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._running = False
        self._set_grasp_accept_available(False, emit=False)
        self._set_release_accept_available(False, emit=False)
        self._set_release_home_available(False, emit=False)
        self._manual_release_requested = False
        self._manual_release_position = None
        self._emit_log("INFO", "Stop requested")
        self._emit_status()

    def warm(self, checkpoint_path: str) -> bool:
        """Pre-load the RDT + SigLIP checkpoint onto the GPU without running a
        pass, so the first real auto-run doesn't pay the cold load. Safe to call
        from a background thread at server startup; idempotent (no-op once the
        same checkpoint is resident). Returns True if the model is loaded."""
        if not checkpoint_path:
            return False
        ckpt = self._resolve(checkpoint_path)
        if self._loaded_ckpt == ckpt and self._rdt is not None:
            return True
        try:
            t0 = time.time()
            self._emit_log("STEP", f"[warm] pre-loading VLA checkpoint {ckpt}")
            self._ensure_models(ckpt)
            self.checkpoint_path = ckpt
            self._emit_log("INFO", f"[warm] VLA ready in {time.time() - t0:.1f}s "
                                   "— CLI/UI runs will skip the cold load")
            self._emit_status()
            return True
        except Exception as exc:                                    # noqa: BLE001
            self._emit_log("WARN", f"[warm] VLA pre-load failed ({exc}); "
                                   "the first run will load it on demand")
            return False

    def _ensure_models(self, ckpt_path: str):
        if self._loaded_ckpt == ckpt_path and self._rdt is not None:
            return
        with self._model_lock:
            # double-checked: another thread (startup warm / a run) may have
            # loaded it while we waited for the lock.
            if self._loaded_ckpt == ckpt_path and self._rdt is not None:
                return
            self._ensure_models_locked(ckpt_path)

    def _ensure_models_locked(self, ckpt_path: str):
        _add_vla_to_path()
        from models.rdt_runner import RDTRunner                                  # noqa: PLC0415
        from models.multimodal_encoder.siglip_encoder import SiglipVisionTower  # noqa: PLC0415

        ema_dir = Path(ckpt_path) / "ema"
        load_path = str(ema_dir) if ema_dir.exists() else ckpt_path
        self._emit_log("STEP", f"Loading RDT from {load_path} ({INFER_DTYPE})")
        self._rdt = RDTRunner.from_pretrained(load_path).to(INFER_DTYPE)
        self._rdt.to(self._device).eval()

        self._emit_log("STEP", "Loading SigLIP")
        self._siglip = SiglipVisionTower("google/siglip-so400m-patch14-384").to(INFER_DTYPE)
        self._siglip.to(self._device).eval()

        self._loaded_ckpt = ckpt_path

    def _load_lang_embed(self, embed_path: str, instruction: str):
        p = Path(embed_path) if embed_path else None
        if p and p.exists():
            data = torch.load(str(p), map_location="cpu", weights_only=True)
            if torch.is_tensor(data):
                # New format: a bare (1, L, 4096) — or (L, 4096) — tensor, no mask stored.
                emb  = data if data.dim() == 3 else data.unsqueeze(0)
                mask = torch.ones(emb.shape[:2])
            else:
                # Old format: dict {embedding: (L, 4096), attention_mask: (L,)}.
                emb  = data["embedding"].unsqueeze(0)
                mask = data["attention_mask"].unsqueeze(0)
            # The mask is fed to SDPA as an additive bias, so it must share the
            # model/query dtype — cast both to INFER_DTYPE.
            emb  = emb.to(self._device, INFER_DTYPE)
            mask = mask.to(self._device, INFER_DTYPE)
            self._emit_log("STEP", "Loaded precomputed embedding")
            return emb, mask

        _add_vla_to_path()
        from models.multimodal_encoder.t5_encoder import T5Embedder  # noqa: PLC0415
        self._emit_log("STEP", "Running T5 online (slow) …")
        t5   = T5Embedder(self._device, from_pretrained="google/t5-v1_1-xxl", model_max_length=1024)
        toks = t5.tokenizer(
            instruction, return_tensors="pt", padding="max_length", truncation=True, max_length=1024
        )
        ids  = toks["input_ids"].to(self._device)
        mask = toks["attention_mask"].to(self._device)
        with torch.no_grad():
            out = t5.model(input_ids=ids, attention_mask=mask)
        emb = out.last_hidden_state
        del t5
        torch.cuda.empty_cache()
        return emb.to(INFER_DTYPE), mask.to(INFER_DTYPE)

    def _grab_frames(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cam1 = cam2 = claw = blank
        src1 = src2 = srcc = "blank"

        if self.rs is not None:
            for key, dest in (("cam1", "cam1"), ("cam2", "cam2")):
                c = self.rs.cams.get(key)
                if c is not None and c.last_rgb is not None:
                    with c.lock:
                        frame = c.last_rgb.copy()
                    if key == "cam1":
                        cam1 = frame
                        src1 = "live"
                    else:
                        cam2 = frame
                        src2 = "live"

        # The 3rd image slot stays BLANK for the model: the dataset was built
        # from cam1+cam2 only, so training filled this slot with a constant
        # blank the model learned to ignore. Feeding the live claw frame is
        # out-of-distribution and measurably degrades x tracking (54-episode
        # ablation: x-corr +0.76 -> +0.84 with the slot blanked).
        if self.claw is not None and self.claw.get_latest_frame() is not None:
            srcc = "live->blanked"

        if self.step == 0:
            self._emit_log(
                "STEP",
                f"frames cam1[{src1}] mean={cam1.mean():.1f} std={cam1.std():.1f} | "
                f"cam2[{src2}] mean={cam2.mean():.1f} std={cam2.std():.1f} | "
                f"claw[{srcc}] mean={claw.mean():.1f} std={claw.std():.1f}",
            )

        return cam1, cam2, claw

    def _pose_to_state(self, pose_mm_deg: list[float], gripper_open: float):
        x, y, z      = pose_mm_deg[0], pose_mm_deg[1], pose_mm_deg[2]
        rx, ry, rz   = pose_mm_deg[3], pose_mm_deg[4], pose_mm_deg[5]
        r6d = _euler_to_rot6d(rx, ry, rz)

        state = np.zeros(STATE_DIM, dtype=np.float32)
        state[0:3] = [x, y, z]
        state[3:9] = r6d
        state[9]   = gripper_open

        mask = np.zeros(STATE_DIM, dtype=np.float32)
        mask[0:10] = 1.0

        return (
            torch.from_numpy(state).unsqueeze(0).unsqueeze(0).to(self._device, INFER_DTYPE),
            torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self._device, INFER_DTYPE),
        )

    # Training data has rx=180, ry=0, rz=0 as exact constants across all episodes; the
    # diffusion head never learned r6d (zero per-sample variance => no gradient) and emits
    # garbage rotations. Decoding them swings Rx/Ry/Rz on the real arm (measured: rz -8..-74,
    # rx flips ~105 within one chunk). Override with the demo orientation — the deploy ready
    # pose is (179.999, 0, 0), so pinning (180,0,0) is a no-op on a correct start and only
    # ever rejects the model's rotation noise. Let only x,y,z and the gripper come from the model.
    _FIXED_ORIENTATION_DEG = (180.0, 0.0, 0.0)

    def _action_to_pose(self, action: np.ndarray) -> tuple[list[float], float]:
        x, y, z = float(action[0]), float(action[1]), float(action[2])
        gripper = float(np.clip(action[9], 0.0, 1.0))
        rx, ry, rz = self._FIXED_ORIENTATION_DEG
        pose_mm_deg = [x, y, z, rx, ry, rz]
        return pose_mm_deg, gripper

    def _loop(
        self,
        ckpt_path: str,
        embed_path: str,
        instruction: str,
        max_steps: int,
        exec_steps: int,
        dry_run: bool,
    ):
        try:
            self._ensure_models(ckpt_path)
            lang_emb, lang_mask = self._load_lang_embed(embed_path, instruction)

            frame_history: list[tuple] = []
            self.status = "running"
            self._emit_status()
            self._emit_log("INFO", f"VLA running: dry_run={dry_run}, max_steps={max_steps}")

            while self._running and self.step < max_steps:
                frames = self._grab_frames()
                frame_history.append(frames)
                if len(frame_history) > IMG_HISTORY:
                    frame_history = frame_history[-IMG_HISTORY:]
                while len(frame_history) < IMG_HISTORY:
                    frame_history.insert(0, frame_history[0])

                img_batch = [_prep_image(f, self._siglip.image_processor)
                             for triplet in frame_history for f in triplet]
                img_t = torch.stack(img_batch, dim=0).to(self._device, INFER_DTYPE)
                with torch.no_grad():
                    img_feat = self._siglip(img_t)                       # (6, 729, 1152)
                    img_tokens = img_feat.reshape(1, -1, img_feat.shape[-1])  # (1, 4374, 1152)

                pose_mm_deg  = [0.0] * 6
                gripper_open = 0.0
                if self.arm is not None and self.arm.current_pose_mm_deg is not None:
                    pose_mm_deg = list(self.arm.current_pose_mm_deg[:6])
                state_t, mask_t = self._pose_to_state(pose_mm_deg, gripper_open)

                with torch.no_grad():
                    actions = self._rdt.predict_action(
                        lang_tokens=lang_emb,
                        lang_attn_mask=lang_mask,
                        img_tokens=img_tokens,
                        state_tokens=state_t,
                        action_mask=mask_t,
                        ctrl_freqs=torch.tensor([CTRL_FREQ], device=self._device),
                    )

                actions_np = actions[0].float().cpu().numpy()  # .float(): numpy has no bfloat16

                for i in range(min(exec_steps, CHUNK_SIZE)):
                    if not self._running:
                        break
                    pose_next, grip_next = self._action_to_pose(actions_np[i])

                    if not dry_run and self.arm is not None and self.arm.ctrl is not None:
                        raw = [int(v * 1000) for v in pose_next]
                        safe, reason = self.arm.check_safety(raw)
                        if not safe:
                            self._emit_log("ERROR", f"Safety boundary reject: {reason}")
                            self.stop()
                            break
                        self.arm.ctrl.move_to(raw, speed=20)

                    self._emit_log(
                        "DEBUG",
                        f"step {self.step}: pos=[{pose_next[0]:.0f},{pose_next[1]:.0f},{pose_next[2]:.0f}]"
                        f" grip={grip_next:.2f}{'  [DRY]' if dry_run else ''}",
                    )
                    self.step += 1
                    self._emit_status()
                    time.sleep(1.0 / CTRL_FREQ)

            self.status = "stopped"
            self._emit_log("INFO", f"Done — {self.step} steps executed")

        except Exception:
            self.status = "error"
            self._emit_log("ERROR", traceback.format_exc())
        finally:
            self._running = False
            self._emit_status()

    # ------------------------------------------------------------------
    # Auto-run mode (three-object demo §3): continuous predict→move →
    # arrival check → servo refinement → scripted tail with UI gates.
    # The legacy start()/_loop() path above is untouched.
    # ------------------------------------------------------------------

    def gate(self, name: str):
        """UI gate button ('pick_done' / 'released') — AGX handoff alternation."""
        self._gate_ack = name
        self._emit_log("INFO", f"[gate] '{name}' acknowledged from UI")

    def accept_grasp(self) -> bool:
        """Accept only a grasp whose live telemetry already proves real close."""
        if not (
            self._running
            and self.auto_stage == "抓取_settle"
            and self.grasp_accept_available
        ):
            self._emit_log(
                "WARN",
                "[auto] manual grasp accept rejected — real close not verified",
            )
            return False
        self._grasp_accept_event.set()
        self._emit_log(
            "INFO",
            "[auto] manual grasp accept requested — holding current real grip",
        )
        return True

    def complete_finger_stage(self) -> bool:
        """Complete only the currently authorized finger-stage action.

        A configured manual release has two distinct authorizations: first stop
        the LSTM and move to the release pose, then return HOME only after that
        measured pose converges and a second action becomes available.
        """
        if self.grasp_accept_available:
            return self.accept_grasp()
        if (
            self._running
            and self.auto_stage == "release_pose_hold"
            and self.release_home_available
        ):
            self._release_home_event.set()
            self._emit_log(
                "INFO",
                "[auto] manual HOME requested — returning fingers to "
                "3072/3072/2048 after verified release pose",
            )
            return True
        if (
            self._running
            and self.auto_stage == "放開_settle"
            and self.release_accept_available
        ):
            self._manual_release_requested = (
                self._manual_release_position is not None
            )
            self._release_accept_event.set()
            next_step = (
                "moving to configured release pose"
                if self._manual_release_requested
                else "opening and verifying HOME"
            )
            self._emit_log(
                "INFO",
                "[auto] manual release completion requested — "
                f"stopping LSTM, then {next_step}",
            )
            return True
        self._emit_log(
            "WARN",
            "[auto] finger-stage completion rejected — no eligible action",
        )
        return False

    def _set_grasp_accept_available(
        self,
        available: bool,
        *,
        emit: bool = True,
    ) -> None:
        available = bool(available)
        changed = available != self.grasp_accept_available
        self.grasp_accept_available = available
        if not available:
            self._grasp_accept_event.clear()
        if changed and emit:
            self._emit_status()

    def _set_release_accept_available(
        self,
        available: bool,
        *,
        emit: bool = True,
    ) -> None:
        available = bool(available)
        changed = available != self.release_accept_available
        self.release_accept_available = available
        if not available:
            self._release_accept_event.clear()
        if changed and emit:
            self._emit_status()

    def _set_release_home_available(
        self,
        available: bool,
        *,
        emit: bool = True,
    ) -> None:
        available = bool(available)
        changed = available != self.release_home_available
        self.release_home_available = available
        if not available:
            self._release_home_event.clear()
        if changed and emit:
            self._emit_status()

    def tune_arrival(self, key: str, value: Any) -> bool:
        """Runtime knob update (conf floor, tolerances, servo caps) from the UI."""
        if self._arrival is None:
            return False
        ok = self._arrival.tune(key, value)
        if ok:
            self._emit_log("INFO", f"[tune] {key} = {getattr(self._arrival, key)}")
        return ok

    def start_auto(
        self,
        object_key: str,
        checkpoint_path: str,
        embed_path: str,
        instruction: str = "",
        mode: str = "vla",              # "vla" | "locator" (§3.6 no-VLA rung)
        run_cfg: dict[str, Any] | None = None,
        tail_cfg: dict[str, Any] | None = None,
        dry_run: bool = False,
    ):
        if self._running:
            self._emit_log("WARN", "VLA already running — stop it first")
            return
        self._running = True
        self.step = 0
        self.status = "loading"
        self.auto_object = object_key
        self.auto_stage = "loading"
        self.awaiting_gate = None
        self._gate_ack = None
        self._set_grasp_accept_available(False, emit=False)
        self._set_release_accept_available(False, emit=False)
        self._set_release_home_available(False, emit=False)
        self._manual_release_requested = False
        self._manual_release_position = None
        self.checkpoint_path = self._resolve(checkpoint_path) if checkpoint_path else ""
        self._emit_status()
        self._thread = threading.Thread(
            target=self._auto_loop,
            args=(object_key, self.checkpoint_path, embed_path, instruction,
                  mode, dict(run_cfg or {}), dict(tail_cfg or {}), dry_run),
            daemon=True,
        )
        self._thread.start()

    def _stage(self, name: str):
        if name != "抓取_settle":
            self._set_grasp_accept_available(False, emit=False)
        if name != "放開_settle":
            self._set_release_accept_available(False, emit=False)
        if name != "release_pose_hold":
            self._set_release_home_available(False, emit=False)
        self.auto_stage = name
        self.status = f"auto:{name}"
        self._emit_log("STEP", f"[auto] stage -> {name}")
        self._emit_status()

    def _wait_gate(self, name: str) -> bool:
        """Block until the UI presses the named gate button (or STOP)."""
        self.awaiting_gate = name
        self._gate_ack = None
        self._emit_status()
        self._emit_log("INFO", f"[auto] GATE — waiting for '{name}' button")
        while self._running and self._gate_ack != name:
            time.sleep(0.2)
        self.awaiting_gate = None
        self._emit_status()
        return self._running

    def _wait_grasp_contact_dwell(self, dwell_s: float) -> bool:
        """Hold at grasp_z to seat the object while keeping STOP responsive."""
        dwell_s = max(0.0, float(dwell_s))
        self._stage("grasp_contact_dwell")
        self._emit_log(
            "INFO",
            f"[auto] grasp_z 觸底後保持接觸 {dwell_s:g}s — 讓物件貼穩後再抓取/上抬",
        )
        deadline = time.time() + dwell_s
        while self._running:
            remaining = deadline - time.time()
            if remaining <= 0:
                return True
            time.sleep(min(0.1, remaining))
        return False

    def _confirm_pick_without_lstm(
        self,
        auto_lift_after_dwell: bool,
        contact_dwell_s: float,
    ) -> bool:
        """Use configured contact dwell as confirmation, otherwise keep the gate."""
        if auto_lift_after_dwell:
            self._stage("auto_pick")
            self._emit_log(
                "INFO",
                f"[auto] board {float(contact_dwell_s):g} 秒接觸等待完成 — "
                "免 pick_done,直接上抬",
            )
            return self._running
        self._stage("gate_pick")
        return self._wait_gate("pick_done")

    def _start_grasp_lift(self, lift_z: float, speed: int,
                          step_timeout: float, start_timeout: float
                          ) -> tuple[threading.Thread, list[Exception]]:
        """Start the normal post-pick lift without blocking finger settling."""
        self._stage("grasp_lift")
        pose = self._auto_pose()
        self._emit_log(
            "INFO",
            f"[auto] 抓取流程連續上抬 {pose[2]:.0f} → {lift_z:.0f}mm",
        )
        errors: list[Exception] = []

        def _move() -> None:
            try:
                self._auto_move(
                    pose[0], pose[1], lift_z, speed,
                    "grasp_lift", step_timeout, start_timeout,
                )
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=_move, name="vla-grasp-lift", daemon=True)
        thread.start()
        return thread, errors

    def _lift_and_start_grasp_at_z(
        self,
        lift_z: float,
        trigger_z: float,
        speed: int,
        step_timeout: float,
        start_timeout: float,
        start_grasp: Callable[[], bool],
    ) -> tuple[bool, threading.Thread, list[Exception]]:
        lift_thread, lift_errors = self._start_grasp_lift(
            lift_z, speed, step_timeout, start_timeout
        )
        self._stage("grasp_trigger_wait")
        while self._running:
            if lift_errors:
                raise lift_errors[0]
            measured_z = float(self._auto_pose()[2])
            if not self._running:
                return False, lift_thread, lift_errors
            if lift_errors:
                raise lift_errors[0]
            if not lift_thread.is_alive():
                raise RuntimeError(
                    f"grasp lift ended at z={measured_z:.1f} "
                    f"before trigger z={trigger_z:.1f}"
                )
            if measured_z >= trigger_z:
                self._emit_log(
                    "INFO",
                    f"[auto] grasp trigger target z={trigger_z:.1f}mm, "
                    f"actual z={measured_z:.1f}mm",
                )
                if not self._running:
                    return False, lift_thread, lift_errors
                if lift_errors:
                    raise lift_errors[0]
                if not lift_thread.is_alive():
                    raise RuntimeError(
                        f"grasp lift ended at z={measured_z:.1f} "
                        f"before trigger z={trigger_z:.1f}"
                    )
                used_lstm = bool(start_grasp())
                return used_lstm, lift_thread, lift_errors
            time.sleep(0.05)
        return False, lift_thread, lift_errors

    def _wait_grasp_goal_ready(self, grip) -> bool:
        """Wait without a deadline until prepared inference has a close goal."""
        self._stage("grasp_goal_wait")
        self._emit_log(
            "INFO",
            "[auto] 等待 LSTM 產生真實閉合 goal — 無逾時限制,STOP 可取消",
        )
        last_step: int | None = None
        while self._running:
            st = grip.lstm_status() or {}
            status = str(st.get("status") or "")
            if status == "error":
                self._set_grasp_accept_available(False)
                raise RuntimeError(
                    f"AGX prepared grasp failed: {st.get('error') or 'unknown error'}"
                )
            step = int(st.get("step") or 0)
            goal = st.get("goal_pos")
            if bool(st.get("goal_ready")):
                self._emit_log(
                    "INFO",
                    f"[auto] LSTM close goal ready step={step} goal={goal} "
                    f"start={st.get('action_start_pos')}",
                )
                return True
            if step != last_step:
                self._emit_log(
                    "INFO",
                    f"[auto] grasp prepare step={step} goal={goal} "
                    f"drive_state={st.get('drive_state')}",
                )
                last_step = step
            time.sleep(0.25)
        return False

    @staticmethod
    def _read_fixed_position(grip) -> list[int]:
        state = grip.get_state() or {}
        pos = state.get("current_pos")
        if not isinstance(pos, list) or len(pos) != 3 or not any(pos):
            raise RuntimeError("AGX motor position unavailable")
        return [int(value) for value in pos]

    def _wait_fixed_fingers_settled(
        self,
        grip,
        *,
        target: list[int],
        start_pos: list[int],
        require_close: bool,
        what: str,
        tolerance_ticks: int,
        settle_ticks: int,
        hold_s: float,
        min_close_ticks: int,
        min_close_fingers: int,
    ) -> bool:
        """Wait for a fixed motor goal to settle, without a success deadline.

        This verifies measured motor movement and target tracking only. It does
        not infer tactile contact or prove that the object is in the fingers.
        """
        self._stage(f"fixed_{what}_settle")
        anchor: list[int] | None = None
        stable_since: float | None = None
        while self._running:
            pos = self._read_fixed_position(grip)
            close_count = sum(
                (start - current) >= min_close_ticks
                for start, current in zip(start_pos, pos)
            )
            moved = not require_close or close_count >= min_close_fingers
            at_target = max(
                abs(current - goal)
                for current, goal in zip(pos, target)
            ) <= tolerance_ticks
            now = time.time()
            if moved and at_target:
                if anchor is None:
                    anchor = list(pos)
                    stable_since = now
                elif max(
                    abs(current - original)
                    for current, original in zip(pos, anchor)
                ) > settle_ticks:
                    anchor = list(pos)
                    stable_since = now
                elif (
                    stable_since is not None
                    and now - stable_since >= hold_s
                ):
                    self._emit_log(
                        "INFO",
                        f"[fixed] {what}收斂 pos={pos} target={target} "
                        "(motor target only; not tactile proof)",
                    )
                    return True
            else:
                anchor = None
                stable_since = None
            time.sleep(0.1)
        return False

    def _command_fixed_finger_goal(
        self,
        grip,
        target: list[int],
        *,
        require_close: bool,
        what: str,
        rc: dict[str, Any],
    ) -> bool:
        start_pos = self._read_fixed_position(grip)
        self._stage(f"fixed_{what}")
        self._emit_log(
            "INFO",
            f"[fixed] command {what} target={target} start={start_pos}",
        )
        ok = grip.set_position(
            target,
            mode="stepped",
            stop_requested=lambda: not self._running,
        )
        if not ok:
            if not self._running:
                return False
            raise RuntimeError(
                f"AGX fixed {what} command failed: "
                f"{grip.last_error or 'unknown error'}"
            )
        return self._wait_fixed_fingers_settled(
            grip,
            target=target,
            start_pos=start_pos,
            require_close=require_close,
            what=what,
            tolerance_ticks=int(
                rc.get("fixed_target_tolerance_ticks", 30)
            ),
            settle_ticks=int(rc.get("fixed_settle_ticks", 12)),
            hold_s=float(rc.get("fixed_settle_hold_s", 1.2)),
            min_close_ticks=int(rc.get("fixed_min_close_ticks", 60)),
            min_close_fingers=int(
                rc.get("fixed_min_close_fingers", 2)
            ),
        )

    def _complete_manual_release_fallback(
        self,
        grip,
        target: list[int],
        rc: dict[str, Any],
    ) -> bool:
        """Reach the fixed release pose, then wait for a distinct HOME action."""
        self._set_release_home_available(False, emit=False)
        reached = self._command_fixed_finger_goal(
            grip,
            target,
            require_close=False,
            what="manual_release_pose",
            rc=rc,
        )
        if not reached:
            return False

        self._stage("release_pose_hold")
        self._set_release_home_available(True)
        self._emit_log(
            "INFO",
            f"[auto] release pose verified at {target} — "
            "press Enter again to return fingers HOME",
        )
        while self._running:
            if self._release_home_event.wait(0.2):
                break
        self._set_release_home_available(False)
        if not self._running:
            return False
        self._manual_release_requested = False
        return True

    def _wait_fingers_settled(self, grip, timeout_s: float | None, settle_ticks: int,
                              hold_s: float, min_step: int,
                              require_displaced: bool, what: str, nxt: str,
                              min_close_ticks: int = 60,
                              min_close_fingers: int = 2,
                              tracking_ticks: int = 30,
                              min_tracking_fingers: int = 3,
                              min_excursion_ticks: int = 0,
                              min_excursion_fingers: int = 0,
                              allow_manual_accept: bool = False,
                              allow_manual_release: bool = False) -> bool:
        """Poll AGX until the fingers converge, with no manual confirmation.

        A grasp must close both its measured motors and commanded goals by the
        configured action-relative amount on enough fingers; merely sitting near
        nominal home cannot pass. Both motors and goals must stay within an
        anchored settle window and track one another. Release omits only the close
        requirement. A release can additionally require evidence that both its
        measured motors and commanded goals actually left the action start before
        a stable plateau is eligible. A None timeout waits until convergence or
        STOP, while a finite timeout warns and returns False. motor_pos [0,0,0]
        is the dead-link sentinel and can never count as settled.
        """
        self._set_grasp_accept_available(False, emit=False)
        self._set_release_accept_available(False, emit=False)
        self._stage(f"{what}_settle")
        self._set_release_accept_available(allow_manual_release)
        self._emit_log("INFO", f"[auto] LSTM {what} — 等待手指收斂後{nxt}(免確認)")
        t0 = time.time()
        deadline = (
            None if timeout_s is None or timeout_s <= 0
            else t0 + timeout_s
        )
        stable_motor_anchor: list[int] | None = None
        stable_goal_anchor: list[int] | None = None
        fallback_start_pos: list[int] | None = None
        stable_since: float | None = None
        warned_dead = False
        last_trace_step: int | None = None
        excursion_seen = min_excursion_ticks <= 0 or min_excursion_fingers <= 0
        while self._running and (deadline is None or time.time() < deadline):
            if allow_manual_release and self._release_accept_event.is_set():
                self._set_release_accept_available(False)
                self._emit_log(
                    "INFO",
                    f"[auto] ✅ {what} manually completed — "
                    "continuing to verified finger HOME",
                )
                return True
            st = grip.lstm_status() or {}
            if allow_manual_release and self._release_accept_event.is_set():
                self._set_release_accept_available(False)
                self._emit_log(
                    "INFO",
                    f"[auto] ✅ {what} manually completed — "
                    "continuing to verified finger HOME",
                )
                return True
            status = str(st.get("status") or "")
            if status == "error":
                raise RuntimeError(
                    f"AGX {what} action failed: "
                    f"{st.get('error') or 'unknown error'}"
                )
            mp = st.get("motor_pos")
            step = int(st.get("step") or 0)
            tac = st.get("tactile")
            goal = st.get("goal_pos")
            goal_pos = (
                [int(value) for value in goal]
                if isinstance(goal, list) and len(goal) == 3
                else None
            )
            goal_step = int(st.get("goal_step") or 0)
            if not (isinstance(mp, list) and len(mp) == 3 and any(mp)):
                self._set_grasp_accept_available(False)
                if not warned_dead:
                    self._emit_log("WARN", "[auto] ⚠️  夾爪馬達無回應 (motor_pos=0) — AGX "
                                           "馬達埠可能掉線(ttyUSB 重編號),無法自動判斷手指")
                    warned_dead = True
                stable_motor_anchor = None
                stable_goal_anchor = None
                stable_since = None
                time.sleep(0.3)
                continue
            if fallback_start_pos is None:
                fallback_start_pos = list(mp)
            reported_start = st.get("action_start_pos")
            start_pos = (
                [int(v) for v in reported_start]
                if isinstance(reported_start, list) and len(reported_start) == 3
                else fallback_start_pos
            )
            if step != last_trace_step:
                home_delta = [m - h for m, h in zip(mp, FINGER_HOME_POS)]
                goal_log = ""
                if goal_pos is not None:
                    tracking_error = [m - g for m, g in zip(mp, goal_pos)]
                    goal_log = (f" goal(step={goal_step})={goal_pos}"
                                f" tracking_err={tracking_error}")
                self._emit_log(
                    "INFO",
                    f"[auto] {what} trace step={step} pos={mp}{goal_log} "
                    f"home_delta={home_delta} tactile={tac}",
                )
                last_trace_step = step
            actual_close_count = sum(
                (s - m) >= min_close_ticks for s, m in zip(start_pos, mp)
            )
            goal_close_count = (
                sum((s - g) >= min_close_ticks for s, g in zip(start_pos, goal_pos))
                if goal_pos is not None
                else actual_close_count
            )
            moved = (not require_displaced) or (
                actual_close_count >= min_close_fingers
                and goal_close_count >= min_close_fingers
            )
            manual_available = bool(allow_manual_accept and moved)
            self._set_grasp_accept_available(manual_available)
            if manual_available and self._grasp_accept_event.is_set():
                self._set_grasp_accept_available(False)
                self._emit_log(
                    "INFO",
                    f"[auto] ✅ {what} manually accepted after real close "
                    f"pos={mp} tactile={tac}",
                )
                return True
            if goal_pos is not None and not excursion_seen:
                actual_excursion_count = sum(
                    abs(motor - start) >= min_excursion_ticks
                    for start, motor in zip(start_pos, mp)
                )
                goal_excursion_count = sum(
                    abs(target - start) >= min_excursion_ticks
                    for start, target in zip(start_pos, goal_pos)
                )
                excursion_seen = (
                    actual_excursion_count >= min_excursion_fingers
                    and goal_excursion_count >= min_excursion_fingers
                )
                if excursion_seen:
                    self._emit_log(
                        "INFO",
                        f"[auto] {what} real-action excursion observed "
                        f"motor_fingers={actual_excursion_count} "
                        f"goal_fingers={goal_excursion_count}",
                    )
            candidate = (
                step >= min_step
                and moved
                and excursion_seen
                and goal_pos is not None
            )
            tracking_count = (
                sum(
                    abs(motor - target) <= tracking_ticks
                    for motor, target in zip(mp, goal_pos)
                )
                if goal_pos is not None
                else 0
            )
            tracking_ok = candidate and tracking_count >= min_tracking_fingers
            if tracking_ok:
                now = time.time()
                if stable_motor_anchor is None or stable_goal_anchor is None:
                    stable_motor_anchor = list(mp)
                    stable_goal_anchor = list(goal_pos)
                    stable_since = now
                else:
                    motor_drift = max(
                        abs(current - anchor)
                        for current, anchor in zip(mp, stable_motor_anchor)
                    )
                    goal_drift = max(
                        abs(current - anchor)
                        for current, anchor in zip(goal_pos, stable_goal_anchor)
                    )
                    if motor_drift > settle_ticks or goal_drift > settle_ticks:
                        stable_motor_anchor = list(mp)
                        stable_goal_anchor = list(goal_pos)
                        stable_since = now
                    elif (
                        stable_since is not None
                        and now - stable_since >= hold_s
                    ):
                        self._emit_log(
                            "INFO",
                            f"[auto] ✅ {what}收斂 pos={mp} tactile={tac} "
                            f"(step {step}) — 自動{nxt}",
                        )
                        self._set_grasp_accept_available(False)
                        self._set_release_accept_available(False)
                        return True
            else:
                stable_motor_anchor = None
                stable_goal_anchor = None
                stable_since = None
            time.sleep(0.25)
        if self._running and deadline is not None:
            self._emit_log("WARN", f"[auto] ⚠️  {what}未在 {timeout_s:.0f}s 內收斂 — 仍自動{nxt},請留意")
        self._set_grasp_accept_available(False)
        self._set_release_accept_available(False)
        return False

    def _verify_fingers_home(self, grip, tol: int = 60) -> bool:
        """After a return-home set_position, CONFIRM the fingers actually reached
        home. set_position returns True on HTTP 200 (goals accepted) — but an
        overload-latched motor (torque-off) silently ignores its goal, so a plain
        home can leave fingers parked mid-travel (seen: release latches finger
        2/3). Rebooting is the only thing that moves a latched motor, so if any
        finger is still off-home, clear the latch + re-home. See ADR 0002 and the
        overload-latch note in TEST_STEPS_zh-TW.md."""
        if grip is None:
            return False
        time.sleep(0.6)                                   # let the stepped ramp run
        st = grip.lstm_status() or {}
        mp = st.get("motor_pos")
        if not (isinstance(mp, list) and len(mp) == 3 and any(mp)):
            return False
        off = [i + 1 for i, (m, h) in enumerate(zip(mp, FINGER_HOME_POS)) if abs(m - h) > tol]
        if not off:
            return True
        self._emit_log("WARN", f"[auto] ⚠️  手指未回到 home {mp}(手指 {off} 卡住/過載鎖死)"
                               " — reboot 清鎖存後重新張開回 home")
        res = grip.reboot_motors(home=True) or {}
        recovered = res.get("motor_pos")
        self._emit_log(
            "INFO",
            f"[auto] reboot 後 motor_pos={recovered} "
            f"(rebooted {res.get('rebooted')})",
        )
        return bool(
            isinstance(recovered, list)
            and len(recovered) == 3
            and any(recovered)
            and all(
                abs(motor - home) <= tol
                for motor, home in zip(recovered, FINGER_HOME_POS)
            )
        )

    def _home_fingers_or_raise(self, grip) -> None:
        self._emit_log("INFO", "[auto] 手指回 home(張開 3072/3072/2048)")
        if not grip.set_position(FINGER_HOME_POS, mode="stepped"):
            raise RuntimeError(
                "finger home command failed: "
                f"{getattr(grip, 'last_error', '') or 'no response'}"
            )
        if not self._verify_fingers_home(grip):
            raise RuntimeError(
                "finger home verification failed; arm remains at release Z"
            )

    def _predict_target(self, lang_emb, lang_mask, frame_history: list, blind_state: bool):
        """One chunk prediction from the current measured pose.

        Returns (actions_np, pose_mm_deg). Refuses to predict if a side cam is
        blank — with blind-state the model would localize from nothing."""
        frames = self._grab_frames()
        cam1, cam2 = frames[0], frames[1]
        if float(cam1.std()) < 1.0 or float(cam2.std()) < 1.0:
            raise RuntimeError("side camera blank — refusing to predict blind")
        frame_history.append(frames)
        del frame_history[:-IMG_HISTORY]
        while len(frame_history) < IMG_HISTORY:
            frame_history.insert(0, frame_history[0])

        img_batch = [_prep_image(f, self._siglip.image_processor)
                     for triplet in frame_history for f in triplet]
        img_t = torch.stack(img_batch, dim=0).to(self._device, INFER_DTYPE)
        with torch.no_grad():
            img_feat = self._siglip(img_t)
            img_tokens = img_feat.reshape(1, -1, img_feat.shape[-1])

        pose_mm_deg = self._auto_pose()
        state_t, mask_t = self._pose_to_state(pose_mm_deg, 0.0)
        if blind_state:
            state_t = torch.zeros_like(state_t)   # parity with --blind_state training
        with torch.no_grad():
            actions = self._rdt.predict_action(
                lang_tokens=lang_emb, lang_attn_mask=lang_mask,
                img_tokens=img_tokens, state_tokens=state_t, action_mask=mask_t,
                ctrl_freqs=torch.tensor([CTRL_FREQ], device=self._device),
            )
        return actions[0].float().cpu().numpy(), pose_mm_deg

    def _auto_pose(self) -> list[float]:
        """Read and publish a measured pose while the auto poller is paused."""
        if self._sim_pose is not None:
            return list(self._sim_pose)
        publish_pose = getattr(self.arm, "read_and_publish_pose", None)
        if callable(publish_pose):
            pose = publish_pose()
            if pose is None:
                raise RuntimeError("invalid live arm pose from Modbus")
            return list(pose[:6])
        # Compatibility for lightweight test doubles that expose only ctrl.
        return self.arm.ctrl.read_current_pose_mm_deg()[:6]

    def _log_auto_motion_settings(
        self, correction: int, travel: int, descend: int
    ) -> None:
        """Expose which speed applies to each stage and the live ramp profile."""
        ctrl = getattr(self.arm, "ctrl", None)
        motion = getattr(ctrl, "motion", {}) or {}
        move_name = (
            "MovL" if motion.get("use_linear_move", False) else "MovP"
        )
        acc = (
            motion.get("acceleration_raw")
            if motion.get("set_acceleration", False)
            else "unchanged"
        )
        dec = (
            motion.get("deceleration_raw")
            if motion.get("set_deceleration", False)
            else "unchanged"
        )
        self._emit_log(
            "INFO",
            "[auto] motion settings "
            f"correction={correction}% travel={travel}% "
            f"descend={descend}% controller={move_name} "
            f"ACC={acc} DEC={dec}",
        )

    def _auto_move(self, x: float, y: float, z: float, speed: int, label: str,
                   step_timeout: float, start_timeout: float):
        """Safety-checked watchdogged move at the demo's fixed orientation."""
        rx, ry, rz = self._FIXED_ORIENTATION_DEG
        raw = [int(x * 1000), int(y * 1000), int(z * 1000),
               int(rx * 1000) - 1, int(ry * 1000), int(rz * 1000)]  # 179999 like ready_pose
        if self.arm is not None:   # arm-less dry run has nothing to protect
            safe, reason = self.arm.check_safety(raw)
            if not safe:
                from src.services.arm_service import FallbackAbort  # noqa: PLC0415
                raise FallbackAbort(f"safety reject {label}: {reason}")
        self._emit_log("STEP", f"[auto] move {label}: ({x:.0f},{y:.0f},{z:.0f}) @ {speed}%"
                               f"{'  [DRY]' if self._sim_pose is not None else ''}")
        if self._sim_pose is not None:
            self._sim_pose[:3] = [x, y, z]
            time.sleep(0.2)
            return
        profile = self.arm._fb_move(
            label, raw, speed, step_timeout, start_timeout
        )
        if isinstance(profile, dict):
            self._emit_log(
                "INFO",
                "[auto] controller confirmed "
                f"speed={profile.get('applied_speed_percent')}% "
                f"{profile.get('move_type', '')} "
                f"ACC={profile.get('acceleration_raw')} "
                f"DEC={profile.get('deceleration_raw')}",
            )

    def _auto_loop(self, object_key: str, ckpt: str, embed: str, instruction: str,
                   mode: str, rc: dict[str, Any], tail: dict[str, Any], dry_run: bool):
        from src.services.arm_service import FallbackAbort       # noqa: PLC0415

        hover_z   = float(rc.get("hover_z_mm", 330.0))
        stride    = int(rc.get("stride", 6))
        max_step  = float(rc.get("max_step_mm", 60.0))
        speed     = int(rc.get("speed_percent", 12))
        travel    = int(rc.get("travel_speed_percent", 40))
        descend_v = int(rc.get("descend_speed_percent", 15))
        max_moves = int(rc.get("max_moves", 40))
        step_to   = float(rc.get("per_step_timeout_s", 25.0))
        start_to  = float(rc.get("motion_start_timeout_s", 3.0))
        # #4 YOLO sanity gate: distrust a VLA jump bigger than this (mm) once the
        # claw already has a fresh, confident box for the object. 0 disables it.
        sanity_mm = float(rc.get("vla_sanity_mm", 40.0))
        claw_restart_after = max(
            1, int(rc.get("claw_restart_after_lost_holds", 3))
        )
        claw_restart_grace = max(
            0, int(rc.get("claw_post_restart_grace_holds", 2))
        )
        grasp_z   = float(tail.get("grasp_z_mm", 175.0))
        # Per-object contact timing. The live knife flow seats at grasp_z, waits,
        # starts one continuous lift, then triggers LSTM grasp from measured Z.
        # The older static pre-grasp backoff remains available but is disabled.
        grasp_contact_dwell = float(tail.get("grasp_contact_dwell_s", 0.0))
        min_close_ticks = int(rc.get("grasp_min_close_ticks", 60))
        prepare_min_close_ticks = int(
            rc.get("grasp_prepare_min_close_ticks", min(20, min_close_ticks))
        )
        min_close_fingers = int(rc.get("grasp_min_close_fingers", 2))
        auto_lift_after_dwell = bool(tail.get("auto_lift_after_dwell", False))
        lift_during_grasp = bool(tail.get("lift_during_grasp", False))
        grasp_start_z_raw = tail.get("grasp_start_z_mm")
        grasp_start_z = (
            float(grasp_start_z_raw) if grasp_start_z_raw is not None else None
        )
        grasp_backoff = float(tail.get("grasp_backoff_mm", 0.0))
        place_pose = tail.get("place_approach_pose")             # raw controller units
        lift_z = float(place_pose[2]) / 1000.0 if place_pose else hover_z
        place_z_mm = tail.get("place_z_mm")
        # place_only: test the release leg alone (lift -> release point ->
        # place-descend -> release gate -> return); skips approach/pick.
        place_only = bool(rc.get("place_only"))

        finger_policy = str(tail.get("finger_policy", "lstm"))
        fixed_policy = finger_policy == "fixed"
        fixed_run = mode == "fixed" or bool(rc.get("fixed_fallback"))
        fixed_grasp_position = tail.get("fixed_grasp_position")
        fixed_release_position = tail.get("fixed_release_position")
        manual_release_raw = tail.get("manual_release_position")
        manual_release_position = (
            [int(value) for value in manual_release_raw]
            if isinstance(manual_release_raw, list)
            and len(manual_release_raw) == 3
            else None
        )
        self._manual_release_position = manual_release_position
        requires_grasp = bool(tail.get("requires_grasp", True))

        # AGX LSTM actions are exclusive to the intelligent finger policy.
        # Fixed fallback uses only measured /set_position commands.
        lstm = dict(tail.get("lstm") or {})
        grip = getattr(self.arm, "gripper", None) if self.arm is not None else None
        lstm_on = (
            not fixed_policy and bool(lstm) and grip is not None and not dry_run
        )
        lstm_active = False
        lifted_during_grasp = False

        def lstm_go(action: str | None) -> bool:
            """Start an AGX LSTM finger action. Returns True ONLY if the LSTM is
            actually driving the fingers; returns False (with a LOUD notice) when
            it is unconfigured / unreachable / failed, so the caller falls back to
            the hardcoded gripper WITHOUT aborting the run (operator request)."""
            nonlocal lstm_active
            if not (lstm_on and action):
                return False
            res = grip.lstm_start(action)
            if res and res.get("ok"):
                lstm_active = True
                self._emit_log("INFO", f"[auto] AGX finger action '{action}' started")
                return True
            self._emit_log("WARN", f"[auto] ⚠️  AGX LSTM '{action}' 沒有啟動 "
                                   f"({(res or {}).get('error', 'unreachable')}) — 不中止,改用硬編 gripper")
            return False

        def lstm_prepare(action: str | None) -> bool:
            """Run grasp inference with motor writes suppressed until activate."""
            nonlocal lstm_active
            if not (lstm_on and action):
                return False
            res = grip.lstm_prepare(
                action,
                min_close_ticks=prepare_min_close_ticks,
                min_close_fingers=min_close_fingers,
            )
            if res and res.get("ok"):
                lstm_active = True
                self._emit_log(
                    "INFO",
                    f"[auto] AGX finger action '{action}' prepared; "
                    "motor drive stays paused; thresholds "
                    f"activation={prepare_min_close_ticks} ticks/"
                    f"{min_close_fingers} fingers, "
                    f"final={min_close_ticks} ticks/"
                    f"{min_close_fingers} fingers",
                )
                return True
            self._emit_log(
                "WARN",
                f"[auto] ⚠️  AGX LSTM '{action}' prepare failed "
                f"({(res or {}).get('error', 'unreachable')})",
            )
            return False

        def lstm_activate() -> bool:
            """Write the cached close goal synchronously at the measured Z trigger."""
            res = grip.lstm_activate()
            if res and res.get("ok"):
                activated_goal = (
                    res.get("goal") or res.get("pending_goal") or res.get("goal_pos")
                )
                self._emit_log(
                    "INFO",
                    f"[auto] AGX prepared grasp activated goal={activated_goal}",
                )
                return True
            self._emit_log(
                "WARN",
                f"[auto] ⚠️  AGX prepared grasp activation failed "
                f"({(res or {}).get('error', 'unreachable')})",
            )
            return False

        def lstm_halt():
            nonlocal lstm_active
            if not lstm_active:
                return
            res = grip.lstm_stop()
            lstm_active = False
            if res and res.get("ok"):
                self._emit_log("INFO", f"[auto] AGX fingers held ({res.get('steps', '?')} steps)")
            else:
                self._emit_log("WARN", "[auto] AGX stop failed — check the fingers!")

        def stop_physical_motion() -> None:
            if (
                self._sim_pose is None
                and self.arm is not None
                and self.arm.ctrl is not None
            ):
                try:
                    self.arm.ctrl.motion_stop()
                except Exception:
                    self._emit_log(
                        "WARN", "[auto] arm motion-stop request failed"
                    )

        def hold_fixed_fingers() -> None:
            if not fixed_policy or grip is None or dry_run:
                return
            try:
                if not grip._hold_measured_position():
                    self._emit_log(
                        "WARN",
                        "[fixed] could not hold measured finger position: "
                        f"{grip.last_error or 'unknown error'}",
                    )
            except Exception as exc:
                self._emit_log(
                    "WARN", f"[fixed] measured finger hold failed: {exc}"
                )

        poll_was_running = False
        self._sim_pose = None
        try:
            if fixed_policy and grip is None:
                raise RuntimeError("fixed fallback requires the AGX gripper")
            if fixed_run:
                servo = None
            else:
                from src.arrival import ArrivalServo             # noqa: PLC0415
                servo = ArrivalServo()
            self._arrival = servo

            if lstm_on:
                def _preload():
                    res = grip.lstm_preload(str(lstm.get("object", object_key)))
                    if res and res.get("ok"):
                        self._emit_log("INFO", f"[auto] AGX preload ok in {res.get('seconds', '?')}s: "
                                               f"{res.get('actions')}")
                    else:
                        self._emit_log("WARN", f"[auto] AGX preload failed "
                                               f"({(res or {}).get('error', 'unreachable')}) — gates go manual")
                threading.Thread(target=_preload, daemon=True).start()
            if servo is not None and not servo.calibrated:
                self._emit_log("WARN", "[auto] claw_servo.yaml calibrated=false — jacobian is the "
                                       "offline fit; verify signs at the rig (§1.3) before trusting servo")

            arm_ok = self.arm is not None and self.arm.ensure_connected()
            if not arm_ok:
                if not dry_run:
                    raise RuntimeError("arm not connected")
                ready = [490.127, 0.0, 425.027, 179.999, 0.0, 0.0]
                self._sim_pose = ready
                self._emit_log("INFO", "[auto] DRY RUN without arm — simulating pose from ready")
            elif dry_run:
                self._sim_pose = self.arm.ctrl.read_current_pose_mm_deg()[:6]

            lang_emb = lang_mask = None
            frame_history: list[tuple] = []
            blind_state = bool(rc.get("blind_state", True))
            if mode == "vla" and not place_only:     # place-only never predicts
                self._ensure_models(ckpt)
                lang_emb, lang_mask = self._load_lang_embed(embed, instruction)

            if arm_ok and not dry_run:
                poll_was_running = self.arm._polling
                self.arm.stop_pose_polling()
                time.sleep(0.15)
                self.arm.ctrl.reset_alarms()
                self.arm.ctrl.servo_on()

            self._log_auto_motion_settings(speed, travel, descend_v)

            # ---------------- stage 1: coarse approach ----------------
            self._stage("approach")
            moves = 0
            # Vision plane: arrival/servo only count at the z the object's
            # jacobian+setpoint were measured at (rig calib json). The board
            # is invisible to the claw above ~z365 — the loop seeks DOWN to
            # its plane when it needs vision, instead of holding blind.
            plane_z = hover_z
            seek_z = servo.servo_z(object_key) if servo is not None else None
            seek_step = float(rc.get("seek_step_mm", 30.0))
            if seek_z is not None and abs(seek_z - hover_z) > 2.0:
                self._emit_log("INFO", f"[auto] {object_key} vision plane z={seek_z:.0f} "
                                       f"(calibrated) vs hover {hover_z:.0f} — will seek down "
                                       "before gating on the claw")
            # Fixed-xy mode skips approach/servo and descends at configured rig
            # coordinates. Fixed fallback requires this path and never touches
            # cameras, locator, or model inference.
            fixed_xy = rc.get("fixed_xy") if not place_only else None
            if fixed_run and not (
                isinstance(fixed_xy, (list, tuple))
                and len(fixed_xy) >= 2
            ):
                raise RuntimeError("fixed fallback requires configured fixed_xy")
            if fixed_xy and isinstance(fixed_xy, (list, tuple)) and len(fixed_xy) >= 2:
                fx, fy = float(fixed_xy[0]), float(fixed_xy[1])
                prefix = "[fixed]" if fixed_run else "[auto]"
                self._emit_log("INFO", f"{prefix} {object_key} XY mode → moving to the point. " #FIXED-XY 模式 → 直接移到設定點
                                       f"({fx:.1f},{fy:.1f})mm, descenting... ") #不接近/不伺服,直接下降
                self._auto_move(fx, fy, plane_z, travel, "fixed_xy", step_to, start_to)
                moves += 1
            elif mode == "locator" and not place_only:
                from src.locator import SideCamLocator           # noqa: PLC0415
                cam = self.rs.cams.get("cam1") if self.rs else None
                if cam is None or cam.last_rgb is None:
                    raise RuntimeError("locator mode: cam1 frame unavailable")
                with cam.lock:
                    frame = cam.last_rgb.copy()
                res = SideCamLocator().locate(frame, object_key)
                if res is None:
                    raise RuntimeError(f"locator: no confident cam1 box for {object_key}")
                self._emit_log("INFO", f"[auto] locator -> ({res.x_mm:.0f},{res.y_mm:.0f})mm "
                                       f"conf={res.conf:.2f}")
                self._auto_move(res.x_mm, res.y_mm, plane_z, travel, "locator_hover", step_to, start_to)
                moves += 1

            # ------- stage 2: predict/servo alternation until arrival -------
            # fixed-xy arrives immediately (no vision); place-only skips to tail.
            arrived = place_only or bool(fixed_xy)
            if not arrived and servo is None:
                raise RuntimeError("vision servo unavailable for non-fixed approach")
            if place_only:
                self._emit_log("INFO", "[auto] PLACE-ONLY test: skipping approach/pick — "
                                       "running the release leg from the current pose")
            corrections = 0
            stalled_holds = 0
            lost_holds = 0
            # Fine-phase latch: once the servo engages (condition C handed
            # over near the target), the claw is the sole authority and the
            # VLA is PAUSED. Without this the two fight: a model with a
            # ~25 mm residual bias keeps yanking the arm back off the servo's
            # converged spot and condition C blocks arrival forever (seen
            # live 2026-07-14: y oscillated 182<->206 for 35 moves).
            fine = False
            # Board locator fallback: once a fresh claw detection produces a
            # COMPLETE (uncapped) servo correction and that move succeeds, its
            # robot-space target remains valid if the stationary board becomes
            # occluded. Never reuse the old wrist-camera box after arm motion.
            last_target_fallback = object_key == "board" and mode == "locator"
            last_servo_target: tuple[float, float, float] | None = None
            claw_restarted_for_loss = False
            post_restart_holds = 0

            def claw_boxes():
                """Detections + WORST-CASE age. A wedged USB camera keeps
                serving one frozen frame: YOLO re-detects it so the detect
                timestamp stays fresh while the IMAGE is stale (seen live
                2026-07-14: identical err across 60 mm of arm motion). Gate
                on max(frame age, detect age) so frozen frames go stale."""
                if self.claw is None:
                    return [], None
                boxes, d_age = self.claw.get_detections()
                try:
                    f_age = self.claw.worker.frame_age() if self.claw.worker else None
                except Exception:
                    f_age = None
                if f_age is not None:
                    d_age = f_age if d_age is None else max(d_age, f_age)
                return boxes, d_age

            while self._running and moves < max_moves and not arrived:
                boxes, age = claw_boxes()

                if mode == "vla" and not fine:
                    actions, pose = self._predict_target(lang_emb, lang_mask, frame_history, blind_state)
                    tgt = actions[min(stride, CHUNK_SIZE) - 1]
                    dx, dy = float(tgt[0]) - pose[0], float(tgt[1]) - pose[1]
                    pred_step = (dx * dx + dy * dy) ** 0.5
                else:
                    pose = self._auto_pose()
                    pred_step = None            # fine phase / locator: claw only

                d = servo.decide(boxes, object_key, pred_step, age)
                self._emit_log("DEBUG", f"[auto] move {moves}: pred_step="
                                        f"{'—' if pred_step is None else f'{pred_step:.0f}mm'} "
                                        f"err=({d.err_cx:+.3f},{d.err_cy:+.3f}) {d.reason}")

                # #4 YOLO sanity gate: the claw is a wrist cam looking straight
                # down, so once it holds a fresh, confident box the object is
                # already roughly under it. A VLA prediction that still wants a
                # big jump has gone OOD — trust YOLO instead: null the prediction
                # so the claw servo (below) takes over and converges on the box.
                if (mode == "vla" and not fine and pred_step is not None
                        and sanity_mm > 0 and pred_step >= sanity_mm
                        and d.box is not None and d.checks.get("fresh", True)
                        and d.checks.get("conf", False)):
                    self._emit_log("WARN",
                        f"[auto] VLA 想跳 {pred_step:.0f}mm,但爪相機已看到 {object_key} "
                        f"(conf {d.box['conf']:.2f}) — 不信模型,改用 YOLO 伺服")
                    pred_step = None

                # Seek the calibrated vision plane before trusting any claw
                # decision (arrival OR servo): descend in bounded steps once
                # the coarse approach is done (model wants to stay / no model).
                want_vision = pred_step is None or pred_step < servo.stay_mm
                if want_vision and seek_z is not None and plane_z > seek_z + 2.0:
                    plane_z = max(seek_z, plane_z - seek_step)
                    self._auto_move(pose[0], pose[1], plane_z, descend_v,
                                    f"seek_z{moves}", step_to, start_to)
                    moves += 1
                    self.step = moves
                    self._emit_status()
                    continue

                if d.arrived:
                    arrived = True
                    break

                box_usable = d.box is not None and d.checks.get("fresh", True)
                if box_usable:
                    lost_holds = 0
                    claw_restarted_for_loss = False
                    post_restart_holds = 0
                servo_ready = box_usable and (pred_step is None or pred_step < servo.stay_mm)
                if servo_ready and corrections < servo.max_corrections:
                    step = servo.servo_step(d, object_key)
                    if step is None:                              # deadband/unusable box
                        stalled_holds += 1
                        if stalled_holds >= 3:
                            self._emit_log("WARN", "[auto] servo stalled inside deadband but not "
                                                   "arrived — check tolerances/setpoint")
                            break
                        time.sleep(0.4)
                        continue
                    if not fine:
                        fine = True
                        self._emit_log("INFO", "[auto] servo engaged — claw takes over, VLA paused")
                    corrections += 1
                    target_x = pose[0] + step[0]
                    target_y = pose[1] + step[1]
                    self._auto_move(target_x, target_y, plane_z, speed,
                                    f"servo{corrections}", step_to, start_to)
                    stalled_holds = 0
                    if last_target_fallback:
                        step_norm = (step[0] * step[0] + step[1] * step[1]) ** 0.5
                        if step_norm < servo.servo_max_step_mm - 1e-6:
                            last_servo_target = (target_x, target_y, plane_z)
                            self._emit_log(
                                "DEBUG",
                                f"[auto] saved verified board servo target "
                                f"({target_x:.0f},{target_y:.0f},{plane_z:.0f})",
                            )
                        else:
                            # A capped correction is only an intermediate
                            # waypoint, so it cannot authorize blind descent.
                            last_servo_target = None
                elif mode == "vla" and pred_step is not None and pred_step >= servo.stay_mm:
                    corrections = 0
                    scale = min(1.0, max_step / max(pred_step, 1e-6))
                    self._auto_move(pose[0] + dx * scale, pose[1] + dy * scale, plane_z, speed,
                                    f"vla{moves}", step_to, start_to)
                elif not box_usable:
                    lost_holds += 1                               # §3.3: box lost/stale -> hold + alert
                    stalled_holds = 0
                    why = "stale frame" if d.box is not None else "box lost"
                    self._emit_log("WARN", f"[auto] {why} ({lost_holds}) — holding at hover")
                    # Force one configured camera-service restart to shake a
                    # wedged capture loose. A saved robot-space target remains
                    # ineligible until that restart succeeds and its missing-box
                    # grace period completes.
                    if claw_restarted_for_loss:
                        post_restart_holds += 1

                    has_verified_target = (
                        last_target_fallback and last_servo_target is not None
                    )
                    restart_due, _, _ = _claw_loss_recovery_decision(
                        lost_holds=lost_holds,
                        post_restart_holds=post_restart_holds,
                        restart_succeeded=claw_restarted_for_loss,
                        has_verified_target=has_verified_target,
                        restart_after_holds=claw_restart_after,
                        post_restart_grace_holds=claw_restart_grace,
                    )
                    if restart_due and self.claw is not None:
                        self._emit_log(
                            "WARN",
                            "[auto] restarting the claw camera service (USB recovery)",
                        )
                        restart_ok = False
                        try:
                            self.claw.stop()
                            self.claw.start()
                            restart_ok = bool(getattr(self.claw, "running", True))
                            if not restart_ok:
                                self._emit_log(
                                    "WARN",
                                    "[auto] claw restart did not come online",
                                )
                        except Exception as exc:
                            self._emit_log(
                                "WARN", f"[auto] claw restart failed: {exc}"
                            )
                        fine = False
                        corrections = 0
                        claw_restarted_for_loss = restart_ok
                        post_restart_holds = 0

                    _, fallback_ready, recovery_pending = (
                        _claw_loss_recovery_decision(
                            lost_holds=lost_holds,
                            post_restart_holds=post_restart_holds,
                            restart_succeeded=claw_restarted_for_loss,
                            has_verified_target=has_verified_target,
                            restart_after_holds=claw_restart_after,
                            post_restart_grace_holds=claw_restart_grace,
                        )
                    )
                    if fallback_ready:
                        tx, ty, tz = last_servo_target
                        self._emit_log(
                            "WARN",
                            f"[auto] board still missing after claw restart — "
                            f"using last verified servo target "
                            f"({tx:.0f},{ty:.0f},{tz:.0f})",
                        )
                        arrived = True
                        break
                    if (
                        lost_holds >= int(rc.get("max_holds", 30))
                        and not recovery_pending
                    ):
                        raise RuntimeError(
                            "object never became visible to the claw — aborted"
                        )
                    time.sleep(1.0)
                    continue
                else:                                             # corrections exhausted -> re-coarse
                    corrections = 0
                    if fine:
                        fine = False
                        self._emit_log("WARN", "[auto] servo hit max corrections without arrival — "
                                               "check jacobian/tolerance; back to coarse approach")
                moves += 1
                self.step = moves
                self._emit_status()

            if not self._running:
                raise FallbackAbort("STOP pressed")
            if not arrived:
                self._stage("hold")
                self._emit_log("WARN", f"[auto] NOT arrived after {moves} moves — holding at hover "
                                       "(no descend without a positive arrival)")
                return

            # Approach-only test mode (§1.4 convergence check): stop AT the
            # vision plane on arrival so the operator can eyeball "directly
            # above" — no descend, no gates, no fingers, arm stays put.
            if bool(rc.get("approach_only")):
                pose = self._auto_pose()
                boxes, age = claw_boxes()
                d = servo.decide(boxes, object_key, None, age)
                self._stage("hover_hold")
                self._emit_log("INFO", f"[auto] APPROACH-ONLY: ARRIVED in {moves} moves at "
                                       f"({pose[0]:.1f},{pose[1]:.1f},{pose[2]:.1f}) — final "
                                       f"err=({d.err_cx:+.3f},{d.err_cy:+.3f}) "
                                       f"conf={d.box['conf']:.2f}" if d.box else
                                       "[auto] APPROACH-ONLY: arrived (box gone at final check)")
                self._emit_log("INFO", "[auto] holding at the vision plane — verify alignment by "
                                       "eye, then jog/descend manually if desired")
                return

            # ---------------- stage 3: scripted tail with gates ----------------
            if not place_only:
                # Descend confirmation: grasp z for board/knife comes from
                # recordings until they're taught — a human eyeball before the
                # blind descent is the anti-table-crash gate.
                if bool(rc.get("confirm_descend", True)):
                    pose = self._auto_pose()
                    self._emit_log("INFO", f"[auto] about to descend {pose[2]:.0f} -> "
                                           f"{grasp_z:.0f} mm  [{tail.get('grasp_z_source', 'config')}]")
                    self._stage("gate_descend")
                    if not self._wait_gate("descend_ok"):
                        raise FallbackAbort("STOP during descend gate")
                self._stage("descend")
                pose = self._auto_pose()
                self._auto_move(pose[0], pose[1], grasp_z, descend_v, "descend", step_to, start_to)

                grasp_action = lstm.get("grasp_action")
                two_phase_grasp = bool(
                    grasp_action
                    and lift_during_grasp
                    and grasp_start_z is not None
                )
                prepared_lstm = False
                if two_phase_grasp and not dry_run:
                    prepared_lstm = lstm_prepare(
                        grasp_action
                    )
                    if not prepared_lstm:
                        raise FallbackAbort("AGX grasp prepare failed")

                # Seat the thin object at the configured contact depth before
                # starting either the fingers or the normal lift.
                if grasp_contact_dwell > 0 and not dry_run:
                    if not self._wait_grasp_contact_dwell(grasp_contact_dwell):
                        raise FallbackAbort("STOP during grasp contact dwell")

                if prepared_lstm:
                    if not self._wait_grasp_goal_ready(grip):
                        raise FallbackAbort("STOP while waiting for real grasp goal")

                # Optional old experiment: a static lift before LSTM start.
                # butter_knife keeps this at zero; its absolute measured-Z
                # trigger below does not split the lift into a backoff waypoint.
                if grasp_action and grasp_backoff > 0 and not dry_run:
                    self._stage("grasp_backoff")
                    p = self._auto_pose()
                    self._emit_log("INFO", f"[auto] 觸底 grasp_z={grasp_z:.0f} → 上抬 "
                                           f"{grasp_backoff:.0f}mm 讓手指離開物件再抓(乾淨基線)")
                    self._auto_move(p[0], p[1], grasp_z + grasp_backoff, travel,
                                    "grasp_backoff", step_to, start_to)

                lift_thread: threading.Thread | None = None
                lift_errors: list[Exception] = []
                used_lstm = False
                if fixed_policy:
                    if requires_grasp:
                        if not (
                            isinstance(fixed_grasp_position, list)
                            and len(fixed_grasp_position) == 3
                        ):
                            raise FallbackAbort(
                                "fixed grasp position missing after preflight"
                            )
                        if lift_during_grasp and grasp_start_z is not None:
                            (
                                used_fixed,
                                lift_thread,
                                lift_errors,
                            ) = self._lift_and_start_grasp_at_z(
                                lift_z,
                                grasp_start_z,
                                travel,
                                step_to,
                                start_to,
                                lambda: self._command_fixed_finger_goal(
                                    grip,
                                    fixed_grasp_position,
                                    require_close=True,
                                    what="抓取",
                                    rc=rc,
                                ),
                            )
                            if not self._running:
                                raise FallbackAbort(
                                    "STOP before fixed grasp height trigger"
                                )
                        else:
                            used_fixed = self._command_fixed_finger_goal(
                                grip,
                                fixed_grasp_position,
                                require_close=True,
                                what="抓取",
                                rc=rc,
                            )
                        if not used_fixed:
                            raise FallbackAbort(
                                "fixed grasp did not converge"
                            )
                elif prepared_lstm:
                    # Knife: issue one uninterrupted move to the normal lift
                    # target. At measured Z>=trigger, synchronously write the
                    # cached real-close goal while that same move remains active.
                    used_lstm, lift_thread, lift_errors = (
                        self._lift_and_start_grasp_at_z(
                            lift_z,
                            grasp_start_z,
                            travel,
                            step_to,
                            start_to,
                            lambda: lstm_activate(),
                        )
                    )
                    if not self._running:
                        raise FallbackAbort("STOP before grasp height trigger")
                else:
                    # Trapezoid and unconfigured objects preserve the previous
                    # start-grasp-first ordering.
                    used_lstm = lstm_go(grasp_action)
                    if used_lstm and lift_during_grasp:
                        lift_thread, lift_errors = self._start_grasp_lift(
                            lift_z, travel, step_to, start_to,
                        )

                if not fixed_policy and used_lstm:
                    self._wait_fingers_settled(
                        grip,
                        timeout_s=None,
                        settle_ticks=int(rc.get("grasp_settle_ticks", 12)),
                        hold_s=float(rc.get("grasp_settle_hold_s", 1.2)),
                        min_step=int(rc.get("grasp_settle_min_step", 20)),
                        require_displaced=True, what="抓取",
                        nxt="完成同步上抬" if lift_thread else "上抬",
                        min_close_ticks=min_close_ticks,
                        min_close_fingers=min_close_fingers,
                        tracking_ticks=int(
                            rc.get("finger_settle_tracking_ticks", 30)
                        ),
                        min_tracking_fingers=int(
                            rc.get(
                                "grasp_min_tracking_fingers",
                                min_close_fingers,
                            )
                        ),
                        allow_manual_accept=True,
                    )
                    lstm_halt()                     # freeze fingers at the settled grip
                    if not self._running:
                        raise FallbackAbort("STOP during grasp")

                # Both the successful LSTM and failed-start/manual paths must
                # finish the already-issued lift before any gate or next move.
                if lift_thread is not None:
                    lift_thread.join()
                    if lift_errors:
                        raise lift_errors[0]
                    lifted_during_grasp = True

                if not fixed_policy and not used_lstm:
                    if grasp_action and grip is not None and not dry_run:
                        # LSTM didn't start. Do NOT fake a grasp: gripper.close()
                        # is only a single 20-tick 'c' jog (record_api jog table),
                        # NOT the tactile grasp — and run_LSTM5_toVLA.py has no
                        # close() to mimic anyway. Per operator request, just
                        # notify loudly and HOLD at the gate so they finish the
                        # pick by hand (UI 夾爪 buttons / manually), then press it.
                        self._emit_log("WARN", "[auto] ⚠️  AGX LSTM 抓取沒啟動 — 不自動夾"
                                               "(close() 只是 20-tick 微動,不是真抓)。"
                                               "請手動完成抓取(UI 夾爪按鈕/手動)後按 pick 閘門")
                    # With a measured-Z trigger the issued lift has completed;
                    # older/manual paths still lift after this confirmation.
                    if not self._confirm_pick_without_lstm(
                        auto_lift_after_dwell, grasp_contact_dwell
                    ):
                        raise FallbackAbort("STOP during pick gate")

            self._stage("to_release")
            pose = self._auto_pose()
            if not lifted_during_grasp:
                self._auto_move(pose[0], pose[1], lift_z, travel, "lift", step_to, start_to)
            else:
                self._emit_log("INFO", "[auto] 同步抓取上抬已完成 — 不重複 lift")
            if place_pose:
                self._auto_move(place_pose[0] / 1000.0, place_pose[1] / 1000.0, lift_z,
                                travel, "release_point", step_to, start_to)
                if place_z_mm is not None:
                    # Same anti-crash gate for the PLACE descent — board/knife
                    # place z are rig-verbal values until confirmed live.
                    if bool(rc.get("confirm_descend", True)):
                        p = self._auto_pose()
                        self._emit_log("INFO", f"[auto] about to PLACE-descend {p[2]:.0f} -> "
                                               f"{float(place_z_mm):.0f} mm at the release point")
                        self._stage("gate_place_descend")
                        if not self._wait_gate("descend_ok"):
                            raise FallbackAbort("STOP during place-descend gate")
                    p = self._auto_pose()
                    self._auto_move(p[0], p[1], float(place_z_mm), descend_v,
                                    "release_down", step_to, start_to)
            else:
                self._emit_log("WARN", "[auto] no place_approach_pose taught — staying put for release")

            self._stage("gate_release")
            release_action = lstm.get("release_action")
            used_lstm = False
            if fixed_policy:
                if not (
                    isinstance(fixed_release_position, list)
                    and len(fixed_release_position) == 3
                ):
                    raise FallbackAbort(
                        "fixed release position missing after preflight"
                    )
                released = self._command_fixed_finger_goal(
                    grip,
                    fixed_release_position,
                    require_close=False,
                    what="放開",
                    rc=rc,
                )
                if not released:
                    raise FallbackAbort("fixed release did not converge")
            else:
                used_lstm = lstm_go(release_action)
            if not fixed_policy and used_lstm:
                # Hands-off release (operator request): wait for the fingers to
                # finish opening (converge) then continue — no manual 'released'
                # gate. require_displaced=False: the release ends with the fingers
                # back near home, not displaced from it.
                released = self._wait_fingers_settled(
                    grip,
                    timeout_s=None,
                    settle_ticks=int(rc.get("grasp_settle_ticks", 12)),
                    hold_s=float(rc.get("grasp_settle_hold_s", 1.2)),
                    min_step=int(rc.get("grasp_settle_min_step", 20)),
                    require_displaced=False,
                    what="放開",
                    nxt="回位",
                    tracking_ticks=int(
                        rc.get("finger_settle_tracking_ticks", 30)
                    ),
                    min_tracking_fingers=int(
                        tail.get("release_min_tracking_fingers", 3)
                    ),
                    min_excursion_ticks=int(
                        tail.get("release_min_excursion_ticks", 60)
                    ),
                    min_excursion_fingers=int(
                        tail.get("release_min_excursion_fingers", 2)
                    ),
                    allow_manual_release=True,
                )
                if not released:
                    raise FallbackAbort("STOP before release convergence")
                if not self._running:
                    raise FallbackAbort("STOP during release")
                lstm_halt()
                if (
                    self._manual_release_requested
                    and manual_release_position is not None
                ):
                    if not self._complete_manual_release_fallback(
                        grip,
                        manual_release_position,
                        rc,
                    ):
                        raise FallbackAbort(
                            "STOP before manual release HOME confirmation"
                        )
            elif not fixed_policy:
                if release_action and grip is not None and not dry_run:
                    # Hardcoded release = ramp the fingers open to HOME_POS.
                    self._emit_log("WARN", "[auto] ⚠️  硬編放開(手指張開回 home)— 放開後按閘門")
                    grip.set_position(FINGER_HOME_POS, mode="stepped")
                # No LSTM telemetry to auto-judge: confirm before the return.
                if not self._wait_gate("released"):
                    raise FallbackAbort("STOP during release gate")
            # #3 fingers return to the OPEN/home state after release. The LSTM
            # stop() FREEZES the fingers wherever the release trajectory ended, so
            # ramp them back to HOME_POS via /set_position (stepped) — the 'o' jog
            # nests state_lock->com_lock against the reader and is unpaced, so it
            # does not reliably reach home. Verify the ramp actually ran.
            if not fixed_policy and grip is not None and not dry_run:
                self._home_fingers_or_raise(grip)

            self._stage("return")
            p = self._auto_pose()
            if place_z_mm is not None and place_pose:
                self._auto_move(p[0], p[1], lift_z, travel, "release_retract", step_to, start_to)
            if self._sim_pose is None:
                self.arm.go_ready()                               # no auto-home mid-flow (§3.4)
                # Ready point ⟺ fingers at home (user request): guarantee the
                # gripper is open at the ready pose. Idempotent — a near no-op
                # when the release already homed them.
                if not fixed_policy and grip is not None:
                    grip.set_position(FINGER_HOME_POS, mode="stepped")
            self._stage("done")
            self.status = "stopped"
            self._emit_log("INFO", f"[auto] COMPLETE {object_key} ({moves} approach moves)")

        except FallbackAbort as exc:
            self.status = "stopped"
            self._emit_log("WARN", f"[auto] aborted: {exc}")
            stop_physical_motion()
            hold_fixed_fingers()
        except Exception:
            self.status = "error"
            stop_physical_motion()
            hold_fixed_fingers()
            self._emit_log("ERROR", traceback.format_exc())
        finally:
            lstm_halt()   # never leave the LSTM loop driving fingers unattended
            if poll_was_running and self.arm is not None:
                self.arm.start_pose_polling()
            self._sim_pose = None
            self._running = False
            self._set_grasp_accept_available(False, emit=False)
            self._set_release_accept_available(False, emit=False)
            self._set_release_home_available(False, emit=False)
            self._manual_release_requested = False
            self._manual_release_position = None
            self.auto_stage = ""
            self.awaiting_gate = None
            self._emit_status()
