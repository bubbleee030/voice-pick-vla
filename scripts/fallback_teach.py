#!/usr/bin/env python3
"""Teach + bake the model-free fallback pick->place config.

This is the Phase 2 recorder for the fallback feature. It connects to the SAME
arm + gripper the demo server uses (via the same config overlay merge), lets you
capture a clean grasp pose, the closed grip, a step-by-step 3-finger release, and
the drop height, then bakes them into ``config/fallback_taught.yaml``.

The runner (ArmService.run_fallback) deep-merges that taught file over the
committed seeds in config/objects.yaml, so re-teaching is just regenerating one
file and can never corrupt the curated config.

It records ONLY arm pose (Modbus) + gripper finger positions (HTTP) -- no
cameras / RealSense / AGX-tactile -- which is the whole point: those were the
fragile pieces. It never commands arm MOTION; you jog the arm yourself (pendant
or hand-guide) and press a key to capture.

IMPORTANT: stop the demo server first. The arm controller typically allows a
single Modbus client, so the server and this tool cannot both own it.

Usage:
    python scripts/fallback_teach.py --object trapezoid
    python scripts/fallback_teach.py --object board --env-config config/env/ubuntu.local.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

# Allow `from src...` when run as `python scripts/fallback_teach.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime_config import (  # noqa: E402
    FALLBACK_TAUGHT_PATH,
    load_runtime_bundle,
    load_taught_fallback,
    merge_fallback,
)
from src.services.arm_service import ArmService, build_fallback_plan  # noqa: E402
from src.services.gripper_service import GripperService  # noqa: E402

DEFAULT_ENV_CFG = "config/env/ubuntu.local.yaml"


def _default_env_config() -> str | None:
    candidate = PROJECT_ROOT / DEFAULT_ENV_CFG
    return DEFAULT_ENV_CFG if candidate.exists() else None


def _read_arm_pose(arm: ArmService, tries: int = 6) -> list[int] | None:
    """Read a valid current pose (raw controller units), skipping all-zero reads."""
    if arm.ctrl is None:
        return None
    for _ in range(max(1, tries)):
        try:
            raw = arm.ctrl.read_current_pose()
        except Exception as exc:  # noqa: BLE001
            print(f"  [arm] read error: {exc}")
            raw = None
        if raw and len(raw) >= 6 and (abs(int(raw[0])) + abs(int(raw[1])) + abs(int(raw[2])) != 0):
            return [int(v) for v in raw[:6]]
        time.sleep(0.1)
    return None


def _read_grip(gripper: GripperService) -> list[int] | None:
    state = gripper.get_state()
    if not isinstance(state, dict):
        return None
    pos = state.get("current_pos")
    if isinstance(pos, list) and len(pos) == 3:
        return [int(v) for v in pos]
    return None


def _fmt_pose(pose: list[int] | None) -> str:
    if not pose:
        return "(none)"
    mm = [pose[0] / 1000.0, pose[1] / 1000.0, pose[2] / 1000.0]
    return f"{pose}  (xyz mm: {mm[0]:.1f}, {mm[1]:.1f}, {mm[2]:.1f})"


def resolve_effective(object_key: str, demo_cfg: dict, objects_cfg: dict) -> tuple[dict, dict, list]:
    """Mirror ArmService.run_fallback's merge: objects.yaml seed + taught overrides."""
    fb_cfg = dict(objects_cfg.get("fallback", {}).get(object_key, {}) or {})
    shared = dict(demo_cfg.get("fallback", {}) or {})
    taught = load_taught_fallback()
    if taught:
        t_obj = taught.get("fallback", {}).get(object_key)
        if isinstance(t_obj, dict):
            fb_cfg = merge_fallback(fb_cfg, t_obj)
        t_shared = taught.get("shared")
        if isinstance(t_shared, dict):
            shared = merge_fallback(shared, t_shared)
    return fb_cfg, shared, demo_cfg.get("ready_pose")


def print_plan(object_key: str, fb_cfg: dict, shared: dict, ready: list | None) -> None:
    """Dry-run print of the exact plan run_fallback would execute. No motion."""
    if not ready:
        print("  [!] ready_pose not configured in demo_config.yaml")
        return
    try:
        plan, warnings = build_fallback_plan(object_key, fb_cfg, shared, ready)
    except ValueError as exc:
        print(f"  [!] cannot build plan: {exc}")
        return
    print(f"\n=== REPLAY PREVIEW (dry-run, NO MOTION): {object_key} ===")
    for w in warnings:
        print(f"  [warn] {w}")
    print(f"  {'#':>2}  {'step':<14} {'kind':<8} detail")
    print(f"  {'-'*2}  {'-'*14} {'-'*8} {'-'*40}")
    for i, s in enumerate(plan, 1):
        kind = s["kind"]
        if kind == "move":
            p = s["pose"]
            detail = (f"speed={s['speed']:>3}%  pose={p}  "
                      f"(xyz mm {p[0]/1000.0:.1f}, {p[1]/1000.0:.1f}, {p[2]/1000.0:.1f})")
        elif kind == "grip":
            detail = f"positions={s['positions']}" if s["positions"] else "close()"
        elif kind == "hold":
            detail = f"{s['seconds']}s"
        elif kind == "release":
            detail = f"{s['steps']}"
        else:
            detail = ""
        print(f"  {i:>2}  {s['label']:<14} {kind:<8} {detail}")
    print("=== end preview ===\n")


HELP = """
Commands:
  pick            capture grasp pose AND closed grip in one shot (recommended)
  pose            capture grasp pose only (current arm pose)
  grip            capture grasp finger position only (current gripper pos)
  state           print current arm pose + gripper finger pos
  open [d]        gripper open (jog); optional settle delay seconds
  close [d]       gripper close (jog)
  set a b c       gripper set_position to [a,b,c] (jog, stepped)
  rstep [delay]   append current gripper pos as a release step (default delay 0.3s)
  ropen [delay]   append a final full-'open' release step
  rundo           remove last release step
  rclear          clear all release steps
  z [mm]          capture place descend height (current arm z, or explicit mm)
  approach        capture SHARED place-approach pose (current arm pose)
  show            show the captured draft
  preview         dry-run the full pick->place plan (draft over saved config; NO motion)
  save            write draft into config/fallback_taught.yaml (merged)
  servo on|off    servo control  (CAUTION: 'off' may let the arm sag)
  help            show this help
  quit            exit (warns if there are unsaved captures)
"""


class TeachSession:
    def __init__(self, object_key: str, arm, gripper, demo_cfg: dict | None = None, objects_cfg: dict | None = None):
        self.object_key = object_key
        self.arm = arm
        self.gripper = gripper
        self.demo_cfg = demo_cfg or {}
        self.objects_cfg = objects_cfg or {}
        self.grasp_pose: list[int] | None = None
        self.grasp_finger_pos: list[int] | None = None
        self.place_z_mm: float | None = None
        self.release_steps: list = []
        self.place_approach_pose: list[int] | None = None
        self.dirty = False

    # ---- captures -------------------------------------------------------
    def cap_pose(self) -> None:
        pose = _read_arm_pose(self.arm)
        if pose is None:
            print("  [!] could not read a valid arm pose (all-zero or no connection)")
            return
        self.grasp_pose = pose
        self.dirty = True
        print(f"  grasp_pose = {_fmt_pose(pose)}")

    def cap_grip(self) -> None:
        pos = _read_grip(self.gripper)
        if pos is None:
            print(f"  [!] could not read gripper position ({self.gripper.last_error or 'no state'})")
            return
        self.grasp_finger_pos = pos
        self.dirty = True
        print(f"  grasp_finger_pos = {pos}")

    def cap_pick(self) -> None:
        self.cap_pose()
        self.cap_grip()

    def cap_z(self, arg: str) -> None:
        if arg:
            try:
                self.place_z_mm = float(arg)
            except ValueError:
                print(f"  [!] not a number: {arg}")
                return
        else:
            pose = _read_arm_pose(self.arm)
            if pose is None:
                print("  [!] could not read arm z")
                return
            self.place_z_mm = round(pose[2] / 1000.0, 1)
        self.dirty = True
        print(f"  place.z_mm = {self.place_z_mm}")

    def cap_approach(self) -> None:
        pose = _read_arm_pose(self.arm)
        if pose is None:
            print("  [!] could not read arm pose")
            return
        self.place_approach_pose = pose
        self.dirty = True
        print(f"  shared.place_approach_pose = {_fmt_pose(pose)}")

    def add_rstep(self, arg: str) -> None:
        pos = _read_grip(self.gripper)
        if pos is None:
            print(f"  [!] could not read gripper position ({self.gripper.last_error or 'no state'})")
            return
        delay = 0.3
        if arg:
            try:
                delay = float(arg)
            except ValueError:
                print(f"  [!] bad delay: {arg}")
                return
        self.release_steps.append({"pos": pos, "delay_s": delay})
        self.dirty = True
        print(f"  release step {len(self.release_steps)}: pos={pos} delay={delay}s")

    def add_ropen(self, arg: str) -> None:
        delay = 0.5
        if arg:
            try:
                delay = float(arg)
            except ValueError:
                print(f"  [!] bad delay: {arg}")
                return
        self.release_steps.append({"open": True, "delay_s": delay})
        self.dirty = True
        print(f"  release step {len(self.release_steps)}: full open, delay={delay}s")

    # ---- gripper jog ----------------------------------------------------
    def jog_open(self, arg: str) -> None:
        ok = self.gripper.open()
        print(f"  gripper open -> {'ok' if ok else 'FAIL: ' + (self.gripper.last_error or '')}")
        self._settle(arg, 0.5)

    def jog_close(self, arg: str) -> None:
        ok = self.gripper.close()
        print(f"  gripper close -> {'ok' if ok else 'FAIL: ' + (self.gripper.last_error or '')}")
        self._settle(arg, 0.5)

    def jog_set(self, parts: list[str]) -> None:
        if len(parts) != 3:
            print("  usage: set a b c")
            return
        try:
            target = [int(p) for p in parts]
        except ValueError:
            print("  [!] positions must be integers")
            return
        ok = self.gripper.set_position(target, mode="stepped")
        print(f"  set_position {target} -> {'ok' if ok else 'FAIL: ' + (self.gripper.last_error or '')}")

    def _settle(self, arg: str, default: float) -> None:
        try:
            time.sleep(float(arg) if arg else default)
        except ValueError:
            time.sleep(default)

    def servo(self, arg: str) -> None:
        if arg == "on":
            self.arm.ctrl.servo_on()
            print("  servo ON")
        elif arg == "off":
            print("  [CAUTION] servo OFF -- support the arm; it may sag under gravity.")
            self.arm.ctrl.servo_off()
            print("  servo OFF")
        else:
            print("  usage: servo on|off")

    def state(self) -> None:
        print(f"  arm pose : {_fmt_pose(_read_arm_pose(self.arm))}")
        print(f"  grip pos : {_read_grip(self.gripper)}")

    def show(self) -> None:
        print(f"\n--- draft for '{self.object_key}' ---")
        print(f"  grasp_pose       : {_fmt_pose(self.grasp_pose)}")
        print(f"  grasp_finger_pos : {self.grasp_finger_pos}")
        print(f"  place.z_mm       : {self.place_z_mm}")
        print(f"  release_steps    : {self.release_steps or '(none)'}")
        print(f"  [shared] place_approach_pose: {_fmt_pose(self.place_approach_pose)}")
        print("------------------------------\n")

    def _draft_block(self) -> dict:
        """The captured (possibly unsaved) values as a fallback config block."""
        obj: dict = {}
        if self.grasp_pose is not None:
            obj["grasp_pose"] = self.grasp_pose
        if self.grasp_finger_pos is not None:
            obj["grasp_finger_pos"] = self.grasp_finger_pos
        place: dict = {}
        if self.place_z_mm is not None:
            place["z_mm"] = self.place_z_mm
        if self.release_steps:
            place["release_steps"] = self.release_steps
        if place:
            obj["place"] = place
        return obj

    def preview(self) -> None:
        """Preview what run_fallback WOULD execute: saved config + current draft."""
        fb_cfg, shared, ready = resolve_effective(self.object_key, self.demo_cfg, self.objects_cfg)
        fb_cfg = merge_fallback(fb_cfg, self._draft_block())
        if self.place_approach_pose is not None:
            shared = merge_fallback(shared, {"place_approach_pose": self.place_approach_pose})
        print_plan(self.object_key, fb_cfg, shared, ready)

    # ---- bake -----------------------------------------------------------
    def save(self) -> None:
        obj_block: dict = {}
        if self.grasp_pose is not None:
            obj_block["grasp_pose"] = self.grasp_pose
        if self.grasp_finger_pos is not None:
            obj_block["grasp_finger_pos"] = self.grasp_finger_pos
        place: dict = {}
        if self.place_z_mm is not None:
            place["z_mm"] = self.place_z_mm
        if self.release_steps:
            place["release_steps"] = self.release_steps
        if place:
            obj_block["place"] = place

        if not obj_block and self.place_approach_pose is None:
            print("  [!] nothing captured yet; nothing to save")
            return

        # Load existing taught file and merge so other objects are preserved.
        existing: dict = {}
        if FALLBACK_TAUGHT_PATH.exists():
            try:
                existing = yaml.safe_load(FALLBACK_TAUGHT_PATH.read_text(encoding="utf-8")) or {}
            except Exception as exc:  # noqa: BLE001
                print(f"  [!] could not read existing taught file ({exc}); starting fresh")
                existing = {}
        if not isinstance(existing, dict):
            existing = {}

        existing.setdefault("fallback", {})
        if obj_block:
            cur = existing["fallback"].get(self.object_key, {})
            cur.update(obj_block)
            existing["fallback"][self.object_key] = cur
        if self.place_approach_pose is not None:
            existing.setdefault("shared", {})
            existing["shared"]["place_approach_pose"] = self.place_approach_pose
        existing["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        existing["_note"] = "AUTO-GENERATED by scripts/fallback_teach.py; merged over config/objects.yaml fallback seeds."

        FALLBACK_TAUGHT_PATH.parent.mkdir(parents=True, exist_ok=True)
        FALLBACK_TAUGHT_PATH.write_text(
            yaml.safe_dump(existing, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        self.dirty = False
        print(f"  saved -> {FALLBACK_TAUGHT_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Teach + bake fallback pick->place config")
    parser.add_argument("--object", required=True, help="object key (e.g. trapezoid, board, tweezers)")
    parser.add_argument("--profile", default="full")
    parser.add_argument("--env-config", default=_default_env_config())
    parser.add_argument("--no-gripper", action="store_true", help="skip gripper connect (pose-only teaching)")
    parser.add_argument("--preview", action="store_true",
                        help="print the dry-run plan for --object and exit (no connection, no motion)")
    args = parser.parse_args()

    bundle = load_runtime_bundle(profile=args.profile, env_config=args.env_config)
    demo_cfg = bundle.demo_config
    modbus_cfg = bundle.modbus_config
    objects_cfg = yaml.safe_load((PROJECT_ROOT / "config" / "objects.yaml").read_text(encoding="utf-8")) or {}

    # Non-interactive preview: no hardware needed.
    if args.preview:
        fb_cfg, shared, ready = resolve_effective(args.object, demo_cfg, objects_cfg)
        print_plan(args.object, fb_cfg, shared, ready)
        return 0

    print("=" * 70)
    print("Fallback teach  ::  STOP the demo server first (arm is single-owner Modbus).")
    print("=" * 70)
    print(f"env overlay : {bundle.env_overlay_path or '(none)'}")

    gripper = GripperService(demo_cfg, socketio=None)
    if not args.no_gripper:
        ep = gripper.resolve_endpoint()
        if ep is None:
            print(f"[gripper] NOT connected ({gripper.last_error}). Re-run with --no-gripper to teach pose only, "
                  "or fix the gripper, then retry.")
            return 2
        print(f"[gripper] connected: {gripper.active_base_url} (api={gripper.api_version})")

    arm = ArmService(demo_cfg, modbus_cfg, socketio=None, gripper_service=gripper)
    if not arm.auto_connect():
        print("[arm] NOT connected. Is the demo server still holding the Modbus port? "
              "Stop it and retry. Connections tried: "
              f"{[c.get('label') for c in demo_cfg.get('arm', {}).get('connections', [])]}")
        return 2
    print(f"[arm] connected: {arm.connection_label}")

    session = TeachSession(args.object, arm, gripper, demo_cfg=demo_cfg, objects_cfg=objects_cfg)
    print(HELP)
    session.state()

    try:
        while True:
            try:
                raw = input(f"teach[{args.object}]> ").strip()
            except EOFError:
                raw = "quit"
            if not raw:
                continue
            parts = raw.split()
            cmd, rest = parts[0].lower(), parts[1:]
            arg = rest[0] if rest else ""

            if cmd in ("quit", "exit", "q"):
                if session.dirty:
                    if input("  unsaved captures -- quit anyway? [y/N] ").strip().lower() != "y":
                        continue
                break
            elif cmd == "help":
                print(HELP)
            elif cmd == "pick":
                session.cap_pick()
            elif cmd == "pose":
                session.cap_pose()
            elif cmd == "grip":
                session.cap_grip()
            elif cmd == "state":
                session.state()
            elif cmd == "open":
                session.jog_open(arg)
            elif cmd == "close":
                session.jog_close(arg)
            elif cmd == "set":
                session.jog_set(rest)
            elif cmd == "rstep":
                session.add_rstep(arg)
            elif cmd == "ropen":
                session.add_ropen(arg)
            elif cmd == "rundo":
                if session.release_steps:
                    session.release_steps.pop()
                    print(f"  removed; {len(session.release_steps)} step(s) left")
                else:
                    print("  no release steps")
            elif cmd == "rclear":
                session.release_steps = []
                print("  release steps cleared")
            elif cmd == "z":
                session.cap_z(arg)
            elif cmd == "approach":
                session.cap_approach()
            elif cmd == "show":
                session.show()
            elif cmd == "preview":
                session.preview()
            elif cmd == "save":
                session.save()
            elif cmd == "servo":
                session.servo(arg)
            else:
                print(f"  unknown command: {cmd} (type 'help')")
    finally:
        try:
            arm.disconnect()
        except Exception:
            pass
        try:
            gripper.stop_monitoring()
        except Exception:
            pass
        print("disconnected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
