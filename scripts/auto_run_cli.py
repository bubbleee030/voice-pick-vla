#!/usr/bin/env python3
"""Terminal runner for the three-object auto pipeline — test everything WITHOUT
the UI (§3; UI wiring is the last step).

Runs the exact same code path the UI's Auto Run button uses
(VLAService.start_auto → arrival/servo → gated tail → AGX fingers), but the
gates are answered with Enter in this terminal and every log line prints here.

Two ways to run:
  * server DOWN  -> this CLI opens its own cameras/arm (single-owner).
  * server UP    -> this CLI ATTACHES to it automatically (--attach): the server
                    keeps the cameras/arm and the WEB UI STAYS LIVE so you can
                    watch the cam scene, while you still start the run and answer
                    gates from this terminal. Use --local to force own devices
                    (only works if the server is actually stopped).

Modes (safest first):
  --observe   No arm commands, no model. Live claw arrival readout at 2 Hz:
              jog the arm over the object (pendant/UI beforehand) and watch
              err(cx,cy) / ARRIVED. Verifies calibration before any motion.
  --dry       Full auto flow, arm simulated (zero motion; needs cameras+GPU).
  (live)      The real thing. First time: low speed, hand near the pendant stop.

Usage:
  conda run --no-capture-output -n voice_pick python scripts/auto_run_cli.py \
      --object trapezoid --observe
  ... --object trapezoid --dry
  ... --object trapezoid              # live VLA→servo→descend→gates
  ... --object board --mode locator   # no-VLA rung (cam1 one-shot + servo)
  ... --object butter_knife           # fixed-xy: recorded pick x,y + blind descend
  ... --no-agx                        # skip the AGX finger calls
  ... --object trapezoid --attach     # server up: drive it, web stays live (auto)
"""
from __future__ import annotations

import argparse
import os
import select
import socket
import sys
import time
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BUNDLE_ROOT))
RELEASE_STAGE_PROMPTS = {
    "release": "Release ready — Enter stops LSTM and returns fingers HOME",
    "release_pose": (
        "Enter stops LSTM and moves fingers to the configured release pose"
    ),
    "release_home": (
        "Release pose verified — Enter returns fingers HOME"
    ),
}


def poll_terminal_enter(stream=sys.stdin) -> bool:
    """Consume one line only when the terminal is readable right now."""
    try:
        readable, _, _ = select.select([stream], [], [], 0.0)
    except (OSError, ValueError, TypeError):
        return False
    if not readable:
        return False
    stream.readline()
    return True


def server_running(port: int = 8090) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def run_attached(args, url: str = "http://127.0.0.1:8090") -> int:
    """Drive the RUNNING demo server over socket.io: start the auto-run, stream
    its logs here, and answer gates with Enter — while the server keeps the
    cameras/arm and the web UI shows the live cam scene. Gates can also be
    answered from the UI buttons; whichever comes first wins."""
    import time as _time
    try:
        import socketio  # python-socketio client (same pkg flask-socketio uses)
    except Exception as exc:
        print(f"[FAIL] python-socketio not available for --attach: {exc}")
        return 1

    overrides: dict = {}
    if args.speed is not None:
        overrides["speed_percent"] = args.speed
    if args.max_moves is not None:
        overrides["max_moves"] = args.max_moves
    if args.approach_only:
        overrides["approach_only"] = True
    if args.place_only:
        overrides["place_only"] = True
    if args.no_agx:
        overrides["no_agx"] = True
    if args.xy is not None:
        overrides["fixed_xy"] = list(args.xy)
    elif args.fixed_xy:
        overrides["use_fixed_xy"] = True

    sio = socketio.Client(reconnection=False)
    st = {
        "await": None,
        "last": None,
        "running": False,
        "started": False,
        "grasp_accept_available": False,
        "grasp_prompted": False,
        "finger_stage_action": None,
        "release_prompted": None,
    }

    @sio.on("vla_log")
    def _on_log(d):
        print(f"  [{d.get('level', '?')}] {d.get('message', '')}")

    @sio.on("vla_status")
    def _on_status(d):
        st["running"] = bool(d.get("running"))
        if st["running"]:
            st["started"] = True
        aw = d.get("awaiting_gate")
        if aw and aw != st["last"]:      # fresh gate (None between gates)
            st["await"] = aw
        st["last"] = aw
        st["grasp_accept_available"] = bool(
            d.get("grasp_accept_available")
        )
        st["finger_stage_action"] = d.get("finger_stage_action")

    try:
        sio.connect(url, wait_timeout=6)
    except Exception as exc:
        print(f"[FAIL] cannot attach to the demo server at {url}: {exc}")
        return 1

    # python-socketio installs its OWN SIGINT handler on connect that just
    # disconnects the client — which would leave the server's run going AND make
    # our vla_stop emit fail ("/ is not a connected namespace"). Reinstall a
    # handler that raises so we STOP the run (emit while still connected) first.
    import signal
    def _raise_kbd(signum, frame):
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGINT, _raise_kbd)
    except Exception:
        pass

    print(f"[attach] connected to {url} — the web UI stays live; watch the cam scene there.")
    print(f"[run] {args.object}  mode={args.mode}  dry={args.dry}  overrides={overrides or '{}'}")
    print("[run] answer gates here with Enter (or use the UI buttons); Ctrl-C = STOP\n")
    sio.emit("vla_auto_start", {"object": args.object, "mode": args.mode,
                                "dry_run": bool(args.dry), "overrides": overrides})

    ok = True
    try:
        t0 = _time.time()
        while not st["started"] and _time.time() - t0 < 15:
            _time.sleep(0.1)
        if not st["started"]:
            print("[WARN] server did not report the run starting (check the UI logs)")
        while st["started"] and st["running"]:
            if st["await"]:
                name = st["await"]
                st["await"] = None
                input(f"\n>>> GATE [{name}] — Enter to continue: ")
                sio.emit("vla_gate", {"name": name})
            action = st["finger_stage_action"]
            if st["grasp_accept_available"]:
                if not st["grasp_prompted"]:
                    print(
                        "\n>>> Real grip verified — press Enter to accept "
                        "the current grip; Ctrl-C = STOP"
                    )
                    st["grasp_prompted"] = True
                if poll_terminal_enter():
                    sio.emit("vla_accept_grasp")
            else:
                st["grasp_prompted"] = False

            if action in RELEASE_STAGE_PROMPTS:
                if st["release_prompted"] != action:
                    print(
                        "\n>>> " + RELEASE_STAGE_PROMPTS[action]
                        + "; Ctrl-C = STOP"
                    )
                    st["release_prompted"] = action
                if poll_terminal_enter():
                    sio.emit("vla_complete_finger_stage")
            else:
                st["release_prompted"] = None
            _time.sleep(0.15)
    except KeyboardInterrupt:
        print("\n[STOP] requested — stopping the run on the server")
        try:
            if sio.connected:
                sio.emit("vla_stop")
                _time.sleep(0.6)
        except Exception as exc:
            print(f"[warn] could not send STOP over the socket ({exc}) — press the UI STOP button!")
        ok = False
    finally:
        try:
            sio.disconnect()
        except Exception:
            pass
    print("\n[done] detached from the server (web UI and server keep running)")
    return 0 if ok else 1


def _reboot_fingers(args) -> int:
    """Clear an AGX finger overload torque-off latch WITHOUT restarting the whole
    service (no model reload). Posts to the AGX :5003 /motor/reboot directly, so
    it works whether or not the demo server is up (the gripper is plain HTTP)."""
    from src.runtime_config import load_runtime_bundle
    from src.services.gripper_service import GripperService
    bundle = load_runtime_bundle(profile="full", env_config=args.env_config)
    grip = GripperService(bundle.demo_config)
    print("[reboot] rebooting AGX finger motors (clear overload torque-off latch)...")
    res = grip.reboot_motors(home=True)
    if res and res.get("ok"):
        print(f"[reboot] OK — motor_pos={res.get('motor_pos')} homed={res.get('homed')} "
              f"rebooted={res.get('rebooted')}")
        return 0
    print(f"[reboot] FAILED: {(res or {}).get('error', 'no response from AGX :5003')}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object", choices=["trapezoid", "board", "butter_knife"],
                    help="object to pick (required unless --reboot-fingers)")
    ap.add_argument("--mode", default="vla", choices=["vla", "locator"])
    ap.add_argument("--observe", action="store_true", help="arrival readout only, no motion")
    ap.add_argument("--dry", action="store_true", help="simulate arm (no motion)")
    ap.add_argument("--approach-only", action="store_true",
                    help="LIVE convergence test (§1.4): VLA+servo until ARRIVED, then hold at "
                         "the vision plane — no descend, no gates, no fingers")
    ap.add_argument("--place-only", action="store_true",
                    help="LIVE release-leg test: from the current pose, lift -> release point -> "
                         "gated place-descend -> release gate -> return. Skips approach/pick.")
    ap.add_argument("--no-agx", action="store_true", help="skip AGX finger actions (gates become plain confirms)")
    ap.add_argument("--fixed-xy", action="store_true",
                    help="skip approach/servo — move to the RECORDED pick x,y (objects.yaml fallback "
                         "grasp_pose) and descend blind. Auto-on for objects with use_recorded_xy "
                         "in config (butter_knife today).")
    ap.add_argument("--xy", nargs=2, type=float, metavar=("X_MM", "Y_MM"), default=None,
                    help="explicit x,y (mm) to use with --fixed-xy, overriding the recorded pose")
    ap.add_argument("--speed", type=int, default=None, help="override vla_auto.speed_percent")
    ap.add_argument("--max-moves", type=int, default=None)
    ap.add_argument("--attach", action="store_true",
                    help="drive the RUNNING demo server (:8090) instead of opening our own "
                         "cameras/arm — the web UI stays live so you can watch the cam scene. "
                         "Auto-selected when the server is up.")
    ap.add_argument("--local", action="store_true",
                    help="force local devices even if the server is up (server must actually be "
                         "stopped — cameras/Modbus are single-owner)")
    ap.add_argument("--reboot-fingers", action="store_true",
                    help="recover a dead/stuck AGX finger (overload torque-off latch): reboot the "
                         "motors + re-enable torque + home via AGX :5003 /motor/reboot. Talks "
                         "straight to the AGX (server up or down), then exits.")
    ap.add_argument("--env-config", default="config/env/ubuntu.local.yaml")
    args = ap.parse_args()

    if args.reboot_fingers:
        return _reboot_fingers(args)
    if not args.object:
        ap.error("--object is required (unless --reboot-fingers)")

    srv_up = server_running()

    # Server up + not forced local + not observe -> ATTACH: let the server keep
    # the cameras/arm and render the web view; we just start the run and answer
    # gates from here. This is what keeps the web alive during a CLI run.
    if srv_up and not args.local and not args.observe:
        return run_attached(args)
    if args.attach and not srv_up:
        print("[FAIL] --attach needs the demo server running on :8090 (start it first)")
        return 1
    if srv_up:   # --local or --observe while the server owns the devices
        print("[FAIL] demo server is running on :8090 — stop it first (single-owner "
              "cameras/Modbus), or drop --local/--observe to attach instead")
        return 1

    from src.runtime_config import load_runtime_bundle, build_auto_run_cfg
    import yaml
    bundle = load_runtime_bundle(profile="full", env_config=args.env_config)
    demo_cfg, modbus_cfg = bundle.demo_config, bundle.modbus_config
    objects_cfg = yaml.safe_load((BUNDLE_ROOT / "config/objects.yaml").read_text(encoding="utf-8")) or {}

    # ---------------- observe mode: claw + calibration only ----------------
    if args.observe:
        from src.services.claw_camera_service import ClawCameraService
        from src.arrival import ArrivalServo
        claw = ClawCameraService(demo_cfg)
        claw.start()
        servo = ArrivalServo()
        sx, sy, cls = servo.setpoint(args.object)
        tx, ty = servo.tols(args.object)
        print(f"[observe] {args.object} (class {cls})  setpoint=({sx:.3f},{sy:.3f})  "
              f"tol=({tx:.3f},{ty:.3f})  plane z={servo.servo_z(args.object)}")
        print("[observe] jog the arm over the object; Ctrl-C to quit")
        try:
            while True:
                boxes, age = claw.get_detections()
                d = servo.decide(boxes, args.object, pred_step_mm=None, detect_age_s=age)
                if d.box is None:
                    print("  no box")
                else:
                    step = servo.servo_step(d, args.object)
                    print(f"  conf={d.box['conf']:.2f} err=({d.err_cx:+.3f},{d.err_cy:+.3f}) "
                          f"{d.reason}"
                          + (f"  servo_would_move=({step[0]:+.1f},{step[1]:+.1f})mm" if step else ""))
                time.sleep(0.5)
        except KeyboardInterrupt:
            claw.stop()
            return 0

    # ---------------- full pipeline (dry or live) ----------------
    resolved = build_auto_run_cfg(demo_cfg, objects_cfg, args.object)
    if resolved is None:
        print(f"[FAIL] no vla_auto config for {args.object}")
        return 1
    # The current checkpoint is trapezoid-only; board/knife have no trained
    # VLA until the threeobj checkpoint passes eval (Day 2). Their working
    # rung TODAY is the side-cam locator (same servo/arrival afterwards).
    ckpt_name = str(demo_cfg.get("vla_auto", {}).get("checkpoint", ""))
    if (args.mode == "vla" and args.object != "trapezoid"
            and "threeobj" not in ckpt_name and not args.place_only):
        print(f"[WARN] checkpoint '{Path(ckpt_name).name}' was trained on the trapezoid only — "
              f"VLA predictions for '{args.object}' will be garbage.\n"
              f"       Use --mode locator for this object until the threeobj checkpoint is deployed.")
        if input("       continue with VLA anyway? [y/N] ").strip().lower() != "y":
            return 1
    run_cfg, tail_cfg, obj_cfg = resolved
    if args.speed is not None:
        run_cfg["speed_percent"] = args.speed
    if args.max_moves is not None:
        run_cfg["max_moves"] = args.max_moves
    if args.approach_only:
        run_cfg["approach_only"] = True
        tail_cfg["lstm"] = {}          # never touch the fingers in this mode
    if args.place_only:
        run_cfg["place_only"] = True
    if args.fixed_xy or args.xy is not None:
        run_cfg["fixed_xy"] = list(args.xy) if args.xy is not None else run_cfg.get("recorded_xy")
        if not run_cfg["fixed_xy"]:
            print(f"[FAIL] --fixed-xy: no recorded x,y for {args.object} "
                  f"(needs a fallback grasp_pose in config/objects.yaml)")
            return 1
    if run_cfg.get("fixed_xy"):
        print(f"[run] FIXED-XY: skipping approach/servo → recorded x,y = "
              f"{run_cfg['fixed_xy']} mm, descend blind on grasp z")
    if args.no_agx:
        tail_cfg["lstm"] = {}

    from src.services.claw_camera_service import ClawCameraService
    from src.services.realsense_service import RealSenseService
    from src.services.gripper_service import GripperService
    from src.services.arm_service import ArmService
    from src.services.vla_service import VLAService

    print("[setup] starting cameras …")
    realsense = RealSenseService(demo_cfg)
    realsense.start()
    claw = ClawCameraService(demo_cfg)
    claw.start()

    gripper = GripperService(demo_cfg)
    if not args.no_agx:
        ep = gripper.resolve_endpoint()
        print(f"[setup] gripper endpoint: {gripper.active_base_url or 'UNREACHABLE (gates go manual)'}"
              + (f"  api={gripper.api_version}" if ep else ""))

    arm = ArmService(demo_cfg, modbus_cfg, socketio=None, gripper_service=gripper)
    connected = arm.auto_connect()
    print(f"[setup] arm: {'connected ' + arm.connection_label if connected else 'NOT connected'}")
    if not connected and not args.dry:
        print("[FAIL] live run needs the arm; use --dry to simulate")
        return 1

    vla = VLAService(socketio=None, arm_service=arm if connected else None,
                     realsense_service=realsense, claw_service=claw)
    vla._emit_log = lambda lvl, msg: print(f"  [{lvl}] {msg}")

    # give the cameras a moment to deliver first frames
    time.sleep(2.0)

    ckpt = str(demo_cfg.get("vla_auto", {}).get("checkpoint", ""))
    print(f"\n[run] {args.object}  mode={args.mode}  dry={args.dry}  ckpt={ckpt}")
    print("[run] Ctrl-C = STOP at any point\n")
    vla.start_auto(
        object_key=args.object,
        checkpoint_path=ckpt,
        embed_path=str(obj_cfg.get("embed", "")),
        instruction=str(obj_cfg.get("instruction", "")),
        mode=args.mode,
        run_cfg=run_cfg,
        tail_cfg=tail_cfg,
        dry_run=args.dry,
    )

    # Terminal gates: Enter plays the role of the UI button.
    prompts = {
        "descend_ok": "CHECK the arm is above the object and the target z is sane — Enter to descend",
        "pick_done": "fingers done PICKING — Enter to freeze them and carry",
        "released": "fingers done RELEASING — Enter to freeze them and return",
    }
    try:
        grasp_prompted = False
        release_prompted = None
        while vla._running:
            gate = vla.awaiting_gate
            if gate:
                input(f"\n>>> GATE [{gate}]  {prompts.get(gate, 'Enter to continue')}: ")
                vla.gate(gate)
            action = vla.vla_status_payload()["finger_stage_action"]
            if vla.grasp_accept_available:
                if not grasp_prompted:
                    print(
                        "\n>>> Real grip verified — press Enter to accept "
                        "the current grip; Ctrl-C = STOP"
                    )
                    grasp_prompted = True
                if poll_terminal_enter():
                    vla.accept_grasp()
            else:
                grasp_prompted = False

            if action in RELEASE_STAGE_PROMPTS:
                if release_prompted != action:
                    print(
                        "\n>>> " + RELEASE_STAGE_PROMPTS[action]
                        + "; Ctrl-C = STOP"
                    )
                    release_prompted = action
                if poll_terminal_enter():
                    vla.complete_finger_stage()
            else:
                release_prompted = None
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[STOP] requested — stopping the loop")
        vla.stop()
        try:
            arm.motion_stop()
        except Exception as exc:
            print(f"[warn] arm motion-stop failed: {exc}")
        time.sleep(2.0)

    ok = vla.status not in ("error",)
    print(f"\n[done] final status: {vla.status}")
    claw.stop()
    try:
        realsense.stop()
    except Exception:
        pass
    # Hard-exit: librealsense/torch worker threads race the interpreter
    # teardown and abort with "terminate called without an active exception"
    # (seen on BOTH successful and failed runs) — skip destructors entirely.
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    sys.exit(main())
