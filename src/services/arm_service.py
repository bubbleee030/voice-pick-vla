from __future__ import annotations

import socket
import threading
import time
from typing import Any

from src.controller import ArmController
from src.runtime_config import load_taught_fallback, merge_fallback, module_enabled


class FallbackAbort(Exception):
    """Raised to abort a fallback pick->place run and trigger a safe-stop."""


def build_fallback_plan(
    object_key: str,
    fb_cfg: dict[str, Any],
    shared: dict[str, Any],
    ready: list[int],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Pure planner for the fallback pick->place sequence.

    Returns (steps, warnings). No I/O and no motion -- this is the single source
    of truth shared by ArmService.run_fallback (execution) and
    scripts/fallback_teach.py (preview), so a preview always matches what runs.

    Step kinds:
      {"kind": "move",    "label", "pose": [6], "speed": int}
      {"kind": "grip",    "label", "positions": [3] | None}   # None -> close()
      {"kind": "hold",    "label", "seconds": float}
      {"kind": "release", "label", "steps": list}             # finger release sequence

    Raises ValueError if grasp_pose is missing/invalid.
    """
    grasp_pose = fb_cfg.get("grasp_pose")
    if not isinstance(grasp_pose, list) or len(grasp_pose) != 6:
        raise ValueError(f"{object_key}: grasp_pose missing/invalid")

    grip_pos = fb_cfg.get("grasp_finger_pos")
    if not (isinstance(grip_pos, list) and len(grip_pos) == 3):
        grip_pos = None

    place_cfg = fb_cfg.get("place", {}) or {}
    place_z_mm = place_cfg.get("z_mm")
    release_steps = place_cfg.get("release_steps") or ["open"]

    place_approach = shared.get("place_approach_pose") or list(ready)
    speeds = shared.get("speeds", {})
    travel = int(speeds.get("travel", 60))
    descend = int(speeds.get("descend", 20))
    grip_hold_s = float(shared.get("settles", {}).get("grip_hold_s", 0.8))

    hover = list(grasp_pose)
    hover[2] += int(shared.get("hover_z_offset_um", 180000))
    pregrasp = list(grasp_pose)
    pregrasp[2] += int(shared.get("pregrasp_z_offset_um", 35000))
    lift = list(grasp_pose)
    lift[2] += int(shared.get("lift_z_offset_um", 200000))

    place_above = list(place_approach)
    place_down = list(place_approach)
    if place_z_mm is not None:
        place_down[2] = int(float(place_z_mm) * 1000.0)  # mm -> controller um

    warnings: list[str] = []
    if grip_pos is None:
        warnings.append("grasp_finger_pos not set -> grip will use close()")
    if list(release_steps) == ["open"]:
        warnings.append("release_steps is the placeholder ['open'] -> teach a real release (Phase 2)")
    if place_z_mm is None:
        warnings.append("place.z_mm not set -> place descends only to place_approach height")
    if not shared.get("place_approach_pose"):
        warnings.append("place_approach_pose not set -> using ready_pose as the drop approach")

    steps: list[dict[str, Any]] = [
        {"kind": "move", "label": "ready", "pose": list(ready), "speed": travel},
        {"kind": "move", "label": "hover", "pose": hover, "speed": travel},
        {"kind": "move", "label": "pregrasp", "pose": pregrasp, "speed": descend},
        {"kind": "move", "label": "grasp", "pose": list(grasp_pose), "speed": descend},
        {"kind": "grip", "label": "grip", "positions": grip_pos},
        {"kind": "hold", "label": "grip_hold", "seconds": grip_hold_s},
        {"kind": "move", "label": "lift", "pose": lift, "speed": travel},
        {"kind": "move", "label": "ready_transit", "pose": list(ready), "speed": travel},
        {"kind": "move", "label": "place_above", "pose": place_above, "speed": travel},
        {"kind": "move", "label": "place_down", "pose": place_down, "speed": descend},
        {"kind": "release", "label": "release", "steps": list(release_steps)},
        {"kind": "move", "label": "retract", "pose": place_above, "speed": travel},
        {"kind": "move", "label": "ready_end", "pose": list(ready), "speed": travel},
    ]
    return steps, warnings


class ArmService:
    def __init__(self, demo_cfg: dict[str, Any], modbus_cfg: dict[str, Any], socketio=None, gripper_service=None, validation_recorder=None):
        self.demo_cfg = demo_cfg
        self.modbus_cfg = modbus_cfg
        self.socketio = socketio
        self.gripper = gripper_service
        self.validation_recorder = validation_recorder
        self.ctrl: Any = None
        self.connected = False
        self.connection_label = ""
        self.current_pose_mm_deg: list[float] | None = None
        self._polling = False
        self._poll_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_zero_pose_warn_ts = 0.0

    def refresh_runtime_config(
        self, demo_cfg: dict[str, Any], modbus_cfg: dict[str, Any]
    ) -> None:
        """Apply next-run demo and motion tuning to this service.

        A connected controller keeps its connection/register identity, while
        motion ramps and safety limits are refreshed atomically by the
        controller. If disconnected, the fresh snapshot is used by the next
        auto-connect instead.
        """
        self.demo_cfg = demo_cfg
        self.modbus_cfg = modbus_cfg
        if self.connected and self.ctrl is not None:
            self.ctrl.refresh_motion_config(modbus_cfg)

    def _emit_log(self, level: str, msg: str):
        ts = time.strftime("%H:%M:%S")
        if self.socketio is not None:
            self.socketio.emit("arm_log", {"level": level, "message": msg, "timestamp": ts})
        if self.validation_recorder is not None:
            self.validation_recorder.log_arm_log(level, msg, ts)

    def _emit_status(self):
        if self.socketio is not None:
            self.socketio.emit(
                "arm_status",
                {"connected": self.connected, "label": self.connection_label, "error": None},
            )

    def _probe_tcp(self, host: str, port: int) -> bool:
        try:
            sock = socket.socket()
            sock.settimeout(1.5)
            sock.connect((host, port))
            sock.close()
            return True
        except Exception:
            return False

    def auto_connect(self) -> bool:
        if not module_enabled(self.demo_cfg, "arm_monitor", default=True):
            self._emit_log("WARN", "Arm module disabled")
            return False
        unit_id = int(self.demo_cfg.get("arm", {}).get("unit_id", 2))
        for conn in self.demo_cfg.get("arm", {}).get("connections", []):
            host = str(conn["host"])
            port = int(conn["port"])
            label = str(conn.get("label", f"{host}:{port}"))
            if not self._probe_tcp(host, port):
                self._emit_log("WARN", f"TCP probe failed: {label}")
                continue
            try:
                ctrl = ArmController(host=host, port=port, unit_id=unit_id, config_data=self.modbus_cfg)
                if ctrl.connect():
                    self.ctrl = ctrl
                    self.connected = True
                    self.connection_label = label
                    self._emit_log("STEP", f"Arm connected via {label}")
                    self._emit_status()
                    return True
            except Exception as exc:
                self._emit_log("WARN", f"Modbus connect failed ({label}): {exc}")
        self.connected = False
        self.connection_label = ""
        self._emit_log("ERROR", "All arm connections failed")
        self._emit_status()
        return False

    def disconnect(self):
        if self.ctrl is not None:
            try:
                self.ctrl.disconnect()
            except Exception:
                pass
        self.connected = False
        self.connection_label = ""
        self.ctrl = None
        self._emit_status()

    def start_pose_polling(self):
        if self._polling:
            return
        self._polling = True
        hz = float(self.demo_cfg.get("arm", {}).get("pose_poll_hz", 4))
        self._poll_thread = threading.Thread(target=self._pose_poll_loop, args=(hz,), daemon=True)
        self._poll_thread.start()

    def stop_pose_polling(self):
        self._polling = False

    def _warn_zero_pose(self) -> None:
        now = time.time()
        if now - self._last_zero_pose_warn_ts > 2.0:
            self._last_zero_pose_warn_ts = now
            self._emit_log(
                "WARN", "Ignoring invalid all-zero arm pose from Modbus"
            )

    def read_and_publish_pose(self) -> list[float] | None:
        """Read one measured pose and publish the same sample to every consumer.

        Auto motion pauses the independent polling thread, so its watchdog calls
        this method directly. Keeping conversion, cache updates, Socket.IO, and
        validation recording here prevents the UI and recorder from disagreeing.
        """
        if not self.connected or self.ctrl is None:
            return None
        raw = self.ctrl.read_current_pose()
        if not raw or len(raw) < 6:
            return None
        if abs(int(raw[0])) + abs(int(raw[1])) + abs(int(raw[2])) == 0:
            self._warn_zero_pose()
            return None

        mm_deg = [value / 1000.0 for value in raw]
        with self._lock:
            self.current_pose_mm_deg = mm_deg
        timestamp = time.time()
        if self.socketio is not None:
            self.socketio.emit(
                "arm_pose",
                {
                    "pose_mm_deg": mm_deg,
                    "raw": raw,
                    "timestamp": timestamp,
                },
            )
        if self.validation_recorder is not None:
            self.validation_recorder.log_arm_pose(mm_deg, timestamp)
        return mm_deg

    def _pose_poll_loop(self, hz: float):
        interval = 1.0 / max(hz, 1.0)
        while self._polling:
            if self.connected and self.ctrl is not None:
                try:
                    self.read_and_publish_pose()
                except Exception as exc:
                    self._emit_log("WARN", f"Arm pose poll failed: {exc}")
            time.sleep(interval)

    def speed_presets(self) -> dict[str, int]:
        presets = self.demo_cfg.get("speed", {}).get("presets", {})
        return {
            "low": int(presets.get("low", 25)),
            "medium": int(presets.get("medium", 60)),
            "high": int(presets.get("high", 100)),
        }

    def resolve_speed(self, preset: str | None = None, custom: int | None = None, fallback: int = 30) -> int:
        presets = self.speed_presets()
        if preset == "custom" and custom is not None:
            return max(1, min(int(custom), 100))
        if preset in presets:
            return presets[preset]
        return max(1, min(int(custom if custom is not None else fallback), 100))

    def check_safety(self, pose: list[int]) -> tuple[bool, str]:
        labels = ["x", "y", "z"]
        for limits_name in ("safety_boundary", "pose_limits"):
            limits = self.demo_cfg.get(limits_name, {})
            for index, axis in enumerate(labels):
                lo = limits.get(f"{axis}_min")
                hi = limits.get(f"{axis}_max")
                if lo is None or hi is None:
                    continue
                value = int(pose[index])
                if value < int(lo) or value > int(hi):
                    return False, f"{axis.upper()}={value} outside [{lo}, {hi}] ({limits_name})"
        return True, ""

    def ensure_connected(self) -> bool:
        if self.connected and self.ctrl is not None:
            return True
        return self.auto_connect()

    def motion_stop(self):
        if self.ensure_connected():
            self.ctrl.motion_stop()
            self._emit_log("STEP", "Motion stop sent")

    def go_ready(self) -> bool:
        if not self.ensure_connected():
            return False
        ready_pose = self.demo_cfg.get("ready_pose")
        if not ready_pose:
            self._emit_log("WARN", "ready_pose not configured")
            return False
        safe, reason = self.check_safety(ready_pose)
        if not safe:
            self._emit_log("ERROR", f"Ready safety reject: {reason}")
            return False
        ready_motion = self.demo_cfg.get("ready_motion", {})
        speed = int(ready_motion.get("speed_percent", 80))
        verify_timeout = float(ready_motion.get("verify_timeout_s", 20.0))
        start_timeout = float(
            ready_motion.get("motion_start_timeout_s", 3.0)
        )
        self.ctrl.reset_alarms()
        self.ctrl.servo_on()
        self._fb_move(
            "ready",
            ready_pose,
            speed,
            verify_timeout,
            start_timeout,
        )
        return True

    def go_home(self) -> bool:
        if not self.ensure_connected():
            return False
        route_cfg = self.demo_cfg.get("home_routing", {})
        strategy = str(route_cfg.get("default_strategy", "native_1405"))
        verify_timeout = float(route_cfg.get("verify_timeout_s", 25.0))
        if strategy == "native_1405":
            try:
                self.ctrl.reset_alarms()
                self.ctrl.servo_on()
                self.ctrl.go_home_native(wait=True, wait_seconds=verify_timeout, verify_timeout_s=verify_timeout)
                return True
            except Exception as exc:
                self._emit_log("WARN", f"Native home failed, trying staged route: {exc}")
        if route_cfg.get("allow_staged_route", True):
            for step in route_cfg.get("staged_route", []):
                pose = step.get("pose")
                if not pose:
                    continue
                safe, reason = self.check_safety(pose)
                if not safe:
                    self._emit_log("ERROR", f"Home staged route safety reject: {reason}")
                    return False
                speed = int(step.get("speed_percent", route_cfg.get("speed_percent", 20)))
                wait_s = float(step.get("wait_seconds", 6.0))
                self._emit_log("STEP", f"Home route: {step.get('label', 'step')}")
                self.ctrl.move_to(pose, speed=speed, wait_seconds=wait_s)
            return True
        self.ctrl.go_home(wait=True)
        return True

    def pick_fixed(self, object_key: str, obj_def: dict[str, Any], speed_preset: str | None = None, custom_speed: int | None = None):
        poses = obj_def.get("fixed_poses", {})
        approach = poses.get("approach")
        pick = poses.get("pick")
        if not approach or not pick:
            self._emit_log("ERROR", f"No fixed_poses defined for {object_key}")
            return
        for label, pose in [("approach", approach), ("pick", pick)]:
            safe, reason = self.check_safety(pose)
            if not safe:
                self._emit_log("ERROR", f"Safety reject ({label}): {reason}")
                return
        if not self.ensure_connected():
            return

        speed_cfg = self.demo_cfg.get("speed", {})
        fast = self.resolve_speed(speed_preset, custom_speed, fallback=int(speed_cfg.get("fast_percent", 60)))
        slow = min(fast, int(speed_cfg.get("slow_percent", 20)))
        seq_cfg = self.demo_cfg.get("pick_sequence", {})
        settle = float(seq_cfg.get("settle_wait_s", 2.0))

        try:
            self.ctrl.reset_alarms()
            self.ctrl.servo_on()
            if seq_cfg.get("home_before", True):
                self._emit_log("STEP", "Home")
                self.go_home()
            self._emit_log("STEP", f"Fast approach -> {object_key} ({fast}%)")
            self.ctrl.move_to(approach, speed=fast, wait_seconds=settle)
            self._emit_log("STEP", f"Slow descend ({slow}%)")
            self.ctrl.move_to(pick, speed=slow, wait_seconds=settle)
            if self.gripper is not None:
                self.gripper.close()
            self._emit_log("STEP", "Lift")
            self.ctrl.move_to(approach, speed=max(10, fast // 2), wait_seconds=settle)
            if seq_cfg.get("return_pose", "ready") == "ready":
                self.go_ready()
            if seq_cfg.get("home_after", False):
                self.go_home()
            self.ctrl.servo_off()
            self._emit_log("STEP", f"Pick complete: {object_key}")
        except Exception as exc:
            self._emit_log("ERROR", f"Pick failed: {exc}")

    # ------------------------------------------------------------------
    # Model-free fallback pick -> place
    # ------------------------------------------------------------------
    def run_fallback(self, object_key: str, fb_cfg: dict[str, Any]) -> bool:
        """Deterministic pick->place that never touches cameras/model.

        Sequence: ready -> hover -> pregrasp -> grasp(grip) -> lift -> ready
                  -> place_approach -> descend(place z) -> release -> retract -> ready.
        Stability guards: pose-poller paused (exclusive Modbus), per-step in-position
        watchdog, explicit STEP logs, gripper retries, and stop+hold on any failure.
        """
        shared = dict(self.demo_cfg.get("fallback", {}))
        ready = self.demo_cfg.get("ready_pose")
        if not ready:
            self._emit_log("ERROR", "[fallback] ready_pose not configured")
            return False

        # Merge machine-taught overrides (config/fallback_taught.yaml) over committed seeds.
        # Loaded per-run so re-teaching takes effect on the next click without a restart.
        taught = load_taught_fallback()
        if taught:
            t_obj = taught.get("fallback", {}).get(object_key)
            if isinstance(t_obj, dict):
                fb_cfg = merge_fallback(fb_cfg, t_obj)
                self._emit_log("STEP", f"[fallback] applied taught overrides for {object_key}")
            t_shared = taught.get("shared")
            if isinstance(t_shared, dict):
                shared = merge_fallback(shared, t_shared)

        # Build the deterministic plan (single source of truth; also used by the
        # CLI preview). Pose synthesis + validation live in build_fallback_plan.
        try:
            plan, warnings = build_fallback_plan(object_key, fb_cfg, shared, ready)
        except ValueError as exc:
            self._emit_log("ERROR", f"[fallback] {exc}")
            return False
        for warning in warnings:
            self._emit_log("WARN", f"[fallback] {warning}")

        wd = shared.get("watchdog", {})
        step_timeout = float(wd.get("per_step_timeout_s", 25.0))
        start_timeout = float(wd.get("motion_start_timeout_s", 3.0))
        gripper_retries = int(wd.get("gripper_retries", 2))
        grip_hold_default = float(shared.get("settles", {}).get("grip_hold_s", 0.8))

        # Pre-flight safety: reject the whole run if any move pose is out of bounds.
        for step in plan:
            if step["kind"] != "move":
                continue
            safe, reason = self.check_safety(step["pose"])
            if not safe:
                self._emit_log("ERROR", f"[fallback] safety reject {step['label']}: {reason}")
                return False

        if not self.ensure_connected():
            self._emit_log("ERROR", "[fallback] arm not connected")
            return False
        if self.gripper is None or not self.gripper.active_base_url:
            self._emit_log("ERROR", "[fallback] gripper not reachable; aborting (needed for grip/release)")
            return False

        # Exclusive Modbus: stop the pose-poller so it can't contend for the lock.
        poll_was_running = self._polling
        self.stop_pose_polling()
        time.sleep(0.15)  # let an in-flight poll read finish

        self._emit_log("STEP", f"[fallback] START {object_key}")
        try:
            self.ctrl.reset_alarms()
            self.ctrl.servo_on()
            for step in plan:
                kind = step["kind"]
                if kind == "move":
                    self._fb_move(step["label"], step["pose"], step["speed"], step_timeout, start_timeout)
                elif kind == "grip":
                    self._fb_grip(step["positions"], gripper_retries)
                elif kind == "hold":
                    time.sleep(float(step.get("seconds", grip_hold_default)))
                elif kind == "release":
                    self._fb_release(step["steps"], gripper_retries)
            self.ctrl.servo_off()
            self._emit_log("STEP", f"[fallback] COMPLETE {object_key}")
            self._emit_progress(object_key, "complete")
            return True
        except FallbackAbort as exc:
            self._fb_safe_stop(str(exc))
            return False
        except Exception as exc:  # noqa: BLE001 - any failure must safe-stop, never die silently
            self._fb_safe_stop(f"unexpected error: {exc}")
            return False
        finally:
            if poll_was_running:
                self.start_pose_polling()

    def _emit_progress(self, object_key: str, phase: str) -> None:
        if self.socketio is not None:
            self.socketio.emit("fallback_progress", {"object": object_key, "phase": phase})

    def _fb_read_flag(self, flag_reg: int | None) -> int | None:
        if flag_reg is None:
            return None
        try:
            return self.ctrl.read_register(flag_reg)
        except Exception:
            return None

    def _fb_move(
        self,
        label: str,
        pose: list[int],
        speed: int,
        step_timeout: float,
        start_timeout: float,
    ) -> dict[str, Any] | None:
        self._emit_log("STEP", f"[fallback] move {label} @ {speed}%")
        try:
            # wait=False: issue the move, then run our own in-position watchdog.
            motion_profile = self.ctrl.move_to(
                pose, speed=speed, wait=False
            )
        except Exception as exc:
            raise FallbackAbort(f"move '{label}' command failed: {exc}") from exc

        pose_warning_emitted = False

        def publish_watchdog_pose() -> None:
            nonlocal pose_warning_emitted
            try:
                self.read_and_publish_pose()
            except Exception as exc:
                if not pose_warning_emitted:
                    pose_warning_emitted = True
                    self._emit_log(
                        "WARN",
                        f"[fallback] live pose unavailable during {label}: {exc}",
                    )

        flag_reg = self.ctrl.regs.get("in_position_flag")
        if flag_reg is None:
            # No in-position flag available: fall back to a bounded fixed settle.
            deadline = time.time() + min(step_timeout, 2.5)
            while time.time() < deadline:
                publish_watchdog_pose()
                time.sleep(min(0.2, max(0.0, deadline - time.time())))
            return motion_profile

        start = time.time()
        # Phase 1: wait for motion to begin (flag clears to 0), bounded.
        while time.time() - start < start_timeout:
            publish_watchdog_pose()
            if self._fb_read_flag(flag_reg) == 0:
                break
            time.sleep(0.1)
        # Phase 2: wait for arrival (flag == 1) within the watchdog window.
        while time.time() - start < step_timeout:
            publish_watchdog_pose()
            if self._fb_read_flag(flag_reg) == 1:
                self._emit_log("STEP", f"[fallback] {label} in-position ({time.time() - start:.1f}s)")
                return motion_profile
            time.sleep(0.2)
        raise FallbackAbort(f"watchdog: '{label}' did not reach in-position within {step_timeout:.0f}s")

    def _fb_grip(self, grip_pos: Any, retries: int) -> None:
        use_pos = isinstance(grip_pos, list) and len(grip_pos) == 3
        label = f"set_position {grip_pos}" if use_pos else "close"
        self._emit_log("STEP", f"[fallback] grip ({label})")
        for attempt in range(max(1, retries)):
            ok = self.gripper.set_position([int(v) for v in grip_pos]) if use_pos else self.gripper.close()
            if ok:
                return
            time.sleep(0.3)
        raise FallbackAbort(f"grip failed: {self.gripper.last_error or 'no response'}")

    def _fb_release(self, release_steps: list[Any], retries: int) -> None:
        self._emit_log("STEP", f"[fallback] release ({len(release_steps)} step(s))")
        for idx, step in enumerate(release_steps):
            is_open = step == "open" or (isinstance(step, dict) and step.get("open"))
            pos = step.get("pos") if isinstance(step, dict) else None
            delay = float(step.get("delay_s", 0.3)) if isinstance(step, dict) else 0.3

            ok = False
            for attempt in range(max(1, retries)):
                if is_open:
                    ok = self.gripper.open()
                    delay = max(delay, 0.5)
                elif isinstance(pos, list) and len(pos) == 3:
                    ok = self.gripper.set_position([int(v) for v in pos], mode="stepped")
                else:
                    self._emit_log("WARN", f"[fallback] release step {idx} malformed, skipping: {step}")
                    ok = True
                if ok:
                    break
                time.sleep(0.3)
            # A failed release must surface (object never dropped), not be ignored.
            if not ok:
                raise FallbackAbort(f"release step {idx} failed: {self.gripper.last_error or 'no response'}")
            time.sleep(delay)

    def _fb_safe_stop(self, reason: str) -> None:
        self._emit_log("ERROR", f"[fallback] ABORT: {reason}")
        try:
            self.ctrl.motion_stop()
        except Exception as exc:  # noqa: BLE001
            self._emit_log("ERROR", f"[fallback] motion_stop also failed: {exc}")
        # Stop + hold: keep servo ON, do NOT release the gripper. Operator recovers.
        self._emit_log("ERROR", "[fallback] arm holding position (servo ON, gripper unchanged). Operator intervention required.")

    def _positions_from_waypoint(self, wp: dict[str, Any]) -> list[int] | None:
        direct = wp.get("gripper_pos")
        if isinstance(direct, list) and len(direct) == 3:
            try:
                return [int(v) for v in direct]
            except (TypeError, ValueError):
                return None
        matched = wp.get("matched_external")
        if isinstance(matched, dict):
            row = matched.get("row")
            if isinstance(row, dict):
                values = [row.get("pos1"), row.get("pos2"), row.get("pos3")]
                if all(v is not None for v in values):
                    try:
                        return [int(float(v)) for v in values]
                    except (TypeError, ValueError):
                        return None
        return None

    def _replay_gripper_timeline(self, timeline: list[dict[str, Any]], stop_event: threading.Event):
        if self.gripper is None:
            return
        replay_start = time.time()
        last_positions = None
        for sample in timeline:
            if stop_event.is_set():
                return
            positions = sample.get("positions")
            if not isinstance(positions, list) or len(positions) != 3:
                continue
            try:
                positions = [int(v) for v in positions]
                target_sec = max(0.0, float(sample.get("t_ms", 0)) / 1000.0)
            except (TypeError, ValueError):
                continue
            wait_sec = target_sec - (time.time() - replay_start)
            if wait_sec > 0:
                time.sleep(wait_sec)
            if positions == last_positions:
                continue
            self.gripper.set_position(positions)
            last_positions = positions

    def replay_recording(self, recording: dict[str, Any], replay_mode: str | None = None):
        waypoints = recording.get("waypoints", [])
        if not waypoints:
            self._emit_log("ERROR", "Recording has no waypoints")
            return
        mode = replay_mode or self.demo_cfg.get("teach", {}).get("default_replay_mode", "raw")
        if mode == "phase_axis_split":
            if len(waypoints) > 1:
                self._emit_log("WARN", "phase_axis_split currently falls back to raw for multi-waypoint recordings")
                mode = "raw"
            else:
                return self._replay_phase_axis_split(recording)
        return self._replay_raw(recording)

    def _replay_raw(self, recording: dict[str, Any]):
        waypoints = recording.get("waypoints", [])
        for wp in waypoints:
            safe, reason = self.check_safety(wp["pose"])
            if not safe:
                self._emit_log("ERROR", f"Safety reject in recording: {reason}")
                return
        if not self.ensure_connected():
            return
        stop_event = threading.Event()
        timeline_thread = None
        external_timeline = recording.get("external_timeline", [])
        if isinstance(external_timeline, list) and external_timeline and self.gripper is not None:
            timeline_thread = threading.Thread(target=self._replay_gripper_timeline, args=(external_timeline, stop_event), daemon=True)
            timeline_thread.start()
        try:
            self.ctrl.reset_alarms()
            self.ctrl.servo_on()
            for index, wp in enumerate(waypoints):
                self.ctrl.move_to(wp["pose"], speed=int(wp.get("speed", 30)), wait_seconds=2.0)
                if timeline_thread is None and self.gripper is not None:
                    positions = self._positions_from_waypoint(wp)
                    if positions is not None:
                        self.gripper.set_position(positions)
                    elif wp.get("gripper") == "close":
                        self.gripper.close()
                    elif wp.get("gripper") == "open":
                        self.gripper.open()
                if self.socketio is not None:
                    self.socketio.emit(
                        "pick_progress",
                        {"step": index + 1, "total": len(waypoints), "name": f"waypoint_{index + 1}"},
                    )
            self.ctrl.servo_off()
            self._emit_log("STEP", "Replay complete")
        except Exception as exc:
            self._emit_log("ERROR", f"Replay failed: {exc}")
        finally:
            stop_event.set()
            if timeline_thread is not None:
                timeline_thread.join(timeout=1.0)

    def _replay_phase_axis_split(self, recording: dict[str, Any]):
        if not self.ensure_connected():
            return
        waypoint = recording["waypoints"][-1]
        target = list(waypoint["pose"])
        safe, reason = self.check_safety(target)
        if not safe:
            self._emit_log("ERROR", f"Safety reject in phase replay: {reason}")
            return
        template = self.demo_cfg.get("teach", {}).get("phase_templates", {}).get("default", {})
        hover = list(target)
        hover[2] += int(template.get("hover_z_offset_um", 180000))
        pregrasp = list(target)
        pregrasp[2] += int(template.get("pregrasp_z_offset_um", 35000))
        lift = list(target)
        lift[2] += int(template.get("lift_z_offset_um", 200000))
        try:
            self.ctrl.reset_alarms()
            self.ctrl.servo_on()
            if self.demo_cfg.get("teach", {}).get("phase_generation", {}).get("use_ready_pose", True):
                self.go_ready()
            for label, pose, speed in [
                ("hover", hover, int(template.get("fast_percent", 60))),
                ("pregrasp", pregrasp, int(template.get("pregrasp_percent", 25))),
                ("grasp", target, int(template.get("grasp_percent", 25))),
            ]:
                safe, reason = self.check_safety(pose)
                if not safe:
                    self._emit_log("ERROR", f"Phase replay safety reject ({label}): {reason}")
                    return
                self._emit_log("STEP", f"Phase replay: {label}")
                self.ctrl.move_to(pose, speed=speed, wait_seconds=float(template.get("settle_s", 0.4)))
            if self.gripper is not None:
                positions = self._positions_from_waypoint(waypoint)
                if positions is not None:
                    self.gripper.set_position(positions, mode=template.get("gripper_position_move_mode", "stepped"))
                else:
                    self.gripper.close()
            time.sleep(float(template.get("grip_hold_s", 0.8)))
            self.ctrl.move_to(lift, speed=int(template.get("lift_percent", 60)), wait_seconds=float(template.get("settle_s", 0.4)))
            if template.get("return_pose", "ready") == "ready":
                self.go_ready()
            self.ctrl.servo_off()
            self._emit_log("STEP", "Phase replay complete")
        except Exception as exc:
            self._emit_log("ERROR", f"Phase replay failed: {exc}")
