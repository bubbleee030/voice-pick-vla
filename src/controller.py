"""
Modbus arm controller for the robot arm.

This version aligns with the newer VLAtest register map and motion flow:
- group error reset
- native home (1405)
- motion stop
- robot speed override
- optional acceleration / posture / frame writes
- readback verification for move mode and speed
"""

from __future__ import annotations

import socket
import threading
import time

from src.utils import (
    int_to_register,
    int32_to_registers,
    load_config,
    pose_to_mm_deg,
    read_current_pose_from_registers,
    registers_to_int32,
)

try:
    from pyModbusTCP.client import ModbusClient
except ImportError:
    ModbusClient = None


class ArmController:
    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        unit_id: int | None = None,
        config_path: str | None = None,
        config_data: dict | None = None,
    ):
        if ModbusClient is None:
            raise RuntimeError("pyModbusTCP not installed. Run: pip install pyModbusTCP")

        if config_data is not None:
            self.cfg = config_data
        else:
            try:
                self.cfg = load_config(config_path or "modbus_config.yaml")
            except Exception:
                print("[WARN] modbus_config.yaml not found, using internal defaults.")
                self.cfg = self._get_default_config()

        conn = self.cfg.get("connection", {})
        self.host = host or conn.get("host", "127.0.0.1")
        self.port = int(port or conn.get("port", 502))
        self.unit_id = int(unit_id or conn.get("unit_id", 2))
        self.timeout = float(conn.get("timeout_s", 1.0))
        self.retries = int(conn.get("connect_retries", 3))
        self.retry_delay = float(conn.get("connect_retry_delay_s", 2.0))

        self.regs = self.cfg["registers"]
        self.cmds = self.cfg["commands"]
        self.motion = self.cfg["motion"]
        self.safety = self.cfg["safety"]

        self.client = None
        self._connected = False
        self._io_lock = threading.RLock()
        self.last_motion_profile: dict[str, object] | None = None

    def _get_default_config(self) -> dict:
        return {
            "connection": {"host": "127.0.0.1", "port": 502, "unit_id": 2},
            "registers": {
                "servo_on_off": 16,
                "alarm_reset_1": 38,
                "alarm_reset_2": 39,
                "alarm_reset_3": 32,
                "group_error_reset": 384,
                "current_pose_base": 240,
                "robot_speed_override": 582,
                "motion_command": 768,
                "acceleration": 778,
                "deceleration": 780,
                "in_position_flag": 799,
                "speed_percent": 804,
                "target_pose_base": 816,
                "user_frame": 828,
                "target_posture": 829,
                "move_mode": 830,
                "tool_frame": 831,
            },
            "commands": {"home": 1405, "motion_stop": 1000, "p2p_move": 301, "linear_move": 302},
            "motion": {
                "use_linear_move": False,
                "move_mode": 3,
                "default_speed_percent": 20,
                "settle_wait_s": 2.0,
                "servo_settle_s": 1.0,
                "verify_speed_write": True,
                "set_robot_speed_override": True,
                "robot_speed_override_percent": 100,
                "set_acceleration": False,
                "acceleration_raw": 0,
                "set_deceleration": False,
                "deceleration_raw": 0,
                "set_target_posture": False,
                "target_posture": 0,
                "set_frames": False,
                "user_frame": 0,
                "tool_frame": 0,
                "protocol_max_speed_percent": 100,
            },
            "home_pose": [444000, 0, 744000, 0, -89999, 179999],
            "safety": {"max_speed_percent": 100},
        }

    def probe(self) -> bool:
        try:
            sock = socket.socket()
            sock.settimeout(2.0)
            sock.connect((self.host, self.port))
            sock.close()
            print(f"[ARM] TCP probe {self.host}:{self.port} OK")
            return True
        except Exception as exc:
            print(f"[ARM] TCP probe {self.host}:{self.port} FAIL: {exc}")
            return False

    def connect(self) -> bool:
        with self._io_lock:
            print(f"[ARM] Connecting to {self.host}:{self.port} (unit={self.unit_id})...")
            if not self.probe():
                return False

            self.client = ModbusClient(host=self.host, port=self.port, unit_id=self.unit_id, auto_open=True)
            self.client.timeout = self.timeout

            for attempt in range(1, self.retries + 1):
                if self.client.open():
                    self._connected = True
                    print(f"[ARM] Connected (attempt {attempt}).")
                    return True
                print(f"[ARM] Connect attempt {attempt}/{self.retries} failed, retrying...")
                time.sleep(self.retry_delay)

            print("[ARM] Failed to connect after all retries.")
            return False

    def disconnect(self):
        with self._io_lock:
            if self.client:
                self.client.close()
                self._connected = False
                print("[ARM] Disconnected.")

    def read_register(self, reg: int) -> int | None:
        with self._io_lock:
            if self.client is None:
                return None
            out = self.client.read_holding_registers(reg, 1)
            if out is None or len(out) != 1:
                return None
            return int(out[0])

    def _read_int32_register(self, reg: int | None) -> int | None:
        if reg is None:
            return None
        with self._io_lock:
            if self.client is None:
                return None
            out = self.client.read_holding_registers(reg, 2)
            if out is None or len(out) != 2:
                return None
            return registers_to_int32([int(out[0]), int(out[1])])

    def refresh_motion_config(self, config_data: dict) -> None:
        """Refresh live motion tuning without changing the active connection.

        Register maps, commands, and network settings describe the connected
        device and are intentionally connection-scoped. Motion ramps and safety
        limits are run-scoped, so operators may edit those for the next run.
        """
        motion = config_data.get("motion")
        safety = config_data.get("safety")
        if not isinstance(motion, dict) or not isinstance(safety, dict):
            raise ValueError(
                "runtime Modbus config requires motion and safety mappings"
            )
        with self._io_lock:
            self.motion = dict(motion)
            self.safety = dict(safety)
            self.cfg["motion"] = dict(motion)
            self.cfg["safety"] = dict(safety)

    def reset_alarms(self):
        with self._io_lock:
            print("[ARM] Resetting alarms...")
            for reg in [
                self.regs["alarm_reset_1"],
                self.regs["alarm_reset_2"],
                self.regs["alarm_reset_3"],
            ]:
                self.client.write_multiple_registers(reg, int_to_register(1))
                self.client.write_multiple_registers(reg, int_to_register(256))
            group_reg = self.regs.get("group_error_reset")
            if group_reg is not None:
                self.client.write_multiple_registers(group_reg, int_to_register(1))
                self.client.write_multiple_registers(group_reg, int_to_register(0))

    def servo_on(self):
        with self._io_lock:
            print("[ARM] Servo ON")
            self.client.write_single_register(self.regs["servo_on_off"], 1)
            # Write the global speed override HERE (once per session), never in
            # move_to: the controller applies 0x0246 asynchronously and couples
            # it with 0x0324 (during a GO, 0x0246 always reads speed% x 10).
            # A per-move override write raced the GO command — when it landed
            # after the GO started it re-sped the move to the override value
            # (audit 2026-07-13: grasp commanded 20%, verified 20% by readback,
            # physically ran ~100%; 2 of 9 moves affected).
            override_reg = self.regs.get("robot_speed_override")
            if self.motion.get("set_robot_speed_override", False) and override_reg is not None:
                override_percent = float(self.motion.get("robot_speed_override_percent", 100))
                override_raw = max(1, min(int(round(override_percent * 10)), 1000))
                self.client.write_single_register(override_reg, override_raw)
        time.sleep(float(self.motion.get("servo_settle_s", 1.0)))

    def servo_off(self):
        with self._io_lock:
            print("[ARM] Servo OFF")
            # Vendor flow uses 2 for OFF in the newer path.
            self.client.write_single_register(self.regs["servo_on_off"], 2)

    def motion_stop(self):
        with self._io_lock:
            stop_cmd = int(self.cmds.get("motion_stop", 1000))
            print(f"[ARM] Motion STOP ({stop_cmd})")
            self.client.write_single_register(self.regs["motion_command"], stop_cmd)

    def read_current_pose(self) -> list[int]:
        with self._io_lock:
            return read_current_pose_from_registers(self.client, self.regs["current_pose_base"])

    def read_current_pose_mm_deg(self) -> list[float]:
        return pose_to_mm_deg(self.read_current_pose())

    def write_target_pose(self, pose: list[int]):
        with self._io_lock:
            base = self.regs["target_pose_base"]
            for i, value in enumerate(pose):
                self.client.write_multiple_registers(base + 2 * i, int32_to_registers(int(value)))

    def move_to(
        self,
        pose: list[int],
        speed: int | None = None,
        wait: bool = True,
        wait_seconds: float | None = None,
        linear: bool | None = None,
    ) -> dict[str, object]:
        requested_speed = int(speed or self.motion["default_speed_percent"])
        protocol_max = int(self.motion.get("protocol_max_speed_percent", 100))
        safety_max = int(self.safety.get("max_speed_percent", 100))
        max_speed = max(1, min(protocol_max, safety_max))
        clamped_speed = max(1, min(requested_speed, max_speed))
        wait_s = float(wait_seconds or self.motion.get("settle_wait_s", 2.0))
        # MovL (302) runs a straight Cartesian line at a consistent tool speed;
        # MovP (301) is joint-interpolated, so the same speed% gives different
        # Cartesian mm/s depending on pose. MovL can alarm near singularities.
        if linear is None:
            linear = bool(self.motion.get("use_linear_move", False))
        move_cmd = int(self.cmds.get("linear_move", 302)) if linear else int(self.cmds["p2p_move"])
        move_name = "MovL" if linear else "MovP"

        with self._io_lock:
            if clamped_speed != requested_speed:
                print(
                    f"[ARM] Move to {pose} @ {clamped_speed}% speed [{move_name}] "
                    f"(requested {requested_speed}%, max {max_speed}%)"
                )
            else:
                print(f"[ARM] Move to {pose} @ {clamped_speed}% speed [{move_name}]")

            self.write_target_pose(pose)

            # NOTE: the 0x0246 speed override is deliberately NOT written here —
            # it is coupled to 0x0324 and applied asynchronously, so a per-move
            # write races the GO command and intermittently forced moves to run
            # at the override value (100%) instead of the commanded speed.
            # It is written once in servo_on() instead.
            override_reg = self.regs.get("robot_speed_override")   # readback only

            # ACC (0x030A) / DEC (0x030C) are DOUBLE-WORD registers (unit:
            # 0.01 um/ms^2 for Cartesian; raw 100 = 1000 mm/s^2). The arm ships
            # with acc=100/dec=10 — the 10x-gentler stop is the long slow-down
            # felt at the end of every move; raise DEC to sharpen it.
            acc_reg = self.regs.get("acceleration")
            if self.motion.get("set_acceleration", False) and acc_reg is not None:
                acc_raw = max(0, int(self.motion.get("acceleration_raw", 0)))
                self.client.write_multiple_registers(
                    acc_reg, [acc_raw & 0xFFFF, (acc_raw >> 16) & 0xFFFF])
            dec_reg = self.regs.get("deceleration")
            if self.motion.get("set_deceleration", False) and dec_reg is not None:
                dec_raw = max(0, int(self.motion.get("deceleration_raw", 0)))
                self.client.write_multiple_registers(
                    dec_reg, [dec_raw & 0xFFFF, (dec_raw >> 16) & 0xFFFF])

            user_frame_reg = self.regs.get("user_frame")
            tool_frame_reg = self.regs.get("tool_frame")
            if self.motion.get("set_frames", False):
                if user_frame_reg is not None:
                    self.client.write_single_register(user_frame_reg, int(self.motion.get("user_frame", 0)))
                if tool_frame_reg is not None:
                    self.client.write_single_register(tool_frame_reg, int(self.motion.get("tool_frame", 0)))

            posture_reg = self.regs.get("target_posture")
            if self.motion.get("set_target_posture", False) and posture_reg is not None:
                self.client.write_single_register(posture_reg, int(self.motion.get("target_posture", 0)))

            verify_speed = bool(
                self.motion.get("verify_speed_write", True)
            )
            verify_attempts = max(
                1, int(self.motion.get("speed_verify_attempts", 3))
            )
            verify_retry_s = max(
                0.0,
                float(self.motion.get("speed_verify_retry_s", 0.05)),
            )
            actual_speed = actual_mode = None
            for attempt in range(verify_attempts):
                self.client.write_single_register(
                    self.regs["move_mode"], int(self.motion["move_mode"])
                )
                self.client.write_single_register(
                    self.regs["speed_percent"], clamped_speed
                )
                actual_speed = self.read_register(self.regs["speed_percent"])
                actual_mode = self.read_register(self.regs["move_mode"])
                if not verify_speed or actual_speed == clamped_speed:
                    break
                if attempt + 1 < verify_attempts and verify_retry_s > 0:
                    time.sleep(verify_retry_s)

            actual_override = (
                self.read_register(override_reg)
                if override_reg is not None
                else None
            )
            actual_acc = self._read_int32_register(acc_reg)
            actual_dec = self._read_int32_register(dec_reg)
            actual_posture = (
                self.read_register(posture_reg)
                if posture_reg is not None
                else None
            )
            actual_user_frame = (
                self.read_register(user_frame_reg)
                if user_frame_reg is not None
                else None
            )
            actual_tool_frame = (
                self.read_register(tool_frame_reg)
                if tool_frame_reg is not None
                else None
            )
            speed_confirmed = actual_speed == clamped_speed
            profile: dict[str, object] = {
                "requested_speed_percent": requested_speed,
                "applied_speed_percent": actual_speed,
                "move_type": move_name,
                "acceleration_raw": actual_acc,
                "deceleration_raw": actual_dec,
                "confirmed": speed_confirmed,
            }
            self.last_motion_profile = dict(profile)

            if verify_speed and not speed_confirmed:
                print(
                    "[ARM-WARN] Motion config rejected before GO: "
                    f"mode wrote {self.motion['move_mode']} read {actual_mode}; "
                    f"speed wrote {clamped_speed}% read {actual_speed}%; "
                    f"override read {actual_override}; acc read {actual_acc}; "
                    f"dec read {actual_dec}"
                )
                raise RuntimeError(
                    "speed register rejected command before GO: "
                    f"requested={requested_speed}% "
                    f"clamped={clamped_speed}% readback={actual_speed}"
                )

            if actual_mode != int(self.motion["move_mode"]):
                print(
                    "[ARM-WARN] Move mode readback mismatch: "
                    f"wrote {self.motion['move_mode']} read {actual_mode}"
                )
            override_msg = (
                f" override={actual_override / 10.0:.1f}%"
                if actual_override is not None
                else ""
            )
            print(
                f"[ARM] Motion config OK: mode={actual_mode} "
                f"speed={actual_speed}%{override_msg} acc={actual_acc} "
                f"dec={actual_dec} posture={actual_posture} "
                f"user_frame={actual_user_frame} tool_frame={actual_tool_frame}"
            )

            self.client.write_single_register(self.regs["motion_command"], move_cmd)

        if wait:
            self._wait_with_live_output("move", wait_s)
        return dict(profile)

    def go_home(self, wait: bool = True):
        home = self.cfg.get("home_pose", [444000, 0, 744000, 0, -89999, 179999])
        print(f"[ARM] Going home: {home}")
        self.move_to(home, speed=20, wait=wait, wait_seconds=5.0)

    def go_home_native(self, wait: bool = True, wait_seconds: float = 20.0, verify_timeout_s: float | None = None):
        with self._io_lock:
            home_cmd = int(self.cmds.get("home", 1405))
            print(f"[ARM] Going home via native command: {home_cmd}")
            self.client.write_single_register(self.regs["motion_command"], home_cmd)
        if wait:
            self._wait_for_in_position(
                timeout_s=verify_timeout_s or wait_seconds,
                fallback_wait_s=wait_seconds,
                label="home_native",
            )

    def wait_until_in_position(
        self,
        timeout_s: float = 20.0,
        fallback_wait_s: float | None = None,
        label: str = "move",
    ):
        self._wait_for_in_position(
            timeout_s=timeout_s,
            fallback_wait_s=fallback_wait_s or timeout_s,
            label=label,
        )

    def pick_at(
        self,
        robot_xyz_mm: list[float],
        orientation: list[int] | None = None,
        approach_height_mm: float = 80,
        lift_height_mm: float = 100,
        speed: int = 20,
    ):
        objects_cfg = load_config("objects.yaml")
        defaults = objects_cfg.get("pick_defaults", {})
        approach_h = approach_height_mm or defaults.get("approach_height_mm", 80)
        lift_h = lift_height_mm or defaults.get("lift_height_mm", 100)
        if orientation is None:
            orientation = [0, -89999, 179999]

        x = int(robot_xyz_mm[0] * 1000)
        y = int(robot_xyz_mm[1] * 1000)
        z = int(robot_xyz_mm[2] * 1000)
        approach_z = int((robot_xyz_mm[2] + approach_h) * 1000)
        lift_z = int((robot_xyz_mm[2] + lift_h) * 1000)

        for step_name, pose in [
            ("approach", [x, y, approach_z] + orientation),
            ("descend", [x, y, z] + orientation),
            ("lift", [x, y, lift_z] + orientation),
        ]:
            print(f"[ARM] Pick step: {step_name}")
            self.move_to(pose, speed=speed)

    def _wait_with_live_output(self, label: str, duration: float):
        start = time.time()
        interval = 0.5
        while time.time() - start < duration:
            elapsed = time.time() - start
            pose = self.read_current_pose_mm_deg()
            print(f"  [{label}] {elapsed:.1f}/{duration:.1f}s pose(mm/deg)={[round(v, 1) for v in pose]}")
            time.sleep(interval)

    def _wait_for_in_position(self, timeout_s: float, fallback_wait_s: float, label: str = "move"):
        flag_reg = self.regs.get("in_position_flag")
        if flag_reg is None:
            self._wait_with_live_output(label, fallback_wait_s)
            return

        start = time.time()
        interval = 0.5
        while time.time() - start < max(timeout_s, interval):
            elapsed = time.time() - start
            flag = self.read_register(flag_reg)
            pose = self.read_current_pose_mm_deg()
            print(
                f"  [{label}] {elapsed:.1f}/{timeout_s:.1f}s "
                f"in_position={flag} pose(mm/deg)={[round(v, 1) for v in pose]}"
            )
            if flag == 1:
                return
            time.sleep(interval)

        print(f"[ARM-WARN] in_position timeout after {timeout_s:.1f}s; fallback wait {fallback_wait_s:.1f}s")
        self._wait_with_live_output(f"{label}_fallback", fallback_wait_s)
