from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.services import arm_service
from src.controller import ArmController


TARGET = [490000, 0, 170000, 179999, 0, 0]


class RecordingSocket:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class PoseWatchdogController:
    def __init__(
        self,
        *,
        poses: list[list[int]],
        flags: list[int] | None = None,
    ) -> None:
        self.regs = {"in_position_flag": 0x031F}
        self._poses = iter(poses)
        self._last_pose = list(poses[-1])
        self._flags = iter(flags or [1])
        self._last_flag = (flags or [1])[-1]
        self.moves: list[tuple[list[int], int, bool]] = []

    def reset_alarms(self) -> None:
        return None

    def servo_on(self) -> None:
        return None

    def read_current_pose(self) -> list[int]:
        self._last_pose = list(next(self._poses, self._last_pose))
        return list(self._last_pose)

    def read_register(self, _register: int) -> int:
        self._last_flag = int(next(self._flags, self._last_flag))
        return self._last_flag

    def move_to(self, pose: list[int], speed: int, wait: bool = False):
        self.moves.append((list(pose), int(speed), bool(wait)))
        return {"applied_speed_percent": int(speed), "confirmed": True}

    def wait_until_in_position(self, **_kwargs) -> None:
        return None


class RecordingModbusClient:
    def __init__(self, *, speed_readback: int) -> None:
        self.speed_readback = int(speed_readback)
        self.single_writes: list[tuple[int, int]] = []
        self.multiple_writes: list[tuple[int, list[int]]] = []
        self.registers: dict[int, int] = {}

    def write_single_register(self, register: int, value: int) -> bool:
        self.single_writes.append((int(register), int(value)))
        self.registers[int(register)] = int(value)
        return True

    def write_multiple_registers(
        self, register: int, values: list[int]
    ) -> bool:
        normalized = [int(value) for value in values]
        self.multiple_writes.append((int(register), normalized))
        for offset, value in enumerate(normalized):
            self.registers[int(register) + offset] = value
        return True

    def read_holding_registers(
        self, register: int, count: int
    ) -> list[int]:
        if int(register) == 0x0324:
            return [self.speed_readback]
        return [
            self.registers.get(int(register) + offset, 0)
            for offset in range(int(count))
        ]


def make_controller(
    *, speed_readback: int, acc: int = 200, dec: int = 50
) -> tuple[ArmController, RecordingModbusClient]:
    ctrl = ArmController.__new__(ArmController)
    ctrl.regs = {
        "motion_command": 0x0300,
        "robot_speed_override": 0x0246,
        "acceleration": 0x030A,
        "deceleration": 0x030C,
        "speed_percent": 0x0324,
        "target_pose_base": 0x0330,
        "user_frame": 0x033C,
        "target_posture": 0x033D,
        "move_mode": 0x033E,
        "tool_frame": 0x033F,
    }
    ctrl.cmds = {"p2p_move": 301, "linear_move": 302}
    ctrl.motion = {
        "use_linear_move": False,
        "move_mode": 3,
        "default_speed_percent": 20,
        "settle_wait_s": 2.0,
        "verify_speed_write": True,
        "speed_verify_attempts": 3,
        "speed_verify_retry_s": 0.0,
        "set_acceleration": True,
        "acceleration_raw": int(acc),
        "set_deceleration": True,
        "deceleration_raw": int(dec),
        "set_target_posture": False,
        "set_frames": False,
        "protocol_max_speed_percent": 100,
    }
    ctrl.safety = {"max_speed_percent": 100}
    ctrl.cfg = {"motion": dict(ctrl.motion), "safety": dict(ctrl.safety)}
    ctrl._io_lock = threading.RLock()
    ctrl.last_motion_profile = None
    client = RecordingModbusClient(speed_readback=speed_readback)
    ctrl.client = client
    return ctrl, client


def make_arm_service(
    *,
    poses: list[list[int]],
    flags: list[int] | None = None,
) -> tuple[arm_service.ArmService, RecordingSocket]:
    socket = RecordingSocket()
    service = arm_service.ArmService({}, {}, socketio=socket)
    service.ctrl = PoseWatchdogController(poses=poses, flags=flags)
    service.connected = True
    return service, socket


class LiveArmPoseTests(unittest.TestCase):
    def test_read_and_publish_pose_updates_cache_and_socket(self) -> None:
        service, socket = make_arm_service(
            poses=[[490000, 0, 126000, 179999, 0, 0]]
        )

        pose = service.read_and_publish_pose()

        self.assertEqual(pose[:3], [490.0, 0.0, 126.0])
        self.assertEqual(service.current_pose_mm_deg, pose)
        self.assertEqual(socket.events[-1][0], "arm_pose")
        self.assertEqual(
            socket.events[-1][1]["pose_mm_deg"][:3], pose[:3]
        )

    def test_fb_move_publishes_pose_while_normal_poller_is_stopped(self) -> None:
        service, socket = make_arm_service(
            flags=[0, 0, 1],
            poses=[
                [490000, 0, 126000, 179999, 0, 0],
                [490000, 0, 150000, 179999, 0, 0],
                [490000, 0, 170000, 179999, 0, 0],
            ],
        )
        service._polling = False

        with patch.object(
            arm_service.time, "sleep", lambda _seconds: None
        ):
            service._fb_move("lift", TARGET, 50, 2.0, 0.5)

        zs = [
            event[1]["pose_mm_deg"][2]
            for event in socket.events
            if event[0] == "arm_pose"
        ]
        self.assertGreaterEqual(len(zs), 2)
        self.assertIn(170.0, zs)

    def test_go_ready_uses_live_watchdog_when_poller_is_stopped(self) -> None:
        service, socket = make_arm_service(
            flags=[0, 0, 1],
            poses=[
                [490000, 0, 170000, 179999, 0, 0],
                [480000, 0, 300000, 179999, 0, 0],
                [444000, 0, 425000, 179999, 0, 0],
            ],
        )
        service.demo_cfg = {
            "ready_pose": [444000, 0, 425000, 179999, 0, 0],
            "ready_motion": {
                "speed_percent": 50,
                "verify_timeout_s": 2.0,
                "motion_start_timeout_s": 0.5,
            },
        }
        service._polling = False

        with patch.object(
            arm_service.time, "sleep", lambda _seconds: None
        ):
            self.assertTrue(service.go_ready())

        zs = [
            event[1]["pose_mm_deg"][2]
            for event in socket.events
            if event[0] == "arm_pose"
        ]
        self.assertGreaterEqual(len(zs), 2)
        self.assertIn(425.0, zs)


class MotionProfileTests(unittest.TestCase):
    def test_speed_mismatch_aborts_before_go(self) -> None:
        ctrl, client = make_controller(speed_readback=20)

        with self.assertRaisesRegex(
            RuntimeError, "speed register rejected"
        ):
            ctrl.move_to(TARGET, speed=60, wait=False)

        self.assertNotIn(
            (ctrl.regs["motion_command"], 301), client.single_writes
        )

    def test_matching_speed_sends_go_and_returns_motion_profile(self) -> None:
        ctrl, client = make_controller(
            speed_readback=60, acc=200, dec=50
        )

        profile = ctrl.move_to(TARGET, speed=60, wait=False)

        self.assertIn(
            (ctrl.regs["motion_command"], 301), client.single_writes
        )
        self.assertEqual(profile["applied_speed_percent"], 60)
        self.assertEqual(profile["acceleration_raw"], 200)
        self.assertEqual(profile["deceleration_raw"], 50)
        self.assertTrue(profile["confirmed"])

    def test_refresh_updates_connected_controller_motion_settings(self) -> None:
        ctrl, _client = make_controller(speed_readback=20)
        service = arm_service.ArmService(
            {"arm": {"pose_poll_hz": 2}},
            {"motion": dict(ctrl.motion), "safety": dict(ctrl.safety)},
        )
        service.ctrl = ctrl
        service.connected = True
        fresh_demo = {"arm": {"pose_poll_hz": 4}}
        fresh_modbus = {
            "motion": {
                **ctrl.motion,
                "acceleration_raw": 350,
                "deceleration_raw": 90,
            },
            "safety": {"max_speed_percent": 80},
        }

        service.refresh_runtime_config(fresh_demo, fresh_modbus)

        self.assertIs(service.demo_cfg, fresh_demo)
        self.assertEqual(ctrl.motion["acceleration_raw"], 350)
        self.assertEqual(ctrl.motion["deceleration_raw"], 90)
        self.assertEqual(ctrl.safety["max_speed_percent"], 80)

    def test_both_auto_entry_points_refresh_runtime_motion_config(self) -> None:
        source = (ROOT / "tools" / "voice_pick_demo.py").read_text(
            encoding="utf-8"
        )
        fallback_handler = source[
            source.index("def on_fallback_run") : source.index(
                '@socketio.on("teach_start")'
            )
        ]
        auto_handler = source[
            source.index("def on_vla_auto_start") : source.index(
                '@socketio.on("vla_gate")'
            )
        ]
        refresh_call = (
            "arm.refresh_runtime_config("
            "fresh.demo_config, fresh.modbus_config"
            ")"
        )

        self.assertIn(refresh_call, fallback_handler)
        self.assertIn(refresh_call, auto_handler)


if __name__ == "__main__":
    unittest.main()
