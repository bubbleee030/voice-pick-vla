from __future__ import annotations

import ast
import inspect
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import runtime_config
from src.services import gripper_service, vla_service


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class SequenceGrip:
    def __init__(self, statuses: list[dict]) -> None:
        self._statuses = iter(statuses)
        self._last = statuses[-1]

    def lstm_status(self) -> dict:
        self._last = next(self._statuses, self._last)
        return self._last


def load_hardware_free_lstm_runner():
    """Load only LSTMRunner's real class body, without module-level serial IO."""
    path = ROOT / "agx" / "lstm_gripper_service.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    runner_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LSTMRunner"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0),
            runner_node,
        ],
        type_ignores=[],
    )
    namespace = {
        "__name__": "lstm_runner_test_module",
        "np": np,
        "torch": torch,
        "threading": threading,
        "time": time.time,
        "sleep": time.sleep,
        "MOTOR_LIMITS": [(1872, 4272), (1872, 4272), (848, 3248)],
        "ACTIONS": {
            "trapezoid_grasp": ("", "", 64, 3, "grasp"),
            "trapezoid_release": ("", "", 64, 3, "release"),
            "knife_grasp": ("", "", 64, 3, "grasp"),
            "knife_release": ("", "", 64, 3, "release"),
            "PCB_release": ("", "", 64, 3, "release"),
        },
        "OBJECT_ACTIONS": {
            "trapezoid": ["trapezoid_grasp", "trapezoid_release"],
            "butter_knife": ["knife_grasp", "knife_release"],
            "board": ["PCB_release"],
        },
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["LSTMRunner"]


def load_hardware_free_tactile_parser():
    path = ROOT / "agx" / "lstm_gripper_service.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    parser_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "TactileFrameParser"
    )
    module = ast.Module(body=[parser_node], type_ignores=[])
    namespace = {"__name__": "tactile_parser_test_module"}
    exec(
        compile(ast.fix_missing_locations(module), str(path), "exec"),
        namespace,
    )
    return namespace["TactileFrameParser"]


class RuntimeConfigTests(unittest.TestCase):
    @staticmethod
    def fixed_fallback() -> dict:
        return {
            "hover_z_mm": 425,
            "travel_speed_percent": 70,
            "descend_speed_percent": 30,
            "per_step_timeout_s": 25.0,
            "motion_start_timeout_s": 3.0,
            "fixed_release_position": [3072, 3072, 2048],
            "fixed_target_tolerance_ticks": 30,
            "fixed_settle_ticks": 12,
            "fixed_settle_hold_s": 1.2,
            "fixed_min_close_ticks": 60,
            "fixed_min_close_fingers": 2,
            "place_approach_pose": [
                490116,
                -203991,
                424908,
                179999,
                0,
                0,
            ],
            "objects": {
                "trapezoid": {
                    "pick_x_mm": 490.116,
                    "pick_y_mm": 177.869,
                    "grasp_z_mm": 170.725,
                    "contact_dwell_s": 0.0,
                    "requires_grasp": True,
                    "fixed_grasp_position": None,
                    "lift_during_grasp": False,
                    "grasp_start_z_mm": None,
                    "release_z_mm": 180,
                },
                "board": {
                    "pick_x_mm": 505.0,
                    "pick_y_mm": 214.0,
                    "grasp_z_mm": 124,
                    "contact_dwell_s": 3.0,
                    "requires_grasp": False,
                    "fixed_grasp_position": None,
                    "lift_during_grasp": False,
                    "grasp_start_z_mm": None,
                    "release_z_mm": 170,
                },
                "butter_knife": {
                    "pick_x_mm": 490.1,
                    "pick_y_mm": 0.0,
                    "grasp_z_mm": 126,
                    "contact_dwell_s": 3.0,
                    "requires_grasp": True,
                    "fixed_grasp_position": None,
                    "lift_during_grasp": True,
                    "grasp_start_z_mm": 170,
                    "release_z_mm": 170,
                },
            },
        }

    def test_plumbs_contact_dwell_and_zero_backoff_into_tail(self) -> None:
        demo_cfg = {
            "fixed_fallback": self.fixed_fallback(),
            "vla_auto": {
                "checkpoint": "unused",
                "objects": {
                    "butter_knife": {
                        "grasp_backoff_mm": 0,
                        "lstm": {
                            "grasp": "knife_grasp",
                            "release": "knife_release",
                        },
                    }
                },
            }
        }

        with patch.object(runtime_config, "load_taught_fallback", return_value={}):
            resolved = runtime_config.build_auto_run_cfg(
                demo_cfg, {"fallback": {}}, "butter_knife"
            )

        self.assertIsNotNone(resolved)
        _, tail, _ = resolved
        self.assertEqual(tail["grasp_backoff_mm"], 0)
        self.assertEqual(tail["grasp_contact_dwell_s"], 3.0)
        self.assertTrue(tail["lift_during_grasp"])
        self.assertEqual(tail["grasp_start_z_mm"], 170)

    def test_knife_resolves_separate_prepare_and_final_thresholds(self) -> None:
        demo_cfg = runtime_config._load_yaml(
            ROOT / "config" / "demo_config.yaml"
        )

        with patch.object(
            runtime_config, "load_taught_fallback", return_value={}
        ):
            resolved = runtime_config.build_auto_run_cfg(
                demo_cfg, {"fallback": {}}, "butter_knife"
            )

        self.assertIsNotNone(resolved)
        run_cfg, _tail, _obj = resolved
        self.assertEqual(run_cfg["grasp_prepare_min_close_ticks"], 20)
        self.assertEqual(run_cfg["grasp_min_close_ticks"], 60)
        self.assertEqual(run_cfg["grasp_min_close_fingers"], 2)

    def test_prepare_threshold_cannot_exceed_final_threshold(self) -> None:
        demo_cfg = runtime_config._load_yaml(
            ROOT / "config" / "demo_config.yaml"
        )
        demo_cfg["vla_auto"]["grasp_prepare_min_close_ticks"] = 61

        with patch.object(
            runtime_config, "load_taught_fallback", return_value={}
        ), self.assertRaisesRegex(
            runtime_config.FixedFallbackConfigError,
            r"grasp_prepare_min_close_ticks.*grasp_min_close_ticks",
        ):
            runtime_config.build_auto_run_cfg(
                demo_cfg, {"fallback": {}}, "butter_knife"
            )

    def test_board_plumbs_three_second_dwell_as_auto_lift_confirmation(self) -> None:
        demo_cfg = {
            "fixed_fallback": self.fixed_fallback(),
            "vla_auto": {
                "checkpoint": "unused",
                "objects": {
                    "board": {
                        "lstm": {"release": "PCB_release"},
                    }
                },
            }
        }

        with patch.object(runtime_config, "load_taught_fallback", return_value={}):
            resolved = runtime_config.build_auto_run_cfg(
                demo_cfg, {"fallback": {}}, "board"
            )

        self.assertIsNotNone(resolved)
        _, tail, _ = resolved
        self.assertEqual(tail["grasp_contact_dwell_s"], 3.0)
        self.assertTrue(tail["auto_lift_after_dwell"])

    def test_board_resolves_one_finger_release_excursion_policy(self) -> None:
        demo_cfg = {
            "fixed_fallback": self.fixed_fallback(),
            "vla_auto": {
                "grasp_min_tracking_fingers": 2,
                "release_min_excursion_ticks": 60,
                "release_min_excursion_fingers": 2,
                "release_min_tracking_fingers": 3,
                "objects": {
                    "board": {
                        "release_min_excursion_fingers": 1,
                        "lstm": {"release": "PCB_release"},
                    }
                },
            },
        }

        resolved = runtime_config.build_auto_run_cfg(
            demo_cfg, {"fallback": {}}, "board"
        )

        self.assertIsNotNone(resolved)
        run_cfg, tail_cfg, _ = resolved
        self.assertEqual(run_cfg["grasp_min_tracking_fingers"], 2)
        self.assertEqual(tail_cfg["release_min_excursion_ticks"], 60)
        self.assertEqual(tail_cfg["release_min_excursion_fingers"], 1)
        self.assertEqual(tail_cfg["release_min_tracking_fingers"], 3)

    def test_trapezoid_inherits_global_release_policy(self) -> None:
        demo_cfg = {
            "fixed_fallback": self.fixed_fallback(),
            "vla_auto": {
                "release_min_excursion_ticks": 60,
                "release_min_excursion_fingers": 2,
                "release_min_tracking_fingers": 3,
                "objects": {
                    "trapezoid": {
                        "lstm": {
                            "grasp": "trapezoid_grasp",
                            "release": "trapezoid_release",
                        }
                    }
                },
            },
        }

        resolved = runtime_config.build_auto_run_cfg(
            demo_cfg, {"fallback": {}}, "trapezoid"
        )

        self.assertIsNotNone(resolved)
        _, tail_cfg, _ = resolved
        self.assertEqual(tail_cfg["release_min_excursion_ticks"], 60)
        self.assertEqual(tail_cfg["release_min_excursion_fingers"], 2)
        self.assertEqual(tail_cfg["release_min_tracking_fingers"], 3)


class VLAServiceDiagnosticTests(unittest.TestCase):
    def test_motion_profile_log_names_stage_speeds_and_controller_ramps(
        self,
    ) -> None:
        service = vla_service.VLAService()
        service.arm = type(
            "Arm",
            (),
            {
                "ctrl": type(
                    "Ctrl",
                    (),
                    {
                        "motion": {
                            "use_linear_move": False,
                            "set_acceleration": True,
                            "acceleration_raw": 200,
                            "set_deceleration": True,
                            "deceleration_raw": 50,
                        }
                    },
                )()
            },
        )()
        logs: list[tuple[str, str]] = []
        service._emit_log = lambda level, message: logs.append(
            (level, message)
        )

        service._log_auto_motion_settings(12, 50, 30)

        message = logs[-1][1]
        self.assertIn("correction=12%", message)
        self.assertIn("travel=50%", message)
        self.assertIn("descend=30%", message)
        self.assertIn("MovP", message)
        self.assertIn("ACC=200", message)
        self.assertIn("DEC=50", message)

    def test_auto_move_reports_controller_confirmed_speed(self) -> None:
        class Arm:
            def check_safety(self, _pose):
                return True, "ok"

            def _fb_move(self, *_args):
                return {
                    "applied_speed_percent": 50,
                    "move_type": "MovP",
                    "acceleration_raw": 200,
                    "deceleration_raw": 50,
                    "confirmed": True,
                }

        service = vla_service.VLAService(arm_service=Arm())
        logs: list[tuple[str, str]] = []
        service._emit_log = lambda level, message: logs.append(
            (level, message)
        )

        service._auto_move(490.0, 0.0, 425.0, 50, "lift", 25.0, 3.0)

        self.assertTrue(
            any(
                "controller confirmed speed=50%" in message
                and "ACC=200" in message
                and "DEC=50" in message
                for _, message in logs
            )
        )

    def test_contact_dwell_is_stoppable_and_reports_its_stage(self) -> None:
        service = vla_service.VLAService()
        service._running = True
        stages: list[str] = []
        logs: list[tuple[str, str]] = []
        service._stage = stages.append
        service._emit_log = lambda level, message: logs.append((level, message))
        clock = FakeClock()

        with patch.object(vla_service.time, "time", clock.time), patch.object(
            vla_service.time, "sleep", clock.sleep
        ):
            completed = service._wait_grasp_contact_dwell(1.5)

        self.assertTrue(completed)
        self.assertEqual(stages, ["grasp_contact_dwell"])
        self.assertGreaterEqual(clock.now, 1.5)
        self.assertTrue(any("保持接觸 1.5s" in message for _, message in logs))

    def test_board_dwell_confirmation_skips_pick_gate(self) -> None:
        service = vla_service.VLAService()
        service._running = True
        stages: list[str] = []
        gates: list[str] = []
        logs: list[tuple[str, str]] = []
        service._stage = stages.append
        service._emit_log = lambda level, message: logs.append((level, message))
        service._wait_gate = lambda name: gates.append(name) or True

        confirmed = service._confirm_pick_without_lstm(
            auto_lift_after_dwell=True,
            contact_dwell_s=1.0,
        )

        self.assertTrue(confirmed)
        self.assertEqual(stages, ["auto_pick"])
        self.assertEqual(gates, [])
        self.assertTrue(
            any("1 秒接觸等待完成" in message for _, message in logs)
        )
        self.assertFalse(
            any("3 秒接觸等待完成" in message for _, message in logs)
        )
        source = inspect.getsource(vla_service.VLAService._auto_loop)
        self.assertIn(
            "if grasp_contact_dwell > 0 and not dry_run",
            source,
        )
        self.assertNotIn(
            "if grasp_action and grasp_contact_dwell > 0",
            source,
        )
        self.assertIn(
            "auto_lift_after_dwell, grasp_contact_dwell",
            source,
        )

    def test_manual_pick_without_lstm_keeps_pick_gate(self) -> None:
        service = vla_service.VLAService()
        service._running = True
        stages: list[str] = []
        gates: list[str] = []
        service._stage = stages.append
        service._wait_gate = lambda name: gates.append(name) or True

        self.assertTrue(service._confirm_pick_without_lstm(False, 0.0))
        self.assertEqual(stages, ["gate_pick"])
        self.assertEqual(gates, ["pick_done"])

    def test_claw_loss_recovery_restarts_at_three_and_falls_back_after_two(
        self,
    ) -> None:
        decide = vla_service._claw_loss_recovery_decision
        common = {
            "has_verified_target": True,
            "restart_after_holds": 3,
            "post_restart_grace_holds": 2,
        }

        self.assertEqual(
            decide(
                lost_holds=2,
                post_restart_holds=0,
                restart_succeeded=False,
                **common,
            ),
            (False, False, True),
        )
        self.assertEqual(
            decide(
                lost_holds=3,
                post_restart_holds=0,
                restart_succeeded=False,
                **common,
            ),
            (True, False, False),
        )
        self.assertEqual(
            decide(
                lost_holds=4,
                post_restart_holds=1,
                restart_succeeded=True,
                **common,
            ),
            (False, False, True),
        )
        self.assertEqual(
            decide(
                lost_holds=5,
                post_restart_holds=2,
                restart_succeeded=True,
                **common,
            ),
            (False, True, False),
        )
        self.assertEqual(
            decide(
                lost_holds=5,
                post_restart_holds=2,
                restart_succeeded=False,
                **common,
            ),
            (False, False, False),
        )

        source = inspect.getsource(vla_service.VLAService._auto_loop)
        self.assertIn('rc.get("claw_restart_after_lost_holds", 3)', source)
        self.assertIn('rc.get("claw_post_restart_grace_holds", 2)', source)
        self.assertIn("_claw_loss_recovery_decision(", source)

    def test_demo_config_uses_fast_claw_recovery_thresholds(self) -> None:
        auto_cfg = runtime_config._load_yaml(
            ROOT / "config" / "demo_config.yaml"
        )["vla_auto"]

        self.assertEqual(auto_cfg["claw_restart_after_lost_holds"], 3)
        self.assertEqual(auto_cfg["claw_post_restart_grace_holds"], 2)

    def test_grasp_lift_starts_as_a_background_motion(self) -> None:
        service = vla_service.VLAService()
        service._running = True
        stages: list[str] = []
        logs: list[tuple[str, str]] = []
        started = threading.Event()
        release = threading.Event()
        calls: list[tuple] = []
        service._stage = stages.append
        service._emit_log = lambda level, message: logs.append((level, message))
        service._auto_pose = lambda: [490.0, 0.0, 126.0, 180.0, 0.0, 0.0]

        def blocking_move(*args):
            calls.append(args)
            started.set()
            release.wait(timeout=1.0)

        service._auto_move = blocking_move
        thread, errors = service._start_grasp_lift(
            lift_z=425.0,
            speed=60,
            step_timeout=25.0,
            start_timeout=3.0,
        )

        self.assertTrue(started.wait(timeout=0.5))
        self.assertTrue(thread.is_alive())
        self.assertEqual(stages, ["grasp_lift"])
        self.assertEqual(calls[0][0:3], (490.0, 0.0, 425.0))
        self.assertEqual(calls[0][4], "grasp_lift")
        release.set()
        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])

    def test_grasp_starts_at_measured_trigger_height_while_lift_is_active(self) -> None:
        service = vla_service.VLAService()
        service._running = True
        stages: list[str] = []
        logs: list[tuple[str, str]] = []
        pose_zs = iter([126.0, 150.0, 169.0, 170.0])
        lift_active = threading.Event()
        release_lift = threading.Event()
        move_target_zs: list[float] = []
        measured_z: float | None = None
        grasp_started_at_z: float | None = None
        lift_was_active_when_grasp_started = False

        service._stage = stages.append
        service._emit_log = lambda level, message: logs.append((level, message))

        def measured_pose() -> list[float]:
            nonlocal measured_z
            next_z = next(pose_zs)
            if next_z > 126.0:
                self.assertTrue(lift_active.wait(timeout=0.5))
            measured_z = next_z
            return [490.0, 0.0, next_z, 180.0, 0.0, 0.0]

        def blocking_move(*args) -> None:
            move_target_zs.append(float(args[2]))
            lift_active.set()
            release_lift.wait(timeout=1.0)
            lift_active.clear()

        def start_grasp() -> bool:
            nonlocal grasp_started_at_z, lift_was_active_when_grasp_started
            grasp_started_at_z = measured_z
            lift_was_active_when_grasp_started = lift_active.is_set()
            return True

        service._auto_pose = measured_pose
        service._auto_move = blocking_move
        try:
            used_lstm, lift_thread, lift_errors = service._lift_and_start_grasp_at_z(
                lift_z=425.0,
                trigger_z=170.0,
                speed=60,
                step_timeout=25.0,
                start_timeout=3.0,
                start_grasp=start_grasp,
            )
        finally:
            release_lift.set()

        lift_thread.join(timeout=1.0)
        self.assertTrue(used_lstm)
        self.assertEqual(move_target_zs, [425.0])
        self.assertEqual(grasp_started_at_z, 170.0)
        self.assertTrue(lift_was_active_when_grasp_started)
        self.assertEqual(stages, ["grasp_lift", "grasp_trigger_wait"])
        self.assertEqual(lift_errors, [])
        self.assertTrue(
            any(
                "target z=170.0" in message and "actual z=170.0" in message
                for _, message in logs
            )
        )

    def test_grasp_trigger_propagates_lift_error_during_pose_read(self) -> None:
        service = vla_service.VLAService()
        service._running = True
        pose_read_started = threading.Event()
        failure_recorded = threading.Event()
        lift_errors: list[Exception] = []
        expected_error = RuntimeError("lift failed during pose read")
        start_grasp_invoked = False
        service._stage = lambda _name: None
        service._emit_log = lambda _level, _message: None

        def fail_during_pose_read() -> None:
            self.assertTrue(pose_read_started.wait(timeout=0.5))
            lift_errors.append(expected_error)
            failure_recorded.set()

        lift_thread = threading.Thread(target=fail_during_pose_read)

        def start_lift(*_args):
            lift_thread.start()
            return lift_thread, lift_errors

        def measured_pose() -> list[float]:
            pose_read_started.set()
            self.assertTrue(failure_recorded.wait(timeout=0.5))
            lift_thread.join(timeout=0.5)
            return [490.0, 0.0, 170.0, 180.0, 0.0, 0.0]

        def start_grasp() -> bool:
            nonlocal start_grasp_invoked
            start_grasp_invoked = True
            return True

        service._auto_pose = measured_pose

        with patch.object(service, "_start_grasp_lift", start_lift):
            with self.assertRaisesRegex(RuntimeError, "lift failed during pose read"):
                service._lift_and_start_grasp_at_z(
                    lift_z=425.0,
                    trigger_z=170.0,
                    speed=60,
                    step_timeout=25.0,
                    start_timeout=3.0,
                    start_grasp=start_grasp,
                )

        lift_thread.join(timeout=0.5)
        self.assertFalse(start_grasp_invoked)

    def test_grasp_trigger_returns_without_grasp_if_stopped_during_pose_read(
        self,
    ) -> None:
        service = vla_service.VLAService()
        service._running = True
        release_lift = threading.Event()
        lift_errors: list[Exception] = []
        start_grasp_invoked = False
        service._stage = lambda _name: None
        service._emit_log = lambda _level, _message: None
        lift_thread = threading.Thread(
            target=release_lift.wait,
            kwargs={"timeout": 1.0},
        )

        def start_lift(*_args):
            lift_thread.start()
            return lift_thread, lift_errors

        def measured_pose() -> list[float]:
            service._running = False
            return [490.0, 0.0, 170.0, 180.0, 0.0, 0.0]

        def start_grasp() -> bool:
            nonlocal start_grasp_invoked
            start_grasp_invoked = True
            return True

        service._auto_pose = measured_pose
        try:
            with patch.object(service, "_start_grasp_lift", start_lift):
                used_lstm, returned_thread, returned_errors = (
                    service._lift_and_start_grasp_at_z(
                        lift_z=425.0,
                        trigger_z=170.0,
                        speed=60,
                        step_timeout=25.0,
                        start_timeout=3.0,
                        start_grasp=start_grasp,
                    )
                )
        finally:
            release_lift.set()

        lift_thread.join(timeout=0.5)
        self.assertFalse(used_lstm)
        self.assertIs(returned_thread, lift_thread)
        self.assertIs(returned_errors, lift_errors)
        self.assertFalse(start_grasp_invoked)

    def test_grasp_trigger_fails_if_lift_ends_during_trigger_pose_read(
        self,
    ) -> None:
        service = vla_service.VLAService()
        service._running = True
        pose_read_started = threading.Event()
        lift_finished = threading.Event()
        lift_errors: list[Exception] = []
        start_grasp_invoked = False
        service._stage = lambda _name: None
        service._emit_log = lambda _level, _message: None

        def end_during_pose_read() -> None:
            self.assertTrue(pose_read_started.wait(timeout=0.5))
            lift_finished.set()

        lift_thread = threading.Thread(target=end_during_pose_read)

        def start_lift(*_args):
            lift_thread.start()
            return lift_thread, lift_errors

        def measured_pose() -> list[float]:
            pose_read_started.set()
            self.assertTrue(lift_finished.wait(timeout=0.5))
            lift_thread.join(timeout=0.5)
            self.assertFalse(lift_thread.is_alive())
            return [490.0, 0.0, 170.0, 180.0, 0.0, 0.0]

        def start_grasp() -> bool:
            nonlocal start_grasp_invoked
            start_grasp_invoked = True
            return True

        service._auto_pose = measured_pose

        with patch.object(service, "_start_grasp_lift", start_lift):
            with self.assertRaisesRegex(
                RuntimeError,
                r"grasp lift ended at z=170\.0 before trigger z=170\.0",
            ):
                service._lift_and_start_grasp_at_z(
                    lift_z=425.0,
                    trigger_z=170.0,
                    speed=60,
                    step_timeout=25.0,
                    start_timeout=3.0,
                    start_grasp=start_grasp,
                )

        lift_thread.join(timeout=0.5)
        self.assertFalse(start_grasp_invoked)

    def test_grasp_trigger_fails_if_lift_ends_below_measured_height(self) -> None:
        service = vla_service.VLAService()
        service._running = True
        lift_finished = threading.Event()
        pose_calls = 0
        start_grasp_invoked = False
        service._stage = lambda _name: None
        service._emit_log = lambda _level, _message: None

        def measured_pose() -> list[float]:
            nonlocal pose_calls
            pose_calls += 1
            if pose_calls > 1:
                self.assertTrue(lift_finished.wait(timeout=0.5))
            return [490.0, 0.0, 160.0, 180.0, 0.0, 0.0]

        def ending_move(*_args) -> None:
            lift_finished.set()

        def start_grasp() -> bool:
            nonlocal start_grasp_invoked
            start_grasp_invoked = True
            return True

        service._auto_pose = measured_pose
        service._auto_move = ending_move

        with self.assertRaisesRegex(
            RuntimeError,
            r"grasp lift ended at z=160\.0 before trigger z=170\.0",
        ):
            service._lift_and_start_grasp_at_z(
                lift_z=425.0,
                trigger_z=170.0,
                speed=60,
                step_timeout=25.0,
                start_timeout=3.0,
                start_grasp=start_grasp,
            )

        self.assertFalse(start_grasp_invoked)

    def test_auto_loop_routes_configured_height_through_continuous_lift(self) -> None:
        source = inspect.getsource(vla_service.VLAService._auto_loop)

        self.assertIn('tail.get("grasp_start_z_mm")', source)
        self.assertIn("_lift_and_start_grasp_at_z", source)
        self.assertIn("lstm_activate", source)
        self.assertRegex(
            source,
            r"_lift_and_start_grasp_at_z\(\s*lift_z,\s*grasp_start_z,\s*travel,",
        )

    def test_two_phase_prepare_is_ready_before_lift_and_activates_at_trigger(self) -> None:
        source = inspect.getsource(vla_service.VLAService._auto_loop)

        threshold_at = source.index("prepare_min_close_ticks = int(")
        helper_at = source.index("def lstm_prepare(")
        helper_end = source.index("def lstm_activate(", helper_at)
        helper_source = source[helper_at:helper_end]
        prepare_at = source.index("prepared_lstm = lstm_prepare(")
        dwell_at = source.index("_wait_grasp_contact_dwell(", prepare_at)
        ready_at = source.index("_wait_grasp_goal_ready(", dwell_at)
        lift_at = source.index("_lift_and_start_grasp_at_z(", ready_at)
        activate_at = source.index("lambda: lstm_activate()", lift_at)

        self.assertLess(threshold_at, helper_at)
        self.assertIn(
            "min_close_ticks=prepare_min_close_ticks",
            helper_source,
        )
        self.assertLess(prepare_at, dwell_at)
        self.assertLess(dwell_at, ready_at)
        self.assertLess(ready_at, lift_at)
        self.assertLess(lift_at, activate_at)
        self.assertIn("timeout_s=None", source)

    def test_grasp_settle_logs_motor_goal_and_tracking_error_by_step(self) -> None:
        statuses = [
            {
                "step": 1,
                "goal_step": 1,
                "motor_pos": [3072, 3072, 2048],
                "goal_pos": [3072, 3072, 2048],
                "tactile": [2400, 1300, 1500],
            },
            {
                "step": 21,
                "goal_step": 21,
                "motor_pos": [2805, 2815, 1805],
                "goal_pos": [2800, 2810, 1800],
                "tactile": [2410, 1310, 1510],
            },
            {
                "step": 22,
                "goal_step": 22,
                "motor_pos": [2805, 2815, 1805],
                "goal_pos": [2800, 2810, 1800],
                "tactile": [2410, 1310, 1510],
            },
            {
                "step": 23,
                "goal_step": 23,
                "motor_pos": [2805, 2815, 1805],
                "goal_pos": [2800, 2810, 1800],
                "tactile": [2410, 1310, 1510],
            },
        ]
        service = vla_service.VLAService()
        service._running = True
        service._stage = lambda _name: None
        logs: list[tuple[str, str]] = []
        service._emit_log = lambda level, message: logs.append((level, message))
        clock = FakeClock()

        with patch.object(vla_service.time, "time", clock.time), patch.object(
            vla_service.time, "sleep", clock.sleep
        ):
            settled = service._wait_fingers_settled(
                SequenceGrip(statuses),
                timeout_s=2.0,
                settle_ticks=12,
                hold_s=0.2,
                min_step=20,
                require_displaced=True,
                what="抓取",
                nxt="上抬",
            )

        self.assertTrue(settled)
        trace = next(message for _, message in logs if "trace step=21" in message)
        self.assertIn("pos=[2805, 2815, 1805]", trace)
        self.assertIn("goal(step=21)=[2800, 2810, 1800]", trace)
        self.assertIn("tracking_err=[5, 5, 5]", trace)
        self.assertIn("tactile=[2410, 1310, 1510]", trace)

    def test_near_start_pose_does_not_count_as_a_real_grasp(self) -> None:
        start_pos = [3079, 3038, 2047]
        near_home = {
            "step": 31,
            "goal_step": 31,
            "action_start_pos": start_pos,
            "motor_pos": [3052, 3038, 2038],
            "goal_pos": [3060, 3054, 2044],
            "tactile": [2531, 1336, 1544],
        }
        service = vla_service.VLAService()
        service._running = True
        service._stage = lambda _name: None
        logs: list[tuple[str, str]] = []
        service._emit_log = lambda level, message: logs.append((level, message))
        clock = FakeClock()

        with patch.object(vla_service.time, "time", clock.time), patch.object(
            vla_service.time, "sleep", clock.sleep
        ):
            settled = service._wait_fingers_settled(
                SequenceGrip([near_home]),
                timeout_s=1.0,
                settle_ticks=12,
                hold_s=0.2,
                min_step=20,
                require_displaced=True,
                what="抓取",
                nxt="上抬",
            )

        self.assertFalse(settled)
        self.assertFalse(any("✅" in message for _, message in logs))

    def test_real_close_from_action_start_can_settle(self) -> None:
        start_pos = [3079, 3038, 2047]
        closed = {
            "step": 55,
            "goal_step": 55,
            "action_start_pos": start_pos,
            "motor_pos": [1900, 1950, 900],
            "goal_pos": [1885, 1935, 880],
            "tactile": [2488, 1325, 1549],
        }
        service = vla_service.VLAService()
        service._running = True
        service._stage = lambda _name: None
        service._emit_log = lambda _level, _message: None
        clock = FakeClock()

        with patch.object(vla_service.time, "time", clock.time), patch.object(
            vla_service.time, "sleep", clock.sleep
        ):
            settled = service._wait_fingers_settled(
                SequenceGrip([closed]),
                timeout_s=1.0,
                settle_ticks=12,
                hold_s=0.2,
                min_step=20,
                require_displaced=True,
                what="抓取",
                nxt="上抬",
            )

        self.assertTrue(settled)

    def test_slow_cumulative_motor_and_goal_drift_does_not_settle(self) -> None:
        statuses = []
        for offset in range(12):
            pos = [3072 - 3 * offset, 3072 - 3 * offset, 2048 - 3 * offset]
            statuses.append(
                {
                    "step": 20 + offset,
                    "goal_step": 20 + offset,
                    "motor_pos": pos,
                    "goal_pos": list(pos),
                    "tactile": [2450, 1330, 1510],
                }
            )

        service = vla_service.VLAService()
        service._running = True
        service._stage = lambda _name: None
        service._emit_log = lambda _level, _message: None
        clock = FakeClock()

        with patch.object(vla_service.time, "time", clock.time), patch.object(
            vla_service.time, "sleep", clock.sleep
        ):
            settled = service._wait_fingers_settled(
                SequenceGrip(statuses),
                timeout_s=2.0,
                settle_ticks=12,
                hold_s=1.2,
                min_step=20,
                require_displaced=False,
                what="放開",
                nxt="回位",
            )

        self.assertFalse(settled)

    def test_flat_tracked_motor_and_goal_trace_settles(self) -> None:
        flat = {
            "step": 32,
            "goal_step": 31,
            "motor_pos": [3042, 3071, 2047],
            "goal_pos": [3038, 3065, 2045],
            "tactile": [2447, 1328, 1508],
        }
        service = vla_service.VLAService()
        service._running = True
        service._stage = lambda _name: None
        service._emit_log = lambda _level, _message: None
        clock = FakeClock()

        with patch.object(vla_service.time, "time", clock.time), patch.object(
            vla_service.time, "sleep", clock.sleep
        ):
            settled = service._wait_fingers_settled(
                SequenceGrip([flat]),
                timeout_s=2.0,
                settle_ticks=12,
                hold_s=1.2,
                min_step=20,
                require_displaced=False,
                what="放開",
                nxt="回位",
            )

        self.assertTrue(settled)

    def test_board_initial_plateau_is_not_release_completion(self) -> None:
        plateau = {
            "status": "running",
            "step": 31,
            "goal_step": 31,
            "action_start_pos": [3071, 3073, 2044],
            "motor_pos": [3049, 3073, 2047],
            "goal_pos": [3045, 3064, 2045],
            "tactile": [2358, 1326, 1556],
        }
        service = vla_service.VLAService()
        service._running = True
        service._stage = lambda _name: None
        service._emit_log = lambda _level, _message: None
        clock = FakeClock()

        with patch.object(vla_service.time, "time", clock.time), patch.object(
            vla_service.time, "sleep", clock.sleep
        ):
            settled = service._wait_fingers_settled(
                SequenceGrip([plateau]),
                timeout_s=2.0,
                settle_ticks=12,
                hold_s=1.2,
                min_step=20,
                require_displaced=False,
                what="放開",
                nxt="回位",
                tracking_ticks=30,
                min_tracking_fingers=3,
                min_excursion_ticks=60,
                min_excursion_fingers=1,
            )

        self.assertFalse(settled)

    def test_board_release_settles_after_real_finger_one_excursion(self) -> None:
        start = [3071, 3073, 2044]
        moving = {
            "status": "running",
            "step": 45,
            "goal_step": 45,
            "action_start_pos": start,
            "motor_pos": [2780, 3060, 2040],
            "goal_pos": [2765, 3055, 2038],
            "tactile": [2360, 1320, 1550],
        }
        flat = {
            "status": "running",
            "step": 46,
            "goal_step": 46,
            "action_start_pos": start,
            "motor_pos": [2770, 3058, 2039],
            "goal_pos": [2765, 3055, 2038],
            "tactile": [2360, 1320, 1550],
        }
        service = vla_service.VLAService()
        service._running = True
        service._stage = lambda _name: None
        service._emit_log = lambda _level, _message: None
        service._manual_release_position = [2872, 3072, 2048]
        clock = FakeClock()

        with patch.object(vla_service.time, "time", clock.time), patch.object(
            vla_service.time, "sleep", clock.sleep
        ):
            settled = service._wait_fingers_settled(
                SequenceGrip([moving, flat]),
                timeout_s=3.0,
                settle_ticks=12,
                hold_s=1.2,
                min_step=20,
                require_displaced=False,
                what="放開",
                nxt="回位",
                tracking_ticks=30,
                min_tracking_fingers=3,
                min_excursion_ticks=60,
                min_excursion_fingers=1,
                allow_manual_release=True,
            )

        self.assertTrue(settled)
        self.assertFalse(service._manual_release_requested)
        self.assertFalse(service.release_home_available)
        self.assertFalse(service._release_home_event.is_set())

    def test_contact_blocked_trapezoid_settles_with_two_tracked_fingers(
        self,
    ) -> None:
        closed = {
            "status": "running",
            "step": 306,
            "goal_step": 306,
            "action_start_pos": [3072, 3082, 2047],
            "motor_pos": [2161, 2187, 1230],
            "goal_pos": [2160, 2183, 1170],
            "tactile": [2374, 1253, 1382],
        }
        service = vla_service.VLAService()
        service._running = True
        service._stage = lambda _name: None
        service._emit_log = lambda _level, _message: None
        clock = FakeClock()

        with patch.object(vla_service.time, "time", clock.time), patch.object(
            vla_service.time, "sleep", clock.sleep
        ):
            settled = service._wait_fingers_settled(
                SequenceGrip([closed]),
                timeout_s=2.0,
                settle_ticks=12,
                hold_s=1.2,
                min_step=20,
                require_displaced=True,
                what="抓取",
                nxt="上抬",
                min_close_ticks=60,
                min_close_fingers=2,
                tracking_ticks=30,
                min_tracking_fingers=2,
            )

        self.assertTrue(settled)

    def test_grasp_with_only_one_tracked_finger_does_not_settle(self) -> None:
        closed = {
            "status": "running",
            "step": 80,
            "goal_step": 80,
            "action_start_pos": [3072, 3072, 2048],
            "motor_pos": [2200, 2260, 1270],
            "goal_pos": [2195, 2180, 1170],
            "tactile": [2380, 1290, 1420],
        }
        service = vla_service.VLAService()
        service._running = True
        service._stage = lambda _name: None
        service._emit_log = lambda _level, _message: None
        clock = FakeClock()

        with patch.object(vla_service.time, "time", clock.time), patch.object(
            vla_service.time, "sleep", clock.sleep
        ):
            settled = service._wait_fingers_settled(
                SequenceGrip([closed]),
                timeout_s=2.0,
                settle_ticks=12,
                hold_s=1.2,
                min_step=20,
                require_displaced=True,
                what="抓取",
                nxt="上抬",
                min_close_ticks=60,
                min_close_fingers=2,
                tracking_ticks=30,
                min_tracking_fingers=2,
            )

        self.assertFalse(settled)

    def test_verify_fingers_home_returns_true_for_measured_home(self) -> None:
        class HomeGrip:
            def lstm_status(self):
                return {"motor_pos": [3071, 3073, 2047]}

        service = vla_service.VLAService()
        with patch.object(vla_service.time, "sleep", return_value=None):
            self.assertTrue(service._verify_fingers_home(HomeGrip()))

    def test_verify_fingers_home_returns_false_when_reboot_stays_off_home(
        self,
    ) -> None:
        class StuckGrip:
            def lstm_status(self):
                return {"motor_pos": [2500, 2500, 1400]}

            def reboot_motors(self, home=True):
                return {
                    "ok": True,
                    "motor_pos": [2500, 2500, 1400],
                    "rebooted": [1, 2, 3],
                }

        service = vla_service.VLAService()
        service._emit_log = lambda _level, _message: None
        with patch.object(vla_service.time, "sleep", return_value=None):
            self.assertFalse(service._verify_fingers_home(StuckGrip()))

    def test_home_failure_raises_before_release_retract(self) -> None:
        class FailedHomeGrip:
            last_error = "offline"

            def set_position(self, positions, mode=None):
                return False

        service = vla_service.VLAService()
        service._emit_log = lambda _level, _message: None
        with self.assertRaisesRegex(RuntimeError, "finger home command failed"):
            service._home_fingers_or_raise(FailedHomeGrip())

    def test_release_home_precedes_arm_retract_in_auto_loop(self) -> None:
        source = inspect.getsource(vla_service.VLAService._auto_loop)
        release_at = source.index('self._stage("gate_release")')
        completion_at = source.index(
            "allow_manual_release=True", release_at
        )
        halt_at = source.index("lstm_halt()", completion_at)
        home_at = source.index("self._home_fingers_or_raise(grip)", release_at)
        return_at = source.index('self._stage("return")', home_at)
        retract_at = source.index('"release_retract"', return_at)

        self.assertLess(release_at, completion_at)
        self.assertLess(completion_at, halt_at)
        self.assertLess(halt_at, home_at)
        self.assertLess(home_at, return_at)
        self.assertLess(return_at, retract_at)

    def test_manual_grasp_accept_is_rejected_until_real_close(self) -> None:
        service = vla_service.VLAService()
        service._running = True
        service.auto_stage = "抓取_settle"
        service.grasp_accept_available = False
        logs = []
        service._emit_log = lambda level, message: logs.append((level, message))

        self.assertFalse(service.accept_grasp())
        self.assertFalse(service._grasp_accept_event.is_set())
        self.assertTrue(any("rejected" in message for _, message in logs))

    def test_manual_grasp_accept_sets_event_after_real_close(self) -> None:
        service = vla_service.VLAService()
        service._running = True
        service.auto_stage = "抓取_settle"
        service.grasp_accept_available = True
        service._emit_log = lambda _level, _message: None

        self.assertTrue(service.accept_grasp())
        self.assertTrue(service._grasp_accept_event.is_set())

    def test_status_exposes_grasp_accept_availability(self) -> None:
        service = vla_service.VLAService()
        service.grasp_accept_available = True

        self.assertTrue(service.vla_status_payload()["grasp_accept_available"])

    def test_status_exposes_contextual_finger_stage_action(self) -> None:
        service = vla_service.VLAService()

        self.assertIsNone(
            service.vla_status_payload()["finger_stage_action"]
        )
        service.grasp_accept_available = True
        self.assertEqual(
            service.vla_status_payload()["finger_stage_action"],
            "grasp",
        )
        service.grasp_accept_available = False
        service.release_accept_available = True
        self.assertEqual(
            service.vla_status_payload()["finger_stage_action"],
            "release",
        )

    def test_configured_release_uses_distinct_pose_and_home_actions(self) -> None:
        service = vla_service.VLAService()
        service._running = True
        service.auto_stage = "放開_settle"
        service._manual_release_position = [2872, 3072, 2048]
        service.release_accept_available = True
        service._emit_log = lambda _level, _message: None

        self.assertEqual(
            service.vla_status_payload()["finger_stage_action"],
            "release_pose",
        )
        self.assertTrue(service.complete_finger_stage())
        self.assertTrue(service._manual_release_requested)
        self.assertTrue(service._release_accept_event.is_set())
        self.assertFalse(service._release_home_event.is_set())

        service.release_accept_available = False
        service.auto_stage = "release_pose_hold"
        self.assertFalse(service.complete_finger_stage())
        self.assertFalse(service._release_home_event.is_set())

        service.release_home_available = True
        self.assertEqual(
            service.vla_status_payload()["finger_stage_action"],
            "release_home",
        )
        self.assertTrue(service.complete_finger_stage())
        self.assertTrue(service._release_home_event.is_set())

    def test_two_stage_release_converges_pose_before_home_is_enabled(
        self,
    ) -> None:
        target = [2872, 3072, 2048]

        class TrackingGrip:
            last_error = ""

            def __init__(self) -> None:
                self.current = [2160, 2183, 1170]
                self.commands: list[list[int]] = []

            def get_state(self):
                return {"current_pos": list(self.current)}

            def set_position(
                self,
                positions,
                mode=None,
                stop_requested=None,
            ):
                self.commands.append(list(positions))
                self.current = list(positions)
                return True

            def lstm_status(self):
                return {"motor_pos": list(self.current)}

        service = vla_service.VLAService()
        service._running = True
        service._manual_release_position = target
        service._emit_log = lambda _level, _message: None
        service._emit_status = lambda: None
        grip = TrackingGrip()
        results: list[bool] = []
        runner = service._complete_manual_release_fallback
        service._release_home_event.set()  # stale first-stage key event
        worker = threading.Thread(
            target=lambda: results.append(
                runner(
                    grip,
                    target,
                    {
                        "fixed_target_tolerance_ticks": 0,
                        "fixed_settle_ticks": 0,
                        "fixed_settle_hold_s": 0.0,
                        "fixed_min_close_ticks": 60,
                        "fixed_min_close_fingers": 2,
                    },
                )
            )
        )
        worker.start()

        deadline = time.monotonic() + 2.0
        while (
            not service.release_home_available
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)

        self.assertTrue(service.release_home_available)
        self.assertEqual(service.auto_stage, "release_pose_hold")
        self.assertEqual(grip.commands, [target])
        self.assertTrue(worker.is_alive())
        self.assertTrue(service.complete_finger_stage())
        worker.join(timeout=2.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(results, [True])
        self.assertEqual(grip.commands, [target])
        self.assertFalse(service.release_home_available)

    def test_auto_loop_runs_two_stage_release_before_existing_home(self) -> None:
        source = inspect.getsource(vla_service.VLAService._auto_loop)
        release_at = source.index('self._stage("gate_release")')
        release_source = source[release_at:]
        halt_at = release_source.index("lstm_halt()")
        fallback_at = release_source.index(
            "self._complete_manual_release_fallback("
        )
        home_at = release_source.index("self._home_fingers_or_raise(grip)")

        self.assertIn("manual_release_position", release_source)
        self.assertLess(halt_at, fallback_at)
        self.assertLess(fallback_at, home_at)

    def test_contextual_completion_rejects_release_outside_settle(self) -> None:
        service = vla_service.VLAService()
        service._running = True
        service.auto_stage = "gate_release"
        service.release_accept_available = False
        service._emit_log = lambda _level, _message: None

        self.assertFalse(service.complete_finger_stage())
        self.assertFalse(service._release_accept_event.is_set())

    def test_manual_release_completion_is_available_and_consumed(
        self,
    ) -> None:
        service = vla_service.VLAService()
        service._running = True
        service._stage = lambda name: setattr(service, "auto_stage", name)
        service._emit_log = lambda _level, _message: None
        service._emit_status = lambda: None
        completed: list[bool] = []
        stalled_release = {
            "status": "running",
            "step": 80,
            "goal_step": 80,
            "action_start_pos": [2161, 2187, 1230],
            "motor_pos": [2500, 2520, 1600],
            "goal_pos": [2700, 2710, 1800],
            "tactile": [2374, 1253, 1382],
        }

        class CompleteReleaseWhenAvailableGrip:
            def lstm_status(self):
                if service.release_accept_available and not completed:
                    completed.append(service.complete_finger_stage())
                return stalled_release

        clock = FakeClock()
        with patch.object(vla_service.time, "time", clock.time), patch.object(
            vla_service.time, "sleep", clock.sleep
        ):
            released = service._wait_fingers_settled(
                CompleteReleaseWhenAvailableGrip(),
                timeout_s=2.0,
                settle_ticks=12,
                hold_s=10.0,
                min_step=20,
                require_displaced=False,
                what="放開",
                nxt="回位",
                min_excursion_ticks=60,
                min_excursion_fingers=2,
                allow_manual_release=True,
            )

        self.assertTrue(released)
        self.assertEqual(completed, [True])
        self.assertFalse(service.release_accept_available)
        self.assertFalse(service._release_accept_event.is_set())

    def test_manual_accept_completes_only_after_live_real_close(self) -> None:
        service = vla_service.VLAService()
        service._running = True
        service._stage = lambda name: setattr(service, "auto_stage", name)
        service._emit_log = lambda _level, _message: None
        service._emit_status = lambda: None
        accepted: list[bool] = []
        closed = {
            "status": "running",
            "step": 306,
            "goal_step": 306,
            "action_start_pos": [3072, 3082, 2047],
            "motor_pos": [2161, 2187, 1230],
            "goal_pos": [2160, 2183, 1170],
            "tactile": [2374, 1253, 1382],
        }

        class AcceptWhenAvailableGrip:
            def lstm_status(self):
                if service.grasp_accept_available and not accepted:
                    accepted.append(service.accept_grasp())
                return closed

        clock = FakeClock()
        with patch.object(vla_service.time, "time", clock.time), patch.object(
            vla_service.time, "sleep", clock.sleep
        ):
            settled = service._wait_fingers_settled(
                AcceptWhenAvailableGrip(),
                timeout_s=2.0,
                settle_ticks=12,
                hold_s=10.0,
                min_step=20,
                require_displaced=True,
                what="抓取",
                nxt="上抬",
                min_close_ticks=60,
                min_close_fingers=2,
                tracking_ticks=30,
                min_tracking_fingers=3,
                allow_manual_accept=True,
            )

        self.assertTrue(settled)
        self.assertEqual(accepted, [True])
        self.assertFalse(service.grasp_accept_available)
        self.assertFalse(service._grasp_accept_event.is_set())

    def test_auto_loop_enables_manual_accept_for_grasp_only(self) -> None:
        source = inspect.getsource(vla_service.VLAService._auto_loop)
        grasp_at = source.index("if not fixed_policy and used_lstm:")
        release_at = source.index('self._stage("gate_release")', grasp_at)

        self.assertIn("allow_manual_accept=True", source[grasp_at:release_at])
        self.assertNotIn("allow_manual_accept=True", source[release_at:])

    def test_dead_motor_telemetry_revokes_manual_accept_immediately(
        self,
    ) -> None:
        service = vla_service.VLAService()
        service._running = True
        service._stage = lambda name: setattr(service, "auto_stage", name)
        service._emit_log = lambda _level, _message: None
        service._emit_status = lambda: None
        availability_seen_after_dropout: list[bool] = []
        closed = {
            "status": "running",
            "step": 306,
            "goal_step": 306,
            "action_start_pos": [3072, 3082, 2047],
            "motor_pos": [2161, 2187, 1230],
            "goal_pos": [2160, 2183, 1170],
            "tactile": [2374, 1253, 1382],
        }
        dead = {
            "status": "running",
            "step": 307,
            "goal_step": 307,
            "action_start_pos": [3072, 3082, 2047],
            "motor_pos": [0, 0, 0],
            "goal_pos": [2160, 2183, 1170],
            "tactile": [2374, 1253, 1382],
        }

        class DropoutGrip:
            calls = 0

            def lstm_status(self):
                self.calls += 1
                if self.calls >= 3:
                    availability_seen_after_dropout.append(
                        service.grasp_accept_available
                    )
                return closed if self.calls == 1 else dead

        clock = FakeClock()
        with patch.object(vla_service.time, "time", clock.time), patch.object(
            vla_service.time, "sleep", clock.sleep
        ):
            settled = service._wait_fingers_settled(
                DropoutGrip(),
                timeout_s=0.75,
                settle_ticks=12,
                hold_s=10.0,
                min_step=20,
                require_displaced=True,
                what="抓取",
                nxt="上抬",
                min_tracking_fingers=3,
                allow_manual_accept=True,
            )

        self.assertFalse(settled)
        self.assertEqual(availability_seen_after_dropout, [False])

    def test_stop_revokes_manual_grasp_accept(self) -> None:
        service = vla_service.VLAService()
        service._running = True
        service.grasp_accept_available = True
        service._grasp_accept_event.set()
        service._emit_log = lambda _level, _message: None

        service.stop()

        self.assertFalse(service.grasp_accept_available)
        self.assertFalse(service._grasp_accept_event.is_set())

    def test_stop_revokes_manual_release_completion(self) -> None:
        service = vla_service.VLAService()
        service._running = True
        service.release_accept_available = True
        service._release_accept_event.set()
        service.release_home_available = True
        service._release_home_event.set()
        service._manual_release_requested = True
        service._manual_release_position = [2872, 3072, 2048]
        service._emit_log = lambda _level, _message: None

        service.stop()

        self.assertFalse(service.release_accept_available)
        self.assertFalse(service._release_accept_event.is_set())
        self.assertFalse(service.release_home_available)
        self.assertFalse(service._release_home_event.is_set())
        self.assertFalse(service._manual_release_requested)
        self.assertIsNone(service._manual_release_position)

    def test_release_wait_has_no_elapsed_deadline(self) -> None:
        source = inspect.getsource(vla_service.VLAService._auto_loop)
        release_source = source[source.index('self._stage("gate_release")'):]

        self.assertIn("timeout_s=None", release_source)
        self.assertIn("allow_manual_release=True", release_source)
        self.assertIn("finger_settle_tracking_ticks", release_source)
        self.assertNotIn("release_settle_timeout_s", release_source)

    def test_demo_config_uses_tracking_tolerance_without_release_timeout(
        self,
    ) -> None:
        auto_cfg = runtime_config._load_yaml(
            ROOT / "config" / "demo_config.yaml"
        )["vla_auto"]

        self.assertEqual(auto_cfg["finger_settle_tracking_ticks"], 30)
        self.assertNotIn("release_settle_timeout_s", auto_cfg)

    def test_goal_ready_wait_has_no_elapsed_deadline(self) -> None:
        statuses = [
            {
                "status": "running",
                "step": 31,
                "goal_ready": False,
                "goal_pos": [3060, 3054, 2044],
                "action_start_pos": [3079, 3038, 2047],
            },
            {
                "status": "running",
                "step": 33,
                "goal_ready": True,
                "goal_pos": [2950, 2920, 1900],
                "action_start_pos": [3079, 3038, 2047],
            },
        ]
        service = vla_service.VLAService()
        service._running = True
        service._stage = lambda _name: None
        service._emit_log = lambda _level, _message: None
        clock = FakeClock()

        with patch.object(vla_service.time, "sleep", clock.sleep):
            ready = service._wait_grasp_goal_ready(SequenceGrip(statuses))

        self.assertTrue(ready)
        self.assertGreaterEqual(clock.now, 0.25)


class AGXActionStartTests(unittest.TestCase):
    @staticmethod
    def make_runner(preloaded_object: str | None = None):
        runner_class = load_hardware_free_lstm_runner()

        class FakeFingers:
            def __init__(self) -> None:
                self.lstm_busy = False
                self.home_calls = 0

            def snapshot(self):
                return [3079, 3038, 2047], [2453, 1332, 1542]

            def home(self) -> None:
                self.home_calls += 1

        runner = runner_class(FakeFingers())
        runner.loaded = {}
        runner.preloaded_object = preloaded_object
        runner.status = "ready"
        runner.action = None
        runner.step = 0
        runner.last_goal_step = 0
        runner.last_goal_pos = None
        runner.error = ""
        runner._stop = threading.Event()
        runner._thread = None
        runner._lock = threading.Lock()
        runner.load_action = lambda action: runner.loaded.setdefault(action, {})
        runner._loop = lambda _action: None
        return runner

    def test_release_actions_never_home_before_lstm(self) -> None:
        for action in ("knife_release", "trapezoid_release", "PCB_release"):
            with self.subTest(action=action):
                runner = self.make_runner()

                result = runner.start(action)
                runner._thread.join(timeout=1.0)

                self.assertTrue(result["ok"])
                self.assertEqual(runner.f.home_calls, 0)
                self.assertEqual(runner.action_start_pos, runner.f.snapshot()[0])

    def test_matching_preloaded_grasp_skips_home(self) -> None:
        runner = self.make_runner(preloaded_object="butter_knife")

        result = runner.start("knife_grasp")
        runner._thread.join(timeout=1.0)

        self.assertTrue(result["ok"])
        self.assertEqual(runner.f.home_calls, 0)
        self.assertIsNone(runner.preloaded_object)

    def test_unprepared_grasp_keeps_home_fallback(self) -> None:
        runner = self.make_runner()

        result = runner.start("knife_grasp")
        runner._thread.join(timeout=1.0)

        self.assertTrue(result["ok"])
        self.assertEqual(runner.f.home_calls, 1)

    def test_direct_grasp_records_measured_action_start_position(self) -> None:
        runner = self.make_runner(preloaded_object="butter_knife")

        result = runner.start("knife_grasp")
        runner._thread.join(timeout=1.0)

        self.assertTrue(result["ok"])
        self.assertEqual(
            runner.action_start_pos,
            [3079, 3038, 2047],
        )


class AGXTactileParserTests(unittest.TestCase):
    def test_split_serial_frame_is_not_published_until_newline(self) -> None:
        parser = load_hardware_free_tactile_parser()(minimum=0, maximum=4095)

        self.assertIsNone(parser.feed(b"2341 1334 1"))
        self.assertEqual(parser.feed(b"498\n"), [2341, 1334, 1498])

    def test_latest_complete_frame_wins_and_trailing_fragment_is_kept(
        self,
    ) -> None:
        parser = load_hardware_free_tactile_parser()(minimum=0, maximum=4095)

        sample = parser.feed(b"2300 1300 1400\n2310 1310 1410\n23")

        self.assertEqual(sample, [2310, 1310, 1410])
        self.assertEqual(parser.feed(b"20 1320 1420\n"), [2320, 1320, 1420])

    def test_out_of_range_frame_is_ignored(self) -> None:
        parser = load_hardware_free_tactile_parser()(minimum=0, maximum=4095)

        self.assertEqual(
            parser.feed(b"2346 1295 1530\n"),
            [2346, 1295, 1530],
        )
        self.assertIsNone(parser.feed(b"47764 1305 1495\n"))


class GripperClientTwoPhaseTests(unittest.TestCase):
    def test_prepare_and_activate_use_dedicated_agx_routes(self) -> None:
        grip = gripper_service.GripperService.__new__(
            gripper_service.GripperService
        )
        calls: list[tuple[str, dict | None, float]] = []

        def record(path, payload, timeout):
            calls.append((path, payload, timeout))
            return {"ok": True}

        grip._lstm_post = record

        prepared = grip.lstm_prepare(
            "knife_grasp",
            min_close_ticks=60,
            min_close_fingers=2,
        )
        activated = grip.lstm_activate()

        self.assertEqual(prepared, {"ok": True})
        self.assertEqual(activated, {"ok": True})
        self.assertEqual(
            calls,
            [
                (
                    "/lstm/prepare",
                    {
                        "action": "knife_grasp",
                        "min_close_ticks": 60,
                        "min_close_fingers": 2,
                    },
                    5.0,
                ),
                ("/lstm/activate", None, 5.0),
            ],
        )


class AGXPreparedGraspTests(unittest.TestCase):
    @staticmethod
    def make_runner(
        goal: list[float],
        start_pos: list[int] | None = None,
    ):
        runner_class = load_hardware_free_lstm_runner()
        measured_start = list(start_pos or [3079, 3038, 2047])

        class FakeFingers:
            def __init__(self) -> None:
                self.lstm_busy = False
                self.home_calls = 0
                self.writes: list[list[int]] = []
                self.written = threading.Event()

            def snapshot(self):
                return list(measured_start), [2453, 1332, 1542]

            def home(self) -> None:
                self.home_calls += 1

            def write_goals(self, target) -> None:
                self.writes.append(list(target))
                self.written.set()

            def freeze(self) -> None:
                pass

        class IdentityInputScaler:
            @staticmethod
            def transform(rows):
                return np.asarray(rows, dtype=np.float32)

        class FixedOutputScaler:
            @staticmethod
            def inverse_transform(_rows):
                return np.asarray([goal], dtype=np.float32)

        class ZeroModel:
            @staticmethod
            def __call__(_inputs, _lengths):
                return torch.zeros((1, 1, 3), dtype=torch.float32)

        fingers = FakeFingers()
        runner = runner_class(fingers)
        runner.preloaded_object = "butter_knife"
        runner.loaded = {
            "knife_grasp": {
                "model": ZeroModel(),
                "sx": IdentityInputScaler(),
                "sy": FixedOutputScaler(),
            }
        }
        return runner, fingers

    def test_observed_incremental_goal_uses_smaller_prepare_threshold(self) -> None:
        observed_start = [3081, 3116, 2054]
        incremental_goal = [3055.0, 3072.0, 2036.0]

        strict_runner, strict_fingers = self.make_runner(
            incremental_goal, observed_start
        )
        try:
            result = strict_runner.prepare(
                "knife_grasp", min_close_ticks=60, min_close_fingers=2
            )

            self.assertTrue(result["ok"])
            self.assertFalse(strict_runner._goal_ready.wait(timeout=1.3))
            self.assertEqual(strict_fingers.writes, [])
        finally:
            strict_runner.stop()

        runner, fingers = self.make_runner(
            incremental_goal, observed_start
        )
        try:
            result = runner.prepare(
                "knife_grasp", min_close_ticks=20, min_close_fingers=2
            )

            self.assertTrue(result["ok"])
            self.assertTrue(runner._goal_ready.wait(timeout=2.5))
            self.assertEqual(fingers.writes, [])
            self.assertEqual(runner.payload()["action_start_pos"], observed_start)

            activated = runner.activate()

            self.assertTrue(activated["ok"])
            self.assertTrue(fingers.written.wait(timeout=0.5))
            self.assertEqual(fingers.writes[0], [3055, 3072, 2036])
        finally:
            runner.stop()

    def test_prepare_latches_real_close_without_writing_until_activate(self) -> None:
        runner, fingers = self.make_runner([2950.0, 2920.0, 1900.0])
        try:
            result = runner.prepare(
                "knife_grasp", min_close_ticks=60, min_close_fingers=2
            )

            self.assertTrue(result["ok"])
            self.assertTrue(runner._goal_ready.wait(timeout=2.5))
            self.assertEqual(fingers.writes, [])
            self.assertEqual(runner.payload()["drive_state"], "ready")

            activated = runner.activate()

            self.assertTrue(activated["ok"])
            self.assertTrue(fingers.written.wait(timeout=0.5))
            self.assertEqual(fingers.writes[0], [2950, 2920, 1900])
            self.assertEqual(runner.payload()["drive_state"], "active")
        finally:
            runner.stop()

    def test_stop_invalidates_cached_goal_before_late_activate(self) -> None:
        runner, fingers = self.make_runner([2950.0, 2920.0, 1900.0])
        try:
            result = runner.prepare(
                "knife_grasp", min_close_ticks=60, min_close_fingers=2
            )
            self.assertTrue(result["ok"])
            self.assertTrue(runner._goal_ready.wait(timeout=2.5))
            self.assertEqual(fingers.writes, [])

            runner.stop()
            late_activate = runner.activate()

            self.assertFalse(late_activate["ok"])
            self.assertEqual(fingers.writes, [])
            self.assertEqual(runner.payload()["drive_state"], "holding")
            self.assertFalse(runner.payload()["goal_ready"])
            self.assertIsNone(runner.payload()["pending_goal"])
        finally:
            if runner.status == "running":
                runner.stop()

    def test_prepare_near_home_goal_never_becomes_ready_or_writes(self) -> None:
        runner, fingers = self.make_runner([3060.0, 3054.0, 2044.0])
        try:
            result = runner.prepare(
                "knife_grasp", min_close_ticks=60, min_close_fingers=2
            )

            self.assertTrue(result["ok"])
            self.assertFalse(runner._goal_ready.wait(timeout=1.3))
            self.assertEqual(fingers.writes, [])
            self.assertEqual(runner.payload()["drive_state"], "preparing")
        finally:
            runner.stop()


class AGXGoalTelemetryTests(unittest.TestCase):
    def test_loop_records_the_goal_that_payload_exposes(self) -> None:
        runner_class = load_hardware_free_lstm_runner()
        runner = runner_class.__new__(runner_class)
        stop = threading.Event()

        class FakeFingers:
            lstm_busy = True

            def __init__(self) -> None:
                self.writes: list[list[int]] = []

            def snapshot(self):
                return [3072, 3072, 2048], [2400, 1300, 1500]

            def write_goals(self, target):
                self.writes.append(list(target))
                stop.set()

        class IdentityInputScaler:
            @staticmethod
            def transform(rows):
                return np.asarray(rows, dtype=np.float32)

        class FixedOutputScaler:
            @staticmethod
            def inverse_transform(_rows):
                return np.asarray([[2800.0, 2810.0, 1800.0]], dtype=np.float32)

        class ZeroModel:
            @staticmethod
            def __call__(_inputs, _lengths):
                return torch.zeros((1, 1, 3), dtype=torch.float32)

        clock = FakeClock()
        original_time = runner._loop.__globals__.get("time")
        original_sleep = runner._loop.__globals__.get("sleep")
        runner._loop.__globals__["time"] = clock.time
        runner._loop.__globals__["sleep"] = clock.sleep
        runner.f = FakeFingers()
        runner.loaded = {
            "knife_grasp": {
                "model": ZeroModel(),
                "sx": IdentityInputScaler(),
                "sy": FixedOutputScaler(),
            }
        }
        runner.status = "running"
        runner.action = "knife_grasp"
        runner.step = 0
        runner.error = ""
        runner.device = torch.device("cpu")
        runner.preloaded_object = "butter_knife"
        runner._stop = stop
        try:
            runner._loop("knife_grasp")
        finally:
            runner._loop.__globals__["time"] = original_time
            runner._loop.__globals__["sleep"] = original_sleep

        self.assertEqual(runner.error, "")
        self.assertEqual(runner.f.writes, [[2800, 2810, 1800]])
        self.assertEqual(runner.last_goal_pos, [2800, 2810, 1800])
        self.assertEqual(runner.last_goal_step, 1)
        payload = runner.payload()
        self.assertEqual(payload["goal_pos"], [2800, 2810, 1800])
        self.assertEqual(payload["goal_step"], 1)


if __name__ == "__main__":
    unittest.main()
