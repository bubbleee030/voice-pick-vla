from __future__ import annotations

import inspect
import runpy
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import runtime_config
from src.services import gripper_service, vla_service
from tools import voice_pick_demo


HOME = [3072, 3072, 2048]


def demo_config() -> dict:
    return runtime_config._load_yaml(ROOT / "config" / "demo_config.yaml")


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FixedFallbackConfigTests(unittest.TestCase):
    def test_unset_trapezoid_goal_rejects_before_run_build(self) -> None:
        with self.assertRaisesRegex(
            runtime_config.FixedFallbackConfigError,
            r"fixed_fallback\.objects\.trapezoid\.fixed_grasp_position",
        ):
            runtime_config.build_fixed_fallback_cfg(
                demo_config(), "trapezoid"
            )

    def test_unset_knife_goal_rejects_before_run_build(self) -> None:
        with self.assertRaisesRegex(
            runtime_config.FixedFallbackConfigError,
            r"fixed_fallback\.objects\.butter_knife\.fixed_grasp_position",
        ):
            runtime_config.build_fixed_fallback_cfg(
                demo_config(), "butter_knife"
            )

    def test_board_requires_no_grasp_goal(self) -> None:
        cfg = demo_config()
        fixed = cfg["fixed_fallback"]
        board = fixed["objects"]["board"]
        run_cfg, tail_cfg = runtime_config.build_fixed_fallback_cfg(
            cfg, "board"
        )

        self.assertEqual(
            run_cfg["fixed_xy"],
            [board["pick_x_mm"], board["pick_y_mm"]],
        )
        self.assertEqual(
            run_cfg["travel_speed_percent"],
            fixed["travel_speed_percent"],
        )
        self.assertEqual(
            run_cfg["descend_speed_percent"],
            fixed["descend_speed_percent"],
        )
        self.assertEqual(tail_cfg["grasp_z_mm"], board["grasp_z_mm"])
        self.assertEqual(
            tail_cfg["grasp_contact_dwell_s"],
            board["contact_dwell_s"],
        )
        self.assertFalse(tail_cfg["requires_grasp"])
        self.assertIsNone(tail_cfg["fixed_grasp_position"])
        self.assertTrue(tail_cfg["auto_lift_after_dwell"])
        self.assertEqual(tail_cfg["lstm"], {})

    def test_editing_xy_changes_target_but_not_constant_z(self) -> None:
        cfg = demo_config()
        cfg["fixed_fallback"]["objects"]["board"]["pick_x_mm"] = 512.5
        cfg["fixed_fallback"]["objects"]["board"]["pick_y_mm"] = 207.25

        run_cfg, tail_cfg = runtime_config.build_fixed_fallback_cfg(
            cfg, "board"
        )

        self.assertEqual(run_cfg["fixed_xy"], [512.5, 207.25])
        self.assertEqual(
            tail_cfg["grasp_z_mm"],
            cfg["fixed_fallback"]["objects"]["board"]["grasp_z_mm"],
        )

    def test_valid_fixed_grasp_builds_no_lstm_policy(self) -> None:
        cfg = demo_config()
        cfg["fixed_fallback"]["objects"]["trapezoid"][
            "fixed_grasp_position"
        ] = [2140, 2150, 1130]

        run_cfg, tail_cfg = runtime_config.build_fixed_fallback_cfg(
            cfg, "trapezoid"
        )

        self.assertTrue(run_cfg["fixed_fallback"])
        self.assertEqual(run_cfg["fixed_xy"], [490.116, 177.869])
        self.assertEqual(tail_cfg["grasp_contact_dwell_s"], 0.0)
        self.assertEqual(tail_cfg["finger_policy"], "fixed")
        self.assertEqual(tail_cfg["lstm"], {})

    def test_invalid_motor_goal_is_rejected_with_exact_key(self) -> None:
        cfg = demo_config()
        cfg["fixed_fallback"]["objects"]["trapezoid"][
            "fixed_grasp_position"
        ] = [1, 2150, 1130]

        with self.assertRaisesRegex(
            runtime_config.FixedFallbackConfigError,
            r"fixed_grasp_position\[0\]",
        ):
            runtime_config.build_fixed_fallback_cfg(cfg, "trapezoid")

    def test_invalid_shared_speed_is_rejected_before_run_build(self) -> None:
        cfg = demo_config()
        cfg["fixed_fallback"]["travel_speed_percent"] = 0

        with self.assertRaisesRegex(
            runtime_config.FixedFallbackConfigError,
            r"fixed_fallback\.travel_speed_percent",
        ):
            runtime_config.build_fixed_fallback_cfg(cfg, "board")

    def test_normal_auto_uses_fixed_geometry_and_keeps_lstm_actions(self) -> None:
        cfg = demo_config()
        with patch.object(
            runtime_config,
            "load_taught_fallback",
            side_effect=AssertionError("legacy taught data must not be read"),
        ):
            resolved = runtime_config.build_auto_run_cfg(
                cfg, {"fallback": {}}, "butter_knife"
            )

        self.assertIsNotNone(resolved)
        run_cfg, tail_cfg, _ = resolved
        self.assertEqual(run_cfg["recorded_xy"], [490.1, 0.0])
        self.assertEqual(
            tail_cfg["grasp_z_mm"],
            float(cfg["fixed_fallback"]["objects"]["butter_knife"]["grasp_z_mm"]),
        )
        self.assertEqual(tail_cfg["place_z_mm"], 170.0)
        self.assertEqual(
            tail_cfg["lstm"],
            {
                "object": "butter_knife",
                "grasp_action": "knife_grasp",
                "release_action": "knife_release",
            },
        )


class CancellableGripperPositionTests(unittest.TestCase):
    def make_gripper(self):
        grip = gripper_service.GripperService.__new__(
            gripper_service.GripperService
        )
        grip.grip_cfg = {
            "open_step_ticks": 24,
            "close_step_ticks": 10,
            "open_step_delay_s": 0.0,
        }
        grip.last_error = ""
        states = iter(
            [
                {"current_pos": list(HOME)},
                {"current_pos": [3062, 3062, 2038]},
            ]
        )
        grip.get_state = lambda: next(
            states, {"current_pos": [3062, 3062, 2038]}
        )
        writes: list[list[int]] = []
        grip._set_position_direct = (
            lambda goal: writes.append(list(goal)) or True
        )
        return grip, writes

    def test_stop_cancels_future_steps_and_holds_measured_position(self) -> None:
        grip, writes = self.make_gripper()
        checks = iter([False, True])

        ok = grip._set_position_stepped(
            [3000, 3000, 1900],
            stop_requested=lambda: next(checks, True),
        )

        self.assertFalse(ok)
        self.assertEqual(writes[0], [3062, 3062, 2038])
        self.assertEqual(writes[-1], [3062, 3062, 2038])
        self.assertEqual(grip.last_error, "position move stopped")

    def test_existing_callers_can_omit_stop_callback(self) -> None:
        signature = inspect.signature(
            gripper_service.GripperService.set_position
        )
        self.assertIsNone(signature.parameters["stop_requested"].default)


class SequenceFixedGrip:
    def __init__(self, positions: list[list[int]]) -> None:
        self.positions = iter(positions)
        self.last = positions[-1]
        self.commands: list[tuple[list[int], str | None, object]] = []
        self.last_error = ""

    def get_state(self):
        self.last = next(self.positions, self.last)
        return {"current_pos": list(self.last)}

    def set_position(self, goal, mode=None, stop_requested=None):
        self.commands.append((list(goal), mode, stop_requested))
        return True


class FixedFingerConvergenceTests(unittest.TestCase):
    def make_service(self) -> vla_service.VLAService:
        service = vla_service.VLAService()
        service._running = True
        service._stage = lambda _name: None
        service._emit_log = lambda _level, _message: None
        return service

    def test_fixed_grasp_requires_close_and_target_convergence(self) -> None:
        service = self.make_service()
        clock = FakeClock()
        grip = SequenceFixedGrip([[2800, 2810, 1800]] * 20)

        with patch.object(vla_service.time, "time", clock.time), patch.object(
            vla_service.time, "sleep", clock.sleep
        ):
            settled = service._wait_fixed_fingers_settled(
                grip,
                target=[2800, 2810, 1800],
                start_pos=list(HOME),
                require_close=True,
                what="抓取",
                tolerance_ticks=30,
                settle_ticks=12,
                hold_s=1.2,
                min_close_ticks=60,
                min_close_fingers=2,
            )

        self.assertTrue(settled)
        self.assertGreaterEqual(clock.now, 1.2)

    def test_fixed_release_can_wait_beyond_eight_seconds(self) -> None:
        service = self.make_service()
        clock = FakeClock()
        grip = SequenceFixedGrip([list(HOME)] * 100)

        with patch.object(vla_service.time, "time", clock.time), patch.object(
            vla_service.time, "sleep", clock.sleep
        ):
            settled = service._wait_fixed_fingers_settled(
                grip,
                target=list(HOME),
                start_pos=[2800, 2810, 1800],
                require_close=False,
                what="放開",
                tolerance_ticks=30,
                settle_ticks=12,
                hold_s=9.0,
                min_close_ticks=60,
                min_close_fingers=2,
            )

        self.assertTrue(settled)
        self.assertGreaterEqual(clock.now, 9.0)

    def test_missing_motor_state_is_failure_not_false_success(self) -> None:
        service = self.make_service()
        grip = SequenceFixedGrip([[0, 0, 0]])

        with self.assertRaisesRegex(RuntimeError, "motor position unavailable"):
            service._wait_fixed_fingers_settled(
                grip,
                target=list(HOME),
                start_pos=[2800, 2810, 1800],
                require_close=False,
                what="放開",
                tolerance_ticks=30,
                settle_ticks=12,
                hold_s=1.2,
                min_close_ticks=60,
                min_close_fingers=2,
            )


class FakeFixedGrip:
    def __init__(self, events: list[tuple], fail_grasp: bool = False) -> None:
        self.events = events
        self.current = list(HOME)
        self.last_error = ""
        self.fail_grasp = fail_grasp
        self.arm = None
        self.grasp_commanded = threading.Event()
        self.hold_calls = 0

    def get_state(self):
        return {"current_pos": list(self.current)}

    def set_position(self, goal, mode=None, stop_requested=None):
        target = [int(value) for value in goal]
        active = bool(self.arm and self.arm.ctrl.lift_active.is_set())
        z_mm = float(self.arm.ctrl.read_current_pose_mm_deg()[2])
        self.events.append(("finger", target, active, z_mm, mode))
        if target != HOME and self.fail_grasp:
            self.last_error = "injected AGX write failure"
            return False
        self.current = target
        if target != HOME:
            self.grasp_commanded.set()
        return True

    def _hold_measured_position(self) -> bool:
        self.hold_calls += 1
        self.events.append(("finger_hold", list(self.current)))
        return True

    def __getattr__(self, name):
        if name.startswith("lstm_"):
            raise AssertionError(f"fixed fallback called LSTM method {name}")
        raise AttributeError(name)


class FakeFixedCtrl:
    def __init__(self, events: list[tuple], grip: FakeFixedGrip) -> None:
        self.events = events
        self.grip = grip
        self.pose = [490.0, 0.0, 425.0, 180.0, 0.0, 0.0]
        self.lift_active = threading.Event()
        self.motion_stopped = threading.Event()
        self._lock = threading.Lock()

    def read_current_pose_mm_deg(self):
        with self._lock:
            if self.lift_active.is_set() and self.pose[2] < 170.0:
                self.pose[2] = min(170.0, self.pose[2] + 22.0)
            return list(self.pose)

    def reset_alarms(self) -> None:
        pass

    def servo_on(self) -> None:
        pass

    def motion_stop(self) -> None:
        self.events.append(("motion_stop",))
        self.motion_stopped.set()

    def set_pose_from_raw(self, raw: list[int]) -> None:
        with self._lock:
            self.pose[:3] = [raw[0] / 1000.0, raw[1] / 1000.0, raw[2] / 1000.0]


class FakeFixedArm:
    def __init__(self, events: list[tuple], grip: FakeFixedGrip) -> None:
        self.events = events
        self.gripper = grip
        self.ctrl = FakeFixedCtrl(events, grip)
        self._polling = False
        grip.arm = self

    def ensure_connected(self) -> bool:
        return True

    def stop_pose_polling(self) -> None:
        pass

    def start_pose_polling(self) -> None:
        pass

    def check_safety(self, _raw):
        return True, ""

    def _fb_move(
        self, label, raw, speed, _step_timeout, _start_timeout
    ) -> None:
        self.events.append(("move", label, raw[2] / 1000.0, speed))
        if label == "grasp_lift":
            self.ctrl.lift_active.set()
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if (
                    self.gripper.grasp_commanded.is_set()
                    or self.ctrl.motion_stopped.is_set()
                ):
                    break
                time.sleep(0.005)
            if not self.ctrl.motion_stopped.is_set():
                self.ctrl.set_pose_from_raw(raw)
            self.ctrl.lift_active.clear()
            return
        self.ctrl.set_pose_from_raw(raw)

    def go_ready(self) -> None:
        self.events.append(("ready",))

    def motion_stop(self) -> None:
        self.ctrl.motion_stop()


class FixedFallbackFlowTests(unittest.TestCase):
    def make_run(self, object_key: str, *, fail_grasp: bool = False):
        events: list[tuple] = []
        grip = FakeFixedGrip(events, fail_grasp=fail_grasp)
        arm = FakeFixedArm(events, grip)
        service = vla_service.VLAService(arm_service=arm)
        service._running = True
        service._emit_log = lambda _level, _message: None
        service._emit_status = lambda: None
        service._wait_gate = lambda name: (_ for _ in ()).throw(
            AssertionError(f"fixed fallback reached gate {name}")
        )
        service._ensure_models = lambda *_args: (_ for _ in ()).throw(
            AssertionError("fixed fallback loaded VLA model")
        )
        service._load_lang_embed = lambda *_args: (_ for _ in ()).throw(
            AssertionError("fixed fallback loaded language embedding")
        )
        service._predict_target = lambda *_args: (_ for _ in ()).throw(
            AssertionError("fixed fallback used camera/model prediction")
        )
        service._wait_grasp_contact_dwell = (
            lambda seconds: events.append(("dwell", float(seconds))) or True
        )

        xy_by_object = {
            "trapezoid": [490.116, 177.869],
            "board": [505.0, 214.0],
            "butter_knife": [490.1, 0.0],
        }
        z_by_object = {
            "trapezoid": 170.725,
            "board": 124.0,
            "butter_knife": 126.0,
        }
        dwell_by_object = {
            "trapezoid": 0.0,
            "board": 3.0,
            "butter_knife": 3.0,
        }
        requires_grasp = object_key != "board"
        run_cfg = {
            "fixed_fallback": True,
            "fixed_xy": xy_by_object[object_key],
            "hover_z_mm": 425.0,
            "travel_speed_percent": 70,
            "descend_speed_percent": 30,
            "per_step_timeout_s": 25.0,
            "motion_start_timeout_s": 3.0,
            "confirm_descend": False,
            "fixed_target_tolerance_ticks": 30,
            "fixed_settle_ticks": 12,
            "fixed_settle_hold_s": 0.0,
            "fixed_min_close_ticks": 60,
            "fixed_min_close_fingers": 2,
        }
        tail_cfg = {
            "finger_policy": "fixed",
            "requires_grasp": requires_grasp,
            "fixed_grasp_position": (
                [2800, 2810, 1800] if requires_grasp else None
            ),
            "fixed_release_position": list(HOME),
            "grasp_z_mm": z_by_object[object_key],
            "grasp_contact_dwell_s": dwell_by_object[object_key],
            "auto_lift_after_dwell": object_key == "board",
            "lift_during_grasp": object_key == "butter_knife",
            "grasp_start_z_mm": (
                170.0 if object_key == "butter_knife" else None
            ),
            "grasp_backoff_mm": 0.0,
            "place_approach_pose": [
                490116,
                -203991,
                424908,
                179999,
                0,
                0,
            ],
            "place_z_mm": 180.0 if object_key == "trapezoid" else 170.0,
            "lstm": {},
        }
        with patch(
            "src.arrival.ArrivalServo",
            side_effect=AssertionError("fixed fallback constructed vision servo"),
        ):
            service._auto_loop(
                object_key,
                "",
                "",
                "",
                "fixed",
                run_cfg,
                tail_cfg,
                False,
            )
        return service, arm, grip, events

    def test_trapezoid_grasps_immediately_then_lifts(self) -> None:
        service, _arm, _grip, events = self.make_run("trapezoid")

        labels = [event[1] for event in events if event[0] == "move"]
        self.assertEqual(
            labels,
            [
                "fixed_xy",
                "descend",
                "lift",
                "release_point",
                "release_down",
                "release_retract",
            ],
        )
        self.assertFalse(any(event[0] == "dwell" for event in events))
        descend_index = events.index(next(e for e in events if e[:2] == ("move", "descend")))
        grasp_index = events.index(next(e for e in events if e[0] == "finger" and e[1] != HOME))
        lift_index = events.index(next(e for e in events if e[:2] == ("move", "lift")))
        self.assertLess(descend_index, grasp_index)
        self.assertLess(grasp_index, lift_index)
        self.assertEqual(service.status, "stopped")

    def test_board_dwells_then_lifts_without_grasp(self) -> None:
        service, _arm, _grip, events = self.make_run("board")

        self.assertIn(("dwell", 3.0), events)
        self.assertFalse(
            any(event[0] == "finger" and event[1] != HOME for event in events)
        )
        dwell_index = events.index(("dwell", 3.0))
        lift_index = events.index(next(e for e in events if e[:2] == ("move", "lift")))
        self.assertLess(dwell_index, lift_index)
        self.assertEqual(service.status, "stopped")

    def test_knife_commands_grasp_during_single_uninterrupted_lift(self) -> None:
        service, _arm, _grip, events = self.make_run("butter_knife")

        lift_events = [e for e in events if e[:2] == ("move", "grasp_lift")]
        self.assertEqual(len(lift_events), 1)
        self.assertEqual(lift_events[0][2:], (424.908, 70))
        grasp_event = next(
            e for e in events if e[0] == "finger" and e[1] != HOME
        )
        self.assertTrue(grasp_event[2])
        self.assertGreaterEqual(grasp_event[3], 170.0)
        self.assertNotIn("lift", [e[1] for e in events if e[0] == "move"])
        self.assertEqual(service.status, "stopped")

    def test_knife_agx_failure_stops_arm_and_attempts_finger_hold(self) -> None:
        service, arm, grip, events = self.make_run(
            "butter_knife", fail_grasp=True
        )

        self.assertEqual(service.status, "error")
        self.assertTrue(arm.ctrl.motion_stopped.is_set())
        self.assertGreaterEqual(grip.hold_calls, 1)
        self.assertIn(("motion_stop",), events)


class FixedFallbackIntegrationTests(unittest.TestCase):
    def test_fixed_mode_has_no_lstm_or_vision_startup(self) -> None:
        source = inspect.getsource(vla_service.VLAService._auto_loop)
        self.assertIn('finger_policy == "fixed"', source)
        self.assertIn('mode == "fixed"', source)
        self.assertIn("_command_fixed_finger_goal", source)

    def test_knife_fixed_grasp_reuses_live_lift_trigger_helper(self) -> None:
        source = inspect.getsource(vla_service.VLAService._auto_loop)
        fixed_start = source.index('finger_policy == "fixed"')
        fixed_source = source[fixed_start:]
        self.assertIn("_lift_and_start_grasp_at_z", fixed_source)
        self.assertIn("fixed_grasp_position", fixed_source)

    def test_fixed_release_uses_measured_convergence_without_gate(self) -> None:
        source = inspect.getsource(vla_service.VLAService._auto_loop)
        release_source = source[source.index('self._stage("gate_release")'):]
        self.assertIn("_command_fixed_finger_goal", release_source)
        self.assertIn("fixed_release_position", release_source)

    def test_server_fallback_no_longer_calls_legacy_arm_runner(self) -> None:
        source = inspect.getsource(voice_pick_demo.create_app)
        handler = source[
            source.index("def on_fallback_run"):
            source.index('@socketio.on("teach_start")')
        ]
        self.assertIn("build_fixed_fallback_cfg", handler)
        self.assertIn('mode="fixed"', handler)
        self.assertIn("gripper.get_state", handler)
        self.assertNotIn("arm.run_fallback", handler)

    def test_all_arm_stop_handlers_cancel_shared_runner(self) -> None:
        source = inspect.getsource(voice_pick_demo.create_app)
        self.assertGreaterEqual(source.count("vla.stop()"), 3)

    def test_local_cli_ctrl_c_stops_arm_motion_too(self) -> None:
        source = (ROOT / "scripts" / "auto_run_cli.py").read_text(
            encoding="utf-8"
        )
        handler = source[source.rindex("except KeyboardInterrupt:"):]
        self.assertIn("vla.stop()", handler)
        self.assertIn("arm.motion_stop()", handler)

    def test_cli_enter_poll_is_non_blocking(self) -> None:
        namespace = runpy.run_path(str(ROOT / "scripts" / "auto_run_cli.py"))
        poll_terminal_enter = namespace["poll_terminal_enter"]
        stream = Mock()
        stream.readline.return_value = "\n"

        with patch("select.select", return_value=([], [], [])):
            self.assertFalse(poll_terminal_enter(stream))
            stream.readline.assert_not_called()

        with patch("select.select", return_value=([stream], [], [])):
            self.assertTrue(poll_terminal_enter(stream))
            stream.readline.assert_called_once_with()

    def test_server_bridges_guarded_grasp_accept_event(self) -> None:
        source = inspect.getsource(voice_pick_demo.create_app)
        handler_at = source.index('@socketio.on("vla_accept_grasp")')
        handler = source[handler_at:source.index('@socketio.on("vla_auto_start")')]

        self.assertIn("vla.accept_grasp()", handler)
        self.assertIn('emit("vla_status"', handler)

    def test_server_bridges_contextual_finger_completion_event(self) -> None:
        source = inspect.getsource(voice_pick_demo.create_app)
        handler_at = source.index(
            '@socketio.on("vla_complete_finger_stage")'
        )
        handler = source[
            handler_at:source.index('@socketio.on("vla_auto_start")')
        ]

        self.assertIn("vla.complete_finger_stage()", handler)
        self.assertIn('emit("vla_status"', handler)

    def test_cli_wires_guarded_grasp_accept_for_attached_and_local_runs(
        self,
    ) -> None:
        source = (ROOT / "scripts" / "auto_run_cli.py").read_text(
            encoding="utf-8"
        )

        self.assertIn('d.get("grasp_accept_available")', source)
        self.assertIn('sio.emit("vla_accept_grasp")', source)
        self.assertIn("vla.grasp_accept_available", source)
        self.assertIn("vla.accept_grasp()", source)

    def test_cli_wires_contextual_release_actions_for_attached_and_local_runs(
        self,
    ) -> None:
        source = (ROOT / "scripts" / "auto_run_cli.py").read_text(
            encoding="utf-8"
        )

        self.assertIn('d.get("finger_stage_action")', source)
        self.assertIn('"release_pose"', source)
        self.assertIn('"release_home"', source)
        self.assertIn(
            'sio.emit("vla_complete_finger_stage")',
            source,
        )
        self.assertIn(
            'vla.vla_status_payload()["finger_stage_action"]',
            source,
        )
        self.assertIn("vla.complete_finger_stage()", source)


class FixedFallbackUiTests(unittest.TestCase):
    def test_both_uis_offer_the_three_supported_objects(self) -> None:
        for relative in (
            "tools/static/index.html",
            "tools/static/index.classic.html",
        ):
            html = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn('id="btn-fallback-trapezoid"', html)
            self.assertIn('id="btn-fallback-board"', html)
            self.assertIn('id="btn-fallback-butter_knife"', html)
            self.assertNotIn('id="btn-fallback-tweezers"', html)

    def test_both_clients_emit_butter_knife_key(self) -> None:
        modern = (ROOT / "tools/static/app.js").read_text(encoding="utf-8")
        classic = (
            ROOT / "tools/static/app.classic.js"
        ).read_text(encoding="utf-8")
        self.assertIn('object_key: "butter_knife"', modern)
        self.assertIn('"butter_knife"', classic)


if __name__ == "__main__":
    unittest.main()
