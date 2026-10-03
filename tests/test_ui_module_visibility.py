from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import runtime_config


class ModuleVisibilityPrecedenceTests(unittest.TestCase):
    def test_full_profile_keeps_base_teach_hidden_but_enabled(self) -> None:
        bundle = runtime_config.load_runtime_bundle(
            profile="full", env_config=None
        )
        self.assertEqual(
            bundle.demo_config["modules"]["teach"],
            {"enabled": True, "show_in_ui": False},
        )

    def test_recording_disables_voice_but_base_keeps_it_visible(self) -> None:
        bundle = runtime_config.load_runtime_bundle(
            profile="recording", env_config=None
        )
        self.assertEqual(
            bundle.demo_config["modules"]["voice"],
            {"enabled": False, "show_in_ui": True},
        )

    def test_environment_changes_enabled_but_not_base_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            overlay = Path(directory) / "overlay.yaml"
            overlay.write_text(
                yaml.safe_dump({
                    "modules": {
                        "teach": {
                            "enabled": False,
                            "show_in_ui": True,
                        }
                    }
                }),
                encoding="utf-8",
            )
            bundle = runtime_config.load_runtime_bundle(
                profile="full", env_config=str(overlay)
            )
        self.assertEqual(
            bundle.demo_config["modules"]["teach"],
            {"enabled": False, "show_in_ui": False},
        )

    def test_public_config_publishes_authoritative_visibility(self) -> None:
        bundle = runtime_config.load_runtime_bundle(
            profile="full", env_config=None
        )
        public = runtime_config.public_runtime_config(bundle)
        self.assertFalse(public["modules"]["teach"]["show_in_ui"])
        self.assertTrue(public["modules"]["teach"]["enabled"])

    def test_launch_profiles_no_longer_claim_ui_authority(self) -> None:
        profiles = runtime_config._load_yaml(
            ROOT / "config" / "launch_profiles.yaml"
        )["profiles"]
        for profile in profiles.values():
            modules = (
                profile.get("demo_overrides", {}).get("modules", {})
            )
            for module in modules.values():
                if isinstance(module, dict):
                    self.assertNotIn("show_in_ui", module)


class ExtendedUiVisibilityConfigTests(unittest.TestCase):
    EXPECTED_OBJECTS = ["trapezoid", "board", "butter_knife"]

    def test_demo_modules_hide_vla_and_dataset_but_keep_them_enabled(
        self,
    ) -> None:
        bundle = runtime_config.load_runtime_bundle(
            profile="full", env_config=None
        )
        self.assertEqual(
            bundle.demo_config["modules"]["vla"],
            {"enabled": True, "show_in_ui": False},
        )
        self.assertEqual(
            bundle.demo_config["modules"]["dataset_capture"],
            {"enabled": True, "show_in_ui": False},
        )

    def test_public_config_publishes_quick_pick_controls(self) -> None:
        bundle = runtime_config.load_runtime_bundle(
            profile="full", env_config=None
        )
        public = runtime_config.public_runtime_config(bundle)
        self.assertEqual(
            public["ui"]["controls"]["quick_pick"],
            {
                "show_arm_controls": True,
                "show_reload_config": False,
                "show_catalog_picker": False,
                "fixed_fallback_objects": self.EXPECTED_OBJECTS,
            },
        )

    def test_environment_cannot_override_base_ui_presentation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            overlay = Path(directory) / "overlay.yaml"
            overlay.write_text(
                yaml.safe_dump({
                    "modules": {
                        "vla": {
                            "enabled": False,
                            "show_in_ui": True,
                        },
                    },
                    "ui": {
                        "controls": {
                            "quick_pick": {
                                "show_arm_controls": True,
                                "fixed_fallback_objects": ["tweezers"],
                            },
                        },
                    },
                }),
                encoding="utf-8",
            )
            bundle = runtime_config.load_runtime_bundle(
                profile="full", env_config=str(overlay)
            )

        self.assertEqual(
            bundle.demo_config["modules"]["vla"],
            {"enabled": False, "show_in_ui": False},
        )
        self.assertEqual(
            bundle.demo_config["ui"]["controls"]["quick_pick"],
            {
                "show_arm_controls": True,
                "show_reload_config": False,
                "show_catalog_picker": False,
                "fixed_fallback_objects": self.EXPECTED_OBJECTS,
            },
        )

    def test_missing_controls_preserve_legacy_visible_defaults(self) -> None:
        cfg = {"ui": {"mode": "classic"}}
        runtime_config.normalize_demo_config(cfg)
        self.assertEqual(
            cfg["ui"]["controls"]["quick_pick"],
            {
                "show_arm_controls": True,
                "show_reload_config": True,
                "show_catalog_picker": True,
                "fixed_fallback_objects": self.EXPECTED_OBJECTS,
            },
        )

    def test_allowlist_deduplicates_and_ignores_non_strings(self) -> None:
        cfg = {
            "ui": {
                "controls": {
                    "quick_pick": {
                        "fixed_fallback_objects": [
                            "board",
                            "board",
                            7,
                            "",
                            "future_object",
                        ],
                    },
                },
            },
        }
        runtime_config.normalize_demo_config(cfg)
        self.assertEqual(
            cfg["ui"]["controls"]["quick_pick"][
                "fixed_fallback_objects"
            ],
            ["board", "future_object"],
        )

    def test_explicit_empty_allowlist_stays_empty(self) -> None:
        cfg = {
            "ui": {
                "controls": {
                    "quick_pick": {
                        "fixed_fallback_objects": [],
                    },
                },
            },
        }
        runtime_config.normalize_demo_config(cfg)
        self.assertEqual(
            cfg["ui"]["controls"]["quick_pick"][
                "fixed_fallback_objects"
            ],
            [],
        )


class ModernModuleVisibilityTests(unittest.TestCase):
    def test_modern_dom_declares_module_ownership(self) -> None:
        html = (ROOT / "tools/static/index.html").read_text(encoding="utf-8")
        for marker in (
            'data-module="teach"',
            'data-module="cam1"',
            'data-module="cam2"',
            'data-module="claw_cam"',
            'data-module="arm_monitor"',
            'data-module="trajectory"',
            'data-module="gripper"',
            'data-module="validation"',
            'data-module="quick_pick"',
            'data-module="voice"',
            'data-module="logs"',
            'data-modules-any="cam1 cam2 claw_cam"',
            'data-modules-any="arm_monitor trajectory"',
        ):
            self.assertIn(marker, html)

    def test_modern_profile_update_applies_visibility(self) -> None:
        source = (ROOT / "tools/static/app.js").read_text(encoding="utf-8")
        update = source[
            source.index("function updateProfileView"):
            source.index("function fillObjectSelects")
        ]
        self.assertIn("applyModuleVisibility();", update)
        self.assertIn("function moduleVisible", source)
        self.assertIn("function applyModuleVisibility", source)

    def test_modern_reload_unwraps_server_config_envelope(self) -> None:
        source = (ROOT / "tools/static/app.js").read_text(encoding="utf-8")
        handler = source[
            source.index('socket.on("config_reloaded"'):
            source.index('document.getElementById("btn-refresh-health"')
        ]
        self.assertIn("(data && data.config) || data || {}", handler)

    def test_classic_applies_initial_and_reloaded_visibility(self) -> None:
        source = (
            ROOT / "tools/static/app.classic.js"
        ).read_text(encoding="utf-8")
        self.assertIn("function applyModuleVisibility()", source)
        self.assertIn("applyModuleVisibility();", source)
        self.assertIn("applyRuntimeConfig(data.config);", source)


class ExtendedUiDomVisibilityTests(unittest.TestCase):
    FALLBACK_MARKERS = (
        'data-quick-fallback-object="trapezoid"',
        'data-quick-fallback-object="board"',
        'data-quick-fallback-object="butter_knife"',
    )

    def test_classic_panels_and_quick_children_declare_ownership(
        self,
    ) -> None:
        html = (
            ROOT / "tools/static/index.classic.html"
        ).read_text(encoding="utf-8")
        server = (
            ROOT / "tools/voice_pick_demo.py"
        ).read_text(encoding="utf-8")

        self.assertIn('id="panel-vla"', html)
        self.assertIn('data-module="vla"', html)
        self.assertIn('data-module="dataset_capture"', server)
        for marker in (
            'data-ui-control="quick_pick.show_arm_controls"',
            'data-ui-control="quick_pick.show_reload_config"',
            'data-ui-control="quick_pick.show_catalog_picker"',
            "data-ui-controls-any=",
            "data-quick-fallback-group",
            *self.FALLBACK_MARKERS,
        ):
            self.assertIn(marker, html)

    def test_modern_panels_and_quick_children_declare_ownership(
        self,
    ) -> None:
        html = (ROOT / "tools/static/index.html").read_text(
            encoding="utf-8"
        )
        for marker in (
            'data-module="vla"',
            'data-module="dataset_capture"',
            'data-ui-control="quick_pick.show_catalog_picker"',
            "data-quick-fallback-group",
            *self.FALLBACK_MARKERS,
        ):
            self.assertIn(marker, html)

    def test_both_clients_apply_child_visibility_on_config_updates(
        self,
    ) -> None:
        for relative in (
            "tools/static/app.js",
            "tools/static/app.classic.js",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("var uiControlConfig = {};", source)
            self.assertIn("function uiControlValue(", source)
            self.assertIn("function uiControlVisible(", source)
            self.assertIn("function fixedFallbackAllowlist()", source)
            self.assertIn("function applyUiControlVisibility()", source)
            self.assertIn("applyUiControlVisibility();", source)

    def test_classic_maps_vla_and_dataset_panels(self) -> None:
        source = (
            ROOT / "tools/static/app.classic.js"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'setPanelVisible("vla", moduleVisible("vla"));',
            source,
        )
        self.assertIn(
            'setPanelVisible("dataset-capture", '
            'moduleVisible("dataset_capture"));',
            source,
        )

    def test_operator_docs_cover_new_visibility_controls(self) -> None:
        docs = (ROOT / "docs/TEST_STEPS_zh-TW.md").read_text(
            encoding="utf-8"
        )
        for marker in (
            "modules.vla.show_in_ui",
            "modules.dataset_capture.show_in_ui",
            "ui.controls.quick_pick",
            "show_arm_controls",
            "show_reload_config",
            "show_catalog_picker",
            "fixed_fallback_objects",
            "Complete Grasp (Enter)",
            "Complete Release + Open (Enter)",
            "manual_release_position: [2872, 3072, 2048]",
            "Move to Release Pose (Enter)",
            "Return Fingers HOME (Enter)",
            "第一次 Enter",
            "第二次 Enter",
            "HOME 成功後手臂才會上抬／縮回",
            "curl -X POST http://127.0.0.1:8090/api/reload_config",
        ):
            self.assertIn(marker, docs)


class FingerStageCompletionUiTests(unittest.TestCase):
    def test_both_uis_have_one_hidden_completion_button_in_quick_pick(
        self,
    ) -> None:
        for relative in (
            "tools/static/index.html",
            "tools/static/index.classic.html",
        ):
            html = (ROOT / relative).read_text(encoding="utf-8")
            self.assertEqual(
                html.count('id="btn-finger-stage-complete"'),
                1,
            )
            button_at = html.index('id="btn-finger-stage-complete"')
            button = html[button_at - 160:button_at + 320]
            self.assertIn("hidden", button)
            self.assertIn("Complete Finger Action", button)

        classic = (
            ROOT / "tools/static/index.classic.html"
        ).read_text(encoding="utf-8")
        quick = classic[
            classic.index('id="panel-quick"'):
            classic.index('id="panel-trajectory"')
        ]
        self.assertIn('id="btn-finger-stage-complete"', quick)

        modern = (ROOT / "tools/static/index.html").read_text(
            encoding="utf-8"
        )
        quick_at = modern.index("data-quick-fallback-group")
        quick = modern[quick_at:quick_at + 1800]
        self.assertIn('id="btn-finger-stage-complete"', quick)

    def test_both_clients_render_and_emit_contextual_completion(self) -> None:
        for relative in (
            "tools/static/app.js",
            "tools/static/app.classic.js",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn('socket.on("vla_status"', source)
            self.assertIn("finger_stage_action", source)
            self.assertIn("Complete Grasp (Enter)", source)
            self.assertIn("Complete Release + Open (Enter)", source)
            self.assertIn("Move to Release Pose (Enter)", source)
            self.assertIn("Return Fingers HOME (Enter)", source)
            self.assertIn('"release_pose"', source)
            self.assertIn('"release_home"', source)
            self.assertIn(
                'socket.emit("vla_complete_finger_stage")',
                source,
            )

    def test_both_clients_guard_the_enter_shortcut_from_form_focus(
        self,
    ) -> None:
        for relative in (
            "tools/static/app.js",
            "tools/static/app.classic.js",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("function isFormControlFocused()", source)
            self.assertIn('e.key === "Enter"', source)
            self.assertIn("e.repeat", source)
            for tag in ("INPUT", "TEXTAREA", "SELECT", "BUTTON"):
                self.assertIn('tag === "' + tag + '"', source)


if __name__ == "__main__":
    unittest.main()
