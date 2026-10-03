"""Per-object default rung/mode for the auto-run (ADR 0003).

board rides the locator rung (the deployed checkpoint is trapezoid-only), the
trapezoid rides VLA, butter_knife rides fixed-xy (mode moot). The default mode
is config-driven in vla_auto.objects and surfaced on run_cfg so voice, the Auto
Run button, and the CLI all resolve the same rung; explicit CLI flags override.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import runtime_config
from src.runtime_config import build_auto_run_cfg


def _cfgs():
    demo = runtime_config._load_yaml(ROOT / "config" / "demo_config.yaml")
    objs = yaml.safe_load((ROOT / "config" / "objects.yaml").read_text(encoding="utf-8"))
    return demo, objs


class AutoRunDefaultModeTests(unittest.TestCase):
    def _default_mode(self, object_key: str) -> str:
        demo, objs = _cfgs()
        run_cfg, _tail, _obj = build_auto_run_cfg(demo, objs, object_key)
        return run_cfg["default_mode"]

    def test_board_default_mode_is_locator(self) -> None:
        self.assertEqual(self._default_mode("board"), "locator")

    def test_trapezoid_default_mode_is_vla(self) -> None:
        self.assertEqual(self._default_mode("trapezoid"), "vla")

    def test_butter_knife_default_mode_is_vla(self) -> None:
        # fixed-xy makes the mode moot, but it must still resolve to a valid rung.
        self.assertEqual(self._default_mode("butter_knife"), "vla")


class ManualReleasePositionConfigTests(unittest.TestCase):
    def _manual_release_position(self, object_key: str):
        demo, objs = _cfgs()
        _run, tail, _obj = build_auto_run_cfg(demo, objs, object_key)
        return tail["manual_release_position"]

    def test_board_manual_release_position(self) -> None:
        self.assertEqual(
            self._manual_release_position("board"),
            [2872, 3072, 2048],
        )

    def test_butter_knife_manual_release_position(self) -> None:
        self.assertEqual(
            self._manual_release_position("butter_knife"),
            [2872, 3072, 2048],
        )

    def test_trapezoid_has_no_manual_release_position(self) -> None:
        self.assertIsNone(self._manual_release_position("trapezoid"))

    def test_invalid_manual_release_position_is_rejected(self) -> None:
        demo, objs = _cfgs()
        demo["vla_auto"]["objects"]["board"]["manual_release_position"] = [2872, 3072]
        with self.assertRaises(runtime_config.FixedFallbackConfigError):
            build_auto_run_cfg(demo, objs, "board")


if __name__ == "__main__":
    unittest.main()
