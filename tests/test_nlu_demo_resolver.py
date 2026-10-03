"""Voice → auto-run object resolution (ADR 0003).

The voice input recognizes EXACTLY the three auto-run objects (trapezoid, board,
butter_knife). Every knife-like word resolves to butter_knife; anything else is
out of scope. The critical trap: 剪刀 (scissors) *contains* 刀 (knife) as a
substring, and must NOT be mis-resolved to the butter knife.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nlu import IntentParser


class DemoObjectResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parser = IntentParser()

    def resolve(self, text: str):
        return self.parser.resolve_demo_object(text)

    # ---- the three objects, both languages ----------------------------------
    def test_trapezoid_chinese(self) -> None:
        r = self.resolve("拿起梯形")
        self.assertEqual(r.object_key, "trapezoid")
        self.assertEqual(r.status, "matched")

    def test_trapezoid_english(self) -> None:
        self.assertEqual(self.resolve("pick up the trapezoid").object_key, "trapezoid")

    def test_board_chinese(self) -> None:
        r = self.resolve("拿電路板")
        self.assertEqual(r.object_key, "board")
        self.assertEqual(r.status, "matched")

    def test_board_english_circuit_board(self) -> None:
        self.assertEqual(self.resolve("grab the circuit board").object_key, "board")

    # ---- knife-words all fold into butter_knife -----------------------------
    def test_full_butter_knife_name_chinese(self) -> None:
        self.assertEqual(self.resolve("拿起奶油刀").object_key, "butter_knife")

    def test_bare_knife_word_chinese(self) -> None:
        # 刀子 is the standalone `knife` object in objects.yaml — in the voice
        # demo it means the butter knife.
        self.assertEqual(self.resolve("刀子").object_key, "butter_knife")

    def test_cleaver_word_chinese(self) -> None:
        self.assertEqual(self.resolve("菜刀").object_key, "butter_knife")

    def test_knife_english(self) -> None:
        self.assertEqual(self.resolve("grab a knife").object_key, "butter_knife")

    def test_butter_knife_english(self) -> None:
        self.assertEqual(self.resolve("pick up the butter knife").object_key, "butter_knife")

    # ---- THE TRAP: 剪刀 contains 刀 but is NOT the knife ---------------------
    def test_scissors_is_out_of_scope_not_knife(self) -> None:
        r = self.resolve("剪刀")
        self.assertIsNone(r.object_key)
        self.assertEqual(r.status, "out_of_scope")

    def test_scissors_english_out_of_scope(self) -> None:
        r = self.resolve("hand me the scissors")
        self.assertIsNone(r.object_key)
        self.assertEqual(r.status, "out_of_scope")

    # ---- other catalog objects are out of scope -----------------------------
    def test_spoon_out_of_scope(self) -> None:
        self.assertEqual(self.resolve("湯匙").status, "out_of_scope")

    def test_chopsticks_out_of_scope(self) -> None:
        self.assertEqual(self.resolve("pick up the chopsticks").status, "out_of_scope")

    # ---- empty / no object --------------------------------------------------
    def test_empty_is_empty(self) -> None:
        r = self.resolve("")
        self.assertIsNone(r.object_key)
        self.assertEqual(r.status, "empty")

    def test_pure_verb_no_object_out_of_scope(self) -> None:
        # a pick verb with no recognizable object is not one of the three
        self.assertIsNone(self.resolve("幫我拿一下").object_key)

    # ---- two demo objects at once is ambiguous, not a silent pick -----------
    def test_two_demo_objects_is_ambiguous(self) -> None:
        r = self.resolve("拿梯形和電路板")
        self.assertIsNone(r.object_key)
        self.assertEqual(r.status, "ambiguous")


if __name__ == "__main__":
    unittest.main()
