"""ASR-tolerant resolution of the three demo objects.

Two failure modes seen on the rig drive this module:

1. Chrome's Web Speech API is an *open-vocabulary* recognizer. A bare short word
   ("knife", ~0.4s) gives its acoustic model almost nothing, so its general
   language model dominates and can return high-prior junk like "OK Google" as
   alternative 0 while the real word sits at alternative 1 or 2. We ask for 3
   alternatives, so we must actually look at all of them.

2. zh-TW ASR returns homophones for short words: a bare 刀 (dāo) very often comes
   back as 到/倒/道/島, and 形/型 + 板/版 are routinely swapped. An exact synonym
   match cannot recover any of these.

The safety rule that must survive both: 剪刀 (scissors) contains 刀, and an
operator who said "scissors" must never get the butter knife.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nlu import IntentParser


class CandidateResolutionTests(unittest.TestCase):
    """resolve_demo_from_candidates: pick the right transcript out of N."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.parser = IntentParser()

    def resolve(self, candidates):
        return self.parser.resolve_demo_from_candidates(candidates)

    # ---- the reported bug: real word is not alternative 0 -------------------
    def test_hallucinated_top_alternative_is_skipped(self) -> None:
        """The exact symptom reported: said "knife", alt0 came back "OK Google"."""
        r = self.resolve(["OK Google", "knife", "nice"])
        self.assertEqual(r.object_key, "butter_knife")
        self.assertEqual(r.status, "matched")
        self.assertEqual(r.candidate_index, 1)

    def test_single_clean_candidate_still_matches(self) -> None:
        r = self.resolve(["knife"])
        self.assertEqual(r.object_key, "butter_knife")
        self.assertEqual(r.candidate_index, 0)

    def test_exact_match_beats_fuzzy_match_ranked_above_it(self) -> None:
        """到 (fuzzy knife) at alt0 must not pre-empt an exact 刀子 at alt1."""
        r = self.resolve(["到", "刀子"])
        self.assertEqual(r.object_key, "butter_knife")
        self.assertEqual(r.status, "matched")
        self.assertEqual(r.candidate_index, 1)

    def test_all_candidates_junk_is_out_of_scope(self) -> None:
        r = self.resolve(["OK Google", "hey Siri", "what's the weather"])
        self.assertIsNone(r.object_key)
        self.assertEqual(r.status, "out_of_scope")

    def test_empty_candidate_list(self) -> None:
        self.assertEqual(self.resolve([]).status, "empty")

    def test_blank_candidates_are_ignored(self) -> None:
        r = self.resolve(["", "   ", "board"])
        self.assertEqual(r.object_key, "board")

    # ---- the 剪刀 safety rule, now under multi-candidate + fuzzy ------------
    def test_scissors_alone_never_becomes_the_knife(self) -> None:
        r = self.resolve(["剪刀"])
        self.assertIsNone(r.object_key)
        self.assertEqual(r.status, "out_of_scope")

    def test_top_ranked_scissors_vetoes_a_lower_knife_candidate(self) -> None:
        """Operator said scissors; a knife transcript below it must not win."""
        r = self.resolve(["剪刀", "刀子"])
        self.assertIsNone(r.object_key)
        self.assertEqual(r.status, "out_of_scope")

    def test_scissors_blocks_the_fuzzy_pass_entirely(self) -> None:
        """An exact non-demo match is decisive: no fuzzy rescue may run after it."""
        r = self.resolve(["剪刀", "到"])
        self.assertIsNone(r.object_key)
        self.assertEqual(r.status, "out_of_scope")

    def test_other_in_scope_object_still_wins_when_junk_ranks_first(self) -> None:
        r = self.resolve(["hey Google", "電路板"])
        self.assertEqual(r.object_key, "board")


class HomophoneToleranceTests(unittest.TestCase):
    """The fuzzy pass: only runs when nothing matched exactly, anywhere."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.parser = IntentParser()

    def resolve(self, text: str):
        return self.parser.resolve_demo_from_candidates([text])

    # ---- zh-TW homophones for a bare 刀 -------------------------------------
    def test_dao_homophone_dao4_arrive(self) -> None:
        self.assertEqual(self.resolve("到").object_key, "butter_knife")

    def test_dao_homophone_in_carrier_phrase(self) -> None:
        self.assertEqual(self.resolve("拿起倒").object_key, "butter_knife")

    def test_dao_homophone_island(self) -> None:
        self.assertEqual(self.resolve("島").object_key, "butter_knife")

    # ---- 形/型 and 板/版 orthographic swaps ---------------------------------
    def test_trapezoid_xing_variant(self) -> None:
        self.assertEqual(self.resolve("梯型").object_key, "trapezoid")

    def test_trapezoid_body_type_homophone(self) -> None:
        self.assertEqual(self.resolve("體型").object_key, "trapezoid")

    def test_board_ban_variant(self) -> None:
        self.assertEqual(self.resolve("電路版").object_key, "board")

    # ---- English near-misses ------------------------------------------------
    def test_english_knife_misheard_as_life(self) -> None:
        self.assertEqual(self.resolve("life").object_key, "butter_knife")

    def test_english_trapezoid_plural(self) -> None:
        self.assertEqual(self.resolve("trapezoids").object_key, "trapezoid")

    def test_english_board_misheard_as_bored(self) -> None:
        self.assertEqual(self.resolve("bored").object_key, "board")

    def test_fuzzy_result_is_flagged_as_fuzzy(self) -> None:
        exact = self.resolve("刀子")
        fuzzy = self.resolve("到")
        self.assertFalse(exact.fuzzy)
        self.assertTrue(fuzzy.fuzzy)

    # ---- the fuzzy pass must not become a free-for-all ---------------------
    def test_unrelated_word_is_not_fuzzed_into_an_object(self) -> None:
        for junk in ("banana", "什麼", "hello there", "stop"):
            with self.subTest(junk=junk):
                self.assertIsNone(self.resolve(junk).object_key)

    def test_other_objects_are_not_fuzzed_into_demo_objects(self) -> None:
        """Real non-demo objects stay out of scope, exactly as before."""
        for other in ("剪刀", "scissors", "湯匙", "spoon", "筷子"):
            with self.subTest(other=other):
                r = self.resolve(other)
                self.assertIsNone(r.object_key)
                self.assertEqual(r.status, "out_of_scope")


class RigObservedConfusionTests(unittest.TestCase):
    """Confusions actually logged on the rig (voice_pick_full.log, 2026-07-22).

    These are the misses where the correct word was absent from EVERY browser
    alternative. They are seeded from real data, not linguistics — the recognizer
    turns 刀子 into common "X-zi" nouns and 梯形 into its near-homophone 提醒,
    especially when the carrier verb is 查 ("search"), which primes the language
    model toward query words.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.parser = IntentParser()

    def resolve(self, *candidates):
        return self.parser.resolve_demo_from_candidates(list(candidates))

    # ---- 刀子 (dāozi) → common "-zi" nouns ---------------------------------
    def test_baozi_bun_becomes_knife(self) -> None:
        self.assertEqual(self.resolve("包子").object_key, "butter_knife")

    def test_taozi_peach_becomes_knife(self) -> None:
        self.assertEqual(self.resolve("桃子").object_key, "butter_knife")

    def test_search_carrier_baozi(self) -> None:
        self.assertEqual(self.resolve("幫我查包子").object_key, "butter_knife")

    def test_pure_miss_candidate_list_recovers_knife(self) -> None:
        """The exact logged miss: all three alts are wrong, all -zi nouns."""
        r = self.resolve("幫我查桃子", "幫我查包子", "幫我查房子")
        self.assertEqual(r.object_key, "butter_knife")
        self.assertTrue(r.fuzzy)

    # ---- 梯形 (tīxíng) → 提醒 (tíxǐng) near-homophone -----------------------
    def test_tixing_reminder_becomes_trapezoid(self) -> None:
        self.assertEqual(self.resolve("提醒").object_key, "trapezoid")

    def test_search_carrier_reminder(self) -> None:
        self.assertEqual(self.resolve("幫我查提醒").object_key, "trapezoid")

    def test_pure_miss_candidate_list_recovers_trapezoid(self) -> None:
        r = self.resolve("幫我查詢", "幫我查提醒", "幫我找尋")
        self.assertEqual(r.object_key, "trapezoid")
        self.assertTrue(r.fuzzy)

    # ---- precision: do NOT reach for words that are genuinely too far -------
    def test_house_does_not_become_knife(self) -> None:
        """房子 (house) shares only 子; too weak a signal to claim the knife."""
        self.assertIsNone(self.resolve("房子").object_key)

    def test_query_word_alone_is_out_of_scope(self) -> None:
        for junk in ("查詢", "找尋", "包裹", "手機", "東西"):
            with self.subTest(junk=junk):
                self.assertIsNone(self.resolve(junk).object_key)

    def test_exact_still_beats_new_aliases(self) -> None:
        """If 刀子 is present anywhere, it wins as an exact match, not fuzzy."""
        r = self.resolve("包子", "高職", "刀子")
        self.assertEqual(r.object_key, "butter_knife")
        self.assertFalse(r.fuzzy)
        self.assertEqual(r.candidate_index, 2)


class NormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parser = IntentParser()

    def resolve(self, text: str):
        return self.parser.resolve_demo_from_candidates([text])

    def test_trailing_punctuation_is_stripped(self) -> None:
        self.assertEqual(self.resolve("Knife.").object_key, "butter_knife")

    def test_uppercase_matches(self) -> None:
        self.assertEqual(self.resolve("BOARD").object_key, "board")

    def test_fullwidth_punctuation_is_stripped(self) -> None:
        self.assertEqual(self.resolve("拿起梯形。").object_key, "trapezoid")

    def test_english_carrier_phrase(self) -> None:
        self.assertEqual(self.resolve("pick up the butter knife").object_key, "butter_knife")


if __name__ == "__main__":
    unittest.main(verbosity=2)
