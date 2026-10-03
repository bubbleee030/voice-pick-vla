"""
Fuzzy Natural Language Understanding (NLU) for voice/text commands.

Handles ambiguous input like:
  - "幫我拿那個紅色的" → pick, but which object? → ask user
  - "grab the red thing" → pick, could be apple or tomato
  - "蘋果拿起來" → pick apple
  - "Pick up apple" → pick apple
  - "幫我 pick the apple" → pick apple (code-switching)

Uses rule-based matching with synonym dictionaries from config/objects.yaml.
"""

import re
from difflib import SequenceMatcher
from typing import Optional
from src.utils import load_config


class IntentResult:
    """Result of intent parsing."""

    def __init__(self, intent: str, object_key: Optional[str],
                 confidence: float, candidates: list = None,
                 need_confirmation: bool = False, raw_text: str = ""):
        self.intent = intent              # "pick", "place", "home", "quit", "unknown"
        self.object_key = object_key      # "apple", "box", etc. or None
        self.confidence = confidence       # 0.0 - 1.0
        self.candidates = candidates or []  # Multiple matches if ambiguous
        self.need_confirmation = need_confirmation
        self.raw_text = raw_text

    def __repr__(self):
        return (f"IntentResult(intent={self.intent!r}, object={self.object_key!r}, "
                f"conf={self.confidence:.2f}, candidates={self.candidates})")


# The voice input drives the three-object auto-run (ADR 0003). Only these three
# object keys are reachable by voice; every other objects.yaml class is out of
# scope. The standalone `knife` class folds into `butter_knife` — in the voice
# demo there is no separate knife.
DEMO_OBJECTS = ("trapezoid", "board", "butter_knife")
_KNIFE_ALIAS_OF_BUTTER_KNIFE = "knife"

# Canonical display labels for the three (the "only these 3" operator message).
# butter_knife shows 奶油刀 — the word used in its VLA instruction 拿起奶油刀 —
# rather than objects.yaml's first synonym 奶油抹刀.
DEMO_OBJECT_LABELS = {
    "trapezoid": {"zh": "梯形", "en": "trapezoid"},
    "board": {"zh": "電路板", "en": "board"},
    "butter_knife": {"zh": "奶油刀", "en": "butter knife"},
}

# ---------------------------------------------------------------------------
# ASR tolerance
# ---------------------------------------------------------------------------
# Browser speech recognition is open-vocabulary: it does not know our three
# objects exist, so a short utterance gets resolved by its general language
# model. These are the resulting confusions, and they are of two kinds:
#
#   zh — purely PHONETIC. A bare 刀 (dāo) is one short syllable, and the
#        recognizer emits whichever character with that sound is most frequent
#        in general Chinese (到 by a wide margin). Edit distance cannot see this
#        at all — 到 and 刀 share no strokes — so the confusions must be listed.
#        形/型 and 板/版 are true homophones and get swapped constantly.
#   en — ORTHOGRAPHIC neighbours ("knife" → "life": the k is silent, so the
#        acoustic evidence is genuinely identical). Edit distance does catch
#        most of these; the list covers the ones that fall below threshold.
#
# Entries tagged "rig 2026-07-22" are seeded from voice_pick_full.log — real
# misses where the correct word appeared in NO browser alternative — not from
# linguistics. See ADR 0004.
_DEMO_FUZZY_ALIASES = {
    "butter_knife": (
        "到", "倒", "道", "島", "導", "蹈",          # dāo/dào/dǎo homophones
        "奶油到", "奶油倒", "奶油道", "抹到", "抹倒",
        # rig 2026-07-22: 刀子 (dāozi) → common "X-zi" nouns. The recognizer
        # keeps the -āozi rhyme and swaps the initial. Only the ones sharing the
        # -ao vowel are listed; 房子/高職 are too far and deliberately excluded so
        # the operator is not offered the knife on a wild mishear.
        "包子", "桃子", "刨子", "袍子",
        "life", "nice", "wife", "night", "naif", "knive", "knifes",
        "butter life", "butter nice", "butterknife",
    ),
    "trapezoid": (
        "梯型", "體型", "體形", "提行", "替行", "踢行",  # tī xíng homophones
        "提醒", "提行", "剃刑",                          # rig 2026-07-22: 提醒 (tíxǐng) ≈ 梯形 (tīxíng)
        "trapezoids", "trapezoidal", "trapeze", "trapezoi", "trapizoid",
    ),
    "board": (
        "電路版", "電路辦", "電腦板", "電路半",          # bǎn/bàn homophones
        "pcb版", "pcb板", "pc平板", "tcp版", "tcb版",     # rig 2026-07-22: "PCB" letter-salad, all end in 板/版
        "boards", "bored", "broad", "boar", "circuit bored", "circuit boards",
    ),
}

# Command scaffolding stripped before judging how much of the utterance the
# matched word actually covers. Without this, "拿起倒" would look like a 33%
# match; with it the remainder is just 倒 and the intent is unambiguous.
_DEMO_CARRIER_WORDS = (
    "拿起來", "拿起", "撿起", "給我", "幫我", "我要", "請幫", "一個", "那個", "這個",
    "拿", "抓", "夾", "撿", "把", "請", "的",
    # 查/查詢 ("search") — the carrier that most poisons recognition (it primes
    # the LM toward query words); stripped so the object word can still surface.
    "查詢", "查一下", "查", "找",
    "pick up", "pick", "grab", "take", "get", "please", "the", "a", "an", "me", "up",
)

# A listed confusion only counts if it covers at least this much of what is left
# after carrier stripping — so 道 rescues "道" but not "我不知道".
_DEMO_FUZZY_MIN_COVERAGE = 0.5
# Latin edit-distance safety net for spellings not in the list above.
_DEMO_FUZZY_MIN_RATIO = 0.85


class DemoResolution:
    """Result of resolving voice text to one of the three auto-run objects.

    status is one of:
      "matched"      — object_key is one of DEMO_OBJECTS
      "out_of_scope" — a word was heard but it is not one of the three
      "ambiguous"    — two or more different demo objects were named at once
      "empty"        — no text
    """

    def __init__(self, object_key, status, matched_phrase="",
                 other_object=None, candidate_index=-1, fuzzy=False):
        self.object_key = object_key      # one of DEMO_OBJECTS, or None
        self.status = status
        self.matched_phrase = matched_phrase
        # A real objects.yaml class that *is* named but is not one of the three
        # (e.g. scissors). Its presence is decisive: it means the operator named
        # a known object, so no homophone rescue may override it.
        self.other_object = other_object
        # Which ASR alternative won (-1 = not from a candidate list), and whether
        # it only matched after homophone tolerance.
        self.candidate_index = candidate_index
        self.fuzzy = fuzzy

    def __repr__(self):
        return (f"DemoResolution(object={self.object_key!r}, status={self.status!r}, "
                f"matched={self.matched_phrase!r}, alt={self.candidate_index}, "
                f"fuzzy={self.fuzzy})")


class IntentParser:
    """
    Parse voice/text input into structured intent + object.
    Supports Mandarin Taiwan, English, and mixed input.
    """

    DEMO_OBJECTS = DEMO_OBJECTS

    # Intent keywords
    PICK_KEYWORDS_ZH = ["拿", "拿起", "撿", "撿起", "抓", "抓起", "取", "拿給我",
                        "幫我拿", "幫我撿", "pick", "拿起來", "拿一下"]
    PICK_KEYWORDS_EN = ["pick", "grab", "take", "get", "fetch", "grasp",
                        "pick up", "grab me", "get me"]
    PLACE_KEYWORDS_ZH = ["放", "放下", "放到", "放在"]
    PLACE_KEYWORDS_EN = ["place", "put", "put down", "set down"]
    HOME_KEYWORDS = ["home", "回家", "回原點", "歸位", "reset"]
    QUIT_KEYWORDS = ["quit", "exit", "bye", "結束", "離開", "停止", "q"]

    def __init__(self):
        """Load object synonyms from config."""
        self.objects_cfg = load_config("objects.yaml")
        self._build_synonym_map()

    def _build_synonym_map(self):
        """Build a flat synonym → object_key mapping."""
        self.synonym_to_object = {}

        for obj_key, obj_def in self.objects_cfg.get("classes", {}).items():
            # Add the key itself
            self.synonym_to_object[obj_key.lower()] = obj_key

            # Add Chinese synonyms
            for syn in obj_def.get("chinese", []):
                self.synonym_to_object[syn.lower()] = obj_key

            # Add English synonyms
            for syn in obj_def.get("english", []):
                self.synonym_to_object[syn.lower()] = obj_key

        # Sort by length (longest first) for greedy matching
        self._sorted_synonyms = sorted(
            self.synonym_to_object.keys(), key=len, reverse=True
        )

    def parse(self, text: str) -> IntentResult:
        """
        Parse a text command into an IntentResult.

        Args:
            text: Raw voice/text input (Chinese, English, or mixed).

        Returns:
            IntentResult with parsed intent and object.
        """
        if not text or not text.strip():
            return IntentResult("unknown", None, 0.0, raw_text=text)

        text_clean = text.strip()
        text_lower = text_clean.lower()

        # Check quit
        for kw in self.QUIT_KEYWORDS:
            if kw in text_lower:
                return IntentResult("quit", None, 1.0, raw_text=text_clean)

        # Check home
        for kw in self.HOME_KEYWORDS:
            if kw in text_lower:
                return IntentResult("home", None, 1.0, raw_text=text_clean)

        # Detect intent
        intent = self._detect_intent(text_lower)

        # Extract object
        obj_key, candidates, conf = self._extract_object(text_lower)

        if obj_key and len(candidates) == 1:
            return IntentResult(intent, obj_key, conf, candidates, False, text_clean)
        elif len(candidates) > 1:
            # Ambiguous — multiple candidates
            return IntentResult(intent, None, conf * 0.5, candidates, True, text_clean)
        else:
            # No object found
            return IntentResult(intent, None, 0.3, [], True, text_clean)

    def _detect_intent(self, text: str) -> str:
        """Detect the action intent from text."""
        # Check place first (less common)
        for kw in self.PLACE_KEYWORDS_ZH + self.PLACE_KEYWORDS_EN:
            if kw in text:
                return "place"

        # Check pick (most common)
        for kw in self.PICK_KEYWORDS_ZH + self.PICK_KEYWORDS_EN:
            if kw in text:
                return "pick"

        # Default to pick if an object is mentioned
        return "pick"

    def _extract_object(self, text: str) -> tuple:
        """
        Extract object key from text using synonym matching.

        Returns:
            (best_match, all_candidates, confidence)
        """
        matched = set()

        for synonym in self._sorted_synonyms:
            if synonym in text:
                obj_key = self.synonym_to_object[synonym]
                matched.add(obj_key)

        candidates = list(matched)

        if len(candidates) == 1:
            return candidates[0], candidates, 0.95
        elif len(candidates) > 1:
            # Multiple matches (e.g., "紅色的" matches both apple and tomato)
            return None, candidates, 0.5
        else:
            return None, [], 0.0

    def _remap_demo(self, obj_key: str) -> str:
        """Fold the standalone `knife` class into `butter_knife` for the voice demo."""
        return "butter_knife" if obj_key == _KNIFE_ALIAS_OF_BUTTER_KNIFE else obj_key

    def resolve_demo_object(self, text: str) -> DemoResolution:
        """Resolve voice text to one of the three auto-run objects (ADR 0003).

        Longest matching synonym wins, and a shorter synonym nested inside a
        longer match is discarded. That is what stops 剪刀 (scissors) — which
        merely *contains* 刀 — from being mis-read as the knife: 剪刀 is the
        longer match and shadows 刀, and scissors is out of scope.
        """
        if not text or not text.strip():
            return DemoResolution(None, "empty")

        t = text.strip().lower()

        # _sorted_synonyms is longest-first, so present-matches keep that order.
        kept: list[str] = []
        for syn in self._sorted_synonyms:
            if syn in t and not any(syn in longer for longer in kept):
                kept.append(syn)

        # Distinct in-scope objects, in longest-first order of their trigger word.
        demo_present: list[tuple] = []
        seen: set = set()
        for syn in kept:
            obj = self._remap_demo(self.synonym_to_object[syn])
            if obj in DEMO_OBJECTS and obj not in seen:
                seen.add(obj)
                demo_present.append((obj, syn))

        if not demo_present:
            # Note *which* object was heard, if any. "剪刀" is out of scope
            # because it is scissors, not because nothing was understood, and
            # that distinction vetoes the homophone pass downstream.
            other = self.synonym_to_object[kept[0]] if kept else None
            return DemoResolution(None, "out_of_scope", other_object=other)
        if len(demo_present) > 1:
            return DemoResolution(None, "ambiguous")
        obj, syn = demo_present[0]
        return DemoResolution(obj, "matched", syn)

    # ---- ASR tolerance ---------------------------------------------------

    def _demo_latin_synonyms(self) -> dict:
        """Latin synonyms of the three, long enough to edit-distance safely."""
        if getattr(self, "_demo_latin_cache", None) is None:
            table: dict = {obj: [] for obj in DEMO_OBJECTS}
            for syn, obj_key in self.synonym_to_object.items():
                obj = self._remap_demo(obj_key)
                if obj in DEMO_OBJECTS and len(syn) >= 4 and syn.isascii():
                    table[obj].append(syn)
            self._demo_latin_cache = table
        return self._demo_latin_cache

    def _strip_demo_carriers(self, text: str) -> str:
        """Drop command scaffolding ("拿起", "pick up") and punctuation."""
        t = re.sub(r"[^\w\s一-鿿]+", " ", (text or "").lower())
        for word in _DEMO_CARRIER_WORDS:
            t = t.replace(word, " ")
        return re.sub(r"\s+", " ", t).strip()

    def _fuzzy_demo_match(self, text: str):
        """Resolve a near-miss transcript. Returns (object_key, phrase) or (None, "").

        Only reached when no candidate matched any known object exactly, so this
        can never talk over an operator who named a real out-of-scope object.
        """
        stripped = self._strip_demo_carriers(text)
        if not stripped:
            return None, ""
        body_len = len(stripped.replace(" ", ""))

        # 1. Listed confusions — the only workable route for Han characters.
        #    Longest alias wins so 奶油到 beats a bare 到.
        best = None
        for obj in DEMO_OBJECTS:
            for alias in _DEMO_FUZZY_ALIASES.get(obj, ()):
                if alias not in stripped:
                    continue
                coverage = len(alias.replace(" ", "")) / max(body_len, 1)
                if coverage < _DEMO_FUZZY_MIN_COVERAGE:
                    continue
                if best is None or len(alias) > len(best[1]):
                    best = (obj, alias)
        if best:
            return best

        # 2. Latin-only edit distance. Meaningless for Han characters, where
        #    confusions are phonetic rather than orthographic.
        tokens = [tok for tok in re.split(r"[^a-z0-9]+", stripped) if len(tok) >= 4]
        if not tokens:
            return None, ""
        best_ratio, best_hit = 0.0, None
        for obj, synonyms in self._demo_latin_synonyms().items():
            for syn in synonyms:
                for tok in tokens:
                    ratio = SequenceMatcher(None, tok, syn).ratio()
                    if ratio > best_ratio:
                        best_ratio, best_hit = ratio, (obj, tok)
        if best_hit and best_ratio >= _DEMO_FUZZY_MIN_RATIO:
            return best_hit
        return None, ""

    def resolve_demo_from_candidates(self, candidates) -> DemoResolution:
        """Resolve one utterance given every transcript the recognizer offered.

        Browser speech returns up to `maxAlternatives` ranked guesses. Because it
        is open-vocabulary, a bare "knife" can come back as "OK Google" at rank 0
        with the real word at rank 1 — so all alternatives get a look.

        Rank is respected: the first candidate naming ANY known object decides,
        in scope or not. A top-ranked 剪刀 therefore vetoes a 刀子 below it,
        rather than the demo opportunistically claiming the in-scope match.
        """
        if isinstance(candidates, str):
            candidates = [candidates]
        texts = [c.strip() for c in (candidates or [])
                 if isinstance(c, str) and c.strip()]
        if not texts:
            return DemoResolution(None, "empty")

        # Pass 1 — exact, in the recognizer's own confidence order.
        for idx, text in enumerate(texts):
            res = self.resolve_demo_object(text)
            if res.status in ("matched", "ambiguous") or res.other_object:
                res.candidate_index = idx
                return res

        # Pass 2 — nothing known was named anywhere: tolerate homophones.
        for idx, text in enumerate(texts):
            obj, phrase = self._fuzzy_demo_match(text)
            if obj:
                return DemoResolution(obj, "matched", phrase,
                                      candidate_index=idx, fuzzy=True)

        return DemoResolution(None, "out_of_scope")

    def get_disambiguation_prompt(self, result: IntentResult) -> str:
        """
        Generate a Chinese disambiguation prompt for the user.

        Args:
            result: An ambiguous IntentResult.

        Returns:
            Chinese string asking user to clarify.
        """
        if not result.candidates:
            # No candidates at all — list all available objects
            all_objects = list(self.objects_cfg.get("classes", {}).keys())
            names_zh = []
            for obj in all_objects:
                zh_names = self.objects_cfg["classes"][obj].get("chinese", [obj])
                names_zh.append(zh_names[0] if zh_names else obj)
            return f"我沒有聽懂要拿什麼。桌上有: {', '.join(names_zh)}。請再說一次？"

        # Multiple candidates
        names_zh = []
        for obj in result.candidates:
            zh_names = self.objects_cfg["classes"].get(obj, {}).get("chinese", [obj])
            names_zh.append(zh_names[0] if zh_names else obj)

        return f"你說的可能是 {' 或 '.join(names_zh)}，請問要拿哪一個？"

    def list_objects(self) -> str:
        """Return a formatted list of all known objects (for help display)."""
        lines = []
        for obj_key, obj_def in self.objects_cfg.get("classes", {}).items():
            zh = obj_def.get("chinese", ["?"])[0]
            en = obj_def.get("english", ["?"])[0]
            lines.append(f"  {zh} / {en}")
        return "\n".join(lines)
