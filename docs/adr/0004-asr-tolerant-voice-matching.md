# ADR 0004 — ASR-tolerant voice matching

**Date:** 2026-07-21
**Status:** Accepted
**Supersedes:** nothing. Extends [ADR 0003](0003-voice-drives-auto-run.md).

## Context

ADR 0003 wired voice to the three-object auto-run. On the rig the recognition
rate was poor, with two distinct symptoms:

1. **Hallucinated transcripts in English.** Saying "knife" produced "OK Google".
2. **Chinese misses**, especially 刀子.

Investigation found three independent causes, none of them in the NLU's
vocabulary:

- **Discarded alternatives.** `app.classic.js` requested
  `maxAlternatives = 3` but read only `result[0]`. Browser speech is
  *open-vocabulary* and ranks by a general language model that has never heard
  of our three objects; for a bare 0.4 s word the acoustic evidence is thin and
  the LM prior dominates. "OK Google" is one of the highest-prior phrases in
  Google's English LM — hence the hallucination. The correct word was frequently
  already computed at rank 1 or 2 and thrown away.
- **Exact-match-only NLU.** `resolve_demo_object` required a literal substring
  hit against `objects.yaml`. A bare 刀 (dāo) is one short syllable and the
  recognizer emits the most frequent character with that sound — usually 到.
  形/型 and 板/版 are true homophones and swap constantly. One character off was
  a hard miss.
- **A fallback into a dead mode.** A single `network` error called
  `setAsrMode("offline", …)`, which persists to `localStorage`. The offline
  endpoint `/api/asr/transcribe` is a hardcoded 501 stub in this bundle, and the
  UI labelled it "Offline backend (whisper)". One transient Google outage
  therefore downgraded the operator, permanently and across reloads, into a mode
  where the mic could never work.

## Decision

**1. The client forwards every ASR alternative; the server chooses.**
`voice_text` gains a `candidates` array. Selection logic lives in `src/nlu.py`
(`resolve_demo_from_candidates`), not the browser, because the vocabulary
belongs with the NLU and Python is where this repo can test it.

**2. Rank is respected, and an exact match anywhere beats a fuzzy one.**
Candidates are scanned in the recognizer's own order, and **the first candidate
naming any known object decides — in scope or not**. A top-ranked 剪刀 therefore
vetoes a 刀子 ranked below it, instead of the demo opportunistically claiming
the in-scope match. Only if no candidate names anything known does the
homophone pass run.

**3. Homophone tolerance is a curated list for zh, edit distance for en.**
Chinese confusions are *phonetic* (到/刀 share no strokes), so edit distance
cannot see them and the confusions are listed explicitly. English confusions are
orthographic neighbours ("knife"/"life" — the k is silent, so the acoustics are
genuinely identical), so `difflib` at ratio ≥ 0.85 covers them, with a list for
the ones that fall below.

**4. A fuzzy hit must cover the utterance, after stripping carrier words.**
"拿起" / "pick up" are removed, then the matched word must account for ≥ 50% of
what remains. This rescues 「到」 and 「拿起倒」 while leaving 「我不知道」 alone.

**5. Never auto-switch into offline ASR.** A network error now reports and stays
put. If offline is selected manually and returns 501, the UI bounces back to
browser mode and disables the button. `offlineAsrLabel()` no longer names an
engine the server did not advertise.

**6. Fuzzy matches are visibly weaker.** They report confidence 0.7 and the card
reads "Homophone match - check before running". Confirm remains the safety gate.

## Consequences

- A fuzzy match can be wrong. That is acceptable only because voice is
  Confirm-gated (ADR 0003) and the operator sees the resolved object first.
- The homophone table is **seeded from linguistics, not from rig data**. Every
  voice command now logs its full alternative list and which one won
  (`ASR alternatives [...] → object (altN, homophone)`), so the table can be
  extended from real misses rather than guesses.
- Browser speech remains open-vocabulary and will keep hallucinating on bare
  single words. The cheapest operator-side mitigation is a **carrier phrase** —
  「拿起刀子」 / "pick up the knife" — which gives the acoustic model context to
  lock onto. Substring matching means carrier phrases already work.
- A real offline recognizer (Whisper with an initial-prompt bias toward the three
  objects) remains the durable fix and is still not implemented in this bundle.

## Addendum — first rig data (2026-07-22)

22 live utterances (`data/runtime/logs/voice_pick_full.log`) confirmed the model
and refined it:

- **The carrier verb dominates the outcome.** Every `幫我拿X` ("take X") surfaced
  the correct word in some alternative; every `幫我查X` ("search X") missed it
  entirely — 查 primes Google's LM toward query words (查詢, 提醒, 包裹, 找尋). No
  server logic can recover a word the browser never emitted, so the primary
  mitigation is operator coaching: **say 拿, not 查**. 查/查詢 were added to the
  carrier-strip list as a partial defence for the fuzzy pass.
- **Most "wrong" results were mis-*displayed*, not mis-matched.** The transcript
  box showed alt0 (the loudest guess, e.g. 包子) while the NLU had correctly
  resolved butter_knife from alt1 (刀子). Fixed: the UI now shows the *understood*
  object plus the winning candidate ("✓ 奶油抹刀 ⟵「幫我拿刀子」"), and the server
  emits `matched_text`/`heard_top` to support it.
- **The homophone table was seeded from real misses.** 刀子→包子/桃子 (the
  recognizer keeps the -āozi rhyme, swaps the initial), 梯形→提醒 (near-homophone),
  and the PCB letter-salad (PC平板/tcp版/…). Deliberately excluded: 房子/高職, too
  far from 刀子 to offer the knife without eroding operator trust.

The durable fix is still an offline recognizer; toneless-pinyin fuzzy matching
(pypinyin, confirmed installable) would generalise the zh homophone handling and
is the recommended next step if list maintenance becomes a burden.
