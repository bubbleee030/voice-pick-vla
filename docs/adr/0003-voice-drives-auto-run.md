# ADR 0003 — Voice input drives the live three-object auto-run (Confirm-gated)

Date: 2026-07-20 · Status: accepted

## Context

The demo has three separate "pick" engines wired to different triggers:

- **Voice** (typed Send + browser mic) → `voice_text` → NLU → `confirm_pick` →
  `arm.pick_fixed` — the legacy **fixed-pose** pick over the full 11-object
  `objects.yaml` catalog.
- **Fallback P→P** buttons → `fallback_run` → `arm.run_fallback` (hardcoded).
- **Auto Run** button *and* the terminal CLI (`scripts/auto_run_cli.py`) →
  `vla_auto_start` → `vla.start_auto` — the real three-object pipeline
  (VLA/locator → arrival/servo → gated tail → AGX fingers).

The CLI path is the one we've validated on the rig. The operator wants the voice
input to "work like the CLI" — i.e. drive `start_auto`, not the legacy
fixed-pose pick — for trapezoid, board, and (butter) knife, in zh-TW and English.

## Decision

Re-point the voice input at the auto-run engine. Specifically:

1. **Vocabulary = the 3 auto-run objects only.** Voice recognizes exactly
   `trapezoid` (梯形…), `board` (電路板/板子/板), and `butter_knife`. Every
   knife-like word — 刀子/刀/菜刀/knife/cutter **and** 奶油抹刀/抹刀/奶油刀/butter
   knife/spreader — resolves to `butter_knife`; the standalone `knife` object is
   not in the voice demo. Any other catalog word (剪刀, 湯匙, 筷子, …) does **not**
   match. Care is required so a word that merely *contains* 刀 (e.g. 剪刀) is NOT
   mis-mapped to the knife.

2. **Engine = `start_auto`.** A confirmed voice command emits the same
   `vla_auto_start` event the Auto Run button and CLI use. The legacy
   voice → `confirm_pick` → `pick_fixed` path is retired **for voice**; the
   quick-pick grid buttons are out of scope and unchanged.

3. **Per-object rung is config-driven; CLI flags override.** Each object's mode
   lives in `vla_auto.objects` — trapezoid = `vla`, board = `locator` (the
   deployed checkpoint is trapezoid-only), butter_knife = fixed-xy (via existing
   `use_recorded_xy`, which skips approach). Voice and the Auto Run button
   default to it; explicit `auto_run_cli.py` flags still win.

4. **Safety = Confirm click, always LIVE.** Flow: speak/type → NLU match card
   (shows object + the mode it will use + that it is a LIVE run) → operator
   clicks **Confirm** → live `start_auto`. Voice does **not** consult the
   `vla-dry-run` checkbox — the Confirm click is the single safety gate. (The
   checkbox still governs the Auto Run button.)

5. **Bilingual via the existing ZH/EN toggle.** The toggle sets the browser
   recognizer's language per utterance; typed text matches either language with
   no toggle. No auto-detect (browser SpeechRecognition can't do it reliably).

6. **Out-of-scope feedback names the 3.** Non-demo or garbled input shows, on the
   card and in the arm log, that only 梯形 / 電路板 / 奶油刀 are available by
   voice — no motion, no Confirm offered.

## Consequences

- Voice becomes the **one** trigger that always runs live regardless of the
  dry-run checkbox. This is deliberate (the operator asked for it) but surprising
  next to every other trigger — hence this ADR. The Confirm button is load-bearing
  as the sole gate; its card must clearly read "LIVE".
- The legacy fixed-pose pick is now reachable only via the quick-pick grid, not
  voice. If the grid is later removed, `pick_fixed` loses its last UI entry point.
- Because board rides the locator rung and butter_knife rides fixed-xy, voice
  inherits those objects' existing reliability caveats — voice does not add or
  fix any grasp/locator behavior, it only chooses the rung.
