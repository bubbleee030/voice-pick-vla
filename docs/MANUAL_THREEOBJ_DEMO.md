# Three-Object Demo — Rig Operator Manual

> 中文逐步測試手冊(終端機優先,UI 最後):`docs/TEST_STEPS_zh-TW.md`

Derived from `HANDOFF_threeobj_laptop.md` (2026-07-14). **Quick demo TOMORROW
(07-15); final in ~3 days.** This is the operator runbook: what to do, in
order, with exact commands. The §3 software (auto-run, arrival, servo) is a
build task for the laptop-side Claude session, not an operator step — it gets
a one-paragraph summary at the end.

**Roles:** only OUR stack commands the arm; the AGX (teammates) only commands
the fingers. The UI is the control surface; the AGX terminal sits on screen 2.

---

## Step 0 — Sync files from the 5090 bundle ✅ DONE (verified 2026-07-14 evening)

All files below are present and checked. Two notes from the verification:
- The plan doc arrived at `docs/superpowers/plan/` (singular) and was moved to
  the canonical `docs/superpowers/plans/` that all references point to.
- The handoff now exists twice (root `HANDOFF_threeobj_laptop.md` and
  `docs/HANDOFF_threeobj_laptop.md`, identical copies) — treat the root one as
  canonical.
- The calib scripts' local dependencies are confirmed present too:
  `data/models/yolo/3_white_trapezoid.pt` and `data/recordings/` (6 episode
  dirs: board, board0, butter_knife, trapezoid, trapezoid0, …).

| file | needed by |
|---|---|
| `scripts/claw_arrival_calib.py` | Step 3 (servo calibration) |
| `scripts/sidecam_locator_calib.py` | demo-room recalibration (mandatory, see below) |
| `scripts/vla_eval_recordings.py` | Day-2 checkpoint checks |
| `data/calibration/claw_arrival.json` | Step 3 + arrival module |
| `data/calibration/sidecam_locator.json` | side-cam locator mode + demo-room recalibration |
| `data/vla_embed_board_zh.pt`, `data/vla_embed_butter_knife_zh.pt`, `data/vla_embed_trapezoid_zh256.pt` | tomorrow's VLA run + Day 2 |
| `CONTEXT.md` | project glossary (don't create a local one — this is the canonical copy) |
| `docs/superpowers/plans/2026-07-14-three-object-demo.md` | full plan reference |
| `HANDOFF_threeobj_laptop.md` | already here |

---

## Step 1 — Teach board + butter_knife scripted picks (§1.1 — TODAY's highest priority)

This is tomorrow's demo floor: DONE = all 3 objects pick→place via the
scripted fallback. Trapezoid was already taught (2026-06-25,
`config/fallback_taught.yaml`) — it only needs a confirmation run.

**Config prep — already done 2026-07-14, no action needed:** `butter_knife`
was added to `config/demo_config.yaml → fallback.objects` and seeded in
`config/objects.yaml → fallback:`. It takes effect at the next server start
(configs load once at startup), which happens naturally in the flow below.

### 1a. Stop the demo server (arm + cameras are single-owner)

```bash
ps aux | grep "src.launcher" | grep -v grep   # find the PID
kill <exact PID>                              # never pkill -f patterns
```

### 1b. Teach each object (board, then butter_knife)

```bash
conda run --no-capture-output -n voice_pick python scripts/fallback_teach.py --object board
# then repeat with: --object butter_knife
```

(If the gripper won't connect, add `--no-gripper` to teach pose-only and
capture grips later.)

At the `teach[board]>` prompt, the flow that worked for trapezoid:

1. `state` — confirm pose + finger readout are live.
2. `open` — open the gripper.
3. Position the tool over the object at grasp height: `servo off`, guide the
   arm by hand (**caution: the arm can sag — support it**), then `servo on`.
   Or jog from the pendant if you prefer.
4. `pick` — captures grasp pose AND closed grip in one shot (it closes the
   gripper; have the object in place).
5. Release recipe: move/`set a b c` the fingers to the staged positions and
   `rstep` after each; finish with `ropen`. (`rundo`/`rclear` to fix mistakes.)
6. `z <mm>` — place descend height (or move the arm there and plain `z`).
7. `preview` — dry-run prints the exact 9-step plan, NO motion. Read it.
8. `save` — writes into `config/fallback_taught.yaml` (merged, non-destructive).
9. `quit`.

### 1c. Verify all 3 objects

```bash
# no-hardware sanity check of each plan:
conda run -n voice_pick python scripts/fallback_teach.py --object board --preview
conda run -n voice_pick python scripts/fallback_teach.py --object butter_knife --preview
conda run -n voice_pick python scripts/fallback_teach.py --object trapezoid --preview
```

Then restart the server (Step 2's command) and run the full fallback from the
UI for **each** object: board, butter_knife, trapezoid. 3/3 clean pick→place =
Step 1 done and tomorrow has its floor.

---

## Step 2 — Speed / smoothness A/B (§1.2 — already queued from last session)

The server config already has the candidates enabled: MovL on
(`use_linear_move: true` in the env overlay), speeds travel 75 / descend 30,
and NEW acceleration registers **ACC=200 / DEC=50** (2×/5× the ship values) in
`config/modbus_config.yaml`. The open question from last session: do the
ACC/DEC registers actually lift MovL's ~17 mm/s² acceleration limit?

1. Start the server (this also loads the butter_knife config from Step 1):

   ```bash
   conda run -n voice_pick python -m src.launcher start full --host 0.0.0.0 \
     --port 8090 --ui-mode classic --env-config config/env/ubuntu.local.yaml
   ```

2. In a **second terminal**, start the read-only audit (safe to run alongside
   anything):

   ```bash
   conda run --no-capture-output -n voice_pick python scripts/arm_speed_audit.py
   ```

3. Run a fallback pick from the UI, then Ctrl-C the audit for the summary.

4. **Read the result against the 2026-07-13 baseline** (peaks 58–66 mm/s on
   the 190–241 mm travel moves):
   - Long-move peaks rose to ~80–90 mm/s → the registers work; we can tune
     ACC/DEC up or down for the demo feel (and re-check that 30% descend is
     still comfortable).
   - Peaks unchanged → the accel limit lives in the vendor app (DRAStudio
     .msi) system parameters — bring the app and we'll find the exact one.

5. Pick the demo speed numbers and leave them set in
   `config/demo_config.yaml → fallback.speeds` (restart to apply).

---

## Step 3 — Servo calibration at hover (§1.3, ~10 min — needs Step 0 sync)

Four numbers, measured over the **trapezoid** at hover z (record which z you
used — write it down):

1. **x direction sign** — jog the arm +20 mm in world x (stepthrough nudge
   `x+20` or UI jog); note which way the YOLO box moves in the claw pane.
2. **y direction sign** — same with `y+20`.
3. **mm-per-pixel** — from the same jogs: known mm moved ÷ pixels the box
   center shifted.
4. **Setpoint check** — jog until the tool is *physically* centered over the
   trapezoid, then read the box center and compare with
   `data/calibration/claw_arrival.json` (exact values now on disk: trapezoid
   cx=0.4837, board cx=0.4747, butter_knife cx=0.3667). cx should match;
   **cy must be re-derived at YOUR hover z** (parallax — the shipped tol_cy
   values are 0.28–0.50 wide for exactly this reason) — run
   `claw_arrival_calib.py` filtered to that z, or just record the cy you read
   while centered.

   Class-name note: the claw model (`3_white_trapezoid.pt`) reports the
   objects as `board`, `knife`, and `trapezoid_white` — that's the label
   you'll see on the boxes, already mapped per object in `claw_arrival.json`.

Reminders: the claw cam is off-axis, so the setpoint is NOT the image center.
Detection degrades below z≈200 (object under the fingers) — calibrate at
hover only. **Never move the two side cams** — that invalidates the VLA
checkpoint.

Give the 4 numbers + hover z back to the Claude session: they go into the §3
servo/arrival build.

## Step 4 — Servo + arrival test on trapezoid (§1.4) — software is READY

The §3 build is done (2026-07-14 evening, all sim checks passing). At the rig,
after Step 3's numbers are in `data/calibration/claw_servo.yaml`
(jacobian + `calibrated: true`):

1. Restart the server. In the VLA panel there is a new row: object selector,
   VLA/locator mode selector, **Auto Run** button.
2. First run: **Dry run checked** — the loop predicts, servos, and logs every
   move `[DRY]` without touching the arm. Watch the step log converge.
3. Then live: dry-run off, object at a taught spot, low speeds (defaults:
   strides 12%, descend 15%), hand near STOP. Flow: approach → servo →
   ARRIVED → descend to grasp z → **PICK DONE button** (AGX picks) → carry to
   release point → **RELEASED button** → back to ready.
4. Knob row (conf / tol_cx / tol_cy / stay / srv + Apply) tunes the arrival
   thresholds live — no restart (demo-room lighting).

---

## TOMORROW — quick demo run-of-show (§2)

- **board + butter_knife: scripted fallback** from the UI (Step 1's result).
- **trapezoid: VLA.** Stop the demo server first (cameras + Modbus are
  single-owner), place the object at the ready/center spots, then:

  ```bash
  conda run --no-capture-output -n voice_pick python scripts/vla_stepthrough.py \
    --blind-state --freeze-z --linear \
    --embed data/vla_embed_trapezoid_zh256.pt
  ```

  (`--blind-state` is REQUIRED with the blind checkpoint; default checkpoint
  is already `trapezoid_full_blind_nchc_20260707_024347`. Keys at the prompt:
  Enter=go, `x+20 y-15` nudge, `r` re-predict, `+`/`-` speed, `l`
  MovP↔MovL, `s` stop.)

  Flow: VLA approach to hover → descend to the z=175 handoff plane
  (stepthrough stops there by design) → **AGX teammates run the fingers** →
  place + return via the existing hardcoded flow, same as today's runs.

Pre-demo checklist:
- [ ] **One object on the table at a time** (per the plan: placed at the
      taught spots / model-accurate spots — arrival logic assumes a single
      class-filtered top-confidence box).
- [ ] Server restarted fresh; all 3 cameras streaming (claw watchdog will
      self-heal USB wedges within ~4 s, but check the pane before starting).
- [ ] Audit tool available in a second terminal (optional but nice for logs).
- [ ] Demo speeds from Step 2 in place.
- [ ] AGX terminal up on screen 2; agree on the verbal go/no-go for the
      finger handoff.
- [ ] If the rig moved since the last calibration: run the DEMO ROOM
      recalibration section below FIRST (mandatory).

---

## Day 2 — the multi-object checkpoint (§4)

The 5090 side evals `threeobj_blind_nchc_*` offline first (placements +
recordings evals, all 3 objects).

- **PASS** → sync the new checkpoint into `data/datasets/`, then run each
  object through the same stepthrough flow with `--checkpoint <new dir>` and
  the per-object embed (`--embed data/vla_embed_board_zh.pt` etc. — the
  instruction embed selects the object). board/knife then go
  VLA→servo→descend like trapezoid.
- **FAIL** → the final demo keeps tomorrow's composition: still 3/3 objects,
  voice-driven, smooth.

---

## ⚠ DEMO ROOM — mandatory recalibration whenever the rig moves (§3.7, ~10 min)

Moving the rig changes cam1's pose, which makes **every side-cam affine stale**.
The claw setpoints survive (arm-mounted), but do this at the demo room BEFORE
the demo — the side-cam locator mode is dead without it:

1. Place the trapezoid at 4–5 spread-out spots on the table. For each spot:
   jog the arm directly over it (that pose's x,y = ground truth), save a cam1
   frame, and append a CSV row:
   `trapezoid,<image_path>,<true_x>,<true_y>`.
2. Refit — this also auto-transfers the camera shift to board and knife
   (verified mechanism; a 5-point refit lands ≈21/10 mm LOO, well inside the
   servo catch basin):

   ```bash
   python scripts/sidecam_locator_calib.py --points-csv <csv> \
     --transfer-old data/calibration/sidecam_locator.json
   ```

3. If the lighting is odd, sanity-check one detection per object at conf 0.25.
4. Re-verify ONE claw setpoint (the Step 3 jog check) in case the claw mount
   was bumped in transport.

Placement note for the locator mode: **butter_knife's affine has only ever
seen x≈490** (its 4 episodes) — place the knife near x≈490, or add 2–3 knife
rows to the refit CSV.

---

## §3 software — BUILT (2026-07-14 evening; sim-verified, awaiting rig calibration)

What exists now (spec: `HANDOFF_threeobj_laptop.md` §3; plan:
`docs/superpowers/plans/2026-07-14-three-object-demo.md`):

| piece | where | verified by |
|---|---|---|
| Arrival check + pixel servo (§3.2/3.3) | `src/arrival.py` + `data/calibration/claw_servo.yaml` | `scripts/arrival_offline_eval.py` vs recordings: 0 false-arrivals on all 3 objects; servo steps point the right way 77/81, closing ~8–12 mm/step |
| Side-cam locator (§3.6) | `src/locator.py` | `scripts/locator_offline_eval.py`: 22/22 episodes located, MAE ≤ 6 mm, max 12 mm (needs ±100) |
| Auto-run loop + gated tail (§3.1/3.4) | `src/services/vla_service.py` (`start_auto`) | `scripts/test_auto_run_sim.py`: full flow end-to-end, 10/10 checks |
| UI controls + runtime knobs (§3.5) | VLA panel (`vla_compact.js`, socket events `vla_auto_start`/`vla_gate`/`vla_tune`) | syntax + config checks |
| Per-object config | `config/demo_config.yaml → vla_auto:` (embeds, zh instructions, grasp z 175/140/150, speeds) | loaded + validated |

The fallback ladder per object is now:
**VLA → side-cam locator → taught pose → full hardcode** — the first two
rungs share the same arrival/servo/tail; the mode selector switches them.

**Rig calibration DONE 2026-07-14 20:18–20:33** — `servo_calibrate.py` wrote
`data/calibration/claw_servo_calibration.json` (per-object jacobian, setpoint,
and measured z). `ArrivalServo` loads that file as **authoritative** whenever
it exists: it survives `claw_arrival.json` regeneration, flips
`calibrated: true`, and carries each object's vision plane. Current values:

| object | vision plane z | setpoint (cx,cy) | tol @ 12 mm (cx,cy) |
|---|---|---|---|
| trapezoid | 425 | 0.531, 0.582 | 0.021, 0.037 |
| board | **364.9** | 0.530, 0.589 | 0.020, 0.043 |
| butter_knife | 425 | 0.396, 0.460 | 0.018, 0.037 |

Two behaviors that came out of the calibration session:

- **Tolerances are now computed, not hand-set.** The `tol_*` fields in
  `claw_arrival.json` are ignored by the runtime. The single knob is
  `arrival_tol_mm` (default 12) in `claw_servo.yaml` / the UI's `tol_mm`
  field — it converts to pixel tolerances per object through the measured
  jacobian. Lower it for a tighter arrival (more servo steps), raise it if
  arrival never triggers.
- **Board seek-descend.** The board is invisible to the claw above ~z365, so
  the auto loop runs arrival/servo at each object's *calibrated* plane: once
  the coarse approach settles, it descends in 30 mm steps (`seek_step_mm`)
  from the 425 travel plane to the object's plane before gating on vision.
  Sim-verified: board scenario (invisible above z380) completes end-to-end.

## AGX finger integration (ADR 0001 — built 2026-07-14 night, NOT yet deployed)

The teammates' LSTM program is now callable from our pipeline. Design (full
rationale: `docs/adr/0001-agx-single-finger-service.md`):

- **One service owns the fingers**: `agx/lstm_gripper_service.py` (in this
  repo, to be copied to the AGX) replaces `gripper_record_api_v2.py` on port
  5003. It serves the same routes the fallback uses (`/state`, `/command`,
  `/set_position`) plus `/lstm/preload`, `/lstm/start`, `/lstm/stop`,
  `/lstm/status`. The teammates' terminal program stays as manual backup
  (never run both — same serial ports).
- **Preload at auto-run start**: pressing Auto Run fires `/lstm/preload`
  (models → GPU + finger homing, ~5–10 s) while the arm is still approaching —
  zero load lag at the handoff.
- **Gates auto-start, button stops**: reaching grasp z auto-starts the object's
  grasp model — `trapezoid_grasp` / `knife_grasp` (butter_knife now does a real
  tactile grasp, see ADR 0002); only **board** stays an open-finger lift-under
  with no grasp action. Reaching the release point auto-starts the object's
  release model (`PCB_release` / `knife_release` / `trapezoid_release`). The
  PICK DONE / RELEASED buttons now stop the fingers (freeze in place) and let
  the arm continue — same human judgment as their 'x' key, zero coordination.
- **Graceful degradation**: AGX down/unreachable → warning in the log, gates
  become plain confirm buttons, arm flow unaffected. Dry runs never touch the
  AGX.

**Deploy checklist (needs SSH access — `ssh-copy-id agx` once):**
1. `scp agx/lstm_gripper_service.py agx:~/Desktop/GP/`
2. On the AGX: stop `gripper_record_api_v2.py`, then
   `python3 ~/Desktop/GP/lstm_gripper_service.py` (same env as their
   program + flask; no sudo — the keyboard module is gone).
3. From the laptop: `curl http://192.168.1.100:5003/health` → `lstm_v1`.
4. Verify fallback still works (open/close via UI), then one preload:
   `curl -X POST http://192.168.1.100:5003/lstm/preload -H 'Content-Type: application/json' -d '{"object":"trapezoid"}'`.
5. **Confirm with teammates**: the service deliberately does NOT home the
   fingers before a release (their menu homes before every action — mid-hold
   that would drop the object). Also verify the `o`/`c` finger poses
   (`OPEN_POS`/`CLOSE_POS` constants) against their record_api_v2 values.

## Safety notes (unchanged, always)

- `--blind-state` REQUIRED with all blind-trained checkpoints.
- Arrival/servo decisions at hover z only; **never gate on vision below
  z≈200** (blind zone under the fingers).
- Safety box + `--max-step` stay active in every mode; first auto-run at low
  speed with a hand near stop.
- Side cams must never move. Arm is never auto-homed mid-flow.
- Kill servers by exact PID; never `pkill -f` patterns.
