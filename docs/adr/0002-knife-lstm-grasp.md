# ADR 0002 — butter_knife becomes a real tactile grasp+release (was lift-under)

Date: 2026-07-18 · Status: accepted · Extends [ADR 0001](0001-agx-single-finger-service.md)

## Context

Until now `butter_knife` (and `board`) were **lift-under** picks: the fingers
stayed open, slid under the object, and lifted it — no grasp model, only a
release. The knife's finger wiring was therefore release-only (`knife_release`,
trained at `hidden64_ly2`).

Teammates have now trained a real tactile LSTM **grasp** for the knife and
retrained its release. Both live on the AGX at
`GP/training_pack/model_pack/LSTM/{knife_grasp,knife_release}/` with three size
variants each (`hidden32_ly2`, `hidden64_ly2`, `hidden64_ly3`). We want the demo
to actually grasp the knife with tactile feedback, like the trapezoid — not just
scoop it.

## Decision

Promote `butter_knife` to a **grasp+release** object (same handoff shape as
`trapezoid`). Wire the new models at the **`hidden64_ly3`** variant for both
(matches `trapezoid_grasp`'s depth; verified against the trained weights —
3 LSTM layers, hidden 64, input 6, output 3).

Two coordinated changes (both must land together):

1. **AGX `agx/lstm_gripper_service.py`** (the model registry):
   - `ACTIONS` += `"knife_grasp" → knife_grasp/hidden64_ly3` (kind `grasp`).
   - `ACTIONS["knife_release"]` variant `hidden64_ly2` → **`hidden64_ly3`**.
   - `OBJECT_ACTIONS["butter_knife"]` = `["knife_grasp", "knife_release"]`
     (was `["knife_release"]`) — so preload warms both models.

2. **Local `config/demo_config.yaml`** (`vla_auto.objects.butter_knife.lstm`):
   - add `grasp: "knife_grasp"` (release name unchanged; the AGX now serves the
     `ly3` weights behind that name).

`board` stays the lone lift-under object (release-only `PCB_release`).

## Consequences

- **The knife now runs a hands-off, two-phase grasp+lift.** The AGX prepares and
  latches a real closing goal without moving the fingers. The laptop then starts
  one uninterrupted lift and activates that goal at measured Z≥170. It waits for
  a real action-relative grasp with no time deadline, so there is no manual
  **PICK DONE** gate and no false success merely because the hand stayed near home.
- **Approach is unchanged.** `use_recorded_xy: true` still holds: the knife
  skips VLA/servo, moves to the recorded pick x,y, and descends blind to its
  `grasp_z_mm`. Only the *finger action at the bottom* changed (open lift-under
  → tactile grasp).
- **Both restarts required** to take effect: AGX service restart (registry) +
  demo server restart (demo_config). Both restarts also carry the earlier
  overload-latch hardening.
- **Variant provenance.** Chosen at deploy time from three trained sizes; if a
  different size validates better, change one line in `ACTIONS` (folder +
  variant string + `hidden,layers`) and redeploy — the filenames re-encode the
  variant, so all three fields must agree.
- **Manual reference not updated.** `scripts/run_LSTM5_toVLA.py` (teammates'
  hand-test menu, ADR 0001) still points `knife_release` at `hidden64_ly2` and
  has no `knife_grasp` branch. It is not part of the demo runtime; anyone using
  it to hand-test the new knife grasp must add that branch there separately.

## Rig sequence + diagnostic follow-up (2026-07-18)

The knife model closes fully in an isolated air trace, but the contact trace and
multiple full demo runs stayed near finger home through step 31. A 30 mm static
pre-grasp backoff did not change the demo result, so the air/contact difference
is still unresolved and must not be explained by another unmeasured guess.

The operator-defined knife sequence is now:

1. descend to the configured `grasp_z_mm` contact depth;
2. call `/lstm/prepare` at the bottom. The AGX records the measured
   `action_start_pos`, runs its tactile baseline and inference, but suppresses
   all motor writes;
3. hold contact for `grasp_contact_dwell_s: 3.0`. After the dwell, remain at the
   bottom without a deadline until inference produces a target that closes at
   least two fingers by 60 ticks from `action_start_pos`;
4. latch that first real-close target, pause inference, and start one
   uninterrupted normal lift toward the place-approach Z at
   `travel_speed_percent: 60`;
5. when measured arm Z first reaches `grasp_start_z_mm: 170`, call
   `/lstm/activate`. It synchronously writes the cached target and resumes live
   inference while the same arm move continues. Z=170 is an event trigger, not a
   waypoint, so the arm does not pause there;
6. accept grasp completion only after both the commanded goal and measured
   motors close at least two fingers by 60 ticks and the hand remains within the
   12-tick stability window for 1.2 seconds. This wait has no elapsed-time limit;
   STOP is the escape. Freeze the fingers, join the lift, then travel to release.

The old static pre-grasp experiment is disabled with `grasp_backoff_mm: 0`.
The LSTM's required 1-second tactile baseline now occurs during prepared contact,
before the lift. Prepared inference cannot move a motor. The first physical close
write occurs inside `/lstm/activate` while the arm is moving from Z=126 toward
Z≈425. CLI `--speed 30` still controls descent, not this 60% lift.

AGX `/lstm/status` exposes `drive_state`, `goal_ready`, `action_start_pos`,
`pending_goal`, `goal_step`, and `goal_pos`. The laptop logs preparation,
the cached goal, activation height, sampled `motor_pos`, tracking error, home
delta, and tactile. A close-valued goal with home-like motor feedback means the
motors did not track a real close command; a home-like goal means the model chose
to hold open in that context. This remains diagnostic evidence, not a claimed
root cause for the air/contact difference.

## Action-start boundary correction (2026-07-18)

Rig logs exposed a second, independent timing problem: the AGX runner called
`Fingers.home()` synchronously before **every** LSTM action. This contradicted
the service API contract and caused two visible failures:

- a release first stepped the closed hand open to `HOME_POS`, then began the
  learned release trajectory;
- a knife grasp that preload had already prepared repeated a blocking home
  operation after the arm crossed the Z=170 trigger.

`LSTMRunner.start()` is now action-kind aware. Release actions never home and
therefore take their baseline from the currently held pose. A matching
preloaded grasp skips the redundant home; a grasp without matching preload
still homes as a safety fallback. Grasp consumes the matching preload marker so
stale readiness cannot be reused later.

The arm schedule did not change: knife still uses one uninterrupted lift to
Z≈425 at `travel_speed_percent: 60`, and Z=170 remains only the measured event
threshold. CLI `--speed 30` does not control this concurrent lift.

## Conditioned completion and board gate policy (2026-07-18)

The reported step-31 trace proved that nominal-home displacement was an invalid
grasp predicate: the hand began slightly off nominal home, remained essentially
at its measured start pose, and was incorrectly declared settled. Grasp success
is now relative to each action's measured start position and requires a real
60-tick close on at least two fingers in both goal and feedback. There is no
grasp timeout. Release retains `release_settle_timeout_s: 8.0`.

`board` remains lift-under, but its manual pick gate is removed by an explicit
per-object policy: it descends to Z=124, holds contact for three seconds, emits
`auto_pick`, and performs the normal lift without `gate_pick` or
`pick_done`. The opt-in `auto_lift_after_dwell: true` is scoped to board;
other no-LSTM/failure paths retain manual confirmation.
