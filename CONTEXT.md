# CONTEXT — voice_pick demo domain glossary

- **ready point** — fixed arm home pose (490, 0, 425); every flow starts and ends here.
- **taught pose** — a per-object grasp hover recorded by jogging the arm over the object
  at the rig (fallback machinery). Source of truth for scripted picks.
- **on-top setpoint** — the (cx, cy) normalized claw-cam pixel where an object's YOLO box
  center sits when the tool is correctly above it. NOT the image center (claw cam is
  mounted off the tool axis). Per-object, calibrated from recordings
  (`data/calibration/claw_arrival.json`).
- **arrival check (A+C)** — decision that the arm is on top and may descend:
  (A) claw-YOLO box center within tolerance of the on-top setpoint, box not edge-clipped,
  at hover z; AND (C) the VLA's predicted next-step displacement is small. Checked at
  hover height only — never below z≈200 (eye-in-hand blind zone).
- **blind zone** — claw view below z≈200 where the object is under the fingers and YOLO
  degrades; no vision decisions there, descent is blind and vertical.
- **servo mode** — closed-loop x,y correction driven by the pixel error between the claw
  YOLO box center and the on-top setpoint, in capped mm steps, at hover z.
- **handoff / handoff gate** — control alternation with the teammates' AGX gripper code:
  we descend to the per-object grasp z, PAUSE (gate) while their LSTM picks, we move to
  the release point, PAUSE while they release, we return to ready. Only we command the
  arm; the AGX only commands the fingers. Reaching a gate point auto-starts the finger
  action; the gate button is the STOP: pressing it freezes the fingers where they are
  and lets the arm continue.
- **finger action** — one AGX LSTM behavior (trapezoid_grasp, trapezoid_release,
  knife_release, PCB_release): a 20 Hz tactile+position control loop that runs until
  stopped. Trapezoid has grasp+release; board and butter_knife have release only.
- **lift-under pick** — how board and butter_knife are picked: fingers stay OPEN, the
  descend slides them under the object, and the lift carries it — no grasp action.
  Only the trapezoid is actually grasped.
- **preload** — loading an object's finger-action models on the AGX (plus finger
  homing) at auto-run START, while the arm is still approaching — so the gate-point
  auto-start has zero model-loading lag.
- **grasp z** — per-object handoff height from recordings: trapezoid ≈175, board ≈140,
  butter_knife ≈150.
- **release point** — fixed place pose the arm carries the object to (fallback's place
  pose, y≈−204 side).
- **blind-state** — training/deploy parity rule: the proprioceptive state token is zeroed
  so the policy localizes from vision. Any checkpoint trained with `--blind_state` must
  be deployed with `--blind-state`.
- **memorized-mean collapse** — failure mode where the policy outputs the dataset-mean
  pose regardless of vision; broken by blind-state training.
- **multi-object model** — single RDT checkpoint finetuned on all three objects'
  episodes, target object selected by the per-episode instruction embedding
  (zh instructions: 拿起梯形 / 拿起電路板 / 拿起奶油刀).
- **side-cam locator** — stage-1 coarse localization: YOLO box center in cam1 (at the
  ready pose) mapped to table (x,y) by a per-object affine calibrated from recordings
  (`data/calibration/sidecam_locator.json`). Bakes in cam1's pose → must be refit when
  the rig moves (the claw servo, arm-mounted, is immune).
- **YOLO pick pipeline** — the classical (no-VLA) pick: side-cam locator → move at
  hover → claw servo to the on-top setpoint → scripted descend. One fallback rung;
  taught fixed spots are another.
- **fallback ladder** — demo execution order per object: VLA → YOLO pick pipeline →
  taught pose → full hardcoded pick→place.
- **voice command vocabulary** — the voice input (typed Send + browser mic, zh-TW and
  English) recognizes EXACTLY the three auto-run objects: trapezoid (梯形), board (電路板),
  butter_knife. Words for any other object in objects.yaml do NOT match (treated as
  "not understood"). In the voice context every knife-like word — 刀子/刀/菜刀/knife/cutter
  as well as 奶油抹刀/抹刀/奶油刀/butter knife/spreader — resolves to butter_knife; there is
  no standalone knife in the demo.
- **ASR tolerance** (ADR 0004) — browser speech is open-vocabulary and does not know the
  three objects exist, so it mis-ranks short words (a bare "knife" has returned "OK Google"
  at rank 0). The client therefore sends ALL of Chrome's alternatives and the server picks:
  first candidate naming any known object wins, so a top-ranked 剪刀 vetoes a 刀子 below it.
  Only if nothing known is named anywhere does a homophone pass run — a curated list for
  Chinese (到/倒/道/島 for 刀, 型↔形, 版↔板, all phonetic and invisible to edit distance) and
  difflib ≥ 0.85 for English. Fuzzy hits show confidence 0.7 and "Homophone match". Every
  command logs its alternatives, so the list can grow from real rig misses.
- **voice → auto-run** — a recognized voice command drives the SAME engine as the terminal
  CLI (auto_run_cli.py) and the Auto Run button: the three-object auto-run
  (VLA/locator → arrival/servo → gated tail → AGX fingers). It does NOT use the legacy
  fixed-pose voice pick. The flow is speak/type → NLU match card → operator clicks
  **Confirm** → run. Voice always runs LIVE (it does not consult the dry-run checkbox);
  the Confirm click is the single safety gate.
- **per-object rung** — each auto-run object has a fixed working mode, config-driven in
  vla_auto.objects: trapezoid = VLA, board = locator (the checkpoint is trapezoid-only),
  butter_knife = fixed-xy (use_recorded_xy, approach skipped). Voice and the Auto Run
  button default to this; explicit auto_run_cli.py flags override it.
- **Isaac Simulation panel** — the operator-facing dashboard surface for the 5070's
  live four-camera simulation view and control status.
  _Avoid_: VNC screen, API panel, 5070 panel.
- **demo object mirror** — a cross-system identity pairing for an object represented
  in both the physical and simulated demos: trapezoid ↔ red, board ↔ pcb, and
  butter_knife ↔ butterknife. The simulation cube has no physical mirror.
  _Avoid_: Object alias, name conversion.
- **best-effort mirror** — a single confirmed object intent that launches the physical
  workflow and its mapped simulation workflow near-simultaneously. Each side follows
  its own validated controller and timing; the physical workflow remains authoritative
  and continues when simulation is unavailable or fails, with both outcomes reported
  separately.
  _Avoid_: Lockstep mirror, synchronized run, atomic run.
- **mirrored stop** — one operator stop intent applied to both sides of a best-effort
  mirror at matching severity: Stop requests cooperative cancellation, while
  Emergency Stop applies the emergency latch.
  _Avoid_: Unified kill, always-emergency stop.
- **simulation mirror target** — the configured Home-TCP-relative destination used
  by a mapped simulation object during a best-effort mirror. It belongs to the
  simulated conveyor and is independent of the physical release point.
  _Avoid_: Converted release point, shared XY.
- **simulation gateway** — the Voice Pick host boundary through which local and
  Tailscale browsers reach the offline 5070 simulation service. Browsers do not
  require a direct route to the 5070 host.
  _Avoid_: Browser-to-5070 connection, direct simulation link.
