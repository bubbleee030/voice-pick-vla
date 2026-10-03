# Fallback Pick-&-Place (Model-Free) — Instructions

A deterministic, **model-free** pick→place for **trapezoid, board, tweezers**, used
when the VLA model path is unavailable or unreliable on the laptop. It uses **only**
the arm (Modbus) and the gripper (HTTP) — **no cameras, RealSense, AGX-tactile, or
CUDA** — which removes the fragile subsystems that used to crash recording/replay.

You **teach once** (grasp pose + grip + finger release) with a CLI tool, the values
are **baked into config**, and at run time you press **one button per object** in the UI.

---

## 1. How it works

For the selected object the runner executes this fixed sequence:

```
ready → hover → pregrasp → grasp (close grip) → hold → lift
      → ready (safe transit)
      → place_above → descend to place z → release (per-object) → retract → ready
```

- The **pick** is synthesized straight-down from a single recorded `grasp_pose`
  (hover/pregrasp/lift are z-offsets above it). One recorded pose, deterministic motion.
- The **place** goes to one shared approach point, descends to a per-object height,
  and runs a per-object finger release.

### Stability guards (why it won't "die" like before)
| Failure mode | Guard |
|---|---|
| Arm freezes mid-motion | Pose-poller is **paused during the run** (exclusive Modbus, no lock contention) + a **per-step in-position watchdog** aborts a move that never arrives. |
| Replay silently stops | **Every step logs** START/done; **any** exception routes to a safe-stop — no quiet thread death. |
| Gripper hangs/fails | Bounded **retries**; a failed **release** is a hard abort (the object never silently stays stuck). |

**On any failure: stop + hold.** The arm sends `motion_stop`, keeps **servo ON**,
freezes in place still holding the object, and logs a loud error. Recovery is manual.

### Config split (defaults vs calibration)
- `config/objects.yaml` → `fallback:` — committed, human-readable **seeds** (placeholders).
- `config/demo_config.yaml` → `fallback:` — shared settings (approach pose, speeds, watchdog).
- `config/fallback_taught.yaml` — **auto-generated** by the teach tool; deep-merged over
  the seeds at every run. Re-teaching just regenerates this file and can never corrupt
  the curated config. If it's missing, the seeds are used.

---

## 2. Prerequisites

- **Stop the demo server before teaching.** The arm controller allows a single Modbus
  client, so the server and the teach tool cannot both own it.
- Arm reachable (e.g. `192.168.1.232:502`) and gripper reachable (e.g. `192.168.1.100:5003`).
- Run everything in the **`voice_pick` conda environment**.
- The UI is in **classic** mode on this machine; the buttons appear in the **Quick Pick**
  panel as a **"Fallback P→P"** row (Trapezoid / Board / Tweezers). They also exist in modern mode.

---

## 3. Teach (Phase A) — CLI

Stop the server, then:

```bash
python scripts/fallback_teach.py --object trapezoid
# default env overlay config/env/ubuntu.local.yaml is auto-detected
```

You jog the arm yourself (pendant or hand-guide); the tool only **reads** pose and
**commands the gripper**. It never moves the arm.

### Commands
| Command | What it does |
|---|---|
| `pick` | Capture grasp **pose + closed grip** at once (recommended) |
| `pose` / `grip` | Capture grasp pose only / grip finger positions only |
| `open [d]` / `close [d]` / `set a b c` | Jog the gripper (to position fingers); optional settle delay |
| `rstep [delay]` | Append the **current finger position** as a release step |
| `ropen [delay]` | Append a final **full-open** release step |
| `rundo` / `rclear` | Remove last / clear all release steps |
| `z [mm]` | Capture drop **descend height** (current arm z, or an explicit mm value) |
| `approach` | Capture the **shared** place-approach pose (current arm pose) |
| `state` / `show` | Print live arm+gripper / the captured draft |
| `preview` | **Dry-run** the full plan (draft over saved config) — NO motion |
| `save` | Write the draft into `config/fallback_taught.yaml` (merged) |
| `servo on\|off` | Servo control (CAUTION: `off` may let the arm sag) |
| `quit` | Exit (warns on unsaved captures) |

### Example session
```
teach[trapezoid]> close          # close fingers on the object
teach[trapezoid]> pick           # capture grasp pose + grip
teach[trapezoid]> approach       # jog above the drop zone first, then capture
teach[trapezoid]> z              # capture the descend height there
teach[trapezoid]> set 40 40 40   # open fingers part-way
teach[trapezoid]> rstep          # snapshot as release step 1
teach[trapezoid]> ropen          # final full open
teach[trapezoid]> preview        # check the plan
teach[trapezoid]> save
teach[trapezoid]> quit
```
Repeat for `board` and `tweezers`.

---

## 4. Preview (dry-run, no motion)

Non-interactive, no hardware needed:

```bash
python scripts/fallback_teach.py --object trapezoid --preview
```

It prints the exact ordered plan the runner would execute — poses (raw + mm), speeds,
grip mode, release steps — plus warnings for any placeholder values still in use. The
preview uses the **same planner** as the runner, so what you see is what will run.

---

## 5. Run (Phase B) — UI

1. **Restart the demo server once** (to load the new code/UI/config). After this,
   re-teaching needs no restart — the taught file is read on every button press.
2. Connect the arm + gripper in the UI.
3. In the **Quick Pick** panel, click **Trapezoid / Board / Tweezers** under "Fallback P→P".
4. Watch the log: `START → move ready → … → release → … → COMPLETE`.

---

## 6. Config reference

`config/objects.yaml`:
```yaml
fallback:
  trapezoid:
    grasp_pose: [x, y, z, rx, ry, rz]   # controller units (um / m-deg)
    grasp_finger_pos: [f1, f2, f3]      # null -> use close()
    place:
      z_mm: 312.5                       # descend height at the drop zone
      release_steps:                    # ordered finger release
        - {pos: [f1, f2, f3], delay_s: 0.3}
        - open
```

`config/demo_config.yaml`:
```yaml
fallback:
  place_approach_pose: [x, y, z, rx, ry, rz]
  speeds: {travel: 60, descend: 20}
  settles: {grip_hold_s: 0.8}
  hover_z_offset_um: 180000
  pregrasp_z_offset_um: 35000
  lift_z_offset_um: 200000
  watchdog: {per_step_timeout_s: 25.0, motion_start_timeout_s: 3.0, gripper_retries: 2}
```

`config/fallback_taught.yaml` (auto-generated; do not hand-edit):
```yaml
fallback:
  trapezoid: { ... taught values ... }
shared:
  place_approach_pose: [ ... ]
```

---

## 7. Tuning (Phase 3)

Defaults seed `place.z_mm` and `place_approach_pose` to the safe **ready height** so the
first runs verify motion **without descending into the table**. Tune on the rig:
- `approach` + `z` to set the real drop location/height,
- `set`/`rstep`/`ropen` to record a clean per-object finger release,
- `preview` after each change, then `save`.

---

## 8. Troubleshooting

| Symptom | Likely cause / action |
|---|---|
| `arm not connected` | Server still owns the Modbus port — stop it. Check arm IP/port. |
| `gripper not reachable` | Gripper endpoint down — check `192.168.1.100:5003/5002`. Release needs the gripper, so the run won't start without it. |
| `safety reject <step>` | A pose is outside `safety_boundary`/`pose_limits`. Re-teach within bounds. |
| `watchdog: '<step>' did not reach in-position` | Arm didn't arrive in time — abort + stop/hold. Check the arm, in-position flag, and speeds. |
| Buttons missing in UI | You're on classic UI and the server wasn't restarted after the update — restart it. |
| Run aborted, arm holding object | Expected "stop + hold" on failure. Recover manually (servo stays ON, gripper unchanged). |

---

## 9. Files

- `scripts/fallback_teach.py` — teach + bake + preview CLI
- `src/services/arm_service.py` — `build_fallback_plan()` (planner) + `run_fallback()` (executor + guards)
- `src/runtime_config.py` — `load_taught_fallback()` / `merge_fallback()`
- `tools/voice_pick_demo.py` — `fallback_run` socket handler
- `tools/static/index.classic.html` / `app.classic.js` (and modern `index.html` / `app.js`) — the 3 buttons
- `config/objects.yaml`, `config/demo_config.yaml`, `config/fallback_taught.yaml` — config
