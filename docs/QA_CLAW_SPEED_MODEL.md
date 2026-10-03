# Q&A — Claw Cam, Model Guidance, Arm Speed

Consolidated answers from the 2026-07-07 → 2026-07-13 debugging sessions, with
step-by-step instructions. Registers reference: `625 arm/DCS, DCV Modbus
Address Manual (User)_20250516.xlsx` (DRS and DRV sheets agree on everything we
use).

---

## Q1. The claw cam pane lags behind the two side cams — why? (FIXED)

**Cause (software, in `src/services/claw_camera_service.py`):** with
`enable_yolo: true`, the stream displayed the **YOLO-annotated frame** — i.e.
the frame YOLO last ran on — instead of the live capture. The pane only
refreshed when a YOLO inference finished (GPU shared with the 1.2B VLA).
Measured before the fix: side cams **14.8 fps** effective refresh, claw only
**4.5 fps** with up to 351 ms stale content.

**Fix (2026-07-07):** the detect loop now stores only the **box coordinates**
(`latest_boxes_norm` + timestamp); the stream always shows the freshest capture
and draws the boxes on top. Boxes older than `detection_stale_s` (default 1.0 s)
are hidden instead of freezing the picture. After the fix: **12.2 fps, ~83 ms**
— visually at parity with the side cams.

**How to verify anytime** (measures exactly what the browser shows):

```bash
# server running; measures delivered fps vs content-change fps
python3 <scratch>/stream_probe.py http://localhost:8090/stream/claw/rgb 6
# or check the service status fields: frame_age_s / detect_age_s
```

**Note — `select() timeout` warnings in the server log** mean the claw USB is
in its known wedge state (opened but not streaming, kernel `usb 3-4.3.2`
disconnects). It usually self-heals in 30–60 s; a `ClawCameraService.stop()` /
`.start()` (or the Restart button / server restart) forces a fresh open. This
is a hardware/USB issue, separate from the lag fix. The fallback pick does not
use the claw, so runs are unaffected.

---

## Q2. Can I adjust the claw cam quality? Will it affect the stream?

Yes. Profiles live in `config/demo_config.yaml → cameras.claw.profiles`
(`quality` = 960×540 capture / 640×360 stream / JPEG 65, `balanced` = default,
`low_latency`). To switch:

1. Edit `config/env/ubuntu.local.yaml`, under `cameras: claw:` add:
   ```yaml
   performance_profile: "quality"
   ```
2. Restart the demo server (conda, **not** venv):
   ```bash
   conda run -n voice_pick python -m src.launcher start full --host 0.0.0.0 \
     --port 8090 --ui-mode classic --env-config config/env/ubuntu.local.yaml
   ```

Effects: sharper claw view, more USB bandwidth (if the `usb 3-4.3.2` error
storms return, go back to `balanced`), slightly slower YOLO passes (boxes
update less often — the video itself stays fresh thanks to the Q1 fix).
**No effect on the VLA model**: the claw is NOT a model input (the model was
trained on the 2 side cams only; deploy feeds a blank for slot 3).

---

## Q3. Can YOLO make the model predict more accurately?

Not by feeding YOLO *into* the model (its input format — 2 side-cam images +
pose — is fixed at training). But YOLO can correct the model's *output*, three
ways, in increasing power:

1. **Sanity gate** — if the claw-cam detection disagrees strongly with the
   model's target, refuse and re-predict.
2. **Visual servoing** (recommended next build) — model gets the arm near the
   object at hover, then a loop: detection-center pixel error → small capped
   x,y moves → until the object is under the tool → scripted descent to z=175.
   Erases the ~50–60 mm x-bias. Needs a one-time 4-number calibration at the
   rig (x/y direction signs, mm-per-pixel at hover z, setpoint pixel — NOT
   image center, because the claw cam is offset from the tool axis). Verified
   feasible: YOLO fires on the trapezoid from hover (`board 0.89`), frames are
   fresh after the Q1 fix.
3. **Auto-labeling** — detections + calibration give true object positions →
   fit `--x-cal` from data, and label episodes for the next finetune.

**Do NOT move the side cams** — the checkpoint was trained with them exactly
where they are; relocating them invalidates the model until retrained. The claw
cam mount is fine as-is. The claw YOLO model is set in
`config/env/ubuntu.local.yaml → cameras.claw.model_path`.

---

## Q4. Can I correct a wrong prediction from the terminal? Can the model learn it realtime?

**Correcting: yes — built into `scripts/vla_stepthrough.py`.** At every move
prompt:

| key | action |
|---|---|
| `Enter` | execute the shown move |
| `x+20`, `y-15`, `x+20 y-15` | nudge the pending target by mm (re-checks box/max-step, does NOT re-predict) |
| `r` | re-predict from the current measured pose |
| `t` | stamp current arm pose as the TRUE object x,y (ground truth log) |
| `+` / `-` | speed ±5% |
| `l` | toggle MovP ↔ MovL |
| `s` / `q` | stop |

**Human-guided re-prediction** also works: refuse a move, jog the arm above the
object (UI / pendant), press `r` — the new prediction starts from the new pose.

**Realtime learning: the model weights cannot learn online** (a 1.2B diffusion
policy only learns by training on the 5090). But your corrections are learned
instantly one level up: every nudge is logged (with `--record`, to
`corrections.csv`), and at session end the tool prints the mean correction and
the exact `--x-cal m,b` to apply next run. The corrections are also labeled
data for the next finetune — that's the permanent version.

---

## Q5. The arm speed doesn't match the settings — what is going on?

**The speed chain (all confirmed against the Modbus manual):**

```
actual speed = 0x0324 (GO/JOG speed %)          ← what move_to writes per move
             × 0x0246 (global override, 0.1%)   ← the old Windows app's slider;
                                                   idle default is 60.0%, but our
                                                   controller writes 100% each move
             × operation mode (0x0139)          ← T1 caps speed; ours reads AUTO (no cap)
             × motion type: MovP vs MovL        ← the big one, see below
```

**MovP (command 301, what we always used) is joint-interpolated:** the % applies
to JOINT speeds, so Cartesian mm/s depends on which joints move. Measured from
the 2026-07-13 fallback run log:

| step | cmd % | path | time | real speed |
|---|---|---|---|---|
| hover (y+z swing) | 60% | 191 mm | 5.9 s | 32 mm/s |
| pregrasp (z down) | 20% | 145 mm | 13.5 s | 11 mm/s |
| place_above (pure y) | 60% | 204 mm | 4.3 s | **47 mm/s** |
| place_down (z down) | 20% | 245 mm | 22.0 s | **11 mm/s** |
| retract (z up) | 60% | 245 mm | 7.9 s | 31 mm/s |

So the % knob works (60% ≈ 3× the 20% speed), but **the same % is ~1.5× faster
sideways than vertically** — that's the "weird in real" feeling. Two more
contributors: each step waits on the in-position flag (~0.5–2 s dead time per
step → stuttery), and the 20% long descents are slow *by design*
(`speeds: travel 60 / descend 20` — raise `descend` or teach an intermediate
point if the crawl bothers you).

**MovL (command 302) = straight Cartesian line, consistent speed. Now supported
(2026-07-13):**

- Per move: `ArmController.move_to(..., linear=True)`
- Stepthrough: `--linear` flag, or press `l` live at any prompt
- Globally (demo UI + fallback too): `config/modbus_config.yaml → motion:
  use_linear_move: true` (overridable from the env overlay under `modbus:`)

Caution: MovL alarms instead of deviating if the straight line passes near a
singularity/joint limit. First try: short mid-workspace move at low %; if it
alarms, reset alarms and toggle back with `l`.

**Known readback quirks** (why we measure instead of trusting readbacks): logs
show 2 moves where speed wrote 80% but read back 100%, 8 garbage mode readbacks,
and user/tool frame always reading 47.

### 2026-07-13 audit findings — the intermittent 100% runaway, ROOT-CAUSED & FIXED

Running `arm_speed_audit.py` alongside a fallback pick caught it red-handed:
the log commanded `grasp @ 20%` and `ready_transit @ 60%`, the readback even
verified those values — yet the registers AND the physical motion showed both
moves ran at **100%** (grasp descended at ~59 mm/s instead of ~14).

**Cause: our own per-move override write.** `move_to` wrote 0x0246 = 1000
(100%) before every move. The audit proved 0x0246 is **coupled** to 0x0324
(during every GO, override ≡ speed% × 10) and applied **asynchronously** — when
the pending "set 100%" landed *after* the GO latched its speed, the move ran at
100%. Roughly 2 in 9 moves that run; also explains the historical
"wrote 80% read 100%" warnings.

**Fix (2026-07-13):** the override is now written once at `servo_on()` (no GO
to race) and never in `move_to`. Verify on the next runs: with the audit tool
running, no move should ever report a speed above its commanded %.

Also confirmed by this audit: the speed % is **linear** (20/60/100% → 12/35/59
mm/s on z, 61/102 mm/s on y), y-moves are ~1.5× faster than z at the same %
(MovP joint geometry), and the 0x00FE Cartesian-speed register is an IEEE754
float32 in mm/s (the audit tool now decodes it correctly as `cart_mms`).

---

## Q6. `arm_speed_audit.py` fails with "no response from 127.0.0.1:1502" — why?

`127.0.0.1:1502` was an old local Modbus forwarder that **no longer runs**
(nothing listens on that port; it's a stale entry in
`config/env/ubuntu.local.yaml → modbus.connection`). The demo server actually
connects **directly** to the controller via
`config/env/ubuntu.local.yaml → demo.arm.connections`
(`192.168.1.232:502`, `192.168.1.233:502`).

**Fixed:** the tool now auto-tries the same endpoints the demo server uses.
Just run:

```bash
conda run --no-capture-output -n voice_pick python scripts/arm_speed_audit.py
# manual override if needed:
#   ... arm_speed_audit.py --host 192.168.1.232 --port 502
```

It is **strictly read-only** (never commands motion). Usage: start it in a
second terminal, run any moves (stepthrough / fallback / UI), Ctrl-C for the
summary table (per move: commanded % / override / op-mode vs measured mm/s).
Logs to `data/observe_logs/speed_audit_*.csv`.

If no endpoint answers at all: the controller is off, or the laptop's route to
the arm flipped between the wired switch and WiFi — check
`systemctl --user status arm-route-autofix.timer` and `ping 192.168.1.232`
before suspecting hardware.

**Live values read 2026-07-13 (arm idle at ready):** `opmode=Auto` (T1 cap
ruled out), `override=600` (60.0% — the pendant default; our moves override it
to 100%, but any external program that doesn't write 0x0246 runs 40% slower
than it thinks), `acc=100 dec=10`, `speed%=60` (left over from the last
fallback move).
