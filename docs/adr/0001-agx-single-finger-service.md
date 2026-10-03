# ADR 0001 — One LSTM-based finger service on the AGX replaces gripper_record_api_v2

Date: 2026-07-14 · Status: accepted

## Context

Two programs need the AGX's fingers, and both open the same single-owner serial
ports (Dynamixel motors on the FTDI FT6RW7NN port, tactile sensor on ttyUSB0):

- `gripper_record_api_v2.py` — the HTTP API our demo server's GripperService
  uses for the scripted fallback (open/close/set_position/state).
- `run_LSTM5_toVLA.py` — the teammates' interactive terminal program running
  the LSTM finger actions (tactile-driven grasp/release), operated by hand on
  a second screen; models load per action (seconds of lag at the handoff), the
  loop stops only on a local keyboard press.

The three-object demo needs the laptop to *call* the finger actions at the
handoff gates, with models preloaded during the arm's approach.

## Decision

Build ONE new service for the AGX from the LSTM program's core
(`agx/lstm_gripper_service.py`, deployed to the AGX, port 5003). It owns both
serial ports and serves BOTH:

- the record_api_v2-compatible routes (`/health`, `/state`, `/command`,
  `/set_position`) so GripperService and the fallback work unchanged, and
- new LSTM routes: `/lstm/preload` (models + finger homing at auto-run start),
  `/lstm/start`, `/lstm/stop` (freeze in place), `/lstm/status`.

`gripper_record_api_v2.py` is retired. The teammates' terminal program remains
untouched on the AGX as the manual backup (mutually exclusive with the
service, same ports).

## Consequences

- Auto-start at the gates: reaching grasp z / the release point triggers the
  finger action; the UI gate button now means "stop the fingers and continue".
- Preload kills the model-load lag; finger homing moves to preload too (it
  takes ~4 s of serial writes).
- Deviation from the teammates' script: their menu HOMES the fingers before
  every action — before a release that would drop the object, so the service
  homes only on preload/grasp, never before a release. Confirm with teammates.
- The service must not import `keyboard` (needs root + local TTY); stopping is
  HTTP-only.
- If the service is down, the auto flow degrades gracefully: gates become
  plain human-confirm buttons (today's behavior).
