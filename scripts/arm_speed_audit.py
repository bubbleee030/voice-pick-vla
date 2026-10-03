#!/usr/bin/env python3
"""PASSIVE arm speed audit — READ-ONLY, never commands motion.

Answers "is the arm actually moving at the speed we set?" with data. Run it in a
second terminal while anything else moves the arm (vla_stepthrough, the demo UI,
the fallback pick). It samples the controller's own telemetry registers and, for
every detected move, reports commanded speed% / global override / operation mode
vs the MEASURED Cartesian speed (both the controller's live speed register and a
pose-derived value, so we can also validate the register's unit).

Registers (DCS/DCV manual, identical in DRS+DRV sheets):
  0x00F0..0x00FB  Cartesian pose x,y,z,rx,ry,rz (DW, um / 0.001deg)   R
  0x00FE          Cartesian speed (DW = IEEE754 float32, mm/s;
                  verified against pose-derived speed 2026-07-13)     R
  0x00E0          motion status (0 stopped, 1 running)                R
  0x0324          JOG/GO speed percent (what move_to writes)          R/W
  0x0246          Robot Speed Override, 0.1% units — COUPLED to 0x0324:
                  during a GO it always reads speed% x 10; writing it
                  per-move raced the GO and forced 100% runs (fixed:
                  written once at servo_on)                           R/W
  0x0139          operation mode (1 T1, 2 T2, 3 Auto) — T-modes cap TCP speed
  0x030A/0x030C   ACC/DEC (DW) — never written by our controller; persisted
  0x031F          GO in-position flag

Usage:
    conda run --no-capture-output -n voice_pick python scripts/arm_speed_audit.py
    # then trigger moves from the other terminal; Ctrl-C to stop and summarize
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import struct
import sys
import time
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BUNDLE_ROOT))

from pyModbusTCP.client import ModbusClient  # noqa: E402

REG_POSE, REG_CART_SPEED, REG_MOTION = 0x00F0, 0x00FE, 0x00E0
REG_SPEED_PCT, REG_OVERRIDE, REG_OPMODE = 0x0324, 0x0246, 0x0139
REG_ACC, REG_DEC, REG_INPOS = 0x030A, 0x030C, 0x031F
OPMODE = {0: "NotWired", 1: "T1", 2: "T2", 3: "Auto"}


def dw(lo: int, hi: int) -> int:
    raw = (hi << 16) | lo
    return raw - 4294967296 if raw & 0x80000000 else raw


class Sampler:
    def __init__(self, host: str, port: int, unit: int):
        self.c = ModbusClient(host=host, port=port, unit_id=unit, auto_open=True)
        self.c.timeout = 1.0

    def read(self) -> dict | None:
        pose_r = self.c.read_holding_registers(REG_POSE, 16)  # 0x00F0..0x00FF: pose + cart speed
        misc = {
            "speed_pct": self.c.read_holding_registers(REG_SPEED_PCT, 1),
            "override": self.c.read_holding_registers(REG_OVERRIDE, 1),
            "opmode": self.c.read_holding_registers(REG_OPMODE, 1),
            "motion": self.c.read_holding_registers(REG_MOTION, 1),
            "inpos": self.c.read_holding_registers(REG_INPOS, 1),
            "acc": self.c.read_holding_registers(REG_ACC, 2),
            "dec": self.c.read_holding_registers(REG_DEC, 2),
        }
        if pose_r is None:
            return None
        out = {"t": time.time()}
        for i, k in enumerate(("x", "y", "z", "rx", "ry", "rz")):
            out[k] = dw(pose_r[2 * i], pose_r[2 * i + 1]) / 1000.0
        # 0x00FE is an IEEE754 float32 in mm/s (verified 2026-07-13: decoded
        # values track pose-derived speeds), not a scaled integer.
        bits = ((pose_r[15] & 0xFFFF) << 16) | (pose_r[14] & 0xFFFF)
        cart = struct.unpack("<f", struct.pack("<I", bits))[0]
        out["cart_mms"] = round(cart, 1) if math.isfinite(cart) and 0 <= cart < 1e5 else 0.0
        for k in ("speed_pct", "override", "opmode", "motion", "inpos"):
            v = misc[k]
            out[k] = v[0] if v else None
        for k in ("acc", "dec"):
            v = misc[k]
            out[k] = dw(v[0], v[1]) if v and len(v) == 2 else None
        return out


def candidate_endpoints() -> list[tuple[str, int]]:
    """Same endpoints the demo server uses: demo.arm.connections from the env
    overlay first (direct controller IPs), then any modbus connection entries.
    The old 127.0.0.1:1502 forwarder no longer runs, so a hardcoded default
    would always FAIL — mirror the server's config instead."""
    import yaml
    cands: list[tuple[str, int]] = []

    def add(host, port):
        if host and (host, int(port)) not in cands:
            cands.append((host, int(port)))

    for path in (BUNDLE_ROOT / "config/env/ubuntu.local.yaml",
                 BUNDLE_ROOT / "config/demo_config.yaml"):
        try:
            cfg = yaml.safe_load(path.read_text()) or {}
        except Exception:
            continue
        demo = cfg.get("demo", cfg)  # overlay nests under demo:, demo_config may not
        for c in ((demo.get("arm") or {}).get("connections") or []):
            add(c.get("host"), c.get("port", 502))
        mb = (cfg.get("modbus") or {}).get("connection") or {}
        add(mb.get("host"), mb.get("port", 502))
    try:
        mbc = yaml.safe_load((BUNDLE_ROOT / "config/modbus_config.yaml").read_text()) or {}
        conn = mbc.get("connection") or {}
        add(conn.get("host"), conn.get("port", 502))
    except Exception:
        pass
    return cands


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only arm speed audit (no motion commands).")
    ap.add_argument("--host", default=None,
                    help="controller/forwarder host; default = auto-try the demo server's "
                         "configured arm endpoints (demo.arm.connections, then modbus)")
    ap.add_argument("--port", type=int, default=502)
    ap.add_argument("--unit", type=int, default=2)
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--move-thresh", type=float, default=0.8,
                    help="mm between samples that counts as 'moving'")
    ap.add_argument("--out", default=None, help="CSV path (default data/observe_logs/speed_audit_<ts>.csv)")
    args = ap.parse_args()

    out = args.out or str(BUNDLE_ROOT / "data/observe_logs" / f"speed_audit_{time.strftime('%m%d_%H%M%S')}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    endpoints = [(args.host, args.port)] if args.host else candidate_endpoints()
    s = first = None
    for host, port in endpoints:
        s = Sampler(host, port, args.unit)
        first = s.read()
        if first is not None:
            print(f"[ok] arm answered at {host}:{port}")
            break
        print(f"[--] no response from {host}:{port}")
    if first is None:
        print(f"[FAIL] no arm on any endpoint {endpoints} (unit {args.unit}).\n"
              "       Is the controller powered on and reachable? Try: "
              "ping 192.168.1.232, then --host 192.168.1.232 --port 502")
        return 1
    print(f"[ok] connected. pose x={first['x']:.1f} y={first['y']:.1f} z={first['z']:.1f}  "
          f"speed%={first['speed_pct']} override={first['override']} "
          f"opmode={OPMODE.get(first['opmode'], first['opmode'])} "
          f"acc={first['acc']} dec={first['dec']}")
    print(f"[log] {out}")
    print("[audit] waiting for motion (Ctrl-C to stop)...")

    cols = ["t", "x", "y", "z", "rx", "ry", "rz", "cart_mms",
            "speed_pct", "override", "opmode", "motion", "inpos", "derived_mms"]
    f = open(out, "w", newline="")
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()

    period = 1.0 / max(1.0, args.hz)
    prev = first
    moving = False
    move: dict = {}
    moves: list[dict] = []

    def finish_move():
        nonlocal move
        if move.get("n", 0) >= 2:
            dur = move["t1"] - move["t0"]
            print(f"[move {len(moves)+1}] cmd={move['cmd']}% override={move['ovr']} "
                  f"mode={OPMODE.get(move['mode'], move['mode'])} | "
                  f"path={move['path']:.0f}mm dur={dur:.2f}s "
                  f"avg={move['path']/dur if dur > 0 else 0:.0f}mm/s "
                  f"peak_derived={move['peak_derived']:.0f}mm/s "
                  f"peak_reg={move['peak_reg']:.0f}mm/s")
            moves.append(dict(move, dur=dur))
        move = {}

    try:
        while True:
            time.sleep(period)
            cur = s.read()
            if cur is None:
                continue
            dt = cur["t"] - prev["t"]
            step = math.dist((cur["x"], cur["y"], cur["z"]), (prev["x"], prev["y"], prev["z"]))
            derived = step / dt if dt > 0 else 0.0
            cur["derived_mms"] = round(derived, 1)
            w.writerow({k: cur.get(k) for k in cols})
            if step > args.move_thresh:
                if not moving:
                    moving = True
                    move = {"t0": prev["t"], "t1": cur["t"], "n": 0, "path": 0.0,
                            "cmd": cur["speed_pct"], "ovr": cur["override"], "mode": cur["opmode"],
                            "peak_derived": 0.0, "peak_reg": 0.0}
                    print(f"[motion start] cmd={move['cmd']}% override={move['ovr']} "
                          f"mode={OPMODE.get(move['mode'], move['mode'])}")
                move["t1"] = cur["t"]
                move["n"] += 1
                move["path"] += step
                move["peak_derived"] = max(move["peak_derived"], derived)
                if cur["cart_mms"]:
                    move["peak_reg"] = max(move["peak_reg"], cur["cart_mms"])
            elif moving and cur["t"] - move["t1"] > 0.5:
                moving = False
                finish_move()
            prev = cur
    except KeyboardInterrupt:
        if moving:
            finish_move()
        f.close()
        print(f"\n[done] {len(moves)} move(s) captured -> {out}")
        if moves:
            print(f"{'cmd%':>5} {'mode':>5} {'path_mm':>8} {'dur_s':>6} {'avg_mm/s':>9} {'peak_mm/s':>9}")
            for m in moves:
                print(f"{str(m['cmd']):>5} {OPMODE.get(m['mode'], m['mode']):>5} "
                      f"{m['path']:>8.0f} {m['dur']:>6.2f} "
                      f"{m['path']/m['dur'] if m['dur'] > 0 else 0:>9.0f} {m['peak_derived']:>9.0f}")
            print("\nSame cmd% with very different avg mm/s across moves = MovP joint-space "
                  "geometry (or accel-limited short strides). Consistent ratio = unit scaling.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
