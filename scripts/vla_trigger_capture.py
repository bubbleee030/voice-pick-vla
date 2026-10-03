"""
Trigger a VLA run over the live socketio server (the same 'vla_start' event the
UI button emits) and capture vla_log / vla_status into a timestamped logfile.

Usage:
    conda run -n voice_pick python scripts/vla_trigger_capture.py \
        --instruction 拿起梯形 --embed data/vla_embed_trapezoid_en.pt \
        --max-steps 16 --dry-run 1
"""
from __future__ import annotations
import argparse, json, threading, time, urllib.request, datetime, os, sys

import socketio

URL = "http://127.0.0.1:8090"
CKPT = "data/datasets/checkpoints/combined_nchc_20260529_120309"


def health():
    try:
        with urllib.request.urlopen(URL + "/api/health", timeout=8) as r:
            return json.load(r)
    except Exception as e:
        return {"_error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", default="拿起梯形")
    ap.add_argument("--embed", default="data/vla_embed_trapezoid_en.pt")
    ap.add_argument("--max-steps", type=int, default=16)
    ap.add_argument("--exec-steps", type=int, default=8)
    ap.add_argument("--dry-run", type=int, default=1)
    ap.add_argument("--connect-arm", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=240)
    args = ap.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join("data/runtime/logs", f"vla_dryrun_{ts}.log")
    out = open(outpath, "w", buffering=1)

    def w(line):
        print(line, flush=True)
        out.write(line + "\n")

    sio = socketio.Client(reconnection=False)
    done = threading.Event()
    saw_active = {"v": False}   # ignore the stale 'stopped' the server emits on connect

    @sio.on("vla_log")
    def on_log(d):
        w(f"{d.get('timestamp','')} [{d.get('level','')}] {d.get('message','')}")

    @sio.on("vla_status")
    def on_status(d):
        st = d.get("status")
        w(f"  <status> {st} step={d.get('step')}")
        if st in ("loading", "running"):
            saw_active["v"] = True
        if st in ("stopped", "error") and saw_active["v"]:
            done.set()

    w(f"# capture -> {outpath}")
    w(f"# args: {vars(args)}")
    sio.connect(URL, wait_timeout=10)
    w("# socketio connected")

    if args.connect_arm:
        w("# emitting arm_connect (motionless) ...")
        sio.emit("arm_connect")
        time.sleep(5)
    h = health()
    a = h.get("arm", {})
    w(f"# arm.connected={a.get('connected')} pose={a.get('current_pose_mm_deg')}")
    c = h.get("cameras", {})
    w(f"# cam1.running={c.get('cam1',{}).get('running')} cam2.running={c.get('cam2',{}).get('running')} "
      f"claw.running={h.get('claw',{}).get('running')} claw.src={h.get('claw',{}).get('source')}")

    w(f"# emitting vla_start dry_run={bool(args.dry_run)} instruction={args.instruction}")
    sio.emit("vla_start", {
        "checkpoint": CKPT,
        "embed_path": args.embed,
        "instruction": args.instruction,
        "max_steps": args.max_steps,
        "exec_steps": args.exec_steps,
        "dry_run": bool(args.dry_run),
    })

    done.wait(timeout=args.timeout)
    if not done.is_set():
        w("# TIMEOUT waiting for stopped/error — emitting vla_stop")
        sio.emit("vla_stop")
        time.sleep(3)
    sio.disconnect()
    out.close()
    print(f"\n# saved: {outpath}")


if __name__ == "__main__":
    main()
