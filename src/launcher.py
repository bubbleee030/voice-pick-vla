from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from src.runtime_config import (
    DEFAULT_PROFILE,
    LAUNCHER_STATE_PATH,
    LOG_DIR,
    PROJECT_ROOT,
    RUNTIME_DIR,
    load_runtime_bundle,
    read_launcher_state,
    write_launcher_state,
)


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8090


def _backend_command(
    profile: str,
    env_config: str | None,
    host: str,
    port: int,
    ui_mode: str | None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "voice_pick_demo.py"),
        "--host",
        host,
        "--port",
        str(port),
        "--profile",
        profile,
    ]
    if env_config:
        cmd.extend(["--env-config", env_config])
    if ui_mode:
        cmd.extend(["--ui-mode", ui_mode])
    return cmd


def _backend_env() -> dict[str, str]:
    env = dict(os.environ)
    current = env.get("PYTHONPATH", "")
    project = str(PROJECT_ROOT)
    env["PYTHONPATH"] = f"{project}{os.pathsep}{current}" if current else project
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _current_running_state() -> dict:
    state = read_launcher_state()
    pid = int(state.get("pid", 0) or 0)
    state["running"] = bool(pid and _is_pid_alive(pid))
    return state


def cmd_start(args: argparse.Namespace) -> int:
    bundle = load_runtime_bundle(profile=args.profile, env_config=args.env_config)
    current = _current_running_state()
    if current.get("running") and not args.force:
        print(f"voice_pick already running with pid={current.get('pid')}")
        return 1

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"voice_pick_{bundle.profile_name}.log"
    effective_ui_mode = args.ui_mode or bundle.demo_config.get("ui", {}).get("mode", "modern")
    cmd = _backend_command(bundle.profile_name, bundle.env_overlay_path, args.host, args.port, effective_ui_mode)
    env = _backend_env()

    if args.attach:
        write_launcher_state(
            {
                "pid": os.getpid(),
                "profile": bundle.profile_name,
                "host": args.host,
                "port": args.port,
                "env_overlay_path": bundle.env_overlay_path,
                "ui_mode": effective_ui_mode,
                "log_path": None,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "command": cmd,
            }
        )
        os.execvpe(cmd[0], cmd, env)

    with open(log_path, "a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

    write_launcher_state(
        {
            "pid": proc.pid,
            "profile": bundle.profile_name,
            "host": args.host,
            "port": args.port,
            "env_overlay_path": bundle.env_overlay_path,
            "ui_mode": effective_ui_mode,
            "log_path": str(log_path),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "command": cmd,
        }
    )
    print(f"Started voice_pick pid={proc.pid} profile={bundle.profile_name} ui_mode={effective_ui_mode}")
    print(f"Log: {log_path}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    state = _current_running_state()
    if not state:
        print("voice_pick not launched via launcher")
        return 0
    for key in ("running", "pid", "profile", "host", "port", "ui_mode", "env_overlay_path", "log_path", "started_at"):
        print(f"{key}: {state.get(key)}")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    state = _current_running_state()
    pid = int(state.get("pid", 0) or 0)
    if not pid or not state.get("running"):
        print("voice_pick is not running")
        return 0
    sig = signal.SIGKILL if args.force else signal.SIGTERM
    os.kill(pid, sig)
    print(f"Sent {sig.name} to pid={pid}")
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    state = _current_running_state()
    log_path = Path(state.get("log_path") or "")
    if not log_path.exists():
        print("No launcher log file found")
        return 1
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    tail = lines[-args.lines :]
    print("\n".join(tail))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Voice Pick launcher")
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="Start a launch profile")
    start.add_argument("profile", nargs="?", default=DEFAULT_PROFILE)
    start.add_argument("--env-config", default=None, help="Optional config overlay path")
    start.add_argument("--host", default=DEFAULT_HOST)
    start.add_argument("--port", type=int, default=DEFAULT_PORT)
    start.add_argument("--ui-mode", choices=("modern", "classic"), default=None)
    start.add_argument("--attach", action="store_true", help="Replace the current process with the backend")
    start.add_argument("--force", action="store_true", help="Ignore existing running state")
    start.set_defaults(func=cmd_start)

    status = sub.add_parser("status", help="Show launcher status")
    status.set_defaults(func=cmd_status)

    stop = sub.add_parser("stop", help="Stop a running backend")
    stop.add_argument("--force", action="store_true", help="Use SIGKILL")
    stop.set_defaults(func=cmd_stop)

    logs = sub.add_parser("logs", help="Print launcher log tail")
    logs.add_argument("--lines", type=int, default=80)
    logs.set_defaults(func=cmd_logs)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
