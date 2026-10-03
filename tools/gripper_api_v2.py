from __future__ import annotations

import argparse
import os
import time

import keyboard
import requests


VALID_KEYS = ["s", "e", "v", "c", "o", "1", "2", "3", "4", "5", "6", "7"]


def resolve_base_url(host: str, ports: list[int]) -> str:
    for port in ports:
        base_url = f"http://{host}:{port}"
        for probe in ("/health", "/state"):
            try:
                resp = requests.get(f"{base_url}{probe}", timeout=1.0)
                if resp.status_code == 200:
                    return base_url
            except requests.exceptions.RequestException:
                continue
    return f"http://{host}:{ports[0]}"


def send_command(command_url: str, key: str):
    try:
        response = requests.post(command_url, json={"action": key}, timeout=1)
        if response.status_code != 200:
            print(f"request failed: HTTP {response.status_code}")
            return
        data = response.json()
        if key == "s":
            print(f"recording started: {data.get('recording_session_id') or data.get('file')}")
        elif key == "e":
            print(f"recording stopped: {data.get('recording_session_id') or data.get('status')}")
        else:
            print(f"sent {key} -> current_pos={data.get('current_pos')}")
    except requests.exceptions.RequestException as exc:
        print(f"network error: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keyboard gripper control client")
    parser.add_argument("--host", default=os.getenv("AGX_IP", "192.168.1.100"))
    parser.add_argument(
        "--ports",
        default=os.getenv("AGX_PORTS", "5003,5002"),
        help="Comma-separated candidate ports. Default prefers v2 on 5003, then legacy on 5002.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ports = [int(p.strip()) for p in str(args.ports).split(",") if p.strip()]
    base_url = resolve_base_url(args.host, ports)
    command_url = f"{base_url}/command"

    print(f"connecting to gripper controller at {base_url}")
    print("controls:")
    print("  [s] start recording | [e] stop recording")
    print("  [v/c] open/close all | [o] home")
    print("  [1-6] per-finger motion | [7] special action")
    print("  [x] exit")

    try:
        while True:
            for key in VALID_KEYS:
                if keyboard.is_pressed(key):
                    send_command(command_url, key)
                    time.sleep(0.15 if key in ["s", "e"] else 0.05)

            if keyboard.is_pressed("x"):
                print("exit requested")
                break

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nforced exit")


if __name__ == "__main__":
    main()
