"""
Shared utilities for the voice-controlled pick system.
"""

import struct
import time
import yaml
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def load_config(name: str) -> dict:
    """Load a YAML config file from config/ directory."""
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(name: str, data: dict):
    """Save a dict to a YAML config file."""
    path = CONFIG_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ---- Modbus integer encoding (matches colleague's register format) ----

def int_to_register(value: int) -> list:
    """Convert a signed integer to a single Modbus register value."""
    if value < 0:
        return [value + 65536]
    if value < 32767:
        return [value]
    return [0]


def int32_to_registers(value: int) -> list:
    """
    Convert a signed 32-bit integer to two Modbus registers (low, high).
    Matches the legacy vendor integer scaling convention.
    """
    if value < 0:
        f = value + 4294967296
        b_str = bin(f)
        b = int(b_str[:18], 2)
        a = int(b_str[18:], 2)
        return [a, b]
    if value < 65536:
        return [value, 0]
    f = bin(value)
    a = int(f[-16:], 2)
    b = int(f[:-16], 2)
    return [a, b]


def registers_to_int32(regs: list) -> int:
    """
    Convert two Modbus registers (low, high) back to a signed 32-bit integer.
    """
    if len(regs) < 2:
        return 0
    low, high = regs[0], regs[1]
    value = (high << 16) | low
    if value >= 2147483648:
        value -= 4294967296
    return value


def read_current_pose_from_registers(client, base_reg: int = 0x00F0) -> list:
    """
    Read current Cartesian pose (X, Y, Z, RX, RY, RZ) from Modbus registers.
    Returns list of 6 integers in controller units (mm * 1000 for XYZ, deg * 1000 for RPY).
    """
    regs = client.read_holding_registers(base_reg, 12)
    if regs is None or len(regs) < 12:
        return [0, 0, 0, 0, 0, 0]
    pose = []
    for i in range(6):
        pose.append(registers_to_int32(regs[i * 2: i * 2 + 2]))
    return pose


def pose_to_mm_deg(pose: list) -> list:
    """Convert controller pose (units * 1000) to mm and degrees."""
    return [v / 1000.0 for v in pose]


def timestamp_str() -> str:
    """Return a timestamp string for filenames."""
    return time.strftime("%Y%m%d_%H%M%S")
