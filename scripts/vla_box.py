"""Pure safety-box geometry for the step-through tool — no hardware/torch imports so it
is unit-testable. `in_box` always enforces x,y; z is enforced only when ignore_z is False
(freeze-z / XY-only mode pins z itself, so physical z-wobble must not trip the box)."""
from __future__ import annotations

import argparse


def parse_box(s: str) -> tuple[float, ...]:
    parts = [float(v) for v in s.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("box must be x_min,x_max,y_min,y_max,z_min,z_max")
    return tuple(parts)


def in_box(xyz, box, ignore_z: bool = False) -> bool:
    x, y, z = xyz
    if not ((box[0] <= x <= box[1]) and (box[2] <= y <= box[3])):
        return False
    if ignore_z:
        return True
    return box[4] <= z <= box[5]
