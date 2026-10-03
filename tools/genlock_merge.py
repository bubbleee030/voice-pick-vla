"""
Merge timestamp-based streams for either:

1. one recorded episode directory (`trajectory.csv` as base timeline), or
2. one teach recording JSON (`waypoints[*].t_ms` + `start_unix` as base timeline).

Supports:
- local episode trajectory.csv (base timeline)
- local gripper_stream.csv (optional)
- external CSV from another computer (manual/relative time supported)
- optional augmented teach JSON output when base is a teach recording

Examples:
    python tools/genlock_merge.py \
      --episode-dir data/recordings/apple/episode_001 \
      --external-csv data/claw/tactile_data_123.csv

    python tools/genlock_merge.py \
      --teach-json data/teach_recordings/chopsticks_pick_v1.json \
      --external-csv data/claw/tactile_data_123.csv \
      --teach-output-json data/teach_recordings/chopsticks_pick_v1.merged.json
"""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genlock merge for episode streams or teach recordings")
    parser.add_argument("--episode-dir", default="", help="Episode directory path")
    parser.add_argument("--teach-json", default="", help="Teach recording JSON path")
    parser.add_argument("--teach-start-unix", type=float, default=None,
                        help="Override teach recording start_unix if JSON does not include it")

    parser.add_argument("--base-csv", default="trajectory.csv", help="Base timeline CSV in episode-dir")
    parser.add_argument("--base-time-col", default="timestamp_unix", help="Timestamp column in base CSV")

    parser.add_argument("--gripper-csv", default="gripper_stream.csv",
                        help="Gripper CSV filename in episode-dir, or an explicit path")
    parser.add_argument("--gripper-time-col", default="timestamp_unix", help="Timestamp column in gripper CSV")

    parser.add_argument("--external-csv", default="", help="Optional external CSV path")
    parser.add_argument("--external-time-col", default="timestamp_unix", help="Timestamp column in external CSV")
    parser.add_argument("--external-relative", action="store_true",
                        help="Interpret external time column as elapsed seconds")
    parser.add_argument("--external-start-unix", type=float, default=None,
                        help="Required when --external-relative is set")
    parser.add_argument("--external-offset-sec", type=float, default=0.0,
                        help="Additional offset applied after parsing external timestamps")

    parser.add_argument("--max-delta", type=float, default=0.10,
                        help="Max allowed nearest-neighbor delta in seconds")
    parser.add_argument("--output", default="genlock_merged.csv", help="Output CSV filename or path")
    parser.add_argument("--report", default="genlock_report.json", help="Output report filename or path")
    parser.add_argument("--teach-output-json", default="",
                        help="When using --teach-json, write an augmented merged teach JSON here")

    args = parser.parse_args()

    if bool(args.episode_dir) == bool(args.teach_json):
        parser.error("Provide exactly one of --episode-dir or --teach-json")

    if args.external_relative and args.external_start_unix is None:
        parser.error("--external-start-unix is required when --external-relative is set")

    return args


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_unix_ts(
    row: dict[str, Any],
    time_col: str,
    relative: bool = False,
    start_unix: float | None = None,
    offset_sec: float = 0.0,
) -> float | None:
    raw = row.get(time_col)
    if raw is None or raw == "":
        return None
    try:
        t = float(raw)
    except ValueError:
        return None

    if relative:
        if start_unix is None:
            return None
        t = start_unix + t

    return t + offset_sec


def prepare_timed_rows(
    rows: list[dict[str, str]],
    time_col: str,
    relative: bool = False,
    start_unix: float | None = None,
    offset_sec: float = 0.0,
) -> tuple[list[tuple[float, dict[str, str]]], list[str]]:
    timed: list[tuple[float, dict[str, str]]] = []
    keys: set[str] = set()
    for row in rows:
        ts = parse_unix_ts(
            row,
            time_col,
            relative=relative,
            start_unix=start_unix,
            offset_sec=offset_sec,
        )
        if ts is None:
            continue
        timed.append((ts, row))
        keys.update(row.keys())
    timed.sort(key=lambda item: item[0])
    return timed, sorted(keys)


def nearest_row(
    timed_rows: list[tuple[float, dict[str, str]]],
    target_ts: float,
    cursor: int,
    max_delta: float,
) -> tuple[tuple[float, dict[str, str]] | None, int, float | None]:
    if not timed_rows:
        return None, cursor, None

    n = len(timed_rows)
    i = min(max(0, cursor), n - 1)

    while i + 1 < n and timed_rows[i + 1][0] <= target_ts:
        i += 1
    while i > 0 and timed_rows[i][0] > target_ts:
        i -= 1

    best_i = i
    best_dt = abs(timed_rows[i][0] - target_ts)
    if i + 1 < n:
        dt2 = abs(timed_rows[i + 1][0] - target_ts)
        if dt2 < best_dt:
            best_i = i + 1
            best_dt = dt2

    if best_dt > max_delta:
        return None, best_i, None
    return timed_rows[best_i], best_i, best_dt


def load_teach_rows(path: Path, start_unix_override: float | None) -> tuple[list[dict[str, str]], dict[str, Any], float]:
    with open(path, "r", encoding="utf-8") as f:
        teach_data = json.load(f)

    start_unix = teach_data.get("start_unix")
    if start_unix is None:
        start_unix = start_unix_override
    if start_unix is None:
        raise SystemExit(
            f"Teach recording missing start_unix: {path}. "
            "Pass --teach-start-unix or re-record with the updated recorder."
        )

    rows: list[dict[str, str]] = []
    for idx, wp in enumerate(teach_data.get("waypoints", [])):
        t_ms = wp.get("t_ms")
        if t_ms is None:
            continue
        pose = wp.get("pose", [])
        row = {
            "waypoint_index": str(idx),
            "t_ms": str(t_ms),
            "timestamp_unix": f"{float(start_unix) + (float(t_ms) / 1000.0):.6f}",
            "gripper": str(wp.get("gripper", "none")),
            "speed": str(wp.get("speed", "")),
        }
        pose_labels = ["x_um", "y_um", "z_um", "rx_mdeg", "ry_mdeg", "rz_mdeg"]
        for pose_idx, label in enumerate(pose_labels):
            row[label] = str(pose[pose_idx]) if pose_idx < len(pose) else ""
        rows.append(row)

    return rows, teach_data, float(start_unix)


def resolve_side_csv(base_dir: Path, value: str) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / value).resolve()


def resolve_output_path(default_dir: Path, default_name: str, user_value: str) -> Path:
    user_path = Path(user_value)
    if user_value == default_name:
        return (default_dir / default_name).resolve()
    if user_path.is_absolute():
        return user_path.resolve()
    return (Path.cwd() / user_path).resolve()


def inject_matches_into_teach(
    teach_data: dict[str, Any],
    base_timed: list[tuple[float, dict[str, str]]],
    external_matches: dict[int, tuple[float, dict[str, str], float]],
    gripper_matches: dict[int, tuple[float, dict[str, str], float]],
    report: dict[str, Any],
    external_timeline: list[dict[str, Any]] | None = None,
    merged_name: str | None = None,
) -> dict[str, Any]:
    merged = deepcopy(teach_data)
    merged["genlock_merge"] = report
    if merged_name:
        merged["name"] = merged_name
    if external_timeline:
        merged["external_timeline"] = external_timeline

    waypoints = merged.get("waypoints", [])
    for idx, wp in enumerate(waypoints):
        if idx >= len(base_timed):
            continue
        base_ts, _base_row = base_timed[idx]
        wp["timestamp_unix"] = round(base_ts, 6)

        gripper_match = gripper_matches.get(idx)
        if gripper_match is not None:
            g_ts, g_row, g_dt = gripper_match
            wp["matched_gripper"] = {
                "timestamp_unix": round(g_ts, 6),
                "dt_sec": round(g_dt, 6),
                "row": g_row,
            }

        external_match = external_matches.get(idx)
        if external_match is not None:
            e_ts, e_row, e_dt = external_match
            wp["matched_external"] = {
                "timestamp_unix": round(e_ts, 6),
                "dt_sec": round(e_dt, 6),
                "row": e_row,
            }

            # Optional convenience: if the external row already carries a symbolic action,
            # copy it onto the waypoint so replay can consume it directly.
            action = (
                e_row.get("gripper")
                or e_row.get("action")
                or e_row.get("command")
            )
            if action in {"open", "close"} and wp.get("gripper", "none") in {"", "none"}:
                wp["gripper"] = action

    return merged


def build_external_timeline(
    external_timed: list[tuple[float, dict[str, str]]],
    teach_start_unix: float,
    teach_end_unix: float,
) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    if not external_timed:
        return timeline

    for external_ts, external_row in external_timed:
        if external_ts < teach_start_unix or external_ts > teach_end_unix:
            continue
        values = [
            external_row.get("pos1", external_row.get("remote_pos1")),
            external_row.get("pos2", external_row.get("remote_pos2")),
            external_row.get("pos3", external_row.get("remote_pos3")),
        ]
        if any(v in {None, ""} for v in values):
            continue
        try:
            positions = [int(float(v)) for v in values]
        except (TypeError, ValueError):
            continue

        sample = {
            "t_ms": int(round((external_ts - teach_start_unix) * 1000.0)),
            "timestamp_unix": round(external_ts, 6),
            "positions": positions,
        }
        tactile_data = external_row.get("tactile_data")
        if tactile_data not in {None, ""}:
            sample["tactile_data"] = tactile_data
        elif all(external_row.get(key) not in {None, ""} for key in ("val1", "val2", "val3")):
            sample["tactile_data"] = [
                int(float(external_row["val1"])),
                int(float(external_row["val2"])),
                int(float(external_row["val3"])),
            ]
        if external_row.get("sensor_connected") not in {None, ""}:
            sample["sensor_connected"] = str(external_row.get("sensor_connected")).strip() not in {"0", "false", "False"}
        if external_row.get("sample_valid") not in {None, ""}:
            sample["sample_valid"] = str(external_row.get("sample_valid")).strip() not in {"0", "false", "False"}
        timeline.append(sample)
    return timeline


def main() -> None:
    args = parse_args()

    teach_data: dict[str, Any] | None = None
    teach_path: Path | None = None
    mode = "episode"

    if args.teach_json:
        mode = "teach"
        teach_path = Path(args.teach_json).resolve()
        base_dir = teach_path.parent
        base_rows, teach_data, teach_start_unix = load_teach_rows(teach_path, args.teach_start_unix)
        base_timed, base_keys = prepare_timed_rows(base_rows, "timestamp_unix")
        if not base_timed:
            raise SystemExit(f"No valid waypoint timestamps found in teach recording: {teach_path}")
        output_path = (
            resolve_output_path(teach_path.parent, teach_path.stem + ".genlock_merged.csv", args.output)
            if args.output != "genlock_merged.csv"
            else (teach_path.parent / f"{teach_path.stem}.genlock_merged.csv").resolve()
        )
        report_path = (
            resolve_output_path(teach_path.parent, teach_path.stem + ".genlock_report.json", args.report)
            if args.report != "genlock_report.json"
            else (teach_path.parent / f"{teach_path.stem}.genlock_report.json").resolve()
        )
    else:
        episode_dir = Path(args.episode_dir).resolve()
        base_dir = episode_dir
        base_path = episode_dir / args.base_csv
        output_path = resolve_output_path(episode_dir, "genlock_merged.csv", args.output)
        report_path = resolve_output_path(episode_dir, "genlock_report.json", args.report)

        base_rows = read_csv_rows(base_path)
        if not base_rows:
            raise SystemExit(f"Base CSV not found or empty: {base_path}")

        base_timed, base_keys = prepare_timed_rows(base_rows, args.base_time_col)
        if not base_timed:
            raise SystemExit(
                f"No valid timestamps in base CSV using column '{args.base_time_col}': {base_path}"
            )

    gripper_path = resolve_side_csv(base_dir, args.gripper_csv)
    gripper_rows = read_csv_rows(gripper_path) if gripper_path is not None else []
    gripper_timed, gripper_keys = prepare_timed_rows(gripper_rows, args.gripper_time_col)

    external_timed: list[tuple[float, dict[str, str]]] = []
    external_keys: list[str] = []
    external_path = Path(args.external_csv).resolve() if args.external_csv else None
    if external_path is not None and external_path.exists():
        ext_rows = read_csv_rows(external_path)
        external_timed, external_keys = prepare_timed_rows(
            ext_rows,
            args.external_time_col,
            relative=args.external_relative,
            start_unix=args.external_start_unix,
            offset_sec=args.external_offset_sec,
        )

    out_header = [
        "base_ts_unix",
        "base_elapsed_s",
        "gripper_dt_sec",
        "external_dt_sec",
    ]
    out_header.extend([f"base_{key}" for key in base_keys])
    out_header.extend([f"gripper_{key}" for key in gripper_keys])
    out_header.extend([f"external_{key}" for key in external_keys])

    gripper_cursor = 0
    external_cursor = 0
    matched_gripper = 0
    matched_external = 0
    gripper_match_map: dict[int, tuple[float, dict[str, str], float]] = {}
    external_match_map: dict[int, tuple[float, dict[str, str], float]] = {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_header)
        writer.writeheader()

        base_start = base_timed[0][0]
        for idx, (base_ts, base_row) in enumerate(base_timed):
            row = {
                "base_ts_unix": f"{base_ts:.6f}",
                "base_elapsed_s": f"{base_ts - base_start:.6f}",
                "gripper_dt_sec": "",
                "external_dt_sec": "",
            }

            for key in base_keys:
                row[f"base_{key}"] = base_row.get(key, "")

            g_match, gripper_cursor, g_dt = nearest_row(
                gripper_timed, base_ts, gripper_cursor, args.max_delta
            )
            if g_match is not None and g_dt is not None:
                matched_gripper += 1
                g_ts, g_row = g_match
                row["gripper_dt_sec"] = f"{(g_ts - base_ts):+.6f}"
                for key in gripper_keys:
                    row[f"gripper_{key}"] = g_row.get(key, "")
                gripper_match_map[idx] = (g_ts, g_row, g_ts - base_ts)
            else:
                for key in gripper_keys:
                    row[f"gripper_{key}"] = ""

            e_match, external_cursor, e_dt = nearest_row(
                external_timed, base_ts, external_cursor, args.max_delta
            )
            if e_match is not None and e_dt is not None:
                matched_external += 1
                e_ts, e_row = e_match
                row["external_dt_sec"] = f"{(e_ts - base_ts):+.6f}"
                for key in external_keys:
                    row[f"external_{key}"] = e_row.get(key, "")
                external_match_map[idx] = (e_ts, e_row, e_ts - base_ts)
            else:
                for key in external_keys:
                    row[f"external_{key}"] = ""

            writer.writerow(row)

    report = {
        "mode": mode,
        "episode_dir": args.episode_dir or None,
        "teach_json": str(teach_path) if teach_path is not None else None,
        "teach_start_unix": teach_start_unix if mode == "teach" else None,
        "base_csv": args.base_csv if mode == "episode" else None,
        "gripper_csv": str(gripper_path) if gripper_path is not None else None,
        "external_csv": str(external_path) if external_path else None,
        "base_rows": len(base_timed),
        "gripper_rows": len(gripper_timed),
        "external_rows": len(external_timed),
        "matched_gripper_rows": matched_gripper,
        "matched_external_rows": matched_external,
        "max_delta_sec": args.max_delta,
        "external_relative": args.external_relative,
        "external_start_unix": args.external_start_unix,
        "external_offset_sec": args.external_offset_sec,
        "output_csv": str(output_path),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if mode == "teach" and teach_data is not None:
        teach_output_path: Path | None = None
        if args.teach_output_json:
            teach_output_path = Path(args.teach_output_json).resolve()
        elif external_path is not None:
            teach_output_path = (teach_path.parent / f"{teach_path.stem}.merged.json").resolve()  # type: ignore[union-attr]

        if teach_output_path is not None:
            teach_end_unix = base_timed[-1][0]
            external_timeline = build_external_timeline(
                external_timed,
                teach_start_unix,
                teach_end_unix,
            )
            merged_teach = inject_matches_into_teach(
                teach_data,
                base_timed,
                external_match_map,
                gripper_match_map,
                report,
                external_timeline=external_timeline,
                merged_name=teach_output_path.stem,
            )
            teach_output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(teach_output_path, "w", encoding="utf-8") as f:
                json.dump(merged_teach, f, indent=2, ensure_ascii=False)
            report["teach_output_json"] = str(teach_output_path)
            report["external_timeline_samples"] = len(external_timeline)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

    print("Genlock merge complete")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
