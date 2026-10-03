"""
Prepare datasets from one recording source for two downstream uses:
1) VLA fine-tuning format
2) YOLO training/annotation format

This lets one recording session feed both pipelines.

Examples:
    python tools/prepare_dataset.py --mode both
    python tools/prepare_dataset.py --mode yolo --auto-label --image-stride 8
    python tools/prepare_dataset.py --mode vla --copy
"""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config

GRIPPER_MATCH_MAX_DELTA_S = 0.10


@dataclass
class Episode:
    object_name: str
    episode_name: str
    path: Path
    cam_frames: dict
    timeline_rows: list
    pose_rows: list
    poses: list
    gripper_rows: list
    metadata: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare VLA/YOLO datasets from recordings")
    parser.add_argument("--recordings-root", default="data/recordings", help="Source recordings root")
    parser.add_argument("--mode", choices=["both", "vla", "yolo"], default="both",
                        help="Which dataset(s) to generate")
    parser.add_argument("--vla-out", default="data/datasets/own3_processed", help="VLA output root")
    parser.add_argument("--yolo-out", default="data/datasets/yolo_voice_pick", help="YOLO output root")
    parser.add_argument("--cameras", default="cam1,cam2",
                        help="Comma separated camera folders to use: cam1,cam2")
    parser.add_argument("--image-stride", type=int, default=6,
                        help="Use one of every N frames for YOLO export")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Validation split ratio for YOLO")
    parser.add_argument("--limit-episodes", type=int, default=0,
                        help="Only process first N discovered episodes (0 = all)")
    parser.add_argument("--min-frames", type=int, default=30,
                        help="Skip episodes shorter than this many frames after trimming")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of hard-linking when possible")
    parser.add_argument("--trim-motion", dest="trim_motion", action="store_true", default=True,
                        help="Trim to movement window based on arm trajectory (default: enabled)")
    parser.add_argument("--no-trim-motion", dest="trim_motion", action="store_false",
                        help="Disable movement-based trimming and keep full episode")
    parser.add_argument("--motion-threshold-mm", type=float, default=3.0,
                        help="Movement threshold in mm for detecting motion start/end")
    parser.add_argument("--motion-pre-buffer-s", type=float, default=3.0,
                        help="Seconds to keep before detected motion start")
    parser.add_argument("--motion-post-buffer-s", type=float, default=3.0,
                        help="Seconds to keep after detected motion end")
    parser.add_argument("--gripper-min", default="0,0,0",
                        help="Gripper position minima as a,b,c for VLA normalization")
    parser.add_argument("--gripper-max", default="4095,4095,4095",
                        help="Gripper position maxima as a,b,c for VLA normalization")
    parser.add_argument("--auto-label", action="store_true",
                        help="Use pretrained YOLO to generate weak labels automatically")
    parser.add_argument("--yolo-model", default="yolov8s.pt", help="YOLO model for --auto-label")
    parser.add_argument("--min-conf", type=float, default=0.3,
                        help="Min confidence for auto labels")
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path, force_copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if force_copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def stable_split_key(text: str) -> float:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / float(0xFFFFFFFF)


def display_path(path: Path, root: Path = PROJECT_ROOT) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _to_float(value, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_triplet_arg(value: str, flag_name: str) -> list[float]:
    parts = [p.strip() for p in str(value).split(",")]
    if len(parts) != 3:
        raise SystemExit(f"{flag_name} must contain exactly 3 comma-separated values")
    out = []
    for part in parts:
        try:
            out.append(float(part))
        except ValueError as exc:
            raise SystemExit(f"{flag_name} contains a non-numeric value: {part}") from exc
    return out


def parse_timeline_rows(csv_path: Path) -> list:
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, r in enumerate(reader):
            elapsed = _to_float(r.get("elapsed_s"), default=float(idx))
            ts_unix = _to_float(r.get("timestamp_unix"))
            rows.append({
                "timestamp_unix": None if math.isnan(ts_unix) else float(ts_unix),
                "elapsed_s": float(idx) if math.isnan(elapsed) else float(elapsed),
                "valid": int(_to_float(r.get("valid"), default=0.0)) if r.get("valid") not in (None, "") else 0,
            })
    return rows


def parse_pose_rows(csv_path: Path) -> list:
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            x = _to_float(r.get("x_mm"))
            y = _to_float(r.get("y_mm"))
            z = _to_float(r.get("z_mm"))
            rx = _to_float(r.get("rx_deg"))
            ry = _to_float(r.get("ry_deg"))
            rz = _to_float(r.get("rz_deg"))
            elapsed = _to_float(r.get("elapsed_s"), default=float(len(rows)))
            ts_unix = _to_float(r.get("timestamp_unix"))

            if np.isnan([x, y, z, rx, ry, rz]).any():
                continue

            if math.isnan(elapsed):
                elapsed = float(len(rows))

            rows.append({
                "timestamp_unix": None if math.isnan(ts_unix) else float(ts_unix),
                "elapsed_s": float(elapsed),
                "pose": [float(x), float(y), float(z), float(rx), float(ry), float(rz)],
            })
    return rows


def parse_gripper_rows(csv_path: Path) -> list:
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            status_value = str(r.get("status", "")).strip().lower()
            if "poll_ok" in r:
                poll_ok = int(_to_float(r.get("poll_ok"), default=1.0))
            elif status_value:
                poll_ok = 1 if status_value == "ok" else 0
            else:
                poll_ok = 1
            pos = [
                _to_float(r.get("pos1", r.get("remote_pos1"))),
                _to_float(r.get("pos2", r.get("remote_pos2"))),
                _to_float(r.get("pos3", r.get("remote_pos3"))),
            ]
            if poll_ok <= 0 or np.isnan(pos).any():
                continue

            elapsed = _to_float(r.get("elapsed_s"))
            ts_unix = _to_float(r.get("timestamp_unix"))
            if math.isnan(elapsed) and math.isnan(ts_unix):
                continue

            server_time = _to_float(r.get("server_time_unix", r.get("remote_server_time_unix")))
            tactile = [
                _to_float(r.get("val1")),
                _to_float(r.get("val2")),
                _to_float(r.get("val3")),
            ]
            sensor_connected = _to_float(r.get("sensor_connected"), default=1.0)
            sample_valid = _to_float(r.get("sample_valid"), default=1.0)
            tactile_ts = _to_float(r.get("tactile_timestamp_unix"))
            rows.append({
                "timestamp_unix": None if math.isnan(ts_unix) else float(ts_unix),
                "elapsed_s": None if math.isnan(elapsed) else float(elapsed),
                "server_time_unix": None if math.isnan(server_time) else float(server_time),
                "positions": [float(v) for v in pos],
                "tactile": None if np.isnan(tactile).any() else [float(v) for v in tactile],
                "sensor_connected": bool(sensor_connected > 0.5),
                "sample_valid": bool(sample_valid > 0.5),
                "tactile_timestamp_unix": None if math.isnan(tactile_ts) else float(tactile_ts),
            })
    return rows


def estimate_pose_hz(pose_rows: list, metadata: dict) -> float:
    pose_hz = _to_float(metadata.get("pose_hz"), default=float("nan")) if metadata else float("nan")
    if not math.isnan(pose_hz) and pose_hz > 0:
        return float(pose_hz)

    if len(pose_rows) >= 2:
        elapsed = np.array([float(r["elapsed_s"]) for r in pose_rows], dtype=np.float64)
        diffs = np.diff(elapsed)
        diffs = diffs[diffs > 1e-6]
        if diffs.size > 0:
            return float(1.0 / np.median(diffs))

    return 10.0


def detect_motion_window(
    pose_rows: list,
    pre_buffer_s: float,
    post_buffer_s: float,
    threshold_mm: float,
    fallback_pose_hz: float,
) -> tuple:
    n = len(pose_rows)
    if n == 0:
        return 0, 0, {"motion_found": False, "reason": "no_pose_rows"}
    if n == 1:
        return 0, 1, {"motion_found": False, "reason": "single_pose_row"}

    xyz = np.array([row["pose"][:3] for row in pose_rows], dtype=np.float64)
    elapsed = np.array([float(row["elapsed_s"]) for row in pose_rows], dtype=np.float64)

    if np.any(np.diff(elapsed) < 0):
        elapsed = np.arange(n, dtype=np.float64) / max(fallback_pose_hz, 1e-6)

    step_mm = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    moving = np.where(step_mm >= threshold_mm)[0] + 1

    if moving.size == 0:
        drift_mm = np.linalg.norm(xyz - xyz[0], axis=1)
        moving = np.where(drift_mm >= threshold_mm)[0]

    if moving.size == 0:
        return 0, n, {
            "motion_found": False,
            "reason": "no_motion_detected",
            "threshold_mm": threshold_mm,
        }

    raw_start_idx = int(moving[0])
    raw_end_idx = int(moving[-1])

    start_t = max(0.0, float(elapsed[raw_start_idx] - pre_buffer_s))
    end_t = min(float(elapsed[-1]), float(elapsed[raw_end_idx] + post_buffer_s))

    start_idx = int(np.searchsorted(elapsed, start_t, side="left"))
    end_idx = int(np.searchsorted(elapsed, end_t, side="right"))
    end_idx = max(end_idx, start_idx + 1)
    end_idx = min(end_idx, n)

    return start_idx, end_idx, {
        "motion_found": True,
        "raw_start_idx": raw_start_idx,
        "raw_end_idx": raw_end_idx,
        "trim_start_idx": start_idx,
        "trim_end_idx": end_idx,
        "threshold_mm": threshold_mm,
        "pre_buffer_s": pre_buffer_s,
        "post_buffer_s": post_buffer_s,
        "trim_start_elapsed_s": float(elapsed[start_idx]),
        "trim_end_elapsed_s": float(elapsed[end_idx - 1]),
    }


def compute_trim_window(ep: Episode, max_len: int, args: argparse.Namespace) -> tuple:
    start_idx = 0
    end_idx = max_len
    trim_info = {"enabled": bool(args.trim_motion), "applied": False, "reason": "disabled"}

    if args.trim_motion and ep.pose_rows:
        pose_hz = estimate_pose_hz(ep.pose_rows[:max_len], ep.metadata)
        p_start, p_end, motion_info = detect_motion_window(
            pose_rows=ep.pose_rows[:max_len],
            pre_buffer_s=args.motion_pre_buffer_s,
            post_buffer_s=args.motion_post_buffer_s,
            threshold_mm=args.motion_threshold_mm,
            fallback_pose_hz=pose_hz,
        )
        start_idx = max(0, min(p_start, max_len - 1))
        end_idx = max(start_idx + 1, min(p_end, max_len))
        trim_info = {
            "enabled": True,
            "applied": (start_idx > 0 or end_idx < max_len),
            "pose_hz": pose_hz,
            **motion_info,
        }
    elif args.trim_motion:
        trim_info = {"enabled": True, "applied": False, "reason": "no_pose_rows"}

    keep_len = max(0, end_idx - start_idx)
    trim_info["start_idx"] = start_idx
    trim_info["end_idx"] = end_idx
    trim_info["kept_frames"] = keep_len
    return start_idx, end_idx, trim_info


def discover_episodes(recordings_root: Path, cameras: list) -> list:
    episodes = []
    if not recordings_root.exists():
        return episodes

    for object_dir in sorted(p for p in recordings_root.iterdir() if p.is_dir()):
        object_name = object_dir.name
        for episode_dir in sorted(p for p in object_dir.iterdir() if p.is_dir() and p.name.startswith("episode_")):
            cam_frames = {}
            for cam in cameras:
                rgb_dir = episode_dir / f"{cam}_rgb"
                cam_frames[cam] = sorted(rgb_dir.glob("*.jpg")) if rgb_dir.exists() else []

            metadata = {}
            metadata_path = episode_dir / "metadata.json"
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    metadata = {}

            trajectory_path = episode_dir / "trajectory.csv"
            timeline_rows = parse_timeline_rows(trajectory_path)
            pose_rows = parse_pose_rows(trajectory_path)
            poses = [row["pose"] for row in pose_rows]
            gripper_rows = parse_gripper_rows(episode_dir / "gripper_stream.csv")
            episodes.append(
                Episode(
                    object_name=object_name,
                    episode_name=episode_dir.name,
                    path=episode_dir,
                    cam_frames=cam_frames,
                    timeline_rows=timeline_rows,
                    pose_rows=pose_rows,
                    poses=poses,
                    gripper_rows=gripper_rows,
                    metadata=metadata,
                )
            )
    return episodes


def estimate_frame_timeline(ep: Episode, frame_count: int) -> list:
    timeline = []
    target_fps = _to_float(ep.metadata.get("target_fps"), default=10.0) if ep.metadata else 10.0
    target_fps = max(float(target_fps), 1e-6)
    base_rows = ep.timeline_rows

    for idx in range(frame_count):
        if idx < len(base_rows):
            row = base_rows[idx]
            timeline.append({
                "timestamp_unix": row.get("timestamp_unix"),
                "elapsed_s": float(row.get("elapsed_s", idx / target_fps)),
            })
            continue

        elapsed = float(idx) / target_fps
        ts_unix = None
        if base_rows:
            last = base_rows[-1]
            last_elapsed = float(last.get("elapsed_s", (len(base_rows) - 1) / target_fps))
            elapsed = last_elapsed + ((idx - len(base_rows) + 1) / target_fps)
            if last.get("timestamp_unix") is not None:
                ts_unix = float(last["timestamp_unix"]) + (elapsed - last_elapsed)
        timeline.append({
            "timestamp_unix": ts_unix,
            "elapsed_s": elapsed,
        })

    return timeline


def nearest_timed_row(
    timed_rows: list[tuple[float, dict]],
    target_value: float,
    cursor: int,
    max_delta: float,
) -> tuple[dict | None, int, float | None]:
    if not timed_rows:
        return None, cursor, None

    n = len(timed_rows)
    i = min(max(0, cursor), n - 1)

    while i + 1 < n and timed_rows[i + 1][0] <= target_value:
        i += 1
    while i > 0 and timed_rows[i][0] > target_value:
        i -= 1

    best_i = i
    best_dt = abs(timed_rows[i][0] - target_value)
    if i + 1 < n:
        dt2 = abs(timed_rows[i + 1][0] - target_value)
        if dt2 < best_dt:
            best_i = i + 1
            best_dt = dt2

    if best_dt > max_delta:
        return None, best_i, None
    return timed_rows[best_i][1], best_i, best_dt


def align_gripper_samples(
    ep: Episode,
    frame_timeline: list,
    trim_start: int,
    trim_end: int,
    gripper_min: list[float],
    gripper_max: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    keep_len = max(0, trim_end - trim_start)
    positions = np.zeros((keep_len, 3), dtype=np.float32)
    gripper_open = np.zeros((keep_len,), dtype=np.float32)
    tactile = np.zeros((keep_len, 3), dtype=np.float32)
    tactile_valid = np.zeros((keep_len,), dtype=np.float32)
    sensor_connected = np.zeros((keep_len,), dtype=np.float32)

    if not ep.gripper_rows:
        return positions, gripper_open, tactile, tactile_valid, sensor_connected, {
            "missing": True,
            "matched_frames": 0,
            "total_samples": 0,
            "valid_tactile_frames": 0,
            "sensor_connected_frames": 0,
            "match_max_delta_s": GRIPPER_MATCH_MAX_DELTA_S,
            "match_mode": "none",
        }

    timed_unix = sorted(
        (float(row["timestamp_unix"]), row)
        for row in ep.gripper_rows
        if row.get("timestamp_unix") is not None
    )
    timed_elapsed = sorted(
        (float(row["elapsed_s"]), row)
        for row in ep.gripper_rows
        if row.get("elapsed_s") is not None
    )

    cursor_unix = 0
    cursor_elapsed = 0
    matched = 0
    matched_unix = 0
    matched_elapsed = 0
    tactile_matched = 0
    sensor_connected_matched = 0

    for src_idx in range(trim_start, trim_end):
        frame_idx = src_idx - trim_start
        timeline = frame_timeline[src_idx] if src_idx < len(frame_timeline) else {
            "timestamp_unix": None,
            "elapsed_s": float(frame_idx),
        }

        match = None
        if timeline.get("timestamp_unix") is not None and timed_unix:
            match, cursor_unix, _ = nearest_timed_row(
                timed_unix,
                float(timeline["timestamp_unix"]),
                cursor_unix,
                GRIPPER_MATCH_MAX_DELTA_S,
            )
            if match is not None:
                matched_unix += 1

        if match is None and timeline.get("elapsed_s") is not None and timed_elapsed:
            match, cursor_elapsed, _ = nearest_timed_row(
                timed_elapsed,
                float(timeline["elapsed_s"]),
                cursor_elapsed,
                GRIPPER_MATCH_MAX_DELTA_S,
            )
            if match is not None:
                matched_elapsed += 1

        if match is None:
            continue

        positions[frame_idx] = np.asarray(match["positions"], dtype=np.float32)
        if match.get("tactile") is not None:
            tactile[frame_idx] = np.asarray(match["tactile"], dtype=np.float32)
        tactile_valid[frame_idx] = 1.0 if bool(match.get("sample_valid")) else 0.0
        sensor_connected[frame_idx] = 1.0 if bool(match.get("sensor_connected")) else 0.0
        if tactile_valid[frame_idx] > 0.5:
            tactile_matched += 1
        if sensor_connected[frame_idx] > 0.5:
            sensor_connected_matched += 1
        matched += 1

    min_arr = np.asarray(gripper_min, dtype=np.float32)
    max_arr = np.asarray(gripper_max, dtype=np.float32)
    denom = np.maximum(max_arr - min_arr, 1e-6)
    clipped = np.clip(positions, min_arr, max_arr)
    gripper_open = ((clipped - min_arr) / denom).mean(axis=1).astype(np.float32)

    match_mode = "none"
    if matched_unix > 0 and matched_elapsed > 0:
        match_mode = "mixed"
    elif matched_unix > 0:
        match_mode = "timestamp_unix"
    elif matched_elapsed > 0:
        match_mode = "elapsed_s"

    return positions, gripper_open, tactile, tactile_valid, sensor_connected, {
        "missing": False,
        "matched_frames": matched,
        "matched_by_timestamp_unix": matched_unix,
        "matched_by_elapsed_s": matched_elapsed,
        "total_samples": len(ep.gripper_rows),
        "valid_tactile_frames": tactile_matched,
        "sensor_connected_frames": sensor_connected_matched,
        "match_max_delta_s": GRIPPER_MATCH_MAX_DELTA_S,
        "match_mode": match_mode,
    }


def chinese_hint(objects_cfg: dict, object_name: str) -> str:
    info = objects_cfg.get("classes", {}).get(object_name, {})
    zh = info.get("chinese", [])
    if isinstance(zh, list) and zh:
        return str(zh[0])
    return object_name


def prepare_vla(
    episodes: list,
    out_root: Path,
    cameras: list,
    force_copy: bool,
    objects_cfg: dict,
    args: argparse.Namespace,
) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "episodes_total": 0,
        "episodes_written": 0,
        "episodes_skipped_short": 0,
        "episodes_trimmed_motion": 0,
        "output_root": str(out_root),
        "cameras": cameras,
        "min_frames": int(args.min_frames),
        "trim_motion": bool(args.trim_motion),
    }

    written = []
    skipped_short = []
    ep_idx = 0

    for ep in episodes:
        available_cams = [cam for cam in cameras if ep.cam_frames.get(cam)]
        if not available_cams:
            continue

        frame_counts = [len(ep.cam_frames[cam]) for cam in available_cams]
        min_len = min(frame_counts)
        if ep.poses:
            min_len = min(min_len, len(ep.poses))
        if min_len <= 0:
            continue

        frame_timeline = estimate_frame_timeline(ep, min_len)

        trim_start, trim_end, trim_info = compute_trim_window(ep, min_len, args)
        keep_len = trim_end - trim_start
        if keep_len < args.min_frames:
            skipped_short.append({
                "source_episode": display_path(ep.path),
                "kept_frames": int(keep_len),
                "min_frames": int(args.min_frames),
                "trim": trim_info,
            })
            continue

        if trim_info.get("applied"):
            summary["episodes_trimmed_motion"] += 1

        dst_ep = out_root / f"episode_{ep_idx}"
        if dst_ep.exists():
            shutil.rmtree(dst_ep)
        dst_ep.mkdir(parents=True, exist_ok=True)

        for cam_i, cam in enumerate(available_cams, start=1):
            cam_out = dst_ep / f"camera{cam_i}"
            cam_out.mkdir(parents=True, exist_ok=True)
            for i in range(trim_start, trim_end):
                src = ep.cam_frames[cam][i]
                frame_idx = i - trim_start
                dst = cam_out / f"frame_{frame_idx:06d}{src.suffix.lower()}"
                link_or_copy(src, dst, force_copy)

        if ep.poses:
            poses = np.array(ep.poses[trim_start:trim_end], dtype=np.float32)
        else:
            poses = np.zeros((keep_len, 6), dtype=np.float32)
        np.save(dst_ep / "ee_poses.npy", poses)
        gripper_pos, gripper_open, tactile_values, tactile_valid, tactile_connected, gripper_meta = align_gripper_samples(
            ep=ep,
            frame_timeline=frame_timeline,
            trim_start=trim_start,
            trim_end=trim_end,
            gripper_min=args.gripper_min,
            gripper_max=args.gripper_max,
        )
        np.save(dst_ep / "gripper_pos.npy", gripper_pos)
        np.save(dst_ep / "gripper_open.npy", gripper_open)
        np.save(dst_ep / "tactile.npy", tactile_values)
        np.save(dst_ep / "tactile_valid.npy", tactile_valid)
        np.save(dst_ep / "tactile_connected.npy", tactile_connected)

        zh_name = chinese_hint(objects_cfg, ep.object_name)
        episode_meta = {
            "object": ep.object_name,
            "object_zh": zh_name,
            "instruction_en": f"pick {ep.object_name}",
            "instruction_zh": f"拿起{zh_name}",
            "source_episode": display_path(ep.path),
            "num_frames": int(keep_len),
            "cameras": available_cams,
            "trim": trim_info,
            "gripper_missing": bool(gripper_meta["missing"]),
            "gripper": {
                **gripper_meta,
                "source_csv": display_path(ep.path / "gripper_stream.csv"),
                "min": [float(v) for v in args.gripper_min],
                "max": [float(v) for v in args.gripper_max],
            },
            "tactile": {
                "valid_frames": int(float(tactile_valid.sum())),
                "connected_frames": int(float(tactile_connected.sum())),
                "source_csv": display_path(ep.path / "gripper_stream.csv"),
            },
        }
        with open(dst_ep / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(episode_meta, f, indent=2, ensure_ascii=False)

        written.append(episode_meta)
        ep_idx += 1

    dataset_info = {
        "name": "own3_processed",
        "num_episodes": len(written),
        "episodes": written,
        "skipped_short": skipped_short,
    }
    with open(out_root / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    summary["episodes_total"] = len(episodes)
    summary["episodes_written"] = len(written)
    summary["episodes_skipped_short"] = len(skipped_short)
    return summary


def build_class_maps(objects_cfg: dict, episodes: list) -> tuple:
    class_names = list(objects_cfg.get("classes", {}).keys())
    discovered = sorted(set(ep.object_name for ep in episodes))
    for name in discovered:
        if name not in class_names:
            class_names.append(name)
    class_to_id = {name: i for i, name in enumerate(class_names)}
    return class_names, class_to_id


def make_yolo_layout(yolo_root: Path) -> dict:
    paths = {
        "train_img": yolo_root / "images" / "train",
        "val_img": yolo_root / "images" / "val",
        "train_lbl": yolo_root / "labels" / "train",
        "val_lbl": yolo_root / "labels" / "val",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def yolo_line_from_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int, cls_id: int) -> str:
    cx = ((x1 + x2) / 2.0) / float(w)
    cy = ((y1 + y2) / 2.0) / float(h)
    bw = (x2 - x1) / float(w)
    bh = (y2 - y1) / float(h)
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def prepare_yolo(
    episodes: list,
    yolo_root: Path,
    cameras: list,
    stride: int,
    val_ratio: float,
    force_copy: bool,
    objects_cfg: dict,
    auto_label: bool,
    yolo_model: str,
    min_conf: float,
    args: argparse.Namespace,
) -> dict:
    if stride <= 0:
        raise ValueError("image-stride must be >= 1")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val-ratio must be between 0 and 1")

    layout = make_yolo_layout(yolo_root)
    class_names, class_to_id = build_class_maps(objects_cfg, episodes)

    detector = None
    if auto_label:
        from src.detector import ObjectDetector
        detector = ObjectDetector(yolo_model=yolo_model)

    total_images = 0
    auto_labeled = 0
    empty_labels = 0
    episodes_used = 0
    episodes_skipped_short = 0
    manifest_path = yolo_root / "annotations_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for ep in episodes:
            cam_lengths = [len(ep.cam_frames.get(cam, [])) for cam in cameras if ep.cam_frames.get(cam)]
            if not cam_lengths:
                continue

            max_len = min(cam_lengths)
            if ep.poses:
                max_len = min(max_len, len(ep.poses))
            if max_len <= 0:
                continue

            trim_start, trim_end, _ = compute_trim_window(ep, max_len, args)
            keep_len = trim_end - trim_start
            if keep_len < args.min_frames:
                episodes_skipped_short += 1
                continue

            episodes_used += 1
            cls_id = class_to_id[ep.object_name]
            for cam in cameras:
                frames = ep.cam_frames.get(cam, [])
                if not frames:
                    continue

                cap = min(len(frames), max_len)
                for i in range(trim_start, min(trim_end, cap), stride):
                    src = frames[i]
                    stem = f"{ep.object_name}_{ep.episode_name}_{cam}_{src.stem}"
                    split_key = stable_split_key(stem)
                    split = "val" if split_key < val_ratio else "train"

                    dst_img = layout[f"{split}_img"] / f"{stem}.jpg"
                    dst_lbl = layout[f"{split}_lbl"] / f"{stem}.txt"
                    link_or_copy(src, dst_img, force_copy)

                    lines = []
                    label_status = "empty"
                    if detector is not None:
                        import cv2

                        img = cv2.imread(str(src))
                        if img is not None:
                            dets = detector.detect_from_image(
                                img,
                                target_object=ep.object_name,
                                confidence_threshold=min_conf,
                            )
                            if dets:
                                x1, y1, x2, y2 = dets[0]["bbox"]
                                h, w = img.shape[:2]
                                lines = [yolo_line_from_bbox(x1, y1, x2, y2, w, h, cls_id)]
                                label_status = "auto"

                    with open(dst_lbl, "w", encoding="utf-8") as f:
                        if lines:
                            f.write("\n".join(lines) + "\n")

                    if lines:
                        auto_labeled += 1
                    else:
                        empty_labels += 1

                    total_images += 1
                    row = {
                        "image": str(dst_img.relative_to(yolo_root)),
                        "label": str(dst_lbl.relative_to(yolo_root)),
                        "object": ep.object_name,
                        "class_id": cls_id,
                        "label_status": label_status,
                        "source": display_path(src),
                    }
                    manifest.write(json.dumps(row, ensure_ascii=False) + "\n")

    data_yaml = yolo_root / "data.yaml"
    names_dict = {i: n for i, n in enumerate(class_names)}
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {yolo_root}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for i, name in names_dict.items():
            f.write(f"  {i}: {name}\n")

    return {
        "images_total": total_images,
        "images_auto_labeled": auto_labeled,
        "images_empty_label": empty_labels,
        "episodes_used": episodes_used,
        "episodes_skipped_short": episodes_skipped_short,
        "min_frames": int(args.min_frames),
        "trim_motion": bool(args.trim_motion),
        "output_root": str(yolo_root),
        "data_yaml": str(data_yaml),
        "manifest": str(manifest_path),
    }


def main() -> None:
    args = parse_args()
    args.gripper_min = parse_triplet_arg(args.gripper_min, "--gripper-min")
    args.gripper_max = parse_triplet_arg(args.gripper_max, "--gripper-max")
    if args.min_frames <= 0:
        raise SystemExit("--min-frames must be >= 1")
    if args.motion_threshold_mm <= 0:
        raise SystemExit("--motion-threshold-mm must be > 0")
    if args.motion_pre_buffer_s < 0 or args.motion_post_buffer_s < 0:
        raise SystemExit("--motion-pre-buffer-s/--motion-post-buffer-s must be >= 0")
    for idx, (gmin, gmax) in enumerate(zip(args.gripper_min, args.gripper_max), start=1):
        if gmax <= gmin:
            raise SystemExit(f"--gripper-max axis {idx} must be greater than --gripper-min")

    recordings_root = (PROJECT_ROOT / args.recordings_root).resolve()
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    if not cameras:
        raise SystemExit("No cameras selected")

    episodes = discover_episodes(recordings_root, cameras)
    if args.limit_episodes > 0:
        episodes = episodes[:args.limit_episodes]

    if not episodes:
        raise SystemExit(f"No episodes found under {recordings_root}")

    objects_cfg = load_config("objects.yaml")
    report = {
        "recordings_root": str(recordings_root),
        "episodes_found": len(episodes),
        "mode": args.mode,
        "trim_motion": bool(args.trim_motion),
        "motion_threshold_mm": float(args.motion_threshold_mm),
        "motion_pre_buffer_s": float(args.motion_pre_buffer_s),
        "motion_post_buffer_s": float(args.motion_post_buffer_s),
        "min_frames": int(args.min_frames),
        "gripper_min": [float(v) for v in args.gripper_min],
        "gripper_max": [float(v) for v in args.gripper_max],
        "gripper_match_max_delta_s": GRIPPER_MATCH_MAX_DELTA_S,
    }

    if args.mode in {"both", "vla"}:
        vla_out = (PROJECT_ROOT / args.vla_out).resolve()
        report["vla"] = prepare_vla(
            episodes=episodes,
            out_root=vla_out,
            cameras=cameras,
            force_copy=args.copy,
            objects_cfg=objects_cfg,
            args=args,
        )

    if args.mode in {"both", "yolo"}:
        yolo_out = (PROJECT_ROOT / args.yolo_out).resolve()
        report["yolo"] = prepare_yolo(
            episodes=episodes,
            yolo_root=yolo_out,
            cameras=cameras,
            stride=args.image_stride,
            val_ratio=args.val_ratio,
            force_copy=args.copy,
            objects_cfg=objects_cfg,
            auto_label=args.auto_label,
            yolo_model=args.yolo_model,
            min_conf=args.min_conf,
            args=args,
        )

    report_path = (PROJECT_ROOT / "data" / "datasets" / "prepare_dataset_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Dataset preparation complete")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
