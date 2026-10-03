from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.utils import CONFIG_DIR, PROJECT_ROOT


RUNTIME_DIR = PROJECT_ROOT / "data" / "runtime"
LOG_DIR = PROJECT_ROOT / "data" / "runtime" / "logs"
LAUNCHER_STATE_PATH = RUNTIME_DIR / "launcher_state.json"
DEFAULT_PROFILE = "full"


DEFAULT_MODULES: dict[str, dict[str, bool]] = {
    "voice": {"enabled": True, "show_in_ui": True},
    "cam1": {"enabled": True, "show_in_ui": True},
    "cam2": {"enabled": True, "show_in_ui": True},
    "claw_cam": {"enabled": True, "show_in_ui": True},
    "sensor_chart": {"enabled": True, "show_in_ui": True},
    "arm_monitor": {"enabled": True, "show_in_ui": True},
    "gripper": {"enabled": True, "show_in_ui": True},
    "validation": {"enabled": False, "show_in_ui": False},
    "quick_pick": {"enabled": True, "show_in_ui": True},
    "vla": {"enabled": True, "show_in_ui": True},
    "dataset_capture": {"enabled": True, "show_in_ui": True},
    "trajectory": {"enabled": True, "show_in_ui": True},
    "teach": {"enabled": True, "show_in_ui": True},
    "logs": {"enabled": True, "show_in_ui": True},
    "virtual_env": {"enabled": False, "show_in_ui": False},
    "virtual_env_data": {"enabled": False, "show_in_ui": False},
}


DEFAULT_QUICK_PICK_CONTROLS: dict[str, Any] = {
    "show_arm_controls": True,
    "show_reload_config": True,
    "show_catalog_picker": True,
    "fixed_fallback_objects": [
        "trapezoid",
        "board",
        "butter_knife",
    ],
}


@dataclass
class RuntimeBundle:
    profile_name: str
    launch_profile: dict[str, Any]
    demo_config: dict[str, Any]
    modbus_config: dict[str, Any]
    launch_profiles: dict[str, Any]
    env_overlay_path: str | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"YAML root must be a mapping: {path}")
    return data


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_optional_config(path_or_name: str | None) -> Path | None:
    if not path_or_name:
        return None
    path = Path(path_or_name)
    if path.is_absolute():
        return path
    candidate = CONFIG_DIR / path
    if candidate.exists():
        return candidate
    return (PROJECT_ROOT / path).resolve()


def _normalize_modules(demo_cfg: dict[str, Any]) -> None:
    modules = demo_cfg.setdefault("modules", {})
    for name, defaults in DEFAULT_MODULES.items():
        current = modules.get(name, {})
        if not isinstance(current, dict):
            current = {"enabled": bool(current), "show_in_ui": bool(current)}
        modules[name] = {
            "enabled": bool(current.get("enabled", defaults["enabled"])),
            "show_in_ui": bool(current.get("show_in_ui", defaults["show_in_ui"])),
        }


def _base_module_visibility(demo_cfg: dict[str, Any]) -> dict[str, bool]:
    normalized = copy.deepcopy(demo_cfg)
    _normalize_modules(normalized)
    return {
        name: bool(node["show_in_ui"])
        for name, node in normalized["modules"].items()
        if isinstance(node, dict) and "show_in_ui" in node
    }


def _restore_module_visibility(
    demo_cfg: dict[str, Any],
    base_visibility: dict[str, bool],
) -> None:
    modules = demo_cfg.setdefault("modules", {})
    for name, visible in base_visibility.items():
        node = modules.get(name)
        if not isinstance(node, dict):
            node = {
                "enabled": bool(node),
                "show_in_ui": visible,
            }
            modules[name] = node
        node["show_in_ui"] = visible


def _base_ui_controls(demo_cfg: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(demo_cfg)
    _normalize_ui_cfg(normalized)
    return copy.deepcopy(normalized["ui"]["controls"])


def _restore_ui_controls(
    demo_cfg: dict[str, Any],
    base_controls: dict[str, Any],
) -> None:
    ui_cfg = demo_cfg.setdefault("ui", {})
    ui_cfg["controls"] = copy.deepcopy(base_controls)


def _normalize_gripper_endpoints(demo_cfg: dict[str, Any]) -> None:
    gripper_cfg = demo_cfg.setdefault("gripper", {})
    endpoints = gripper_cfg.get("endpoints")
    if not isinstance(endpoints, list) or not endpoints:
        host = gripper_cfg.get("agx_ip", "127.0.0.1")
        port = int(gripper_cfg.get("agx_port", 5002))
        gripper_cfg["endpoints"] = [{
            "host": host,
            "port": port,
            "label": f"{host}:{port}",
        }]
    remote_recording = gripper_cfg.setdefault("remote_recording", {})
    remote_recording.setdefault("enabled", True)
    remote_recording.setdefault("filename_prefix", "tactile_v2")
    remote_recording.setdefault("output_dir", "data/agx_tactile_raw")
    remote_recording.setdefault("start_timeout_s", 2.0)
    remote_recording.setdefault("stop_timeout_s", 4.0)


def _normalize_arm_connections(demo_cfg: dict[str, Any]) -> None:
    arm_cfg = demo_cfg.setdefault("arm", {})
    connections = arm_cfg.get("connections")
    if not isinstance(connections, list) or not connections:
        arm_cfg["connections"] = [{
            "host": "127.0.0.1",
            "port": 1502,
            "label": "default_local",
        }]


def _normalize_dataset_capture(demo_cfg: dict[str, Any]) -> None:
    teach_cfg = demo_cfg.setdefault("teach", {})
    dataset_cfg = teach_cfg.setdefault("dataset_capture", {})
    dataset_cfg.setdefault("enabled", True)
    dataset_cfg.setdefault("output_root", "data/recordings")
    dataset_cfg.setdefault("image_fps", 6)
    dataset_cfg.setdefault("telemetry_hz", 10)
    dataset_cfg.setdefault(
        "camera_streams",
        ["cam1_rgb", "cam1_depth", "cam2_rgb", "cam2_depth", "claw_rgb"],
    )
    auto_label = teach_cfg.setdefault("auto_label", {})
    auto_label.setdefault("enabled", True)
    auto_label.setdefault("streams", ["claw_rgb"])
    auto_label.setdefault("model_path", "data/models/yolo/yolo12sbest.pt")
    auto_label.setdefault("conf", 0.35)
    auto_label.setdefault("export_format", "cvat_yolo")


def _normalize_camera_cfg(demo_cfg: dict[str, Any]) -> None:
    cameras = demo_cfg.setdefault("cameras", {})
    claw = cameras.setdefault("claw", {})
    claw.setdefault("enabled", True)
    claw.setdefault("source", 0)
    claw.setdefault("label", "Claw Cam")
    claw.setdefault("performance_profile", "balanced")
    # capture_* must be a NATIVE camera mode (see demo_config.yaml note): the
    # Realtek claw only offers 1920x1080 MJPG @ 24/15/10 — a non-native request
    # stalls the UVC stream and triggers reopen churn. Downscale in software.
    claw.setdefault("profiles", {
        "quality": {
            "capture_w": 1920,
            "capture_h": 1080,
            "stream_w": 640,
            "stream_h": 360,
            "infer_w": 640,
            "infer_h": 360,
            "stream_fps": 15,
            "jpeg_quality": 65,
            "infer_interval_s": 0.12,
        },
        "balanced": {
            "capture_w": 1920,
            "capture_h": 1080,
            "stream_w": 424,
            "stream_h": 240,
            "infer_w": 424,
            "infer_h": 240,
            "stream_fps": 15,
            "jpeg_quality": 50,
            "infer_interval_s": 0.12,
        },
        "low_latency": {
            "capture_w": 1920,
            "capture_h": 1080,
            "stream_w": 320,
            "stream_h": 180,
            "infer_w": 320,
            "infer_h": 180,
            "stream_fps": 10,
            "jpeg_quality": 40,
            "infer_interval_s": 0.18,
        },
    })
    claw.setdefault("overrides", {})
    claw.setdefault("device", "auto")
    claw.setdefault("enable_yolo", True)
    claw.setdefault("model_path", "data/models/yolo/yolo12sbest.pt")
    claw.setdefault("conf", 0.6)


def _normalize_ui_cfg(demo_cfg: dict[str, Any]) -> None:
    ui_cfg = demo_cfg.setdefault("ui", {})
    mode = str(ui_cfg.get("mode", "modern")).strip().lower()
    if mode not in {"modern", "classic"}:
        mode = "modern"
    ui_cfg["mode"] = mode

    controls = ui_cfg.get("controls")
    if not isinstance(controls, dict):
        controls = {}
        ui_cfg["controls"] = controls
    quick_pick = controls.get("quick_pick")
    if not isinstance(quick_pick, dict):
        quick_pick = {}

    raw_objects = quick_pick.get(
        "fixed_fallback_objects",
        DEFAULT_QUICK_PICK_CONTROLS["fixed_fallback_objects"],
    )
    if not isinstance(raw_objects, list):
        raw_objects = DEFAULT_QUICK_PICK_CONTROLS[
            "fixed_fallback_objects"
        ]
    objects: list[str] = []
    for raw in raw_objects:
        if not isinstance(raw, str):
            continue
        object_key = raw.strip()
        if object_key and object_key not in objects:
            objects.append(object_key)

    controls["quick_pick"] = {
        "show_arm_controls": bool(quick_pick.get(
            "show_arm_controls",
            DEFAULT_QUICK_PICK_CONTROLS["show_arm_controls"],
        )),
        "show_reload_config": bool(quick_pick.get(
            "show_reload_config",
            DEFAULT_QUICK_PICK_CONTROLS["show_reload_config"],
        )),
        "show_catalog_picker": bool(quick_pick.get(
            "show_catalog_picker",
            DEFAULT_QUICK_PICK_CONTROLS["show_catalog_picker"],
        )),
        "fixed_fallback_objects": objects,
    }


def normalize_demo_config(demo_cfg: dict[str, Any]) -> dict[str, Any]:
    _normalize_modules(demo_cfg)
    _normalize_gripper_endpoints(demo_cfg)
    _normalize_arm_connections(demo_cfg)
    _normalize_dataset_capture(demo_cfg)
    _normalize_camera_cfg(demo_cfg)
    _normalize_ui_cfg(demo_cfg)
    return demo_cfg


def load_runtime_bundle(profile: str | None = None, env_config: str | None = None) -> RuntimeBundle:
    launch_profiles = _load_yaml(CONFIG_DIR / "launch_profiles.yaml")
    profiles = launch_profiles.get("profiles", {})
    profile_name = profile or launch_profiles.get("default_profile", DEFAULT_PROFILE)
    if profile_name not in profiles:
        raise RuntimeError(f"Unknown launch profile: {profile_name}")

    demo_cfg = _load_yaml(CONFIG_DIR / "demo_config.yaml")
    base_visibility = _base_module_visibility(demo_cfg)
    base_ui_controls = _base_ui_controls(demo_cfg)
    modbus_cfg = _load_yaml(CONFIG_DIR / "modbus_config.yaml")
    launch_profile = profiles[profile_name]

    if isinstance(launch_profile.get("demo_overrides"), dict):
        demo_cfg = _deep_merge(demo_cfg, launch_profile["demo_overrides"])
    if isinstance(launch_profile.get("modbus_overrides"), dict):
        modbus_cfg = _deep_merge(modbus_cfg, launch_profile["modbus_overrides"])

    env_overlay_path = _resolve_optional_config(env_config)
    if env_overlay_path is not None and env_overlay_path.exists():
        overlay = _load_yaml(env_overlay_path)
        if isinstance(overlay.get("demo"), dict):
            demo_cfg = _deep_merge(demo_cfg, overlay["demo"])
        else:
            demo_cfg = _deep_merge(demo_cfg, overlay)
        if isinstance(overlay.get("modbus"), dict):
            modbus_cfg = _deep_merge(modbus_cfg, overlay["modbus"])

    normalize_demo_config(demo_cfg)
    _restore_module_visibility(demo_cfg, base_visibility)
    _restore_ui_controls(demo_cfg, base_ui_controls)
    return RuntimeBundle(
        profile_name=profile_name,
        launch_profile=launch_profile,
        demo_config=demo_cfg,
        modbus_config=modbus_cfg,
        launch_profiles=launch_profiles,
        env_overlay_path=str(env_overlay_path) if env_overlay_path is not None else None,
    )


FALLBACK_TAUGHT_PATH = CONFIG_DIR / "fallback_taught.yaml"


def load_taught_fallback() -> dict[str, Any]:
    """Load auto-generated fallback overrides written by scripts/fallback_teach.py.

    Returns an empty dict if the file is absent or unreadable, so a missing teach
    file simply falls back to the committed seeds in config/objects.yaml.
    """
    if not FALLBACK_TAUGHT_PATH.exists():
        return {}
    try:
        with open(FALLBACK_TAUGHT_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def merge_fallback(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge taught overrides over committed fallback config (public wrapper)."""
    return _deep_merge(base, overlay)


FINGER_MOTOR_LIMITS = ((1872, 4272), (1872, 4272), (848, 3248))


class FixedFallbackConfigError(ValueError):
    """A fixed fallback route is incomplete or unsafe to start."""


def _finite_number(node: dict[str, Any], key: str, path: str) -> float:
    value = node.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise FixedFallbackConfigError(
            f"{path}.{key} must be a finite number"
        )
    return float(value)


def _bounded_int(
    node: dict[str, Any],
    key: str,
    path: str,
    low: int,
    high: int,
) -> int:
    value = node.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise FixedFallbackConfigError(f"{path}.{key} must be an integer")
    if value < low or value > high:
        raise FixedFallbackConfigError(
            f"{path}.{key} must be between {low} and {high}"
        )
    return value


def _boolean(node: dict[str, Any], key: str, path: str) -> bool:
    value = node.get(key)
    if not isinstance(value, bool):
        raise FixedFallbackConfigError(f"{path}.{key} must be boolean")
    return value


def _finger_position(
    value: Any,
    path: str,
    *,
    required: bool,
) -> list[int] | None:
    if value is None and not required:
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise FixedFallbackConfigError(f"{path} must be three motor integers")
    result: list[int] = []
    for index, raw in enumerate(value):
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise FixedFallbackConfigError(
                f"{path}[{index}] must be an integer"
            )
        low, high = FINGER_MOTOR_LIMITS[index]
        if raw < low or raw > high:
            raise FixedFallbackConfigError(
                f"{path}[{index}]={raw} is outside [{low}, {high}]"
            )
        result.append(raw)
    return result


def _check_axis_mm(
    demo_cfg: dict[str, Any],
    axis: str,
    value_mm: float,
    path: str,
) -> None:
    boundary = demo_cfg.get("safety_boundary") or {}
    low = boundary.get(f"{axis}_min")
    high = boundary.get(f"{axis}_max")
    value_um = value_mm * 1000.0
    if low is not None and value_um < float(low):
        raise FixedFallbackConfigError(f"{path} is below {axis}_min")
    if high is not None and value_um > float(high):
        raise FixedFallbackConfigError(f"{path} is above {axis}_max")


def _fixed_fallback_object(
    demo_cfg: dict[str, Any],
    object_key: str,
    *,
    require_grasp_goal: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fixed = demo_cfg.get("fixed_fallback")
    if not isinstance(fixed, dict):
        raise FixedFallbackConfigError("fixed_fallback must be configured")
    objects = fixed.get("objects")
    if not isinstance(objects, dict):
        raise FixedFallbackConfigError(
            "fixed_fallback.objects must be configured"
        )
    node = objects.get(object_key)
    if not isinstance(node, dict):
        raise FixedFallbackConfigError(
            f"fixed_fallback.objects.{object_key} is not configured"
        )

    path = f"fixed_fallback.objects.{object_key}"
    x_mm = _finite_number(node, "pick_x_mm", path)
    y_mm = _finite_number(node, "pick_y_mm", path)
    grasp_z_mm = _finite_number(node, "grasp_z_mm", path)
    release_z_mm = _finite_number(node, "release_z_mm", path)
    contact_dwell_s = _finite_number(node, "contact_dwell_s", path)
    hover_z_mm = _finite_number(
        fixed, "hover_z_mm", "fixed_fallback"
    )
    if contact_dwell_s < 0:
        raise FixedFallbackConfigError(
            f"{path}.contact_dwell_s cannot be negative"
        )
    for axis, value, key in (
        ("x", x_mm, f"{path}.pick_x_mm"),
        ("y", y_mm, f"{path}.pick_y_mm"),
        ("z", grasp_z_mm, f"{path}.grasp_z_mm"),
        ("z", release_z_mm, f"{path}.release_z_mm"),
        ("z", hover_z_mm, "fixed_fallback.hover_z_mm"),
    ):
        _check_axis_mm(demo_cfg, axis, value, key)

    place_pose = fixed.get("place_approach_pose")
    if (
        not isinstance(place_pose, list)
        or len(place_pose) != 6
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in place_pose
        )
    ):
        raise FixedFallbackConfigError(
            "fixed_fallback.place_approach_pose must contain six integers"
        )
    for axis, value_mm, key in (
        (
            "x",
            place_pose[0] / 1000.0,
            "fixed_fallback.place_approach_pose[0]",
        ),
        (
            "y",
            place_pose[1] / 1000.0,
            "fixed_fallback.place_approach_pose[1]",
        ),
        (
            "z",
            place_pose[2] / 1000.0,
            "fixed_fallback.place_approach_pose[2]",
        ),
    ):
        _check_axis_mm(demo_cfg, axis, value_mm, key)

    requires_grasp = _boolean(node, "requires_grasp", path)
    lift_during_grasp = _boolean(node, "lift_during_grasp", path)
    grasp_position = _finger_position(
        node.get("fixed_grasp_position"),
        f"{path}.fixed_grasp_position",
        required=require_grasp_goal and requires_grasp,
    )
    release_position = _finger_position(
        fixed.get("fixed_release_position"),
        "fixed_fallback.fixed_release_position",
        required=True,
    )

    trigger = node.get("grasp_start_z_mm")
    if trigger is not None:
        trigger = _finite_number(node, "grasp_start_z_mm", path)
        if trigger < grasp_z_mm or trigger > hover_z_mm:
            raise FixedFallbackConfigError(
                f"{path}.grasp_start_z_mm must be between grasp and hover Z"
            )
    if lift_during_grasp and requires_grasp and trigger is None:
        raise FixedFallbackConfigError(
            f"{path}.grasp_start_z_mm is required when lift_during_grasp is true"
        )

    return fixed, {
        **node,
        "pick_x_mm": x_mm,
        "pick_y_mm": y_mm,
        "grasp_z_mm": grasp_z_mm,
        "release_z_mm": release_z_mm,
        "contact_dwell_s": contact_dwell_s,
        "grasp_start_z_mm": trigger,
        "requires_grasp": requires_grasp,
        "lift_during_grasp": lift_during_grasp,
        "fixed_grasp_position": grasp_position,
        "fixed_release_position": release_position,
        "place_approach_pose": list(place_pose),
    }


def build_fixed_fallback_cfg(
    demo_cfg: dict[str, Any],
    object_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve and fully validate one deterministic fallback before motion."""
    fixed, obj = _fixed_fallback_object(
        demo_cfg, object_key, require_grasp_goal=True
    )
    travel = _bounded_int(
        fixed, "travel_speed_percent", "fixed_fallback", 1, 100
    )
    descend = _bounded_int(
        fixed, "descend_speed_percent", "fixed_fallback", 1, 100
    )
    step_timeout = _finite_number(
        fixed, "per_step_timeout_s", "fixed_fallback"
    )
    start_timeout = _finite_number(
        fixed, "motion_start_timeout_s", "fixed_fallback"
    )
    settle_hold = _finite_number(
        fixed, "fixed_settle_hold_s", "fixed_fallback"
    )
    if step_timeout <= 0:
        raise FixedFallbackConfigError(
            "fixed_fallback.per_step_timeout_s must be positive"
        )
    if start_timeout <= 0:
        raise FixedFallbackConfigError(
            "fixed_fallback.motion_start_timeout_s must be positive"
        )
    if settle_hold < 0:
        raise FixedFallbackConfigError(
            "fixed_fallback.fixed_settle_hold_s cannot be negative"
        )

    run_cfg = {
        "fixed_fallback": True,
        "fixed_xy": [obj["pick_x_mm"], obj["pick_y_mm"]],
        "hover_z_mm": float(fixed["hover_z_mm"]),
        "travel_speed_percent": travel,
        "descend_speed_percent": descend,
        "per_step_timeout_s": step_timeout,
        "motion_start_timeout_s": start_timeout,
        "confirm_descend": False,
        "fixed_target_tolerance_ticks": _bounded_int(
            fixed,
            "fixed_target_tolerance_ticks",
            "fixed_fallback",
            0,
            2400,
        ),
        "fixed_settle_ticks": _bounded_int(
            fixed, "fixed_settle_ticks", "fixed_fallback", 0, 2400
        ),
        "fixed_settle_hold_s": settle_hold,
        "fixed_min_close_ticks": _bounded_int(
            fixed, "fixed_min_close_ticks", "fixed_fallback", 1, 2400
        ),
        "fixed_min_close_fingers": _bounded_int(
            fixed, "fixed_min_close_fingers", "fixed_fallback", 1, 3
        ),
    }
    tail_cfg = {
        "finger_policy": "fixed",
        "requires_grasp": obj["requires_grasp"],
        "fixed_grasp_position": obj["fixed_grasp_position"],
        "fixed_release_position": obj["fixed_release_position"],
        "grasp_z_mm": obj["grasp_z_mm"],
        "grasp_z_source": "fixed_fallback.grasp_z_mm (rig constant)",
        "grasp_contact_dwell_s": obj["contact_dwell_s"],
        "auto_lift_after_dwell": not obj["requires_grasp"],
        "lift_during_grasp": obj["lift_during_grasp"],
        "grasp_start_z_mm": obj["grasp_start_z_mm"],
        "grasp_backoff_mm": 0.0,
        "place_approach_pose": obj["place_approach_pose"],
        "place_z_mm": obj["release_z_mm"],
        "lstm": {},
    }
    return run_cfg, tail_cfg


def build_auto_run_cfg(demo_cfg: dict[str, Any], objects_cfg: dict[str, Any],
                       object_key: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """Resolve one object's auto-run configuration (three-object demo §3).

    Single source of truth for the UI socket handler AND the terminal CLI:
    returns (run_cfg, tail_cfg, obj_cfg) from demo_config's vla_auto block,
    with all rig geometry taken from fixed_fallback. LSTM action names and
    intelligent-approach settings remain in vla_auto. None = unknown object.
    """
    auto_cfg = dict(demo_cfg.get("vla_auto", {}))
    obj_cfg = dict((auto_cfg.get("objects") or {}).get(object_key) or {})
    if not obj_cfg:
        return None
    run_cfg = {k: v for k, v in auto_cfg.items() if k not in ("objects", "checkpoint")}

    grasp_thresholds = {
        "grasp_prepare_min_close_ticks": auto_cfg.get(
            "grasp_prepare_min_close_ticks", 20
        ),
        "grasp_min_close_ticks": auto_cfg.get("grasp_min_close_ticks", 60),
    }
    grasp_prepare_min_close_ticks = _bounded_int(
        grasp_thresholds,
        "grasp_prepare_min_close_ticks",
        "vla_auto",
        1,
        2400,
    )
    grasp_min_close_ticks = _bounded_int(
        grasp_thresholds,
        "grasp_min_close_ticks",
        "vla_auto",
        1,
        2400,
    )
    if grasp_prepare_min_close_ticks > grasp_min_close_ticks:
        raise FixedFallbackConfigError(
            "vla_auto.grasp_prepare_min_close_ticks must not exceed "
            "vla_auto.grasp_min_close_ticks"
        )
    run_cfg["grasp_prepare_min_close_ticks"] = (
        grasp_prepare_min_close_ticks
    )
    run_cfg["grasp_min_close_ticks"] = grasp_min_close_ticks

    _, fixed_obj = _fixed_fallback_object(
        demo_cfg, object_key, require_grasp_goal=False
    )
    recorded_xy = [fixed_obj["pick_x_mm"], fixed_obj["pick_y_mm"]]
    run_cfg["recorded_xy"] = recorded_xy
    run_cfg["fixed_xy"] = (
        recorded_xy if obj_cfg.get("use_recorded_xy") else None
    )
    # Per-object default rung (ADR 0003): voice/Auto-Run/CLI fall back to this
    # when no explicit mode is given; explicit CLI flags override it.
    run_cfg["default_mode"] = str(obj_cfg.get("mode") or "vla").strip().lower()

    lstm_cfg = dict(obj_cfg.get("lstm") or {})
    manual_release_position = _finger_position(
        obj_cfg.get("manual_release_position"),
        f"vla_auto.objects.{object_key}.manual_release_position",
        required=False,
    )
    release_policy = {
        "release_min_excursion_ticks": obj_cfg.get(
            "release_min_excursion_ticks",
            auto_cfg.get("release_min_excursion_ticks", 60),
        ),
        "release_min_excursion_fingers": obj_cfg.get(
            "release_min_excursion_fingers",
            auto_cfg.get("release_min_excursion_fingers", 2),
        ),
        "release_min_tracking_fingers": obj_cfg.get(
            "release_min_tracking_fingers",
            auto_cfg.get("release_min_tracking_fingers", 3),
        ),
    }
    release_min_excursion_ticks = _bounded_int(
        release_policy,
        "release_min_excursion_ticks",
        "vla_auto.release_policy",
        0,
        2400,
    )
    release_min_excursion_fingers = _bounded_int(
        release_policy,
        "release_min_excursion_fingers",
        "vla_auto.release_policy",
        1,
        3,
    )
    release_min_tracking_fingers = _bounded_int(
        release_policy,
        "release_min_tracking_fingers",
        "vla_auto.release_policy",
        1,
        3,
    )
    tail_cfg = {
        "grasp_z_mm": fixed_obj["grasp_z_mm"],
        "grasp_z_source": "fixed_fallback.grasp_z_mm (rig constant)",
        "grasp_backoff_mm": obj_cfg.get("grasp_backoff_mm", 0.0),
        "grasp_contact_dwell_s": fixed_obj["contact_dwell_s"],
        "auto_lift_after_dwell": not fixed_obj["requires_grasp"],
        "lift_during_grasp": fixed_obj["lift_during_grasp"],
        "grasp_start_z_mm": fixed_obj["grasp_start_z_mm"],
        "place_approach_pose": fixed_obj["place_approach_pose"],
        "place_z_mm": fixed_obj["release_z_mm"],
        "manual_release_position": manual_release_position,
        "release_min_excursion_ticks": release_min_excursion_ticks,
        "release_min_excursion_fingers": release_min_excursion_fingers,
        "release_min_tracking_fingers": release_min_tracking_fingers,
        "lstm": {
            "object": object_key,
            "grasp_action": lstm_cfg.get("grasp"),
            "release_action": lstm_cfg.get("release"),
        } if lstm_cfg else {},
    }
    return run_cfg, tail_cfg, obj_cfg


def module_enabled(demo_cfg: dict[str, Any], module_name: str, default: bool = False) -> bool:
    node = demo_cfg.get("modules", {}).get(module_name)
    if isinstance(node, dict):
        return bool(node.get("enabled", default))
    if node is None:
        return default
    return bool(node)


def module_visible(demo_cfg: dict[str, Any], module_name: str, default: bool = False) -> bool:
    node = demo_cfg.get("modules", {}).get(module_name)
    if isinstance(node, dict):
        return bool(node.get("show_in_ui", default))
    if node is None:
        return default
    return bool(node)


def public_runtime_config(bundle: RuntimeBundle) -> dict[str, Any]:
    demo_cfg = bundle.demo_config
    return {
        "profile": bundle.profile_name,
        "env_overlay_path": bundle.env_overlay_path,
        "ui": {
            "mode": demo_cfg.get("ui", {}).get("mode", "modern"),
            "available_modes": ["modern", "classic"],
            "controls": copy.deepcopy(
                demo_cfg.get("ui", {}).get("controls", {})
            ),
        },
        "modules": copy.deepcopy(demo_cfg.get("modules", {})),
        "safety_boundary": copy.deepcopy(demo_cfg.get("safety_boundary", {})),
        "pose_limits": copy.deepcopy(demo_cfg.get("pose_limits", {})),
        "speed": copy.deepcopy(demo_cfg.get("speed", {})),
        "home_pose": copy.deepcopy(demo_cfg.get("home_pose", [])),
        "ready_pose": copy.deepcopy(demo_cfg.get("ready_pose", [])),
        "ready_motion": copy.deepcopy(demo_cfg.get("ready_motion", {})),
        "home_routing": copy.deepcopy(demo_cfg.get("home_routing", {})),
        "virtual_env": copy.deepcopy(demo_cfg.get("virtual_env", {})),
        "trajectory": copy.deepcopy(demo_cfg.get("trajectory", {})),
        "teach": {
            "default_replay_mode": demo_cfg.get("teach", {}).get("default_replay_mode", "raw"),
            "dataset_capture": copy.deepcopy(demo_cfg.get("teach", {}).get("dataset_capture", {})),
        },
        "cameras": {
            "cam1": copy.deepcopy(demo_cfg.get("cameras", {}).get("cam1", {})),
            "cam2": copy.deepcopy(demo_cfg.get("cameras", {}).get("cam2", {})),
            "claw": copy.deepcopy(demo_cfg.get("cameras", {}).get("claw", {})),
        },
        "gripper": {
            "enabled": bool(demo_cfg.get("gripper", {}).get("enabled", False)),
            "position_move_mode": demo_cfg.get("gripper", {}).get("position_move_mode", "direct"),
            "endpoints": copy.deepcopy(demo_cfg.get("gripper", {}).get("endpoints", [])),
        },
        "sensor_api": copy.deepcopy(demo_cfg.get("sensor_api", {})),
        "arm": {
            "connections": copy.deepcopy(demo_cfg.get("arm", {}).get("connections", [])),
            "unit_id": demo_cfg.get("arm", {}).get("unit_id", 2),
            "pose_poll_hz": demo_cfg.get("arm", {}).get("pose_poll_hz", 4),
        },
        "profiles": {
            name: {"description": str(node.get("description", ""))}
            for name, node in bundle.launch_profiles.get("profiles", {}).items()
        },
    }


def write_launcher_state(payload: dict[str, Any]) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LAUNCHER_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_launcher_state() -> dict[str, Any]:
    if not LAUNCHER_STATE_PATH.exists():
        return {}
    with open(LAUNCHER_STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
