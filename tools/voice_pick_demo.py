"""
Voice Pick Demo Server.

Ubuntu-oriented main control backend:
- Flask + Socket.IO UI backend
- RealSense wall
- Claw cam low-latency UVC wall
- Arm / gripper / dataset capture orchestration
- Teach / replay / validation workflows
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, send_from_directory
from flask_socketio import SocketIO, emit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nlu import IntentParser, DEMO_OBJECTS, DEMO_OBJECT_LABELS
from src.runtime_config import (
    FixedFallbackConfigError,
    build_auto_run_cfg,
    build_fixed_fallback_cfg,
    load_runtime_bundle,
    module_enabled,
    public_runtime_config,
    read_launcher_state,
)
from src.services.arm_service import ArmService
from src.services.claw_camera_service import ClawCameraService
from src.services.gripper_service import GripperService
from src.services.health_service import HealthService
from src.services.realsense_service import RealSenseService
from src.services.recording_service import DatasetCaptureManager, TeachManager, ValidationSessionRecorder
from src.services.vla_service import VLAService
from src.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("voice_pick_demo")


def _classic_ui_available(static_dir: Path) -> bool:
    return (static_dir / "index.classic.html").exists()


def _classic_dataset_panel_html() -> str:
    return """
        <section class="panel dashboard-panel" id="panel-dataset-capture"
            data-module="dataset_capture"
            data-panel-id="dataset-capture" data-default-x="14" data-default-y="11" data-default-w="4" data-default-h="3"
            data-min-w="4" data-min-h="3">
            <div class="panel-header">
                <span class="panel-title">Dataset Capture</span>
                <span class="panel-badge" id="dataset-capture-badge">Idle</span>
            </div>
            <div class="panel-content panel-scroll">
                <div class="validation-controls">
                    <select id="dataset-object-select">
                        <option value="">-- Select Object --</option>
                    </select>
                    <button class="btn-validate" id="btn-dataset-start">Start</button>
                    <button class="btn-validate fail" id="btn-dataset-stop" disabled>Stop</button>
                </div>
                <div class="teach-controls" style="margin-top:8px;">
                    <input type="number" id="dataset-fps" value="6" min="1" max="30" title="Image FPS">
                    <input type="number" id="dataset-telemetry-hz" value="10" min="1" max="60" title="Telemetry Hz">
                    <input type="number" id="dataset-duration-s" value="0" min="0" step="1" title="Duration seconds, 0=manual stop">
                </div>
                <div class="teach-replay" style="margin-top:8px; align-items:flex-start;">
                    <label><input type="checkbox" id="dataset-stream-cam1-rgb" checked> cam1_rgb</label>
                    <label><input type="checkbox" id="dataset-stream-cam2-rgb" checked> cam2_rgb</label>
                    <label><input type="checkbox" id="dataset-stream-cam1-depth"> cam1_depth</label>
                    <label><input type="checkbox" id="dataset-stream-cam2-depth"> cam2_depth</label>
                    <label><input type="checkbox" id="dataset-stream-claw-rgb"> claw_rgb</label>
                </div>
                <div class="teach-controls" style="margin-top:8px;">
                    <input type="text" id="dataset-notes" placeholder="Notes..." value="">
                </div>
                <div style="margin-top:10px;">
                    <div style="height:8px; background:#1a2332; border-radius:999px; overflow:hidden;">
                        <div id="dataset-capture-progress-fill" style="height:100%; width:0%; background:linear-gradient(90deg,#5b8dff,#45f2a1); transition:width 160ms ease;"></div>
                    </div>
                    <div id="dataset-capture-progress-label" style="margin-top:6px; font-size:12px; color:#8b9bb4;">Idle</div>
                </div>
                <div class="teach-waypoints" id="dataset-capture-summary">No dataset capture running</div>
            </div>
        </section>
    """


def _serve_classic_ui(static_dir: Path) -> Response:
    html = (static_dir / "index.classic.html").read_text(encoding="utf-8")
    html = html.replace('href="style.css"', 'href="/classic-assets/style.css"')
    html = html.replace('src="app.js"', 'src="/classic-assets/app.js"')
    html = html.replace("href='style.css'", "href='/classic-assets/style.css'")
    html = html.replace("src='app.js'", "src='/classic-assets/app.js'")
    if 'id="panel-dataset-capture"' not in html:
        html = html.replace("</div>\n</main>", _classic_dataset_panel_html() + "\n    </div>\n</main>")
    if 'src="/classic-assets/classic_dataset_compat.js"' not in html:
        html = html.replace("</body>", '    <script src="/classic-assets/classic_dataset_compat.js"></script>\n</body>')
    if 'src="/classic-assets/vla_compact.js"' not in html:
        html = html.replace("</body>", '    <script src="/classic-assets/vla_compact.js"></script>\n</body>')
    return Response(html, mimetype="text/html")


def _sensor_payload_from_gripper_status(status: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    state = status.get("state", {}) if isinstance(status, dict) else {}
    tactile = state.get("tactile_data")
    if not isinstance(tactile, list):
        tactile = state.get("tactile")
    if not isinstance(tactile, list):
        tactile = []
    values = [tactile[i] if i < len(tactile) else None for i in range(3)]
    connected = bool(state.get("sensor_connected"))
    sample_valid = bool(state.get("sample_valid"))
    error = status.get("last_error") or state.get("last_error") or ""
    channels = []
    for value in values:
        channels.append({
            "ok": bool(sample_valid and connected and value is not None),
            "last_error": None if sample_valid and connected else (error or "sensor_disconnected"),
        })
    sample_payload = {
        "analog_value1": values[0],
        "analog_value2": values[1],
        "analog_value3": values[2],
        "timestamp_unix": state.get("tactile_timestamp_unix"),
        "channels": channels,
        "sensor_connected": connected,
        "sample_valid": sample_valid,
    }
    state_payload = {
        "connected": connected,
        "sensor_connected": connected,
        "sample_valid": sample_valid,
        "values": values,
        "timestamp_unix": state.get("tactile_timestamp_unix"),
        "api_version": state.get("api_version") or status.get("api_version"),
        "error": error or None,
        "last_error": error or None,
        "status": "connected" if connected else "disconnected",
        "last_sample": sample_payload,
    }
    data_payload = {
        "sample": sample_payload,
        "history": [],
        "analog_value1": values[0],
        "analog_value2": values[1],
        "analog_value3": values[2],
        "connected": connected,
        "sensor_connected": connected,
        "sample_valid": sample_valid,
        "timestamp_unix": state.get("tactile_timestamp_unix"),
        "last_error": error or None,
        "raw": state,
    }
    return state_payload, data_payload


def _virtual_env_sync_payload(demo_cfg: dict[str, Any]) -> dict[str, Any]:
    virtual_cfg = demo_cfg.get("virtual_env", {})
    enabled = bool(virtual_cfg.get("enabled", False))
    arm_cfg = virtual_cfg.get("arm_sync", {}) if isinstance(virtual_cfg.get("arm_sync"), dict) else {}
    grip_cfg = virtual_cfg.get("gripper_sync", {}) if isinstance(virtual_cfg.get("gripper_sync"), dict) else {}
    return {
        "enabled": enabled,
        "connected": False,
        "last_error": None if enabled else "virtual env disabled",
        "arm": {
            "enabled": bool(arm_cfg.get("enabled", enabled)),
            "mode": arm_cfg.get("mode", "direct_ee_move"),
            "ok_count": 0,
            "fail_count": 0,
            "calibration": arm_cfg.get("calibration", {}),
        },
        "gripper": {
            "enabled": bool(grip_cfg.get("enabled", enabled)),
            "ok_count": 0,
            "fail_count": 0,
        },
    }


def _camera_status_payload(realsense: RealSenseService, claw: ClawCameraService) -> dict[str, Any]:
    payload = realsense.status()
    claw_status = claw.status()
    payload["claw"] = {
        **claw_status,
        "depth_enabled": False,
        "yolo_ready": bool(claw_status.get("yolo_enabled")) and not claw_status.get("last_error"),
        "last_labels": [str(d.get("label", "")) for d in claw_status.get("detections", []) if d.get("label")],
    }
    return payload


def _objects_catalog_payload(objects_cfg: dict[str, Any], validation: ValidationSessionRecorder) -> dict[str, Any]:
    classes = objects_cfg.get("classes", {})
    result = {}
    for key, obj in classes.items():
        result[key] = {
            "chinese": obj.get("chinese", []),
            "english": obj.get("english", []),
            "has_fixed_poses": "fixed_poses" in obj,
            "slot_id": obj.get("slot_id", ""),
            "table_set_id": obj.get("table_set_id", ""),
            "default_teach_recording": validation.default_teach_recording(key),
        }
    return result


def _demo_available_labels() -> list[dict]:
    """The three voice-reachable objects, for the 'only these 3' operator hint."""
    return [{"key": k, "zh": DEMO_OBJECT_LABELS[k]["zh"], "en": DEMO_OBJECT_LABELS[k]["en"]}
            for k in DEMO_OBJECTS]


def _demo_scope_message(status: str) -> str:
    """Feedback when voice input isn't one of the three (ADR 0003)."""
    zh = " / ".join(DEMO_OBJECT_LABELS[k]["zh"] for k in DEMO_OBJECTS)
    en = " / ".join(DEMO_OBJECT_LABELS[k]["en"] for k in DEMO_OBJECTS)
    if status == "ambiguous":
        return f"一次只能拿一個：{zh}（one at a time: {en}）"
    return f"只支援 {zh}（only {en}）"


def create_app(profile: str, env_config: str | None, ui_mode: str | None = None) -> tuple[Flask, SocketIO]:
    bundle = load_runtime_bundle(profile=profile, env_config=env_config)
    demo_cfg = bundle.demo_config
    modbus_cfg = bundle.modbus_config
    static_dir = Path(__file__).parent / "static"
    effective_ui_mode = str(ui_mode or demo_cfg.get("ui", {}).get("mode", "modern")).strip().lower()
    if effective_ui_mode not in {"modern", "classic"}:
        effective_ui_mode = "modern"

    app = Flask(__name__, static_folder=str(static_dir))
    app.config["SECRET_KEY"] = "voice-pick-demo"
    app.config["VOICE_PICK_UI_MODE"] = effective_ui_mode
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

    objects_cfg = load_config("objects.yaml")
    nlu = IntentParser()
    latest_speech: dict[str, Any] = {}

    realsense = RealSenseService(demo_cfg)
    claw = ClawCameraService(demo_cfg)
    validation = ValidationSessionRecorder(socketio, objects_cfg)
    gripper = GripperService(demo_cfg, socketio=socketio)
    arm = ArmService(demo_cfg, modbus_cfg, socketio=socketio, gripper_service=gripper, validation_recorder=validation)
    teach = TeachManager(demo_cfg, arm, gripper, socketio)
    dataset_capture = DatasetCaptureManager(demo_cfg, socketio, arm, gripper, claw, realsense)
    vla = VLAService(socketio=socketio, arm_service=arm, realsense_service=realsense, claw_service=claw)
    # Pre-warm the VLA checkpoint in the background so the first CLI/UI auto-run
    # skips the cold RDT+SigLIP load (~6 s). Server boot is NOT blocked (daemon
    # thread); disable with vla_auto.preload_on_startup: false.
    _vla_auto_cfg = demo_cfg.get("vla_auto", {}) or {}
    if _vla_auto_cfg.get("preload_on_startup", True) and _vla_auto_cfg.get("checkpoint"):
        threading.Thread(target=vla.warm, args=(_vla_auto_cfg["checkpoint"],),
                         name="vla-warm", daemon=True).start()
    health = HealthService(demo_cfg, bundle.profile_name, realsense, claw, arm, gripper, dataset_capture)
    sensor_cfg = demo_cfg.get("sensor_api", {})
    sensor_enabled = module_enabled(demo_cfg, "sensor_chart", default=True) and bool(sensor_cfg.get("enabled", True))
    sensor_history = deque(maxlen=max(int(sensor_cfg.get("history_points", 180)), 20))
    virtual_env_sync = _virtual_env_sync_payload(demo_cfg)

    def emit_config_payload(target_emit=None):
        fresh_bundle = load_runtime_bundle(profile=profile, env_config=env_config)
        payload = {
            "ok": True,
            "message": "Runtime config reloaded",
            "notes": [],
            "config": public_runtime_config(fresh_bundle),
        }
        if target_emit is not None:
            target_emit("config_reloaded", payload)
        else:
            socketio.emit("config_reloaded", payload)
        return payload

    if demo_cfg.get("modules", {}).get("cam1", {}).get("enabled", True) or demo_cfg.get("modules", {}).get("cam2", {}).get("enabled", True):
        realsense.start()
    if demo_cfg.get("modules", {}).get("claw_cam", {}).get("enabled", True):
        claw.start()
    if demo_cfg.get("modules", {}).get("gripper", {}).get("enabled", True):
        gripper.start_monitoring()
    arm.start_pose_polling()

    def _status_broadcaster():
        while True:
            try:
                gripper_status = gripper.status()
                sensor_state, sensor_data = _sensor_payload_from_gripper_status(gripper_status)
                sensor_state["enabled"] = sensor_enabled
                sensor_data["enabled"] = sensor_enabled
                if sensor_data.get("sample") is not None:
                    sensor_history.append(dict(sensor_data["sample"]))
                sensor_data["history"] = list(sensor_history)
                socketio.emit("camera_status", _camera_status_payload(realsense, claw))
                socketio.emit("claw_status", claw.status())
                socketio.emit("gripper_state", gripper_status)
                socketio.emit("sensor_state", sensor_state)
                socketio.emit("sensor_data", sensor_data)
                socketio.emit("virtual_env_sync", virtual_env_sync)
                socketio.emit("dataset_capture_state", dataset_capture.dataset_capture_status())
                socketio.emit("system_health", health.snapshot())
            except Exception:
                pass
            time.sleep(1.0)

    threading.Thread(target=_status_broadcaster, daemon=True).start()

    @app.route("/")
    def index():
        if app.config.get("VOICE_PICK_UI_MODE") == "classic":
            if _classic_ui_available(static_dir):
                return _serve_classic_ui(static_dir)
            return send_from_directory(static_dir, "legacy.html")
        return send_from_directory(static_dir, "index.html")

    @app.route("/modern")
    def modern_index():
        return send_from_directory(static_dir, "index.html")

    @app.route("/legacy")
    def legacy_index():
        return send_from_directory(static_dir, "legacy.html")

    @app.route("/classic")
    def classic_index():
        if _classic_ui_available(static_dir):
            return _serve_classic_ui(static_dir)
        return send_from_directory(static_dir, "legacy.html")

    @app.route("/classic-assets/<path:filename>")
    def classic_assets(filename: str):
        mapping = {
            "app.js": "app.classic.js",
            "style.css": "style.classic.css",
            "index.html": "index.classic.html",
        }
        target = mapping.get(filename, filename)
        return send_from_directory(static_dir, target)

    @app.route("/<path:filename>")
    def static_files(filename):
        return send_from_directory(static_dir, filename)

    def _mjpeg_gen(fn):
        while True:
            frame = fn()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(1.0 / 15)

    @app.route("/stream/<cam_key>/<stream_type>")
    def stream_camera(cam_key: str, stream_type: str):
        if cam_key in ("cam1", "cam2") and stream_type in ("rgb", "depth"):
            return Response(
                _mjpeg_gen(lambda: realsense.get_jpeg(cam_key, stream_type)),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )
        if cam_key == "claw" and stream_type == "rgb":
            return Response(
                _mjpeg_gen(claw.get_jpeg),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )
        return "Not found", 404

    @app.route("/api/objects")
    def api_objects():
        return jsonify(_objects_catalog_payload(objects_cfg, validation))

    @app.route("/api/recordings")
    def api_recordings():
        return jsonify(teach.list_recordings())

    @app.route("/api/teach_recordings")
    def api_teach_recordings():
        return jsonify(teach.list_recordings())

    @app.route("/api/config")
    def api_config():
        return jsonify(public_runtime_config(bundle))

    @app.route("/config")
    def api_config_alias():
        return jsonify(public_runtime_config(bundle))

    @app.route("/reload_config", methods=["GET", "POST"])
    def api_reload_config_alias():
        return jsonify(emit_config_payload())

    @app.route("/reload-config", methods=["GET", "POST"])
    def api_reload_config_hyphen_alias():
        return jsonify(emit_config_payload())

    @app.route("/api/reload_config", methods=["GET", "POST"])
    def api_reload_config():
        return jsonify(emit_config_payload())

    @app.route("/api/reload-config", methods=["GET", "POST"])
    def api_reload_config_hyphen():
        return jsonify(emit_config_payload())

    @app.route("/api/config/reload", methods=["GET", "POST"])
    def api_config_reload():
        return jsonify(emit_config_payload())

    @app.route("/api/health")
    def api_health():
        return jsonify(health.snapshot())

    @app.route("/api/asr/transcribe", methods=["POST"])
    def api_asr_transcribe():
        return jsonify({"ok": False, "error": "offline ASR endpoint not enabled in this bundle"}), 501

    @app.route("/api/cameras/<cam_key>/restart", methods=["POST"])
    def api_restart_camera(cam_key: str):
        try:
            if cam_key in ("cam1", "cam2"):
                realsense.stop()
                realsense.start()
                return jsonify({"ok": True, "message": f"Restarted RealSense pipeline for {cam_key}"})
            if cam_key == "claw":
                claw.stop()
                claw.start()
                return jsonify({"ok": True, "message": "Restarted claw camera"})
            return jsonify({"ok": False, "error": f"Unknown camera: {cam_key}"}), 404
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.route("/api/panel_recordings/start", methods=["POST"])
    def api_panel_recording_start():
        return jsonify({"ok": False, "error": "panel recording backend is not enabled in this bundle"}), 501

    @app.route("/api/panel_recordings/upload", methods=["POST"])
    def api_panel_recording_upload():
        return jsonify({"ok": False, "error": "panel recording backend is not enabled in this bundle"}), 501

    @app.route("/api/panel_recordings/frame", methods=["POST"])
    def api_panel_recording_frame():
        return jsonify({"ok": False, "error": "panel recording backend is not enabled in this bundle"}), 501

    @app.route("/api/panel_recordings/finish", methods=["POST"])
    def api_panel_recording_finish():
        return jsonify({"ok": False, "error": "panel recording backend is not enabled in this bundle"}), 501

    @app.route("/api/virtual_env/calibrate_current", methods=["POST"])
    def api_virtual_env_calibrate_current():
        return jsonify({"ok": False, "error": "virtual env calibration is not enabled in this bundle"}), 501

    @app.route("/api/virtual_env/reset_calibration", methods=["POST"])
    def api_virtual_env_reset_calibration():
        return jsonify({"ok": False, "error": "virtual env calibration is not enabled in this bundle"}), 501

    @app.route("/api/launcher/status")
    def api_launcher_status():
        return jsonify(read_launcher_state())

    @app.route("/api/profiles")
    def api_profiles():
        return jsonify(public_runtime_config(bundle).get("profiles", {}))

    @socketio.on("connect")
    def on_connect():
        log.info("Client connected")
        emit("arm_status", {"connected": arm.connected, "label": arm.connection_label, "error": None})
        emit("camera_status", _camera_status_payload(realsense, claw))
        emit("claw_status", claw.status())
        emit("teach_list", teach.list_recordings())
        emit("validation_state", validation.state_payload())
        gripper_status = gripper.status()
        sensor_state, sensor_data = _sensor_payload_from_gripper_status(gripper_status)
        sensor_state["enabled"] = sensor_enabled
        sensor_data["enabled"] = sensor_enabled
        if sensor_data.get("sample") is not None and not sensor_history:
            sensor_history.append(dict(sensor_data["sample"]))
        sensor_data["history"] = list(sensor_history)
        emit("gripper_state", gripper_status)
        emit("sensor_state", sensor_state)
        emit("sensor_data", sensor_data)
        emit("virtual_env_sync", virtual_env_sync)
        emit("objects_catalog", _objects_catalog_payload(objects_cfg, validation))
        emit("dataset_capture_state", dataset_capture.dataset_capture_status())
        emit("vla_status", vla.vla_status_payload())
        emit("system_health", health.snapshot())

    @socketio.on("voice_text")
    def on_voice_text(data):
        text = data.get("text", "").strip()
        lang = data.get("lang", "zh-TW")
        # Browser speech is open-vocabulary and ranks by a general language
        # model, so the word the operator actually said is often not
        # alternative 0 (a bare "knife" has been seen returning "OK Google").
        # The client forwards every alternative; the NLU picks among them.
        raw_alts = data.get("candidates") or []
        alts = [a.strip() for a in raw_alts if isinstance(a, str) and a.strip()]
        if text and text not in alts:
            alts.insert(0, text)
        if not alts:
            return
        text = alts[0]
        # ADR 0003: voice drives the three-object auto-run. Resolve to one of the
        # three (every knife-word → butter_knife); anything else is out of scope
        # and gets no Confirm. The Confirm click then runs LIVE (start_auto).
        res = nlu.resolve_demo_from_candidates(alts)
        obj = res.object_key
        # The candidate that actually won — not necessarily alt0, which is only
        # the recognizer's loudest guess and is often the wrong homophone. The UI
        # displays this so the operator sees what was understood, not the noise.
        winning_text = alts[res.candidate_index] if 0 <= res.candidate_index < len(alts) else ""
        if len(alts) > 1 or res.fuzzy:
            picked = ("alt%d" % res.candidate_index) if res.candidate_index >= 0 else "none"
            arm._emit_log(
                "INFO",
                "ASR alternatives %s → %s (%s%s)"
                % (alts, obj or res.status, picked, ", homophone" if res.fuzzy else ""),
            )
        matched = res.status == "matched"
        obj_mode = None
        if obj:
            obj_cfg = ((demo_cfg.get("vla_auto", {}).get("objects") or {}).get(obj) or {})
            obj_mode = str(obj_cfg.get("mode") or "vla").strip().lower()
        payload = {
            "intent": "pick",
            "object": obj,
            "status": res.status,                 # matched|out_of_scope|ambiguous|empty
            "in_scope": matched,
            "mode": obj_mode,                     # rung the auto-run will use
            "run_live": True,                     # voice always runs live (ADR 0003)
            "matched_phrase": res.matched_phrase,
            "fuzzy": bool(res.fuzzy),             # matched only after homophone tolerance
            "heard": alts,                        # every ASR alternative, ranked
            "heard_top": alts[0],                 # the loudest guess (may be wrong)
            "matched_text": winning_text,         # the candidate that actually resolved
            "available": _demo_available_labels(),
            "raw_text": text,
            # legacy fields the NLU card still reads:
            "confidence": (0.7 if res.fuzzy else 1.0) if matched else 0.0,
            "candidates": [obj] if matched else [],
            "need_confirm": matched,              # Confirm offered only for the three
            "disambiguation": "" if matched else _demo_scope_message(res.status),
        }
        latest_speech.clear()
        latest_speech.update({"requested_text": text, "lang": lang, "nlu_result": payload})
        emit("nlu_result", payload)

    @socketio.on("confirm_pick")
    def on_confirm_pick(data):
        obj_key = data.get("object_key")
        method = data.get("method", "fixed")
        requested_text = data.get("requested_text", "").strip() or latest_speech.get("requested_text", "") or obj_key
        recording_name = data.get("recording_name", "")
        source = data.get("source", "ui")
        speed_preset = data.get("speed_preset")
        custom_speed = data.get("custom_speed")
        replay_mode = data.get("replay_mode")

        if not obj_key:
            emit("arm_log", {"level": "ERROR", "message": "No object specified", "timestamp": time.strftime("%H:%M:%S")})
            return
        obj_def = objects_cfg.get("classes", {}).get(obj_key, {})
        if not obj_def:
            emit("arm_log", {"level": "ERROR", "message": f"Unknown object: {obj_key}", "timestamp": time.strftime("%H:%M:%S")})
            return
        if method == "teach" and not recording_name:
            recording_name = validation.default_teach_recording(obj_key)

        ok, info = validation.start_session(
            object_key=obj_key,
            obj_def=obj_def,
            mode=method,
            requested_text=requested_text,
            resolved_object_key=obj_key,
            teach_recording_name=recording_name if method == "teach" else "",
        )
        if not ok:
            emit("arm_log", {"level": "WARN", "message": info, "timestamp": time.strftime("%H:%M:%S")})
            emit("validation_state", validation.state_payload())
            return

        if source == "voice" and latest_speech:
            validation.log_speech("voice_text", {"requested_text": latest_speech.get("requested_text", ""), "lang": latest_speech.get("lang", "")})
            validation.log_speech("nlu_result", latest_speech.get("nlu_result", {}))
        else:
            validation.log_speech("ui_request", {"requested_text": requested_text, "resolved_object_key": obj_key})

        def _run():
            if method == "teach":
                rec = teach.load_recording(recording_name)
                if rec is None:
                    arm._emit_log("ERROR", f"Recording not found: {recording_name}")
                    return
                arm.replay_recording(rec, replay_mode=replay_mode)
            else:
                arm.pick_fixed(obj_key, obj_def, speed_preset=speed_preset, custom_speed=custom_speed)

        threading.Thread(target=_run, daemon=True).start()

    @socketio.on("arm_connect")
    def on_arm_connect():
        threading.Thread(target=arm.auto_connect, daemon=True).start()

    @socketio.on("arm_home")
    def on_arm_home():
        threading.Thread(target=arm.go_home, daemon=True).start()

    @socketio.on("arm_ready")
    def on_arm_ready():
        def _ready_and_home_fingers():
            arm.go_ready()
            # Ready button also opens the fingers to home (operator request):
            # same HOME_POS [3072,3072,2048] the auto-run returns to. Harmless
            # no-op / 409 if an LSTM finger action is mid-run (stop the run first).
            grip = getattr(arm, "gripper", None)
            if grip is not None:
                grip.set_position([3072, 3072, 2048], mode="stepped")
        threading.Thread(target=_ready_and_home_fingers, daemon=True).start()

    @socketio.on("arm_stop")
    def on_arm_stop():
        vla.stop()
        threading.Thread(target=arm.motion_stop, daemon=True).start()

    @socketio.on("gripper_reboot")
    def on_gripper_reboot(data=None):
        # Manual recovery for a Dynamixel overload torque-off latch (a finger
        # reads fine but won't drive to goal). Reboots the motors + re-enables
        # torque + homes them open — no full AGX service restart / model reload.
        def _reboot():
            home = bool(data.get("home", True)) if isinstance(data, dict) else True
            res = gripper.reboot_motors(home=home)
            print(f"[server] gripper reboot -> {res}")
            socketio.emit("gripper_state", gripper.status())
        threading.Thread(target=_reboot, daemon=True).start()

    @socketio.on("arm_reset_alarms")
    def on_arm_reset_alarms():
        def _run():
            if not arm.ensure_connected():
                return
            try:
                arm.ctrl.reset_alarms()
                arm._emit_log("STEP", "Arm alarms reset")
            except Exception as exc:
                arm._emit_log("ERROR", f"Reset alarms failed: {exc}")

        threading.Thread(target=_run, daemon=True).start()

    @socketio.on("arm_emergency_stop")
    def on_arm_emergency_stop():
        vla.stop()
        threading.Thread(target=arm.motion_stop, daemon=True).start()

    @socketio.on("fallback_run")
    def on_fallback_run(data):
        obj_key = str((data or {}).get("object_key", "")).strip()
        try:
            fresh = load_runtime_bundle(
                profile=profile, env_config=env_config
            )
            arm.refresh_runtime_config(fresh.demo_config, fresh.modbus_config)
            run_cfg, tail_cfg = build_fixed_fallback_cfg(
                fresh.demo_config, obj_key
            )
        except FixedFallbackConfigError as exc:
            emit("vla_log", {
                "level": "ERROR",
                "message": f"[fixed] preflight rejected: {exc}",
                "timestamp": time.strftime("%H:%M:%S"),
            })
            return
        if gripper.resolve_endpoint() is None:
            emit("vla_log", {
                "level": "ERROR",
                "message": (
                    "[fixed] preflight rejected: AGX service unreachable "
                    f"({gripper.last_error or 'no endpoint'})"
                ),
                "timestamp": time.strftime("%H:%M:%S"),
            })
            return
        state = gripper.get_state() or {}
        current_pos = state.get("current_pos")
        if not (
            isinstance(current_pos, list)
            and len(current_pos) == 3
            and any(current_pos)
        ):
            emit("vla_log", {
                "level": "ERROR",
                "message": (
                    "[fixed] preflight rejected: "
                    "AGX motor position unavailable"
                ),
                "timestamp": time.strftime("%H:%M:%S"),
            })
            return
        emit("vla_log", {
            "level": "STEP",
            "message": f"[fixed] deterministic pick->place requested: {obj_key}",
            "timestamp": time.strftime("%H:%M:%S"),
        })
        vla.start_auto(
            object_key=obj_key,
            checkpoint_path="",
            embed_path="",
            instruction=f"fixed fallback {obj_key}",
            mode="fixed",
            run_cfg=run_cfg,
            tail_cfg=tail_cfg,
            dry_run=False,
        )
        emit("vla_status", vla.vla_status_payload())

    @socketio.on("teach_start")
    def on_teach_start(data):
        teach.start(data.get("name", f"recording_{int(time.time())}"))

    @socketio.on("teach_waypoint")
    def on_teach_waypoint(data):
        teach.save_waypoint(data.get("gripper", "none"), int(data.get("speed", 30)))

    @socketio.on("teach_stop")
    def on_teach_stop():
        teach.stop()
        emit("teach_list", teach.list_recordings())

    @socketio.on("teach_replay")
    def on_teach_replay(data):
        name = data.get("name", "")
        replay_mode = data.get("replay_mode")
        rec = teach.load_recording(name)
        if rec is None:
            emit("arm_log", {"level": "ERROR", "message": f"Recording not found: {name}", "timestamp": time.strftime("%H:%M:%S")})
            return
        threading.Thread(target=arm.replay_recording, args=(rec, replay_mode), daemon=True).start()

    @socketio.on("teach_regenerate_phase")
    def on_teach_regenerate_phase(data):
        name = data.get("name", "")
        emit("arm_log", {"level": "WARN", "message": f"Phase regeneration is not implemented in this bundle for {name}", "timestamp": time.strftime("%H:%M:%S")})

    @socketio.on("validation_mark")
    def on_validation_mark(data):
        ok, info = validation.mark_result(data.get("result", ""))
        emit("arm_log", {"level": "STEP" if ok else "WARN", "message": f"Validation mark: {info}", "timestamp": time.strftime("%H:%M:%S")})
        emit("validation_state", validation.state_payload())

    @socketio.on("dataset_capture_start")
    def on_dataset_capture_start(data):
        ok, info = dataset_capture.start(data or {})
        emit("arm_log", {"level": "STEP" if ok else "WARN", "message": info, "timestamp": time.strftime("%H:%M:%S")})
        emit("dataset_capture_state", dataset_capture.dataset_capture_status())

    @socketio.on("dataset_capture_stop")
    def on_dataset_capture_stop():
        ok, info = dataset_capture.stop()
        emit("arm_log", {"level": "STEP" if ok else "WARN", "message": info, "timestamp": time.strftime("%H:%M:%S")})
        emit("dataset_capture_state", dataset_capture.dataset_capture_status())

    @socketio.on("vla_start")
    def on_vla_start(data):
        d = data or {}
        vla.start(
            checkpoint_path=str(d.get("checkpoint", "")),
            embed_path=str(d.get("embed_path", "")),
            instruction=str(d.get("instruction", "拿起梯形")),
            max_steps=int(d.get("max_steps", 200)),
            exec_steps=int(d.get("exec_steps", 8)),
            dry_run=bool(d.get("dry_run", True)),
        )
        emit("vla_status", vla.vla_status_payload())

    @socketio.on("vla_stop")
    def on_vla_stop():
        vla.stop()
        threading.Thread(target=arm.motion_stop, daemon=True).start()
        emit("vla_status", vla.vla_status_payload())

    @socketio.on("vla_accept_grasp")
    def on_vla_accept_grasp():
        accepted = vla.accept_grasp()
        emit("vla_status", vla.vla_status_payload())
        return {"ok": accepted}

    @socketio.on("vla_complete_finger_stage")
    def on_vla_complete_finger_stage():
        accepted = vla.complete_finger_stage()
        emit("vla_status", vla.vla_status_payload())
        return {"ok": accepted}

    @socketio.on("vla_auto_start")
    def on_vla_auto_start(data):
        """Three-object demo auto-run (§3): object key + mode is all the UI sends;
        checkpoint/embed/speeds come from demo_config vla_auto, while route
        geometry comes from the authoritative fixed_fallback table."""
        d = data or {}
        obj_key = str(d.get("object", "")).strip()
        fresh = load_runtime_bundle(profile=profile, env_config=env_config)
        try:
            arm.refresh_runtime_config(fresh.demo_config, fresh.modbus_config)
            resolved = build_auto_run_cfg(
                fresh.demo_config, objects_cfg, obj_key
            )
        except FixedFallbackConfigError as exc:
            emit("vla_log", {
                "level": "ERROR",
                "timestamp": time.strftime("%H:%M:%S"),
                "message": f"vla_auto geometry rejected: {exc}",
            })
            return
        if resolved is None:
            emit("vla_log", {"level": "ERROR", "timestamp": time.strftime("%H:%M:%S"),
                             "message": f"vla_auto: unknown object '{obj_key or '(none)'}'"})
            return
        run_cfg, tail_cfg, obj_cfg = resolved
        auto_cfg = dict(fresh.demo_config.get("vla_auto", {}))

        # Optional overrides (e.g. from auto_run_cli.py --attach): let the
        # terminal driver tweak the run without editing config, same knobs the
        # local CLI exposes. Server still owns cameras/arm so the web stays live.
        ov = d.get("overrides") or {}
        if ov.get("speed_percent") is not None:
            run_cfg["speed_percent"] = int(ov["speed_percent"])
        if ov.get("max_moves") is not None:
            run_cfg["max_moves"] = int(ov["max_moves"])
        if ov.get("approach_only"):
            run_cfg["approach_only"] = True
        if ov.get("place_only"):
            run_cfg["place_only"] = True
        if ov.get("fixed_xy") is not None:
            run_cfg["fixed_xy"] = ov["fixed_xy"]
        elif ov.get("use_fixed_xy"):
            run_cfg["fixed_xy"] = run_cfg.get("recorded_xy")
        if ov.get("no_agx"):
            tail_cfg["lstm"] = {}

        # Mode priority (ADR 0003): explicit event mode (Auto Run dropdown / CLI
        # flag) wins; otherwise fall back to the per-object config rung. Voice
        # sends the config rung it previewed, so this is consistent either way.
        mode = str(d.get("mode") or run_cfg.get("default_mode", "vla")).strip().lower()

        vla.start_auto(
            object_key=obj_key,
            checkpoint_path=str(d.get("checkpoint", auto_cfg.get("checkpoint", ""))),
            embed_path=str(obj_cfg.get("embed", "")),
            instruction=str(obj_cfg.get("instruction", "")),
            mode=mode,
            run_cfg=run_cfg,
            tail_cfg=tail_cfg,
            dry_run=bool(d.get("dry_run", True)),
        )
        emit("vla_status", vla.vla_status_payload())

    @socketio.on("vla_gate")
    def on_vla_gate(data):
        vla.gate(str((data or {}).get("name", "")))

    @socketio.on("vla_tune")
    def on_vla_tune(data):
        d = data or {}
        if not vla.tune_arrival(str(d.get("key", "")), d.get("value")):
            emit("vla_log", {"level": "WARN", "timestamp": time.strftime("%H:%M:%S"),
                             "message": f"vla_tune: rejected {d.get('key')} (no auto run active, or unknown knob)"})

    @socketio.on("reload_config")
    def on_reload_config():
        try:
            emit_config_payload(emit)
            emit("arm_log", {"level": "STEP", "message": "Config reloaded from disk", "timestamp": time.strftime("%H:%M:%S")})
        except Exception as exc:
            emit("arm_log", {"level": "ERROR", "message": f"Reload config failed: {exc}", "timestamp": time.strftime("%H:%M:%S")})

    @socketio.on("frontend_log")
    def on_frontend_log(data):
        level = str((data or {}).get("level", "INFO"))
        message = str((data or {}).get("message", ""))
        if message:
            log.info("frontend_log[%s] %s", level, message)

    @socketio.on("frontend_log_clear")
    def on_frontend_log_clear():
        log.info("frontend_log_clear")

    return app, socketio


def main():
    parser = argparse.ArgumentParser(description="Voice Pick Demo Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--profile", default="full")
    parser.add_argument("--env-config", default=None)
    parser.add_argument("--ui-mode", choices=("modern", "classic"), default=None)
    args = parser.parse_args()

    app, socketio = create_app(profile=args.profile, env_config=args.env_config, ui_mode=args.ui_mode)
    log.info(
        "Starting Voice Pick Demo at http://%s:%d ui_mode=%s",
        args.host,
        args.port,
        app.config.get("VOICE_PICK_UI_MODE", "modern"),
    )
    socketio.run(app, host=args.host, port=args.port, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
