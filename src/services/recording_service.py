from __future__ import annotations

import csv
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2

from src.demo_runtime import DemoRealtimeSession, FeatureToggles, RuntimeHooks, SessionConfig, next_episode_index
from src.runtime_config import PROJECT_ROOT

try:
    import requests
except ImportError:
    requests = None


log = logging.getLogger("dataset_capture")


STAGE_PROGRESS = {
    "starting": 5,
    "pausing_preview": 12,
    "recording": 20,
    "stop_requested": 35,
    "finalizing": 50,
    "stopping_agx": 62,
    "flushing_aux": 74,
    "committing_episode": 86,
    "writing_metadata": 94,
    "restoring_preview": 97,
    "completed": 100,
    "completed_empty": 100,
}


class ValidationSessionRecorder:
    def __init__(self, socketio, objects_cfg: dict[str, Any]):
        self.sio = socketio
        self.objects_cfg = objects_cfg
        self.root_dir = PROJECT_ROOT / "data" / "validation_sessions"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._current: Optional[dict[str, Any]] = None
        self._last_completed: Optional[dict[str, Any]] = None

    def default_teach_recording(self, object_key: str) -> str:
        obj_def = self.objects_cfg.get("classes", {}).get(object_key, {})
        return obj_def.get("default_teach_recording", f"{object_key}_pick_v1")

    def state_payload(self) -> dict[str, Any]:
        with self._lock:
            return {"active": self._public_session(self._current), "last_completed": self._last_completed}

    def emit_state(self) -> None:
        self.sio.emit("validation_state", self.state_payload())

    def _public_session(self, session: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if session is None:
            return None
        return {
            "object_key": session["object_key"],
            "slot_id": session["slot_id"],
            "table_set_id": session["table_set_id"],
            "mode": session["mode"],
            "requested_text": session["requested_text"],
            "resolved_object_key": session["resolved_object_key"],
            "teach_recording_name": session["teach_recording_name"],
            "operator_result": session["operator_result"],
            "start_time": session["start_time"],
            "session_dir": session["session_dir"],
        }

    def _write_session_json(self, session: dict[str, Any]) -> None:
        payload = {
            "object_key": session["object_key"],
            "slot_id": session["slot_id"],
            "table_set_id": session["table_set_id"],
            "mode": session["mode"],
            "ready_pose": session["ready_pose"],
            "pick_pose": session["pick_pose"],
            "teach_recording_name": session["teach_recording_name"],
            "requested_text": session["requested_text"],
            "resolved_object_key": session["resolved_object_key"],
            "operator_result": session["operator_result"],
            "start_time": session["start_time"],
            "end_time": session["end_time"],
        }
        with open(session["session_json_path"], "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _write_jsonl(self, handle, payload: dict[str, Any]) -> None:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()

    def start_session(
        self,
        object_key: str,
        obj_def: dict[str, Any],
        mode: str,
        requested_text: str,
        resolved_object_key: str,
        teach_recording_name: str = "",
    ) -> tuple[bool, str]:
        with self._lock:
            if self._current is not None and self._current["operator_result"] is None:
                return False, "validation session already active; mark success/fail first"

            ts = time.time()
            slug = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
            session_dir = self.root_dir / f"{slug}_{int(ts * 1000) % 1000:03d}_{object_key}"
            session_dir.mkdir(parents=True, exist_ok=True)

            trajectory_path = session_dir / "arm_trajectory.csv"
            arm_logs_path = session_dir / "arm_logs.jsonl"
            speech_events_path = session_dir / "speech_events.jsonl"
            session_json_path = session_dir / "session.json"

            trajectory_file = open(trajectory_path, "w", newline="", encoding="utf-8")
            trajectory_writer = csv.writer(trajectory_file)
            trajectory_writer.writerow(["timestamp_unix", "x_mm", "y_mm", "z_mm", "rx_deg", "ry_deg", "rz_deg"])

            arm_logs_file = open(arm_logs_path, "w", encoding="utf-8", buffering=1)
            speech_events_file = open(speech_events_path, "w", encoding="utf-8", buffering=1)

            fixed_poses = obj_def.get("fixed_poses", {})
            session = {
                "object_key": object_key,
                "slot_id": obj_def.get("slot_id", ""),
                "table_set_id": obj_def.get("table_set_id", ""),
                "mode": mode,
                "ready_pose": fixed_poses.get("approach", []),
                "pick_pose": fixed_poses.get("pick", []),
                "teach_recording_name": teach_recording_name,
                "requested_text": requested_text,
                "resolved_object_key": resolved_object_key,
                "operator_result": None,
                "start_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)),
                "end_time": None,
                "session_dir": str(session_dir),
                "session_json_path": session_json_path,
                "trajectory_file": trajectory_file,
                "trajectory_writer": trajectory_writer,
                "arm_logs_file": arm_logs_file,
                "speech_events_file": speech_events_file,
            }
            self._current = session
            self._write_session_json(session)

        self.emit_state()
        return True, str(session_dir)

    def log_arm_pose(self, pose_mm_deg: list[float], timestamp: float) -> None:
        with self._lock:
            session = self._current
            if session is None or session["operator_result"] is not None:
                return
            session["trajectory_writer"].writerow(
                [f"{timestamp:.6f}", f"{pose_mm_deg[0]:.3f}", f"{pose_mm_deg[1]:.3f}", f"{pose_mm_deg[2]:.3f}", f"{pose_mm_deg[3]:.3f}", f"{pose_mm_deg[4]:.3f}", f"{pose_mm_deg[5]:.3f}"]
            )
            session["trajectory_file"].flush()

    def log_arm_log(self, level: str, message: str, timestamp: str) -> None:
        with self._lock:
            session = self._current
            if session is None or session["operator_result"] is not None:
                return
            self._write_jsonl(session["arm_logs_file"], {"timestamp": timestamp, "level": level, "message": message})

    def log_speech(self, event_type: str, payload: dict[str, Any]) -> None:
        with self._lock:
            session = self._current
            if session is None or session["operator_result"] is not None:
                return
            self._write_jsonl(
                session["speech_events_file"],
                {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()), "event_type": event_type, "payload": payload},
            )

    def mark_result(self, result: str) -> tuple[bool, str]:
        if result not in {"success", "fail"}:
            return False, "invalid validation result"
        with self._lock:
            session = self._current
            if session is None:
                return False, "no active validation session"
            session["operator_result"] = result
            session["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
            self._write_session_json(session)
            self._last_completed = {
                "object_key": session["object_key"],
                "slot_id": session["slot_id"],
                "table_set_id": session["table_set_id"],
                "mode": session["mode"],
                "requested_text": session["requested_text"],
                "resolved_object_key": session["resolved_object_key"],
                "teach_recording_name": session["teach_recording_name"],
                "operator_result": session["operator_result"],
                "start_time": session["start_time"],
                "end_time": session["end_time"],
                "session_dir": session["session_dir"],
            }
            session["trajectory_file"].close()
            session["arm_logs_file"].close()
            session["speech_events_file"].close()
            self._current = None
        self.emit_state()
        return True, result


class TeachManager:
    def __init__(self, demo_cfg: dict[str, Any], arm_service, gripper_service, socketio):
        self.demo_cfg = demo_cfg
        self.arm = arm_service
        self.gripper = gripper_service
        self.sio = socketio
        self.save_dir = PROJECT_ROOT / demo_cfg.get("teach", {}).get("save_dir", "data/teach_recordings")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.recording = False
        self.current_name = ""
        self.waypoints: list[dict[str, Any]] = []
        self.start_time = 0.0

    def start(self, name: str):
        self.current_name = name
        self.waypoints = []
        self.start_time = time.time()
        self.recording = True
        self.sio.emit("teach_data", {"waypoints": [], "count": 0})
        self.sio.emit("arm_log", {"level": "STEP", "message": f"Teach recording started: {name}", "timestamp": time.strftime("%H:%M:%S")})

    def save_waypoint(self, gripper: str = "none", speed: int = 30):
        if not self.recording:
            return
        pose = self.arm.current_pose_mm_deg
        if pose is None:
            self.sio.emit("arm_log", {"level": "WARN", "message": "Cannot save waypoint: no pose data", "timestamp": time.strftime("%H:%M:%S")})
            return
        raw_pose = [int(v * 1000) for v in pose]
        gripper_state = self.gripper.get_state() if self.gripper is not None else None
        wp = {"t_ms": int((time.time() - self.start_time) * 1000), "pose": raw_pose, "gripper": gripper, "speed": speed}
        if isinstance(gripper_state, dict):
            current_pos = gripper_state.get("current_pos")
            if isinstance(current_pos, list) and len(current_pos) == 3:
                wp["gripper_pos"] = [int(v) for v in current_pos]
            if "server_time_unix" in gripper_state:
                wp["gripper_server_time_unix"] = gripper_state.get("server_time_unix")
            if "tactile_data" in gripper_state:
                wp["tactile_data"] = gripper_state.get("tactile_data")
        self.waypoints.append(wp)
        self.sio.emit("teach_data", {"waypoints": self.waypoints, "count": len(self.waypoints)})

    def stop(self) -> Optional[str]:
        if not self.recording:
            return None
        self.recording = False
        if not self.waypoints:
            self.sio.emit("teach_data", {"waypoints": [], "count": 0})
            return None
        data = {
            "name": self.current_name,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "start_unix": self.start_time,
            "waypoints": self.waypoints,
        }
        path = self.save_dir / f"{self.current_name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.sio.emit("teach_data", {"waypoints": [], "count": 0})
        return str(path)

    def list_recordings(self) -> list[dict[str, Any]]:
        results = []
        for path in sorted(self.save_dir.glob("*.json")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                results.append(
                    {
                        "id": path.stem,
                        "name": data.get("name", path.stem),
                        "created": data.get("created", ""),
                        "count": len(data.get("waypoints", [])),
                        "has_external_timeline": bool(data.get("external_timeline")),
                    }
                )
            except Exception:
                pass
        return results

    def load_recording(self, name: str) -> Optional[dict[str, Any]]:
        path = self.save_dir / f"{name}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


class DatasetCaptureManager:
    def __init__(self, demo_cfg: dict[str, Any], socketio, arm_service, gripper_service, claw_service, realsense_service=None):
        self.demo_cfg = demo_cfg
        self.socketio = socketio
        self.arm = arm_service
        self.gripper = gripper_service
        self.claw = claw_service
        self.realsense = realsense_service
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._session: DemoRealtimeSession | None = None
        self._state: dict[str, Any] = {"running": False}
        self._stage_name: str | None = None
        self._stage_started_monotonic: float | None = None

    def _update_state(self, **fields: Any) -> None:
        with self._lock:
            self._state.update(fields)
        self._emit_state()

    def _set_stage(self, stage: str, message: str | None = None, **extra: Any) -> None:
        now_mono = time.monotonic()
        prev_stage = self._stage_name
        prev_started = self._stage_started_monotonic
        if prev_stage is not None and prev_started is not None and prev_stage != stage:
            elapsed = now_mono - prev_started
            log.info("[dataset_capture] stage_done stage=%s elapsed=%.3fs", prev_stage, elapsed)
        self._stage_name = stage
        self._stage_started_monotonic = now_mono
        payload = {"stage": stage}
        if message is not None:
            payload["status_message"] = message
        payload["progress_pct"] = int(STAGE_PROGRESS.get(stage, 0))
        payload["stage_elapsed_s"] = 0.0
        payload.update(extra)
        self._update_state(**payload)
        log.info(
            "[dataset_capture] stage_start stage=%s progress=%s%% message=%s",
            stage,
            payload["progress_pct"],
            message or stage,
        )

    def _remote_recording_payload(self, object_name: str, episode_index: int, notes: str) -> dict[str, Any]:
        remote_cfg = self.demo_cfg.get("gripper", {}).get("remote_recording", {})
        return {
            "object_name": object_name,
            "episode_index": episode_index,
            "notes": notes,
            "filename_prefix": str(remote_cfg.get("filename_prefix", "tactile_v2")),
            "output_dir": str(remote_cfg.get("output_dir", "data/agx_tactile_raw")),
        }

    def _pause_realsense_preview(self) -> bool:
        if self.realsense is None:
            return False
        try:
            self.realsense.stop()
            log.info("[dataset_capture] paused UI RealSense preview")
            return True
        except Exception as exc:
            log.warning("[dataset_capture] failed to pause UI RealSense preview: %s", exc)
            return False

    def _restore_realsense_preview(self) -> None:
        if self.realsense is None:
            return
        try:
            self.realsense.start()
            log.info("[dataset_capture] restored UI RealSense preview")
        except Exception as exc:
            log.warning("[dataset_capture] failed to restore UI RealSense preview: %s", exc)

    def _emit_state(self):
        self.socketio.emit("dataset_capture_state", self.dataset_capture_status())

    def dataset_capture_status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def start(self, payload: dict[str, Any]) -> tuple[bool, str]:
        with self._lock:
            if self._state.get("running"):
                return False, "dataset capture already running"

        dataset_cfg = self.demo_cfg.get("teach", {}).get("dataset_capture", {})
        object_name = str(payload.get("object") or payload.get("object_name") or "demo")
        streams = payload.get("streams") or dataset_cfg.get("camera_streams", [])
        fps = float(payload.get("fps") or dataset_cfg.get("image_fps", 6))
        telemetry_hz = float(payload.get("telemetry_hz") or dataset_cfg.get("telemetry_hz", 10))
        duration_s = float(payload.get("duration_s") or payload.get("duration") or 0.0)
        notes = str(payload.get("notes") or "")
        output_root = str(dataset_cfg.get("output_root", "data/recordings"))

        output_path = PROJECT_ROOT / output_root / object_name
        episode_index = next_episode_index(output_path)
        final_output_dir = output_path / f"episode_{episode_index:03d}"
        arm_conn = self.demo_cfg.get("arm", {}).get("connections", [{}])[0]
        gripper_url = self.gripper.active_base_url if self.gripper is not None else ""
        remote_recording = {"ok": False, "status": "not_requested"}
        if self.gripper is not None and gripper_url:
            remote_recording = self.gripper.start_remote_recording(
                self._remote_recording_payload(object_name, episode_index, notes)
            )
            if not bool(remote_recording.get("ok")):
                self.socketio.emit(
                    "arm_log",
                    {
                        "level": "WARN",
                        "message": f"AGX tactile backup did not start: {remote_recording.get('message') or remote_recording.get('status')}",
                        "timestamp": time.strftime("%H:%M:%S"),
                    },
                )

        features = FeatureToggles(
            cam1_enabled="cam1_rgb" in streams or "cam1_depth" in streams,
            cam2_enabled="cam2_rgb" in streams or "cam2_depth" in streams,
            rgb_enabled=any(s.endswith("_rgb") for s in streams if s.startswith("cam")),
            depth_enabled=any(s.endswith("_depth") for s in streams),
            align_enabled=True,
            arm_log_enabled=True,
            gripper_log_enabled=bool(gripper_url),
            yolo_enabled=False,
            preview_window_enabled=False,
            disk_output_enabled=True,
        )

        session_cfg = SessionConfig(
            object_name=object_name,
            cam1_serial=str(self.demo_cfg.get("cameras", {}).get("cam1", {}).get("serial", "")),
            cam2_serial=str(self.demo_cfg.get("cameras", {}).get("cam2", {}).get("serial", "")) or None,
            episode_index=episode_index,
            duration_s=duration_s,
            fps=fps,
            pose_hz=telemetry_hz,
            arm_monitor_hz=float(self.demo_cfg.get("arm", {}).get("pose_poll_hz", 4)),
            safe_profile=True,
            output_root=output_root,
            notes=notes,
            arm_host=str(arm_conn.get("host", "127.0.0.1")),
            arm_port=int(arm_conn.get("port", 1502)),
            gripper_api_url=gripper_url,
            gripper_poll_hz=max(telemetry_hz, 10.0),
            features=features,
        )

        session = DemoRealtimeSession(config=session_cfg, hooks=RuntimeHooks(on_telemetry=self._on_telemetry))

        thread = threading.Thread(
            target=self._run_capture,
            args=(session, object_name, episode_index, streams, fps, notes, remote_recording),
            daemon=True,
        )
        with self._lock:
            self._session = session
            self._thread = thread
            self._state = {
                "running": True,
                "stopping": False,
                "stage": "starting",
                "status_message": "Starting dataset capture",
                "object": object_name,
                "episode_index": episode_index,
                "streams": streams,
                "fps": fps,
                "telemetry_hz": telemetry_hz,
                "notes": notes,
                "output_dir": str(final_output_dir),
                "staging_dir": None,
                "summary": None,
                "remote_recording": remote_recording,
            }
        thread.start()
        self._emit_state()
        log.info(
            "[dataset_capture] start object=%s episode=%03d final_dir=%s streams=%s fps=%s telemetry_hz=%s",
            object_name,
            episode_index,
            final_output_dir,
            ",".join(streams),
            fps,
            telemetry_hz,
        )
        return True, f"dataset capture started for {object_name}"

    def stop(self) -> tuple[bool, str]:
        with self._lock:
            session = self._session
            thread = self._thread
            running = bool(self._state.get("running"))
        if not running or session is None:
            return False, "no dataset capture running"
        self._set_stage("stop_requested", "Stop requested, waiting for finalize", stopping=True)
        session.request_stop()
        if thread is not None:
            thread.join()
        with self._lock:
            output_dir = self._state.get("output_dir")
            summary = self._state.get("summary")
        if self._stage_name is not None and self._stage_started_monotonic is not None:
            elapsed = time.monotonic() - self._stage_started_monotonic
            log.info("[dataset_capture] stage_done stage=%s elapsed=%.3fs", self._stage_name, elapsed)
            self._stage_name = None
            self._stage_started_monotonic = None
        log.info("[dataset_capture] stop complete output_dir=%s summary=%s", output_dir, summary)
        return True, "stop completed"

    def _on_telemetry(self, telemetry: dict[str, Any]) -> None:
        with self._lock:
            self._state["telemetry"] = telemetry
        self._emit_state()

    def _run_capture(
        self,
        session: DemoRealtimeSession,
        object_name: str,
        episode_index: int,
        streams: list[str],
        fps: float,
        notes: str,
        remote_recording_start: dict[str, Any],
    ):
        output_root = PROJECT_ROOT / self.demo_cfg.get("teach", {}).get("dataset_capture", {}).get("output_root", "data/recordings")
        claw_staging = PROJECT_ROOT / "data" / "runtime" / f"claw_capture_{object_name}_{episode_index:03d}_{int(time.time())}"
        claw_dir = claw_staging / "claw_rgb"
        sensor_path = claw_staging / "sensor_stream.csv"
        claw_dir.mkdir(parents=True, exist_ok=True)
        log.info("[dataset_capture] claw staging dir=%s", claw_staging)

        claw_stop = threading.Event()
        claw_stats = {"frames": 0}
        sensor_stats = {"rows": 0}

        claw_thread = threading.Thread(target=self._capture_claw_frames, args=(claw_stop, claw_dir, fps, claw_stats), daemon=True)
        sensor_thread = threading.Thread(target=self._capture_sensor_rows, args=(claw_stop, sensor_path, sensor_stats), daemon=True)
        gripper_status = self.gripper.status() if self.gripper is not None else {}
        gripper_api_version = str(gripper_status.get("api_version") or "")
        use_integrated_tactile = gripper_api_version == "v2"
        if "claw_rgb" in streams:
            claw_thread.start()
        if self.demo_cfg.get("sensor_api", {}).get("enabled", False) and not use_integrated_tactile:
            sensor_thread.start()

        preview_paused = False
        try:
            self._set_stage("pausing_preview", "Pausing UI RealSense preview for exclusive recording access")
            preview_paused = self._pause_realsense_preview()
            self._set_stage(
                "recording",
                f"Recording in progress. Final dir: {output_root / object_name / f'episode_{episode_index:03d}'}",
                staging_dir=str(claw_staging),
            )
            summary = session.run()
            self._set_stage("finalizing", f"Session stopped. stop_reason={summary.get('stop_reason')}")
            remote_recording_stop = {"ok": False, "status": "not_requested"}
            if self.gripper is not None and bool(remote_recording_start.get("ok")):
                self._set_stage("stopping_agx", "Stopping AGX tactile recording")
                remote_recording_stop = self.gripper.stop_remote_recording()
            self._set_stage("flushing_aux", "Waiting for claw/sensor capture threads to flush")
            claw_stop.set()
            if claw_thread.is_alive():
                claw_thread.join()
            if sensor_thread.is_alive():
                sensor_thread.join()

            output_dir = summary.get("output_dir")
            if output_dir:
                self._set_stage("committing_episode", f"Committing episode to {output_dir}", output_dir=output_dir)
                final_dir = Path(output_dir)
                if claw_dir.exists() and any(claw_dir.glob("*.jpg")):
                    final_claw_dir = final_dir / "claw_rgb"
                    final_claw_dir.mkdir(parents=True, exist_ok=True)
                    for path in sorted(claw_dir.glob("*.jpg")):
                        path.replace(final_claw_dir / path.name)
                if sensor_path.exists() and sensor_path.stat().st_size > 0:
                    sensor_path.replace(final_dir / "sensor_stream.csv")
                metadata_path = final_dir / "metadata.json"
                if metadata_path.exists():
                    self._set_stage("writing_metadata", f"Writing metadata to {metadata_path}", output_dir=output_dir)
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    metadata["claw_capture"] = claw_stats
                    metadata["sensor_capture"] = sensor_stats
                    metadata["requested_streams"] = streams
                    metadata["notes"] = notes
                    metadata["gripper_recording"] = {
                        "api_version": gripper_api_version or None,
                        "start": remote_recording_start,
                        "stop": remote_recording_stop,
                        "integrated_tactile": use_integrated_tactile,
                        "fallback_used": bool(remote_recording_start.get("fallback_used") or remote_recording_stop.get("fallback_used")),
                    }
                    disconnect_count = metadata.get("gripper", {}).get("sensor_disconnect_count")
                    metadata["gripper_recording"]["sensor_disconnect_count"] = disconnect_count
                    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            else:
                self._set_stage("completed_empty", "Capture ended without committed output directory")
                log.warning("[dataset_capture] session ended without committed output dir")
        finally:
            claw_stop.set()
            if claw_thread.is_alive():
                claw_thread.join()
            if sensor_thread.is_alive():
                sensor_thread.join()
            if preview_paused:
                self._set_stage("restoring_preview", "Restoring UI RealSense preview")
                self._restore_realsense_preview()

        with self._lock:
            self._state = {
                "running": False,
                "stopping": False,
                "stage": "completed",
                "status_message": f"Completed. Output dir: {output_dir or 'none'}",
                "object": object_name,
                "episode_index": episode_index,
                "streams": streams,
                "fps": fps,
                "output_dir": output_dir,
                "staging_dir": str(claw_staging),
                "summary": summary,
                "remote_recording": {
                    "start": remote_recording_start,
                    "stop": remote_recording_stop,
                },
            }
            self._session = None
            self._thread = None
        self._emit_state()

    def _capture_claw_frames(self, stop_event: threading.Event, output_dir: Path, fps: float, stats: dict[str, int]):
        interval = 1.0 / max(fps, 1.0)
        frame_idx = 0
        while not stop_event.is_set():
            frame = self.claw.get_latest_frame()
            if frame is not None:
                cv2.imwrite(str(output_dir / f"frame_{frame_idx:06d}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                frame_idx += 1
                stats["frames"] = frame_idx
            time.sleep(interval)

    def _capture_sensor_rows(self, stop_event: threading.Event, output_path: Path, stats: dict[str, int]):
        if requests is None:
            return
        sensor_cfg = self.demo_cfg.get("sensor_api", {})
        base_url = str(sensor_cfg.get("base_url", "")).rstrip("/")
        endpoint = str(sensor_cfg.get("endpoint", "/get_sensor"))
        poll_hz = float(sensor_cfg.get("poll_hz", 15))
        timeout = float(sensor_cfg.get("request_timeout_s", 1.0))
        if not base_url:
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_unix", "analog_value1", "analog_value2", "analog_value3", "raw"])
            while not stop_event.is_set():
                row = [f"{time.time():.6f}", "", "", "", ""]
                try:
                    resp = requests.get(f"{base_url}{endpoint}", timeout=timeout)
                    resp.raise_for_status()
                    data = resp.json()
                    row = [
                        f"{time.time():.6f}",
                        data.get("analog_value1", ""),
                        data.get("analog_value2", ""),
                        data.get("analog_value3", ""),
                        json.dumps(data, ensure_ascii=False),
                    ]
                except Exception:
                    pass
                writer.writerow(row)
                stats["rows"] += 1
                time.sleep(1.0 / max(poll_hz, 1.0))
