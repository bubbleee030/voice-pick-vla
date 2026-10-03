from __future__ import annotations

import os
from typing import Any

from src.runtime_config import module_enabled, read_launcher_state


class HealthService:
    def __init__(self, demo_cfg: dict[str, Any], profile_name: str, realsense_service, claw_service, arm_service, gripper_service, recording_service):
        self.demo_cfg = demo_cfg
        self.profile_name = profile_name
        self.realsense = realsense_service
        self.claw = claw_service
        self.arm = arm_service
        self.gripper = gripper_service
        self.recording = recording_service

    def snapshot(self) -> dict[str, Any]:
        launcher = read_launcher_state()
        pid = launcher.get("pid")
        launcher_running = bool(pid and self._pid_alive(int(pid)))
        return {
            "profile": self.profile_name,
            "modules": self.demo_cfg.get("modules", {}),
            "launcher": {**launcher, "running": launcher_running},
            "cameras": self.realsense.status(),
            "claw": self.claw.status(),
            "arm": {
                "connected": self.arm.connected,
                "label": self.arm.connection_label,
                "current_pose_mm_deg": self.arm.current_pose_mm_deg,
            },
            "gripper": self.gripper.status(),
            "recording": self.recording.dataset_capture_status(),
        }

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
