#!/usr/bin/env python3
"""Regression test: RealSense startup survives the brief post-replug USB delay."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services import realsense_service as service


class _FakeConfig:
    stream_requests: list[tuple[object, ...]] = []

    def enable_device(self, serial):
        self.serial = serial

    def enable_stream(self, *args):
        type(self).stream_requests.append(args)


class _FakePipeline:
    attempts = 0
    failures_before_success = 2

    def start(self, config):
        type(self).attempts += 1
        if type(self).attempts <= type(self).failures_before_success:
            raise RuntimeError("Couldn't resolve requests")

    def stop(self):
        pass


class _FakeRS:
    pipeline = _FakePipeline
    config = _FakeConfig
    stream = type("Stream", (), {"color": object(), "depth": object()})
    format = type("Format", (), {"bgr8": object(), "z16": object()})

    @staticmethod
    def align(stream):
        return object()


class RealSenseRecoveryTest(unittest.TestCase):
    def test_retries_until_post_replug_pipeline_starts(self):
        _FakePipeline.attempts = 0
        _FakePipeline.failures_before_success = 2
        cfg = {
            "cameras": {
                "resolution_w": 424,
                "resolution_h": 240,
                "fps": 15,
                "init_delay_s": 0,
                "reconnect_attempts": 3,
                "reconnect_delay_s": 0,
                "cam1": {"enabled": True, "serial": "cam-1", "enable_depth": True},
                "cam2": {"enabled": False},
            }
        }
        with patch.object(service, "RS_AVAILABLE", True), patch.object(service, "rs", _FakeRS), patch.object(
            service.SingleCameraStream, "_capture_loop", lambda self: None
        ):
            cameras = service.RealSenseService(cfg)
            cameras.start()

        self.assertEqual(_FakePipeline.attempts, 3)
        self.assertTrue(cameras.cams["cam1"].running)
        self.assertIsNone(cameras.cams["cam1"].last_error)

    def test_leaves_depth_profile_selection_to_realsense(self):
        _FakePipeline.attempts = 0
        _FakePipeline.failures_before_success = 0
        _FakeConfig.stream_requests = []
        cfg = {
            "cameras": {
                "resolution_w": 424,
                "resolution_h": 240,
                "fps": 15,
                "init_delay_s": 0,
                "cam1": {"enabled": True, "serial": "cam-1", "enable_depth": True},
                "cam2": {"enabled": False},
            }
        }
        with patch.object(service, "RS_AVAILABLE", True), patch.object(service, "rs", _FakeRS), patch.object(
            service.SingleCameraStream, "_capture_loop", lambda self: None
        ):
            cameras = service.RealSenseService(cfg)
            cameras.start()

        depth_requests = [args for args in _FakeConfig.stream_requests if args[0] is _FakeRS.stream.depth]
        self.assertEqual(depth_requests, [(_FakeRS.stream.depth,)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
