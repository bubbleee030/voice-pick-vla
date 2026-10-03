# -*- coding: utf-8 -*-
"""AGX finger service — LSTM actions + record_api_v2-compatible gripper API.

Replaces gripper_record_api_v2.py (ADR 0001): ONE process owns the Dynamixel
motor port and the tactile sensor, serving both the simple gripper API the
laptop's fallback uses and the teammates' LSTM finger actions, remotely
triggered by the laptop's auto-run gates.

Deploy:  scp to agx:~/Desktop/GP/lstm_gripper_service.py
Run:     python3 lstm_gripper_service.py          (same env as run_LSTM5_toVLA.py,
         plus flask; NO sudo needed — the keyboard module is gone)
Stop the old gripper_record_api_v2.py first — same serial ports.

Routes (compat, mirrors record_api_v2 for GripperService):
  GET  /health                      {ok, api_version, status}
  GET  /state                       {current_pos, tactile, lstm, api_version}
  POST /command      {"action":"o"|"c"}
  POST /set_position {"positions":[a,b,c]}
Routes (new, the auto-run gates):
  POST /lstm/preload {"object":"trapezoid"|"board"|"butter_knife"}
       loads that object's models to GPU + homes fingers. Call at auto-run
       START (arm approach hides the latency).
  POST /lstm/start   {"action":"trapezoid_grasp"|...}
       1 s tactile baseline, then the 20 Hz control loop. Grasp actions home
       first IF preload didn't; release actions NEVER home (fingers are
       holding the object — homing would drop it).
  POST /lstm/prepare {"action":"knife_grasp","min_close_ticks":60,
                      "min_close_fingers":2}
       runs baseline/inference but does not drive motors; latches the first
       real closing goal.
  POST /lstm/activate
       synchronously writes the latched close goal, then resumes live control.
  POST /lstm/stop    stop the loop, freeze fingers at current position
  GET  /lstm/status  {status, action, step, loaded, tactile, motor_pos,
                      action_start_pos, goal_step, goal_pos}

Behavior preserved from run_LSTM5_toVLA.py: shutdown-protection disable +
reboot, PID gains, full-sequence LSTM inference, MOTOR_LIMITS clipping,
baseline subtraction, 0.05 s loop interval, freeze-in-place on stop.
"""
from __future__ import annotations

import sys
import numpy as np
try:
    import numpy._core
except ModuleNotFoundError:  # torch pickle compat across numpy versions
    from types import ModuleType
    _core = ModuleType('numpy._core')
    sys.modules['numpy._core'] = _core
    import numpy.core.multiarray as _multiarray
    sys.modules['numpy._core.multiarray'] = _multiarray
    _core.multiarray = _multiarray

import os
import threading
from time import sleep, time

import joblib
import serial
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dynamixel_sdk import (COMM_SUCCESS, GroupSyncRead, PacketHandler,
                           PortHandler)
from flask import Flask, jsonify, request

# ---------------------------------------------------------------- LSTM model
class GripperControllerLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=0.2 if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm(x_packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True)
        return self.fc(out)


# ---------------------------------------------------------------- registry
GP = os.environ.get("GP_MODEL_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "training_pack", "model_pack", "LSTM"))
ACTIONS = {
    # action: (subdir, variant, hidden, layers, kind)
    "trapezoid_grasp":   (f"{GP}/trapezoid_grasp/hidden64_ly3",   "trapezoid_grasp_hidden64_ly3",   64, 3, "grasp"),
    "trapezoid_release": (f"{GP}/trapezoid_release/hidden64_ly3", "trapezoid_release_hidden64_ly3", 64, 3, "release"),
    "knife_grasp":       (f"{GP}/knife_grasp/hidden64_ly3",       "knife_grasp_hidden64_ly3",       64, 3, "grasp"),
    "knife_release":     (f"{GP}/knife_release/hidden64_ly3",     "knife_release_hidden64_ly3",     64, 3, "release"),
    "PCB_release":       (f"{GP}/PCB_release/hidden64_ly3",       "PCB_release_hidden64_ly3",       64, 3, "release"),
}
# object -> finger actions (board is a lift-under pick: release only.
# butter_knife is now a real tactile grasp+release — see knife_grasp above).
OBJECT_ACTIONS = {
    "trapezoid":    ["trapezoid_grasp", "trapezoid_release"],
    "board":        ["PCB_release"],
    "butter_knife": ["knife_grasp", "knife_release"],
}

# ---------------------------------------------------------------- hardware
ADDR_SHUTDOWN, ADDR_TORQUE_ENABLE = 63, 64
ADDR_HARDWARE_ERROR_STATUS = 70        # nonzero = latched fault (overload/overheat…)
ADDR_GOAL_POSITION, ADDR_PRESENT_POSITION = 116, 132
ADDR_TELEMETRY_START, ADDR_TELEMETRY_LEN = 124, 8
ADDR_POSITION_D_GAIN, ADDR_POSITION_I_GAIN, ADDR_POSITION_P_GAIN = 80, 82, 84
TORQUE_ENABLE, TORQUE_DISABLE = 1, 0
DXL_IDS = [1, 2, 3]
MOTOR_LIMITS = [(1872, 4272), (1872, 4272), (848, 3248)]
HOME_POS = [3072, 3072, 2048]          # = record_api_v2 ORIGIN_POS; open / lift-under carry
# Jog semantics verified against gripper_record_api_v2.py 2026-07-14:
# 'v' opens / 'c' closes all fingers by GRIPPER_SPEED ticks per call (jog,
# not a pose snap); 'o' snaps to ORIGIN_POS in the legacy reversed order.
GRIPPER_SPEED = int(os.getenv("GRIPPER_STEP_TICKS", "20"))

DEVICENAME_motor = os.getenv(
    "GRIPPER_MOTOR_PORT",
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT6RW7NN-if00-port0")
BAUDRATE_motor = 57600
DEVICENAME_sensor = os.getenv("GRIPPER_SENSOR_PORT", "/dev/ttyUSB0")
BAUDRATE_sensor = 115200
TACTILE_MIN = int(os.getenv("GRIPPER_TACTILE_MIN", "0"))
TACTILE_MAX = int(os.getenv("GRIPPER_TACTILE_MAX", "4095"))


class TactileFrameParser:
    """Buffer serial bytes and publish only complete, plausible samples."""

    def __init__(self, minimum: int, maximum: int):
        self.minimum = int(minimum)
        self.maximum = int(maximum)
        self._buffer = ""

    def feed(self, raw: bytes) -> list[int] | None:
        self._buffer += raw.decode("utf-8", errors="ignore")
        frames = self._buffer.split("\n")
        self._buffer = frames.pop()
        if len(self._buffer) > 256:
            self._buffer = self._buffer[-256:]

        latest = None
        for frame in frames:
            parts = frame.strip().split()
            if len(parts) < 3:
                continue
            try:
                values = [int(parts[0]), int(parts[1]), int(parts[2])]
            except ValueError:
                continue
            if not all(
                self.minimum <= value <= self.maximum for value in values
            ):
                continue
            latest = values
        return latest


class Fingers:
    """Owns the serial ports; all motor/sensor IO goes through here."""

    def __init__(self):
        self.port = PortHandler(DEVICENAME_motor)
        self.pkt = PacketHandler(2.0)
        if not self.port.openPort():
            raise RuntimeError(f"cannot open motor port {DEVICENAME_motor}")
        self.port.setBaudRate(BAUDRATE_motor)

        for i in DXL_IDS:                       # disable overload shutdown, reboot
            self.pkt.write1ByteTxRx(self.port, i, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
            sleep(0.05)
            self.pkt.write1ByteTxRx(self.port, i, ADDR_SHUTDOWN, 21)
            sleep(0.05)
        for i in DXL_IDS:
            self.pkt.reboot(self.port, i)
        sleep(1)

        self.group_read = GroupSyncRead(self.port, self.pkt, ADDR_PRESENT_POSITION, 4)
        self.group_tel = GroupSyncRead(self.port, self.pkt, ADDR_TELEMETRY_START, ADDR_TELEMETRY_LEN)
        for i in DXL_IDS:
            self.group_read.addParam(i)
            self.group_tel.addParam(i)
        for i in DXL_IDS:
            self.pkt.write2ByteTxRx(self.port, i, ADDR_POSITION_P_GAIN, 300)
            self.pkt.write2ByteTxRx(self.port, i, ADDR_POSITION_I_GAIN, 40)
            self.pkt.write2ByteTxRx(self.port, i, ADDR_POSITION_D_GAIN, 20)
        for i in DXL_IDS:
            self.pkt.write1ByteTxRx(self.port, i, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

        try:
            self.ser = serial.Serial(DEVICENAME_sensor, BAUDRATE_sensor, timeout=1)
            sleep(2)
        except Exception as e:
            print(f"[sensor] cannot connect: {e}")
            self.ser = None

        self.com_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.tactile = [0, 0, 0]
        self.tactile_parser = TactileFrameParser(TACTILE_MIN, TACTILE_MAX)
        self.motor_pos = list(HOME_POS)
        # v2 semantics: jogs move a TARGET cursor (shared_pos), not the
        # measured position — repeated 'c' calls keep closing from the cursor.
        self.shared_pos = list(HOME_POS)
        self.running = True
        self._dead_reads = 0          # consecutive motor reads that came back dead
        self._last_reopen = 0.0       # cooldown so we don't thrash reopen()
        self.lstm_busy = False        # LSTMRunner sets this while a loop is driving
        self._last_hwerr_check = 0.0  # throttle the reg-70 scan
        self._last_reboot_heal = 0.0  # cooldown so we don't reboot-loop
        threading.Thread(target=self._background, daemon=True).start()

    def _background(self):
        while self.running:
            if self.ser and self.ser.in_waiting > 0:
                try:
                    raw = self.ser.read(self.ser.in_waiting)
                    values = self.tactile_parser.feed(raw)
                    if values is not None:
                        with self.state_lock:
                            self.tactile = values
                except Exception:
                    pass
            with self.com_lock:
                res = self.group_read.txRxPacket()
                self.group_tel.txRxPacket()
                cur = []
                for i in DXL_IDS:
                    p = self.group_read.getData(i, ADDR_PRESENT_POSITION, 4)
                    if p > 0x7FFFFFFF:
                        p -= 0x100000000
                    cur.append(p)
            # Keep publishing cur even when dead: [0,0,0] is the sentinel the
            # laptop's grasp-settle guard relies on to refuse a phantom lift.
            with self.state_lock:
                self.motor_pos = cur
            # Dead motor link (FTDI re-enumerated -> our fd is stale): COMM failure
            # or an all-zero read (0 is outside the valid Dynamixel range). Reopen
            # the by-id path — it now resolves to the live ttyUSB node — instead of
            # holding the dead handle until someone restarts the service. ~1.5 s of
            # failures (150 * 10 ms) before acting, 5 s cooldown between reopens.
            if res != COMM_SUCCESS or not any(cur):
                self._dead_reads += 1
                if self._dead_reads >= 150 and (time() - self._last_reopen) > 5.0:
                    print(f"[lstm_gripper_service] motor reads dead "
                          f"({self._dead_reads}) — reopening {DEVICENAME_motor}")
                    self._reopen_motor()
                    self._last_reopen = time()
                    self._dead_reads = 0
            else:
                self._dead_reads = 0
                # Overload torque-off LATCH (distinct from the dead-link case: the
                # motor still READS a valid value, it just won't drive to goal).
                # Reg 70 goes nonzero and torque disables when a finger clamps too
                # hard. A Dynamixel reboot clears it. Only auto-heal when NOT mid
                # LSTM (rebooting a live grasp would drop the object) and throttle.
                now = time()
                if (not self.lstm_busy and now - self._last_hwerr_check > 1.0
                        and now - self._last_reboot_heal > 8.0):
                    self._last_hwerr_check = now
                    errs = []
                    with self.com_lock:
                        for i in DXL_IDS:
                            val, r, _ = self.pkt.read1ByteTxRx(self.port, i, ADDR_HARDWARE_ERROR_STATUS)
                            if r == COMM_SUCCESS and val:
                                errs.append(i)
                    if errs:
                        print(f"[lstm_gripper_service] HW-error latch on {errs} "
                              "(overload) — auto-reboot to clear (no home)")
                        self.reboot_motors(home=False, ids=errs)
                        self._last_reboot_heal = time()
            sleep(0.01)

    def snapshot(self) -> tuple[list[int], list[int]]:
        with self.state_lock:
            return list(self.motor_pos), list(self.tactile)

    def write_goals(self, target: list[int], per_write_sleep: float = 0.0):
        with self.com_lock:
            for idx, i in enumerate(DXL_IDS):
                self.pkt.write4ByteTxRx(self.port, i, ADDR_GOAL_POSITION, int(target[idx]))
                if per_write_sleep:
                    sleep(per_write_sleep)

    def reboot_motors(self, home: bool = True, ids: list[int] | None = None) -> dict:
        """Clear a Dynamixel fault latch (overload torque-off) by rebooting the
        motor(s), then re-arm gains + torque — same recipe as startup. home()
        runs OUTSIDE com_lock (it takes com_lock itself -> would deadlock). Used
        by the auto-heal (home=False, just clear) and the manual /motor/reboot
        (home=True, open back to home)."""
        ids = list(ids or DXL_IDS)
        with self.com_lock:
            for i in ids:
                self.pkt.write1ByteTxRx(self.port, i, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
                sleep(0.02)
                self.pkt.write1ByteTxRx(self.port, i, ADDR_SHUTDOWN, 21)
                sleep(0.02)
                self.pkt.reboot(self.port, i)
            sleep(1.0)
            for i in ids:
                self.pkt.write2ByteTxRx(self.port, i, ADDR_POSITION_P_GAIN, 300)
                self.pkt.write2ByteTxRx(self.port, i, ADDR_POSITION_I_GAIN, 40)
                self.pkt.write2ByteTxRx(self.port, i, ADDR_POSITION_D_GAIN, 20)
                self.pkt.write1ByteTxRx(self.port, i, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
        if home:
            self.home()
        pos, _ = self.snapshot()
        return {"rebooted": ids, "homed": home, "motor_pos": pos}

    def freeze(self):
        """Hold at the current position (their post-'x' behavior)."""
        with self.com_lock:
            cur = []
            for i in DXL_IDS:
                p, res, _ = self.pkt.read4ByteTxRx(self.port, i, ADDR_PRESENT_POSITION)
                if res != COMM_SUCCESS:
                    p = None
                elif p > 0x7FFFFFFF:
                    p -= 0x100000000
                cur.append(p)
            for idx, i in enumerate(DXL_IDS):
                if cur[idx] is not None:
                    self.pkt.write4ByteTxRx(self.port, i, ADDR_GOAL_POSITION, cur[idx])
        with self.state_lock:
            self.shared_pos = [c if c is not None else s
                               for c, s in zip(cur, self.shared_pos)]

    def command(self, action: str) -> bool:
        """record_api_v2's jog table (shared_pos cursor). Goal writes happen
        OUTSIDE state_lock: the reader takes com_lock then state_lock, so holding
        state_lock while taking com_lock here (the old 'o' path did) can deadlock
        and leave the fingers stuck mid-move (the '手指回 home 沒有正確回' bug)."""
        reversed_open = False
        with self.state_lock:
            sp = self.shared_pos
            if action == "v":
                for i in range(3):
                    sp[i] += GRIPPER_SPEED
            elif action == "c":
                for i in range(3):
                    sp[i] -= GRIPPER_SPEED
            elif action == "o":
                self.shared_pos = sp = list(HOME_POS)
                reversed_open = True
            elif action in ("1", "2", "3"):
                sp[int(action) - 1] -= GRIPPER_SPEED
            elif action in ("4", "5", "6"):
                sp[int(action) - 4] += GRIPPER_SPEED
            elif action == "7":
                sp[1] += GRIPPER_SPEED
                sp[0] -= GRIPPER_SPEED
            else:
                return False
            target = list(sp)
        if reversed_open:                          # legacy reversed motor order
            with self.com_lock:
                for idx in range(len(DXL_IDS) - 1, -1, -1):
                    sleep(0.1)
                    self.pkt.write4ByteTxRx(self.port, DXL_IDS[idx],
                                            ADDR_GOAL_POSITION, int(target[idx]))
        else:
            self.write_goals(target)
        return True

    def home(self):
        """Establish the known-open pose for startup, preload, or fallback grasp.
        Skip the 4 s pacing sleeps when the fingers are already home."""
        pos, _ = self.snapshot()
        already = all(abs(p - h) <= 30 for p, h in zip(pos, HOME_POS))
        self.write_goals(HOME_POS, per_write_sleep=0.0 if already else 1.0)
        sleep(0.3 if already else 1.0)
        with self.state_lock:
            self.shared_pos = list(HOME_POS)

    def _reopen_motor(self) -> bool:
        """FTDI re-enumerated (ttyUSB renumber) -> our fd is dead. Reopen the
        by-id path (now the live node) and re-arm the sync-read groups + PID/torque
        WITHOUT a reboot: a USB re-enum doesn't reset the motors, only the serial
        link. Held under com_lock so no write races a half-open port."""
        with self.com_lock:
            try:
                self.port.closePort()
            except Exception:
                pass
            self.port = PortHandler(DEVICENAME_motor)
            if not self.port.openPort():
                return False
            self.port.setBaudRate(BAUDRATE_motor)
            self.group_read = GroupSyncRead(self.port, self.pkt, ADDR_PRESENT_POSITION, 4)
            self.group_tel = GroupSyncRead(self.port, self.pkt, ADDR_TELEMETRY_START, ADDR_TELEMETRY_LEN)
            for i in DXL_IDS:
                self.group_read.addParam(i)
                self.group_tel.addParam(i)
                self.pkt.write2ByteTxRx(self.port, i, ADDR_POSITION_P_GAIN, 300)
                self.pkt.write2ByteTxRx(self.port, i, ADDR_POSITION_I_GAIN, 40)
                self.pkt.write2ByteTxRx(self.port, i, ADDR_POSITION_D_GAIN, 20)
                self.pkt.write1ByteTxRx(self.port, i, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
        print(f"[lstm_gripper_service] motor port reopened: {DEVICENAME_motor}")
        return True


# ---------------------------------------------------------------- LSTM runner
class LSTMRunner:
    def __init__(self, fingers: Fingers):
        self.f = fingers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loaded: dict[str, dict] = {}        # action -> {model, scaler_x, scaler_y}
        self.preloaded_object: str | None = None
        self.status = "idle"                     # idle|preloading|ready|running|holding|error
        self.action: str | None = None
        self.step = 0
        self.last_goal_step = 0
        self.last_goal_pos: list[int] | None = None
        self.error = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._armed = False
        self._drive_enabled = threading.Event()
        self._drive_enabled.set()
        self._goal_ready = threading.Event()
        self._pending_goal: list[int] | None = None
        self.action_start_pos: list[int] | None = None
        self._min_close_ticks = 60
        self._min_close_fingers = 2

    def load_action(self, action: str):
        if action in self.loaded:
            return
        d, variant, hidden, layers, _ = ACTIONS[action]
        # Trusted local files only (teammates' training_pack on this AGX);
        # sklearn scalers require pickle — never point ACTIONS at a download.
        scaler_x = joblib.load(f"{d}/LSTM_scaler_x_{variant}.pkl")
        scaler_y = joblib.load(f"{d}/LSTM_scaler_y_{variant}.pkl")
        model = GripperControllerLSTM(input_size=6, hidden_size=hidden,
                                      num_layers=layers, output_size=3).to(self.device)
        model.load_state_dict(torch.load(f"{d}/LSTM_{variant}.pth",
                                         map_location=self.device, weights_only=True))
        model.eval()
        # warm the CUDA graph so the first live step isn't the slow one
        with torch.no_grad():
            dummy = torch.zeros((1, 1, 6), dtype=torch.float32).to(self.device)
            model(dummy, torch.tensor([1], dtype=torch.long))
        self.loaded[action] = {"model": model, "sx": scaler_x, "sy": scaler_y}

    def preload(self, object_key: str) -> dict:
        with self._lock:
            if self.status == "running":
                return {"ok": False, "error": "action running — stop first"}
            self.status = "preloading"
        t0 = time()
        try:
            for a in OBJECT_ACTIONS[object_key]:
                self.load_action(a)
            self.f.home()                        # fingers open/carry pose, ~4 s
            self.preloaded_object = object_key
            self.status = "ready"
            return {"ok": True, "object": object_key,
                    "actions": OBJECT_ACTIONS[object_key], "seconds": round(time() - t0, 1)}
        except Exception as e:
            self.status = "error"
            self.error = str(e)
            return {"ok": False, "error": str(e)}

    def start(self, action: str) -> dict:
        return self._begin(action, armed=False)

    def prepare(self, action: str, min_close_ticks: int = 60,
                min_close_fingers: int = 2) -> dict:
        return self._begin(
            action,
            armed=True,
            min_close_ticks=min_close_ticks,
            min_close_fingers=min_close_fingers,
        )

    def _begin(self, action: str, armed: bool,
               min_close_ticks: int = 60,
               min_close_fingers: int = 2) -> dict:
        if action not in ACTIONS:
            return {"ok": False, "error": f"unknown action {action}"}
        _, _, _, _, kind = ACTIONS[action]
        if armed and kind != "grasp":
            return {"ok": False, "error": "only grasp actions can be prepared"}
        preloaded_for_action = action in OBJECT_ACTIONS.get(
            self.preloaded_object or "", ()
        )
        with self._lock:
            if self.status == "running":
                return {"ok": False, "error": "already running"}
            self.status = "running"
        self.f.lstm_busy = True                   # pause auto-heal while driving
        self.load_action(action)                 # no-op when preloaded
        # Preload already established the open pose for the normal auto-grasp
        # path. Only an unprepared grasp needs the safety home here. A release
        # must begin from the held pose; homing it first opens/drops the object
        # before the release model receives its first sample.
        if kind == "grasp" and not preloaded_for_action:
            self.f.home()
        if kind == "grasp":
            self.preloaded_object = None          # consume open-pose readiness
        self._stop.clear()
        self._armed = bool(armed)
        self._drive_enabled.clear()
        if not armed:
            self._drive_enabled.set()
        self._goal_ready.clear()
        self._pending_goal = None
        self._min_close_ticks = max(1, int(min_close_ticks))
        self._min_close_fingers = max(1, min(3, int(min_close_fingers)))
        self.action_start_pos = list(self.f.snapshot()[0])
        self.action = action
        self.step = 0
        self.last_goal_step = 0
        self.last_goal_pos = None
        self.error = ""
        self._thread = threading.Thread(target=self._loop, args=(action,), daemon=True)
        self._thread.start()
        return {"ok": True, "action": action, "drive_state": self._drive_state()}

    def activate(self) -> dict:
        with self._lock:
            if (self.status != "running" or not self._armed
                    or not self._goal_ready.is_set()
                    or self._pending_goal is None):
                return {"ok": False, "error": "prepared close goal not ready"}
            goal = list(self._pending_goal)
            self.f.write_goals(goal)
            self._armed = False
            self._drive_enabled.set()
        return {"ok": True, "action": self.action, "goal": goal}

    def stop(self) -> dict:
        with self._lock:
            self.status = "stopping"
        self._stop.set()
        self._drive_enabled.set()                 # release a prepared-loop wait
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._lock:
            self._armed = False
            self._goal_ready.clear()
            self._pending_goal = None
        self.f.freeze()
        self.f.lstm_busy = False
        self.status = "holding"
        return {"ok": True, "held_at": self.f.snapshot()[0], "steps": self.step}

    def _loop(self, action: str):
        try:
            entry = self.loaded[action]
            model, sx, sy = entry["model"], entry["sx"], entry["sy"]
            # 1 s tactile baseline (their initialization balance)
            samples = []
            t0 = time()
            while time() - t0 < 1.0 and not self._stop.is_set():
                _, tac = self.f.snapshot()
                samples.append(tac)
                sleep(0.01)
            base = np.mean(np.array(samples), axis=0).round().astype(int) if samples else np.zeros(3, int)

            seq_history = []
            while not self._stop.is_set():
                self.step += 1
                pos, tac = self.f.snapshot()
                feat = [tac[0] - base[0], tac[1] - base[1], tac[2] - base[2],
                        pos[0], pos[1], pos[2]]
                seq_history.append(sx.transform([feat])[0])
                inputs = torch.from_numpy(np.array([seq_history], dtype=np.float32)).to(self.device)
                lengths = torch.tensor([len(seq_history)], dtype=torch.long)
                with torch.no_grad():
                    out = model(inputs, lengths)
                    pred = out[0, -1, :].cpu().numpy()
                pred_pos = sy.inverse_transform([pred])[0]
                target = [max(MOTOR_LIMITS[i][0], min(int(pred_pos[i]), MOTOR_LIMITS[i][1]))
                          for i in range(3)]
                self.last_goal_step = self.step
                self.last_goal_pos = list(target)
                if getattr(self, "_armed", False) and not self._drive_enabled.is_set():
                    start = self.action_start_pos or pos
                    close_count = sum(
                        (s - g) >= self._min_close_ticks
                        for s, g in zip(start, target)
                    )
                    if close_count >= self._min_close_fingers:
                        with self._lock:
                            self._pending_goal = list(target)
                            self._goal_ready.set()
                        while (not self._drive_enabled.is_set()
                               and not self._stop.is_set()):
                            sleep(0.01)
                        if self._stop.is_set():
                            break
                    sleep(0.05)
                    continue
                if self._stop.is_set():
                    break
                self.f.write_goals(target)
                sleep(0.05)
        except Exception as e:
            self.status = "error"
            self.error = str(e)
            self.f.lstm_busy = False

    def _drive_state(self) -> str:
        armed = bool(getattr(self, "_armed", False))
        goal_event = getattr(self, "_goal_ready", None)
        if armed:
            return "ready" if goal_event is not None and goal_event.is_set() else "preparing"
        return "active" if self.status == "running" else self.status

    def payload(self) -> dict:
        pos, tac = self.f.snapshot()
        goal_event = getattr(self, "_goal_ready", None)
        pending = getattr(self, "_pending_goal", None)
        return {"status": self.status, "action": self.action, "step": self.step,
                "loaded": sorted(self.loaded), "preloaded_object": self.preloaded_object,
                "drive_state": self._drive_state(),
                "goal_ready": bool(goal_event is not None and goal_event.is_set()),
                "action_start_pos": getattr(self, "action_start_pos", None),
                "pending_goal": list(pending) if pending is not None else None,
                "goal_step": self.last_goal_step,
                "goal_pos": list(self.last_goal_pos) if self.last_goal_pos is not None else None,
                "motor_pos": pos, "tactile": tac, "error": self.error}


# ---------------------------------------------------------------- HTTP app
app = Flask(__name__)
fingers = Fingers()
runner = LSTMRunner(fingers)


@app.get("/health")
def health():
    return jsonify({"ok": True, "api_version": "lstm_v1", "status": runner.status})


@app.get("/state")
def state():
    pos, tac = fingers.snapshot()
    return jsonify({"ok": True, "api_version": "lstm_v1", "current_pos": pos,
                    "tactile": tac, "lstm": runner.payload()})


@app.post("/motor/reboot")
def motor_reboot():
    """Clear a fault latch (overload torque-off) WITHOUT a full service restart:
    stop any running LSTM, reboot the motor(s), re-enable torque, and (default)
    home them open. Backs the UI 'Reboot Fingers' button + CLI --reboot-fingers."""
    body = request.get_json(silent=True) or {}
    home = bool(body.get("home", True))
    if runner.status == "running":
        runner.stop()
    res = fingers.reboot_motors(home=home)
    print(f"[lstm_gripper_service] /motor/reboot -> {res}")
    return jsonify({"ok": True, **res})


@app.post("/command")
def command():
    if runner.status == "running":
        return jsonify({"ok": False, "error": "LSTM action running — /lstm/stop first"}), 409
    action = str((request.get_json(silent=True) or {}).get("action", ""))
    if not fingers.command(action):
        return jsonify({"ok": False, "status": "error",
                        "error": f"unknown action {action}"}), 400
    return jsonify({"ok": True, "status": "success", "action": action,
                    "current_pos": list(fingers.shared_pos)})


@app.post("/set_position")
def set_position():
    if runner.status == "running":
        return jsonify({"ok": False, "error": "LSTM action running — /lstm/stop first"}), 409
    p = (request.get_json(silent=True) or {}).get("positions")
    if not isinstance(p, list) or len(p) != 3:
        return jsonify({"ok": False, "error": "positions must be [a,b,c]"}), 400
    target = [max(MOTOR_LIMITS[i][0], min(int(p[i]), MOTOR_LIMITS[i][1])) for i in range(3)]
    fingers.write_goals(target)
    return jsonify({"ok": True, "target": target})


@app.post("/lstm/preload")
def lstm_preload():
    obj = str((request.get_json(silent=True) or {}).get("object", ""))
    if obj not in OBJECT_ACTIONS:
        return jsonify({"ok": False, "error": f"unknown object {obj}"}), 400
    return jsonify(runner.preload(obj))


@app.post("/lstm/start")
def lstm_start():
    action = str((request.get_json(silent=True) or {}).get("action", ""))
    res = runner.start(action)
    return jsonify(res), (200 if res.get("ok") else 409)


@app.post("/lstm/prepare")
def lstm_prepare():
    body = request.get_json(silent=True) or {}
    action = str(body.get("action", ""))
    res = runner.prepare(
        action,
        min_close_ticks=int(body.get("min_close_ticks", 60)),
        min_close_fingers=int(body.get("min_close_fingers", 2)),
    )
    return jsonify(res), (200 if res.get("ok") else 409)


@app.post("/lstm/activate")
def lstm_activate():
    res = runner.activate()
    return jsonify(res), (200 if res.get("ok") else 409)


@app.post("/lstm/stop")
def lstm_stop():
    return jsonify(runner.stop())


@app.get("/lstm/status")
def lstm_status():
    return jsonify(runner.payload())


if __name__ == "__main__":
    print(f"[lstm_gripper_service] device={runner.device}  motor={DEVICENAME_motor}")
    # Restart should leave the fingers OPEN at home (user request): home once at
    # startup so a service restart always resets the gripper to a known-open pose.
    try:
        fingers.home()
        print(f"[lstm_gripper_service] fingers homed to {HOME_POS}")
    except Exception as exc:
        print(f"[lstm_gripper_service] startup home failed: {exc}")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5003")), threaded=True)
