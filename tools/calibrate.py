"""
Hand-Eye Calibration Tool.

Computes the transformation matrix from camera coordinates to robot base coordinates.
This allows YOLO-detected 3D positions to be converted into arm target positions.

=== What is "hand-eye calibration"? ===

Your cameras are fixed next to the arm (eye-to-hand setup).
The arm knows its own position in "robot coordinates."
The cameras see the world in "camera coordinates."
We need a matrix to convert between them.

=== How it works ===

1. Print a checkerboard pattern and place it on the desk.
2. Move the arm's end-effector to touch a known point on the board.
3. Record: arm's position (from Modbus) + checkerboard detection (from camera).
4. Repeat 15+ times with the arm in different positions.
5. OpenCV computes the transformation matrix.

=== What are "15 known poses"? ===

You physically move the robot arm's gripper/tool tip to 15 different positions
on or near the checkerboard. At each position:
  - The script reads the arm's Cartesian position from Modbus registers
  - The camera takes a photo and detects the checkerboard corners
  - Both data points are saved as a calibration pair

More positions = more accurate calibration. 15 is a practical minimum.
Space them around the workspace: center, corners, different heights.

=== What is a "checkerboard pattern"? ===

A grid of alternating black and white squares, like a chess board.
OpenCV can detect the inner corner points with sub-pixel accuracy.
Print the generated PNG at actual size (no scaling) on a flat surface.

Usage:
    # Step 1: Generate checkerboard PNG for printing
    python tools/calibrate.py --generate-board --paper-size A3

    # Step 2: Run calibration (requires camera + arm connected)
    python tools/calibrate.py --run --camera 1

    # Step 3: Re-compute from saved data (optional)
    python tools/calibrate.py --compute --camera 1
"""

import argparse
import json
import sys
import time
import numpy as np
import cv2
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, save_config


PAPER_SIZES_MM = {
    "A4": (210.0, 297.0),
    "A3": (297.0, 420.0),
}


def generate_checkerboard(rows: int = 7, cols: int = 9,
                          square_size_mm: float = 25.0,
                          output_path: str | None = None,
                          paper_size: str = "A4",
                          landscape: bool = True):
    """
    Generate a checkerboard image for printing.

    Args:
        rows: Number of inner corners vertically.
        cols: Number of inner corners horizontally.
        square_size_mm: Size of each square in mm.
        output_path: Where to save the image.
        paper_size: Paper size name, e.g. A4 or A3.
        landscape: True for landscape orientation, False for portrait.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive integers.")
    if square_size_mm <= 0:
        raise ValueError("square_size_mm must be > 0.")

    paper_key = paper_size.upper()
    if paper_key not in PAPER_SIZES_MM:
        supported = ", ".join(sorted(PAPER_SIZES_MM.keys()))
        raise ValueError(f"Unsupported paper size '{paper_size}'. Supported: {supported}")

    base_w_mm, base_h_mm = PAPER_SIZES_MM[paper_key]
    if landscape:
        page_w_mm, page_h_mm = base_h_mm, base_w_mm
        orientation = "landscape"
    else:
        page_w_mm, page_h_mm = base_w_mm, base_h_mm
        orientation = "portrait"

    # Render output at 300 DPI for printing.
    dpi = 300
    px_per_mm = dpi / 25.4

    # Calculate board size
    board_squares_x = cols + 1  # squares = corners + 1
    board_squares_y = rows + 1
    board_w_mm = board_squares_x * square_size_mm
    board_h_mm = board_squares_y * square_size_mm

    # Fail early if checkerboard does not fit the selected page.
    if board_w_mm > page_w_mm or board_h_mm > page_h_mm:
        max_square_w = page_w_mm / board_squares_x
        max_square_h = page_h_mm / board_squares_y
        max_square_mm = min(max_square_w, max_square_h)
        raise ValueError(
            "Checkerboard does not fit on selected paper. "
            f"Board={board_w_mm:.1f}x{board_h_mm:.1f}mm, "
            f"Paper={page_w_mm:.1f}x{page_h_mm:.1f}mm ({paper_key} {orientation}). "
            f"Use square_size_mm <= {max_square_mm:.1f}, choose A3, or switch orientation."
        )

    square_px = int(square_size_mm * px_per_mm)

    board_w = board_squares_x * square_px
    board_h = board_squares_y * square_px

    # Create page canvas
    page_w = int(page_w_mm * px_per_mm)
    page_h = int(page_h_mm * px_per_mm)
    page = np.ones((page_h, page_w), dtype=np.uint8) * 255

    # Center the board on the page
    start_x = (page_w - board_w) // 2
    start_y = (page_h - board_h) // 2

    # Draw checkerboard
    for row in range(board_squares_y):
        for col in range(board_squares_x):
            if (row + col) % 2 == 0:
                x1 = start_x + col * square_px
                y1 = start_y + row * square_px
                x2 = x1 + square_px
                y2 = y1 + square_px
                page[y1:y2, x1:x2] = 0  # Black square

    # Save
    if output_path is None:
        output_path = str(PROJECT_ROOT / "data" / "checkerboard.png")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, page)

    print(f"Checkerboard saved to: {output_path}")
    print(f"  Paper: {paper_key} ({orientation}), {page_w_mm:.0f} x {page_h_mm:.0f} mm")
    print(f"  Inner corners: {cols} x {rows}")
    print(f"  Square size: {square_size_mm} mm")
    print(f"  Board dimensions: {board_squares_x * square_size_mm:.0f} x "
          f"{board_squares_y * square_size_mm:.0f} mm")
    print(f"\nPrint this image at 100% scale (no scaling/fit-to-page).")
    print("Place it flat on the desk within camera view.")

    return output_path


def _convert_color_frame_to_bgr(color_frame, color_format_name: str):
    """Convert RealSense color frame to OpenCV BGR image."""
    color_image = np.asanyarray(color_frame.get_data())

    if color_format_name == "rgb8":
        return cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

    if color_format_name == "yuyv":
        if color_image.ndim == 2:
            if color_image.dtype == np.uint16:
                color_image = color_image.view(np.uint8).reshape(
                    color_image.shape[0], color_image.shape[1], 2
                )
            else:
                if color_image.shape[1] % 2 != 0:
                    raise RuntimeError(f"Unexpected YUYV shape: {color_image.shape}")
                color_image = color_image.reshape(
                    color_image.shape[0], color_image.shape[1] // 2, 2
                )
        elif color_image.ndim == 3 and color_image.shape[2] == 1:
            w2 = color_image.shape[1]
            if w2 % 2 != 0:
                raise RuntimeError(f"Unexpected YUYV shape: {color_image.shape}")
            color_image = color_image.reshape(color_image.shape[0], w2 // 2, 2)
        elif color_image.ndim != 3 or color_image.shape[2] != 2:
            raise RuntimeError(f"Unsupported YUYV frame shape: {color_image.shape}")

        return cv2.cvtColor(color_image, cv2.COLOR_YUV2BGR_YUY2)

    # bgr8 arrives in OpenCV-ready BGR order.
    return color_image


def _detect_checkerboard(gray: np.ndarray, board_cols: int, board_rows: int):
    """Detect checkerboard corners with robust multi-scale fallback."""
    pattern_size = (board_cols, board_rows)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # More robust than classic detector for perspective/light variation.
    sb_flags = (
        getattr(cv2, "CALIB_CB_NORMALIZE_IMAGE", 0)
        | getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0)
        | getattr(cv2, "CALIB_CB_ACCURACY", 0)
    )
    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    # If board is small in the frame, upscaling helps corner localization.
    for scale in (1.0, 1.5, 2.0):
        if scale == 1.0:
            work = gray
        else:
            work = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        found = False
        corners = None

        if hasattr(cv2, "findChessboardCornersSB"):
            found, corners = cv2.findChessboardCornersSB(
                work, pattern_size, flags=sb_flags
            )
            if found and corners is not None:
                corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)

        if (not found) or corners is None:
            found, corners = cv2.findChessboardCorners(
                work, pattern_size, flags=classic_flags
            )
            if found and corners is not None:
                corners = cv2.cornerSubPix(work, corners, (11, 11), (-1, -1), criteria)

        if found and corners is not None:
            if scale != 1.0:
                corners = corners / scale
            return True, corners.astype(np.float32)

    return False, None


def _start_color_stream_with_fallback(rs_lib, camera_serial: str | None = None):
    """Start RealSense color stream, prioritizing 640x480@30 with retries."""
    profile_candidates = [
        # Match tools/realtime_view.py primary profile exactly.
        {
            "w": 640,
            "h": 480,
            "fps": 30,
            "fmt": rs_lib.format.rgb8,
            "name": "rgb8",
            "enable_depth": True,
            "retries": 8,
        },
        # Fallbacks mirror realtime_view profile ladder.
        {
            "w": 640,
            "h": 480,
            "fps": 15,
            "fmt": rs_lib.format.rgb8,
            "name": "rgb8",
            "enable_depth": True,
            "retries": 2,
        },
        {
            "w": 424,
            "h": 240,
            "fps": 15,
            "fmt": rs_lib.format.rgb8,
            "name": "rgb8",
            "enable_depth": True,
            "retries": 2,
        },
        {
            "w": 424,
            "h": 240,
            "fps": 15,
            "fmt": rs_lib.format.yuyv,
            "name": "yuyv",
            "enable_depth": True,
            "retries": 2,
        },
    ]

    attempted = []
    for candidate in profile_candidates:
        retries = int(candidate.get("retries", 1))
        for attempt_idx in range(1, retries + 1):
            pipeline = rs_lib.pipeline()
            config = rs_lib.config()
            if camera_serial:
                config.enable_device(camera_serial)
            if candidate.get("enable_depth"):
                config.enable_stream(
                    rs_lib.stream.depth,
                    candidate["w"],
                    candidate["h"],
                    rs_lib.format.z16,
                    candidate["fps"],
                )
            config.enable_stream(
                rs_lib.stream.color,
                candidate["w"],
                candidate["h"],
                candidate["fmt"],
                candidate["fps"],
            )

            label = (
                f"{candidate['w']}x{candidate['h']}@{candidate['fps']} "
                f"{candidate['name']} depth={'on' if candidate.get('enable_depth') else 'off'}"
            )

            try:
                pipeline.start(config)
                for _ in range(10):
                    pipeline.wait_for_frames(timeout_ms=10000)
                frames = pipeline.wait_for_frames(timeout_ms=10000)
                if not frames.get_color_frame():
                    raise RuntimeError("No color frame after warmup")

                chosen = dict(candidate)
                chosen["attempt"] = attempt_idx
                chosen["attempts_total"] = retries
                return pipeline, chosen
            except Exception as e:
                attempted.append(f"{label} attempt {attempt_idx}/{retries}: {e}")
                try:
                    pipeline.stop()
                except Exception:
                    pass
                time.sleep(0.15)

    detail = "\n  - ".join(attempted)
    raise RuntimeError(
        "Could not start RealSense color stream with any known profile.\n"
        f"  - {detail}"
    )


def _build_checkerboard_object_points(board_rows: int, board_cols: int, square_mm: float) -> np.ndarray:
    """Build checkerboard corner coordinates in checkerboard frame (mm)."""
    objp = np.zeros((board_rows * board_cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    objp *= float(square_mm)
    return objp


def _parse_corner_input(text: str, board_rows: int, board_cols: int):
    """Parse user corner input as col,row and return (idx, col, row)."""
    raw = text.strip().lower()
    if raw in {"q", "quit", "exit"}:
        return None

    compact = raw.replace(" ", "")
    parts = compact.split(",")
    if len(parts) != 2:
        raise ValueError("Use format col,row (example: 0,0)")

    col = int(parts[0])
    row = int(parts[1])
    if not (0 <= col < board_cols and 0 <= row < board_rows):
        raise ValueError(
            f"Corner out of range. col in [0,{board_cols - 1}], row in [0,{board_rows - 1}]"
        )

    idx = row * board_cols + col
    return idx, col, row


def _estimate_rigid_transform(camera_points_mm: np.ndarray, robot_points_mm: np.ndarray):
    """Estimate rigid transform X_robot = R * X_camera + t from point pairs."""
    if camera_points_mm.shape != robot_points_mm.shape:
        raise ValueError("camera_points_mm and robot_points_mm shape mismatch")
    if camera_points_mm.shape[0] < 3:
        raise ValueError("Need at least 3 point pairs")

    c_centroid = camera_points_mm.mean(axis=0)
    r_centroid = robot_points_mm.mean(axis=0)

    c0 = camera_points_mm - c_centroid
    r0 = robot_points_mm - r_centroid

    H = c0.T @ r0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Enforce proper rotation (det=+1)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = r_centroid.reshape(3, 1) - R @ c_centroid.reshape(3, 1)
    return R, t


def _read_valid_arm_pose_mm_deg(arm_ctrl: Any,
                                retries: int = 6,
                                retry_delay_s: float = 0.08):
    """Read arm pose with retries; reject transient all-zero register reads."""
    last_pose = None
    for _ in range(max(1, retries)):
        try:
            pose_raw = arm_ctrl.read_current_pose()
        except Exception:
            pose_raw = None

        if pose_raw is None or len(pose_raw) < 6:
            time.sleep(retry_delay_s)
            continue

        last_pose = pose_raw
        # Some Modbus read failures are surfaced as [0,0,0,0,0,0].
        if abs(int(pose_raw[0])) + abs(int(pose_raw[1])) + abs(int(pose_raw[2])) == 0:
            time.sleep(retry_delay_s)
            continue

        return pose_raw, [v / 1000.0 for v in pose_raw], False

    if last_pose is not None:
        return last_pose, [v / 1000.0 for v in last_pose], True
    return None, None, True


def collect_point_based_data(camera_serial: str | None = None,
                             num_points: int = 20,
                             modbus_host: str = "127.0.0.1",
                             modbus_port: int = 1502,
                             preview: bool = False):
    """
    Collect point-pair calibration data for fixed-camera + fixed-board setups.

    At each sample:
      1) user places TCP on a known checkerboard corner
      2) capture image and detect checkerboard
      3) read robot TCP pose
      4) derive that corner's 3D camera coordinate via solvePnP
    """
    cam_cfg = load_config("camera_calibration.yaml")
    board_rows = int(cam_cfg["checkerboard"]["rows"])
    board_cols = int(cam_cfg["checkerboard"]["cols"])
    square_mm = float(cam_cfg["checkerboard"]["square_size_mm"])
    objp = _build_checkerboard_object_points(board_rows, board_cols, square_mm)

    print(f"\n{'='*60}")
    print("Point-Based Camera-to-Robot Calibration Data Collection")
    print(f"{'='*60}")
    print(f"Checkerboard: {board_cols}x{board_rows} corners, {square_mm}mm squares")
    print(f"Target samples: {num_points}")
    print("\nFor each sample:")
    print("  1. Move TCP tip to a checkerboard inner corner")
    print("  2. Keep board fully visible")
    print("  3. Capture frame")
    print("  4. Enter corner index as col,row (example: 0,0 top-left)")
    if preview:
        print("  Preview keys: c=capture, r=reconnect arm, q=quit")
    print()

    # Connect camera
    try:
        import pyrealsense2 as rs_lib
    except ImportError:
        print("ERROR: pyrealsense2 not installed.")
        return []

    try:
        pipeline, stream_profile = _start_color_stream_with_fallback(rs_lib, camera_serial)
        print(
            "Camera stream ready: "
            f"{stream_profile['w']}x{stream_profile['h']}@{stream_profile['fps']} "
            f"{stream_profile['name']} depth={'on' if stream_profile.get('enable_depth') else 'off'} "
            f"(attempt {stream_profile.get('attempt', 1)}/{stream_profile.get('attempts_total', 1)})"
        )
    except Exception as e:
        print(f"ERROR: {e}")
        return []

    arm: Any | None = None
    arm_status = "not connected"
    arm_pose_mm_deg_live: list[float] | None = None
    arm_pose_live_ts = 0.0

    def ensure_arm_connected(verbose: bool = True) -> bool:
        nonlocal arm, arm_status
        if arm is not None:
            arm_status = "connected"
            return True
        try:
            from src.controller import ArmController
            arm = ArmController(host=modbus_host, port=modbus_port)
            if not arm.connect():
                if verbose:
                    print("  Cannot connect to arm yet. Check tunnel/connection.")
                arm = None
                arm_status = "not connected"
                return False
            if verbose:
                print("  Arm connected.")
            arm_status = "connected"
            return True
        except Exception as e:
            if verbose:
                print(f"  Cannot connect to arm yet: {e}")
            arm = None
            arm_status = "not connected"
            return False

    def reconnect_arm() -> bool:
        nonlocal arm, arm_status
        if arm is not None:
            try:
                arm.disconnect()
            except Exception:
                pass
        arm = None
        arm_status = "not connected"
        return ensure_arm_connected()

    def poll_arm_pose_live() -> None:
        nonlocal arm_pose_mm_deg_live, arm_pose_live_ts, arm_status
        if not ensure_arm_connected(verbose=False):
            return
        arm_ctrl = cast(Any, arm)
        if arm_ctrl is None:
            arm_status = "not connected"
            return
        try:
            pose_raw = arm_ctrl.read_current_pose()
        except Exception:
            arm_status = "read error"
            return
        if pose_raw is None or len(pose_raw) < 6:
            arm_status = "connected (no pose)"
            return
        arm_pose_mm_deg_live = [v / 1000.0 for v in pose_raw]
        arm_pose_live_ts = time.time()
        arm_status = "connected"

    def render_arm_monitor() -> np.ndarray:
        panel = np.zeros((300, 430, 3), dtype=np.uint8)
        panel[:] = (20, 20, 20)
        cv2.putText(panel, "Arm Monitor", (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        status_color = (0, 220, 0) if arm_status.startswith("connected") else (0, 180, 255)
        cv2.putText(panel, f"Status: {arm_status}", (16, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        if arm_pose_mm_deg_live is None:
            cv2.putText(panel, "Pose: unavailable", (16, 98),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        else:
            labels = ["X", "Y", "Z", "RX", "RY", "RZ"]
            units = ["mm", "mm", "mm", "deg", "deg", "deg"]
            y = 100
            for idx, (label, unit) in enumerate(zip(labels, units)):
                value = arm_pose_mm_deg_live[idx]
                cv2.putText(panel, f"{label:>2}: {value:8.2f} {unit}", (16, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
                y += 28
            age_s = time.time() - arm_pose_live_ts
            cv2.putText(panel, f"Last update: {age_s:.2f}s ago", (16, 282),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        return panel

    data = []
    out_dir = PROJECT_ROOT / "data" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_window = "Point Calibration Preview"
    arm_monitor_window = "Arm Monitor"
    arm_poll_interval_s = 0.25
    last_arm_poll_ts = 0.0

    user_quit = False
    for i in range(num_points):
        print(f"\n--- Point Sample {i+1}/{num_points} ---")
        color_image = None
        color_frame = None
        corners = None

        if preview:
            print("Move TCP to target checkerboard corner, then press 'c' in preview window.")
            while True:
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=10000)
                except RuntimeError as e:
                    print(f"  Frame timeout: {e}. Try again.")
                    continue

                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                try:
                    frame_bgr = _convert_color_frame_to_bgr(color_frame, stream_profile["name"])
                except Exception:
                    continue

                now_ts = time.time()
                if now_ts - last_arm_poll_ts >= arm_poll_interval_s:
                    poll_arm_pose_live()
                    last_arm_poll_ts = now_ts

                preview_img = frame_bgr.copy()
                gray_prev = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                found_prev, corners_prev = _detect_checkerboard(gray_prev, board_cols, board_rows)
                if found_prev and corners_prev is not None:
                    cv2.drawChessboardCorners(preview_img, (board_cols, board_rows), corners_prev, True)
                    status_text = "Checkerboard: detected"
                    status_color = (0, 200, 0)
                else:
                    status_text = "Checkerboard: not detected"
                    status_color = (0, 0, 255)

                cv2.putText(preview_img,
                            f"Sample {i+1}/{num_points} | c=capture r=reconnect q=quit",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(preview_img, status_text, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                try:
                    cv2.imshow(preview_window, preview_img)
                    cv2.imshow(arm_monitor_window, render_arm_monitor())
                except cv2.error as e:
                    print(f"  Preview disabled: {e}")
                    preview = False
                    break

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    user_quit = True
                    break
                if key == ord('r'):
                    ok = reconnect_arm()
                    print("  Manual reconnect: arm connected." if ok else "  Manual reconnect: arm still not connected.")
                    continue
                if key == ord('c'):
                    if not found_prev or corners_prev is None:
                        print("  Capture rejected: checkerboard not detected.")
                        continue
                    color_image = frame_bgr
                    corners = corners_prev
                    break

            if user_quit:
                break
            if color_image is None or corners is None:
                continue
        else:
            print("Press Enter to capture frame (or 'q' to finish).")
            user = input("> ").strip().lower()
            if user == 'q':
                break
            try:
                frames = pipeline.wait_for_frames(timeout_ms=10000)
            except RuntimeError as e:
                print(f"  Frame timeout: {e}. Try again.")
                continue
            color_frame = frames.get_color_frame()
            if not color_frame:
                print("  No frame captured. Try again.")
                continue
            color_image = _convert_color_frame_to_bgr(color_frame, stream_profile["name"])
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            ret, corners = _detect_checkerboard(gray, board_cols, board_rows)
            if not ret or corners is None:
                print("  Checkerboard NOT detected. Try again.")
                continue

        # Ask which checkerboard corner the TCP is touching.
        while True:
            print(
                f"  Enter touched corner as col,row "
                f"(col:0..{board_cols-1}, row:0..{board_rows-1}, example 0,0 top-left), or q to stop:"
            )
            raw = input("  corner> ")
            try:
                parsed = _parse_corner_input(raw, board_rows, board_cols)
            except Exception as e:
                print(f"  Invalid input: {e}")
                continue
            if parsed is None:
                user_quit = True
                break
            corner_idx, corner_col, corner_row = parsed
            break
        if user_quit:
            break

        if not ensure_arm_connected():
            print("  Skip this sample: arm is not connected yet.")
            continue
        arm_ctrl = cast(Any, arm)
        if arm_ctrl is None:
            print("  Skip this sample: arm handle unavailable.")
            continue
        arm_pose, arm_pose_mm, likely_invalid = _read_valid_arm_pose_mm_deg(arm_ctrl)
        if arm_pose is None or arm_pose_mm is None:
            print("  Skip this sample: arm returned no pose.")
            continue
        if likely_invalid:
            print("  Skip this sample: arm pose read looks invalid (likely transient Modbus read as zeros).")
            print("  Tip: wait 1 second and capture again. If persistent, press 'r' to reconnect arm.")
            continue
        robot_point_mm = np.array(arm_pose_mm[:3], dtype=np.float64)

        # Compute checkerboard pose in camera frame.
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        camera_matrix = np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.array(intr.coeffs, dtype=np.float64)
        ret_pnp, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
        if not ret_pnp:
            print("  PnP solve failed. Try again.")
            continue

        R_tc, _ = cv2.Rodrigues(rvec)
        target_corner_mm = objp[corner_idx].reshape(3, 1).astype(np.float64)
        camera_point_mm = (R_tc @ target_corner_mm + tvec.reshape(3, 1)).flatten()

        entry = {
            "sample_index": i,
            "corner_index": int(corner_idx),
            "corner_col": int(corner_col),
            "corner_row": int(corner_row),
            "camera_point_mm": camera_point_mm.tolist(),
            "robot_point_mm": robot_point_mm.tolist(),
            "arm_pose_raw": arm_pose,
            "arm_pose_mm_deg": arm_pose_mm,
            "rvec": rvec.flatten().tolist(),
            "tvec": tvec.flatten().tolist(),
        }
        data.append(entry)

        print(f"  Corner ({corner_col},{corner_row})")
        print(f"  Camera point (mm): {[round(v, 2) for v in camera_point_mm.tolist()]}")
        print(f"  Robot point  (mm): {[round(v, 2) for v in robot_point_mm.tolist()]}")
        print(f"  Saved! ({len(data)} samples)")

    if preview:
        try:
            cv2.destroyWindow(arm_monitor_window)
        except cv2.error:
            pass
        cv2.destroyAllWindows()

    pipeline.stop()
    if arm is not None:
        arm.disconnect()

    data_path = out_dir / "point_calibration_data.json"
    with open(data_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nCollected {len(data)} point samples.")
    print(f"Data saved to: {data_path}")
    return data


def compute_point_based_calibration(data: list | None = None, camera_id: int = 1):
    """Compute camera->robot transform from 3D point correspondences."""
    def mark_calibration_failed(reason: str, error_mm: float | None = None):
        cam_cfg = load_config("camera_calibration.yaml")
        key = f"camera{camera_id}_to_robot"
        cur = dict(cam_cfg["hand_eye"].get(key, {}))
        cur["calibrated"] = False
        cur["calibration_error_mm"] = None if error_mm is None else round(float(error_mm), 2)
        cur["calibration_date"] = time.strftime("%Y-%m-%d %H:%M:%S")
        cur["failure_reason"] = reason
        cur["method"] = "point-based"
        cam_cfg["hand_eye"][key] = cur
        save_config("camera_calibration.yaml", cam_cfg)
        print("Point-based calibration marked as FAILED in config/camera_calibration.yaml")

    if data is None:
        data_path = PROJECT_ROOT / "data" / "calibration" / "point_calibration_data.json"
        try:
            with open(data_path) as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"ERROR: Point data file not found: {data_path}")
            mark_calibration_failed("Point data file not found")
            return None

    if not data or len(data) < 4:
        reason = f"Need at least 4 point samples, got {0 if not data else len(data)}."
        print(f"ERROR: {reason}")
        mark_calibration_failed(reason)
        return None

    camera_pts = np.array([e["camera_point_mm"] for e in data], dtype=np.float64)
    robot_pts = np.array([e["robot_point_mm"] for e in data], dtype=np.float64)

    # Basic observability check.
    cam_span = np.ptp(camera_pts, axis=0)
    if float(np.linalg.norm(cam_span)) < 30.0:
        reason = "Camera points have too little spread. Use corners across the whole checkerboard."
        print(f"ERROR: {reason}")
        mark_calibration_failed(reason)
        return None

    R, t = _estimate_rigid_transform(camera_pts, robot_pts)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()

    pred = (R @ camera_pts.T + t).T
    errs = np.linalg.norm(pred - robot_pts, axis=1)
    mean_error = float(np.mean(errs))
    max_error = float(np.max(errs))

    print(f"\nPoint-based calibration matrix (camera{camera_id} -> robot base):")
    print(T)
    print("\nPoint-fit error:")
    print(f"  Mean: {mean_error:.2f} mm")
    print(f"  Max:  {max_error:.2f} mm")

    if mean_error > 80.0:
        print("WARNING: Mean error is high. Re-collect with better TCP corner touching and wider corner spread.")

    if np.allclose(T, np.eye(4), atol=1e-6) or mean_error > 200.0:
        reason = (
            "Point-based result rejected: "
            f"identity={bool(np.allclose(T, np.eye(4), atol=1e-6))}, "
            f"mean_error_mm={mean_error:.2f}."
        )
        print(f"ERROR: {reason}")
        mark_calibration_failed(reason, error_mm=mean_error)
        return None

    cam_cfg = load_config("camera_calibration.yaml")
    key = f"camera{camera_id}_to_robot"
    cur = dict(cam_cfg["hand_eye"].get(key, {}))
    cur.pop("failure_reason", None)
    cam_cfg["hand_eye"][key] = {
        **cur,
        "matrix": T.flatten().tolist(),
        "calibrated": True,
        "calibration_error_mm": round(mean_error, 2),
        "calibration_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "point-based",
    }
    save_config("camera_calibration.yaml", cam_cfg)
    print("\nSaved point-based result to config/camera_calibration.yaml")
    return T


def collect_calibration_data(camera_serial: str | None = None,
                             num_poses: int = 15,
                             modbus_host: str = "127.0.0.1",
                             modbus_port: int = 1502,
                             preview: bool = False):
    """
    Collect calibration data pairs: (camera checkerboard corners, arm pose).

    Interactive process:
    1. Move arm to a position near the checkerboard
    2. Press Enter to capture
    3. Script records camera image + arm position
    4. Repeat num_poses times

    Returns:
        List of dicts with camera corners and arm poses.
    """
    cam_cfg = load_config("camera_calibration.yaml")
    board_rows = cam_cfg["checkerboard"]["rows"]
    board_cols = cam_cfg["checkerboard"]["cols"]
    square_mm = cam_cfg["checkerboard"]["square_size_mm"]

    # Prepare object points (3D points in checkerboard frame)
    objp = np.zeros((board_rows * board_cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    objp *= square_mm  # In mm

    print(f"\n{'='*60}")
    print("Hand-Eye Calibration Data Collection")
    print(f"{'='*60}")
    print(f"Checkerboard: {board_cols}x{board_rows} corners, {square_mm}mm squares")
    print(f"Target poses: {num_poses}")
    print(f"\nInstructions:")
    print(f"  1. Place the printed checkerboard flat on the desk")
    print(f"  2. Make sure the camera can see the entire board")
    print(f"  3. Move the arm to different positions near the board")
    print(f"  4. Press Enter to capture at each position")
    print(f"  5. Try to cover the workspace: center, corners, different heights")
    print(f"\nTips for good calibration:")
    print(f"  - Space poses evenly across the workspace")
    print(f"  - Include different arm orientations")
    print(f"  - Don't move too fast between captures")
    print(f"  - Keep the board visible and flat")
    if preview:
        print(f"  - Preview mode: press 'c' to capture, 'r' to reconnect arm, 'q' to quit")
        print(f"  - Separate Arm Monitor window shows live X/Y/Z/RX/RY/RZ")
    print()

    # Connect camera
    try:
        import pyrealsense2 as rs_lib
    except ImportError:
        print("ERROR: pyrealsense2 not installed.")
        return []

    try:
        pipeline, stream_profile = _start_color_stream_with_fallback(rs_lib, camera_serial)
        print(
            "Camera stream ready: "
            f"{stream_profile['w']}x{stream_profile['h']}@{stream_profile['fps']} "
            f"{stream_profile['name']} depth={'on' if stream_profile.get('enable_depth') else 'off'} "
            f"(attempt {stream_profile.get('attempt', 1)}/{stream_profile.get('attempts_total', 1)})"
        )
    except Exception as e:
        print(f"ERROR: {e}")
        print("Tip: confirm the device appears in `bash scripts/preflight_ubuntu.sh`.")
        return []

    arm: Any | None = None
    arm_status = "not connected"
    arm_pose_mm_deg_live: list[float] | None = None
    arm_pose_live_ts = 0.0

    def ensure_arm_connected(verbose: bool = True) -> bool:
        """Connect arm lazily so preview can be used before Modbus is ready."""
        nonlocal arm, arm_status
        if arm is not None:
            arm_status = "connected"
            return True

        try:
            from src.controller import ArmController
            arm = ArmController(host=modbus_host, port=modbus_port)
            if not arm.connect():
                if verbose:
                    print("  Cannot connect to arm yet. Check tunnel/connection.")
                arm = None
                arm_status = "not connected"
                return False
            if verbose:
                print("  Arm connected.")
            arm_status = "connected"
            return True
        except Exception as e:
            if verbose:
                print(f"  Cannot connect to arm yet: {e}")
            arm = None
            arm_status = "not connected"
            return False

    def reconnect_arm() -> bool:
        """Force reconnect arm controller and update status immediately."""
        nonlocal arm, arm_status
        if arm is not None:
            try:
                arm.disconnect()
            except Exception:
                pass
        arm = None
        arm_status = "not connected"
        return ensure_arm_connected()

    def poll_arm_pose_live() -> None:
        """Read arm pose for monitor window without noisy terminal logs."""
        nonlocal arm_pose_mm_deg_live, arm_pose_live_ts, arm_status
        if not ensure_arm_connected(verbose=False):
            return

        arm_ctrl = cast(Any, arm)
        if arm_ctrl is None:
            arm_status = "not connected"
            return

        try:
            pose_raw = arm_ctrl.read_current_pose()
        except Exception:
            arm_status = "read error"
            return

        if pose_raw is None or len(pose_raw) < 6:
            arm_status = "connected (no pose)"
            return

        arm_pose_mm_deg_live = [v / 1000.0 for v in pose_raw]
        arm_pose_live_ts = time.time()
        arm_status = "connected"

    def render_arm_monitor() -> np.ndarray:
        """Render arm telemetry as a separate panel image."""
        panel = np.zeros((300, 430, 3), dtype=np.uint8)
        panel[:] = (20, 20, 20)

        cv2.putText(panel, "Arm Monitor", (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        status_color = (0, 220, 0) if arm_status.startswith("connected") else (0, 180, 255)
        cv2.putText(panel, f"Status: {arm_status}", (16, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        if arm_pose_mm_deg_live is None:
            cv2.putText(panel, "Pose: unavailable", (16, 98),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        else:
            labels = ["X", "Y", "Z", "RX", "RY", "RZ"]
            units = ["mm", "mm", "mm", "deg", "deg", "deg"]
            y = 100
            for idx, (label, unit) in enumerate(zip(labels, units)):
                value = arm_pose_mm_deg_live[idx]
                cv2.putText(panel, f"{label:>2}: {value:8.2f} {unit}", (16, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
                y += 28

            age_s = time.time() - arm_pose_live_ts
            cv2.putText(panel, f"Last update: {age_s:.2f}s ago", (16, 282),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        return panel

    # Collect data
    data = []
    out_dir = PROJECT_ROOT / "data" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_window = "Calibration Preview"
    arm_monitor_window = "Arm Monitor"
    arm_poll_interval_s = 0.25
    last_arm_poll_ts = 0.0

    for i in range(num_poses):
        print(f"\n--- Pose {i+1}/{num_poses} ---")
        color_image = None
        color_frame = None

        if preview:
            print("Move arm to a new position, then press 'c' in preview window to capture.")
            print("(press 'q' in preview window to finish early)")
            while True:
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=10000)
                except RuntimeError as e:
                    print(f"  Frame timeout: {e}. Try again.")
                    continue

                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                try:
                    frame_bgr = _convert_color_frame_to_bgr(color_frame, stream_profile["name"])
                except Exception:
                    continue

                now_ts = time.time()
                if now_ts - last_arm_poll_ts >= arm_poll_interval_s:
                    poll_arm_pose_live()
                    last_arm_poll_ts = now_ts

                preview_img = frame_bgr.copy()
                gray_prev = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                found_prev, corners_prev = _detect_checkerboard(
                    gray_prev, board_cols, board_rows
                )
                if found_prev and corners_prev is not None:
                    cv2.drawChessboardCorners(
                        preview_img, (board_cols, board_rows), corners_prev, True
                    )
                    pts_prev = corners_prev.reshape(-1, 2)
                    x_prev, y_prev, w_prev, h_prev = cv2.boundingRect(pts_prev.astype(np.int32))
                    min_side_px = min(w_prev, h_prev)
                    if min_side_px < 120:
                        status_text = f"Checkerboard: detected (small: {min_side_px}px)"
                        status_color = (0, 170, 255)
                    else:
                        status_text = "Checkerboard: detected"
                        status_color = (0, 200, 0)
                else:
                    status_text = "Checkerboard: not detected"
                    status_color = (0, 0, 255)

                cv2.putText(
                    preview_img,
                    f"Pose {i+1}/{num_poses} | c=capture r=reconnect q=quit",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    preview_img,
                    status_text,
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    status_color,
                    2,
                )

                arm_text = f"Arm: {arm_status}"
                arm_color = (0, 200, 0) if arm_status == "connected" else (0, 170, 255)
                cv2.putText(
                    preview_img,
                    arm_text,
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    arm_color,
                    2,
                )

                try:
                    cv2.imshow(preview_window, preview_img)
                    cv2.imshow(arm_monitor_window, render_arm_monitor())
                except cv2.error as e:
                    print(f"  Preview disabled: {e}")
                    preview = False
                    break

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    color_image = None
                    user_quit = True
                    break
                if key == ord('r'):
                    ok = reconnect_arm()
                    if ok:
                        print("  Manual reconnect: arm connected.")
                    else:
                        print("  Manual reconnect: arm still not connected.")
                    continue
                if key == ord('c'):
                    color_image = frame_bgr
                    color_frame = frames.get_color_frame()
                    user_quit = False
                    break

            if preview and 'user_quit' in locals() and user_quit:
                break
            if color_image is None:
                continue
        else:
            print("Move arm to a new position, then press Enter to capture.")
            print("(or 'q' to finish early)")

            user = input("> ").strip()
            if user.lower() == 'q':
                break

            # Capture camera frame
            try:
                frames = pipeline.wait_for_frames(timeout_ms=10000)
            except RuntimeError as e:
                print(f"  Frame timeout: {e}. Try again.")
                continue

            color_frame = frames.get_color_frame()
            if not color_frame:
                print("  No frame captured. Try again.")
                continue

            try:
                color_image = _convert_color_frame_to_bgr(color_frame, stream_profile["name"])
            except Exception as e:
                print(f"  Color conversion failed: {e}. Try again.")
                continue

        # Detect checkerboard
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        ret, corners = _detect_checkerboard(gray, board_cols, board_rows)

        if not ret or corners is None:
            print("  Checkerboard NOT detected. Keep full board visible and move closer if needed.")
            # Save failed image for debugging
            cv2.imwrite(str(out_dir / f"fail_{i:03d}.jpg"), color_image)
            continue

        # Read arm position
        if not ensure_arm_connected():
            print("  Skip this pose: arm is not connected yet.")
            continue

        arm_ctrl = cast(Any, arm)
        if arm_ctrl is None:
            print("  Skip this pose: arm handle unavailable.")
            continue

        arm_pose, arm_pose_mm, likely_invalid = _read_valid_arm_pose_mm_deg(arm_ctrl)
        if arm_pose is None or arm_pose_mm is None:
            print("  Skip this pose: arm returned no pose.")
            continue
        if likely_invalid:
            print("  Skip this pose: arm pose read looks invalid (likely transient Modbus read as zeros).")
            print("  Tip: wait 1 second and capture again. If persistent, press 'r' to reconnect arm.")
            continue

        print(f"  Checkerboard detected ({len(corners)} corners)")
        print(f"  Arm pose (mm/deg): {[round(v, 1) for v in arm_pose_mm]}")

        # Solve PnP for camera pose
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        camera_matrix = np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.array(intr.coeffs, dtype=np.float64)

        ret_pnp, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)

        if not ret_pnp:
            print("  PnP solve failed. Try again.")
            continue

        # Save data point
        entry = {
            "pose_index": i,
            "arm_pose_raw": arm_pose,
            "arm_pose_mm_deg": arm_pose_mm,
            "rvec": rvec.flatten().tolist(),
            "tvec": tvec.flatten().tolist(),
            "corners": corners.reshape(-1, 2).tolist(),
        }
        data.append(entry)

        # Save annotated image
        annotated = cv2.drawChessboardCorners(
            color_image.copy(), (board_cols, board_rows), corners, True
        )
        cv2.imwrite(str(out_dir / f"pose_{i:03d}.jpg"), annotated)

        print(f"  Saved! ({len(data)} poses collected)")

    if preview:
        try:
            cv2.destroyWindow(arm_monitor_window)
        except cv2.error:
            pass
        cv2.destroyAllWindows()

    pipeline.stop()
    if arm is not None:
        arm.disconnect()

    # Save collected data
    data_path = out_dir / "calibration_data.json"
    with open(data_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nCollected {len(data)} valid poses.")
    print(f"Data saved to: {data_path}")

    return data


def compute_hand_eye(data: list | None = None, camera_id: int = 1):
    """
    Compute the hand-eye transformation matrix from collected data.

    Uses OpenCV's calibrateHandEye with the Tsai method.
    """
    if data is None:
        data_path = PROJECT_ROOT / "data" / "calibration" / "calibration_data.json"
        with open(data_path) as f:
            data = json.load(f)

    if data is None:
        print("ERROR: No calibration data provided.")
        return None

    if len(data) < 5:
        print(f"ERROR: Need at least 5 poses, got {len(data)}.")
        return None

    print(f"\nComputing hand-eye calibration from {len(data)} poses...")

    def mark_calibration_failed(reason: str, error_mm: float | None = None):
        cam_cfg = load_config("camera_calibration.yaml")
        key = f"camera{camera_id}_to_robot"
        cur = dict(cam_cfg["hand_eye"].get(key, {}))
        cur["calibrated"] = False
        cur["calibration_error_mm"] = None if error_mm is None else round(float(error_mm), 2)
        cur["calibration_date"] = time.strftime("%Y-%m-%d %H:%M:%S")
        cur["failure_reason"] = reason
        cam_cfg["hand_eye"][key] = cur
        save_config("camera_calibration.yaml", cam_cfg)
        print("\nCalibration marked as FAILED in config/camera_calibration.yaml")

    # Observability check: for calibrateHandEye, checkerboard pose in camera frame
    # must vary across captures. If camera and checkerboard are both fixed, this
    # method is not observable regardless of robot motions.
    cam_rvec_arr = np.array([entry["rvec"] for entry in data], dtype=np.float64)
    cam_tvec_arr = np.array([entry["tvec"] for entry in data], dtype=np.float64)
    cam_rvec_span = np.ptp(cam_rvec_arr, axis=0)
    cam_tvec_span = np.ptp(cam_tvec_arr, axis=0)
    cam_tvec_diag = float(np.linalg.norm(cam_tvec_span))
    cam_rot_span_max = float(np.max(np.abs(cam_rvec_span)))
    if cam_rot_span_max < 0.02 and cam_tvec_diag < 15.0:
        reason = (
            "Checkerboard pose is almost static in camera frame "
            f"(rvec_span_max={cam_rot_span_max:.4f} rad, "
            f"tvec_span_diag={cam_tvec_diag:.2f} mm). "
            "Current Tsai hand-eye setup needs target pose changes across captures. "
            "If camera and checkerboard are both fixed, switch to a point-based "
            "camera-to-robot calibration workflow."
        )
        print(f"ERROR: {reason}")
        mark_calibration_failed(reason)
        return None

    # Tsai method needs informative rotations; pure translations often fail.
    arm_pose_arr = np.array([entry["arm_pose_mm_deg"] for entry in data], dtype=np.float64)
    arm_rot_deg = arm_pose_arr[:, 3:6]
    arm_rot_unwrapped_deg = np.rad2deg(np.unwrap(np.deg2rad(arm_rot_deg), axis=0))
    angle_spans = np.ptp(arm_rot_unwrapped_deg, axis=0)
    max_angle_span = float(np.max(angle_spans))
    if max_angle_span < 8.0:
        reason = (
            "Not enough informative rotations. "
            f"Observed angle spans (deg): rx={angle_spans[0]:.2f}, "
            f"ry={angle_spans[1]:.2f}, rz={angle_spans[2]:.2f}. "
            "Collect poses with larger wrist rotations."
        )
        print(f"ERROR: {reason}")
        mark_calibration_failed(reason)
        return None

    # Extract rotation matrices and translations
    # Robot poses: gripper-to-base
    R_gripper2base = []
    t_gripper2base = []

    # Camera poses: target-to-camera (from solvePnP)
    R_target2cam = []
    t_target2cam = []

    for entry in data:
        # Camera: target → camera (from PnP)
        rvec = np.array(entry["rvec"])
        tvec = np.array(entry["tvec"])
        R_tc, _ = cv2.Rodrigues(rvec)
        R_target2cam.append(R_tc)
        t_target2cam.append(tvec.reshape(3, 1))

        # Robot: gripper → base (from arm pose)
        # For now, simplified: use XYZ as translation, identity rotation
        # A full implementation would convert RX/RY/RZ to rotation matrix
        pose_mm = entry["arm_pose_mm_deg"]
        t_gb = np.array([[pose_mm[0]], [pose_mm[1]], [pose_mm[2]]])
        # Simple rotation from euler angles (controller uses degrees * 1000)
        rx_deg = pose_mm[3]
        ry_deg = pose_mm[4]
        rz_deg = pose_mm[5]
        R_gb = euler_to_rotation_matrix(rx_deg, ry_deg, rz_deg)
        R_gripper2base.append(R_gb)
        t_gripper2base.append(t_gb)

    # Compute hand-eye calibration (eye-to-hand: camera is fixed)
    try:
        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI
        )
    except cv2.error as e:
        reason = f"OpenCV calibrateHandEye failed: {e}"
        print(f"ERROR: {reason}")
        mark_calibration_failed(reason)
        return None

    # Build 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R_cam2base
    T[:3, 3] = t_cam2base.flatten()

    print(f"Calibration matrix (camera{camera_id} → robot base):")
    print(T)

    # Compute reprojection error
    errors = []
    for entry in data:
        rvec = np.array(entry["rvec"])
        tvec = np.array(entry["tvec"]).reshape(3, 1)
        R_tc, _ = cv2.Rodrigues(rvec)
        # Camera coords of board origin
        cam_point = tvec.flatten()
        # Transform to robot coords
        robot_point = (R_cam2base @ cam_point.reshape(3, 1) + t_cam2base).flatten()
        # Compare with actual arm position
        arm_xyz = np.array(entry["arm_pose_mm_deg"][:3])
        error = np.linalg.norm(robot_point - arm_xyz)
        errors.append(error)

    mean_error = float(np.mean(errors))
    max_error = float(np.max(errors))
    print(f"\nCalibration error:")
    print(f"  Mean: {mean_error:.2f} mm")
    print(f"  Max:  {max_error:.2f} mm")

    # Guard against obvious failed solutions that still return a matrix.
    if np.allclose(T, np.eye(4), atol=1e-6) or mean_error > 200.0:
        reason = (
            "Calibration result rejected: "
            f"identity={bool(np.allclose(T, np.eye(4), atol=1e-6))}, "
            f"mean_error_mm={mean_error:.2f}."
        )
        print(f"ERROR: {reason}")
        mark_calibration_failed(reason, error_mm=mean_error)
        return None

    # Save to config
    cam_cfg = load_config("camera_calibration.yaml")
    key = f"camera{camera_id}_to_robot"
    cur = dict(cam_cfg["hand_eye"].get(key, {}))
    cur.pop("failure_reason", None)
    cam_cfg["hand_eye"][key] = {
        **cur,
        "matrix": T.flatten().tolist(),
        "calibrated": True,
        "calibration_error_mm": round(mean_error, 2),
        "calibration_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_config("camera_calibration.yaml", cam_cfg)
    print(f"\nSaved to config/camera_calibration.yaml")

    return T


def euler_to_rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float):
    """Convert Euler angles (degrees) to 3x3 rotation matrix."""
    rx = np.radians(rx_deg)
    ry = np.radians(ry_deg)
    rz = np.radians(rz_deg)

    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])

    return Rz @ Ry @ Rx


def main():
    parser = argparse.ArgumentParser(
        description="Hand-eye calibration tool for RealSense + robot arm."
    )
    parser.add_argument("--generate-board", action="store_true",
                        help="Generate a checkerboard image for printing")
    parser.add_argument("--paper-size", type=str, default="A3", choices=["A4", "A3"],
                        help="Paper size for checkerboard output (default: A3)")
    parser.add_argument("--portrait", action="store_true",
                        help="Use portrait orientation (default: landscape)")
    parser.add_argument("--rows", type=int, default=None,
                        help="Override checkerboard rows (inner corners)")
    parser.add_argument("--cols", type=int, default=None,
                        help="Override checkerboard cols (inner corners)")
    parser.add_argument("--square-mm", type=float, default=None,
                        help="Override checkerboard square size in mm")
    parser.add_argument("--run", action="store_true",
                        help="Run the calibration data collection")
    parser.add_argument("--run-point", action="store_true",
                        help="Run point-based calibration for fixed camera + fixed board")
    parser.add_argument("--compute", action="store_true",
                        help="Compute calibration from collected data")
    parser.add_argument("--compute-point", action="store_true",
                        help="Compute point-based calibration from saved point data")
    parser.add_argument("--camera", type=int, default=1, choices=[1, 2],
                        help="Which camera to calibrate (1 or 2)")
    parser.add_argument("--camera-serial", type=str, default=None,
                        help="Camera serial number (auto-detect if not set)")
    parser.add_argument("--poses", type=int, default=15,
                        help="Number of calibration poses to collect (default: 15)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Modbus host (default: 127.0.0.1 for tunnel)")
    parser.add_argument("--port", type=int, default=1502,
                        help="Modbus port (default: 1502 for tunnel)")
    parser.add_argument("--preview", action="store_true",
                        help="Show live camera preview during capture (press c to capture)")
    args = parser.parse_args()

    if args.generate_board:
        cam_cfg = load_config("camera_calibration.yaml")
        cb = cam_cfg["checkerboard"]
        rows = args.rows if args.rows is not None else cb["rows"]
        cols = args.cols if args.cols is not None else cb["cols"]
        square_mm = args.square_mm if args.square_mm is not None else cb["square_size_mm"]
        try:
            generate_checkerboard(
                rows=rows,
                cols=cols,
                square_size_mm=square_mm,
                paper_size=args.paper_size,
                landscape=not args.portrait,
            )
        except ValueError as e:
            print(f"ERROR: {e}")
        return

    if args.run:
        data = collect_calibration_data(
            camera_serial=args.camera_serial,
            num_poses=args.poses,
            modbus_host=args.host,
            modbus_port=args.port,
            preview=args.preview,
        )
        if len(data) >= 5:
            T = compute_hand_eye(data, camera_id=args.camera)
            if T is None:
                print("Calibration failed. See failure_reason in config/camera_calibration.yaml")
        else:
            print(f"Only {len(data)} poses collected. Need at least 5.")
            print("Run again with more poses, then use --compute.")
        return

    if args.run_point:
        data = collect_point_based_data(
            camera_serial=args.camera_serial,
            num_points=args.poses,
            modbus_host=args.host,
            modbus_port=args.port,
            preview=args.preview,
        )
        if len(data) >= 4:
            T = compute_point_based_calibration(data, camera_id=args.camera)
            if T is None:
                print("Point-based calibration failed.")
        else:
            print(f"Only {len(data)} point samples collected. Need at least 4.")
            print("Run again with more samples, then use --compute-point.")
        return

    if args.compute:
        T = compute_hand_eye(camera_id=args.camera)
        if T is None:
            print("Calibration failed. See failure_reason in config/camera_calibration.yaml")
        return

    if args.compute_point:
        T = compute_point_based_calibration(camera_id=args.camera)
        if T is None:
            print("Point-based calibration failed.")
        return

    # Default: show help
    parser.print_help()
    print("\n\nQuick start:")
    print("  1. Generate checkerboard:  python tools/calibrate.py --generate-board --paper-size A3")
    print("  2. Print the checkerboard PNG at 100% scale")
    print("  3. Place it flat on the desk within camera view")
    print("  4. Hand-eye mode:          python tools/calibrate.py --run --camera 1")
    print("  5. Point-based mode:       python tools/calibrate.py --run-point --camera 1 --preview")
    print("  6. Repeat for camera 2 with --camera 2")


if __name__ == "__main__":
    main()
