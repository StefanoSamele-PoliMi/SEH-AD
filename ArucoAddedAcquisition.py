import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json
import threading
import sys
import select
from datetime import datetime

# --- 1. Basic Configuration ---
CONFIG_FILE = "config_default.json"
IMAGE_DIR = "imgs/acquisitions"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR, exist_ok=True)

# Adjustable crop parameters
PADDING_SIZE = 0.01   # Physical margin (in meters) added around the target object relative to the original image coordinates.
VIEW_WIDTH = 300      # The output image width in pixels, which defines the scale relative to the physical width of the box.

# Physical parameters (in meters) (ArUco localization and cropping)
MARKER_SIZE = 0.06
BOX_W = 0.063
BOX_H = 0.183
BOX_T = 0.025
OFFSET_X = 0.05
DIST_Y = BOX_H - MARKER_SIZE


# --- 2. Core Algorithms ---

def compensate_green(image):
    """Green channel color compensation"""
    matrix = np.array([[0.96, 0, 0], [0, 1.06, 0], [0, 0, 0.96]])
    # matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    corrected = cv2.transform(image, matrix)
    return np.clip(corrected, 0, 255).astype(np.uint8)


def get_crop_view(frame, mtx, dist, padding_size=PADDING_SIZE):
    """
    Detect ArUco markers and generate a 2D top-down view
    :param padding_size: Edge padding distance (meters), increasing this value expands crop range
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(
        aruco_dict,
        cv2.aruco.DetectorParameters()
    )

    corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None and len(ids) >= 2:
        id_map = {int(id): c for id, c in zip(ids.flatten(), corners)}
        if 0 in id_map and 1 in id_map:
            # 3D reference points
            m0_3d = np.array([
                [-MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
                [MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
                [MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
                [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]
            ], dtype=np.float32)

            m1_3d = m0_3d + np.array([0, DIST_Y, 0], dtype=np.float32)

            _, rvec, tvec = cv2.solvePnP(
                np.vstack((m0_3d, m1_3d)),
                np.vstack((id_map[0][0], id_map[1][0])),
                mtx,
                dist
            )

            # Define target top surface crop region using padding_size
            box_top_3d = np.array([
                [OFFSET_X - padding_size, BOX_H - MARKER_SIZE / 2 + padding_size, BOX_T],
                [OFFSET_X + BOX_W + padding_size, BOX_H - MARKER_SIZE / 2 + padding_size, BOX_T],
                [OFFSET_X + BOX_W + padding_size, -MARKER_SIZE / 2 - padding_size, BOX_T],
                [OFFSET_X - padding_size, -MARKER_SIZE / 2 - padding_size, BOX_T]
            ], dtype=np.float32)

            img_pts, _ = cv2.projectPoints(box_top_3d, rvec, tvec, mtx, dist)

            view_w = VIEW_WIDTH
            view_h = int(view_w * (BOX_H / BOX_W))
            dst_pts = np.array([[0, 0], [view_w, 0], [view_w, view_h], [0, view_h]], dtype=np.float32)

            M = cv2.getPerspectiveTransform(img_pts.reshape(-1, 2).astype(np.float32), dst_pts)

            return cv2.warpPerspective(
                frame,
                M,
                (view_w, view_h),
                flags=cv2.INTER_CUBIC
            )
    return None


# --- 3. Global Synchronization Variables ---
capture_event = threading.Event()
exit_event = threading.Event()
label_input = "default"


# --- 4. Camera Main Logic ---

def run_camera():
    global capture_event, exit_event, label_input

    # Load JSON configuration
    with open(CONFIG_FILE) as f:
        jsonObj = json.load(f)
    json_string = json.dumps(jsonObj)

    pipeline = rs.pipeline()
    config = rs.config()

    # Enable color stream
    config.enable_stream(
        rs.stream.color,
        int(jsonObj['viewer']['stream-width']),
        int(jsonObj['viewer']['stream-height']),
        rs.format.bgr8,
        int(jsonObj['viewer']['stream-fps'])
    )

    try:
        profile = pipeline.start(config)
        dev = profile.get_device()

        # Load Advanced Mode hardware parameters
        advnc_mode = rs.rs400_advanced_mode(dev)
        advnc_mode.load_json(json_string)

        # Get camera intrinsics
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        mtx = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
        dist = np.array(intr.coeffs)

        # Hardware warm-up
        print("Camera starting, applying hardware parameters (from JSON)...")
        for _ in range(20):
            pipeline.wait_for_frames()

        print("\n--- System Ready ---")
        print("Commands: [p] Capture, [q] Quit")

        while not exit_event.is_set():
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                print("Warning: Unable to capture image. Please check the camera connection!")
                continue

            if capture_event.is_set():
                # Capture current frame
                raw_img = np.asanyarray(color_frame.get_data())

                # Color compensation
                final_image = compensate_green(raw_img)

                # ArUco crop (call function with parameter)
                cropped_box = get_crop_view(final_image, mtx, dist, padding_size=PADDING_SIZE)

                # Save files
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                raw_name = f"RAW_{label_input}_{timestamp}.png"
                cv2.imwrite(os.path.join(IMAGE_DIR, raw_name), final_image)
                print(f"\n Raw image saved: {raw_name}")

                if cropped_box is not None:
                    crop_name = f"CROP_{label_input}_{timestamp}.png"
                    cv2.imwrite(os.path.join(IMAGE_DIR, crop_name), cropped_box)
                    print(f"[OK] Cropped image saved: {crop_name}")
                else:
                    print(f"[!!] Warning: Markers not fully detected, only raw image saved.")

                capture_event.clear()
                print("\nWaiting for command (p/q): ", end="", flush=True)

    except Exception as e:
        print(f"Camera thread error: {e}")
    finally:
        pipeline.stop()


# --- 5. Command Listener Logic ---

def input_listener():
    global capture_event, exit_event, label_input

    while not exit_event.is_set():
        if select.select([sys.stdin], [], [], 0.1)[0]:
            cmd = sys.stdin.readline().strip().lower()
            if cmd == 'p':
                # Blocking input to ensure label_input updates correctly
                label_input = input("Enter label/name for this image: ").strip() or "unnamed"
                capture_event.set()
            elif cmd == 'q':
                exit_event.set()
                break


# --- 6. main ---

if __name__ == "__main__":
    t_cam = threading.Thread(target=run_camera)
    t_in = threading.Thread(target=input_listener)

    t_cam.start()
    t_in.start()

    t_in.join()
    t_cam.join()