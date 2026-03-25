import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json
import threading
import sys
import select
from datetime import datetime

# Green channel compensation core logic
def compensate_green(image):
    """
    Apply a simple color correction matrix to slightly boost
    the green channel and reduce red/blue channels.
    """
    matrix = np.array([[0.96, 0, 0],
                       [0, 1.06, 0],
                       [0, 0, 0.96]])
    corrected = cv2.transform(image, matrix)
    return np.clip(corrected, 0, 255).astype(np.uint8)

# File paths
CONFIG_FILE = "config_default.json"
MAPPING_FILE = "camera_mapping.json"
IMAGE_DIR = "imgs/acquisitions"

# Load configuration
with open(CONFIG_FILE) as f:
    jsonObj = json.load(f)

json_string = str(jsonObj).replace("'", '\"')


# Detect connected RealSense cameras
ctx = rs.context()
serials = [dev.get_info(rs.camera_info.serial_number) for dev in ctx.devices]


def save_mapping(mapping):
    """Save camera mapping to file."""
    with open(MAPPING_FILE, "w") as f:
        json.dump(mapping, f)


def load_mapping():
    """Load camera mapping if file exists."""
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE) as f:
            return json.load(f)
    return {}

# Thread synchronization primitives
capture_event = threading.Event()

# Thread-safe exit signal (prevents thread hang)
exit_event = threading.Event()

label_input = ""
cam_mapping = {}


def pipeline_health_check(pipeline):
    """
    Check if pipeline responds within timeout.
    Prevents infinite blocking if camera is unresponsive.
    """
    try:
        pipeline.wait_for_frames(timeout_ms=3000)
        return True
    except:
        return False


def run_camera(serial, label=None, stream_mode=False):
    """
    Main camera worker thread.
    Handles streaming, frame capture, and image saving.
    """
    global capture_event, exit_event, label_input

    config = rs.config()
    pipeline = rs.pipeline()
    config.enable_device(serial)

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    # Skip device if no RGB sensor available
    if not any(s.get_info(rs.camera_info.name) == 'RGB Camera' for s in device.sensors):
        print(f"Camera {serial} has no RGB sensor. Skipping.")
        return

    # Configure color stream
    config.enable_stream(
        rs.stream.color,
        int(jsonObj['viewer']['stream-width']),
        int(jsonObj['viewer']['stream-height']),
        rs.format.bgr8,
        int(jsonObj['viewer']['stream-fps'])
    )

    try:
        cfg = pipeline.start(config)
    except Exception as e:
        print(f"Error starting pipeline for {serial}: {e}. Try reconnecting the device.")
        exit_event.set()
        return

    dev = cfg.get_device()

    # Load advanced mode JSON configuration
    advnc_mode = rs.rs400_advanced_mode(dev)
    advnc_mode.load_json(json_string)

    # Hardware warm-up (avoid unstable first frames)
    for _ in range(20):
        try:
            pipeline.wait_for_frames(timeout_ms=1000)  #The program will wait for a maximum of 1s for a frame; otherwise, it times out.
        except:
            print(f"[{serial}] Warmup frame timeout")
            break

    try:
        if not stream_mode:
            # Single snapshot mode
            print("Pipeline response in 3 seconds?:", pipeline_health_check(pipeline))

            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                color = frames.get_color_frame()
            except:
                print(f"[{serial}] Snapshot timeout")
                return

            img = np.asanyarray(color.get_data())
            img = compensate_green(img)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"sample_{label or serial}_{timestamp}.png"

            cv2.imwrite(os.path.join(IMAGE_DIR, filename), img)
            print(f"[{serial}] Snapshot saved: {filename}")

        else:
            # Continuous streaming mode
            while not exit_event.is_set():
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                except:
                    print(f"[{serial}] Frame timeout, retrying...")
                    continue

                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())

                # Capture triggered by user input
                if capture_event.is_set():
                    final_image = compensate_green(color_image)

                    label = cam_mapping.get(serial, serial)
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = f"sample_{label}_{label_input}_{timestamp}.png"

                    cv2.imwrite(os.path.join(IMAGE_DIR, filename), final_image)
                    print(f"[{serial}] Captured: {filename}")

                    capture_event.clear()

    except Exception as e:
        print(
            f"Error capturing image for {serial}: {e}. Device may be disconnected. Restart required.")
        dev.hardware_reset()
        exit_event.set()

    finally:
        pipeline.stop()
        print(f"[{serial}] Stopped")


def input_listener():
    """
    Listen for keyboard commands:
    - 'p' to capture
    - 'q' to quit
    """
    global capture_event, exit_event, label_input

    print("Press 'p' + [Enter] to capture, 'q' + [Enter] to quit.")

    while not exit_event.is_set():
        if select.select([sys.stdin], [], [], 0.1)[0]:
            cmd = sys.stdin.readline().strip()

            if cmd == 'p':
                label_input = input("Enter label for captured images: ").strip()
                capture_event.set()

            elif cmd == 'q':
                exit_event.set()
                break

def interactive_mapping():
    """
    Interactive camera direction mapping.
    """
    global cam_mapping

    print(f"Detected cameras: {serials}")

    ans = input("Do you already know camera directions? (y/n): ").strip().lower()

    if ans == 'y':
        for sn in serials:
            direction = input(f"Enter direction for camera {sn} (e.g. left/right): ").strip().lower()
            cam_mapping[sn] = direction

        with open(MAPPING_FILE, "w") as f:
            json.dump(cam_mapping, f)

    else:
        print("Capturing images to help identify cameras...")
        for sn in serials:
            run_camera(sn, stream_mode=False)

        print(f"Images saved in {IMAGE_DIR}. Please review and re-run the program.")

# Load or create camera mapping
if os.path.exists(MAPPING_FILE):
    with open(MAPPING_FILE) as f:
        cam_mapping = json.load(f)
else:
    interactive_mapping()
    if not os.path.exists(MAPPING_FILE):
        exit()

# Start camera threads
threads = []

for sn in serials:
    label = cam_mapping.get(sn, sn)
    t = threading.Thread(target=run_camera, args=(sn, label, True))
    t.start()
    threads.append(t)

# Start input listener thread
input_thread = threading.Thread(target=input_listener)
input_thread.start()
input_thread.join()

# Wait for all camera threads to finish
for t in threads:
    t.join()