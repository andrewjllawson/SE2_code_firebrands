"""
Updated on Mar 25 2026

@author: Andrew Lawson

Emberometer Video & Thermal Acquisition
---------------------------------------

This module is launched by the top-level controller (run.py) to perform
time-synchronised recording of:

  - Thermal imagery from the MLX90640 sensor (via I2C)
  - RGB video from the Raspberry Pi camera (via Picamera2)
  - Per-pixel thermal raw data to CSV (for post-processing / calibration)
  - Local ambient temperature from a MAX31855 thermocouple (via SPI)

Each recording session:

  - Creates a new directory: recording_session_YYYYMMDD_HHMMSS
  - Stores:
      * Segmented thermal video (.avi)
      * Segmented RGB video (.avi)
      * Merged final thermal and RGB videos
      * Thermal per-frame raw CSV
      * RGB and thermal timestamp metadata files
      * GNSS-based recording start time (GNSS_adjustment.txt)
      * Temperature log from the MAX31855 (temperature_log.csv)
      * A simple recording_complete.txt flag

The controller (run.py) is responsible for:

  - Ensuring GNSS lock before starting this process
  - Passing GPS_RECORDING_START_UTC in the environment
  - Handling user button input and LEDs
  - Managing safe shutdown, reboot and high-level state

This file focuses on:

  - Sensor I/O and capture
  - Threaded recording pipelines
  - File I/O and segmentation
  - Graceful termination on SIGTERM or KeyboardInterrupt
"""

#!/home/pi/my-env2/bin/python3
import csv
import time
import threading
import queue
import cv2
import numpy as np
from picamera2 import Picamera2
import board
import busio
import adafruit_mlx90640
from scipy import ndimage
import os
import signal
import sys
import subprocess
import spidev
from pathlib import Path
import socket

EMBEROMETER_DIR = Path("/home/pi/Emberometer")
MASTER_SPEC_PATH = EMBEROMETER_DIR / "device_spec.txt"


# ============================================================================
# WRITE DEVICE SPECIFICATION FILE FOR THIS SESSION
# ============================================================================

def write_session_device_spec(session_dir: str) -> None:
    session_dir = Path(session_dir)
    out_path = session_dir / "device_spec.txt"

    user_id = socket.gethostname()

    if MASTER_SPEC_PATH.exists():
        master_text = MASTER_SPEC_PATH.read_text(encoding="utf-8")
    else:
        master_text = "WARNING: device_spec.txt not found\n"

    out_text = f"User ID: {user_id}\n" + master_text.lstrip("\n")
    out_path.write_text(out_text, encoding="utf-8")
    

# ============================================================================
# MAX31855 THERMOCOUPLE INTERFACE
# ============================================================================

class MAX31855:
    """
    Simple driver for the MAX31855 thermocouple amplifier over SPI.

    Provides:
      - raw 16-bit reads
      - Celsius and Fahrenheit temperature conversion
      - Open-circuit detection (no thermocouple connected)
    """
    def __init__(self, bus=0, device=0, max_speed_hz=500000):
        self.bus = bus
        self.device = device
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = max_speed_hz
        # MAX31855 uses SPI mode 0 (CPOL=0, CPHA=0)
        self.spi.mode = 0

    def read_raw(self):
        """
        Read 32 bits of raw data from the MAX31855.

        Returns:
            int: 32-bit value representing thermocouple state and temperature.

        Raises:
            RuntimeError: if fewer than 4 bytes are read.
        """
        raw = self.spi.readbytes(4)
        if len(raw) != 4:
            raise RuntimeError("Failed to read 4 bytes from MAX31855")
        value = (raw[0] << 24) | (raw[1] << 16) | (raw[2] << 8) | raw[3]
        return value

    def read_celsius(self):
        """
        Read temperature in degrees Celsius.

        The MAX31855 encodes:
          - Bit 2 (0x4): open thermocouple indicator when set to 1.
          - Bits 3..14: 16-bit temperature, step size 0.25 °C.

        Returns:
            float: Temperature in °C.

        Raises:
            RuntimeError: if thermocouple is not connected.
        """
        value = self.read_raw()
            
        #Fault bit
        if (value >> 16) & 0x1:
            raise RuntimeError("No thermocouple connected (open circuit)")
        
        temp_raw = (value >> 18) & 0x3FFF

        # Bit 2 (0x4) is D2: '1' means no thermocouple connected
        if temp_raw & 0x2000:
            temp_raw -= 1 << 14
            
        temp_c = temp_raw *0.25
        return temp_c

    def read_fahrenheit(self):
        """
        Read temperature in degrees Fahrenheit.

        Returns:
            float: Temperature in °F.
        """
        temp_c = self.read_celsius()
        return temp_c * 9.0 / 5.0 + 32.0

    def close(self):
        """Close the underlying SPI device handle."""
        self.spi.close()

# Thermocouple sampling interval (seconds)
TEMP_LOG_INTERVAL = 1.0

def temp_logging_thread(session_dir, rgb_start_time, stop_event):
    """
    Periodically log ambient temperature from MAX31855 to CSV.

    Samples once per second (TEMP_LOG_INTERVAL) until stop_event is set, and
    writes:

        time_s, temperature_C

    into temperature_log.csv located inside the recording session directory.

    Args:
        session_dir (str): Path to the recording session directory.
        rgb_start_time (float): Unix timestamp of RGB recording start reference.
        stop_event (threading.Event): Cooperative stop flag for clean exit.
    """
    csv_path = os.path.join(session_dir, "temperature_log.csv")
    sensor = None

    # Try initialising the MAX31855; if that fails, still create header and exit.
    try:
        sensor = MAX31855(bus=0, device=0)
    except Exception as e:
        print(f"Failed to initialize MAX31855: {e}")
        with open(csv_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["time_s", "temperature_C"])
        return

    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["time_s", "temperature_C"])

        try:
            while not stop_event.is_set():
                now = time.time()
                # Relative time (s) from RGB start reference.
                t_rel = now - rgb_start_time

                try:
                    temp_c = sensor.read_celsius()
                except RuntimeError as e:
                    # For example: thermocouple disconnected
                    print(f"Temp read error: {e}")
                    temp_c = float("nan")

                # Write relative time and temperature in Celsius only
                writer.writerow([f"{t_rel:.3f}", temp_c])
                csvfile.flush()

                time.sleep(TEMP_LOG_INTERVAL)
        finally:
            sensor.close()

    print("Temperature logging stopped.")



# ============================================================================
# SIGNAL HANDLING
# ============================================================================

def graceful_exit(signum, frame):
    """
    Handle SIGTERM for graceful shutdown.

    Called when the parent process (run.py) sends SIGTERM. Ensures that the
    process exits cleanly and triggers the 'finally' block in main() for
    thread shutdown and file closure.
    """
    print("SIGTERM received. Cleaning up and exiting...")
    sys.exit(0)

signal.signal(signal.SIGTERM, graceful_exit)


# ============================================================================
# MLX90640 THERMAL SENSOR INITIALISATION
# ============================================================================

# I2C bus for MLX90640
i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_32_HZ

# MLX90640 native resolution and interpolation factor
mlx_shape = (24, 32)
mlx_interp_val = 1  # can be > 1 if you want upscaled thermal images


# ============================================================================
# RASPBERRY PI CAMERA INITIALISATION
# ============================================================================

picam2 = Picamera2()
picam2_config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(picam2_config)
picam2.set_controls({"AwbEnable": True})
picam2.start()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_thermal_image(max_retries=3):
    """
    Capture a single frame from the MLX90640 and return as a 2D numpy array.

    Retries up to max_retries times if the underlying driver raises a ValueError
    (for example, the 'math domain error' seen in the Adafruit library).

    Args:
        max_retries (int): Maximum number of attempts for a valid frame.

    Returns:
        np.ndarray or None: 2D array of temperatures (24x32, possibly interpolated),
                            or None if all attempts fail.
    """
    frame = [0] * (24 * 32)
    for attempt in range(max_retries):
        try:
            mlx.getFrame(frame)
            break  # success
        except ValueError as e:
            # This is the "math domain error"
            print(f"[THERMAL] MLX90640 ValueError on getFrame (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(0.05)
    else:
        # All attempts failed; caller will skip this frame
        return None

    data_array = np.reshape(frame, (24, 32))
    data_array = ndimage.zoom(data_array, mlx_interp_val)
    return data_array


def process_thermal_image(data_array):
    """
    Convert a thermal 2D array into a color-mapped RGB image for video.

    Performs normalization to 0–255, converts to uint8 and applies a JET
    colourmap.

    Args:
        data_array (np.ndarray): 2D thermal data array.

    Returns:
        np.ndarray: BGR image suitable for OpenCV video writing.
    """
    min_val, max_val = np.min(data_array), np.max(data_array)
    image = (data_array - min_val) / (max_val - min_val) * 255
    image = np.uint8(image)
    image = cv2.flip(image, 1)  # Mirror flip horizontally if desired
    image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return image

def add_timestamp(frame, timestamp):
    """
    Overlay a human-readable timestamp onto an image frame.

    Args:
        frame (np.ndarray): Input image frame (BGR).
        timestamp (float): Unix timestamp (seconds since epoch).

    Returns:
        np.ndarray: Frame with timestamp text overlay.
    """
    timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp)) + f".{(timestamp % 1):.3f}"
    cv2.putText(frame, timestamp_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return frame

# Segment length for video files in seconds (300 s = 5 minutes)
SEGMENT_DURATION = 300


# ============================================================================
# CAPTURE THREADS (THERMAL & RGB) WITH SEGMENTED RECORDING
# ============================================================================

def thermal_capture_thread(thermal_video_base, target_fps, stop_event,
                           raw_data_queue, fps_queue, metadata_file):
    """
    Capture and encode thermal video segments, and push raw frames to a queue.

    - Writes segmented .avi files named thermal_video_base_<index>.avi
    - Each frame is written to a CSV writer via raw_data_queue (handled elsewhere)
    - Metadata file receives the frame timestamps (one per line)
    - FPS timestamps are pushed to fps_queue for live FPS estimation

    Args:
        thermal_video_base (str): Base filepath (without segment suffix).
        target_fps (float): Desired thermal frame rate.
        stop_event (threading.Event): Cooperative stop flag.
        raw_data_queue (queue.Queue): Queue for (data_array, timestamp).
        fps_queue (queue.Queue): Queue for ("thermal", frame_time) stamps.
        metadata_file (file): Open text file to append timestamps.
    """
    seg_index = 0
    segment_start_time = time.time()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    current_video_path = f"{thermal_video_base}_{seg_index}.avi"
    thermal_out = cv2.VideoWriter(
        current_video_path, fourcc, target_fps, (640, 480), isColor=True
    )
    period = 1.0 / target_fps
    last_time = time.time()

    while not stop_event.is_set():
        current_time = time.time()
        
        # Rotate to a new segment if we exceed SEGMENT_DURATION
        if current_time - segment_start_time >= SEGMENT_DURATION:
            thermal_out.release()
            seg_index += 1
            segment_start_time = current_time
            current_video_path = f"{thermal_video_base}_{seg_index}.avi"
            thermal_out = cv2.VideoWriter(current_video_path, fourcc, target_fps, (640, 480), isColor=True)
            print(f"Started new thermal segment: {current_video_path}")
        
        # Enforce approximate frame rate
        if current_time - last_time >= period:
            timestamp = time.time()
            data_array = get_thermal_image()
            if data_array is None:
                # No valid frame; drop this frame and continue
                print("[THERMAL] Dropping frame due to invalid MLX90640 data.")
                last_time = current_time
                continue
        
            # Send raw data for CSV logging
            try:
                raw_data_queue.put((data_array, timestamp), timeout=0.1)
            except queue.Full:
                print("Raw data queue full, dropping thermal data.")
            
            # Colour-map and write video frame
            thermal_image = process_thermal_image(data_array)
            thermal_resized = cv2.resize(thermal_image, (640, 480))
            thermal_resized = add_timestamp(thermal_resized, timestamp)
            thermal_out.write(thermal_resized)

            # Log timestamp to metadata file
            metadata_file.write(f"{timestamp}\n")
            metadata_file.flush()
            
            # Notify FPS calculation thread
            try:
                fps_queue.put(("thermal", time.time()), timeout=0.1)
            except queue.Full:
                pass
                
            last_time = current_time
            
        time.sleep(0.001)

    thermal_out.release()
    print("Thermal recording stopped.")


def rgb_capture_thread(rgb_video_base, target_fps, stop_event, fps_queue, metadata_file):
    """
    Capture and encode RGB video segments from the Pi camera.

    - Writes segmented .avi files named rgb_video_base_<index>.avi
    - Adds timestamps to frames
    - Writes timestamps to metadata file
    - Publishes frame times to fps_queue for FPS monitoring

    Args:
        rgb_video_base (str): Base filepath (without segment suffix).
        target_fps (float): Desired RGB frame rate.
        stop_event (threading.Event): Cooperative stop flag.
        fps_queue (queue.Queue): Queue for ("rgb", frame_timestamp) entries.
        metadata_file (file): Open text file to append timestamps.
    """
    seg_index = 0
    segment_start_time = time.time()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    current_video_path = f"{rgb_video_base}_{seg_index}.avi"
    rgb_out = cv2.VideoWriter(current_video_path, fourcc, target_fps, (640, 480), isColor=True)
    period = 1.0 / target_fps
    last_time = time.time()

    while not stop_event.is_set():
        current_time = time.time()
        
        # Rotate RGB video segment
        if current_time - segment_start_time >= SEGMENT_DURATION:
            rgb_out.release()
            seg_index += 1
            segment_start_time = current_time
            current_video_path = f"{rgb_video_base}_{seg_index}.avi"
            rgb_out = cv2.VideoWriter(current_video_path, fourcc, target_fps, (640, 480), isColor=True)
            print(f"Started new RGB segment: {current_video_path}")
        
        # Enforce approximate frame rate
        if current_time - last_time >= period:
            timestamp = time.time()
            pi_cam_frame = picam2.capture_array()
            pi_cam_frame = add_timestamp(pi_cam_frame, timestamp)
            rgb_out.write(pi_cam_frame)

            # Log timestamp for each RGB frame
            metadata_file.write(f"{timestamp}\n")
            metadata_file.flush()
            
            # Publish frame time for FPS calculation
            try:
                fps_queue.put(("rgb", time.time()), timeout=0.1)
            except queue.Full:
                pass

            last_time = current_time
            
        time.sleep(0.001)

    rgb_out.release()
    print("RGB recording stopped.")

def calculate_fps(frame_times):
    """
    Compute approximate FPS from a list of frame timestamps.

    Args:
        frame_times (list[float]): List of timestamps (seconds).

    Returns:
        float: Estimated frames per second.
    """
    if len(frame_times) < 2:
        return 0.0
    time_diffs = [frame_times[i+1] - frame_times[i] for i in range(len(frame_times) - 1)]
    avg_period = sum(time_diffs) / len(time_diffs)
    return 1.0 / avg_period if avg_period > 0 else 0.0

def data_saving_thread(raw_data_queue, thermal_raw_data_path, stop_event):
    """
    Save raw thermal data frames from the queue into a CSV file.

    CSV format:
      Pixel_1, Pixel_2, ..., Pixel_768, Timestamp

    Terminates when:
      - stop_event is set and no more data arrives, or
      - a sentinel value of None is put on the queue.

    Args:
        raw_data_queue (queue.Queue): Source queue of (data_array, timestamp).
        thermal_raw_data_path (str): Output CSV path for raw thermal data.
        stop_event (threading.Event): Cooperative stop flag.
    """
    with open(thermal_raw_data_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f'Pixel_{i + 1}' for i in range(24 * 32)] + ['Timestamp'])
        while True:
            try:
                item = raw_data_queue.get(timeout=1)
            except queue.Empty:
                if stop_event.is_set():
                    break
                continue

            # Sentinel used to signal clean shutdown
            if item is None:
                break

            data_array, timestamp = item
            flattened_data = data_array.flatten()
            writer.writerow(list(flattened_data) + [timestamp])
            
    print("Thermal data saving stopped.")


# ============================================================================
# MERGING SEGMENTED VIDEO FILES
# ============================================================================

def merge_video_segments(video_base, final_output, session_dir):
    """
    Use ffmpeg to merge segmented AVI files into a single final video.

    Files are detected in session_dir by matching the base name prefix and
    extension .avi, e.g.:

        thermal_video_YYYYMMDD_HHMMSS_0.avi
        thermal_video_YYYYMMDD_HHMMSS_1.avi
        ...

    Args:
        video_base (str): Base path used when segments were created.
        final_output (str): Final merged AVI filename to write.
        session_dir (str): Directory where segments and final output live.
    """
    base_name = os.path.basename(video_base)
    seg_files = sorted([f for f in os.listdir(session_dir)
                        if f.startswith(base_name) and f.endswith('.avi')])
    filelist_path = os.path.join(session_dir, "filelist.txt")
    with open(filelist_path, "w") as f:
        for seg in seg_files:
            abs_path = os.path.abspath(os.path.join(session_dir, seg))
            f.write(f"file '{abs_path}'\n")
    merge_cmd = [
        "ffmpeg", "-f", "concat", "-safe", "0",
        "-i", filelist_path,
        "-c", "copy",
        os.path.join(session_dir, final_output)
    ]
    subprocess.run(merge_cmd)
    print(f"Merged segments into {final_output}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Entry point for the recording process.

    Responsibilities:
      - Create a new recording session directory
      - Log GNSS-based recording start time (from environment)
      - Start thermal, RGB, thermocouple and CSV threads
      - Periodically compute and print FPS
      - On termination:
          * Stop all threads
          * Close metadata files
          * Write recording_complete.txt
          * Merge segmented videos into final_*.avi outputs
    """
    target_thermal_fps = 8
    target_rgb_fps = 32
    
    temp_thread = None

    # -------------------------
    # Persist GPS-based recording start time
    # -------------------------  
        
    recording_start_utc = os.environ.get("GPS_RECORDING_START_UTC", "UNKNOWN")
    recording_start_lat = os.environ.get("GPS_RECORDING_START_LAT", "UNKNOWN")
    recording_start_lon = os.environ.get("GPS_RECORDING_START_LON", "UNKNOWN")
    
    # -------------------------
    # Create unique session directory
    # -------------------------
    from datetime import datetime

    ME_identifier = socket.gethostname()

    try:
        # Parse full ISO UTC string
        dt = datetime.fromisoformat(recording_start_utc.replace("Z", "+00:00"))
        
        # Format safely (no colons)
        session_timestamp = dt.strftime("D%Y%m%d_T%H%M%S")

    except Exception:
        # Fallback if GPS time missing or malformed
        dt = datetime.utcnow()
        session_timestamp = dt.strftime("D%Y%m%d_T%H%M%S")

    session_dir = f"{ME_identifier}_recording_session_{session_timestamp}"
    os.makedirs(session_dir, exist_ok=True)

    
    write_session_device_spec(session_dir)

    gps_rel_path = os.path.join(session_dir, "GNSS_adjustment.txt")
    try:
        with open(gps_rel_path, "w") as f:
            f.write(f"UTC,{recording_start_utc}\n")
            f.write(f"LAT,{recording_start_lat}\n")
            f.write(f"LON,{recording_start_lon}\n")
            f.write(f"ID,{ME_identifier}\n")
        print(f"Wrote GPS recording start time + location to {gps_rel_path}")
    except Exception as e:
        print(f"Failed to write GNSS_adjustment.txt: {e}")


    # -------------------------
    # Define base paths for this session's outputs
    # -------------------------
    thermal_video_base = os.path.join(session_dir, f"thermal_video_{session_timestamp}")
    rgb_video_base = os.path.join(session_dir, f"rgb_video_{session_timestamp}")
    thermal_raw_data_path = os.path.join(session_dir, f"thermal_raw_data_{session_timestamp}.csv")
    thermal_metadata_path = os.path.join(session_dir, f"thermal_metadata_{session_timestamp}.txt")
    rgb_metadata_path = os.path.join(session_dir, f"rgb_metadata_{session_timestamp}.txt")

    # Reference time for RGB and temperature logging
    rgb_start_time = time.time()

    stop_event = threading.Event()

    # Open timestamp metadata files
    thermal_metadata_file = open(thermal_metadata_path, "w")
    rgb_metadata_file = open(rgb_metadata_path, "w")

    # Queues for raw data and FPS events
    raw_data_queue = queue.Queue(maxsize=100)
    fps_queue = queue.Queue()

    print(f"Recording started: Saving thermal video segments with base {thermal_video_base}")
    print(f"Recording started: Saving RGB video segments with base {rgb_video_base}")
    print(f"Saving raw thermal data to {thermal_raw_data_path}")
    
    # Storage for FPS calculations
    thermal_frame_times = []
    rgb_frame_times = []
    fps_measurement_interval = 5  # seconds

    # -------------------------
    # Start capture and logging threads
    # ------------------------
    thermal_thread = threading.Thread(target=thermal_capture_thread,
                                      args=(thermal_video_base, target_thermal_fps, stop_event,
                                            raw_data_queue, fps_queue, thermal_metadata_file))
    rgb_thread = threading.Thread(target=rgb_capture_thread,
                                  args=(rgb_video_base, target_rgb_fps, stop_event,
                                        fps_queue, rgb_metadata_file))
    data_saving_thread_instance = threading.Thread(target=data_saving_thread,
                                                   args=(raw_data_queue, thermal_raw_data_path, stop_event))

    thermal_thread.daemon = True
    rgb_thread.daemon = True
    data_saving_thread_instance.daemon = True

    thermal_thread.start()
    rgb_thread.start()
    data_saving_thread_instance.start()

    # Temperature logging thread using a relative time reference close to RGB start
    rgb_start_time = time.time()
    temp_thread = threading.Thread(
        target=temp_logging_thread,
        args=(session_dir, rgb_start_time, stop_event)
    )
    temp_thread.daemon = True
    temp_thread.start()

    measurement_start_time = time.time()

    try:
        # Run up to ~5 hours or until externally stopped
        while time.time() - measurement_start_time < 18000 and not stop_event.is_set():
            time.sleep(1)
            while not fps_queue.empty():
                stream_type, frame_time = fps_queue.get()
                if stream_type == "thermal":
                    thermal_frame_times.append(frame_time)
                elif stream_type == "rgb":
                    rgb_frame_times.append(frame_time)
            elapsed_time = time.time() - measurement_start_time
            if elapsed_time >= fps_measurement_interval:
                thermal_fps = calculate_fps(thermal_frame_times)
                rgb_fps = calculate_fps(rgb_frame_times)
                print(f"Thermal FPS: {thermal_fps:.2f}, RGB FPS: {rgb_fps:.2f}")
                thermal_frame_times = []
                rgb_frame_times = []
                measurement_start_time = time.time()
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Stopping recording...")
        stop_event.set()
    finally:
        print("Stopping threads...")
        stop_event.set()
        # Send sentinel to terminate the data_saving_thread
        raw_data_queue.put(None)
        
        thermal_thread.join()
        rgb_thread.join()
        data_saving_thread_instance.join()
        if temp_thread is not None:
            temp_thread.join()
        
        thermal_metadata_file.close()
        rgb_metadata_file.close()
        
        # Flag file to mark session completion
        with open(os.path.join(session_dir, "recording_complete.txt"), "w") as f:
            f.write("complete")
        print("Recording stopped.")

        # Merge segmented videos into final outputs
        merge_video_segments(thermal_video_base, "final_thermal_video.avi", session_dir)
        merge_video_segments(rgb_video_base, "final_rgb_video.avi", session_dir)

if __name__ == '__main__':
    main()

