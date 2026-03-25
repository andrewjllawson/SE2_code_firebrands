"""
Updated on Mar 25 2026

@author: Andrew Lawson

Emberometer Field Recording Controller
--------------------------------------

This program acts as the primary control interface for the Emberometer sensor
platform. It manages GNSS timing, recording control, system status indication,
and safe reboot behaviour for unattended field operation.

Core responsibilities:
  • Continuously monitor a GNSS receiver for valid RMC time data
  • Indicate GNSS lock status via dual status LEDs:
        - Searching: LEDs blink
        - Locked:    LEDs solid ON
        - Recording: LEDs OFF (to avoid false state reporting)
  • Enforce that thermal/RGB video recording may only begin once GNSS lock exists
  • Launch and terminate the video acquisition process (`video_record.py`)
  • Export the GNSS UTC timestamp corresponding to the start of each recording
  • Support physical button input with two modes:
        - Short press (<5s): start/stop recording
        - Long press  (≥5s): initiate a clean system reboot (disabled during recording)
  • Maintain robust behaviour across power cycles using systemd autostart

GPS usage:
  GNSS time is parsed from NMEA $xxRMC sentences delivered at 115200 baud.
  Once a valid fix is acquired, GNSS time is treated as the authoritative
  source for timestamping and alignment of captured sensor data.

Field operation notes:
  - The device may be deployed outdoors without network access; GNSS provides
    both the timing reference and readiness indication.
  - No recording begins without GNSS lock to ensure all captures are time-aligned.
  - The reboot long-press provides an operator-friendly method to restart
    the instrument without needing SSH, keyboard, or UI access.

This file intentionally contains:
  • No video-capture logic (handled by `video_record.py`)
  • No image processing
  • No networking assumptions
  • No user-facing UI beyond LEDs and physical buttons

Designed for:
  - unattended field deployment
  - deterministic timing
  - time-synchronised multi-sensor post-processing
  - robustness under power loss, reboot, and environmental variability
"""

#!home/pi/my-env2/bin/python3
import os
import subprocess
import RPi.GPIO as GPIO
import time
import signal
import threading

# --- GPS imports ---
import serial
from datetime import datetime, timezone
# --------------------

# ----------------------------------------
# System initialisation / Working directory
# ----------------------------------------
os.chdir("/home/pi/Emberometer")

# ----------------------------------------
# GPIO pin configuration
# ----------------------------------------
RECORD_BUTTON = 22      # Short press -> toggle record, long press (5s) -> reboot
ANALYZE_BUTTON = 23     # Unused (future expansion)

GPIO_LED1 = 5           # Status LED A
GPIO_LED2 = 6           # Status LED B

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(GPIO_LED1, GPIO.OUT)
GPIO.setup(GPIO_LED2, GPIO.OUT)

# LED logic (active-high)
LED_ON_LEVEL = GPIO.HIGH
LED_OFF_LEVEL = GPIO.LOW

def leds_on(msg=None):
    """Turn both LEDs on."""
    if msg:
        print(msg)
    GPIO.output(GPIO_LED1, LED_ON_LEVEL)
    GPIO.output(GPIO_LED2, LED_ON_LEVEL)

def leds_off(msg=None):
    """Turn both LEDs off."""
    if msg:
        print(msg)
    GPIO.output(GPIO_LED1, LED_OFF_LEVEL)
    GPIO.output(GPIO_LED2, LED_OFF_LEVEL)

# ----------------------------------------
# GPS configuration + state flags
# ----------------------------------------
GPS_PORT = "/dev/serial0"
GPS_BAUD = 115200

gps_locked = False          # True once valid GNSS fix received
recording_active = False    # True while video_record.py running
gps_last_fix_dt = None      # Stores latest valid GPS UTC timestamp
gps_last_fix_pos = None     # (lat_deg, lon_deg)
gps_stop_event = threading.Event()

recording_process = None    # Handle for the video_record subprocess

# ----------------------------------------
# GPS NMEA parsing (RMC for UTC time + fix state)
# ----------------------------------------
def parse_rmc(nmea: str):
    """
    Parse a $..RMC NMEA sentence and return (datetime_utc, lat_deg, lon_deg)
    if valid, else None.

    lat_deg / lon_deg are decimal degrees (WGS84), or None if parsing fails.
    """
    if not nmea or not nmea.startswith("$"):
        return None
    
    nmea_no_cs = nmea.split("*", 1)[0]

    parts = nmea_no_cs.split(",")
    if len(parts) < 10:
        return None
        
    header = parts[0]
    if len(header) < 6 or header[-3:] != "RMC":
        return None

    time_str = parts[1]  # hhmmss.sss
    status   = parts[2]
    lat_str  = parts[3]  # ddmm.mmmm
    lat_hemi = parts[4]  # N/S
    lon_str  = parts[5]  # dddmm.mmmm
    lon_hemi = parts[6]  # E/W
    date_str = parts[9]  # ddmmyy
    
    if status != "A":  # 'A' = valid fix
        return None
    if not time_str or not date_str or len(date_str) < 6 or len(time_str) < 6:
        return None

    try:
        # ----- Time -----
        hh = int(time_str[0:2])
        mm = int(time_str[2:4])
        ss = int(time_str[4:6])
        us = 0
        if "." in time_str:
            frac = time_str.split(".", 1)[1]
            frac = (frac + "000000")[:6]
            us = int(frac)

        # ----- Date -----
        day = int(date_str[0:2])
        month = int(date_str[2:4])
        year = 2000 + int(date_str[4:6])

        dt_utc = datetime(year, month, day, hh, mm, ss, us, tzinfo=timezone.utc)

        # ----- Latitude: ddmm.mmmm -> decimal degrees -----
        lat_deg = None
        if lat_str and lat_hemi in ("N", "S"):
            lat_val = float(lat_str)
            lat_deg_int = int(lat_val // 100)
            lat_min = lat_val - lat_deg_int * 100
            lat_deg = lat_deg_int + lat_min / 60.0
            if lat_hemi == "S":
                lat_deg = -lat_deg

        # ----- Longitude: dddmm.mmmm -> decimal degrees -----
        lon_deg = None
        if lon_str and lon_hemi in ("E", "W"):
            lon_val = float(lon_str)
            lon_deg_int = int(lon_val // 100)
            lon_min = lon_val - lon_deg_int * 100
            lon_deg = lon_deg_int + lon_min / 60.0
            if lon_hemi == "W":
                lon_deg = -lon_deg

        return dt_utc, lat_deg, lon_deg

    except Exception:
        return None

# ----------------------------------------
# Background GPS monitor thread
# Handles:
#   - Reading RMC data
#   - Updating fix status
#   - LED behaviour:
#         Searching   -> blink
#         Locked      -> solid ON
#         Recording   -> LEDs off (controlled externally)
# ----------------------------------------
def gps_monitor_thread():
    global gps_locked, gps_last_fix_dt, gps_last_fix_pos

    print("Starting GPS monitor thread (LEDs will indicate status)...")
    blink_state = False

    try:
        ser = serial.Serial(GPS_PORT, GPS_BAUD, timeout=1)
    except Exception as e:
        print(f"Could not open GPS serial port {GPS_PORT}: {e}")
        leds_off()
        return

    with ser:
        while not gps_stop_event.is_set():
            # During recording, force LEDs off but continue updating GPS state internally
            if recording_active:
                leds_off()
                try:
                    _ = ser.readline()
                except Exception:
                    pass
                time.sleep(0.2)
                continue

            try:
                raw = ser.readline()
            except Exception as e:
                print(f"Error reading GPS serial: {e}")
                leds_off()
                time.sleep(1.0)
                continue

            if raw:
                try:
                    line = raw.decode(errors="ignore").strip()
                except Exception:
                    line = ""
                if line:
                    parsed = parse_rmc(line)
                    if parsed:
                        dt, lat_deg, lon_deg = parsed
                        gps_last_fix_dt = dt
                        gps_last_fix_pos = (lat_deg, lon_deg)

                        if not gps_locked:
                            gps_locked = True
                            print(f"GPS lock acquired at GNSS UTC: {dt.isoformat()}")
                            if lat_deg is not None and lon_deg is not None:
                                print(f"GPS position at lock: lat={lat_deg:.6f}, lon={lon_deg:.6f}")


            # LED indication logic
            if gps_locked:
                # solid ON when locked
                leds_on()
                time.sleep(0.5)
            else:
                # blink while searching
                blink_state = not blink_state
                if blink_state:
                    leds_on()
                else:
                    leds_off()
                time.sleep(0.5)

# ----------------------------------------
# Recording session folder helper
# ----------------------------------------
def get_latest_session():
    recordings_dir = "/home/pi/Emberometer"
    sessions = [d for d in os.listdir(recordings_dir)
                if os.path.isdir(os.path.join(recordings_dir, d)) and d.startswith("recording_session_")]
    if not sessions:
        return None
    latest_session = max(sessions, key=lambda x: os.path.getmtime(os.path.join(recordings_dir, x)))
    return os.path.join(recordings_dir, latest_session)

# ----------------------------------------
# Recording toggle (short press behaviour)
# ----------------------------------------
def toggle_recording():
    global recording_process, recording_active, gps_locked, gps_last_fix_dt
    if recording_process is None:
        print("Starting recording...")

        # Block until GNSS lock
        while not gps_locked:
            print("Waiting for GPS lock before starting recording...")
            time.sleep(0.5)

        recording_active = True
        leds_off()

        # GPS time when recording actually starts
        start_dt = gps_last_fix_dt
        start_pos = gps_last_fix_pos

        env = os.environ.copy()

        if start_dt is not None:
            env["GPS_RECORDING_START_UTC"] = start_dt.isoformat(timespec="milliseconds")
        else:
            env["GPS_RECORDING_START_UTC"] = "UNKNOWN"

        if start_pos is not None and start_pos[0] is not None and start_pos[1] is not None:
            env["GPS_RECORDING_START_LAT"] = f"{start_pos[0]:.8f}"
            env["GPS_RECORDING_START_LON"] = f"{start_pos[1]:.8f}"
        else:
            env["GPS_RECORDING_START_LAT"] = "UNKNOWN"
            env["GPS_RECORDING_START_LON"] = "UNKNOWN"

        recording_process = subprocess.Popen(
            ["python3", "/home/pi/Emberometer/video_record.py"],
            cwd="/home/pi/Emberometer",
            env=env,
        )
        
    else:
        print("Stopping recording...")
        recording_process.send_signal(signal.SIGTERM)
        recording_process.wait()
        recording_process = None
        recording_active = False
        leds_off()
        latest_session = get_latest_session()
        if latest_session:
            completion_file = os.path.join(latest_session, "recording_complete.txt")
            with open(completion_file, "w") as f:
                f.write("Recording complete.")
            print(f"Recording complete flag generated in {latest_session}")
        else:
            print("No session directory found to generate the recording complete flag.")

# ----------------------------------------
# Long-press reboot (5s hold)
# ----------------------------------------
def reboot_system():
    global recording_process, recording_active

    # Safety: block reboot if a recording is in progress
    if recording_process is not None or recording_active:
        print("Cannot reboot while recording. Stop recording first.")
        return

    print("Reboot button pressed. Rebooting system...")

    # Short visual feedback
    for _ in range(5):
        GPIO.output(GPIO_LED1, GPIO.HIGH)
        GPIO.output(GPIO_LED2, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(GPIO_LED1, GPIO.LOW)
        GPIO.output(GPIO_LED2, GPIO.LOW)
        time.sleep(0.1)

    os.system("sudo reboot")
    
# ----------------------------------------
# Main event loop
# ----------------------------------------
print("Initializing GPIO...")
GPIO.setmode(GPIO.BCM)
GPIO.setup(RECORD_BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(ANALYZE_BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
print("GPIO Initialized. Entering main loop...")

# Start GPS monitor in the background
gps_thread = threading.Thread(target=gps_monitor_thread, daemon=True)
gps_thread.start()

try:
    while True:
        if GPIO.input(RECORD_BUTTON) == GPIO.HIGH:
            press_time = time.time()

            # Wait while button remains pressed
            while GPIO.input(RECORD_BUTTON) == GPIO.HIGH:
                time.sleep(0.05)
                # Check if held long enough
                if time.time() - press_time >= 5.0:
                    print("Long press detected (>=5s). Rebooting system...")
                    
                    # Block reboot if recording active
                    if recording_process is None:
                        # LED feedback
                        for _ in range(5):
                            GPIO.output(GPIO_LED1, GPIO.HIGH)
                            GPIO.output(GPIO_LED2, GPIO.HIGH)
                            time.sleep(0.1)
                            GPIO.output(GPIO_LED1, GPIO.LOW)
                            GPIO.output(GPIO_LED2, GPIO.LOW)
                            time.sleep(0.1)
                        os.system("sudo reboot")
                    else:
                        print("Recording in progress — reboot aborted.")
                    # Skip normal short-press logic
                    break
            else:
                # Only runs if button released before 5s -> short press
                print("Short press detected — toggling recording")
                toggle_recording()
            time.sleep(0.3)  # debounce
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Exiting due to KeyboardInterrupt...")
except Exception as e:
    print("An unexpected error occurred:", e)
finally:
    print("Cleaning up GPIO...")
    gps_stop_event.set()
    try:
        gps_thread.join(timeout=1.0)
    except Exception:
        pass
    GPIO.cleanup()

