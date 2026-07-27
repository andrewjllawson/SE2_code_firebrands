"""
Updated on June 19 2026

@author: Andrew Lawson

Oconee thermocouple-guided video trimming workflow

Purpose
-------
This script uses thermocouple recordings to estimate the main fire / thermal
arrival time for each emberometer recording, creates a reviewable Excel trim
plan, and can optionally trim the corresponding AVI videos using ffmpeg.

Workflow
--------
1. Search thermocouple Excel workbooks for sheets corresponding to IDs 1–24.
2. Search the video folder for AVI files corresponding to the same IDs.
3. Clean and smooth each thermocouple trace using a Savitzky-Golay filter.
4. Estimate fire arrival using:
      - primary method: first dT/dt > 2 °C/s above the baseline temperature
      - fallback method: peak smoothed temperature
5. Ignore weak traces where the peak temperature rise is < 10 °C above baseline.
6. Create a ±10 minute video trim window around the estimated arrival time.
7. Write the proposed trim windows to an Excel file for manual review.
8. If PROCEED_WITH_TRIMMING is set to True, trim only rows marked ready_to_trim.

Important
---------
Run the script first with PROCEED_WITH_TRIMMING = False.
Check tc_video_windows.xlsx manually before enabling video trimming.

ffmpeg must be installed and callable from the command line for trimming.
"""

from pathlib import Path
import re
import subprocess
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

# ============================================================
# USER SETTINGS
# ============================================================

# Folder containing your thermocouple Excel files
TC_FOLDER = Path(r"Path")

# Folder containing your .avi video files
VIDEO_FOLDER = Path(r"Path")

# Folder where trimmed videos will be saved
OUTPUT_VIDEO_FOLDER = VIDEO_FOLDER / "trimmed_videos"

# Excel file where detected fire arrival times and trim windows are saved
OUTPUT_XLSX = TC_FOLDER / "tc_video_windows.xlsx"

# IMPORTANT:
# Keep this as False first.
# The script will only create the Excel trim plan.
# After checking the Excel file, change this to True to trim videos.
PROCEED_WITH_TRIMMING = False

# Fire arrival detection settings
DTDT_THRESHOLD = 2.0                  # °C/s, from the paper
BASELINE_DURATION_S = 300             # first 5 minutes used for baseline
MIN_PEAK_RISE_ABOVE_BASELINE_C = 10   # ignore weak traces with peak < baseline + 10°C

# Savitzky-Golay smoothing settings
SAVGOL_WINDOW_S = 7                   # smoothing window in seconds
SAVGOL_POLY_ORDER = 2                 # polynomial order for smoothing

# Video trimming settings
TRIM_BEFORE_S = 10 * 60               # start trimmed video 10 minutes before arrival
TRIM_AFTER_S = 10 * 60                # end trimmed video 10 minutes after arrival

# Expected row / recording range
MIN_ROW = 1
MAX_ROW = 24


# ============================================================
# TIME FORMATTING
# ============================================================

def seconds_to_hhmmss(seconds):
    """
    Convert seconds into HH:MM:SS format.
    Example:
    8389 -> 02:19:49
    """

    if seconds is None or pd.isna(seconds):
        return ""

    seconds = max(0, int(round(seconds)))

    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    return f"{h:02d}:{m:02d}:{s:02d}"


# ============================================================
# NUMBER EXTRACTION
# ============================================================

def extract_first_number(name):
    """
    Extract the first number from a string.

    Examples:
    Sheet1       -> 1
    TC_05        -> 5
    recording_24 -> 24
    """

    match = re.search(r"\d+", str(name))

    if match:
        return int(match.group())

    return None


def extract_video_number(video_path):
    """
    Extract a recording number from a video filename.

    This is used to match videos numbered 1–24.

    Examples:
    1.avi              -> 1
    01.avi             -> 1
    rgb_video_05.avi   -> 5
    recording_24.avi   -> 24
    """

    return extract_first_number(video_path.stem)


# ============================================================
# VIDEO MATCHING
# ============================================================

def build_video_map():
    """
    Find all .avi files in VIDEO_FOLDER and map them to row numbers.

    Output example:
    {
        1: Path(".../rgb_video_01.avi"),
        2: Path(".../rgb_video_02.avi")
    }
    """

    video_map = {}

    avi_files = list(VIDEO_FOLDER.glob("*.avi"))

    for video_file in avi_files:
        video_number = extract_video_number(video_file)

        if video_number is not None and MIN_ROW <= video_number <= MAX_ROW:
            video_map[video_number] = video_file

    return video_map


# ============================================================
# SAVGOL WINDOW
# ============================================================

def make_savgol_window(window_seconds, sampling_rate, data_length):
    """
    Convert a smoothing window in seconds into a valid Savitzky-Golay
    window length in samples.

    Savitzky-Golay requires:
    - odd window length
    - window length greater than polynomial order
    - window length less than or equal to data length
    """

    window_samples = int(round(window_seconds * sampling_rate))

    minimum_window = SAVGOL_POLY_ORDER + 2

    if minimum_window % 2 == 0:
        minimum_window += 1

    window_samples = max(window_samples, minimum_window)

    if window_samples % 2 == 0:
        window_samples += 1

    if window_samples >= data_length:
        window_samples = data_length - 1

    if window_samples % 2 == 0:
        window_samples -= 1

    if window_samples < minimum_window:
        return None

    return window_samples


# ============================================================
# CLEAN TC DATA
# ============================================================

def clean_tc_dataframe(df):
    """
    Clean one thermocouple sheet.

    Assumption:
    - first column = time in seconds
    - second column = temperature in °C

    Extra columns are ignored.
    """

    df = df.copy()

    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    if df.shape[1] < 2:
        return None

    time_col = df.columns[0]
    temp_col = df.columns[1]

    data = df[[time_col, temp_col]].copy()
    data.columns = ["time_s", "temperature_C"]

    data["time_s"] = pd.to_numeric(data["time_s"], errors="coerce")
    data["temperature_C"] = pd.to_numeric(data["temperature_C"], errors="coerce")

    data = data.dropna()
    data = data.sort_values("time_s")
    data = data.drop_duplicates(subset="time_s", keep="first")
    data = data.reset_index(drop=True)

    if len(data) < 20:
        return None

    return data


# ============================================================
# DETECT FIRE ARRIVAL
# ============================================================

def detect_arrival_time(df):
    """
    Detect fire / thermal arrival time from one thermocouple sheet.

    Primary method:
    - Smooth temperature
    - Calculate dT/dt
    - Find first point where:
        dT/dt > 2°C/s
        AND
        temperature > baseline

    Fallback method:
    - If primary method fails, use peak temperature time

    Ignore rule:
    - If peak temperature is less than baseline + 10°C,
      ignore the sheet completely

    Returns:
    - arrival_time_s
    - method
    """

    data = clean_tc_dataframe(df)

    if data is None:
        return None, ""

    dt = data["time_s"].diff().median()

    if pd.isna(dt) or dt <= 0:
        return None, ""

    sampling_rate = 1 / dt

    window_samples = make_savgol_window(
        window_seconds=SAVGOL_WINDOW_S,
        sampling_rate=sampling_rate,
        data_length=len(data)
    )

    if window_samples is None:
        return None, ""

    try:
        data["temp_smooth_C"] = savgol_filter(
            data["temperature_C"].to_numpy(),
            window_length=window_samples,
            polyorder=SAVGOL_POLY_ORDER
        )
    except Exception:
        return None, ""

    data["dTdt_C_per_s"] = np.gradient(
        data["temp_smooth_C"].to_numpy(),
        data["time_s"].to_numpy()
    )

    baseline_region = data[data["time_s"] <= BASELINE_DURATION_S]

    if len(baseline_region) >= 5:
        baseline_temp = baseline_region["temperature_C"].median()
    else:
        baseline_temp = data["temperature_C"].head(20).median()

    peak_index = data["temp_smooth_C"].idxmax()
    peak_temp = data.loc[peak_index, "temp_smooth_C"]
    peak_time_s = data.loc[peak_index, "time_s"]

    peak_rise_above_baseline = peak_temp - baseline_temp

    if peak_rise_above_baseline < MIN_PEAK_RISE_ABOVE_BASELINE_C:
        return None, ""

    for i in range(len(data)):
        temp_now = data.loc[i, "temp_smooth_C"]
        rate_now = data.loc[i, "dTdt_C_per_s"]

        condition_rate = rate_now > DTDT_THRESHOLD
        condition_baseline = temp_now > baseline_temp

        if condition_rate and condition_baseline:
            return data.loc[i, "time_s"], "gradient"

    return peak_time_s, "peak_fallback"


# ============================================================
# READ TC EXCEL FILES
# ============================================================

def build_tc_sheet_map():
    """
    Read thermocouple Excel files and map row numbers 1–24 to sheets.

    This assumes sheets are named with numbers, such as:
    1, 2, 3...
    Sheet1, Sheet2...
    TC_01, TC_02...
    """

    sheet_map = {}

    excel_files = (
        list(TC_FOLDER.glob("*.xlsx"))
        + list(TC_FOLDER.glob("*.xls"))
        + list(TC_FOLDER.glob("*.xlsm"))
    )

    for excel_file in excel_files:
        try:
            workbook = pd.ExcelFile(excel_file)
        except Exception as e:
            print(f"Could not open TC workbook {excel_file.name}: {e}")
            continue

        for sheet_name in workbook.sheet_names:
            sheet_number = extract_first_number(sheet_name)

            if sheet_number is not None and MIN_ROW <= sheet_number <= MAX_ROW:
                sheet_map[sheet_number] = {
                    "excel_file": excel_file,
                    "sheet_name": sheet_name
                }

    return sheet_map


# ============================================================
# CREATE TRIM PLAN
# ============================================================

def create_trim_plan():
    """
    Create an Excel trim plan.

    This does not trim videos yet.
    It only detects arrival times and matches them to video files.
    """

    results = []

    tc_sheet_map = build_tc_sheet_map()
    video_map = build_video_map()

    for row_number in range(MIN_ROW, MAX_ROW + 1):

        arrival_time = ""
        trim_start = ""
        trim_end = ""
        method = ""
        tc_file_name = ""
        tc_sheet_name = ""
        video_file_name = ""
        trim_status = ""

        arrival_time_s = None
        trim_start_s = None
        trim_end_s = None

        if row_number in tc_sheet_map:
            tc_info = tc_sheet_map[row_number]
            tc_file = tc_info["excel_file"]
            sheet_name = tc_info["sheet_name"]

            tc_file_name = tc_file.name
            tc_sheet_name = sheet_name

            try:
                df = pd.read_excel(tc_file, sheet_name=sheet_name)

                arrival_time_s, method = detect_arrival_time(df)

                if arrival_time_s is not None:
                    trim_start_s = max(0, arrival_time_s - TRIM_BEFORE_S)
                    trim_end_s = arrival_time_s + TRIM_AFTER_S

                    arrival_time = seconds_to_hhmmss(arrival_time_s)
                    trim_start = seconds_to_hhmmss(trim_start_s)
                    trim_end = seconds_to_hhmmss(trim_end_s)
                else:
                    trim_status = "ignored_no_valid_temperature_event"

            except Exception as e:
                trim_status = f"tc_error: {e}"

        else:
            trim_status = "missing_tc_sheet"

        if row_number in video_map:
            video_file = video_map[row_number]
            video_file_name = video_file.name
        else:
            video_file = None

            if trim_status:
                trim_status += "; missing_video"
            else:
                trim_status = "missing_video"

        if arrival_time_s is not None and video_file is not None:
            trim_status = "ready_to_trim"

        results.append({
            "row": row_number,
            "tc_file": tc_file_name,
            "tc_sheet": tc_sheet_name,
            "video_file": video_file_name,
            "arrival_time": arrival_time,
            "trim_start": trim_start,
            "trim_end": trim_end,
            "method": method,
            "status": trim_status
        })

    trim_plan = pd.DataFrame(results)

    trim_plan.to_excel(OUTPUT_XLSX, index=False)

    print()
    print("Trim plan created.")
    print(f"Saved to: {OUTPUT_XLSX}")

    return trim_plan


# ============================================================
# TRIM VIDEO
# ============================================================

def trim_video_ffmpeg(input_video, output_video, trim_start, trim_end):
    """
    Trim one video using ffmpeg.

    ffmpeg must be installed and available from the command line.

    trim_start and trim_end should be HH:MM:SS strings.
    """

    output_video.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-ss", trim_start,
        "-to", trim_end,
        "-i", str(input_video),
        "-c", "copy",
        str(output_video)
    ]

    subprocess.run(command, check=True)


# ============================================================
# TRIM VIDEOS FROM REVIEWED PLAN
# ============================================================

def trim_videos_from_plan():
    """
    Trim videos using the Excel trim plan.

    This only trims rows marked ready_to_trim.
    """

    if not OUTPUT_XLSX.exists():
        print("No trim plan found. Create the Excel trim plan first.")
        return

    plan = pd.read_excel(OUTPUT_XLSX)

    for _, row in plan.iterrows():

        if row["status"] != "ready_to_trim":
            continue

        row_number = int(row["row"])

        video_file_name = row["video_file"]
        trim_start = row["trim_start"]
        trim_end = row["trim_end"]

        input_video = VIDEO_FOLDER / video_file_name

        output_video = OUTPUT_VIDEO_FOLDER / f"recording_{row_number:02d}_trimmed.avi"

        print(f"Trimming row {row_number}: {video_file_name}")

        try:
            trim_video_ffmpeg(
                input_video=input_video,
                output_video=output_video,
                trim_start=trim_start,
                trim_end=trim_end
            )

        except Exception as e:
            print(f"  Failed to trim row {row_number}: {e}")

    print()
    print("Video trimming complete.")
    print(f"Trimmed videos saved to: {OUTPUT_VIDEO_FOLDER}")


# ============================================================
# MAIN SCRIPT
# ============================================================

def main():

    create_trim_plan()

    if PROCEED_WITH_TRIMMING:
        print()
        print("PROCEED_WITH_TRIMMING is True.")
        print("Starting video trimming...")
        trim_videos_from_plan()
    else:
        print()
        print("PROCEED_WITH_TRIMMING is False.")
        print("No videos have been trimmed yet.")
        print("Check the Excel trim plan first.")
        print("Then set PROCEED_WITH_TRIMMING = True to trim the videos.")


if __name__ == "__main__":
    main()