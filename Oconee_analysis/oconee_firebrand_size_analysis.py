"""
Updated on July 18 2026

@author: Andrew Lawson

Oconee firebrand apparent-size analysis workflow

Purpose
-------
This script analyses multiple trimmed wildfire videos using the trained YOLO
firebrand detector and extracts the apparent size of every retained detection.

The script is designed for the Oconee SE2 dataset and performs batch processing
across all compatible videos in a selected folder. For each video, it detects
firebrands, removes overlapping duplicate detections using non-maximum
suppression, converts bounding-box area from pixels squared to millimetres
squared using the supplied image calibration, and produces both per-video and
combined size-distribution outputs.

Workflow
--------
1. Find all supported video files in VIDEO_DIR.
2. Load the trained YOLO detector once.
3. Sample each video at FPS_SAMPLE frames per second.
4. Pad smaller 640 x 480 frames to the 640 x 640 model input size.
5. Run YOLO inference at the selected confidence threshold.
6. Ignore detections that fall entirely inside the padded image region.
7. Apply non-maximum suppression to remove duplicate overlapping detections.
8. Calculate each retained bounding-box width, height and area.
9. Convert bounding-box area from px² to apparent mm² using the calibration.
10. Optionally remove detections outside chosen minimum / maximum area limits.
11. Save one detection-level CSV and several histograms for each video.
12. Pool all detections across all videos and save combined distributions.
13. Save per-video summary statistics to an Excel workbook.

Important interpretation note
-----------------------------
The calculated area represents an apparent ML-derived bounding-box area rather
than the true physical projected area of the firebrand. The value can be
affected by particle orientation, motion blur, glow, focus, camera geometry,
perspective and bounding-box fit.

The pixel-to-mm² conversion is only valid where the supplied calibration
represents the camera geometry used in the analysed recording.
"""

from pathlib import Path
import math
import csv

import cv2
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO


# ============================================================
# USER SETTINGS
# ============================================================

# Folder containing videos to analyse
VIDEO_DIR = Path(r"Path")

# Path to trained YOLO model
MODEL_PATH = r"Path"

# Output folder
OUTPUT_DIR = VIDEO_DIR / "firebrand_size_distribution_results"

# Video extensions to process
VIDEO_EXTENSIONS = [".avi", ".mp4", ".mov"]

# YOLO confidence threshold
CONF = 0.4

# Sampling rate
FPS_SAMPLE = 2.0

# YOLO image size
TILE = 640

# NMS threshold
IOU_NMS = 0.5

# Pixel calibration
# Change these if using a different camera geometry/calibration
PX_PER_MM_X = 10.7
PX_PER_MM_Y = 8.0

# Histogram settings
HIST_BINS = 50

# Optional filtering of extreme detections
# Set to None to disable upper filtering
MIN_AREA_MM2 = 0.0
MAX_AREA_MM2 = None

# Save detailed detection table?
SAVE_DETECTIONS_CSV = True


# ============================================================
# DERIVED SETTINGS
# ============================================================

PX2_PER_MM2 = PX_PER_MM_X * PX_PER_MM_Y


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def iou_xyxy(a, b):
    """
    Compute Intersection-over-Union between two boxes.
    Boxes are in [x1, y1, x2, y2] format.
    """

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)

    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection + 1e-9

    return intersection / union


def nms(boxes, confs, iou_threshold):
    """
    Non-maximum suppression to remove duplicate overlapping detections.
    """

    idxs = sorted(range(len(boxes)), key=lambda i: confs[i], reverse=True)
    keep = []

    while idxs:
        i = idxs.pop(0)
        keep.append(i)

        idxs = [
            j for j in idxs
            if iou_xyxy(boxes[i], boxes[j]) < iou_threshold
        ]

    return keep


def clean_name(path):
    """
    Clean video filename stem for safe output filenames.
    """

    return path.stem.replace(" ", "_").replace(".", "_")


def find_videos(video_dir):
    """
    Find all videos in the folder matching the allowed extensions.
    """

    videos = []

    for ext in VIDEO_EXTENSIONS:
        videos.extend(video_dir.glob(f"*{ext}"))

    return sorted(videos)


def percentile(values, p):
    """
    Calculate percentile using linear interpolation.
    """

    if not values:
        return 0.0

    values = sorted(values)

    if len(values) == 1:
        return float(values[0])

    k = (len(values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return float(values[int(k)])

    return float(values[f] * (c - k) + values[c] * (k - f))


def pad_frame_to_square(frame, tile_size):
    """
    Pad a frame to tile_size x tile_size if it is smaller.

    This is important for 640 x 480 SE2 videos.
    The frame becomes 640 x 640 before YOLO inference.
    Detections in the padded region are ignored later.
    """

    h, w = frame.shape[:2]

    pad_bottom = max(0, tile_size - h)
    pad_right = max(0, tile_size - w)

    if pad_bottom > 0 or pad_right > 0:
        frame_padded = cv2.copyMakeBorder(
            frame,
            0,
            pad_bottom,
            0,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )
    else:
        frame_padded = frame

    return frame_padded


# ============================================================
# VIDEO ANALYSIS
# ============================================================

def analyse_video_size_distribution(video_path, model):
    """
    Analyse one video and return a dataframe of individual firebrand detections.

    Each row corresponds to one deduplicated ML detection.
    """

    print(f"\nAnalysing: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"  Could not open video: {video_path}")
        return pd.DataFrame()

    native_fps = cap.get(cv2.CAP_PROP_FPS)

    if not native_fps or native_fps <= 0:
        print(f"  Could not read FPS for: {video_path.name}")
        cap.release()
        return pd.DataFrame()

    step = max(1, int(round(native_fps / FPS_SAMPLE)))

    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Resolution: {original_w} x {original_h}")
    print(f"  Native FPS: {native_fps:.2f}")
    print(f"  Sampling every {step} frames")

    detection_rows = []

    frame_idx = -1
    sample_idx = -1

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame_idx += 1

        if frame_idx % step != 0:
            continue

        sample_idx += 1
        time_s = sample_idx / FPS_SAMPLE

        input_img = pad_frame_to_square(frame, TILE)

        result = model.predict(
            input_img,
            imgsz=TILE,
            conf=CONF,
            verbose=False
        )[0]

        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes = []
        confs = []

        for b in result.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0])

            # Ignore detections fully outside the original frame
            # This prevents counting detections in the padded black region
            if x1 >= original_w or y1 >= original_h:
                continue

            # Clip boxes back to original image region
            x1 = max(0.0, min(x1, original_w))
            x2 = max(0.0, min(x2, original_w))
            y1 = max(0.0, min(y1, original_h))
            y2 = max(0.0, min(y2, original_h))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append((x1, y1, x2, y2))
            confs.append(conf)

        if not boxes:
            continue

        keep = nms(boxes, confs, IOU_NMS)

        for det_id, i in enumerate(keep):
            x1, y1, x2, y2 = boxes[i]
            conf = confs[i]

            width_px = x2 - x1
            height_px = y2 - y1

            area_px = width_px * height_px
            area_mm2 = area_px / PX2_PER_MM2

            # Optional filtering
            if area_mm2 <= MIN_AREA_MM2:
                continue

            if MAX_AREA_MM2 is not None and area_mm2 > MAX_AREA_MM2:
                continue

            detection_rows.append({
                "video": video_path.name,
                "sample_index": sample_idx,
                "time_s": time_s,
                "det_id": det_id,
                "conf": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width_px": width_px,
                "height_px": height_px,
                "area_px": area_px,
                "area_mm2": area_mm2
            })

    cap.release()

    df = pd.DataFrame(detection_rows)

    print(f"  Detections retained: {len(df)}")

    return df


# ============================================================
# PLOTTING
# ============================================================

def plot_size_distribution(df, video_path, output_dir):
    """
    Save size distribution histograms for one video.
    The graph title includes the original video filename.
    """

    if df.empty:
        print(f"  No detections to plot for {video_path.name}")
        return

    video_name = clean_name(video_path)

    # --------------------------------------------------------
    # Histogram in mm²
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))
    plt.hist(df["area_mm2"], bins=HIST_BINS)
    plt.xlabel("Apparent bounding-box area (mm$^2$)")
    plt.ylabel("Frequency")
    plt.title(f"Firebrand apparent size distribution\nVideo: {video_path.name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{video_name}_area_distribution_mm2.png", dpi=300)
    plt.close()

    # --------------------------------------------------------
    # Histogram in px²
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))
    plt.hist(df["area_px"], bins=HIST_BINS)
    plt.xlabel("Apparent bounding-box area (px$^2$)")
    plt.ylabel("Frequency")
    plt.title(f"Firebrand apparent size distribution\nVideo: {video_path.name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{video_name}_area_distribution_px2.png", dpi=300)
    plt.close()

    # --------------------------------------------------------
    # Log-scale histogram in mm²
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))
    plt.hist(df["area_mm2"], bins=HIST_BINS)
    plt.xscale("log")
    plt.xlabel("Apparent bounding-box area (mm$^2$) [log scale]")
    plt.ylabel("Frequency")
    plt.title(f"Firebrand apparent size distribution, log scale\nVideo: {video_path.name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{video_name}_area_distribution_mm2_log.png", dpi=300)
    plt.close()

    print(f"  Saved size distribution plots for {video_path.name}")


def plot_combined_size_distribution(all_df, output_dir):
    """
    Save one combined size distribution plot for all videos.
    """

    if all_df.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(all_df["area_mm2"], bins=HIST_BINS)
    plt.xlabel("Bounding-box area (mm$^2$)")
    plt.ylabel("Frequency")
    plt.title("Combined firebrand size distribution across all videos")
    plt.tight_layout()
    plt.savefig(output_dir / "combined_area_distribution_mm2.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(all_df["area_mm2"], bins=HIST_BINS)
    plt.xscale("log")
    plt.xlabel("Bounding-box area (mm$^2$) [log scale]")
    plt.ylabel("Frequency")
    plt.title("Combined firebrand size distribution across all videos")
    plt.tight_layout()
    plt.savefig(output_dir / "combined_area_distribution_mm2_log.png", dpi=300)
    plt.close()


# ============================================================
# SUMMARY
# ============================================================

def build_summary(df, video_path):
    """
    Build summary statistics for one video.
    """

    if df.empty:
        return {
            "video": video_path.name,
            "n_detections": 0,
            "mean_area_mm2": 0,
            "median_area_mm2": 0,
            "p95_area_mm2": 0,
            "max_area_mm2": 0,
            "mean_area_px": 0,
            "median_area_px": 0,
            "p95_area_px": 0,
            "max_area_px": 0
        }

    area_mm2 = df["area_mm2"].tolist()
    area_px = df["area_px"].tolist()

    return {
        "video": video_path.name,
        "n_detections": len(df),
        "mean_area_mm2": sum(area_mm2) / len(area_mm2),
        "median_area_mm2": percentile(area_mm2, 50),
        "p95_area_mm2": percentile(area_mm2, 95),
        "max_area_mm2": max(area_mm2),
        "mean_area_px": sum(area_px) / len(area_px),
        "median_area_px": percentile(area_px, 50),
        "p95_area_px": percentile(area_px, 95),
        "max_area_px": max(area_px)
    }


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = find_videos(VIDEO_DIR)

    if not videos:
        print(f"No videos found in: {VIDEO_DIR}")
        return

    print(f"Found {len(videos)} videos.")

    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    all_detection_tables = []
    summary_rows = []

    for video_path in videos:
        df = analyse_video_size_distribution(video_path, model)

        summary_rows.append(build_summary(df, video_path))

        if not df.empty:
            all_detection_tables.append(df)

            # Save one CSV per video
            if SAVE_DETECTIONS_CSV:
                video_name = clean_name(video_path)
                df.to_csv(
                    OUTPUT_DIR / f"{video_name}_detections_with_size.csv",
                    index=False
                )

            # Save graphs
            plot_size_distribution(df, video_path, OUTPUT_DIR)

    # Combine all videos
    if all_detection_tables:
        all_df = pd.concat(all_detection_tables, ignore_index=True)

        all_df.to_csv(
            OUTPUT_DIR / "all_videos_detections_with_size.csv",
            index=False
        )

        plot_combined_size_distribution(all_df, OUTPUT_DIR)

    else:
        all_df = pd.DataFrame()

    # Save summary table
    summary_df = pd.DataFrame(summary_rows)

    summary_excel = OUTPUT_DIR / "firebrand_size_distribution_summary.xlsx"

    with pd.ExcelWriter(summary_excel, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)

        if not all_df.empty:
            all_df.to_excel(writer, sheet_name="all_detections", index=False)

    print()
    print("Done.")
    print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
