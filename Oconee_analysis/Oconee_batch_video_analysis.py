"""
Updated on July 27 2026

@author: Andrew Lawson

Batch firebrand video analysis for Oconee SE2 recordings.

This script applies a trained YOLO firebrand detector to every trimmed AVI video
in a selected folder. Each sampled frame is padded from 640x480 to 640x640
before inference, detections are deduplicated using non-maximum suppression,
and frame-level count/size statistics are calculated.

Outputs:
- one Excel workbook containing one sheet per video;
- an overview sheet containing summary statistics for all videos;
- firebrand count-vs-time plots for each video;
- cumulative firebrand count-vs-time plots for each video.

The script is intended for batch analysis of Oconee recordings after the long
raw videos have already been trimmed to the fire-event analysis window.
"""

import math
from pathlib import Path

import cv2
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO


# ============================================================
# USER CONFIG
# ============================================================

# Folder containing the trimmed Oconee .avi videos to analyse.
# Every .avi file directly inside this folder will be processed.
VIDEO_DIR = Path(r"Path")

# Path to the trained YOLO firebrand detector weights.
# Keep the model fixed when comparing videos within the same analysis run.
MODEL_PATH = r"runs/detect/train7/weights/best.pt"

# Combined Excel workbook. One worksheet is written for each video,
# plus an overview sheet containing one summary row per recording.
OUTPUT_XLSX = VIDEO_DIR / "output.xlsx"

# Folder used to store count-vs-time and cumulative-count plots.
PLOTS_DIR = VIDEO_DIR / "firebrand_graphs"

# ============================================================
# FIXED CONFIG
# ============================================================

# Effective video sampling rate used for ML inference (frames per second).
# At 0.5 FPS, one frame is analysed every 2 seconds.
FPS_SAMPLE = 0.5

# YOLO inference image size. Oconee videos are 640x480 and are padded to
# 640x640 before inference so that no valid portion of the frame is discarded.
TILE = 640

# Retained for compatibility with the earlier tiled workflow. The current
# 640x480 analysis uses one padded full frame rather than overlapping tiles.
OVERLAP = 0.25
STRIDE = int(TILE * (1 - OVERLAP))

# YOLO confidence threshold. Set this to the chosen Oconee working value
# (for example 0.4) before running the analysis.
CONF = 0.4

# IoU threshold for non-maximum suppression. Boxes that overlap above this
# level are treated as duplicate detections and the lower-confidence box is removed.
IOU_NMS = 0.5

# Smoothing-window setting retained from the original single-video script.
WINDOW_S = 5

# Physical collection/deposition area used when normalising count to count/m^2.
DEPOSITION_AREA_M2 = 0.03


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def iou_xyxy(a, b):
    """
    Compute intersection-over-union (IoU) for two bounding boxes.

    IoU measures how strongly two boxes overlap and is used by NMS to decide
    whether two predicted boxes are likely to represent the same firebrand.
    """

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)

    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter + 1e-9

    return inter / union


def save_video_plots(video_path, frame_df):
    """
    Save two plots for one video:

    1. Firebrand count vs time
    2. Cumulative firebrand count vs time

    The plots are saved as PNG files in PLOTS_DIR.
    """

    # Create the plots folder if it does not already exist
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # If there is no frame data, skip plotting
    if frame_df.empty:
        print(f"  No frame data available for plotting: {video_path.name}")
        return

    # Extract time values from dataframe
    time_s = frame_df["time_s"].tolist()

    # Extract firebrand count per sampled frame
    counts = frame_df["count_dedup"].tolist()

    # Calculate cumulative count over time
    cumulative_counts = frame_df["count_dedup"].cumsum().tolist()

    # Clean video name for output filenames
    video_stem = video_path.stem

    # ------------------------------------------------------------
    # Plot 1: firebrand count vs time
    # ------------------------------------------------------------

    plt.figure()

    plt.plot(time_s, counts)

    plt.xlabel("Time (s)")
    plt.ylabel("Firebrand count per sampled frame")
    plt.title(f"Firebrand count vs time\n{video_path.name}")

    plt.tight_layout()

    count_plot_path = PLOTS_DIR / f"{video_stem}_count_vs_time.png"

    plt.savefig(count_plot_path, dpi=300)

    plt.close()

    # ------------------------------------------------------------
    # Plot 2: cumulative firebrand count vs time
    # ------------------------------------------------------------

    plt.figure()

    plt.plot(time_s, cumulative_counts)

    plt.xlabel("Time (s)")
    plt.ylabel("Cumulative firebrand count")
    plt.title(f"Cumulative firebrand count vs time\n{video_path.name}")

    plt.tight_layout()

    cumulative_plot_path = PLOTS_DIR / f"{video_stem}_cumulative_count_vs_time.png"

    plt.savefig(cumulative_plot_path, dpi=300)

    plt.close()

    print(f"  Saved plots for: {video_path.name}")


def nms(boxes, confs, iou_thr):
    """
    Apply non-maximum suppression (NMS) to remove duplicate boxes.

    Detections are sorted from highest to lowest confidence. Lower-confidence
    boxes are discarded when they overlap a retained box by more than iou_thr.
    """

    idxs = sorted(range(len(boxes)), key=lambda i: confs[i], reverse=True)
    keep = []

    while idxs:
        i = idxs.pop(0)
        keep.append(i)

        idxs = [
            j for j in idxs
            if iou_xyxy(boxes[i], boxes[j]) < iou_thr
        ]

    return keep


def tile_coords(W, H, tile=TILE, stride=STRIDE):
    """
    Generate top-left tile coordinates that cover a full frame.

    This helper is retained from the original tiled-inference workflow. The
    current Oconee 640x480 method instead pads the full frame to 640x640.
    """

    xs = list(range(0, max(1, W - tile + 1), stride))
    ys = list(range(0, max(1, H - tile + 1), stride))

    if xs[-1] != W - tile:
        xs.append(max(0, W - tile))

    if ys[-1] != H - tile:
        ys.append(max(0, H - tile))

    xs = sorted(set(xs))
    ys = sorted(set(ys))

    for y0 in ys:
        for x0 in xs:
            yield x0, y0


def rolling_mean(values, window):
    """
    Compute a centred rolling mean for a one-dimensional series.

    This helper can be used to smooth noisy frame-by-frame count or size data.
    """

    out = []

    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2)
        out.append(sum(values[start:end]) / max(1, end - start))

    return out


def percentile(sorted_vals, p):
    """
    Calculate a percentile from a sorted sequence using linear interpolation.
    """

    if not sorted_vals:
        return 0.0

    if len(sorted_vals) == 1:
        return float(sorted_vals[0])

    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return float(sorted_vals[int(k)])

    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)

    return float(d0 + d1)


def clean_sheet_name(name):
    """
    Excel sheet names cannot be longer than 31 characters and cannot contain
    certain special characters.
    """

    bad_chars = ["\\", "/", "*", "?", ":", "[", "]"]

    for char in bad_chars:
        name = name.replace(char, "_")

    return name[:31]


# ============================================================
# MAIN ANALYSIS FUNCTION FOR ONE VIDEO
# ============================================================

def analyse_video(video_path, model):
    """
    Analyse one 640x480 AVI video and return:
    - frame-level results dataframe
    - detection-level results dataframe
    - summary statistics dataframe

    This version pads each 640x480 frame to 640x640 before YOLO inference.
    This avoids the issue where the old tiling method skipped frames that
    were not exactly 640x640.
    """

    print(f"Analysing video: {video_path.name}")

    # ------------------------------------------------------------
    # OPEN VIDEO
    # ------------------------------------------------------------

    # Open the video using OpenCV
    cap = cv2.VideoCapture(str(video_path))

    # Check that OpenCV successfully opened the video
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Read the native frame rate of the video
    native_fps = cap.get(cv2.CAP_PROP_FPS)

    # Check that FPS was read correctly
    if not native_fps or native_fps <= 0:
        raise RuntimeError(f"Could not read FPS for video: {video_path}")

    # Work out how many frames to skip to sample at FPS_SAMPLE
    # Example: if native_fps = 30 and FPS_SAMPLE = 2, step = 15
    step = max(1, int(round(native_fps / FPS_SAMPLE)))

    # Read video frame width
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Read video frame height
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Print resolution so you can confirm the video is 640x480
    print(f"  Resolution: {W} x {H}")
    print(f"  Native FPS: {native_fps:.2f}")
    print(f"  Sampling every {step} frames")

    # ------------------------------------------------------------
    # STORAGE
    # ------------------------------------------------------------

    # This stores one summary row per analysed frame
    frame_rows = []

    # This stores one row per individual detection
    det_rows = []

    # Native video frame index
    frame_idx = -1

    # Analysed/sampled frame index
    sample_idx = -1

    # ------------------------------------------------------------
    # MAIN VIDEO LOOP
    # ------------------------------------------------------------

    while True:

        # Read the next frame from the video
        ok, frame = cap.read()

        # Stop if there are no more frames
        if not ok:
            break

        # Increase native frame counter
        frame_idx += 1

        # Skip frames so we only analyse FPS_SAMPLE frames per second
        if frame_idx % step != 0:
            continue

        # Increase analysed frame counter
        sample_idx += 1

        # Time assigned to this analysed frame
        time_s_val = sample_idx / FPS_SAMPLE

        # --------------------------------------------------------
        # PAD FRAME TO 640x640
        # --------------------------------------------------------

        # Current frame height and width
        frame_h, frame_w = frame.shape[:2]

        # Amount of padding required at bottom and right
        pad_bottom = max(0, TILE - frame_h)
        pad_right = max(0, TILE - frame_w)

        # If the frame is smaller than 640x640, pad it with black pixels
        # For 640x480 video, this adds 160 black pixels at the bottom
        if pad_bottom > 0 or pad_right > 0:
            input_img = cv2.copyMakeBorder(
                frame,
                0,
                pad_bottom,
                0,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )

        # If the frame is already 640x640, use it directly
        else:
            input_img = frame

        # --------------------------------------------------------
        # RUN YOLO ON PADDED FRAME
        # --------------------------------------------------------

        # Run YOLO inference on the padded frame
        res = model.predict(
            input_img,
            imgsz=TILE,
            conf=CONF,
            verbose=False
        )[0]

        # Storage for detections in the original 640x480 frame
        full_boxes = []
        full_confs = []

        # If YOLO found detections, process them
        if res.boxes is not None and len(res.boxes) > 0:

            # Loop through each detected box
            for b in res.boxes:

                # Extract bounding box coordinates
                x1, y1, x2, y2 = b.xyxy[0].tolist()

                # Extract confidence score
                conf = float(b.conf[0])

                # Ignore detections that are fully inside the padded black region
                # For 640x480 video, anything with y1 >= 480 is in the padded region
                if y1 >= H:
                    continue

                # Ignore detections that are fully beyond the right edge
                if x1 >= W:
                    continue

                # Clip boxes so they stay inside the original video frame
                x1 = max(0, min(x1, W))
                x2 = max(0, min(x2, W))
                y1 = max(0, min(y1, H))
                y2 = max(0, min(y2, H))

                # Ignore invalid boxes after clipping
                if x2 <= x1 or y2 <= y1:
                    continue

                # Store full-frame box coordinates
                full_boxes.append((x1, y1, x2, y2))

                # Store confidence
                full_confs.append(conf)

        # --------------------------------------------------------
        # OPTIONAL NMS
        # --------------------------------------------------------

        # With one padded full-frame image, duplicates are less likely than
        # with overlapping tiles, but NMS is still safe to keep.
        if full_boxes:
            keep = nms(full_boxes, full_confs, IOU_NMS)
            boxes = [full_boxes[i] for i in keep]
            confs = [full_confs[i] for i in keep]
        else:
            boxes = []
            confs = []

        # --------------------------------------------------------
        # CALCULATE DETECTION AREAS
        # --------------------------------------------------------

        areas = []

        # Calculate bounding-box area for each retained detection
        for x1, y1, x2, y2 in boxes:

            # Width of bounding box in pixels
            wpx = max(0.0, x2 - x1)

            # Height of bounding box in pixels
            hpx = max(0.0, y2 - y1)

            # Area in pixels squared
            areas.append(wpx * hpx)

        # --------------------------------------------------------
        # FRAME-LEVEL METRICS
        # --------------------------------------------------------

        # Number of firebrands detected in this sampled frame
        count = len(boxes)

        # Count normalised by deposition area
        count_per_m2 = count / DEPOSITION_AREA_M2

        # If there are detections, calculate size/confidence statistics
        if areas:

            # Sort areas for median and percentile calculations
            areas_sorted = sorted(areas)

            # Sum of all bounding-box areas
            area_sum = sum(areas_sorted)

            # Mean bounding-box area
            area_mean = area_sum / count

            # Median bounding-box area
            if count % 2:
                area_median = areas_sorted[count // 2]
            else:
                area_median = 0.5 * (
                    areas_sorted[count // 2 - 1]
                    + areas_sorted[count // 2]
                )

            # 95th percentile bounding-box area
            area_p95 = percentile(areas_sorted, 95)

            # Mean YOLO confidence for this frame
            conf_mean = sum(confs) / count

        # If there are no detections, all size/confidence values are zero
        else:
            area_sum = 0.0
            area_mean = 0.0
            area_median = 0.0
            area_p95 = 0.0
            conf_mean = 0.0

        # Store one row for this analysed frame
        frame_rows.append({
            "sample_index": sample_idx,
            "time_s": round(time_s_val, 3),
            "count_dedup": count,
            "count_per_m2": round(count_per_m2, 3),
            "area_sum_px": round(area_sum, 2),
            "area_mean_px": round(area_mean, 2),
            "area_median_px": round(area_median, 2),
            "area_p95_px": round(area_p95, 2),
            "conf_mean": round(conf_mean, 3)
        })

        # --------------------------------------------------------
        # DETECTION-LEVEL METRICS
        # --------------------------------------------------------

        # Store one row per individual detection
        for det_id, ((x1, y1, x2, y2), conf, area) in enumerate(
            zip(boxes, confs, areas)
        ):
            det_rows.append({
                "sample_index": sample_idx,
                "time_s": round(time_s_val, 3),
                "det_id": det_id,
                "conf": round(conf, 3),
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
                "area_px": round(area, 2)
            })

    # ------------------------------------------------------------
    # RELEASE VIDEO
    # ------------------------------------------------------------

    # Close the video file
    cap.release()

    # Convert frame-level results to dataframe
    frame_df = pd.DataFrame(frame_rows)

    # Convert detection-level results to dataframe
    det_df = pd.DataFrame(det_rows)

    # ------------------------------------------------------------
    # HANDLE EMPTY VIDEO CASE
    # ------------------------------------------------------------

    # If no frames were analysed, return an error-style summary
    if frame_df.empty:
        summary_df = pd.DataFrame({
            "metric": ["video_name", "status"],
            "value": [video_path.name, "no_frames_analysed"]
        })

        return frame_df, det_df, summary_df

    # ------------------------------------------------------------
    # SUMMARY STATISTICS
    # ------------------------------------------------------------

    # Extract counts and times from frame dataframe
    counts = frame_df["count_dedup"].tolist()
    counts_per_m2 = frame_df["count_per_m2"].tolist()
    time_s = frame_df["time_s"].tolist()

    # Number of analysed frames
    n = len(counts)

    # Total detections across sampled frames
    total_count = sum(counts)

    # Mean count per analysed frame
    mean_count_frame = total_count / n if n else 0.0

    # Approximate mean count per second
    mean_count_sec = mean_count_frame * FPS_SAMPLE

    # Mean count per square metre per analysed frame
    mean_count_per_m2_frame = sum(counts_per_m2) / n if n else 0.0

    # Variance of count per frame
    variance = (
        sum((c - mean_count_frame) ** 2 for c in counts) / n
        if n else 0.0
    )

    # Standard deviation of count per frame
    std_count = math.sqrt(variance) if n else 0.0

    # Peak count in any analysed frame
    peak_count = max(counts) if counts else 0

    # Peak count per square metre
    peak_count_per_m2 = max(counts_per_m2) if counts_per_m2 else 0.0

    # Time at which peak count occurred
    peak_time = time_s[counts.index(peak_count)] if counts else 0.0

    # Number of frames with at least one detection
    active_frames = sum(1 for c in counts if c > 0)

    # Duration with at least one detection
    active_duration_s = active_frames / FPS_SAMPLE

    # Build summary table
    summary_rows = [
        ["video_name", video_path.name],
        ["video_path", str(video_path)],
        ["native_fps", native_fps],
        ["fps_sample", FPS_SAMPLE],
        ["frame_width_px", W],
        ["frame_height_px", H],
        ["frames_analysed", n],
        ["total_count_sum_frames", total_count],
        ["mean_count_per_frame", mean_count_frame],
        ["mean_count_per_second", mean_count_sec],
        ["mean_count_per_m2_per_frame", mean_count_per_m2_frame],
        ["std_count_per_frame", std_count],
        ["peak_count", peak_count],
        ["peak_count_per_m2", peak_count_per_m2],
        ["peak_time_s", peak_time],
        ["active_duration_s", active_duration_s],
        ["tile_px", TILE],
        ["conf_threshold", CONF],
        ["nms_iou", IOU_NMS],
        ["deposition_area_m2", DEPOSITION_AREA_M2],
        ["method_note", "640x480 frame padded to 640x640 before YOLO inference"]
    ]

    # Convert summary to dataframe
    summary_df = pd.DataFrame(summary_rows, columns=["metric", "value"])

    # Return all three result tables
    return frame_df, det_df, summary_df


# ============================================================
# WRITE ONE SHEET PER VIDEO
# ============================================================

def write_video_sheet(writer, video_path, frame_df, summary_df):
    """
    Write one Excel sheet for one video.

    The sheet contains:
    - summary statistics at the top
    - frame-level results below
    """

    sheet_name = clean_sheet_name(video_path.stem)

    start_row_for_frame_data = len(summary_df) + 3

    summary_df.to_excel(
        writer,
        sheet_name=sheet_name,
        index=False,
        startrow=0
    )

    frame_df.to_excel(
        writer,
        sheet_name=sheet_name,
        index=False,
        startrow=start_row_for_frame_data
    )


# ============================================================
# MAIN SCRIPT
# ============================================================

def main():
    """Run the complete batch-analysis workflow for every AVI in VIDEO_DIR."""

    # Find every AVI file directly inside the selected input folder.
    avi_files = sorted(VIDEO_DIR.glob("*.avi"))

    if not avi_files:
        print(f"No .avi files found in: {VIDEO_DIR}")
        return

    print(f"Found {len(avi_files)} .avi videos.")

    # Load the trained detector once and reuse it for every video.
    # Reloading the model for each file would add unnecessary overhead.
    model = YOLO(MODEL_PATH)

    # Collect one compact summary dictionary per video for the overview sheet.
    all_summary_rows = []

    # Keep one Excel writer open for the complete batch so all recordings are
    # written into the same workbook.
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:

        # Process each recording independently. If one fails, the remaining
        # recordings are still analysed.
        for video_path in avi_files:

            try:
                frame_df, det_df, summary_df = analyse_video(video_path, model)

                write_video_sheet(
                    writer=writer,
                    video_path=video_path,
                    frame_df=frame_df,
                    summary_df=summary_df
                )
                
                save_video_plots(
                    video_path=video_path,
                    frame_df=frame_df
                )

                # Also collect one compact row for the overview sheet
                summary_dict = dict(zip(summary_df["metric"], summary_df["value"]))
                all_summary_rows.append(summary_dict)

            except Exception as e:
                print(f"Error processing {video_path.name}: {e}")

                error_df = pd.DataFrame({
                    "metric": ["video_name", "status", "error"],
                    "value": [video_path.name, "failed", str(e)]
                })

                sheet_name = clean_sheet_name(video_path.stem)
                error_df.to_excel(writer, sheet_name=sheet_name, index=False)

                all_summary_rows.append({
                    "video_name": video_path.name,
                    "status": "failed",
                    "error": str(e)
                })

        # Convert the collected summaries into a single campaign-level table.
        overview_df = pd.DataFrame(all_summary_rows)

        # Write the overview sheet after all individual video sheets.
        overview_df.to_excel(
            writer,
            sheet_name="overview",
            index=False
        )

    print()
    print("Done.")
    print(f"Saved Excel results to: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
