"""
Updated on Mar 25 2026

@author: Andrew Lawson

Tile-based firebrand inference and statistics for 640x480 SE2 video.

Outputs:
- counts and size statistics by frame
- detections by frame CSV
- summary CSV and TXT
- count/size plots
- histogram plots
- temperature plots (if metadata exists)
- annotated sampled frames
- cropped firebrand detections
"""

import csv
import math
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ---------------- USER CONFIG ----------------

# Name of the recording folder to analyse when using the SE2 folder structure
RECORDING_DIR_NAME = "recording_analysis_insert_here"

# Path to the trained YOLO weights file
MODEL_PATH = r"runs/detect/train5/weights/best.pt"

# Set to False when analysing a standalone video without the recording metadata folder structure
USE_METADATA = False

# Direct path to the video file when USE_METADATA is False
DIRECT_VIDEO_PATH = r"dataset_eval/videos/rgb_video_DYYYYMMDD_THHMMSS.mp4"

# --------------------------------------------


# ---------------- FIXED CONFIG ----------------

# Desired sampled analysis frame rate in Hz
FPS_SAMPLE = 10.0

# Tile size used for inference on 640x480 imagery
# This should match the tile size used during model training
TILE = 320

# Fractional overlap between adjacent tiles
OVERLAP = 0.25

# Tile step size derived from tile size and overlap
STRIDE = int(TILE * (1 - OVERLAP))   # 240

# Detection confidence threshold passed to YOLO
CONF = 0.5

# IoU threshold for frame-level non-maximum suppression
IOU_NMS = 0.5

# Rolling-mean window in seconds for smoothed time-series plots
WINDOW_S = 5

# Save detailed CSV containing every retained detection
SAVE_DETECTIONS_CSV = True

# Save a copy of each sampled frame with retained detections drawn on it
SAVE_ANNOTATED_FRAMES = True

# Save cropped image patches for each retained detection
SAVE_DETECTION_CROPS = True

# ---------------------------------------------


# Resolve the directory containing this script
BASE_DIR = Path(__file__).resolve().parent

# Resolve the chosen recording directory
RECORDING_DIR = BASE_DIR / RECORDING_DIR_NAME

# Set input/output paths based on whether full metadata structure is being used
if USE_METADATA:
    # Standard SE2 folder layout
    VIDEO_PATH   = RECORDING_DIR / "inputs" / "videos" / "final_rgb_video.mp4"
    METADATA_DIR = RECORDING_DIR / "inputs" / "metadata"
    TEMP_PATH    = METADATA_DIR / "temperature_log.csv"
    GNSS_PATH    = METADATA_DIR / "GNSS_adjustment.txt"
    RESULTS_DIR  = RECORDING_DIR / "results"
else:
    # Standalone evaluation mode using a direct video path
    VIDEO_PATH   = BASE_DIR / DIRECT_VIDEO_PATH
    TEMP_PATH    = None
    GNSS_PATH    = None
    RESULTS_DIR  = BASE_DIR / "results_se2_eval"

# Create main results directory if it does not already exist
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Output directory for annotated sampled frames
ANNOTATED_DIR = RESULTS_DIR / "annotated_frames"

# Output directory for individual cropped detections
CROPS_DIR = RESULTS_DIR / "detected_firebrands"

# Create optional output folders only if those outputs are enabled
if SAVE_ANNOTATED_FRAMES:
    ANNOTATED_DIR.mkdir(exist_ok=True)
if SAVE_DETECTION_CROPS:
    CROPS_DIR.mkdir(exist_ok=True)


def iou_xyxy(a, b):
    """
    Compute intersection-over-union (IoU) for two bounding boxes in
    [x1, y1, x2, y2] format.

    IoU is used during non-maximum suppression to remove overlapping
    duplicate detections caused by tile overlap.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    # Coordinates of the overlap region
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    # Overlap width and height
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    # Individual box areas
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    # Union with tiny epsilon to avoid divide-by-zero
    union = area_a + area_b - inter + 1e-9

    return inter / union


def nms(boxes, confs, iou_thr):
    """
    Apply non-maximum suppression to a list of bounding boxes.

    Detections are sorted by confidence, and lower-confidence boxes are
    removed when they overlap too strongly with a higher-confidence box.
    """
    idxs = sorted(range(len(boxes)), key=lambda i: confs[i], reverse=True)
    keep = []

    while idxs:
        # Keep the highest-confidence remaining detection
        i = idxs.pop(0)
        keep.append(i)

        # Remove detections that overlap too much with the kept one
        idxs = [j for j in idxs if iou_xyxy(boxes[i], boxes[j]) < iou_thr]

    return keep


def tile_coords(W, H, tile=TILE, stride=STRIDE):
    """
    Generate the top-left coordinates for all tiles needed to cover a frame.

    The tiling includes overlap and ensures the right and bottom edges are
    still covered even if the frame size is not an exact multiple of stride.
    """
    xs = list(range(0, max(1, W - tile + 1), stride))
    ys = list(range(0, max(1, H - tile + 1), stride))

    # Ensure last tile reaches the right edge
    if xs[-1] != W - tile:
        xs.append(max(0, W - tile))

    # Ensure last tile reaches the bottom edge
    if ys[-1] != H - tile:
        ys.append(max(0, H - tile))

    xs = sorted(set(xs))
    ys = sorted(set(ys))

    for y0 in ys:
        for x0 in xs:
            yield x0, y0


def rolling_mean(values, window):
    """
    Compute a centred rolling mean of a 1D list.

    This is used to smooth the count and size time series.
    """
    out = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        out.append(sum(values[start:end]) / max(1, (end - start)))
    return out


def percentile(sorted_vals, p):
    """
    Compute the p-th percentile from a sorted list using linear interpolation.
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


def parse_gnss_adjustment(path: Path):
    """
    Read GNSS metadata from GNSS_adjustment.txt.

    Returns:
    - latitude in degrees
    - longitude in degrees
    - UTC start time
    """
    lat = lon = None
    t0 = None

    if path is None or not path.exists():
        return lat, lon, t0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue

            key, val = [s.strip() for s in line.split("=", 1)]
            key = key.upper()

            if key == "UTC":
                t0 = datetime.fromisoformat(val)
            elif key == "LAT":
                lat = float(val)
            elif key == "LON":
                lon = float(val)

    return lat, lon, t0


def read_temperature_log(path: Path):
    """
    Read temperature metadata from temperature_log.csv.

    Returns:
    - list of times in seconds
    - list of temperatures in degrees C
    """
    times = []
    temps = []

    if path is None or not path.exists():
        return times, temps

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time_s"]))
            temps.append(float(row["temperature_C"]))

    return times, temps


def temp_at_time(t, t_list, v_list):
    """
    Return the nearest recorded temperature value for a given frame time.
    """
    diffs = [abs(tt - t) for tt in t_list]
    if not diffs:
        return None
    j = diffs.index(min(diffs))
    return v_list[j]


# ---------------- Load metadata ----------------

# Read GNSS metadata if available
lat_deg, lon_deg, t0_utc = parse_gnss_adjustment(GNSS_PATH)

# Read temperature metadata if available
temp_times, temp_vals = read_temperature_log(TEMP_PATH)


# ---------------- Load model ----------------

# Load trained YOLO model
model = YOLO(MODEL_PATH)


# ---------------- Open video ----------------

# Open the selected video file
cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

# Read the native frame rate of the video
native_fps = cap.get(cv2.CAP_PROP_FPS)
if not native_fps or native_fps <= 0:
    raise RuntimeError("Could not read video FPS.")

# Compute how many native frames to skip between analysed frames
step = max(1, int(round(native_fps / FPS_SAMPLE)))

# Read frame resolution
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Print basic run information for traceability
print(f"Video: {VIDEO_PATH}")
print(f"Resolution: {W}x{H}")
print(f"Native FPS: {native_fps:.3f}")
print(f"Sampling step: {step}")
print(f"Tile={TILE}, Overlap={OVERLAP}, Stride={STRIDE}")


# Storage for frame-level and detection-level outputs
frame_rows = []
det_rows = []

frame_idx = -1
sample_idx = -1
global_det_id = 0


# ---------------- Main video analysis loop ----------------

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame_idx += 1

    # Only analyse frames at the chosen sampled rate
    if frame_idx % step != 0:
        continue

    sample_idx += 1
    time_s_val = sample_idx / FPS_SAMPLE

    # Storage for tile detections converted into full-frame coordinates
    full_boxes = []
    full_confs = []

    # Run detection tile by tile across the 640x480 frame
    for x0, y0 in tile_coords(W, H):
        tile_img = frame[y0:y0 + TILE, x0:x0 + TILE]

        # Skip partial or invalid tiles
        if tile_img.shape[0] != TILE or tile_img.shape[1] != TILE:
            continue

        # YOLO inference on one tile
        res = model.predict(tile_img, imgsz=TILE, conf=CONF, verbose=False)[0]

        if res.boxes is None or len(res.boxes) == 0:
            continue

        # Convert tile-local boxes back into full-frame coordinates
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0])

            full_boxes.append((x1 + x0, y1 + y0, x2 + x0, y2 + y0))
            full_confs.append(conf)

    # Deduplicate overlapping detections from adjacent tiles
    if full_boxes:
        keep = nms(full_boxes, full_confs, IOU_NMS)
        boxes = [full_boxes[i] for i in keep]
        confs = [full_confs[i] for i in keep]
    else:
        boxes, confs = [], []

    # Compute per-detection size metrics from retained boxes
    areas = []
    widths = []
    heights = []

    for (x1, y1, x2, y2) in boxes:
        wpx = max(0.0, x2 - x1)
        hpx = max(0.0, y2 - y1)
        widths.append(wpx)
        heights.append(hpx)
        areas.append(wpx * hpx)

    # Number of deduplicated detections in this sampled frame
    count = len(boxes)

    # Frame-level summary statistics for apparent size and confidence
    if areas:
        areas_sorted = sorted(areas)
        area_sum = sum(areas_sorted)
        area_mean = area_sum / count
        area_med = (
            areas_sorted[count // 2]
            if count % 2
            else 0.5 * (areas_sorted[count // 2 - 1] + areas_sorted[count // 2])
        )
        area_p95 = percentile(areas_sorted, 95)
        conf_mean = sum(confs) / count
    else:
        area_sum = area_mean = area_med = area_p95 = 0.0
        conf_mean = 0.0

    # Store frame-level results
    frame_rows.append([
        sample_idx,
        f"{time_s_val:.3f}",
        count,
        f"{area_sum:.2f}",
        f"{area_mean:.2f}",
        f"{area_med:.2f}",
        f"{area_p95:.2f}",
        f"{conf_mean:.3f}"
    ])

    # Save the sampled frame with detections drawn on top
    if SAVE_ANNOTATED_FRAMES:
        vis = frame.copy()
        for (x1, y1, x2, y2), c in zip(boxes, confs):
            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
            cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), (0, 255, 0), 1)
            cv2.putText(
                vis,
                f"{c:.2f}",
                (x1i, max(12, y1i - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )
        cv2.imwrite(str(ANNOTATED_DIR / f"sample_{sample_idx:06d}.jpg"), vis)

    # Save detection-level CSV rows and optional cropped images
    if SAVE_DETECTIONS_CSV and boxes:
        for k, ((x1, y1, x2, y2), c, a) in enumerate(zip(boxes, confs, areas)):
            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])

            # Clamp crop coordinates to valid image bounds
            x1i = max(0, min(x1i, W - 1))
            y1i = max(0, min(y1i, H - 1))
            x2i = max(0, min(x2i, W))
            y2i = max(0, min(y2i, H))

            wpx = x2i - x1i
            hpx = y2i - y1i

            # Store detailed detection record
            det_rows.append([
                sample_idx,
                f"{time_s_val:.3f}",
                global_det_id,
                f"{c:.3f}",
                f"{x1i:.1f}",
                f"{y1i:.1f}",
                f"{x2i:.1f}",
                f"{y2i:.1f}",
                f"{a:.2f}"
            ])

            # Save cropped image patch for this detection
            if SAVE_DETECTION_CROPS and wpx > 0 and hpx > 0:
                crop = frame[y1i:y2i, x1i:x2i]
                crop_name = (
                    f"sample_{sample_idx:06d}_det_{k:03d}"
                    f"_conf_{c:.2f}_x{x1i}_y{y1i}_w{wpx}_h{hpx}.jpg"
                )
                cv2.imwrite(str(CROPS_DIR / crop_name), crop)

            global_det_id += 1

# Release the video file after processing is complete
cap.release()


# ---------------- Write CSV outputs ----------------

# Save frame-level count and size summary
frame_csv = RESULTS_DIR / "counts_and_sizes_by_frame.csv"
with open(frame_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "sample_index", "time_s", "count_dedup",
        "area_sum_px", "area_mean_px", "area_median_px", "area_p95_px",
        "conf_mean"
    ])
    w.writerows(frame_rows)

# Save detailed per-detection results
if SAVE_DETECTIONS_CSV:
    det_csv = RESULTS_DIR / "detections_by_frame.csv"
    with open(det_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "sample_index", "time_s", "det_id",
            "conf", "x1", "y1", "x2", "y2", "area_px"
        ])
        w.writerows(det_rows)


# ---------------- Prepare series for plotting/statistics ----------------

# Convert frame-level rows into analysis series
time_s = [float(r[1]) for r in frame_rows]
counts = [int(float(r[2])) for r in frame_rows]
area_mean_series = [float(r[4]) for r in frame_rows]
area_p95_series = [float(r[6]) for r in frame_rows]

n = len(counts)

# Basic count statistics
total_count = sum(counts)
mean_count_frame = total_count / n if n else 0.0
mean_count_sec = mean_count_frame * FPS_SAMPLE
variance = sum((c - mean_count_frame) ** 2 for c in counts) / n if n else 0.0
std_count = math.sqrt(variance) if n else 0.0
peak_count = max(counts) if counts else 0
peak_time = time_s[counts.index(peak_count)] if counts else 0.0
active_frames = sum(1 for c in counts if c > 0)
active_duration_s = active_frames / FPS_SAMPLE


# ---------------- Save summary files ----------------

# Save machine-readable summary CSV
summary_csv = RESULTS_DIR / "summary_stats.csv"
with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerow(["recording_dir", RECORDING_DIR_NAME])
    w.writerow(["video_path", str(VIDEO_PATH)])
    w.writerow(["model_path", MODEL_PATH])
    w.writerow(["fps_sample", FPS_SAMPLE])
    w.writerow(["tile_px", TILE])
    w.writerow(["overlap", OVERLAP])
    w.writerow(["stride_px", STRIDE])
    w.writerow(["conf_threshold", CONF])
    w.writerow(["nms_iou", IOU_NMS])

    if lat_deg is not None:
        w.writerow(["lat_deg", lat_deg])
    if lon_deg is not None:
        w.writerow(["lon_deg", lon_deg])
    if t0_utc is not None:
        w.writerow(["gnss_start_utc", t0_utc.isoformat()])

    w.writerow(["frames_analysed", n])
    w.writerow(["total_count_sum_frames", total_count])
    w.writerow(["mean_count_per_frame", mean_count_frame])
    w.writerow(["mean_count_per_second", mean_count_sec])
    w.writerow(["std_count_per_frame", std_count])
    w.writerow(["peak_count", peak_count])
    w.writerow(["peak_time_s", peak_time])
    w.writerow(["active_duration_s", active_duration_s])

# Save human-readable summary TXT
summary_txt = RESULTS_DIR / "summary_stats.txt"
with open(summary_txt, "w", encoding="utf-8") as f:
    f.write("Firebrand video evaluation summary\n")
    f.write("=================================\n\n")
    f.write(f"Recording dir           : {RECORDING_DIR_NAME}\n")
    f.write(f"Video                   : {VIDEO_PATH}\n")
    f.write(f"Model                   : {MODEL_PATH}\n")
    f.write(f"Sampling FPS            : {FPS_SAMPLE}\n")
    f.write(f"Tile                    : {TILE}px, Overlap: {OVERLAP}, Stride: {STRIDE}px\n")
    f.write(f"CONF                    : {CONF}, NMS IoU: {IOU_NMS}\n")

    if lat_deg is not None and lon_deg is not None:
        f.write(f"Latitude (deg)          : {lat_deg:.8f}\n")
        f.write(f"Longitude (deg)         : {lon_deg:.8f}\n")
    if t0_utc is not None:
        f.write(f"GNSS start UTC          : {t0_utc.isoformat()}\n")

    f.write("\n")
    f.write(f"Frames analysed         : {n}\n")
    f.write(f"Total count (sum frames): {total_count}\n")
    f.write(f"Mean count per frame    : {mean_count_frame:.2f}\n")
    f.write(f"Mean count per second   : {mean_count_sec:.2f}\n")
    f.write(f"Std. deviation/frame    : {std_count:.2f}\n")
    f.write(f"Peak count/frame        : {peak_count}\n")
    f.write(f"Time of peak (s)        : {peak_time:.2f}\n")
    f.write(f"Active duration (s)     : {active_duration_s:.2f}\n")


# ---------------- Plots: counts and size over time ----------------

# Convert smoothing window from seconds to samples
window = max(1, int(WINDOW_S * FPS_SAMPLE))

# Smoothed series for count and size plots
counts_smooth = rolling_mean(counts, window)
area_mean_smooth = rolling_mean(area_mean_series, window)
area_p95_smooth = rolling_mean(area_p95_series, window)

# Raw deduplicated count time series
plt.figure()
plt.plot(time_s, counts)
plt.xlabel("Time (s)")
plt.ylabel("Firebrand count (deduplicated)")
plt.title("Deduplicated firebrand count over time")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "counts_raw.png", dpi=300)
plt.close()

# Raw + smoothed deduplicated count time series
plt.figure()
plt.plot(time_s, counts, alpha=0.4, label="raw")
plt.plot(time_s, counts_smooth, label=f"{WINDOW_S}s rolling mean")
plt.xlabel("Time (s)")
plt.ylabel("Firebrand count (deduplicated)")
plt.title("Deduplicated firebrand count over time")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "counts_smoothed.png", dpi=300)
plt.close()

# Raw mean apparent size time series
plt.figure()
plt.plot(time_s, area_mean_series)
plt.xlabel("Time (s)")
plt.ylabel("Mean bbox area (px^2)")
plt.title("Mean apparent firebrand size over time")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "area_mean_raw.png", dpi=300)
plt.close()

# Raw + smoothed mean apparent size time series
plt.figure()
plt.plot(time_s, area_mean_series, alpha=0.4, label="raw")
plt.plot(time_s, area_mean_smooth, label=f"{WINDOW_S}s rolling mean")
plt.xlabel("Time (s)")
plt.ylabel("Mean bbox area (px^2)")
plt.title("Mean apparent firebrand size over time")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "area_mean_smoothed.png", dpi=300)
plt.close()

# Raw 95th-percentile apparent size time series
plt.figure()
plt.plot(time_s, area_p95_series)
plt.xlabel("Time (s)")
plt.ylabel("95th percentile bbox area (px^2)")
plt.title("95th percentile apparent firebrand size over time")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "area_p95_raw.png", dpi=300)
plt.close()

# Raw + smoothed 95th-percentile apparent size time series
plt.figure()
plt.plot(time_s, area_p95_series, alpha=0.4, label="raw")
plt.plot(time_s, area_p95_smooth, label=f"{WINDOW_S}s rolling mean")
plt.xlabel("Time (s)")
plt.ylabel("95th percentile bbox area (px^2)")
plt.title("95th percentile apparent firebrand size over time")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "area_p95_smoothed.png", dpi=300)
plt.close()


# ---------------- Histogram plots ----------------

# Gather all non-zero detection areas
all_areas = []
if SAVE_DETECTIONS_CSV:
    for r in det_rows:
        all_areas.append(float(r[8]))

all_areas = [a for a in all_areas if a > 0]

if all_areas:
    # Overall area histogram
    plt.figure()
    plt.hist(all_areas, bins=50)
    plt.xlabel("Bounding-box area (px^2)")
    plt.ylabel("Frequency")
    plt.title("Firebrand apparent size distribution (all detections)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "area_hist_overall.png", dpi=300)
    plt.close()

    # Same histogram on log x-axis
    plt.figure()
    plt.hist(all_areas, bins=50)
    plt.xscale("log")
    plt.xlabel("Bounding-box area (px^2) [log scale]")
    plt.ylabel("Frequency")
    plt.title("Apparent firebrand size distribution (log scale)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "area_hist_log_overall.png", dpi=300)
    plt.close()

    # Build time-area pairs for early/mid/late comparison
    time_area = []
    for r in det_rows:
        t = float(r[1])
        a = float(r[8])
        if a > 0:
            time_area.append((t, a))

    if time_area:
        times = [t for t, _ in time_area]
        t_min, t_max = min(times), max(times)

        # Split event duration into three equal windows
        t1 = t_min + (t_max - t_min) / 3.0
        t2 = t_min + 2.0 * (t_max - t_min) / 3.0

        early = [a for t, a in time_area if t <= t1]
        mid   = [a for t, a in time_area if t1 < t <= t2]
        late  = [a for t, a in time_area if t > t2]

        # Overlay histograms for early, mid, and late phases
        plt.figure()
        if early:
            plt.hist(early, bins=40, alpha=0.5, label="early")
        if mid:
            plt.hist(mid, bins=40, alpha=0.5, label="mid")
        if late:
            plt.hist(late, bins=40, alpha=0.5, label="late")
        plt.xlabel("Bounding-box area (px^2)")
        plt.ylabel("Frequency")
        plt.title("Firebrand size distribution by time window")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "area_hist_windowed.png", dpi=300)
        plt.close()
else:
    print("No detection areas available for histogram plots.")


# ---------------- Temperature plots ----------------

if temp_times and temp_vals:
    # Temperature time series
    plt.figure()
    plt.plot(temp_times, temp_vals)
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (°C)")
    plt.title("Ambient/box temperature vs time")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "temperature_vs_time.png", dpi=300)
    plt.close()

    # Map each analysed frame to nearest temperature sample
    temp_for_frames = [temp_at_time(t, temp_times, temp_vals) for t in time_s]

    # Count and temperature against time on dual y-axes
    plt.figure()
    ax1 = plt.gca()
    ax1.plot(time_s, counts, label="Firebrand count", color="tab:blue")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Count (deduplicated)", color="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(time_s, temp_for_frames, label="Temperature (°C)", color="tab:red", alpha=0.7)
    ax2.set_ylabel("Temperature (°C)", color="tab:red")

    plt.title("Firebrand count and temperature vs time")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "counts_and_temperature_vs_time.png", dpi=300)
    plt.close()

    # Scatter plot of count against nearest temperature
    valid_pairs = [(c, tf) for c, tf in zip(counts, temp_for_frames) if tf is not None]
    if valid_pairs:
        c_vals, t_vals = zip(*valid_pairs)
        plt.figure()
        plt.scatter(t_vals, c_vals, alpha=0.5)
        plt.xlabel("Temperature (°C)")
        plt.ylabel("Firebrand count per frame")
        plt.title("Firebrand count vs temperature")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "scatter_counts_vs_temperature.png", dpi=300)
        plt.close()

# Print output folder location at the end of the run
print(f"All outputs saved to: {RESULTS_DIR.resolve()}")

