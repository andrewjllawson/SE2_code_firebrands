import csv
import math
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ---------------- USER CONFIG (only change these per experiment) ----------------

# Name of the recording folder to analyse
RECORDING_DIR_NAME = "recording_analysis_insert_here"

# Path to the trained YOLO model weights
MODEL_PATH = r"runs/detect/train5/weights/best.pt"

# -------------------------------------------------------------------------------


# ---------------- FIXED CONFIG ----------------

# Effective frame sampling rate used for analysis (Hz)
FPS_SAMPLE = 2.0

# Tile size used to split the full frame for inference
TILE = 640

# Fractional overlap between adjacent tiles
OVERLAP = 0.25

# Tile stride computed from tile size and overlap
STRIDE = int(TILE * (1 - OVERLAP))  # 480 for 25% overlap

# Detection confidence threshold for YOLO inference
CONF = 0.5

# IoU threshold used for frame-level non-maximum suppression
IOU_NMS = 0.5

# Rolling-mean window length used for smoothed plots
WINDOW_S = 5

# Whether to save a detailed CSV of every detection
SAVE_DETECTIONS_CSV = True

# Constant deposition area used to normalise count to count per m^2
DEPOSITION_AREA_M2 = 0.03

# ---------------------------------------------


# Resolve the directory containing this script
BASE_DIR = Path(__file__).resolve().parent

# Recording directory for the selected experiment
RECORDING_DIR = BASE_DIR / RECORDING_DIR_NAME

# Expected input paths
VIDEO_PATH   = RECORDING_DIR / "inputs" / "videos"   / "final_rgb_video.mp4"
METADATA_DIR = RECORDING_DIR / "inputs" / "metadata"
TEMP_PATH    = METADATA_DIR / "temperature_log.csv"
GNSS_PATH    = METADATA_DIR / "GNSS_adjustment.txt"

# Directory where all outputs will be written
RESULTS_DIR  = RECORDING_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


def iou_xyxy(a, b):
    """
    Compute the intersection-over-union (IoU) of two bounding boxes in
    [x1, y1, x2, y2] format.

    This is used during non-maximum suppression to identify overlapping
    detections that likely refer to the same firebrand.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    # Coordinates of overlapping region
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    # Width and height of overlap
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    # Areas of the two boxes
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    # Union area with a small epsilon to avoid division by zero
    union = area_a + area_b - inter + 1e-9

    return inter / union


def nms(boxes, confs, iou_thr):
    """
    Apply non-maximum suppression (NMS) to a list of boxes.

    Boxes are sorted by confidence, and lower-confidence boxes are removed
    if they overlap too strongly with a higher-confidence box.
    """
    idxs = sorted(range(len(boxes)), key=lambda i: confs[i], reverse=True)
    keep = []

    while idxs:
        # Keep the highest-confidence remaining box
        i = idxs.pop(0)
        keep.append(i)

        # Remove boxes with IoU above the chosen threshold
        idxs = [j for j in idxs if iou_xyxy(boxes[i], boxes[j]) < iou_thr]

    return keep


def tile_coords(W, H, tile=TILE, stride=STRIDE):
    """
    Generate the top-left coordinates of all tiles needed to cover a frame.

    Tiles are laid out with overlap. Additional edge tiles are added so that
    the full frame is covered right to the boundaries.
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
    Compute a simple rolling mean for a 1D list.

    The smoothing is centred approximately on each point and is used to
    reduce short-term variability in count and size time series.
    """
    out = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2)
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

    Expected keys are:
    - UTC
    - LAT
    - LON

    Returns latitude, longitude, and UTC start time if present.
    """
    lat = lon = None
    t0 = None

    if not path.exists():
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
    Read the temperature log CSV and return:
    - times in seconds
    - temperatures in degrees C
    """
    times = []
    temps = []

    if not path.exists():
        return times, temps

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time_s"]))
            temps.append(float(row["temperature_C"]))

    return times, temps


# ---------------- Load metadata ----------------

# Read GNSS metadata, if available
lat_deg, lon_deg, t0_utc = parse_gnss_adjustment(GNSS_PATH)

# Read temperature metadata, if available
temp_times, temp_vals = read_temperature_log(TEMP_PATH)


# ---------------- Load model ----------------

# Load the trained YOLO detector
model = YOLO(MODEL_PATH)


# ---------------- Open video ----------------

# Open the RGB video file for analysis
cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

# Read the native video frame rate
native_fps = cap.get(cv2.CAP_PROP_FPS)
if not native_fps or native_fps <= 0:
    raise RuntimeError("Could not read video FPS.")

# Determine how many native frames to skip between sampled frames
step = max(1, int(round(native_fps / FPS_SAMPLE)))

# Read frame dimensions
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# Storage for per-frame summary rows and per-detection rows
frame_rows = []
det_rows = []

frame_idx = -1
sample_idx = -1


# ---------------- Main video analysis loop ----------------

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame_idx += 1

    # Only analyse frames according to the desired sampling rate
    if frame_idx % step != 0:
        continue

    sample_idx += 1

    # Time assigned to this analysed frame
    time_s_val = sample_idx / FPS_SAMPLE

    # Full-frame detection storage before deduplication
    full_boxes = []
    full_confs = []

    # Run detection tile by tile across the full image
    for x0, y0 in tile_coords(W, H):
        tile_img = frame[y0:y0 + TILE, x0:x0 + TILE]

        # Skip partial tiles, though edge logic should usually prevent these
        if tile_img.shape[0] != TILE or tile_img.shape[1] != TILE:
            continue

        # Run YOLO inference on this tile
        res = model.predict(tile_img, imgsz=TILE, conf=CONF, verbose=False)[0]

        # Skip tiles with no detections
        if res.boxes is None or len(res.boxes) == 0:
            continue

        # Convert tile-local boxes into full-frame coordinates
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0])

            full_boxes.append((x1 + x0, y1 + y0, x2 + x0, y2 + y0))
            full_confs.append(conf)

    # Deduplicate detections caused by overlapping tiles
    if full_boxes:
        keep = nms(full_boxes, full_confs, IOU_NMS)
        boxes = [full_boxes[i] for i in keep]
        confs = [full_confs[i] for i in keep]
    else:
        boxes, confs = [], []

    # Compute bounding-box area for each retained detection
    areas = []
    for (x1, y1, x2, y2) in boxes:
        wpx = max(0.0, x2 - x1)
        hpx = max(0.0, y2 - y1)
        areas.append(wpx * hpx)

    # Number of deduplicated firebrands in this sampled frame
    count = len(boxes)

    # Area-normalised count
    count_per_m2 = count / DEPOSITION_AREA_M2

    # Compute frame-level size and confidence statistics
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

    # Store one row of summary data for this sampled frame
    frame_rows.append([
        sample_idx,
        f"{time_s_val:.3f}",
        count,
        f"{count_per_m2:.3f}",
        f"{area_sum:.2f}",
        f"{area_mean:.2f}",
        f"{area_med:.2f}",
        f"{area_p95:.2f}",
        f"{conf_mean:.3f}"
    ])

    # Optionally store every retained detection in detail
    if SAVE_DETECTIONS_CSV and boxes:
        for k, ((x1, y1, x2, y2), c, a) in enumerate(zip(boxes, confs, areas)):
            det_rows.append([
                sample_idx,
                f"{time_s_val:.3f}",
                k,
                f"{c:.3f}",
                f"{x1:.1f}",
                f"{y1:.1f}",
                f"{x2:.1f}",
                f"{y2:.1f}",
                f"{a:.2f}"
            ])

# Release the video file when processing is complete
cap.release()


# ---------------- Write CSV outputs ----------------

# Write frame-level summary table
frame_csv = RESULTS_DIR / "counts_and_sizes_by_frame.csv"
with open(frame_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "sample_index", "time_s", "count_dedup", "count_per_m2",
        "area_sum_px", "area_mean_px", "area_median_px", "area_p95_px",
        "conf_mean"
    ])
    w.writerows(frame_rows)

# Write detailed per-detection table
if SAVE_DETECTIONS_CSV:
    det_csv = RESULTS_DIR / "detections_by_frame.csv"
    with open(det_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "sample_index", "time_s", "det_id", "conf",
            "x1", "y1", "x2", "y2", "area_px"
        ])
        w.writerows(det_rows)


# ---------------- Prepare series for plotting/statistics ----------------

# Extract time and metric series from frame_rows
time_s = [float(r[1]) for r in frame_rows]
counts = [int(float(r[2])) for r in frame_rows]
counts_per_m2 = [float(r[3]) for r in frame_rows]
area_mean = [float(r[5]) for r in frame_rows]
area_p95 = [float(r[7]) for r in frame_rows]

n = len(counts)

# Basic count statistics
total_count = sum(counts)
mean_count_frame = total_count / n if n else 0.0
mean_count_sec = mean_count_frame * FPS_SAMPLE
mean_count_per_m2_frame = sum(counts_per_m2) / n if n else 0.0
variance = sum((c - mean_count_frame) ** 2 for c in counts) / n if n else 0.0
std_count = math.sqrt(variance) if n else 0.0
peak_count = max(counts) if counts else 0
peak_count_per_m2 = max(counts_per_m2) if counts_per_m2 else 0.0
peak_time = time_s[counts.index(peak_count)] if counts else 0.0
active_frames = sum(1 for c in counts if c > 0)
active_duration_s = active_frames / FPS_SAMPLE


# ---------------- Save summary files ----------------

# Save summary statistics as CSV
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
    w.writerow(["deposition_area_m2", DEPOSITION_AREA_M2])

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
    w.writerow(["mean_count_per_m2_per_frame", mean_count_per_m2_frame])
    w.writerow(["std_count_per_frame", std_count])
    w.writerow(["peak_count", peak_count])
    w.writerow(["peak_count_per_m2", peak_count_per_m2])
    w.writerow(["peak_time_s", peak_time])
    w.writerow(["active_duration_s", active_duration_s])

# Save the same summary as a human-readable text file
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
    f.write(f"Deposition area (m^2)   : {DEPOSITION_AREA_M2}\n")

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
    f.write(f"Mean count per m^2/frame: {mean_count_per_m2_frame:.2f}\n")
    f.write(f"Std. deviation/frame    : {std_count:.2f}\n")
    f.write(f"Peak count/frame        : {peak_count}\n")
    f.write(f"Peak count per m^2      : {peak_count_per_m2:.2f}\n")
    f.write(f"Time of peak (s)        : {peak_time:.2f}\n")
    f.write(f"Active duration (s)     : {active_duration_s:.2f}\n")


# ---------------- Plots: counts and size over time ----------------

# Convert smoothing window from seconds to samples
window = max(1, int(WINDOW_S * FPS_SAMPLE))

# Smoothed series for plotting
counts_per_m2_smooth = rolling_mean(counts_per_m2, window)
area_mean_smooth = rolling_mean(area_mean, window)
area_p95_smooth = rolling_mean(area_p95, window)

# Raw count-per-area time series
plt.figure()
plt.plot(time_s, counts_per_m2)
plt.xlabel("Time (s)")
plt.ylabel(r"Firebrand count per m$^2$")
plt.title(r"Deduplicated firebrand count per m$^2$ over time")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "counts_per_m2_raw.png", dpi=300)
plt.close()

# Raw + smoothed count-per-area time series
plt.figure()
plt.plot(time_s, counts_per_m2, alpha=0.4, label="raw")
plt.plot(time_s, counts_per_m2_smooth, label=f"{WINDOW_S}s rolling mean")
plt.xlabel("Time (s)")
plt.ylabel(r"Firebrand count per m$^2$")
plt.title(r"Deduplicated firebrand count per m$^2$ over time")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "counts_per_m2_smoothed.png", dpi=300)
plt.close()

# Raw mean apparent size time series
plt.figure()
plt.plot(time_s, area_mean)
plt.xlabel("Time (s)")
plt.ylabel("Mean bbox area (px^2)")
plt.title("Mean apparent firebrand size over time")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "area_mean_raw.png", dpi=300)
plt.close()

# Raw + smoothed mean apparent size time series
plt.figure()
plt.plot(time_s, area_mean, alpha=0.4, label="raw")
plt.plot(time_s, area_mean_smooth, label=f"{WINDOW_S}s rolling mean")
plt.xlabel("Time (s)")
plt.ylabel("Mean bbox area (px^2)")
plt.title("Mean apparent firebrand size over time")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "area_mean_smoothed.png", dpi=300)
plt.close()

# Raw p95 apparent size time series
plt.figure()
plt.plot(time_s, area_p95)
plt.xlabel("Time (s)")
plt.ylabel("95th percentile bbox area (px^2)")
plt.title("95th percentile apparent firebrand size over time")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "area_p95_raw.png", dpi=300)
plt.close()

# Raw + smoothed p95 apparent size time series
plt.figure()
plt.plot(time_s, area_p95, alpha=0.4, label="raw")
plt.plot(time_s, area_p95_smooth, label=f"{WINDOW_S}s rolling mean")
plt.xlabel("Time (s)")
plt.ylabel("95th percentile bbox area (px^2)")
plt.title("95th percentile apparent firebrand size over time")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "area_p95_smoothed.png", dpi=300)
plt.close()


# ---------------- Histogram plots (bbox area) ----------------

# Collect all detection areas from the detailed detection rows
all_areas = []
if SAVE_DETECTIONS_CSV:
    for r in det_rows:
        all_areas.append(float(r[8]))

# Remove zero or invalid areas
all_areas = [a for a in all_areas if a > 0]

if all_areas:
    # Overall size distribution histogram
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

    # Build a time-tagged list of areas for early/mid/late comparison
    time_area = []
    if SAVE_DETECTIONS_CSV:
        for r in det_rows:
            t = float(r[1])
            a = float(r[8])
            if a > 0:
                time_area.append((t, a))

    if time_area:
        times = [t for t, _ in time_area]
        t_min, t_max = min(times), max(times)

        # Divide duration into three equal time windows
        t1 = t_min + (t_max - t_min) / 3.0
        t2 = t_min + 2.0 * (t_max - t_min) / 3.0

        early = [a for t, a in time_area if t <= t1]
        mid   = [a for t, a in time_area if t1 < t <= t2]
        late  = [a for t, a in time_area if t > t2]

        # Overlay histograms for early, mid, and late detections
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


# ---------------- Temperature plots (if metadata present) ----------------

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

    def temp_at_time(t, t_list, v_list):
        """
        Find the nearest recorded temperature value for a given frame time.
        """
        diffs = [abs(tt - t) for tt in t_list]
        if not diffs:
            return None
        j = diffs.index(min(diffs))
        return v_list[j]

    # Map each analysed frame time to its nearest temperature reading
    temp_for_frames = [temp_at_time(t, temp_times, temp_vals) for t in time_s]

    # Count-per-area and temperature on dual y-axes
    plt.figure()
    ax1 = plt.gca()
    ax1.plot(time_s, counts_per_m2, label="Firebrand count per m$^2$", color="tab:blue")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel(r"Count per m$^2$", color="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(time_s, temp_for_frames, label="Temperature (°C)", color="tab:red", alpha=0.7)
    ax2.set_ylabel("Temperature (°C)", color="tab:red")

    plt.title("Firebrand count per m$^2$ and temperature vs time")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "counts_per_m2_and_temperature_vs_time.png", dpi=300)
    plt.close()

    # Scatter plot of count-per-area against temperature
    valid_pairs = [(c, tf) for c, tf in zip(counts_per_m2, temp_for_frames) if tf is not None]
    if valid_pairs:
        c_vals, t_vals = zip(*valid_pairs)
        plt.figure()
        plt.scatter(t_vals, c_vals, alpha=0.5)
        plt.xlabel("Temperature (°C)")
        plt.ylabel(r"Firebrand count per m$^2$")
        plt.title(r"Firebrand count per m$^2$ vs temperature")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "scatter_counts_per_m2_vs_temperature.png", dpi=300)
        plt.close()


# Print output location when complete
print(f"All outputs saved to: {RESULTS_DIR.resolve()}")
