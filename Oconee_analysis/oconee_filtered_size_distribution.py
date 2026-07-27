"""
Updated on July 22 2026

@author: Andrew Lawson

Oconee filtered firebrand size-distribution workflow

Purpose
-------
This script combines the individual detection-level CSV files produced by the
Oconee firebrand size-analysis workflow and creates pooled apparent-size
distributions after applying a minimum and optional maximum size threshold.

The script is useful for investigating the upper part of the apparent
firebrand-size distribution while deliberately excluding smaller detections.
For example, setting MIN_AREA_MM2 = 100.0 retains only detections with an
apparent ML-derived bounding-box area of at least 100 mm².

Workflow
--------
1. Search RESULTS_FOLDER for per-video detection CSV files.
2. Load each CSV and locate either:
      - area_mm2, if already available, or
      - area_px, which is converted to mm² using the supplied calibration.
3. Remove invalid / missing area measurements.
4. Discard detections smaller than MIN_AREA_MM2.
5. Optionally discard detections larger than MAX_AREA_MM2.
6. Report how many detections were removed and retained from each video.
7. Pool all retained detection rows across all videos.
8. Save:
      - a linear-scale combined histogram,
      - a log-scale combined histogram,
      - a combined filtered CSV,
      - an Excel workbook containing summary statistics and all retained rows.

Important interpretation note
-----------------------------
The values represent apparent ML-derived bounding-box areas, not true physical
firebrand areas. Large detections may represent genuine large firebrands, but
they can also be influenced by motion blur, moving vegetation, enclosure
movement, poor focus, background objects, merged particles or other false
detections.

The minimum-size threshold should therefore be interpreted as an analysis filter
rather than a physical classification unless it has been independently
validated.

Calibration note
----------------
The conversion from px² to mm² uses:

    PX2_PER_MM2 = PX_PER_MM_X * PX_PER_MM_Y

and is only valid where the supplied pixel calibration is appropriate for the
camera geometry used in the source recording.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

# Folder containing the individual per-video CSV files
RESULTS_FOLDER = Path(
    r"Path"
)

# Output folder
OUTPUT_FOLDER = RESULTS_FOLDER / "combined_size_distribution_min_filtered"

# Pixel calibration
# Used only if area_mm2 is missing from the CSV files
PX_PER_MM_X = 10.7
PX_PER_MM_Y = 8.0

# Histogram settings
BINS = 50

# ------------------------------------------------------------
# SIZE FILTER
# ------------------------------------------------------------

# Discard all detected firebrands SMALLER than this value
# Example:
# 1.0 = ignore anything below 1 mm²
# 2.0 = ignore anything below 2 mm²
MIN_AREA_MM2 = 100.0

# Optional upper size limit
# Leave as None if you do not want to remove large detections
MAX_AREA_MM2 = None

# CSV filename pattern
CSV_PATTERN = "*_detections_with_size.csv"


# ============================================================
# DERIVED SETTINGS
# ============================================================

PX2_PER_MM2 = PX_PER_MM_X * PX_PER_MM_Y


# ============================================================
# FUNCTIONS
# ============================================================

def find_detection_csvs(folder):
    """
    Find individual per-video detection CSV files.
    """

    return sorted(folder.glob(CSV_PATTERN))


def load_detection_csv(csv_path):
    """
    Load one per-video detection CSV.

    Each row represents an individual ML detection.

    Detections smaller than MIN_AREA_MM2 are discarded.
    """

    print(f"Reading CSV: {csv_path.name}")

    try:
        df = pd.read_csv(csv_path)

    except Exception as e:
        print(f"  Could not read {csv_path.name}: {e}")
        return pd.DataFrame()

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    # --------------------------------------------------------
    # Ensure area_mm2 exists
    # --------------------------------------------------------

    if "area_mm2" in df.columns:

        df["area_mm2"] = pd.to_numeric(
            df["area_mm2"],
            errors="coerce"
        )

    elif "area_px" in df.columns:

        df["area_px"] = pd.to_numeric(
            df["area_px"],
            errors="coerce"
        )

        df["area_mm2"] = (
            df["area_px"] / PX2_PER_MM2
        )

    else:

        print(
            f"  Skipped {csv_path.name}: "
            "no area_mm2 or area_px column found"
        )

        return pd.DataFrame()

    # --------------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------------

    df = df.dropna(
        subset=["area_mm2"]
    )

    original_count = len(df)

    # --------------------------------------------------------
    # Remove detections smaller than chosen threshold
    # --------------------------------------------------------

    df = df[
        df["area_mm2"] >= MIN_AREA_MM2
    ]

    # Optional upper limit
    if MAX_AREA_MM2 is not None:

        df = df[
            df["area_mm2"] <= MAX_AREA_MM2
        ]

    filtered_count = len(df)
    removed_count = original_count - filtered_count

    print(f"  Original detections: {original_count}")
    print(f"  Removed below {MIN_AREA_MM2} mm²: {removed_count}")
    print(f"  Remaining detections: {filtered_count}")

    # Add source filename
    df["source_csv"] = csv_path.name

    # If video column is missing, infer it from filename
    if "video" not in df.columns:

        df["video"] = (
            csv_path.stem
            .replace("_detections_with_size", "")
        )

    return df


def plot_combined_histogram(all_df):
    """
    Plot combined firebrand size distribution
    after small detections have been removed.
    """

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    total_detections = len(all_df)
    total_videos = all_df["video"].nunique()

    plt.figure(figsize=(9, 5))

    plt.hist(
        all_df["area_mm2"],
        bins=BINS
    )

    plt.xlabel(
        "Apparent bounding-box area (mm$^2$)"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Firebrand size distribution across all videos\n"
        f"Detections ≥ {MIN_AREA_MM2} mm² | "
        f"{total_detections} detections from "
        f"{total_videos} videos"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_FOLDER /
        "combined_firebrand_size_distribution_filtered_mm2.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(f"Saved: {output_path}")


def plot_combined_histogram_log(all_df):
    """
    Plot same filtered distribution using a log x-axis.
    """

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    total_detections = len(all_df)
    total_videos = all_df["video"].nunique()

    plt.figure(figsize=(9, 5))

    plt.hist(
        all_df["area_mm2"],
        bins=BINS
    )

    plt.xscale("log")

    plt.xlabel(
        "Apparent bounding-box area (mm$^2$) [log scale]"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Firebrand size distribution across all videos, log scale\n"
        f"Detections ≥ {MIN_AREA_MM2} mm² | "
        f"{total_detections} detections from "
        f"{total_videos} videos"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_FOLDER /
        "combined_firebrand_size_distribution_filtered_mm2_log.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(f"Saved: {output_path}")


def save_combined_outputs(all_df):
    """
    Save filtered detection table and summary statistics.
    """

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save combined CSV
    # --------------------------------------------------------

    combined_csv = (
        OUTPUT_FOLDER /
        "combined_filtered_detection_sizes.csv"
    )

    all_df.to_csv(
        combined_csv,
        index=False
    )

    print(f"Saved: {combined_csv}")

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    summary = {

        "minimum_size_filter_mm2":
            MIN_AREA_MM2,

        "total_detections_after_filtering":
            len(all_df),

        "total_videos":
            all_df["video"].nunique(),

        "mean_area_mm2":
            all_df["area_mm2"].mean(),

        "median_area_mm2":
            all_df["area_mm2"].median(),

        "p05_area_mm2":
            all_df["area_mm2"].quantile(0.05),

        "p25_area_mm2":
            all_df["area_mm2"].quantile(0.25),

        "p75_area_mm2":
            all_df["area_mm2"].quantile(0.75),

        "p95_area_mm2":
            all_df["area_mm2"].quantile(0.95),

        "max_area_mm2":
            all_df["area_mm2"].max()
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_excel = (
        OUTPUT_FOLDER /
        "combined_filtered_detection_size_summary.xlsx"
    )

    with pd.ExcelWriter(
        summary_excel,
        engine="openpyxl"
    ) as writer:

        summary_df.to_excel(
            writer,
            sheet_name="summary",
            index=False
        )

        all_df.to_excel(
            writer,
            sheet_name="all_filtered_detections",
            index=False
        )

    print(f"Saved: {summary_excel}")


# ============================================================
# MAIN
# ============================================================

def main():

    csv_files = find_detection_csvs(
        RESULTS_FOLDER
    )

    if not csv_files:

        print(
            f"No detection CSV files found in: "
            f"{RESULTS_FOLDER}"
        )

        return

    print(
        f"Found {len(csv_files)} "
        "detection CSV files."
    )

    print()
    print(
        f"Minimum retained area: "
        f"{MIN_AREA_MM2} mm²"
    )

    all_tables = []

    for csv_file in csv_files:

        df = load_detection_csv(
            csv_file
        )

        if not df.empty:

            all_tables.append(df)

    if not all_tables:

        print(
            "No detections remain after filtering."
        )

        return

    all_df = pd.concat(
        all_tables,
        ignore_index=True
    )

    print()
    print(
        f"Total detections after filtering: "
        f"{len(all_df)}"
    )

    print(
        f"Total videos: "
        f"{all_df['video'].nunique()}"
    )

    print()
    print("Videos included:")

    for video in sorted(
        all_df["video"].dropna().unique()
    ):

        count = len(
            all_df[
                all_df["video"] == video
            ]
        )

        print(
            f"  - {video}: "
            f"{count} detections"
        )

    # Produce plots
    plot_combined_histogram(all_df)

    plot_combined_histogram_log(all_df)

    # Save filtered data
    save_combined_outputs(all_df)

    print()
    print("Done.")

    print(
        f"Outputs saved to: "
        f"{OUTPUT_FOLDER}"
    )


if __name__ == "__main__":
    main()