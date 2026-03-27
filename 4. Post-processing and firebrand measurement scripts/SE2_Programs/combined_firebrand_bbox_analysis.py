"""

Updated on Mar 25 2026

@author: Andrew Lawson

This script reads bounding-box detection data from all valid sheets in an
Excel workbook, combines the detections into a single dataset, converts
apparent firebrand area from px^2 to mm^2 using a specified calibration,
and produces summary statistics and combined plots.

Outputs:
- combined summary statistics for all detections
- logarithmic histogram of apparent firebrand size in mm^2
- time series of the combined 95th percentile apparent firebrand size
- 5 s rolling-mean smoothed p95 size plot

Expected sheet columns:
- time_s
- area_px

"""

# Import pandas for reading Excel sheets and handling tabular data
import pandas as pd

# Import matplotlib for plotting graphs
import matplotlib.pyplot as plt

# Import Path for safer file path handling
from pathlib import Path


# ---------------- USER SETTINGS ----------------

# Path to the Excel workbook containing multiple sheets of detection data
EXCEL_PATH = r"Graph_Compilation_bbox.xlsx"

# Output filename for the combined size-distribution histogram
OUTPUT_HIST = r"combined_firebrand_size_distribution_mm2_log.png"

# Output filename for the combined p95 time-series plot
OUTPUT_P95 = r"combined_p95_apparent_size_over_time_mm2.png"

# Calibration factors used to convert pixels to millimetres
PX_PER_MM_X = 10.7
PX_PER_MM_Y = 8.0

# Rolling-mean window length used to smooth the p95 time series
ROLLING_WINDOW_S = 5.0

# Number of bins used in the histogram
BINS = 20

# ----------------------------------------------


# Conversion factor from pixel area to mm^2
# Since area scales in two dimensions, multiply x and y calibration terms
PX2_PER_MM2 = PX_PER_MM_X * PX_PER_MM_Y  # 85.6 px^2 per mm^2


def load_all_sheets(excel_path: str) -> pd.DataFrame:
    """
    Read every valid sheet from the Excel workbook and combine them into a
    single DataFrame.

    Each sheet must contain:
    - time_s
    - area_px

    The function:
    - checks required columns exist
    - converts columns to numeric
    - removes invalid rows
    - converts area from px^2 to mm^2
    - tags each row with its sheet name
    """
    # Open the Excel workbook
    xls = pd.ExcelFile(excel_path)

    # List to store cleaned DataFrames from each sheet
    frames = []

    print(f"Reading workbook: {excel_path}")
    print(f"Found sheets: {xls.sheet_names}")

    # Loop over every worksheet in the workbook
    for sheet in xls.sheet_names:
        # Read the current sheet
        df = pd.read_excel(xls, sheet_name=sheet)

        # Required columns for analysis
        required_cols = {"time_s", "area_px"}

        # Skip sheets that do not contain the required data
        if not required_cols.issubset(df.columns):
            print(f"Skipping sheet '{sheet}': missing required columns {required_cols}")
            continue

        # Make a copy so edits do not affect the original read
        df = df.copy()

        # Store the sheet name for traceability
        df["sheet_name"] = sheet

        # Force time and area columns to numeric values
        # Invalid strings are converted to NaN
        df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
        df["area_px"] = pd.to_numeric(df["area_px"], errors="coerce")

        # Remove rows with missing or invalid values
        df = df.dropna(subset=["time_s", "area_px"])

        # Keep only positive detection areas
        df = df[df["area_px"] > 0]

        # Skip empty sheets after cleaning
        if df.empty:
            print(f"Skipping sheet '{sheet}': no valid detections.")
            continue

        # Convert apparent firebrand area from px^2 to mm^2
        df["area_mm2"] = df["area_px"] / PX2_PER_MM2

        # Add cleaned sheet data to the list
        frames.append(df)

        print(f"Loaded sheet '{sheet}': {len(df)} detections")

    # Stop if no sheets were usable
    if not frames:
        raise ValueError("No valid sheets found with 'time_s' and 'area_px'.")

    # Combine all valid sheets into one DataFrame
    return pd.concat(frames, ignore_index=True)


def print_summary_stats(df_all: pd.DataFrame) -> None:
    """
    Print basic summary statistics for the combined area data across all sheets.
    """
    # Extract the converted apparent area series
    area = df_all["area_mm2"]

    print("\nCombined summary statistics (all sheets):")
    print(f"Detections : {len(area)}")
    print(f"Mean       : {area.mean():.4f} mm^2")
    print(f"Median     : {area.median():.4f} mm^2")
    print(f"P95        : {area.quantile(0.95):.4f} mm^2")


def plot_combined_histogram(df_all: pd.DataFrame, output_path: str) -> None:
    """
    Plot and save a combined histogram of apparent firebrand size in mm^2
    using a logarithmic x-axis.
    """
    plt.figure(figsize=(8, 5))

    # Plot histogram of all apparent areas combined
    plt.hist(df_all["area_mm2"], bins=BINS)

    # Log scale helps visualise wide size distributions more clearly
    plt.xscale("log")

    plt.xlabel("Bounding-box area (mm$^2$) [log scale]")
    plt.ylabel("Frequency")
    plt.title("Firebrand size distribution (log scale)")
    plt.tight_layout()

    # Save the figure to file
    plt.savefig(output_path, dpi=300)

    # Display the figure
    plt.show()


def build_combined_p95_timeseries(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Build a combined time series of the 95th percentile apparent firebrand
    area across all sheets.

    For each unique time_s value:
    - pool detections from all sheets
    - compute the 95th percentile area in mm^2
    - apply a rolling mean for smoothing
    """
    # Group all detections by time and compute p95 apparent area
    ts = (
        df_all.groupby("time_s")["area_mm2"]
        .quantile(0.95)
        .reset_index()
        .rename(columns={"area_mm2": "p95_area_mm2"})
        .sort_values("time_s")
    )

    # Estimate time spacing between consecutive samples
    dt = ts["time_s"].diff().median()

    # Stop if a sensible time step cannot be inferred
    if pd.isna(dt) or dt <= 0:
        raise ValueError("Could not determine valid time spacing from time_s.")

    # Convert desired rolling window from seconds to number of samples
    window_n = max(1, int(round(ROLLING_WINDOW_S / dt)))

    # Apply rolling mean to smooth the p95 series
    ts["p95_area_mm2_roll"] = (
        ts["p95_area_mm2"]
        .rolling(window=window_n, min_periods=1)
        .mean()
    )

    return ts


def plot_combined_p95(ts: pd.DataFrame, output_path: str) -> None:
    """
    Plot and save the combined p95 apparent size time series, including both
    the raw series and the 5 s rolling-mean smoothed series.
    """
    plt.figure(figsize=(8, 5))

    # Plot raw p95 values with transparency
    plt.plot(ts["time_s"], ts["p95_area_mm2"], label="raw", alpha=0.35)

    # Plot smoothed p95 values
    plt.plot(ts["time_s"], ts["p95_area_mm2_roll"], label="5 s rolling mean")

    plt.xlabel("Time (s)")
    plt.ylabel("95th percentile bbox area (mm$^2$)")
    plt.title("95th percentile apparent firebrand size over time")
    plt.legend()
    plt.tight_layout()

    # Save the figure to file
    plt.savefig(output_path, dpi=300)

    # Display the figure
    plt.show()


def main():
    """
    Main workflow:
    1. Check the Excel workbook exists
    2. Load and combine valid sheets
    3. Print summary statistics
    4. Plot combined histogram
    5. Build and plot combined p95 time series
    6. Print output filenames
    """
    # Convert configured Excel path to a Path object
    excel_path = Path(EXCEL_PATH)

    # Check the workbook exists before continuing
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    # Load all valid detection data from the workbook
    df_all = load_all_sheets(str(excel_path))

    # Print summary statistics for the combined dataset
    print_summary_stats(df_all)

    # Create and save the combined histogram
    plot_combined_histogram(df_all, OUTPUT_HIST)

    # Build the combined p95 time series
    ts = build_combined_p95_timeseries(df_all)

    # Create and save the p95 time-series plot
    plot_combined_p95(ts, OUTPUT_P95)

    # Report saved output locations
    print(f"\nSaved histogram to: {OUTPUT_HIST}")
    print(f"Saved p95 plot to : {OUTPUT_P95}")


# Run the script only when executed directly
if __name__ == "__main__":
    main()